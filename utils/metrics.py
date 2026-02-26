"""
In-process metrics collection for RAG Legal Search.

Provides lightweight, thread-safe metrics without external dependencies.
Supports counters, gauges, and histograms with Prometheus-compatible export.

Usage:
    from utils.metrics import get_collector

    metrics = get_collector()
    metrics.increment("search_requests_total", labels={"status": "success"})
    metrics.observe_histogram("search_latency_ms", 42.5)
    metrics.set_gauge("active_connections", 3)
"""

import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class HistogramBucket:
    """Pre-defined histogram bucket boundaries."""

    # Search latency buckets (ms)
    LATENCY_MS: Tuple[float, ...] = (
        5,
        10,
        25,
        50,
        100,
        250,
        500,
        1000,
        2500,
        5000,
        10000,
    )

    # Result count buckets
    RESULT_COUNT: Tuple[float, ...] = (0, 1, 5, 10, 25, 50, 100)

    # Document size buckets (bytes)
    DOC_SIZE: Tuple[float, ...] = (
        256,
        1024,
        4096,
        16384,
        65536,
        262144,
        1048576,
    )


@dataclass
class Histogram:
    """Thread-safe histogram with configurable buckets."""

    buckets: Tuple[float, ...]
    _counts: Dict[float, int] = field(default_factory=lambda: defaultdict(int))
    _sum: float = 0.0
    _count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, value: float) -> None:
        """Record an observation."""
        with self._lock:
            self._sum += value
            self._count += 1
            for bound in self.buckets:
                if value <= bound:
                    self._counts[bound] += 1
            self._counts[math.inf] += 1  # +Inf bucket

    def snapshot(self) -> Dict:
        """Get a snapshot of the histogram state."""
        with self._lock:
            cumulative = 0
            bucket_data = {}
            for bound in self.buckets:
                cumulative += self._counts.get(bound, 0)
                bucket_data[str(bound)] = cumulative
            cumulative += self._counts.get(math.inf, 0) - cumulative
            bucket_data["+Inf"] = self._count

            return {
                "buckets": bucket_data,
                "sum": round(self._sum, 4),
                "count": self._count,
                "avg": round(self._sum / self._count, 4) if self._count > 0 else 0,
            }


class MetricsCollector:
    """
    Central metrics collector supporting counters, gauges, and histograms.

    Thread-safe and designed for concurrent access from multiple
    request handlers.
    """

    def __init__(self):
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._labeled_counters: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._lock = threading.Lock()
        self._start_time = time.time()

        # Register default histograms
        self._histograms["search_latency_ms"] = Histogram(
            buckets=HistogramBucket.LATENCY_MS
        )
        self._histograms["embedding_latency_ms"] = Histogram(
            buckets=HistogramBucket.LATENCY_MS
        )
        self._histograms["result_count"] = Histogram(
            buckets=HistogramBucket.RESULT_COUNT
        )
        self._histograms["document_size_bytes"] = Histogram(
            buckets=HistogramBucket.DOC_SIZE
        )

    def increment(
        self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Increment a counter.

        Args:
            name: Metric name
            value: Amount to increment by
            labels: Optional label key-value pairs
        """
        with self._lock:
            if labels:
                label_key = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
                self._labeled_counters[name][label_key] += value
            else:
                self._counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        """
        Set a gauge to an absolute value.

        Args:
            name: Metric name
            value: Current value
        """
        with self._lock:
            self._gauges[name] = value

    def observe_histogram(self, name: str, value: float) -> None:
        """
        Record an observation in a histogram.

        Creates the histogram with default latency buckets if it doesn't exist.

        Args:
            name: Metric name
            value: Observed value
        """
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(buckets=HistogramBucket.LATENCY_MS)
        self._histograms[name].observe(value)

    def snapshot(self) -> Dict:
        """
        Get a complete snapshot of all metrics.

        Returns a dictionary suitable for JSON serialization and
        API endpoint responses.
        """
        with self._lock:
            uptime = time.time() - self._start_time

            result = {
                "uptime_seconds": round(uptime, 2),
                "counters": dict(self._counters),
                "labeled_counters": {
                    name: dict(labels)
                    for name, labels in self._labeled_counters.items()
                },
                "gauges": dict(self._gauges),
                "histograms": {
                    name: hist.snapshot() for name, hist in self._histograms.items()
                },
            }

        return result

    def to_prometheus(self) -> str:
        """
        Export metrics in Prometheus text exposition format.

        Returns:
            Multi-line string in Prometheus format
        """
        lines: List[str] = []
        snap = self.snapshot()

        # Uptime gauge
        lines.append("# HELP app_uptime_seconds Application uptime in seconds")
        lines.append("# TYPE app_uptime_seconds gauge")
        lines.append(f"app_uptime_seconds {snap['uptime_seconds']}")
        lines.append("")

        # Counters
        for name, value in snap["counters"].items():
            safe_name = name.replace(".", "_").replace("-", "_")
            lines.append(f"# TYPE {safe_name} counter")
            lines.append(f"{safe_name} {value}")
            lines.append("")

        # Labeled counters
        for name, labels_dict in snap["labeled_counters"].items():
            safe_name = name.replace(".", "_").replace("-", "_")
            lines.append(f"# TYPE {safe_name} counter")
            for label_str, value in labels_dict.items():
                lines.append(f"{safe_name}{{{label_str}}} {value}")
            lines.append("")

        # Gauges
        for name, value in snap["gauges"].items():
            safe_name = name.replace(".", "_").replace("-", "_")
            lines.append(f"# TYPE {safe_name} gauge")
            lines.append(f"{safe_name} {value}")
            lines.append("")

        # Histograms
        for name, hist_data in snap["histograms"].items():
            safe_name = name.replace(".", "_").replace("-", "_")
            lines.append(f"# TYPE {safe_name} histogram")
            for bound, count in hist_data["buckets"].items():
                lines.append(f'{safe_name}_bucket{{le="{bound}"}} {count}')
            lines.append(f"{safe_name}_sum {hist_data['sum']}")
            lines.append(f"{safe_name}_count {hist_data['count']}")
            lines.append("")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._labeled_counters.clear()
            for hist in self._histograms.values():
                hist._counts.clear()
                hist._sum = 0.0
                hist._count = 0


# Singleton collector
_collector: Optional[MetricsCollector] = None
_collector_lock = threading.Lock()


def get_collector() -> MetricsCollector:
    """Get or create the global metrics collector singleton."""
    global _collector
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                _collector = MetricsCollector()
    return _collector
