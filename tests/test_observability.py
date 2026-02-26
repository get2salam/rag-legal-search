"""
Tests for the observability stack: logging, metrics, and health checks.
"""

import json
import logging
import time
import threading

import pytest

from utils.logging_config import (
    StructuredFormatter,
    new_correlation_id,
    get_correlation_id,
    correlation_id,
    timed,
)
from utils.metrics import MetricsCollector, Histogram, HistogramBucket


# ── Structured Logging Tests ────────────────────────────────────────────


class TestStructuredFormatter:
    """Tests for JSON structured log formatting."""

    def test_basic_format(self):
        formatter = StructuredFormatter(service_name="test-svc")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Hello world",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["message"] == "Hello world"
        assert parsed["level"] == "info"
        assert parsed["service"] == "test-svc"
        assert "timestamp" in parsed

    def test_error_level_mapping(self):
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.CRITICAL,
            pathname="",
            lineno=0,
            msg="Critical failure",
            args=(),
            exc_info=None,
        )
        parsed = json.loads(formatter.format(record))
        assert parsed["level"] == "fatal"

    def test_exception_included(self):
        formatter = StructuredFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg="Something broke",
                args=(),
                exc_info=sys.exc_info(),
            )
        parsed = json.loads(formatter.format(record))
        assert "error" in parsed
        assert parsed["error"]["type"] == "ValueError"
        assert "test error" in parsed["error"]["message"]

    def test_correlation_id_included(self):
        formatter = StructuredFormatter()
        token = correlation_id.set("abc123")
        try:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="traced",
                args=(),
                exc_info=None,
            )
            parsed = json.loads(formatter.format(record))
            assert parsed["correlation_id"] == "abc123"
        finally:
            correlation_id.reset(token)


class TestCorrelationId:
    """Tests for request correlation."""

    def test_new_correlation_id(self):
        cid = new_correlation_id()
        assert len(cid) == 32  # UUID hex
        assert cid == correlation_id.get()

    def test_get_creates_if_missing(self):
        token = correlation_id.set("")
        try:
            cid = get_correlation_id()
            assert len(cid) == 32
        finally:
            correlation_id.reset(token)

    def test_isolation_across_threads(self):
        """Verify correlation IDs are thread-local via contextvars."""
        results = {}

        def worker(name):
            cid = new_correlation_id()
            time.sleep(0.01)  # Simulate work
            results[name] = (cid, correlation_id.get())

        t1 = threading.Thread(target=worker, args=("a",))
        t2 = threading.Thread(target=worker, args=("b",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Each thread should have its own ID
        assert results["a"][0] == results["a"][1]
        assert results["b"][0] == results["b"][1]


class TestTimedDecorator:
    """Tests for the @timed performance decorator."""

    def test_timed_logs_duration(self, caplog):
        logger = logging.getLogger("test.timed")

        @timed(logger=logger, metric_name="test_op")
        def slow_op():
            time.sleep(0.05)
            return 42

        with caplog.at_level(logging.INFO, logger="test.timed"):
            result = slow_op()

        assert result == 42
        assert any("Completed" in r.message for r in caplog.records)

    def test_timed_logs_exceptions(self, caplog):
        logger = logging.getLogger("test.timed.err")

        @timed(logger=logger)
        def failing_op():
            raise RuntimeError("boom")

        with caplog.at_level(logging.ERROR, logger="test.timed.err"):
            with pytest.raises(RuntimeError, match="boom"):
                failing_op()

        assert any("Failed" in r.message for r in caplog.records)


# ── Metrics Tests ────────────────────────────────────────────────────────


class TestMetricsCollector:
    """Tests for the metrics collection system."""

    @pytest.fixture
    def collector(self):
        c = MetricsCollector()
        yield c
        c.reset()

    def test_counter_increment(self, collector):
        collector.increment("requests_total")
        collector.increment("requests_total")
        collector.increment("requests_total", value=3)
        snap = collector.snapshot()
        assert snap["counters"]["requests_total"] == 5

    def test_labeled_counter(self, collector):
        collector.increment("http_status", labels={"code": "200"})
        collector.increment("http_status", labels={"code": "200"})
        collector.increment("http_status", labels={"code": "500"})
        snap = collector.snapshot()
        labeled = snap["labeled_counters"]["http_status"]
        assert labeled['code="200"'] == 2
        assert labeled['code="500"'] == 1

    def test_gauge_set(self, collector):
        collector.set_gauge("temperature", 72.5)
        collector.set_gauge("temperature", 73.1)
        snap = collector.snapshot()
        assert snap["gauges"]["temperature"] == 73.1

    def test_histogram_observe(self, collector):
        for val in [10, 20, 50, 100, 500, 1000]:
            collector.observe_histogram("search_latency_ms", val)

        snap = collector.snapshot()
        hist = snap["histograms"]["search_latency_ms"]
        assert hist["count"] == 6
        assert hist["sum"] == 1680.0
        assert hist["avg"] == 280.0

    def test_histogram_buckets(self):
        hist = Histogram(buckets=(10, 50, 100))
        hist.observe(5)
        hist.observe(30)
        hist.observe(75)
        hist.observe(200)

        snap = hist.snapshot()
        assert snap["buckets"]["10"] == 1  # 5
        assert snap["buckets"]["50"] == 2  # 5, 30
        assert snap["buckets"]["100"] == 3  # 5, 30, 75
        assert snap["count"] == 4

    def test_prometheus_export(self, collector):
        collector.increment("test_counter", value=5)
        collector.set_gauge("test_gauge", 3.14)
        collector.observe_histogram("search_latency_ms", 42.0)

        prom = collector.to_prometheus()
        assert "test_counter 5" in prom
        assert "test_gauge 3.14" in prom
        assert "search_latency_ms_count 1" in prom
        assert "app_uptime_seconds" in prom

    def test_thread_safety(self, collector):
        """Verify counters are accurate under concurrent access."""
        errors = []

        def incrementer():
            try:
                for _ in range(1000):
                    collector.increment("concurrent_counter")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=incrementer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        snap = collector.snapshot()
        assert snap["counters"]["concurrent_counter"] == 10000

    def test_reset(self, collector):
        collector.increment("counter", value=10)
        collector.set_gauge("gauge", 5.0)
        collector.reset()

        snap = collector.snapshot()
        assert snap["counters"] == {}
        assert snap["gauges"] == {}

    def test_uptime_increases(self, collector):
        snap1 = collector.snapshot()
        time.sleep(0.1)
        snap2 = collector.snapshot()
        assert snap2["uptime_seconds"] > snap1["uptime_seconds"]


class TestHistogramBucket:
    """Tests for pre-defined bucket configurations."""

    def test_latency_buckets_ordered(self):
        buckets = HistogramBucket.LATENCY_MS
        assert buckets == tuple(sorted(buckets))
        assert buckets[0] > 0

    def test_result_count_buckets(self):
        buckets = HistogramBucket.RESULT_COUNT
        assert 0 in buckets
        assert max(buckets) >= 100
