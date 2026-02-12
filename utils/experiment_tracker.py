"""
Lightweight experiment tracker for RAG retrieval experiments.

Logs experiment runs (parameters, metrics, artifacts) to JSON files
for comparison and reproducibility — no external service required.

Usage:
    from utils.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker("experiments/")

    with tracker.start_run("baseline_bm25") as run:
        run.log_params({"model": "BM25", "k": 10})
        run.log_metrics({"mrr": 0.72, "map": 0.65})
        run.log_artifact("report.json", report_data)

    # Compare runs
    tracker.compare(["baseline_bm25", "hybrid_v1"])
"""

import json
import os
import time
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    """Immutable record of a single experiment run."""

    run_id: str
    name: str
    status: str = "created"  # created | running | completed | failed
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    notes: str = ""


class Run:
    """Active experiment run — collects params, metrics, and artifacts."""

    def __init__(self, record: RunRecord, base_dir: Path):
        self._record = record
        self._base_dir = base_dir
        self._artifact_dir = base_dir / "artifacts" / record.run_id
        self._start_ts: float = 0.0

    @property
    def run_id(self) -> str:
        return self._record.run_id

    @property
    def name(self) -> str:
        return self._record.name

    @property
    def metrics(self) -> Dict[str, float]:
        return dict(self._record.metrics)

    # ---- logging helpers ------------------------------------------------

    def log_params(self, params: Dict[str, Any]) -> None:
        """Record hyper-parameters for this run."""
        self._record.params.update(params)
        logger.debug("Run %s: logged %d params", self.run_id, len(params))

    def log_param(self, key: str, value: Any) -> None:
        """Record a single parameter."""
        self._record.params[key] = value

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Record evaluation metrics."""
        self._record.metrics.update(metrics)
        logger.debug("Run %s: logged %d metrics", self.run_id, len(metrics))

    def log_metric(self, key: str, value: float) -> None:
        """Record a single metric."""
        self._record.metrics[key] = value

    def set_tags(self, tags: List[str]) -> None:
        """Replace run tags."""
        self._record.tags = list(tags)

    def add_tag(self, tag: str) -> None:
        """Add a tag to the run."""
        if tag not in self._record.tags:
            self._record.tags.append(tag)

    def set_notes(self, notes: str) -> None:
        """Set free-form notes."""
        self._record.notes = notes

    def log_artifact(self, name: str, data: Any) -> str:
        """
        Save an artifact (dict/list as JSON, str as text).

        Returns the path to the saved artifact.
        """
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        path = self._artifact_dir / name

        if isinstance(data, (dict, list)):
            path = path.with_suffix(".json") if not path.suffix else path
            path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        elif isinstance(data, str):
            path.write_text(data, encoding="utf-8")
        elif isinstance(data, bytes):
            path.write_bytes(data)
        else:
            path = path.with_suffix(".json")
            path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

        rel = str(path.relative_to(self._base_dir))
        self._record.artifacts.append(rel)
        logger.debug("Run %s: saved artifact %s", self.run_id, rel)
        return str(path)

    # ---- lifecycle (called by tracker) -----------------------------------

    def _begin(self) -> None:
        self._start_ts = time.monotonic()
        self._record.status = "running"
        self._record.start_time = datetime.now(timezone.utc).isoformat()

    def _end(self, status: str = "completed") -> None:
        self._record.status = status
        self._record.end_time = datetime.now(timezone.utc).isoformat()
        self._record.duration_seconds = round(time.monotonic() - self._start_ts, 3)


class ExperimentTracker:
    """
    File-based experiment tracker.

    Stores each run as ``<base_dir>/runs/<run_id>.json``.
    """

    def __init__(self, base_dir: str = "experiments"):
        self._base_dir = Path(base_dir)
        self._runs_dir = self._base_dir / "runs"
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ExperimentTracker initialised at %s", self._base_dir)

    # ---- run management --------------------------------------------------

    def _generate_id(self, name: str) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        safe = name.replace(" ", "_").replace("/", "_")[:40]
        return f"{safe}_{ts}"

    def _save_run(self, record: RunRecord) -> None:
        path = self._runs_dir / f"{record.run_id}.json"
        path.write_text(json.dumps(asdict(record), indent=2, default=str), encoding="utf-8")

    @contextmanager
    def start_run(self, name: str, tags: Optional[List[str]] = None):
        """
        Context-manager that creates, tracks, and persists a run.

        Example::

            with tracker.start_run("hybrid_search_v2") as run:
                run.log_params({"model": "bge-m3", "top_k": 10})
                run.log_metrics(evaluator.evaluate(...).to_dict())
        """
        run_id = self._generate_id(name)
        record = RunRecord(run_id=run_id, name=name, tags=tags or [])
        run = Run(record, self._base_dir)

        run._begin()
        logger.info("Started run: %s (%s)", name, run_id)

        try:
            yield run
            run._end("completed")
            logger.info(
                "Run completed: %s — %.1fs, %d metrics",
                run_id,
                record.duration_seconds,
                len(record.metrics),
            )
        except Exception:
            run._end("failed")
            logger.exception("Run failed: %s", run_id)
            raise
        finally:
            self._save_run(record)

    # ---- querying --------------------------------------------------------

    def list_runs(
        self,
        name_filter: Optional[str] = None,
        tag_filter: Optional[str] = None,
        status_filter: Optional[str] = None,
        limit: int = 50,
    ) -> List[RunRecord]:
        """List stored runs with optional filters, newest first."""
        records: List[RunRecord] = []

        for path in sorted(self._runs_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                rec = RunRecord(**{k: v for k, v in data.items() if k in RunRecord.__dataclass_fields__})
            except Exception:
                continue

            if name_filter and name_filter.lower() not in rec.name.lower():
                continue
            if tag_filter and tag_filter not in rec.tags:
                continue
            if status_filter and rec.status != status_filter:
                continue

            records.append(rec)
            if len(records) >= limit:
                break

        return records

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        """Load a specific run by ID."""
        path = self._runs_dir / f"{run_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return RunRecord(**{k: v for k, v in data.items() if k in RunRecord.__dataclass_fields__})

    def delete_run(self, run_id: str) -> bool:
        """Remove a run and its artifacts."""
        path = self._runs_dir / f"{run_id}.json"
        if not path.exists():
            return False
        path.unlink()

        artifact_dir = self._base_dir / "artifacts" / run_id
        if artifact_dir.exists():
            import shutil
            shutil.rmtree(artifact_dir)

        logger.info("Deleted run: %s", run_id)
        return True

    # ---- comparison ------------------------------------------------------

    def compare(
        self,
        run_ids: Optional[List[str]] = None,
        names: Optional[List[str]] = None,
        metric_keys: Optional[List[str]] = None,
    ) -> str:
        """
        Compare multiple runs side-by-side.

        Args:
            run_ids: Explicit run IDs to compare.
            names: If run_ids not given, pick the latest run matching each name.
            metric_keys: Specific metrics to compare (default: all).

        Returns:
            Formatted comparison table as a string.
        """
        records: List[RunRecord] = []

        if run_ids:
            for rid in run_ids:
                rec = self.get_run(rid)
                if rec:
                    records.append(rec)
        elif names:
            for name in names:
                runs = self.list_runs(name_filter=name, limit=1)
                if runs:
                    records.append(runs[0])

        if not records:
            return "No runs found for comparison."

        # Collect all metric keys
        all_keys: List[str] = []
        if metric_keys:
            all_keys = metric_keys
        else:
            seen = set()
            for rec in records:
                for k in rec.metrics:
                    if k not in seen:
                        all_keys.append(k)
                        seen.add(k)

        # Build table
        header = f"{'Metric':<25}" + "".join(f"{r.name[:20]:<22}" for r in records)
        separator = "-" * len(header)
        rows = [separator, header, separator]

        for key in all_keys:
            values = []
            best_val = None
            for rec in records:
                v = rec.metrics.get(key)
                if v is not None and (best_val is None or v > best_val):
                    best_val = v
                values.append(v)

            cells = []
            for v in values:
                if v is None:
                    cells.append(f"{'—':<22}")
                elif v == best_val and len(records) > 1:
                    cells.append(f"{v:<20.4f} *")
                else:
                    cells.append(f"{v:<22.4f}")

            rows.append(f"{key:<25}" + "".join(cells))

        rows.append(separator)
        rows.append("* = best")
        return "\n".join(rows)

    def best_run(self, metric: str, higher_is_better: bool = True) -> Optional[RunRecord]:
        """Find the run with the best value for a given metric."""
        runs = self.list_runs(status_filter="completed", limit=500)
        if not runs:
            return None

        best = None
        best_val = None

        for run in runs:
            val = run.metrics.get(metric)
            if val is None:
                continue
            if best_val is None:
                best, best_val = run, val
            elif higher_is_better and val > best_val:
                best, best_val = run, val
            elif not higher_is_better and val < best_val:
                best, best_val = run, val

        return best
