"""
Tests for retrieval evaluation metrics and experiment tracking.
"""

import json
from pathlib import Path

import pytest

from utils.evaluation import RetrievalEvaluator, EvaluationReport
from utils.experiment_tracker import ExperimentTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def evaluator():
    return RetrievalEvaluator(k_values=[1, 3, 5, 10])


@pytest.fixture
def experiment_dir(tmp_path):
    d = tmp_path / "experiments"
    d.mkdir()
    return str(d)


@pytest.fixture
def tracker(experiment_dir):
    return ExperimentTracker(experiment_dir)


# ---------------------------------------------------------------------------
# Precision@K
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    def test_perfect_precision(self, evaluator):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert evaluator.precision_at_k(retrieved, relevant, 3) == 1.0

    def test_zero_precision(self, evaluator):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b", "c"}
        assert evaluator.precision_at_k(retrieved, relevant, 3) == 0.0

    def test_partial_precision(self, evaluator):
        retrieved = ["a", "x", "b", "y"]
        relevant = {"a", "b"}
        assert evaluator.precision_at_k(retrieved, relevant, 4) == 0.5

    def test_k_larger_than_results(self, evaluator):
        retrieved = ["a", "b"]
        relevant = {"a", "b"}
        # Only 2 docs retrieved, even though k=5
        assert evaluator.precision_at_k(retrieved, relevant, 5) == 1.0

    def test_k_zero(self, evaluator):
        assert evaluator.precision_at_k(["a"], {"a"}, 0) == 0.0

    def test_empty_retrieved(self, evaluator):
        assert evaluator.precision_at_k([], {"a", "b"}, 5) == 0.0


# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_perfect_recall(self, evaluator):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert evaluator.recall_at_k(retrieved, relevant, 3) == 1.0

    def test_partial_recall(self, evaluator):
        retrieved = ["a", "x", "y"]
        relevant = {"a", "b", "c"}
        assert evaluator.recall_at_k(retrieved, relevant, 3) == pytest.approx(1 / 3)

    def test_no_relevant_docs(self, evaluator):
        # If no docs are relevant, recall is 0 (avoid div by zero)
        assert evaluator.recall_at_k(["a", "b"], set(), 2) == 0.0

    def test_recall_increases_with_k(self, evaluator):
        retrieved = ["x", "a", "y", "b"]
        relevant = {"a", "b"}
        r2 = evaluator.recall_at_k(retrieved, relevant, 2)
        r4 = evaluator.recall_at_k(retrieved, relevant, 4)
        assert r4 >= r2


# ---------------------------------------------------------------------------
# Reciprocal Rank
# ---------------------------------------------------------------------------


class TestReciprocalRank:
    def test_first_position(self, evaluator):
        assert evaluator.reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_third_position(self, evaluator):
        assert evaluator.reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_not_found(self, evaluator):
        assert evaluator.reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0

    def test_multiple_relevant(self, evaluator):
        # RR only cares about the *first* relevant doc
        assert evaluator.reciprocal_rank(["x", "a", "b"], {"a", "b"}) == 0.5


# ---------------------------------------------------------------------------
# Average Precision
# ---------------------------------------------------------------------------


class TestAveragePrecision:
    def test_perfect_ranking(self, evaluator):
        # All relevant docs at the top
        retrieved = ["a", "b", "c", "x", "y"]
        relevant = {"a", "b", "c"}
        ap = evaluator.average_precision(retrieved, relevant)
        # AP = (1/1 + 2/2 + 3/3) / 3 = 1.0
        assert ap == pytest.approx(1.0)

    def test_worst_ranking(self, evaluator):
        # All relevant docs at the bottom
        retrieved = ["x", "y", "z", "a", "b"]
        relevant = {"a", "b"}
        ap = evaluator.average_precision(retrieved, relevant)
        # AP = (1/4 + 2/5) / 2 = 0.325
        assert ap == pytest.approx(0.325)

    def test_empty_relevant(self, evaluator):
        assert evaluator.average_precision(["a", "b"], set()) == 0.0

    def test_single_relevant(self, evaluator):
        retrieved = ["x", "a", "y"]
        relevant = {"a"}
        ap = evaluator.average_precision(retrieved, relevant)
        assert ap == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# NDCG@K
# ---------------------------------------------------------------------------


class TestNDCG:
    def test_perfect_binary(self, evaluator):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert evaluator.ndcg_at_k(retrieved, relevant, 3) == pytest.approx(1.0)

    def test_graded_relevance(self, evaluator):
        retrieved = ["a", "b", "c"]
        relevance = {"a": 3.0, "b": 2.0, "c": 1.0}
        # Perfect order → NDCG should be 1.0
        assert evaluator.ndcg_at_k(retrieved, relevance, 3) == pytest.approx(1.0)

    def test_reversed_graded(self, evaluator):
        retrieved = ["c", "b", "a"]
        relevance = {"a": 3.0, "b": 2.0, "c": 1.0}
        ndcg = evaluator.ndcg_at_k(retrieved, relevance, 3)
        # Reversed order → NDCG < 1.0
        assert 0.0 < ndcg < 1.0

    def test_no_relevant_in_top_k(self, evaluator):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert evaluator.ndcg_at_k(retrieved, relevant, 3) == 0.0

    def test_k_zero(self, evaluator):
        assert evaluator.ndcg_at_k(["a"], {"a"}, 0) == 0.0

    def test_empty_relevance(self, evaluator):
        assert evaluator.ndcg_at_k(["a", "b"], set(), 2) == 0.0


# ---------------------------------------------------------------------------
# Hit Rate
# ---------------------------------------------------------------------------


class TestHitRate:
    def test_hit(self, evaluator):
        assert evaluator.hit_at_k(["x", "a"], {"a"}, 2) is True

    def test_miss(self, evaluator):
        assert evaluator.hit_at_k(["x", "y"], {"a"}, 2) is False

    def test_hit_outside_k(self, evaluator):
        assert evaluator.hit_at_k(["x", "y", "a"], {"a"}, 2) is False


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------


class TestEvaluationPipeline:
    def test_single_query(self, evaluator):
        report = evaluator.evaluate(
            queries=["test query"],
            retrieved_ids=[["a", "x", "b", "y"]],
            relevant_ids=[{"a", "b"}],
        )

        assert isinstance(report, EvaluationReport)
        assert report.num_queries == 1
        assert report.mrr > 0
        assert report.map_score > 0
        assert len(report.per_query) == 1
        assert report.per_query[0].query == "test query"

    def test_multiple_queries(self, evaluator):
        report = evaluator.evaluate(
            queries=["q1", "q2", "q3"],
            retrieved_ids=[
                ["a", "b"],
                ["x", "c"],
                ["d", "y"],
            ],
            relevant_ids=[
                {"a"},
                {"c"},
                {"z"},  # not found
            ],
        )

        assert report.num_queries == 3
        assert report.hit_rate == pytest.approx(2 / 3)

    def test_mismatched_lengths_raises(self, evaluator):
        with pytest.raises(ValueError):
            evaluator.evaluate(
                queries=["q1"],
                retrieved_ids=[["a"], ["b"]],
                relevant_ids=[{"a"}],
            )

    def test_report_summary(self, evaluator):
        report = evaluator.evaluate(
            queries=["q1"],
            retrieved_ids=[["a"]],
            relevant_ids=[{"a"}],
        )
        summary = report.summary()
        assert "MRR" in summary
        assert "MAP" in summary

    def test_report_json_serializable(self, evaluator):
        report = evaluator.evaluate(
            queries=["q1"],
            retrieved_ids=[["a", "b"]],
            relevant_ids=[{"a"}],
        )
        data = json.loads(report.to_json())
        assert data["num_queries"] == 1
        assert "mrr" in data

    def test_graded_relevance_pipeline(self, evaluator):
        report = evaluator.evaluate(
            queries=["graded test"],
            retrieved_ids=[["a", "b", "c"]],
            relevant_ids=[{"a": 3.0, "b": 1.0, "c": 2.0}],
        )
        assert report.mrr == 1.0
        assert report.mean_ndcg_at_k[3] > 0


# ---------------------------------------------------------------------------
# Experiment Tracker
# ---------------------------------------------------------------------------


class TestExperimentTracker:
    def test_create_and_list_runs(self, tracker):
        with tracker.start_run("test_run") as run:
            run.log_params({"model": "bm25", "k": 10})
            run.log_metrics({"mrr": 0.75, "map": 0.68})

        runs = tracker.list_runs()
        assert len(runs) == 1
        assert runs[0].name == "test_run"
        assert runs[0].status == "completed"
        assert runs[0].metrics["mrr"] == 0.75

    def test_run_with_artifacts(self, tracker):
        with tracker.start_run("artifact_run") as run:
            path = run.log_artifact("test_data", {"key": "value"})
            assert Path(path).exists()

        runs = tracker.list_runs()
        assert len(runs[0].artifacts) == 1

    def test_run_failure(self, tracker):
        with pytest.raises(ValueError):
            with tracker.start_run("failing_run") as run:
                run.log_params({"should": "fail"})
                raise ValueError("Intentional failure")

        runs = tracker.list_runs()
        assert runs[0].status == "failed"

    def test_filter_by_name(self, tracker):
        with tracker.start_run("alpha_run") as run:
            run.log_metrics({"mrr": 0.5})

        with tracker.start_run("beta_run") as run:
            run.log_metrics({"mrr": 0.8})

        alpha_runs = tracker.list_runs(name_filter="alpha")
        assert len(alpha_runs) == 1
        assert alpha_runs[0].name == "alpha_run"

    def test_filter_by_tag(self, tracker):
        with tracker.start_run("tagged_run", tags=["benchmark"]) as run:
            run.log_metrics({"mrr": 0.6})

        with tracker.start_run("untagged_run") as run:
            run.log_metrics({"mrr": 0.7})

        tagged = tracker.list_runs(tag_filter="benchmark")
        assert len(tagged) == 1

    def test_compare_runs(self, tracker):
        with tracker.start_run("model_a") as run:
            run.log_metrics({"mrr": 0.6, "map": 0.5})

        with tracker.start_run("model_b") as run:
            run.log_metrics({"mrr": 0.8, "map": 0.7})

        comparison = tracker.compare(names=["model_a", "model_b"])
        assert "mrr" in comparison
        assert "*" in comparison  # best marker

    def test_best_run(self, tracker):
        with tracker.start_run("weak") as run:
            run.log_metrics({"mrr": 0.4})

        with tracker.start_run("strong") as run:
            run.log_metrics({"mrr": 0.9})

        best = tracker.best_run("mrr")
        assert best is not None
        assert best.metrics["mrr"] == 0.9

    def test_delete_run(self, tracker):
        with tracker.start_run("deletable") as run:
            run.log_artifact("data", {"x": 1})

        runs = tracker.list_runs()
        assert len(runs) == 1

        tracker.delete_run(runs[0].run_id)
        assert len(tracker.list_runs()) == 0

    def test_get_run(self, tracker):
        with tracker.start_run("fetchable") as run:
            run.log_metrics({"mrr": 0.55})
            run_id = run.run_id

        fetched = tracker.get_run(run_id)
        assert fetched is not None
        assert fetched.metrics["mrr"] == 0.55

    def test_get_nonexistent_run(self, tracker):
        assert tracker.get_run("nonexistent_id_12345") is None

    def test_run_tags_and_notes(self, tracker):
        with tracker.start_run("annotated") as run:
            run.add_tag("production")
            run.add_tag("v2")
            run.set_notes("Testing new embedding model")

        rec = tracker.list_runs()[0]
        assert "production" in rec.tags
        assert "v2" in rec.tags
        assert "embedding" in rec.notes
