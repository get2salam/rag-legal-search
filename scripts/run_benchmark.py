"""
Benchmark script for evaluating retrieval configurations.

Runs a suite of test queries against the search engine with different
settings and logs results via the experiment tracker.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --config benchmarks/config.json
    python scripts/run_benchmark.py --quick   # subset of queries
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.evaluation import RetrievalEvaluator, EvaluationReport
from utils.experiment_tracker import ExperimentTracker


# ---------------------------------------------------------------------------
# Sample benchmark dataset (real projects would load from a curated file)
# ---------------------------------------------------------------------------

SAMPLE_BENCHMARK = {
    "name": "legal_retrieval_v1",
    "description": "Baseline benchmark for legal case retrieval quality",
    "queries": [
        {
            "query": "breach of contract remedies and damages",
            "relevant_ids": [
                "contract_001",
                "contract_015",
                "contract_022",
                "remedy_003",
            ],
        },
        {
            "query": "wrongful termination in employment",
            "relevant_ids": ["employment_005", "employment_012", "employment_031"],
        },
        {
            "query": "intellectual property software copyright",
            "relevant_ids": ["ip_002", "ip_019", "ip_044", "ip_007", "copyright_001"],
        },
        {
            "query": "constitutional right to fair trial",
            "relevant_ids": ["const_001", "const_033", "human_rights_012"],
        },
        {
            "query": "landlord tenant eviction procedure",
            "relevant_ids": [
                "property_008",
                "property_021",
                "tenant_005",
                "eviction_001",
            ],
        },
        {
            "query": "corporate merger shareholder approval",
            "relevant_ids": ["corporate_011", "corporate_034", "merger_002"],
        },
        {
            "query": "negligence duty of care standard",
            "relevant_ids": [
                "tort_001",
                "tort_023",
                "negligence_005",
                "negligence_019",
            ],
        },
        {
            "query": "criminal sentencing guidelines appeal",
            "relevant_ids": ["criminal_007", "criminal_044", "sentencing_001"],
        },
        {
            "query": "data protection privacy GDPR compliance",
            "relevant_ids": ["privacy_002", "gdpr_011", "data_protection_008"],
        },
        {
            "query": "arbitration clause enforceability international",
            "relevant_ids": ["arbitration_001", "arbitration_015", "international_003"],
        },
    ],
}


def load_benchmark(config_path: str | None = None) -> dict:
    """Load benchmark config from file or use built-in sample."""
    if config_path and Path(config_path).exists():
        return json.loads(Path(config_path).read_text(encoding="utf-8"))
    return SAMPLE_BENCHMARK


def simulate_retrieval(
    query: str,
    relevant_ids: list[str],
    noise_docs: int = 7,
    hit_rate: float = 0.6,
) -> list[str]:
    """
    Simulate a retrieval run for benchmarking without a live index.

    In production, this would call your actual retriever. Here we
    create a synthetic ranking that includes some relevant docs
    mixed with noise, allowing the metrics to produce realistic
    (non-trivial) numbers for demonstration.
    """
    import random

    rng = random.Random(hash(query))
    retrieved: list[str] = []

    # Decide which relevant docs appear in the result
    for doc_id in relevant_ids:
        if rng.random() < hit_rate:
            retrieved.append(doc_id)

    # Add noise documents
    for i in range(noise_docs):
        retrieved.append(f"noise_{hash(query) % 1000}_{i:03d}")

    # Shuffle to simulate imperfect ranking (but bias relevant ones toward top)
    rng.shuffle(retrieved)

    # Boost: move one relevant doc to position 1 with 50 % chance
    for i, doc in enumerate(retrieved):
        if doc in relevant_ids and rng.random() < 0.5:
            retrieved.insert(0, retrieved.pop(i))
            break

    return retrieved[:10]


def run_benchmark(
    config_path: str | None = None,
    quick: bool = False,
    experiment_dir: str = "experiments",
) -> EvaluationReport:
    """Execute benchmark and return evaluation report."""
    benchmark = load_benchmark(config_path)
    queries_data = benchmark["queries"]

    if quick:
        queries_data = queries_data[:3]

    print(f"📊 Running benchmark: {benchmark['name']}")
    print(f"   Queries: {len(queries_data)}")
    print()

    evaluator = RetrievalEvaluator(k_values=[1, 3, 5, 10])
    tracker = ExperimentTracker(experiment_dir)

    queries = [q["query"] for q in queries_data]
    relevant = [set(q["relevant_ids"]) for q in queries_data]

    # Simulate retrieval (replace with real retriever in production)
    retrieved = [
        simulate_retrieval(q["query"], q["relevant_ids"]) for q in queries_data
    ]

    # Evaluate
    report = evaluator.evaluate(queries, retrieved, relevant)

    # Log experiment
    with tracker.start_run(benchmark["name"], tags=["benchmark"]) as run:
        run.log_params(
            {
                "benchmark": benchmark["name"],
                "num_queries": len(queries),
                "k_values": [1, 3, 5, 10],
                "retriever": "simulated",
                "quick_mode": quick,
            }
        )

        run.log_metrics(
            {
                "mrr": report.mrr,
                "map": report.map_score,
                "hit_rate": report.hit_rate,
            }
        )

        for k in report.k_values:
            run.log_metric(f"precision@{k}", report.mean_precision_at_k.get(k, 0))
            run.log_metric(f"recall@{k}", report.mean_recall_at_k.get(k, 0))
            run.log_metric(f"ndcg@{k}", report.mean_ndcg_at_k.get(k, 0))

        # Save full report as artifact
        run.log_artifact("evaluation_report.json", report.to_dict())
        run.set_notes(f"Benchmark run with {len(queries)} queries")

    # Print summary
    print(report.summary())
    print()
    print(f"✅ Results saved to {experiment_dir}/")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval benchmark")
    parser.add_argument("--config", help="Path to benchmark config JSON")
    parser.add_argument("--quick", action="store_true", help="Run quick subset")
    parser.add_argument(
        "--experiment-dir", default="experiments", help="Experiment directory"
    )

    args = parser.parse_args()
    run_benchmark(args.config, args.quick, args.experiment_dir)
