"""
Retrieval evaluation metrics for RAG search quality assessment.

Implements standard IR metrics:
- Precision@K, Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Mean Average Precision (MAP)
- Hit Rate / Success@K

Usage:
    from utils.evaluation import RetrievalEvaluator

    evaluator = RetrievalEvaluator()
    results = evaluator.evaluate(
        queries=["breach of contract remedies"],
        retrieved_ids=[["doc_1", "doc_5", "doc_3"]],
        relevant_ids=[{"doc_1", "doc_3", "doc_7"}],
    )
    print(results.summary())
"""

import math
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Set, Dict, Optional, Sequence, Union
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a single query evaluation."""

    query: str
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    reciprocal_rank: float = 0.0
    average_precision: float = 0.0
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    hit: bool = False
    num_retrieved: int = 0
    num_relevant: int = 0
    num_relevant_retrieved: int = 0


@dataclass
class EvaluationReport:
    """Aggregate evaluation report across all queries."""

    timestamp: str = ""
    num_queries: int = 0
    k_values: List[int] = field(default_factory=list)
    mean_precision_at_k: Dict[int, float] = field(default_factory=dict)
    mean_recall_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    map_score: float = 0.0
    mean_ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    hit_rate: float = 0.0
    per_query: List[QueryMetrics] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary of evaluation results."""
        lines = [
            "=" * 60,
            "RETRIEVAL EVALUATION REPORT",
            f"Timestamp: {self.timestamp}",
            f"Queries evaluated: {self.num_queries}",
            "=" * 60,
            "",
            f"MRR:       {self.mrr:.4f}",
            f"MAP:       {self.map_score:.4f}",
            f"Hit Rate:  {self.hit_rate:.4f}",
            "",
        ]

        for k in self.k_values:
            lines.append(
                f"@{k}  — P: {self.mean_precision_at_k.get(k, 0):.4f}  "
                f"R: {self.mean_recall_at_k.get(k, 0):.4f}  "
                f"NDCG: {self.mean_ndcg_at_k.get(k, 0):.4f}"
            )

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serializable dictionary representation."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """JSON string representation."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


class RetrievalEvaluator:
    """
    Evaluates retrieval quality using standard IR metrics.

    Supports both binary relevance (set of relevant IDs) and
    graded relevance (dict mapping ID → relevance score).
    """

    def __init__(self, k_values: Optional[List[int]] = None):
        """
        Args:
            k_values: Cut-off values for @K metrics.
                      Defaults to [1, 3, 5, 10].
        """
        self.k_values = sorted(k_values or [1, 3, 5, 10])

    # ------------------------------------------------------------------
    # Core metric functions (static for reuse)
    # ------------------------------------------------------------------

    @staticmethod
    def precision_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
        """Fraction of top-k results that are relevant."""
        if k <= 0:
            return 0.0
        top_k = retrieved[:k]
        if not top_k:
            return 0.0
        return sum(1 for doc in top_k if doc in relevant) / len(top_k)

    @staticmethod
    def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
        """Fraction of relevant docs found in top-k."""
        if not relevant:
            return 0.0
        top_k = retrieved[:k]
        return sum(1 for doc in top_k if doc in relevant) / len(relevant)

    @staticmethod
    def reciprocal_rank(retrieved: Sequence[str], relevant: Set[str]) -> float:
        """1 / rank of the first relevant document (0 if none found)."""
        for i, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                return 1.0 / i
        return 0.0

    @staticmethod
    def average_precision(retrieved: Sequence[str], relevant: Set[str]) -> float:
        """
        Average of precision values at each relevant hit position.
        Standard MAP component.
        """
        if not relevant:
            return 0.0

        hits = 0
        sum_precision = 0.0

        for i, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                hits += 1
                sum_precision += hits / i

        return sum_precision / len(relevant) if relevant else 0.0

    @staticmethod
    def ndcg_at_k(
        retrieved: Sequence[str],
        relevance: Union[Set[str], Dict[str, float]],
        k: int,
    ) -> float:
        """
        Normalized Discounted Cumulative Gain at K.

        Supports both binary relevance (set) and graded relevance (dict).
        For binary: relevant = 1.0, non-relevant = 0.0.
        """
        if k <= 0:
            return 0.0

        # Normalise relevance to a score dict
        if isinstance(relevance, set):
            rel_scores: Dict[str, float] = {doc: 1.0 for doc in relevance}
        else:
            rel_scores = relevance

        def dcg(ranking: Sequence[str], limit: int) -> float:
            total = 0.0
            for i, doc in enumerate(ranking[:limit], start=1):
                gain = rel_scores.get(doc, 0.0)
                total += gain / math.log2(i + 1)
            return total

        # Ideal ranking: sort all relevant docs by descending score
        ideal_ranking = sorted(rel_scores.keys(), key=lambda d: rel_scores[d], reverse=True)
        ideal = dcg(ideal_ranking, k)

        if ideal == 0.0:
            return 0.0

        return dcg(retrieved, k) / ideal

    @staticmethod
    def hit_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> bool:
        """Whether any relevant document appears in the top-k."""
        return any(doc in relevant for doc in retrieved[:k])

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate_query(
        self,
        query: str,
        retrieved: Sequence[str],
        relevant: Union[Set[str], Dict[str, float]],
    ) -> QueryMetrics:
        """Compute all metrics for a single query."""
        relevant_set = set(relevant) if isinstance(relevant, dict) else relevant

        metrics = QueryMetrics(
            query=query,
            reciprocal_rank=self.reciprocal_rank(retrieved, relevant_set),
            average_precision=self.average_precision(retrieved, relevant_set),
            hit=self.hit_at_k(retrieved, relevant_set, max(self.k_values)),
            num_retrieved=len(retrieved),
            num_relevant=len(relevant_set),
            num_relevant_retrieved=sum(1 for d in retrieved if d in relevant_set),
        )

        for k in self.k_values:
            metrics.precision_at_k[k] = self.precision_at_k(retrieved, relevant_set, k)
            metrics.recall_at_k[k] = self.recall_at_k(retrieved, relevant_set, k)
            metrics.ndcg_at_k[k] = self.ndcg_at_k(retrieved, relevant, k)

        return metrics

    def evaluate(
        self,
        queries: List[str],
        retrieved_ids: List[Sequence[str]],
        relevant_ids: List[Union[Set[str], Dict[str, float]]],
    ) -> EvaluationReport:
        """
        Evaluate retrieval quality across multiple queries.

        Args:
            queries: List of query strings.
            retrieved_ids: For each query, ordered list of retrieved doc IDs.
            relevant_ids: For each query, set of relevant IDs (binary)
                          or dict of ID → relevance score (graded).

        Returns:
            EvaluationReport with aggregate and per-query metrics.
        """
        if not (len(queries) == len(retrieved_ids) == len(relevant_ids)):
            raise ValueError(
                "queries, retrieved_ids, and relevant_ids must have the same length"
            )

        per_query: List[QueryMetrics] = []

        for q, ret, rel in zip(queries, retrieved_ids, relevant_ids):
            qm = self.evaluate_query(q, ret, rel)
            per_query.append(qm)

        # Aggregate
        n = len(per_query)
        report = EvaluationReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            num_queries=n,
            k_values=self.k_values,
            mrr=sum(qm.reciprocal_rank for qm in per_query) / n if n else 0.0,
            map_score=sum(qm.average_precision for qm in per_query) / n if n else 0.0,
            hit_rate=sum(1 for qm in per_query if qm.hit) / n if n else 0.0,
            per_query=per_query,
        )

        for k in self.k_values:
            report.mean_precision_at_k[k] = (
                sum(qm.precision_at_k.get(k, 0) for qm in per_query) / n if n else 0.0
            )
            report.mean_recall_at_k[k] = (
                sum(qm.recall_at_k.get(k, 0) for qm in per_query) / n if n else 0.0
            )
            report.mean_ndcg_at_k[k] = (
                sum(qm.ndcg_at_k.get(k, 0) for qm in per_query) / n if n else 0.0
            )

        logger.info(
            "Evaluation complete: %d queries, MRR=%.4f, MAP=%.4f",
            n,
            report.mrr,
            report.map_score,
        )

        return report
