"""
Passage fusion module for RAG retrieval quality improvement.

Provides three fusion strategies that combine or re-rank retrieved passages
to improve recall and precision:

- ``SentenceWindowFusion`` – small-to-big retrieval (Parent Document pattern)
- ``MultiQueryFusion``     – expand query into variants, fuse with RRF
- ``HyDEFusion``           – Hypothetical Document Embeddings fusion

Usage::

    from utils.passage_fusion import get_fusion, MultiQueryFusion

    fusion = get_fusion("multi_query", max_variants=5)
    result = fusion.fuse("contract breach remedy", retrieve_fn=my_retriever)
    print(result.results)

    # Or use HyDE fusion:
    hyde = get_fusion("hyde", domain="legal")
    fused = hyde.fuse(query_results, hypothesis_results, alpha=0.3)
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from utils.query_expansion import SynonymExpander

__all__ = [
    "SentenceWindowFusion",
    "MultiQueryFusion",
    "MultiQueryFusionResult",
    "HyDEFusion",
    "get_fusion",
]

# ---------------------------------------------------------------------------
# Stop-words for keyword extraction (subset, no external deps)
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "that",
        "this",
        "it",
        "its",
        "as",
        "not",
        "no",
        "what",
        "which",
        "who",
        "how",
        "when",
        "where",
        "why",
        "about",
        "between",
        "into",
        "through",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using basic regex."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _rrf_fuse(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    weights: Optional[List[float]] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion over multiple ranked lists.

    Args:
        ranked_lists: Each inner list is a ranked list of result dicts.
            Each dict must have an ``"id"`` key.
        k: RRF constant (default 60).
        weights: Optional per-list weight multipliers.  If ``None``, all
            lists are weighted equally at ``1.0``.
        top_k: Number of results to return.

    Returns:
        Top-k results sorted by fused RRF score (descending).  Each dict
        is augmented with an ``"rrf_score"`` key.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: Dict[str, float] = {}
    docs: Dict[str, Dict[str, Any]] = {}

    for weight, ranked in zip(weights, ranked_lists):
        for rank, doc in enumerate(ranked, start=1):
            doc_id = doc.get("id", str(uuid.uuid4()))
            rrf_score = weight / (k + rank)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score
            if doc_id not in docs:
                docs[doc_id] = dict(doc)

    sorted_ids = sorted(scores, key=lambda d: scores[d], reverse=True)[:top_k]
    results = []
    for doc_id in sorted_ids:
        entry = dict(docs[doc_id])
        entry["rrf_score"] = scores[doc_id]
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class MultiQueryFusionResult:
    """Result container for :class:`MultiQueryFusion`.

    Attributes:
        query: Original query.
        variants: List of generated query variants.
        results: Fused result list (sorted by RRF score).
        variant_hits: Mapping of variant → number of results it contributed.
    """

    query: str
    variants: List[str]
    results: List[Dict[str, Any]]
    variant_hits: Dict[str, int]


# ---------------------------------------------------------------------------
# SentenceWindowFusion
# ---------------------------------------------------------------------------


class SentenceWindowFusion:
    """Small-to-big retrieval (Parent Document Retriever pattern).

    Stores original large chunks (*parents*) alongside smaller sub-chunks
    (*children*).  At retrieval time, child matches are mapped back to their
    parent documents for richer context.

    Args:
        window_size: Number of sentences per child window.  Default ``3``.
        parent_overlap: If ``True`` (default), consecutive windows share
            overlapping sentences.

    Usage::

        fusion = SentenceWindowFusion(window_size=3)
        fusion.index([
            {"id": "doc1", "text": "Sentence one. Sentence two. Sentence three. Sentence four."},
        ])
        results = fusion.fuse([{"id": "doc1_w0", "score": 0.9}], top_k=5)
    """

    def __init__(
        self,
        window_size: int = 3,
        parent_overlap: bool = True,
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self.window_size = window_size
        self.parent_overlap = parent_overlap
        self._parents: Dict[str, Dict[str, Any]] = {}
        self._children: Dict[str, Dict[str, Any]] = {}
        self._child_to_parent: Dict[str, str] = {}

    def index(self, chunks: List[Dict[str, Any]]) -> None:
        """Index parent chunks and generate child windows.

        Args:
            chunks: List of parent chunk dicts.  Each must contain at
                least ``"id"`` and ``"text"`` keys.

        Raises:
            ValueError: If a chunk is missing ``"id"`` or ``"text"``.
        """
        for chunk in chunks:
            if "id" not in chunk or "text" not in chunk:
                raise ValueError("Each chunk must have 'id' and 'text' keys")

            parent_id = chunk["id"]
            self._parents[parent_id] = dict(chunk)

            sentences = _split_sentences(chunk["text"])
            if not sentences:
                continue

            step = 1 if self.parent_overlap else self.window_size
            for i in range(0, len(sentences), step):
                window = sentences[i : i + self.window_size]
                if not window:
                    continue
                child_id = f"{parent_id}_w{i}"
                child = {
                    "id": child_id,
                    "text": " ".join(window),
                    "parent_id": parent_id,
                    "window_start": i,
                }
                self._children[child_id] = child
                self._child_to_parent[child_id] = parent_id

    def fuse(
        self,
        retrieved_children: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Map child results back to deduplicated parent documents.

        Args:
            retrieved_children: Child chunk results, each with ``"id"``
                and ``"score"`` keys.
            top_k: Maximum parents to return.

        Returns:
            Parent documents sorted by aggregated child score (sum),
            each augmented with ``"aggregated_score"`` and
            ``"child_count"`` keys.
        """
        parent_scores: Dict[str, float] = {}
        parent_child_counts: Dict[str, int] = {}

        for child in retrieved_children:
            child_id = child.get("id", "")
            score = child.get("score", 0.0)
            parent_id = self._child_to_parent.get(child_id)
            if parent_id is None:
                continue
            parent_scores[parent_id] = parent_scores.get(parent_id, 0.0) + score
            parent_child_counts[parent_id] = parent_child_counts.get(parent_id, 0) + 1

        sorted_parents = sorted(
            parent_scores, key=lambda pid: parent_scores[pid], reverse=True
        )[:top_k]

        results = []
        for pid in sorted_parents:
            entry = dict(self._parents[pid])
            entry["aggregated_score"] = parent_scores[pid]
            entry["child_count"] = parent_child_counts[pid]
            results.append(entry)
        return results

    @property
    def parent_count(self) -> int:
        """Number of indexed parent documents."""
        return len(self._parents)

    @property
    def child_count(self) -> int:
        """Number of generated child windows."""
        return len(self._children)


# ---------------------------------------------------------------------------
# MultiQueryFusion
# ---------------------------------------------------------------------------


class MultiQueryFusion:
    """Expand a query into N variants and fuse retrievals with RRF.

    Generates query variants via simple perturbations (keywords-only,
    reversed word order, question prefix, synonym expansion) and fuses
    the results from each variant using Reciprocal Rank Fusion.

    Args:
        max_variants: Maximum number of query variants (including the
            original).  Default ``5``.
        rrf_k: RRF constant.  Default ``60``.

    Usage::

        fusion = MultiQueryFusion(max_variants=5)
        result = fusion.fuse("breach of contract", retrieve_fn=my_retriever)
        print(result.variant_hits)
    """

    def __init__(
        self,
        max_variants: int = 5,
        rrf_k: int = 60,
    ) -> None:
        if max_variants < 1:
            raise ValueError(f"max_variants must be >= 1, got {max_variants}")
        self.max_variants = max_variants
        self.rrf_k = rrf_k
        self._synonym_expander = SynonymExpander(max_synonyms_per_term=1)

    def generate_variants(self, query: str) -> List[str]:
        """Produce query variants from the original query.

        Variants generated (deduplicated, up to ``max_variants``):

        1. Original query
        2. Keywords-only (stop-words removed)
        3. Reversed word order
        4. With question prefix (``"What is ..."`` or ``"How does ..."``)
        5. Synonym-expanded version

        Args:
            query: Original query string.

        Returns:
            List of unique query variants (always includes the original).
        """
        variants: List[str] = [query]
        seen = {query.strip().lower()}

        def _add(v: str) -> None:
            normalised = v.strip().lower()
            if normalised and normalised not in seen:
                seen.add(normalised)
                variants.append(v.strip())

        # Keywords-only: remove stop-words
        words = query.split()
        keywords = [w for w in words if w.lower() not in _STOP_WORDS]
        if keywords:
            _add(" ".join(keywords))

        # Reversed word order
        reversed_words = list(reversed(words))
        _add(" ".join(reversed_words))

        # Question prefix
        first_word = words[0].lower() if words else ""
        if first_word in ("what", "how", "why", "when", "where", "who"):
            _add(query)  # already a question, won't add duplicate
        else:
            _add(f"What is {query}")
            _add(f"How does {query}")

        # Synonym-expanded
        expanded = self._synonym_expander.expand(query)
        if expanded.expanded_query != query:
            _add(expanded.expanded_query)

        return variants[: self.max_variants]

    def fuse(
        self,
        query: str,
        retrieve_fn: Callable[[str], List[Dict[str, Any]]],
        top_k: int = 10,
    ) -> MultiQueryFusionResult:
        """Retrieve for each variant and fuse results with RRF.

        Args:
            query: Original query.
            retrieve_fn: Callable that takes a query string and returns
                a ranked list of result dicts (each must have ``"id"``).
            top_k: Number of fused results to return.

        Returns:
            :class:`MultiQueryFusionResult` with fused results and
            per-variant hit counts.
        """
        variants = self.generate_variants(query)
        ranked_lists: List[List[Dict[str, Any]]] = []
        variant_hits: Dict[str, int] = {}

        for variant in variants:
            results = retrieve_fn(variant)
            ranked_lists.append(results)
            variant_hits[variant] = len(results)

        fused = _rrf_fuse(ranked_lists, k=self.rrf_k, top_k=top_k)
        return MultiQueryFusionResult(
            query=query,
            variants=variants,
            results=fused,
            variant_hits=variant_hits,
        )


# ---------------------------------------------------------------------------
# HyDEFusion
# ---------------------------------------------------------------------------


class HyDEFusion:
    """Hypothetical Document Embeddings fusion.

    Generates a synthetic passage from the query using templates (no LLM
    required), then blends rankings from the real query and the hypothesis
    using weighted RRF.

    Args:
        domain: Domain keyword inserted into the hypothesis template.
            Default ``"legal"``.

    Usage::

        hyde = HyDEFusion(domain="legal")
        hypothesis = hyde.generate_hypothesis("breach of contract")
        fused = hyde.fuse(query_results, hypo_results, alpha=0.3)
    """

    def __init__(self, domain: str = "legal") -> None:
        self.domain = domain

    def generate_hypothesis(self, query: str) -> str:
        """Create a plausible synthetic passage for the query.

        Uses template-based generation (deterministic, no LLM).

        Args:
            query: Original query string.

        Returns:
            Synthetic passage containing the query terms and domain.
        """
        return (
            f"In the context of {self.domain}, {query} refers to a situation "
            f"where the relevant principles and regulations apply. "
            f"The key aspects include the interpretation of {query} under "
            f"established {self.domain} frameworks, the applicable standards, "
            f"and the procedural requirements that govern such matters."
        )

    def fuse(
        self,
        query_results: List[Dict[str, Any]],
        hypothesis_results: List[Dict[str, Any]],
        alpha: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Blend query and hypothesis result rankings with weighted RRF.

        Args:
            query_results: Results retrieved using the original query.
            hypothesis_results: Results retrieved using the hypothesis.
            alpha: Weight for hypothesis results.  Query results receive
                weight ``1 - alpha``.  Default ``0.3``.

        Returns:
            Fused results sorted by weighted RRF score.

        Raises:
            ValueError: If *alpha* is not in ``[0, 1]``.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        if not query_results and not hypothesis_results:
            return []

        weights = [1.0 - alpha, alpha]
        ranked_lists = [query_results, hypothesis_results]
        return _rrf_fuse(
            ranked_lists,
            weights=weights,
            top_k=max(len(query_results), len(hypothesis_results), 10),
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_FUSION_REGISTRY: Dict[str, type] = {
    "sentence_window": SentenceWindowFusion,
    "multi_query": MultiQueryFusion,
    "hyde": HyDEFusion,
}


def get_fusion(method: str = "multi_query", **kwargs: Any) -> Any:
    """Instantiate a fusion strategy by name.

    Args:
        method: One of ``"sentence_window"``, ``"multi_query"``, or
            ``"hyde"``.
        **kwargs: Forwarded to the fusion class constructor.

    Returns:
        Configured fusion instance.

    Raises:
        ValueError: If *method* is not recognised.

    Usage::

        fusion = get_fusion("multi_query", max_variants=3)
        hyde = get_fusion("hyde", domain="medical")
    """
    if method not in _FUSION_REGISTRY:
        raise ValueError(
            f"Unknown fusion method '{method}'. Choose from: {sorted(_FUSION_REGISTRY)}"
        )
    cls = _FUSION_REGISTRY[method]
    return cls(**kwargs)
