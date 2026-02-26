"""
Result re-ranking module for RAG legal search.

Provides multiple re-ranking strategies:
- TFIDFReranker: term-overlap scoring using TF-IDF weighted features
- ScoreReranker: blends original vector similarity with lexical overlap
- MMRReranker: Maximal Marginal Relevance for diversity-aware reranking

Usage::

    from utils.reranker import MMRReranker

    reranker = MMRReranker(lambda_param=0.6)
    reranked = reranker.rerank(query="breach of contract", results=results, top_k=5)
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
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
    }
)


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stop words."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def _term_freq(tokens: List[str]) -> Counter:
    return Counter(tokens)


def _idf(term: str, doc_token_lists: List[List[str]]) -> float:
    """Compute IDF for *term* across *doc_token_lists*."""
    n_docs = len(doc_token_lists)
    df = sum(1 for tl in doc_token_lists if term in tl)
    if df == 0:
        return 0.0
    return math.log((n_docs + 1) / (df + 1)) + 1.0


def _tfidf_vector(
    tokens: List[str],
    vocab: List[str],
    idf_scores: Dict[str, float],
) -> List[float]:
    """Return a TF-IDF vector over *vocab*."""
    tf = _term_freq(tokens)
    n = len(tokens) or 1
    return [(tf.get(t, 0) / n) * idf_scores.get(t, 0.0) for t in vocab]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length float lists."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------


class BaseReranker(ABC):
    """Abstract base class for all re-rankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """Re-rank *results* for *query* and return the top-k slice.

        Args:
            query: Original user query.
            results: List of result dicts, each containing at least ``score``
                     and at least one text field (``excerpt`` or ``summary``).
            top_k: Maximum number of results to return (``None`` → all).

        Returns:
            Re-ranked list of result dicts, each augmented with
            ``rerank_score`` and ``rerank_method`` fields.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_text(result: Dict) -> str:
        """Extract the best available text representation from a result."""
        parts = []
        if result.get("title"):
            parts.append(result["title"])
        if result.get("summary"):
            parts.append(result["summary"])
        if result.get("excerpt"):
            parts.append(result["excerpt"][:500])
        return " ".join(parts)

    @staticmethod
    def _slice(results: List[Dict], top_k: Optional[int]) -> List[Dict]:
        if top_k is None:
            return results
        return results[:top_k]


# ---------------------------------------------------------------------------
# TFIDFReranker
# ---------------------------------------------------------------------------


class TFIDFReranker(BaseReranker):
    """Re-rank results using TF-IDF cosine similarity between query and documents.

    Does **not** use the original vector-store score; ranking is purely based
    on lexical TF-IDF overlap.  Best used as a lightweight baseline or when
    keyword precision matters more than semantic coverage.

    Args:
        min_score: Results whose TF-IDF score falls below this threshold are
                   moved to the end of the list (still returned, not dropped).
    """

    def __init__(self, min_score: float = 0.0) -> None:
        self.min_score = min_score

    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        if not results:
            return []

        query_tokens = _tokenize(query)
        doc_token_lists = [_tokenize(self._get_text(r)) for r in results]

        # Build shared vocabulary
        vocab = sorted({t for tl in [query_tokens] + doc_token_lists for t in tl})

        # Compute IDF over all documents (query excluded from IDF corpus)
        idf_scores = {t: _idf(t, doc_token_lists) for t in vocab}

        query_vec = _tfidf_vector(query_tokens, vocab, idf_scores)

        scored: List[Dict] = []
        for result, doc_tokens in zip(results, doc_token_lists):
            doc_vec = _tfidf_vector(doc_tokens, vocab, idf_scores)
            sim = _cosine_similarity(query_vec, doc_vec)
            entry = dict(result)
            entry["rerank_score"] = round(sim, 6)
            entry["rerank_method"] = "tfidf"
            scored.append(entry)

        scored.sort(key=lambda r: r["rerank_score"], reverse=True)
        return self._slice(scored, top_k)


# ---------------------------------------------------------------------------
# ScoreReranker
# ---------------------------------------------------------------------------


class ScoreReranker(BaseReranker):
    """Blend original vector-store similarity with TF-IDF lexical overlap.

    The final re-rank score is::

        rerank_score = alpha * vector_score + (1 - alpha) * tfidf_score

    This balances semantic recall from embeddings against keyword precision
    from lexical matching — a common hybrid technique in production RAG systems.

    Args:
        alpha: Weight given to the original vector score (0 = pure TF-IDF,
               1 = no re-ranking, 0.5 = equal blend).  Default is ``0.6``.
        score_field: Name of the field containing the original similarity
                     score.  Default is ``"score"``.
    """

    def __init__(self, alpha: float = 0.6, score_field: str = "score") -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.score_field = score_field

    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        if not results:
            return []

        # Compute TF-IDF scores in positional order (same indices as results)
        query_tokens = _tokenize(query)
        doc_token_lists = [_tokenize(self._get_text(r)) for r in results]
        vocab = sorted({t for tl in [query_tokens] + doc_token_lists for t in tl})
        idf_scores = {t: _idf(t, doc_token_lists) for t in vocab}
        query_vec = _tfidf_vector(query_tokens, vocab, idf_scores)
        tfidf_scores = [
            _cosine_similarity(query_vec, _tfidf_vector(tl, vocab, idf_scores))
            for tl in doc_token_lists
        ]

        # Normalise original scores to [0, 1]
        raw_scores = [r.get(self.score_field, 0.0) for r in results]
        max_score = max(raw_scores) or 1.0
        min_score = min(raw_scores)
        score_range = max_score - min_score or 1.0

        blended: List[Dict] = []
        for i, result in enumerate(results):
            norm_vec = (result.get(self.score_field, 0.0) - min_score) / score_range
            tfidf_score = tfidf_scores[i]
            blended_score = self.alpha * norm_vec + (1 - self.alpha) * tfidf_score
            entry = dict(result)
            entry["rerank_score"] = round(blended_score, 6)
            entry["rerank_method"] = f"score_blend(alpha={self.alpha})"
            blended.append(entry)

        blended.sort(key=lambda r: r["rerank_score"], reverse=True)
        return self._slice(blended, top_k)


# ---------------------------------------------------------------------------
# MMRReranker
# ---------------------------------------------------------------------------


class MMRReranker(BaseReranker):
    """Maximal Marginal Relevance (MMR) re-ranker for diverse result sets.

    MMR iteratively selects the result that maximises::

        MMR = lambda_param * Sim(doc, query) - (1 - lambda_param) * max_j Sim(doc, d_j)

    where ``d_j`` ranges over already-selected documents.  This balances
    relevance against redundancy — useful when the top results from a vector
    store are near-duplicate chunks.

    Args:
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0).
                      Default is ``0.5``.
        score_field: Original similarity score field name.  Default ``"score"``.
    """

    def __init__(
        self,
        lambda_param: float = 0.5,
        score_field: str = "score",
    ) -> None:
        if not 0.0 <= lambda_param <= 1.0:
            raise ValueError(f"lambda_param must be in [0, 1], got {lambda_param}")
        self.lambda_param = lambda_param
        self.score_field = score_field

    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        if not results:
            return []

        n = len(results)
        k = min(top_k, n) if top_k is not None else n

        # Build TF-IDF vectors for query + all docs
        query_tokens = _tokenize(query)
        doc_token_lists = [_tokenize(self._get_text(r)) for r in results]

        vocab = sorted({t for tl in [query_tokens] + doc_token_lists for t in tl})
        idf_scores = {t: _idf(t, doc_token_lists) for t in vocab}

        query_vec = _tfidf_vector(query_tokens, vocab, idf_scores)
        doc_vecs = [_tfidf_vector(tl, vocab, idf_scores) for tl in doc_token_lists]

        # Relevance of each doc to the query
        query_sims = [_cosine_similarity(query_vec, dv) for dv in doc_vecs]

        selected_indices: List[int] = []
        candidate_indices = list(range(n))

        for _ in range(k):
            if not candidate_indices:
                break

            best_idx: Optional[int] = None
            best_mmr: float = float("-inf")

            for idx in candidate_indices:
                rel = self.lambda_param * query_sims[idx]

                if selected_indices:
                    max_sim_to_selected = max(
                        _cosine_similarity(doc_vecs[idx], doc_vecs[s])
                        for s in selected_indices
                    )
                    red = (1 - self.lambda_param) * max_sim_to_selected
                else:
                    red = 0.0

                mmr = rel - red
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            assert best_idx is not None  # noqa: S101 — only fails if candidate list empty
            selected_indices.append(best_idx)
            candidate_indices.remove(best_idx)

        reranked: List[Dict] = []
        for rank, idx in enumerate(selected_indices):
            entry = dict(results[idx])
            entry["rerank_score"] = round(query_sims[idx], 6)
            entry["rerank_method"] = f"mmr(lambda={self.lambda_param})"
            entry["mmr_rank"] = rank + 1
            reranked.append(entry)

        return reranked


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_RERANKER_REGISTRY: Dict[str, type] = {
    "tfidf": TFIDFReranker,
    "score": ScoreReranker,
    "mmr": MMRReranker,
}


def get_reranker(method: str = "mmr", **kwargs) -> BaseReranker:
    """Instantiate a re-ranker by name.

    Args:
        method: One of ``"tfidf"``, ``"score"``, or ``"mmr"``.
        **kwargs: Forwarded to the reranker constructor.

    Returns:
        Configured :class:`BaseReranker` instance.

    Raises:
        ValueError: If *method* is not recognised.
    """
    if method not in _RERANKER_REGISTRY:
        raise ValueError(
            f"Unknown reranker '{method}'. Choose from: {sorted(_RERANKER_REGISTRY)}"
        )
    return _RERANKER_REGISTRY[method](**kwargs)
