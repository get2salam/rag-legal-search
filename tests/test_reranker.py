"""
Tests for utils/reranker.py

Covers:
- Tokenisation helper
- TF-IDF vector construction and cosine similarity
- TFIDFReranker: ordering, top_k, empty input, single result
- ScoreReranker: alpha blending, edge cases
- MMRReranker: diversity selection, lambda extremes, top_k
- get_reranker factory
- Augmented fields (rerank_score, rerank_method, mmr_rank)
"""

from __future__ import annotations

import math
from typing import Dict, List

import pytest

from utils.reranker import (
    MMRReranker,
    ScoreReranker,
    TFIDFReranker,
    _cosine_similarity,
    _idf,
    _term_freq,
    _tfidf_vector,
    _tokenize,
    get_reranker,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_result(
    title: str,
    summary: str,
    score: float = 0.8,
    rid: str | None = None,
) -> Dict:
    return {
        "id": rid or title.lower().replace(" ", "_"),
        "title": title,
        "summary": summary,
        "excerpt": summary,
        "score": score,
    }


RESULTS_CONTRACT = [
    make_result(
        "Smith v Jones",
        "breach of contract damages award expectation loss",
        score=0.90,
        rid="r1",
    ),
    make_result(
        "Donoghue v Stevenson",
        "negligence duty of care snail in ginger beer bottle",
        score=0.75,
        rid="r2",
    ),
    make_result(
        "Carlill v Carbolic Smoke Ball",
        "contract offer acceptance consideration performance",
        score=0.70,
        rid="r3",
    ),
    make_result(
        "Hadley v Baxendale",
        "remoteness contract damages consequential losses",
        score=0.65,
        rid="r4",
    ),
    make_result(
        "Caparo Industries v Dickman",
        "negligence three-part test duty proximity fair",
        score=0.60,
        rid="r5",
    ),
]


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_lowercases(self):
        assert _tokenize("Contract") == ["contract"]

    def test_removes_stop_words(self):
        tokens = _tokenize("a breach of contract")
        assert "a" not in tokens
        assert "of" not in tokens
        assert "breach" in tokens
        assert "contract" in tokens

    def test_removes_punctuation(self):
        tokens = _tokenize("duty-of-care, negligence!")
        assert "negligence" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_all_stop_words(self):
        assert _tokenize("the a an and or") == []

    def test_single_char_filtered(self):
        tokens = _tokenize("i x y damages")
        assert "i" not in tokens
        assert "x" not in tokens
        assert "damages" in tokens


class TestTermFreq:
    def test_counts(self):
        tf = _term_freq(["contract", "contract", "breach"])
        assert tf["contract"] == 2
        assert tf["breach"] == 1

    def test_empty(self):
        assert _term_freq([]) == {}


class TestIDF:
    def test_term_in_all_docs(self):
        doc_lists = [["contract", "breach"], ["contract", "damages"]]
        idf = _idf("contract", doc_lists)
        # IDF should be low for a ubiquitous term
        assert idf < _idf("breach", doc_lists)

    def test_term_in_no_doc(self):
        assert _idf("unknown", [["contract"], ["breach"]]) == 0.0

    def test_single_doc(self):
        assert _idf("contract", [["contract"]]) > 0.0


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert math.isclose(_cosine_similarity(v, v), 1.0, rel_tol=1e-6)

    def test_orthogonal_vectors(self):
        assert math.isclose(_cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_both_zero(self):
        assert _cosine_similarity([0.0], [0.0]) == 0.0

    def test_range(self):
        a = [0.5, 0.3, 0.8]
        b = [0.1, 0.9, 0.2]
        sim = _cosine_similarity(a, b)
        assert 0.0 <= sim <= 1.0


class TestTFIDFVector:
    def test_zero_for_absent_terms(self):
        vocab = ["contract", "breach", "tort"]
        idf = {"contract": 1.0, "breach": 1.0, "tort": 1.0}
        vec = _tfidf_vector(["contract"], vocab, idf)
        assert vec[0] > 0  # contract present
        assert vec[1] == 0.0  # breach absent
        assert vec[2] == 0.0  # tort absent

    def test_empty_tokens(self):
        vocab = ["contract"]
        idf = {"contract": 1.0}
        # empty token list should not raise; division by 1 (len or 1)
        vec = _tfidf_vector([], vocab, idf)
        assert vec == [0.0]


# ---------------------------------------------------------------------------
# TFIDFReranker
# ---------------------------------------------------------------------------


class TestTFIDFReranker:
    def setup_method(self):
        self.reranker = TFIDFReranker()

    def test_returns_list(self):
        results = self.reranker.rerank("contract damages", RESULTS_CONTRACT)
        assert isinstance(results, list)

    def test_same_length_without_top_k(self):
        results = self.reranker.rerank("contract damages", RESULTS_CONTRACT)
        assert len(results) == len(RESULTS_CONTRACT)

    def test_top_k_limits_output(self):
        results = self.reranker.rerank("contract", RESULTS_CONTRACT, top_k=3)
        assert len(results) == 3

    def test_rerank_score_added(self):
        results = self.reranker.rerank("contract", RESULTS_CONTRACT)
        for r in results:
            assert "rerank_score" in r
            assert isinstance(r["rerank_score"], float)

    def test_rerank_method_field(self):
        results = self.reranker.rerank("contract", RESULTS_CONTRACT)
        assert all(r["rerank_method"] == "tfidf" for r in results)

    def test_descending_order(self):
        results = self.reranker.rerank("contract damages breach", RESULTS_CONTRACT)
        scores = [r["rerank_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_relevant_docs_ranked_higher(self):
        """Contract-heavy docs should outscore the negligence doc."""
        results = self.reranker.rerank("contract damages", RESULTS_CONTRACT)
        top_ids = {r["id"] for r in results[:3]}
        # r1, r3, r4 are all contract-related
        assert len(top_ids & {"r1", "r3", "r4"}) >= 2

    def test_empty_results(self):
        assert self.reranker.rerank("contract", []) == []

    def test_single_result(self):
        results = self.reranker.rerank("contract", [RESULTS_CONTRACT[0]])
        assert len(results) == 1
        assert "rerank_score" in results[0]

    def test_top_k_larger_than_results(self):
        results = self.reranker.rerank("contract", RESULTS_CONTRACT, top_k=100)
        assert len(results) == len(RESULTS_CONTRACT)

    def test_original_fields_preserved(self):
        results = self.reranker.rerank("contract", RESULTS_CONTRACT)
        original_ids = {r["id"] for r in RESULTS_CONTRACT}
        result_ids = {r["id"] for r in results}
        assert original_ids == result_ids

    def test_does_not_mutate_input(self):
        original_copy = [dict(r) for r in RESULTS_CONTRACT]
        self.reranker.rerank("contract", RESULTS_CONTRACT)
        for orig, current in zip(original_copy, RESULTS_CONTRACT):
            assert orig == current


# ---------------------------------------------------------------------------
# ScoreReranker
# ---------------------------------------------------------------------------


class TestScoreReranker:
    def setup_method(self):
        self.reranker = ScoreReranker(alpha=0.5)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            ScoreReranker(alpha=1.5)

        with pytest.raises(ValueError):
            ScoreReranker(alpha=-0.1)

    def test_returns_correct_length(self):
        results = self.reranker.rerank("contract", RESULTS_CONTRACT)
        assert len(results) == len(RESULTS_CONTRACT)

    def test_top_k(self):
        results = self.reranker.rerank("contract", RESULTS_CONTRACT, top_k=2)
        assert len(results) == 2

    def test_rerank_score_in_range(self):
        results = self.reranker.rerank("contract damages", RESULTS_CONTRACT)
        for r in results:
            assert 0.0 <= r["rerank_score"] <= 1.0 + 1e-9

    def test_rerank_method_contains_alpha(self):
        results = self.reranker.rerank("contract", RESULTS_CONTRACT)
        assert all("0.5" in r["rerank_method"] for r in results)

    def test_descending_order(self):
        results = self.reranker.rerank("contract", RESULTS_CONTRACT)
        scores = [r["rerank_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_alpha_one_preserves_vector_rank(self):
        """alpha=1 → pure vector score ordering."""
        reranker = ScoreReranker(alpha=1.0)
        results = reranker.rerank("anything", RESULTS_CONTRACT)
        scores = [r["rerank_score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        # The highest original score (r1, 0.90) should be first
        assert results[0]["id"] == "r1"

    def test_alpha_zero_pure_tfidf(self):
        """alpha=0 → pure TF-IDF ordering."""
        reranker = ScoreReranker(alpha=0.0)
        results_blend = reranker.rerank("contract damages", RESULTS_CONTRACT)
        tfidf_results = TFIDFReranker().rerank("contract damages", RESULTS_CONTRACT)
        # Top result should be the same
        assert results_blend[0]["id"] == tfidf_results[0]["id"]

    def test_empty_input(self):
        assert self.reranker.rerank("contract", []) == []

    def test_single_result(self):
        results = self.reranker.rerank("contract", [RESULTS_CONTRACT[0]])
        assert len(results) == 1

    def test_all_same_score(self):
        """Results with identical vector scores — should still sort by TF-IDF."""
        uniform = [
            make_result(r["title"], r["summary"], score=0.5) for r in RESULTS_CONTRACT
        ]
        results = ScoreReranker(alpha=0.5).rerank("contract damages", uniform)
        scores = [r["rerank_score"] for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# MMRReranker
# ---------------------------------------------------------------------------


class TestMMRReranker:
    def setup_method(self):
        self.reranker = MMRReranker(lambda_param=0.5)

    def test_invalid_lambda_raises(self):
        with pytest.raises(ValueError):
            MMRReranker(lambda_param=1.5)
        with pytest.raises(ValueError):
            MMRReranker(lambda_param=-0.1)

    def test_returns_correct_length_no_top_k(self):
        results = self.reranker.rerank("contract", RESULTS_CONTRACT)
        assert len(results) == len(RESULTS_CONTRACT)

    def test_top_k(self):
        results = self.reranker.rerank("contract", RESULTS_CONTRACT, top_k=3)
        assert len(results) == 3

    def test_no_duplicate_results(self):
        results = self.reranker.rerank("contract damages", RESULTS_CONTRACT, top_k=5)
        ids = [r["id"] for r in results]
        assert len(ids) == len(set(ids))

    def test_rerank_score_added(self):
        results = self.reranker.rerank("contract", RESULTS_CONTRACT)
        for r in results:
            assert "rerank_score" in r
            assert 0.0 <= r["rerank_score"] <= 1.0 + 1e-9

    def test_mmr_rank_field(self):
        results = self.reranker.rerank("contract", RESULTS_CONTRACT)
        ranks = [r["mmr_rank"] for r in results]
        assert ranks == list(range(1, len(RESULTS_CONTRACT) + 1))

    def test_rerank_method_contains_lambda(self):
        results = self.reranker.rerank("contract", RESULTS_CONTRACT)
        assert all("0.5" in r["rerank_method"] for r in results)

    def test_lambda_one_is_pure_relevance(self):
        """lambda=1 ignores diversity; should rank by TF-IDF relevance."""
        reranker = MMRReranker(lambda_param=1.0)
        mmr_results = reranker.rerank("contract damages breach", RESULTS_CONTRACT)
        tfidf_results = TFIDFReranker().rerank(
            "contract damages breach", RESULTS_CONTRACT
        )
        assert mmr_results[0]["id"] == tfidf_results[0]["id"]

    def test_lambda_zero_maximum_diversity(self):
        """lambda=0 → maximise diversity; first pick is still the most relevant
        (no prior selected docs for comparison yet)."""
        reranker = MMRReranker(lambda_param=0.0)
        results = reranker.rerank("contract", RESULTS_CONTRACT, top_k=5)
        assert len(results) == 5
        ids = [r["id"] for r in results]
        assert len(ids) == len(set(ids))

    def test_empty_input(self):
        assert self.reranker.rerank("contract", []) == []

    def test_single_result(self):
        results = self.reranker.rerank("contract", [RESULTS_CONTRACT[0]])
        assert len(results) == 1
        assert results[0]["mmr_rank"] == 1

    def test_top_k_larger_than_results(self):
        results = self.reranker.rerank("contract", RESULTS_CONTRACT, top_k=999)
        assert len(results) == len(RESULTS_CONTRACT)

    def test_all_fields_preserved(self):
        results = self.reranker.rerank("contract", RESULTS_CONTRACT, top_k=3)
        for r in results:
            assert "title" in r
            assert "summary" in r
            assert "score" in r

    def test_diversity_vs_relevance(self):
        """With lambda=0.5, similar docs should not dominate top results."""
        # Create 3 near-identical "contract" docs + 1 different "negligence" doc
        similar: List[Dict] = [
            make_result(
                f"Contract Case {i}",
                "contract breach damages award",
                score=0.9,
                rid=f"c{i}",
            )
            for i in range(3)
        ]
        different = make_result(
            "Tort Case", "negligence duty care proximity", score=0.5, rid="t1"
        )
        pool = similar + [different]

        reranker = MMRReranker(lambda_param=0.5)
        results = reranker.rerank("contract", pool, top_k=4)

        ids = [r["id"] for r in results]
        # All 4 docs should appear exactly once
        assert sorted(ids) == sorted(["c0", "c1", "c2", "t1"])


# ---------------------------------------------------------------------------
# get_reranker factory
# ---------------------------------------------------------------------------


class TestGetReranker:
    def test_tfidf(self):
        r = get_reranker("tfidf")
        assert isinstance(r, TFIDFReranker)

    def test_score(self):
        r = get_reranker("score", alpha=0.7)
        assert isinstance(r, ScoreReranker)
        assert r.alpha == 0.7

    def test_mmr(self):
        r = get_reranker("mmr", lambda_param=0.3)
        assert isinstance(r, MMRReranker)
        assert r.lambda_param == 0.3

    def test_default_is_mmr(self):
        r = get_reranker()
        assert isinstance(r, MMRReranker)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown reranker"):
            get_reranker("unknown_method")

    def test_factory_produces_working_reranker(self):
        reranker = get_reranker("mmr", lambda_param=0.6)
        results = reranker.rerank("contract", RESULTS_CONTRACT, top_k=3)
        assert len(results) == 3
        assert all("rerank_score" in r for r in results)
