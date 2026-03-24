"""
Tests for utils/passage_fusion.py

Covers:
- SentenceWindowFusion: indexing, fuse deduplication, score aggregation, edge cases
- MultiQueryFusion: variant generation, RRF fusion, variant_hits, empty results
- HyDEFusion: hypothesis generation, alpha blending, edge cases
- get_fusion factory: known/unknown methods
- RRF helper: weighted fusion, deduplication
"""

from __future__ import annotations

import pytest

from utils.passage_fusion import (
    HyDEFusion,
    MultiQueryFusion,
    MultiQueryFusionResult,
    SentenceWindowFusion,
    _rrf_fuse,
    get_fusion,
)


# ---------------------------------------------------------------------------
# SentenceWindowFusion — indexing
# ---------------------------------------------------------------------------


class TestSentenceWindowIndex:
    def test_index_single_chunk(self):
        fusion = SentenceWindowFusion(window_size=2)
        fusion.index([{"id": "d1", "text": "First sentence. Second sentence."}])
        assert fusion.parent_count == 1
        assert fusion.child_count >= 1

    def test_index_three_chunks(self):
        fusion = SentenceWindowFusion(window_size=2)
        chunks = [
            {"id": f"d{i}", "text": f"Sentence A{i}. Sentence B{i}. Sentence C{i}."}
            for i in range(3)
        ]
        fusion.index(chunks)
        assert fusion.parent_count == 3

    def test_index_ten_chunks(self):
        fusion = SentenceWindowFusion(window_size=3)
        chunks = [
            {"id": f"d{i}", "text": f"S{i}a. S{i}b. S{i}c. S{i}d. S{i}e."}
            for i in range(10)
        ]
        fusion.index(chunks)
        assert fusion.parent_count == 10
        # With overlap and window_size=3 on 5 sentences, expect 3 windows
        assert fusion.child_count >= 10

    def test_index_missing_id_raises(self):
        fusion = SentenceWindowFusion()
        with pytest.raises(ValueError, match="id"):
            fusion.index([{"text": "Hello."}])

    def test_index_missing_text_raises(self):
        fusion = SentenceWindowFusion()
        with pytest.raises(ValueError, match="text"):
            fusion.index([{"id": "x"}])

    def test_index_empty_text_no_children(self):
        fusion = SentenceWindowFusion()
        fusion.index([{"id": "empty", "text": ""}])
        assert fusion.parent_count == 1
        assert fusion.child_count == 0

    def test_index_no_overlap_fewer_children(self):
        text = "S1. S2. S3. S4. S5. S6."
        overlap = SentenceWindowFusion(window_size=2, parent_overlap=True)
        overlap.index([{"id": "a", "text": text}])

        no_overlap = SentenceWindowFusion(window_size=2, parent_overlap=False)
        no_overlap.index([{"id": "a", "text": text}])

        assert no_overlap.child_count <= overlap.child_count

    def test_window_size_validation(self):
        with pytest.raises(ValueError):
            SentenceWindowFusion(window_size=0)

    def test_index_preserves_extra_fields(self):
        fusion = SentenceWindowFusion(window_size=2)
        fusion.index([{"id": "d1", "text": "A. B.", "source": "test"}])
        result = fusion.fuse([{"id": "d1_w0", "score": 1.0}])
        assert result[0]["source"] == "test"


# ---------------------------------------------------------------------------
# SentenceWindowFusion — fuse
# ---------------------------------------------------------------------------


class TestSentenceWindowFuse:
    @pytest.fixture()
    def indexed_fusion(self):
        fusion = SentenceWindowFusion(window_size=2)
        fusion.index(
            [
                {"id": "p1", "text": "Alpha one. Alpha two. Alpha three."},
                {"id": "p2", "text": "Beta one. Beta two. Beta three."},
            ]
        )
        return fusion

    def test_fuse_maps_child_to_parent(self, indexed_fusion):
        results = indexed_fusion.fuse([{"id": "p1_w0", "score": 0.9}])
        assert len(results) == 1
        assert results[0]["id"] == "p1"

    def test_fuse_deduplication_same_parent(self, indexed_fusion):
        """Two children from the same parent → one parent returned."""
        results = indexed_fusion.fuse(
            [
                {"id": "p1_w0", "score": 0.5},
                {"id": "p1_w1", "score": 0.4},
            ]
        )
        assert len(results) == 1
        assert results[0]["id"] == "p1"

    def test_fuse_score_aggregation(self, indexed_fusion):
        """Sum of child scores for the same parent."""
        results = indexed_fusion.fuse(
            [
                {"id": "p1_w0", "score": 0.5},
                {"id": "p1_w1", "score": 0.3},
            ]
        )
        assert results[0]["aggregated_score"] == pytest.approx(0.8)

    def test_fuse_child_count(self, indexed_fusion):
        results = indexed_fusion.fuse(
            [
                {"id": "p1_w0", "score": 0.5},
                {"id": "p1_w1", "score": 0.3},
            ]
        )
        assert results[0]["child_count"] == 2

    def test_fuse_empty_input(self, indexed_fusion):
        results = indexed_fusion.fuse([])
        assert results == []

    def test_fuse_unknown_child_ignored(self, indexed_fusion):
        results = indexed_fusion.fuse([{"id": "unknown_w0", "score": 0.9}])
        assert results == []

    def test_fuse_top_k_limits_output(self, indexed_fusion):
        results = indexed_fusion.fuse(
            [
                {"id": "p1_w0", "score": 0.9},
                {"id": "p2_w0", "score": 0.8},
            ],
            top_k=1,
        )
        assert len(results) == 1

    def test_fuse_ordering_by_score(self, indexed_fusion):
        results = indexed_fusion.fuse(
            [
                {"id": "p2_w0", "score": 0.9},
                {"id": "p1_w0", "score": 0.1},
            ]
        )
        assert results[0]["id"] == "p2"
        assert results[1]["id"] == "p1"

    def test_fuse_multiple_children_beats_single_high_score(self, indexed_fusion):
        """Parent with 2 children scoring 0.4 each (sum=0.8) beats one child at 0.7."""
        results = indexed_fusion.fuse(
            [
                {"id": "p1_w0", "score": 0.4},
                {"id": "p1_w1", "score": 0.4},
                {"id": "p2_w0", "score": 0.7},
            ]
        )
        assert results[0]["id"] == "p1"


# ---------------------------------------------------------------------------
# MultiQueryFusion — variant generation
# ---------------------------------------------------------------------------


class TestMultiQueryVariants:
    def test_returns_at_least_three_variants(self):
        fusion = MultiQueryFusion(max_variants=5)
        variants = fusion.generate_variants("data retrieval methods")
        assert len(variants) >= 3

    def test_original_query_always_first(self):
        fusion = MultiQueryFusion()
        variants = fusion.generate_variants("test query")
        assert variants[0] == "test query"

    def test_no_duplicate_variants(self):
        fusion = MultiQueryFusion(max_variants=5)
        variants = fusion.generate_variants("information retrieval")
        lower = [v.lower().strip() for v in variants]
        assert len(lower) == len(set(lower))

    def test_keywords_only_variant_removes_stopwords(self):
        fusion = MultiQueryFusion(max_variants=5)
        variants = fusion.generate_variants("the breach of a contract")
        # At least one variant should lack "the" and "of" and "a"
        has_keywords_only = any(
            "the" not in v.lower().split() and "of" not in v.lower().split()
            for v in variants[1:]
        )
        assert has_keywords_only

    def test_max_variants_respected(self):
        fusion = MultiQueryFusion(max_variants=2)
        variants = fusion.generate_variants("test query words here")
        assert len(variants) <= 2

    def test_single_word_query(self):
        fusion = MultiQueryFusion()
        variants = fusion.generate_variants("negligence")
        assert len(variants) >= 1
        assert variants[0] == "negligence"

    def test_empty_query(self):
        fusion = MultiQueryFusion()
        variants = fusion.generate_variants("")
        assert len(variants) >= 1

    def test_question_query_no_extra_prefix(self):
        fusion = MultiQueryFusion(max_variants=5)
        variants = fusion.generate_variants("What is a remedy")
        # Should not add "What is What is a remedy"
        doubled = [v for v in variants if v.lower().startswith("what is what")]
        assert len(doubled) == 0

    def test_max_variants_validation(self):
        with pytest.raises(ValueError):
            MultiQueryFusion(max_variants=0)


# ---------------------------------------------------------------------------
# MultiQueryFusion — fuse
# ---------------------------------------------------------------------------


class TestMultiQueryFuse:
    def _mock_retrieve(self, results_map):
        """Return a retrieve_fn that returns different results per query."""

        def retrieve_fn(query):
            return results_map.get(query, [])

        return retrieve_fn

    def test_fuse_basic(self):
        fusion = MultiQueryFusion(max_variants=2, rrf_k=60)

        def retrieve_fn(query):
            return [
                {"id": "doc1", "text": "Result 1"},
                {"id": "doc2", "text": "Result 2"},
            ]

        result = fusion.fuse("test query", retrieve_fn, top_k=5)
        assert isinstance(result, MultiQueryFusionResult)
        assert len(result.results) >= 1

    def test_fuse_variant_hits_correct(self):
        fusion = MultiQueryFusion(max_variants=2, rrf_k=60)
        call_counts = {}

        def retrieve_fn(query):
            docs = [{"id": f"d_{query}_{i}", "text": f"R{i}"} for i in range(3)]
            call_counts[query] = len(docs)
            return docs

        result = fusion.fuse("test", retrieve_fn, top_k=10)
        for variant, hits in result.variant_hits.items():
            assert hits == 3

    def test_fuse_empty_retrieve(self):
        fusion = MultiQueryFusion(max_variants=3)

        def retrieve_fn(query):
            return []

        result = fusion.fuse("nothing here", retrieve_fn)
        assert result.results == []
        for hits in result.variant_hits.values():
            assert hits == 0

    def test_fuse_deduplication_across_variants(self):
        """Same doc from multiple variants should appear once in fused results."""
        fusion = MultiQueryFusion(max_variants=2, rrf_k=60)

        def retrieve_fn(query):
            return [{"id": "shared_doc", "text": "Shared"}]

        result = fusion.fuse("test", retrieve_fn, top_k=5)
        ids = [r["id"] for r in result.results]
        assert ids.count("shared_doc") == 1

    def test_fuse_rrf_score_present(self):
        fusion = MultiQueryFusion(max_variants=2)

        def retrieve_fn(query):
            return [{"id": "doc1", "text": "Hello"}]

        result = fusion.fuse("test", retrieve_fn)
        assert "rrf_score" in result.results[0]
        assert result.results[0]["rrf_score"] > 0

    def test_fuse_top_k_limits(self):
        fusion = MultiQueryFusion(max_variants=2)

        def retrieve_fn(query):
            return [{"id": f"d{i}", "text": f"R{i}"} for i in range(20)]

        result = fusion.fuse("test", retrieve_fn, top_k=3)
        assert len(result.results) <= 3

    def test_fuse_result_has_query(self):
        fusion = MultiQueryFusion(max_variants=2)
        result = fusion.fuse("original", lambda q: [], top_k=5)
        assert result.query == "original"


# ---------------------------------------------------------------------------
# HyDEFusion — hypothesis generation
# ---------------------------------------------------------------------------


class TestHyDEHypothesis:
    def test_hypothesis_contains_query_terms(self):
        hyde = HyDEFusion(domain="research")
        hyp = hyde.generate_hypothesis("data retrieval")
        assert "data retrieval" in hyp.lower()

    def test_hypothesis_contains_domain(self):
        hyde = HyDEFusion(domain="medical")
        hyp = hyde.generate_hypothesis("treatment options")
        assert "medical" in hyp.lower()

    def test_hypothesis_default_domain(self):
        hyde = HyDEFusion()
        hyp = hyde.generate_hypothesis("contract law")
        assert "legal" in hyp.lower()

    def test_hypothesis_nonempty(self):
        hyde = HyDEFusion()
        hyp = hyde.generate_hypothesis("test")
        assert len(hyp) > 10

    def test_hypothesis_short_query(self):
        hyde = HyDEFusion()
        hyp = hyde.generate_hypothesis("x")
        assert "x" in hyp


# ---------------------------------------------------------------------------
# HyDEFusion — fuse
# ---------------------------------------------------------------------------


class TestHyDEFuse:
    def _make_results(self, ids):
        return [{"id": doc_id, "text": f"Doc {doc_id}"} for doc_id in ids]

    def test_fuse_alpha_zero_query_order(self):
        """alpha=0 means hypothesis has zero weight → query order dominates."""
        hyde = HyDEFusion()
        query_results = self._make_results(["a", "b", "c"])
        hypo_results = self._make_results(["c", "b", "a"])

        fused = hyde.fuse(query_results, hypo_results, alpha=0.0)
        ids = [r["id"] for r in fused]
        assert ids[0] == "a"

    def test_fuse_alpha_one_hypothesis_order(self):
        """alpha=1 means query has zero weight → hypothesis order dominates."""
        hyde = HyDEFusion()
        query_results = self._make_results(["a", "b", "c"])
        hypo_results = self._make_results(["c", "b", "a"])

        fused = hyde.fuse(query_results, hypo_results, alpha=1.0)
        ids = [r["id"] for r in fused]
        assert ids[0] == "c"

    def test_fuse_mixed_produces_ranking(self):
        hyde = HyDEFusion()
        query_results = self._make_results(["a", "b"])
        hypo_results = self._make_results(["b", "c"])

        fused = hyde.fuse(query_results, hypo_results, alpha=0.5)
        assert len(fused) >= 2
        # "b" appears in both lists, should rank highest
        assert fused[0]["id"] == "b"

    def test_fuse_empty_both(self):
        hyde = HyDEFusion()
        assert hyde.fuse([], []) == []

    def test_fuse_empty_hypothesis(self):
        hyde = HyDEFusion()
        query_results = self._make_results(["a", "b"])
        fused = hyde.fuse(query_results, [], alpha=0.3)
        assert len(fused) == 2

    def test_fuse_empty_query_results(self):
        hyde = HyDEFusion()
        hypo_results = self._make_results(["x", "y"])
        fused = hyde.fuse([], hypo_results, alpha=0.3)
        assert len(fused) == 2

    def test_fuse_invalid_alpha_low(self):
        hyde = HyDEFusion()
        with pytest.raises(ValueError, match="alpha"):
            hyde.fuse([], [], alpha=-0.1)

    def test_fuse_invalid_alpha_high(self):
        hyde = HyDEFusion()
        with pytest.raises(ValueError, match="alpha"):
            hyde.fuse([], [], alpha=1.5)

    def test_fuse_rrf_scores_present(self):
        hyde = HyDEFusion()
        fused = hyde.fuse(
            self._make_results(["a"]),
            self._make_results(["b"]),
            alpha=0.5,
        )
        for r in fused:
            assert "rrf_score" in r

    def test_fuse_identical_scores(self):
        """Documents with identical RRF contributions should all appear."""
        hyde = HyDEFusion()
        query_results = self._make_results(["a", "b", "c"])
        hypo_results = self._make_results(["a", "b", "c"])
        fused = hyde.fuse(query_results, hypo_results, alpha=0.5)
        assert len(fused) == 3


# ---------------------------------------------------------------------------
# get_fusion factory
# ---------------------------------------------------------------------------


class TestGetFusion:
    def test_sentence_window(self):
        f = get_fusion("sentence_window")
        assert isinstance(f, SentenceWindowFusion)

    def test_multi_query(self):
        f = get_fusion("multi_query")
        assert isinstance(f, MultiQueryFusion)

    def test_hyde(self):
        f = get_fusion("hyde")
        assert isinstance(f, HyDEFusion)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown fusion"):
            get_fusion("unknown")

    def test_kwargs_forwarded(self):
        f = get_fusion("hyde", domain="medical")
        assert f.domain == "medical"

    def test_default_is_multi_query(self):
        f = get_fusion()
        assert isinstance(f, MultiQueryFusion)


# ---------------------------------------------------------------------------
# RRF helper
# ---------------------------------------------------------------------------


class TestRRFHelper:
    def test_single_list(self):
        results = _rrf_fuse(
            [[{"id": "a"}, {"id": "b"}]],
            k=60,
            top_k=2,
        )
        assert results[0]["id"] == "a"
        assert results[1]["id"] == "b"

    def test_weighted_fusion(self):
        list1 = [{"id": "x"}, {"id": "y"}]
        list2 = [{"id": "y"}, {"id": "x"}]
        results = _rrf_fuse([list1, list2], k=60, weights=[1.0, 1.0], top_k=2)
        # Both appear with equal total weight, "x" and "y" both at rank 1+2
        assert len(results) == 2

    def test_empty_lists(self):
        assert _rrf_fuse([[], []], k=60, top_k=5) == []
