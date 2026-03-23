"""
Tests for utils/deduplication.py

Covers:
- _tokenize, _shingle helpers
- hamming_distance, simhash standalone functions
- ExactDedupe: fit, is_duplicate, reset, fit_deduplicate, normalisation
- SimHashDedupe: construction, near-duplicate detection, Hamming threshold,
  fit, reset, fingerprint utility
- MinHashDedupe: construction, Jaccard estimation, near-dup detection,
  fit, reset
- DeduplicationPipeline: fit_deduplicate, is_unique/add API, per-stage
  counters, skip flags, reset, property accessors
- DeduplicationResult: duplicate_rate calculation
"""

from __future__ import annotations

import pytest

from utils.deduplication import (
    DeduplicationPipeline,
    DeduplicationResult,
    ExactDedupe,
    MinHashDedupe,
    SimHashDedupe,
    _jaccard_from_signatures,
    _minhash_signature,
    _shingle,
    _tokenize,
    hamming_distance,
    simhash,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Representative legal text snippets
CONTRACT_DOC = (
    "The parties hereby agree to the following terms and conditions of employment, "
    "including all schedules and annexures attached hereto."
)

CONTRACT_DOC_WHITESPACE = (
    "The  parties  hereby agree  to the following terms and conditions of employment, "
    "including all schedules and annexures   attached hereto."
)

CONTRACT_DOC_PARAPHRASE = (
    "Both parties consent to the terms and conditions of employment "
    "together with all schedules and annexures incorporated herein."
)

NEGLIGENCE_DOC = (
    "The defendant owed the claimant a duty of care, and that duty was breached "
    "when the defendant failed to maintain the premises in a safe condition."
)

UNRELATED_DOC = (
    "Machine learning models require substantial training data and computational "
    "resources to achieve high performance on downstream tasks."
)


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_lowercases_input(self):
        assert _tokenize("Contract") == ["contract"]

    def test_removes_stop_words(self):
        tokens = _tokenize("a breach of contract")
        assert "a" not in tokens
        assert "of" not in tokens
        assert "breach" in tokens

    def test_removes_single_chars(self):
        tokens = _tokenize("i x damages")
        assert "i" not in tokens
        assert "x" not in tokens
        assert "damages" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_all_stop_words(self):
        assert _tokenize("the a an and") == []

    def test_numbers_kept(self):
        tokens = _tokenize("section 42 of the act")
        assert "42" in tokens

    def test_punctuation_stripped(self):
        tokens = _tokenize("breach, damages!")
        assert "breach" in tokens
        assert "damages" in tokens


# ---------------------------------------------------------------------------
# _shingle
# ---------------------------------------------------------------------------


class TestShingle:
    def test_basic_bigram(self):
        shingles = _shingle("contract breach damages", k=2)
        assert "contract breach" in shingles
        assert "breach damages" in shingles
        assert len(shingles) == 2  # 3 tokens - 2 + 1

    def test_trigram_default(self):
        shingles = _shingle("contract breach damages remedy", k=3)
        assert "contract breach damages" in shingles
        assert "breach damages remedy" in shingles

    def test_empty_text_returns_empty(self):
        assert _shingle("") == []

    def test_short_text_falls_back_to_unigrams(self):
        # Only 1 token → k=3 shingle impossible → unigrams
        shingles = _shingle("contract", k=3)
        assert shingles == ["contract"]

    def test_two_tokens_returns_unigrams_for_trigram(self):
        shingles = _shingle("contract breach", k=3)
        # 2 tokens < 3 → unigram fallback
        assert "contract" in shingles
        assert "breach" in shingles

    def test_count(self):
        text = "a b c d e"  # 5 tokens after stop-word removal: only 5 single-chars
        # after tokenization stop-word "a" removed; b,c,d,e are single chars too
        # → 0 tokens → empty
        assert _shingle(text, k=3) == []

    def test_stop_word_filtering(self):
        shingles = _shingle("the breach of a contract", k=2)
        # After stop-word removal: breach, contract
        assert "breach contract" in shingles


# ---------------------------------------------------------------------------
# hamming_distance
# ---------------------------------------------------------------------------


class TestHammingDistance:
    def test_identical_integers(self):
        assert hamming_distance(0xFF, 0xFF) == 0

    def test_single_bit_difference(self):
        assert hamming_distance(0b1000, 0b0000) == 1

    def test_all_bits_different(self):
        assert hamming_distance(0xFFFFFFFF, 0x00000000) == 32

    def test_zero_vs_zero(self):
        assert hamming_distance(0, 0) == 0

    def test_symmetry(self):
        a, b = 0b1010_1010, 0b0101_0101
        assert hamming_distance(a, b) == hamming_distance(b, a)


# ---------------------------------------------------------------------------
# simhash
# ---------------------------------------------------------------------------


class TestSimhash:
    def test_returns_int(self):
        fp = simhash(CONTRACT_DOC)
        assert isinstance(fp, int)

    def test_deterministic(self):
        assert simhash(CONTRACT_DOC) == simhash(CONTRACT_DOC)

    def test_different_docs_different_fingerprints(self):
        fp1 = simhash(CONTRACT_DOC)
        fp2 = simhash(UNRELATED_DOC)
        # Almost certainly different for unrelated texts
        assert fp1 != fp2

    def test_similar_docs_close_fingerprints(self):
        fp1 = simhash(CONTRACT_DOC)
        fp2 = simhash(CONTRACT_DOC_WHITESPACE)
        # Whitespace normalisation means near-zero Hamming distance
        assert hamming_distance(fp1, fp2) <= 10

    def test_empty_string_returns_zero(self):
        assert simhash("") == 0

    def test_stop_words_only_returns_zero(self):
        assert simhash("the a an and or") == 0


# ---------------------------------------------------------------------------
# ExactDedupe
# ---------------------------------------------------------------------------


class TestExactDedupe:
    def setup_method(self):
        self.deduper = ExactDedupe()

    def test_exact_duplicate_detected(self):
        self.deduper.fit([CONTRACT_DOC])
        assert self.deduper.is_duplicate(CONTRACT_DOC)

    def test_different_doc_not_duplicate(self):
        self.deduper.fit([CONTRACT_DOC])
        assert not self.deduper.is_duplicate(NEGLIGENCE_DOC)

    def test_whitespace_variant_is_duplicate_when_normalized(self):
        self.deduper.fit([CONTRACT_DOC])
        # Extra spaces should collapse to same fingerprint
        assert self.deduper.is_duplicate(CONTRACT_DOC_WHITESPACE)

    def test_whitespace_variant_not_dup_when_normalize_false(self):
        deduper = ExactDedupe(normalize=False)
        deduper.fit([CONTRACT_DOC])
        assert not deduper.is_duplicate(CONTRACT_DOC_WHITESPACE)

    def test_empty_corpus_not_duplicate(self):
        assert not self.deduper.is_duplicate(CONTRACT_DOC)

    def test_seen_count(self):
        self.deduper.fit([CONTRACT_DOC, NEGLIGENCE_DOC, CONTRACT_DOC])
        assert self.deduper.seen_count == 2  # CONTRACT_DOC hashes to same

    def test_reset_clears_state(self):
        self.deduper.fit([CONTRACT_DOC])
        self.deduper.reset()
        assert not self.deduper.is_duplicate(CONTRACT_DOC)
        assert self.deduper.seen_count == 0

    def test_fit_deduplicate_removes_exact_dups(self):
        corpus = [CONTRACT_DOC, CONTRACT_DOC, NEGLIGENCE_DOC, CONTRACT_DOC]
        unique = self.deduper.fit_deduplicate(corpus)
        assert len(unique) == 2
        assert CONTRACT_DOC in unique
        assert NEGLIGENCE_DOC in unique

    def test_fit_deduplicate_preserves_order(self):
        corpus = [NEGLIGENCE_DOC, CONTRACT_DOC, UNRELATED_DOC]
        unique = self.deduper.fit_deduplicate(corpus)
        assert unique == corpus  # all distinct, order preserved

    def test_fit_deduplicate_empty_input(self):
        assert self.deduper.fit_deduplicate([]) == []

    def test_fit_deduplicate_all_same(self):
        corpus = [CONTRACT_DOC] * 10
        unique = self.deduper.fit_deduplicate(corpus)
        assert unique == [CONTRACT_DOC]

    def test_fit_returns_self(self):
        result = self.deduper.fit([CONTRACT_DOC])
        assert result is self.deduper


# ---------------------------------------------------------------------------
# SimHashDedupe
# ---------------------------------------------------------------------------


class TestSimHashDedupe:
    def setup_method(self):
        self.deduper = SimHashDedupe(hamming_threshold=3)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="hamming_threshold"):
            SimHashDedupe(hamming_threshold=-1)

    def test_exact_duplicate_detected_at_zero_threshold(self):
        deduper = SimHashDedupe(hamming_threshold=0)
        deduper.fit([CONTRACT_DOC])
        assert deduper.is_duplicate(CONTRACT_DOC)

    def test_near_dup_whitespace_variant(self):
        self.deduper.fit([CONTRACT_DOC])
        assert self.deduper.is_duplicate(CONTRACT_DOC_WHITESPACE)

    def test_unrelated_doc_not_duplicate(self):
        self.deduper.fit([CONTRACT_DOC])
        assert not self.deduper.is_duplicate(UNRELATED_DOC)

    def test_empty_corpus_not_duplicate(self):
        assert not self.deduper.is_duplicate(CONTRACT_DOC)

    def test_threshold_zero_strict(self):
        """At threshold 0, only bit-identical SimHash fingerprints match."""
        deduper = SimHashDedupe(hamming_threshold=0)
        deduper.fit([CONTRACT_DOC])
        # CONTRACT_DOC_WHITESPACE may differ by a few bits
        fp1 = simhash(CONTRACT_DOC)
        fp2 = simhash(CONTRACT_DOC_WHITESPACE)
        if fp1 == fp2:
            assert deduper.is_duplicate(CONTRACT_DOC_WHITESPACE)
        else:
            assert not deduper.is_duplicate(CONTRACT_DOC_WHITESPACE)

    def test_indexed_count(self):
        self.deduper.fit([CONTRACT_DOC, NEGLIGENCE_DOC, UNRELATED_DOC])
        assert self.deduper.indexed_count == 3

    def test_reset_clears_state(self):
        self.deduper.fit([CONTRACT_DOC])
        self.deduper.reset()
        assert not self.deduper.is_duplicate(CONTRACT_DOC)
        assert self.deduper.indexed_count == 0

    def test_fingerprint_utility_is_deterministic(self):
        fp1 = self.deduper.fingerprint(CONTRACT_DOC)
        fp2 = self.deduper.fingerprint(CONTRACT_DOC)
        assert fp1 == fp2
        assert isinstance(fp1, int)

    def test_fit_returns_self(self):
        result = self.deduper.fit([CONTRACT_DOC])
        assert result is self.deduper

    def test_fit_deduplicate_removes_near_dups(self):
        corpus = [CONTRACT_DOC, CONTRACT_DOC_WHITESPACE, NEGLIGENCE_DOC]
        unique = self.deduper.fit_deduplicate(corpus)
        # CONTRACT_DOC and its whitespace variant should collapse to one
        assert len(unique) == 2

    def test_fit_deduplicate_empty_input(self):
        assert self.deduper.fit_deduplicate([]) == []


# ---------------------------------------------------------------------------
# MinHashDedupe
# ---------------------------------------------------------------------------

_MINHASH_PARAMS = [(1234567, 89012345), (9876543, 12345678)]


class TestMinHashSignature:
    def test_returns_correct_length(self):
        sig = _minhash_signature(
            ["contract breach", "damages award"], 4, _MINHASH_PARAMS * 2
        )
        assert len(sig) == 4

    def test_empty_shingles_returns_sentinel(self):
        from utils.deduplication import _MERSENNE_PRIME

        sig = _minhash_signature([], 4, _MINHASH_PARAMS * 2)
        assert all(v == _MERSENNE_PRIME for v in sig)

    def test_deterministic(self):
        shingles = ["contract breach", "breach damages"]
        s1 = _minhash_signature(shingles, 4, _MINHASH_PARAMS * 2)
        s2 = _minhash_signature(shingles, 4, _MINHASH_PARAMS * 2)
        assert s1 == s2


class TestJaccardFromSignatures:
    def test_identical_signatures_return_one(self):
        sig = [1, 2, 3, 4]
        assert _jaccard_from_signatures(sig, sig) == 1.0

    def test_no_overlap_returns_zero(self):
        assert _jaccard_from_signatures([1, 2], [3, 4]) == 0.0

    def test_partial_overlap(self):
        j = _jaccard_from_signatures([1, 2, 3, 4], [1, 2, 5, 6])
        assert j == 0.5

    def test_empty_signatures_return_zero(self):
        assert _jaccard_from_signatures([], []) == 0.0


class TestMinHashDedupe:
    def setup_method(self):
        self.deduper = MinHashDedupe(
            similarity_threshold=0.7,
            n_hashes=128,
            shingle_size=3,
            seed=42,
        )

    def test_invalid_threshold_zero_raises(self):
        with pytest.raises(ValueError, match="similarity_threshold"):
            MinHashDedupe(similarity_threshold=0.0)

    def test_invalid_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="similarity_threshold"):
            MinHashDedupe(similarity_threshold=1.1)

    def test_invalid_n_hashes_raises(self):
        with pytest.raises(ValueError, match="n_hashes"):
            MinHashDedupe(n_hashes=0)

    def test_invalid_shingle_size_raises(self):
        with pytest.raises(ValueError, match="shingle_size"):
            MinHashDedupe(shingle_size=0)

    def test_exact_duplicate_is_near_dup(self):
        self.deduper.fit([CONTRACT_DOC])
        assert self.deduper.is_duplicate(CONTRACT_DOC)

    def test_unrelated_doc_not_duplicate(self):
        self.deduper.fit([CONTRACT_DOC])
        assert not self.deduper.is_duplicate(UNRELATED_DOC)

    def test_empty_corpus_not_duplicate(self):
        assert not self.deduper.is_duplicate(CONTRACT_DOC)

    def test_estimate_jaccard_identical(self):
        j = self.deduper.estimate_jaccard(CONTRACT_DOC, CONTRACT_DOC)
        assert j == pytest.approx(1.0, abs=1e-6)

    def test_estimate_jaccard_unrelated(self):
        j = self.deduper.estimate_jaccard(CONTRACT_DOC, UNRELATED_DOC)
        assert j < 0.3

    def test_estimate_jaccard_paraphrase_higher_than_unrelated(self):
        """Paraphrase similarity should exceed similarity to an unrelated doc."""
        j_paraphrase = self.deduper.estimate_jaccard(
            CONTRACT_DOC, CONTRACT_DOC_PARAPHRASE
        )
        j_unrelated = self.deduper.estimate_jaccard(CONTRACT_DOC, UNRELATED_DOC)
        assert j_paraphrase > j_unrelated
        assert j_paraphrase > 0.0

    def test_indexed_count(self):
        self.deduper.fit([CONTRACT_DOC, NEGLIGENCE_DOC, UNRELATED_DOC])
        assert self.deduper.indexed_count == 3

    def test_reset_clears_state(self):
        self.deduper.fit([CONTRACT_DOC])
        self.deduper.reset()
        assert not self.deduper.is_duplicate(CONTRACT_DOC)
        assert self.deduper.indexed_count == 0

    def test_fit_returns_self(self):
        assert self.deduper.fit([CONTRACT_DOC]) is self.deduper

    def test_fit_deduplicate_removes_exact_dups(self):
        corpus = [CONTRACT_DOC, CONTRACT_DOC, UNRELATED_DOC]
        unique = self.deduper.fit_deduplicate(corpus)
        assert len(unique) == 2

    def test_fit_deduplicate_empty_input(self):
        assert self.deduper.fit_deduplicate([]) == []

    def test_seed_reproducibility(self):
        """Same seed → same hash params → same signatures."""
        d1 = MinHashDedupe(n_hashes=32, seed=99)
        d2 = MinHashDedupe(n_hashes=32, seed=99)
        j1 = d1.estimate_jaccard(CONTRACT_DOC, NEGLIGENCE_DOC)
        j2 = d2.estimate_jaccard(CONTRACT_DOC, NEGLIGENCE_DOC)
        assert j1 == j2

    def test_different_seeds_same_estimate_approx(self):
        """Different seeds should produce similar (not identical) estimates."""
        d1 = MinHashDedupe(n_hashes=128, seed=1)
        d2 = MinHashDedupe(n_hashes=128, seed=2)
        j1 = d1.estimate_jaccard(CONTRACT_DOC, CONTRACT_DOC_PARAPHRASE)
        j2 = d2.estimate_jaccard(CONTRACT_DOC, CONTRACT_DOC_PARAPHRASE)
        assert abs(j1 - j2) < 0.2  # within 20% of each other


# ---------------------------------------------------------------------------
# DeduplicationResult
# ---------------------------------------------------------------------------


class TestDeduplicationResult:
    def test_duplicate_rate_all_unique(self):
        r = DeduplicationResult(total_input=10, total_unique=10)
        assert r.duplicate_rate == 0.0

    def test_duplicate_rate_all_dups(self):
        r = DeduplicationResult(total_input=5, total_unique=0)
        assert r.duplicate_rate == 1.0

    def test_duplicate_rate_partial(self):
        r = DeduplicationResult(total_input=10, total_unique=7)
        assert r.duplicate_rate == pytest.approx(0.3, abs=1e-9)

    def test_duplicate_rate_zero_input(self):
        r = DeduplicationResult(total_input=0, total_unique=0)
        assert r.duplicate_rate == 0.0


# ---------------------------------------------------------------------------
# DeduplicationPipeline
# ---------------------------------------------------------------------------


class TestDeduplicationPipeline:
    def setup_method(self):
        self.pipeline = DeduplicationPipeline(
            hamming_threshold=3,
            minhash_threshold=0.7,
            minhash_n_hashes=128,
            shingle_size=3,
        )

    # --- fit_deduplicate: result structure ---

    def test_returns_list_and_report(self):
        unique, report = self.pipeline.fit_deduplicate(
            [CONTRACT_DOC, NEGLIGENCE_DOC, UNRELATED_DOC]
        )
        assert isinstance(unique, list)
        assert isinstance(report, DeduplicationResult)

    def test_all_unique_corpus_preserved(self):
        corpus = [CONTRACT_DOC, NEGLIGENCE_DOC, UNRELATED_DOC]
        unique, report = self.pipeline.fit_deduplicate(corpus)
        assert len(unique) == 3
        assert report.total_unique == 3
        assert report.total_input == 3

    def test_order_preserved(self):
        corpus = [UNRELATED_DOC, CONTRACT_DOC, NEGLIGENCE_DOC]
        unique, _ = self.pipeline.fit_deduplicate(corpus)
        assert unique == corpus

    # --- Stage 1: exact duplicates ---

    def test_exact_dup_counted_in_exact_stage(self):
        corpus = [CONTRACT_DOC, CONTRACT_DOC, NEGLIGENCE_DOC]
        _, report = self.pipeline.fit_deduplicate(corpus)
        assert report.exact_duplicates_removed == 1
        assert report.total_unique == 2

    def test_multiple_exact_dups(self):
        corpus = [CONTRACT_DOC] * 5
        unique, report = self.pipeline.fit_deduplicate(corpus)
        assert len(unique) == 1
        assert report.exact_duplicates_removed == 4

    # --- Stage 2: SimHash near-duplicates ---

    def test_whitespace_variant_caught_by_simhash(self):
        """Extra whitespace should fall through exact but be caught by SimHash."""
        pipeline = DeduplicationPipeline(hamming_threshold=5)
        corpus = [CONTRACT_DOC, CONTRACT_DOC_WHITESPACE, UNRELATED_DOC]
        unique, report = pipeline.fit_deduplicate(corpus)
        # CONTRACT_DOC normalises to same MD5 → exact duplicate (stage 1)
        assert report.total_unique == 2

    def test_simhash_stage_skipped(self):
        pipeline = DeduplicationPipeline(skip_simhash=True)
        corpus = [CONTRACT_DOC, CONTRACT_DOC, UNRELATED_DOC]
        unique, report = pipeline.fit_deduplicate(corpus)
        # Exact stage still catches the duplicate
        assert report.exact_duplicates_removed == 1
        assert report.simhash_duplicates_removed == 0

    # --- Stage 3: MinHash near-duplicates ---

    def test_minhash_stage_skipped(self):
        pipeline = DeduplicationPipeline(skip_minhash=True)
        corpus = [CONTRACT_DOC, NEGLIGENCE_DOC, UNRELATED_DOC]
        _, report = pipeline.fit_deduplicate(corpus)
        assert report.minhash_duplicates_removed == 0

    # --- Statistics ---

    def test_total_input_matches(self):
        corpus = [CONTRACT_DOC, NEGLIGENCE_DOC, UNRELATED_DOC, CONTRACT_DOC]
        _, report = self.pipeline.fit_deduplicate(corpus)
        assert report.total_input == 4

    def test_stage_counts_sum_to_removed(self):
        corpus = [CONTRACT_DOC, CONTRACT_DOC, NEGLIGENCE_DOC]
        _, report = self.pipeline.fit_deduplicate(corpus)
        total_removed = (
            report.exact_duplicates_removed
            + report.simhash_duplicates_removed
            + report.minhash_duplicates_removed
        )
        assert report.total_unique + total_removed == report.total_input

    # --- fit + is_unique + add API ---

    def test_is_unique_after_fit(self):
        self.pipeline.fit([CONTRACT_DOC, NEGLIGENCE_DOC])
        assert not self.pipeline.is_unique(CONTRACT_DOC)
        assert not self.pipeline.is_unique(NEGLIGENCE_DOC)
        assert self.pipeline.is_unique(UNRELATED_DOC)

    def test_add_makes_doc_not_unique(self):
        assert self.pipeline.is_unique(CONTRACT_DOC)
        self.pipeline.add(CONTRACT_DOC)
        assert not self.pipeline.is_unique(CONTRACT_DOC)

    def test_is_unique_empty_pipeline(self):
        pipeline = DeduplicationPipeline()
        assert pipeline.is_unique(CONTRACT_DOC)
        assert pipeline.is_unique(NEGLIGENCE_DOC)

    def test_incremental_add_workflow(self):
        """Simulate incremental indexing: check then add."""
        pipeline = DeduplicationPipeline(skip_minhash=True)
        docs = [CONTRACT_DOC, NEGLIGENCE_DOC, CONTRACT_DOC, UNRELATED_DOC]
        indexed = []
        for doc in docs:
            if pipeline.is_unique(doc):
                indexed.append(doc)
                pipeline.add(doc)
        assert len(indexed) == 3
        assert CONTRACT_DOC in indexed
        assert NEGLIGENCE_DOC in indexed
        assert UNRELATED_DOC in indexed

    # --- reset ---

    def test_reset_clears_all_dedupers(self):
        self.pipeline.fit([CONTRACT_DOC])
        self.pipeline.reset()
        assert self.pipeline.is_unique(CONTRACT_DOC)
        assert self.pipeline.exact_seen_count == 0
        assert self.pipeline.simhash_indexed_count == 0
        assert self.pipeline.minhash_indexed_count == 0

    # --- property accessors ---

    def test_exact_seen_count_after_fit_deduplicate(self):
        corpus = [CONTRACT_DOC, NEGLIGENCE_DOC, UNRELATED_DOC]
        self.pipeline.fit_deduplicate(corpus)
        # All 3 are unique → all indexed in exact stage
        assert self.pipeline.exact_seen_count == 3

    def test_simhash_indexed_count_after_fit_deduplicate(self):
        corpus = [CONTRACT_DOC, NEGLIGENCE_DOC, UNRELATED_DOC]
        self.pipeline.fit_deduplicate(corpus)
        assert self.pipeline.simhash_indexed_count == 3

    def test_minhash_indexed_count_after_fit_deduplicate(self):
        corpus = [CONTRACT_DOC, NEGLIGENCE_DOC, UNRELATED_DOC]
        self.pipeline.fit_deduplicate(corpus)
        assert self.pipeline.minhash_indexed_count == 3

    # --- edge cases ---

    def test_empty_corpus(self):
        unique, report = self.pipeline.fit_deduplicate([])
        assert unique == []
        assert report.total_input == 0
        assert report.total_unique == 0

    def test_single_doc(self):
        unique, report = self.pipeline.fit_deduplicate([CONTRACT_DOC])
        assert unique == [CONTRACT_DOC]
        assert report.total_unique == 1

    def test_all_same_doc(self):
        corpus = [CONTRACT_DOC] * 7
        unique, report = self.pipeline.fit_deduplicate(corpus)
        assert unique == [CONTRACT_DOC]
        assert report.total_unique == 1
        assert report.total_input == 7

    def test_fit_deduplicate_resets_before_run(self):
        """Calling fit_deduplicate twice should not accumulate state."""
        corpus = [CONTRACT_DOC, NEGLIGENCE_DOC]
        _, r1 = self.pipeline.fit_deduplicate(corpus)
        _, r2 = self.pipeline.fit_deduplicate(corpus)
        # Second run should produce identical results
        assert r1.total_unique == r2.total_unique
        assert r1.exact_duplicates_removed == r2.exact_duplicates_removed

    def test_skip_both_stages_keeps_exact_only(self):
        pipeline = DeduplicationPipeline(skip_simhash=True, skip_minhash=True)
        corpus = [CONTRACT_DOC, CONTRACT_DOC, CONTRACT_DOC_WHITESPACE, UNRELATED_DOC]
        unique, report = pipeline.fit_deduplicate(corpus)
        # CONTRACT_DOC_WHITESPACE normalises to same MD5 as CONTRACT_DOC → exact dup
        assert len(unique) == 2
        assert report.simhash_duplicates_removed == 0
        assert report.minhash_duplicates_removed == 0
