"""
Tests for the contextual compression module.

Covers:
- SentenceScoreCompressor: extraction, budget, min_score, min_sentences
- KeywordWindowCompressor: window logic, gap ellipsis, no-hit fallback
- CompressionPipeline: chaining, original_text preservation, step count
- CompressedChunk: compression_ratio, to_dict
- get_compressor: factory success and error paths
- Edge cases: empty input, single sentence, no-match query
"""

from __future__ import annotations

import pytest

from utils.compression import (
    CompressedChunk,
    CompressionPipeline,
    KeywordWindowCompressor,
    LLMCompressor,
    SentenceScoreCompressor,
    get_compressor,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LEGAL_TEXT = (
    "The court held that the defendant was liable for breach of contract. "
    "The plaintiff demonstrated clear reliance on the defendant's representations. "
    "Damages were assessed at the plaintiff's expectation interest. "
    "The appellate court affirmed the decision. "
    "The judge noted that specific performance was unavailable. "
    "Compensatory damages were therefore the appropriate remedy. "
    "The defendant's appeal was dismissed on all grounds. "
    "Costs were awarded to the plaintiff."
)

MULTI_PARA_TEXT = (
    "Formation requires offer, acceptance, and consideration.\n\n"
    "A material breach excuses the non-breaching party from further performance.\n\n"
    "The remedy for breach is typically compensatory damages.\n\n"
    "Courts may also award specific performance in equity."
)

CHUNK_BREACH = {
    "text": LEGAL_TEXT,
    "score": 0.82,
    "title": "Contract Breach Case",
    "id": "case_001",
}

CHUNK_SHORT = {
    "text": "The court dismissed the claim.",
    "score": 0.60,
    "id": "case_002",
}


def make_chunks(*texts: str) -> list:
    return [{"text": t, "score": 0.75} for t in texts]


# ---------------------------------------------------------------------------
# CompressedChunk
# ---------------------------------------------------------------------------


class TestCompressedChunk:
    def test_compression_ratio_full(self):
        c = CompressedChunk(text="hello world", original_text="hello world", score=1.0)
        assert c.compression_ratio == pytest.approx(1.0)

    def test_compression_ratio_half(self):
        c = CompressedChunk(text="ab", original_text="abcd", score=0.5)
        assert c.compression_ratio == pytest.approx(0.5)

    def test_compression_ratio_empty_original(self):
        c = CompressedChunk(text="", original_text="", score=0.0)
        assert c.compression_ratio == pytest.approx(1.0)

    def test_to_dict_keys(self):
        c = CompressedChunk(
            text="compressed",
            original_text="compressed original",
            score=0.9,
            metadata={"id": "x"},
        )
        d = c.to_dict()
        assert "text" in d
        assert "original_text" in d
        assert "score" in d
        assert "compression_ratio" in d
        assert "metadata" in d

    def test_to_dict_ratio_rounded(self):
        c = CompressedChunk(text="ab", original_text="abcde", score=0.0)
        d = c.to_dict()
        assert isinstance(d["compression_ratio"], float)
        assert d["compression_ratio"] == round(2 / 5, 3)


# ---------------------------------------------------------------------------
# SentenceScoreCompressor
# ---------------------------------------------------------------------------


class TestSentenceScoreCompressor:
    def test_returns_one_result_per_chunk(self):
        comp = SentenceScoreCompressor()
        results = comp.compress("breach damages", [CHUNK_BREACH, CHUNK_SHORT])
        assert len(results) == 2

    def test_result_is_compressed_chunk(self):
        comp = SentenceScoreCompressor()
        results = comp.compress("breach damages", [CHUNK_BREACH])
        assert isinstance(results[0], CompressedChunk)

    def test_score_passthrough(self):
        comp = SentenceScoreCompressor()
        results = comp.compress("breach", [CHUNK_BREACH])
        assert results[0].score == pytest.approx(0.82)

    def test_original_text_preserved(self):
        comp = SentenceScoreCompressor()
        results = comp.compress("breach", [CHUNK_BREACH])
        assert results[0].original_text == LEGAL_TEXT

    def test_max_chars_respected(self):
        comp = SentenceScoreCompressor(max_chars=100)
        results = comp.compress("breach damages", [CHUNK_BREACH])
        assert len(results[0].text) <= 100 + 50  # small tolerance for last sentence

    def test_relevant_sentences_prioritised(self):
        comp = SentenceScoreCompressor(max_chars=400)
        results = comp.compress("breach remedy damages", [CHUNK_BREACH])
        text = results[0].text.lower()
        assert any(word in text for word in ["breach", "damages", "remedy"])

    def test_empty_chunk_list(self):
        comp = SentenceScoreCompressor()
        assert comp.compress("query", []) == []

    def test_empty_text_in_chunk(self):
        comp = SentenceScoreCompressor()
        results = comp.compress("query", [{"text": "", "score": 0.5}])
        assert len(results) == 1
        # Empty text should not crash; text may be empty or fallback
        assert isinstance(results[0], CompressedChunk)

    def test_min_sentences_fallback(self):
        """Even when min_score is very high, min_sentences sentences are returned."""
        comp = SentenceScoreCompressor(min_score=0.99, min_sentences=1)
        results = comp.compress("xyzzy", [CHUNK_BREACH])
        assert results[0].text.strip() != ""

    def test_metadata_contains_compressor_key(self):
        comp = SentenceScoreCompressor()
        results = comp.compress("breach", [CHUNK_BREACH])
        assert results[0].metadata.get("compressor") == "sentence_score"

    def test_metadata_kept_and_total_sentences(self):
        comp = SentenceScoreCompressor()
        results = comp.compress("breach", [CHUNK_BREACH])
        meta = results[0].metadata
        assert "kept_sentences" in meta
        assert "total_sentences" in meta
        assert meta["kept_sentences"] <= meta["total_sentences"]

    def test_non_text_metadata_passed_through(self):
        """Extra fields like 'id' and 'title' survive compression."""
        comp = SentenceScoreCompressor()
        results = comp.compress("breach", [CHUNK_BREACH])
        # CHUNK_BREACH has 'title' and 'id'
        assert results[0].metadata.get("id") == "case_001"
        assert results[0].metadata.get("title") == "Contract Breach Case"

    def test_single_sentence_chunk(self):
        comp = SentenceScoreCompressor()
        chunk = {"text": "The court dismissed all claims.", "score": 0.7}
        results = comp.compress("dismissed", [chunk])
        assert "dismissed" in results[0].text.lower()


# ---------------------------------------------------------------------------
# KeywordWindowCompressor
# ---------------------------------------------------------------------------


class TestKeywordWindowCompressor:
    def test_returns_one_result_per_chunk(self):
        comp = KeywordWindowCompressor()
        results = comp.compress("breach damages", [CHUNK_BREACH, CHUNK_SHORT])
        assert len(results) == 2

    def test_hit_sentences_included(self):
        comp = KeywordWindowCompressor(window=0)
        results = comp.compress("breach", [CHUNK_BREACH])
        assert "breach" in results[0].text.lower()

    def test_window_expands_context(self):
        comp_w0 = KeywordWindowCompressor(window=0)
        comp_w2 = KeywordWindowCompressor(window=2)
        r0 = comp_w0.compress("breach", [CHUNK_BREACH])[0]
        r2 = comp_w2.compress("breach", [CHUNK_BREACH])[0]
        assert len(r2.text) >= len(r0.text)

    def test_max_chars_respected(self):
        comp = KeywordWindowCompressor(window=5, max_chars=100)
        results = comp.compress("breach", [CHUNK_BREACH])
        assert len(results[0].text) <= 200  # some tolerance for last sentence

    def test_no_match_fallback(self):
        """Queries with no keyword hits fall back to first 1-2 sentences."""
        comp = KeywordWindowCompressor(window=1)
        chunk = {"text": LEGAL_TEXT, "score": 0.5}
        results = comp.compress("xyzzy foobar", [chunk])
        assert results[0].text.strip() != ""

    def test_ellipsis_on_gap(self):
        """Ellipsis is inserted when there is a gap between kept windows."""
        # Use a long text with keywords far apart
        text = (
            "Alpha damages clause was invoked. "
            "Second sentence has nothing useful. "
            "Third sentence also unrelated. "
            "Fourth sentence too. "
            "Beta remedy awarded here."
        )
        comp = KeywordWindowCompressor(window=0, max_chars=500)
        chunk = {"text": text, "score": 0.6}
        results = comp.compress("damages remedy", [chunk])
        assert "..." in results[0].text

    def test_metadata_hit_count(self):
        comp = KeywordWindowCompressor()
        results = comp.compress("breach damages", [CHUNK_BREACH])
        assert "hit_count" in results[0].metadata
        assert results[0].metadata["hit_count"] >= 1

    def test_empty_chunk_list(self):
        comp = KeywordWindowCompressor()
        assert comp.compress("breach", []) == []

    def test_compression_ratio_leq_one(self):
        comp = KeywordWindowCompressor()
        results = comp.compress("breach", [CHUNK_BREACH])
        assert results[0].compression_ratio <= 1.0 + 1e-9

    def test_original_text_preserved(self):
        comp = KeywordWindowCompressor()
        results = comp.compress("breach", [CHUNK_BREACH])
        assert results[0].original_text == LEGAL_TEXT


# ---------------------------------------------------------------------------
# CompressionPipeline
# ---------------------------------------------------------------------------


class TestCompressionPipeline:
    def test_requires_at_least_one_compressor(self):
        with pytest.raises(ValueError, match="at least one"):
            CompressionPipeline([])

    def test_single_compressor_equivalent(self):
        solo = SentenceScoreCompressor(max_chars=400)
        pipeline = CompressionPipeline([SentenceScoreCompressor(max_chars=400)])
        q = "breach damages"
        chunks = [CHUNK_BREACH]
        r_solo = solo.compress(q, chunks)
        r_pipe = pipeline.compress(q, chunks)
        assert r_solo[0].text == r_pipe[0].text

    def test_two_stage_compression(self):
        pipeline = CompressionPipeline(
            [
                SentenceScoreCompressor(max_chars=800),
                KeywordWindowCompressor(window=1, max_chars=400),
            ]
        )
        results = pipeline.compress("breach damages", [CHUNK_BREACH])
        assert len(results) == 1
        assert isinstance(results[0], CompressedChunk)

    def test_pipeline_smaller_than_solo(self):
        """Two-stage should be ≤ single-stage in length."""
        solo = SentenceScoreCompressor(max_chars=800)
        pipeline = CompressionPipeline(
            [
                SentenceScoreCompressor(max_chars=800),
                KeywordWindowCompressor(window=1, max_chars=300),
            ]
        )
        q = "breach damages remedy"
        r_solo = solo.compress(q, [CHUNK_BREACH])[0]
        r_pipe = pipeline.compress(q, [CHUNK_BREACH])[0]
        assert len(r_pipe.text) <= len(r_solo.text) + 20

    def test_original_text_is_first_stage_input(self):
        """original_text must always equal the input to stage 1, not stage 2."""
        pipeline = CompressionPipeline(
            [
                SentenceScoreCompressor(max_chars=400),
                KeywordWindowCompressor(window=1, max_chars=200),
            ]
        )
        results = pipeline.compress("breach", [CHUNK_BREACH])
        assert results[0].original_text == LEGAL_TEXT

    def test_pipeline_steps_in_metadata(self):
        pipeline = CompressionPipeline(
            [
                SentenceScoreCompressor(),
                KeywordWindowCompressor(),
            ]
        )
        results = pipeline.compress("breach", [CHUNK_BREACH])
        assert results[0].metadata.get("pipeline_steps") == 2

    def test_empty_chunks(self):
        pipeline = CompressionPipeline([SentenceScoreCompressor()])
        assert pipeline.compress("breach", []) == []

    def test_multiple_chunks(self):
        pipeline = CompressionPipeline([SentenceScoreCompressor(max_chars=300)])
        results = pipeline.compress("breach", [CHUNK_BREACH, CHUNK_SHORT])
        assert len(results) == 2


# ---------------------------------------------------------------------------
# get_compressor factory
# ---------------------------------------------------------------------------


class TestGetCompressor:
    def test_sentence_score(self):
        comp = get_compressor("sentence_score")
        assert isinstance(comp, SentenceScoreCompressor)

    def test_keyword_window(self):
        comp = get_compressor("keyword_window")
        assert isinstance(comp, KeywordWindowCompressor)

    def test_llm(self):
        comp = get_compressor("llm")
        assert isinstance(comp, LLMCompressor)

    def test_kwargs_forwarded(self):
        comp = get_compressor("sentence_score", max_chars=123, min_score=0.2)
        assert comp.max_chars == 123
        assert comp.min_score == pytest.approx(0.2)

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="Unknown compression strategy"):
            get_compressor("nonexistent_strategy")

    def test_error_message_lists_valid_strategies(self):
        try:
            get_compressor("bad")
        except ValueError as exc:
            msg = str(exc)
            assert "keyword_window" in msg
            assert "sentence_score" in msg
            assert "llm" in msg


# ---------------------------------------------------------------------------
# Integration: compress → to_dict round-trip
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_sentence_score_to_dict_roundtrip(self):
        comp = SentenceScoreCompressor(max_chars=500)
        results = comp.compress("breach damages", [CHUNK_BREACH])
        d = results[0].to_dict()
        assert d["text"] != ""
        assert d["original_text"] == LEGAL_TEXT
        assert 0.0 <= d["compression_ratio"] <= 1.0

    def test_keyword_window_to_dict_roundtrip(self):
        comp = KeywordWindowCompressor(window=2)
        results = comp.compress("remedy compensatory", [CHUNK_BREACH])
        d = results[0].to_dict()
        assert "remedy" in d["text"].lower() or "compens" in d["text"].lower()

    def test_pipeline_full_flow(self):
        """End-to-end: score → window → to_dict."""
        pipeline = CompressionPipeline(
            [
                SentenceScoreCompressor(max_chars=500),
                KeywordWindowCompressor(window=1, max_chars=300),
            ]
        )
        chunks = [CHUNK_BREACH, CHUNK_SHORT]
        results = pipeline.compress("breach remedy", chunks)
        assert len(results) == 2
        for r in results:
            d = r.to_dict()
            assert d["compression_ratio"] <= 1.0 + 1e-9
            assert d["metadata"]["pipeline_steps"] == 2
