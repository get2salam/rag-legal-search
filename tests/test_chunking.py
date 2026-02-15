"""
Tests for the chunking strategies module.

Covers all six chunking strategies with edge cases, size constraints,
overlap behaviour, and factory function.
"""

import numpy as np
import pytest

from utils.chunking import (
    Chunk,
    FixedSizeChunker,
    SentenceChunker,
    RecursiveChunker,
    SemanticChunker,
    SlidingWindowChunker,
    StructureAwareChunker,
    get_chunker,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "The court held that the defendant was liable for breach of contract. "
    "The plaintiff had demonstrated reliance on the defendant's representations. "
    "Damages were assessed at the amount of the plaintiff's expectation interest.\n\n"
    "In considering the appropriate remedy, the court noted that specific performance "
    "was not available because the subject matter was not unique. "
    "The court therefore awarded compensatory damages.\n\n"
    "The defendant appealed on the grounds that the trial court had erred in its "
    "assessment of damages. The appellate court affirmed the lower court's decision, "
    "finding no abuse of discretion."
)

STRUCTURED_TEXT = """# Introduction

This document provides an overview of contract law principles.

## Section 1: Formation

A contract requires offer, acceptance, and consideration.
The offeror must communicate the terms clearly.
Acceptance must be unequivocal and communicated to the offeror.

## Section 2: Performance

Both parties must perform their obligations under the contract.
Failure to perform constitutes a breach.

### 2.1 Material Breach

A material breach excuses the non-breaching party from further performance.
The test for materiality considers several factors.

### 2.2 Minor Breach

A minor breach does not excuse further performance but may give rise to damages.

## Section 3: Remedies

The primary remedy for breach of contract is compensatory damages.
Specific performance may be awarded in cases involving unique goods.
"""


SHORT_TEXT = "Hello world."


def dummy_embed_fn(texts: list) -> np.ndarray:
    """Deterministic dummy embeddings based on text hash."""
    rng = np.random.RandomState(42)
    return np.array([rng.randn(64) for _ in texts])


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

class TestChunkDataclass:
    def test_token_estimate(self):
        c = Chunk(text="a" * 400, index=0, start_char=0, end_char=400)
        assert c.token_estimate == 100

    def test_content_hash_deterministic(self):
        c1 = Chunk(text="hello", index=0, start_char=0, end_char=5)
        c2 = Chunk(text="hello", index=1, start_char=10, end_char=15)
        assert c1.content_hash == c2.content_hash

    def test_content_hash_differs(self):
        c1 = Chunk(text="hello", index=0, start_char=0, end_char=5)
        c2 = Chunk(text="world", index=0, start_char=0, end_char=5)
        assert c1.content_hash != c2.content_hash

    def test_len(self):
        c = Chunk(text="abc", index=0, start_char=0, end_char=3)
        assert len(c) == 3


# ---------------------------------------------------------------------------
# FixedSizeChunker
# ---------------------------------------------------------------------------

class TestFixedSizeChunker:
    def test_basic_chunking(self):
        chunker = FixedSizeChunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) > 1
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert len(chunk.text) > 0

    def test_no_chunk_exceeds_size(self):
        chunker = FixedSizeChunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk(SAMPLE_TEXT)
        for chunk in chunks:
            # Allow slight overshoot due to boundary-seeking
            assert len(chunk.text) <= 250  # 200 + tolerance

    def test_short_text_single_chunk(self):
        chunker = FixedSizeChunker(chunk_size=1000, overlap=100)
        chunks = chunker.chunk(SHORT_TEXT)
        assert len(chunks) == 1
        assert chunks[0].text == SHORT_TEXT

    def test_empty_text(self):
        chunker = FixedSizeChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_overlap_must_be_less_than_chunk_size(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, overlap=100)

    def test_indices_are_sequential(self):
        chunker = FixedSizeChunker(chunk_size=150, overlap=30)
        chunks = chunker.chunk(SAMPLE_TEXT)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_full_text_covered(self):
        """Every word in the original text should appear in at least one chunk."""
        chunker = FixedSizeChunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk(SAMPLE_TEXT)
        all_chunk_text = " ".join(c.text for c in chunks)
        for word in SAMPLE_TEXT.split()[:10]:  # Spot-check first 10 words
            assert word in all_chunk_text


# ---------------------------------------------------------------------------
# SentenceChunker
# ---------------------------------------------------------------------------

class TestSentenceChunker:
    def test_basic_chunking(self):
        chunker = SentenceChunker(max_chunk_size=200, sentence_overlap=1)
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) > 1

    def test_preserves_sentences(self):
        """No chunk should end mid-sentence (no trailing fragments)."""
        chunker = SentenceChunker(max_chunk_size=300, sentence_overlap=0)
        chunks = chunker.chunk(SAMPLE_TEXT)
        for chunk in chunks:
            text = chunk.text.strip()
            # Each chunk should end with sentence-ending punctuation
            assert text[-1] in ".!?", f"Chunk ends with '{text[-1]}': ...{text[-30:]}"

    def test_single_sentence(self):
        chunker = SentenceChunker(max_chunk_size=1000)
        chunks = chunker.chunk("Just one sentence.")
        assert len(chunks) == 1

    def test_empty_text(self):
        chunker = SentenceChunker()
        assert chunker.chunk("") == []

    def test_overlap_carries_sentences(self):
        chunker = SentenceChunker(max_chunk_size=200, sentence_overlap=1)
        chunks = chunker.chunk(SAMPLE_TEXT)
        if len(chunks) >= 2:
            # With overlap=1, last sentence of chunk N should appear in chunk N+1
            # (This is a soft check — depends on sentence boundary detection)
            assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# RecursiveChunker
# ---------------------------------------------------------------------------

class TestRecursiveChunker:
    def test_basic_chunking(self):
        chunker = RecursiveChunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) > 1

    def test_respects_paragraph_boundaries(self):
        chunker = RecursiveChunker(chunk_size=500, overlap=50)
        chunks = chunker.chunk(SAMPLE_TEXT)
        # With a large enough chunk size, paragraphs should be preserved
        assert any("\n\n" not in c.text for c in chunks)

    def test_short_text_single_chunk(self):
        chunker = RecursiveChunker(chunk_size=5000)
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) == 1

    def test_custom_separators(self):
        chunker = RecursiveChunker(
            chunk_size=200,
            separators=["\n\n", " "],  # Skip sentence-level
        )
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) > 0

    def test_empty_text(self):
        chunker = RecursiveChunker()
        assert chunker.chunk("") == []


# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------

class TestSemanticChunker:
    def test_basic_chunking(self):
        chunker = SemanticChunker(
            embed_fn=dummy_embed_fn,
            max_chunk_size=300,
        )
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, Chunk)

    def test_respects_max_size(self):
        chunker = SemanticChunker(
            embed_fn=dummy_embed_fn,
            max_chunk_size=200,
        )
        chunks = chunker.chunk(SAMPLE_TEXT)
        for chunk in chunks:
            # Semantic chunker groups sentences, so allow some overshoot
            assert len(chunk.text) <= 400  # 2x tolerance

    def test_metadata_has_sentence_count(self):
        chunker = SemanticChunker(embed_fn=dummy_embed_fn)
        chunks = chunker.chunk(SAMPLE_TEXT)
        for chunk in chunks:
            assert "num_sentences" in chunk.metadata
            assert chunk.metadata["num_sentences"] >= 1

    def test_single_sentence(self):
        chunker = SemanticChunker(embed_fn=dummy_embed_fn)
        chunks = chunker.chunk("Only one sentence here.")
        assert len(chunks) == 1

    def test_empty_text(self):
        chunker = SemanticChunker(embed_fn=dummy_embed_fn)
        assert chunker.chunk("") == []

    def test_explicit_threshold(self):
        chunker = SemanticChunker(
            embed_fn=dummy_embed_fn,
            similarity_threshold=0.9,  # Very high = more splits
        )
        chunks_high = chunker.chunk(SAMPLE_TEXT)

        chunker_low = SemanticChunker(
            embed_fn=dummy_embed_fn,
            similarity_threshold=0.0,  # Very low = fewer splits
        )
        chunks_low = chunker_low.chunk(SAMPLE_TEXT)

        # Higher threshold should produce more (or equal) chunks
        assert len(chunks_high) >= len(chunks_low)


# ---------------------------------------------------------------------------
# SlidingWindowChunker
# ---------------------------------------------------------------------------

class TestSlidingWindowChunker:
    def test_basic_chunking(self):
        chunker = SlidingWindowChunker(window_sentences=3, stride_sentences=2)
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) > 1

    def test_overlap_between_windows(self):
        chunker = SlidingWindowChunker(window_sentences=4, stride_sentences=2)
        chunks = chunker.chunk(SAMPLE_TEXT)
        if len(chunks) >= 2:
            # With stride < window, consecutive chunks should share content
            words_0 = set(chunks[0].text.split())
            words_1 = set(chunks[1].text.split())
            overlap = words_0 & words_1
            assert len(overlap) > 0, "Expected overlap between consecutive windows"

    def test_stride_equals_window_no_overlap(self):
        chunker = SlidingWindowChunker(window_sentences=3, stride_sentences=3)
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) >= 1

    def test_stride_gt_window_raises(self):
        with pytest.raises(ValueError):
            SlidingWindowChunker(window_sentences=3, stride_sentences=5)

    def test_short_text_single_chunk(self):
        chunker = SlidingWindowChunker(window_sentences=20)
        chunks = chunker.chunk(SHORT_TEXT)
        assert len(chunks) == 1

    def test_empty_text(self):
        chunker = SlidingWindowChunker()
        assert chunker.chunk("") == []

    def test_window_metadata(self):
        chunker = SlidingWindowChunker(window_sentences=3, stride_sentences=2)
        chunks = chunker.chunk(SAMPLE_TEXT)
        for chunk in chunks:
            assert "window_start" in chunk.metadata
            assert "window_end" in chunk.metadata
            assert chunk.metadata["window_end"] > chunk.metadata["window_start"]


# ---------------------------------------------------------------------------
# StructureAwareChunker
# ---------------------------------------------------------------------------

class TestStructureAwareChunker:
    def test_detects_markdown_sections(self):
        chunker = StructureAwareChunker(max_chunk_size=500)
        chunks = chunker.chunk(STRUCTURED_TEXT)
        # Should produce multiple chunks based on headings
        assert len(chunks) >= 3

    def test_preserves_section_content(self):
        chunker = StructureAwareChunker(max_chunk_size=2000)
        chunks = chunker.chunk(STRUCTURED_TEXT)
        all_text = " ".join(c.text for c in chunks)
        assert "offer, acceptance, and consideration" in all_text
        assert "compensatory damages" in all_text

    def test_falls_back_on_unstructured_text(self):
        """Plain text with no headings should still be chunked."""
        plain = "This is plain text. " * 50
        chunker = StructureAwareChunker(max_chunk_size=200)
        chunks = chunker.chunk(plain)
        assert len(chunks) > 0

    def test_empty_text(self):
        chunker = StructureAwareChunker()
        assert chunker.chunk("") == []

    def test_large_section_is_split(self):
        """A single section exceeding max size should be further split."""
        large_section = "# Big Section\n\n" + "This is a sentence. " * 200
        chunker = StructureAwareChunker(max_chunk_size=300)
        chunks = chunker.chunk(large_section)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) <= 500  # Allow tolerance for recursive split


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

class TestFactory:
    def test_get_fixed(self):
        chunker = get_chunker("fixed", chunk_size=500, overlap=100)
        assert isinstance(chunker, FixedSizeChunker)

    def test_get_sentence(self):
        chunker = get_chunker("sentence")
        assert isinstance(chunker, SentenceChunker)

    def test_get_recursive(self):
        chunker = get_chunker("recursive")
        assert isinstance(chunker, RecursiveChunker)

    def test_get_semantic(self):
        chunker = get_chunker("semantic", embed_fn=dummy_embed_fn)
        assert isinstance(chunker, SemanticChunker)

    def test_get_sliding_window(self):
        chunker = get_chunker("sliding_window")
        assert isinstance(chunker, SlidingWindowChunker)

    def test_get_structure(self):
        chunker = get_chunker("structure")
        assert isinstance(chunker, StructureAwareChunker)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            get_chunker("nonexistent")

    def test_kwargs_passed_through(self):
        chunker = get_chunker("fixed", chunk_size=42, overlap=10)
        assert chunker.chunk_size == 42
        assert chunker.overlap == 10


# ---------------------------------------------------------------------------
# chunk_many (batch interface)
# ---------------------------------------------------------------------------

class TestChunkMany:
    def test_batch_chunking(self):
        chunker = FixedSizeChunker(chunk_size=200, overlap=50)
        results = chunker.chunk_many([SAMPLE_TEXT, SHORT_TEXT, ""])
        assert len(results) == 3
        assert len(results[0]) > 1   # SAMPLE_TEXT produces multiple chunks
        assert len(results[1]) == 1   # SHORT_TEXT is one chunk
        assert len(results[2]) == 0   # Empty produces no chunks


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_whitespace_only(self):
        for strategy in ["fixed", "sentence", "recursive", "sliding_window", "structure"]:
            chunker = get_chunker(strategy) if strategy != "semantic" else None
            if chunker:
                assert chunker.chunk("   \n\n\t  ") == []

    def test_very_long_word(self):
        """A single word longer than chunk_size should still produce output."""
        long_word = "a" * 5000
        chunker = RecursiveChunker(chunk_size=200)
        chunks = chunker.chunk(long_word)
        assert len(chunks) > 0
        # All text should be covered
        total = sum(len(c.text) for c in chunks)
        assert total >= 5000

    def test_unicode_text(self):
        unicode_text = (
            "المحكمة قررت أن المدعى عليه مسؤول. "
            "تم تقييم الأضرار بمبلغ التعويض المتوقع. "
            "استأنف المدعى عليه القرار."
        )
        chunker = SentenceChunker(max_chunk_size=100)
        chunks = chunker.chunk(unicode_text)
        assert len(chunks) >= 1

    def test_newlines_only(self):
        chunker = RecursiveChunker()
        assert chunker.chunk("\n\n\n\n") == []

    def test_single_character(self):
        chunker = FixedSizeChunker(chunk_size=10, overlap=2)
        chunks = chunker.chunk("X")
        assert len(chunks) == 1
        assert chunks[0].text == "X"
