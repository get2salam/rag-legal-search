"""
Advanced document chunking strategies for RAG pipelines.

Provides multiple chunking approaches optimized for different document types
and retrieval scenarios. Each strategy implements the same interface for
easy swapping and benchmarking.

Strategies:
    - FixedSizeChunker: Simple character-based chunking with overlap
    - SentenceChunker: Sentence-boundary-aware chunking
    - RecursiveChunker: Hierarchical splitting (paragraphs → sentences → words)
    - SemanticChunker: Embedding-based chunking that groups semantically similar sentences
    - SlidingWindowChunker: Overlapping windows with configurable stride
    - StructureAwareChunker: Respects document structure (headings, sections, lists)
"""

from __future__ import annotations

import re
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Chunk data model
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A single chunk produced by a chunking strategy."""

    text: str
    index: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        """Rough token estimate (chars / 4)."""
        return max(1, len(self.text) // 4)

    @property
    def content_hash(self) -> str:
        """SHA-256 digest of the chunk text (useful for dedup)."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]

    def __len__(self) -> int:
        return len(self.text)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseChunker(ABC):
    """Abstract base for all chunking strategies."""

    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """Split *text* into a list of Chunk objects."""

    def chunk_many(self, texts: Sequence[str]) -> List[List[Chunk]]:
        """Chunk multiple documents."""
        return [self.chunk(t) for t in texts]

    # Helpers shared by subclasses ---------------------

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """
        Split text into sentences.

        Uses a two-pass approach: first protect known abbreviations with
        placeholders, split on sentence boundaries, then restore them.
        This avoids variable-width lookbehind issues across Python versions.
        """
        abbrevs = [
            "Mr.",
            "Mrs.",
            "Ms.",
            "Dr.",
            "Prof.",
            "Jr.",
            "Sr.",
            "Inc.",
            "Ltd.",
            "Corp.",
            "Co.",
            "No.",
            "Art.",
            "Sec.",
            "Vol.",
            "Ch.",
            "Cl.",
            "Para.",
            "Sched.",
            "Pt.",
            "vs.",
            "v.",
        ]
        # Replace abbreviations with placeholders
        protected = text
        for i, abbr in enumerate(abbrevs):
            protected = protected.replace(abbr, f"\x00ABBR{i}\x00")

        # Split on sentence-ending punctuation followed by whitespace
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z([\"])", protected)

        # Restore abbreviations
        result: List[str] = []
        for part in parts:
            restored = part
            for i, abbr in enumerate(abbrevs):
                restored = restored.replace(f"\x00ABBR{i}\x00", abbr)
            restored = restored.strip()
            if restored:
                result.append(restored)
        return result

    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        """Split text on double newlines (paragraph boundaries)."""
        parts = re.split(r"\n\s*\n", text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _merge_small(segments: List[str], min_size: int) -> List[str]:
        """Merge consecutive segments shorter than *min_size*."""
        if not segments:
            return segments
        merged: List[str] = []
        buffer = segments[0]
        for seg in segments[1:]:
            if len(buffer) < min_size:
                buffer = buffer + "\n\n" + seg
            else:
                merged.append(buffer)
                buffer = seg
        merged.append(buffer)
        return merged


# ---------------------------------------------------------------------------
# 1. Fixed-size chunking
# ---------------------------------------------------------------------------


class FixedSizeChunker(BaseChunker):
    """
    Split text into fixed character-length chunks with optional overlap.

    Tries to break at paragraph or sentence boundaries when possible.
    Falls back to word boundaries, then hard character split.
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        if overlap >= chunk_size:
            raise ValueError("overlap must be < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[Chunk]:
        if not text.strip():
            return []

        chunks: List[Chunk] = []
        start = 0
        idx = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Try to break at a paragraph boundary
            if end < len(text):
                para = text.rfind("\n\n", start + self.chunk_size // 2, end)
                if para != -1:
                    end = para
                else:
                    # Try sentence boundary
                    sent = max(
                        text.rfind(". ", start + self.chunk_size // 2, end),
                        text.rfind("? ", start + self.chunk_size // 2, end),
                        text.rfind("! ", start + self.chunk_size // 2, end),
                    )
                    if sent != -1:
                        end = sent + 1  # Include the punctuation
                    else:
                        # Try word boundary
                        space = text.rfind(" ", start + self.chunk_size // 2, end)
                        if space != -1:
                            end = space

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        index=idx,
                        start_char=start,
                        end_char=end,
                    )
                )
                idx += 1

            # Advance with overlap
            start = end - self.overlap if end < len(text) else len(text)

        return chunks


# ---------------------------------------------------------------------------
# 2. Sentence-based chunking
# ---------------------------------------------------------------------------


class SentenceChunker(BaseChunker):
    """
    Aggregate sentences into chunks up to *max_chunk_size* characters.

    Each chunk contains whole sentences — never splits mid-sentence.
    Overlap is expressed as a number of trailing sentences carried forward.
    """

    def __init__(
        self,
        max_chunk_size: int = 1000,
        sentence_overlap: int = 2,
        min_chunk_size: int = 100,
    ):
        self.max_chunk_size = max_chunk_size
        self.sentence_overlap = sentence_overlap
        self.min_chunk_size = min_chunk_size

    def chunk(self, text: str) -> List[Chunk]:
        if not text.strip():
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return [Chunk(text=text.strip(), index=0, start_char=0, end_char=len(text))]

        chunks: List[Chunk] = []
        current_sents: List[str] = []
        current_len = 0
        idx = 0
        char_pos = 0

        for sent in sentences:
            if current_len + len(sent) > self.max_chunk_size and current_sents:
                chunk_text = " ".join(current_sents)
                start = text.find(current_sents[0], char_pos)
                if start == -1:
                    start = char_pos
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        index=idx,
                        start_char=start,
                        end_char=start + len(chunk_text),
                    )
                )
                idx += 1
                char_pos = start + len(chunk_text)

                # Overlap: keep last N sentences
                overlap_sents = current_sents[-self.sentence_overlap :]
                current_sents = overlap_sents
                current_len = sum(len(s) for s in current_sents)

            current_sents.append(sent)
            current_len += len(sent)

        # Flush remaining
        if current_sents:
            chunk_text = " ".join(current_sents)
            start = text.find(current_sents[0], char_pos)
            if start == -1:
                start = char_pos
            # Merge into previous if too small
            if len(chunk_text) < self.min_chunk_size and chunks:
                prev = chunks[-1]
                merged = prev.text + " " + chunk_text
                chunks[-1] = Chunk(
                    text=merged,
                    index=prev.index,
                    start_char=prev.start_char,
                    end_char=start + len(chunk_text),
                )
            else:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        index=idx,
                        start_char=start,
                        end_char=start + len(chunk_text),
                    )
                )

        return chunks


# ---------------------------------------------------------------------------
# 3. Recursive chunking
# ---------------------------------------------------------------------------


class RecursiveChunker(BaseChunker):
    """
    Hierarchical splitting inspired by LangChain's RecursiveCharacterTextSplitter.

    Tries separators in order: double-newline → single-newline → sentence-end → space → char.
    This preserves document structure as much as possible while staying within size limits.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, text: str) -> List[Chunk]:
        if not text.strip():
            return []
        raw = self._recursive_split(text, self.separators)
        # Build Chunk objects with position tracking
        chunks: List[Chunk] = []
        search_from = 0
        for idx, piece in enumerate(raw):
            pos = text.find(piece, search_from)
            if pos == -1:
                pos = search_from
            chunks.append(
                Chunk(
                    text=piece,
                    index=idx,
                    start_char=pos,
                    end_char=pos + len(piece),
                )
            )
            search_from = pos + len(piece)
        return chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if not text.strip():
            return []
        if len(text) <= self.chunk_size:
            return [text.strip()]

        # Find the best separator
        sep = separators[0] if separators else ""
        remaining_seps = separators[1:] if len(separators) > 1 else [""]

        if sep == "":
            # Hard character split (last resort)
            return self._hard_split(text)

        parts = text.split(sep)
        parts = [p for p in parts if p.strip()]

        result: List[str] = []
        buffer = ""

        for part in parts:
            candidate = (buffer + sep + part).strip() if buffer else part.strip()

            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    result.append(buffer)
                # If the single part is too big, recurse with finer separators
                if len(part.strip()) > self.chunk_size:
                    result.extend(self._recursive_split(part.strip(), remaining_seps))
                    buffer = ""
                else:
                    buffer = part.strip()

        if buffer:
            result.append(buffer)

        return result

    def _hard_split(self, text: str) -> List[str]:
        """Character-level split as last resort."""
        step = max(1, self.chunk_size - self.overlap)
        pieces: List[str] = []
        for i in range(0, len(text), step):
            piece = text[i : i + self.chunk_size].strip()
            if piece:
                pieces.append(piece)
        return pieces


# ---------------------------------------------------------------------------
# 4. Semantic chunking
# ---------------------------------------------------------------------------


class SemanticChunker(BaseChunker):
    """
    Group consecutive sentences based on embedding similarity.

    Sentences are embedded, and chunk boundaries are placed where the
    cosine similarity between adjacent sentence groups drops below a
    dynamic threshold (percentile-based or absolute).

    Requires an embedding function: ``embed_fn(texts) -> np.ndarray``.
    """

    def __init__(
        self,
        embed_fn: Callable[[List[str]], np.ndarray],
        max_chunk_size: int = 1500,
        similarity_threshold: Optional[float] = None,
        percentile_cutoff: int = 25,
    ):
        self.embed_fn = embed_fn
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        self.percentile_cutoff = percentile_cutoff

    def chunk(self, text: str) -> List[Chunk]:
        if not text.strip():
            return []

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [Chunk(text=text.strip(), index=0, start_char=0, end_char=len(text))]

        # Embed all sentences
        embeddings = self.embed_fn(sentences)
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)

        # Compute pairwise cosine similarities between consecutive sentences
        similarities: List[float] = []
        for i in range(len(embeddings) - 1):
            a, b = embeddings[i], embeddings[i + 1]
            cos_sim = float(
                np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
            )
            similarities.append(cos_sim)

        # Determine threshold
        if self.similarity_threshold is not None:
            threshold = self.similarity_threshold
        else:
            threshold = float(np.percentile(similarities, self.percentile_cutoff))

        # Find split points
        split_indices = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]

        # Build groups
        groups: List[List[str]] = []
        prev = 0
        for split_idx in split_indices:
            groups.append(sentences[prev:split_idx])
            prev = split_idx
        groups.append(sentences[prev:])

        # Enforce max_chunk_size — merge tiny groups, split large ones
        merged_groups = self._enforce_size_limits(groups)

        # Convert to Chunk objects
        chunks: List[Chunk] = []
        search_from = 0
        for idx, group in enumerate(merged_groups):
            chunk_text = " ".join(group)
            pos = text.find(group[0], search_from)
            if pos == -1:
                pos = search_from
            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=idx,
                    start_char=pos,
                    end_char=pos + len(chunk_text),
                    metadata={"num_sentences": len(group)},
                )
            )
            search_from = pos + len(chunk_text)

        return chunks

    def _enforce_size_limits(self, groups: List[List[str]]) -> List[List[str]]:
        """Merge tiny groups and split oversized ones."""
        result: List[List[str]] = []

        for group in groups:
            group_text = " ".join(group)

            if len(group_text) > self.max_chunk_size:
                # Split large group in half, recursively
                mid = len(group) // 2
                result.extend(self._enforce_size_limits([group[:mid], group[mid:]]))
            elif (
                result
                and len(" ".join(result[-1]) + " " + group_text)
                <= self.max_chunk_size // 2
            ):
                # Merge small groups
                result[-1].extend(group)
            else:
                result.append(group)

        return result


# ---------------------------------------------------------------------------
# 5. Sliding window chunking
# ---------------------------------------------------------------------------


class SlidingWindowChunker(BaseChunker):
    """
    Fixed-window chunking with configurable stride.

    Unlike FixedSizeChunker which tries to break at boundaries,
    this creates uniformly-sized windows — useful for models that
    benefit from consistent input lengths (e.g., bi-encoders).

    Window and stride are expressed in **sentences** (not characters),
    producing more meaningful chunk boundaries.
    """

    def __init__(self, window_sentences: int = 5, stride_sentences: int = 3):
        if stride_sentences > window_sentences:
            raise ValueError("stride must be <= window size")
        self.window = window_sentences
        self.stride = stride_sentences

    def chunk(self, text: str) -> List[Chunk]:
        if not text.strip():
            return []

        sentences = self._split_sentences(text)
        if len(sentences) <= self.window:
            return [Chunk(text=text.strip(), index=0, start_char=0, end_char=len(text))]

        chunks: List[Chunk] = []
        search_from = 0

        for idx, start in enumerate(range(0, len(sentences), self.stride)):
            window_sents = sentences[start : start + self.window]
            if not window_sents:
                break

            chunk_text = " ".join(window_sents)
            pos = text.find(window_sents[0], search_from)
            if pos == -1:
                pos = search_from

            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=idx,
                    start_char=pos,
                    end_char=pos + len(chunk_text),
                    metadata={
                        "window_start": start,
                        "window_end": min(start + self.window, len(sentences)),
                    },
                )
            )
            # Only advance search_from by stride worth of text
            if start + self.stride < len(sentences):
                stride_text = " ".join(sentences[start : start + self.stride])
                search_from = pos + len(stride_text)

            # Stop if we've covered all sentences
            if start + self.window >= len(sentences):
                break

        return chunks


# ---------------------------------------------------------------------------
# 6. Structure-aware chunking
# ---------------------------------------------------------------------------


class StructureAwareChunker(BaseChunker):
    """
    Respects document structure markers (headings, numbered sections, lists).

    Designed for structured legal / technical documents that use headings,
    numbered sections, or bullet lists. Each structural section becomes
    a chunk (or is split further if it exceeds max size).

    Section patterns detected:
        - Markdown headings (# / ## / ###)
        - Numbered sections (1. / 1.1 / (a) / (i))
        - ALL-CAPS headings
        - Horizontal rules (---)
    """

    SECTION_PATTERN = re.compile(
        r"(?:^|\n)"
        r"(?:"
        r"#{1,4}\s+.+"  # Markdown headings
        r"|(?:\d+\.)+\s+[A-Z]"  # Numbered sections: 1. / 1.1.
        r"|\([a-z]\)\s+"  # Lettered subsections: (a) / (b)
        r"|\([ivxlcdm]+\)\s+"  # Roman numeral subsections: (i) / (ii)
        r"|[A-Z][A-Z\s]{5,}"  # ALL-CAPS headings (min 6 chars)
        r"|---+"  # Horizontal rules
        r")",
        re.MULTILINE,
    )

    def __init__(self, max_chunk_size: int = 1500, min_chunk_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(self, text: str) -> List[Chunk]:
        if not text.strip():
            return []

        # Find section boundaries
        matches = list(self.SECTION_PATTERN.finditer(text))

        if not matches:
            # No structure detected — fall back to recursive
            fallback = RecursiveChunker(chunk_size=self.max_chunk_size)
            return fallback.chunk(text)

        # Extract sections
        sections: List[str] = []
        positions: List[int] = []

        # Text before first section marker
        first_pos = matches[0].start()
        if first_pos > 0 and text[:first_pos].strip():
            sections.append(text[:first_pos].strip())
            positions.append(0)

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            if section_text:
                sections.append(section_text)
                positions.append(start)

        # Merge small sections, split large ones
        merged = self._merge_small(sections, self.min_chunk_size)

        # Build chunks
        chunks: List[Chunk] = []
        search_from = 0
        idx = 0

        for section in merged:
            if len(section) > self.max_chunk_size:
                # Split oversized section with recursive chunker
                sub = RecursiveChunker(chunk_size=self.max_chunk_size)
                for sub_chunk in sub.chunk(section):
                    pos = text.find(sub_chunk.text[:50], search_from)
                    if pos == -1:
                        pos = search_from
                    chunks.append(
                        Chunk(
                            text=sub_chunk.text,
                            index=idx,
                            start_char=pos,
                            end_char=pos + len(sub_chunk.text),
                            metadata={"source": "structure_split"},
                        )
                    )
                    idx += 1
                    search_from = pos + len(sub_chunk.text)
            else:
                pos = text.find(section[:50], search_from)
                if pos == -1:
                    pos = search_from
                chunks.append(
                    Chunk(
                        text=section,
                        index=idx,
                        start_char=pos,
                        end_char=pos + len(section),
                        metadata={"source": "structure"},
                    )
                )
                idx += 1
                search_from = pos + len(section)

        return chunks


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

STRATEGY_MAP = {
    "fixed": FixedSizeChunker,
    "sentence": SentenceChunker,
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
    "sliding_window": SlidingWindowChunker,
    "structure": StructureAwareChunker,
}


def get_chunker(strategy: str = "recursive", **kwargs) -> BaseChunker:
    """
    Factory function to get a chunker by strategy name.

    Args:
        strategy: One of 'fixed', 'sentence', 'recursive', 'semantic',
                  'sliding_window', 'structure'.
        **kwargs: Passed to the chunker constructor.

    Returns:
        An instance of the requested chunker.

    Raises:
        ValueError: If the strategy name is not recognized.
    """
    cls = STRATEGY_MAP.get(strategy)
    if cls is None:
        valid = ", ".join(sorted(STRATEGY_MAP.keys()))
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Valid: {valid}")
    return cls(**kwargs)
