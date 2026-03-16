"""
Contextual compression for RAG pipelines.

Post-retrieval compression extracts only the sentences (or spans) from each
retrieved chunk that are *directly relevant* to the query.  This reduces the
context window sent to the LLM, cuts cost, and often improves answer quality
by removing distracting content.

Compressors
-----------
- ``SentenceScoreCompressor``
    Scores every sentence against the query with a TF-IDF–style term-overlap
    approach.  Keeps the top-scoring sentences up to a character budget.

- ``KeywordWindowCompressor``
    Locates query-keyword occurrences and expands a ±N-sentence window around
    each hit.  Good for exact-match retrieval where context around keywords
    matters most.

- ``LLMCompressor``
    Delegates extraction to an LLM (Anthropic or OpenAI-compatible) which is
    asked to return only the spans relevant to the query.  Highest quality
    but uses tokens; recommended as a final step after a cheaper pre-filter.

- ``CompressionPipeline``
    Chains compressors in order, feeding each one's output into the next.

Usage::

    from utils.compression import SentenceScoreCompressor, CompressionPipeline

    compressor = SentenceScoreCompressor(max_chars=600, min_score=0.1)
    results = compressor.compress(query="breach of contract remedy", chunks=retrieved)
    # Each result now has a smaller 'text' and metadata about what was kept.
"""

from __future__ import annotations

import math
import os
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CompressedChunk:
    """A retrieval result after contextual compression."""

    text: str
    """The compressed text (subset of the original chunk)."""

    original_text: str
    """The full original chunk text before compression."""

    score: float
    """Original retrieval similarity score (pass-through)."""

    metadata: Dict = field(default_factory=dict)
    """Arbitrary metadata from the retrieval result, plus compression info."""

    @property
    def compression_ratio(self) -> float:
        """Fraction of original length retained (1.0 = no compression)."""
        if not self.original_text:
            return 1.0
        return len(self.text) / len(self.original_text)

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "original_text": self.original_text,
            "score": self.score,
            "compression_ratio": round(self.compression_ratio, 3),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Shared helpers
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
        "can",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "not",
        "no",
        "nor",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "any",
        "all",
        "some",
        "such",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "when",
        "where",
        "how",
        "there",
        "their",
        "they",
        "he",
        "she",
        "we",
        "i",
        "you",
        "me",
        "him",
        "her",
        "us",
        "my",
        "your",
        "his",
        "our",
        "than",
        "then",
    }
)


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokeniser, strips stop words."""
    tokens = re.findall(r"\b[a-z]{2,}\b", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS]


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using a two-pass abbreviation-protection approach.

    Protects common abbreviations before splitting on [.!?] + whitespace.
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
        "cf.",
        "e.g.",
        "i.e.",
    ]
    protected = text
    for i, abbr in enumerate(abbrevs):
        protected = protected.replace(abbr, f"\x00ABBR{i}\x00")

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z([\"])", protected)

    result: List[str] = []
    for part in parts:
        restored = part
        for i, abbr in enumerate(abbrevs):
            restored = restored.replace(f"\x00ABBR{i}\x00", abbr)
        restored = restored.strip()
        if restored:
            result.append(restored)
    return result or [text.strip()]


def _idf_weights(sentences: List[List[str]]) -> Dict[str, float]:
    """Compute inverse-document-frequency weights over a sentence corpus."""
    N = max(len(sentences), 1)
    df: Counter = Counter()
    for sent_tokens in sentences:
        for tok in set(sent_tokens):
            df[tok] += 1
    return {tok: math.log(N / (count + 1)) + 1.0 for tok, count in df.items()}


def _score_sentence(
    query_tokens: List[str],
    sent_tokens: List[str],
    idf: Dict[str, float],
) -> float:
    """
    Score a sentence against query terms using TF-IDF cosine similarity.

    Both vectors are unit-normalised so the result is in [0, 1].
    """
    if not query_tokens or not sent_tokens:
        return 0.0

    sent_tf = Counter(sent_tokens)
    query_set = set(query_tokens)

    dot = sum(idf.get(tok, 1.0) * sent_tf[tok] for tok in sent_tf if tok in query_set)
    sent_mag = math.sqrt(
        sum((idf.get(tok, 1.0) * cnt) ** 2 for tok, cnt in sent_tf.items())
    )
    query_mag = math.sqrt(sum(idf.get(tok, 1.0) ** 2 for tok in query_tokens))
    denom = sent_mag * query_mag
    return dot / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Base compressor
# ---------------------------------------------------------------------------


class BaseCompressor(ABC):
    """Abstract base for all contextual compressors."""

    @abstractmethod
    def compress(
        self,
        query: str,
        chunks: Sequence[Dict],
    ) -> List[CompressedChunk]:
        """
        Compress a list of retrieval results relative to *query*.

        Each item in *chunks* should be a dict with at least:
            - ``"text"``  : str  — chunk text
            - ``"score"`` : float — retrieval similarity score (optional)
        Additional keys are passed through as metadata.

        Returns a list of :class:`CompressedChunk` objects, one per input.
        """

    def _make_compressed(
        self,
        original: Dict,
        compressed_text: str,
        extra_meta: Optional[Dict] = None,
    ) -> CompressedChunk:
        meta = {k: v for k, v in original.items() if k not in {"text", "score"}}
        if extra_meta:
            meta.update(extra_meta)
        return CompressedChunk(
            text=compressed_text or original.get("text", ""),
            original_text=original.get("text", ""),
            score=original.get("score", 0.0),
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# 1. Sentence-score compressor
# ---------------------------------------------------------------------------


class SentenceScoreCompressor(BaseCompressor):
    """
    Extract the highest-scoring sentences from each chunk.

    Sentences are ranked by TF-IDF cosine similarity against the query.
    The top sentences are selected in *document order* (not score order)
    until the character budget is reached.

    Parameters
    ----------
    max_chars : int
        Maximum characters to retain per chunk. Default 600.
    min_score : float
        Minimum sentence score to be eligible. Default 0.05.
    min_sentences : int
        Always keep at least this many sentences even if below min_score.
        Default 1.
    """

    def __init__(
        self,
        max_chars: int = 600,
        min_score: float = 0.05,
        min_sentences: int = 1,
    ) -> None:
        self.max_chars = max_chars
        self.min_score = min_score
        self.min_sentences = min_sentences

    def compress(
        self,
        query: str,
        chunks: Sequence[Dict],
    ) -> List[CompressedChunk]:
        if not chunks:
            return []

        query_tokens = _tokenize(query)
        results: List[CompressedChunk] = []

        for chunk in chunks:
            text = chunk.get("text", "")
            sentences = _split_sentences(text)
            tokenized_sents = [_tokenize(s) for s in sentences]

            idf = _idf_weights(tokenized_sents)
            scores = [
                _score_sentence(query_tokens, toks, idf) for toks in tokenized_sents
            ]

            # Rank and select
            ranked = sorted(
                range(len(sentences)),
                key=lambda i: scores[i],
                reverse=True,
            )

            # Always include at least min_sentences
            eligible = [i for i in ranked if scores[i] >= self.min_score]
            if len(eligible) < self.min_sentences:
                eligible = ranked[: self.min_sentences]

            # Greedily fill budget, preserve document order
            eligible_set = set(eligible)
            budget = self.max_chars
            selected: List[int] = []
            for i in sorted(eligible_set):
                if budget <= 0:
                    break
                selected.append(i)
                budget -= len(sentences[i])

            compressed_text = " ".join(sentences[i] for i in selected)
            results.append(
                self._make_compressed(
                    chunk,
                    compressed_text,
                    extra_meta={
                        "kept_sentences": len(selected),
                        "total_sentences": len(sentences),
                        "compressor": "sentence_score",
                    },
                )
            )

        return results


# ---------------------------------------------------------------------------
# 2. Keyword-window compressor
# ---------------------------------------------------------------------------


class KeywordWindowCompressor(BaseCompressor):
    """
    Keep a ±``window`` sentence context around each query-keyword hit.

    Good for keyword-heavy queries where the LLM needs the surrounding
    sentences for context (e.g. "what did the court hold on damages?").

    Parameters
    ----------
    window : int
        Number of sentences to include before and after each hit. Default 2.
    max_chars : int
        Hard cap on output length. Default 800.
    """

    def __init__(self, window: int = 2, max_chars: int = 800) -> None:
        self.window = window
        self.max_chars = max_chars

    def compress(
        self,
        query: str,
        chunks: Sequence[Dict],
    ) -> List[CompressedChunk]:
        if not chunks:
            return []

        query_tokens = set(_tokenize(query))
        results: List[CompressedChunk] = []

        for chunk in chunks:
            text = chunk.get("text", "")
            sentences = _split_sentences(text)
            tokenized_sents = [set(_tokenize(s)) for s in sentences]

            # Find sentences that contain at least one query token
            hit_indices = {
                i for i, toks in enumerate(tokenized_sents) if toks & query_tokens
            }

            # Expand windows
            keep: set = set()
            for hi in hit_indices:
                for offset in range(-self.window, self.window + 1):
                    idx = hi + offset
                    if 0 <= idx < len(sentences):
                        keep.add(idx)

            if not keep:
                # Nothing matched — fall back to first two sentences
                keep = {0, 1} if len(sentences) > 1 else {0}

            # Collect in document order, respect budget
            compressed_parts: List[str] = []
            char_count = 0
            prev_idx: Optional[int] = None

            for i in sorted(keep):
                if char_count >= self.max_chars:
                    break
                # Add ellipsis if there is a gap between kept sentences
                if prev_idx is not None and i > prev_idx + 1:
                    compressed_parts.append("...")
                compressed_parts.append(sentences[i])
                char_count += len(sentences[i])
                prev_idx = i

            compressed_text = " ".join(compressed_parts)
            results.append(
                self._make_compressed(
                    chunk,
                    compressed_text,
                    extra_meta={
                        "kept_sentences": len(keep),
                        "total_sentences": len(sentences),
                        "hit_count": len(hit_indices),
                        "compressor": "keyword_window",
                    },
                )
            )

        return results


# ---------------------------------------------------------------------------
# 3. LLM compressor
# ---------------------------------------------------------------------------


class LLMCompressor(BaseCompressor):
    """
    Use an LLM to extract only the relevant spans from each chunk.

    The LLM is instructed to return verbatim sentences that answer or
    support the query.  If the chunk is already short (≤ *skip_if_under*
    characters) it is passed through unchanged to save tokens.

    Parameters
    ----------
    model : str
        Model identifier (e.g. ``"claude-3-haiku-20240307"`` or
        ``"gpt-4o-mini"``).  Defaults to the ``LLM_COMPRESSION_MODEL``
        env var, then ``"gpt-4o-mini"``.
    max_tokens : int
        Max tokens for the extraction response.  Default 512.
    skip_if_under : int
        Skip compression if chunk is already ≤ this many chars. Default 300.
    """

    _SYSTEM = (
        "You are a legal document analyst. "
        "Extract only the sentences from the provided text that directly answer "
        "or support the user's query. "
        "Return the extracted sentences verbatim, separated by spaces. "
        "If nothing is relevant, return the single word: NONE."
    )

    def __init__(
        self,
        model: Optional[str] = None,
        max_tokens: int = 512,
        skip_if_under: int = 300,
    ) -> None:
        self.model = model or os.getenv("LLM_COMPRESSION_MODEL", "gpt-4o-mini")
        self.max_tokens = max_tokens
        self.skip_if_under = skip_if_under
        self._client: Optional[object] = None
        self._provider: Optional[str] = None

    def _init_client(self) -> None:
        if self._client is not None:
            return
        if self.model.startswith("claude"):
            from anthropic import Anthropic  # type: ignore[import]

            self._client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self._provider = "anthropic"
        else:
            from openai import OpenAI  # type: ignore[import]

            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self._provider = "openai"

    def _call_llm(self, query: str, chunk_text: str) -> str:
        self._init_client()
        user_msg = f"QUERY: {query}\n\nTEXT:\n{chunk_text}"
        try:
            if self._provider == "anthropic":
                resp = self._client.messages.create(  # type: ignore[union-attr]
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=self._SYSTEM,
                    messages=[{"role": "user", "content": user_msg}],
                )
                return resp.content[0].text.strip()
            else:
                resp = self._client.chat.completions.create(  # type: ignore[union-attr]
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                )
                return resp.choices[0].message.content.strip()
        except Exception:
            return chunk_text  # fail-open: return original

    def compress(
        self,
        query: str,
        chunks: Sequence[Dict],
    ) -> List[CompressedChunk]:
        if not chunks:
            return []

        results: List[CompressedChunk] = []
        for chunk in chunks:
            text = chunk.get("text", "")
            if len(text) <= self.skip_if_under:
                results.append(
                    self._make_compressed(
                        chunk,
                        text,
                        extra_meta={"compressor": "llm_skipped"},
                    )
                )
                continue

            extracted = self._call_llm(query, text)
            if extracted.upper() == "NONE" or not extracted:
                extracted = text[: self.skip_if_under]

            results.append(
                self._make_compressed(
                    chunk,
                    extracted,
                    extra_meta={"compressor": "llm"},
                )
            )

        return results


# ---------------------------------------------------------------------------
# 4. Compression pipeline
# ---------------------------------------------------------------------------


class CompressionPipeline(BaseCompressor):
    """
    Chain multiple compressors sequentially.

    Each compressor's output is passed as input to the next one.
    The ``text`` field of each :class:`CompressedChunk` from stage N
    becomes the ``text`` used in stage N+1, while ``original_text``
    always reflects the very first input.

    Example::

        pipeline = CompressionPipeline([
            SentenceScoreCompressor(max_chars=1000, min_score=0.05),
            KeywordWindowCompressor(window=1, max_chars=600),
        ])
        results = pipeline.compress(query, chunks)
    """

    def __init__(self, compressors: List[BaseCompressor]) -> None:
        if not compressors:
            raise ValueError("Pipeline requires at least one compressor.")
        self.compressors = compressors

    def compress(
        self,
        query: str,
        chunks: Sequence[Dict],
    ) -> List[CompressedChunk]:
        if not chunks:
            return []

        # Preserve original texts from the very first input
        originals = [c.get("text", "") for c in chunks]

        current: List[Dict] = list(chunks)
        final: List[CompressedChunk] = []

        for step, compressor in enumerate(self.compressors):
            compressed = compressor.compress(query, current)
            if step < len(self.compressors) - 1:
                # Feed into next stage: use compressed text as new 'text'
                current = [
                    {**c.metadata, "text": c.text, "score": c.score} for c in compressed
                ]
            else:
                final = compressed

        # Restore original_text from the first-stage inputs
        patched: List[CompressedChunk] = []
        for orig_text, result in zip(originals, final):
            patched.append(
                CompressedChunk(
                    text=result.text,
                    original_text=orig_text,
                    score=result.score,
                    metadata={
                        **result.metadata,
                        "pipeline_steps": len(self.compressors),
                    },
                )
            )

        return patched


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_COMPRESSOR_MAP = {
    "sentence_score": SentenceScoreCompressor,
    "keyword_window": KeywordWindowCompressor,
    "llm": LLMCompressor,
}


def get_compressor(strategy: str = "sentence_score", **kwargs) -> BaseCompressor:
    """
    Return a compressor instance by strategy name.

    Parameters
    ----------
    strategy : str
        One of ``"sentence_score"``, ``"keyword_window"``, ``"llm"``.
    **kwargs
        Forwarded to the compressor constructor.

    Raises
    ------
    ValueError
        If *strategy* is not recognised.
    """
    cls = _COMPRESSOR_MAP.get(strategy)
    if cls is None:
        valid = ", ".join(sorted(_COMPRESSOR_MAP.keys()))
        raise ValueError(f"Unknown compression strategy '{strategy}'. Valid: {valid}")
    return cls(**kwargs)
