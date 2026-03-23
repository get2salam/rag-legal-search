"""
Document deduplication and near-duplicate detection for RAG pipelines.

Deduplication is essential before indexing to prevent vector stores from being
polluted with near-identical content, which inflates index size, biases
retrieval toward over-represented topics, and wastes embedding API budget.

Three complementary strategies are provided:

``ExactDedupe``
    MD5 fingerprint for perfect bit-for-bit duplicate detection.  O(n) time.

``SimHashDedupe``
    Charikar's SimHash (2002) — projects each document into a compact 64-bit
    fingerprint that preserves *cosine similarity*: documents that differ by
    fewer than ``hamming_threshold`` bits are considered near-duplicates.
    Ideal for text corpora with minor reformatting or whitespace artefacts.

``MinHashDedupe``
    Broder's MinHash sketch (1997) estimates *Jaccard similarity* between
    document shingle sets.  Documents with estimated Jaccard ≥
    ``similarity_threshold`` are flagged as near-duplicates.  Effective for
    content that has been paraphrased or lightly reworded.

``DeduplicationPipeline``
    Chains ExactDedupe → SimHashDedupe → MinHashDedupe in sequence, with
    per-stage counters and an incremental ``is_unique`` / ``add`` API.

Usage::

    from utils.deduplication import DeduplicationPipeline

    deduper = DeduplicationPipeline(
        hamming_threshold=3,
        minhash_threshold=0.8,
        minhash_n_hashes=128,
        shingle_size=3,
    )
    unique_docs, report = deduper.fit_deduplicate(corpus)
    print(report)  # DeduplicationResult(input=500, unique=423, ...)

    # Incremental API — check + index one document at a time
    deduper2 = DeduplicationPipeline()
    deduper2.fit(existing_corpus)
    for doc in new_docs:
        if deduper2.is_unique(doc):
            vector_store.add(doc)
            deduper2.add(doc)
"""

from __future__ import annotations

import hashlib
import random
import re
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import FrozenSet, Iterable, List, Tuple

# ---------------------------------------------------------------------------
# Stop-word list
# ---------------------------------------------------------------------------

_STOP_WORDS: FrozenSet[str] = frozenset(
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
        "i",
        "we",
        "he",
        "she",
        "they",
        "what",
        "which",
        "who",
    }
)

# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """Return lowercase alphanumeric tokens with stop-words and single chars removed."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def _shingle(text: str, k: int = 3) -> List[str]:
    """Return overlapping k-word shingles from *text*.

    Shingles (k-grams of words) capture local phrase order and produce a
    better near-duplicate signal than a simple bag-of-words approach.

    Args:
        text: Input document string.
        k:    Shingle size in words.  Default ``3``.

    Returns:
        List of shingle strings, or unigrams if fewer than *k* content tokens.
    """
    words = _tokenize(text)
    if len(words) < k:
        return list(words)  # fallback to unigrams for short texts
    return [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class DeduplicationResult:
    """Summary statistics from a :class:`DeduplicationPipeline` run."""

    total_input: int = 0
    total_unique: int = 0
    exact_duplicates_removed: int = 0
    simhash_duplicates_removed: int = 0
    minhash_duplicates_removed: int = 0
    duplicate_pairs: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def duplicate_rate(self) -> float:
        """Fraction of input documents classified as duplicates."""
        if self.total_input == 0:
            return 0.0
        return (self.total_input - self.total_unique) / self.total_input

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"DeduplicationResult("
            f"input={self.total_input}, unique={self.total_unique}, "
            f"exact={self.exact_duplicates_removed}, "
            f"simhash={self.simhash_duplicates_removed}, "
            f"minhash={self.minhash_duplicates_removed}, "
            f"rate={self.duplicate_rate:.1%})"
        )


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------


class BaseDedupe(ABC):
    """Abstract base class for deduplication strategies."""

    @abstractmethod
    def fit(self, texts: Iterable[str]) -> "BaseDedupe":
        """Index *texts* into the seen-set without deduplicating.

        Returns:
            ``self`` for method chaining.
        """

    @abstractmethod
    def is_duplicate(self, text: str) -> bool:
        """Return ``True`` if *text* matches something already indexed."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal state."""

    @abstractmethod
    def _add(self, text: str) -> None:
        """Add *text* to the seen-set (internal; called after uniqueness check)."""

    def fit_deduplicate(self, texts: Iterable[str]) -> List[str]:
        """Index *texts* and return only the unique documents in input order.

        Args:
            texts: Iterable of document strings.

        Returns:
            Deduplicated list preserving first-occurrence order.
        """
        self.reset()
        unique: List[str] = []
        for text in texts:
            if not self.is_duplicate(text):
                unique.append(text)
                self._add(text)
        return unique


# ---------------------------------------------------------------------------
# ExactDedupe
# ---------------------------------------------------------------------------


class ExactDedupe(BaseDedupe):
    """Detect exact duplicates using MD5 content fingerprints.

    Time complexity: O(n · |text|) for hashing.
    Space complexity: O(n · 16) for 16-byte MD5 digests.

    Args:
        normalize: If ``True`` (default), collapse internal whitespace before
                   hashing so extra spaces or ``\\r\\n`` artefacts do not
                   produce false negatives.
    """

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize
        self._seen: set[bytes] = set()

    def _fingerprint(self, text: str) -> bytes:
        s = " ".join(text.split()) if self.normalize else text
        return hashlib.md5(s.encode(), usedforsecurity=False).digest()

    # -- BaseDedupe --

    def fit(self, texts: Iterable[str]) -> "ExactDedupe":
        for text in texts:
            self._seen.add(self._fingerprint(text))
        return self

    def is_duplicate(self, text: str) -> bool:
        return self._fingerprint(text) in self._seen

    def reset(self) -> None:
        self._seen.clear()

    def _add(self, text: str) -> None:
        self._seen.add(self._fingerprint(text))

    @property
    def seen_count(self) -> int:
        """Number of unique fingerprints currently indexed."""
        return len(self._seen)


# ---------------------------------------------------------------------------
# SimHash helpers
# ---------------------------------------------------------------------------

_HASH_BITS = 64


def _sha256_int(text: str) -> int:
    """Return a 64-bit integer from the first 8 bytes of SHA-256(*text*)."""
    digest = hashlib.sha256(text.encode()).digest()
    return struct.unpack(">Q", digest[:8])[0]


def simhash(text: str, shingle_size: int = 3) -> int:
    """Compute a 64-bit SimHash fingerprint for *text*.

    SimHash maps a document to a compact integer such that the **Hamming
    distance** between two fingerprints correlates with the **cosine
    distance** between the documents' shingle frequency vectors.

    Algorithm (Charikar 2002):

    1. Tokenise *text* into overlapping k-word shingles.
    2. For each shingle compute a 64-bit hash.
    3. For each bit position keep a running sum: +1 if the bit is set in
       the shingle hash, −1 otherwise.
    4. The final SimHash bit is 1 iff the running sum is positive.

    Args:
        text:         Input document string.
        shingle_size: k-word shingle width.  Default ``3``.

    Returns:
        64-bit integer SimHash fingerprint (0 for empty/stop-word-only text).
    """
    shingles = _shingle(text, k=shingle_size)
    if not shingles:
        return 0

    sums = [0] * _HASH_BITS
    for s in shingles:
        h = _sha256_int(s)
        for i in range(_HASH_BITS):
            sums[i] += 1 if (h >> (_HASH_BITS - 1 - i)) & 1 else -1

    fingerprint = 0
    for i, v in enumerate(sums):
        if v > 0:
            fingerprint |= 1 << (_HASH_BITS - 1 - i)
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    """Count bit positions where *a* and *b* differ (Hamming distance)."""
    xor = a ^ b
    count = 0
    while xor:
        count += xor & 1
        xor >>= 1
    return count


# ---------------------------------------------------------------------------
# SimHashDedupe
# ---------------------------------------------------------------------------


class SimHashDedupe(BaseDedupe):
    """Near-duplicate detection via SimHash Hamming distance.

    Two documents are considered near-duplicates when the Hamming distance
    between their SimHash fingerprints is ≤ ``hamming_threshold``.

    Typical thresholds:

    * ``0`` — only bit-identical fingerprints (very strict)
    * ``3`` — common default; catches reformatting and single-word changes
    * ``≥ 8`` — aggressive; may flag loosely related documents

    Args:
        hamming_threshold: Maximum Hamming distance to classify as duplicate.
            Default ``3``.
        shingle_size:      k-word shingle size passed to :func:`simhash`.
            Default ``3``.

    Note:
        Uses a linear scan over indexed fingerprints.  For very large corpora
        (> 1 M docs) consider building a banding LSH index on top.
    """

    def __init__(
        self,
        hamming_threshold: int = 3,
        shingle_size: int = 3,
    ) -> None:
        if hamming_threshold < 0:
            raise ValueError(f"hamming_threshold must be >= 0, got {hamming_threshold}")
        self.hamming_threshold = hamming_threshold
        self.shingle_size = shingle_size
        self._fingerprints: List[int] = []

    def _fp(self, text: str) -> int:
        return simhash(text, shingle_size=self.shingle_size)

    def _is_near_dup(self, fp: int) -> bool:
        return any(
            hamming_distance(fp, seen) <= self.hamming_threshold
            for seen in self._fingerprints
        )

    # -- BaseDedupe --

    def fit(self, texts: Iterable[str]) -> "SimHashDedupe":
        for text in texts:
            self._fingerprints.append(self._fp(text))
        return self

    def is_duplicate(self, text: str) -> bool:
        return self._is_near_dup(self._fp(text))

    def reset(self) -> None:
        self._fingerprints.clear()

    def _add(self, text: str) -> None:
        self._fingerprints.append(self._fp(text))

    def fingerprint(self, text: str) -> int:
        """Return the 64-bit SimHash fingerprint for *text* (public utility)."""
        return self._fp(text)

    @property
    def indexed_count(self) -> int:
        """Number of fingerprints currently indexed."""
        return len(self._fingerprints)


# ---------------------------------------------------------------------------
# MinHash helpers
# ---------------------------------------------------------------------------

_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def _minhash_signature(
    shingles: List[str],
    n_hashes: int,
    hash_params: List[Tuple[int, int]],
) -> List[int]:
    """Compute an *n_hashes*-dimensional MinHash signature.

    Uses the standard linear-hashing trick::

        h_i(x) = (a_i * hash(x) + b_i) % p

    where *p* is a Mersenne prime for fast modular arithmetic.

    Args:
        shingles:     List of shingle strings for the document.
        n_hashes:     Length of the signature vector.
        hash_params:  Pre-generated list of (a, b) pairs, one per hash fn.

    Returns:
        List of *n_hashes* minimum hash values (all ``_MERSENNE_PRIME`` if
        *shingles* is empty — a sentinel larger than any valid hash output).
    """
    # Sentinel: _MERSENNE_PRIME is the modulus so h is always in
    # [0, _MERSENNE_PRIME - 1]; the sentinel is guaranteed to be replaced
    # by the first valid hash encountered.
    sig = [_MERSENNE_PRIME] * n_hashes
    for s in shingles:
        x = (
            int(hashlib.md5(s.encode(), usedforsecurity=False).hexdigest(), 16)
            & _MAX_HASH
        )
        for i, (a, b) in enumerate(hash_params):
            h = (a * x + b) % _MERSENNE_PRIME
            if h < sig[i]:
                sig[i] = h
    return sig


def _jaccard_from_signatures(sig_a: List[int], sig_b: List[int]) -> float:
    """Estimate Jaccard similarity from two equal-length MinHash signatures."""
    if not sig_a or not sig_b:
        return 0.0
    return sum(a == b for a, b in zip(sig_a, sig_b)) / len(sig_a)


# ---------------------------------------------------------------------------
# MinHashDedupe
# ---------------------------------------------------------------------------


class MinHashDedupe(BaseDedupe):
    """Near-duplicate detection via MinHash Jaccard similarity estimation.

    Each document is represented as a compact *n_hashes*-length integer
    vector.  Estimated Jaccard similarity ≥ ``similarity_threshold`` between
    a candidate and any indexed document classifies the candidate as a
    near-duplicate.

    MinHash is especially effective for:

    * Paraphrased or lightly reworded content
    * Documents with sentence reordering
    * Scraped content with minor editorial changes

    Args:
        similarity_threshold: Minimum estimated Jaccard to classify as
            duplicate.  Range ``(0, 1]``.  Default ``0.8``.
        n_hashes:             Signature length.  Higher values → better
            accuracy but more memory.  Default ``128``.
        shingle_size:         k-word shingle size.  Default ``3``.
        seed:                 Seed for reproducible hash parameters.
            Default ``42``.

    Accuracy:
        Standard deviation of the Jaccard estimate:
        ``σ = sqrt(J * (1 - J) / n_hashes)``.
        For J=0.8, n_hashes=128: σ ≈ 0.035.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        n_hashes: int = 128,
        shingle_size: int = 3,
        seed: int = 42,
    ) -> None:
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be in (0, 1], got {similarity_threshold}"
            )
        if n_hashes < 1:
            raise ValueError(f"n_hashes must be >= 1, got {n_hashes}")
        if shingle_size < 1:
            raise ValueError(f"shingle_size must be >= 1, got {shingle_size}")

        self.similarity_threshold = similarity_threshold
        self.n_hashes = n_hashes
        self.shingle_size = shingle_size
        self.seed = seed

        rng = random.Random(seed)
        self._hash_params: List[Tuple[int, int]] = [
            (
                rng.randint(1, _MERSENNE_PRIME - 1),
                rng.randint(0, _MERSENNE_PRIME - 1),
            )
            for _ in range(n_hashes)
        ]
        self._signatures: List[List[int]] = []

    def _signature(self, text: str) -> List[int]:
        shingles = _shingle(text, k=self.shingle_size)
        return _minhash_signature(shingles, self.n_hashes, self._hash_params)

    def _is_near_dup(self, sig: List[int]) -> bool:
        return any(
            _jaccard_from_signatures(sig, seen) >= self.similarity_threshold
            for seen in self._signatures
        )

    # -- BaseDedupe --

    def fit(self, texts: Iterable[str]) -> "MinHashDedupe":
        for text in texts:
            self._signatures.append(self._signature(text))
        return self

    def is_duplicate(self, text: str) -> bool:
        return self._is_near_dup(self._signature(text))

    def reset(self) -> None:
        self._signatures.clear()

    def _add(self, text: str) -> None:
        self._signatures.append(self._signature(text))

    def estimate_jaccard(self, text_a: str, text_b: str) -> float:
        """Estimate the Jaccard similarity between *text_a* and *text_b*."""
        sig_a = self._signature(text_a)
        sig_b = self._signature(text_b)
        return _jaccard_from_signatures(sig_a, sig_b)

    @property
    def indexed_count(self) -> int:
        """Number of signatures currently indexed."""
        return len(self._signatures)


# ---------------------------------------------------------------------------
# DeduplicationPipeline
# ---------------------------------------------------------------------------


class DeduplicationPipeline:
    """Sequential deduplication pipeline: Exact → SimHash → MinHash.

    Each stage targets a different class of duplicates:

    1. **Exact** — bit-for-bit identical documents (whitespace normalised)
    2. **SimHash** — minor reformatting, tag stripping, whitespace changes
    3. **MinHash** — paraphrased or lightly reworded content

    Documents that pass all three stages are considered unique.  Per-stage
    counters in :class:`DeduplicationResult` explain what was removed and
    why.

    Args:
        hamming_threshold:  SimHash Hamming distance threshold.  Default ``3``.
        minhash_threshold:  MinHash Jaccard similarity threshold.  Default ``0.8``.
        minhash_n_hashes:   MinHash signature length.  Default ``128``.
        shingle_size:       Shared k-word shingle size.  Default ``3``.
        skip_simhash:       Disable the SimHash stage.  Default ``False``.
        skip_minhash:       Disable the MinHash stage.  Default ``False``.
    """

    def __init__(
        self,
        hamming_threshold: int = 3,
        minhash_threshold: float = 0.8,
        minhash_n_hashes: int = 128,
        shingle_size: int = 3,
        skip_simhash: bool = False,
        skip_minhash: bool = False,
    ) -> None:
        self.exact = ExactDedupe()
        self.simhash_dedupe = SimHashDedupe(
            hamming_threshold=hamming_threshold,
            shingle_size=shingle_size,
        )
        self.minhash_dedupe = MinHashDedupe(
            similarity_threshold=minhash_threshold,
            n_hashes=minhash_n_hashes,
            shingle_size=shingle_size,
        )
        self.skip_simhash = skip_simhash
        self.skip_minhash = skip_minhash

    def reset(self) -> None:
        """Clear all internal state from all three dedupers."""
        self.exact.reset()
        self.simhash_dedupe.reset()
        self.minhash_dedupe.reset()

    def fit(self, texts: Iterable[str]) -> "DeduplicationPipeline":
        """Index *texts* into all three dedupers without deduplicating.

        Use this to prime the pipeline on an existing corpus before calling
        :meth:`is_unique` on new documents.

        Returns:
            ``self`` for method chaining.
        """
        texts_list = list(texts)
        self.exact.fit(texts_list)
        self.simhash_dedupe.fit(texts_list)
        self.minhash_dedupe.fit(texts_list)
        return self

    def is_unique(self, text: str) -> bool:
        """Return ``True`` if *text* is unique w.r.t. all currently indexed docs.

        Does **not** add *text* to the index.  Use :meth:`add` after confirming
        uniqueness if you want future documents to be checked against it.
        """
        if self.exact.is_duplicate(text):
            return False
        if not self.skip_simhash and self.simhash_dedupe.is_duplicate(text):
            return False
        if not self.skip_minhash and self.minhash_dedupe.is_duplicate(text):
            return False
        return True

    def add(self, text: str) -> None:
        """Index *text* in all three dedupers (without checking uniqueness)."""
        self.exact._add(text)
        self.simhash_dedupe._add(text)
        self.minhash_dedupe._add(text)

    def fit_deduplicate(
        self,
        texts: Iterable[str],
    ) -> Tuple[List[str], DeduplicationResult]:
        """Deduplicate *texts* and return unique documents + a statistics report.

        The pipeline processes each document through three stages in order,
        stopping at the first stage that classifies it as a duplicate.
        Only documents that pass all active stages are added to the internal
        indexes (so duplicates of duplicates are still detected transitively).

        Args:
            texts: Iterable of document strings.

        Returns:
            Tuple of ``(unique_texts, DeduplicationResult)`` where
            ``unique_texts`` preserves the original first-occurrence order.
        """
        self.reset()
        texts_list = list(texts)
        result = DeduplicationResult(total_input=len(texts_list))
        unique: List[str] = []

        for text in texts_list:
            # Stage 1: exact duplicate
            if self.exact.is_duplicate(text):
                result.exact_duplicates_removed += 1
                continue

            # Stage 2: SimHash near-duplicate
            if not self.skip_simhash and self.simhash_dedupe.is_duplicate(text):
                result.simhash_duplicates_removed += 1
                # Track exact fingerprint so future exact copies are caught
                self.exact._add(text)
                continue

            # Stage 3: MinHash near-duplicate
            if not self.skip_minhash and self.minhash_dedupe.is_duplicate(text):
                result.minhash_duplicates_removed += 1
                self.exact._add(text)
                self.simhash_dedupe._add(text)
                continue

            # Passed all active stages — unique
            unique.append(text)
            self.exact._add(text)
            self.simhash_dedupe._add(text)
            self.minhash_dedupe._add(text)

        result.total_unique = len(unique)
        return unique, result

    # -- Convenience properties --

    @property
    def exact_seen_count(self) -> int:
        """Total distinct exact fingerprints indexed across all stages."""
        return self.exact.seen_count

    @property
    def simhash_indexed_count(self) -> int:
        """Total SimHash fingerprints indexed."""
        return self.simhash_dedupe.indexed_count

    @property
    def minhash_indexed_count(self) -> int:
        """Total MinHash signatures indexed."""
        return self.minhash_dedupe.indexed_count


# ---------------------------------------------------------------------------
# Standalone utilities (exported for direct use)
# ---------------------------------------------------------------------------

__all__ = [
    # helpers
    "_tokenize",
    "_shingle",
    "_minhash_signature",
    "_jaccard_from_signatures",
    "hamming_distance",
    "simhash",
    # classes
    "BaseDedupe",
    "ExactDedupe",
    "SimHashDedupe",
    "MinHashDedupe",
    "DeduplicationPipeline",
    "DeduplicationResult",
]
