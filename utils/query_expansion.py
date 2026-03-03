"""
Query expansion module for RAG search quality improvement.

Expands user queries with synonyms, acronym resolution, and pseudo-relevance
feedback (PRF) to improve recall without sacrificing precision.

Strategies
----------
- ``SynonymExpander``        – term-level synonym injection from a built-in vocab
- ``AcronymExpander``        – resolves common abbreviations to full phrases
- ``PRFExpander``            – Pseudo-Relevance Feedback using top retrieved docs
- ``QueryExpansionPipeline`` – chains multiple expanders in sequence

Usage::

    from utils.query_expansion import QueryExpansionPipeline, SynonymExpander, AcronymExpander

    pipeline = QueryExpansionPipeline([
        AcronymExpander(),
        SynonymExpander(max_synonyms_per_term=2),
    ])
    result = pipeline.expand("NLP contract breach remedy")
    print(result.expanded_query)
    # e.g. "natural language processing agreement breach violation remedy redress"
    print(result.added_terms)
    # {'natural language processing', 'agreement', 'violation', 'redress'}
"""

from __future__ import annotations

import re
import math
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ExpandedQuery:
    """Result of a query expansion operation."""

    original_query: str
    expanded_query: str
    added_terms: Set[str] = field(default_factory=set)
    expansion_steps: List[str] = field(default_factory=list)

    @property
    def all_terms(self) -> List[str]:
        """Whitespace-tokenised list of terms in the expanded query."""
        return self.expanded_query.lower().split()

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"ExpandedQuery(original={self.original_query!r}, "
            f"expanded={self.expanded_query!r}, "
            f"added={len(self.added_terms)} terms)"
        )


# ---------------------------------------------------------------------------
# Helpers
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


def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenisation; preserves multi-word phrases split on spaces."""
    return re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", text.lower())


def _meaningful_tokens(text: str) -> List[str]:
    """Return tokens minus stop-words and single characters."""
    return [t for t in _tokenize(text) if t not in _STOP_WORDS and len(t) > 1]


def _term_freq(tokens: List[str]) -> Counter:
    return Counter(tokens)


def _idf(term: str, doc_token_lists: List[List[str]], n_docs: int) -> float:
    """BM25-style IDF: log((N - df + 0.5) / (df + 0.5) + 1)."""
    df = sum(1 for tl in doc_token_lists if term in tl)
    return math.log((n_docs - df + 0.5) / (df + 0.5) + 1)


# ---------------------------------------------------------------------------
# Built-in synonym vocabulary
# ---------------------------------------------------------------------------

#: Mapping from canonical term → set of synonyms.
#: Terms are stored lowercase; multi-word synonyms are space-separated.
_SYNONYM_VOCAB: Dict[str, List[str]] = {
    # General legal
    "agreement": ["contract", "covenant", "deed", "pact", "arrangement"],
    "contract": ["agreement", "covenant", "deed", "pact"],
    "breach": ["violation", "infringement", "contravention", "default"],
    "violation": ["breach", "infringement", "contravention", "transgression"],
    "remedy": ["relief", "redress", "compensation", "cure", "restitution"],
    "damages": ["compensation", "restitution", "indemnity", "reparation"],
    "liability": ["obligation", "responsibility", "accountability", "culpability"],
    "negligence": ["carelessness", "recklessness", "fault", "dereliction"],
    "duty": ["obligation", "responsibility", "burden"],
    "obligation": ["duty", "responsibility", "commitment", "undertaking"],
    "jurisdiction": ["authority", "competence", "venue"],
    "statute": ["legislation", "act", "law", "enactment", "ordinance"],
    "legislation": ["statute", "act", "law", "code", "regulation"],
    "regulation": ["rule", "statute", "directive", "ordinance", "provision"],
    "appeal": ["review", "challenge", "application", "petition"],
    "judgment": ["decision", "ruling", "verdict", "order", "decree"],
    "ruling": ["judgment", "decision", "verdict", "determination", "finding"],
    "injunction": ["restraining order", "order", "prohibition", "interdict"],
    "plaintiff": ["claimant", "complainant", "petitioner", "applicant"],
    "defendant": ["respondent", "accused", "accused party"],
    "evidence": ["proof", "testimony", "documentation", "exhibit"],
    "testimony": ["evidence", "statement", "deposition", "affidavit"],
    "precedent": ["authority", "case law", "ruling", "decision"],
    "tort": ["wrong", "civil wrong", "delict"],
    "fraud": ["deception", "misrepresentation", "deceit", "dishonesty"],
    "consent": ["agreement", "approval", "assent", "acquiescence"],
    # Search & retrieval
    "search": ["retrieve", "find", "query", "look up", "locate"],
    "retrieve": ["fetch", "extract", "obtain", "pull"],
    "query": ["search", "request", "inquiry", "question"],
    "document": ["record", "file", "text", "paper", "report"],
    "result": ["outcome", "finding", "match", "hit"],
    "index": ["catalogue", "inventory", "register", "database"],
    "rank": ["order", "sort", "prioritise", "score"],
    "relevance": ["pertinence", "applicability", "appropriateness"],
    # ML / NLP
    "embedding": ["vector", "representation", "encoding", "feature"],
    "vector": ["embedding", "representation", "array"],
    "similarity": ["resemblance", "likeness", "closeness", "proximity"],
    "classification": ["categorisation", "labelling", "tagging"],
    "summarise": ["summarize", "condense", "abstract", "distil"],
    "summarize": ["summarise", "condense", "abstract", "distil"],
    "extract": ["retrieve", "obtain", "pull", "derive"],
    "analyse": ["analyze", "examine", "assess", "evaluate", "review"],
    "analyze": ["analyse", "examine", "assess", "evaluate", "review"],
    "accuracy": ["precision", "correctness", "fidelity", "exactness"],
    "performance": ["efficiency", "effectiveness", "speed", "throughput"],
    "model": ["algorithm", "system", "architecture", "network"],
    "training": ["learning", "fitting", "optimisation"],
    "inference": ["prediction", "scoring", "evaluation"],
    # General
    "improve": ["enhance", "optimise", "optimize", "refine", "boost"],
    "optimise": ["improve", "enhance", "refine", "tune"],
    "optimize": ["improve", "enhance", "refine", "tune"],
    "error": ["fault", "mistake", "defect", "bug", "issue"],
    "issue": ["problem", "error", "concern", "matter"],
    "problem": ["issue", "challenge", "difficulty", "obstacle"],
    "solution": ["resolution", "answer", "approach", "fix"],
    "method": ["approach", "technique", "procedure", "strategy"],
    "process": ["procedure", "workflow", "operation", "pipeline"],
    "data": ["information", "records", "content", "dataset"],
    "information": ["data", "knowledge", "content", "details"],
    "report": ["document", "analysis", "summary", "finding"],
    "review": ["assessment", "evaluation", "examination", "audit"],
    "update": ["modify", "amend", "revise", "change"],
    "remove": ["delete", "eliminate", "drop", "exclude"],
    "add": ["include", "insert", "append", "incorporate"],
}

#: Build reverse index: synonym → canonical term (for look-ups from either direction)
_REVERSE_SYNONYM: Dict[str, str] = {}
for _canonical, _syns in _SYNONYM_VOCAB.items():
    for _syn in _syns:
        _key = _syn.replace(" ", "_")
        if _key not in _REVERSE_SYNONYM:
            _REVERSE_SYNONYM[_key] = _canonical


# ---------------------------------------------------------------------------
# Built-in acronym / abbreviation dictionary
# ---------------------------------------------------------------------------

_ACRONYM_MAP: Dict[str, str] = {
    # NLP / ML
    "nlp": "natural language processing",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "llm": "large language model",
    "rag": "retrieval augmented generation",
    "bert": "bidirectional encoder representations transformers",
    "gpt": "generative pre-trained transformer",
    "tfidf": "term frequency inverse document frequency",
    "tf-idf": "term frequency inverse document frequency",
    "bm25": "best match 25",
    "ndcg": "normalised discounted cumulative gain",
    "mrr": "mean reciprocal rank",
    "map": "mean average precision",
    "rrf": "reciprocal rank fusion",
    "mmr": "maximal marginal relevance",
    "ner": "named entity recognition",
    "pos": "part of speech",
    "ocr": "optical character recognition",
    "api": "application programming interface",
    "sdk": "software development kit",
    "cli": "command line interface",
    "ui": "user interface",
    "ux": "user experience",
    "ci": "continuous integration",
    "cd": "continuous deployment",
    "sql": "structured query language",
    "nosql": "non relational database",
    # Legal / domain
    "ip": "intellectual property",
    "ipc": "intellectual property code",
    "nda": "non disclosure agreement",
    "tos": "terms of service",
    "gdpr": "general data protection regulation",
    "adr": "alternative dispute resolution",
    "mou": "memorandum of understanding",
    "sla": "service level agreement",
    "ipo": "initial public offering",
    "m&a": "mergers and acquisitions",
    "aml": "anti money laundering",
    "kyc": "know your customer",
}


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseExpander(ABC):
    """Abstract base for all query expanders."""

    @abstractmethod
    def expand(self, query: str) -> ExpandedQuery:
        """Expand *query* and return an :class:`ExpandedQuery`."""

    def _make_result(
        self,
        original: str,
        expanded: str,
        added: Set[str],
        step_label: str,
    ) -> ExpandedQuery:
        return ExpandedQuery(
            original_query=original,
            expanded_query=expanded,
            added_terms=added,
            expansion_steps=[step_label],
        )


# ---------------------------------------------------------------------------
# SynonymExpander
# ---------------------------------------------------------------------------


class SynonymExpander(BaseExpander):
    """Expand query tokens using a built-in synonym vocabulary.

    For each content token in the query the expander looks up synonyms from
    :data:`_SYNONYM_VOCAB` (and its reverse index) and appends the top-N
    synonyms not already present in the query.

    Args:
        max_synonyms_per_term: Maximum number of synonyms to add per token.
            Default ``2``.
        vocab: Custom synonym dictionary that **overrides** the built-in one.
            Format: ``{term: [synonym1, synonym2, ...]}``.  If ``None``, the
            built-in :data:`_SYNONYM_VOCAB` is used.

    Example::

        expander = SynonymExpander(max_synonyms_per_term=1)
        result = expander.expand("contract breach remedy")
        # expanded_query might be: "contract breach remedy agreement violation relief"
    """

    def __init__(
        self,
        max_synonyms_per_term: int = 2,
        vocab: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        if max_synonyms_per_term < 0:
            raise ValueError(
                f"max_synonyms_per_term must be >= 0, got {max_synonyms_per_term}"
            )
        self.max_synonyms_per_term = max_synonyms_per_term
        self._vocab: Dict[str, List[str]] = (
            vocab if vocab is not None else _SYNONYM_VOCAB
        )

    def _get_synonyms(self, term: str) -> List[str]:
        """Return synonyms for *term*, checking both canonical and reverse index."""
        syns: List[str] = []
        # Direct lookup
        if term in self._vocab:
            syns.extend(self._vocab[term])
        # Reverse lookup (term is itself a synonym of something)
        canonical = _REVERSE_SYNONYM.get(term)
        if canonical and canonical in self._vocab and canonical not in syns:
            syns.append(canonical)
        return syns

    def expand(self, query: str) -> ExpandedQuery:
        tokens = _meaningful_tokens(query)
        existing_lower = set(_tokenize(query))
        added: Set[str] = set()
        extra_terms: List[str] = []

        for token in tokens:
            syns = self._get_synonyms(token)
            count = 0
            for syn in syns:
                if count >= self.max_synonyms_per_term:
                    break
                syn_tokens = _tokenize(syn)
                # Skip if all synonym tokens already appear in query
                if all(t in existing_lower for t in syn_tokens):
                    continue
                extra_terms.append(syn)
                added.add(syn)
                existing_lower.update(syn_tokens)
                count += 1

        expanded = (
            (query + " " + " ".join(extra_terms)).strip() if extra_terms else query
        )
        return self._make_result(
            query, expanded, added, f"SynonymExpander(n={self.max_synonyms_per_term})"
        )


# ---------------------------------------------------------------------------
# AcronymExpander
# ---------------------------------------------------------------------------


class AcronymExpander(BaseExpander):
    """Replace acronyms and abbreviations with their full-text equivalents.

    Replaces each recognised acronym token in the query with its expansion.
    The original acronym is kept alongside the expansion so that exact-match
    retrieval still works.

    Args:
        acronym_map: Custom mapping ``{acronym: expansion}`` (lowercase keys).
            Merged with the built-in :data:`_ACRONYM_MAP`; custom entries win
            on conflict.
        keep_original: If ``True`` (default), the original acronym token is
            retained in the expanded query alongside the expansion.

    Example::

        expander = AcronymExpander()
        result = expander.expand("NLP models for RAG pipelines")
        # expanded_query:
        # "NLP natural language processing models for RAG retrieval augmented generation pipelines"
    """

    def __init__(
        self,
        acronym_map: Optional[Dict[str, str]] = None,
        keep_original: bool = True,
    ) -> None:
        self._map: Dict[str, str] = dict(_ACRONYM_MAP)
        if acronym_map:
            self._map.update({k.lower(): v for k, v in acronym_map.items()})
        self.keep_original = keep_original

    def expand(self, query: str) -> ExpandedQuery:
        tokens = query.split()
        result_tokens: List[str] = []
        added: Set[str] = set()

        for token in tokens:
            key = token.lower().strip(".,;:!?")
            if key in self._map:
                expansion = self._map[key]
                if self.keep_original:
                    result_tokens.append(token)
                result_tokens.append(expansion)
                added.add(expansion)
            else:
                result_tokens.append(token)

        expanded = " ".join(result_tokens)
        return self._make_result(query, expanded, added, "AcronymExpander")


# ---------------------------------------------------------------------------
# PRFExpander  (Pseudo-Relevance Feedback)
# ---------------------------------------------------------------------------


class PRFExpander(BaseExpander):
    """Pseudo-Relevance Feedback (PRF) query expansion.

    Treats the top-K retrieved documents as pseudo-relevant and extracts
    their most discriminative terms (by TF-IDF score relative to the full
    collection).  The top-scoring terms not already in the query are appended
    to create an expanded query.

    This is a lightweight implementation of Rocchio-style relevance feedback
    that requires no user interaction.

    Args:
        top_k_docs: Number of top documents to treat as pseudo-relevant.
            Default ``3``.
        top_k_terms: Number of expansion terms to add from the feedback docs.
            Default ``5``.
        min_doc_freq: Minimum number of pseudo-relevant docs a term must appear
            in before it is considered for expansion.  Default ``1``.

    Example::

        expander = PRFExpander(top_k_docs=3, top_k_terms=4)
        docs = [r["summary"] for r in top_results[:3]]
        result = expander.expand("contract breach", feedback_docs=docs)
    """

    def __init__(
        self,
        top_k_docs: int = 3,
        top_k_terms: int = 5,
        min_doc_freq: int = 1,
    ) -> None:
        if top_k_docs < 1:
            raise ValueError(f"top_k_docs must be >= 1, got {top_k_docs}")
        if top_k_terms < 1:
            raise ValueError(f"top_k_terms must be >= 1, got {top_k_terms}")
        if min_doc_freq < 1:
            raise ValueError(f"min_doc_freq must be >= 1, got {min_doc_freq}")
        self.top_k_docs = top_k_docs
        self.top_k_terms = top_k_terms
        self.min_doc_freq = min_doc_freq

    def expand(
        self,
        query: str,
        feedback_docs: Optional[Sequence[str]] = None,
    ) -> ExpandedQuery:
        """Expand *query* using *feedback_docs* as pseudo-relevant evidence.

        Args:
            query: Original query string.
            feedback_docs: Text content of top retrieved documents.  If
                ``None`` or empty, the original query is returned unchanged.

        Returns:
            :class:`ExpandedQuery` with PRF terms appended.
        """
        if not feedback_docs:
            return self._make_result(query, query, set(), "PRFExpander(no_docs)")

        docs = list(feedback_docs[: self.top_k_docs])
        n_docs = len(docs)
        doc_token_lists = [_meaningful_tokens(d) for d in docs]

        # Aggregate term frequency across pseudo-relevant docs
        global_tf: Counter = Counter()
        doc_freq: Counter = Counter()
        for tl in doc_token_lists:
            global_tf.update(tl)
            doc_freq.update(set(tl))

        # Filter terms by min_doc_freq
        candidates = {term for term, df in doc_freq.items() if df >= self.min_doc_freq}

        # Compute TF-IDF score for each candidate
        query_tokens_set = set(_meaningful_tokens(query))
        scored: List[Tuple[float, str]] = []

        for term in candidates:
            if term in query_tokens_set:
                continue  # already in query
            tf_score = global_tf[term] / max(sum(global_tf.values()), 1)
            idf_score = _idf(term, doc_token_lists, n_docs)
            scored.append((tf_score * idf_score, term))

        scored.sort(reverse=True)
        top_terms = [term for _, term in scored[: self.top_k_terms]]

        added: Set[str] = set(top_terms)
        expanded = (query + " " + " ".join(top_terms)).strip() if top_terms else query
        step = f"PRFExpander(docs={n_docs},terms={len(top_terms)})"
        return self._make_result(query, expanded, added, step)


# ---------------------------------------------------------------------------
# QueryExpansionPipeline
# ---------------------------------------------------------------------------


class QueryExpansionPipeline:
    """Chain multiple expanders in sequence.

    Each expander receives the output of the previous one, accumulating
    added terms and expansion steps.

    Args:
        expanders: Ordered list of :class:`BaseExpander` instances.
            Executed left-to-right.

    Example::

        pipeline = QueryExpansionPipeline([
            AcronymExpander(),
            SynonymExpander(max_synonyms_per_term=2),
        ])
        result = pipeline.expand("NLP contract breach")
        print(result.expanded_query)
    """

    def __init__(self, expanders: Sequence[BaseExpander]) -> None:
        if not expanders:
            raise ValueError("At least one expander must be provided")
        self.expanders = list(expanders)

    def expand(
        self,
        query: str,
        feedback_docs: Optional[Sequence[str]] = None,
    ) -> ExpandedQuery:
        """Run the pipeline.

        Args:
            query: Original user query.
            feedback_docs: Passed through to any :class:`PRFExpander` in the
                pipeline; ignored by other expanders.

        Returns:
            Merged :class:`ExpandedQuery` from all pipeline stages.
        """
        current_query = query
        all_added: Set[str] = set()
        all_steps: List[str] = []

        for expander in self.expanders:
            if isinstance(expander, PRFExpander):
                result = expander.expand(current_query, feedback_docs=feedback_docs)
            else:
                result = expander.expand(current_query)

            current_query = result.expanded_query
            all_added.update(result.added_terms)
            all_steps.extend(result.expansion_steps)

        return ExpandedQuery(
            original_query=query,
            expanded_query=current_query,
            added_terms=all_added,
            expansion_steps=all_steps,
        )

    def __repr__(self) -> str:  # pragma: no cover
        names = [type(e).__name__ for e in self.expanders]
        return f"QueryExpansionPipeline([{', '.join(names)}])"


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

_EXPANDER_REGISTRY: Dict[str, type] = {
    "synonym": SynonymExpander,
    "acronym": AcronymExpander,
    "prf": PRFExpander,
}


def get_expander(name: str, **kwargs: object) -> BaseExpander:
    """Instantiate an expander by name.

    Args:
        name: One of ``"synonym"``, ``"acronym"``, or ``"prf"``.
        **kwargs: Forwarded to the expander constructor.

    Returns:
        Configured expander instance.

    Raises:
        ValueError: If *name* is not recognised.
    """
    if name not in _EXPANDER_REGISTRY:
        raise ValueError(
            f"Unknown expander '{name}'. Choose from: {sorted(_EXPANDER_REGISTRY)}"
        )
    cls = _EXPANDER_REGISTRY[name]
    return cls(**kwargs)  # type: ignore[call-arg]
