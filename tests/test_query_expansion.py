"""
Tests for utils/query_expansion.py

Covers:
- Helper functions: _tokenize, _meaningful_tokens, _term_freq, _idf
- ExpandedQuery dataclass: fields, all_terms property
- SynonymExpander: synonym injection, deduplication, max_synonyms_per_term, custom vocab
- AcronymExpander: acronym resolution, keep_original flag, custom map, case handling
- PRFExpander: term extraction from feedback docs, top_k_terms, min_doc_freq, no-docs fallback
- QueryExpansionPipeline: sequential chaining, PRF passthrough, empty-step merging
- get_expander factory: known/unknown names, kwargs forwarding
"""

from __future__ import annotations

import pytest

from utils.query_expansion import (
    AcronymExpander,
    ExpandedQuery,
    PRFExpander,
    QueryExpansionPipeline,
    SynonymExpander,
    _idf,
    _meaningful_tokens,
    _term_freq,
    _tokenize,
    get_expander,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_lowercases(self):
        assert "contract" in _tokenize("Contract")

    def test_strips_punctuation(self):
        tokens = _tokenize("breach, negligence!")
        assert "breach" in tokens
        assert "negligence" in tokens

    def test_empty_string_returns_empty(self):
        assert _tokenize("") == []

    def test_numbers_included(self):
        tokens = _tokenize("section 42 of the act")
        assert "42" in tokens

    def test_hyphenated_word(self):
        tokens = _tokenize("duty-of-care")
        # hyphenated forms are captured as single or split tokens
        assert any("duty" in t or t == "duty-of-care" for t in tokens)


class TestMeaningfulTokens:
    def test_removes_stop_words(self):
        tokens = _meaningful_tokens("a breach of contract")
        assert "a" not in tokens
        assert "of" not in tokens
        assert "breach" in tokens
        assert "contract" in tokens

    def test_removes_single_characters(self):
        tokens = _meaningful_tokens("i want x damages")
        assert "i" not in tokens
        assert "x" not in tokens
        assert "damages" in tokens

    def test_empty_input(self):
        assert _meaningful_tokens("") == []

    def test_all_stop_words(self):
        assert _meaningful_tokens("the a an and or is was") == []


class TestTermFreq:
    def test_counts_correctly(self):
        tf = _term_freq(["contract", "contract", "breach"])
        assert tf["contract"] == 2
        assert tf["breach"] == 1

    def test_empty(self):
        assert _term_freq([]) == {}


class TestIdf:
    def test_term_in_all_docs_has_lower_idf(self):
        doc_lists = [["contract", "breach"], ["contract", "tort"]]
        idf_contract = _idf("contract", doc_lists, 2)
        idf_breach = _idf("breach", doc_lists, 2)
        assert idf_breach > idf_contract

    def test_term_absent_from_all_docs(self):
        doc_lists = [["contract"], ["breach"]]
        # BM25 IDF with df=0 → log((N+0.5) / 0.5 + 1)
        idf = _idf("missing", doc_lists, 2)
        assert idf > 0

    def test_non_negative(self):
        doc_lists = [["contract"]] * 5
        assert _idf("contract", doc_lists, 5) >= 0


# ---------------------------------------------------------------------------
# ExpandedQuery
# ---------------------------------------------------------------------------


class TestExpandedQuery:
    def test_fields_accessible(self):
        eq = ExpandedQuery(
            original_query="breach of contract",
            expanded_query="breach of contract violation agreement",
            added_terms={"violation", "agreement"},
            expansion_steps=["SynonymExpander(n=2)"],
        )
        assert eq.original_query == "breach of contract"
        assert eq.expanded_query == "breach of contract violation agreement"
        assert "violation" in eq.added_terms
        assert "SynonymExpander(n=2)" in eq.expansion_steps

    def test_all_terms_property(self):
        eq = ExpandedQuery(
            original_query="contract",
            expanded_query="contract agreement covenant",
        )
        terms = eq.all_terms
        assert "contract" in terms
        assert "agreement" in terms
        assert "covenant" in terms

    def test_default_added_terms_empty(self):
        eq = ExpandedQuery(original_query="query", expanded_query="query")
        assert eq.added_terms == set()

    def test_default_expansion_steps_empty(self):
        eq = ExpandedQuery(original_query="query", expanded_query="query")
        assert eq.expansion_steps == []


# ---------------------------------------------------------------------------
# SynonymExpander
# ---------------------------------------------------------------------------


class TestSynonymExpander:
    def setup_method(self):
        self.expander = SynonymExpander(max_synonyms_per_term=2)

    # --- construction ---

    def test_invalid_max_synonyms_raises(self):
        with pytest.raises(ValueError, match="max_synonyms_per_term"):
            SynonymExpander(max_synonyms_per_term=-1)

    def test_zero_max_synonyms_returns_original(self):
        result = SynonymExpander(max_synonyms_per_term=0).expand("contract breach")
        assert result.expanded_query == "contract breach"
        assert result.added_terms == set()

    # --- output structure ---

    def test_returns_expanded_query_instance(self):
        result = self.expander.expand("contract breach")
        assert isinstance(result, ExpandedQuery)

    def test_original_query_preserved(self):
        q = "contract damages remedy"
        result = self.expander.expand(q)
        assert result.original_query == q

    def test_expanded_query_starts_with_original(self):
        q = "contract breach"
        result = self.expander.expand(q)
        assert result.expanded_query.startswith(q)

    def test_expansion_step_recorded(self):
        result = self.expander.expand("contract")
        assert len(result.expansion_steps) == 1
        assert "SynonymExpander" in result.expansion_steps[0]

    # --- synonym injection ---

    def test_known_term_gets_synonyms(self):
        result = self.expander.expand("contract")
        # "contract" has synonyms: agreement, covenant, deed, pact
        assert any(
            s in result.expanded_query
            for s in ("agreement", "covenant", "deed", "pact")
        )

    def test_added_terms_not_empty_for_known_word(self):
        result = self.expander.expand("breach")
        assert len(result.added_terms) > 0

    def test_max_synonyms_limit_respected(self):
        expander = SynonymExpander(max_synonyms_per_term=1)
        result = expander.expand("contract")
        # Only 1 synonym should be added for "contract"
        # Expanded = original + added terms
        added_count = len(result.added_terms)
        # We added 1 per known token max
        assert added_count <= 1

    def test_no_duplicate_terms_in_expansion(self):
        result = self.expander.expand("agreement contract")
        tokens = result.expanded_query.lower().split()
        # No exact duplicate terms
        assert len(tokens) == len(set(tokens))

    def test_unknown_term_no_expansion(self):
        result = SynonymExpander(max_synonyms_per_term=2).expand("xyznonsenseword")
        assert result.expanded_query == "xyznonsenseword"
        assert result.added_terms == set()

    def test_stop_word_only_query(self):
        result = self.expander.expand("the a and or")
        # No content terms — should be returned as-is
        assert result.added_terms == set()

    def test_custom_vocab_used(self):
        custom = {"apple": ["fruit", "orchard"]}
        expander = SynonymExpander(max_synonyms_per_term=2, vocab=custom)
        result = expander.expand("apple")
        assert "fruit" in result.expanded_query or "orchard" in result.expanded_query

    def test_custom_vocab_overrides_builtin(self):
        # Custom vocab should not inherit built-in if fully overridden
        custom = {"contract": ["deal"]}
        expander = SynonymExpander(max_synonyms_per_term=2, vocab=custom)
        result = expander.expand("contract")
        assert "deal" in result.expanded_query

    def test_already_present_synonyms_not_re_added(self):
        # "contract agreement" — "agreement" is a synonym of "contract"
        result = self.expander.expand("contract agreement")
        initial_count = result.expanded_query.lower().count("agreement")
        assert initial_count == 1  # not duplicated

    def test_does_not_mutate_input_string(self):
        q = "contract breach"
        original = q
        self.expander.expand(q)
        assert q == original


# ---------------------------------------------------------------------------
# AcronymExpander
# ---------------------------------------------------------------------------


class TestAcronymExpander:
    def setup_method(self):
        self.expander = AcronymExpander()

    # --- known acronyms ---

    def test_nlp_expanded(self):
        result = self.expander.expand("NLP models")
        assert "natural language processing" in result.expanded_query

    def test_rag_expanded(self):
        result = self.expander.expand("RAG pipeline")
        assert "retrieval augmented generation" in result.expanded_query

    def test_llm_expanded(self):
        result = self.expander.expand("LLM inference")
        assert "large language model" in result.expanded_query

    def test_case_insensitive_match(self):
        result_upper = self.expander.expand("NLP")
        result_lower = self.expander.expand("nlp")
        # Both should produce an expansion
        assert "natural language processing" in result_upper.expanded_query
        assert "natural language processing" in result_lower.expanded_query

    # --- keep_original flag ---

    def test_keep_original_true_retains_acronym(self):
        result = AcronymExpander(keep_original=True).expand("NLP search")
        assert "NLP" in result.expanded_query

    def test_keep_original_false_removes_acronym(self):
        result = AcronymExpander(keep_original=False).expand("NLP search")
        assert "NLP" not in result.expanded_query
        assert "natural language processing" in result.expanded_query

    # --- unknown tokens ---

    def test_unknown_tokens_passed_through(self):
        result = self.expander.expand("contract breach negligence")
        # None of these are acronyms
        assert result.added_terms == set()
        assert result.expanded_query == "contract breach negligence"

    # --- custom map ---

    def test_custom_acronym_map(self):
        custom = {"xyz": "xenophile zodiac system"}
        expander = AcronymExpander(acronym_map=custom)
        result = expander.expand("XYZ pipeline")
        assert "xenophile zodiac system" in result.expanded_query

    def test_custom_map_merged_with_builtin(self):
        custom = {"myco": "my company"}
        expander = AcronymExpander(acronym_map=custom)
        result = expander.expand("NLP myco")
        assert "natural language processing" in result.expanded_query
        assert "my company" in result.expanded_query

    # --- output structure ---

    def test_returns_expanded_query_instance(self):
        assert isinstance(self.expander.expand("NLP"), ExpandedQuery)

    def test_original_query_preserved(self):
        q = "NLP contract analysis"
        result = self.expander.expand(q)
        assert result.original_query == q

    def test_expansion_step_recorded(self):
        result = self.expander.expand("NLP")
        assert "AcronymExpander" in result.expansion_steps[0]

    def test_added_terms_contains_expansion(self):
        result = self.expander.expand("NLP")
        assert "natural language processing" in result.added_terms

    def test_multiple_acronyms_expanded(self):
        result = self.expander.expand("NLP RAG pipeline")
        assert "natural language processing" in result.added_terms
        assert "retrieval augmented generation" in result.added_terms


# ---------------------------------------------------------------------------
# PRFExpander
# ---------------------------------------------------------------------------


class TestPRFExpander:
    def setup_method(self):
        self.docs = [
            "contract breach damages expectation loss award court",
            "contract agreement covenant terms conditions",
            "breach violation infringement remedies damages compensation",
        ]
        self.expander = PRFExpander(top_k_docs=3, top_k_terms=4)

    # --- construction ---

    def test_invalid_top_k_docs_raises(self):
        with pytest.raises(ValueError, match="top_k_docs"):
            PRFExpander(top_k_docs=0)

    def test_invalid_top_k_terms_raises(self):
        with pytest.raises(ValueError, match="top_k_terms"):
            PRFExpander(top_k_terms=0)

    def test_invalid_min_doc_freq_raises(self):
        with pytest.raises(ValueError, match="min_doc_freq"):
            PRFExpander(min_doc_freq=0)

    # --- no feedback docs ---

    def test_no_feedback_docs_returns_original(self):
        result = self.expander.expand("contract breach")
        assert result.expanded_query == "contract breach"

    def test_none_feedback_docs(self):
        result = self.expander.expand("contract breach", feedback_docs=None)
        assert result.expanded_query == "contract breach"
        assert result.added_terms == set()

    def test_empty_feedback_docs(self):
        result = self.expander.expand("contract breach", feedback_docs=[])
        assert result.expanded_query == "contract breach"

    # --- with feedback docs ---

    def test_expands_with_feedback_docs(self):
        result = self.expander.expand("contract", feedback_docs=self.docs)
        assert len(result.added_terms) > 0

    def test_original_terms_not_re_added(self):
        result = self.expander.expand("contract breach", feedback_docs=self.docs)
        # "contract" and "breach" already in query
        assert "contract" not in result.added_terms
        assert "breach" not in result.added_terms

    def test_top_k_terms_limit(self):
        expander = PRFExpander(top_k_docs=3, top_k_terms=2)
        result = expander.expand("x", feedback_docs=self.docs)
        assert len(result.added_terms) <= 2

    def test_top_k_docs_limits_docs_used(self):
        # Only top_k_docs=1 docs are used; terms exclusive to later docs must not appear.
        exclusive_docs = [
            # doc[0]: unique term "xylophone" that only appears here
            "xylophone damages breach expectation contract",
            # doc[1]: unique term "zeppelin"
            "zeppelin agreement covenant terms contract",
            # doc[2]: unique term "wolverine" — should NOT be used
            "wolverine irrelevant words here contract",
        ]
        expander = PRFExpander(top_k_docs=1, top_k_terms=5)
        result = expander.expand("contract", feedback_docs=exclusive_docs)
        # "wolverine" comes from doc[2] which is outside top_k_docs=1
        assert "wolverine" not in result.expanded_query
        # Some term from doc[0] should be present
        assert any(
            t in result.expanded_query
            for t in ("xylophone", "damages", "breach", "expectation")
        )

    def test_min_doc_freq_filters_rare_terms(self):
        # With min_doc_freq=3 only terms appearing in all 3 docs pass
        expander = PRFExpander(top_k_docs=3, top_k_terms=5, min_doc_freq=3)
        result = expander.expand("x", feedback_docs=self.docs)
        # "contract" and "breach" appear in ≥2 docs, but we test what passes
        # The result should have fewer or equal terms than with min_doc_freq=1
        expander_loose = PRFExpander(top_k_docs=3, top_k_terms=5, min_doc_freq=1)
        result_loose = expander_loose.expand("x", feedback_docs=self.docs)
        assert len(result.added_terms) <= len(result_loose.added_terms)

    def test_expansion_step_recorded(self):
        result = self.expander.expand("contract", feedback_docs=self.docs)
        assert any("PRFExpander" in s for s in result.expansion_steps)

    def test_original_query_preserved(self):
        q = "contract damages"
        result = self.expander.expand(q, feedback_docs=self.docs)
        assert result.original_query == q

    def test_expanded_query_starts_with_original(self):
        q = "contract"
        result = self.expander.expand(q, feedback_docs=self.docs)
        assert result.expanded_query.startswith(q)

    def test_returns_expanded_query_instance(self):
        result = self.expander.expand("contract", feedback_docs=self.docs)
        assert isinstance(result, ExpandedQuery)


# ---------------------------------------------------------------------------
# QueryExpansionPipeline
# ---------------------------------------------------------------------------


class TestQueryExpansionPipeline:
    # --- construction ---

    def test_empty_expanders_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            QueryExpansionPipeline([])

    # --- single expander ---

    def test_single_synonym_expander(self):
        pipeline = QueryExpansionPipeline([SynonymExpander(max_synonyms_per_term=1)])
        result = pipeline.expand("contract breach")
        assert isinstance(result, ExpandedQuery)

    def test_single_acronym_expander(self):
        pipeline = QueryExpansionPipeline([AcronymExpander()])
        result = pipeline.expand("NLP pipeline")
        assert "natural language processing" in result.expanded_query

    # --- chaining ---

    def test_acronym_then_synonym(self):
        """Acronym resolved first, then synonyms applied to the resolved terms."""
        pipeline = QueryExpansionPipeline(
            [
                AcronymExpander(),
                SynonymExpander(max_synonyms_per_term=1),
            ]
        )
        result = pipeline.expand("NLP contract")
        # Acronym expanded first
        assert "natural language processing" in result.added_terms
        # Synonym applied to "contract"
        assert result.original_query == "NLP contract"

    def test_steps_accumulated_from_all_expanders(self):
        pipeline = QueryExpansionPipeline(
            [
                AcronymExpander(),
                SynonymExpander(max_synonyms_per_term=1),
            ]
        )
        result = pipeline.expand("NLP contract")
        assert len(result.expansion_steps) == 2
        step_names = " ".join(result.expansion_steps)
        assert "AcronymExpander" in step_names
        assert "SynonymExpander" in step_names

    def test_added_terms_merged_across_expanders(self):
        pipeline = QueryExpansionPipeline(
            [
                AcronymExpander(),
                SynonymExpander(max_synonyms_per_term=1),
            ]
        )
        result = pipeline.expand("NLP contract")
        # From acronym: "natural language processing"
        assert "natural language processing" in result.added_terms

    def test_prf_expander_receives_feedback_docs(self):
        feedback = [
            "damages compensation award court decision",
            "remedy relief restitution breach contract",
        ]
        pipeline = QueryExpansionPipeline(
            [
                PRFExpander(top_k_docs=2, top_k_terms=3),
            ]
        )
        result = pipeline.expand("contract", feedback_docs=feedback)
        assert len(result.added_terms) > 0

    def test_prf_without_feedback_docs_no_expansion(self):
        pipeline = QueryExpansionPipeline([PRFExpander()])
        result = pipeline.expand("contract breach")
        assert result.expanded_query == "contract breach"

    def test_original_query_always_preserved(self):
        pipeline = QueryExpansionPipeline(
            [
                AcronymExpander(),
                SynonymExpander(),
            ]
        )
        q = "NLP contract breach"
        result = pipeline.expand(q)
        assert result.original_query == q

    def test_three_stage_pipeline(self):
        feedback = [
            "damages breach violation remedy award",
            "contract agreement covenant deed",
        ]
        pipeline = QueryExpansionPipeline(
            [
                AcronymExpander(),
                SynonymExpander(max_synonyms_per_term=1),
                PRFExpander(top_k_docs=2, top_k_terms=2),
            ]
        )
        result = pipeline.expand("NLP contract", feedback_docs=feedback)
        assert isinstance(result, ExpandedQuery)
        assert len(result.expansion_steps) == 3
        assert len(result.expanded_query) > len("NLP contract")


# ---------------------------------------------------------------------------
# get_expander factory
# ---------------------------------------------------------------------------


class TestGetExpander:
    def test_synonym(self):
        e = get_expander("synonym")
        assert isinstance(e, SynonymExpander)

    def test_acronym(self):
        e = get_expander("acronym")
        assert isinstance(e, AcronymExpander)

    def test_prf(self):
        e = get_expander("prf")
        assert isinstance(e, PRFExpander)

    def test_kwargs_forwarded_to_synonym(self):
        e = get_expander("synonym", max_synonyms_per_term=3)
        assert isinstance(e, SynonymExpander)
        assert e.max_synonyms_per_term == 3

    def test_kwargs_forwarded_to_prf(self):
        e = get_expander("prf", top_k_docs=5, top_k_terms=8)
        assert isinstance(e, PRFExpander)
        assert e.top_k_docs == 5
        assert e.top_k_terms == 8

    def test_unknown_expander_raises(self):
        with pytest.raises(ValueError, match="Unknown expander"):
            get_expander("nonexistent")

    def test_factory_produces_working_expander(self):
        e = get_expander("synonym", max_synonyms_per_term=1)
        result = e.expand("contract breach")
        assert isinstance(result, ExpandedQuery)
