"""
Tests for the RAG Legal Search retriever.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestEmbeddingModel:
    """Tests for the embedding model wrapper."""
    
    def test_local_model_init(self):
        """Test local model initialization."""
        with patch("utils.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model
            
            from utils.embeddings import EmbeddingModel
            model = EmbeddingModel("local")
            
            assert model.provider == "local"
            assert model.dimensions == 384
    
    def test_embed_empty_list(self):
        """Test that embedding an empty list returns empty."""
        with patch("utils.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model
            
            from utils.embeddings import EmbeddingModel
            model = EmbeddingModel("local")
            
            result = model.embed([])
            assert result == []
    
    def test_embed_documents_batching(self):
        """Test that large document lists are batched."""
        with patch("utils.embeddings.SentenceTransformer") as mock_st:
            import numpy as np
            
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_model.encode.return_value = np.random.rand(10, 384)
            mock_st.return_value = mock_model
            
            from utils.embeddings import EmbeddingModel
            model = EmbeddingModel("local")
            
            docs = [f"document {i}" for i in range(100)]
            result = model.embed_documents(docs)
            
            # Should have called encode multiple times (batch_size=32 for local)
            assert mock_model.encode.call_count > 1
            assert len(result) == 100


class TestLegalRetriever:
    """Tests for the legal retriever."""
    
    def test_build_filters_empty(self):
        """Test filter building with no effective filters."""
        with patch("utils.retriever.EmbeddingModel"), \
             patch("utils.retriever.get_vector_store"), \
             patch("utils.retriever.LegalRetriever._init_llm"):
            
            from utils.retriever import LegalRetriever
            retriever = LegalRetriever.__new__(LegalRetriever)
            
            filters = {"courts": ["All Courts"], "categories": ["All Categories"]}
            result = retriever._build_filters(filters)
            assert result is None
    
    def test_build_filters_with_courts(self):
        """Test filter building with court selection."""
        with patch("utils.retriever.EmbeddingModel"), \
             patch("utils.retriever.get_vector_store"), \
             patch("utils.retriever.LegalRetriever._init_llm"):
            
            from utils.retriever import LegalRetriever
            retriever = LegalRetriever.__new__(LegalRetriever)
            
            filters = {
                "courts": ["Supreme Court", "High Court"],
                "categories": ["All Categories"]
            }
            result = retriever._build_filters(filters)
            assert result is not None
            assert "court" in str(result)
    
    def test_format_result(self):
        """Test result formatting."""
        with patch("utils.retriever.EmbeddingModel"), \
             patch("utils.retriever.get_vector_store"), \
             patch("utils.retriever.LegalRetriever._init_llm"):
            
            from utils.retriever import LegalRetriever
            retriever = LegalRetriever.__new__(LegalRetriever)
            
            raw = {
                "id": "test_001",
                "text": "Some case text here...",
                "metadata": {
                    "title": "Test Case",
                    "citation": "[2024] TEST 1",
                    "court": "Supreme Court",
                    "date": "2024-01-01",
                    "category": "Contract Law",
                    "summary": "Test summary"
                },
                "score": 0.85
            }
            
            result = retriever._format_result(raw)
            
            assert result['title'] == "Test Case"
            assert result['score'] == 0.85
            assert result['court'] == "Supreme Court"
    
    def test_parse_explanations(self):
        """Test explanation parsing from LLM response."""
        with patch("utils.retriever.EmbeddingModel"), \
             patch("utils.retriever.get_vector_store"), \
             patch("utils.retriever.LegalRetriever._init_llm"):
            
            from utils.retriever import LegalRetriever
            retriever = LegalRetriever.__new__(LegalRetriever)
            
            llm_response = """Case 1: This case is relevant because it deals with breach of contract.
Case 2: Employment law precedent for wrongful dismissal.
Case 3: Software licensing and IP rights."""
            
            explanations = retriever._parse_explanations(llm_response)
            
            assert len(explanations) == 3
            assert "breach of contract" in explanations[0]


class TestDemoResults:
    """Test the demo mode functionality."""
    
    def test_demo_results_returned_without_api_key(self):
        """Test that demo results are returned when no API key is set."""
        import os
        
        # Remove API key if set
        original = os.environ.pop('OPENAI_API_KEY', None)
        
        try:
            from app import get_demo_results
            results = get_demo_results("breach of contract")
            
            assert len(results) > 0
            assert all('title' in r for r in results)
            assert all('score' in r for r in results)
            assert all(r['score'] > 0 for r in results)
        finally:
            if original:
                os.environ['OPENAI_API_KEY'] = original
