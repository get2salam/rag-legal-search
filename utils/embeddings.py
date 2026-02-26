"""
Embedding utilities for generating vector representations of text.
"""

import os
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential


class EmbeddingModel:
    """
    Wrapper for embedding models.
    Supports OpenAI and local sentence-transformers models.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding model.

        Args:
            model_name: Model identifier. Options:
                - "openai" or "text-embedding-3-small" (default)
                - "openai-large" or "text-embedding-3-large"
                - "local" or any sentence-transformers model name
        """
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "openai")
        self._init_model()

    def _init_model(self):
        """Initialize the appropriate model based on model_name."""
        if self.model_name in ["openai", "text-embedding-3-small"]:
            self._init_openai("text-embedding-3-small")
        elif self.model_name in ["openai-large", "text-embedding-3-large"]:
            self._init_openai("text-embedding-3-large")
        elif self.model_name == "local":
            self._init_local("all-MiniLM-L6-v2")
        else:
            # Assume it's a sentence-transformers model name
            self._init_local(self.model_name)

    def _init_openai(self, model: str):
        """Initialize OpenAI embeddings."""
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.provider = "openai"
        self.dimensions = 1536 if model == "text-embedding-3-small" else 3072

    def _init_local(self, model: str):
        """Initialize local sentence-transformers model."""
        from sentence_transformers import SentenceTransformer

        self.client = SentenceTransformer(model)
        self.model = model
        self.provider = "local"
        self.dimensions = self.client.get_sentence_embedding_dimension()

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        if self.provider == "openai":
            response = self.client.embeddings.create(model=self.model, input=texts)
            return [item.embedding for item in response.data]
        else:
            # Local model
            embeddings = self.client.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.

        Args:
            query: Query string

        Returns:
            Embedding vector
        """
        return self.embed([query])[0]

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        Handles batching for large lists.

        Args:
            documents: List of document texts

        Returns:
            List of embedding vectors
        """
        batch_size = 100 if self.provider == "openai" else 32
        all_embeddings = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            embeddings = self.embed(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings


# Convenience function
def get_embeddings(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    """
    Get embeddings for a list of texts using the default model.

    Args:
        texts: List of strings to embed
        model: Optional model override

    Returns:
        List of embedding vectors
    """
    embedding_model = EmbeddingModel(model)
    return embedding_model.embed(texts)
