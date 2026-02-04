"""
Vector store management for RAG.
Supports ChromaDB (local) and Pinecone (cloud).
"""

import os
from typing import List, Dict, Optional, Any
from utils.embeddings import EmbeddingModel


class VectorStore:
    """
    Abstract base class for vector stores.
    """
    
    def add_documents(self, documents: List[Dict], embeddings: List[List[float]]) -> None:
        raise NotImplementedError
    
    def search(self, query_embedding: List[float], top_k: int = 10) -> List[Dict]:
        raise NotImplementedError
    
    def delete(self, ids: List[str]) -> None:
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    """
    ChromaDB vector store for local development.
    """
    
    def __init__(self, collection_name: str = "legal_cases", persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistent storage
        """
        import chromadb
        from chromadb.config import Settings
        
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(
        self,
        documents: List[Dict],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dicts with 'text' and metadata
            embeddings: Corresponding embeddings
            ids: Optional document IDs
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        texts = [doc.get('text', '') for doc in documents]
        metadatas = [{k: v for k, v in doc.items() if k != 'text'} for doc in documents]
        
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            where: Optional metadata filter
            
        Returns:
            List of results with scores
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        return formatted
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        self.collection.delete(ids=ids)


class PineconeVectorStore(VectorStore):
    """
    Pinecone vector store for production.
    """
    
    def __init__(
        self,
        index_name: Optional[str] = None,
        api_key: Optional[str] = None,
        environment: Optional[str] = None
    ):
        """
        Initialize Pinecone store.
        
        Args:
            index_name: Pinecone index name
            api_key: Pinecone API key
            environment: Pinecone environment
        """
        from pinecone import Pinecone
        
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.environment = environment or os.getenv("PINECONE_ENVIRONMENT")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "legal-cases")
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY not found")
        
        self.pc = Pinecone(api_key=self.api_key)
        self.index = self.pc.Index(self.index_name)
    
    def add_documents(
        self,
        documents: List[Dict],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add documents to Pinecone."""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        vectors = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            vectors.append({
                'id': ids[i],
                'values': embedding,
                'metadata': {
                    'text': doc.get('text', '')[:1000],  # Pinecone metadata limit
                    **{k: v for k, v in doc.items() if k != 'text'}
                }
            })
        
        # Batch upsert
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[i:i + batch_size])
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter: Optional[Dict] = None
    ) -> List[Dict]:
        """Search Pinecone index."""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filter,
            include_metadata=True
        )
        
        formatted = []
        for match in results['matches']:
            formatted.append({
                'id': match['id'],
                'text': match.get('metadata', {}).get('text', ''),
                'metadata': match.get('metadata', {}),
                'score': match['score']
            })
        
        return formatted
    
    def delete(self, ids: List[str]) -> None:
        """Delete vectors by ID."""
        self.index.delete(ids=ids)


def get_vector_store(store_type: Optional[str] = None) -> VectorStore:
    """
    Factory function to get the appropriate vector store.
    
    Args:
        store_type: "chroma" or "pinecone"
        
    Returns:
        VectorStore instance
    """
    store_type = store_type or os.getenv("VECTOR_STORE", "chroma")
    
    if store_type == "pinecone":
        return PineconeVectorStore()
    else:
        return ChromaVectorStore()
