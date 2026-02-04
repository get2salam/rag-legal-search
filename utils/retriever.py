"""
RAG Retriever for legal document search.
Combines vector search with LLM-powered relevance explanations.
"""

import os
from typing import List, Dict, Optional
from utils.embeddings import EmbeddingModel
from utils.vector_store import get_vector_store


class LegalRetriever:
    """
    RAG retriever specialized for legal case law search.
    """
    
    def __init__(
        self,
        embedding_model: Optional[str] = None,
        vector_store_type: Optional[str] = None,
        llm_model: Optional[str] = None
    ):
        """
        Initialize the retriever.
        
        Args:
            embedding_model: Embedding model to use
            vector_store_type: Vector store type ("chroma" or "pinecone")
            llm_model: LLM model for generating explanations
        """
        self.embeddings = EmbeddingModel(embedding_model)
        self.vector_store = get_vector_store(vector_store_type)
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "gpt-4-turbo")
        self._init_llm()
    
    def _init_llm(self):
        """Initialize the LLM for generating relevance explanations."""
        if self.llm_model.startswith("claude"):
            from anthropic import Anthropic
            self.llm_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.llm_provider = "anthropic"
        else:
            from openai import OpenAI
            self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.llm_provider = "openai"
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.5,
        filters: Optional[Dict] = None,
        include_explanations: bool = True
    ) -> List[Dict]:
        """
        Search for relevant legal cases.
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            filters: Optional filters (year, court, category)
            include_explanations: Whether to generate AI explanations
            
        Returns:
            List of search results with relevance explanations
        """
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Build filter dict for vector store
        vector_filters = self._build_filters(filters) if filters else None
        
        # Search vector store
        raw_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get more results to filter by threshold
            where=vector_filters
        )
        
        # Filter by threshold and limit
        filtered_results = [
            r for r in raw_results
            if r.get('score', 0) >= threshold
        ][:top_k]
        
        # Format results
        formatted_results = []
        for result in filtered_results:
            formatted = self._format_result(result)
            formatted_results.append(formatted)
        
        # Add AI explanations if requested
        if include_explanations and formatted_results:
            formatted_results = self._add_explanations(query, formatted_results)
        
        return formatted_results
    
    def _build_filters(self, filters: Dict) -> Optional[Dict]:
        """Build vector store filter from search filters."""
        conditions = []
        
        if filters.get('year_from') and filters.get('year_to'):
            conditions.append({
                "year": {"$gte": filters['year_from'], "$lte": filters['year_to']}
            })
        
        if filters.get('courts') and 'All Courts' not in filters['courts']:
            conditions.append({
                "court": {"$in": filters['courts']}
            })
        
        if filters.get('categories') and 'All Categories' not in filters['categories']:
            conditions.append({
                "category": {"$in": filters['categories']}
            })
        
        if not conditions:
            return None
        
        return {"$and": conditions} if len(conditions) > 1 else conditions[0]
    
    def _format_result(self, result: Dict) -> Dict:
        """Format a raw result into the display format."""
        metadata = result.get('metadata', {})
        
        return {
            'id': result.get('id', ''),
            'title': metadata.get('title', 'Untitled Case'),
            'citation': metadata.get('citation', 'Citation not available'),
            'court': metadata.get('court', 'Unknown Court'),
            'date': metadata.get('date', 'Date unknown'),
            'category': metadata.get('category', 'Uncategorized'),
            'score': result.get('score', 0),
            'summary': metadata.get('summary', ''),
            'excerpt': result.get('text', '')[:2000],
            'relevance_explanation': ''  # Will be filled by _add_explanations
        }
    
    def _add_explanations(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Add AI-generated relevance explanations to results.
        """
        # Build prompt for batch explanation
        cases_text = "\n\n".join([
            f"Case {i+1}: {r['title']}\nSummary: {r['summary'][:300]}..."
            for i, r in enumerate(results[:5])  # Limit to top 5 for efficiency
        ])
        
        prompt = f"""Given the following legal research query and search results, 
explain briefly (1-2 sentences each) why each case is relevant to the query.

QUERY: {query}

CASES:
{cases_text}

Provide explanations in the format:
Case 1: [explanation]
Case 2: [explanation]
...

Be specific about the legal principles or facts that make each case relevant."""

        try:
            if self.llm_provider == "anthropic":
                response = self.llm_client.messages.create(
                    model=self.llm_model,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                explanation_text = response.content[0].text
            else:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a legal research assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                explanation_text = response.choices[0].message.content
            
            # Parse explanations
            explanations = self._parse_explanations(explanation_text)
            
            # Add to results
            for i, result in enumerate(results):
                if i < len(explanations):
                    result['relevance_explanation'] = explanations[i]
                else:
                    result['relevance_explanation'] = "Matches your search criteria based on semantic similarity."
        
        except Exception as e:
            # Fallback if LLM fails
            for result in results:
                result['relevance_explanation'] = "Matches your search criteria based on semantic similarity."
        
        return results
    
    def _parse_explanations(self, text: str) -> List[str]:
        """Parse case explanations from LLM response."""
        explanations = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('Case ') and ':' in line:
                explanation = line.split(':', 1)[1].strip()
                explanations.append(explanation)
        
        return explanations
    
    def add_case(self, case: Dict) -> None:
        """
        Add a single case to the vector store.
        
        Args:
            case: Case dict with title, citation, text, and metadata
        """
        text = case.get('text', '')
        embedding = self.embeddings.embed_query(text)
        
        self.vector_store.add_documents(
            documents=[case],
            embeddings=[embedding],
            ids=[case.get('id', f"case_{hash(case.get('citation', ''))}")]
        )
    
    def add_cases_batch(self, cases: List[Dict]) -> None:
        """
        Add multiple cases to the vector store.
        
        Args:
            cases: List of case dicts
        """
        texts = [case.get('text', '') for case in cases]
        embeddings = self.embeddings.embed_documents(texts)
        ids = [case.get('id', f"case_{i}") for i, case in enumerate(cases)]
        
        self.vector_store.add_documents(
            documents=cases,
            embeddings=embeddings,
            ids=ids
        )
