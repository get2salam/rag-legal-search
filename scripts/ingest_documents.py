"""
Bulk document ingestion for RAG Legal Search.
Ingests case law documents from various formats (JSON, PDF, text).

Usage:
    python scripts/ingest_documents.py --source data/cases.json
    python scripts/ingest_documents.py --source data/pdfs/ --format pdf
    python scripts/ingest_documents.py --source data/cases/ --format text
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.embeddings import EmbeddingModel
from utils.vector_store import get_vector_store
from utils.chunking import get_chunker


def load_json_cases(path: str) -> List[Dict]:
    """Load cases from a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'cases' in data:
        return data['cases']
    else:
        raise ValueError(f"Unexpected JSON structure in {path}")


def load_pdf_cases(directory: str) -> List[Dict]:
    """Load cases from PDF files in a directory."""
    try:
        import pdfplumber
    except ImportError:
        print("❌ pdfplumber not installed. Run: pip install pdfplumber")
        sys.exit(1)
    
    cases = []
    pdf_dir = Path(directory)
    
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        print(f"  📄 Processing: {pdf_path.name}")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
            
            cases.append({
                "id": f"pdf_{pdf_path.stem}",
                "title": pdf_path.stem.replace("_", " ").replace("-", " ").title(),
                "text": text,
                "source": str(pdf_path),
                "format": "pdf"
            })
        except Exception as e:
            print(f"  ⚠️ Failed to process {pdf_path.name}: {e}")
    
    return cases


def load_text_cases(directory: str) -> List[Dict]:
    """Load cases from text files in a directory."""
    cases = []
    text_dir = Path(directory)
    
    for txt_path in sorted(text_dir.glob("*.txt")):
        print(f"  📄 Processing: {txt_path.name}")
        
        text = txt_path.read_text(encoding='utf-8')
        cases.append({
            "id": f"txt_{txt_path.stem}",
            "title": txt_path.stem.replace("_", " ").replace("-", " ").title(),
            "text": text,
            "source": str(txt_path),
            "format": "text"
        })
    
    return cases


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks (legacy wrapper).

    For new code, prefer ``utils.chunking.get_chunker()`` which supports
    multiple strategies (fixed, sentence, recursive, semantic, sliding_window,
    structure) with richer Chunk metadata.
    """
    chunker = get_chunker("recursive", chunk_size=chunk_size, overlap=overlap)
    return [c.text for c in chunker.chunk(text)]


def ingest(
    source: str,
    format: str = "json",
    model: str = "local",
    store: str = "chroma",
    chunk_size: int = 1000,
    batch_size: int = 50,
    chunk_strategy: str = "recursive",
):
    """Ingest documents into the vector store."""
    print(f"📥 Ingesting documents from: {source}")
    print(f"   Format: {format}")
    print(f"   Model: {model}")
    print(f"   Store: {store}")
    print(f"   Chunking strategy: {chunk_strategy}")
    print()
    
    # Load cases
    print("📂 Loading documents...")
    if format == "json":
        cases = load_json_cases(source)
    elif format == "pdf":
        cases = load_pdf_cases(source)
    elif format == "text":
        cases = load_text_cases(source)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    print(f"   Loaded {len(cases)} documents")
    
    # Chunk documents using the selected strategy
    print(f"\n✂️ Chunking documents (strategy: {chunk_strategy})...")
    chunker = get_chunker(chunk_strategy, chunk_size=chunk_size, overlap=min(200, chunk_size // 5))
    chunked_cases = []
    for case in cases:
        text = case.get('text', '')
        if len(text) > chunk_size:
            chunks = chunker.chunk(text)
            for chunk_obj in chunks:
                chunked_case = {**case}
                chunked_case['id'] = f"{case.get('id', 'doc')}_{chunk_obj.index}"
                chunked_case['text'] = chunk_obj.text
                chunked_case['chunk_index'] = chunk_obj.index
                chunked_case['total_chunks'] = len(chunks)
                chunked_case['chunk_start'] = chunk_obj.start_char
                chunked_case['chunk_end'] = chunk_obj.end_char
                chunked_cases.append(chunked_case)
        else:
            chunked_cases.append(case)
    
    print(f"   Created {len(chunked_cases)} chunks from {len(cases)} documents")
    
    # Initialize components
    print("\n📦 Loading embedding model...")
    embeddings = EmbeddingModel(model)
    
    print("💾 Connecting to vector store...")
    vector_store = get_vector_store(store)
    
    # Process in batches
    total_batches = (len(chunked_cases) + batch_size - 1) // batch_size
    print(f"\n🔢 Processing {total_batches} batches...")
    
    for i in range(0, len(chunked_cases), batch_size):
        batch = chunked_cases[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        print(f"   Batch {batch_num}/{total_batches} ({len(batch)} chunks)...", end=" ")
        
        texts = [c['text'] for c in batch]
        vectors = embeddings.embed_documents(texts)
        ids = [c['id'] for c in batch]
        
        vector_store.add_documents(
            documents=batch,
            embeddings=vectors,
            ids=ids
        )
        
        print("✅")
    
    print(f"\n🎉 Ingestion complete!")
    print(f"   Documents: {len(cases)}")
    print(f"   Chunks: {len(chunked_cases)}")
    print(f"   Vector store: {store}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into RAG Legal Search")
    parser.add_argument("--source", required=True, help="Path to source file or directory")
    parser.add_argument("--format", default="json", choices=["json", "pdf", "text"],
                       help="Source format (default: json)")
    parser.add_argument("--model", default="local", choices=["local", "openai", "openai-large"],
                       help="Embedding model (default: local)")
    parser.add_argument("--store", default="chroma", choices=["chroma", "pinecone"],
                       help="Vector store (default: chroma)")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in chars")
    parser.add_argument("--chunk-strategy", default="recursive",
                       choices=["fixed", "sentence", "recursive", "sliding_window", "structure"],
                       help="Chunking strategy (default: recursive)")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing")
    
    args = parser.parse_args()
    ingest(args.source, args.format, args.model, args.store, args.chunk_size, args.batch_size, args.chunk_strategy)
