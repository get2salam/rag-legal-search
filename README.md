# RAG Legal Search 🔍⚖️

A semantic search engine for legal case law using Retrieval-Augmented Generation (RAG). Ask questions in natural language, get relevant cases with AI-powered summaries.

## 🎯 What It Does

- **Natural Language Search** — "What are the precedents for breach of contract in employment?"
- **Semantic Matching** — Finds conceptually similar cases, not just keyword matches
- **AI Summaries** — Each result includes a plain-English explanation of relevance
- **Source Citations** — Full citations and links to original documents
- **Filtering** — By court, date range, jurisdiction, topic

## 🚀 Live Demo

[Try it on Streamlit Cloud](https://rag-legal-search.streamlit.app) *(coming soon)*

## 🖼️ Screenshots

![Search Interface](screenshots/search.png)
![Results View](screenshots/results.png)

## 🛠️ Tech Stack

- **Embeddings:** OpenAI text-embedding-3-small / Sentence Transformers
- **Vector Store:** ChromaDB (local) / Pinecone (production)
- **LLM:** GPT-4 / Claude for summaries
- **Backend:** Python, FastAPI
- **Frontend:** Streamlit
- **Document Processing:** LangChain, PyPDF2

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                               │
│                    "breach of contract cases"                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Query Embedding                             │
│              Convert query to vector representation              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Vector Search                               │
│        Find top-k similar document chunks in vector DB           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Context Assembly                            │
│          Combine relevant chunks with original query             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLM Generation                              │
│        Generate summary and relevance explanation                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Search Results                              │
│           Ranked cases with summaries and citations              │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/get2salam/rag-legal-search.git
cd rag-legal-search

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Initialize the vector database with sample data
python scripts/init_db.py

# Run the app
streamlit run app.py
```

## 🔑 Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-your-key-here

# Optional - for production vector store
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=us-east-1

# Optional - for Anthropic
ANTHROPIC_API_KEY=sk-ant-your-key
```

## 📁 Project Structure

```
rag-legal-search/
├── app.py                    # Streamlit application
├── requirements.txt          # Dependencies
├── .env.example             # Environment template
├── utils/
│   ├── embeddings.py        # Embedding generation
│   ├── vector_store.py      # ChromaDB/Pinecone interface
│   ├── retriever.py         # RAG retrieval logic
│   └── llm.py               # LLM integration
├── scripts/
│   ├── init_db.py           # Initialize vector DB
│   └── ingest_documents.py  # Bulk document ingestion
├── data/
│   └── sample_cases/        # Sample case law for demo
└── tests/
    └── test_retriever.py    # Unit tests
```

## 🔧 Configuration

### Embedding Models

```python
# In utils/embeddings.py
EMBEDDING_MODELS = {
    "openai": "text-embedding-3-small",      # Best quality, paid
    "openai-large": "text-embedding-3-large", # Higher dim, more $
    "local": "all-MiniLM-L6-v2",              # Free, runs locally
}
```

### Chunking Strategy

```python
# Optimized for legal documents
CHUNK_CONFIG = {
    "chunk_size": 1000,        # Characters per chunk
    "chunk_overlap": 200,      # Overlap for context
    "separator": "\n\n",       # Prefer paragraph breaks
}
```

## 📊 Sample Queries

```
"cases about wrongful termination in the UK"
"precedents for intellectual property infringement"  
"contract law breach of duty cases 2020-2024"
"landlord tenant disputes security deposit"
"employment discrimination age-based"
```

## 🎯 Features

### Implemented ✅
- [x] Semantic search with embeddings
- [x] Natural language queries
- [x] AI-generated result summaries
- [x] Source citations
- [x] Date range filtering
- [x] Relevance scoring

### Roadmap 🗺️
- [ ] Multi-jurisdiction support
- [ ] Case relationship mapping
- [ ] Saved searches
- [ ] Export results to PDF
- [ ] API endpoint for integration
- [ ] Fine-tuned legal embeddings

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=utils tests/
```

## 📈 Performance

| Metric | Value |
|--------|-------|
| Index Size | 10,000 cases |
| Query Latency | ~500ms |
| Embedding Dim | 1536 (OpenAI) |
| Top-k Results | 10 |

## 👨‍💻 Author

**Abdul Salam**
- MS in Artificial Intelligence
- LLM in Commercial Law | LLB
- Building AI tools for legal research

[LinkedIn](https://linkedin.com/in/abdul-salam-6539aa11b) | [GitHub](https://github.com/get2salam)

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

*Part of the Qanoon.com project — AI-powered legal research for Pakistan* 🇵🇰
