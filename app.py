"""
RAG Legal Search
Semantic search for legal case law using Retrieval-Augmented Generation.
"""

import streamlit as st
import os
from dotenv import load_dotenv
from utils.retriever import LegalRetriever

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="RAG Legal Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1E3A5F;
    }
    .relevance-high {
        border-left-color: #28a745 !important;
    }
    .relevance-medium {
        border-left-color: #ffc107 !important;
    }
    .relevance-low {
        border-left-color: #dc3545 !important;
    }
    .citation {
        font-family: 'Georgia', serif;
        font-style: italic;
        color: #555;
    }
    .score-badge {
        background-color: #1E3A5F;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize session state."""
    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""


def render_sidebar():
    """Render sidebar with filters and settings."""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/search--v1.png", width=80)
        st.markdown("### 🔧 Search Settings")

        # Number of results
        top_k = st.slider(
            "Number of Results",
            min_value=5,
            max_value=25,
            value=10,
            help="How many cases to return",
        )

        # Similarity threshold
        threshold = st.slider(
            "Relevance Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum similarity score (higher = more relevant)",
        )

        st.markdown("---")
        st.markdown("### 📅 Date Filter")

        col1, col2 = st.columns(2)
        with col1:
            year_from = st.number_input(
                "From Year", min_value=1900, max_value=2026, value=2000
            )
        with col2:
            year_to = st.number_input(
                "To Year", min_value=1900, max_value=2026, value=2026
            )

        st.markdown("---")
        st.markdown("### 🏛️ Court Filter")

        courts = st.multiselect(
            "Select Courts",
            options=[
                "Supreme Court",
                "High Court",
                "Court of Appeal",
                "District Court",
                "All Courts",
            ],
            default=["All Courts"],
        )

        st.markdown("---")
        st.markdown("### 📚 Categories")

        categories = st.multiselect(
            "Legal Categories",
            options=[
                "Contract Law",
                "Criminal Law",
                "Employment Law",
                "Family Law",
                "Property Law",
                "Constitutional Law",
                "Corporate Law",
                "All Categories",
            ],
            default=["All Categories"],
        )

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **RAG Legal Search** uses AI to find 
        semantically similar cases, not just 
        keyword matches.
        
        Built with:
        - OpenAI Embeddings
        - ChromaDB Vector Store
        - LangChain RAG
        """)

        st.markdown("---")
        st.markdown("### 👨‍💻 Built by")
        st.markdown("[Abdul Salam](https://linkedin.com/in/abdul-salam-6539aa11b)")

        return {
            "top_k": top_k,
            "threshold": threshold,
            "year_from": year_from,
            "year_to": year_to,
            "courts": courts,
            "categories": categories,
        }


def render_result(result: dict, index: int):
    """Render a single search result."""
    score = result.get("score", 0)

    # Determine relevance class
    if score >= 0.8:
        relevance_class = "relevance-high"
    elif score >= 0.6:
        relevance_class = "relevance-medium"
    else:
        relevance_class = "relevance-low"

    st.markdown(
        f"""
    <div class="result-card {relevance_class}">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div>
                <h3 style="margin: 0; color: #1E3A5F;">{index}. {result.get("title", "Untitled Case")}</h3>
                <p class="citation">{result.get("citation", "Citation not available")}</p>
            </div>
            <span class="score-badge">{score:.0%} match</span>
        </div>
        
        <p style="margin: 1rem 0;">
            <strong>Court:</strong> {result.get("court", "N/A")} | 
            <strong>Date:</strong> {result.get("date", "N/A")} |
            <strong>Category:</strong> {result.get("category", "N/A")}
        </p>
        
        <p><strong>Summary:</strong></p>
        <p>{result.get("summary", "No summary available.")}</p>
        
        <p><strong>Why this is relevant:</strong></p>
        <p style="color: #555; font-style: italic;">{result.get("relevance_explanation", "AI analysis not available.")}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Expandable full text
    with st.expander("📄 View Full Text Excerpt"):
        st.markdown(result.get("excerpt", "Full text not available."))


def get_demo_results(query: str) -> list:
    """Return demo results for testing without API key."""
    return [
        {
            "title": "Smith v. Jones Construction Ltd",
            "citation": "[2023] UKSC 45",
            "court": "Supreme Court",
            "date": "2023-06-15",
            "category": "Contract Law",
            "score": 0.92,
            "summary": "Landmark case establishing the principle that implied terms in construction contracts must be necessary for business efficacy. The court held that a contractor's failure to complete works on time constituted a fundamental breach entitling the employer to terminate.",
            "relevance_explanation": f"This case is highly relevant to your query about '{query}' because it establishes key principles regarding breach of contract and termination rights in commercial agreements.",
            "excerpt": "LORD REED (with whom Lord Hodge, Lady Arden, Lord Kitchin and Lord Burrows agree):\n\n1. This appeal concerns the circumstances in which a party to a contract may terminate the contract for breach by the other party...",
        },
        {
            "title": "Ahmed v. British Airways Plc",
            "citation": "[2022] EWCA Civ 1234",
            "court": "Court of Appeal",
            "date": "2022-11-20",
            "category": "Employment Law",
            "score": 0.85,
            "summary": "Employment tribunal's decision on wrongful dismissal upheld. The Court of Appeal confirmed that employers must follow proper procedures even in cases of gross misconduct, and that failure to do so can render a dismissal unfair.",
            "relevance_explanation": f"Relevant to your search for '{query}' as it addresses procedural requirements in termination scenarios and the consequences of breach.",
            "excerpt": "LADY JUSTICE SIMLER:\n\n1. This is an appeal against the decision of the Employment Appeal Tribunal which upheld the Employment Tribunal's finding of unfair dismissal...",
        },
        {
            "title": "Global Tech Solutions v. DataCorp International",
            "citation": "[2024] EWHC 567 (Comm)",
            "court": "High Court",
            "date": "2024-03-08",
            "category": "Corporate Law",
            "score": 0.78,
            "summary": "Dispute over software licensing agreement. The court found that the defendant's use of the software exceeded the scope of the license, constituting both breach of contract and copyright infringement.",
            "relevance_explanation": f"This case relates to '{query}' through its analysis of contractual interpretation and the remedies available for breach.",
            "excerpt": "MR JUSTICE MILES:\n\n1. This is a claim for breach of contract and copyright infringement arising from the defendant's use of the claimant's proprietary software...",
        },
        {
            "title": "Khan Family Trust v. Revenue Commissioners",
            "citation": "[2023] UKFTT 789 (TC)",
            "court": "First-tier Tribunal",
            "date": "2023-09-12",
            "category": "Tax Law",
            "score": 0.71,
            "summary": "Tax tribunal case concerning the treatment of trust distributions. The tribunal held that distributions to beneficiaries were correctly assessed as income rather than capital, applying established principles from earlier Supreme Court decisions.",
            "relevance_explanation": f"While not directly about '{query}', this case demonstrates the application of contractual interpretation principles in the context of trust deeds.",
            "excerpt": "JUDGE THOMAS SCOTT:\n\n1. This appeal concerns the tax treatment of distributions made by the Khan Family Trust to its beneficiaries during the tax years 2019-20 to 2021-22...",
        },
        {
            "title": "Manchester United Football Club v. Premier League",
            "citation": "[2024] EWHC 123 (Ch)",
            "court": "High Court",
            "date": "2024-01-22",
            "category": "Sports Law",
            "score": 0.65,
            "summary": "Challenge to Premier League financial regulations. The court upheld the League's right to impose spending limits but required clearer procedural safeguards for clubs facing sanctions.",
            "relevance_explanation": f"Tangentially relevant to '{query}' through its discussion of regulatory enforcement and contractual compliance in membership organizations.",
            "excerpt": "MR JUSTICE ZACAROLI:\n\n1. This claim raises important questions about the procedural fairness requirements applicable to disciplinary proceedings conducted by sporting bodies...",
        },
    ]


def main():
    """Main application."""
    init_session_state()

    # Sidebar
    filters = render_sidebar()

    # Main content
    st.markdown(
        '<p class="main-header">🔍 RAG Legal Search</p>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">AI-powered semantic search for legal case law</p>',
        unsafe_allow_html=True,
    )

    # Search input
    query = st.text_input(
        "Enter your legal research query",
        placeholder="e.g., breach of contract cases in employment law",
        key="search_input",
    )

    # Example queries
    st.markdown("**Try these example queries:**")
    col1, col2, col3 = st.columns(3)

    example_queries = [
        "wrongful termination employment UK",
        "breach of contract remedies damages",
        "intellectual property infringement software",
    ]

    for col, example in zip([col1, col2, col3], example_queries):
        with col:
            if st.button(f"📝 {example[:30]}...", key=f"example_{example}"):
                st.session_state.search_query = example
                st.rerun()

    # Use example query if clicked
    if st.session_state.search_query:
        query = st.session_state.search_query
        st.session_state.search_query = ""

    # Search button
    if st.button("🔍 Search Cases", type="primary", use_container_width=True) or query:
        if query:
            # Check for API key
            api_key = os.getenv("OPENAI_API_KEY")

            if not api_key:
                st.warning("⚠️ No API key configured. Showing demo results.")
                st.info("Set OPENAI_API_KEY in your .env file for real search.")
                results = get_demo_results(query)
            else:
                with st.spinner("🔍 Searching case law database..."):
                    try:
                        retriever = LegalRetriever()
                        results = retriever.search(
                            query=query,
                            top_k=filters["top_k"],
                            threshold=filters["threshold"],
                            filters={
                                "year_from": filters["year_from"],
                                "year_to": filters["year_to"],
                                "courts": filters["courts"],
                                "categories": filters["categories"],
                            },
                        )
                    except Exception as e:
                        st.error(f"Search failed: {str(e)}")
                        results = get_demo_results(query)

            st.session_state.search_results = results

    # Display results
    if st.session_state.search_results:
        results = st.session_state.search_results

        st.markdown("---")
        st.markdown(f"### 📋 Found {len(results)} relevant cases")

        for i, result in enumerate(results, 1):
            render_result(result, i)

        # Export option
        st.markdown("---")
        st.markdown("### 📥 Export Results")

        import json
        import csv
        from io import StringIO

        export_data = json.dumps(results, indent=2)

        # Generate CSV
        csv_buffer = StringIO()
        if results:
            fieldnames = [
                "title",
                "citation",
                "court",
                "date",
                "category",
                "score",
                "summary",
            ]
            writer = csv.DictWriter(
                csv_buffer, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        csv_data = csv_buffer.getvalue()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "📄 Download JSON",
                data=export_data,
                file_name="legal_search_results.json",
                mime="application/json",
            )
        with col2:
            st.download_button(
                "📊 Download CSV",
                data=csv_data,
                file_name="legal_search_results.csv",
                mime="text/csv",
            )
        with col3:
            # Copy citations to clipboard
            citations = "\n".join(
                [r.get("citation", "") for r in results if r.get("citation")]
            )
            st.download_button(
                "📋 Citations Only",
                data=citations,
                file_name="citations.txt",
                mime="text/plain",
            )


if __name__ == "__main__":
    main()
