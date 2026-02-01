"""
Streamlit Web Interface for Cancer Medical Literature Search

A user-friendly web interface for searching and analyzing cancer research papers.

Run with:
    streamlit run app.py
"""

import streamlit as st
from semantic_search import CancerSearchApp, Config
import os


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Cancer Literature Search",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

@st.cache_resource
def initialize_app():
    """Initialize the search app (cached to avoid rebuilding)."""
    app = CancerSearchApp()
    try:
        app.build_or_load_index()
        return app, None
    except Exception as e:
        return None, str(e)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def display_result(result, index):
    """
    Display a single search result in a nice card format.
    
    Args:
        result: Result dictionary
        index: Result number (for display)
    """
    with st.expander(
        f"**Result {index}** - {result['pdf_file']} (Page {result['page']}) "
        f"| Similarity: {result['similarity']:.3f}",
        expanded=(index == 1)  # Auto-expand first result
    ):
        st.markdown(f"**Source:** {result['pdf_file']}")
        st.markdown(f"**Page:** {result['page']}")
        st.markdown(f"**Similarity Score:** {result['similarity']:.4f}")
        st.markdown("**Text:**")
        st.text_area(
            "Content",
            value=result['text'],
            height=150,
            key=f"result_{index}",
            label_visibility="collapsed"
        )


def display_search_results(response):
    """
    Display all search results.
    
    Args:
        response: Response dictionary from search
    """
    # Display summary if available
    if "summary" in response and response["summary"]:
        st.markdown("### 📝 Summary")
        st.info(response["summary"])
        st.markdown("---")
    
    # Display results
    st.markdown(f"### 📊 Top {len(response['results'])} Results")
    
    for i, result in enumerate(response['results'], 1):
        display_result(result, i)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application."""
    
    # Header
    st.title("🔬 Cancer Medical Literature Search")
    st.markdown(
        "Semantic search engine powered by AI for exploring cancer research papers"
    )
    st.markdown("---")
    
    # Initialize app
    app, error = initialize_app()
    
    if error:
        st.error(f"❌ Failed to initialize: {error}")
        st.info(
            "Please ensure the 'papers' directory exists and contains PDF files."
        )
        return
    
    if not app:
        st.error("Failed to initialize application")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Search settings
        st.subheader("Search Settings")
        top_k = st.slider(
            "Number of results",
            min_value=3,
            max_value=20,
            value=Config.TOP_K,
            help="How many search results to return"
        )
        
        enhance_results = st.checkbox(
            "Enable AI Summary",
            value=True,
            help="Generate an AI summary of search results"
        )
        
        st.markdown("---")
        
        # Mode selection
        st.subheader("🎯 Search Mode")
        search_mode = st.radio(
            "Choose mode:",
            ["🔍 Semantic Search", "💬 Question Answering", "📊 Comparative Analysis"],
            help="Different ways to interact with the papers"
        )
        
        st.markdown("---")
        
        # Info
        st.subheader("ℹ️ About")
        st.markdown(
            """
            This tool uses:
            - **Sentence Transformers** for semantic embeddings
            - **Cosine Similarity** for search
            - **Claude AI** for summaries and Q&A
            
            **Tips:**
            - Be specific in queries
            - Use medical terminology
            - Try different search modes
            """
        )
        
        # Stats
        if app.is_indexed:
            st.markdown("---")
            st.subheader("📈 Index Stats")
            st.metric("Total Chunks", len(app.index.chunks))
            
            # Count unique PDFs
            unique_pdfs = len(set(c['pdf_file'] for c in app.index.chunks))
            st.metric("PDF Files", unique_pdfs)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Search mode: Semantic Search
        if search_mode == "🔍 Semantic Search":
            st.header("🔍 Semantic Search")
            st.markdown(
                "Search for relevant passages across all papers using natural language."
            )
            
            query = st.text_input(
                "Enter your search query:",
                placeholder="e.g., immunotherapy side effects, BRCA1 mutations, targeted therapy...",
                key="search_query"
            )
            
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
            
            if search_button and query:
                with st.spinner("Searching..."):
                    try:
                        response = app.search(
                            query,
                            top_k=top_k,
                            enhance=enhance_results
                        )
                        display_search_results(response)
                    except Exception as e:
                        st.error(f"Search failed: {e}")
        
        # Question Answering mode
        elif search_mode == "💬 Question Answering":
            st.header("💬 Question Answering")
            st.markdown(
                "Ask questions and get AI-generated answers based on the research papers."
            )
            
            question = st.text_input(
                "Ask your question:",
                placeholder="e.g., What are the main types of immunotherapy? How does CAR-T work?",
                key="qa_question"
            )
            
            ask_button = st.button("💬 Ask", type="primary", use_container_width=True)
            
            if ask_button and question:
                with st.spinner("Thinking..."):
                    try:
                        answer = app.answer_question(question, top_k=top_k)
                        
                        st.markdown("### 📝 Answer")
                        st.markdown(answer)
                        
                        # Show source chunks
                        with st.expander("📚 View Source Chunks"):
                            results = app.index.search(question, top_k=top_k)
                            for i, result in enumerate(results, 1):
                                display_result(result, i)
                    except Exception as e:
                        st.error(f"Failed to answer: {e}")
        
        # Comparative Analysis mode
        elif search_mode == "📊 Comparative Analysis":
            st.header("📊 Comparative Analysis")
            st.markdown(
                "Compare findings across multiple papers on a topic."
            )
            
            compare_query = st.text_input(
                "Enter topic to compare:",
                placeholder="e.g., checkpoint inhibitors vs chemotherapy, different CAR-T approaches...",
                key="compare_query"
            )
            
            compare_button = st.button("📊 Compare", type="primary", use_container_width=True)
            
            if compare_button and compare_query:
                with st.spinner("Analyzing..."):
                    try:
                        # Get results
                        results = app.index.search(compare_query, top_k=top_k)
                        
                        # Generate comparison
                        comparison = app.llm.compare_results(results)
                        
                        st.markdown("### 📊 Comparative Analysis")
                        st.markdown(comparison)
                        
                        # Show source chunks
                        with st.expander("📚 View Source Chunks"):
                            for i, result in enumerate(results, 1):
                                display_result(result, i)
                    except Exception as e:
                        st.error(f"Comparison failed: {e}")
    
    with col2:
        # Example queries
        st.markdown("### 💡 Example Queries")
        
        examples = {
            "🔍 Search": [
                "BRCA1 mutations",
                "immunotherapy resistance",
                "tumor microenvironment",
                "targeted therapy mechanisms"
            ],
            "💬 Questions": [
                "What is pembrolizumab?",
                "How does radiation therapy work?",
                "What are common side effects of chemotherapy?",
                "What is precision medicine?"
            ],
            "📊 Comparisons": [
                "surgery vs radiation",
                "different immunotherapy approaches",
                "early vs late stage treatment",
                "liquid biopsy techniques"
            ]
        }
        
        for category, items in examples.items():
            with st.expander(category):
                for item in items:
                    st.markdown(f"• {item}")
        
        # Quick tips
        st.markdown("### 🎯 Tips")
        st.markdown(
            """
            - **Be specific**: "KRAS mutations" > "mutations"
            - **Use medical terms**: Better results
            - **Ask follow-ups**: Dive deeper
            - **Try different modes**: Each reveals different insights
            """
        )


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
