"""
Streamlit Web Interface for Cancer Medical Literature Search
WITH RATE LIMITING - 10 Questions Per User Per Day

Run with:
    streamlit run app_with_limits.py
"""

import streamlit as st
from semantic_search import CancerSearchApp, Config
import os
from datetime import datetime, timedelta
import hashlib

# Get API key from Streamlit secrets
api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    st.error("Please set ANTHROPIC_API_KEY in Streamlit secrets")


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
# RATE LIMITING SYSTEM
# ============================================================================

def get_user_id():
    """
    Get a unique identifier for the user.
    Uses session state to track users across page reloads.
    """
    if 'user_id' not in st.session_state:
        # Create unique ID based on session
        import uuid
        st.session_state.user_id = str(uuid.uuid4())
    return st.session_state.user_id


def init_rate_limit():
    """Initialize rate limiting in session state."""
    if 'question_count' not in st.session_state:
        st.session_state.question_count = 0
        st.session_state.last_question_time = None
        st.session_state.questions_asked_today = []


def can_ask_question():
    """
    Check if user can ask another question.
    Returns (can_ask: bool, message: str)
    """
    init_rate_limit()
    
    # Get today's date
    today = datetime.now().date()
    
    # Filter questions asked today
    today_questions = [
        q for q in st.session_state.questions_asked_today 
        if q.date() == today
    ]
    
    # Update count
    st.session_state.questions_asked_today = today_questions
    
    # Check limit
    DAILY_LIMIT = 10  # Change this number to allow more questions per day
    
    if len(today_questions) >= DAILY_LIMIT:
        # Calculate time until reset
        tomorrow = datetime.combine(today + timedelta(days=1), datetime.min.time())
        time_until_reset = tomorrow - datetime.now()
        hours = int(time_until_reset.total_seconds() // 3600)
        minutes = int((time_until_reset.total_seconds() % 3600) // 60)
        
        return False, f"⏰ Daily limit reached ({DAILY_LIMIT} questions/day). Try again in {hours}h {minutes}m."
    
    return True, ""


def record_question():
    """Record that a question was asked."""
    init_rate_limit()
    st.session_state.questions_asked_today.append(datetime.now())
    st.session_state.question_count += 1


def get_usage_stats():
    """Get usage statistics for display."""
    init_rate_limit()
    today = datetime.now().date()
    today_count = len([
        q for q in st.session_state.questions_asked_today 
        if q.date() == today
    ])
    
    DAILY_LIMIT = 10
    remaining = max(0, DAILY_LIMIT - today_count)
    
    return {
        'used_today': today_count,
        'remaining_today': remaining,
        'total_all_time': st.session_state.question_count
    }


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
    """Display a single search result in a nice card format."""
    with st.expander(
        f"**Result {index}** - {result['pdf_file']} (Page {result['page']}) "
        f"| Similarity: {result['similarity']:.3f}",
        expanded=(index == 1)
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
    """Display all search results."""
    if "summary" in response and response["summary"]:
        st.markdown("### 📝 Summary")
        st.info(response["summary"])
        st.markdown("---")
    
    st.markdown(f"### 📊 Top {len(response['results'])} Results")
    
    for i, result in enumerate(response['results'], 1):
        display_result(result, i)


def display_usage_stats():
    """Display usage statistics in sidebar."""
    stats = get_usage_stats()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Your Usage")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Today", stats['used_today'])
    with col2:
        st.metric("Remaining", stats['remaining_today'])
    
    # Progress bar
    DAILY_LIMIT = 1
    progress = min(stats['used_today'] / DAILY_LIMIT, 1.0)
    st.sidebar.progress(progress)
    
    if stats['remaining_today'] == 0:
        st.sidebar.error("🚫 Daily limit reached")
    elif stats['remaining_today'] == 1:
        st.sidebar.warning("⚠️ Last question remaining today")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application."""
    
    # Initialize rate limiting
    init_rate_limit()
    
    # Header
    st.title("🔬 Cancer Medical Literature Search")
    st.markdown(
        "Semantic search engine powered by AI for exploring cancer research papers"
    )
    
    # Show rate limit warning at top
    stats = get_usage_stats()
    if stats['remaining_today'] == 0:
        st.error("⏰ You've used your daily question limit. Search mode is still available, or come back tomorrow for Q&A!")
    
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
            help="Generate an AI summary of search results (uses API credits)"
        )
        
        st.markdown("---")
        
        # Mode selection
        st.subheader("🎯 Search Mode")
        search_mode = st.radio(
            "Choose mode:",
            ["🔍 Semantic Search", "💬 Question Answering", "📊 Comparative Analysis"],
            help="Different ways to interact with the papers"
        )
        
        # Display usage stats
        display_usage_stats()
        
        # Admin reset button (for testing)
        #st.markdown("---")
        #if st.button("🔄 Reset Usage (Testing)", help="Reset your daily usage counter"):
            #st.session_state.questions_asked_today = []
            #st.session_state.question_count = 0
            #st.success("✅ Usage reset! Refresh the page.")
            #st.rerun()
        
        st.markdown("---")
        
        # Info
        st.subheader("ℹ️ About")
        st.markdown(
            """
            This tool uses:
            - **Sentence Transformers** for semantic embeddings
            - **Cosine Similarity** for search
            - **Claude AI** for summaries and Q&A
            
            **Usage Limits:**
            - 🔍 Semantic Search: **Unlimited**
            - 💬 Q&A: **10 questions/day**
            - 📊 Comparison: **10 analysis/day**
            
            **Tips:**
            - Be specific in queries
            - Use medical terminology
            - Try different search modes
            """
        )
        
        # Stats
        #if app.is_indexed:
            #st.markdown("---")
            #st.subheader("📈 Index Stats")
            #st.metric("Total Chunks", len(app.index.chunks))
            
            #unique_pdfs = len(set(c['pdf_file'] for c in app.index.chunks))
            #st.metric("PDF Files", unique_pdfs)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Search mode: Semantic Search (NO LIMITS)
        if search_mode == "🔍 Semantic Search":
            st.header("🔍 Semantic Search")
            st.markdown(
                "Search for relevant passages across all papers using natural language. **Unlimited searches!**"
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
        
        # Question Answering mode (WITH LIMITS)
        elif search_mode == "💬 Question Answering":
            st.header("💬 Question Answering")
            st.markdown(
                "Ask questions and get AI-generated answers based on the research papers. **Limited to 10 questions per day.**"
            )
            
            # Check rate limit
            can_ask, limit_msg = can_ask_question()
            
            if not can_ask:
                st.warning(limit_msg)
                st.info("💡 **Tip:** Use Semantic Search mode for unlimited searches!")
            
            question = st.text_input(
                "Ask your question:",
                placeholder="e.g., What are the main types of immunotherapy? How does CAR-T work?",
                key="qa_question",
                disabled=not can_ask
            )
            
            ask_button = st.button(
                "💬 Ask", 
                type="primary", 
                use_container_width=True,
                disabled=not can_ask
            )
            
            if ask_button and question and can_ask:
                with st.spinner("Thinking..."):
                    try:
                        # Get the answer first
                        answer = app.answer_question(question, top_k=top_k)
                        
                        # Record the question AFTER getting answer
                        record_question()
                        
                        st.markdown("### 📝 Answer")
                        st.markdown(answer)
                        
                        # Show source chunks
                        with st.expander("📚 View Source Chunks"):
                            results = app.index.search(question, top_k=top_k)
                            for i, result in enumerate(results, 1):
                                display_result(result, i)
                        
                        # Show success message
                        st.success(f"✅ Question answered! You have {get_usage_stats()['remaining_today']} questions remaining today.")
                        
                    except Exception as e:
                        st.error(f"Failed to answer: {e}")
        
        # Comparative Analysis mode (WITH LIMITS)
        elif search_mode == "📊 Comparative Analysis":
            st.header("📊 Comparative Analysis")
            st.markdown(
                "Compare findings across multiple papers on a topic. **Limited to 10 analyses per day.**"
            )
            
            # Check rate limit
            can_ask, limit_msg = can_ask_question()
            
            if not can_ask:
                st.warning(limit_msg)
                st.info("💡 **Tip:** Use Semantic Search mode for unlimited searches!")
            
            compare_query = st.text_input(
                "Enter topic to compare:",
                placeholder="e.g., checkpoint inhibitors vs chemotherapy, different CAR-T approaches...",
                key="compare_query",
                disabled=not can_ask
            )
            
            compare_button = st.button(
                "📊 Compare", 
                type="primary", 
                use_container_width=True,
                disabled=not can_ask
            )
            
            if compare_button and compare_query and can_ask:
                with st.spinner("Analyzing..."):
                    try:
                        # Get results first
                        results = app.index.search(compare_query, top_k=top_k)
                        comparison = app.llm.compare_results(results)
                        
                        # Record the question AFTER getting results
                        record_question()
                        
                        st.markdown("### 📊 Comparative Analysis")
                        st.markdown(comparison)
                        
                        # Show source chunks
                        with st.expander("📚 View Source Chunks"):
                            for i, result in enumerate(results, 1):
                                display_result(result, i)
                        
                        # Show success message
                        st.success(f"✅ Analysis complete! You have {get_usage_stats()['remaining_today']} analyses remaining today.")
                        
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
                "What are common side effects?",
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
            - **Search is unlimited**: Use it freely!
            - **Q&A is limited**: 10 per day to control costs
            - **Be specific**: "KRAS mutations" > "mutations"
            - **Use medical terms**: Better results
            """
        )


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
