import streamlit as st
import sys
import os
import warnings
from datetime import datetime
from typing import List, Dict, Any, Generator
import asyncio
import time

# Suppress warnings
warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Import your RAG agent
from rag_agent import RAGAgent

# Page configuration
st.set_page_config(
    page_title="JEA Search Assistant",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f4e79;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f8f9fa;
        border-left: 4px solid #28a745;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    .chat-timestamp {
        font-size: 0.8rem;
        color: #666;
        margin-bottom: 0.5rem;
    }
    .source-summary {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_agent():
    """Load the RAG agent (cached for performance)"""
    return RAGAgent()

def get_confidence_color(confidence):
    """Get color class based on confidence score"""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"

def initialize_chat_history():
    """Initialize chat history in session state"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = []

def add_to_chat_history(query: str, response: Any):
    """Add a query-response pair to chat history"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    chat_entry = {
        "timestamp": timestamp,
        "query": query,
        "response": response,
        "sources": response.sources if hasattr(response, 'sources') else []
    }
    
    st.session_state.chat_history.append(chat_entry)
    
    # Add to conversation context for the RAG agent (keep last 5 exchanges)
    st.session_state.conversation_context.append({
        "query": query,
        "answer": response.answer if hasattr(response, 'answer') else str(response)
    })
    
    # Keep only last 5 exchanges to prevent context from getting too long
    if len(st.session_state.conversation_context) > 5:
        st.session_state.conversation_context = st.session_state.conversation_context[-5:]

def build_conversation_context():
    """Build conversation context string for the RAG agent"""
    if not st.session_state.conversation_context:
        return ""
    
    context_parts = []
    for i, exchange in enumerate(st.session_state.conversation_context[-3:], 1):  # Last 3 exchanges
        context_parts.append(f"Previous Q{i}: {exchange['query']}")
        context_parts.append(f"Previous A{i}: {exchange['answer'][:200]}...")
    
    return "\n".join(context_parts)

def display_chat_history():
    """Display the chat history"""
    if not st.session_state.chat_history:
        st.info("💬 Your conversation will appear here. Ask a question to get started!")
        return
    
    st.markdown("## 💬 Conversation History")
    
    # Reverse order to show most recent first
    for i, entry in enumerate(reversed(st.session_state.chat_history)):
        with st.container():
            # User message
            st.markdown(f"""
            <div class="chat-message user-message">
                <div class="chat-timestamp">👤 You - {entry['timestamp']}</div>
                <strong>{entry['query']}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Assistant response
            confidence_class = get_confidence_color(entry['response'].confidence)
            source_count = len(entry['sources'])
            
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <div class="chat-timestamp">🤖 JEA Assistant - {entry['timestamp']}</div>
                {entry['response'].answer}
                <div class="source-summary">
                    <span class="{confidence_class}">Confidence: {entry['response'].confidence:.2f}</span> • 
                    {source_count} sources found
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show sources in an expander for recent messages
            if i < 3 and entry['sources']:  # Only show sources for last 3 messages
                with st.expander(f"📚 View {len(entry['sources'])} sources", expanded=False):
                    for j, source in enumerate(entry['sources'], 1):
                        st.markdown(f"**{j}. {source.title}** (Relevance: {source.similarity_score:.3f})")
                        st.markdown(f"🔗 [{source.url}]({source.url})")
                        st.markdown(f"_{source.chunk_text[:200]}..._")
                        st.markdown("---")

def stream_response_generator(rag_agent, query: str, conversation_context: str, high_reasoning: bool = True) -> Generator[str, None, None]:
    """Generator that yields tokens from the RAG agent response"""
    try:
        # Enhance query with conversation context if available
        enhanced_query = query
        if conversation_context:
            enhanced_query = f"Context from previous conversation:\n{conversation_context}\n\nCurrent question: {query}"
        
        # Get streaming response from RAG agent
        for token in rag_agent.stream_query(
            query=enhanced_query,
            user_context=None,
            top_k=5,
            min_similarity=0.3,
            high_reasoning=high_reasoning
        ):
            yield token
            
    except Exception as e:
        yield f"❌ Error: {str(e)}"

def add_to_chat_history_streaming(query: str, full_response: str, sources: List[Any] = None):
    """Add a query-response pair to chat history for streaming responses"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Create a mock response object for compatibility
    class StreamedResponse:
        def __init__(self, answer, sources, confidence=0.8):
            self.answer = answer
            self.sources = sources or []
            self.confidence = confidence
    
    response = StreamedResponse(full_response, sources)
    
    chat_entry = {
        "timestamp": timestamp,
        "query": query,
        "response": response,
        "sources": sources or []
    }
    
    st.session_state.chat_history.append(chat_entry)
    
    # Add to conversation context
    st.session_state.conversation_context.append({
        "query": query,
        "answer": full_response
    })
    
    # Keep only last 5 exchanges
    if len(st.session_state.conversation_context) > 5:
        st.session_state.conversation_context = st.session_state.conversation_context[-5:]

def main():
    # Initialize chat history
    initialize_chat_history()
    
    # Header
    st.markdown('<h1 class="main-header">🔍 JEA Search Assistant</h1>', unsafe_allow_html=True)
    st.markdown("Ask questions about JEA services, rates, policies, and more!")
    
    # Sidebar configuration
    st.sidebar.markdown("### ⚙️ Search Settings")
    
    high_reasoning = st.sidebar.checkbox(
        "🧠 High Reasoning Mode",
        value=True,
        help="Enable comprehensive search with multiple query variants and enhanced filtering. Unchecking will use faster, simpler search."
    )
    
    if high_reasoning:
        st.sidebar.success("🔬 Using comprehensive search with multiple strategies")
    else:
        st.sidebar.info("⚡ Using fast search with 2 query variants")
    
    # Load RAG agent
    try:
        rag_agent = load_rag_agent()
    except Exception as e:
        st.error(f"Failed to load RAG agent: {e}")
        return
    
    # Main search interface
    st.markdown("### 🔍 Ask JEA Assistant")
    
    # Use a form to handle Enter key submission
    with st.form(key="search_form", clear_on_submit=False):
        # Text input
        query = st.text_input(
            "What would you like to know about JEA services?",
            placeholder="e.g., What are the current electric rates?",
            help="Ask questions about JEA services, rates, policies, contact information, and more."
        )
        
        # Submit button with dynamic text
        search_text = "🔍 Search (High Reasoning)" if high_reasoning else "⚡ Search (Fast)"
        search_button = st.form_submit_button(search_text, type="primary", use_container_width=True)
    
    # Process search with streaming
    if search_button and query:
        search_mode = "comprehensive search" if high_reasoning else "fast search"
        with st.spinner(f"🔍 Connecting to JEA knowledge base ({search_mode})..."):
            try:
                # FORCE complete reset of RAG agent state
                rag_agent.reset_state()
                
                # Build conversation context
                conversation_context = build_conversation_context()
                
                # Create placeholder for streaming response
                response_placeholder = st.empty()
                
                # Stream the response
                full_response = ""
                for token in stream_response_generator(rag_agent, query, conversation_context, high_reasoning):
                    full_response += token
                    
                    # Update the display in real-time
                    with response_placeholder.container():
                        st.markdown("### 🤖 JEA Assistant Response:")
                        st.markdown(full_response + "▋")  # Add cursor effect
                
                # Final update without cursor
                with response_placeholder.container():
                    st.markdown("### 🤖 JEA Assistant Response:")
                    st.markdown(full_response)
                
                # Get sources after streaming is complete - with delay to ensure they're set
                time.sleep(0.1)  # Small delay to ensure sources are properly set
                sources = rag_agent.get_last_sources()
                
                # Add to chat history
                add_to_chat_history_streaming(query, full_response, sources)
                
                # Show sources
                if sources:
                    mode_info = " (High Reasoning)" if high_reasoning else " (Fast Search)"
                    with st.expander(f"📚 View {len(sources)} sources{mode_info}", expanded=False):
                        for j, source in enumerate(sources, 1):
                            st.markdown(f"**{j}. {source.title}** (Relevance: {source.similarity_score:.3f})")
                            st.markdown(f"🔗 [{source.url}]({source.url})")
                            st.markdown(f"_{source.chunk_text[:200]}..._")
                            st.markdown("---")
                else:
                    st.info("No sources found for this query.")
                
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
                st.exception(e)
    
    # Show current conversation stats
    if st.session_state.chat_history:
        st.sidebar.markdown("### 📊 Conversation Stats")
        st.sidebar.metric("Questions Asked", len(st.session_state.chat_history))
        
        # Average confidence
        avg_confidence = sum(entry['response'].confidence for entry in st.session_state.chat_history) / len(st.session_state.chat_history)
        st.sidebar.metric("Average Confidence", f"{avg_confidence:.2f}")
        
        # Most recent confidence
        if st.session_state.chat_history:
            recent_confidence = st.session_state.chat_history[-1]['response'].confidence
            st.sidebar.metric("Last Response Confidence", f"{recent_confidence:.2f}")
    
    # Footer
    st.markdown("---")
    mode_tip = "High Reasoning mode uses comprehensive search for better accuracy" if high_reasoning else "Fast mode uses simpler search for quicker responses"
    st.markdown(f"💡 **Tip:** {mode_tip}. The assistant remembers your recent questions for context!")

if __name__ == "__main__":
    main() 