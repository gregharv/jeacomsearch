import streamlit as st
import sys
import os
import warnings
from datetime import datetime
from typing import List, Dict, Any, Generator
import asyncio
import time
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re

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
    .model-status {
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    .model-available { background-color: #d4edda; color: #155724; }
    .model-fallback { background-color: #fff3cd; color: #856404; }
    .model-unavailable { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_agent(preferred_model: str = "gemini"):
    """Load the RAG agent with model preference (cached for performance)"""
    return RAGAgent(preferred_model=preferred_model)

def get_confidence_color(confidence):
    """Get color class based on confidence score"""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"

def get_model_status_info(agent):
    """Get formatted model status information"""
    try:
        model_info = agent.security_router.get_model_info()
        
        # Determine status class and message
        preferred = model_info.get('preferred_model', 'unknown')
        external = model_info.get('external_model', 'none')
        models_config = model_info.get('models_configured', {})
        
        status_info = {
            'preferred_model': preferred,
            'active_model': external,
            'gemini_available': models_config.get('gemini', False),
            'openai_available': models_config.get('openai', False)
        }
        
        # Determine status message and class
        if external == preferred:
            status_info['status'] = "optimal"
            status_info['message'] = f"✅ Using preferred model: {external.upper()}"
            status_info['css_class'] = "model-available"
        elif external and external != preferred:
            status_info['status'] = "fallback"
            status_info['message'] = f"⚠️ Using fallback model: {external.upper()} (preferred: {preferred.upper()} unavailable)"
            status_info['css_class'] = "model-fallback"
        else:
            status_info['status'] = "error"
            status_info['message'] = "❌ No AI models available"
            status_info['css_class'] = "model-unavailable"
        
        return status_info
    except Exception as e:
        return {
            'status': 'error',
            'message': f"❌ Error checking model status: {e}",
            'css_class': 'model-unavailable',
            'preferred_model': 'unknown',
            'active_model': 'none',
            'gemini_available': False,
            'openai_available': False
        }

def initialize_session_state():
    """Initialize all session state variables"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = []
    if "pending_feedback" not in st.session_state:
        st.session_state.pending_feedback = {}
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = set()

def setup_feedback_database():
    """Create feedback database table if it doesn't exist"""
    conn = sqlite3.connect("crawler.db")
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            is_helpful BOOLEAN NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            sources_count INTEGER,
            confidence REAL
        )
    ''')
    
    conn.commit()
    conn.close()

def store_feedback(query: str, response: str, is_perfect: bool, sources_count: int = 0, confidence: float = 0.0):
    """Store user feedback in database"""
    try:
        conn = sqlite3.connect("crawler.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_feedback (query, response, is_helpful, sources_count, confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (query, response, is_perfect, sources_count, confidence))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error storing feedback: {e}")
        return False

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
            
            # Get feedback status for display
            feedback_status = entry.get('feedback_status')
            if feedback_status == "perfect":
                feedback_indicator = " ✅ Marked as Perfect"
            elif feedback_status == "somethings_wrong":
                feedback_indicator = " ❌ Something's Wrong"
            else:
                feedback_indicator = ""
            
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <div class="chat-timestamp">🤖 JEA Assistant - {entry['timestamp']}{feedback_indicator}</div>
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

def stream_response_generator(rag_agent, query: str, conversation_context: str) -> Generator[str, None, None]:
    """Generator that yields tokens from the RAG agent response"""
    try:
        # Enhance query with conversation context if available
        enhanced_query = query
        if conversation_context:
            enhanced_query = f"Context from previous conversation:\n{conversation_context}\n\nCurrent question: {query}"
        
        # Get streaming response from RAG agent with adjusted parameters
        for token in rag_agent.stream_query(
            query=enhanced_query,
            user_context=None,
            top_k=5,
            min_similarity=0.3,
            high_reasoning=True
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
        "sources": sources or [],
        "entry_id": len(st.session_state.chat_history),  # Add unique ID for feedback
        "feedback_status": None  # Track feedback status: None, "perfect", "somethings_wrong"
    }
    
    st.session_state.chat_history.append(chat_entry)
    
    # Add to pending feedback (will be removed once feedback is given)
    entry_id = chat_entry["entry_id"]
    st.session_state.pending_feedback[entry_id] = {
        "query": query,
        "response": full_response,
        "sources": sources or [],
        "confidence": response.confidence,
        "cached": False  # Track if this was cached pending feedback
    }
    
    # Add to conversation context
    st.session_state.conversation_context.append({
        "query": query,
        "answer": full_response
    })
    
    # Keep only last 5 exchanges
    if len(st.session_state.conversation_context) > 5:
        st.session_state.conversation_context = st.session_state.conversation_context[-5:]

def get_cache_analysis():
    """Get comprehensive cache analysis data"""
    try:
        conn = sqlite3.connect("crawler.db")
        cursor = conn.cursor()
        
        # Check if query_cache table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='query_cache'")
        if not cursor.fetchone():
            return None
        
        # Basic stats
        cursor.execute('SELECT COUNT(*) FROM query_cache')
        total_queries = cursor.fetchone()[0]
        
        if total_queries == 0:
            return {
                'total_queries': 0,
                'most_accessed': [],
                'recent_entries': [],
                'confidence_distribution': [],
                'security_confidence': [],
                'top_words': [],
                'query_patterns': [],
                'cache_performance': {}
            }
        
        # Most accessed queries
        cursor.execute('''
            SELECT original_query, access_count, confidence, security_level, 
                   timestamp, last_accessed
            FROM query_cache 
            ORDER BY access_count DESC 
            LIMIT 15
        ''')
        most_accessed = cursor.fetchall()
        
        # Recent cache entries
        cursor.execute('''
            SELECT original_query, confidence, security_level, timestamp, 
                   access_count, last_accessed
            FROM query_cache 
            ORDER BY timestamp DESC 
            LIMIT 15
        ''')
        recent_entries = cursor.fetchall()
        
        # Confidence distribution
        cursor.execute('''
            SELECT 
                CASE 
                    WHEN confidence >= 0.8 THEN 'High (≥0.8)'
                    WHEN confidence >= 0.6 THEN 'Medium (0.6-0.8)'
                    WHEN confidence >= 0.4 THEN 'Low (0.4-0.6)'
                    ELSE 'Very Low (<0.4)'
                END as confidence_range,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence
            FROM query_cache 
            GROUP BY confidence_range
            ORDER BY avg_confidence DESC
        ''')
        confidence_distribution = cursor.fetchall()
        
        # Security level analysis
        cursor.execute('''
            SELECT security_level, AVG(confidence) as avg_confidence, 
                   COUNT(*) as count, SUM(access_count) as total_accesses
            FROM query_cache 
            GROUP BY security_level
            ORDER BY count DESC
        ''')
        security_confidence = cursor.fetchall()
        
        # Query pattern analysis
        cursor.execute('SELECT original_query, access_count FROM query_cache')
        query_data = cursor.fetchall()
        
        # Analyze query patterns
        query_patterns = analyze_query_patterns([row[0] for row in query_data])
        
        # Word frequency analysis
        all_words = []
        for query, _ in query_data:
            # Clean and tokenize
            words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
            all_words.extend(words)
        
        word_counter = Counter(all_words)
        top_words = word_counter.most_common(20)
        
        # Cache performance metrics
        cursor.execute('''
            SELECT AVG(access_count) as avg_access_count,
                   MAX(access_count) as max_access_count,
                   AVG(confidence) as avg_confidence,
                   COUNT(CASE WHEN access_count > 1 THEN 1 END) as reused_queries
            FROM query_cache
        ''')
        perf_data = cursor.fetchone()
        
        cache_performance = {
            'avg_access_count': perf_data[0] or 0,
            'max_access_count': perf_data[1] or 0,
            'avg_confidence': perf_data[2] or 0,
            'reused_queries': perf_data[3] or 0,
            'cache_hit_rate': (perf_data[3] / total_queries * 100) if total_queries > 0 else 0
        }
        
        conn.close()
        
        return {
            'total_queries': total_queries,
            'most_accessed': most_accessed,
            'recent_entries': recent_entries,
            'confidence_distribution': confidence_distribution,
            'security_confidence': security_confidence,
            'top_words': top_words,
            'query_patterns': query_patterns,
            'cache_performance': cache_performance
        }
        
    except Exception as e:
        st.error(f"Error analyzing cache: {e}")
        return None

def analyze_query_patterns(queries):
    """Analyze patterns in cached queries"""
    patterns = {
        'question_types': Counter(),
        'topics': Counter(),
        'length_distribution': {'short': 0, 'medium': 0, 'long': 0}
    }
    
    question_starters = {
        'what': 'what',
        'how': 'how', 
        'when': 'when',
        'where': 'where',
        'who': 'who',
        'why': 'why',
        'can': 'can/ability',
        'is': 'is/definition',
        'do': 'do/action'
    }
    
    topic_keywords = {
        'rates': ['rate', 'cost', 'price', 'bill', 'charge'],
        'service': ['service', 'connect', 'disconnect', 'setup'],
        'payment': ['pay', 'payment', 'bill', 'assistance'],
        'outage': ['outage', 'power', 'electricity', 'restore'],
        'contact': ['contact', 'phone', 'number', 'address'],
        'hours': ['hours', 'time', 'schedule', 'open'],
        'water': ['water', 'sewer', 'wastewater'],
        'electric': ['electric', 'electricity', 'power']
    }
    
    for query in queries:
        query_lower = query.lower()
        
        # Analyze question types
        first_word = query_lower.split()[0] if query_lower.split() else ''
        for starter, category in question_starters.items():
            if query_lower.startswith(starter):
                patterns['question_types'][category] += 1
                break
        else:
            patterns['question_types']['other'] += 1
        
        # Analyze topics
        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                patterns['topics'][topic] += 1
        
        # Analyze length
        word_count = len(query.split())
        if word_count <= 5:
            patterns['length_distribution']['short'] += 1
        elif word_count <= 12:
            patterns['length_distribution']['medium'] += 1
        else:
            patterns['length_distribution']['long'] += 1
    
    return patterns

def get_feedback_history():
    """Get feedback history from the database"""
    try:
        conn = sqlite3.connect("crawler.db")
        cursor = conn.cursor()
        
        # Check if user_feedback table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_feedback'")
        if not cursor.fetchone():
            return None
        
        # Get all feedback entries
        cursor.execute('''
            SELECT query, is_helpful, timestamp, confidence, sources_count
            FROM user_feedback 
            ORDER BY timestamp DESC
        ''')
        
        feedback_data = cursor.fetchall()
        conn.close()
        
        return feedback_data
        
    except Exception as e:
        st.error(f"Error loading feedback history: {e}")
        return None

def display_prompt_analysis_tab():
    """Display the simplified feedback analysis tab"""
    st.markdown("### 📊 Question History & Feedback")
    st.markdown("View all historical questions and their feedback status.")
    
    # Load feedback history
    with st.spinner("Loading question history..."):
        feedback_data = get_feedback_history()
    
    if not feedback_data:
        st.info("No feedback history found. Ask questions and provide feedback to see data here!")
        return
    
    if len(feedback_data) == 0:
        st.info("No questions with feedback found yet. Start asking questions and providing feedback!")
        return
    
    # Convert to DataFrame for easier display
    df = pd.DataFrame(feedback_data, columns=[
        'Question', 'Feedback', 'Timestamp', 'Confidence', 'Sources'
    ])
    
    # Convert feedback boolean to readable format
    df['Feedback'] = df['Feedback'].apply(lambda x: '✅ Perfect' if x else '❌ Something Wrong')
    
    # Format timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Format confidence as percentage
    df['Confidence'] = df['Confidence'].apply(lambda x: f"{x:.1%}")
    
    # Summary metrics
    st.markdown("#### 📈 Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    total_questions = len(df)
    perfect_count = len(df[df['Feedback'] == '✅ Perfect'])
    wrong_count = len(df[df['Feedback'] == '❌ Something Wrong'])
    avg_confidence = feedback_data[0][3] if feedback_data else 0  # Get from raw data for calculation
    
    with col1:
        st.metric("Total Questions", total_questions)
    
    with col2:
        st.metric("Perfect Responses", perfect_count)
    
    with col3:
        st.metric("Needs Improvement", wrong_count)
    
    with col4:
        if total_questions > 0:
            # Calculate average confidence from raw data
            avg_conf = sum([row[3] for row in feedback_data]) / len(feedback_data)
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
        else:
            st.metric("Avg Confidence", "N/A")
    
    # Filter options
    st.markdown("#### 🔍 Filter Options")
    col1, col2 = st.columns(2)
    
    with col1:
        feedback_filter = st.selectbox(
            "Filter by feedback:",
            ["All", "✅ Perfect", "❌ Something Wrong"]
        )
    
    with col2:
        # Sort options
        sort_by = st.selectbox(
            "Sort by:",
            ["Newest First", "Oldest First", "Highest Confidence", "Lowest Confidence"]
        )
    
    # Apply filters
    filtered_df = df.copy()
    
    if feedback_filter != "All":
        filtered_df = filtered_df[filtered_df['Feedback'] == feedback_filter]
    
    # Apply sorting
    if sort_by == "Newest First":
        filtered_df = filtered_df.sort_values('Timestamp', ascending=False)
    elif sort_by == "Oldest First":
        filtered_df = filtered_df.sort_values('Timestamp', ascending=True)
    elif sort_by == "Highest Confidence":
        # Convert percentage back to float for sorting
        filtered_df['ConfidenceSort'] = filtered_df['Confidence'].str.rstrip('%').astype(float)
        filtered_df = filtered_df.sort_values('ConfidenceSort', ascending=False)
        filtered_df = filtered_df.drop('ConfidenceSort', axis=1)
    elif sort_by == "Lowest Confidence":
        filtered_df['ConfidenceSort'] = filtered_df['Confidence'].str.rstrip('%').astype(float)
        filtered_df = filtered_df.sort_values('ConfidenceSort', ascending=True)
        filtered_df = filtered_df.drop('ConfidenceSort', axis=1)
    
    # Display the dataframe
    st.markdown("#### 📋 Question History")
    
    if len(filtered_df) == 0:
        st.info(f"No questions found with the selected filter: {feedback_filter}")
    else:
        # Configure column widths and display
        st.dataframe(
            filtered_df,
            use_container_width=True,
            column_config={
                "Question": st.column_config.TextColumn(
                    "Question",
                    width="large",
                    help="The user's question"
                ),
                "Feedback": st.column_config.TextColumn(
                    "Feedback",
                    width="small",
                    help="User feedback on the response"
                ),
                "Timestamp": st.column_config.TextColumn(
                    "When Asked",
                    width="medium",
                    help="When the question was asked"
                ),
                "Confidence": st.column_config.TextColumn(
                    "Confidence",
                    width="small",
                    help="AI confidence in the response"
                ),
                "Sources": st.column_config.NumberColumn(
                    "Sources",
                    width="small",
                    help="Number of sources found"
                )
            },
            hide_index=True
        )
        
        st.markdown(f"*Showing {len(filtered_df)} of {total_questions} total questions*")
    
    # Export functionality
    st.markdown("---")
    st.markdown("#### 💾 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Download as CSV", use_container_width=True):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV File",
                data=csv,
                file_name=f"jea_questions_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    
    # Quick insights
    if total_questions > 0:
        st.markdown("---")
        st.markdown("#### 💡 Quick Insights")
        
        perfect_rate = (perfect_count / total_questions) * 100
        
        if perfect_rate >= 80:
            st.success(f"🎉 Great job! {perfect_rate:.1f}% of responses were marked as perfect!")
        elif perfect_rate >= 60:
            st.info(f"👍 Good performance! {perfect_rate:.1f}% of responses were marked as perfect.")
        else:
            st.warning(f"⚠️ Room for improvement: Only {perfect_rate:.1f}% of responses were marked as perfect.")
        
        # Show most recent problematic questions if any
        recent_wrong = df[df['Feedback'] == '❌ Something Wrong'].head(3)
        if len(recent_wrong) > 0:
            with st.expander("🔍 Recent Questions Marked as 'Something Wrong'", expanded=False):
                for _, row in recent_wrong.iterrows():
                    st.markdown(f"**Q:** {row['Question']}")
                    st.markdown(f"*Asked: {row['Timestamp']} • Confidence: {row['Confidence']} • Sources: {row['Sources']}*")
                    st.markdown("---")

def main():
    # Initialize chat history and feedback system
    initialize_session_state()
    setup_feedback_database()
    
    # Create tabs
    tab1, tab2 = st.tabs(["🔍 JEA Search Assistant", "🎯 Prompt Analysis"])
    
    with tab1:
        # Header
        st.markdown('<h1 class="main-header">🔍 JEA Search Assistant</h1>', unsafe_allow_html=True)
        st.markdown("Ask questions about JEA services, rates, policies, and more!")
        
        # Enhanced sidebar with model status (automatic selection)
        with st.sidebar:
            st.markdown("### 🤖 AI Model Status")
            
            # Load RAG agent with automatic model selection (Gemini preferred, OpenAI fallback)
            try:
                rag_agent = load_rag_agent(preferred_model="gemini")  # Always prefer Gemini first
                model_status = get_model_status_info(rag_agent)
                
                # Display model status
                st.markdown(f"""
                <div class="model-status {model_status['css_class']}">
                    {model_status['message']}
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Failed to load RAG agent: {e}")
                rag_agent = None
                
            st.markdown("---")
            
            # Feedback section
            st.markdown("### 💬 Feedback")
            
            # Show pending feedback for recent responses
            pending_items = list(st.session_state.pending_feedback.items())
            if pending_items and rag_agent is not None:
                st.markdown("**How was the most recent answer?**")
                
                # Get the most recent response that needs feedback
                most_recent_id = max(st.session_state.pending_feedback.keys())
                recent_item = st.session_state.pending_feedback[most_recent_id]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("✅ Perfect", key=f"perfect_{most_recent_id}", use_container_width=True):
                        # Mark as perfect and cache the response
                        query = recent_item["query"]
                        response = recent_item["response"]
                        sources = recent_item["sources"]
                        confidence = recent_item["confidence"]
                        
                        # Store positive feedback
                        store_feedback(query, response, True, len(sources), confidence)
                        
                        # Update chat history entry with feedback status
                        for entry in st.session_state.chat_history:
                            if entry.get("entry_id") == most_recent_id:
                                entry["feedback_status"] = "perfect"
                                break
                        
                        # Cache the response since it's marked as perfect
                        try:
                            if not recent_item.get("cached", False):
                                query_embedding = rag_agent.encode_query(query)
                                
                                # Create response object for caching
                                class FeedbackResponse:
                                    def __init__(self, answer, sources, confidence):
                                        self.answer = answer
                                        self.sources = sources
                                        self.confidence = confidence
                                
                                response_obj = FeedbackResponse(response, sources, confidence)
                                rag_agent.query_cache.cache_query_response(
                                    query, query_embedding, response_obj, "external"
                                )
                                st.success("✅ Perfect response cached for future use!")
                            else:
                                st.success("✅ Thank you for confirming this was perfect!")
                        except Exception as e:
                            st.warning(f"Feedback recorded, but caching failed: {e}")
                        
                        # Remove from pending feedback but keep the response visible
                        del st.session_state.pending_feedback[most_recent_id]
                        st.rerun()
                
                with col2:
                    if st.button("❌ Something's Wrong", key=f"wrong_{most_recent_id}", use_container_width=True):
                        # Mark as wrong and do NOT cache
                        query = recent_item["query"]
                        response = recent_item["response"]
                        sources = recent_item["sources"]
                        confidence = recent_item["confidence"]
                        
                        # Store negative feedback
                        store_feedback(query, response, False, len(sources), confidence)
                        
                        # Update chat history entry with feedback status
                        for entry in st.session_state.chat_history:
                            if entry.get("entry_id") == most_recent_id:
                                entry["feedback_status"] = "somethings_wrong"
                                break
                        
                        # Remove from pending feedback
                        del st.session_state.pending_feedback[most_recent_id]
                        st.success("❌ Feedback recorded. This response will not be cached.")
                        st.rerun()
            
            else:
                if rag_agent is None:
                    st.warning("⚠️ AI model not available for feedback.")
                else:
                    st.info("Ask a question to provide feedback on the response!")
            
            # Show feedback summary if there's chat history
            if st.session_state.chat_history:
                st.markdown("---")
                st.markdown("#### 📈 Feedback Summary")
                
                perfect_count = len([entry for entry in st.session_state.chat_history if entry.get("feedback_status") == "perfect"])
                wrong_count = len([entry for entry in st.session_state.chat_history if entry.get("feedback_status") == "somethings_wrong"])
                total_responses = len(st.session_state.chat_history)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("✅ Perfect", perfect_count)
                with col2:
                    st.metric("❌ Wrong", wrong_count)
                
                if total_responses > 0:
                    feedback_rate = (perfect_count + wrong_count) / total_responses * 100
                    st.metric("Feedback Rate", f"{feedback_rate:.0f}%")
        
        # Check if agent is available before proceeding
        if rag_agent is None:
            st.error("❌ Cannot proceed without a working AI model. Please check your configuration.")
            return
        
        # Main search interface
        st.markdown("### 🔍 Ask JEA Assistant")
        
        with st.form(key="search_form", clear_on_submit=False):
            query = st.text_input(
                "What would you like to know about JEA services?",
                placeholder="e.g., What are the current electric rates?",
                help="Ask questions about JEA services, rates, policies, contact information, and more."
            )
            
            search_button = st.form_submit_button("🔍 Search", type="primary", use_container_width=True)
        
        # Process search with streaming and error handling
        if search_button and query:
            with st.spinner("🔍 Searching JEA knowledge base..."):
                try:
                    # Reset state
                    rag_agent.reset_state()
                    
                    # Build conversation context
                    conversation_context = build_conversation_context()
                    
                    # Create placeholder for streaming response
                    response_placeholder = st.empty()
                    
                    # Stream the response
                    full_response = ""
                    error_occurred = False
                    timeout_occurred = False
                    
                    for token in stream_response_generator(rag_agent, query, conversation_context):
                        full_response += token
                        
                        # Check for network error patterns in the response
                        if "Network Connection Issue" in token or "connectivity issues" in token:
                            error_occurred = True
                        
                        # Check for timeout patterns
                        if "Request Timed Out" in token or "timed out" in token.lower():
                            timeout_occurred = True
                        
                        # Update the display in real-time
                        with response_placeholder.container():
                            st.markdown("### 🤖 JEA Assistant Response:")
                            if error_occurred:
                                st.warning("⚠️ Network connectivity issue detected")
                            elif timeout_occurred:
                                st.warning("⚠️ Request timed out - showing available sources")
                            st.markdown(full_response + "▋")
                    
                    # Final update without cursor
                    with response_placeholder.container():
                        st.markdown("### 🤖 JEA Assistant Response:")
                        if error_occurred:
                            st.warning("⚠️ Please try your question again in a few moments. SSL certificate Error.")
                        elif timeout_occurred:
                            st.warning("⚠️ Response timed out - but here are relevant sources that may contain your answer")
                        st.markdown(full_response)
                    
                    # Get sources after streaming - even if there were errors/timeouts
                    time.sleep(0.1)
                    sources = rag_agent.get_last_sources()
                    
                    # Show sources even when there are network issues or timeouts
                    if sources and (error_occurred or timeout_occurred):
                        st.markdown("### 📚 Relevant Sources Found")
                        st.info("💡 **The answer to your question may be found in one of these sources below:**")
                        
                        with st.expander(f"📖 View {len(sources)} relevant JEA sources", expanded=True):
                            for j, source in enumerate(sources, 1):
                                st.markdown(f"**{j}. {source.title}** (Relevance: {source.similarity_score:.3f})")
                                st.markdown(f"🔗 [{source.url}]({source.url})")
                                st.markdown(f"_{source.chunk_text[:300]}..._")
                                if j < len(sources):  # Don't add separator after last item
                                    st.markdown("---")
                        
                        st.markdown("""
                        💡 **Tip:** Click on the links above to visit the JEA pages directly, or try asking your question again in a few moments.
                        """)
                    
                    # For successful responses, add to chat history and show sources normally
                    elif not error_occurred and not timeout_occurred:
                        # Add to chat history (this will add to pending feedback)
                        add_to_chat_history_streaming(query, full_response, sources)
                        
                        # Show sources normally
                        if sources:
                            with st.expander(f"📚 View {len(sources)} sources", expanded=False):
                                for j, source in enumerate(sources, 1):
                                    st.markdown(f"**{j}. {source.title}** (Relevance: {source.similarity_score:.3f})")
                                    st.markdown(f"🔗 [{source.url}]({source.url})")
                                    st.markdown(f"_{source.chunk_text[:200]}..._")
                                    st.markdown("---")
                        else:
                            st.info("No sources found for this query.")
                    
                    # For error cases, don't add to chat history but still show helpful message
                    if error_occurred or timeout_occurred:
                        if not sources:
                            st.info("💡 **Tip:** Network issues are usually temporary. Please try your question again!")
                        # Don't add error responses to chat history
                    
                except Exception as e:
                    error_message = str(e).lower()
                    
                    # Check if it's a network-related error
                    if any(term in error_message for term in ['ssl', 'certificate', 'handshake', 'connection', 'network', 'timeout']):
                        st.error("🌐 **Network Connection Issue**")
                        st.markdown("""
                        We're experiencing connectivity issues. This could be due to:
                        - SSL certificate verification problems  
                        - Network connectivity issues
                        - External service unavailability
                        
                        **Please try again in a few moments.** If the issue persists:
                        - Contact JEA Customer Service at **(904) 665-6000**
                        - Visit **jea.com** directly
                        """)
                        
                        # Try to get sources even after an exception
                        try:
                            sources = rag_agent.get_last_sources()
                            if sources:
                                st.markdown("### 📚 Available Sources")
                                st.info("💡 **Your answer might be found in these JEA sources:**")
                                
                                with st.expander(f"📖 View {len(sources)} JEA sources", expanded=True):
                                    for j, source in enumerate(sources, 1):
                                        st.markdown(f"**{j}. {source.title}**")
                                        st.markdown(f"🔗 [{source.url}]({source.url})")
                                        st.markdown(f"_{source.chunk_text[:250]}..._")
                                        if j < len(sources):
                                            st.markdown("---")
                        except:
                            pass  # If we can't get sources, just continue
                            
                    else:
                        st.error(f"❌ An error occurred: {e}")
                        st.exception(e)
        
        # Always display chat history
        display_chat_history()
    
    with tab2:
        display_prompt_analysis_tab()

if __name__ == "__main__":
    main() 