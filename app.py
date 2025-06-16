from fasthtml.common import *
from fasthtml.core import Response
from starlette.responses import RedirectResponse
from rag_agent import RAGAgent
import json
import uuid
from datetime import datetime
from urllib.parse import urlparse
import sqlite3
from starlette.middleware.sessions import SessionMiddleware
import os

# Run with python app.py --port 8050 for local testing

# Store conversation history and pending responses
conversations = {}
pending_responses = {}

def _create_source_favicon(source, number):
    """Creates a simple favicon element for sources."""
    domain = urlparse(source.url).netloc
    return A(
        Img(
            src=f"https://www.google.com/s2/favicons?domain={domain}&sz=20",
            alt=f"Source {number}",
            cls="source-favicon-simple",
            onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAyMCAyMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjIwIiBoZWlnaHQ9IjIwIiBmaWxsPSIjZjNmNGY2IiByeD0iMyIvPgo8cGF0aCBkPSJNMTAgNUwxNSAxMEwxMCAxNUw1IDEwTDEwIDVaIiBmaWxsPSIjOWNhM2FmIi8+Cjwvc3ZnPgo='"
        ),
        href=source.url,
        target="_blank",
        cls="source-favicon-link",
        title=source.url
    )

def _create_sources_section(sources):
    """Creates the entire sources section Div."""
    if not sources:
        return None
    
    source_favicons = [_create_source_favicon(s, i) for i, s in enumerate(sources, 1)]
    
    return Div(
        Span("Sources: ", cls="sources-label"),
        Div(*source_favicons, cls="sources-favicons"),
        cls="sources-simple"
    )

def setup_feedback_database():
    """Create feedback database table if it doesn't exist and add missing columns"""
    from rag_agent import get_app_db_path
    app_db_path = get_app_db_path()  # Use app database for feedback data
    
    conn = sqlite3.connect(app_db_path)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
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
    
    # Check if message_id column exists and add it if not
    cursor.execute("PRAGMA table_info(user_feedback)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'message_id' not in columns:
        print("Adding message_id column to user_feedback table...")
        cursor.execute('ALTER TABLE user_feedback ADD COLUMN message_id TEXT')
    
    if 'session_id' not in columns:
        print("Adding session_id column to user_feedback table...")
        cursor.execute('ALTER TABLE user_feedback ADD COLUMN session_id TEXT')
    
    if 'interaction_id' not in columns:
        print("Adding interaction_id column to user_feedback table...")
        cursor.execute('ALTER TABLE user_feedback ADD COLUMN interaction_id INTEGER')
    
    # Enhance rag_interactions table with missing fields
    cursor.execute("PRAGMA table_info(rag_interactions)")
    rag_columns = [column[1] for column in cursor.fetchall()]
    
    if 'message_id' not in rag_columns:
        print("Adding message_id column to rag_interactions table...")
        cursor.execute('ALTER TABLE rag_interactions ADD COLUMN message_id TEXT')
    
    if 'session_id' not in rag_columns:
        print("Adding session_id column to rag_interactions table...")
        cursor.execute('ALTER TABLE rag_interactions ADD COLUMN session_id TEXT')
    
    if 'response_text' not in rag_columns:
        print("Adding response_text column to rag_interactions table...")
        cursor.execute('ALTER TABLE rag_interactions ADD COLUMN response_text TEXT')
    
    if 'sources_json' not in rag_columns:
        print("Adding sources_json column to rag_interactions table...")
        cursor.execute('ALTER TABLE rag_interactions ADD COLUMN sources_json TEXT')
    
    if 'cached_from_query_cache' not in rag_columns:
        print("Adding cached_from_query_cache column to rag_interactions table...")
        cursor.execute('ALTER TABLE rag_interactions ADD COLUMN cached_from_query_cache BOOLEAN DEFAULT FALSE')
    
    conn.commit()
    conn.close()
    print("✅ Feedback database setup complete")
    print(f"📊 Using app database: {app_db_path}")

def store_interaction(message_id: str, session_id: str, query: str, response_text: str, 
                     sources: list, security_level: str = "external", confidence: float = 0.0, 
                     cached_from_query_cache: bool = False, user_context: dict = None):
    """Store all interactions in rag_interactions table"""
    try:
        from rag_agent import get_app_db_path
        app_db_path = get_app_db_path()  # Use app database
        
        conn = sqlite3.connect(app_db_path)
        cursor = conn.cursor()
        
        # Serialize sources for storage
        sources_json = json.dumps([{
            'url': s.url,
            'title': s.title,
            'similarity_score': s.similarity_score
        } for s in sources]) if sources else None
        
        user_context_json = json.dumps(user_context) if user_context else None
        
        cursor.execute('''
            INSERT INTO rag_interactions (
                message_id, session_id, query, response_text, security_level, 
                num_sources_retrieved, confidence_score, sources_json, 
                cached_from_query_cache, user_context_json, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ''', (
            message_id, session_id, query, response_text, security_level,
            len(sources) if sources else 0, confidence, sources_json,
            cached_from_query_cache, user_context_json
        ))
        
        interaction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"✅ Interaction stored in rag_interactions table: ID {interaction_id}")
        return interaction_id
        
    except Exception as e:
        print(f"❌ Error storing interaction: {e}")
        return None

def store_feedback(message_id: str, query: str, response: str, is_helpful: bool, sources_count: int = 0, confidence: float = 0.0, session_id: str = None):
    """Store user feedback in database"""
    try:
        from rag_agent import get_app_db_path
        app_db_path = get_app_db_path()  # Use app database
        
        conn = sqlite3.connect(app_db_path)
        cursor = conn.cursor()
        
        # Find the interaction_id for this message_id
        cursor.execute('SELECT id FROM rag_interactions WHERE message_id = ?', (message_id,))
        interaction_result = cursor.fetchone()
        interaction_id = interaction_result[0] if interaction_result else None
        
        # Insert feedback
        cursor.execute('''
            INSERT INTO user_feedback (message_id, query, response, is_helpful, sources_count, confidence, session_id, interaction_id, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ''', (message_id, query, response, is_helpful, sources_count, confidence, session_id, interaction_id))
        
        conn.commit()
        
        # If positive feedback, add to cache (if not already cached)
        if is_helpful:
            print(f"✅ Positive feedback - checking if should be cached...")
            add_positive_response_to_cache(message_id, query, response, confidence, cursor)
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error storing feedback: {e}")
        return False

def add_positive_response_to_cache(message_id: str, query: str, response: str, confidence: float, cursor):
    """Add positively rated response to query cache"""
    try:
        # Check if this interaction was already cached
        cursor.execute('SELECT cached_from_query_cache FROM rag_interactions WHERE message_id = ?', (message_id,))
        result = cursor.fetchone()
        was_cached = result[0] if result else False
        
        if was_cached:
            print(f"   Response was from cache, updating access count...")
            # Update access count for existing cache entry
            query_embedding = rag_agent.encode_query(query)
            query_hash = rag_agent.query_cache._generate_query_hash(query, "external")
            cursor.execute('UPDATE query_cache SET access_count = access_count + 1, last_accessed = datetime("now") WHERE query_hash = ?', (query_hash,))
        else:
            print(f"   Adding new positive response to cache...")
            # Add new entry to cache
            query_embedding = rag_agent.encode_query(query)
            
            # Get sources from the interaction
            cursor.execute('SELECT sources_json FROM rag_interactions WHERE message_id = ?', (message_id,))
            sources_result = cursor.fetchone()
            sources_json = sources_result[0] if sources_result else "[]"
            
            # Use the RAG agent's caching mechanism
            from rag_agent import RetrievedDocument, RAGResponse
            
            # Parse sources back to objects for caching
            sources_data = json.loads(sources_json) if sources_json else []
            sources = [RetrievedDocument(
                document_id=0,  # Not available from JSON
                url=s['url'],
                title=s['title'],
                chunk_text="",  # Not stored in JSON
                similarity_score=s['similarity_score'],
                source_type="",  # Not available
                chunk_index=0   # Not available
            ) for s in sources_data]
            
            rag_response = RAGResponse(
                answer=response,
                sources=sources,
                reasoning="",
                confidence=confidence,
                security_level="external"
            )
            
            rag_agent.query_cache.cache_query_response(query, query_embedding, rag_response, "external")
            
    except Exception as e:
        print(f"   ⚠️ Error adding to cache: {e}")

# Initialize the RAG agent with separate database paths
from rag_agent import get_knowledge_db_path, get_app_db_path
rag_agent = RAGAgent(
    knowledge_db_path=get_knowledge_db_path(),  # For documents and embeddings
    app_db_path=get_app_db_path(),              # For interactions, feedback, cache
    preferred_model="gemini"
)

app, rt = fast_app(
    pico=False,  # Disable Pico CSS to use our custom CSS
    hdrs=(
        # Force disable any Pico CSS
        Style("/* Disable any inherited Pico CSS */ .pico { display: none !important; }"),
        # Load our custom CSS with cache busting
        Link(rel="stylesheet", href=f"/styles.css?v={int(datetime.now().timestamp())}", type="text/css"),
        # Load JavaScript libraries  
        Script(type="text/javascript", src="/htmx.min.js"),
        Script(src="/marked.min.js")
    )
)

# Add session middleware
app.add_middleware(SessionMiddleware, secret_key=os.getenv('SECRET_KEY', 'your-secret-key-change-in-production'))

# Add CORS middleware for iframe support
@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["X-Frame-Options"] = "ALLOWALL"  # Allow iframe embedding
    return response

# Setup feedback database after RAG agent is initialized
setup_feedback_database()

def create_message_element(content, is_user=False, sources=None, message_id=None, show_feedback=False):
    """Create a modern message element with avatars"""
    if is_user:
        return Div(
            Div(
                Div("👤", cls="user-avatar"),
                Div(
                    Div(content, cls="message-text"),
                    cls="message-content"
                ),
                cls="message"
            ),
            cls="message-wrapper"
        )
    
    # Assistant message content
    message_content_elements = [
        Div(content, cls="message-text", **{"data-markdown": "true"})
    ]
    
    if sources:
        message_content_elements.append(_create_sources_section(sources))
    
    if show_feedback and message_id:
        message_content_elements.append(
            Div(
                Button("👍 Helpful", 
                       hx_post=f"/feedback/{message_id}/positive",
                       hx_target=f"#feedback-{message_id}",
                       cls="positive"),
                Button("👎 Not helpful", 
                       hx_post=f"/feedback/{message_id}/negative",
                       hx_target=f"#feedback-{message_id}",
                       cls="negative"),
                id=f"feedback-{message_id}",
                cls="feedback"
            )
        )
    
    return Div(
        Div(
            Div("🤖", cls="assistant-avatar"),
            Div(*message_content_elements, cls="message-content"),
            cls="message"
        ),
        cls="message-wrapper"
    )

@rt("/")
def home(session):
    """Main chat interface"""
    # Use persistent session ID or create new one
    session_id = session.get('chat_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['chat_session_id'] = session_id
        print(f"🆕 Created new session: {session_id}")
    else:
        print(f"🔄 Using existing session: {session_id}")
    
    if session_id not in conversations:
        conversations[session_id] = []
    
    # Get actual conversation history from database for this session only
    conversation_items = get_session_conversations(session_id, limit=10)
    
    # Load CSS content directly
    css_content = ""
    try:
        with open("styles.css", "r") as f:
            css_content = f.read()
        print(f"✅ Loaded CSS content directly, size: {len(css_content)} characters")
    except Exception as e:
        print(f"❌ Error loading CSS content: {e}")
        css_content = "/* CSS loading failed */"
    
    return Html(
        Head(
            Title("JEA Customer Service Assistant"),
            Meta(charset="utf-8"),
            Meta(name="viewport", content="width=device-width, initial-scale=1"),
            # Load CSS with proper iframe isolation
            Link(rel="stylesheet", href="/styles.css", type="text/css"),
            Style(f"""
                /* Iframe-specific CSS isolation - only reset potentially conflicting styles */
                html, body {{
                    margin: 0 !important;
                    padding: 0 !important;
                    height: 100% !important;
                    overflow: hidden !important;
                }}
                
                /* Ensure our variables work in iframe context */
                :root {{
                    --bg-primary: #ffffff;
                    --bg-secondary: #f7f7f8;
                    --text-primary: #2d333a;
                    --text-secondary: #565869;
                    --border-light: #e5e5e6;
                    --accent-blue: #10a37f;
                    --accent-blue-hover: #0d8f6b;
                }}
            """),
            Script(f"""
                console.log("🎨 CSS Inline Loading Debug:");
                console.log("Session ID:", "{session_id}");
                
                // Check if styles are applied after DOM loads
                document.addEventListener('DOMContentLoaded', function() {{
                    setTimeout(() => {{
                        const body = document.body;
                        const computedStyle = window.getComputedStyle(body);
                        console.log("Body background color:", computedStyle.backgroundColor);
                        console.log("Body font family:", computedStyle.fontFamily);
                        console.log("Body margin:", computedStyle.margin);
                        
                        // Check for specific elements and their styles
                        const appContainer = document.querySelector('.app-container');
                        const chatTitle = document.querySelector('.chat-title');
                        
                        if (appContainer && chatTitle) {{
                            const containerStyle = window.getComputedStyle(appContainer);
                            const titleStyle = window.getComputedStyle(chatTitle);
                            console.log("App container display:", containerStyle.display);
                            console.log("Chat title color:", titleStyle.color);
                            console.log("Chat title font-size:", titleStyle.fontSize);
                            
                            // Check if our CSS variables are working
                            const rootStyles = window.getComputedStyle(document.documentElement);
                            console.log("CSS variable --bg-primary:", rootStyles.getPropertyValue('--bg-primary'));
                            
                            if (containerStyle.display === 'flex' && titleStyle.fontSize) {{
                                console.log("✅ CSS styles applied correctly!");
                            }} else {{
                                console.log("❌ CSS styles not fully applied.");
                            }}
                        }} else {{
                            console.log("❌ Required elements not found (.app-container, .chat-title).");
                        }}
                        
                        // Library checks
                        console.log("Libraries check: HTMX (" + (typeof htmx !== 'undefined') + "), marked.js (" + (typeof marked !== 'undefined') + ")");
                        if (typeof marked !== 'undefined') {{
                            console.log("✅ marked.js is available!");
                        }}
                        
                        if (typeof htmx !== 'undefined') {{
                            console.log("✅ Switching to HTMX form...");
                        }}
                    }}, 500);
                }});
            """)
        ),
        Body(
            Script(src="/htmx.min.js"),
            Script(src="/marked.min.js"),
            Div(
                # Sidebar
                Div(
                    Div(
                        Button(
                            "✨ New Chat",
                            cls="new-chat-btn",
                            onclick="window.location.reload()"
                        ),
                        cls="sidebar-header"
                    ),
                    Div(
                        Div(
                            *([
                                Div(
                                    conv['display_text'],
                                    cls="conversation-item",
                                    onclick=f"loadConversation('{conv['session_id']}', '{conv['message_id']}')",
                                    title=conv['query']  # Show full query on hover
                                ) for conv in conversation_items
                            ] if conversation_items else [
                                Div(
                                    "No conversations yet",
                                    cls="conversation-item empty-state",
                                    style="color: var(--text-secondary); font-style: italic; cursor: default;"
                                )
                            ]),
                            id="conversation-list-items"
                        ),
                        cls="conversation-list"
                    ),
                    cls="sidebar"
                ),
                
                # Main Chat Area
                Div(
                    # Header
                    Div(
                        Div("JEA Customer Service Assistant", cls="chat-title"),
                        Div(
                            Button("🌙", cls="theme-toggle", onclick="toggleTheme()"),
                            A("📊", href="/feedback", target="_blank", cls="header-btn", title="View Feedback"),
                            A("📈", href="/stats", target="_blank", cls="header-btn", title="Statistics"),
                            cls="header-actions"
                        ),
                        cls="chat-header"
                    ),
                    
                    # Messages
                    Div(
                        # Welcome screen with suggested prompts
                        Div(
                            Div(
                                "JEA Customer Service Assistant",
                                cls="welcome-title"
                            ),
                            Div(
                                "I can help you with questions about electric and water services, rates, payment options, outage reporting, and much more. Try asking me something!",
                                cls="welcome-subtitle"
                            ),
                            Div(
                                Div(
                                    Div("💡 Electric Rates", cls="prompt-title"),
                                    Div("What are the current electric rate structures and how can I save money?", cls="prompt-description"),
                                    cls="prompt-card",
                                    onclick=f"askQuestion('What are the current electric rate structures and how can I save money?', '{session_id}')"
                                ),
                                Div(
                                    Div("💧 Water Services", cls="prompt-title"),
                                    Div("How do I set up new water service or report a water leak?", cls="prompt-description"),
                                    cls="prompt-card",
                                    onclick=f"askQuestion('How do I set up new water service or report a water leak?', '{session_id}')"
                                ),
                                Div(
                                    Div("⚡ Outage Reporting", cls="prompt-title"),
                                    Div("How do I report a power outage and track restoration progress?", cls="prompt-description"),
                                    cls="prompt-card",
                                    onclick=f"askQuestion('How do I report a power outage and track restoration progress?', '{session_id}')"
                                ),
                                Div(
                                    Div("💳 Payment Options", cls="prompt-title"),
                                    Div("What payment methods are available and how do I set up autopay?", cls="prompt-description"),
                                    cls="prompt-card",
                                    onclick=f"askQuestion('What payment methods are available and how do I set up autopay?', '{session_id}')"
                                ),
                                cls="suggested-prompts"
                            ),
                            cls="welcome-screen",
                            id="welcome-screen"
                        ),
                        id="messages",
                        cls="messages-container"
                    ),
                    
                    # Input Area
                    Div(
                        Div(
                            # Regular form (fallback)
                            Form(
                                Input(
                                    type="text",
                                    name="question",
                                    placeholder="Ask about JEA rates, services, or policies...",
                                    required=True,
                                    autocomplete="off",
                                    id="question-input"
                                ),
                                Button("Send", type="submit", cls="send-button"),
                                method="post",
                                action=f"/chat/{session_id}",
                                cls="input-form"
                            ),
                            # HTMX form (preferred)
                            Form(
                                Input(
                                    type="text",
                                    name="question",
                                    placeholder="Ask about JEA rates, services, or policies...",
                                    required=True,
                                    autocomplete="off",
                                    id="htmx-question-input"
                                ),
                                Button("Send", type="submit", cls="send-button"),
                                cls="input-form",
                                style="display: none;",
                                id="htmx-form",
                                hx_post=f"/chat/{session_id}",
                                hx_target="#messages",
                                hx_swap="beforeend",
                                **{"hx-on::after-request": "document.getElementById('htmx-question-input').value = ''; document.getElementById('htmx-question-input').focus();"}
                            ),
                            cls="input-wrapper"
                        ),
                        cls="input-container"
                    ),
                    
                    cls="main-chat"
                ),
                
                cls="app-container"
            ),
            Script("""
                // Modern chat interface initialization
                console.log('Session ID:', '""" + session_id + """');
                
                // Theme management
                function toggleTheme() {
                    const body = document.body;
                    const currentTheme = body.getAttribute('data-theme');
                    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
                    body.setAttribute('data-theme', newTheme);
                    localStorage.setItem('theme', newTheme);
                    
                    // Update theme toggle icon
                    const toggle = document.querySelector('.theme-toggle');
                    toggle.textContent = newTheme === 'dark' ? '☀️' : '🌙';
                }
                
                // Initialize theme from localStorage
                function initTheme() {
                    const savedTheme = localStorage.getItem('theme') || 'light';
                    document.body.setAttribute('data-theme', savedTheme);
                    const toggle = document.querySelector('.theme-toggle');
                    if (toggle) {
                        toggle.textContent = savedTheme === 'dark' ? '☀️' : '🌙';
                    }
                }
                
                // Form switching and setup
                function setupForms() {
                    const htmxForm = document.getElementById('htmx-form');
                    const regularForm = document.querySelector('form[action]');
                    
                    if (typeof htmx !== 'undefined' && htmxForm && regularForm) {
                        console.log('✅ Switching to HTMX form...');
                        regularForm.style.display = 'none';
                        htmxForm.style.display = 'flex';
                        htmxForm.querySelector('input').focus();
                    } else {
                        console.log('Using regular form fallback');
                        if (regularForm) {
                            regularForm.querySelector('input').focus();
                        }
                    }
                }
                
                // Markdown rendering
                function setupMarkdown() {
                    if (typeof marked !== 'undefined') {
                        document.querySelectorAll('[data-markdown="true"]:not([data-rendered])').forEach(el => {
                            try {
                                el.innerHTML = marked.parse(el.textContent);
                                el.setAttribute('data-rendered', 'true');
                                
                                // Make all links in markdown content open in new tab
                                el.querySelectorAll('a').forEach(link => {
                                    link.setAttribute('target', '_blank');
                                    link.setAttribute('rel', 'noopener noreferrer');
                                });
                            } catch (e) {
                                console.warn('Markdown parsing error:', e);
                            }
                        });
                    }
                }
                
                // Auto-scroll to bottom
                function scrollToBottom() {
                    const messagesContainer = document.querySelector('.messages-container');
                    if (messagesContainer) {
                        messagesContainer.scrollTop = messagesContainer.scrollHeight;
                    }
                }
                
                // Scroll to show the most recent question-answer pair
                function scrollToShowLatestQA() {
                    console.log('🔍 scrollToShowLatestQA called');
                    const messagesContainer = document.querySelector('.messages-container');
                    if (!messagesContainer) {
                        console.log('❌ No messages container found');
                        return;
                    }
                    
                    const messageWrappers = messagesContainer.querySelectorAll('.message-wrapper');
                    console.log(`📝 Found ${messageWrappers.length} message wrappers`);
                    
                    if (messageWrappers.length >= 2) {
                        // Look for the last user message by searching backwards
                        let userMessageIndex = -1;
                        for (let i = messageWrappers.length - 1; i >= 0; i--) {
                            const isUserMessage = messageWrappers[i].querySelector('.user-avatar');
                            if (isUserMessage) {
                                userMessageIndex = i;
                                break;
                            }
                        }
                        
                        console.log(`👤 Last user message found at index: ${userMessageIndex}`);
                        
                        if (userMessageIndex >= 0) {
                            const userMessage = messageWrappers[userMessageIndex];
                            console.log('✅ Scrolling to user message');
                            
                            // Scroll to show the question at the top with some padding
                            userMessage.scrollIntoView({ 
                                behavior: 'smooth', 
                                block: 'start',
                                inline: 'nearest'
                            });
                            return;
                        }
                    }
                    
                    console.log('⚠️ Fallback to scrollToBottom');
                    // Fallback to normal scroll behavior
                    scrollToBottom();
                }
                
                // Handle suggested prompt clicks
                function askQuestion(question, sessionId) {
                    const activeInput = document.querySelector('.input-form:not([style*="display: none"]) input');
                    if (activeInput) {
                        activeInput.value = question;
                        
                        // Hide welcome screen
                        const welcomeScreen = document.getElementById('welcome-screen');
                        if (welcomeScreen) {
                            welcomeScreen.style.display = 'none';
                        }
                        
                        // Trigger form submission
                        const form = activeInput.closest('form');
                        if (form) {
                            // Create a submit event
                            const event = new Event('submit', { bubbles: true, cancelable: true });
                            form.dispatchEvent(event);
                        }
                    }
                }
                
                // Load and display a previous conversation
                function loadConversation(sessionId, messageId) {
                    console.log('Loading conversation:', sessionId, messageId);
                    
                    // Hide welcome screen if showing
                    const welcomeScreen = document.getElementById('welcome-screen');
                    if (welcomeScreen) {
                        welcomeScreen.style.display = 'none';
                    }
                    
                    // Clear current messages and show the conversation
                    const messagesContainer = document.getElementById('messages');
                    
                    // Load the conversation via HTMX
                    if (typeof htmx !== 'undefined') {
                        htmx.ajax('GET', `/load-conversation/${sessionId}/${messageId}`, {
                            target: '#messages',
                            swap: 'innerHTML'
                        });
                    } else {
                        // Fallback for non-HTMX
                        fetch(`/load-conversation/${sessionId}/${messageId}`)
                            .then(response => response.text())
                            .then(html => {
                                messagesContainer.innerHTML = html;
                                scrollToShowLatestQA();
                                setupMarkdown();
                            })
                            .catch(error => {
                                console.error('Error loading conversation:', error);
                            });
                    }
                    
                    // Update sidebar to show active conversation
                    document.querySelectorAll('.conversation-item').forEach(item => {
                        item.classList.remove('active');
                    });
                    event.target.classList.add('active');
                }
                
                // Initialize everything
                function initialize() {
                    initTheme();
                    setupForms();
                    setupMarkdown();
                    // Don't auto-scroll on init - let messages stay where they are
                }
                
                // Update sidebar with new conversations
                function updateSidebar(sessionId) {
                    if (typeof htmx !== 'undefined') {
                        htmx.ajax('GET', `/session-conversations/${sessionId}`, {
                            target: '#conversation-list-items',
                            swap: 'innerHTML'
                        });
                    }
                }
                
                // Event listeners
                document.addEventListener('DOMContentLoaded', initialize);
                document.body.addEventListener('htmx:afterSwap', (event) => {
                    setTimeout(() => {
                        setupMarkdown();
                        
                        // Use different scroll behavior based on what was updated
                        const targetId = event.detail.target.id;
                        console.log('🔄 HTMX afterSwap for element:', targetId);
                        
                        if (targetId === 'messages' || targetId.startsWith('response-')) {
                            console.log('📱 Chat-related swap detected - calling scrollToShowLatestQA');
                            // For chat responses, scroll to show the question
                            scrollToShowLatestQA();
                            
                            // Update sidebar for any chat-related change
                            const sessionId = '""" + session_id + """';
                            updateSidebar(sessionId);
                        } else if (targetId === 'conversation-list-items') {
                            console.log('📋 Sidebar update - no scrolling needed');
                            // Don't scroll for sidebar updates
                        } else {
                            console.log('🔄 Other swap - using scrollToBottom');
                            // For other updates, use normal scroll behavior
                            scrollToBottom();
                        }
                    }, 200);
                });
                
                // Keyboard shortcuts
                document.addEventListener('keydown', (e) => {
                    if (e.key === 'Escape') {
                        // Clear input on Escape
                        const activeInput = document.querySelector('.input-form:not([style*="display: none"]) input');
                        if (activeInput) activeInput.value = '';
                    }
                });
                
                // Make functions globally accessible
                window.toggleTheme = toggleTheme;
                window.askQuestion = askQuestion;
                window.loadConversation = loadConversation;
                
                // Initialize immediately if DOM is already loaded
                if (document.readyState !== 'loading') {
                    initialize();
                }
            """)
        )
    )

@rt("/chat/{session_id}", methods=["POST"])
def chat(session_id: str, question: str, request, session):
    """Handle chat message"""
    print(f"💬 Chat endpoint called: session={session_id}, question='{question}'")
    
    # Validate that session_id matches the session's stored session_id
    stored_session_id = session.get('chat_session_id')
    print(f"🔧 URL session_id: {session_id}")
    print(f"🔧 Stored session_id: {stored_session_id}")
    
    if stored_session_id and session_id != stored_session_id:
        print(f"⚠️ Session ID mismatch! Using stored session ID: {stored_session_id}")
        session_id = stored_session_id
    
    # Check if this is an HTMX request
    is_htmx = request.headers.get('hx-request') == 'true'
    print(f"🔧 Is HTMX request: {is_htmx}")
    
    if session_id not in conversations:
        conversations[session_id] = []
    
    message_id = str(uuid.uuid4())
    
    # Add user message to conversation
    conversations[session_id].append({
        'type': 'user',
        'content': question,
        'timestamp': datetime.now()
    })
    
    # Store pending response info
    pending_responses[message_id] = {
        'session_id': session_id,
        'question': question,
        'response_text': '',
        'sources': []
    }
    
    # If this is a regular form submission (not HTMX), redirect back with a simple response
    if not is_htmx:
        print("🔧 Regular form submission - redirecting to simple response page")
        return RedirectResponse(url=f"/simple-chat/{session_id}/{message_id}")
    
    # Return user message and trigger bot response using HTMX
    return (
        create_message_element(question, is_user=True),
        Div(
            Div(
                Div("🤖", cls="assistant-avatar"),
                Div(
                    Div(
                        "JEA Assistant is thinking",
                        Div(
                            Span(), Span(), Span(),
                            cls="typing-dots"
                        ),
                        cls="thinking-indicator"
                    ),
                    cls="message-content"
                ),
                cls="message thinking"
            ),
            id=f"response-{message_id}",
            cls="message-wrapper",
            hx_get=f"/stream/{message_id}",
            hx_trigger="load delay:500ms",
            hx_swap="outerHTML"
        )
    )

@rt("/simple-chat/{session_id}/{message_id}")
def simple_chat_response(session_id: str, message_id: str):
    """Handle simple chat response for non-HTMX requests"""
    print(f"🔧 Simple chat response for session {session_id}, message {message_id}")
    
    if message_id not in pending_responses:
        return f"Error: Message not found. <a href='/'>← Back to chat</a>"
    
    pending = pending_responses[message_id]
    question = pending['question']
    
    try:
        # Process the question same as HTMX version
        rag_agent.reset_state()
        response_parts = []
        
        for chunk in rag_agent.query_response(question, top_k=5, min_similarity=0.3):
            if chunk:
                response_parts.append(str(chunk))
        
        full_response = ''.join(response_parts) if response_parts else "Sorry, I couldn't generate a response."
        try:
            sources = rag_agent.get_last_sources()
            confidence = rag_agent.get_last_confidence()
            print(f"🔗 Simple chat - Found {len(sources)} sources")
        except Exception as e:
            print(f"⚠️ Simple chat - Error getting sources: {e}")
            sources = []
            confidence = 0.0
        
        # Store interaction
        store_interaction(
            message_id=message_id,
            session_id=session_id,
            query=question,
            response_text=full_response,
            sources=sources,
            confidence=confidence,
            cached_from_query_cache=getattr(rag_agent, '_is_cached_response', False)
        )
        
        # Create sources display for simple chat using FastHTML components
        sources_section = _create_sources_section(sources)

        return Html(
            Head(
                Title("JEA Chat Response"),
                Script(src="/marked.min.js"),
                Style("""
                    .markdown-content h1, .markdown-content h2, .markdown-content h3 { margin: 1rem 0 0.5rem 0; }
                    .markdown-content ul, .markdown-content ol { margin: 0.5rem 0; padding-left: 1.5rem; }
                    .markdown-content p { margin: 0.5rem 0; line-height: 1.5; }
                    .markdown-content code { background: #f3f4f6; padding: 0.2rem 0.4rem; border-radius: 3px; font-family: monospace; }
                    .markdown-content pre { background: #f3f4f6; padding: 1rem; border-radius: 5px; overflow-x: auto; margin: 0.5rem 0; }
                    .markdown-content strong { font-weight: 600; }
                    
                    /* Source card styles for simple page */
                    .sources-grid {
                        display: flex;
                        gap: 0.75rem;
                        overflow-x: auto;
                        padding: 0.5rem 0;
                        scrollbar-width: thin;
                        scrollbar-color: #cbd5e1 #f8fafc;
                    }
                    .sources-grid::-webkit-scrollbar { height: 6px; }
                    .sources-grid::-webkit-scrollbar-track { background: #f8fafc; border-radius: 3px; }
                    .sources-grid::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
                    .source-card {
                        flex: 0 0 auto;
                        width: 120px;
                        height: 80px;
                        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
                        border: 1px solid #e2e8f0;
                        border-radius: 8px;
                        padding: 0.75rem;
                        text-decoration: none;
                        color: inherit;
                        transition: all 0.2s ease;
                        cursor: pointer;
                        position: relative;
                        display: flex;
                        flex-direction: column;
                        justify-content: space-between;
                        overflow: hidden;
                    }
                    .source-card:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 4px 12px rgba(30, 64, 175, 0.15);
                        border-color: #1e40af;
                        background: linear-gradient(135deg, #eff6ff, #dbeafe);
                    }
                    .source-favicon { width: 16px; height: 16px; border-radius: 3px; margin-bottom: 0.5rem; flex-shrink: 0; }
                    .source-title { font-size: 0.75rem; font-weight: 500; line-height: 1.2; color: #374151; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; margin-bottom: 0.25rem; }
                    .source-domain { font-size: 0.65rem; color: #6b7280; font-weight: 400; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
                    .source-number { position: absolute; top: 0.25rem; right: 0.25rem; background: #1e40af; color: white; font-size: 0.6rem; width: 16px; height: 16px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; }
                    .source-tooltip { position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%); background: #1f2937; color: white; padding: 0.5rem 0.75rem; border-radius: 6px; font-size: 0.75rem; white-space: nowrap; opacity: 0; visibility: hidden; transition: all 0.2s ease; z-index: 10; max-width: 300px; word-break: break-all; margin-bottom: 0.5rem; }
                    .source-tooltip::after { content: ''; position: absolute; top: 100%; left: 50%; transform: translateX(-50%); border: 5px solid transparent; border-top-color: #1f2937; }
                    .source-card:hover .source-tooltip { opacity: 1; visibility: visible; }
                """)
            ),
            Body(
                H1("JEA Customer Service"),
                Div(f"You asked: {question}", style="padding: 1rem; background: #f0f4ff; border-radius: 5px; margin: 1rem 0;"),
                Div(
                    full_response,
                    cls="markdown-content",
                    style="padding: 1rem; background: #f9f9f9; border-radius: 5px; margin: 1rem 0;",
                    **{"data-markdown": "true"}
                ),
                sources_section if sources_section else "",
                A("← Ask another question", href="/", style="display: inline-block; padding: 0.5rem 1rem; background: #1e40af; color: white; text-decoration: none; border-radius: 5px;"),
                style="max-width: 800px; margin: 2rem auto; padding: 2rem; font-family: sans-serif;",
                onload="if(typeof marked !== 'undefined') { document.querySelectorAll('[data-markdown]').forEach(el => { el.innerHTML = marked.parse(el.textContent); el.querySelectorAll('a').forEach(link => { link.setAttribute('target', '_blank'); link.setAttribute('rel', 'noopener noreferrer'); }); }); }"
            )
        )
        
    except Exception as e:
        return f"Error processing question: {e}. <a href='/'>← Back to chat</a>"

@rt("/stream/{message_id}")
def stream_response(message_id: str):
    """Generate streaming response"""
    print(f"🔍 Stream request received for message: {message_id}")
    
    if message_id not in pending_responses:
        print(f"❌ Message ID not found: {message_id}")
        return Div(
            "❌ Error: Message not found",
            cls="message bot-message",
            style="color: #ef4444;"
        )
    
    pending = pending_responses[message_id]
    question = pending['question']
    print(f"📝 Processing question: {question}")
    
    try:
        # Reset agent state for fresh query
        rag_agent.reset_state()
        print("🔄 RAG agent reset")
        
        # Test basic response first
        print("🤖 Starting RAG query...")
        
        # Collect the streaming response
        response_parts = []
        chunk_count = 0
        
        try:
            for chunk in rag_agent.query_response(question, top_k=5, min_similarity=0.3):
                if chunk:
                    response_parts.append(str(chunk))
                    chunk_count += 1
            
            print(f"✅ Received {chunk_count} chunks from RAG agent")
            
        except Exception as e:
            print(f"❌ Error during RAG streaming: {e}")
            # Fallback to simple response
            response_parts = [f"I'm having trouble processing your question right now. Please try asking: '{question}' again."]
        
        # Get complete response and sources
        full_response = ''.join(response_parts) if response_parts else "Sorry, I couldn't generate a response."
        print(f"📄 Full response length: {len(full_response)}")
        
        try:
            sources = rag_agent.get_last_sources()
            print(f"🔗 Found {len(sources)} sources")
            for i, source in enumerate(sources, 1):
                print(f"  Source {i}: {source.title[:50]}... - {source.url}")
        except Exception as e:
            print(f"⚠️ Error getting sources: {e}")
            sources = []
        
        # Calculate confidence
        try:
            confidence = rag_agent.get_last_confidence()
        except:
            confidence = 0.0
        
        # Store response data
        pending['response_text'] = full_response
        pending['sources'] = sources
        
        # Store interaction in rag_interactions table (regardless of feedback)
        store_interaction(
            message_id=message_id,
            session_id=pending['session_id'],
            query=question,
            response_text=full_response,
            sources=sources,
            security_level="external",
            confidence=confidence,
            cached_from_query_cache=False,  # This was a fresh query
            user_context=None
        )
        
        # Add to conversation history
        conversations[pending['session_id']].append({
            'type': 'bot',
            'content': full_response,
            'sources': sources,
            'timestamp': datetime.now(),
            'message_id': message_id
        })
        
        # Create feedback section using FastHTML components
        feedback_section = Div(
            Button("👍 Helpful", 
                   hx_post=f"/feedback/{message_id}/positive",
                   hx_target=f"#feedback-{message_id}",
                   cls="positive"),
            Button("👎 Not helpful", 
                   hx_post=f"/feedback/{message_id}/negative",
                   hx_target=f"#feedback-{message_id}",
                   cls="negative"),
            id=f"feedback-{message_id}",
            cls="feedback"
        )
        
        # Use the existing create_message_element function for consistency
        print(f"✅ Returning response with {len(sources)} sources")
        return create_message_element(
            content=full_response,
            is_user=False,
            sources=sources,
            message_id=message_id,
            show_feedback=True
        )
        
    except Exception as e:
        print(f"❌ Error in stream_response: {e}")
        error_message = f"Sorry, I encountered an error: {str(e)}\n\nPlease try asking your question again or contact JEA customer service at (904) 665-6000."
        return create_message_element(
            content=error_message,
            is_user=False,
            sources=None,
            message_id=message_id,
            show_feedback=False
        )

@rt("/feedback/{message_id}/{feedback_type}")
def handle_feedback(message_id: str, feedback_type: str):
    """Handle user feedback on responses"""
    try:
        # Get the conversation data for this message
        feedback_entry = None
        for session_id, messages in conversations.items():
            for msg in messages:
                if msg.get('message_id') == message_id:
                    feedback_entry = msg
                    break
            if feedback_entry:
                break
        
        if not feedback_entry:
            print(f"❌ Feedback: No conversation found for message {message_id}")
            return Div(
                Span("⚠️ Message not found", cls="feedback-result error"),
                style="color: #dc2626; margin: 0.5rem;"
            )
        
        is_helpful = feedback_type == "positive"
        query = feedback_entry.get('query', '')
        response = feedback_entry.get('response', '')
        sources_count = len(feedback_entry.get('sources', []))
        confidence = feedback_entry.get('confidence', 0.0)
        
        print(f"📝 Feedback received for message {message_id}: {'👍 Positive' if is_helpful else '👎 Negative'}")
        print(f"   Query: {query[:100]}...")
        print(f"   Response length: {len(response)} chars")
        print(f"   Sources: {sources_count}, Confidence: {confidence}")
        
        # Store in database
        success = store_feedback(
            message_id=message_id,
            query=query,
            response=response, 
            is_helpful=is_helpful,
            sources_count=sources_count,
            confidence=confidence,
            session_id=feedback_entry.get('session_id')
        )
        
        if success:
            # Remove from pending feedback
            if message_id in pending_responses:
                del pending_responses[message_id]
            
            # Mark as feedback given in conversation
            feedback_entry['feedback_given'] = True
            feedback_entry['feedback_type'] = feedback_type
            
            result_message = "✅ Thank you for your feedback!" if is_helpful else "✅ Feedback noted - we'll work to improve!"
            print(f"✅ Feedback stored successfully for message {message_id}")
            
            return Div(
                Span(result_message, cls="feedback-result success"),
                style="color: #059669; margin: 0.5rem; font-weight: 500;"
            )
        else:
            print(f"❌ Failed to store feedback for message {message_id}")
            return Div(
                Span("❌ Error saving feedback", cls="feedback-result error"),
                style="color: #dc2626; margin: 0.5rem;"
            )
            
    except Exception as e:
        print(f"❌ Error in feedback handler: {e}")
        return Div(
            Span("❌ Error processing feedback", cls="feedback-result error"),
            style="color: #dc2626; margin: 0.5rem;"
        )

@rt("/htmx.min.js")
def serve_htmx():
    """Serve local HTMX file"""
    print("🔧 HTMX file requested from main page!")
    try:
        with open("htmx.min.js", "r") as f:
            content = f.read()
        print(f"✅ HTMX file served successfully, size: {len(content)} characters")
        headers = {
            "Cache-Control": "public, max-age=86400",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "Content-Type"
        }
        return Response(content, media_type="application/javascript", headers=headers)
    except Exception as e:
        print(f"❌ Error loading HTMX file: {e}")
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "Content-Type"
        }
        return Response(f"// Error loading HTMX: {e}", media_type="application/javascript", headers=headers)

@rt("/marked.min.js")
def serve_marked():
    """Serve local marked.js file"""
    print("📝 Marked.js file requested!")
    try:
        with open("marked.min.js", "r") as f:
            content = f.read()
        print(f"✅ Marked.js file served successfully, size: {len(content)} characters")
        headers = {
            "Cache-Control": "public, max-age=86400",
            "Access-Control-Allow-Origin": "*"
        }
        return Response(content, media_type="application/javascript", headers=headers)
    except Exception as e:
        print(f"❌ Error loading marked.js file: {e}")
        headers = {
            "Access-Control-Allow-Origin": "*"
        }
        return Response(f"// Error loading marked.js: {e}", media_type="application/javascript", headers=headers)

@rt("/styles.css")
def serve_css(request):
    """Serve CSS with proper headers"""
    print("🎨 CSS file requested")
    try:
        with open("styles.css", "r") as f:
            content = f.read()
        print(f"✅ CSS file served successfully, size: {len(content)} characters")
        headers = {
            "Cache-Control": "public, max-age=86400",
            "Access-Control-Allow-Origin": "*"
        }
        return Response(content, media_type="text/css", headers=headers)
    except Exception as e:
        print(f"❌ Error loading CSS file: {e}")
        return Response(f"/* Error loading CSS: {e} */", media_type="text/css")

@rt("/health")
def health_check():
    """Health check endpoint"""
    try:
        model_info = rag_agent.security_router.get_model_info()
        cache_stats = rag_agent.get_cache_stats()
        
        return {
            "status": "healthy",
            "model_info": model_info,
            "cache_stats": cache_stats,
            "active_conversations": len(conversations),
            "pending_responses": len(pending_responses)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@rt("/stats")
def stats():
    """Show system statistics"""
    try:
        model_info = rag_agent.security_router.get_model_info()
        cache_stats = rag_agent.get_cache_stats()
        
        return Html(
            Head(Title("JEA Chatbot - Statistics")),
            Body(
                H1("JEA Chatbot Statistics"),
                H2("Model Information"),
                Pre(json.dumps(model_info, indent=2)),
                H2("Cache Statistics"),
                Pre(json.dumps(cache_stats, indent=2)),
                P(f"Active Conversations: {len(conversations)}"),
                P(f"Pending Responses: {len(pending_responses)}"),
                A("← Back to Chat", href="/")
            )
        )
    except Exception as e:
        return f"Error: {str(e)}"

@rt("/feedback-test")
def feedback_test():
    """Test endpoint to view feedback and interaction data"""
    try:
        app_db_path = rag_agent.app_db_path
        conn = sqlite3.connect(app_db_path)
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_feedback'")
        feedback_table_exists = cursor.fetchone() is not None
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rag_interactions'")
        interactions_table_exists = cursor.fetchone() is not None
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='query_cache'")
        cache_table_exists = cursor.fetchone() is not None
        
        # Get feedback data
        feedback_data = []
        if feedback_table_exists:
            cursor.execute('''
                SELECT id, message_id, query, is_helpful, timestamp, sources_count, confidence, session_id, interaction_id
                FROM user_feedback 
                ORDER BY timestamp DESC 
                LIMIT 5
            ''')
            feedback_data = cursor.fetchall()
        
        # Get interaction data
        interaction_data = []
        if interactions_table_exists:
            cursor.execute('''
                SELECT id, message_id, query, confidence_score, timestamp, num_sources_retrieved, cached_from_query_cache
                FROM rag_interactions 
                ORDER BY timestamp DESC 
                LIMIT 10
            ''')
            interaction_data = cursor.fetchall()
        
        # Get cache stats
        cache_count = 0
        if cache_table_exists:
            cursor.execute('SELECT COUNT(*) FROM query_cache')
            cache_count = cursor.fetchone()[0]
        
        conn.close()
        
        return Html(
            Head(Title("Database Test")),
            Body(
                H1("Database Structure Test"),
                P(f"App database path: {app_db_path}"),
                P(f"Knowledge database path: {rag_agent.knowledge_db_path}"),
                
                H2("Table Status:"),
                P(f"✅ user_feedback table exists: {feedback_table_exists}"),
                P(f"✅ rag_interactions table exists: {interactions_table_exists}"),
                P(f"✅ query_cache table exists: {cache_table_exists}"),
                P(f"📊 Query cache entries: {cache_count}"),
                
                H2("Recent Interactions (All questions):"),
                P(f"Total interactions: {len(interaction_data)}"),
                *([
                    *[Div(
                        f"ID: {row[0]}, Message: {row[1][:15] if row[1] else 'None'}..., Query: '{row[2][:40]}...', Confidence: {row[3]:.2f if row[3] else 0}, Cached: {row[6]}, Time: {row[4]}"
                    ) for row in interaction_data]
                ] if interaction_data else [P("No interaction data found.")]),
                
                H2("Recent Feedback (Only rated questions):"),
                P(f"Total feedback entries: {len(feedback_data)}"),
                *([
                    *[Div(
                        f"ID: {row[0]}, Message: {row[1][:15] if row[1] else 'None'}..., Query: '{row[2][:40]}...', Helpful: {'✅' if row[3] else '❌'}, Sources: {row[5]}, Interaction_ID: {row[8]}, Time: {row[4]}"
                    ) for row in feedback_data]
                ] if feedback_data else [P("No feedback data found.")]),
                
                Hr(),
                P("🔄 Logic: ALL questions go to rag_interactions. Only questions with feedback go to user_feedback. Only POSITIVE feedback gets cached."),
                A("← Back to chat", href="/")
            )
        )
        
    except Exception as e:
        return f"Error: {e}"

@rt("/feedback")
def feedback_page():
    """Display a page with questions, answers, and user feedback."""
    try:
        app_db_path = rag_agent.app_db_path
        conn = sqlite3.connect(app_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                ri.query,
                ri.response_text,
                uf.is_helpful
            FROM
                rag_interactions ri
            LEFT JOIN
                user_feedback uf ON ri.id = uf.interaction_id
            ORDER BY
                ri.timestamp DESC
            LIMIT 50
        """)
        
        interactions = cursor.fetchall()
        conn.close()

        def get_feedback_status(is_helpful):
            if is_helpful is None:
                return Span("Not Rated", style="color: #6b7280; font-style: italic;")
            elif is_helpful:
                return Span("👍 Helpful", style="color: #10b981; font-weight: bold;")
            else:
                return Span("👎 Not Helpful", style="color: #ef4444; font-weight: bold;")

        cards = []
        for query, response, is_helpful in interactions:
            cards.append(
                Card(
                    H4(query),
                    Div(
                        response, 
                        **{"data-markdown": "true"},
                        style="background-color: #f9f9f9; padding: 1rem; border-radius: 5px; margin-bottom: 1rem; max-height: 200px; overflow-y: auto;"
                    ),
                    footer=Div(
                        "Feedback: ",
                        get_feedback_status(is_helpful),
                        style="text-align: right; font-size: 0.9rem;"
                    ),
                    style="margin-bottom: 1.5rem;"
                )
            )
        
        return Titled(
            "User Feedback",
            *cards,
            Script(src="/marked.min.js"),
            Script("""
                document.addEventListener('DOMContentLoaded', () => {
                    if (typeof marked !== 'undefined') {
                        document.querySelectorAll('[data-markdown="true"]').forEach(el => {
                            el.innerHTML = marked.parse(el.textContent);
                        });
                    }
                });
            """)
        )

    except Exception as e:
        return Titled("Error", P(f"An error occurred: {e}"))

def get_session_conversations(session_id, limit=10):
    """Get conversations for the current session only"""
    try:
        app_db_path = rag_agent.app_db_path
        conn = sqlite3.connect(app_db_path)
        cursor = conn.cursor()
        
        # Get conversations for this session only
        cursor.execute('''
            SELECT 
                session_id,
                query,
                timestamp,
                message_id
            FROM rag_interactions 
            WHERE session_id = ? AND query IS NOT NULL 
            ORDER BY timestamp ASC
            LIMIT ?
        ''', (session_id, limit))
        
        conversations_data = cursor.fetchall()
        conn.close()
        
        # Format conversations for sidebar display
        conversation_items = []
        for session_id, query, timestamp, message_id in conversations_data:
            # Truncate long queries for display
            display_text = query[:50] + "..." if len(query) > 50 else query
            conversation_items.append({
                'session_id': session_id,
                'query': query,
                'display_text': display_text,
                'timestamp': timestamp,
                'message_id': message_id
            })
        
        return conversation_items
        
    except Exception as e:
        print(f"Error getting session conversations: {e}")
        return []

@rt("/conversation/{session_id}/{message_id}")
def get_conversation(session_id: str, message_id: str):
    """Get conversation data for loading previous chats"""
    try:
        app_db_path = rag_agent.app_db_path
        conn = sqlite3.connect(app_db_path)
        cursor = conn.cursor()
        
        # Get the conversation data
        cursor.execute('''
            SELECT query, response_text, timestamp, sources_json
            FROM rag_interactions 
            WHERE session_id = ? AND message_id = ?
        ''', (session_id, message_id))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            query, response_text, timestamp, sources_json = result
            return {
                "session_id": session_id,
                "message_id": message_id,
                "query": query,
                "response": response_text,
                "timestamp": timestamp,
                "sources": json.loads(sources_json) if sources_json else []
            }
        else:
            return {"error": "Conversation not found"}
            
    except Exception as e:
        print(f"Error getting conversation: {e}")
        return {"error": str(e)}

@rt("/load-conversation/{session_id}/{message_id}")
def load_conversation_display(session_id: str, message_id: str):
    """Load and display a conversation (question + answer) in the chat"""
    try:
        app_db_path = rag_agent.app_db_path
        conn = sqlite3.connect(app_db_path)
        cursor = conn.cursor()
        
        # Get the conversation data
        cursor.execute('''
            SELECT query, response_text, timestamp, sources_json
            FROM rag_interactions 
            WHERE session_id = ? AND message_id = ?
        ''', (session_id, message_id))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            query, response_text, timestamp, sources_json = result
            
            # Parse sources
            sources = []
            if sources_json:
                try:
                    sources_data = json.loads(sources_json)
                    from rag_agent import RetrievedDocument
                    sources = [RetrievedDocument(
                        document_id=0,
                        url=s['url'],
                        title=s['title'],
                        chunk_text="",
                        similarity_score=s.get('similarity_score', 0.0),
                        source_type="",
                        chunk_index=0
                    ) for s in sources_data]
                except:
                    sources = []
            
            # Create message elements
            user_message = create_message_element(query, is_user=True)
            bot_message = create_message_element(
                response_text, 
                is_user=False, 
                sources=sources, 
                message_id=message_id, 
                show_feedback=True
            )
            
            return Div(user_message, bot_message)
        else:
            return Div(
                Div(
                    "Conversation not found",
                    cls="message-wrapper"
                )
            )
            
    except Exception as e:
        print(f"Error loading conversation display: {e}")
        return Div(
            Div(
                f"Error loading conversation: {str(e)}",
                cls="message-wrapper"
            )
        )

@rt("/session-conversations/{session_id}")
def get_session_conversations_api(session_id: str):
    """Get updated sidebar conversation list for current session"""
    try:
        conversation_items = get_session_conversations(session_id, limit=20)
        
        if conversation_items:
            sidebar_items = []
            for conv in conversation_items:
                sidebar_items.append(
                    Div(
                        conv['display_text'],
                        cls="conversation-item",
                        onclick=f"loadConversation('{conv['session_id']}', '{conv['message_id']}')",
                        title=conv['query']
                    )
                )
            return Div(*sidebar_items, id="conversation-list-items")
        else:
            return Div(
                Div(
                    "No conversations yet",
                    cls="conversation-item empty-state",
                    style="color: var(--text-secondary); font-style: italic; cursor: default;"
                ),
                id="conversation-list-items"
            )
            
    except Exception as e:
        print(f"Error getting session conversations API: {e}")
        return Div(
            Div("Error loading conversations", cls="conversation-item empty-state"),
            id="conversation-list-items"
        )

if __name__ == "__main__":
    import uvicorn
    import argparse
    import os
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='JEA Customer Service Chatbot')
    parser.add_argument('--port', type=int, help='Port to run the server on')
    args = parser.parse_args()
    
    # Determine port from arguments, environment variable, or default
    port = args.port or int(os.environ.get('PORT', 8050))
    
    print("🚀 Starting JEA Chatbot Server...")
    print("🤖 RAG Agent initialized")
    print(f"💬 Chat interface available at: http://localhost:{port}")
    print(f"📊 Statistics available at: http://localhost:{port}/stats")
    print(f"🏥 Health check available at: http://localhost:{port}/health")
    uvicorn.run(app, host="0.0.0.0", port=port) 