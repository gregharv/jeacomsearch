import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
import logging
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dataclasses import dataclass
import re
import os
from urllib.parse import urlparse
from dotenv import load_dotenv
import ssl
import urllib3
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import threading
import time



# Add OpenAI import
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI library not installed. Install with: pip install openai")

@dataclass
class CachedQuery:
    """Represents a cached query with its response"""
    query_id: int
    original_query: str
    query_embedding: np.ndarray
    response_answer: str
    response_sources: str  # JSON string (not parsed list)
    confidence: float
    security_level: str
    timestamp: datetime
    access_count: int
    similarity_threshold: float = 0.85  # Default threshold for cache hits

@dataclass
class RetrievedDocument:
    """Represents a retrieved document with metadata"""
    document_id: int
    url: str
    title: str
    chunk_text: str
    similarity_score: float
    source_type: str
    chunk_index: int

@dataclass
class RAGResponse:
    """Represents a complete RAG response"""
    answer: str
    sources: List[RetrievedDocument]
    reasoning: str
    confidence: float
    security_level: str
    is_cached: bool = False  # New field to indicate if response came from cache

class SecurityLevelRouter:
    """Routes queries to appropriate LLMs based on security level"""
    
    def __init__(self, preferred_model: str = "gemini"):
        """
        Initialize SecurityLevelRouter with model preference
        
        Args:
            preferred_model: "gemini" for Gemini Flash 2.0, "openai" for GPT-4o-mini
        """
        # Load environment variables from .env file
        load_dotenv()
        self.preferred_model = preferred_model.lower()
        self.setup_models()
    
    def setup_models(self):
        """Initialize LLM models for different security levels"""
        self.external_model = None
        self.openai_client = None
        
        # Setup Gemini Flash 2.0
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if gemini_api_key:
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            genai.configure(api_key=gemini_api_key)
        else:
            print("⚠️  GEMINI_API_KEY not found. Gemini queries will not work.")
            self.gemini_model = None
        
        # Setup OpenAI GPT-4o-mini
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if openai_api_key and OPENAI_AVAILABLE:
            try:
                import httpx
                
                # Use standard SSL verification
                http_client = httpx.Client(verify=True)
                self.openai_client = openai.OpenAI(api_key=openai_api_key, http_client=http_client)
                print("✅ OpenAI GPT-4o-mini model configured successfully")

            except Exception as e:
                print(f"⚠️  Failed to configure OpenAI model: {e}")
                self.openai_client = None
        else:
            if not openai_api_key:
                print("⚠️  OPENAI_API_KEY not found. OpenAI queries will not work.")
            if not OPENAI_AVAILABLE:
                print("⚠️  OpenAI library not available. Install with: pip install openai")
            self.openai_client = None
        
        # Set external model based on preference and availability
        if self.preferred_model == "openai" and self.openai_client:
            self.external_model = "openai"
            print(f"🤖 Using OpenAI GPT-4o-mini as primary external model")
        elif self.preferred_model == "gemini" and self.gemini_model:
            self.external_model = "gemini"
            print(f"🤖 Using Gemini Flash 2.0 as primary external model")
        else:
            # Fallback logic
            if self.gemini_model:
                self.external_model = "gemini"
                print(f"🤖 Fallback: Using Gemini Flash 2.0 as external model")
            elif self.openai_client:
                self.external_model = "openai"
                print(f"🤖 Fallback: Using OpenAI GPT-4o-mini as external model")
            else:
                self.external_model = None
                print("❌ No external models available!")
        
        # Internal level - placeholder for future LLM
        self.internal_model = None  # TBD
        
        # Sensitive level - placeholder for self-hosted Llama
        self.sensitive_model = None  # Self-hosted Llama (not implemented)
    
    def get_model(self, security_level: str):
        """Get appropriate model for security level"""
        if security_level == "external":
            return self.external_model
        elif security_level == "internal":
            return self.internal_model
        elif security_level == "sensitive":
            return self.sensitive_model
        else:
            return self.sensitive_model  # Default fallback
    
    def generate_content(self, prompt: str, stream: bool = False):
        """
        Generate content using the configured external model
        
        Args:
            prompt: The input prompt
            stream: Whether to stream the response
            
        Returns:
            Response object with text attribute (for compatibility)
        """
        if self.external_model == "openai" and self.openai_client:
            return self._generate_openai_content(prompt, stream)
        elif self.external_model == "gemini" and self.gemini_model:
            return self._generate_gemini_content(prompt, stream)
        else:
            raise Exception("No external model available for content generation")
    
    def _generate_openai_content(self, prompt: str, stream: bool = False):
        """Generate content using OpenAI GPT-4o-mini"""
        try:
            if stream:
                return self._openai_stream_response(prompt)
            else:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.7
                )
                
                # Create a response object with text attribute for compatibility
                class OpenAIResponse:
                    def __init__(self, text):
                        self.text = text
                
                return OpenAIResponse(response.choices[0].message.content)
                
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")
    
    def _generate_gemini_content(self, prompt: str, stream: bool = False):
        """Generate content using Gemini Flash 2.0"""
        try:
            return self.gemini_model.generate_content(prompt, stream=stream)
        except Exception as e:
            raise Exception(f"Gemini API error: {e}")
    
    def _openai_stream_response(self, prompt: str):
        """Handle OpenAI streaming response"""
        try:
            stream = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7,
                stream=True
            )
            
            # Create a generator that yields text chunks
            class OpenAIStreamResponse:
                def __init__(self, stream):
                    self.stream = stream
                
                def __iter__(self):
                    for chunk in self.stream:
                        if chunk.choices[0].delta.content:
                            # Create chunk object with text attribute for compatibility
                            class Chunk:
                                def __init__(self, text):
                                    self.text = text
                            yield Chunk(chunk.choices[0].delta.content)
            
            return OpenAIStreamResponse(stream)
            
        except Exception as e:
            raise Exception(f"OpenAI streaming error: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        return {
            "preferred_model": self.preferred_model,
            "external_model": self.external_model,
            "gemini_available": self.gemini_model is not None,
            "openai_available": self.openai_client is not None,
            "models_configured": {
                "gemini": self.gemini_model is not None,
                "openai": self.openai_client is not None
            }
        }

class QueryCache:
    """Manages caching of queries and responses"""
    
    def __init__(self, app_db_path: str, similarity_threshold: float = 0.85):
        self.app_db_path = app_db_path
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)
        self.setup_cache_tables()
    
    def setup_cache_tables(self):
        """Create cache tables if they don't exist"""
        conn = sqlite3.connect(self.app_db_path)
        cursor = conn.cursor()
        
        # Create query cache table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT UNIQUE NOT NULL,
                original_query TEXT NOT NULL,
                query_embedding BLOB NOT NULL,
                response_answer TEXT NOT NULL,
                response_sources TEXT NOT NULL,
                confidence REAL NOT NULL,
                security_level TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1,
                last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index for faster lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_query_hash ON query_cache(query_hash)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_security_level ON query_cache(security_level)
        ''')
        
        conn.commit()
        conn.close()
        self.logger.info("✅ Query cache tables initialized")
    
    def _generate_query_hash(self, query: str, security_level: str) -> str:
        """Generate a hash for the query and security level"""
        content = f"{query.lower().strip()}:{security_level}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _serialize_sources(self, sources: List[RetrievedDocument]) -> str:
        """Serialize sources to JSON string"""
        serialized = []
        for source in sources:
            serialized.append({
                'document_id': source.document_id,
                'url': source.url,
                'title': source.title,
                'chunk_text': source.chunk_text,
                'similarity_score': source.similarity_score,
                'source_type': source.source_type,
                'chunk_index': source.chunk_index
            })
        return json.dumps(serialized)
    
    def _deserialize_sources(self, sources_json: str) -> List[RetrievedDocument]:
        """Deserialize sources from JSON string"""
        sources_data = json.loads(sources_json)
        sources = []
        for data in sources_data:
            sources.append(RetrievedDocument(**data))
        return sources
    
    def find_similar_cached_query(self, query: str, query_embedding: np.ndarray, 
                                 security_level: str) -> Optional[CachedQuery]:
        """Find a similar cached query using embeddings"""
        # Always create a new connection in the current thread to avoid threading issues
        try:
            conn = sqlite3.connect(self.app_db_path)
            cursor = conn.cursor()
            
            # Use a simple timeout approach for the database query
            cursor.execute('''
                SELECT id, original_query, query_embedding, response_answer, 
                       response_sources, confidence, security_level, timestamp, access_count
                FROM query_cache 
                WHERE security_level = ?
                ORDER BY access_count DESC, last_accessed DESC
                LIMIT 100
            ''', (security_level,))
            
            cached_queries = cursor.fetchall()
            
            if not cached_queries:
                conn.close()
                return None
            
            best_match = None
            best_similarity = 0.0
            
            for row in cached_queries:
                (cache_id, original_query, embedding_blob, response_answer, 
                 response_sources, confidence, sec_level, timestamp, access_count) = row
                
                try:
                    # Decode cached embedding
                    cached_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    
                    # Calculate similarity
                    similarity = cosine_similarity([query_embedding], [cached_embedding])[0][0]
                    
                    if similarity > best_similarity and similarity >= self.similarity_threshold:
                        best_similarity = similarity
                        best_match = CachedQuery(
                            query_id=cache_id,
                            original_query=original_query,
                            query_embedding=cached_embedding,
                            response_answer=response_answer,
                            response_sources=response_sources,  # Keep as JSON string, don't parse here
                            confidence=confidence,
                            security_level=sec_level,
                            timestamp=datetime.fromisoformat(timestamp),
                            access_count=access_count
                        )
                
                except Exception as e:
                    self.logger.warning(f"Error processing cached query {cache_id}: {e}")
                    continue
            
            if best_match:
                self.logger.info(f"🎯 Cache hit! Found similar query with {best_similarity:.3f} similarity")
                self.logger.info(f"   Original: '{query}'")
                self.logger.info(f"   Cached: '{best_match.original_query}'")
                
                # Update access count and last accessed time
                cursor.execute('''
                    UPDATE query_cache 
                    SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (best_match.query_id,))
                conn.commit()
            
            conn.close()
            return best_match
            
        except Exception as e:
            self.logger.error(f"Error searching cache: {e}")
            if 'conn' in locals():
                try:
                    conn.close()
                except:
                    pass
            return None
    
    def cache_query_response(self, query: str, query_embedding: np.ndarray, 
                           response: RAGResponse, security_level: str):
        """Cache a query and its response"""
        try:
            conn = sqlite3.connect(self.app_db_path)
            cursor = conn.cursor()
            
            query_hash = self._generate_query_hash(query, security_level)
            sources_json = self._serialize_sources(response.sources)
            
            # Store in cache (replace if exists)
            cursor.execute('''
                INSERT OR REPLACE INTO query_cache 
                (query_hash, original_query, query_embedding, response_answer, 
                 response_sources, confidence, security_level)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                query_hash,
                query,
                query_embedding.tobytes(),
                response.answer,
                sources_json,
                response.confidence,
                security_level
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"💾 Cached query: '{query[:50]}...' (security: {security_level})")
            
        except Exception as e:
            self.logger.error(f"Error caching query: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        conn = sqlite3.connect(self.app_db_path)
        cursor = conn.cursor()
        
        try:
            # Total cached queries
            cursor.execute('SELECT COUNT(*) FROM query_cache')
            total_queries = cursor.fetchone()[0]
            
            # Most accessed queries
            cursor.execute('''
                SELECT original_query, access_count 
                FROM query_cache 
                ORDER BY access_count DESC 
                LIMIT 5
            ''')
            top_queries = cursor.fetchall()
            
            # Cache by security level
            cursor.execute('''
                SELECT security_level, COUNT(*) 
                FROM query_cache 
                GROUP BY security_level
            ''')
            by_security = dict(cursor.fetchall())
            
            return {
                'total_cached_queries': total_queries,
                'top_queries': top_queries,
                'by_security_level': by_security
            }
            
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {}
        finally:
            conn.close()
    
    def clear_old_cache_entries(self, days_old: int = 30):
        """Clear cache entries older than specified days"""
        conn = sqlite3.connect(self.app_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                DELETE FROM query_cache 
                WHERE timestamp < datetime('now', '-{} days')
            '''.format(days_old))
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            self.logger.info(f"🧹 Cleared {deleted_count} old cache entries (>{days_old} days)")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error clearing old cache: {e}")
            return 0
        finally:
            conn.close()

def get_knowledge_db_path():
    """Get knowledge base database path"""
    db_path = os.getenv('KNOWLEDGE_DB_PATH')
    if db_path:
        return os.path.abspath(db_path)
    else:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge.db")

def get_app_db_path():
    """Get application database path"""
    db_path = os.getenv('APP_DB_PATH')
    if db_path:
        return os.path.abspath(db_path)
    else:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.db")

class RAGAgent:
    def __init__(self, knowledge_db_path: str = None, app_db_path: str = None,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 cache_similarity_threshold: float = 0.85,
                 preferred_model: str = "gemini"):
        """
        Initialize RAG Agent
        
        Args:
            knowledge_db_path: Path to the knowledge base SQLite database (documents, embeddings)
            app_db_path: Path to the application SQLite database (interactions, feedback, cache)
            embedding_model: Name of the sentence transformer model
            cache_similarity_threshold: Threshold for cache hit similarity
            preferred_model: "gemini" for Gemini Flash 2.0, "openai" for GPT-4o-mini
        """
        # Set up database paths
        self.knowledge_db_path = knowledge_db_path if knowledge_db_path else get_knowledge_db_path()
        self.app_db_path = app_db_path if app_db_path else get_app_db_path()
        
        # Keep old db_path for backward compatibility (use app_db_path)
        self.db_path = self.app_db_path
        
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.preferred_model = preferred_model
        self.security_router = SecurityLevelRouter(preferred_model=preferred_model)
        self.setup_logging()
        
        # Initialize query cache with app database
        self.query_cache = QueryCache(self.app_db_path, cache_similarity_threshold)
        
        # Simple model cache directory
        self.model_cache_dir = r"C:\python\jeasearch\models\sentence-transformers"
        
        # Initialize embedding model
        try:
            self.embedding_model = self._load_embedding_model()
            self.logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
        
        # Log model configuration and database paths
        model_info = self.security_router.get_model_info()
        self.logger.info(f"🤖 Model configuration: {model_info}")
        self.logger.info(f"📚 Knowledge DB: {self.knowledge_db_path}")
        self.logger.info(f"💾 Application DB: {self.app_db_path}")
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_embedding_model(self):
        """Load embedding model from models folder"""
        model_name = "all-MiniLM-L6-v2"
        model_path = os.path.join(self.model_cache_dir, model_name)
        
        self.logger.info(f"Loading embedding model from: {model_path}")
        
        try:
            model = SentenceTransformer(model_path, device='cpu')
            self.logger.info("✅ Successfully loaded model")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {e}")
            raise Exception(f"Cannot load embedding model from: {model_path}")

    def determine_security_level(self, query: str, user_context: Dict[str, Any] = None) -> str:
        """Determine appropriate security level based on query and user context"""
        # For now, simple implementation - can be enhanced with more sophisticated routing
        if user_context and user_context.get('user_type') == 'employee':
            return 'internal'
        elif user_context and user_context.get('user_type') == 'privileged':
            return 'sensitive'
        else:
            return 'external'
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode query into embedding vector"""
        if self.embedding_model is None:
            self.embedding_model = self._load_embedding_model()
        
        return self.embedding_model.encode(query, convert_to_numpy=True, show_progress_bar=False)
    
    def calculate_similarity(self, query_embedding: np.ndarray, doc_embedding: np.ndarray) -> float:
        """Calculate cosine similarity between query and document embeddings"""
        # Normalize vectors
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norm = doc_embedding / np.linalg.norm(doc_embedding)
        
        # Calculate cosine similarity
        similarity = np.dot(query_norm, doc_norm)
        return float(similarity)
    
    def retrieve_documents(self, query: str, security_level: str = "external", 
                          top_k: int = 5, min_similarity: float = 0.3) -> List[RetrievedDocument]:
        """Retrieve relevant documents from the knowledge base with forced refresh"""
        
        # Force clear any cached results
        if hasattr(self, '_cached_results'):
            delattr(self, '_cached_results')
        
        self.logger.info(f"🔍 RETRIEVE_DOCUMENTS: Starting fresh search for: '{query}'")
        
        try:
            # Encode the query
            query_embedding = self.encode_query(query)
            self.logger.info(f"🔢 Generated new embedding for query (shape: {query_embedding.shape})")
            
            # Connect to knowledge base database
            conn = sqlite3.connect(self.knowledge_db_path)
            cursor = conn.cursor()
            
            # Check what tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            self.logger.info(f"📊 Available tables in knowledge database: {tables}")
            
            # First, let's check what columns actually exist in documents table
            cursor.execute("PRAGMA table_info(documents)")
            columns = [row[1] for row in cursor.fetchall()]
            self.logger.info(f"📊 Available columns in documents table: {columns}")
            
            # Check if there's an embeddings table
            if 'embeddings' in tables:
                cursor.execute("PRAGMA table_info(embeddings)")
                embed_columns = [row[1] for row in cursor.fetchall()]
                self.logger.info(f"📊 Available columns in embeddings table: {embed_columns}")
                
                # Query documents with embeddings
                cursor.execute("""
                    SELECT d.id, d.title, d.url, d.extracted_text, e.embedding
                    FROM documents d
                    JOIN embeddings e ON d.id = e.document_id
                    WHERE d.embedding_status = 'completed'
                    ORDER BY d.id
                """)
            else:
                # Check if embedding is stored directly in documents table
                if 'embedding' in columns:
                    cursor.execute("""
                        SELECT id, title, url, extracted_text, embedding
                        FROM documents 
                        WHERE embedding_status = 'completed'
                        ORDER BY id
                    """)
                else:
                    # No embeddings found - let's see what we have
                    self.logger.error("No embedding column found in documents table and no embeddings table exists")
                    cursor.execute("""
                        SELECT id, title, url, extracted_text
                        FROM documents 
                        WHERE extracted_text IS NOT NULL AND extracted_text != ''
                        ORDER BY id
                        LIMIT 10
                    """)
                    sample_rows = cursor.fetchall()
                    self.logger.info(f"📊 Sample documents (first 10): {len(sample_rows)} found")
                    for i, row in enumerate(sample_rows[:3]):
                        self.logger.info(f"  Sample {i+1}: ID={row[0]}, Title='{row[1][:50]}...'")
                    conn.close()
                    return []
            
            results = []
            processed_count = 0
            
            for row in cursor.fetchall():
                processed_count += 1
                
                doc_id, title, url, extracted_text, embedding_blob = row
                
                if not extracted_text or extracted_text.strip() == '':
                    continue
                    
                try:
                    # Decode embedding
                    doc_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    
                    # Calculate similarity
                    similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
                    
                    # Apply similarity threshold
                    if similarity >= min_similarity:
                        results.append(RetrievedDocument(
                            document_id=doc_id,
                            url=url,
                            title=title,
                            chunk_text=extracted_text,  # Use extracted_text as chunk_text
                            similarity_score=float(similarity),
                            source_type="external",  # Default since no security_level column
                            chunk_index=0
                        ))
                        
                except Exception as e:
                    self.logger.warning(f"Error processing document {doc_id}: {e}")
                    continue
            
            conn.close()
            
            # Sort by similarity and take top results
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            final_results = results[:top_k]
            
            self.logger.info(f"🎯 RETRIEVE_DOCUMENTS: Processed {processed_count} docs, found {len(results)} above threshold, returning top {len(final_results)}")
            
            # Log the actual results we're returning
            for i, doc in enumerate(final_results):
                self.logger.info(f"  Result {i+1}: '{doc.title}' (score: {doc.similarity_score:.4f}) - ID: {doc.document_id}")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error in retrieve_documents: {e}")
            return []

    def _check_cache(self, query: str, security_level: str):
        """Check cache for a similar query."""
        try:
            query_embedding = self.encode_query(query)
            cached_result = self.query_cache.find_similar_cached_query(
                query, query_embedding, security_level
            )
            
            if cached_result:
                self.logger.info("✅ Returning cached response")
                self.last_sources = self.query_cache._deserialize_sources(cached_result.response_sources)
                self.last_confidence = cached_result.confidence
                self._is_cached_response = True
                return cached_result.response_answer
        except Exception as e:
            if self._is_network_error(e):
                self.logger.warning(f"Network error during cache lookup: {e}")
            else:
                self.logger.warning(f"Cache lookup failed: {e}")
        return None

    def _retrieve_and_process_documents(self, query: str, security_level: str, top_k: int, min_similarity: float):
        """Retrieve documents and handle cases with no results."""
        self.logger.info("🔍 Starting document retrieval...")
        retrieved_docs = self.retrieve_documents(query, security_level, top_k, min_similarity)
        
        self.logger.info(f"📄 Retrieved {len(retrieved_docs)} documents:")
        for i, doc in enumerate(retrieved_docs[:3]):
            self.logger.info(f"  {i+1}. {doc.title} (score: {doc.similarity_score:.3f})")
        
        import copy
        self.last_sources = copy.deepcopy(retrieved_docs)
        self.logger.info(f"✅ Stored {len(self.last_sources)} sources for display (even if AI fails)")

        if not retrieved_docs:
            self.last_sources = []
            self.logger.info("❌ No documents found - sources explicitly set to empty")
            return None, "I couldn't find relevant information in the JEA knowledge base to answer your question. You might want to try rephrasing your question or contact JEA customer service directly at (904) 665-6000."

        return retrieved_docs, None

    def _generate_response_with_timeout(self, prompt: str, timeout: float = 30.0):
        """Generate response from LLM with a timeout."""
        response_container = [None]
        exception_container = [None]

        def api_thread():
            try:
                self.logger.info("🚀 Starting API call...")
                response = self.security_router.generate_content(prompt, stream=False)
                self.logger.info("✅ Received complete response from API")
                response_container[0] = response.text
            except Exception as e:
                exception_container[0] = e

        thread = threading.Thread(target=api_thread)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            self.logger.error("API call timed out")
            return None, "\n\n🕐 **Request Timed Out**\n\nThe request is taking longer than expected. This could be due to:\n- Network connectivity issues\n\n💡 **Note:** I've found relevant sources that may contain your answer - check the sources section below!"
        
        if exception_container[0]:
            raise exception_container[0]

        return response_container[0], None

    def _cache_new_response(self, query: str, response_text: str, retrieved_docs: List[RetrievedDocument], security_level: str):
        """Cache the newly generated response."""
        if len(response_text.strip()) > 10:
            try:
                calculated_confidence = self.calculate_confidence(query, retrieved_docs, response_text)
                self.last_confidence = calculated_confidence
                
                response_obj = RAGResponse(
                    answer=response_text,
                    sources=retrieved_docs,
                    reasoning="Generated via API call",
                    confidence=calculated_confidence,
                    security_level=security_level
                )
                
                query_embedding = self.encode_query(query)
                self.query_cache.cache_query_response(
                    query, query_embedding, response_obj, security_level
                )
                self.logger.info("✅ Cached complete response")
            except Exception as e:
                if self._is_network_error(e):
                    self.logger.warning(f"Network error while caching: {e}")
                else:
                    self.logger.warning(f"Failed to cache response: {e}")

    def query_response(self, query: str, user_context=None, top_k: int = 5, 
             min_similarity: float = 0.3, high_reasoning: bool = True):
        """Process query and return complete response (no streaming)"""
        try:
            self.reset_state()
            
            mode = "high reasoning" if high_reasoning else "standard"
            self.logger.info(f"🔄 QUERY: Starting {mode} query: '{query}' - All state cleared")
            
            search_query = query
            if "Current question:" in query:
                search_query = query.split("Current question:")[-1].strip()
                self.logger.info(f"📝 Extracted current question for search: '{search_query}'")
            
            security_level = self.determine_security_level(search_query, user_context)
            
            # Step 1: Check cache
            cached_response = self._check_cache(search_query, security_level)
            if cached_response:
                yield cached_response
                return

            # Step 2: Retrieve documents
            retrieved_docs, error_message = self._retrieve_and_process_documents(
                search_query, security_level, top_k, min_similarity
            )
            if error_message:
                yield error_message
                return

            # Step 3: Generate response
            context = self.build_context_from_docs(retrieved_docs)
            prompt = self.build_rag_prompt(query, context, user_context)
            
            full_response, error_message = self._generate_response_with_timeout(prompt)
            if error_message:
                yield error_message
                return

            if full_response:
                # Step 4: Cache new response
                self._cache_new_response(search_query, full_response, retrieved_docs, security_level)
                yield full_response
            else:
                yield "I'm sorry, but I couldn't generate a response at this time."
                        
        except Exception as e:
            if self._is_network_error(e):
                self.logger.error(f"Network error during response generation: {e}")
                yield self._handle_network_error(e, "generating response")
            else:
                self.logger.error(f"Error during response generation: {e}")
                yield f"I encountered an error while generating the response: {str(e)}"
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get query cache statistics"""
        stats = self.query_cache.get_cache_stats()
        stats["similarity_threshold"] = self.query_cache.similarity_threshold
        return stats
    
    def clear_cache(self, days_old: int = None):
        """Clear query cache"""
        if days_old:
            return self.query_cache.clear_old_cache_entries(days_old)
        else:
            # Clear all cache from app database
            conn = sqlite3.connect(self.app_db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM query_cache')
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            self.logger.info(f"🧹 Cleared all {deleted_count} cache entries")
            return deleted_count

    def build_context_from_docs(self, retrieved_docs: List[RetrievedDocument]) -> str:
        """Build context string from retrieved documents"""
        if not retrieved_docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"Source {i} (from {doc.title}):")
            context_parts.append(doc.chunk_text)
            context_parts.append(f"URL: {doc.url}")
            context_parts.append("---")
        
        return "\n".join(context_parts)

    def build_rag_prompt(self, query: str, context: str, user_context: Dict[str, Any] = None) -> str:
        """Build the RAG prompt for the LLM"""
        
        prompt = f"""You are a helpful AI assistant for JEA, a municipal utility company in Jacksonville, Florida. Your role is to provide accurate, helpful information about JEA's services, rates, policies, and procedures.

CONTEXT FROM JEA KNOWLEDGE BASE:
{context}

CUSTOMER QUESTION: {query}

SPECIAL HANDLING FOR PAYMENT ASSISTANCE:
If the customer is asking about payment difficulties, bill assistance, or can't pay their bill, prioritize information about:
- Payment plan options and arrangements
- Financial assistance programs
- Low-income assistance programs
- Emergency payment assistance
- Budget billing options
- Ways to reduce bills or manage costs
- Contact information for customer assistance representatives

INSTRUCTIONS:
1. Use ONLY the information provided in the context above to answer the question
2. If the context doesn't contain enough information to fully answer the question, say so clearly
3. Be conversational and helpful, as if you're a knowledgeable JEA customer service representative
4. Include specific details like rates, phone numbers, URLs when available in the context
5. If you mention rates or fees, include any relevant caveats about when they apply
6. For contact information, always provide phone numbers and relevant websites when available
7. For payment difficulties, be empathetic and focus on available assistance options
8. If the question is about something not covered in the context, suggest contacting JEA customer service

RESPONSE FORMAT:
- Write in a clear, friendly, professional tone
- Use bullet points or numbered lists when helpful for readability
- Include relevant URLs or phone numbers from the context
- For payment assistance, organize information clearly with available options
- End with a helpful suggestion if appropriate

Please provide your response:"""

        return prompt

    def _is_network_error(self, error: Exception) -> bool:
        """Check if an error is network-related"""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Common network error patterns
        network_indicators = [
            'ssl_error_ssl',
            'certificate_verify_failed',
            'handshake failed',
            'connection error',
            'timeout',
            'network error',
            'ssl error',
            'connection refused',
            'connection timeout',
            'name resolution failed',
            'dns',
            'socket error',
            'http error',
            'unable to connect',
            'connection aborted',
            'connection reset',
            'certificate',
            'ssl routines',
            'openssl_internal'
        ]
        
        # Check error message and type
        is_network = any(indicator in error_str for indicator in network_indicators)
        is_network = is_network or any(indicator in error_type for indicator in ['connection', 'timeout', 'ssl', 'http'])
        
        return is_network

    def _handle_network_error(self, error: Exception, context: str = "processing your request") -> str:
        """Generate user-friendly message for network errors"""
        self.logger.error(f"Network error during {context}: {error}")
        
        return """🌐 **Network Connection Issue**

I'm experiencing connectivity issues while processing your request. This could be due to:
- SSL certificate verification problems
- Network connectivity issues
- External service unavailability

**Please try again in a few moments.** If the issue persists, you can:
- Visit **jea.com** directly for information

I apologize for the inconvenience!"""

    def get_last_sources(self) -> List[RetrievedDocument]:
        """Get the sources from the last query"""
        return getattr(self, 'last_sources', [])
    
    def get_last_confidence(self) -> float:
        """Get the confidence from the last query"""
        return getattr(self, 'last_confidence', 0.8)
    
    def reset_state(self):
        """Reset the agent's state"""
        self.last_sources = []
        self.last_confidence = 0.8
        self._is_cached_response = False
        if hasattr(self, '_cached_results'):
            delattr(self, '_cached_results')
        self.logger.info("🔄 Agent state reset")

    def calculate_confidence(self, query: str, retrieved_docs: List[RetrievedDocument], response_text: str) -> float:
        """Calculate confidence score for the response"""
        if not retrieved_docs:
            return 0.0
        
        # Base confidence on similarity scores of retrieved documents
        avg_similarity = sum(doc.similarity_score for doc in retrieved_docs) / len(retrieved_docs)
        
        # Adjust based on number of sources
        source_factor = min(1.0, len(retrieved_docs) / 3.0)  # Optimal around 3 sources
        
        # Adjust based on response length (longer responses might be more detailed)
        length_factor = min(1.0, len(response_text) / 500.0)  # Normalize around 500 chars
        
        # Combined confidence
        confidence = (avg_similarity * 0.5) + (source_factor * 0.3) + (length_factor * 0.2)
        
        return min(0.95, max(0.1, confidence))  # Clamp between 0.1 and 0.95

def format_response(response: RAGResponse) -> str:
    """Format RAG response for display"""
    output = []
    output.append(f"**Answer** (Security Level: {response.security_level}):")
    output.append(response.answer)
    output.append("")
    
    if response.sources:
        output.append("**Sources:**")
        for i, source in enumerate(response.sources, 1):
            domain = urlparse(source.url).netloc
            output.append(f"{i}. {source.title} - {domain}")
            output.append(f"   URL: {source.url}")
            output.append(f"   Relevance: {source.similarity_score:.3f}")
            output.append("")
    
    output.append(f"**Confidence:** {response.confidence:.2f}")
    output.append(f"**Reasoning:** {response.reasoning}")
    
    return "\n".join(output)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="JEA RAG Agent")
    parser.add_argument("query", nargs="*", help="Query to ask")
    parser.add_argument("--security-level", choices=["external", "internal", "sensitive"], 
                       help="Override security level")
    parser.add_argument("--model", choices=["gemini", "openai"], default="gemini",
                       help="Choose AI model: 'gemini' for Gemini Flash 2.0, 'openai' for GPT-4o-mini")
    parser.add_argument("--top-k", type=int, default=5, help="Number of sources to retrieve")
    parser.add_argument("--min-similarity", type=float, default=0.3, help="Minimum similarity threshold")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    
    args = parser.parse_args()
    
    # Initialize RAG agent with selected model
    agent = RAGAgent(preferred_model=args.model)
    
    # Display model info
    model_info = agent.security_router.get_model_info()
    print(f"🤖 Using model: {model_info['external_model']} (preference: {model_info['preferred_model']})")
    print(f"📊 Available models: Gemini={model_info['gemini_available']}, OpenAI={model_info['openai_available']}")
    print()
    
    if args.interactive:
        print("JEA RAG Agent - Interactive Mode")
        print("Type 'quit' to exit\n")
        
        while True:
            query = input("Query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            user_context = {}
            if args.security_level:
                user_context['user_type'] = 'employee' if args.security_level == 'internal' else 'external'
            
            # Get the complete response for interactive mode
            print("Response:")
            for chunk in agent.query_response(
                query, 
                user_context=user_context,
                top_k=args.top_k,
                min_similarity=args.min_similarity
            ):
                print(chunk, end='', flush=True)
            
            # Display sources
            sources = agent.get_last_sources()
            if sources:
                print("\n\n**Sources:**")
                for i, source in enumerate(sources, 1):
                    domain = urlparse(source.url).netloc
                    print(f"{i}. {source.title} - {domain}")
                    print(f"   Relevance: {source.similarity_score:.3f}")
                    print(f"   URL: {source.url}")
            
            print("\n" + "="*80 + "\n")
    
    elif args.query:
        query = " ".join(args.query)
        
        user_context = {}
        if args.security_level:
            user_context['user_type'] = 'employee' if args.security_level == 'internal' else 'external'
        
        # Get the complete response for command-line query
        print("Response:")
        for chunk in agent.query_response(
            query,
            user_context=user_context,
            top_k=args.top_k,
            min_similarity=args.min_similarity
        ):
            print(chunk, end='', flush=True)
        
        # Display sources
        sources = agent.get_last_sources()
        if sources:
            print("\n\n**Sources:**")
            for i, source in enumerate(sources, 1):
                domain = urlparse(source.url).netloc
                print(f"{i}. {source.title} - {domain}")
                print(f"   Relevance: {source.similarity_score:.3f}")
                print(f"   URL: {source.url}")
        print()
    
    else:
        parser.print_help() 
