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

@dataclass
class CachedQuery:
    """Represents a cached query with its response"""
    query_id: int
    original_query: str
    query_embedding: np.ndarray
    response_answer: str
    response_sources: List[Dict[str, Any]]  # Serialized sources
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
    
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        self.setup_models()
    
    def setup_models(self):
        """Initialize LLM models for different security levels"""
        # External level - Google Gemini Flash 2
        self.external_model = None
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.external_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("✅ Gemini model configured successfully")
        else:
            print("⚠️  GEMINI_API_KEY not found. External queries will not work.")
        
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
            return self.external_model  # Default fallback

class QueryCache:
    """Manages caching of queries and responses"""
    
    def __init__(self, db_path: str, similarity_threshold: float = 0.85):
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)
        self.setup_cache_tables()
    
    def setup_cache_tables(self):
        """Create cache tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get all cached queries for this security level
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
                            response_sources=json.loads(response_sources),
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
                
                return best_match
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error searching cache: {e}")
            return None
        finally:
            conn.close()
    
    def cache_query_response(self, query: str, query_embedding: np.ndarray, 
                           response: RAGResponse, security_level: str):
        """Cache a query and its response"""
        try:
            conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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

class RAGAgent:
    def __init__(self, db_path: str = "crawler.db", embedding_model: str = "all-MiniLM-L6-v2",
                 cache_similarity_threshold: float = 0.85):
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.security_router = SecurityLevelRouter()
        self.setup_logging()
        
        # Initialize query cache (always enabled)
        self.query_cache = QueryCache(db_path, cache_similarity_threshold)
        
        # Set up local model cache directory
        self.model_cache_dir = r"\\jeasas2p1\Utility Analytics\Load Research\Projects\jeacomsearch\models\sentence-transformers"
        
        # Initialize embedding model with local caching
        try:
            self.embedding_model = self._load_embedding_model()
            self.logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_embedding_model(self):
        """Load embedding model with local caching and UNC path support"""
        model_name = "all-MiniLM-L6-v2"
        local_model_path = os.path.join(self.model_cache_dir, model_name)
        
        try:
            # Try to load from local cache first
            if os.path.exists(local_model_path) and os.listdir(local_model_path):
                self.logger.info(f"Loading embedding model from cache: {local_model_path}")
                
                # Check if required files exist
                required_files = ['config.json', 'sentence_bert_config.json', 'modules.json']
                missing_files = []
                for file in required_files:
                    if not os.path.exists(os.path.join(local_model_path, file)):
                        missing_files.append(file)
                
                if missing_files:
                    self.logger.warning(f"Missing required files: {missing_files}")
                    raise FileNotFoundError(f"Missing model files: {missing_files}")
                
                # Method 1: Convert UNC path to a format that works with SentenceTransformer
                try:
                    # For UNC paths, we need to copy to a local temp directory or use a different approach
                    import tempfile
                    import shutil
                    
                    # Create a temporary local copy if on UNC path
                    if local_model_path.startswith('\\\\'):
                        self.logger.info("Detected UNC path, creating local copy...")
                        
                        # Create temp directory
                        temp_dir = tempfile.mkdtemp()
                        temp_model_path = os.path.join(temp_dir, model_name)
                        
                        # Copy model files to temp directory
                        shutil.copytree(local_model_path, temp_model_path)
                        
                        # Load from temp directory
                        model = SentenceTransformer(temp_model_path, device='cpu')
                        self.logger.info("✅ Successfully loaded model from temporary local copy")
                        
                        # Clean up temp directory
                        try:
                            shutil.rmtree(temp_dir)
                        except:
                            pass  # Don't fail if cleanup fails
                        
                        return model
                    else:
                        # Regular local path
                        model = SentenceTransformer(local_model_path, device='cpu')
                        self.logger.info("✅ Successfully loaded model from local path")
                        return model
                        
                except Exception as e:
                    self.logger.warning(f"Method 1 (direct loading) failed: {e}")
                    
                    # Method 2: Load components manually from UNC path
                    try:
                        from transformers import AutoTokenizer, AutoModel
                        
                        self.logger.info("Trying manual component loading...")
                        
                        # Load tokenizer and model manually
                        tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
                        transformer_model = AutoModel.from_pretrained(local_model_path, local_files_only=True)
                        
                        # Create SentenceTransformer from components
                        from sentence_transformers.models import Transformer, Pooling
                        
                        # Create the transformer component
                        word_embedding_model = Transformer(local_model_path)
                        
                        # Create pooling component
                        pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())
                        
                        # Create the sentence transformer
                        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cpu')
                        
                        self.logger.info("✅ Successfully loaded model using manual component loading")
                        return model
                        
                    except Exception as e2:
                        self.logger.warning(f"Method 2 (manual loading) failed: {e2}")
                        
                        # Method 3: Try using local_files_only with the model name
                        try:
                            model = SentenceTransformer(model_name, cache_folder=os.path.dirname(self.model_cache_dir), local_files_only=True, device='cpu')
                            self.logger.info("✅ Successfully loaded model using cache folder approach")
                            return model
                        except Exception as e3:
                            self.logger.warning(f"Method 3 failed: {e3}")
                            raise e3
            
            # If not cached, try to download
            self.logger.info(f"Model not found locally. Attempting to download: {model_name}")
            
            # Disable SSL verification temporarily (only for model download)
            original_ssl_context = ssl._create_default_https_context
            ssl._create_default_https_context = ssl._create_unverified_context
            
            # Disable urllib3 warnings
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            try:
                # Download and cache the model
                model = SentenceTransformer(model_name, cache_folder=os.path.dirname(self.model_cache_dir), device='cpu')
                
                # Save to specific local path for future use
                model.save(local_model_path)
                self.logger.info(f"Model downloaded and cached to: {local_model_path}")
                
                return model
                
            finally:
                # Restore original SSL context
                ssl._create_default_https_context = original_ssl_context
            
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise Exception(f"Cannot load embedding model. Please ensure all required files are in: {local_model_path}")

    def _check_huggingface_cache(self, model_name):
        """Check if model exists in HuggingFace cache"""
        try:
            from transformers import AutoTokenizer
            # This will succeed if the model is in HF cache
            AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            return True
        except:
            return False
    
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
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check what tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            self.logger.info(f"📊 Available tables in database: {tables}")
            
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
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine intent and information needs"""
        query_lower = query.lower()
        
        # Simple intent classification - can be enhanced
        intents = {
            'factual': ['what', 'who', 'when', 'where', 'how much', 'how many'],
            'procedural': ['how to', 'how do', 'process', 'procedure', 'steps'],
            'policy': ['policy', 'rule', 'regulation', 'requirement', 'allowed', 'permitted'],
            'contact': ['contact', 'phone', 'email', 'address', 'reach'],
            'comparison': ['compare', 'difference', 'versus', 'vs', 'better'],
        }
        
        detected_intents = []
        for intent, keywords in intents.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_intents.append(intent)
        
        if not detected_intents:
            detected_intents = ['general']
        
        return {
            'primary_intent': detected_intents[0],
            'all_intents': detected_intents,
            'requires_specific_sources': any(word in query_lower for word in ['policy', 'regulation', 'procedure', 'contact']),
            'is_comparative': 'comparison' in detected_intents
        }
    
    def analyze_query_ambiguity_with_llm(self, query: str, retrieved_docs: List[RetrievedDocument], security_level: str) -> Dict[str, Any]:
        """Use LLM to analyze if query is ambiguous and needs clarification"""
        
        model = self.security_router.get_model(security_level)
        
        if model is None:
            # Fallback to simple analysis if no LLM available
            return {
                'is_ambiguous': False,
                'needs_clarification': False,
                'clarifying_questions': [],
                'confidence_reduction': 0.0,
                'reasoning': "No LLM available for ambiguity analysis"
            }
        
        # Prepare context from retrieved documents
        context_preview = ""
        if retrieved_docs:
            context_preview = "\n".join([
                f"- {doc.title}: {doc.chunk_text[:200]}..." 
                for doc in retrieved_docs[:3]
            ])
        
        ambiguity_analysis_prompt = f"""You are an expert at analyzing user queries for ambiguity and determining when clarification is needed.

Analyze this user query and determine if it's ambiguous or needs clarification to provide an accurate answer.

USER QUERY: "{query}"

AVAILABLE CONTEXT (top 3 most relevant documents):
{context_preview if context_preview else "No relevant documents found"}

IMPORTANT ASSUMPTIONS TO MAKE:
- For rates/pricing: Assume CURRENT rates unless historical rates are explicitly requested
- For schedules/hours: Assume CURRENT schedule unless otherwise specified
- For policies/procedures: Assume CURRENT/ACTIVE policies unless historical versions are requested
- For time-sensitive information: Assume TODAY/NOW unless a specific time is mentioned
- For contact information: Assume CURRENT contact details

DEFAULT ASSUMPTIONS FOR UTILITY QUERIES:
- "electric rates" → Assume RESIDENTIAL electric rates (most common request)
- "water rates" → Assume RESIDENTIAL water rates
- "fuel rates" → Assume JEA's electricity generation fuel rates (what customers pay)
- "break down rates" → Provide standard residential rate components
- "hours" → Assume CUSTOMER SERVICE hours (most common)
- "contact" → Assume CUSTOMER SERVICE contact info
- "apply for service" → Assume RESIDENTIAL service connection
- "pay bill" → Assume standard online/phone payment methods

GUIDANCE:
- Utility customers usually want the most common/standard information
- Only request clarification if the query would lead to SIGNIFICANTLY different answers
- If you can provide a helpful answer with reasonable defaults, do NOT request clarification
- For rate queries, provide the standard residential information and briefly mention alternatives exist
- Err strongly on the side of being helpful rather than asking for clarification
- Most customers are residential, so default to residential information

Your task:
1. Determine if the query is ambiguous AFTER applying all assumptions above
2. Consider if the available context provides enough information to give a useful answer
3. Only suggest clarification if absolutely necessary for accuracy

Respond in this JSON format:
{{
    "is_ambiguous": true/false,
    "needs_clarification": true/false,
    "confidence_in_current_context": 0.0-1.0,
    "clarifying_questions": ["question 1", "question 2"],
    "reasoning": "Brief explanation of your analysis",
    "assumptions_applied": ["assumption 1", "assumption 2"],
    "default_response_possible": true/false
}}

Examples of queries that should NOT need clarification (provide default answer):
- "What are the electric rates?" → Provide residential rates, mention commercial available
- "Break down the residential electric rate" → Provide rate components (basic charge, energy tiers, fuel)
- "What are fuel rates?" → Provide current JEA fuel rates for electricity generation
- "What are JEA's hours?" → Provide customer service hours, mention other departments  
- "How do I pay my bill?" → Provide standard payment methods
- "What is JEA's phone number?" → Provide main customer service number

Examples of queries that DO need clarification:
- "How much will my bill be?" → Need usage/property details
- "When will my service be restored?" → Need specific outage information
- "Can I get a discount?" → Need customer type/situation details

Be extremely conservative about requesting clarification for standard utility information."""

        try:
            if security_level == "external" and model:
                response = model.generate_content(ambiguity_analysis_prompt)
                response_text = response.text.strip()
                
                # Try to parse JSON response
                try:
                    # Clean up the response to extract JSON
                    if "```json" in response_text:
                        json_start = response_text.find("```json") + 7
                        json_end = response_text.find("```", json_start)
                        response_text = response_text[json_start:json_end].strip()
                    elif "{" in response_text and "}" in response_text:
                        # Extract JSON from response
                        json_start = response_text.find("{")
                        json_end = response_text.rfind("}") + 1
                        response_text = response_text[json_start:json_end]
                    
                    analysis = json.loads(response_text)
                    
                    # Validate the response structure
                    required_keys = ['is_ambiguous', 'needs_clarification', 'confidence_in_current_context', 'reasoning']
                    if all(key in analysis for key in required_keys):
                        # If default response is possible, reduce need for clarification
                        if analysis.get('default_response_possible', False):
                            analysis['needs_clarification'] = False
                            analysis['confidence_in_current_context'] = max(0.7, analysis['confidence_in_current_context'])
                        
                        # Calculate confidence reduction based on ambiguity
                        confidence_reduction = max(0.0, (1.0 - analysis['confidence_in_current_context']) * 0.2)
                        
                        return {
                            'is_ambiguous': analysis['is_ambiguous'],
                            'needs_clarification': analysis['needs_clarification'],
                            'clarifying_questions': analysis.get('clarifying_questions', []),
                            'confidence_reduction': confidence_reduction,
                            'reasoning': analysis['reasoning'],
                            'llm_confidence': analysis['confidence_in_current_context'],
                            'assumptions_applied': analysis.get('assumptions_applied', []),
                            'default_response_possible': analysis.get('default_response_possible', False)
                        }
                    else:
                        raise ValueError("Missing required keys in LLM response")
                    
                except (json.JSONDecodeError, ValueError) as e:
                    self.logger.warning(f"Failed to parse LLM ambiguity analysis: {e}")
                    # Fallback: be more lenient for common queries
                    query_lower = query.lower()
                    common_queries = ['rate', 'hour', 'contact', 'phone', 'pay', 'bill']
                    is_common = any(word in query_lower for word in common_queries)
                    
                    return {
                        'is_ambiguous': False,
                        'needs_clarification': False,
                        'clarifying_questions': [],
                        'confidence_reduction': 0.1 if not is_common else 0.0,
                        'reasoning': "LLM response parsing failed, used fallback analysis with common query detection",
                        'assumptions_applied': ["current/today assumed for time-sensitive queries", "residential assumed for rate queries"],
                        'default_response_possible': is_common
                    }
            
            else:
                # Fallback for non-external security levels
                return {
                    'is_ambiguous': False,
                    'needs_clarification': False,
                    'clarifying_questions': [],
                    'confidence_reduction': 0.0,
                    'reasoning': f"LLM not available for {security_level} security level",
                    'assumptions_applied': ["current/today assumed for time-sensitive queries", "residential assumed for rate queries"]
                }
            
        except Exception as e:
            self.logger.error(f"Error in LLM ambiguity analysis: {e}")
            return {
                'is_ambiguous': False,
                'needs_clarification': False,
                'clarifying_questions': [],
                'confidence_reduction': 0.0,
                'reasoning': f"Error in ambiguity analysis: {str(e)}",
                'assumptions_applied': []
            }

    def generate_response(self, query: str, retrieved_docs: List[RetrievedDocument], 
                         query_intent: Dict[str, Any], security_level: str) -> RAGResponse:
        """Generate response using appropriate LLM with LLM-powered ambiguity handling"""
        
        # Use LLM to analyze query ambiguity
        ambiguity_analysis = self.analyze_query_ambiguity_with_llm(query, retrieved_docs, security_level)
        
        model = self.security_router.get_model(security_level)
        
        if model is None:
            return RAGResponse(
                answer="I apologize, but the appropriate language model for your security level is not available at the moment.",
                sources=[],
                reasoning="No LLM available for security level: " + security_level,
                confidence=0.0,
                security_level=security_level
            )
        
        # Prepare context from retrieved documents
        context_chunks = []
        for i, doc in enumerate(retrieved_docs):
            source_info = f"Source {i+1}: {doc.title} ({doc.url})"
            context_chunks.append(f"{source_info}\n{doc.chunk_text}\n")
        
        context = "\n".join(context_chunks)
        
        # Handle ambiguous queries that need clarification
        if ambiguity_analysis['needs_clarification'] and ambiguity_analysis['clarifying_questions']:
            
            clarification_prompt = f"""You are an intelligent assistant for JEA.

The user has asked a question that could have multiple interpretations. Your analysis indicates this needs clarification.

USER QUERY: "{query}"

AMBIGUITY ANALYSIS:
- Reasoning: {ambiguity_analysis['reasoning']}
- Suggested clarifying questions: {ambiguity_analysis['clarifying_questions']}

AVAILABLE CONTEXT from JEA sources:
{context}

Your task:
1. Acknowledge that you found relevant information
2. Explain briefly why clarification would help provide a better answer
3. Ask the suggested clarifying questions in a natural, helpful way
4. Optionally provide a brief overview of what information is available

Be conversational and helpful. Show that you have information but need clarification to give the best answer.

Security Level: {security_level}"""

            try:
                if security_level == "external" and model:
                    response = model.generate_content(clarification_prompt)
                    answer = response.text
                    confidence = max(0.4, 0.8 - ambiguity_analysis['confidence_reduction'])
                else:
                    # Fallback for other security levels
                    answer = f"I found relevant information about your query, but to provide the most accurate answer, I need some clarification:\n\n"
                    for i, question in enumerate(ambiguity_analysis['clarifying_questions'], 1):
                        answer += f"{i}. {question}\n"
                    confidence = 0.4
                
            except Exception as e:
                self.logger.error(f"Error generating clarification response: {e}")
                answer = "I found information that might be relevant to your question, but I need clarification to provide the best answer. Could you please be more specific about what you're looking for?"
                confidence = 0.3
        
        else:
            # Handle clear queries with standard response generation
            system_prompt = f"""You are an intelligent assistant for JEA, a municipal utility company in Jacksonville, Florida. 
            
Your role is to provide accurate, helpful responses based on the provided context from JEA's official sources.

IMPORTANT ASSUMPTIONS TO MAKE:
- For rates/pricing: Assume CURRENT rates unless historical rates are explicitly requested
- For schedules/hours: Assume CURRENT schedule unless otherwise specified  
- For policies/procedures: Assume CURRENT/ACTIVE policies unless historical versions are requested
- For time-sensitive information: Assume TODAY/NOW unless a specific time is mentioned
- For contact information: Assume CURRENT contact details

DEFAULT ASSUMPTIONS FOR COMMON QUERIES:
- "electric rates" → Provide RESIDENTIAL electric rates first, then mention commercial options
- "water rates" → Provide RESIDENTIAL water rates first
- "hours" → Provide CUSTOMER SERVICE hours first, mention other departments if relevant
- "contact" → Provide CUSTOMER SERVICE contact info first
- "fuel rates" → Provide JEA's electricity generation fuel rates (what customers pay)

Guidelines:
1. Answer the user's question directly using the information provided in the context
2. Apply the assumptions above when interpreting queries
3. For rate queries, provide the most common type first (residential) and mention alternatives exist
4. Be concise and factual - avoid phrases like "I can help you" or "I found information"
5. Always cite your sources using the source numbers provided (e.g., "According to Source 1...")
6. Provide specific numbers, dates, and details when available in the context
7. If the context doesn't contain enough information to answer fully, say so briefly
8. Focus on giving useful information rather than asking for clarification

TONE: Direct, professional, informative. Get straight to the answer.

Security Level: {security_level}
Query Intent: {query_intent['primary_intent']}

Context from JEA sources:
{context}

Provide a comprehensive answer with proper source citations. Start with the most relevant information first."""

            user_prompt = f"Question: {query}"
            
            try:
                if security_level == "external" and model:
                    full_prompt = f"{system_prompt}\n\n{user_prompt}"
                    response = model.generate_content(full_prompt)
                    answer = response.text
                    confidence = max(0.5, 0.8 - ambiguity_analysis['confidence_reduction'])
                else:
                    answer = f"I can see relevant information about your query, but the appropriate language model for {security_level} level access is not yet configured. The retrieved sources contain information that could answer your question."
                    confidence = 0.5
                
            except Exception as e:
                self.logger.error(f"Error generating response: {e}")
                answer = "I apologize, but I encountered an error while generating a response. Please try rephrasing your question."
                confidence = 0.0
        
        # Generate reasoning
        if retrieved_docs:
            reasoning = f"Retrieved {len(retrieved_docs)} relevant documents with similarities ranging from {min([d.similarity_score for d in retrieved_docs]):.3f} to {max([d.similarity_score for d in retrieved_docs]):.3f}. "
            reasoning += f"Ambiguity analysis: {ambiguity_analysis['reasoning']}"
        else:
            reasoning = "No relevant documents found"
        
        return RAGResponse(
            answer=answer,
            sources=retrieved_docs,
            reasoning=reasoning,
            confidence=confidence,
            security_level=security_level
        )
    
    def clear_sources(self):
        """Explicitly clear stored sources"""
        self.last_sources = []
        self.logger.info("🔄 Sources explicitly cleared")

    def query(self, query: str, user_context: Dict[str, Any] = None, 
              top_k: int = 5, min_similarity: float = 0.3) -> RAGResponse:
        """Main query interface with caching support"""
        
        # Clear previous sources at the start of a new query
        self.last_sources = []
        self.logger.info(f"🔄 Starting new query: '{query}' - Sources cleared")
        
        # Step 1: Determine security level
        security_level = self.determine_security_level(query, user_context)
        
        # Step 2: Check cache first
        try:
            # Encode query for cache lookup
            query_embedding = self.encode_query(query)
            
            # Look for similar cached query
            cached_result = self.query_cache.find_similar_cached_query(
                query, query_embedding, security_level
            )
            
            if cached_result:
                # Return cached response
                sources = self._deserialize_sources(cached_result.response_sources)
                self.last_sources = sources.copy()
                
                self.logger.info(f"✅ Returning cached response for query")
                
                return RAGResponse(
                    answer=cached_result.response_answer,
                    sources=sources,
                    reasoning=f"Retrieved from cache (similarity: high, accessed {cached_result.access_count} times)",
                    confidence=cached_result.confidence,
                    security_level=security_level,
                    is_cached=True
                )
                
        except Exception as e:
            self.logger.warning(f"Cache lookup failed: {e}")
            # Continue with normal processing if cache fails
        
        # Step 3: Normal processing if no cache hit
        # Analyze query intent
        query_intent = self.analyze_query_intent(query)
        
        # Retrieve relevant documents using enhanced retrieval
        retrieved_docs = self.enhanced_retrieve_documents(
            query, 
            security_level=security_level, 
            top_k=top_k, 
            min_similarity=min_similarity
        )
        
        # Store sources immediately after retrieval
        self.last_sources = retrieved_docs.copy()
        self.logger.info(f"✅ Stored {len(self.last_sources)} sources")
        
        # Generate response
        response = self.generate_response(query, retrieved_docs, query_intent, security_level)
        
        # Cache the response if it meets quality threshold
        if response.confidence >= 0.5:
            try:
                query_embedding = self.encode_query(query)
                self.query_cache.cache_query_response(
                    query, query_embedding, response, security_level
                )
            except Exception as e:
                self.logger.warning(f"Failed to cache response: {e}")
        
        # Log the interaction
        self.log_interaction(query, response, user_context)
        
        return response
    
    def log_interaction(self, query: str, response: RAGResponse, user_context: Dict[str, Any] = None):
        """Log user interactions for monitoring and improvement"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create interaction log table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rag_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query TEXT NOT NULL,
                security_level TEXT,
                num_sources_retrieved INTEGER,
                confidence_score FLOAT,
                user_context_json TEXT
            )
        ''')
        
        user_context_json = json.dumps(user_context) if user_context else None
        
        cursor.execute('''
            INSERT INTO rag_interactions 
            (query, security_level, num_sources_retrieved, confidence_score, user_context_json)
            VALUES (?, ?, ?, ?, ?)
        ''', (query, response.security_level, len(response.sources), response.confidence, user_context_json))
        
        conn.commit()
        conn.close()

    def stream_query(self, query: str, user_context=None, top_k: int = 5, 
                    min_similarity: float = 0.3, high_reasoning: bool = True):
        """Stream tokens from the query response with caching support"""
        try:
            # FORCE clear all previous state at the very beginning
            self.last_sources = []
            if hasattr(self, '_cached_results'):
                delattr(self, '_cached_results')
            
            mode = "high reasoning" if high_reasoning else "standard"
            self.logger.info(f"🔄 STREAM_QUERY: Starting {mode} query: '{query}' - All state cleared")
            
            # Extract the actual question from enhanced query with conversation context
            search_query = query
            if "Current question:" in query:
                parts = query.split("Current question:")
                if len(parts) > 1:
                    search_query = parts[-1].strip()
                    self.logger.info(f"📝 Extracted current question for search: '{search_query}'")
            
            # Determine security level
            security_level = self.determine_security_level(search_query, user_context)
            
            # Step 1: Check cache first
            try:
                # Use the extracted search query for cache lookup
                query_embedding = self.encode_query(search_query)
                
                cached_result = self.query_cache.find_similar_cached_query(
                    search_query, query_embedding, security_level
                )
                
                if cached_result:
                    # Stream cached response
                    sources = self._deserialize_sources(cached_result.response_sources)
                    self.last_sources = sources.copy()
                    
                    self.logger.info(f"✅ Streaming cached response")
                    
                    # Stream the cached answer
                    cached_answer = cached_result.response_answer
                    words = cached_answer.split()
                    
                    for i, word in enumerate(words):
                        yield f" {word}" if i > 0 else word
                        
                        # Small delay for streaming effect
                        import time
                        time.sleep(0.05)
                    
                    return
                    
            except Exception as e:
                self.logger.warning(f"Cache lookup failed in streaming: {e}")
                # Continue with normal processing
            
            # Step 2: Normal processing if no cache hit
            self.logger.info(f"🔍 Starting {'enhanced' if high_reasoning else 'fast'} document retrieval...")
            retrieved_docs = self.enhanced_retrieve_documents(search_query, security_level, top_k, min_similarity, high_reasoning)
            
            # Log what we actually retrieved
            self.logger.info(f"📄 Retrieved {len(retrieved_docs)} documents:")
            for i, doc in enumerate(retrieved_docs[:3]):
                self.logger.info(f"  {i+1}. {doc.title} (score: {doc.similarity_score:.3f})")
            
            if not retrieved_docs:
                yield "I couldn't find relevant information in the JEA knowledge base to answer your question. "
                yield "You might want to try rephrasing your question or contact JEA customer service directly at (904) 665-6000."
                self.last_sources = []
                self.logger.info("❌ No documents found - sources explicitly set to empty")
                return
            
            # Store sources immediately after retrieval
            import copy
            self.last_sources = copy.deepcopy(retrieved_docs)
            self.logger.info(f"✅ Stored {len(self.last_sources)} sources for display (deep copy)")
            
            # Build context and stream response
            context = self.build_context_from_docs(retrieved_docs)
            model = self.security_router.get_model(security_level)
            
            if model is None:
                yield "I apologize, but the AI model is not available at the moment. "
                yield "Please try again later or contact JEA customer service at (904) 665-6000."
                return
            
            # Build the prompt and stream
            prompt = self.build_rag_prompt(query, context, user_context)
            
            # Collect full response for caching
            full_response = ""
            
            try:
                response = model.generate_content(prompt, stream=True)
                
                for chunk in response:
                    if chunk.text:
                        full_response += chunk.text
                        yield chunk.text
                
                # Cache the response after streaming
                if len(full_response.strip()) > 10:
                    try:
                        # Create a response object for caching
                        response_obj = RAGResponse(
                            answer=full_response,
                            sources=retrieved_docs,
                            reasoning="Generated via streaming",
                            confidence=0.8,  # Default confidence for streamed responses
                            security_level=security_level
                        )
                        
                        query_embedding = self.encode_query(search_query)
                        self.query_cache.cache_query_response(
                            search_query, query_embedding, response_obj, security_level
                        )
                        self.logger.info("✅ Cached streamed response")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to cache streamed response: {e}")
                        
            except Exception as e:
                self.logger.error(f"Error during streaming: {e}")
                yield f"I encountered an error while generating the response: {str(e)}"
            
        except Exception as e:
            self.logger.error(f"Error in stream_query: {e}")
            yield f"Error processing query: {str(e)}"
            self.last_sources = []

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
            # Clear all cache
            conn = sqlite3.connect(self.db_path)
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

    def analyze_query_for_search_strategy(self, query: str) -> Dict[str, Any]:
        """Analyze the query to determine the best search strategy"""
        
        # Get model for analysis
        model = self.security_router.get_model("external")
        
        if not model:
            # Fallback to simple keyword analysis
            return self._fallback_query_analysis(query)
        
        analysis_prompt = f"""Analyze this customer query and respond with ONLY a JSON object (no other text):

QUERY: "{query}"

Respond with this exact JSON structure:
{{
    "query_type": "specific_rate",
    "key_concepts": ["electric", "rates", "pricing"],
    "search_terms": ["electric rate", "electricity pricing", "rate schedule"],
    "alternative_phrasings": ["electricity prices", "power costs"],
    "specificity_level": "high",
    "context_needed": "current",
    "reasoning": "Customer wants current electricity pricing information"
}}

JSON Response:"""
        
        try:
            response = model.generate_content(analysis_prompt)
            response_text = response.text.strip()
            
            # Try to extract JSON from response
            import json
            import re
            
            # Look for JSON object in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                analysis = json.loads(json_str)
                self.logger.info(f"Query analysis: {analysis.get('reasoning', 'No reasoning provided')}")
                return analysis
            else:
                self.logger.warning("No JSON found in LLM response")
                return self._fallback_query_analysis(query)
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze query with LLM: {e}")
            return self._fallback_query_analysis(query)

    def _fallback_query_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback query analysis using simple heuristics"""
        query_lower = query.lower()
        
        # Determine query type
        if any(term in query_lower for term in ['rate', 'price', 'cost', 'fee', 'charge']):
            query_type = "specific_rate"
        elif any(term in query_lower for term in ['contact', 'phone', 'call', 'email', 'address']):
            query_type = "contact_info"
        elif any(term in query_lower for term in ['pay', 'payment', 'bill', 'assistance', 'help pay', "can't pay"]):
            query_type = "payment_assistance"
        elif any(term in query_lower for term in ['how to', 'process', 'procedure', 'steps']):
            query_type = "process_help"
        else:
            query_type = "general_info"
        
        # Extract key terms
        import re
        words = re.findall(r'\b\w+\b', query_lower)
        key_concepts = [word for word in words if len(word) > 3 and word not in ['what', 'when', 'where', 'how', 'does', 'have', 'they', 'this', 'that']]
        
        return {
            "query_type": query_type,
            "key_concepts": key_concepts[:3],
            "search_terms": key_concepts[:5],
            "alternative_phrasings": [],
            "specificity_level": "medium",
            "context_needed": "current",
            "reasoning": "Fallback analysis using keyword matching"
        }

    def enhanced_retrieve_documents(self, query: str, security_level: str = "external", 
                                  top_k: int = 5, min_similarity: float = 0.3, 
                                  high_reasoning: bool = True) -> List[RetrievedDocument]:
        """Enhanced document retrieval with multiple search strategies and result combination"""
        
        # Force clear any cached data
        if hasattr(self, '_cached_results'):
            delattr(self, '_cached_results')
        
        self.logger.info(f"🚀 ENHANCED_RETRIEVE: Starting {'high reasoning' if high_reasoning else 'standard'} search for: '{query}'")
        
        if high_reasoning:
            # Full enhanced search with multiple variants
            analysis = self.analyze_query_intent(query)
            search_queries = self.generate_search_variants(query, analysis)
            self.logger.info(f"🔍 High reasoning: Created {len(search_queries)} search queries: {search_queries}")
        else:
            # Faster search with just 2 queries
            search_queries = self.generate_fast_search_variants(query)
            self.logger.info(f"🔍 Standard search: Created {len(search_queries)} search queries: {search_queries}")
        
        # Collect results from all search variants
        all_results = []
        unique_doc_ids = set()
        
        for i, search_query in enumerate(search_queries, 1):
            self.logger.info(f"🔍 Search {i}/{len(search_queries)}: '{search_query}'")
            
            # Get results for this search variant
            results = self.retrieve_documents(search_query, security_level, top_k*2, min_similarity)
            self.logger.info(f"  → Found {len(results)} documents for this search")
            
            # Add unique results only
            new_results = 0
            for result in results:
                if result.document_id not in unique_doc_ids:
                    all_results.append(result)
                    unique_doc_ids.add(result.document_id)
                    new_results += 1
            
            self.logger.info(f"  → Added {new_results} new unique documents")
        
        self.logger.info(f"📊 Total unique results before processing: {len(all_results)}")
        
        if not all_results:
            self.logger.info("✅ ENHANCED_RETRIEVE: Returning 0 final documents:")
            return []
        
        # Re-rank results based on original query (with error handling)
        self.logger.info("📊 Re-ranking results based on original query...")
        try:
            ranked_results = self.re_rank_by_original_query(all_results, query)
        except Exception as e:
            self.logger.error(f"Re-ranking failed: {e}, using original results")
            ranked_results = all_results
            # Sort by original similarity scores as fallback
            ranked_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Apply enhanced scoring and filtering (only in high reasoning mode)
        if high_reasoning:
            analysis = self.analyze_query_intent(query)
            final_results = self.apply_enhanced_scoring(ranked_results, query, analysis)
        else:
            final_results = ranked_results
        
        # Take top results
        final_results = final_results[:top_k]
        
        self.logger.info(f"📊 After filtering: {len(final_results)} results")
        self.logger.info(f"✅ ENHANCED_RETRIEVE: Returning {len(final_results)} final documents:")
        
        for i, doc in enumerate(final_results, 1):
            self.logger.info(f"  Final {i}: '{doc.title}' (score: {doc.similarity_score:.4f}) - ID: {doc.document_id}")
        
        return final_results

    def re_rank_by_original_query(self, documents: List[RetrievedDocument], original_query: str) -> List[RetrievedDocument]:
        """Re-rank documents by similarity to the original query"""
        if not documents:
            return documents
        
        self.logger.info(f"🔄 Re-ranking {len(documents)} documents by original query")
        
        try:
            # Generate embedding for original query
            original_embedding = self.encode_query(original_query)
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Re-score each document using embeddings table
            for doc in documents:
                try:
                    # Get embedding from embeddings table, not documents table
                    cursor.execute("""
                        SELECT embedding 
                        FROM embeddings 
                        WHERE document_id = ? 
                        ORDER BY id DESC 
                        LIMIT 1
                    """, (doc.document_id,))
                    
                    result = cursor.fetchone()
                    if result and result[0]:
                        doc_embedding = np.frombuffer(result[0], dtype=np.float32)
                        new_similarity = cosine_similarity([original_embedding], [doc_embedding])[0][0]
                        doc.similarity_score = float(new_similarity)
                    else:
                        # If no embedding found, keep original score but log warning
                        self.logger.warning(f"No embedding found for document {doc.document_id}, keeping original score")
                        
                except Exception as e:
                    self.logger.warning(f"Error re-ranking document {doc.document_id}: {e}")
                    # Keep original similarity score if re-ranking fails
                    continue
            
            conn.close()
            
            # Sort by new similarity scores
            documents.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Log top scores for debugging
            top_scores = [f"{doc.similarity_score:.4f}" for doc in documents[:3]]
            self.logger.info(f"✅ Re-ranking complete. Top 3 scores: {top_scores}")
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Error in re-ranking: {e}")
            # Return original documents if re-ranking fails completely
            return documents

    def apply_enhanced_scoring(self, results: List[RetrievedDocument], query: str, analysis: Dict[str, Any]) -> List[RetrievedDocument]:
        """Apply enhanced scoring and filtering based on query analysis"""
        
        query_type = analysis.get('query_type', '')
        
        if not query_type or len(results) <= 3:
            return results  # Don't filter if we have few results
        
        filtered = []
        
        for doc in results:
            content_lower = doc.chunk_text.lower()
            should_include = True
            
            # Filter based on query type
            if query_type == 'specific_rate':
                # For rate queries, prefer documents with pricing info
                if not any(term in content_lower for term in ['rate', 'price', 'cost', 'fee', 'charge', '$', 'kwh']):
                    if len([r for r in filtered if any(term in r.chunk_text.lower() for term in ['rate', 'price', '$'])]) >= 3:
                        should_include = False
            
            elif query_type == 'payment_assistance':
                # For payment assistance, prefer documents with assistance info
                if not any(term in content_lower for term in ['payment', 'assistance', 'help', 'plan', 'program', 'financial']):
                    if len([r for r in filtered if any(term in r.chunk_text.lower() for term in ['payment', 'assistance', 'plan'])]) >= 3:
                        should_include = False
            
            if should_include:
                filtered.append(doc)
        
        return filtered

    def debug_sources_state(self):
        """Debug method to check current sources state"""
        sources = getattr(self, 'last_sources', None)
        self.logger.info(f"🔍 DEBUG - Current sources state:")
        self.logger.info(f"  - Sources exist: {sources is not None}")
        self.logger.info(f"  - Sources count: {len(sources) if sources else 0}")
        if sources:
            for i, source in enumerate(sources[:3]):
                self.logger.info(f"  - Source {i+1}: {source.title}")
        return sources

    def reset_state(self):
        """Explicitly reset all agent state"""
        self.logger.info("🔄 RESET_STATE: Clearing all cached data")
        
        # Clear all possible cached attributes
        attrs_to_clear = [
            'last_sources', '_cached_results', '_last_query', '_last_results', 
            '_query_cache', '_cached_docs', '_last_embeddings'
        ]
        
        for attr in attrs_to_clear:
            if hasattr(self, attr):
                delattr(self, attr)
        
        # Explicitly set last_sources to empty
        self.last_sources = []
        
        self.logger.info("✅ RESET_STATE: Complete")

    def _create_search_queries(self, original_query: str, analysis: Dict[str, Any]) -> List[str]:
        """Create multiple search queries based on analysis"""
        queries = [original_query]  # Always include original
        
        # Add key concepts as search terms
        if analysis.get('key_concepts'):
            queries.append(' '.join(analysis['key_concepts']))
        
        # Add search terms
        if analysis.get('search_terms'):
            queries.append(' '.join(analysis['search_terms'][:3]))
        
        # Add alternative phrasings
        for alt in analysis.get('alternative_phrasings', [])[:2]:
            queries.append(alt)
        
        # Query-type specific enhancements
        query_type = analysis.get('query_type', '')
        
        if query_type == 'specific_rate':
            queries.extend([
                f"current {' '.join(analysis.get('key_concepts', []))} schedule",
                f"{' '.join(analysis.get('key_concepts', []))} pricing structure"
            ])
        elif query_type == 'payment_assistance':
            queries.extend([
                "payment plan assistance program",
                "financial help bill payment",
                "low income assistance program"
            ])
        elif query_type == 'contact_info':
            queries.extend([
                "customer service phone number",
                "contact information customer support"
            ])
        
        # Remove duplicates and limit
        unique_queries = list(dict.fromkeys(queries))[:5]
        return unique_queries

    def generate_search_variants(self, original_query: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate multiple search variants based on query analysis"""
        queries = [original_query]  # Always include original
        
        # Add key concepts as search terms
        if analysis.get('key_concepts'):
            queries.append(' '.join(analysis['key_concepts']))
        
        # Add search terms
        if analysis.get('search_terms'):
            queries.append(' '.join(analysis['search_terms'][:3]))
        
        # Add alternative phrasings
        for alt in analysis.get('alternative_phrasings', [])[:2]:
            queries.append(alt)
        
        # Query-type specific enhancements
        query_type = analysis.get('query_type', '')
        
        if query_type == 'specific_rate':
            queries.extend([
                f"current {' '.join(analysis.get('key_concepts', []))} schedule",
                f"{' '.join(analysis.get('key_concepts', []))} pricing structure"
            ])
        elif query_type == 'payment_assistance':
            queries.extend([
                "payment plan assistance program",
                "financial help bill payment",
                "low income assistance program"
            ])
        elif query_type == 'contact_info':
            queries.extend([
                "customer service phone number",
                "contact information customer support"
            ])
        
        # Remove duplicates and limit
        unique_queries = list(dict.fromkeys(queries))[:5]
        return unique_queries

    def generate_fast_search_variants(self, original_query: str) -> List[str]:
        """Generate just 2 search variants for faster processing"""
        queries = [original_query]  # Always include original
        
        # Simple keyword extraction for second query
        query_lower = original_query.lower()
        
        # Extract key terms (simple approach)
        import re
        words = re.findall(r'\b\w+\b', query_lower)
        key_words = [word for word in words if len(word) > 3 and word not in [
            'what', 'when', 'where', 'how', 'does', 'have', 'they', 'this', 'that', 
            'about', 'with', 'from', 'would', 'could', 'should'
        ]]
        
        # Create a second query with just key terms
        if key_words:
            queries.append(' '.join(key_words[:3]))
        
        # Limit to exactly 2 queries
        return queries[:2]

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
    parser.add_argument("--top-k", type=int, default=5, help="Number of sources to retrieve")
    parser.add_argument("--min-similarity", type=float, default=0.3, help="Minimum similarity threshold")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    
    args = parser.parse_args()
    
    # Initialize RAG agent
    agent = RAGAgent()
    
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
            
            response = agent.query(
                query, 
                user_context=user_context,
                top_k=args.top_k,
                min_similarity=args.min_similarity
            )
            
            print("\n" + format_response(response))
            print("\n" + "="*80 + "\n")
    
    elif args.query:
        query = " ".join(args.query)
        
        user_context = {}
        if args.security_level:
            user_context['user_type'] = 'employee' if args.security_level == 'internal' else 'external'
        
        response = agent.query(
            query,
            user_context=user_context,
            top_k=args.top_k,
            min_similarity=args.min_similarity
        )
        
        print(format_response(response))
    
    else:
        parser.print_help() 