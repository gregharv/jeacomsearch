import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import hashlib
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import logging
import re

class EmbeddingGenerator:
    def __init__(self, db_path: str = "crawler.db", model_name: str = "modernbert-base"):
        self.db_path = db_path
        self.model_name = model_name
        self.model = None
        self.chunk_size = 512  # tokens per chunk
        self.chunk_overlap = 50  # overlap between chunks
        self.init_database()
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def init_database(self):
        """Initialize embedding-related tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create embeddings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedding_model TEXT NOT NULL,
                created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                chunk_hash TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents (id),
                UNIQUE(document_id, chunk_index, embedding_model)
            )
        ''')
        
        # Create index for fast similarity search
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_embeddings_document 
            ON embeddings (document_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_embeddings_model 
            ON embeddings (embedding_model)
        ''')
        
        # Add embedding status to documents table if not exists (with error handling)
        try:
            cursor.execute('''
                ALTER TABLE documents 
                ADD COLUMN embedding_status TEXT DEFAULT 'pending'
            ''')
        except sqlite3.OperationalError:
            # Column already exists
            pass
        
        try:
            cursor.execute('''
                ALTER TABLE documents 
                ADD COLUMN embedding_model TEXT
            ''')
        except sqlite3.OperationalError:
            # Column already exists
            pass
        
        try:
            cursor.execute('''
                ALTER TABLE documents 
                ADD COLUMN embedding_date DATETIME
            ''')
        except sqlite3.OperationalError:
            # Column already exists
            pass
        
        conn.commit()
        conn.close()
        
    def load_model(self):
        """Load the ModernBERT model"""
        if self.model is None:
            self.logger.info(f"Loading model: {self.model_name}")
            try:
                # Try to load ModernBERT through sentence-transformers
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                self.logger.warning(f"Could not load {self.model_name}, falling back to all-MiniLM-L6-v2")
                # Fallback to a known working model
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.model_name = 'all-MiniLM-L6-v2'
            
            self.logger.info(f"Model loaded successfully: {self.model_name}")
            
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text before chunking and embedding"""
        if not text:
            return ""
        
        # Remove markdown-style links [text](url) and keep just the text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove standalone URLs (http/https links that aren't in markdown)
        text = re.sub(r'https?://[^\s]+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Clean up HTML entities that might have been missed
        html_entities = {
            '&nbsp;': ' ',
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'",
            '&hellip;': '...',
        }
        for entity, replacement in html_entities.items():
            text = text.replace(entity, replacement)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove lines that are just navigation or UI elements
        lines = text.split('\n')
        filtered_lines = []
        skip_patterns = [
            r'^(home|about|contact|login|search|menu|navigation)$',
            r'^(click here|read more|learn more|see more)$',
            r'^(previous|next|back|forward)$',
            r'^\d+$',  # Just numbers
            r'^[^\w]*$',  # Just punctuation
        ]
        
        for line in lines:
            line = line.strip()
            if len(line) < 3:  # Skip very short lines
                continue
            
            skip_line = False
            for pattern in skip_patterns:
                if re.match(pattern, line.lower()):
                    skip_line = True
                    break
            
            if not skip_line:
                filtered_lines.append(line)
        
        # Rejoin and clean up
        text = ' '.join(filtered_lines)
        text = text.strip()
        
        return text
    
    def chunk_text_memory_efficient(self, text):
        """Memory-efficient text chunking that properly splits large documents"""
        if not text or len(text) < self.chunk_size:
            return [text] if text.strip() else []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position
            end = min(start + self.chunk_size, text_length)
            chunk = text[start:end]
            
            # If not at the end of text, find a good breaking point
            if end < text_length:
                # Look for sentence endings first
                sentence_endings = ['. ', '? ', '! ', '.\n', '?\n', '!\n']
                best_break = -1
                
                for ending in sentence_endings:
                    pos = chunk.rfind(ending)
                    if pos > self.chunk_size * 0.6:  # At least 60% of chunk size
                        best_break = max(best_break, pos + len(ending))
                
                # If no good sentence break, look for paragraph breaks
                if best_break == -1:
                    paragraph_breaks = ['\n\n', '\n']
                    for break_char in paragraph_breaks:
                        pos = chunk.rfind(break_char)
                        if pos > self.chunk_size * 0.6:
                            best_break = max(best_break, pos + len(break_char))
                
                # Fall back to word boundary
                if best_break == -1:
                    pos = chunk.rfind(' ')
                    if pos > self.chunk_size * 0.5:  # At least 50% of chunk size
                        best_break = pos
                
                # Apply the break if found
                if best_break > 0:
                    chunk = text[start:start + best_break].strip()
                else:
                    # No good break found, use the full chunk
                    chunk = chunk.strip()
            else:
                chunk = chunk.strip()
            
            # Add chunk if it has meaningful content
            if chunk and len(chunk) > 20:  # Minimum meaningful chunk size
                chunks.append(chunk)
            
            # Calculate next start position with overlap
            if end >= text_length:
                break
            
            # Move start position
            actual_chunk_length = len(chunk)
            start += max(actual_chunk_length - self.chunk_overlap, self.chunk_size // 2)
        
        return chunks
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if self.model is None:
            self.load_model()
            
        try:
            # Disable sentence-transformers progress bar to avoid clutter
            embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            return None
    
    def get_documents_needing_embeddings(self, security_level: str = None) -> List[Dict]:
        """Get documents that need embeddings generated"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Base query
        query = '''
            SELECT d.id, d.url, d.extracted_text, d.title, s.source_type
            FROM documents d
            JOIN sources s ON d.source_id = s.id
            WHERE d.extracted_text IS NOT NULL 
            AND d.extracted_text != ''
            AND (d.embedding_status IS NULL OR d.embedding_status = 'pending'
                 OR d.embedding_model != ?)
        '''
        
        params = [self.model_name]
        
        # Add security level filter if specified
        if security_level:
            query += ' AND s.metadata_json LIKE ?'
            params.append(f'%"security_level":"{security_level}"%')
        
        cursor.execute(query, params)
        
        documents = []
        for row in cursor.fetchall():
            documents.append({
                'id': row[0],
                'url': row[1],
                'text': row[2],
                'title': row[3],
                'source_type': row[4]
            })
        
        conn.close()
        return documents
    
    def save_embeddings(self, document_id: int, chunks: List[str], embeddings: List[np.ndarray]):
        """Save embeddings to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Clear existing embeddings for this document and model
            cursor.execute('''
                DELETE FROM embeddings 
                WHERE document_id = ? AND embedding_model = ?
            ''', (document_id, self.model_name))
            
            # Insert new embeddings
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                if embedding is not None:
                    chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
                    embedding_blob = embedding.tobytes()
                    
                    cursor.execute('''
                        INSERT INTO embeddings (
                            document_id, chunk_index, chunk_text, embedding,
                            embedding_model, chunk_hash
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    ''', (document_id, i, chunk, embedding_blob, self.model_name, chunk_hash))
            
            # Update document embedding status
            cursor.execute('''
                UPDATE documents 
                SET embedding_status = 'completed',
                    embedding_model = ?,
                    embedding_date = ?
                WHERE id = ?
            ''', (self.model_name, datetime.now().isoformat(), document_id))
            
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error saving embeddings for document {document_id}: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def process_document(self, document):
        """Process a single document with fixed chunking and memory efficiency"""
        try:
            doc_id = document['id']
            text = document['text']
            
            if not text or len(text.strip()) < 50:
                logging.warning(f"Document {doc_id} has insufficient text")
                return False
            
            # Use the fixed chunking algorithm
            chunks = self.chunk_text_memory_efficient(text)
            
            if not chunks:
                logging.warning(f"No chunks generated for document {doc_id}")
                return False
            
            logging.info(f"Generated {len(chunks)} chunks for document {doc_id}")
            
            # Load model only when needed
            self.load_model()
            
            # Process chunks in batches to save memory
            batch_size = 10  # Process 10 chunks at a time
            conn = sqlite3.connect('crawler.db')
            cursor = conn.cursor()
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                
                # Generate embeddings for this batch
                try:
                    embeddings = self.model.encode(batch_chunks, convert_to_numpy=True)
                except Exception as e:
                    logging.error(f"Failed to generate embeddings for document {doc_id}: {e}")
                    return False
                
                # Insert batch into database
                for j, (chunk_text, embedding) in enumerate(zip(batch_chunks, embeddings)):
                    chunk_index = i + j
                    chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()
                    
                    cursor.execute('''
                        INSERT INTO embeddings 
                        (document_id, chunk_index, chunk_text, embedding, embedding_model, created_date, chunk_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        doc_id,
                        chunk_index,
                        chunk_text,
                        embedding.tobytes(),
                        self.model_name,
                        datetime.now().isoformat(),
                        chunk_hash
                    ))
                
                # Clear batch from memory
                del batch_chunks, embeddings
            
            # Update document status
            cursor.execute('''
                UPDATE documents 
                SET embedding_status = 'completed',
                    embedding_date = ?,
                    embedding_model = ?
                WHERE id = ?
            ''', (datetime.now().isoformat(), self.model_name, doc_id))
            
            conn.commit()
            conn.close()
            
            logging.info(f"Successfully processed document {doc_id} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logging.error(f"Error processing document {doc_id}: {e}")
            return False
    
    def mark_embedding_failed(self, document_id: int, error_message: str):
        """Mark document embedding as failed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE documents 
            SET embedding_status = 'failed'
            WHERE id = ?
        ''', (document_id,))
        
        # Log the error
        cursor.execute('''
            INSERT INTO crawl_log (document_id, event_type, message)
            VALUES (?, 'embedding_failed', ?)
        ''', (document_id, error_message))
        
        conn.commit()
        conn.close()
    
    def generate_embeddings_batch(self, security_level: str = None, batch_size: int = 10):
        """Generate embeddings for all documents needing them"""
        documents = self.get_documents_needing_embeddings(security_level)
        
        if not documents:
            self.logger.info("No documents need embedding generation")
            return
        
        self.logger.info(f"Found {len(documents)} documents needing embeddings")
        
        # Load model once for the batch
        self.load_model()
        
        # Process documents with detailed progress
        successful = 0
        failed = 0
        total_chunks = 0
        
        # Main progress bar for documents
        for doc in tqdm(documents, desc="Processing documents", unit="doc"):
            try:
                if self.process_document(doc):
                    successful += 1
                    # Count chunks for successful documents
                    chunks = self.chunk_text_memory_efficient(doc['text'])
                    total_chunks += len(chunks)
                else:
                    failed += 1
            except Exception as e:
                self.logger.error(f"Failed to process document {doc['id']}: {e}")
                failed += 1
        
        self.logger.info(f"Embedding generation complete: {successful} successful ({total_chunks} total chunks), {failed} failed")
        
        # Show failure analysis
        if failed > 0:
            self.logger.info("Checking failure reasons...")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT message, COUNT(*) 
                FROM crawl_log 
                WHERE event_type = 'embedding_failed' 
                GROUP BY message 
                ORDER BY COUNT(*) DESC
            ''')
            failure_reasons = cursor.fetchall()
            conn.close()
            
            if failure_reasons:
                self.logger.info("Top failure reasons:")
                for reason, count in failure_reasons[:5]:
                    self.logger.info(f"  {count}x: {reason}")
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about embeddings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count documents by embedding status
        cursor.execute('''
            SELECT embedding_status, COUNT(*) 
            FROM documents 
            WHERE extracted_text IS NOT NULL AND extracted_text != ''
            GROUP BY embedding_status
        ''')
        status_counts = dict(cursor.fetchall())
        
        # Count total embeddings
        cursor.execute('SELECT COUNT(*) FROM embeddings')
        total_embeddings = cursor.fetchone()[0]
        
        # Count by model
        cursor.execute('''
            SELECT embedding_model, COUNT(*) 
            FROM embeddings 
            GROUP BY embedding_model
        ''')
        model_counts = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'status_counts': status_counts,
            'total_embeddings': total_embeddings,
            'model_counts': model_counts
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for RAG system")
    parser.add_argument("--security-level", choices=["external", "internal", "sensitive"], 
                       help="Generate embeddings for specific security level")
    parser.add_argument("--batch-size", type=int, default=10, 
                       help="Batch size for processing")
    parser.add_argument("--model", default="modernbert-base", 
                       help="Model name to use for embeddings")
    parser.add_argument("--stats", action="store_true", 
                       help="Show embedding statistics")
    
    args = parser.parse_args()
    
    generator = EmbeddingGenerator(model_name=args.model)
    
    if args.stats:
        stats = generator.get_embedding_stats()
        print("Embedding Statistics:")
        print(f"  Status counts: {stats['status_counts']}")
        print(f"  Total embeddings: {stats['total_embeddings']}")
        print(f"  Model counts: {stats['model_counts']}")
    else:
        generator.generate_embeddings_batch(
            security_level=args.security_level,
            batch_size=args.batch_size
        ) 