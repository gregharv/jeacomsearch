import sqlite3
import json
from datetime import datetime
from typing import Optional, Dict, Any
import hashlib
from urllib.parse import urlparse, urlunparse

class CrawlerDatabase:
    def __init__(self, db_path: str = "crawler.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with the required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sources table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,
                base_url TEXT,
                file_path TEXT,
                last_crawled_start_time DATETIME,
                last_crawled_finish_time DATETIME,
                status TEXT DEFAULT 'pending',
                metadata_json TEXT
            )
        ''')
        
        # Create documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                url TEXT UNIQUE NOT NULL,
                content_hash TEXT,
                extracted_text TEXT,
                title TEXT,
                first_crawled_date DATETIME,
                last_crawled_date DATETIME,
                last_modified_date_on_source DATETIME,
                http_status_code INTEGER,
                metadata_json TEXT,
                processing_status TEXT DEFAULT 'pending_embedding',
                FOREIGN KEY (source_id) REFERENCES sources (id)
            )
        ''')
        
        # Create crawl_log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS crawl_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER,
                document_id INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                message TEXT,
                details_json TEXT,
                FOREIGN KEY (source_id) REFERENCES sources (id),
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        ''')
        
        # Create crawl_queue table for managing URLs to crawl
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS crawl_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                url TEXT UNIQUE NOT NULL,
                priority INTEGER DEFAULT 0,
                added_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending',
                FOREIGN KEY (source_id) REFERENCES sources (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_source(self, source_type: str, base_url: str = None, file_path: str = None, 
                   metadata: Dict[str, Any] = None) -> int:
        """Add a new source to crawl"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute('''
            INSERT INTO sources (source_type, base_url, file_path, metadata_json)
            VALUES (?, ?, ?, ?)
        ''', (source_type, base_url, file_path, metadata_json))
        
        source_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return source_id
    
    def update_source_status(self, source_id: int, status: str, 
                           start_time: datetime = None, finish_time: datetime = None):
        """Update source crawling status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        updates = ["status = ?"]
        params = [status]
        
        if start_time:
            updates.append("last_crawled_start_time = ?")
            params.append(start_time.isoformat())
        
        if finish_time:
            updates.append("last_crawled_finish_time = ?")
            params.append(finish_time.isoformat())
        
        params.append(source_id)
        
        cursor.execute(f'''
            UPDATE sources SET {", ".join(updates)} WHERE id = ?
        ''', params)
        
        conn.commit()
        conn.close()
    
    def add_to_queue(self, source_id: int, url: str, priority: int = 0) -> bool:
        """Add URL to crawl queue. Returns True if added, False if already exists."""
        # Normalize the URL before storing
        parsed = urlparse(url)
        path = parsed.path.lower()
        if path != '/' and path.endswith('/'):
            path = path[:-1]
        
        normalized_url = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            parsed.params,
            '', ''  # Remove query and fragment
        ))
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO crawl_queue (source_id, url, priority)
                VALUES (?, ?, ?)
            ''', (source_id, normalized_url, priority))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            # URL already in queue
            conn.close()
            return False
    
    def get_next_queue_item(self, source_id: int) -> Optional[tuple]:
        """Get the next URL to crawl from the queue"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, url FROM crawl_queue 
            WHERE source_id = ? AND status = 'pending'
            ORDER BY priority DESC, id ASC
            LIMIT 1
        ''', (source_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        return result
    
    def mark_queue_item_processing(self, queue_id: int):
        """Mark a queue item as being processed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE crawl_queue 
            SET status = 'processing'
            WHERE id = ?
        ''', (queue_id,))
        
        conn.commit()
        conn.close()
    
    def mark_queue_item_completed(self, queue_id: int):
        """Mark a queue item as completed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE crawl_queue 
            SET status = 'completed'
            WHERE id = ?
        ''', (queue_id,))
        
        conn.commit()
        conn.close()
    
    def get_source_base_url(self, source_id: int) -> str:
        """Get the base URL for a source"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT base_url FROM sources WHERE id = ?', (source_id,))
        result = cursor.fetchone()
        
        conn.close()
        
        return result[0] if result else None
    
    def upsert_document(self, source_id: int, url: str, content_hash: str,
                       extracted_text: str, title: str = None, 
                       http_status_code: int = None, 
                       last_modified: datetime = None,
                       metadata: Dict[str, Any] = None) -> int:
        """Insert or update document"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else None
        last_modified_iso = last_modified.isoformat() if last_modified else None
        
        # Check if document exists
        cursor.execute('SELECT id, content_hash FROM documents WHERE url = ?', (url,))
        existing = cursor.fetchone()
        
        if existing:
            doc_id, existing_hash = existing
            # Update existing document
            cursor.execute('''
                UPDATE documents SET 
                    content_hash = ?, extracted_text = ?, title = ?,
                    last_crawled_date = ?, http_status_code = ?,
                    last_modified_date_on_source = ?, metadata_json = ?,
                    processing_status = 'pending_embedding'
                WHERE id = ?
            ''', (content_hash, extracted_text, title, now, http_status_code,
                  last_modified_iso, metadata_json, doc_id))
        else:
            # Insert new document
            cursor.execute('''
                INSERT INTO documents (
                    source_id, url, content_hash, extracted_text, title,
                    first_crawled_date, last_crawled_date, http_status_code,
                    last_modified_date_on_source, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (source_id, url, content_hash, extracted_text, title,
                  now, now, http_status_code, last_modified_iso, metadata_json))
            doc_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return doc_id
    
    def log_event(self, event_type: str, message: str, source_id: int = None,
                  document_id: int = None, details: Dict[str, Any] = None):
        """Log crawling events"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        details_json = json.dumps(details) if details else None
        
        cursor.execute('''
            INSERT INTO crawl_log (source_id, document_id, event_type, message, details_json)
            VALUES (?, ?, ?, ?, ?)
        ''', (source_id, document_id, event_type, message, details_json))
        
        conn.commit()
        conn.close()
    
    def get_crawl_stats(self, source_id: int) -> Dict[str, Any]:
        """Get crawling statistics for a source"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get document counts
        cursor.execute('''
            SELECT COUNT(*) FROM documents WHERE source_id = ?
        ''', (source_id,))
        total_docs = cursor.fetchone()[0]
        
        # Get queue status
        cursor.execute('''
            SELECT status, COUNT(*) FROM crawl_queue 
            WHERE source_id = ? GROUP BY status
        ''', (source_id,))
        queue_stats = dict(cursor.fetchall())
        
        # Get source info
        cursor.execute('''
            SELECT status, last_crawled_start_time, last_crawled_finish_time
            FROM sources WHERE id = ?
        ''', (source_id,))
        source_info = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_documents': total_docs,
            'queue_stats': queue_stats,
            'source_status': source_info[0] if source_info else None,
            'last_start': source_info[1] if source_info else None,
            'last_finish': source_info[2] if source_info else None
        } 