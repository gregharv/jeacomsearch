#!/usr/bin/env python3
"""
JEA RAG Agent Cache Management Script

This script allows you to manage the query cache for the JEA RAG system.
Note: Cache data is stored in the application database, separate from the knowledge database.
"""

import sqlite3
import argparse
import sys
import os
from datetime import datetime

def get_app_db_path():
    """Get application database path"""
    try:
        from rag_agent import get_app_db_path
        return get_app_db_path()
    except ImportError:
        # Fallback for standalone usage
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.db")

def get_cache_stats(app_db_path=None):
    """Get cache statistics"""
    if app_db_path is None:
        app_db_path = get_app_db_path()
    
    try:
        conn = sqlite3.connect(app_db_path)
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='query_cache'")
        if not cursor.fetchone():
            print(f"⚠️  No query_cache table found in {app_db_path}")
            return None
        
        # Total cached queries
        cursor.execute('SELECT COUNT(*) FROM query_cache')
        total_queries = cursor.fetchone()[0]
        
        # Cache by security level
        cursor.execute('''
            SELECT security_level, COUNT(*) 
            FROM query_cache 
            GROUP BY security_level
        ''')
        by_security = dict(cursor.fetchall())
        
        # Most accessed queries
        cursor.execute('''
            SELECT original_query, access_count, timestamp
            FROM query_cache 
            ORDER BY access_count DESC 
            LIMIT 5
        ''')
        top_queries = cursor.fetchall()
        
        # Recent queries
        cursor.execute('''
            SELECT original_query, timestamp, access_count
            FROM query_cache 
            ORDER BY timestamp DESC 
            LIMIT 5
        ''')
        recent_queries = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_queries': total_queries,
            'by_security_level': by_security,
            'top_queries': top_queries,
            'recent_queries': recent_queries
        }
        
    except Exception as e:
        print(f"Error getting cache stats: {e}")
        return None

def clear_cache(app_db_path=None, days_old=None, security_level=None, confirm=True):
    """Clear cache entries"""
    if app_db_path is None:
        app_db_path = get_app_db_path()
    
    try:
        conn = sqlite3.connect(app_db_path)
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='query_cache'")
        if not cursor.fetchone():
            print(f"⚠️  No query_cache table found in {app_db_path}")
            return 0
        
        # Build query based on parameters
        if days_old is not None and security_level is not None:
            query = '''
                DELETE FROM query_cache 
                WHERE timestamp < datetime('now', '-{} days') 
                AND security_level = ?
            '''.format(days_old)
            params = (security_level,)
            description = f"entries older than {days_old} days with security level '{security_level}'"
            clear_feedback = False
        elif days_old is not None:
            query = '''
                DELETE FROM query_cache 
                WHERE timestamp < datetime('now', '-{} days')
            '''.format(days_old)
            params = ()
            description = f"entries older than {days_old} days"
            clear_feedback = False
        elif security_level is not None:
            query = 'DELETE FROM query_cache WHERE security_level = ?'
            params = (security_level,)
            description = f"entries with security level '{security_level}'"
            clear_feedback = False
        else:
            query = 'DELETE FROM query_cache'
            params = ()
            description = "ALL cache entries and user feedback"
            clear_feedback = True
        
        # Get count of entries that would be deleted
        count_query = query.replace('DELETE', 'SELECT COUNT(*)')
        cursor.execute(count_query, params)
        count_to_delete = cursor.fetchone()[0]
        
        # Also count user_feedback entries if we're doing a clear-all
        feedback_count = 0
        if clear_feedback:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_feedback'")
            if cursor.fetchone():
                cursor.execute('SELECT COUNT(*) FROM user_feedback')
                feedback_count = cursor.fetchone()[0]
        
        total_count = count_to_delete + feedback_count
        
        if total_count == 0:
            print(f"No entries found matching criteria: {description}")
            conn.close()
            return 0
        
        if clear_feedback and feedback_count > 0:
            print(f"Found {count_to_delete} cache entries and {feedback_count} user feedback entries")
        else:
            print(f"Found {count_to_delete} cache {description}")
        
        if confirm:
            if clear_feedback:
                response = input(f"Are you sure you want to delete {count_to_delete} cache entries and {feedback_count} user feedback entries? (y/N): ")
            else:
                response = input(f"Are you sure you want to delete {count_to_delete} {description}? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("Operation cancelled.")
                conn.close()
                return 0
        
        # Execute deletion
        cursor.execute(query, params)
        deleted_count = cursor.rowcount
        
        # Also clear user_feedback if doing clear-all
        if clear_feedback and feedback_count > 0:
            cursor.execute('DELETE FROM user_feedback')
            feedback_deleted = cursor.rowcount
            deleted_count += feedback_deleted
        
        conn.commit()
        conn.close()
        
        if clear_feedback:
            print(f"✅ Successfully deleted {count_to_delete} cache entries and {feedback_count} user feedback entries")
        else:
            print(f"✅ Successfully deleted {deleted_count} cache entries")
        return deleted_count
        
    except Exception as e:
        print(f"Error clearing cache: {e}")
        return 0

def list_cache_entries(app_db_path=None, security_level=None, limit=10):
    """List cache entries"""
    if app_db_path is None:
        app_db_path = get_app_db_path()
    
    try:
        conn = sqlite3.connect(app_db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='query_cache'")
        if not cursor.fetchone():
            print(f"⚠️  No query_cache table found in {app_db_path}")
            return
        
        if security_level:
            cursor.execute('''
                SELECT original_query, security_level, timestamp, access_count, confidence
                FROM query_cache 
                WHERE security_level = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (security_level, limit))
        else:
            cursor.execute('''
                SELECT original_query, security_level, timestamp, access_count, confidence
                FROM query_cache 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
        
        entries = cursor.fetchall()
        conn.close()
        
        if not entries:
            print("No cache entries found.")
            return
        
        print(f"\n{'Query':<50} {'Security':<10} {'Timestamp':<20} {'Access':<7} {'Confidence':<10}")
        print("-" * 100)
        
        for query, sec_level, timestamp, access_count, confidence in entries:
            query_short = query[:47] + "..." if len(query) > 50 else query
            print(f"{query_short:<50} {sec_level:<10} {timestamp:<20} {access_count:<7} {confidence:<10.2f}")
        
    except Exception as e:
        print(f"Error listing cache entries: {e}")

def get_cache_analysis(app_db_path=None):
    """Get detailed cache analysis for prompt analysis"""
    if app_db_path is None:
        app_db_path = get_app_db_path()
    
    try:
        conn = sqlite3.connect(app_db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='query_cache'")
        if not cursor.fetchone():
            print(f"⚠️  No query_cache table found in {app_db_path}")
            return None
        
        # Basic stats
        cursor.execute('SELECT COUNT(*) FROM query_cache')
        total_queries = cursor.fetchone()[0]
        
        # Cache hit patterns (most accessed)
        cursor.execute('''
            SELECT original_query, access_count, confidence, security_level, timestamp
            FROM query_cache 
            ORDER BY access_count DESC 
            LIMIT 10
        ''')
        most_accessed = cursor.fetchall()
        
        # Recent cache entries
        cursor.execute('''
            SELECT original_query, confidence, security_level, timestamp, access_count
            FROM query_cache 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''')
        recent_entries = cursor.fetchall()
        
        # Cache by confidence levels
        cursor.execute('''
            SELECT 
                CASE 
                    WHEN confidence >= 0.8 THEN 'High (0.8+)'
                    WHEN confidence >= 0.6 THEN 'Medium (0.6-0.8)'
                    ELSE 'Low (<0.6)'
                END as confidence_range,
                COUNT(*) as count
            FROM query_cache 
            GROUP BY confidence_range
            ORDER BY count DESC
        ''')
        confidence_distribution = cursor.fetchall()
        
        # Average confidence by security level
        cursor.execute('''
            SELECT security_level, AVG(confidence) as avg_confidence, COUNT(*) as count
            FROM query_cache 
            GROUP BY security_level
        ''')
        security_confidence = cursor.fetchall()
        
        # Query patterns (word frequency analysis)
        cursor.execute('SELECT original_query FROM query_cache')
        all_queries = [row[0] for row in cursor.fetchall()]
        
        # Simple word frequency analysis
        word_freq = {}
        for query in all_queries:
            words = query.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Top words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        
        conn.close()
        
        return {
            'total_queries': total_queries,
            'most_accessed': most_accessed,
            'recent_entries': recent_entries,
            'confidence_distribution': confidence_distribution,
            'security_confidence': security_confidence,
            'top_words': top_words,
            'all_queries': all_queries
        }
        
    except Exception as e:
        print(f"Error getting cache analysis: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="JEA RAG Agent Cache Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clear_cache.py --stats                    # Show cache statistics
  python clear_cache.py --list                     # List recent cache entries
  python clear_cache.py --clear-all                # Clear all cache entries
  python clear_cache.py --clear-old 30             # Clear entries older than 30 days
  python clear_cache.py --clear-old 7 --security external  # Clear external entries older than 7 days
  python clear_cache.py --app-db-path /path/to/app.db --stats  # Use custom database path
        """
    )
    
    parser.add_argument("--app-db-path", help="Path to application database file (optional)")
    parser.add_argument("--stats", action="store_true", help="Show cache statistics")
    parser.add_argument("--list", action="store_true", help="List cache entries")
    parser.add_argument("--limit", type=int, default=10, help="Limit for list operation")
    parser.add_argument("--clear-all", action="store_true", help="Clear all cache entries")
    parser.add_argument("--clear-old", type=int, metavar="DAYS", help="Clear entries older than N days")
    parser.add_argument("--security", choices=["external", "internal", "sensitive"], 
                       help="Filter by security level")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompts")
    
    args = parser.parse_args()
    
    # Get app database path
    app_db_path = args.app_db_path if args.app_db_path else get_app_db_path()
    
    # Check if database exists
    if not os.path.exists(app_db_path):
        print(f"Error: Application database file '{app_db_path}' not found.")
        print("Tip: Run the web application first to create the database, or use --app-db-path to specify location.")
        sys.exit(1)
    
    # Show stats
    if args.stats:
        print("📊 Cache Statistics")
        print("=" * 50)
        
        stats = get_cache_stats(app_db_path)
        if stats:
            print(f"Total cached queries: {stats['total_queries']}")
            print(f"By security level: {stats['by_security_level']}")
            
            if stats['top_queries']:
                print(f"\nTop accessed queries:")
                for i, (query, count, timestamp) in enumerate(stats['top_queries'], 1):
                    query_short = query[:60] + "..." if len(query) > 60 else query
                    print(f"  {i}. {query_short} (accessed {count} times)")
            
            if stats['recent_queries']:
                print(f"\nRecent queries:")
                for i, (query, timestamp, count) in enumerate(stats['recent_queries'], 1):
                    query_short = query[:60] + "..." if len(query) > 60 else query
                    print(f"  {i}. {query_short} ({timestamp})")
    
    # List entries
    elif args.list:
        print("📋 Cache Entries")
        print("=" * 50)
        list_cache_entries(app_db_path, args.security, args.limit)
    
    # Clear operations
    elif args.clear_all:
        clear_cache(app_db_path, confirm=not args.force)
    
    elif args.clear_old is not None:
        clear_cache(app_db_path, days_old=args.clear_old, security_level=args.security, confirm=not args.force)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 