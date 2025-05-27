import argparse
import sys
from datetime import datetime
from crawler import WebCrawler
from database import CrawlerDatabase

def main():
    parser = argparse.ArgumentParser(description='Web Crawler for RAG System')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start crawl command
    start_parser = subparsers.add_parser('start', help='Start crawling a website')
    start_parser.add_argument('url', help='Base URL to crawl')
    start_parser.add_argument('--max-pages', type=int, help='Maximum pages to crawl')
    start_parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests (seconds)')
    start_parser.add_argument('--javascript', action='store_true', 
                            help='Enable JavaScript execution (slower but more complete)')
    start_parser.add_argument('--debug', action='store_true', 
                            help='Enable debug output')
    
    # Resume crawl command
    resume_parser = subparsers.add_parser('resume', help='Resume crawling a source')
    resume_parser.add_argument('source_id', type=int, help='Source ID to resume')
    resume_parser.add_argument('--max-pages', type=int, help='Maximum additional pages to crawl')
    resume_parser.add_argument('--javascript', action='store_true', 
                             help='Enable JavaScript execution')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show crawl status')
    status_parser.add_argument('source_id', type=int, nargs='?', help='Source ID (optional)')
    
    # List sources command
    list_parser = subparsers.add_parser('list', help='List all sources')
    
    # Stop command (for future use with background processes)
    stop_parser = subparsers.add_parser('stop', help='Stop crawling')
    stop_parser.add_argument('source_id', type=int, help='Source ID to stop')
    
    # View command
    view_parser = subparsers.add_parser('view', help='View extracted content from a document')
    view_parser.add_argument('document_id', type=int, help='Document ID to view')
    view_parser.add_argument('--full', action='store_true', help='Show full content instead of preview')
    
    # Add reset command
    reset_parser = subparsers.add_parser('reset', help='Reset the database (delete all data)')
    reset_parser.add_argument('--confirm', action='store_true', help='Confirm the reset')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    db = CrawlerDatabase()
    crawler = None
    
    try:
        if args.command == 'start':
            print(f"Starting crawl of {args.url}")
            if args.javascript:
                print("JavaScript execution enabled (using Playwright)")
            else:
                print("JavaScript execution disabled (using requests)")
            
            crawler = WebCrawler(
                delay=args.delay,
                use_javascript=args.javascript
            )
            
            # Enable debug mode if requested
            if getattr(args, 'debug', False):
                crawler.debug = True
            
            source_id = crawler.start_crawl(args.url)
            print(f"Created source ID: {source_id}")
            
            success = crawler.crawl_source(source_id, max_pages=args.max_pages)
            if success:
                print("Crawl completed successfully")
            else:
                print("Crawl failed or was stopped")
                
            # Show final stats
            stats = crawler.get_stats(source_id)
            print(f"\nFinal Statistics:")
            print(f"Total documents: {stats['total_documents']}")
            print(f"Queue status: {stats['queue_stats']}")
        
        elif args.command == 'resume':
            print(f"Resuming crawl for source ID: {args.source_id}")
            if getattr(args, 'javascript', False):
                print("JavaScript execution enabled")
            
            crawler = WebCrawler(
                use_javascript=getattr(args, 'javascript', False)
            )
            
            success = crawler.crawl_source(args.source_id, max_pages=args.max_pages)
            if success:
                print("Crawl completed successfully")
            else:
                print("Crawl failed or was stopped")
        
        elif args.command == 'status':
            if args.source_id:
                crawler = WebCrawler()  # Just for stats, no JS needed
                stats = crawler.get_stats(args.source_id)
                print(f"Source ID: {args.source_id}")
                print(f"Status: {stats['source_status']}")
                print(f"Total documents: {stats['total_documents']}")
                print(f"Last started: {stats['last_start']}")
                print(f"Last finished: {stats['last_finish']}")
                print(f"Queue status: {stats['queue_stats']}")
            else:
                # Show all sources
                import sqlite3
                conn = sqlite3.connect(db.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, source_type, base_url, status, 
                           last_crawled_start_time, last_crawled_finish_time
                    FROM sources ORDER BY id
                ''')
                sources = cursor.fetchall()
                conn.close()
                
                if sources:
                    print("All Sources:")
                    print("-" * 80)
                    for source in sources:
                        print(f"ID: {source[0]}, Type: {source[1]}, URL: {source[2]}")
                        print(f"  Status: {source[3]}, Last Start: {source[4]}, Last Finish: {source[5]}")
                        print()
                else:
                    print("No sources found")
        
        elif args.command == 'list':
            import sqlite3
            conn = sqlite3.connect(db.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT id, base_url, status FROM sources WHERE source_type = "website"')
            sources = cursor.fetchall()
            conn.close()
            
            if sources:
                print("Website Sources:")
                for source_id, url, status in sources:
                    print(f"  {source_id}: {url} ({status})")
            else:
                print("No website sources found")
        
        elif args.command == 'stop':
            print(f"Stopping crawl for source ID: {args.source_id}")
            # This would be more useful with background processes
            # For now, just update the status
            db.update_source_status(args.source_id, 'stopped')
            print("Status updated to 'stopped'")
        
        elif args.command == 'view':
            import sqlite3
            conn = sqlite3.connect(db.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT url, title, extracted_text, http_status_code, metadata_json
                FROM documents WHERE id = ?
            ''', (args.document_id,))
            doc = cursor.fetchone()
            conn.close()
            
            if doc:
                url, title, text, status, metadata = doc
                print(f"Document ID: {args.document_id}")
                print(f"URL: {url}")
                print(f"Title: {title}")
                print(f"Status: {status}")
                print(f"Content length: {len(text)} characters")
                
                if metadata:
                    import json
                    meta = json.loads(metadata)
                    print(f"Meaningful content: {meta.get('meaningful_content', 'Unknown')}")
                
                print("\nExtracted Text:")
                print("-" * 50)
                if args.full:
                    print(text)
                else:
                    print(text[:1000] + "..." if len(text) > 1000 else text)
            else:
                print(f"Document ID {args.document_id} not found")
        
        elif args.command == 'reset':
            if not args.confirm:
                print("This will delete all crawled data. Use --confirm to proceed.")
                return
            
            import os
            if os.path.exists(db.db_path):
                os.remove(db.db_path)
                print("Database reset successfully")
            else:
                print("Database file not found")
    
    finally:
        # Always clean up crawler resources
        if crawler:
            crawler.close()

if __name__ == '__main__':
    main() 