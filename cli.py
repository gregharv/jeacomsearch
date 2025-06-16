import argparse
import sys
from datetime import datetime
from crawler import WebCrawler
from database import CrawlerDatabase
import click
from rag_agent import RAGAgent, format_response, get_knowledge_db_path, get_app_db_path

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
    start_parser.add_argument('--disable-ssl-verify', action='store_true',
                            help='Disable SSL certificate verification (use only for trusted sites with cert issues)')
    
    # Resume crawl command
    resume_parser = subparsers.add_parser('resume', help='Resume crawling a source')
    resume_parser.add_argument('source_id', type=int, help='Source ID to resume')
    resume_parser.add_argument('--max-pages', type=int, help='Maximum additional pages to crawl')
    resume_parser.add_argument('--javascript', action='store_true', 
                             help='Enable JavaScript execution')
    resume_parser.add_argument('--disable-ssl-verify', action='store_true',
                             help='Disable SSL certificate verification (use only for trusted sites with cert issues)')
    
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
    reset_parser = subparsers.add_parser('reset', help='Reset the databases (delete all data)')
    reset_parser.add_argument('--confirm', action='store_true', help='Confirm the reset')
    reset_parser.add_argument('--knowledge-only', action='store_true', help='Reset only knowledge database')
    reset_parser.add_argument('--app-only', action='store_true', help='Reset only application database')
    
    # Add embeddings command
    embeddings_parser = subparsers.add_parser('generate-embeddings', help='Generate embeddings for documents')
    embeddings_parser.add_argument('--security-level', choices=['external', 'internal', 'sensitive'],
                                  help='Generate embeddings for specific security level')
    embeddings_parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing')
    embeddings_parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Model name for embeddings')
    
    # Add embedding stats command
    stats_parser = subparsers.add_parser('embedding-stats', help='Show embedding statistics')
    
    # Add RAG query command
    query_parser = subparsers.add_parser('rag-query', help='Query the RAG agent')
    query_parser.add_argument('query', nargs='*', help='Query text')
    query_parser.add_argument('--security-level', choices=['external', 'internal', 'sensitive'],
                             help='Security level for the query')
    query_parser.add_argument('--top-k', type=int, default=5, help='Number of sources to retrieve')
    query_parser.add_argument('--min-similarity', type=float, default=0.3, help='Minimum similarity threshold')
    query_parser.add_argument('--interactive', action='store_true', help='Start interactive mode')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    db = CrawlerDatabase()  # Will use knowledge database by default
    crawler = None
    
    try:
        if args.command == 'start':
            print(f"Starting crawl of {args.url}")
            if args.javascript:
                print("JavaScript execution enabled (using Playwright)")
            else:
                print("JavaScript execution disabled (using requests)")
            
            # Handle SSL verification if requested
            if getattr(args, 'disable_ssl_verify', False):
                print("⚠️  SSL certificate verification DISABLED (use only for trusted sites)")
            
            crawler = WebCrawler(
                delay=args.delay,
                use_javascript=args.javascript,
                disable_ssl_verify=getattr(args, 'disable_ssl_verify', False)
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
            
            # Handle SSL verification if requested
            if getattr(args, 'disable_ssl_verify', False):
                print("⚠️  SSL certificate verification DISABLED (use only for trusted sites)")
            
            crawler = WebCrawler(
                use_javascript=getattr(args, 'javascript', False),
                disable_ssl_verify=getattr(args, 'disable_ssl_verify', False)
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
                print("This will delete all data. Use --confirm to proceed.")
                return
            
            import os
            
            # Determine which databases to reset
            reset_knowledge = not args.app_only
            reset_app = not args.knowledge_only
            
            if reset_knowledge:
                knowledge_db_path = get_knowledge_db_path()
                if os.path.exists(knowledge_db_path):
                    os.remove(knowledge_db_path)
                    print(f"✅ Knowledge database reset: {knowledge_db_path}")
                else:
                    print(f"⚠️  Knowledge database not found: {knowledge_db_path}")
            
            if reset_app:
                app_db_path = get_app_db_path()
                if os.path.exists(app_db_path):
                    os.remove(app_db_path)
                    print(f"✅ Application database reset: {app_db_path}")
                else:
                    print(f"⚠️  Application database not found: {app_db_path}")
            
            print("Database(s) reset successfully. You can now start fresh crawling and embeddings.")
        
        elif args.command == 'generate-embeddings':
            print("Generating embeddings for documents...")
            from embeddings import EmbeddingGenerator
            
            # Use knowledge database path
            generator = EmbeddingGenerator(knowledge_db_path=get_knowledge_db_path(), model_name=args.model)
            generator.generate_embeddings_batch(
                security_level=args.security_level,
                batch_size=args.batch_size
            )
        
        elif args.command == 'embedding-stats':
            print("Embedding Statistics:")
            from embeddings import EmbeddingGenerator
            
            # Use knowledge database path
            generator = EmbeddingGenerator(knowledge_db_path=get_knowledge_db_path())
            stats = generator.get_embedding_stats()
            
            print(f"  Status counts: {stats['status_counts']}")
            print(f"  Total embeddings: {stats['total_embeddings']}")
            print(f"  Model counts: {stats['model_counts']}")
        
        elif args.command == 'rag-query':
            # Use separated database paths
            agent = RAGAgent(
                knowledge_db_path=get_knowledge_db_path(),
                app_db_path=get_app_db_path()
            )
            
            if args.interactive:
                print("JEA RAG Agent - Interactive Mode")
                print("Type 'quit' to exit\n")
                
                while True:
                    import click
                    query_text = click.prompt("Query", default="", show_default=False)
                    if query_text.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if not query_text:
                        continue
                    
                    user_context = {}
                    if args.security_level:
                        user_context['user_type'] = 'employee' if args.security_level == 'internal' else 'external'
                    
                    # Get complete response from generator
                    full_response = ""
                    for chunk in agent.query_response(
                        query_text,
                        user_context=user_context,
                        top_k=args.top_k,
                        min_similarity=args.min_similarity
                    ):
                        full_response += chunk
                    
                    # Create a response object for compatibility
                    from rag_agent import RAGResponse
                    response = RAGResponse(
                        answer=full_response,
                        sources=agent.get_last_sources(),
                        reasoning="Generated via CLI",
                        confidence=agent.get_last_confidence(),
                        security_level="external"
                    )
                    
                    print("\n" + format_response(response))
                    print("\n" + "="*80 + "\n")
            
            elif args.query:
                query_text = " ".join(args.query)
                
                user_context = {}
                if args.security_level:
                    user_context['user_type'] = 'employee' if args.security_level == 'internal' else 'external'
                
                # Get complete response from generator
                full_response = ""
                for chunk in agent.query_response(
                    query_text,
                    user_context=user_context,
                    top_k=args.top_k,
                    min_similarity=args.min_similarity
                ):
                    full_response += chunk
                
                # Create a response object for compatibility
                from rag_agent import RAGResponse
                response = RAGResponse(
                    answer=full_response,
                    sources=agent.get_last_sources(),
                    reasoning="Generated via CLI",
                    confidence=agent.get_last_confidence(),
                    security_level="external"
                )
                
                print(format_response(response))
            
            else:
                print("Please provide a query or use --interactive mode")
    
    finally:
        # Always clean up crawler resources
        if crawler:
            crawler.close()

@click.command()
@click.option('--security-level', type=click.Choice(['external', 'internal', 'sensitive']),
              help='Generate embeddings for specific security level')
@click.option('--batch-size', default=10, help='Batch size for processing')
@click.option('--model', default='all-MiniLM-L6-v2', help='Model name for embeddings')
def generate_embeddings(security_level, batch_size, model):
    """Generate embeddings for documents"""
    from embeddings import EmbeddingGenerator
    
    # Use knowledge database path
    generator = EmbeddingGenerator(knowledge_db_path=get_knowledge_db_path(), model_name=model)
    generator.generate_embeddings_batch(security_level=security_level, batch_size=batch_size)

@click.command()
def embedding_stats():
    """Show embedding statistics"""
    from embeddings import EmbeddingGenerator
    
    # Use knowledge database path
    generator = EmbeddingGenerator(knowledge_db_path=get_knowledge_db_path())
    stats = generator.get_embedding_stats()
    
    click.echo("Embedding Statistics:")
    click.echo(f"  Status counts: {stats['status_counts']}")
    click.echo(f"  Total embeddings: {stats['total_embeddings']}")
    click.echo(f"  Model counts: {stats['model_counts']}")

@click.command()
@click.argument('query', nargs=-1)
@click.option('--security-level', type=click.Choice(['external', 'internal', 'sensitive']),
              help='Security level for the query')
@click.option('--top-k', default=5, help='Number of sources to retrieve')
@click.option('--min-similarity', default=0.3, help='Minimum similarity threshold')
@click.option('--interactive', is_flag=True, help='Start interactive mode')
def rag_query(query, security_level, top_k, min_similarity, interactive):
    """Query the RAG agent"""
    # Use separated database paths
    agent = RAGAgent(
        knowledge_db_path=get_knowledge_db_path(),
        app_db_path=get_app_db_path()
    )
    
    if interactive:
        click.echo("JEA RAG Agent - Interactive Mode")
        click.echo("Type 'quit' to exit\n")
        
        while True:
            query_text = click.prompt("Query", default="", show_default=False)
            if query_text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query_text:
                continue
            
            user_context = {}
            if security_level:
                user_context['user_type'] = 'employee' if security_level == 'internal' else 'external'
            
            # Get complete response from generator
            full_response = ""
            for chunk in agent.query_response(
                query_text,
                user_context=user_context,
                top_k=top_k,
                min_similarity=min_similarity
            ):
                full_response += chunk
            
            # Create a response object for compatibility
            from rag_agent import RAGResponse
            response = RAGResponse(
                answer=full_response,
                sources=agent.get_last_sources(),
                reasoning="Generated via CLI",
                confidence=agent.get_last_confidence(),
                security_level="external"
            )
            
            click.echo("\n" + format_response(response))
            click.echo("\n" + "="*80 + "\n")
    
    elif query:
        query_text = " ".join(query)
        
        user_context = {}
        if security_level:
            user_context['user_type'] = 'employee' if security_level == 'internal' else 'external'
        
        # Get complete response from generator
        full_response = ""
        for chunk in agent.query_response(
            query_text,
            user_context=user_context,
            top_k=top_k,
            min_similarity=min_similarity
        ):
            full_response += chunk
        
        # Create a response object for compatibility
        from rag_agent import RAGResponse
        response = RAGResponse(
            answer=full_response,
            sources=agent.get_last_sources(),
            reasoning="Generated via CLI",
            confidence=agent.get_last_confidence(),
            security_level="external"
        )
        
        click.echo(format_response(response))
    
    else:
        click.echo("Please provide a query or use --interactive mode")

if __name__ == '__main__':
    main() 