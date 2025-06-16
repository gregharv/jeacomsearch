#!/usr/bin/env python3
"""
Database Initialization Script for JEA RAG System

This script initializes both the knowledge and application databases
with the proper structure for the separated database architecture.
"""

import os
import sys
from pathlib import Path

def main():
    """Initialize both databases"""
    try:
        # Import modules that create databases
        from rag_agent import get_knowledge_db_path, get_app_db_path, RAGAgent
        from database import CrawlerDatabase
        from embeddings import EmbeddingGenerator
        
        print("🗄️  JEA RAG System Database Initialization")
        print("=" * 50)
        
        # Get database paths
        knowledge_db_path = get_knowledge_db_path()
        app_db_path = get_app_db_path()
        
        print(f"📚 Knowledge Database: {knowledge_db_path}")
        print(f"💾 Application Database: {app_db_path}")
        print()
        
        # Initialize knowledge database (documents, embeddings, crawl data)
        print("1. Initializing knowledge database...")
        try:
            # CrawlerDatabase creates knowledge database structure
            crawler_db = CrawlerDatabase(knowledge_db_path)
            print("   ✅ Knowledge database structure created")
            
            # EmbeddingGenerator ensures embedding tables exist
            embedding_gen = EmbeddingGenerator(knowledge_db_path)
            print("   ✅ Embedding tables verified")
            
        except Exception as e:
            print(f"   ❌ Error initializing knowledge database: {e}")
            return False
        
        # Initialize application database (cache, interactions, feedback)
        print("2. Initializing application database...")
        try:
            # RAGAgent creates app database structure
            rag_agent = RAGAgent(
                knowledge_db_path=knowledge_db_path,
                app_db_path=app_db_path
            )
            print("   ✅ Application database structure created")
            print("   ✅ Query cache tables verified")
            
        except Exception as e:
            print(f"   ❌ Error initializing application database: {e}")
            return False
        
        print()
        print("✅ Database initialization complete!")
        print()
        print("Next steps:")
        print("1. Run web crawler: python cli.py start <url>")
        print("2. Generate embeddings: python cli.py generate-embeddings")
        print("3. Start web application: python app.py")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 