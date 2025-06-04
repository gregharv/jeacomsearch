#!/usr/bin/env python3
"""
Test script to verify feedback system functionality
"""

import sqlite3
import os
from datetime import datetime
from streamlit_app import get_db_path, setup_feedback_database

def test_feedback_system():
    """Test the feedback system functionality"""
    print("🧪 Testing Feedback System")
    print("=" * 50)
    
    # Get database path
    db_path = get_db_path()
    print(f"📍 Database path: {db_path}")
    print(f"📂 Database exists: {os.path.exists(db_path)}")
    
    # Setup feedback database
    try:
        setup_feedback_database()
        print("✅ Feedback database setup completed")
    except Exception as e:
        print(f"❌ Error setting up feedback database: {e}")
        return False
    
    # Test database connection
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_feedback'")
        table_exists = cursor.fetchone() is not None
        print(f"📊 user_feedback table exists: {table_exists}")
        
        if not table_exists:
            print("❌ user_feedback table not found!")
            conn.close()
            return False
        
        # Check table structure
        cursor.execute("PRAGMA table_info(user_feedback)")
        columns = cursor.fetchall()
        print(f"📋 Table columns: {[col[1] for col in columns]}")
        
        # Test insert
        test_query = "What are the electric rates?"
        test_response = "The residential electric rates are..."
        test_is_helpful = True
        test_sources_count = 3
        test_confidence = 0.85
        
        cursor.execute('''
            INSERT INTO user_feedback (query, response, is_helpful, sources_count, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
        ''', (test_query, test_response, test_is_helpful, test_sources_count, test_confidence))
        
        row_id = cursor.lastrowid
        print(f"✅ Test feedback inserted with ID: {row_id}")
        
        # Commit and verify
        conn.commit()
        
        # Verify the data was saved
        cursor.execute("SELECT * FROM user_feedback WHERE id = ?", (row_id,))
        saved_data = cursor.fetchone()
        
        if saved_data:
            print(f"✅ Data verified in database:")
            print(f"   ID: {saved_data[0]}")
            print(f"   Query: {saved_data[1][:50]}...")
            print(f"   Is Helpful: {saved_data[3]}")
            print(f"   Sources Count: {saved_data[5]}")
            print(f"   Confidence: {saved_data[6]}")
            print(f"   Timestamp: {saved_data[4]}")
        else:
            print("❌ Data was not saved properly!")
            conn.close()
            return False
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM user_feedback")
        total_count = cursor.fetchone()[0]
        print(f"📊 Total feedback entries: {total_count}")
        
        # Clean up test data
        cursor.execute("DELETE FROM user_feedback WHERE id = ?", (row_id,))
        conn.commit()
        print(f"🧹 Test data cleaned up")
        
        conn.close()
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        if 'conn' in locals():
            conn.close()
        return False

if __name__ == "__main__":
    success = test_feedback_system()
    if success:
        print("\n🎉 Feedback system is working correctly!")
    else:
        print("\n💥 Feedback system has issues that need to be fixed!") 