#!/usr/bin/env python3
"""
Direct database test for feedback system
"""

import sqlite3
from streamlit_app import get_db_path

def test_direct_db():
    """Test direct database connectivity"""
    print("🧪 Testing Direct Database Access")
    print("=" * 40)
    
    db_path = get_db_path()
    print(f"📍 Database: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check initial count
        cursor.execute('SELECT COUNT(*) FROM user_feedback')
        initial_count = cursor.fetchone()[0]
        print(f"📊 Initial feedback count: {initial_count}")
        
        # Insert test feedback
        cursor.execute("""
            INSERT INTO user_feedback (query, response, is_helpful, sources_count, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
        """, ('test query', 'test response', True, 3, 0.8))
        
        conn.commit()
        row_id = cursor.lastrowid
        print(f"✅ Inserted test feedback with ID: {row_id}")
        
        # Verify count increased
        cursor.execute('SELECT COUNT(*) FROM user_feedback')
        new_count = cursor.fetchone()[0]
        print(f"📊 New feedback count: {new_count}")
        
        # Clean up
        cursor.execute('DELETE FROM user_feedback WHERE id = ?', (row_id,))
        conn.commit()
        
        # Final check
        cursor.execute('SELECT COUNT(*) FROM user_feedback')
        final_count = cursor.fetchone()[0]
        print(f"📊 Final feedback count after cleanup: {final_count}")
        
        conn.close()
        
        if new_count == initial_count + 1 and final_count == initial_count:
            print("✅ Database operations working correctly!")
            return True
        else:
            print("❌ Database operations failed!")
            return False
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_direct_db()
    if success:
        print("\n🎉 Database is working - issue must be in Streamlit app logic")
    else:
        print("\n💥 Database has issues!") 