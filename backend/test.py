import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_database():
    """Test database connection and basic operations"""
    print("=== Database Connection Test ===")
    
    try:
        # Import database class directly
        from backend.models.database import Database
        
        # Initialize database
        db = Database()
        
        if not db.connection:
            print("❌ Database connection failed")
            return False
        
        if not db.connection.is_connected():
            print("❌ Database not connected")
            return False
        
        print("✅ Database connected successfully")
        
        # Test basic queries
        cursor = db.connection.cursor()
        
        # Test users table
        try:
            cursor.execute("SELECT COUNT(*) FROM users")
            users_count = cursor.fetchone()[0]
            print(f"✅ Users table: {users_count} records")
        except Exception as e:
            print(f"❌ Users table error: {e}")
        
        # Test students table  
        try:
            cursor.execute("SELECT COUNT(*) FROM students")
            students_count = cursor.fetchone()[0]
            print(f"✅ Students table: {students_count} records")
        except Exception as e:
            print(f"❌ Students table error: {e}")
        
        # Test PDF history table
        try:
            cursor.execute("SELECT COUNT(*) FROM pdf_history")
            pdfs_count = cursor.fetchone()[0]
            print(f"✅ PDF history table: {pdfs_count} records")
        except Exception as e:
            print(f"❌ PDF history table error: {e}")
        
        # Test activity logs table
        try:
            cursor.execute("SELECT COUNT(*) FROM activity_logs")
            activities_count = cursor.fetchone()[0]
            print(f"✅ Activity logs table: {activities_count} records")
        except Exception as e:
            print(f"❌ Activity logs table error: {e}")
        
        cursor.close()
        db.close()
        
        print("=== Database test completed ===")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_database()