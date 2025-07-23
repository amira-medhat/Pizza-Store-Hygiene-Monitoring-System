import sqlite3
import os
import time

def init_db():
    db_path = os.environ.get("DB_PATH", "/app/shared/violations.db")
    print(f"Checking database at {db_path}...")
    
    # Wait for shared volume to be ready
    max_retries = 10
    retries = 0
    while retries < max_retries:
        try:
            # Check if database exists and has the violations table
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='violations'")
                if cursor.fetchone():
                    print("Database already initialized.")
                    conn.close()
                    return
                conn.close()
            
            # Initialize the database
            print(f"Initializing database at {db_path}...")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Apply performance PRAGMA settings
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.execute("PRAGMA cache_size = 10000")
            
            # Create table if not exists
            cursor.execute('''CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                frame_path TEXT,
                labels TEXT,
                boxes TEXT,
                is_violation INTEGER DEFAULT 0,
                is_safe_pickup INTEGER DEFAULT 0
            )''')
            
            # Optional: create indexes for summary queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_violation ON violations(is_violation)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_safe_pickup ON violations(is_safe_pickup)")
            
            # Insert some sample data
            cursor.execute('''
                INSERT INTO violations (timestamp, frame_path, labels, boxes, is_violation, is_safe_pickup)
                VALUES (?, ?, ?, ?, ?, ?)''',
                ("2025-07-23T22:40:00", "", "[]", "[]", 0, 1)
            )
            
            conn.commit()
            conn.close()
            print("Database initialized successfully!")
            return
            
        except sqlite3.OperationalError as e:
            print(f"Database error: {e}. Retrying in 2 seconds...")
            retries += 1
            time.sleep(2)
        except Exception as e:
            print(f"Unexpected error: {e}. Retrying in 2 seconds...")
            retries += 1
            time.sleep(2)
    
    print("Failed to initialize database after multiple attempts.")

if __name__ == "__main__":
    init_db()