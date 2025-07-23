import sqlite3
from datetime import datetime

# ────────────────────────────────────────────────
# Initialize Database: DROP + Create + PRAGMAs
# ────────────────────────────────────────────────
def init_db(path):
    print("Initializing database (testing mode with DROP)...")
    conn = sqlite3.connect(path, check_same_thread=False)
    cursor = conn.cursor()

    # Apply performance PRAGMA settings
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.execute("PRAGMA cache_size = 10000")

    # Drop existing table (for testing only)
    cursor.execute("DROP TABLE IF EXISTS violations")

    # Recreate table
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

    conn.commit()
    return conn, cursor  # Return for reuse


# ────────────────────────────────────────────────
# Save Violation Record (No commit inside)
# ────────────────────────────────────────────────
def save_violation(timestamp, path, labels, boxes, is_violation, is_safe_pickup, db_path):
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO violations (timestamp, frame_path, labels, boxes, is_violation, is_safe_pickup)
        VALUES (?, ?, ?, ?, ?, ?)''',
        (timestamp, path, str(labels), str(boxes), int(is_violation), int(is_safe_pickup))
    )

    conn.commit()
    conn.close()



# ────────────────────────────────────────────────
# Commit Wrapper (Use in your main loop every N writes)
# ────────────────────────────────────────────────
def commit_changes(conn):
    conn.commit()
