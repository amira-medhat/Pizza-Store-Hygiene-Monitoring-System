import sqlite3
import os
from database import init_db
from shared.config import DB_PATH

def initialize_database():
    print(f"Initializing database at {DB_PATH}...")
    try:
        # Initialize the database
        init_db(DB_PATH)
        print("Database initialized successfully!")
    except Exception as e:
        print(f"Error initializing database: {e}")

if __name__ == "__main__":
    initialize_database()