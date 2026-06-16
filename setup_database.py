import sqlite3
from config import DB_PATH


def setup_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        face_embedding BLOB,
        device_id TEXT,
        enrollment_date DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')

    # Auth log table
    c.execute('''CREATE TABLE IF NOT EXISTS auth_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        user_name TEXT,
        face_score REAL,
        device_id TEXT,
        final_decision TEXT
    )''')

    # Migration: add columns if missing (safe, idempotent)
    for column, col_type in [('device_id', 'TEXT'), ('face_score', 'REAL')]:
        try:
            c.execute(f"ALTER TABLE auth_log ADD COLUMN {column} {col_type}")
            print(f"✅ Added {column} column to auth_log")
        except sqlite3.OperationalError:
            print(f"⚠️  {column} column already exists in auth_log")

    conn.commit()
    conn.close()
    print("✅ Database ready:", DB_PATH)


if __name__ == '__main__':
    setup_database()
