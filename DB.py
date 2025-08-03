import sqlite3
import os

DB_FOLDER = "D:\\LangChain\\chromadb"
DB_FILE = os.path.join(DB_FOLDER, "pdf_metadata.db")

os.makedirs(DB_FOLDER, exist_ok=True)

def migrate_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE pdf_files ADD COLUMN file_type TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE pdf_files ADD COLUMN session_id TEXT")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

migrate_db()  # Run migration
