import sqlite3


def create_query_history_table():

    conn = sqlite3.connect(
        "logs/query_history.db"
    )

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS query_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        question TEXT,
        search_mode TEXT,
        response_time REAL,
        confidence REAL,
        quality TEXT
    )
    """)

    conn.commit()
    conn.close()


def create_feedback_table():

    conn = sqlite3.connect(
        "logs/query_history.db"
    )

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        feedback TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


create_query_history_table()
create_feedback_table()

print("Database initialized")