from qdrant_client import QdrantClient
import sqlite3
import pandas as pd


def get_query_history():

    conn = sqlite3.connect(
        "logs/query_history.db"
    )

    df = pd.read_sql_query(
        "SELECT * FROM query_history",
        conn
    )

    conn.close()

    return df
def get_daily_query_counts():

    conn = sqlite3.connect(
        "logs/query_history.db"
    )

    query = """
    SELECT
        DATE(timestamp) as date,
        COUNT(*) as total_queries
    FROM query_history
    GROUP BY DATE(timestamp)
    ORDER BY DATE(timestamp)
    """

    df = pd.read_sql_query(
        query,
        conn
    )

    conn.close()

    return df

def get_top_questions(limit=10):

    conn = sqlite3.connect(
        "logs/query_history.db"
    )

    query = """
    SELECT
        question,
        COUNT(*) as count
    FROM query_history
    GROUP BY question
    ORDER BY count DESC
    LIMIT ?
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=(limit,)
    )

    conn.close()

    return df

def get_daily_activity():

    conn = sqlite3.connect(
        "logs/query_history.db"
    )

    query = """
    SELECT
        DATE(timestamp) as date,
        COUNT(*) as queries
    FROM query_history
    GROUP BY DATE(timestamp)
    ORDER BY DATE(timestamp)
    """

    df = pd.read_sql_query(
        query,
        conn
    )

    conn.close()

    return df

def get_quality_distribution():

    conn = sqlite3.connect(
        "logs/query_history.db"
    )

    query = """
    SELECT
        quality,
        COUNT(*) as count
    FROM query_history
    GROUP BY quality
    """

    df = pd.read_sql_query(
        query,
        conn
    )

    conn.close()

    return df

def get_total_chunks():

    client = QdrantClient(
        host="localhost",
        port=6333
    )

    info = client.get_collection(
        "business_docs"
    )

    return info.points_count

def save_feedback(question, feedback):

    conn = sqlite3.connect(
        "logs/query_history.db"
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO feedback
        (
            question,
            feedback
        )
        VALUES (?, ?)
        """,
        (
            question,
            feedback
        )
    )

    conn.commit()
    conn.close()
