"""
tools/memory.py — Memory tool for caching past research queries.

Uses SQLite for persistent storage and embedding similarity to detect
when a new query is similar to a previously researched topic.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import sys

from tools.embeddings import embed_query

logger = logging.getLogger(__name__)

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "memory")
DB_PATH = os.path.join(DB_DIR, "past_searches.db")

# Similarity threshold for considering a query as "already researched"
SIMILARITY_THRESHOLD = 0.80

# Maximum number of entries to keep in memory
MAX_MEMORY_ENTRIES = 50


def _get_connection() -> sqlite3.Connection:
    """Get a SQLite connection, creating the database if needed."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS past_searches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            query_embedding BLOB NOT NULL,
            timestamp TEXT NOT NULL,
            sources_count INTEGER,
            summary_path TEXT,
            summary_json TEXT,
            strategy TEXT
        )
    """)
    conn.commit()
    return conn


def check_memory(query: str) -> Optional[dict]:
    """Check if a similar query has been researched before.

    Args:
        query: The new search query.

    Returns:
        dict with past search info if a similar query exists, None otherwise.
        Dict keys: id, query, timestamp, sources_count, summary_path, similarity
    """
    conn = None
    try:
        conn = _get_connection()
        cursor = conn.execute(
            "SELECT id, query, query_embedding, timestamp, sources_count, summary_path, summary_json "
            "FROM past_searches ORDER BY timestamp DESC"
        )

        query_emb = embed_query(query)
        best_match = None
        best_score = -1.0

        for row in cursor.fetchall():
            row_id, past_query, emb_blob, timestamp, sources_count, summary_path, summary_json = row

            # Deserialize the stored embedding
            past_emb = np.frombuffer(emb_blob, dtype=np.float32)

            # Cosine similarity (both are already normalized)
            similarity = float(np.dot(query_emb, past_emb))

            if similarity > best_score:
                best_score = similarity
                best_match = {
                    "id": row_id,
                    "query": past_query,
                    "timestamp": timestamp,
                    "sources_count": sources_count,
                    "summary_path": summary_path,
                    "similarity": round(similarity, 4),
                }
                # Also try to parse the stored summary
                if summary_json:
                    try:
                        best_match["summary"] = json.loads(summary_json)
                    except json.JSONDecodeError:
                        pass

        if best_match and best_score >= SIMILARITY_THRESHOLD:
            logger.info(
                f"Memory hit: '{query}' matches past query '{best_match['query']}' "
                f"(similarity={best_score:.4f})"
            )
            return best_match
        else:
            logger.info(f"No memory hit for '{query}' (best score: {best_score:.4f})")
            return None

    except Exception as e:
        logger.error(f"Memory check failed: {e}")
        return None
    finally:
        if conn:
            conn.close()


def store_search(
    query: str,
    sources_count: int,
    summary_path: Optional[str],
    summary_dict: Optional[dict],
    strategy: str = "general_web",
) -> None:
    """Store a completed search in memory.

    Args:
        query: The search query.
        sources_count: Number of sources used.
        summary_path: Path to the exported report file.
        summary_dict: The structured summary dict (stored as JSON).
        strategy: The search strategy used.
    """
    conn = None
    try:
        conn = _get_connection()
        query_emb = embed_query(query)
        emb_blob = query_emb.tobytes()

        summary_json = json.dumps(summary_dict) if summary_dict else None

        conn.execute(
            "INSERT INTO past_searches "
            "(query, query_embedding, timestamp, sources_count, summary_path, summary_json, strategy) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                query,
                emb_blob,
                datetime.now().isoformat(),
                sources_count,
                summary_path,
                summary_json,
                strategy,
            ),
        )

        # Prune old entries to prevent unbounded growth
        conn.execute(
            "DELETE FROM past_searches WHERE id NOT IN "
            "(SELECT id FROM past_searches ORDER BY timestamp DESC LIMIT ?)",
            (MAX_MEMORY_ENTRIES,),
        )

        conn.commit()
        logger.info(f"Stored search in memory: '{query}'")

    except Exception as e:
        logger.error(f"Failed to store search in memory: {e}")
    finally:
        if conn:
            conn.close()
