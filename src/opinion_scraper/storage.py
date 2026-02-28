"""SQLite storage layer for scraped opinions."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Opinion:
    """A single scraped opinion post."""

    platform: str
    post_id: str
    author: str
    text: str
    created_at: datetime
    query: str
    likes: int = 0
    reposts: int = 0
    sentiment_score: float | None = None
    sentiment_label: str | None = None


class OpinionStore:
    """SQLite-backed storage for opinions."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS opinions (
                    platform TEXT NOT NULL,
                    post_id TEXT PRIMARY KEY,
                    author TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    query TEXT NOT NULL,
                    likes INTEGER DEFAULT 0,
                    reposts INTEGER DEFAULT 0,
                    sentiment_score REAL,
                    sentiment_label TEXT
                )
            """)

    def save(self, opinion: Opinion):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO opinions
                   (platform, post_id, author, text, created_at, query, likes, reposts)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (opinion.platform, opinion.post_id, opinion.author, opinion.text,
                 opinion.created_at.isoformat(), opinion.query,
                 opinion.likes, opinion.reposts),
            )

    def save_batch(self, opinions: list[Opinion]):
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """INSERT OR IGNORE INTO opinions
                   (platform, post_id, author, text, created_at, query, likes, reposts)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (o.platform, o.post_id, o.author, o.text,
                     o.created_at.isoformat(), o.query, o.likes, o.reposts)
                    for o in opinions
                ],
            )

    def get_all(self) -> list[Opinion]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM opinions").fetchall()
        return [self._row_to_opinion(r) for r in rows]

    def get_unanalyzed(self) -> list[Opinion]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM opinions WHERE sentiment_score IS NULL"
            ).fetchall()
        return [self._row_to_opinion(r) for r in rows]

    def update_sentiment(self, post_id: str, score: float, label: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE opinions SET sentiment_score = ?, sentiment_label = ? WHERE post_id = ?",
                (score, label, post_id),
            )

    def count_by_platform(self) -> dict[str, int]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT platform, COUNT(*) FROM opinions GROUP BY platform"
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    @staticmethod
    def _row_to_opinion(row) -> Opinion:
        return Opinion(
            platform=row[0],
            post_id=row[1],
            author=row[2],
            text=row[3],
            created_at=datetime.fromisoformat(row[4]),
            query=row[5],
            likes=row[6],
            reposts=row[7],
            sentiment_score=row[8],
            sentiment_label=row[9],
        )
