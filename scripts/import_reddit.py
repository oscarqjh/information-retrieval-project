"""Import Reddit JSONL data into the opinions SQLite database."""

import hashlib
import json
import sys
from datetime import datetime, timezone

from opinion_scraper.storage import Opinion, OpinionStore


def make_post_id(text: str, parent_title: str) -> str:
    """Generate a deterministic post ID from text + parent title."""
    h = hashlib.md5(f"{parent_title}:{text}".encode()).hexdigest()[:12]
    return f"reddit_{h}"


def import_reddit_jsonl(jsonl_path: str, db_path: str = "opinions.db"):
    store = OpinionStore(db_path)
    opinions = []

    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            text = record["text"]
            subreddit = record.get("subreddit", "unknown")
            parent_title = record.get("parent_title", "")

            opinion = Opinion(
                platform="reddit",
                post_id=make_post_id(text, parent_title),
                author="unknown",
                text=text,
                created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                query=f"r/{subreddit}: {parent_title}",
                likes=0,
                reposts=0,
                is_reply=True,
                parent_post_id=make_post_id(parent_title, subreddit),
            )
            opinions.append(opinion)

    store.save_batch(opinions)
    print(f"Imported {len(opinions)} Reddit comments into {db_path}")


if __name__ == "__main__":
    jsonl_path = sys.argv[1] if len(sys.argv) > 1 else "3cleaned_comment_records.jsonl"
    db_path = sys.argv[2] if len(sys.argv) > 2 else "opinions.db"
    import_reddit_jsonl(jsonl_path, db_path)
