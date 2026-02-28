"""Tests for SQLite storage layer."""

import sqlite3
from datetime import datetime, timezone

import pytest

from opinion_scraper.storage import OpinionStore, Opinion


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test.db")
    return OpinionStore(db_path)


@pytest.fixture
def sample_opinion():
    return Opinion(
        platform="twitter",
        post_id="123456",
        author="testuser",
        text="I think AI tools are amazing for productivity",
        created_at=datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        query="AI tools",
        likes=42,
        reposts=5,
    )


def test_store_creates_table(store):
    conn = sqlite3.connect(store.db_path)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='opinions'"
    )
    assert cursor.fetchone() is not None
    conn.close()


def test_save_and_retrieve_opinion(store, sample_opinion):
    store.save(sample_opinion)
    results = store.get_all()
    assert len(results) == 1
    assert results[0].post_id == "123456"
    assert results[0].text == "I think AI tools are amazing for productivity"


def test_duplicate_post_id_is_ignored(store, sample_opinion):
    store.save(sample_opinion)
    store.save(sample_opinion)
    results = store.get_all()
    assert len(results) == 1


def test_save_batch(store):
    opinions = [
        Opinion(
            platform="twitter",
            post_id=f"id_{i}",
            author=f"user_{i}",
            text=f"Opinion {i} on AI tools",
            created_at=datetime(2026, 1, i + 1, tzinfo=timezone.utc),
            query="AI tools",
        )
        for i in range(5)
    ]
    store.save_batch(opinions)
    results = store.get_all()
    assert len(results) == 5


def test_get_unanalyzed(store, sample_opinion):
    store.save(sample_opinion)
    unanalyzed = store.get_unanalyzed()
    assert len(unanalyzed) == 1

    store.update_sentiment(sample_opinion.post_id, 0.85, "positive")
    unanalyzed = store.get_unanalyzed()
    assert len(unanalyzed) == 0


def test_count_by_platform(store):
    for i in range(3):
        store.save(Opinion(
            platform="twitter", post_id=f"tw_{i}", author="u",
            text="text", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), query="q",
        ))
    for i in range(2):
        store.save(Opinion(
            platform="bluesky", post_id=f"bs_{i}", author="u",
            text="text", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), query="q",
        ))
    counts = store.count_by_platform()
    assert counts["twitter"] == 3
    assert counts["bluesky"] == 2


def test_save_reply_opinion(store):
    reply = Opinion(
        platform="bluesky",
        post_id="bsky_reply_1",
        author="replier.bsky.social",
        text="I totally agree with this take on AI tools",
        created_at=datetime(2026, 1, 20, tzinfo=timezone.utc),
        query="AI tools",
        is_reply=True,
        parent_post_id="bsky_xyz789",
    )
    store.save(reply)
    results = store.get_all()
    assert len(results) == 1
    assert results[0].is_reply is True
    assert results[0].parent_post_id == "bsky_xyz789"


def test_save_opinion_with_relevance(store):
    opinion = Opinion(
        platform="twitter",
        post_id="tw_rel_1",
        author="user1",
        text="AI tools are great",
        created_at=datetime(2026, 1, 20, tzinfo=timezone.utc),
        query="AI tools",
    )
    store.save(opinion)
    store.update_relevance("tw_rel_1", 0.92, "relevant")
    results = store.get_all()
    assert results[0].relevance_score == 0.92
    assert results[0].relevance_label == "relevant"


def test_get_unfiltered(store):
    for i in range(3):
        store.save(Opinion(
            platform="twitter", post_id=f"uf_{i}", author="u",
            text="text about AI", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            query="q",
        ))
    store.update_relevance("uf_0", 0.95, "relevant")
    unfiltered = store.get_unfiltered()
    assert len(unfiltered) == 2
    assert all(o.relevance_label is None for o in unfiltered)


def test_default_reply_fields(store, sample_opinion):
    store.save(sample_opinion)
    results = store.get_all()
    assert results[0].is_reply is False
    assert results[0].parent_post_id is None
    assert results[0].relevance_score is None
    assert results[0].relevance_label is None
