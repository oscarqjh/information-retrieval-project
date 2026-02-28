"""Tests for CSV/JSON export."""

import csv
import json
from datetime import datetime, timezone

import pytest

from opinion_scraper.export import OpinionExporter
from opinion_scraper.storage import Opinion


@pytest.fixture
def sample_opinions():
    return [
        Opinion(
            platform="twitter", post_id="tw_1", author="alice",
            text="AI tools are amazing for productivity",
            created_at=datetime(2026, 1, 10, 8, 0, 0, tzinfo=timezone.utc),
            query="AI tools", likes=25, reposts=3,
            sentiment_score=0.75, sentiment_label="positive",
        ),
        Opinion(
            platform="bluesky", post_id="bsky_2", author="bob.bsky.social",
            text="I find AI tools overhyped and frustrating",
            created_at=datetime(2026, 1, 12, 14, 30, 0, tzinfo=timezone.utc),
            query="AI tools", likes=10, reposts=1,
            sentiment_score=-0.62, sentiment_label="negative",
        ),
        Opinion(
            platform="twitter", post_id="tw_3", author="carol",
            text="AI tools exist, some are useful",
            created_at=datetime(2026, 1, 15, 9, 0, 0, tzinfo=timezone.utc),
            query="AI tools", likes=5, reposts=0,
            sentiment_score=0.0, sentiment_label="neutral",
        ),
    ]


@pytest.fixture
def exporter():
    return OpinionExporter()


def test_to_csv(exporter, sample_opinions, tmp_path):
    path = str(tmp_path / "out.csv")
    exporter.to_csv(sample_opinions, path)

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 3
    assert rows[0]["platform"] == "twitter"
    assert rows[0]["post_id"] == "tw_1"
    assert rows[0]["author"] == "alice"
    assert rows[0]["text"] == "AI tools are amazing for productivity"
    assert rows[0]["sentiment_label"] == "positive"
    assert "sentiment_score" in rows[0]
    assert "created_at" in rows[0]
    assert "likes" in rows[0]
    assert "reposts" in rows[0]


def test_to_json(exporter, sample_opinions, tmp_path):
    path = str(tmp_path / "out.json")
    exporter.to_json(sample_opinions, path)

    with open(path) as f:
        data = json.load(f)

    assert len(data) == 3
    assert data[0]["platform"] == "twitter"
    assert data[0]["post_id"] == "tw_1"
    assert data[0]["author"] == "alice"
    assert data[1]["sentiment_label"] == "negative"
    assert isinstance(data[2]["likes"], int)


def test_to_csv_empty(exporter, tmp_path):
    path = str(tmp_path / "empty.csv")
    exporter.to_csv([], path)

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 0
    assert reader.fieldnames is not None


def test_to_json_empty(exporter, tmp_path):
    path = str(tmp_path / "empty.json")
    exporter.to_json([], path)

    with open(path) as f:
        data = json.load(f)

    assert data == []
