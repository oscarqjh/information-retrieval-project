"""Tests for sentiment analysis module."""

from datetime import datetime, timezone

import pytest

from opinion_scraper.analysis import SentimentAnalyzer
from opinion_scraper.storage import Opinion


@pytest.fixture
def analyzer():
    return SentimentAnalyzer()


@pytest.fixture
def positive_opinion():
    return Opinion(
        platform="twitter", post_id="pos1", author="user",
        text="I absolutely love AI tools, they make my work so much easier!",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), query="AI tools",
    )


@pytest.fixture
def negative_opinion():
    return Opinion(
        platform="twitter", post_id="neg1", author="user",
        text="AI tools are terrible and useless, complete waste of time",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), query="AI tools",
    )


@pytest.fixture
def neutral_opinion():
    return Opinion(
        platform="twitter", post_id="neu1", author="user",
        text="AI tools exist and people use them for work",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), query="AI tools",
    )


def test_analyze_positive(analyzer, positive_opinion):
    result = analyzer.analyze(positive_opinion)
    assert result.sentiment_score is not None
    assert result.sentiment_score > 0.05
    assert result.sentiment_label == "positive"


def test_analyze_negative(analyzer, negative_opinion):
    result = analyzer.analyze(negative_opinion)
    assert result.sentiment_score is not None
    assert result.sentiment_score < -0.05
    assert result.sentiment_label == "negative"


def test_analyze_neutral(analyzer, neutral_opinion):
    result = analyzer.analyze(neutral_opinion)
    assert result.sentiment_score is not None
    assert result.sentiment_label == "neutral"


def test_analyze_batch(analyzer, positive_opinion, negative_opinion, neutral_opinion):
    opinions = [positive_opinion, negative_opinion, neutral_opinion]
    results = analyzer.analyze_batch(opinions)
    assert len(results) == 3
    assert all(r.sentiment_score is not None for r in results)


def test_summarize(analyzer):
    opinions = [
        Opinion(
            platform="twitter", post_id=f"id_{i}", author="u",
            text="great" if i < 6 else "bad",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), query="q",
            sentiment_score=0.5 if i < 6 else -0.5,
            sentiment_label="positive" if i < 6 else "negative",
        )
        for i in range(10)
    ]
    summary = analyzer.summarize(opinions)
    assert summary["total"] == 10
    assert summary["positive"] == 6
    assert summary["negative"] == 4
    assert "avg_score" in summary
