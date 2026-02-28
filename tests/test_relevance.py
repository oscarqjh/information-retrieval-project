"""Tests for ML-based relevance classification."""

from unittest.mock import patch, MagicMock

import pytest

from opinion_scraper.relevance import RelevanceClassifier


@pytest.fixture
def mock_pipeline():
    """Mock the transformers pipeline to avoid loading real models in tests."""
    with patch("opinion_scraper.relevance.pipeline") as mock_pipe:
        mock_classifier = MagicMock()
        mock_pipe.return_value = mock_classifier
        yield mock_classifier


def test_classify_single_relevant(mock_pipeline):
    mock_pipeline.return_value = {
        "labels": ["personal opinion about AI tools", "unrelated to AI tools", "spam or advertisement"],
        "scores": [0.92, 0.05, 0.03],
    }
    classifier = RelevanceClassifier(model_name="test-model")
    score, label = classifier.classify("AI tools have really helped my workflow")
    assert label == "relevant"
    assert score == pytest.approx(0.92)


def test_classify_single_spam(mock_pipeline):
    mock_pipeline.return_value = {
        "labels": ["spam or advertisement", "unrelated to AI tools", "personal opinion about AI tools"],
        "scores": [0.88, 0.08, 0.04],
    }
    classifier = RelevanceClassifier(model_name="test-model")
    score, label = classifier.classify("Buy now! Click here for free AI tools giveaway")
    assert label == "spam"
    assert score == pytest.approx(0.88)


def test_classify_single_off_topic(mock_pipeline):
    mock_pipeline.return_value = {
        "labels": ["unrelated to AI tools", "personal opinion about AI tools", "spam or advertisement"],
        "scores": [0.85, 0.10, 0.05],
    }
    classifier = RelevanceClassifier(model_name="test-model")
    score, label = classifier.classify("The weather is nice today")
    assert label == "off_topic"
    assert score == pytest.approx(0.85)


def test_classify_batch(mock_pipeline):
    mock_pipeline.return_value = [
        {"labels": ["personal opinion about AI tools", "unrelated to AI tools", "spam or advertisement"],
         "scores": [0.90, 0.06, 0.04]},
        {"labels": ["spam or advertisement", "unrelated to AI tools", "personal opinion about AI tools"],
         "scores": [0.80, 0.12, 0.08]},
    ]
    classifier = RelevanceClassifier(model_name="test-model")
    results = classifier.classify_batch(["Good AI text", "Spam text"])
    assert len(results) == 2
    assert results[0] == (pytest.approx(0.90), "relevant")
    assert results[1] == (pytest.approx(0.80), "spam")
