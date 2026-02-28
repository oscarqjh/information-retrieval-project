"""Tests for rule-based filtering."""

import pytest

from opinion_scraper.filter import RuleFilter


@pytest.fixture
def rule_filter():
    return RuleFilter()


def test_accepts_normal_opinion(rule_filter):
    text = "I think AI tools are really helpful for my daily work"
    assert rule_filter.is_acceptable(text, lang="en") is True
    assert rule_filter.rejection_reason(text, lang="en") is None


def test_rejects_short_text(rule_filter):
    assert rule_filter.is_acceptable("lol", lang="en") is False
    assert rule_filter.rejection_reason("lol", lang="en") == "min_text_length"


def test_rejects_non_english(rule_filter):
    text = "Les outils IA sont formidables pour le travail quotidien"
    assert rule_filter.is_acceptable(text, lang="fr") is False
    assert rule_filter.rejection_reason(text, lang="fr") == "language"


def test_language_none_uses_detection(rule_filter):
    # English text with no lang hint should pass
    text = "AI tools are transforming the way we work every single day"
    assert rule_filter.is_acceptable(text, lang=None) is True


def test_rejects_url_spam(rule_filter):
    text = "Check out https://a.com https://b.com https://c.com https://d.com great AI tools"
    assert rule_filter.is_acceptable(text, lang="en") is False
    assert rule_filter.rejection_reason(text, lang="en") == "url_density"


def test_rejects_hashtag_spam(rule_filter):
    text = "AI tools #ai #ml #tech #coding #dev #chatgpt are great"
    assert rule_filter.is_acceptable(text, lang="en") is False
    assert rule_filter.rejection_reason(text, lang="en") == "hashtag_density"


def test_rejects_blocklist_keywords(rule_filter):
    text = "Buy now the best AI tools at half price! Limited offer click here"
    assert rule_filter.is_acceptable(text, lang="en") is False
    assert rule_filter.rejection_reason(text, lang="en") == "keyword_blocklist"


def test_rejects_near_duplicate(rule_filter):
    text = "AI tools have completely changed my workflow for the better"
    assert rule_filter.is_acceptable(text, lang="en") is True
    # Nearly identical text should be rejected
    duplicate = "AI tools have completely changed my workflow for the better!"
    assert rule_filter.is_acceptable(duplicate, lang="en") is False
    assert rule_filter.rejection_reason(duplicate, lang="en") == "near_duplicate"


def test_reset_clears_duplicate_memory(rule_filter):
    text = "AI tools have completely changed my workflow for the better"
    rule_filter.is_acceptable(text, lang="en")
    rule_filter.reset()
    # After reset, same text should be accepted again
    assert rule_filter.is_acceptable(text, lang="en") is True
