"""Tests for configuration management."""

from opinion_scraper.config import ScraperConfig


def test_default_config():
    config = ScraperConfig()
    assert config.search_queries == ["AI tools"]
    assert config.max_results == 100
    assert config.language == "en"
    assert config.db_path == "opinions.db"


def test_custom_config():
    config = ScraperConfig(
        search_queries=["ChatGPT", "Claude AI"],
        max_results=500,
        language="en",
        db_path="custom.db",
    )
    assert config.search_queries == ["ChatGPT", "Claude AI"]
    assert config.max_results == 500
    assert config.db_path == "custom.db"


def test_default_search_queries_for_ai_opinions():
    config = ScraperConfig.ai_opinions_preset()
    assert len(config.search_queries) > 1
    assert any("AI tools" in q for q in config.search_queries)
