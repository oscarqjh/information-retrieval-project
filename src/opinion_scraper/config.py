"""Configuration for the opinion scraper."""

from dataclasses import dataclass, field


@dataclass
class ScraperConfig:
    """Configuration for scraping parameters."""

    search_queries: list[str] = field(default_factory=lambda: ["AI tools"])
    max_results: int = 100
    language: str = "en"
    db_path: str = "opinions.db"
    exclude_retweets: bool = True

    @classmethod
    def ai_opinions_preset(cls) -> "ScraperConfig":
        """Preset configuration targeting AI tool opinions."""
        return cls(
            search_queries=[
                '"AI tools" lang:en -is:retweet',
                '("AI tools" OR "AI assistants") ("I think" OR "my experience" OR "opinion") lang:en -is:retweet',
                '(ChatGPT OR Claude OR Gemini OR Copilot) (love OR hate OR amazing OR terrible) lang:en -is:retweet',
                '"generative AI" (overrated OR underrated OR "game changer") lang:en -is:retweet',
            ],
            max_results=200,
        )
