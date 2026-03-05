"""Configuration for the opinion scraper."""

from dataclasses import dataclass, field


@dataclass
class ScraperConfig:
    """Configuration for scraping parameters."""

    search_queries: list[str] = field(default_factory=lambda: ["AI tools"])
    max_results: int = 100
    language: str = "en"
    db_path: str = "opinions.db"

    @classmethod
    def ai_opinions_preset(cls) -> "ScraperConfig":
        """Preset configuration targeting AI tool opinions on Bluesky."""
        return cls(
            search_queries=[
                # # Broad AI tools mentions
                # "AI tools",
                # "AI assistants opinion",
                # "AI opinion",
                # # Named tools — chat & general
                # "ChatGPT",
                # "Claude AI",
                # "Gemini AI",
                # "Copilot AI",
                # "Perplexity AI",
                # "NotebookLM",
                # # Named tools — coding
                # "GitHub Copilot",
                # "Cursor AI",
                # "Codeium",
                # "Tabnine",
                # # Named tools — creative
                # "Midjourney",
                # "Stable Diffusion",
                # "DALL-E",
                # "Suno AI",
                # "Runway AI",
                # # Named tools — productivity
                # "Notion AI",
                # "Grammarly AI",
                # "Jasper AI",
                # Use case opinions
                "AI for coding",
                "AI for writing",
                "AI for research",
                "AI in education",
                "AI at work",
                # Broader sentiment
                "generative AI",
                "AI replacing jobs",
                "AI hype",
                "AI bubble",
                "AI overrated",
                "AI game changer",
                # Trust and ethics
                "AI ethics",
                "AI bias",
                "trust AI",
                # Productivity vs threat
                "AI productivity",
                "AI workflow",
            ],
            max_results=1000,
        )
