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
                # Broad AI tools mentions
                '"AI tools" lang:en -is:retweet',
                '("AI tools" OR "AI assistants") ("I think" OR "my experience" OR "opinion") lang:en -is:retweet',
                # Named tools — chat & general
                '(ChatGPT OR Claude OR Gemini OR Copilot) (love OR hate OR amazing OR terrible) lang:en -is:retweet',
                '(Perplexity OR "NotebookLM" OR "Pi AI") ("I use" OR "switched to" OR "better than" OR "worse than") lang:en -is:retweet',
                # Named tools — coding
                '("GitHub Copilot" OR "Cursor AI" OR Codeium OR Tabnine) (helpful OR useless OR "saves time" OR frustrating) lang:en -is:retweet',
                # Named tools — creative
                '(Midjourney OR "Stable Diffusion" OR "DALL-E" OR Suno OR Runway) (impressive OR "kills art" OR creative OR scary) lang:en -is:retweet',
                # Named tools — productivity
                '("Notion AI" OR "Grammarly AI" OR "Otter AI" OR "Jasper AI") (review OR opinion OR recommend) lang:en -is:retweet',
                # Use case opinions
                '("AI for coding" OR "AI for writing" OR "AI for research") ("my experience" OR review OR opinion) lang:en -is:retweet',
                '("AI in education" OR "AI in healthcare" OR "AI at work") (helpful OR dangerous OR promising OR concerning) lang:en -is:retweet',
                # Broader sentiment — generative AI
                '"generative AI" (overrated OR underrated OR "game changer") lang:en -is:retweet',
                '("AI replacing" OR "AI taking") (jobs OR artists OR writers OR developers) lang:en -is:retweet',
                '("AI hype" OR "AI bubble" OR "AI revolution") ("I think" OR "I believe" OR honestly) lang:en -is:retweet',
                # Trust and ethics
                '("trust AI" OR "don\'t trust AI" OR "AI bias" OR "AI ethics") (opinion OR think OR feel) lang:en -is:retweet',
                # Productivity vs threat
                '("AI productivity" OR "AI workflow") (increased OR decreased OR "waste of time" OR "game changer") lang:en -is:retweet',
            ],
            max_results=1000,
        )
