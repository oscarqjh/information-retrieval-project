"""Abstract base class for social media scrapers."""

from abc import ABC, abstractmethod

from opinion_scraper.storage import Opinion


class BaseScraper(ABC):
    """Interface that all platform scrapers must implement."""

    @abstractmethod
    async def scrape(self, query: str, max_results: int = 100) -> list[Opinion]:
        """Scrape posts matching the query.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.

        Returns:
            List of Opinion objects.
        """
        ...

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the platform name (e.g., 'twitter', 'bluesky')."""
        ...
