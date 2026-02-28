"""Abstract base class for social media scrapers."""

import asyncio
import random
from abc import ABC, abstractmethod
from collections.abc import Callable

from opinion_scraper.filter import RuleFilter
from opinion_scraper.storage import Opinion

ProgressCallback = Callable[[int], None]  # called with number of new items


class BaseScraper(ABC):
    """Interface that all platform scrapers must implement."""

    async def _random_delay(self, min_seconds: float = 1.0, max_seconds: float = 3.0):
        """Sleep for a random duration to mimic human-like pacing."""
        delay = random.uniform(min_seconds, max_seconds)
        await asyncio.sleep(delay)

    @abstractmethod
    async def scrape(
        self, query: str, max_results: int = 100, on_progress: ProgressCallback | None = None,
        rule_filter: RuleFilter | None = None,
    ) -> list[Opinion]:
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
