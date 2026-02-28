"""Twitter/X scraper using twscrape."""

from datetime import datetime, timezone

from twscrape import API

from opinion_scraper.scraper.base import BaseScraper
from opinion_scraper.storage import Opinion


class TwitterScraper(BaseScraper):
    """Scrapes Twitter/X using twscrape's GraphQL API."""

    def __init__(self, api: API | None = None):
        self._api = api or API()

    @property
    def platform_name(self) -> str:
        return "twitter"

    async def add_account(self, username: str, password: str, email: str, email_password: str):
        """Add a Twitter account to the pool for scraping."""
        await self._api.pool.add_account(username, password, email, email_password)
        await self._api.pool.login_all()

    async def scrape(self, query: str, max_results: int = 100, on_progress=None) -> list[Opinion]:
        """Scrape tweets matching the query."""
        opinions = []
        count = 0
        async for tweet in self._api.search(query, limit=max_results):
            if count >= max_results:
                break
            opinions.append(self._tweet_to_opinion(tweet, query))
            count += 1
            if on_progress:
                on_progress(1)
            if count % 20 == 0:
                await self._random_delay()
        return opinions

    @staticmethod
    def _tweet_to_opinion(tweet, query: str) -> Opinion:
        return Opinion(
            platform="twitter",
            post_id=str(tweet.id),
            author=tweet.user.username,
            text=tweet.rawContent,
            created_at=tweet.date,
            query=query,
            likes=tweet.likeCount,
            reposts=tweet.retweetCount,
        )
