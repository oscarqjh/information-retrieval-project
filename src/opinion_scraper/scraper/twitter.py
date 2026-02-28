"""Twitter/X scraper using twscrape."""

from datetime import datetime, timezone

from twscrape import API

from opinion_scraper.filter import RuleFilter
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

    async def scrape(self, query: str, max_results: int = 100, on_progress=None, rule_filter: RuleFilter | None = None) -> list[Opinion]:
        """Scrape tweets matching the query."""
        opinions = []
        count = 0
        async for tweet in self._api.search(query, limit=max_results):
            if count >= max_results:
                break
            if rule_filter:
                lang = getattr(tweet, "lang", None)
                if not rule_filter.is_acceptable(tweet.rawContent, lang=lang):
                    continue
            opinions.append(self._tweet_to_opinion(tweet, query))
            count += 1
            if on_progress:
                on_progress(1)
            if count % 20 == 0:
                await self._random_delay()
        return opinions

    async def scrape_replies(
        self, tweet_id: int, query: str, max_replies: int = 50,
        rule_filter: RuleFilter | None = None,
    ) -> list[Opinion]:
        """Fetch replies to a tweet using conversation_id search."""
        replies = []
        search_query = f"conversation_id:{tweet_id} is:reply"
        async for tweet in self._api.search(search_query, limit=max_replies):
            if rule_filter:
                lang = getattr(tweet, "lang", None)
                if not rule_filter.is_acceptable(tweet.rawContent, lang=lang):
                    continue
            opinion = self._tweet_to_opinion(tweet, query)
            opinion.is_reply = True
            opinion.parent_post_id = str(tweet_id)
            replies.append(opinion)
        return replies

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
