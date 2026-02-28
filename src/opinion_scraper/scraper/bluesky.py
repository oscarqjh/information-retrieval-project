"""Bluesky scraper using the AT Protocol SDK."""

from datetime import datetime, timezone

from atproto import Client
from atproto_client.request import Request

from opinion_scraper.filter import RuleFilter
from opinion_scraper.scraper.base import BaseScraper
from opinion_scraper.storage import Opinion


class BlueskyScraper(BaseScraper):
    """Scrapes Bluesky using the AT Protocol API."""

    def __init__(self, handle: str, password: str, timeout: int = 30):
        self._handle = handle
        self._password = password
        self._client = Client(request=Request(timeout=timeout))
        self._logged_in = False

    @property
    def platform_name(self) -> str:
        return "bluesky"

    def _ensure_login(self):
        if not self._logged_in:
            self._client.login(self._handle, self._password)
            self._logged_in = True

    async def scrape(self, query: str, max_results: int = 100, on_progress=None, rule_filter: RuleFilter | None = None) -> list[Opinion]:
        """Scrape Bluesky posts matching the query."""
        self._ensure_login()
        opinions = []
        cursor = None
        is_first_page = True

        while len(opinions) < max_results:
            if not is_first_page:
                await self._random_delay()
            is_first_page = False

            limit = min(25, max_results - len(opinions))
            params = {"q": query, "limit": limit}
            if cursor:
                params["cursor"] = cursor

            response = self._client.app.bsky.feed.search_posts(params)

            batch_count = 0
            for post in response.posts:
                if rule_filter:
                    lang = None
                    if hasattr(post.record, "langs") and post.record.langs:
                        lang = post.record.langs[0]
                    if not rule_filter.is_acceptable(post.record.text, lang=lang):
                        continue
                opinions.append(self._post_to_opinion(post, query))
                batch_count += 1
            if on_progress and batch_count:
                on_progress(batch_count)

            cursor = response.cursor
            if not cursor or not response.posts:
                break

        return opinions[:max_results]

    def _post_to_opinion(self, post, query: str) -> Opinion:
        return Opinion(
            platform="bluesky",
            post_id=self._extract_post_id(post.uri),
            author=post.author.handle,
            text=post.record.text,
            created_at=datetime.fromisoformat(
                post.record.created_at.replace("Z", "+00:00")
            ),
            query=query,
            likes=getattr(post, "like_count", 0) or 0,
            reposts=getattr(post, "repost_count", 0) or 0,
        )

    async def scrape_replies(
        self, post_uri: str, parent_post_id: str, query: str,
        depth: int = 6, rule_filter: RuleFilter | None = None,
    ) -> list[Opinion]:
        """Fetch reply thread for a post and return replies as Opinion objects."""
        self._ensure_login()
        response = self._client.app.bsky.feed.get_post_thread(
            {"uri": post_uri, "depth": depth}
        )
        replies: list[Opinion] = []
        self._traverse_replies(response.thread, parent_post_id, query, replies, rule_filter)
        return replies

    def _traverse_replies(
        self, node, parent_post_id: str, query: str,
        results: list[Opinion], rule_filter: RuleFilter | None = None,
    ):
        """Recursively extract replies from a ThreadViewPost tree."""
        if not hasattr(node, "replies") or not node.replies:
            return
        for reply_node in node.replies:
            if getattr(reply_node, "py_type", "") != "app.bsky.feed.defs#threadViewPost":
                continue
            post = reply_node.post
            if rule_filter:
                lang = None
                if hasattr(post.record, "langs") and post.record.langs:
                    lang = post.record.langs[0]
                if not rule_filter.is_acceptable(post.record.text, lang=lang):
                    continue
            reply_opinion = Opinion(
                platform="bluesky",
                post_id=self._extract_post_id(post.uri),
                author=post.author.handle,
                text=post.record.text,
                created_at=datetime.fromisoformat(
                    post.record.created_at.replace("Z", "+00:00")
                ),
                query=query,
                likes=getattr(post, "like_count", 0) or 0,
                reposts=getattr(post, "repost_count", 0) or 0,
                is_reply=True,
                parent_post_id=parent_post_id,
            )
            results.append(reply_opinion)
            # Recurse into nested replies
            reply_id = self._extract_post_id(post.uri)
            self._traverse_replies(reply_node, reply_id, query, results, rule_filter)

    @staticmethod
    def _extract_post_id(uri: str) -> str:
        """Extract a unique post ID from an AT Protocol URI."""
        rkey = uri.split("/")[-1]
        return f"bsky_{rkey}"
