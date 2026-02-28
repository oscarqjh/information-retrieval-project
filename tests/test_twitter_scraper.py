"""Tests for Twitter/X scraper using twscrape."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from opinion_scraper.filter import RuleFilter
from opinion_scraper.scraper.twitter import TwitterScraper
from opinion_scraper.storage import Opinion


@pytest.fixture
def mock_tweet():
    """Create a mock tweet object matching twscrape's Tweet model."""
    tweet = MagicMock()
    tweet.id = 123456789
    tweet.user = MagicMock()
    tweet.user.username = "testuser"
    tweet.rawContent = "I think AI tools are changing everything"
    tweet.date = datetime(2026, 2, 1, 10, 30, 0, tzinfo=timezone.utc)
    tweet.likeCount = 15
    tweet.retweetCount = 3
    return tweet


@pytest.fixture
def scraper():
    return TwitterScraper()


def test_platform_name(scraper):
    assert scraper.platform_name == "twitter"


@pytest.mark.asyncio
async def test_scrape_converts_tweets_to_opinions(scraper, mock_tweet):
    with patch.object(scraper, "_api") as mock_api:
        mock_api.search.return_value = AsyncIteratorMock([mock_tweet])
        results = await scraper.scrape("AI tools", max_results=10)

    assert len(results) == 1
    opinion = results[0]
    assert isinstance(opinion, Opinion)
    assert opinion.platform == "twitter"
    assert opinion.post_id == "123456789"
    assert opinion.author == "testuser"
    assert opinion.text == "I think AI tools are changing everything"
    assert opinion.likes == 15
    assert opinion.reposts == 3


@pytest.mark.asyncio
async def test_scrape_respects_max_results(scraper, mock_tweet):
    tweets = [MagicMock(**{
        "id": i, "user.username": f"user{i}",
        "rawContent": f"text {i}",
        "date": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "likeCount": 0, "retweetCount": 0,
    }) for i in range(20)]
    # Set attributes properly for MagicMock
    for i, t in enumerate(tweets):
        t.id = i
        t.user = MagicMock()
        t.user.username = f"user{i}"
        t.rawContent = f"text {i}"
        t.date = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t.likeCount = 0
        t.retweetCount = 0

    with patch.object(scraper, "_api") as mock_api:
        mock_api.search.return_value = AsyncIteratorMock(tweets[:5])
        results = await scraper.scrape("AI tools", max_results=5)

    assert len(results) == 5


@pytest.mark.asyncio
async def test_scrape_filters_spam_tweets(scraper):
    good_tweet = MagicMock()
    good_tweet.id = 111
    good_tweet.user = MagicMock()
    good_tweet.user.username = "legit_user"
    good_tweet.rawContent = "AI tools have genuinely improved my productivity at work"
    good_tweet.date = datetime(2026, 2, 1, tzinfo=timezone.utc)
    good_tweet.likeCount = 10
    good_tweet.retweetCount = 2
    good_tweet.lang = "en"

    spam_tweet = MagicMock()
    spam_tweet.id = 222
    spam_tweet.user = MagicMock()
    spam_tweet.user.username = "spammer"
    spam_tweet.rawContent = "Buy now the best AI tools! Click here for free giveaway"
    spam_tweet.date = datetime(2026, 2, 1, tzinfo=timezone.utc)
    spam_tweet.likeCount = 0
    spam_tweet.retweetCount = 0
    spam_tweet.lang = "en"

    rule_filter = RuleFilter()
    with patch.object(scraper, "_api") as mock_api:
        mock_api.search.return_value = AsyncIteratorMock([good_tweet, spam_tweet])
        results = await scraper.scrape("AI tools", max_results=10, rule_filter=rule_filter)

    assert len(results) == 1
    assert results[0].author == "legit_user"


@pytest.mark.asyncio
async def test_scrape_replies_uses_conversation_id(scraper):
    """Test that scrape_replies searches by conversation_id."""
    reply1 = MagicMock()
    reply1.id = 999001
    reply1.user = MagicMock()
    reply1.user.username = "replier1"
    reply1.rawContent = "Great take on AI tools, I totally agree with this"
    reply1.date = datetime(2026, 2, 1, 11, 0, 0, tzinfo=timezone.utc)
    reply1.likeCount = 3
    reply1.retweetCount = 0
    reply1.lang = "en"
    reply1.inReplyToTweetId = 123456789

    reply2 = MagicMock()
    reply2.id = 999002
    reply2.user = MagicMock()
    reply2.user.username = "replier2"
    reply2.rawContent = "I disagree, AI tools are overhyped in my opinion"
    reply2.date = datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    reply2.likeCount = 1
    reply2.retweetCount = 0
    reply2.lang = "en"
    reply2.inReplyToTweetId = 123456789

    with patch.object(scraper, "_api") as mock_api:
        mock_api.search.return_value = AsyncIteratorMock([reply1, reply2])
        replies = await scraper.scrape_replies(
            tweet_id=123456789,
            query="AI tools",
            max_replies=50,
        )

    # Verify correct search query was used
    mock_api.search.assert_called_once_with(
        "conversation_id:123456789 is:reply", limit=50,
    )
    assert len(replies) == 2
    assert all(r.is_reply for r in replies)
    assert all(r.parent_post_id == "123456789" for r in replies)
    assert replies[0].author == "replier1"
    assert replies[1].author == "replier2"


class AsyncIteratorMock:
    """Helper to mock async iterators."""

    def __init__(self, items):
        self.items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.items)
        except StopIteration:
            raise StopAsyncIteration
