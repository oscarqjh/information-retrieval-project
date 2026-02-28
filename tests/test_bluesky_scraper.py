"""Tests for Bluesky scraper using AT Protocol."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from opinion_scraper.filter import RuleFilter
from opinion_scraper.scraper.bluesky import BlueskyScraper
from opinion_scraper.storage import Opinion


@pytest.fixture
def mock_post():
    """Create a mock Bluesky post object."""
    post = MagicMock()
    post.uri = "at://did:plc:abc123/app.bsky.feed.post/xyz789"
    post.author = MagicMock()
    post.author.handle = "testuser.bsky.social"
    post.record = MagicMock()
    post.record.text = "AI tools are genuinely helpful for coding"
    post.record.created_at = "2026-02-01T10:30:00.000Z"
    post.record.langs = ["en"]
    post.like_count = 8
    post.repost_count = 2
    return post


@pytest.fixture
def scraper():
    return BlueskyScraper(handle="test.bsky.social", password="testpass")


def test_platform_name(scraper):
    assert scraper.platform_name == "bluesky"


@pytest.mark.asyncio
async def test_scrape_converts_posts_to_opinions(scraper, mock_post):
    mock_response = MagicMock()
    mock_response.posts = [mock_post]
    mock_response.cursor = None

    with patch.object(scraper, "_client") as mock_client:
        mock_client.app = MagicMock()
        mock_client.app.bsky.feed.search_posts.return_value = mock_response
        results = await scraper.scrape("AI tools", max_results=10)

    assert len(results) == 1
    opinion = results[0]
    assert isinstance(opinion, Opinion)
    assert opinion.platform == "bluesky"
    assert opinion.author == "testuser.bsky.social"
    assert opinion.text == "AI tools are genuinely helpful for coding"


def test_extract_post_id():
    uri = "at://did:plc:abc123/app.bsky.feed.post/xyz789"
    assert BlueskyScraper._extract_post_id(uri) == "bsky_xyz789"


@pytest.mark.asyncio
async def test_scrape_filters_spam_posts(scraper):
    good_post = MagicMock()
    good_post.uri = "at://did:plc:abc123/app.bsky.feed.post/good1"
    good_post.author = MagicMock()
    good_post.author.handle = "legit.bsky.social"
    good_post.record = MagicMock()
    good_post.record.text = "I genuinely think AI tools are helpful for coding"
    good_post.record.created_at = "2026-02-01T10:30:00.000Z"
    good_post.record.langs = ["en"]
    good_post.like_count = 5
    good_post.repost_count = 1

    spam_post = MagicMock()
    spam_post.uri = "at://did:plc:abc123/app.bsky.feed.post/spam1"
    spam_post.author = MagicMock()
    spam_post.author.handle = "spammer.bsky.social"
    spam_post.record = MagicMock()
    spam_post.record.text = "lol"
    spam_post.record.created_at = "2026-02-01T11:00:00.000Z"
    spam_post.record.langs = ["en"]
    spam_post.like_count = 0
    spam_post.repost_count = 0

    mock_response = MagicMock()
    mock_response.posts = [good_post, spam_post]
    mock_response.cursor = None

    rule_filter = RuleFilter()
    with patch.object(scraper, "_client") as mock_client:
        mock_client.app = MagicMock()
        mock_client.app.bsky.feed.search_posts.return_value = mock_response
        results = await scraper.scrape("AI tools", max_results=10, rule_filter=rule_filter)

    assert len(results) == 1
    assert results[0].author == "legit.bsky.social"


@pytest.mark.asyncio
async def test_scrape_replies_traverses_thread(scraper):
    """Test that scrape_replies recursively extracts replies from a thread."""
    # Build a mock thread: root -> reply1 -> nested_reply
    nested_reply = MagicMock()
    nested_reply.post = MagicMock()
    nested_reply.post.uri = "at://did:plc:abc123/app.bsky.feed.post/nested1"
    nested_reply.post.author = MagicMock()
    nested_reply.post.author.handle = "nested.bsky.social"
    nested_reply.post.record = MagicMock()
    nested_reply.post.record.text = "Great follow-up point about AI tools"
    nested_reply.post.record.created_at = "2026-02-01T12:00:00.000Z"
    nested_reply.post.record.langs = ["en"]
    nested_reply.post.like_count = 2
    nested_reply.post.repost_count = 0
    nested_reply.replies = []
    nested_reply.py_type = "app.bsky.feed.defs#threadViewPost"

    reply1 = MagicMock()
    reply1.post = MagicMock()
    reply1.post.uri = "at://did:plc:abc123/app.bsky.feed.post/reply1"
    reply1.post.author = MagicMock()
    reply1.post.author.handle = "replier.bsky.social"
    reply1.post.record = MagicMock()
    reply1.post.record.text = "I agree, AI tools have been very useful"
    reply1.post.record.created_at = "2026-02-01T11:00:00.000Z"
    reply1.post.record.langs = ["en"]
    reply1.post.like_count = 5
    reply1.post.repost_count = 1
    reply1.replies = [nested_reply]
    reply1.py_type = "app.bsky.feed.defs#threadViewPost"

    root_thread = MagicMock()
    root_thread.thread = MagicMock()
    root_thread.thread.replies = [reply1]
    root_thread.thread.py_type = "app.bsky.feed.defs#threadViewPost"

    with patch.object(scraper, "_client") as mock_client:
        mock_client.app = MagicMock()
        mock_client.app.bsky.feed.get_post_thread.return_value = root_thread
        replies = await scraper.scrape_replies(
            post_uri="at://did:plc:abc123/app.bsky.feed.post/root1",
            parent_post_id="bsky_root1",
            query="AI tools",
            depth=6,
        )

    assert len(replies) == 2
    assert all(r.is_reply for r in replies)
    assert replies[0].parent_post_id == "bsky_root1"
    assert replies[0].author == "replier.bsky.social"
    assert replies[1].author == "nested.bsky.social"


@pytest.mark.asyncio
async def test_scrape_replies_skips_blocked_posts(scraper):
    """Test that NotFoundPost and BlockedPost nodes are skipped."""
    blocked = MagicMock()
    blocked.py_type = "app.bsky.feed.defs#blockedPost"
    blocked.replies = None

    not_found = MagicMock()
    not_found.py_type = "app.bsky.feed.defs#notFoundPost"
    not_found.replies = None

    root_thread = MagicMock()
    root_thread.thread = MagicMock()
    root_thread.thread.replies = [blocked, not_found]
    root_thread.thread.py_type = "app.bsky.feed.defs#threadViewPost"

    with patch.object(scraper, "_client") as mock_client:
        mock_client.app = MagicMock()
        mock_client.app.bsky.feed.get_post_thread.return_value = root_thread
        replies = await scraper.scrape_replies(
            post_uri="at://did:plc:abc123/app.bsky.feed.post/root2",
            parent_post_id="bsky_root2",
            query="AI tools",
            depth=6,
        )

    assert len(replies) == 0
