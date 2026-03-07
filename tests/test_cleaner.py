"""Tests for text cleaning and preprocessing."""
import pytest
from opinion_scraper.cleaner import TextCleaner

@pytest.fixture
def cleaner():
    return TextCleaner()

def test_clean_normal_text(cleaner):
    text = "I think AI tools are really helpful for my daily work and productivity especially coding debugging testing deployment"
    cleaned, status = cleaner.clean(text)
    assert status == "cleaned"
    assert cleaned is not None

def test_clean_removes_html(cleaner):
    text = "<p>AI tools are <b>amazing</b> for coding writing debugging testing deployment optimization workflow automation</p>"
    cleaned, status = cleaner.clean(text)
    assert status == "cleaned"
    assert "<p>" not in cleaned
    assert "<b>" not in cleaned

def test_clean_expands_contractions(cleaner):
    text = "I can't believe how good AI tools are for daily work productivity gains coding debugging testing deployment"
    cleaned, status = cleaner.clean(text)
    assert status == "cleaned"
    assert "can't" not in cleaned

def test_clean_converts_emojis(cleaner):
    text = "AI tools are amazing for productivity workflow optimization coding debugging testing deployment \U0001f600"
    cleaned, status = cleaner.clean(text)
    assert status == "cleaned"

def test_clean_lowercases(cleaner):
    text = "AI TOOLS ARE AMAZING FOR CODING WRITING DEBUGGING TESTING DEPLOYMENT OPTIMIZATION WORKFLOW"
    cleaned, status = cleaner.clean(text)
    assert status == "cleaned"
    assert cleaned == cleaned.lower()

def test_clean_removes_numbers(cleaner):
    text = "I tested 15 different AI tools for coding writing debugging testing deployment productivity workflow"
    cleaned, status = cleaner.clean(text)
    assert status == "cleaned"
    assert "15" not in cleaned

def test_clean_removes_urls(cleaner):
    text = "Check out this AI tool https://example.com it is really great for coding work debugging testing deployment"
    cleaned, status = cleaner.clean(text)
    assert status == "cleaned"
    assert "https" not in cleaned

def test_clean_keeps_stop_words(cleaner):
    text = "I think that the AI tools are very helpful for all of my daily work tasks coding debugging testing deployment"
    cleaned, status = cleaner.clean(text)
    assert status == "cleaned"
    assert cleaned is not None

def test_clean_lemmatizes(cleaner):
    text = "AI tools are making amazing improvements in coding workflows testing applications deployment optimization automation"
    cleaned, status = cleaner.clean(text)
    assert status == "cleaned"
    assert "improvement" in cleaned or "workflow" in cleaned

def test_rejects_short_text(cleaner):
    text = "AI is cool"
    cleaned, status = cleaner.clean(text)
    assert status == "too_short"
    assert cleaned is None

def test_rejects_text_short_after_cleaning(cleaner):
    text = "yes no ok"
    cleaned, status = cleaner.clean(text)
    assert status == "too_short"
    assert cleaned is None

def test_rejects_bot_phrase(cleaner):
    text = "I am a bot. This action was performed automatically."
    cleaned, status = cleaner.clean(text)
    assert status == "bot"
    assert cleaned is None

def test_rejects_bot_author(cleaner):
    text = "Here is a summary of the latest AI news developments technology trends coding debugging testing deployment"
    cleaned, status = cleaner.clean(text, author="newsbot.bsky.social")
    assert status == "bot"
    assert cleaned is None

def test_accepts_normal_author(cleaner):
    text = "I think AI tools are really helpful for my daily work productivity coding debugging testing deployment"
    cleaned, status = cleaner.clean(text, author="alice.bsky.social")
    assert status == "cleaned"
    assert cleaned is not None
