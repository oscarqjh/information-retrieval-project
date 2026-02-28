"""Rule-based filter for dropping obvious spam and irrelevant posts during scrape."""

import hashlib
import re

from langdetect import detect, LangDetectException


# Spam phrases (case-insensitive matching)
DEFAULT_BLOCKLIST = [
    "buy now", "click here", "dm me", "dm for", "free giveaway",
    "check out my", "follow me", "subscribe to", "limited offer",
    "act now", "sign up free", "earn money", "make money",
    "discount code", "promo code", "use code",
]

# Thresholds
MIN_TEXT_LENGTH = 15
MAX_URLS = 3
MAX_HASHTAGS = 5


class RuleFilter:
    """Applies heuristic rules to filter spam and irrelevant posts."""

    def __init__(self, blocklist: list[str] | None = None):
        self._blocklist = [kw.lower() for kw in (blocklist or DEFAULT_BLOCKLIST)]
        self._seen_hashes: dict[str, str] = {}  # hash -> original raw text

    def is_acceptable(self, text: str, lang: str | None = None) -> bool:
        """Returns True if the post passes all rules.

        Unlike :meth:`rejection_reason`, this method records the text hash
        so that future near-identical texts are flagged as duplicates.
        """
        reason = self.rejection_reason(text, lang)
        if reason is None:
            # Record hash -> original text only when the text is accepted
            text_hash = self._fuzzy_hash(text)
            self._seen_hashes[text_hash] = text
        return reason is None

    def rejection_reason(self, text: str, lang: str | None = None) -> str | None:
        """Returns the rule name that rejected the post, or None if accepted."""
        if len(text.strip()) < MIN_TEXT_LENGTH:
            return "min_text_length"

        if not self._check_language(text, lang):
            return "language"

        url_count = len(re.findall(r"https?://\S+", text))
        if url_count > MAX_URLS:
            return "url_density"

        hashtag_count = len(re.findall(r"#\w+", text))
        if hashtag_count > MAX_HASHTAGS:
            return "hashtag_density"

        text_lower = text.lower()
        for keyword in self._blocklist:
            if keyword in text_lower:
                return "keyword_blocklist"

        text_hash = self._fuzzy_hash(text)
        if text_hash in self._seen_hashes and self._seen_hashes[text_hash] != text:
            return "near_duplicate"

        return None

    def reset(self):
        """Clear duplicate detection memory between scraping sessions."""
        self._seen_hashes.clear()

    @staticmethod
    def _check_language(text: str, lang: str | None) -> bool:
        """Check if the text is in English."""
        if lang is not None:
            return lang.startswith("en")
        try:
            detected = detect(text)
            return detected == "en"
        except LangDetectException:
            return True  # Accept if detection fails

    @staticmethod
    def _fuzzy_hash(text: str) -> str:
        """Generate a normalized hash for near-duplicate detection."""
        normalized = re.sub(r"[^\w\s]", "", text.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return hashlib.md5(normalized.encode()).hexdigest()
