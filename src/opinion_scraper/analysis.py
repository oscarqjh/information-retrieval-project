"""Sentiment analysis for scraped opinions."""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from opinion_scraper.storage import Opinion


class SentimentAnalyzer:
    """Analyze sentiment of opinion posts using VADER."""

    def __init__(self):
        self._vader = SentimentIntensityAnalyzer()

    def analyze(self, opinion: Opinion) -> Opinion:
        """Analyze sentiment of a single opinion. Returns the opinion with sentiment fields set."""
        scores = self._vader.polarity_scores(opinion.text)
        compound = scores["compound"]

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        opinion.sentiment_score = compound
        opinion.sentiment_label = label
        return opinion

    def analyze_batch(self, opinions: list[Opinion]) -> list[Opinion]:
        """Analyze sentiment for a batch of opinions."""
        return [self.analyze(o) for o in opinions]

    @staticmethod
    def summarize(opinions: list[Opinion]) -> dict:
        """Generate summary statistics for analyzed opinions."""
        analyzed = [o for o in opinions if o.sentiment_score is not None]
        if not analyzed:
            return {"total": 0, "positive": 0, "negative": 0, "neutral": 0, "avg_score": 0.0}

        positive = sum(1 for o in analyzed if o.sentiment_label == "positive")
        negative = sum(1 for o in analyzed if o.sentiment_label == "negative")
        neutral = sum(1 for o in analyzed if o.sentiment_label == "neutral")
        avg_score = sum(o.sentiment_score for o in analyzed) / len(analyzed)

        return {
            "total": len(analyzed),
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "avg_score": round(avg_score, 4),
        }
