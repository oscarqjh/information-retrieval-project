"""ML-based relevance classification using zero-shot HuggingFace models."""

from transformers import pipeline


CANDIDATE_LABELS = [
    "personal opinion about AI tools",
    "spam or advertisement",
    "unrelated to AI tools",
]

LABEL_MAP = {
    "personal opinion about AI tools": "relevant",
    "spam or advertisement": "spam",
    "unrelated to AI tools": "off_topic",
}

DEFAULT_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"


class RelevanceClassifier:
    """Zero-shot text classifier for opinion relevance."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: int | str = 0, batch_size: int = 64):
        self._classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device,
            batch_size=batch_size,
        )
        self._batch_size = batch_size

    def classify(self, text: str) -> tuple[float, str]:
        """Classify a single text. Returns (score, label)."""
        result = self._classifier(text, CANDIDATE_LABELS)
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        return top_score, LABEL_MAP[top_label]

    def classify_batch(self, texts: list[str]) -> list[tuple[float, str]]:
        """Classify a batch of texts. Returns list of (score, label) tuples."""
        results = self._classifier(texts, CANDIDATE_LABELS)
        if isinstance(results, dict):
            results = [results]
        return [
            (r["scores"][0], LABEL_MAP[r["labels"][0]])
            for r in results
        ]
