"""Constants for hierarchical classification."""

DEFAULT_BASE_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
DEFAULT_SARCASM_MODEL = DEFAULT_BASE_MODEL

SUBJECTIVITY_LABELS = ["neutral", "opinionated"]
POLARITY_LABELS = ["negative", "positive"]
FINAL_LABELS = ["neutral", "negative", "positive"]
SARCASM_LABELS = ["non-sarcastic", "sarcastic"]

SUBJECTIVITY_COLUMN = "subjectivity_detection"
POLARITY_COLUMN = "polarity_detection"
SARCASM_COLUMN = "sarcasm_detection"
DEFAULT_TEXT_COLUMN = "text"
DEFAULT_FALLBACK_TEXT_COLUMN = "cleaned_text"
