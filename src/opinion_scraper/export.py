"""Export opinions to CSV and JSON formats."""

import csv
import json
from opinion_scraper.storage import Opinion


FIELDS = [
    "platform", "post_id", "author", "text", "created_at",
    "query", "likes", "reposts", "sentiment_score", "sentiment_label",
    "is_reply", "parent_post_id", "relevance_score", "relevance_label",
    "cleaned_text", "clean_status",
    "subjectivity_label", "polarity_label",
]


FIELDS_CLEAN_TEXT_ONLY = [f for f in FIELDS if f not in ("cleaned_text", "clean_status")]


def _fields_for(clean_text_only: bool) -> list[str]:
    return FIELDS_CLEAN_TEXT_ONLY if clean_text_only else FIELDS


class OpinionExporter:
    """Export opinions to file formats."""

    def to_csv(self, opinions: list[Opinion], path: str, clean_text_only: bool = False):
        """Export opinions to a CSV file."""
        fields = _fields_for(clean_text_only)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for o in opinions:
                writer.writerow(self._to_dict(o, clean_text_only))

    def to_json(self, opinions: list[Opinion], path: str, clean_text_only: bool = False):
        """Export opinions to a JSON file."""
        data = [self._to_dict(o, clean_text_only) for o in opinions]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def to_jsonl(self, opinions: list[Opinion], path: str, clean_text_only: bool = False):
        """Export opinions to a JSONL (line-delimited JSON) file."""
        with open(path, "w") as f:
            for o in opinions:
                f.write(json.dumps(self._to_dict(o, clean_text_only), default=str) + "\n")

    @staticmethod
    def _to_dict(opinion: Opinion, clean_text_only: bool = False) -> dict:
        d = {
            "platform": opinion.platform,
            "post_id": opinion.post_id,
            "author": opinion.author,
            "text": opinion.cleaned_text or opinion.text if clean_text_only else opinion.text,
            "created_at": opinion.created_at.isoformat(),
            "query": opinion.query,
            "likes": opinion.likes,
            "reposts": opinion.reposts,
            "sentiment_score": opinion.sentiment_score,
            "sentiment_label": opinion.sentiment_label,
            "is_reply": opinion.is_reply,
            "parent_post_id": opinion.parent_post_id,
            "relevance_score": opinion.relevance_score,
            "relevance_label": opinion.relevance_label,
        }
        if not clean_text_only:
            d["cleaned_text"] = opinion.cleaned_text
            d["clean_status"] = opinion.clean_status
        d["subjectivity_label"] = opinion.subjectivity_label
        d["polarity_label"] = opinion.polarity_label
        return d
