"""Tests for hierarchical batch annotation."""

import csv

from unittest.mock import MagicMock

from opinion_scraper.classification.annotation import (
    format_annotation_score,
    HierarchicalBatchAnnotator,
    is_missing_annotation_value,
    load_csv_records,
    prepare_csv_records_for_annotation,
    update_csv_records_in_place,
)
from opinion_scraper.classification.inference import HierarchicalPrediction


def test_batch_annotator_collects_predictions_and_stats():
    classifier = MagicMock()
    classifier.predict.side_effect = [
        [
            HierarchicalPrediction(
                text="One",
                stage1_label="opinionated",
                stage1_score=0.9,
                stage2_label="positive",
                stage2_score=0.8,
                final_label="positive",
            ),
            HierarchicalPrediction(
                text="Two",
                stage1_label="neutral",
                stage1_score=0.7,
                stage2_label=None,
                stage2_score=None,
                final_label="neutral",
            ),
        ]
    ]
    annotator = HierarchicalBatchAnnotator(classifier=classifier)

    artifacts = annotator.annotate_records(
        records=[
            {"post_id": "p1", "text": "One"},
            {"post_id": "p2", "text": "Two"},
        ],
        batch_size=2,
    )

    assert len(artifacts.predictions) == 2
    assert artifacts.predictions[0]["final_label"] == "positive"
    assert artifacts.stats.total_records == 2
    assert artifacts.stats.total_batches == 1


def test_csv_annotation_helpers_update_rows_in_place(tmp_path):
    csv_path = tmp_path / "opinions.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["post_id", "text", "cleaned_text", "subjectivity_label"])
        writer.writeheader()
        writer.writerow({"post_id": "p1", "text": "hello", "cleaned_text": "", "subjectivity_label": ""})
        writer.writerow({"post_id": "p2", "text": "world", "cleaned_text": "", "subjectivity_label": "neutral"})

    rows, _ = load_csv_records(csv_path)
    prepared = prepare_csv_records_for_annotation(
        rows=rows,
        target_columns=["subjectivity_label", "subjectivity_score"],
        force=False,
    )
    assert [row["post_id"] for row in prepared] == ["p1", "p2"]

    update_csv_records_in_place(
        path=csv_path,
        updates_by_id={"p1": {"subjectivity_label": "opinionated", "subjectivity_score": "0.900000"}},
        new_columns=["subjectivity_label", "subjectivity_score"],
    )

    rows, fieldnames = load_csv_records(csv_path)
    assert "subjectivity_score" in fieldnames
    assert rows[0]["subjectivity_label"] == "opinionated"
    assert rows[0]["subjectivity_score"] == "0.900000"


def test_prepare_csv_records_treats_nan_placeholders_as_missing():
    rows = [
        {
            "post_id": "p1",
            "text": "hello",
            "cleaned_text": "",
            "subjectivity_label": "neutral",
            "subjectivity_score": "nan",
        },
        {
            "post_id": "p2",
            "text": "world",
            "cleaned_text": "",
            "subjectivity_label": "neutral",
            "subjectivity_score": "0.910000",
        },
    ]

    prepared = prepare_csv_records_for_annotation(
        rows=rows,
        target_columns=["subjectivity_label", "subjectivity_score"],
        force=False,
    )

    assert [row["post_id"] for row in prepared] == ["p1"]
    assert is_missing_annotation_value("nan") is True
    assert is_missing_annotation_value("0.910000") is False


def test_format_annotation_score_drops_non_finite_values():
    assert format_annotation_score(0.8754321) == "0.875432"
    assert format_annotation_score(float("nan")) == ""
    assert format_annotation_score(None) == ""
