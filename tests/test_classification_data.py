"""Tests for hierarchical dataset construction."""

from opinion_scraper.classification.data import (
    InMemoryDataset,
    InMemoryDatasetDict,
    ManualLabelDatasetBuilder,
)


def test_builder_filters_and_splits_hierarchical_labels(monkeypatch):
    records = [
        {
            "post_id": "p1",
            "text": "positive opinion",
            "cleaned_text": "",
            "subjectivity_detection": "opinionated",
            "polarity_detection": "positive",
        },
        {
            "post_id": "p2",
            "text": "negative opinion",
            "cleaned_text": "",
            "subjectivity_detection": "opinionated",
            "polarity_detection": "negative",
        },
        {
            "post_id": "p3",
            "text": "neutral statement",
            "cleaned_text": "",
            "subjectivity_detection": "neutral",
            "polarity_detection": "neutral",
        },
        {
            "post_id": "p4",
            "text": "opinionated but neutral polarity",
            "cleaned_text": "",
            "subjectivity_detection": "opinionated",
            "polarity_detection": "neutral",
        },
        {
            "post_id": "p5",
            "text": "neutral row with sentiment polarity",
            "cleaned_text": "",
            "subjectivity_detection": "neutral",
            "polarity_detection": "positive",
        },
        {
            "post_id": "p6",
            "text": "missing subjectivity",
            "cleaned_text": "",
            "subjectivity_detection": "",
            "polarity_detection": "positive",
        },
        {
            "post_id": "p7",
            "text": "opinionated missing polarity",
            "cleaned_text": "",
            "subjectivity_detection": "opinionated",
            "polarity_detection": "",
        },
        {
            "post_id": "p8",
            "text": "",
            "cleaned_text": "fallback negative opinion",
            "subjectivity_detection": "opinionated",
            "polarity_detection": "negative",
        },
        {
            "post_id": "p9",
            "text": "",
            "cleaned_text": "",
            "subjectivity_detection": "neutral",
            "polarity_detection": "neutral",
        },
    ]

    monkeypatch.setattr(
        "opinion_scraper.classification.data.load_xlsx_rows",
        lambda path, sheet_name=None: records,
    )
    monkeypatch.setattr(
        "opinion_scraper.classification.data.get_dataset_backend",
        lambda: (InMemoryDataset, InMemoryDatasetDict),
    )

    builder = ManualLabelDatasetBuilder(validation_ratio=0.2, seed=42)
    bundle = builder.build("ignored.xlsx")

    assert isinstance(bundle.subjectivity_dataset, InMemoryDatasetDict)
    assert isinstance(bundle.polarity_dataset, InMemoryDatasetDict)

    subjectivity_train = bundle.subjectivity_dataset["train"].to_list()
    subjectivity_validation = bundle.subjectivity_dataset["validation"].to_list()
    polarity_train = bundle.polarity_dataset["train"].to_list()
    polarity_validation = bundle.polarity_dataset["validation"].to_list()

    assert bundle.stage_counts["subjectivity"]["total"] == 7
    assert bundle.stage_counts["polarity"]["total"] == 3
    assert len(subjectivity_train) + len(subjectivity_validation) == 7
    assert len(polarity_train) + len(polarity_validation) == 3

    assert {row["label_name"] for row in subjectivity_train + subjectivity_validation} == {
        "neutral",
        "opinionated",
    }
    assert {row["label_name"] for row in polarity_train + polarity_validation} == {
        "positive",
        "negative",
    }
    assert any(row["text"] == "fallback negative opinion" for row in polarity_train + polarity_validation)

    assert bundle.anomaly_stats == {
        "missing_text": 1,
        "missing_subjectivity_label": 1,
        "missing_polarity_label": 1,
        "opinionated_with_neutral_polarity": 1,
        "neutral_with_sentiment_polarity": 1,
    }


def test_builder_creates_shared_ablation_split(monkeypatch):
    records = [
        {
            "post_id": "n1",
            "text": "neutral statement one",
            "cleaned_text": "",
            "subjectivity_detection": "neutral",
            "polarity_detection": "neutral",
        },
        {
            "post_id": "n2",
            "text": "neutral statement two",
            "cleaned_text": "",
            "subjectivity_detection": "neutral",
            "polarity_detection": "neutral",
        },
        {
            "post_id": "p1",
            "text": "positive opinion one",
            "cleaned_text": "",
            "subjectivity_detection": "opinionated",
            "polarity_detection": "positive",
        },
        {
            "post_id": "p2",
            "text": "positive opinion two",
            "cleaned_text": "",
            "subjectivity_detection": "opinionated",
            "polarity_detection": "positive",
        },
        {
            "post_id": "g1",
            "text": "negative opinion one",
            "cleaned_text": "",
            "subjectivity_detection": "opinionated",
            "polarity_detection": "negative",
        },
        {
            "post_id": "g2",
            "text": "negative opinion two",
            "cleaned_text": "",
            "subjectivity_detection": "opinionated",
            "polarity_detection": "negative",
        },
        {
            "post_id": "bad1",
            "text": "bad anomaly",
            "cleaned_text": "",
            "subjectivity_detection": "neutral",
            "polarity_detection": "positive",
        },
    ]

    monkeypatch.setattr(
        "opinion_scraper.classification.data.load_xlsx_rows",
        lambda path, sheet_name=None: records,
    )
    monkeypatch.setattr(
        "opinion_scraper.classification.data.get_dataset_backend",
        lambda: (InMemoryDataset, InMemoryDatasetDict),
    )

    builder = ManualLabelDatasetBuilder(validation_ratio=0.5, seed=7)
    bundle = builder.build_ablation_bundle("ignored.xlsx")

    final_train = bundle.final_label_dataset["train"].to_list()
    final_validation = bundle.final_label_dataset["validation"].to_list()
    assert len(final_train) + len(final_validation) == 6
    assert bundle.stage_counts["final_label"]["total"] == 6
    assert bundle.stage_counts["subjectivity"]["total"] == 6
    assert bundle.stage_counts["polarity"]["total"] == 4
    assert bundle.anomaly_stats["neutral_with_sentiment_polarity"] == 1
    assert {row["label_name"] for row in final_train + final_validation} == {
        "neutral",
        "positive",
        "negative",
    }
