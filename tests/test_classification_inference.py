"""Tests for hierarchical inference."""

import json
from unittest.mock import MagicMock, patch

from opinion_scraper.classification.inference import HierarchicalClassifier


def test_classifier_loads_model_artifacts_before_building_pipelines():
    with patch(
        "opinion_scraper.classification.inference.AutoModelForSequenceClassification.from_pretrained",
        side_effect=["subjectivity-model", "polarity-model"],
    ) as model_loader, patch(
        "opinion_scraper.classification.inference.AutoTokenizer.from_pretrained",
        side_effect=["subjectivity-tokenizer", "polarity-tokenizer"],
    ) as tokenizer_loader, patch(
        "opinion_scraper.classification.inference.pipeline",
        side_effect=[MagicMock(), MagicMock()],
    ) as pipeline_factory:
        HierarchicalClassifier(
            "subjectivity-dir",
            "polarity-dir",
            local_files_only=True,
        )

    assert model_loader.call_args_list[0].kwargs["local_files_only"] is True
    assert tokenizer_loader.call_args_list[0].kwargs["local_files_only"] is True
    assert "local_files_only" not in pipeline_factory.call_args_list[0].kwargs


def test_predict_skips_stage2_when_stage1_is_neutral():
    stage1_pipeline = MagicMock(
        return_value=[
            [
                {"label": "neutral", "score": 0.91},
                {"label": "opinionated", "score": 0.09},
            ]
        ]
    )
    stage2_pipeline = MagicMock()

    with patch(
        "opinion_scraper.classification.inference.AutoModelForSequenceClassification.from_pretrained",
        side_effect=["subjectivity-model", "polarity-model"],
    ), patch(
        "opinion_scraper.classification.inference.AutoTokenizer.from_pretrained",
        side_effect=["subjectivity-tokenizer", "polarity-tokenizer"],
    ), patch(
        "opinion_scraper.classification.inference.pipeline",
        side_effect=[stage1_pipeline, stage2_pipeline],
    ):
        classifier = HierarchicalClassifier("subjectivity-model", "polarity-model")
        predictions = classifier.predict(["A neutral factual statement"])

    assert len(predictions) == 1
    assert predictions[0].stage1_label == "neutral"
    assert predictions[0].stage2_label is None
    assert predictions[0].final_label == "neutral"
    stage2_pipeline.assert_not_called()


def test_predict_runs_stage2_for_opinionated_rows_only():
    stage1_pipeline = MagicMock(
        return_value=[
            [
                {"label": "opinionated", "score": 0.84},
                {"label": "neutral", "score": 0.16},
            ],
            [
                {"label": "neutral", "score": 0.80},
                {"label": "opinionated", "score": 0.20},
            ],
        ]
    )
    stage2_pipeline = MagicMock(
        return_value=[
            [
                {"label": "negative", "score": 0.72},
                {"label": "positive", "score": 0.28},
            ]
        ]
    )

    with patch(
        "opinion_scraper.classification.inference.AutoModelForSequenceClassification.from_pretrained",
        side_effect=["subjectivity-model", "polarity-model"],
    ), patch(
        "opinion_scraper.classification.inference.AutoTokenizer.from_pretrained",
        side_effect=["subjectivity-tokenizer", "polarity-tokenizer"],
    ), patch(
        "opinion_scraper.classification.inference.pipeline",
        side_effect=[stage1_pipeline, stage2_pipeline],
    ):
        classifier = HierarchicalClassifier("subjectivity-model", "polarity-model")
        predictions = classifier.predict(["Strong complaint about AI", "Objective update about AI"])

    assert [prediction.final_label for prediction in predictions] == ["negative", "neutral"]
    assert predictions[0].stage2_label == "negative"
    assert predictions[1].stage2_label is None
    stage2_pipeline.assert_called_once_with(["Strong complaint about AI"], truncation=True, top_k=None)


def test_predict_uses_saved_decision_thresholds(tmp_path):
    subjectivity_dir = tmp_path / "subjectivity"
    polarity_dir = tmp_path / "polarity"
    subjectivity_dir.mkdir()
    polarity_dir.mkdir()

    with open(subjectivity_dir / "label_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "labels": ["neutral", "opinionated"],
                "positive_label": "opinionated",
                "decision_threshold": 0.6,
            },
            handle,
        )
    with open(polarity_dir / "label_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "labels": ["negative", "positive"],
                "positive_label": "positive",
                "decision_threshold": 0.8,
            },
            handle,
        )

    stage1_pipeline = MagicMock(
        return_value=[
            [
                {"label": "opinionated", "score": 0.55},
                {"label": "neutral", "score": 0.45},
            ],
            [
                {"label": "opinionated", "score": 0.91},
                {"label": "neutral", "score": 0.09},
            ],
        ]
    )
    stage2_pipeline = MagicMock(
        return_value=[
            [
                {"label": "positive", "score": 0.75},
                {"label": "negative", "score": 0.25},
            ]
        ]
    )

    with patch(
        "opinion_scraper.classification.inference.AutoModelForSequenceClassification.from_pretrained",
        side_effect=["subjectivity-model", "polarity-model"],
    ), patch(
        "opinion_scraper.classification.inference.AutoTokenizer.from_pretrained",
        side_effect=["subjectivity-tokenizer", "polarity-tokenizer"],
    ), patch(
        "opinion_scraper.classification.inference.pipeline",
        side_effect=[stage1_pipeline, stage2_pipeline],
    ):
        classifier = HierarchicalClassifier(str(subjectivity_dir), str(polarity_dir))
        predictions = classifier.predict(["Borderline opinion", "Strong positive opinion"])

    assert predictions[0].final_label == "neutral"
    assert predictions[1].final_label == "negative"
