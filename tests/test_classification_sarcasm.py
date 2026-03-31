"""Tests for zero-shot sarcasm evaluation and annotation."""

from unittest.mock import MagicMock, patch

from opinion_scraper.classification.sarcasm import (
    SarcasmBatchAnnotator,
    SarcasmClassifier,
    SarcasmPrediction,
    evaluate_sarcasm_on_manual_labels,
)


def test_evaluate_sarcasm_on_manual_labels(monkeypatch):
    monkeypatch.setattr(
        "opinion_scraper.classification.sarcasm.load_xlsx_rows",
        lambda path, sheet_name=None: [
            {"text": "Yeah right, that was amazing", "cleaned_text": "", "sarcasm_detection": "sarcastic"},
            {"text": "This is helpful", "cleaned_text": "", "sarcasm_detection": "non-sarcastic"},
        ],
    )
    fake_classifier = MagicMock()
    fake_classifier.predict.return_value = [
        SarcasmPrediction(text="Yeah right, that was amazing", label="sarcastic", score=0.9),
        SarcasmPrediction(text="This is helpful", label="non-sarcastic", score=0.1),
    ]
    monkeypatch.setattr(
        "opinion_scraper.classification.sarcasm.SarcasmClassifier",
        MagicMock(return_value=fake_classifier),
    )

    artifacts = evaluate_sarcasm_on_manual_labels("manual.xlsx", device=-1)

    assert artifacts.samples == 2
    assert artifacts.metrics["accuracy"] == 1.0
    assert artifacts.metrics["f1"] == 1.0
    assert artifacts.metrics["model_type"] == "zero-shot"


def test_sarcasm_batch_annotator_collects_scores():
    classifier = MagicMock()
    classifier.predict.return_value = [
        SarcasmPrediction(text="One", label="sarcastic", score=0.8),
        SarcasmPrediction(text="Two", label="non-sarcastic", score=0.2),
    ]
    annotator = SarcasmBatchAnnotator(classifier=classifier)

    artifacts = annotator.annotate_records(
        records=[
            {"post_id": "p1", "text": "One"},
            {"post_id": "p2", "text": "Two"},
        ],
        batch_size=2,
        threshold=0.5,
    )

    assert artifacts.predictions[0]["sarcasm_label"] == "sarcastic"
    assert artifacts.predictions[0]["sarcasm_score"] == 0.8
    assert artifacts.stats.total_records == 2


def test_sarcasm_classifier_uses_zero_shot_hypotheses():
    fake_pipeline = MagicMock(
        return_value=[
            {
                "labels": [
                    "This text about an AI tool is sarcastic and expresses criticism or mockery indirectly.",
                    "This text about an AI tool is literal, sincere, and does not use sarcasm.",
                ],
                "scores": [0.8, 0.2],
            }
        ]
    )
    with patch(
        "opinion_scraper.classification.sarcasm.AutoModelForSequenceClassification.from_pretrained",
        return_value=MagicMock(),
    ), patch(
        "opinion_scraper.classification.sarcasm.AutoTokenizer.from_pretrained",
        return_value="tokenizer",
    ), patch(
        "opinion_scraper.classification.sarcasm.pipeline",
        return_value=fake_pipeline,
    ):
        classifier = SarcasmClassifier(model_name="test-model", device=-1)
        predictions = classifier.predict(["sample"], threshold=0.5)

    assert predictions[0].label == "sarcastic"
    assert predictions[0].score == 0.8
    fake_pipeline.assert_called_once()
    assert fake_pipeline.call_args.kwargs["candidate_labels"] == [
        "This text about an AI tool is sarcastic and expresses criticism or mockery indirectly.",
        "This text about an AI tool is literal, sincere, and does not use sarcasm.",
    ]
