"""Tests for hierarchical classification CLI commands."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from click.testing import CliRunner

import opinion_scraper.classification as classification_pkg
from opinion_scraper.cli import main
from opinion_scraper.classification.inference import HierarchicalPrediction


def test_run_hierarchical_ablation_command(monkeypatch, tmp_path):
    fake_builder = MagicMock()
    fake_runner = MagicMock()
    fake_runner.run.return_value = SimpleNamespace(summary={"baseline_hierarchical_finetuned": {"accuracy": 0.8}})
    fake_stage_config_factory = MagicMock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs, stage=kwargs))
    fake_ablation_config_factory = MagicMock(side_effect=lambda **kwargs: kwargs)

    monkeypatch.setattr(classification_pkg, "ManualLabelDatasetBuilder", MagicMock(return_value=fake_builder))
    monkeypatch.setattr(classification_pkg, "AblationRunner", MagicMock(return_value=fake_runner))
    monkeypatch.setattr(classification_pkg, "StageTrainingConfig", fake_stage_config_factory)
    monkeypatch.setattr(classification_pkg, "AblationConfig", fake_ablation_config_factory)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run-hierarchical-ablation",
            "--output-dir",
            str(tmp_path / "ablation"),
            "--flat-num-train-epochs",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert '"accuracy": 0.8' in result.output
    config_payload = fake_runner.run.call_args.kwargs["config"]
    assert config_payload["flat_final"].stage["num_train_epochs"] == 2


def test_classify_hierarchical_command(monkeypatch):
    fake_classifier = MagicMock()
    fake_classifier.predict.return_value = [
        HierarchicalPrediction(
            text="AI is helpful",
            stage1_label="opinionated",
            stage1_score=0.91,
            stage2_label="positive",
            stage2_score=0.88,
            final_label="positive",
        ),
        HierarchicalPrediction(
            text="AI update from the vendor",
            stage1_label="neutral",
            stage1_score=0.84,
            stage2_label=None,
            stage2_score=None,
            final_label="neutral",
        ),
    ]
    monkeypatch.setattr(classification_pkg, "HierarchicalClassifier", MagicMock(return_value=fake_classifier))

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "classify-hierarchical",
            "--text",
            "AI is helpful",
            "--text",
            "AI update from the vendor",
        ],
    )

    assert result.exit_code == 0
    assert '"final_label": "positive"' in result.output
    assert '"final_label": "neutral"' in result.output
    fake_classifier.predict.assert_called_once_with(["AI is helpful", "AI update from the vendor"])
    init_kwargs = classification_pkg.HierarchicalClassifier.call_args.kwargs
    assert init_kwargs["subjectivity_model_path"] == "artifacts/ablation/baseline_hierarchical_finetuned/subjectivity"
    assert init_kwargs["polarity_model_path"] == "artifacts/ablation/baseline_hierarchical_finetuned/polarity"


def test_annotate_hierarchical_command_reports_when_no_rows_need_updates(monkeypatch):
    monkeypatch.setattr(
        classification_pkg,
        "load_csv_records",
        MagicMock(return_value=([{"post_id": "p1", "text": "hello"}], ["post_id", "text"])),
    )
    monkeypatch.setattr(
        classification_pkg,
        "prepare_csv_records_for_annotation",
        MagicMock(return_value=[]),
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "annotate-hierarchical",
            "--subjectivity-model",
            "subjectivity-dir",
            "--polarity-model",
            "polarity-dir",
        ],
    )

    assert result.exit_code == 0
    assert "No rows require hierarchical updates" in result.output


def test_annotate_sarcasm_command_reports_when_no_rows_need_updates(monkeypatch):
    monkeypatch.setattr(
        classification_pkg,
        "load_csv_records",
        MagicMock(return_value=([{"post_id": "p1", "text": "hello"}], ["post_id", "text"])),
    )
    monkeypatch.setattr(
        classification_pkg,
        "prepare_csv_records_for_annotation",
        MagicMock(return_value=[]),
    )

    runner = CliRunner()
    result = runner.invoke(main, ["annotate-sarcasm"])

    assert result.exit_code == 0
    assert "No rows require sarcasm updates" in result.output


def test_evaluate_sarcasm_classifier_command(monkeypatch):
    fake_evaluator = MagicMock(
        return_value=SimpleNamespace(metrics={"accuracy": 0.5, "model_type": "zero-shot"})
    )
    monkeypatch.setattr(
        classification_pkg,
        "evaluate_sarcasm_on_manual_labels",
        fake_evaluator,
    )

    runner = CliRunner()
    result = runner.invoke(main, ["evaluate-sarcasm-classifier", "--device", "-1"])

    assert result.exit_code == 0
    assert '"model_type": "zero-shot"' in result.output
    assert fake_evaluator.call_args.kwargs["model_name"] == "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"


def test_classify_sarcasm_command(monkeypatch):
    fake_classifier = MagicMock()
    fake_classifier.predict.return_value = [
        SimpleNamespace(to_dict=lambda: {"text": "Yeah right", "label": "sarcastic", "score": 0.82}),
        SimpleNamespace(to_dict=lambda: {"text": "Helpful update", "label": "non-sarcastic", "score": 0.14}),
    ]
    monkeypatch.setattr(classification_pkg, "SarcasmClassifier", MagicMock(return_value=fake_classifier))

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "classify-sarcasm",
            "--text",
            "Yeah right",
            "--text",
            "Helpful update",
            "--device",
            "-1",
        ],
    )

    assert result.exit_code == 0
    assert '"label": "sarcastic"' in result.output
    assert '"label": "non-sarcastic"' in result.output
    fake_classifier.predict.assert_called_once_with(["Yeah right", "Helpful update"], threshold=0.5)
