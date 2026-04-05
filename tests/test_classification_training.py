"""Tests for hierarchical fine-tuning."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from transformers import EarlyStoppingCallback

from opinion_scraper.classification.data import (
    HierarchicalDatasetBundle,
    InMemoryDataset,
    InMemoryDatasetDict,
)
from opinion_scraper.classification.training import (
    HierarchicalFineTuner,
    StageTrainingConfig,
    TrainingConfig,
)


def make_dataset_dict(rows):
    return InMemoryDatasetDict(
        {
            "train": InMemoryDataset.from_list(rows),
            "validation": InMemoryDataset.from_list(rows[:1]),
        }
    )


def make_fake_model():
    model = MagicMock()
    model.float.return_value = model
    model.config = SimpleNamespace(torch_dtype=None)
    model.named_parameters.return_value = []
    return model


def test_fine_tuner_trains_two_binary_stages(monkeypatch, tmp_path):
    bundle = HierarchicalDatasetBundle(
        subjectivity_dataset=make_dataset_dict(
            [
                {"text": "neutral example", "label": 0, "label_name": "neutral"},
                {"text": "opinionated example", "label": 1, "label_name": "opinionated"},
            ]
        ),
        polarity_dataset=make_dataset_dict(
            [
                {"text": "negative example", "label": 0, "label_name": "negative"},
                {"text": "positive example", "label": 1, "label_name": "positive"},
            ]
        ),
        anomaly_stats={"missing_text": 0},
        stage_counts={"subjectivity": {"total": 2}, "polarity": {"total": 2}},
    )

    tokenizer_instances = [MagicMock(), MagicMock()]
    trainer_instances = [MagicMock(), MagicMock()]
    trainer_instances[0].evaluate.return_value = {"eval_macro_f1": 0.71, "eval_loss": 0.9}
    trainer_instances[0].predict.return_value = SimpleNamespace(
        predictions=[[0.1, 0.9]],
        label_ids=[1],
    )
    trainer_instances[0].state = SimpleNamespace(best_model_checkpoint="subjectivity-best")
    trainer_instances[1].evaluate.return_value = {"eval_macro_f1": 0.83, "eval_loss": 0.4}
    trainer_instances[1].predict.return_value = SimpleNamespace(
        predictions=[[0.2, 0.8]],
        label_ids=[1],
    )
    trainer_instances[1].state = SimpleNamespace(best_model_checkpoint="polarity-best")

    training_argument_calls = []
    model_calls = []

    monkeypatch.setattr("opinion_scraper.classification.training.ensure_training_dependencies", lambda: None)
    monkeypatch.setattr(
        "opinion_scraper.classification.training.AutoTokenizer.from_pretrained",
        MagicMock(side_effect=tokenizer_instances),
    )

    def fake_model_from_pretrained(*args, **kwargs):
        model_calls.append({"args": args, "kwargs": kwargs})
        return make_fake_model()

    monkeypatch.setattr(
        "opinion_scraper.classification.training.AutoModelForSequenceClassification.from_pretrained",
        fake_model_from_pretrained,
    )

    def fake_training_arguments(**kwargs):
        training_argument_calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr("opinion_scraper.classification.training.TrainingArguments", fake_training_arguments)
    trainer_factory = MagicMock(side_effect=trainer_instances)
    monkeypatch.setattr("opinion_scraper.classification.training.Trainer", trainer_factory)

    fine_tuner = HierarchicalFineTuner()
    artifacts = fine_tuner.train(
        output_dir=tmp_path,
        dataset_bundle=bundle,
        config=TrainingConfig(
            base_model="test-model",
            subjectivity=StageTrainingConfig(
                learning_rate=1e-5,
                warmup_ratio=0.15,
                classifier_dropout=0.2,
                early_stopping_patience=2,
            ),
            polarity=StageTrainingConfig(
                learning_rate=3e-5,
                warmup_ratio=0.05,
                classifier_dropout=0.3,
                early_stopping_patience=1,
            ),
        ),
    )

    assert artifacts.subjectivity_metrics["eval_macro_f1"] == 0.71
    assert artifacts.polarity_metrics["eval_macro_f1"] == 0.83
    assert artifacts.subjectivity_metrics["selected_threshold"] == 0.5
    assert len(model_calls) == 2
    assert model_calls[0]["kwargs"]["num_labels"] == 2
    assert model_calls[0]["kwargs"]["id2label"] == {0: "neutral", 1: "opinionated"}
    assert model_calls[0]["kwargs"]["cls_dropout"] == 0.2
    assert model_calls[1]["kwargs"]["id2label"] == {0: "negative", 1: "positive"}
    assert model_calls[1]["kwargs"]["cls_dropout"] == 0.3
    assert len(training_argument_calls) == 2
    assert training_argument_calls[0]["metric_for_best_model"] == "macro_f1"
    assert training_argument_calls[0]["warmup_ratio"] == 0.15
    assert training_argument_calls[1]["warmup_ratio"] == 0.05

    first_trainer_call = trainer_factory.call_args_list[0].kwargs
    assert isinstance(first_trainer_call["callbacks"][0], EarlyStoppingCallback)
    assert first_trainer_call["callbacks"][0].early_stopping_patience == 2

    first_compute_metrics = first_trainer_call["compute_metrics"]
    binary_metrics = first_compute_metrics(
        SimpleNamespace(
            predictions=[[0.1, 0.9], [0.8, 0.2]],
            label_ids=[1, 0],
        )
    )
    assert binary_metrics["precision"] == 1.0
    assert binary_metrics["recall"] == 1.0
    assert binary_metrics["macro_f1"] == 1.0

    with open(tmp_path / "subjectivity" / "label_config.json", "r", encoding="utf-8") as handle:
        label_config = json.load(handle)
    assert label_config["decision_threshold"] == 0.5


def test_train_uses_builder_when_data_path_is_provided(monkeypatch, tmp_path):
    bundle = HierarchicalDatasetBundle(
        subjectivity_dataset=make_dataset_dict([{"text": "a", "label": 0, "label_name": "neutral"}]),
        polarity_dataset=make_dataset_dict([{"text": "b", "label": 1, "label_name": "positive"}]),
        anomaly_stats={},
        stage_counts={"subjectivity": {"total": 1}, "polarity": {"total": 1}},
    )
    builder = MagicMock()
    builder.build.return_value = bundle

    monkeypatch.setattr("opinion_scraper.classification.training.ensure_training_dependencies", lambda: None)
    monkeypatch.setattr(
        "opinion_scraper.classification.training.HierarchicalFineTuner._train_stage",
        MagicMock(side_effect=[{"eval_macro_f1": 0.5}, {"eval_macro_f1": 0.6}]),
    )

    fine_tuner = HierarchicalFineTuner(dataset_builder=builder)
    fine_tuner.train(
        output_dir=tmp_path,
        data_path="manual.xlsx",
        config=TrainingConfig(validation_ratio=0.3, seed=7),
    )

    assert builder.validation_ratio == 0.3
    assert builder.seed == 7
    builder.build.assert_called_once_with("manual.xlsx", sheet_name=None)
