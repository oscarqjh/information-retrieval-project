"""Training workflow for hierarchical classifiers."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from opinion_scraper.classification.constants import (
    DEFAULT_BASE_MODEL,
    POLARITY_LABELS,
    SUBJECTIVITY_LABELS,
)
from opinion_scraper.classification.data import HierarchicalDatasetBundle, ManualLabelDatasetBuilder
from opinion_scraper.classification.metrics import (
    build_binary_compute_metrics,
    tune_binary_threshold,
)


def ensure_training_dependencies() -> None:
    """Ensure optional training dependencies are available."""
    missing = []
    for package_name in ("datasets", "accelerate"):
        if importlib.util.find_spec(package_name) is None:
            missing.append(package_name)
    if missing:
        joined = ", ".join(missing)
        raise ImportError(
            f"Missing optional training dependencies: {joined}. "
            "Install project dependencies again before running hierarchical fine-tuning."
        )


@dataclass
class HierarchicalTrainingArtifacts:
    """Training outputs for the two-stage classifier."""

    output_dir: str
    subjectivity_model_dir: str
    polarity_model_dir: str
    subjectivity_metrics: dict[str, float | str]
    polarity_metrics: dict[str, float | str]
    anomaly_stats: dict[str, int]


@dataclass
class StageTrainingConfig:
    """Stage-specific hyperparameters for binary fine-tuning."""

    num_train_epochs: int = 6
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    classifier_dropout: float | None = None
    early_stopping_patience: int = 2
    early_stopping_threshold: float = 0.0
    threshold_metric: str = "macro_f1"
    threshold_search_steps: int = 101


@dataclass
class TrainingConfig:
    """Hyperparameters for hierarchical fine-tuning."""

    base_model: str = DEFAULT_BASE_MODEL
    validation_ratio: float = 0.2
    seed: int = 42
    subjectivity: StageTrainingConfig = field(default_factory=StageTrainingConfig)
    polarity: StageTrainingConfig = field(default_factory=StageTrainingConfig)


class HierarchicalFineTuner:
    """Fine-tune two binary sequence-classification models."""

    def __init__(self, dataset_builder: ManualLabelDatasetBuilder | None = None):
        self.dataset_builder = dataset_builder or ManualLabelDatasetBuilder()

    def train(
        self,
        output_dir: str | Path,
        data_path: str | Path | None = None,
        dataset_bundle: HierarchicalDatasetBundle | None = None,
        sheet_name: str | None = None,
        config: TrainingConfig | None = None,
    ) -> HierarchicalTrainingArtifacts:
        """Train both stages and save models, datasets, and metrics."""
        ensure_training_dependencies()
        training_config = config or TrainingConfig()
        if dataset_bundle is None:
            if data_path is None:
                raise ValueError("Either data_path or dataset_bundle must be provided.")
            self.dataset_builder.validation_ratio = training_config.validation_ratio
            self.dataset_builder.seed = training_config.seed
            dataset_bundle = self.dataset_builder.build(data_path, sheet_name=sheet_name)

        output_path = Path(output_dir)
        dataset_root = output_path / "datasets"
        dataset_root.mkdir(parents=True, exist_ok=True)
        dataset_bundle.subjectivity_dataset.save_to_disk(str(dataset_root / "subjectivity"))
        dataset_bundle.polarity_dataset.save_to_disk(str(dataset_root / "polarity"))

        subjectivity_metrics = self._train_stage(
            stage_name="subjectivity",
            dataset_dict=dataset_bundle.subjectivity_dataset,
            label_order=SUBJECTIVITY_LABELS,
            positive_label="opinionated",
            output_dir=output_path / "subjectivity",
            config=training_config,
            stage_config=training_config.subjectivity,
        )
        polarity_metrics = self._train_stage(
            stage_name="polarity",
            dataset_dict=dataset_bundle.polarity_dataset,
            label_order=POLARITY_LABELS,
            positive_label="positive",
            output_dir=output_path / "polarity",
            config=training_config,
            stage_config=training_config.polarity,
        )

        with open(output_path / "anomaly_stats.json", "w", encoding="utf-8") as handle:
            json.dump(dataset_bundle.anomaly_stats, handle, indent=2, ensure_ascii=False)
        with open(output_path / "stage_counts.json", "w", encoding="utf-8") as handle:
            json.dump(dataset_bundle.stage_counts, handle, indent=2, ensure_ascii=False)

        return HierarchicalTrainingArtifacts(
            output_dir=str(output_path),
            subjectivity_model_dir=str(output_path / "subjectivity"),
            polarity_model_dir=str(output_path / "polarity"),
            subjectivity_metrics=subjectivity_metrics,
            polarity_metrics=polarity_metrics,
            anomaly_stats=dataset_bundle.anomaly_stats,
        )

    def _train_stage(
        self,
        stage_name: str,
        dataset_dict: Any,
        label_order: list[str],
        positive_label: str,
        output_dir: Path,
        config: TrainingConfig,
        stage_config: StageTrainingConfig,
    ) -> dict[str, float | str]:
        """Train a single binary stage."""
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model)

        model_kwargs = {
            "num_labels": len(label_order),
            "id2label": {index: label for index, label in enumerate(label_order)},
            "label2id": {label: index for index, label in enumerate(label_order)},
            "ignore_mismatched_sizes": True,
            "torch_dtype": torch.float32,
        }
        if stage_config.classifier_dropout is not None:
            model_kwargs["cls_dropout"] = stage_config.classifier_dropout
            model_kwargs["pooler_dropout"] = stage_config.classifier_dropout

        model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model,
            **model_kwargs,
        )
        model = model.float()
        model.config.torch_dtype = torch.float32

        def tokenize_batch(batch):
            return tokenizer(batch["text"], truncation=True)

        tokenized_dataset = dataset_dict.map(tokenize_batch, batched=True)
        training_arguments = TrainingArguments(
            output_dir=str(output_dir),
            learning_rate=stage_config.learning_rate,
            per_device_train_batch_size=stage_config.per_device_train_batch_size,
            per_device_eval_batch_size=stage_config.per_device_eval_batch_size,
            num_train_epochs=stage_config.num_train_epochs,
            weight_decay=stage_config.weight_decay,
            warmup_ratio=stage_config.warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=stage_config.threshold_metric,
            greater_is_better=True,
            save_total_limit=max(2, stage_config.early_stopping_patience + 1),
            seed=config.seed,
            report_to="none",
        )

        callbacks = []
        if stage_config.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=stage_config.early_stopping_patience,
                    early_stopping_threshold=stage_config.early_stopping_threshold,
                )
            )

        trainer = Trainer(
            model=model,
            args=training_arguments,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            processing_class=tokenizer,
            compute_metrics=build_binary_compute_metrics(
                positive_label=positive_label,
                label_order=label_order,
            ),
            callbacks=callbacks,
        )
        trainer.train()
        raw_metrics = trainer.evaluate()
        prediction_output = trainer.predict(tokenized_dataset["validation"])
        tuned_metrics = self._tune_threshold(
            label_order=label_order,
            positive_label=positive_label,
            label_ids=prediction_output.label_ids,
            logits=prediction_output.predictions,
            stage_config=stage_config,
        )
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        normalized_metrics = self._merge_metrics(
            raw_metrics=raw_metrics,
            tuned_metrics=tuned_metrics,
            best_checkpoint=getattr(trainer.state, "best_model_checkpoint", None),
        )
        with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(normalized_metrics, handle, indent=2, ensure_ascii=False)
        with open(output_dir / "label_config.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "stage_name": stage_name,
                    "labels": label_order,
                    "positive_label": positive_label,
                    "decision_threshold": tuned_metrics["threshold"],
                    "threshold_metric": tuned_metrics["metric_name"],
                    "threshold_metric_value": tuned_metrics["metric_value"],
                    "training_config": asdict(config),
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        return normalized_metrics

    @staticmethod
    def _tune_threshold(
        label_order: list[str],
        positive_label: str,
        label_ids: Any,
        logits: Any,
        stage_config: StageTrainingConfig,
    ) -> dict[str, float | str]:
        """Tune the positive-class decision threshold on the validation split."""
        logits_tensor = torch.as_tensor(logits, dtype=torch.float32)
        positive_index = label_order.index(positive_label)
        positive_scores = torch.softmax(logits_tensor, dim=-1)[:, positive_index].tolist()
        gold_labels = [label_order[int(label_id)] for label_id in label_ids]
        return tune_binary_threshold(
            gold_labels=gold_labels,
            positive_scores=positive_scores,
            positive_label=positive_label,
            label_order=label_order,
            metric_name=stage_config.threshold_metric,
            search_steps=stage_config.threshold_search_steps,
        )

    @staticmethod
    def _merge_metrics(
        raw_metrics: dict[str, Any],
        tuned_metrics: dict[str, float | str],
        best_checkpoint: str | None,
    ) -> dict[str, float | str]:
        """Combine raw trainer metrics and threshold-tuned metrics."""
        normalized_metrics: dict[str, float | str] = {
            key: float(value)
            for key, value in raw_metrics.items()
            if isinstance(value, (int, float))
        }
        if best_checkpoint is not None:
            normalized_metrics["best_checkpoint"] = str(best_checkpoint)

        for key, value in tuned_metrics.items():
            target_key = "selected_threshold" if key == "threshold" else f"threshold_{key}"
            if isinstance(value, (int, float)):
                normalized_metrics[target_key] = float(value)
            else:
                normalized_metrics[target_key] = str(value)
        return normalized_metrics
