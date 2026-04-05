"""Ablation experiments for hierarchical AI-tool comment classification."""

from __future__ import annotations

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
    pipeline,
)

from opinion_scraper.classification.constants import (
    DEFAULT_BASE_MODEL,
    FINAL_LABELS,
    POLARITY_LABELS,
    SUBJECTIVITY_LABELS,
)
from opinion_scraper.classification.data import (
    AblationDatasetBundle,
    HierarchicalDatasetBundle,
    ManualLabelDatasetBuilder,
)
from opinion_scraper.classification.inference import HierarchicalClassifier
from opinion_scraper.classification.metrics import (
    LabelMetrics,
    build_multiclass_compute_metrics,
    compute_multiclass_metrics,
)
from opinion_scraper.classification.training import (
    HierarchicalFineTuner,
    StageTrainingConfig,
    TrainingConfig,
    ensure_training_dependencies,
)


ZERO_SHOT_SUBJECTIVITY_LABELS = ["opinionated", "neutral"]
ZERO_SHOT_SUBJECTIVITY_TEMPLATE = "This text is {} about an AI tool."
ZERO_SHOT_POLARITY_LABELS = ["positive", "negative"]
ZERO_SHOT_POLARITY_TEMPLATE = "The sentiment toward the AI tool in this text is {}."


@dataclass
class AblationConfig:
    """Configuration for baseline and ablation experiments."""

    base_model: str = DEFAULT_BASE_MODEL
    validation_ratio: float = 0.2
    seed: int = 42
    zero_shot_batch_size: int = 8
    zero_shot_local_files_only: bool = False
    subjectivity: StageTrainingConfig = field(default_factory=StageTrainingConfig)
    polarity: StageTrainingConfig = field(default_factory=StageTrainingConfig)
    flat_final: StageTrainingConfig = field(default_factory=StageTrainingConfig)


@dataclass
class AblationArtifacts:
    """Saved experiment paths and summary metrics."""

    output_dir: str
    summary: dict[str, dict[str, Any]]


def _as_output_rows(label_metrics: list[LabelMetrics]) -> list[dict[str, Any]]:
    return [asdict(metric) for metric in label_metrics]


def _normalize_zero_shot_outputs(outputs: Any) -> list[dict[str, Any]]:
    if isinstance(outputs, dict):
        return [outputs]
    return list(outputs)


class AblationRunner:
    """Run baseline and ablation experiments on a shared validation split."""

    def __init__(
        self,
        dataset_builder: ManualLabelDatasetBuilder | None = None,
        fine_tuner: HierarchicalFineTuner | None = None,
    ):
        self.dataset_builder = dataset_builder or ManualLabelDatasetBuilder()
        self.fine_tuner = fine_tuner or HierarchicalFineTuner(dataset_builder=self.dataset_builder)

    def run(
        self,
        output_dir: str | Path,
        data_path: str | Path,
        sheet_name: str | None = None,
        config: AblationConfig | None = None,
    ) -> AblationArtifacts:
        """Run the baseline and two ablation variants on a shared split."""
        ensure_training_dependencies()
        ablation_config = config or AblationConfig()
        self.dataset_builder.validation_ratio = ablation_config.validation_ratio
        self.dataset_builder.seed = ablation_config.seed
        bundle = self.dataset_builder.build_ablation_bundle(data_path, sheet_name=sheet_name)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        baseline_metrics = self._run_hierarchical_baseline(
            bundle=bundle,
            output_dir=output_path / "baseline_hierarchical_finetuned",
            config=ablation_config,
        )
        zero_shot_metrics = self._run_zero_shot_ablation(
            bundle=bundle,
            output_dir=output_path / "ablation_no_finetuning",
            config=ablation_config,
        )
        flat_metrics = self._run_flat_ablation(
            bundle=bundle,
            output_dir=output_path / "ablation_no_hierarchy",
            config=ablation_config,
        )

        summary = {
            "baseline_hierarchical_finetuned": baseline_metrics,
            "ablation_no_finetuning": zero_shot_metrics,
            "ablation_no_hierarchy": flat_metrics,
            "stage_counts": bundle.stage_counts,
            "anomaly_stats": bundle.anomaly_stats,
        }
        with open(output_path / "ablation_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        return AblationArtifacts(output_dir=str(output_path), summary=summary)

    def _run_hierarchical_baseline(
        self,
        bundle: AblationDatasetBundle,
        output_dir: Path,
        config: AblationConfig,
    ) -> dict[str, Any]:
        hierarchical_bundle = HierarchicalDatasetBundle(
            subjectivity_dataset=bundle.subjectivity_dataset,
            polarity_dataset=bundle.polarity_dataset,
            anomaly_stats=bundle.anomaly_stats,
            stage_counts=bundle.stage_counts,
        )
        artifacts = self.fine_tuner.train(
            output_dir=output_dir,
            dataset_bundle=hierarchical_bundle,
            config=TrainingConfig(
                base_model=config.base_model,
                validation_ratio=config.validation_ratio,
                seed=config.seed,
                subjectivity=config.subjectivity,
                polarity=config.polarity,
            ),
        )
        classifier = HierarchicalClassifier(
            subjectivity_model_path=artifacts.subjectivity_model_dir,
            polarity_model_path=artifacts.polarity_model_dir,
            batch_size=max(
                config.subjectivity.per_device_eval_batch_size,
                config.polarity.per_device_eval_batch_size,
            ),
            local_files_only=True,
        )
        validation_rows = list(bundle.final_label_dataset["validation"])
        predictions = classifier.predict([row["text"] for row in validation_rows])
        gold_labels = [row["label_name"] for row in validation_rows]
        predicted_labels = [prediction.final_label for prediction in predictions]
        label_metrics, aggregate_metrics = compute_multiclass_metrics(
            gold_labels=gold_labels,
            predicted_labels=predicted_labels,
            label_order=FINAL_LABELS,
        )
        result = {
            "variant": "baseline_hierarchical_finetuned",
            "accuracy": aggregate_metrics["accuracy"],
            "macro_f1": aggregate_metrics["macro_f1"],
            "weighted_f1": aggregate_metrics["weighted_f1"],
            "label_metrics": _as_output_rows(label_metrics),
            "subjectivity_validation": artifacts.subjectivity_metrics,
            "polarity_validation": artifacts.polarity_metrics,
        }
        with open(output_dir / "final_validation_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        return result

    def _run_zero_shot_ablation(
        self,
        bundle: AblationDatasetBundle,
        output_dir: Path,
        config: AblationConfig,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model,
            local_files_only=config.zero_shot_local_files_only,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            local_files_only=config.zero_shot_local_files_only,
        )
        classifier = pipeline(
            "zero-shot-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1,
            batch_size=config.zero_shot_batch_size,
        )
        validation_rows = list(bundle.final_label_dataset["validation"])
        texts = [row["text"] for row in validation_rows]
        gold_labels = [row["label_name"] for row in validation_rows]

        subjectivity_outputs = _normalize_zero_shot_outputs(
            classifier(
                texts,
                candidate_labels=ZERO_SHOT_SUBJECTIVITY_LABELS,
                hypothesis_template=ZERO_SHOT_SUBJECTIVITY_TEMPLATE,
                multi_label=False,
            )
        )
        predicted_labels: list[str] = []
        opinionated_texts: list[str] = []
        opinionated_indices: list[int] = []

        for index, output in enumerate(subjectivity_outputs):
            stage1_label = str(output["labels"][0]).lower()
            if stage1_label == "opinionated":
                opinionated_indices.append(index)
                opinionated_texts.append(texts[index])
                predicted_labels.append("pending")
            else:
                predicted_labels.append("neutral")

        if opinionated_texts:
            polarity_outputs = _normalize_zero_shot_outputs(
                classifier(
                    opinionated_texts,
                    candidate_labels=ZERO_SHOT_POLARITY_LABELS,
                    hypothesis_template=ZERO_SHOT_POLARITY_TEMPLATE,
                    multi_label=False,
                )
            )
            for row_index, output in zip(opinionated_indices, polarity_outputs):
                predicted_labels[row_index] = str(output["labels"][0]).lower()

        label_metrics, aggregate_metrics = compute_multiclass_metrics(
            gold_labels=gold_labels,
            predicted_labels=predicted_labels,
            label_order=FINAL_LABELS,
        )
        result = {
            "variant": "ablation_no_finetuning",
            "accuracy": aggregate_metrics["accuracy"],
            "macro_f1": aggregate_metrics["macro_f1"],
            "weighted_f1": aggregate_metrics["weighted_f1"],
            "label_metrics": _as_output_rows(label_metrics),
        }
        with open(output_dir / "final_validation_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        return result

    def _run_flat_ablation(
        self,
        bundle: AblationDatasetBundle,
        output_dir: Path,
        config: AblationConfig,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model)

        model_kwargs = {
            "num_labels": len(FINAL_LABELS),
            "id2label": {index: label for index, label in enumerate(FINAL_LABELS)},
            "label2id": {label: index for index, label in enumerate(FINAL_LABELS)},
            "ignore_mismatched_sizes": True,
            "torch_dtype": torch.float32,
        }
        if config.flat_final.classifier_dropout is not None:
            model_kwargs["cls_dropout"] = config.flat_final.classifier_dropout
            model_kwargs["pooler_dropout"] = config.flat_final.classifier_dropout

        model = AutoModelForSequenceClassification.from_pretrained(config.base_model, **model_kwargs)
        model = model.float()
        model.config.torch_dtype = torch.float32

        def tokenize_batch(batch):
            return tokenizer(batch["text"], truncation=True)

        tokenized_dataset = bundle.final_label_dataset.map(tokenize_batch, batched=True)
        training_arguments = TrainingArguments(
            output_dir=str(output_dir),
            learning_rate=config.flat_final.learning_rate,
            per_device_train_batch_size=config.flat_final.per_device_train_batch_size,
            per_device_eval_batch_size=config.flat_final.per_device_eval_batch_size,
            num_train_epochs=config.flat_final.num_train_epochs,
            weight_decay=config.flat_final.weight_decay,
            warmup_ratio=config.flat_final.warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            save_total_limit=max(2, config.flat_final.early_stopping_patience + 1),
            seed=config.seed,
            report_to="none",
        )
        callbacks = []
        if config.flat_final.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=config.flat_final.early_stopping_patience,
                    early_stopping_threshold=config.flat_final.early_stopping_threshold,
                )
            )
        trainer = Trainer(
            model=model,
            args=training_arguments,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            processing_class=tokenizer,
            compute_metrics=build_multiclass_compute_metrics(FINAL_LABELS),
            callbacks=callbacks,
        )
        trainer.train()
        raw_metrics = trainer.evaluate()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        predicted_output = trainer.predict(tokenized_dataset["validation"])
        predicted_ids = [
            int(max(range(len(row)), key=row.__getitem__))
            for row in predicted_output.predictions
        ]
        gold_labels = [FINAL_LABELS[int(label_id)] for label_id in predicted_output.label_ids]
        predicted_labels = [FINAL_LABELS[predicted_id] for predicted_id in predicted_ids]
        label_metrics, aggregate_metrics = compute_multiclass_metrics(
            gold_labels=gold_labels,
            predicted_labels=predicted_labels,
            label_order=FINAL_LABELS,
        )
        result = {
            "variant": "ablation_no_hierarchy",
            "accuracy": aggregate_metrics["accuracy"],
            "macro_f1": aggregate_metrics["macro_f1"],
            "weighted_f1": aggregate_metrics["weighted_f1"],
            "label_metrics": _as_output_rows(label_metrics),
            "trainer_metrics": {
                key: float(value) for key, value in raw_metrics.items() if isinstance(value, (int, float))
            },
            "best_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
        }
        with open(output_dir / "final_validation_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        return result
