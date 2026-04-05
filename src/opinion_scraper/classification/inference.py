"""Hierarchical inference workflow."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from opinion_scraper.classification.constants import POLARITY_LABELS, SUBJECTIVITY_LABELS
from opinion_scraper.classification.metrics import apply_threshold_to_scores


@dataclass
class HierarchicalPrediction:
    """A single hierarchical classification result."""

    text: str
    stage1_label: str
    stage1_score: float
    stage2_label: str | None
    stage2_score: float | None
    final_label: str

    def to_dict(self) -> dict[str, object]:
        """Serialize the prediction to a plain dictionary."""
        return asdict(self)


def _normalize_label(raw_label: str, label_order: list[str]) -> str:
    normalized = raw_label.strip().lower()
    if normalized in label_order:
        return normalized
    if normalized.startswith("label_"):
        index = int(normalized.split("_", maxsplit=1)[1])
        return label_order[index]
    raise ValueError(f"Unsupported model label: {raw_label}")


def _ensure_sequence_list(outputs: Any) -> list[list[dict[str, float | str]]]:
    if isinstance(outputs, dict):
        return [[outputs]]
    if outputs and isinstance(outputs[0], dict):
        return [outputs]
    return outputs


def _load_label_config(model_path: str, label_order: list[str], positive_label: str) -> dict[str, Any]:
    config_path = Path(model_path) / "label_config.json"
    if not config_path.exists():
        return {
            "labels": label_order,
            "positive_label": positive_label,
            "decision_threshold": 0.5,
        }
    with open(config_path, "r", encoding="utf-8") as handle:
        config_data = json.load(handle)
    config_data.setdefault("labels", label_order)
    config_data.setdefault("positive_label", positive_label)
    config_data.setdefault("decision_threshold", 0.5)
    return config_data


def _select_binary_label(
    scores: dict[str, float],
    label_order: list[str],
    positive_label: str,
    threshold: float,
) -> tuple[str, float]:
    predicted_label = apply_threshold_to_scores(
        positive_scores=[scores[positive_label]],
        positive_label=positive_label,
        label_order=label_order,
        threshold=threshold,
    )[0]
    return predicted_label, float(scores[predicted_label])


class HierarchicalClassifier:
    """Run subjectivity then polarity classification."""

    def __init__(
        self,
        subjectivity_model_path: str,
        polarity_model_path: str,
        device: int = -1,
        batch_size: int = 8,
        local_files_only: bool = False,
    ):
        subjectivity_model = AutoModelForSequenceClassification.from_pretrained(
            subjectivity_model_path,
            local_files_only=local_files_only,
        )
        subjectivity_tokenizer = AutoTokenizer.from_pretrained(
            subjectivity_model_path,
            local_files_only=local_files_only,
        )
        self._subjectivity_pipeline = pipeline(
            "text-classification",
            model=subjectivity_model,
            tokenizer=subjectivity_tokenizer,
            device=device,
            batch_size=batch_size,
        )
        polarity_model = AutoModelForSequenceClassification.from_pretrained(
            polarity_model_path,
            local_files_only=local_files_only,
        )
        polarity_tokenizer = AutoTokenizer.from_pretrained(
            polarity_model_path,
            local_files_only=local_files_only,
        )
        self._polarity_pipeline = pipeline(
            "text-classification",
            model=polarity_model,
            tokenizer=polarity_tokenizer,
            device=device,
            batch_size=batch_size,
        )
        self._subjectivity_config = _load_label_config(
            subjectivity_model_path,
            label_order=SUBJECTIVITY_LABELS,
            positive_label="opinionated",
        )
        self._polarity_config = _load_label_config(
            polarity_model_path,
            label_order=POLARITY_LABELS,
            positive_label="positive",
        )

    def predict(self, texts: str | list[str]) -> list[HierarchicalPrediction]:
        """Predict final labels for one or more texts."""
        if isinstance(texts, str):
            text_list = [texts]
        else:
            text_list = list(texts)
        if not text_list:
            return []

        stage1_outputs = _ensure_sequence_list(
            self._subjectivity_pipeline(text_list, truncation=True, top_k=None)
        )
        predictions: list[HierarchicalPrediction] = []
        opinionated_indices: list[int] = []
        opinionated_texts: list[str] = []

        stage1_threshold = float(self._subjectivity_config["decision_threshold"])
        stage1_positive_label = str(self._subjectivity_config["positive_label"])
        stage1_label_order = list(self._subjectivity_config["labels"])

        for index, (text, output_scores) in enumerate(zip(text_list, stage1_outputs)):
            normalized_scores = {
                _normalize_label(item["label"], SUBJECTIVITY_LABELS): float(item["score"])
                for item in output_scores
            }
            stage1_label, stage1_score = _select_binary_label(
                scores=normalized_scores,
                label_order=stage1_label_order,
                positive_label=stage1_positive_label,
                threshold=stage1_threshold,
            )
            prediction = HierarchicalPrediction(
                text=text,
                stage1_label=stage1_label,
                stage1_score=stage1_score,
                stage2_label=None,
                stage2_score=None,
                final_label="neutral" if stage1_label == "neutral" else "pending",
            )
            predictions.append(prediction)
            if stage1_label == "opinionated":
                opinionated_indices.append(index)
                opinionated_texts.append(text)

        if opinionated_texts:
            stage2_outputs = _ensure_sequence_list(
                self._polarity_pipeline(opinionated_texts, truncation=True, top_k=None)
            )
            stage2_threshold = float(self._polarity_config["decision_threshold"])
            stage2_positive_label = str(self._polarity_config["positive_label"])
            stage2_label_order = list(self._polarity_config["labels"])

            for prediction_index, output_scores in zip(opinionated_indices, stage2_outputs):
                normalized_scores = {
                    _normalize_label(item["label"], POLARITY_LABELS): float(item["score"])
                    for item in output_scores
                }
                stage2_label, stage2_score = _select_binary_label(
                    scores=normalized_scores,
                    label_order=stage2_label_order,
                    positive_label=stage2_positive_label,
                    threshold=stage2_threshold,
                )
                predictions[prediction_index].stage2_label = stage2_label
                predictions[prediction_index].stage2_score = stage2_score
                predictions[prediction_index].final_label = stage2_label

        return predictions
