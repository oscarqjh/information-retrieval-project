"""Independent sarcasm-detection workflow and evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from opinion_scraper.classification.annotation import AnnotationArtifacts, AnnotationStats
from opinion_scraper.classification.constants import (
    DEFAULT_FALLBACK_TEXT_COLUMN,
    DEFAULT_SARCASM_MODEL,
    DEFAULT_TEXT_COLUMN,
    SARCASM_COLUMN,
    SARCASM_LABELS,
)
from opinion_scraper.classification.data import load_xlsx_rows
from opinion_scraper.classification.metrics import compute_binary_metrics, tune_binary_threshold


@dataclass
class SarcasmPrediction:
    """A single sarcasm prediction."""

    text: str
    label: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SarcasmEvaluationArtifacts:
    """Manual-label evaluation output for sarcasm detection."""

    samples: int
    metrics: dict[str, Any]


ZERO_SHOT_SARCASM_LABELS = ["sarcastic", "non_sarcastic"]
ZERO_SHOT_SARCASM_HYPOTHESES = {
    "sarcastic": "This text about an AI tool is sarcastic and expresses criticism or mockery indirectly.",
    "non_sarcastic": "This text about an AI tool is literal, sincere, and does not use sarcasm.",
}


def _normalize_sarcasm_label(raw_label: str) -> str:
    normalized = raw_label.strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if normalized in {"sarcastic", "sarcasm"}:
        return "sarcastic"
    if normalized in {"nonsarcastic", "nonsarcasm", "notsarcastic"}:
        return "non-sarcastic"
    raise ValueError(f"Unsupported sarcasm label: {raw_label}")


def _load_manual_sarcasm_rows(
    data_path: str | Path,
    sheet_name: str | None,
    text_column: str,
    fallback_text_column: str,
) -> tuple[list[str], list[str]]:
    """Load labeled sarcasm examples from the manual spreadsheet."""
    rows = load_xlsx_rows(Path(data_path), sheet_name=sheet_name)
    texts: list[str] = []
    gold_labels: list[str] = []
    for row in rows:
        label = str(row.get(SARCASM_COLUMN, "") or "").strip().lower()
        if label not in {"sarcastic", "non-sarcastic"}:
            continue
        text = str(row.get(text_column, "") or "").strip() or str(row.get(fallback_text_column, "") or "").strip()
        if not text:
            continue
        texts.append(text)
        gold_labels.append(label)
    return texts, gold_labels


def _build_sarcasm_evaluation_metrics(
    gold_labels: list[str],
    predicted_labels: list[str],
    positive_scores: list[float],
    threshold: float,
    threshold_metric: str,
    threshold_search_steps: int,
) -> dict[str, Any]:
    """Build binary sarcasm metrics plus a recommended tuned threshold."""
    label_metrics, aggregate_metrics = compute_binary_metrics(
        gold_labels=gold_labels,
        predicted_labels=predicted_labels,
        positive_label="sarcastic",
        label_order=SARCASM_LABELS,
    )
    tuned_metrics = tune_binary_threshold(
        gold_labels=gold_labels,
        positive_scores=positive_scores,
        positive_label="sarcastic",
        label_order=SARCASM_LABELS,
        metric_name=threshold_metric,
        search_steps=threshold_search_steps,
    )
    return {
        "samples": len(gold_labels),
        "threshold": threshold,
        "accuracy": aggregate_metrics["accuracy"],
        "precision": aggregate_metrics["precision"],
        "recall": aggregate_metrics["recall"],
        "f1": aggregate_metrics["f1"],
        "macro_f1": aggregate_metrics["macro_f1"],
        "weighted_f1": aggregate_metrics["weighted_f1"],
        "label_metrics": [asdict(metric) for metric in label_metrics],
        "best_threshold": tuned_metrics["threshold"],
        "best_threshold_metric": tuned_metrics["metric_name"],
        "best_threshold_metric_value": tuned_metrics["metric_value"],
    }


def _normalize_zero_shot_sarcasm_candidate(raw_label: str) -> str:
    """Map zero-shot candidate labels or hypotheses back to canonical sarcasm labels."""
    normalized = raw_label.strip()
    if normalized in ZERO_SHOT_SARCASM_HYPOTHESES.values():
        for label_key, hypothesis in ZERO_SHOT_SARCASM_HYPOTHESES.items():
            if normalized == hypothesis:
                return _normalize_sarcasm_label(label_key)
    return _normalize_sarcasm_label(normalized)


class SarcasmClassifier:
    """Run sarcasm classification with a zero-shot NLI model."""

    def __init__(
        self,
        model_name: str = DEFAULT_SARCASM_MODEL,
        device: int = 0,
        batch_size: int = 16,
        local_files_only: bool = False,
        hypothesis_template: str = "{}",
    ):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        tokenizer = self._load_tokenizer(
            model_name=model_name,
            local_files_only=local_files_only,
        )
        self._hypothesis_template = hypothesis_template
        self._pipeline = pipeline(
            "zero-shot-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
        )

    def predict(
        self,
        texts: str | list[str],
        threshold: float = 0.5,
    ) -> list[SarcasmPrediction]:
        """Predict sarcasm labels and positive-class scores."""
        if isinstance(texts, str):
            text_list = [texts]
        else:
            text_list = list(texts)
        if not text_list:
            return []

        outputs = self._pipeline(
            text_list,
            candidate_labels=list(ZERO_SHOT_SARCASM_HYPOTHESES.values()),
            hypothesis_template=self._hypothesis_template,
            multi_label=False,
        )
        normalized_outputs = outputs if isinstance(outputs, list) else [outputs]
        predictions: list[SarcasmPrediction] = []
        for text, output in zip(text_list, normalized_outputs):
            normalized_scores = {
                _normalize_zero_shot_sarcasm_candidate(str(label)): float(score)
                for label, score in zip(output["labels"], output["scores"])
            }
            sarcastic_score = normalized_scores["sarcastic"]
            predictions.append(
                SarcasmPrediction(
                    text=text,
                    label="sarcastic" if sarcastic_score >= threshold else "non-sarcastic",
                    score=sarcastic_score,
                )
            )
        return predictions

    @staticmethod
    def _load_tokenizer(
        model_name: str,
        local_files_only: bool,
    ):
        try:
            return AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )
        except (ImportError, ValueError):
            return AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                use_fast=False,
            )


class SarcasmBatchAnnotator:
    """Batch sarcasm annotation with throughput statistics."""

    def __init__(self, classifier: SarcasmClassifier):
        self.classifier = classifier

    def annotate_records(
        self,
        records: list[dict[str, Any]],
        text_key: str = "text",
        id_key: str = "post_id",
        batch_size: int = 32,
        threshold: float = 0.5,
    ) -> AnnotationArtifacts:
        """Annotate records and collect runtime metrics."""
        prepared_records = [record for record in records if str(record.get(text_key, "")).strip()]
        text_lengths = [len(str(record[text_key])) for record in prepared_records]
        predictions: list[dict[str, Any]] = []
        batch_durations: list[float] = []

        started_at = perf_counter()
        for index in range(0, len(prepared_records), max(1, batch_size)):
            batch = prepared_records[index:index + max(1, batch_size)]
            texts = [str(record[text_key]) for record in batch]
            batch_started_at = perf_counter()
            batch_predictions = self.classifier.predict(texts, threshold=threshold)
            batch_durations.append(perf_counter() - batch_started_at)
            for record, prediction in zip(batch, batch_predictions):
                predictions.append(
                    {
                        "post_id": record.get(id_key),
                        "text": record.get(text_key),
                        "sarcasm_label": prediction.label,
                        "sarcasm_score": prediction.score,
                    }
                )

        elapsed_seconds = perf_counter() - started_at
        total_records = len(prepared_records)
        total_batches = len(batch_durations)
        stats = AnnotationStats(
            total_records=total_records,
            total_batches=total_batches,
            elapsed_seconds=elapsed_seconds,
            records_per_second=(total_records / elapsed_seconds) if elapsed_seconds else 0.0,
            batches_per_second=(total_batches / elapsed_seconds) if elapsed_seconds else 0.0,
            avg_seconds_per_batch=(sum(batch_durations) / total_batches) if total_batches else 0.0,
            avg_characters_per_text=(sum(text_lengths) / total_records) if total_records else 0.0,
            max_characters_per_text=max(text_lengths) if text_lengths else 0,
            p95_characters_per_text=sorted(text_lengths)[int(round((len(text_lengths) - 1) * 0.95))]
            if text_lengths
            else 0,
        )
        return AnnotationArtifacts(predictions=predictions, stats=stats)

    @staticmethod
    def save_jsonl(predictions: list[dict[str, Any]], output_path: str | Path) -> None:
        """Persist predictions to JSONL."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def save_stats(stats: AnnotationStats, output_path: str | Path) -> None:
        """Persist run metrics to JSON."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(asdict(stats), handle, indent=2, ensure_ascii=False)


def evaluate_sarcasm_on_manual_labels(
    data_path: str | Path,
    model_name: str = DEFAULT_SARCASM_MODEL,
    sheet_name: str | None = None,
    text_column: str = DEFAULT_TEXT_COLUMN,
    fallback_text_column: str = DEFAULT_FALLBACK_TEXT_COLUMN,
    batch_size: int = 16,
    device: int = 0,
    local_files_only: bool = False,
    threshold: float = 0.5,
    threshold_metric: str = "f1",
    threshold_search_steps: int = 101,
    hypothesis_template: str = "{}",
    metrics_out: str | Path | None = None,
) -> SarcasmEvaluationArtifacts:
    """Evaluate zero-shot sarcasm agreement on the manually labeled spreadsheet."""
    texts, gold_labels = _load_manual_sarcasm_rows(
        data_path=data_path,
        sheet_name=sheet_name,
        text_column=text_column,
        fallback_text_column=fallback_text_column,
    )

    classifier = SarcasmClassifier(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        local_files_only=local_files_only,
        hypothesis_template=hypothesis_template,
    )
    predictions = classifier.predict(texts, threshold=threshold)
    predicted_labels = [prediction.label for prediction in predictions]
    positive_scores = [prediction.score for prediction in predictions]
    metrics = _build_sarcasm_evaluation_metrics(
        gold_labels=gold_labels,
        predicted_labels=predicted_labels,
        positive_scores=positive_scores,
        threshold=threshold,
        threshold_metric=threshold_metric,
        threshold_search_steps=threshold_search_steps,
    )
    metrics["model_name"] = model_name
    metrics["model_type"] = "zero-shot"
    metrics["candidate_labels"] = list(ZERO_SHOT_SARCASM_LABELS)
    metrics["candidate_hypotheses"] = dict(ZERO_SHOT_SARCASM_HYPOTHESES)
    metrics["hypothesis_template"] = hypothesis_template
    if metrics_out is not None:
        output = Path(metrics_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=False)
    return SarcasmEvaluationArtifacts(samples=len(gold_labels), metrics=metrics)
