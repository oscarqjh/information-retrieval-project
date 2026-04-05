"""Batch annotation workflow for hierarchical classifiers."""

from __future__ import annotations

import csv
import json
from math import isfinite
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Iterable

from opinion_scraper.classification.inference import HierarchicalClassifier


@dataclass
class AnnotationStats:
    """Performance and scalability statistics for one annotation run."""

    total_records: int
    total_batches: int
    elapsed_seconds: float
    records_per_second: float
    batches_per_second: float
    avg_seconds_per_batch: float
    avg_characters_per_text: float
    max_characters_per_text: int
    p95_characters_per_text: int


@dataclass
class AnnotationArtifacts:
    """Predictions and run-level metrics."""

    predictions: list[dict[str, Any]]
    stats: AnnotationStats


def select_annotation_text(
    record: dict[str, Any],
    primary_key: str = "cleaned_text",
    fallback_key: str = "text",
) -> str:
    """Choose the preferred text source for annotation."""
    primary = str(record.get(primary_key, "") or "").strip()
    if primary:
        return primary
    return str(record.get(fallback_key, "") or "").strip()


def load_csv_records(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    """Load CSV rows while preserving column order."""
    csv_path = Path(path)
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def is_missing_annotation_value(value: Any) -> bool:
    """Treat empty and placeholder cells as missing annotation values."""
    normalized = str(value or "").strip().lower()
    return normalized in {"", "nan", "none", "null"}


def format_annotation_score(value: Any) -> str:
    """Serialize numeric scores while dropping invalid values."""
    if value is None:
        return ""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return ""
    if not isfinite(numeric_value):
        return ""
    return f"{numeric_value:.6f}"


def prepare_csv_records_for_annotation(
    rows: list[dict[str, str]],
    target_columns: list[str],
    force: bool = False,
    id_key: str = "post_id",
    primary_text_key: str = "cleaned_text",
    fallback_text_key: str = "text",
) -> list[dict[str, str]]:
    """Filter CSV rows down to the ones that still need annotation."""
    prepared_records: list[dict[str, str]] = []
    for row in rows:
        if not force and all(
            not is_missing_annotation_value(row.get(column, ""))
            for column in target_columns
        ):
            continue
        text = select_annotation_text(row, primary_key=primary_text_key, fallback_key=fallback_text_key)
        if not text:
            continue
        prepared_records.append(
            {
                id_key: str(row.get(id_key, "") or ""),
                "text": text,
            }
        )
    return prepared_records


def update_csv_records_in_place(
    path: str | Path,
    updates_by_id: dict[str, dict[str, Any]],
    new_columns: list[str],
    id_key: str = "post_id",
) -> None:
    """Apply prediction updates to a CSV file in place."""
    rows, fieldnames = load_csv_records(path)
    merged_fieldnames = list(fieldnames)
    for column in new_columns:
        if column not in merged_fieldnames:
            merged_fieldnames.append(column)

    csv_path = Path(path)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged_fieldnames)
        writer.writeheader()
        for row in rows:
            row_id = str(row.get(id_key, "") or "")
            if row_id in updates_by_id:
                for key, value in updates_by_id[row_id].items():
                    row[key] = value
            writer.writerow({column: row.get(column, "") for column in merged_fieldnames})


def _chunked(items: list[Any], chunk_size: int) -> Iterable[list[Any]]:
    for index in range(0, len(items), chunk_size):
        yield items[index:index + chunk_size]


def _p95(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    position = int(round((len(ordered) - 1) * 0.95))
    return ordered[position]


class HierarchicalBatchAnnotator:
    """Apply the hierarchical classifier to unlabeled or refreshable records."""

    def __init__(self, classifier: HierarchicalClassifier):
        self.classifier = classifier

    def annotate_records(
        self,
        records: list[dict[str, Any]],
        text_key: str = "text",
        id_key: str = "post_id",
        batch_size: int = 32,
    ) -> AnnotationArtifacts:
        """Annotate dictionary records and collect runtime statistics."""
        prepared_records = [record for record in records if str(record.get(text_key, "")).strip()]
        text_lengths = [len(str(record[text_key])) for record in prepared_records]
        predictions: list[dict[str, Any]] = []
        batch_durations: list[float] = []

        started_at = perf_counter()
        for batch in _chunked(prepared_records, max(1, batch_size)):
            texts = [str(record[text_key]) for record in batch]
            batch_started_at = perf_counter()
            batch_predictions = self.classifier.predict(texts)
            batch_durations.append(perf_counter() - batch_started_at)
            for record, prediction in zip(batch, batch_predictions):
                payload = {
                    "post_id": record.get(id_key),
                    "text": record.get(text_key),
                    "stage1_label": prediction.stage1_label,
                    "stage1_score": prediction.stage1_score,
                    "stage2_label": prediction.stage2_label,
                    "stage2_score": prediction.stage2_score,
                    "final_label": prediction.final_label,
                }
                predictions.append(payload)

        elapsed_seconds = perf_counter() - started_at
        total_records = len(prepared_records)
        total_batches = len(batch_durations)
        stats = AnnotationStats(
            total_records=total_records,
            total_batches=total_batches,
            elapsed_seconds=elapsed_seconds,
            records_per_second=(total_records / elapsed_seconds) if elapsed_seconds else 0.0,
            batches_per_second=(total_batches / elapsed_seconds) if elapsed_seconds else 0.0,
            avg_seconds_per_batch=mean(batch_durations) if batch_durations else 0.0,
            avg_characters_per_text=mean(text_lengths) if text_lengths else 0.0,
            max_characters_per_text=max(text_lengths) if text_lengths else 0,
            p95_characters_per_text=_p95(text_lengths),
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
