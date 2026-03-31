"""Metrics helpers for hierarchical classification."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf


@dataclass
class LabelMetrics:
    """Metrics for a single class label."""

    label: str
    support: int
    precision: float
    recall: float
    f1: float


def compute_binary_metrics(
    gold_labels: list[str],
    predicted_labels: list[str],
    positive_label: str,
    label_order: list[str],
) -> tuple[list[LabelMetrics], dict[str, float | int | str]]:
    """Compute binary metrics with a fixed positive label."""
    negative_label = next(label for label in label_order if label != positive_label)

    tp = sum(
        1
        for gold, pred in zip(gold_labels, predicted_labels)
        if gold == positive_label and pred == positive_label
    )
    fp = sum(
        1
        for gold, pred in zip(gold_labels, predicted_labels)
        if gold != positive_label and pred == positive_label
    )
    tn = sum(
        1
        for gold, pred in zip(gold_labels, predicted_labels)
        if gold != positive_label and pred != positive_label
    )
    fn = sum(
        1
        for gold, pred in zip(gold_labels, predicted_labels)
        if gold == positive_label and pred != positive_label
    )

    total = len(gold_labels)
    positive_support = sum(1 for gold in gold_labels if gold == positive_label)
    negative_support = total - positive_support

    positive_precision = tp / (tp + fp) if tp + fp else 0.0
    positive_recall = tp / (tp + fn) if tp + fn else 0.0
    positive_f1 = (
        2 * positive_precision * positive_recall / (positive_precision + positive_recall)
        if positive_precision + positive_recall
        else 0.0
    )

    negative_precision = tn / (tn + fn) if tn + fn else 0.0
    negative_recall = tn / (tn + fp) if tn + fp else 0.0
    negative_f1 = (
        2 * negative_precision * negative_recall / (negative_precision + negative_recall)
        if negative_precision + negative_recall
        else 0.0
    )

    label_metrics = [
        LabelMetrics(
            label=positive_label,
            support=positive_support,
            precision=positive_precision,
            recall=positive_recall,
            f1=positive_f1,
        ),
        LabelMetrics(
            label=negative_label,
            support=negative_support,
            precision=negative_precision,
            recall=negative_recall,
            f1=negative_f1,
        ),
    ]

    accuracy = (tp + tn) / total if total else 0.0
    macro_precision = (positive_precision + negative_precision) / 2
    macro_recall = (positive_recall + negative_recall) / 2
    macro_f1 = (positive_f1 + negative_f1) / 2
    weighted_precision = (
        (positive_precision * positive_support) + (negative_precision * negative_support)
    ) / total if total else 0.0
    weighted_recall = (
        (positive_recall * positive_support) + (negative_recall * negative_support)
    ) / total if total else 0.0
    weighted_f1 = (
        (positive_f1 * positive_support) + (negative_f1 * negative_support)
    ) / total if total else 0.0

    return label_metrics, {
        "accuracy": accuracy,
        "precision": positive_precision,
        "recall": positive_recall,
        "f1": positive_f1,
        "micro_precision": positive_precision,
        "micro_recall": positive_recall,
        "micro_f1": positive_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "positive_label": positive_label,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
    }


def build_binary_compute_metrics(positive_label: str, label_order: list[str]):
    """Build a Trainer-compatible binary metrics function."""

    def compute_metrics(eval_prediction) -> dict[str, float]:
        predictions = eval_prediction.predictions
        labels = eval_prediction.label_ids

        predicted_ids: list[int] = []
        for row in predictions:
            best_index = max(range(len(row)), key=row.__getitem__)
            predicted_ids.append(int(best_index))

        gold_labels = [label_order[int(label_id)] for label_id in labels]
        predicted_labels = [label_order[predicted_id] for predicted_id in predicted_ids]
        _, aggregate_metrics = compute_binary_metrics(
            gold_labels=gold_labels,
            predicted_labels=predicted_labels,
            positive_label=positive_label,
            label_order=label_order,
        )

        return {
            "accuracy": float(aggregate_metrics["accuracy"]),
            "precision": float(aggregate_metrics["precision"]),
            "recall": float(aggregate_metrics["recall"]),
            "f1": float(aggregate_metrics["f1"]),
            "macro_f1": float(aggregate_metrics["macro_f1"]),
            "weighted_f1": float(aggregate_metrics["weighted_f1"]),
        }

    return compute_metrics


def compute_multiclass_metrics(
    gold_labels: list[str],
    predicted_labels: list[str],
    label_order: list[str],
) -> tuple[list[LabelMetrics], dict[str, float]]:
    """Compute multiclass metrics for a fixed label space."""
    if len(gold_labels) != len(predicted_labels):
        raise ValueError("gold_labels and predicted_labels must have the same length.")

    total = len(gold_labels)
    label_metrics: list[LabelMetrics] = []
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    for label in label_order:
        tp = sum(1 for gold, pred in zip(gold_labels, predicted_labels) if gold == label and pred == label)
        fp = sum(1 for gold, pred in zip(gold_labels, predicted_labels) if gold != label and pred == label)
        fn = sum(1 for gold, pred in zip(gold_labels, predicted_labels) if gold == label and pred != label)
        support = sum(1 for gold in gold_labels if gold == label)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        label_metrics.append(
            LabelMetrics(
                label=label,
                support=support,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support

    divisor = max(1, len(label_order))
    accuracy = (
        sum(1 for gold, pred in zip(gold_labels, predicted_labels) if gold == pred) / total
        if total
        else 0.0
    )
    return label_metrics, {
        "accuracy": accuracy,
        "macro_precision": macro_precision / divisor,
        "macro_recall": macro_recall / divisor,
        "macro_f1": macro_f1 / divisor,
        "weighted_precision": weighted_precision / total if total else 0.0,
        "weighted_recall": weighted_recall / total if total else 0.0,
        "weighted_f1": weighted_f1 / total if total else 0.0,
    }


def build_multiclass_compute_metrics(label_order: list[str]):
    """Build a Trainer-compatible multiclass metrics function."""

    def compute_metrics(eval_prediction) -> dict[str, float]:
        predictions = eval_prediction.predictions
        labels = eval_prediction.label_ids

        predicted_ids = [int(max(range(len(row)), key=row.__getitem__)) for row in predictions]
        gold_labels = [label_order[int(label_id)] for label_id in labels]
        predicted_labels = [label_order[predicted_id] for predicted_id in predicted_ids]
        _, aggregate_metrics = compute_multiclass_metrics(
            gold_labels=gold_labels,
            predicted_labels=predicted_labels,
            label_order=label_order,
        )
        return {key: float(value) for key, value in aggregate_metrics.items()}

    return compute_metrics


def apply_threshold_to_scores(
    positive_scores: list[float],
    positive_label: str,
    label_order: list[str],
    threshold: float,
) -> list[str]:
    """Convert positive-class scores into discrete labels using a threshold."""
    negative_label = next(label for label in label_order if label != positive_label)
    return [positive_label if score >= threshold else negative_label for score in positive_scores]


def tune_binary_threshold(
    gold_labels: list[str],
    positive_scores: list[float],
    positive_label: str,
    label_order: list[str],
    metric_name: str = "macro_f1",
    search_steps: int = 101,
) -> dict[str, float | str]:
    """Find the threshold that maximizes a chosen validation metric."""
    if not gold_labels:
        raise ValueError("Threshold tuning requires at least one labeled validation sample.")
    if len(gold_labels) != len(positive_scores):
        raise ValueError("gold_labels and positive_scores must have the same length.")

    best_threshold = 0.5
    best_metric = -inf
    best_metrics: dict[str, float | int | str] | None = None

    total_steps = max(2, search_steps)
    for step in range(total_steps):
        threshold = step / (total_steps - 1)
        predicted_labels = apply_threshold_to_scores(
            positive_scores=positive_scores,
            positive_label=positive_label,
            label_order=label_order,
            threshold=threshold,
        )
        _, aggregate_metrics = compute_binary_metrics(
            gold_labels=gold_labels,
            predicted_labels=predicted_labels,
            positive_label=positive_label,
            label_order=label_order,
        )
        current_metric = float(aggregate_metrics[metric_name])
        is_better = current_metric > best_metric
        if not is_better and current_metric == best_metric:
            is_better = abs(threshold - 0.5) < abs(best_threshold - 0.5)
        if is_better:
            best_metric = current_metric
            best_threshold = threshold
            best_metrics = aggregate_metrics

    assert best_metrics is not None
    return {
        "threshold": best_threshold,
        "metric_name": metric_name,
        "metric_value": best_metric,
        **best_metrics,
    }
