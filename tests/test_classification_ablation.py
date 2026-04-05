"""Tests for hierarchical ablation experiments."""

from unittest.mock import MagicMock

from opinion_scraper.classification.ablation import AblationRunner
from opinion_scraper.classification.data import AblationDatasetBundle, InMemoryDataset, InMemoryDatasetDict


def make_dataset_dict(rows):
    return InMemoryDatasetDict(
        {
            "train": InMemoryDataset.from_list(rows),
            "validation": InMemoryDataset.from_list(rows[:1]),
        }
    )


def test_ablation_runner_writes_summary(monkeypatch, tmp_path):
    builder = MagicMock()
    builder.build_ablation_bundle.return_value = AblationDatasetBundle(
        subjectivity_dataset=make_dataset_dict([{"text": "a", "label": 0, "label_name": "neutral"}]),
        polarity_dataset=make_dataset_dict([{"text": "b", "label": 1, "label_name": "positive"}]),
        final_label_dataset=make_dataset_dict([{"text": "c", "label": 2, "label_name": "positive"}]),
        anomaly_stats={"missing_text": 0},
        stage_counts={"final_label": {"total": 1}},
    )

    monkeypatch.setattr("opinion_scraper.classification.ablation.ensure_training_dependencies", lambda: None)
    runner = AblationRunner(dataset_builder=builder)
    monkeypatch.setattr(
        runner,
        "_run_hierarchical_baseline",
        MagicMock(return_value={"accuracy": 0.8, "macro_f1": 0.75}),
    )
    monkeypatch.setattr(
        runner,
        "_run_zero_shot_ablation",
        MagicMock(return_value={"accuracy": 0.7, "macro_f1": 0.68}),
    )
    monkeypatch.setattr(
        runner,
        "_run_flat_ablation",
        MagicMock(return_value={"accuracy": 0.6, "macro_f1": 0.59}),
    )

    artifacts = runner.run(output_dir=tmp_path, data_path="manual.xlsx")

    assert artifacts.summary["baseline_hierarchical_finetuned"]["accuracy"] == 0.8
    assert (tmp_path / "ablation_summary.json").exists()
