"""Data loading and dataset construction for hierarchical classification."""

from __future__ import annotations

import json
import posixpath
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from opinion_scraper.classification.constants import (
    DEFAULT_FALLBACK_TEXT_COLUMN,
    DEFAULT_TEXT_COLUMN,
    FINAL_LABELS,
    POLARITY_COLUMN,
    POLARITY_LABELS,
    SUBJECTIVITY_COLUMN,
    SUBJECTIVITY_LABELS,
)


SPREADSHEET_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
CELL_REF_RE = re.compile(r"([A-Z]+)(\d+)")


class InMemoryDataset:
    """Small fallback dataset used when Hugging Face datasets is unavailable."""

    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows

    @classmethod
    def from_list(cls, rows: list[dict[str, Any]]) -> "InMemoryDataset":
        return cls(list(rows))

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._rows[index]

    def __iter__(self):
        return iter(self._rows)

    def to_list(self) -> list[dict[str, Any]]:
        return list(self._rows)

    @property
    def column_names(self) -> list[str]:
        if not self._rows:
            return []
        return list(self._rows[0].keys())

    def map(self, function, batched: bool = False):
        if batched:
            batch = {key: [row.get(key) for row in self._rows] for key in self.column_names}
            mapped = function(batch)
            new_rows: list[dict[str, Any]] = []
            for index, row in enumerate(self._rows):
                updated = dict(row)
                for key, values in mapped.items():
                    updated[key] = values[index]
                new_rows.append(updated)
            return InMemoryDataset(new_rows)

        return InMemoryDataset([function(row) for row in self._rows])

    def save_to_disk(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        with open(target / "data.json", "w", encoding="utf-8") as handle:
            json.dump(self._rows, handle, indent=2, ensure_ascii=False)


class InMemoryDatasetDict(dict):
    """Small fallback DatasetDict used for tests and offline development."""

    def map(self, function, batched: bool = False):
        return InMemoryDatasetDict(
            {split_name: dataset.map(function, batched=batched) for split_name, dataset in self.items()}
        )

    def save_to_disk(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        for split_name, dataset in self.items():
            dataset.save_to_disk(str(target / split_name))


def get_dataset_backend():
    """Return the Dataset and DatasetDict classes."""
    try:
        from datasets import Dataset, DatasetDict

        return Dataset, DatasetDict
    except ImportError:
        return InMemoryDataset, InMemoryDatasetDict


@dataclass
class HierarchicalDatasetBundle:
    """Built datasets and their bookkeeping."""

    subjectivity_dataset: Any
    polarity_dataset: Any
    anomaly_stats: dict[str, int]
    stage_counts: dict[str, dict[str, int]]


@dataclass
class AblationDatasetBundle:
    """Shared train/validation splits for baseline and ablation experiments."""

    subjectivity_dataset: Any
    polarity_dataset: Any
    final_label_dataset: Any
    anomaly_stats: dict[str, int]
    stage_counts: dict[str, dict[str, int]]


def column_ref_to_index(column_ref: str) -> int:
    """Convert an Excel column reference such as 'AB' to a zero-based index."""
    index = 0
    for char in column_ref:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def load_shared_strings(workbook: ZipFile) -> list[str]:
    """Load the shared string table when present."""
    shared_strings_path = "xl/sharedStrings.xml"
    if shared_strings_path not in workbook.namelist():
        return []

    root = ET.fromstring(workbook.read(shared_strings_path))
    shared_strings: list[str] = []
    for item in root.findall("a:si", SPREADSHEET_NS):
        fragments = [node.text or "" for node in item.findall(".//a:t", SPREADSHEET_NS)]
        shared_strings.append("".join(fragments))
    return shared_strings


def resolve_sheet_path(workbook: ZipFile, sheet_name: str | None) -> str:
    """Resolve a worksheet XML path from a workbook archive."""
    workbook_root = ET.fromstring(workbook.read("xl/workbook.xml"))
    relationships_root = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
    relationship_map = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in relationships_root.findall("r:Relationship", REL_NS)
    }

    sheets = workbook_root.find("a:sheets", SPREADSHEET_NS)
    if sheets is None or not list(sheets):
        raise ValueError("No worksheets were found in the workbook.")

    selected_sheet = None
    if sheet_name is None:
        selected_sheet = list(sheets)[0]
    else:
        for sheet in sheets:
            if sheet.attrib.get("name") == sheet_name:
                selected_sheet = sheet
                break
        if selected_sheet is None:
            available = ", ".join(sheet.attrib.get("name", "<unknown>") for sheet in sheets)
            raise ValueError(f"Sheet '{sheet_name}' was not found. Available sheets: {available}")

    relation_id = selected_sheet.attrib[
        "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    ]
    target = relationship_map[relation_id].lstrip("/")
    if target.startswith("xl/"):
        return posixpath.normpath(target)
    return posixpath.normpath(posixpath.join("xl", target))


def extract_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    """Extract a human-readable value from a worksheet cell."""
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//a:t", SPREADSHEET_NS))
    if cell_type == "s":
        value_node = cell.find("a:v", SPREADSHEET_NS)
        if value_node is None or value_node.text is None:
            return ""
        return shared_strings[int(value_node.text)]

    value_node = cell.find("a:v", SPREADSHEET_NS)
    if value_node is None or value_node.text is None:
        return ""
    return value_node.text


def load_xlsx_rows(path: Path, sheet_name: str | None = None) -> list[dict[str, str]]:
    """Read worksheet rows into a list of dictionaries."""
    with ZipFile(path) as workbook:
        shared_strings = load_shared_strings(workbook)
        sheet_path = resolve_sheet_path(workbook, sheet_name)
        worksheet_root = ET.fromstring(workbook.read(sheet_path))

    sheet_data = worksheet_root.find("a:sheetData", SPREADSHEET_NS)
    if sheet_data is None:
        return []

    raw_rows: list[dict[int, str]] = []
    for row in sheet_data:
        values: dict[int, str] = {}
        for cell in row:
            ref = cell.attrib.get("r")
            if ref is None:
                continue
            match = CELL_REF_RE.fullmatch(ref)
            if match is None:
                continue
            values[column_ref_to_index(match.group(1))] = extract_cell_value(cell, shared_strings)
        raw_rows.append(values)

    if not raw_rows:
        return []

    max_index = max(raw_rows[0].keys())
    headers = [raw_rows[0].get(index, "").strip() for index in range(max_index + 1)]
    return [
        {header: row.get(index, "").strip() for index, header in enumerate(headers) if header}
        for row in raw_rows[1:]
    ]


def choose_text(record: dict[str, str], primary_column: str, fallback_column: str) -> str:
    """Choose the preferred text field for model input."""
    primary = record.get(primary_column, "").strip()
    if primary:
        return primary
    return record.get(fallback_column, "").strip()


def stratified_split(
    rows: list[dict[str, Any]],
    label_key: str,
    validation_ratio: float,
    seed: int,
):
    """Split rows into stratified train and validation sets."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row[label_key]), []).append(dict(row))

    rng = random.Random(seed)
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for label in sorted(grouped):
        items = grouped[label]
        rng.shuffle(items)
        if len(items) <= 1 or validation_ratio <= 0:
            train_rows.extend(items)
            continue

        validation_count = int(round(len(items) * validation_ratio))
        validation_count = max(1, validation_count)
        validation_count = min(validation_count, len(items) - 1)
        validation_rows.extend(items[:validation_count])
        train_rows.extend(items[validation_count:])

    rng.shuffle(train_rows)
    rng.shuffle(validation_rows)
    return train_rows, validation_rows


class ManualLabelDatasetBuilder:
    """Build staged datasets from the manually labeled spreadsheet."""

    def __init__(
        self,
        text_column: str = DEFAULT_TEXT_COLUMN,
        fallback_text_column: str = DEFAULT_FALLBACK_TEXT_COLUMN,
        validation_ratio: float = 0.2,
        seed: int = 42,
    ):
        self.text_column = text_column
        self.fallback_text_column = fallback_text_column
        self.validation_ratio = validation_ratio
        self.seed = seed

    def build(self, path: str | Path, sheet_name: str | None = None) -> HierarchicalDatasetBundle:
        """Build staged dataset dictionaries and anomaly statistics."""
        dataset_cls, dataset_dict_cls = get_dataset_backend()
        records = load_xlsx_rows(Path(path), sheet_name=sheet_name)

        subjectivity_rows: list[dict[str, Any]] = []
        polarity_rows: list[dict[str, Any]] = []
        anomaly_stats = {
            "missing_text": 0,
            "missing_subjectivity_label": 0,
            "missing_polarity_label": 0,
            "opinionated_with_neutral_polarity": 0,
            "neutral_with_sentiment_polarity": 0,
        }

        for record in records:
            text = choose_text(record, self.text_column, self.fallback_text_column)
            if not text:
                anomaly_stats["missing_text"] += 1
                continue

            subjectivity = record.get(SUBJECTIVITY_COLUMN, "").strip()
            polarity = record.get(POLARITY_COLUMN, "").strip()
            post_id = record.get("post_id", "")

            if not subjectivity:
                anomaly_stats["missing_subjectivity_label"] += 1
                continue

            if subjectivity not in SUBJECTIVITY_LABELS:
                continue

            subjectivity_rows.append(
                {
                    "post_id": post_id,
                    "text": text,
                    "label": SUBJECTIVITY_LABELS.index(subjectivity),
                    "label_name": subjectivity,
                }
            )

            if subjectivity == "opinionated":
                if not polarity:
                    anomaly_stats["missing_polarity_label"] += 1
                    continue
                if polarity == "neutral":
                    anomaly_stats["opinionated_with_neutral_polarity"] += 1
                    continue
                if polarity in POLARITY_LABELS:
                    polarity_rows.append(
                        {
                            "post_id": post_id,
                            "text": text,
                            "label": POLARITY_LABELS.index(polarity),
                            "label_name": polarity,
                        }
                    )
            elif polarity in POLARITY_LABELS:
                anomaly_stats["neutral_with_sentiment_polarity"] += 1

        subjectivity_train, subjectivity_validation = stratified_split(
            subjectivity_rows, "label", self.validation_ratio, self.seed
        )
        polarity_train, polarity_validation = stratified_split(
            polarity_rows, "label", self.validation_ratio, self.seed
        )

        subjectivity_dataset = dataset_dict_cls(
            {
                "train": dataset_cls.from_list(subjectivity_train),
                "validation": dataset_cls.from_list(subjectivity_validation),
            }
        )
        polarity_dataset = dataset_dict_cls(
            {
                "train": dataset_cls.from_list(polarity_train),
                "validation": dataset_cls.from_list(polarity_validation),
            }
        )

        return HierarchicalDatasetBundle(
            subjectivity_dataset=subjectivity_dataset,
            polarity_dataset=polarity_dataset,
            anomaly_stats=anomaly_stats,
            stage_counts={
                "subjectivity": {
                    "total": len(subjectivity_rows),
                    "train": len(subjectivity_train),
                    "validation": len(subjectivity_validation),
                },
                "polarity": {
                    "total": len(polarity_rows),
                    "train": len(polarity_train),
                    "validation": len(polarity_validation),
                },
            },
        )

    def build_ablation_bundle(self, path: str | Path, sheet_name: str | None = None) -> AblationDatasetBundle:
        """Build shared splits for baseline and ablation experiments."""
        dataset_cls, dataset_dict_cls = get_dataset_backend()
        records = load_xlsx_rows(Path(path), sheet_name=sheet_name)

        final_rows: list[dict[str, Any]] = []
        anomaly_stats = {
            "missing_text": 0,
            "missing_subjectivity_label": 0,
            "missing_polarity_label": 0,
            "opinionated_with_neutral_polarity": 0,
            "neutral_with_sentiment_polarity": 0,
        }

        for record in records:
            text = choose_text(record, self.text_column, self.fallback_text_column)
            if not text:
                anomaly_stats["missing_text"] += 1
                continue

            subjectivity = record.get(SUBJECTIVITY_COLUMN, "").strip()
            polarity = record.get(POLARITY_COLUMN, "").strip()
            post_id = record.get("post_id", "")

            if not subjectivity:
                anomaly_stats["missing_subjectivity_label"] += 1
                continue
            if subjectivity not in SUBJECTIVITY_LABELS:
                continue

            if subjectivity == "neutral":
                if polarity in POLARITY_LABELS:
                    anomaly_stats["neutral_with_sentiment_polarity"] += 1
                    continue
                final_rows.append(
                    {
                        "post_id": post_id,
                        "text": text,
                        "label": FINAL_LABELS.index("neutral"),
                        "label_name": "neutral",
                    }
                )
                continue

            if not polarity:
                anomaly_stats["missing_polarity_label"] += 1
                continue
            if polarity == "neutral":
                anomaly_stats["opinionated_with_neutral_polarity"] += 1
                continue
            if polarity in POLARITY_LABELS:
                final_rows.append(
                    {
                        "post_id": post_id,
                        "text": text,
                        "label": FINAL_LABELS.index(polarity),
                        "label_name": polarity,
                    }
                )

        final_train, final_validation = stratified_split(
            final_rows, "label", self.validation_ratio, self.seed
        )

        def build_subjectivity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            result = []
            for row in rows:
                label_name = "neutral" if row["label_name"] == "neutral" else "opinionated"
                result.append(
                    {
                        "post_id": row["post_id"],
                        "text": row["text"],
                        "label": SUBJECTIVITY_LABELS.index(label_name),
                        "label_name": label_name,
                    }
                )
            return result

        def build_polarity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            result = []
            for row in rows:
                if row["label_name"] == "neutral":
                    continue
                result.append(
                    {
                        "post_id": row["post_id"],
                        "text": row["text"],
                        "label": POLARITY_LABELS.index(row["label_name"]),
                        "label_name": row["label_name"],
                    }
                )
            return result

        subjectivity_train = build_subjectivity_rows(final_train)
        subjectivity_validation = build_subjectivity_rows(final_validation)
        polarity_train = build_polarity_rows(final_train)
        polarity_validation = build_polarity_rows(final_validation)

        return AblationDatasetBundle(
            subjectivity_dataset=dataset_dict_cls(
                {
                    "train": dataset_cls.from_list(subjectivity_train),
                    "validation": dataset_cls.from_list(subjectivity_validation),
                }
            ),
            polarity_dataset=dataset_dict_cls(
                {
                    "train": dataset_cls.from_list(polarity_train),
                    "validation": dataset_cls.from_list(polarity_validation),
                }
            ),
            final_label_dataset=dataset_dict_cls(
                {
                    "train": dataset_cls.from_list(final_train),
                    "validation": dataset_cls.from_list(final_validation),
                }
            ),
            anomaly_stats=anomaly_stats,
            stage_counts={
                "final_label": {
                    "total": len(final_rows),
                    "train": len(final_train),
                    "validation": len(final_validation),
                },
                "subjectivity": {
                    "total": len(subjectivity_train) + len(subjectivity_validation),
                    "train": len(subjectivity_train),
                    "validation": len(subjectivity_validation),
                },
                "polarity": {
                    "total": len(polarity_train) + len(polarity_validation),
                    "train": len(polarity_train),
                    "validation": len(polarity_validation),
                },
            },
        )
