"""
Evaluate pretrained models on manually labeled dataset.

Tasks:
1. Subjectivity detection (opinionated vs neutral)
2. Polarity detection (positive vs negative vs neutral)
3. Sarcasm detection (sarcastic vs non-sarcastic)

Models used:
- Subjectivity: Zero-shot classification (facebook/bart-large-mnli)
- Polarity: Pretrained sentiment model (cardiffnlp/twitter-roberta-base-sentiment-latest)
- Sarcasm: Pretrained sarcasm model (helinivan/english-sarcasm-detector)
"""

import pandas as pd
import numpy as np
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import time

warnings.filterwarnings("ignore")

DEVICE = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'CUDA' if DEVICE == 0 else 'CPU'}")


def load_data(path="data/manual_label_dataset_v1.xlsx"):
    df = pd.read_excel(path)
    # Drop rows with NaN labels
    df = df.dropna(subset=["subjectivity_detection", "polarity_detection", "sarcasm_detection"])
    print(f"Loaded {len(df)} samples (after dropping NaN labels)")
    # Use cleaned_text if available, fallback to text
    df["input_text"] = df["cleaned_text"].fillna(df["text"])
    return df


# ── Task 1: Subjectivity Detection ──────────────────────────────────────────

def evaluate_subjectivity_zeroshot(df):
    """Use zero-shot classification: opinionated vs neutral."""
    print("\n" + "=" * 70)
    print("TASK 1: SUBJECTIVITY DETECTION (Zero-Shot - BART-large-MNLI)")
    print("=" * 70)

    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=DEVICE,
        batch_size=32,
    )

    labels = ["opinionated", "neutral"]
    texts = df["input_text"].tolist()
    gold = df["subjectivity_detection"].str.lower().tolist()

    start = time.time()
    results = classifier(texts, candidate_labels=labels, batch_size=32)
    elapsed = time.time() - start

    preds = [r["labels"][0] for r in results]

    print(f"\nInference time: {elapsed:.1f}s ({len(texts)/elapsed:.1f} samples/s)")
    print(f"\nClassification Report:")
    print(classification_report(gold, preds, digits=3))
    print(f"Accuracy: {accuracy_score(gold, preds):.3f}")

    return gold, preds


# ── Task 2: Polarity Detection ───────────────────────────────────────────────

def evaluate_polarity_sentiment(df):
    """Use cardiffnlp/twitter-roberta-base-sentiment-latest for 3-class sentiment."""
    print("\n" + "=" * 70)
    print("TASK 2: POLARITY DETECTION (twitter-roberta-base-sentiment-latest)")
    print("=" * 70)

    classifier = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=DEVICE,
        batch_size=32,
        truncation=True,
        max_length=512,
    )

    texts = df["input_text"].tolist()
    gold = df["polarity_detection"].str.lower().tolist()

    start = time.time()
    results = classifier(texts, batch_size=32)
    elapsed = time.time() - start

    # Model outputs: negative, neutral, positive
    preds = [r["label"].lower() for r in results]

    print(f"\nInference time: {elapsed:.1f}s ({len(texts)/elapsed:.1f} samples/s)")
    print(f"\nClassification Report:")
    print(classification_report(gold, preds, labels=["positive", "neutral", "negative"], digits=3))
    print(f"Accuracy: {accuracy_score(gold, preds):.3f}")

    return gold, preds


# ── Task 3: Sarcasm Detection ────────────────────────────────────────────────

def evaluate_sarcasm(df):
    """Use a pretrained sarcasm detector."""
    print("\n" + "=" * 70)
    print("TASK 3: SARCASM DETECTION (helinivan/english-sarcasm-detector)")
    print("=" * 70)

    classifier = pipeline(
        "text-classification",
        model="helinivan/english-sarcasm-detector",
        device=DEVICE,
        batch_size=32,
        truncation=True,
        max_length=512,
    )

    texts = df["input_text"].tolist()
    gold = df["sarcasm_detection"].str.lower().tolist()

    start = time.time()
    results = classifier(texts, batch_size=32)
    elapsed = time.time() - start

    # Map model labels to our labels
    label_map = {"LABEL_0": "non-sarcastic", "LABEL_1": "sarcastic",
                 "non_sarcasm": "non-sarcastic", "sarcasm": "sarcastic",
                 "not_sarcasm": "non-sarcastic"}
    preds = []
    for r in results:
        label = r["label"]
        mapped = label_map.get(label, label.lower().replace("_", "-"))
        preds.append(mapped)

    print(f"\nInference time: {elapsed:.1f}s ({len(texts)/elapsed:.1f} samples/s)")
    print(f"\nPrediction distribution: {pd.Series(preds).value_counts().to_dict()}")
    print(f"Gold distribution: {pd.Series(gold).value_counts().to_dict()}")
    print(f"\nClassification Report:")
    print(classification_report(gold, preds, labels=["sarcastic", "non-sarcastic"], digits=3, zero_division=0))
    print(f"Accuracy: {accuracy_score(gold, preds):.3f}")

    return gold, preds


# ── Task 2 Alternative: Zero-Shot Polarity ───────────────────────────────────

def evaluate_polarity_zeroshot(df):
    """Use zero-shot classification for polarity as an alternative."""
    print("\n" + "=" * 70)
    print("TASK 2 ALT: POLARITY DETECTION (Zero-Shot - BART-large-MNLI)")
    print("=" * 70)

    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=DEVICE,
        batch_size=32,
    )

    labels = ["positive", "negative", "neutral"]
    texts = df["input_text"].tolist()
    gold = df["polarity_detection"].str.lower().tolist()

    start = time.time()
    results = classifier(texts, candidate_labels=labels, batch_size=32)
    elapsed = time.time() - start

    preds = [r["labels"][0] for r in results]

    print(f"\nInference time: {elapsed:.1f}s ({len(texts)/elapsed:.1f} samples/s)")
    print(f"\nClassification Report:")
    print(classification_report(gold, preds, labels=["positive", "neutral", "negative"], digits=3))
    print(f"Accuracy: {accuracy_score(gold, preds):.3f}")

    return gold, preds


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()

    print("\n" + "#" * 70)
    print("# PRETRAINED MODEL EVALUATION ON MANUAL LABELS")
    print("#" * 70)

    # Task 1
    sub_gold, sub_pred = evaluate_subjectivity_zeroshot(df)

    # Task 2: Dedicated sentiment model
    pol_gold, pol_pred = evaluate_polarity_sentiment(df)

    # Task 2 alt: Zero-shot polarity
    pol_zs_gold, pol_zs_pred = evaluate_polarity_zeroshot(df)

    # Task 3
    sar_gold, sar_pred = evaluate_sarcasm(df)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Task':<45} {'Accuracy':>10}")
    print("-" * 55)
    print(f"{'Subjectivity (BART zero-shot)':<45} {accuracy_score(sub_gold, sub_pred):>10.3f}")
    print(f"{'Polarity (twitter-roberta-sentiment)':<45} {accuracy_score(pol_gold, pol_pred):>10.3f}")
    print(f"{'Polarity (BART zero-shot)':<45} {accuracy_score(pol_zs_gold, pol_zs_pred):>10.3f}")
    print(f"{'Sarcasm (english-sarcasm-detector)':<45} {accuracy_score(sar_gold, sar_pred):>10.3f}")
