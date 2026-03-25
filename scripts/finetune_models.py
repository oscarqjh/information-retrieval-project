"""
Finetune pretrained models on manually labeled data.

Tasks:
1. Subjectivity detection (opinionated vs neutral)
2. Polarity detection (positive vs negative vs neutral)

Base model: cardiffnlp/twitter-roberta-base-sentiment-latest
Split: 80:20 stratified, with early stopping on validation loss.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
OUTPUT_DIR = "models"
SEED = 42

print(f"Device: {DEVICE}")
print(f"Base model: {BASE_MODEL}")


# ── Dataset ──────────────────────────────────────────────────────────────────

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


# ── Finetune one task ────────────────────────────────────────────────────────

def finetune_task(task_name, texts, labels, label_names):
    print(f"\n{'=' * 70}")
    print(f"FINETUNING: {task_name}")
    print(f"{'=' * 70}")

    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for i, name in enumerate(label_names)}
    encoded_labels = [label2id[l] for l in labels]

    # Stratified 80:20 split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, encoded_labels, test_size=0.2, random_state=SEED, stratify=encoded_labels
    )

    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")
    print(f"Label distribution (train): {dict(pd.Series([id2label[l] for l in train_labels]).value_counts())}")
    print(f"Label distribution (val):   {dict(pd.Series([id2label[l] for l in val_labels]).value_counts())}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Datasets
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer)

    # Training arguments
    task_output_dir = os.path.join(OUTPUT_DIR, task_name.replace(" ", "_").lower())
    training_args = TrainingArguments(
        output_dir=task_output_dir,
        num_train_epochs=20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=10,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train
    train_result = trainer.train()
    print(f"\nTraining completed in {train_result.metrics['train_runtime']:.1f}s")
    print(f"Best model from epoch with best f1_macro on val set")

    # Evaluate on val set
    eval_result = trainer.evaluate()
    print(f"\nValidation Results:")
    print(f"  Accuracy:    {eval_result['eval_accuracy']:.3f}")
    print(f"  F1 (macro):  {eval_result['eval_f1_macro']:.3f}")
    print(f"  F1 (weighted): {eval_result['eval_f1_weighted']:.3f}")

    # Detailed classification report
    preds = trainer.predict(val_dataset)
    pred_labels = np.argmax(preds.predictions, axis=-1)
    gold_names = [id2label[l] for l in val_labels]
    pred_names = [id2label[l] for l in pred_labels]

    print(f"\nDetailed Classification Report:")
    print(classification_report(gold_names, pred_names, digits=3))

    # Save best model
    best_model_path = os.path.join(task_output_dir, "best_model")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    print(f"Best model saved to: {best_model_path}")

    return {
        "task": task_name,
        "accuracy": eval_result["eval_accuracy"],
        "f1_macro": eval_result["eval_f1_macro"],
        "f1_weighted": eval_result["eval_f1_weighted"],
        "train_size": len(train_texts),
        "val_size": len(val_texts),
        "epochs_trained": train_result.metrics.get("epoch", "?"),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load data
    df = pd.read_excel("data/manual_label_dataset_v1.xlsx")
    df["input_text"] = df["cleaned_text"].fillna(df["text"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []

    # Task 1: Subjectivity (2-class)
    mask = df["subjectivity_detection"].notna()
    sub_df = df[mask]
    res1 = finetune_task(
        task_name="subjectivity_detection",
        texts=sub_df["input_text"].tolist(),
        labels=sub_df["subjectivity_detection"].str.lower().tolist(),
        label_names=["neutral", "opinionated"],
    )
    results.append(res1)

    # Task 2: Polarity (3-class)
    mask = df["polarity_detection"].notna()
    pol_df = df[mask]
    res2 = finetune_task(
        task_name="polarity_detection",
        texts=pol_df["input_text"].tolist(),
        labels=pol_df["polarity_detection"].str.lower().tolist(),
        label_names=["positive", "neutral", "negative"],
    )
    results.append(res2)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINETUNING SUMMARY")
    print("=" * 70)
    print(f"\n{'Task':<30} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Wtd':>10} {'Epochs':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['task']:<30} {r['accuracy']:>10.3f} {r['f1_macro']:>10.3f} {r['f1_weighted']:>10.3f} {r['epochs_trained']:>8}")

    print("\n\nComparison with pretrained (no finetuning):")
    print(f"{'Task':<30} {'Pretrained Acc':>15} {'Finetuned Acc':>15} {'Pretrained F1m':>15} {'Finetuned F1m':>15}")
    print("-" * 90)
    pretrained = {
        "subjectivity_detection": {"acc": 0.675, "f1m": 0.415},
        "polarity_detection": {"acc": 0.626, "f1m": 0.622},
    }
    for r in results:
        pt = pretrained[r["task"]]
        print(f"{r['task']:<30} {pt['acc']:>15.3f} {r['accuracy']:>15.3f} {pt['f1m']:>15.3f} {r['f1_macro']:>15.3f}")

    # Save results
    results_path = os.path.join(OUTPUT_DIR, "finetune_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
