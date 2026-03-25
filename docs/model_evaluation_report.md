# Model Evaluation Report

**Date:** 2026-03-19 | **Dataset:** 982 labeled Bluesky posts (`data/manual_label_dataset_v1.xlsx`) | **Split:** 80:20 stratified

## Pretrained (No Finetuning)

| Task | Model | Accuracy | Macro F1 |
|------|-------|:--------:|:--------:|
| Subjectivity (2-class) | `facebook/bart-large-mnli` (zero-shot) | 0.675 | 0.415 |
| Polarity (3-class) | `cardiffnlp/twitter-roberta-base-sentiment-latest` | 0.626 | 0.622 |
| Polarity (3-class) | `facebook/bart-large-mnli` (zero-shot) | 0.559 | 0.493 |
| Sarcasm (2-class) | `helinivan/english-sarcasm-detector` | 0.923 | 0.505 |

## Finetuned

Base model: `cardiffnlp/twitter-roberta-base-sentiment-latest` | Early stopping patience: 3 | LR: 2e-5

| Task | Accuracy | Macro F1 | Epochs | Accuracy Gain | Macro F1 Gain |
|------|:--------:|:--------:|:------:|:------------:|:-------------:|
| Subjectivity (2-class) | 0.858 | 0.829 | 6 | +18.3pp | +41.4pp |
| Polarity (3-class) | 0.766 | 0.757 | 9 | +14.0pp | +13.5pp |

> Sarcasm excluded from finetuning due to extreme class imbalance (20 sarcastic / 962 non-sarcastic).
