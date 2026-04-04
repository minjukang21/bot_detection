# Bot Detector — Submission README

## Overview

This project detects bot accounts in social media datasets using a supervised machine learning pipeline. It was designed for a competition where accounts are labeled as bots or humans, and predictions are evaluated using an asymmetric scoring scheme:

| Outcome | Points |
|--------|--------|
| True Positive (bot correctly flagged) | +2 |
| False Negative (bot missed) | -2 |
| False Positive (human incorrectly flagged) | -6 |

The detector is optimized to maximize this competition score, meaning it is deliberately precision-aware — a missed bot is only one-third as costly as a wrongly flagged human.

---

## Files

```
detector3.py                        # Main detection script
dataset_bots_eng.txt                # Bot labels for English practice dataset
dataset_bots_fr.txt                 # Bot labels for French practice dataset
detections_eng.txt                  # Submission: flagged bots in dataset 7 (EN)
detections_fr.txt                   # Submission: flagged bots in dataset 8 (FR)
README.md                           # This file
```

---

## Requirements

```bash
pip install numpy scikit-learn
```

Python 3.8+ required.

---

## How to Run

### With a single training dataset:
```bash
python detector3.py \
  --dataset "dataset.posts&users.7.json" \
  --train-dataset "dataset.posts&users.eng.json" \
  --train-bots "dataset_bots_eng.txt" \
  --output detections_eng.txt \
  --lang en
```

### With multiple training datasets (recommended):
Place all practice JSONs and their matching `.txt` label files in a folder (e.g. `practice_dataset/`), then run:

```powershell
# English — trains on all EN practice datasets, scores dataset 7
python detector3.py --dataset "dataset.posts&users.7.json" --lang en --train-dirs "practice_dataset" --output detections_eng.txt --no-hard-rules

# French — trains on all FR practice datasets, scores dataset 8
python detector3.py --dataset "dataset.posts&users.8.json" --lang fr --train-dirs "practice_dataset" --output detections_fr.txt
```

The script automatically discovers all `dataset.posts&users.*.json` files in the training directory and matches them with their corresponding `dataset.bots.*.txt` label files, filtering by language.

### Key arguments

| Argument | Description |
|----------|-------------|
| `--dataset` | Target JSON file to score (users + posts) |
| `--lang` | Language: `en` or `fr` |
| `--train-dirs` | Folder(s) containing practice datasets |
| `--train-dataset` | Single training JSON (alternative to `--train-dirs`) |
| `--train-bots` | Single training label file (used with `--train-dataset`) |
| `--output` | Output file for flagged bot IDs |
| `--model` | Model type: `ensemble` (default), `rf`, or `hgb` |
| `--no-hard-rules` | Disable hard-rule veto gate (recommended for EN dataset 7) |
| `--eval-bots` | Optional ground truth file to evaluate predictions |

---

## How It Works

### 1. Feature Extraction

For each user account, the detector extracts features across several categories:

**Timing features** — gaps between posts, burst fractions (posts within 10s/30s/60s of each other), posting hour entropy, max posts in the same minute. Bots often post in rapid bursts at regular intervals.

**Content features** — text uniqueness, duplicate fraction, max duplicate streak, average/max hashtag count, hashtag diversity, URL fraction, average tweet length and variance, character entropy, vocabulary richness, solicitation keyword fraction.

**Profile features** — empty description/location/name flags, username digit ratio, username starting with `@`, description length and word count, presence of control characters in name.

**Inter-account coordination features** — fraction of posts that appear verbatim in other accounts, number of distinct accounts that co-posted the same text within 60 seconds. These cross-user signals are strong indicators of coordinated inauthentic behavior.

**Composite features** — language-specific composite scores:
- `organic_power_user_hint` (EN): rewards high text uniqueness + spread posting hours + low burst rate
- `bot_automation_pressure`: penalizes burst timing, control characters, `@` usernames
- `fr_organic_hint` / `fr_bot_pressure` (FR): French-tuned equivalents

### 2. Model

The detector uses an **ensemble of two classifiers**, averaging their predicted bot probabilities:

- **Random Forest** (600 trees, class weight 1:2.2 bot-to-human)
- **Histogram Gradient Boosting** (400 iterations, balanced class weights, early stopping)

Both are trained on all available labeled practice data for the target language.

### 3. Threshold Optimization

Rather than using a fixed 0.5 threshold, the script uses **stratified cross-validation** to find the probability threshold that maximizes the competition score (TP×2 + FN×-2 + FP×-6) on the training data. For English, a false positive penalty multiplier of 1.5× is applied during threshold search to further reduce false alarms.

### 4. Hard-Rule Veto Gate (EN only)

For English datasets, an additional precision gate blocks flagging unless at least one strong bot signal is present:
- Burst fraction (10s) > 15%
- Duplicate fraction > 25%
- Username digit ratio > 40%
- Posts in same minute > 45%
- Solicitation keyword fraction > 25%
- Control characters in name
- Shared text fraction > 40%
- 2+ coordinated co-posters

For French datasets this gate is disabled by default, as FR bots in the training data tend to be caught by the model's probability alone without strong hard-signal evidence.

> **Note:** For dataset 7 (EN), `--no-hard-rules` was used because the bots present do not exhibit the classic hard-signal patterns seen in the practice data, but still score high model probability based on subtler profile and content features.

---

## Submission Results

| Dataset | Language | Users | Flagged Bots |
|---------|----------|-------|--------------|
| dataset 7 | English | 436 | 29 |
| dataset 8 | French | 250 | 22 |

### Cross-validation performance on training data

| Language | CV Score | TP | FN | FP | Threshold |
|----------|----------|----|----|----|-----------|
| English | +352 | 206 | 18 | 4 | 0.59 |
| French | +170 | 99 | 11 | 1 | 0.60 |

---

## Notes

- Training and scoring use separate datasets — the practice JSONs are used for training, and datasets 7 and 8 are scored without their labels being seen during training.
- User IDs are deduplicated across training files to prevent data leakage.
- All features are computed from post content and metadata only — no external data sources are used.