# Bot or Not Competition Detector


## Approach

- Extract per-user behavioral/account/text features from `dataset.posts&users.*.json` (timing, duplication, vocabulary, character entropy, French + English promo phrases, etc.).
- Train a supervised model using **all labeled data** for the target language: by default an **ensemble** of `RandomForestClassifier` + `HistGradientBoostingClassifier` (averaged bot probabilities), with `--model rf` or `--model hgb` for a single model:
  - Numbered sets in `dataset/` (`dataset.posts&users.{1..6}.json` + `dataset.bots.{1..6}.txt`), filtered by root `lang` (`en` or `fr`).
  - Plus the matching file in `practice_dataset/` (`dataset.posts&users.eng.json` / `fr.json`).
- Optimize decision threshold with cross-validation for the exact competition scoring:
  - `+2` true positive
  - `-2` false negative
  - `-6` false positive
- Apply the model to the target dataset and write one `user_id` per line.

## Files

- `detector.py`: main bot detector.
- `script.py`: baseline.
- `sorter.py`: exploring the dataset.
- `test_accuracy.py`: evaluate one detections file vs one labeled dataset.
- `test_all_datasets.py`: **leave-one-dataset-out** evaluation on every labeled pair in `dataset/` + `practice_dataset/` (English and French in one run).