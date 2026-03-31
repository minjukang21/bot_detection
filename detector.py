import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict


SCORE_TP = 2
SCORE_FN = -2
SCORE_FP = -6

def safe_stdout():
    # Avoid Windows cp1252 crashes when printing emojis/accents.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def parse_ts(ts_str):
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


def _entropy_from_counts(counts):
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def _char_entropy(s):
    if not s:
        return 0.0
    freq = defaultdict(int)
    for c in s:
        freq[c] += 1
    return _entropy_from_counts(list(freq.values()))


def extract_features(user, posts, dataset_lang=None):
    f = {}
    tweet_count = len(posts)
    f["tweet_count"] = tweet_count
    f["z_score"] = user.get("z_score", 0.0)

    description = (user.get("description") or "").strip()
    username = (user.get("username") or "").strip()
    name = (user.get("name") or "").strip()
    location = (user.get("location") or "").strip()

    f["has_empty_description"] = int(len(description) == 0)
    f["has_empty_location"] = int(len(location) == 0)
    f["has_empty_name"] = int(len(name) == 0)
    f["description_length"] = len(description)
    f["description_word_count"] = len(description.split()) if description else 0
    f["username_starts_with_at"] = int(username.startswith("@"))
    f["username_digit_ratio"] = sum(c.isdigit() for c in username) / max(len(username), 1)
    f["description_hashtag_count"] = len(re.findall(r"#\w+", description))
    f["location_length"] = len(location)
    f["name_has_control_chars"] = int(any(ord(c) < 32 for c in name))

    if tweet_count >= 2:
        timestamps = sorted(parse_ts(p["created_at"]) for p in posts)
        gaps = [
            (timestamps[i + 1] - timestamps[i]).total_seconds()
            for i in range(len(timestamps) - 1)
        ]
        mean_gap = sum(gaps) / len(gaps)
        variance_gap = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
        sorted_gaps = sorted(gaps)
        mid = len(sorted_gaps) // 2
        median_gap = (
            sorted_gaps[mid]
            if len(sorted_gaps) % 2
            else 0.5 * (sorted_gaps[mid - 1] + sorted_gaps[mid])
        )
        f["mean_gap_seconds"] = mean_gap
        f["median_gap_seconds"] = median_gap
        f["max_gap_seconds"] = max(gaps)
        f["gap_variance"] = variance_gap
        f["gap_cv"] = math.sqrt(variance_gap) / max(mean_gap, 1)
        f["min_gap_seconds"] = min(gaps)
        f["burst_fraction_10s"] = sum(1 for g in gaps if g < 10) / len(gaps)
        f["burst_fraction_30s"] = sum(1 for g in gaps if g < 30) / len(gaps)
        f["burst_fraction_60s"] = sum(1 for g in gaps if g < 60) / len(gaps)

        minute_buckets = defaultdict(int)
        hour_counts = [0] * 24
        for ts in timestamps:
            key = (ts.year, ts.month, ts.day, ts.hour, ts.minute)
            minute_buckets[key] += 1
            hour_counts[ts.hour] += 1
        f["posts_same_minute_max_frac"] = max(minute_buckets.values()) / max(tweet_count, 1)
        f["post_hour_entropy"] = _entropy_from_counts(hour_counts)
    else:
        f["mean_gap_seconds"] = 0
        f["median_gap_seconds"] = 0
        f["max_gap_seconds"] = 0
        f["gap_variance"] = 0
        f["gap_cv"] = 0
        f["min_gap_seconds"] = 0
        f["burst_fraction_10s"] = 0
        f["burst_fraction_30s"] = 0
        f["burst_fraction_60s"] = 0
        f["posts_same_minute_max_frac"] = 0
        f["post_hour_entropy"] = 0

    texts = [p["text"] for p in posts]
    normalized = [t.strip().lower() for t in texts]
    unique_texts = len(set(normalized))
    f["text_uniqueness"] = unique_texts / max(tweet_count, 1)
    f["duplicate_fraction"] = 1 - f["text_uniqueness"]

    max_dup_run = 0
    run = 0
    prev = None
    for t in normalized:
        if t == prev and t:
            run += 1
        else:
            run = 1
        prev = t
        max_dup_run = max(max_dup_run, run)
    f["max_duplicate_streak_frac"] = max_dup_run / max(tweet_count, 1)

    hashtag_counts = [len(re.findall(r"#\w+", t)) for t in texts]
    f["avg_hashtags"] = sum(hashtag_counts) / max(tweet_count, 1)
    f["max_hashtags"] = max(hashtag_counts) if hashtag_counts else 0
    all_hashtags = [h.lower() for t in texts for h in re.findall(r"#(\w+)", t)]
    f["unique_hashtag_count"] = len(set(all_hashtags))
    f["hashtag_diversity"] = len(set(all_hashtags)) / max(len(all_hashtags), 1)
    total_hashtag_tokens = sum(hashtag_counts)
    f["unique_hashtag_per_tweet"] = f["unique_hashtag_count"] / max(tweet_count, 1)
    # High = many distinct tags spread across tweets (poll/trend humans); bots often repeat few tags.
    f["hashtag_tokens_per_unique"] = total_hashtag_tokens / max(f["unique_hashtag_count"], 1)

    url_counts = [len(re.findall(r"https://t\.co/\S+", t)) for t in texts]
    f["url_fraction"] = sum(1 for c in url_counts if c > 0) / max(tweet_count, 1)
    f["avg_url_count"] = sum(url_counts) / max(tweet_count, 1)

    lengths = [len(t) for t in texts]
    avg_len = sum(lengths) / max(tweet_count, 1)
    f["avg_tweet_length"] = avg_len
    f["std_tweet_length"] = math.sqrt(
        sum((l - avg_len) ** 2 for l in lengths) / max(tweet_count, 1)
    )

    all_caps = [
        sum(1 for w in t.split() if len(w) > 2 and w.isupper()) / max(len(t.split()), 1)
        for t in texts
    ]
    f["avg_caps_ratio"] = sum(all_caps) / max(tweet_count, 1)

    SOLICITATION = [
        "follow",
        "check my bio",
        "dm me",
        "opt in",
        "click",
        "join",
        "subscribe",
        "free picks",
        "link in bio",
        "suivez",
        "suis-moi",
        "ma bio",
        "cliquez",
        "abonnez",
        "gratuit",
        "message privé",
        "mp moi",
        "gagnez",
        "inscription",
    ]
    f["solicitation_fraction"] = sum(
        1 for t in texts if any(kw in t.lower() for kw in SOLICITATION)
    ) / max(tweet_count, 1)

    all_words = [w for t in texts for w in t.split() if w.isalpha()]
    f["avg_word_length"] = sum(len(w) for w in all_words) / max(len(all_words), 1)
    words_lower = [w.lower() for w in all_words]
    f["vocab_richness"] = len(set(words_lower)) / max(len(words_lower), 1)

    big_text = "\n".join(texts)
    total_chars = len(big_text)
    f["char_entropy"] = _char_entropy(big_text)
    f["punctuation_ratio"] = (
        sum(1 for c in big_text if not c.isalnum() and not c.isspace()) / max(total_chars, 1)
    )
    f["digit_ratio_text"] = sum(c.isdigit() for c in big_text) / max(total_chars, 1)

    mention_counts = [len(re.findall(r"@mention", t)) for t in texts]
    f["avg_mentions"] = sum(mention_counts) / max(tweet_count, 1)
    f["avg_newlines"] = sum(t.count("\n") for t in texts) / max(tweet_count, 1)
    f["emoji_like_ratio"] = sum(
        sum(1 for c in t if ord(c) > 0xFFFF) for t in texts
    ) / max(sum(len(t) for t in texts), 1)

    # Poll / fandom EN accounts: unique text + hours spread + few rapid bursts (FR: disabled).
    burst_10 = f["burst_fraction_10s"]
    ent_h = f["post_hour_entropy"]
    uniq_txt = f["text_uniqueness"]
    if dataset_lang == "en":
        f["organic_power_user_hint"] = (
            uniq_txt * (ent_h / 4.0) * max(0.0, 1.0 - burst_10 * 8.0)
        )
    else:
        f["organic_power_user_hint"] = 0.0

    # Explicit axis for trees: strong bot-like timing/identity artifacts (helps recall vs. pure organic).
    burst_10b = f["burst_fraction_10s"]
    f["bot_automation_pressure"] = min(
        1.0,
        0.42 * min(1.0, burst_10b / 0.11)
        + 0.28 * float(f["name_has_control_chars"])
        + 0.18 * float(f["username_starts_with_at"])
        + 0.12 * min(1.0, f["posts_same_minute_max_frac"] / 0.15),
    )

    return f


def load_dataset(dataset_path):
    with open(dataset_path, encoding="utf-8") as fh:
        data = json.load(fh)

    users = {u["id"]: u for u in data["users"]}
    posts_by_user = defaultdict(list)
    for post in data["posts"]:
        posts_by_user[post["author_id"]].append(post)

    dataset_lang = data.get("lang")
    user_ids = sorted(users.keys())
    feat_rows = [
        extract_features(users[uid], posts_by_user[uid], dataset_lang)
        for uid in user_ids
    ]
    feature_names = sorted(feat_rows[0].keys())
    x = np.array([[row[name] for name in feature_names] for row in feat_rows], dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return user_ids, users, x, feature_names


def load_bot_ids(path):
    with open(path, encoding="utf-8") as fh:
        return set(line.strip() for line in fh if line.strip())


def competition_score_from_preds(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    score = SCORE_TP * tp + SCORE_FN * fn + SCORE_FP * fp
    return score, tp, fn, fp


def find_best_threshold(y_true, probs):
    """
    Maximize competition score; on ties prefer fewer FN, then fewer FP, then higher accuracy.
    """
    best = None
    n = len(y_true)
    for i in range(1, 200):
        thr = i / 200
        y_pred = (probs >= thr).astype(int)
        score, tp, fn, fp = competition_score_from_preds(y_true, y_pred)
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        acc = (tp + tn) / n if n else 0.0
        # Lexicographic tie-break (all maximize except fp, fn).
        key = (score, tp, -fn, -fp, acc, thr)
        if best is None or key > best[0]:
            best = (key, score, thr, tp, fn, fp)
    return best[1], best[2], best[3], best[4], best[5]


class _BlendedProbaClassifier:
    """Average bot probability from two sklearn estimators."""

    def __init__(self, a, b, weight_a=0.5):
        self.a = a
        self.b = b
        self.weight_a = weight_a
        self.weight_b = 1.0 - weight_a

    def predict_proba(self, X):
        p1 = (
            self.weight_a * self.a.predict_proba(X)[:, 1]
            + self.weight_b * self.b.predict_proba(X)[:, 1]
        )
        p1 = np.clip(p1, 0.0, 1.0)
        return np.column_stack((1.0 - p1, p1))


def _make_rf():
    return RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=2,
        class_weight={0: 1.0, 1: 2.2},
        random_state=42,
        n_jobs=-1,
    )


def _make_hgb():
    return HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.06,
        max_depth=7,
        min_samples_leaf=16,
        l2_regularization=0.5,
        random_state=42,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=25,
    )


def _dataset_lang_from_json(path):
    """Root-level lang field from dataset.posts&users.*.json (en/fr)."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("lang")


def discover_pairs_in_dir(base_dir):
    """
    Find matching pairs: dataset.posts&users.<id>.json + dataset.bots.<id>.txt
    """
    p = Path(base_dir)
    if not p.is_dir():
        return []
    pairs = []
    for posts in sorted(p.glob("dataset.posts&users.*.json")):
        stem = posts.stem
        m = re.match(r"dataset\.posts&users\.(.+)$", stem)
        if not m:
            continue
        key = m.group(1)
        bots = p / f"dataset.bots.{key}.txt"
        if bots.is_file():
            pairs.append((str(posts.resolve()), str(bots.resolve())))
    return pairs


def filter_pairs_by_lang(pairs, lang):
    """Keep pairs whose JSON root lang matches (practice eng/fr + numbered sets)."""
    out = []
    for posts_json, bots_txt in pairs:
        stem = Path(posts_json).stem
        if stem.endswith(".eng"):
            if lang != "en":
                continue
        elif stem.endswith(".fr"):
            if lang != "fr":
                continue
        else:
            try:
                dl = _dataset_lang_from_json(posts_json)
            except (OSError, json.JSONDecodeError):
                continue
            if dl != lang:
                continue
        out.append((posts_json, bots_txt))
    return out


def load_training_concat(pairs):
    """Stack rows from multiple labeled datasets into one X, y."""
    if not pairs:
        raise ValueError("No training pairs to load.")

    all_x_rows = []
    all_y = []
    feature_names = None

    for posts_json, bots_txt in pairs:
        user_ids, _, x_block, fn = load_dataset(posts_json)
        if feature_names is None:
            feature_names = fn
        elif fn != feature_names:
            raise ValueError(f"Feature mismatch: {fn} vs {feature_names}")
        bot_ids = load_bot_ids(bots_txt)
        y_block = np.array([1 if uid in bot_ids else 0 for uid in user_ids], dtype=int)
        all_x_rows.append(x_block)
        all_y.append(y_block)

    x_train = np.vstack(all_x_rows)
    y_train = np.concatenate(all_y)
    return x_train, y_train


def train_model(x_train, y_train, verbose=False, model="ensemble"):
    """
    Fit classifier(s) + competition-score threshold (CV when possible).
    model: "rf" | "hgb" | "ensemble" (RF + HistGradientBoosting, averaged probs).
    Returns (clf, threshold, cv_info).
    """
    x_train = np.asarray(x_train, dtype=float)
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)

    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    cv_info = None
    threshold = 0.45

    def _cv_and_threshold(estimator):
        nonlocal threshold, cv_info
        desired = 5 if n_pos >= 20 else 3
        n_splits = min(desired, n_pos, n_neg)
        n_splits = max(n_splits, 2)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_probs = cross_val_predict(
            estimator, x_train, y_train, cv=cv, method="predict_proba"
        )[:, 1]
        best_score, best_thr, tp, fn, fp = find_best_threshold(y_train, cv_probs)
        threshold = best_thr
        cv_info = (best_score, best_thr, tp, fn, fp)
        if verbose:
            print(
                f"  CV threshold: {threshold:.2f}  CV score: {best_score:+d}  "
                f"(TP={tp}, FN={fn}, FP={fp})"
            )

    if n_pos >= 2 and n_neg >= 2:
        if model == "rf":
            clf = _make_rf()
            _cv_and_threshold(clf)
            clf.fit(x_train, y_train)
        elif model == "hgb":
            clf = _make_hgb()
            _cv_and_threshold(clf)
            clf.fit(x_train, y_train)
        else:
            rf = _make_rf()
            hgb = _make_hgb()
            desired = 5 if n_pos >= 20 else 3
            n_splits = min(desired, n_pos, n_neg)
            n_splits = max(n_splits, 2)
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            pr_rf = cross_val_predict(rf, x_train, y_train, cv=cv, method="predict_proba")[
                :, 1
            ]
            pr_hgb = cross_val_predict(hgb, x_train, y_train, cv=cv, method="predict_proba")[
                :, 1
            ]
            blend = 0.5 * (pr_rf + pr_hgb)
            best_score, best_thr, tp, fn, fp = find_best_threshold(y_train, blend)
            threshold = best_thr
            cv_info = (best_score, best_thr, tp, fn, fp)
            if verbose:
                print(
                    f"  CV threshold: {threshold:.2f}  CV score: {best_score:+d}  "
                    f"(TP={tp}, FN={fn}, FP={fp})"
                )
            rf.fit(x_train, y_train)
            hgb.fit(x_train, y_train)
            clf = _BlendedProbaClassifier(rf, hgb, 0.5)
    else:
        if model == "hgb":
            clf = _make_hgb()
        elif model == "rf":
            clf = _make_rf()
        else:
            rf = _make_rf()
            hgb = _make_hgb()
            rf.fit(x_train, y_train)
            hgb.fit(x_train, y_train)
            clf = _BlendedProbaClassifier(rf, hgb, 0.5)
            return clf, threshold, cv_info
        clf.fit(x_train, y_train)

    return clf, threshold, cv_info


def main():
    safe_stdout()
    parser = argparse.ArgumentParser(description="Bot or Not detector (supervised + score-optimized)")
    parser.add_argument("--dataset", required=True, help="Target dataset JSON to score")
    parser.add_argument("--output", required=True, help="Output detections file")
    parser.add_argument(
        "--lang",
        choices=["en", "fr"],
        default=None,
        help="Train on all labeled data for this language (dataset/ + practice_dataset/).",
    )
    parser.add_argument(
        "--train-dirs",
        default="dataset,practice_dataset",
        help="Comma-separated dirs to scan for dataset.posts&users.*.json pairs (with --lang).",
    )
    parser.add_argument("--train-dataset", default=None, help="Single training dataset JSON")
    parser.add_argument("--train-bots", default=None, help="Single training bot IDs TXT")
    parser.add_argument("--eval-bots", default=None, help="Optional bot labels for evaluation of target dataset")
    parser.add_argument(
        "--model",
        choices=["ensemble", "rf", "hgb"],
        default="ensemble",
        help="ensemble = RF + gradient boosting (default); rf / hgb = single model.",
    )
    args = parser.parse_args()

    if args.train_dataset and args.train_bots:
        x_train, y_train = load_training_concat([(args.train_dataset, args.train_bots)])
    else:
        if args.lang is None:
            raise SystemExit("Provide --lang OR both --train-dataset and --train-bots.")
        train_dirs = [d.strip() for d in args.train_dirs.split(",") if d.strip()]
        all_pairs = []
        for d in train_dirs:
            all_pairs.extend(discover_pairs_in_dir(d))
        pairs = filter_pairs_by_lang(all_pairs, args.lang)
        if not pairs:
            raise SystemExit(
                f"No training pairs found for lang={args.lang} in {train_dirs}. "
                "Expected dataset.posts&users.*.json + dataset.bots.*.txt per folder."
            )
        print(f"Training from {len(pairs)} labeled file pair(s):")
        for pj, bj in pairs:
            print(f"  {pj}")
            print(f"  {bj}")
        x_train, y_train = load_training_concat(pairs)

    print(f"Training set users: {len(y_train)} (bots={int(np.sum(y_train))})")
    clf, best_thr, cv_info = train_model(x_train, y_train, verbose=False, model=args.model)
    if cv_info:
        best_score, best_thr, tp, fn, fp = cv_info
        print(f"CV-optimized threshold: {best_thr:.2f}")
        print(f"CV score: {best_score:+d}  (TP={tp}, FN={fn}, FP={fp})")
    else:
        print(f"CV skipped (small classes); using default threshold: {best_thr:.2f}")

    target_user_ids, _, x_target, _ = load_dataset(args.dataset)
    target_probs = clf.predict_proba(x_target)[:, 1]
    flagged = [uid for uid, p in zip(target_user_ids, target_probs) if p >= best_thr]

    with open(args.output, "w", encoding="utf-8") as fh:
        for uid in sorted(flagged):
            fh.write(uid + "\n")
    print(f"Flagged {len(flagged)} / {len(target_user_ids)} users")
    print(f"Detections written to: {args.output}")

    if args.eval_bots:
        gt = load_bot_ids(args.eval_bots)
        y_true = np.array([1 if uid in gt else 0 for uid in target_user_ids], dtype=int)
        y_pred = np.array([1 if uid in set(flagged) else 0 for uid in target_user_ids], dtype=int)
        score, tp, fn, fp = competition_score_from_preds(y_true, y_pred)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        print(f"Eval score: {score:+d}  (TP={tp}, FN={fn}, FP={fp})")
        print(f"Precision={precision:.3f}  Recall={recall:.3f}")


if __name__ == "__main__":
    main()
