"""
Leave-one-dataset-out evaluation: for each labeled JSON+bots pair, train on all
other pairs of the same language and report accuracy / competition metrics on
the held-out set. Runs English and French in one go.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import detector3 as det


def safe_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def short_label(posts_json):
    return Path(posts_json).name


def eval_fold(test_posts, test_bots, train_pairs, verbose_cv=False, model="ensemble"):
    """Train on train_pairs, predict on test_posts; compare to test_bots."""
    x_train, y_train = det.load_training_concat(train_pairs)
    clf, thr, cv_info = det.train_model(x_train, y_train, verbose=verbose_cv, model=model)

    user_ids, _, x_test, feat_names = det.load_dataset(test_posts)
    gt = det.load_bot_ids(test_bots)
    y_true = np.array([1 if uid in gt else 0 for uid in user_ids], dtype=int)
    probs = clf.predict_proba(x_test)[:, 1]
    # --- DEBUG ---
    print(f"\n  [DEBUG] {short_label(test_posts)}")
    for uid, prob, feat_row, true_label in zip(user_ids, probs, x_test, y_true):
        feat_dict = dict(zip(feat_names, feat_row))
        if prob > 0.1:  # low threshold to catch misses too
            status = "BOT" if true_label == 1 else "FP?"
            print(f"    [{status}] {uid}  prob={prob:.3f}  shared={feat_dict['shared_text_frac']:.2f}  copost={feat_dict['coposter_count']:.0f}  fr_bot={feat_dict['fr_bot_pressure']:.2f}  dup={feat_dict['duplicate_fraction']:.2f}  burst={feat_dict['burst_fraction_10s']:.2f}  vocab={feat_dict['vocab_richness']:.2f}")
    # --- END DEBUG ---
    y_pred = (probs >= thr).astype(int)

    score, tp, fn, fp = det.competition_score_from_preds(y_true, y_pred)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    n = len(user_ids)
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    return {
        "label": short_label(test_posts),
        "users": n,
        "bots_gt": int(np.sum(y_true)),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "score": score,
        "threshold": thr,
        "cv_info": cv_info,
    }



def run_lang(lang, train_dirs, verbose_cv, model):
    all_pairs = []
    for d in train_dirs:
        all_pairs.extend(det.discover_pairs_in_dir(d))
    pairs = det.filter_pairs_by_lang(all_pairs, lang)
    if not pairs:
        return []

    rows = []
    for holdout in pairs:
        train = [p for p in pairs if p != holdout]
        if not train:
            print(f"[{lang}] Skip {short_label(holdout[0])}: no other training data.")
            continue
        if verbose_cv:
            print(f"\n[{lang}] Hold out: {short_label(holdout[0])}  |  Train on {len(train)} file(s)")
        m = eval_fold(holdout[0], holdout[1], train, verbose_cv=verbose_cv, model=model)
        rows.append(m)
    return rows

def print_table(rows, title):
    if not rows:
        print(f"\n{title}: no datasets.")
        return
    print(f"\n{title}")
    print(
        f"{'Dataset':<42}  {'Users':>5}  {'Bots':>4}  "
        f"{'TP':>3} {'FN':>3} {'FP':>3}  {'Acc':>6}  {'P':>5} {'R':>5}  {'Score':>6}"
    )
    print("-" * 100)
    total_score = 0
    for m in rows:
        total_score += m["score"]
        print(
            f"{m['label']:<42}  {m['users']:>5}  {m['bots_gt']:>4}  "
            f"{m['tp']:>3} {m['fn']:>3} {m['fp']:>3}  "
            f"{m['accuracy']*100:>5.1f}%  {m['precision']:>5.2f} {m['recall']:>5.2f}  {m['score']:>+6d}"
        )
    print("-" * 100)
    print(f"{'Sum of competition scores (LODO folds)':<42}  {'':>5}  {'':>4}  {'':>3} {'':>3} {'':>3}  {'':>6}  {'':>5} {'':>5}  {total_score:>+6d}")


def main():
    safe_stdout()
    parser = argparse.ArgumentParser(
        description="Leave-one-out test on all labeled datasets (en + fr)."
    )
    parser.add_argument(
        "--train-dirs",
        default="dataset,practice_dataset",
        help="Comma-separated dirs with dataset.posts&users.*.json + dataset.bots.*.txt",
    )
    parser.add_argument(
        "--verbose-cv",
        action="store_true",
        help="Print per-fold CV lines from training.",
    )
    parser.add_argument(
        "--model",
        choices=["ensemble", "rf", "hgb"],
        default="ensemble",
        help="Same as detector.py --model (default: ensemble).",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "fr", "both"],
        default="both",
        help="Which language(s) to evaluate (default: both).",
    )
    args = parser.parse_args()

    train_dirs = [d.strip() for d in args.train_dirs.split(",") if d.strip()]

    langs = ["en", "fr"] if args.lang == "both" else [args.lang]
    all_rows = []
    for lang in langs:
        rows = run_lang(lang, train_dirs, args.verbose_cv, args.model)
        print_table(rows, f"=== Leave-one-out ({lang}) ===")
        all_rows.extend(rows)

    if len(langs) > 1 and all_rows:
        grand = sum(m["score"] for m in all_rows)
        print(f"\n=== Grand total competition score (all folds): {grand:+d} ===")


if __name__ == "__main__":
    main()
