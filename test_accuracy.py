import argparse
import json


SCORE_TP = 2
SCORE_FN = -2
SCORE_FP = -6


def load_ids(path):
    with open(path, encoding="utf-8") as fh:
        return set(line.strip() for line in fh if line.strip())


def load_all_user_ids(dataset_path):
    with open(dataset_path, encoding="utf-8") as fh:
        data = json.load(fh)
    return {u["id"] for u in data["users"]}


def evaluate(detections, bots, all_users):
    tp = len(detections & bots)
    fn = len(bots - detections)
    fp = len((detections - bots) & all_users)
    tn = len(all_users - bots - detections)

    score = SCORE_TP * tp + SCORE_FN * fn + SCORE_FP * fp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(all_users) if all_users else 0.0

    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "score": score,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate bot detections against labeled bots."
    )
    parser.add_argument("--dataset", required=True, help="dataset.posts&users.*.json")
    parser.add_argument("--bots", required=True, help="dataset.bots.*.txt (ground truth)")
    parser.add_argument("--detections", required=True, help="your detections txt")
    args = parser.parse_args()

    all_users = load_all_user_ids(args.dataset)
    bots = load_ids(args.bots)
    detections = load_ids(args.detections)

    unknown_ids = detections - all_users
    if unknown_ids:
        print(f"Warning: {len(unknown_ids)} detection IDs are not in the dataset.")

    metrics = evaluate(detections, bots, all_users)

    print("=== Evaluation ===")
    print(f"Users:      {len(all_users)}")
    print(f"Bots GT:    {len(bots)}")
    print(f"Detections: {len(detections)}")
    print()
    print(f"TP: {metrics['tp']}")
    print(f"FN: {metrics['fn']}")
    print(f"FP: {metrics['fp']}")
    print(f"TN: {metrics['tn']}")
    print()
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print()
    print(f"Competition score: {metrics['score']:+d}")
    print(f"Formula: (TP*{SCORE_TP}) + (FN*{SCORE_FN}) + (FP*{SCORE_FP})")


if __name__ == "__main__":
    main()
