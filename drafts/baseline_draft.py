import json
import re
import math
import argparse
from collections import defaultdict
from datetime import datetime

SCORE_TP =  2
SCORE_FN = -2
SCORE_FP = -6

def parse_ts(ts_str):
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
 
def extract_features(user, posts):
    """
    Compute per-user features from both the user object and their posts.
    Returns a flat dict of numeric values.
    """
    f = {}
    tweet_count = len(posts)
    f["tweet_count"] = tweet_count
    f["z_score"] = user.get("z_score", 0.0)

    description = (user.get("description") or "").strip()
    username    = (user.get("username")    or "").strip()
    name        = (user.get("name")        or "").strip()
    location    = (user.get("location")    or "").strip()
 
    f["has_empty_description"] = int(len(description) == 0)
    f["has_empty_location"]    = int(len(location) == 0)
    f["has_empty_name"]        = int(len(name) == 0)
    f["description_length"]    = len(description)
    f["description_word_count"] = len(description.split()) if description else 0
    f["username_starts_with_at"] = int(username.startswith("@"))

    digit_ratio = sum(c.isdigit() for c in username) / max(len(username), 1)
    f["username_digit_ratio"] = digit_ratio
 
    f["description_hashtag_count"] = len(re.findall(r"#\w+", description))
    f["location_length"] = len(location)
    f["name_has_control_chars"] = int(any(ord(c) < 32 for c in name))
 
    #timing
    if tweet_count >= 2:
        timestamps = sorted(parse_ts(p["created_at"]) for p in posts)
        gaps = [
            (timestamps[i+1] - timestamps[i]).total_seconds()
            for i in range(len(timestamps) - 1)
        ]
        mean_gap = sum(gaps) / len(gaps)
        variance_gap = sum((g - mean_gap)**2 for g in gaps) / len(gaps)
        f["mean_gap_seconds"] = mean_gap
        f["gap_variance"]     = variance_gap
        f["gap_cv"]           = math.sqrt(variance_gap) / max(mean_gap, 1)
        f["min_gap_seconds"]  = min(gaps)
        f["burst_fraction"]   = sum(1 for g in gaps if g < 10) / len(gaps)
    else:
        f["mean_gap_seconds"] = 0
        f["gap_variance"]     = 0
        f["gap_cv"]           = 0
        f["min_gap_seconds"]  = 0
        f["burst_fraction"]   = 0
 
 
    texts = [p["text"] for p in posts]
 
    #hashtag
    hashtag_counts = [len(re.findall(r"#\w+", t)) for t in texts]
    f["avg_hashtags"] = sum(hashtag_counts) / max(tweet_count, 1)
    f["max_hashtags"] = max(hashtag_counts) if hashtag_counts else 0
 
    #url
    url_counts = [len(re.findall(r"https://t\.co/\S+", t)) for t in texts]
    f["url_fraction"]  = sum(1 for c in url_counts if c > 0) / max(tweet_count, 1)
    f["avg_url_count"] = sum(url_counts) / max(tweet_count, 1)
 
    #text uniqueness
    normalized = [t.strip().lower() for t in texts]
    f["text_uniqueness"] = len(set(normalized)) / max(tweet_count, 1)
 

    lengths = [len(t) for t in texts]
    avg_len = sum(lengths) / max(tweet_count, 1)
    f["avg_tweet_length"] = avg_len
    f["std_tweet_length"] = math.sqrt(
        sum((l - avg_len)**2 for l in lengths) / max(tweet_count, 1)
    )
 
    #all caps word ratio
    all_caps = [
        sum(1 for w in t.split() if len(w) > 2 and w.isupper())
        / max(len(t.split()), 1)
        for t in texts
    ]
    f["avg_caps_ratio"] = sum(all_caps) / max(tweet_count, 1)
 
    #solicitation keywords
    SOLICITATION = [
        "follow", "check my bio", "dm me", "opt in",
        "click", "join", "subscribe", "free picks", "link in bio"
    ]
    f["solicitation_fraction"] = sum(
        1 for t in texts if any(kw in t.lower() for kw in SOLICITATION)
    ) / max(tweet_count, 1)
 
    #average word length
    all_words = [w for t in texts for w in t.split() if w.isalpha()]
    f["avg_word_length"] = (
        sum(len(w) for w in all_words) / max(len(all_words), 1)
    )

    all_hashtags = [h.lower() for t in texts for h in re.findall(r"#(\w+)", t)]
    f["unique_hashtag_count"] = len(set(all_hashtags))
    f["hashtag_diversity"] = len(set(all_hashtags)) / max(len(all_hashtags), 1)

    mention_counts = [len(re.findall(r"@mention", t)) for t in texts]
    f["avg_mentions"] = sum(mention_counts) / max(tweet_count, 1)
 
    f["avg_newlines"] = sum(t.count("\n") for t in texts) / max(tweet_count, 1)
 
    return f
 
 
def score_user(features):
    score = 0.0
 
    z = features["z_score"]
    if   z > 3.0: score += 0.30
    elif z > 2.0: score += 0.18
    elif z > 1.5: score += 0.08
 

    if features["username_starts_with_at"]:
        score += 0.25   
 
    if features["name_has_control_chars"]:
        score += 0.20   
    if features["has_empty_description"]:
        score += 0.07
 
    if features["username_digit_ratio"] > 0.4:
        score += 0.10
 
    dw = features["description_word_count"]
    if 1 <= dw <= 6:
        score += 0.08
 
    cv = features["gap_cv"]
    if 0 < cv < 0.3:
        score += 0.12   # suspiciously regular cadence
    if features["min_gap_seconds"] < 5:
        score += 0.15   # machine-gun burst posting
    if features["burst_fraction"] > 0.2:
        score += 0.10
 
    
    if features["avg_hashtags"] > 4:
        score += 0.15
    elif features["avg_hashtags"] > 2.5:
        score += 0.07
 
    if features["url_fraction"] > 0.85:
        score += 0.10  
 
    uniqueness = features["text_uniqueness"]
    if uniqueness < 0.5:
        score += 0.20   
    elif uniqueness < 0.7:
        score += 0.08
 
    if features["solicitation_fraction"] > 0.3:
        score += 0.20
    elif features["solicitation_fraction"] > 0.1:
        score += 0.10
 
   
    awl = features["avg_word_length"]
    if awl > 5.8:
        score += 0.12
    elif awl > 5.2:
        score += 0.06
 
    if 0 < features["std_tweet_length"] < 20:
        score += 0.10
 
    return min(score, 1.0)
 
 
def evaluate(predictions, ground_truth, all_user_ids):
    tp = len(predictions & ground_truth)
    fn = len(ground_truth - predictions)
    fp = len(predictions - ground_truth)
    tn = len(all_user_ids - ground_truth - predictions)
 
    competition_score = SCORE_TP * tp + SCORE_FN * fn + SCORE_FP * fp
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)
 
    print("\n=== Evaluation Results ===")
    print(f"  True Positives  (bots caught):    {tp}")
    print(f"  False Negatives (bots missed):    {fn}")
    print(f"  False Positives (humans flagged): {fp}")
    print(f"  True Negatives  (humans spared):  {tn}")
    print(f"\n  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"\n  Competition score: {competition_score:+d}")
    print(f"    ({tp}x+{SCORE_TP}) + ({fn}x{SCORE_FN}) + ({fp}x{SCORE_FP})")
    print("==========================\n")
    return competition_score
 
 
def main():
    parser = argparse.ArgumentParser(description="Bot or Not baseline detector v2")
    parser.add_argument("--dataset",   required=True, help="dataset.posts_users.json")
    parser.add_argument("--bots",      default=None,  help="dataset.bots.txt (optional, for eval)")
    parser.add_argument("--output",    required=True, help="output detections .txt file")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Score cutoff to flag as bot (default 0.5). "
                             "Raise to 0.55-0.65 to cut false positives.")
    args = parser.parse_args()
 
    # Load dataset
    print(f"Loading {args.dataset} ...")
    with open(args.dataset, encoding="utf-8") as fh:
        data = json.load(fh)
 
    users = {u["id"]: u for u in data["users"]}
    all_user_ids = set(users.keys())
 
    posts_by_user = defaultdict(list)
    for post in data["posts"]:
        posts_by_user[post["author_id"]].append(post)
 
    print(f"{len(users)} users, {len(data['posts'])} posts")
 
    ground_truth = None
    if args.bots:
        with open(args.bots, encoding="utf-8") as fh:
            ground_truth = set(line.strip() for line in fh if line.strip())
        print(f"  Ground truth: {len(ground_truth)} bots labeled")
 
    print(f"\nScoring users (threshold={args.threshold}) ...")
    user_scores = {}
    for uid, user in users.items():
        feats = extract_features(user, posts_by_user[uid])
        user_scores[uid] = score_user(feats)
 
    flagged = {uid for uid, s in user_scores.items() if s >= args.threshold}
    print(f"Flagged {len(flagged)} / {len(users)} users as bots")
 
    if ground_truth is not None:
        evaluate(flagged, ground_truth, all_user_ids)
 
        missed = ground_truth - flagged
        if missed:
            print(f"Missed bots ({len(missed)}) — look for shared patterns:")
            for uid in sorted(missed, key=lambda u: user_scores[u], reverse=True):
                u = users[uid]
                desc = (u.get("description") or "")[:60]
                print(f"  score={user_scores[uid]:.2f}  @{u.get('username','?'):<20}  "
                      f"z={u.get('z_score',0):>5.2f}  \"{desc}\"")

        false_pos = flagged - ground_truth
        if false_pos:
            print(f"\nFalse positives ({len(false_pos)}) — each costs {SCORE_FP} pts:")
            for uid in sorted(false_pos, key=lambda u: user_scores[u], reverse=True):
                u = users[uid]
                desc = (u.get("description") or "")[:60]
                print(f"  score={user_scores[uid]:.2f}  @{u.get('username','?'):<20}  "
                      f"z={u.get('z_score',0):>5.2f}  \"{desc}\"")
 

    with open(args.output, "w", encoding="utf-8") as fh:
        for uid in sorted(flagged):
            fh.write(uid + "\n")
    print(f"\nDetections written to: {args.output}")
 
    print("\nTop 25 most suspicious users:")
    print(f"  {'Score':>5}  {'Username':<22}  {'z':>5}  description")
    print("  " + "-" * 78)
    top = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)[:25]
    for uid, score in top:
        u   = users[uid]
        lbl = " <- BOT" if ground_truth and uid in ground_truth else ""
        desc = (u.get("description") or "")[:48]
        print(f"  {score:.2f}   @{u.get('username','?'):<20}  "
              f"z={u.get('z_score',0):>5.2f}  \"{desc}\"{lbl}")
 
 
if __name__ == "__main__":
    main()