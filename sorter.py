import json
import argparse
from collections import defaultdict
from datetime import datetime
 
 
def parse_ts(ts_str):
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
 
 
def fmt_user(user, posts, label):
    """Format a user's profile + tweets as a readable string block."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"[{label}]  @{user.get('username', '?')}  —  {user.get('name', '?')}")
    lines.append(f"ID:          {user['id']}")
    lines.append(f"Description: {user.get('description') or '(empty)'}")
    lines.append(f"Location:    {user.get('location') or '(empty)'}")
    lines.append(f"Tweet count: {user.get('tweet_count', '?')}  |  z_score: {user.get('z_score', 0):.3f}")
    lines.append("-" * 70)
 
    sorted_posts = sorted(posts, key=lambda p: parse_ts(p["created_at"]))
    for i, post in enumerate(sorted_posts, 1):
        ts = parse_ts(post["created_at"]).strftime("%b %d %H:%M")
        lines.append(f"  [{i:>3}] {ts}  {post['text']}")
 
    lines.append("")
    return "\n".join(lines)
 
 
def print_summary_table(users, posts_by_user, bot_ids, label_filter=None):
    """Print a compact summary table of all users."""
    rows = []
    for uid, user in users.items():
        is_bot = uid in bot_ids
        label = "BOT" if is_bot else "human"
        if label_filter and label != label_filter:
            continue
        tweet_count = len(posts_by_user[uid])
        rows.append((
            label,
            user.get("z_score", 0),
            tweet_count,
            user.get("username", "?"),
            user.get("name", "?"),
            (user.get("description") or "")[:50],
            uid,
        ))
 
    rows.sort(key=lambda r: (r[0], -r[1]))  # bots first, then by z_score desc
 
    print(f"\n{'Label':<6}  {'z':>6}  {'#tw':>4}  {'Username':<22}  {'Name':<22}  Description")
    print("-" * 100)
    for label, z, tc, username, name, desc, uid in rows:
        marker = "*** " if label == "BOT" else "    "
        print(f"{marker}{label:<3}  {z:>6.2f}  {tc:>4}  @{username:<21}  {name:<22}  {desc}")
 
    bots   = sum(1 for r in rows if r[0] == "BOT")
    humans = sum(1 for r in rows if r[0] == "human")
    print(f"\nTotal: {len(rows)} users  ({bots} bots, {humans} humans)")
 
 
def main():
    parser = argparse.ArgumentParser(description="Explore bot vs human profiles")
    parser.add_argument("--dataset", required=True, help="dataset.posts&users.eng.json")
    parser.add_argument("--bots",    required=True, help="dataset.bots.eng.txt")
    parser.add_argument("--user",    default=None,  help="Print full tweets for one user ID")
    parser.add_argument("--save",    action="store_true",
                        help="Save bots.txt and humans.txt with full profiles")
    parser.add_argument("--only",    choices=["bots", "humans"],
                        help="Summary table: show only bots or only humans")
    args = parser.parse_args()
 
    # Load
    print(f"Loading {args.dataset} ...")
    with open(args.dataset, encoding="utf-8") as f:
        data = json.load(f)
 
    users = {u["id"]: u for u in data["users"]}
    posts_by_user = defaultdict(list)
    for post in data["posts"]:
        posts_by_user[post["author_id"]].append(post)
 
    with open(args.bots, encoding="utf-8") as f:
        bot_ids = set(line.strip() for line in f if line.strip())
 
    print(f"  {len(users)} users  |  {len(data['posts'])} posts  |  {len(bot_ids)} labeled bots")
 
    # -- Single user deep-dive ------------------------------------------------
    if args.user:
        uid = args.user.strip()
        if uid not in users:
            print(f"User ID not found: {uid}")
            return
        label = "BOT" if uid in bot_ids else "HUMAN"
        print(fmt_user(users[uid], posts_by_user[uid], label))
        return
 
    # -- Summary table --------------------------------------------------------
    label_filter = None
    if args.only == "bots":
        label_filter = "BOT"
    elif args.only == "humans":
        label_filter = "human"
 
    print_summary_table(users, posts_by_user, bot_ids, label_filter)
 
    # -- Save full profiles to files ------------------------------------------
    #if args.save:
    for filename, ids, label in [
        ("bots_full.txt",   bot_ids,                  "BOT"),
        ("humans_full.txt", set(users) - bot_ids,     "HUMAN"),
    ]:
        with open(filename, "w", encoding="utf-8") as f:
            # Sort by z_score descending
            sorted_ids = sorted(
                ids,
                key=lambda uid: users[uid].get("z_score", 0) if uid in users else 0,
                reverse=True
            )
            for uid in sorted_ids:
                if uid in users:
                    f.write(fmt_user(users[uid], posts_by_user[uid], label))
                    f.write("\n")
        print(f"Saved: {filename}")

 
if __name__ == "__main__":
    main()