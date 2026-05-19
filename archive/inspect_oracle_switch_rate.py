import json
from pathlib import Path
from collections import Counter

def switch_rate(path: str):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    experts = sorted(rows[0]["experts"].keys())
    winners = []
    for r in rows:
        w = max(experts, key=lambda e: r["experts"][e]["f1"])
        winners.append(w)
    cnt = Counter(winners)
    # switch rate = 1 - fraction of most common winner
    top = cnt.most_common(1)[0][1] / len(winners)
    print(Path(path).name, "top_winner_frac=", round(top, 3), "switch_rate=", round(1-top, 3))
    for k,v in cnt.most_common():
        print(" ", k, v)

for p in [
    "prediction/router_train_hotpotqa.jsonl",
    "prediction/router_train_squad_v2.jsonl",
    "prediction/router_train_pubmedqa_v2.jsonl",
]:
    print("\n===", p, "===")
    switch_rate(p)
