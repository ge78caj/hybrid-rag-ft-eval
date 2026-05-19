import json
from collections import Counter
from pathlib import Path

def read_jsonl(p: Path):
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]

def main():
    for ds in ["hotpotqa", "squad_v2", "pubmedqa_v2"]:
        path = Path(f"prediction/router_train_{ds}.jsonl")
        rows = read_jsonl(path)
        winners = Counter()
        avg = Counter()

        for r in rows:
            experts = r["experts"]
            # oracle by F1
            best = max(experts.keys(), key=lambda e: experts[e]["f1"])
            winners[best] += 1

        print("\n===", ds, "===")
        n = sum(winners.values())
        for k,v in winners.most_common():
            print(f"{k:>10}: {v:4d} ({100*v/n:5.1f}%)")

if __name__ == "__main__":
    main()
