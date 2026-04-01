import json
from pathlib import Path
from collections import Counter
from typing import Dict, Any

DEFAULT_PATH = Path("prediction/router_train_hotpotqa.jsonl")


def pick_best_expert(experts: Dict[str, Any]) -> str | None:
    """
    Choose best expert by (f1, em, loose_em) in that priority.
    Returns expert name or None if experts dict is empty/bad.
    """
    if not isinstance(experts, dict) or not experts:
        return None

    best_name = None
    best_tuple = None

    for name, info in experts.items():
        if not isinstance(info, dict):
            continue
        f1 = float(info.get("f1", 0.0) or 0.0)
        em = float(info.get("em", 0.0) or 0.0)
        loose = float(info.get("loose_em", 0.0) or 0.0)
        t = (f1, em, loose)
        if best_tuple is None or t > best_tuple:
            best_tuple = t
            best_name = name

    return best_name


def main(path: Path = DEFAULT_PATH) -> None:
    print(f"Reading router data from: {path}")
    rows = [json.loads(l) for l in path.open("r", encoding="utf-8") if l.strip()]
    print(f"\nTotal examples: {len(rows)}\n")

    counts = Counter()
    missing = 0

    for r in rows:
        # If an explicit label exists, use it. Otherwise derive it from per-expert scores.
        label = r.get("label")
        if label is None:
            label = pick_best_expert(r.get("experts", {}))

        if label is None:
            missing += 1
        else:
            counts[str(label)] += 1

    if missing:
        print(f"[WARN] {missing} rows had no usable label.\n")

    if not counts:
        print("[ERROR] No labels found/derived.")
        return

    total = sum(counts.values())
    print("Label distribution (best expert by stored scores):")
    for expert, cnt in counts.most_common():
        frac = 100.0 * cnt / total
        print(f"  {expert:>8}: {cnt:4d} examples ({frac:5.1f}%)")


if __name__ == "__main__":
    main()
