import json
from pathlib import Path
from collections import Counter


def main():
    path = Path("prediction") / "router_train_hotpotqa.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}. Run build_router_labels_hotpotqa.py first.")

    counts = Counter()
    best_f1_values = []

    print(f"Reading router labels from: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            best_expert = obj.get("best_expert")
            best_f1 = obj.get("best_f1", 0.0)

            counts[best_expert] += 1
            best_f1_values.append(best_f1)

    total = sum(counts.values())
    print(f"\nTotal labeled examples: {total}\n")

    for expert, cnt in counts.items():
        frac = cnt / total * 100 if total > 0 else 0.0
        print(f"  {expert:>5}: {cnt:4d} examples ({frac:5.1f}%)")

    if best_f1_values:
        avg_best_f1 = sum(best_f1_values) / len(best_f1_values)
        print(f"\nAverage best F1 over all examples: {avg_best_f1:.2f}")
    else:
        print("\nNo best_f1 values found (file may be empty).")


if __name__ == "__main__":
    main()
