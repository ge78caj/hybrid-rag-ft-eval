import json
from pathlib import Path

LABEL_PATH = Path("prediction") / "router_train_hotpotqa.jsonl"


def main():
    if not LABEL_PATH.exists():
        raise FileNotFoundError(f"Router label file not found: {LABEL_PATH}")

    n = 0

    # Sums for F1-based oracle
    sum_f1_choice_em = 0.0
    sum_f1_choice_f1 = 0.0

    # Sums for EM-based oracle
    sum_em_choice_em = 0.0
    sum_em_choice_f1 = 0.0

    # Optional: distributions (which expert is chosen in each oracle)
    dist_f1_choice = {}
    dist_em_choice = {}

    with LABEL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            experts = obj["experts"]  # dict: expert_name -> {"em": float, "f1": float}

            # --- Oracle 1: choose by F1 ---
            best_expert_f1, best_f1_data = max(
                experts.items(),
                key=lambda kv: kv[1]["f1"],
            )
            sum_f1_choice_em += best_f1_data["em"]
            sum_f1_choice_f1 += best_f1_data["f1"]
            dist_f1_choice[best_expert_f1] = dist_f1_choice.get(best_expert_f1, 0) + 1

            # --- Oracle 2: choose by EM ---
            best_expert_em, best_em_data = max(
                experts.items(),
                key=lambda kv: kv[1]["em"],
            )
            sum_em_choice_em += best_em_data["em"]
            sum_em_choice_f1 += best_em_data["f1"]
            dist_em_choice[best_expert_em] = dist_em_choice.get(best_expert_em, 0) + 1

            n += 1

    def pct(x):
        return 100.0 * x / n

    print(f"Total examples: {n}\n")

    print("=== Oracle 1: choose best expert by F1 ===")
    print(f"EM: {pct(sum_f1_choice_em):.2f}%")
    print(f"F1: {pct(sum_f1_choice_f1):.2f}%")
    print("Choice distribution (F1-based):")
    for name, cnt in sorted(dist_f1_choice.items(), key=lambda x: -x[1]):
        print(f"  {name:9s}: {cnt:4d} ({100.0 * cnt / n:5.1f}%)")

    print("\n=== Oracle 2: choose best expert by EM ===")
    print(f"EM: {pct(sum_em_choice_em):.2f}%")
    print(f"F1: {pct(sum_em_choice_f1):.2f}%")
    print("Choice distribution (EM-based):")
    for name, cnt in sorted(dist_em_choice.items(), key=lambda x: -x[1]):
        print(f"  {name:9s}: {cnt:4d} ({100.0 * cnt / n:5.1f}%)")


if __name__ == "__main__":
    main()
