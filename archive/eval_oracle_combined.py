import json
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List

# Path to your router labels file
ROUTER_PATH = Path("prediction") / "router_train_hotpotqa.jsonl"


# --- Simple normalization & loose EM -------------------------

ARTICLES = {"a", "an", "the"}


def normalize_text(s: str) -> str:
    """Roughly mimic the Hotpot-style normalization."""
    s = s.lower()

    # Strip <answer> tags if they survived
    s = s.replace("<answer>", " ").replace("</answer>", " ")

    # Remove punctuation -> spaces
    cleaned_chars = []
    for ch in s:
        if ch.isalnum() or ch.isspace():
            cleaned_chars.append(ch)
        else:
            cleaned_chars.append(" ")
    s = "".join(cleaned_chars)

    # Collapse spaces, drop articles
    tokens = [t for t in s.split() if t not in ARTICLES]
    return " ".join(tokens)


def loose_em(pred: str, gold_answers: List[str]) -> float:
    """
    Loose EM:
      1 if (normalized) gold is contained in normalized prediction
      OR prediction is contained in gold.
    Otherwise, 0.
    """
    if not gold_answers:
        return 0.0

    pred_n = normalize_text(pred)
    if not pred_n:
        return 0.0

    for g in gold_answers:
        g_n = normalize_text(g)
        if not g_n:
            continue
        if g_n == pred_n or g_n in pred_n or pred_n in g_n:
            return 1.0

    return 0.0


# --- Loading & oracle computations --------------------------


def load_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def compute_oracle_scores(records: List[Dict[str, Any]]) -> None:
    # We will compute 3 oracles:
    #  - by F1
    #  - by EM
    #  - by combined metric: 0.5 * F1 + 0.3 * EM + 0.2 * loose_EM

    alpha, beta, gamma = 0.5, 0.3, 0.2  # weights for F1, EM, loose_EM

    # Accumulators
    sum_em_f1_oracle = sum_f1_f1_oracle = 0.0
    sum_em_em_oracle = sum_f1_em_oracle = 0.0
    sum_em_comb_oracle = sum_f1_comb_oracle = 0.0

    dist_f1_oracle = Counter()
    dist_em_oracle = Counter()
    dist_comb_oracle = Counter()

    n = len(records)

    for rec in records:
        gold_answers = rec.get("gold_answer", [])
        experts: Dict[str, Dict[str, Any]] = rec["experts"]

        # compute loose_EM & combined score on the fly
        for name, info in experts.items():
            pred_text = info["prediction_text"]
            em = float(info["em"])
            f1 = float(info["f1"])
            lem = loose_em(pred_text, gold_answers)
            info["loose_em"] = lem
            info["combined"] = alpha * f1 + beta * em + gamma * lem

        # Best by F1
        best_f1_name, best_f1_info = max(
            experts.items(), key=lambda kv: kv[1]["f1"]
        )
        sum_em_f1_oracle += best_f1_info["em"]
        sum_f1_f1_oracle += best_f1_info["f1"]
        dist_f1_oracle[best_f1_name] += 1

        # Best by EM
        best_em_name, best_em_info = max(
            experts.items(), key=lambda kv: kv[1]["em"]
        )
        sum_em_em_oracle += best_em_info["em"]
        sum_f1_em_oracle += best_em_info["f1"]
        dist_em_oracle[best_em_name] += 1

        # Best by combined score
        best_comb_name, best_comb_info = max(
            experts.items(), key=lambda kv: kv[1]["combined"]
        )
        sum_em_comb_oracle += best_comb_info["em"]
        sum_f1_comb_oracle += best_comb_info["f1"]
        dist_comb_oracle[best_comb_name] += 1

    def pct(x: float) -> float:
        return 100.0 * x / n

    # --- Print results ---

    print(f"Total examples: {n}\n")

    print("=== Oracle 1: choose best expert by F1 ===")
    print(f"EM: {pct(sum_em_f1_oracle):.2f}%")
    print(f"F1: {pct(sum_f1_f1_oracle):.2f}%")
    print("Choice distribution (F1-based):")
    for name, cnt in dist_f1_oracle.most_common():
        print(f"  {name:8s}: {cnt:4d} ({100.0*cnt/n:5.1f}%)")
    print()

    print("=== Oracle 2: choose best expert by EM ===")
    print(f"EM: {pct(sum_em_em_oracle):.2f}%")
    print(f"F1: {pct(sum_f1_em_oracle):.2f}%")
    print("Choice distribution (EM-based):")
    for name, cnt in dist_em_oracle.most_common():
        print(f"  {name:8s}: {cnt:4d} ({100.0*cnt/n:5.1f}%)")
    print()

    print("=== Oracle 3: choose best expert by combined (0.5*F1 + 0.3*EM + 0.2*loose_EM) ===")
    print(f"EM: {pct(sum_em_comb_oracle):.2f}%")
    print(f"F1: {pct(sum_f1_comb_oracle):.2f}%")
    print("Choice distribution (combined-based):")
    for name, cnt in dist_comb_oracle.most_common():
        print(f"  {name:8s}: {cnt:4d} ({100.0*cnt/n:5.1f}%)")
    print()


def main():
    if not ROUTER_PATH.exists():
        raise FileNotFoundError(f"router labels not found at {ROUTER_PATH}")
    records = load_records(ROUTER_PATH)
    compute_oracle_scores(records)


if __name__ == "__main__":
    main()
