import json
from pathlib import Path
from collections import Counter

import joblib
from metrics import compute_em_f1   # only for sanity if needed

DATA_PATH = Path("prediction/router_train_hotpotqa.jsonl")
MODEL_PATH = Path("methods/multi_lora/router_hotpotqa_tfidf.joblib")


def load_router_data(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(obj)
    return records


def main():
    print(f"Reading router labels from: {DATA_PATH}")
    records = load_router_data(DATA_PATH)
    print(f"Total examples: {len(records)}")

    model_bundle = joblib.load(MODEL_PATH)
    vectorizer = model_bundle["vectorizer"]
    clf = model_bundle["clf"]
    print(f"Loading router model from: {MODEL_PATH}")

    # Collect all expert names present in the file
    all_experts = sorted(list(records[0]["experts"].keys()))
    print(f"Experts in this file: {all_experts}")

    # Prepare text + oracle labels and per-expert F1
    questions = []
    oracle_labels = []
    per_expert_f1 = {exp: [] for exp in all_experts}

    for rec in records:
        questions.append(rec["question"])
        oracle_labels.append(rec["best_expert"])

        for exp in all_experts:
            per_expert_f1[exp].append(rec["experts"][exp]["f1"])

    # Compute EM/F1 for "always choose expert X"
    print("\n=== Offline HotpotQA router evaluation (using stored EM/F1) ===")
    for exp in all_experts:
        f1_list = per_expert_f1[exp]
        avg_f1 = 100.0 * sum(f1_list) / len(f1_list)
        # EM: fraction of examples where that expert has em==1
        em_list = [rec["experts"][exp]["em"] for rec in records]
        avg_em = 100.0 * sum(em_list) / len(em_list)
        print(f"Always {exp:10s} -> EM: {avg_em:5.2f}%, F1: {avg_f1:5.2f}%")

    # Oracle: per question, choose expert with highest F1
    oracle_f1 = 100.0 * sum(rec["best_f1"] for rec in records) / len(records)
    oracle_em = 100.0 * sum(
        rec["experts"][rec["best_expert"]]["em"] for rec in records
    ) / len(records)
    print(f"Oracle (per q)  -> EM: {oracle_em:5.2f}%, F1: {oracle_f1:5.2f}%")

    # Router decision
    X_tfidf = vectorizer.transform(questions)
    router_preds = clf.predict(X_tfidf)

    # Using stored per-expert F1 to score the router’s choices
    router_f1_sum = 0.0
    router_em_sum = 0.0
    for rec, route_exp in zip(records, router_preds):
        router_f1_sum += rec["experts"][route_exp]["f1"]
        router_em_sum += rec["experts"][route_exp]["em"]

    router_f1 = 100.0 * router_f1_sum / len(records)
    router_em = 100.0 * router_em_sum / len(records)
    print(f"Router decision -> EM: {router_em:5.2f}%, F1: {router_f1:5.2f}%")

    # Also show distribution of router choices
    counts = Counter(router_preds)
    print("\nRouter choice distribution:")
    for exp in all_experts:
        print(f"  {exp:10s}: {counts[exp]:3d} ({100.0*counts[exp]/len(records):4.1f}%)")


if __name__ == "__main__":
    main()
