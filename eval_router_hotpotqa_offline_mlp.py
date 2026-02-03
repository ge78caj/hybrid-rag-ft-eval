import json
from pathlib import Path
from collections import Counter

import joblib
from sentence_transformers import SentenceTransformer

from metrics import compute_em_f1

# Paths
ROUTER_PATH = Path("methods/multi_lora/router_hotpotqa_mlp_embed.joblib")
ROUTER_TRAIN_FILE = Path("prediction/router_train_hotpotqa.jsonl")

# ⚠️ MUST match the training script
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_router(path: Path):
    print(f"Loading MLP router model from: {path}")
    router = joblib.load(path)
    return router


def load_router_data(path: Path):
    print(f"Reading router labels from: {path}")
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj)
    print(f"Total examples: {len(data)}")
    return data


def main():
    data = load_router_data(ROUTER_TRAIN_FILE)
    router = load_router(ROUTER_PATH)

    expert_names = list(data[0]["experts"].keys())
    print(f"Experts in this file: {expert_names}")

    # 1) Prepare questions
    questions = [ex["question"] for ex in data]

    # 2) Encode questions to embeddings (same encoder as in training)
    print(f"Encoding {len(questions)} questions with {EMBED_MODEL_NAME} ...")
    encoder = SentenceTransformer(EMBED_MODEL_NAME)
    question_embs = encoder.encode(questions, batch_size=32, show_progress_bar=True)
    # question_embs shape: (N, D) -> exactly what MLP expects

    # 3) Let the router choose an expert for each question
    router_choices = router.predict(question_embs)

    # 4) Baselines: always pick each expert
    def eval_always(expert_name: str):
        ems = []
        f1s = []
        for ex in data:
            scores = ex["experts"][expert_name]
            ems.append(scores["em"])
            f1s.append(scores["f1"])
        return sum(ems) / len(ems), sum(f1s) / len(f1s)

    print("\n=== Offline HotpotQA router evaluation (MLP router, using stored EM/F1) ===")
    for expert_name in expert_names:
        em, f1 = eval_always(expert_name)
        print(f"Always {expert_name:8s} -> EM: {em*100:6.2f}%, F1: {f1*100:6.2f}%")

    # 5) Oracle (per-question best F1)
    oracle_ems = []
    oracle_f1s = []
    for ex in data:
        best_f1 = -1.0
        best_em = 0.0
        for name, scores in ex["experts"].items():
            if scores["f1"] > best_f1:
                best_f1 = scores["f1"]
                best_em = scores["em"]
        oracle_ems.append(best_em)
        oracle_f1s.append(best_f1)

    oracle_em = sum(oracle_ems) / len(oracle_ems)
    oracle_f1 = sum(oracle_f1s) / len(oracle_f1s)
    print(f"Oracle (per q)  -> EM: {oracle_em*100:6.2f}%, F1: {oracle_f1*100:6.2f}%")

    # 6) Router decision (MLP-based)
    router_ems = []
    router_f1s = []
    choice_counts = Counter()

    for ex, choice in zip(data, router_choices):
        scores = ex["experts"][choice]
        router_ems.append(scores["em"])
        router_f1s.append(scores["f1"])
        choice_counts[choice] += 1

    router_em = sum(router_ems) / len(router_ems)
    router_f1 = sum(router_f1s) / len(router_f1s)
    print(f"Router decision -> EM: {router_em*100:6.2f}%, F1: {router_f1*100:6.2f}%\n")

    print("Router choice distribution:")
    for name in expert_names:
        count = choice_counts[name]
        print(f"  {name:8s}: {count:3d} ({count/len(data)*100:4.1f}%)")


if __name__ == "__main__":
    main()
