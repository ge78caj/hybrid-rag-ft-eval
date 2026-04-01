import json
from pathlib import Path
from typing import Dict, Any, List

from metrics import compute_em_f1, extract_prediction


# Folder containing all *_predictions.jsonl files
PRED_DIR = Path("prediction")

"""
Configure the experts here.

Keys = logical expert names (router labels)
Values = filenames inside prediction/

Start with what you have now. Later you just add more entries.
"""
EXPERT_FILES = {
    # base LLaMA-3.1-8B, no RAG
    "base_only": "normal_hotpotqa_False_Llama-3.1-8B-Instruct_predictions.jsonl",

    # base LLaMA-3.1-8B, with RAG
    "base_rag": "normal_hotpotqa_True_Llama-3.1-8B-Instruct_predictions.jsonl",

    # SFT on LLaMA-3.1-8B, no RAG
    "sft_only": "ft_hotpotqa_False_Llama-3.1-8B-Instruct_predictions.jsonl",

    # SFT on LLaMA-3.1-8B, with RAG
    "sft_rag": "ft_hotpotqa_True_Llama-3.1-8B-Instruct_predictions.jsonl",

    # RAFT on LLaMA-3.1-8B, no RAG
    "raft_only": "raft_hotpotqa_False_Llama-3.1-8B-Instruct_predictions.jsonl",

    # RAFT on LLaMA-3.1-8B, with RAG
    "raft_rag": "raft_hotpotqa_True_Llama-3.1-8B-Instruct_predictions.jsonl",
}



def extract_answer_text(prediction_field: str) -> str:
    """
    Use the same extraction logic as metrics.py (extract_prediction),
    so router labeling is aligned with Ryo's evaluation.
    """
    return extract_prediction(prediction_field)


def compute_em_f1_single(prediction_text: str,
                         gold_answers: List[str]) -> Dict[str, float]:
    """
    Wraps compute_em_f1 for a single example.
    Note: compute_em_f1 expects lists of strings.
    """
    gold_list = gold_answers
    pred_list = [prediction_text]
    result = compute_em_f1(gold_list, pred_list)
    return {"em": float(result["em"]), "f1": float(result["f1"])}


def load_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Reads a *_predictions.jsonl file and returns:
        { example_id: {"question": ..., "gold_answer": [...],
                       "prediction_text": ...} }
    """
    data: Dict[str, Dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            ex_id = obj["id"]
            question = obj["question"]
            gold_answer = obj["gold_answer"]  # already a list
            pred_raw = obj["prediction"]

            pred_text = extract_answer_text(pred_raw)

            data[ex_id] = {
                "question": question,
                "gold_answer": gold_answer,
                "prediction_text": pred_text,
            }

    return data


def build_router_labels() -> None:
    # 1) Load predictions for each expert that actually exists
    expert_data: Dict[str, Dict[str, Any]] = {}
    for expert_name, filename in EXPERT_FILES.items():
        path = PRED_DIR / filename
        if not path.exists():
            print(f"[WARN] File for expert '{expert_name}' not found: {path}. Skipping.")
            continue
        print(f"Loading predictions for expert '{expert_name}' from {path} ...")
        expert_data[expert_name] = load_predictions(path)

    if len(expert_data) < 2:
        raise RuntimeError(
            f"Need at least 2 experts to train a router, but only found {len(expert_data)}."
        )

    # 2) Find intersection of IDs across all loaded experts
    expert_ids = [set(d.keys()) for d in expert_data.values()]
    common_ids = set.intersection(*expert_ids)
    print(f"Found {len(common_ids)} common examples across all experts.")

    out_path = PRED_DIR / "router_train_hotpotqa.jsonl"
    n_written = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for ex_id in sorted(common_ids):
            # take question & gold_answer from any expert, they should match
            any_expert_data = next(iter(expert_data.values()))
            question = any_expert_data[ex_id]["question"]
            gold_answer = any_expert_data[ex_id]["gold_answer"]

            experts_result: Dict[str, Any] = {}
            best_expert_name = None
            best_f1 = -1.0

            for expert_name, preds in expert_data.items():
                pred_text = preds[ex_id]["prediction_text"]
                scores = compute_em_f1_single(pred_text, gold_answer)

                experts_result[expert_name] = {
                    "prediction_text": pred_text,
                    "em": scores["em"],
                    "f1": scores["f1"],
                }

                if scores["f1"] > best_f1:
                    best_f1 = scores["f1"]
                    best_expert_name = expert_name

            record = {
                "id": ex_id,
                "question": question,
                "gold_answer": gold_answer,
                "experts": experts_result,
                "best_expert": best_expert_name,
                "best_f1": best_f1,
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} labeled examples to: {out_path}")


if __name__ == "__main__":
    build_router_labels()
