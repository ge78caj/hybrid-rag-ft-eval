import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit("Missing sentence-transformers. Install: pip install sentence-transformers")


# -----------------------
# INPUTS
# -----------------------
ROUTER_TRAIN_FILES = {
    "hotpotqa": Path("prediction/router_train_hotpotqa.jsonl"),
    "squad_v2": Path("prediction/router_train_squad_v2.jsonl"),
    "pubmedqa_v2": Path("prediction/router_train_pubmedqa_v2.jsonl"),
}

MODEL_PATH = Path("results/router_utility_ranker_all.pt")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_expert_block(row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    block = row.get("experts")
    if isinstance(block, dict) and block:
        return block
    raise KeyError("Could not find 'experts' dict in row (schema mismatch).")


def get_metric(v: Dict[str, Any], name: str) -> float:
    aliases = {
        "f1": ["f1", "F1"],
        "em": ["em", "exact_match", "EM"],
        "loose_em": ["loose_em", "looseEM", "loose_exact_match", "lem"],
        "latency": ["latency", "time"],
        "vram_mb": ["vram_mb", "peak_vram_mb"],
    }
    for key in aliases.get(name, [name]):
        if key in v:
            try:
                x = v[key]
                if x is None:
                    return 0.0
                return float(x)
            except Exception:
                return 0.0
    return 0.0


def basic_text_features(q: str) -> np.ndarray:
    ql = len(q)
    nw = len(q.split())
    has_yesno = int(any(x in q.lower() for x in ["yes or no", "yes/no", "true or false"]))
    wh = int(any(q.lower().startswith(x) for x in ["what", "who", "when", "where", "why", "how"]))
    has_number = int(any(c.isdigit() for c in q))
    return np.array([ql, nw, has_yesno, wh, has_number], dtype=np.float32)


class MLPScoreRouter(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"Missing {MODEL_PATH}. Train first: python train_router_utility_ranker_all.py")

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    expert_names = ckpt["expert_names"]
    in_dim = ckpt["in_dim"]
    out_dim = ckpt["out_dim"]

    embed_by_ds = ckpt.get("embed_model_by_dataset", None)
    if not isinstance(embed_by_ds, dict) or not embed_by_ds:
        raise SystemExit(
            "Checkpoint does not contain embed_model_by_dataset. "
            "Make sure you trained with train_router_utility_ranker_all.py"
        )

    model = MLPScoreRouter(in_dim, out_dim)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Build embedders per dataset (don’t reload for every row)
    embedders = {ds: SentenceTransformer(name) for ds, name in embed_by_ds.items()}

    # Aggregate stats
    total_n = 0
    total_acc_best = 0

    per_ds_stats = {}

    for ds_name, path in ROUTER_TRAIN_FILES.items():
        if not path.exists():
            print(f"[WARN] Missing {path}, skipping {ds_name}")
            continue

        rows = read_jsonl(path)
        embedder = embedders.get(ds_name)
        if embedder is None:
            print(f"[WARN] No embedder found for dataset={ds_name}, skipping")
            continue

        chosen_f1, chosen_em, chosen_loose = [], [], []
        best_f1, best_em, best_loose = [], [], []
        acc_best = 0
        n = 0

        for r in rows:
            q = r.get("question") or r.get("q") or r.get("query")
            if not q:
                continue

            block = find_expert_block(r)
            if any(e not in block for e in expert_names):
                continue

            emb = embedder.encode(q, normalize_embeddings=True).astype(np.float32)
            feats = basic_text_features(q)
            x = np.concatenate([emb, feats], axis=0).astype(np.float32)

            # sanity check: dimensions must match training
            if x.shape[0] != in_dim:
                raise SystemExit(
                    f"Feature dim mismatch for {ds_name}: got {x.shape[0]} but model expects {in_dim}.\n"
                    f"Likely you trained with a different embedder dim or changed features."
                )

            x_t = torch.from_numpy(x).unsqueeze(0)

            with torch.no_grad():
                pred_scores = model(x_t).squeeze(0).numpy()

            pick_idx = int(np.argmax(pred_scores))
            pick_expert = expert_names[pick_idx]

            # Oracle best by F1 (offline upper bound)
            true_best = max(expert_names, key=lambda e: get_metric(block[e], "f1"))

            if pick_expert == true_best:
                acc_best += 1

            n += 1
            chosen_f1.append(get_metric(block[pick_expert], "f1"))
            chosen_em.append(get_metric(block[pick_expert], "em"))
            chosen_loose.append(get_metric(block[pick_expert], "loose_em"))

            best_f1.append(get_metric(block[true_best], "f1"))
            best_em.append(get_metric(block[true_best], "em"))
            best_loose.append(get_metric(block[true_best], "loose_em"))

        if n == 0:
            print(f"[WARN] No usable rows for {ds_name}. Check schema/expert names.")
            continue

        stats = {
            "N": n,
            "acc_match_f1_best": acc_best / n,
            "chosen_f1": float(np.mean(chosen_f1)),
            "chosen_em": float(np.mean(chosen_em)),
            "chosen_loose_em": float(np.mean(chosen_loose)),
            "oracle_f1": float(np.mean(best_f1)),
            "oracle_em": float(np.mean(best_em)),
            "oracle_loose_em": float(np.mean(best_loose)),
        }
        per_ds_stats[ds_name] = stats

        total_n += n
        total_acc_best += acc_best

    # Print results
    print("\n==================== OFFLINE ROUTER EVAL (ALL DATASETS) ====================")
    for ds_name, s in per_ds_stats.items():
        print(f"\n--- {ds_name} ---")
        print("N =", s["N"])
        print("Router matches F1-best expert (accuracy):", s["acc_match_f1_best"])
        print("Chosen avg F1     :", s["chosen_f1"])
        print("Chosen avg EM     :", s["chosen_em"])
        print("Chosen avg looseEM:", s["chosen_loose_em"])
        print("Oracle avg F1     :", s["oracle_f1"])
        print("Oracle avg EM     :", s["oracle_em"])
        print("Oracle avg looseEM:", s["oracle_loose_em"])

    if total_n > 0:
        print("\n--- OVERALL ---")
        print("Total N =", total_n)
        print("Overall router matches F1-best expert (accuracy):", total_acc_best / total_n)
    else:
        print("[ERROR] No datasets evaluated.")


if __name__ == "__main__":
    main()
