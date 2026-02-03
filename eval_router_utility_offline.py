import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit("Missing sentence-transformers. Install: pip install sentence-transformers")

CFG_PATH = Path("configs/router_config.json")
RESULTS_DIR = Path("results")
FEAT_DIM = 5


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def basic_text_features(q: str) -> np.ndarray:
    ql = len(q)
    nw = len(q.split())
    has_yesno = int(any(x in q.lower() for x in ["yes or no", "yes/no", "true or false"]))
    wh = int(any(q.lower().startswith(x) for x in ["what", "who", "when", "where", "why", "how"]))
    has_number = int(any(c.isdigit() for c in q))
    return np.array([ql, nw, has_yesno, wh, has_number], dtype=np.float32)


def get_metric(v: Dict[str, Any], name: str) -> float:
    aliases = {
        "f1": ["f1", "F1"],
        "em": ["em", "exact_match", "EM"],
        "loose_em": ["loose_em", "looseEM", "loose_exact_match", "lem"],
    }
    for key in aliases.get(name, [name]):
        if key in v:
            try:
                return float(v[key])
            except Exception:
                return 0.0
    return 0.0


def _try_load_embedder(name: str):
    try:
        return SentenceTransformer(name)
    except Exception as e:
        print(f"[WARN] Failed to load embedder '{name}': {e}")
        return None


def load_embedder_list(names: List[str]) -> Tuple[List[str], List[SentenceTransformer]]:
    used_names, used = [], []
    for n in names:
        m = _try_load_embedder(n)
        if m is not None:
            used_names.append(n)
            used.append(m)
    if not used:
        fallback = "sentence-transformers/all-mpnet-base-v2"
        m = _try_load_embedder(fallback)
        if m is None:
            raise SystemExit("Could not load any embedder (even fallback).")
        used_names, used = [fallback], [m]
    return used_names, used


class GatedMultiEmbedRouter(nn.Module):
    def __init__(
            self,
            embed_dims: List[int],
            feat_dim: int,
            proj_dim: int,
            hidden_dim: int,
            out_dim: int,
            dropout: float,
    ):
        super().__init__()
        self.n_enc = len(embed_dims)
        self.proj = nn.ModuleList([nn.Linear(d, proj_dim) for d in embed_dims])

        self.gate = nn.Sequential(
            nn.Linear(proj_dim + feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.n_enc),
        )

        self.trunk = nn.Sequential(
            nn.Linear(proj_dim + feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head_experts = nn.Linear(hidden_dim, out_dim)

    def forward(self, enc_embs: List[torch.Tensor], feats: torch.Tensor):
        proj_embs = [self.proj[i](enc_embs[i]) for i in range(self.n_enc)]
        ref = proj_embs[0]
        gate_in = torch.cat([ref, feats], dim=1)
        gate_w = F.softmax(self.gate(gate_in), dim=1)

        stacked = torch.stack(proj_embs, dim=1)
        mixed = (stacked * gate_w.unsqueeze(-1)).sum(dim=1)

        x = torch.cat([mixed, feats], dim=1)
        h = self.trunk(x)
        logits = self.head_experts(h)
        return logits, gate_w


def eval_one_dataset(dataset_name: str, router_train_path: str):
    model_path = RESULTS_DIR / f"router_{dataset_name}_utility_ranker.pt"
    if not model_path.exists():
        raise SystemExit(f"[{dataset_name}] Missing model: {model_path} (run train_router_all_utility_ranker.py)")

    ckpt = torch.load(model_path, map_location="cpu")

    expert_names = ckpt["expert_names"]
    out_dim = int(ckpt["out_dim"])
    embed_names = ckpt.get("embed_models_used", ckpt.get("embed_models_requested", []))
    embed_dims = ckpt["embed_dims"]
    proj_dim = int(ckpt["proj_dim"])
    hidden_dim = int(ckpt["hidden_dim"])
    dropout = float(ckpt["dropout"])

    _, embedders = load_embedder_list(embed_names)

    model = GatedMultiEmbedRouter(
        embed_dims=embed_dims,
        feat_dim=FEAT_DIM,
        proj_dim=proj_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        dropout=dropout,
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    rows = read_jsonl(Path(router_train_path))

    chosen_f1, chosen_em, chosen_loose = [], [], []
    oracle_f1, oracle_em, oracle_loose = [], [], []

    match_best = 0
    n = 0

    for r in rows:
        q = r.get("question") or ""
        if not q:
            continue
        block = r.get("experts", {})

        # ensure only experts used in training are evaluated
        if any(e not in block for e in expert_names):
            continue

        embs = [e.encode(q, normalize_embeddings=True).astype(np.float32) for e in embedders]
        feats = basic_text_features(q).astype(np.float32)

        enc_t = [torch.from_numpy(v).unsqueeze(0) for v in embs]
        feats_t = torch.from_numpy(feats).unsqueeze(0)

        with torch.no_grad():
            logits, _ = model(enc_t, feats_t)
            pred_scores = logits.squeeze(0).numpy()

        pick_idx = int(np.argmax(pred_scores))
        pick_expert = expert_names[pick_idx]

        best_expert = max(expert_names, key=lambda e: get_metric(block[e], "f1"))

        if pick_expert == best_expert:
            match_best += 1
        n += 1

        chosen_f1.append(get_metric(block[pick_expert], "f1"))
        chosen_em.append(get_metric(block[pick_expert], "em"))
        chosen_loose.append(get_metric(block[pick_expert], "loose_em"))

        oracle_f1.append(get_metric(block[best_expert], "f1"))
        oracle_em.append(get_metric(block[best_expert], "em"))
        oracle_loose.append(get_metric(block[best_expert], "loose_em"))

    if n == 0:
        raise SystemExit(f"[{dataset_name}] No usable rows found in {router_train_path}")

    print(f"\n--- {dataset_name} ---")
    print("N =", n)
    print("Router matches F1-best expert (accuracy):", match_best / n)
    print("Chosen avg F1     :", float(np.mean(chosen_f1)))
    print("Chosen avg EM     :", float(np.mean(chosen_em)))
    print("Chosen avg looseEM:", float(np.mean(chosen_loose)))
    print("Oracle avg F1     :", float(np.mean(oracle_f1)))
    print("Oracle avg EM     :", float(np.mean(oracle_em)))
    print("Oracle avg looseEM:", float(np.mean(oracle_loose)))
    return n, match_best


def main():
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8-sig"))

    print("\n==================== OFFLINE ROUTER EVAL (PER-DATASET MODELS) ====================")

    total_n = 0
    total_match = 0

    for dataset_name, ds_cfg in cfg["datasets"].items():
        n, match = eval_one_dataset(dataset_name, ds_cfg["router_train_out"])
        total_n += n
        total_match += match

    print("\n--- OVERALL ---")
    print("Total N =", total_n)
    print("Overall router matches F1-best expert (accuracy):", total_match / total_n)


if __name__ == "__main__":
    main()
