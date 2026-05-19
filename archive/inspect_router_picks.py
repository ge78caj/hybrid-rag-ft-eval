# inspect_router_picks.py
# ------------------------------------------------------------
# Loads router checkpoints and prints:
#  - per-dataset expert pick distribution
#  - mean logits and mean softmax probabilities
#
# Backward/forward compatible with checkpoints that may use:
#  - head_experts.*  (older)
#  - head_scores.*   (newer ranking version)
# ------------------------------------------------------------

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

CFG_PATH = Path("configs/router_config.json")
RESULTS_DIR = Path("results")
FEAT_DIM = 5


def basic_text_features(q: str) -> np.ndarray:
    ql = len(q)
    nw = len(q.split())
    has_yesno = int(any(x in q.lower() for x in ["yes or no", "yes/no", "true or false"]))
    wh = int(any(q.lower().startswith(x) for x in ["what", "who", "when", "where", "why", "how"]))
    has_number = int(any(c.isdigit() for c in q))
    return np.array([ql, nw, has_yesno, wh, has_number], dtype=np.float32)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class GatedMultiEmbedRouter(nn.Module):
    """
    Router used by inspect/eval scripts.
    NOTE: We keep the output head name as `head_experts` to stay compatible
    with older scripts; we also remap checkpoint keys if they used head_scores.
    """

    def __init__(self, embed_dims, feat_dim, proj_dim, hidden_dim, out_dim, dropout):
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

        # IMPORTANT: keep this name
        self.head_experts = nn.Linear(hidden_dim, out_dim)

    def forward(self, enc_embs, feats):
        proj_embs = [self.proj[i](enc_embs[i]) for i in range(self.n_enc)]
        ref = proj_embs[0]
        gate_w = F.softmax(self.gate(torch.cat([ref, feats], dim=1)), dim=1)
        mixed = (torch.stack(proj_embs, dim=1) * gate_w.unsqueeze(-1)).sum(dim=1)
        h = self.trunk(torch.cat([mixed, feats], dim=1))
        return self.head_experts(h)


def _safe_get_list(ckpt: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in ckpt and ckpt[k]:
            return ckpt[k]
    return default


def _remap_head_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Make checkpoints compatible when the training script used a different head name.
    - Newer: head_scores.{weight,bias}
    - Older/this script: head_experts.{weight,bias}
    """
    sd = dict(state_dict)  # shallow copy

    if "head_scores.weight" in sd and "head_experts.weight" not in sd:
        sd["head_experts.weight"] = sd.pop("head_scores.weight")
        sd["head_experts.bias"] = sd.pop("head_scores.bias")

    # (If someday the reverse happens, handle it too)
    if "head_experts.weight" in sd and "head_scores.weight" not in sd:
        # no-op for this script, but kept for completeness
        pass

    return sd


def main():
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8-sig"))

    for dataset_name, ds_cfg in cfg["datasets"].items():
        ckpt_path = RESULTS_DIR / f"router_{dataset_name}_utility_ranker.pt"
        if not ckpt_path.exists():
            print(f"[WARN] missing {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        expert_names = ckpt["expert_names"]

        embed_names = _safe_get_list(
            ckpt,
            keys=["embed_models_used", "embed_models_requested", "embed_names", "embed_models"],
            default=[],
        )
        if not embed_names:
            embed_names = ["sentence-transformers/all-mpnet-base-v2"]

        embed_dims = ckpt["embed_dims"]

        proj_dim = int(ckpt["proj_dim"])
        hidden_dim = int(ckpt["hidden_dim"])
        dropout = float(ckpt["dropout"])
        out_dim = int(ckpt["out_dim"])

        embedders = [SentenceTransformer(n) for n in embed_names]

        model = GatedMultiEmbedRouter(embed_dims, FEAT_DIM, proj_dim, hidden_dim, out_dim, dropout)

        # ---- key fix: accept head_scores checkpoints ----
        sd = _remap_head_keys(ckpt["state_dict"])
        model.load_state_dict(sd, strict=True)
        model.eval()

        rows = read_jsonl(Path(ds_cfg["router_train_out"]))

        counts = {e: 0 for e in expert_names}
        n = 0

        sum_logits = np.zeros((len(expert_names),), dtype=np.float64)
        sum_probs = np.zeros((len(expert_names),), dtype=np.float64)

        for r in rows:
            q = r.get("question") or ""
            if not q:
                continue

            embs = [e.encode(q, normalize_embeddings=True).astype(np.float32) for e in embedders]
            feats = basic_text_features(q).astype(np.float32)

            enc_t = [torch.from_numpy(v).unsqueeze(0) for v in embs]
            feats_t = torch.from_numpy(feats).unsqueeze(0)

            with torch.no_grad():
                logits_t = model(enc_t, feats_t).squeeze(0)  # [E]
                probs_t = torch.softmax(logits_t, dim=0)     # [E]

            logits = logits_t.detach().cpu().numpy()
            probs = probs_t.detach().cpu().numpy()

            sum_logits += logits
            sum_probs += probs

            pick = expert_names[int(np.argmax(logits))]
            counts[pick] += 1
            n += 1

        print(f"\n=== Router pick distribution: {dataset_name} ===")
        for e, c in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{e:>10}: {c:4d} ({(c/max(1,n))*100:5.1f}%)")
        print("TOTAL:", n)

        if n > 0:
            mean_probs = sum_probs / n
            mean_logits = sum_logits / n

            print("\nMean softmax probs:")
            for e, p in sorted(zip(expert_names, mean_probs), key=lambda x: x[1], reverse=True):
                print(f"{e:>10}: {p:.4f}")

            print("\nMean logits:")
            for e, l in sorted(zip(expert_names, mean_logits), key=lambda x: x[1], reverse=True):
                print(f"{e:>10}: {l:.4f}")


if __name__ == "__main__":
    main()
