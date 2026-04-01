# eval_router_all_offline_utility.py
# ------------------------------------------------------------
# Offline evaluation: for each dataset, run the trained router on
# router_train_*.jsonl and compare chosen expert vs oracle (utility-best).
#
# Compatible with checkpoints that may use either:
#   - head_experts.*  (older)
#   - head_scores.*   (newer ranker version)
#
# Prints:
#   - Router matches utility-best expert (accuracy)
#   - Chosen avg F1/EM/looseEM
#   - Oracle avg F1/EM/looseEM
# ------------------------------------------------------------

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

CFG_PATH = Path("configs/router_config.json")
RESULTS_DIR = Path("results")
FEAT_DIM = 5


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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


def _log1p_cost(x: float) -> float:
    return math.log1p(max(0.0, float(x)))


def _latency_cap_seconds(util_cfg: Dict[str, Any], dataset_name: str, expert_name: str) -> float:
    caps = util_cfg.get("latency_caps_seconds", {}) or {}
    default_cap = float(caps.get("default", 0.0) or 0.0)

    by_ds = caps.get("by_dataset", {}) or {}
    ds_caps = by_ds.get(dataset_name, {}) or {}
    ds_default = float(ds_caps.get("default", default_cap) or default_cap)

    # expert override inside dataset
    return float(ds_caps.get(expert_name, ds_default) or ds_default)


def compute_utility(metrics: Dict[str, Any], util_cfg: Dict[str, Any], dataset_name: str, expert_name: str) -> float:
    """
    utility = alpha*f1 + beta*em + gamma*loose_em
              - lambda*cost(latency) - mu*cost(vram)

    Supports:
      - old keys: alpha_f1, beta_em, gamma_loose_em, lambda_latency, mu_vram
      - new keys: ALPHA_F1, BETA_EM, GAMMA_LOOSE, LAMBDA_LAT, MU_VRAM
      - COST_FN: "log1p" (default) or "linear"
      - latency_caps_seconds in util_cfg (optional)
    """
    f1 = get_metric(metrics, "f1")
    em = get_metric(metrics, "em")
    loose = get_metric(metrics, "loose_em")
    lat = get_metric(metrics, "latency")
    vram = get_metric(metrics, "vram_mb")

    # accept multiple naming styles
    alpha = float(util_cfg.get("ALPHA_F1", util_cfg.get("alpha_f1", 1.0)))
    beta = float(util_cfg.get("BETA_EM", util_cfg.get("beta_em", 0.5)))
    gamma = float(util_cfg.get("GAMMA_LOOSE", util_cfg.get("gamma_loose_em", util_cfg.get("gamma_loose", 0.0))))
    lam = float(util_cfg.get("LAMBDA_LAT", util_cfg.get("lambda_latency", util_cfg.get("lam", 0.0))))
    mu = float(util_cfg.get("MU_VRAM", util_cfg.get("mu_vram", util_cfg.get("mu", 0.0))))

    cost_fn = util_cfg.get("COST_FN", util_cfg.get("cost_fn", "log1p"))

    # Apply latency cap (if configured)
    cap = _latency_cap_seconds(util_cfg, dataset_name, expert_name)
    lat_used = min(float(lat), cap) if (cap and cap > 0) else float(lat)

    acc = (alpha * f1) + (beta * em) + (gamma * loose)

    if str(cost_fn).lower() == "linear":
        cost = (lam * lat_used) + (mu * float(vram))
    else:
        cost = (lam * _log1p_cost(lat_used)) + (mu * _log1p_cost(vram))

    return acc - cost



class GatedMultiEmbedRouter(nn.Module):
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

        # IMPORTANT: keep name stable for loading
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
    Accept checkpoints that used head_scores.* (newer) instead of head_experts.* (older).
    """
    sd = dict(state_dict)
    if "head_scores.weight" in sd and "head_experts.weight" not in sd:
        sd["head_experts.weight"] = sd.pop("head_scores.weight")
        sd["head_experts.bias"] = sd.pop("head_scores.bias")
    return sd


def eval_one_dataset(dataset_name: str, ds_cfg: Dict[str, Any], util_cfg: Dict[str, Any]) -> Tuple[int, float]:
    model_path = RESULTS_DIR / f"router_{dataset_name}_utility_ranker.pt"
    if not model_path.exists():
        raise SystemExit(f"[{dataset_name}] Missing model: {model_path} (run train_router_all_utility_ranker.py)")

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    expert_names = ckpt["expert_names"]

    embed_names = _safe_get_list(
        ckpt,
        keys=["embed_models_used", "embed_models_requested", "embed_names", "embed_models"],
        default=[],
    )
    if not embed_names:
        embed_names = ["sentence-transformers/all-mpnet-base-v2"]

    embed_dims = ckpt["embed_dims"]

    model = GatedMultiEmbedRouter(
        embed_dims=embed_dims,
        feat_dim=int(ckpt.get("feat_dim", FEAT_DIM)),
        proj_dim=int(ckpt["proj_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        out_dim=int(ckpt["out_dim"]),
        dropout=float(ckpt["dropout"]),
    )

    sd = _remap_head_keys(ckpt["state_dict"])
    model.load_state_dict(sd, strict=True)
    model.eval()

    embedders = [SentenceTransformer(n) for n in embed_names]
    rows = read_jsonl(Path(ds_cfg["router_train_out"]))

    n = 0
    match_oracle = 0

    chosen_f1, chosen_em, chosen_loose = [], [], []
    oracle_f1, oracle_em, oracle_loose = [], [], []

    for r in rows:
        q = r.get("question") or ""
        if not q:
            continue
        block = r.get("experts", {})
        if not isinstance(block, dict) or not block:
            continue
        if any(e not in block for e in expert_names):
            continue

        embs = [e.encode(q, normalize_embeddings=True).astype(np.float32) for e in embedders]
        feats = basic_text_features(q).astype(np.float32)

        enc_t = [torch.from_numpy(v).unsqueeze(0) for v in embs]
        feats_t = torch.from_numpy(feats).unsqueeze(0)

        with torch.no_grad():
            logits_t = model(enc_t, feats_t).squeeze(0)
            logits = logits_t.detach().cpu().numpy()

        pick = expert_names[int(np.argmax(logits))]

        # Oracle = best by UTILITY
        best = max(expert_names, key=lambda e: compute_utility(block[e], util_cfg, dataset_name, e))




        if pick == best:
            match_oracle += 1
        n += 1

        chosen_f1.append(get_metric(block[pick], "f1"))
        chosen_em.append(get_metric(block[pick], "em"))
        chosen_loose.append(get_metric(block[pick], "loose_em"))

        oracle_f1.append(get_metric(block[best], "f1"))
        oracle_em.append(get_metric(block[best], "em"))
        oracle_loose.append(get_metric(block[best], "loose_em"))

    print(f"\n--- {dataset_name} ---")
    print("N =", n)
    print("Router matches utility-best expert:", match_oracle / max(1, n))
    print("Chosen avg F1     :", float(np.mean(chosen_f1)) if chosen_f1 else 0.0)
    print("Chosen avg EM     :", float(np.mean(chosen_em)) if chosen_em else 0.0)
    print("Chosen avg looseEM:", float(np.mean(chosen_loose)) if chosen_loose else 0.0)
    print("Oracle avg F1     :", float(np.mean(oracle_f1)) if oracle_f1 else 0.0)
    print("Oracle avg EM     :", float(np.mean(oracle_em)) if oracle_em else 0.0)
    print("Oracle avg looseEM:", float(np.mean(oracle_loose)) if oracle_loose else 0.0)

    return n, match_oracle / max(1, n)


def main():
    if not CFG_PATH.exists():
        raise SystemExit(f"Missing {CFG_PATH}")

    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8-sig"))

    # Prefer utility weights saved inside each checkpoint if present (most consistent),
    # otherwise fall back to configs/router_config.json utility block.
    default_util_cfg = cfg.get("utility", {})

    print("\n==================== OFFLINE ROUTER EVAL (PER-DATASET MODELS) ====================")

    total_n = 0
    weighted_acc = 0.0

    for dataset_name, ds_cfg in cfg["datasets"].items():
        # load ckpt utility config if available
        ckpt_path = RESULTS_DIR / f"router_{dataset_name}_utility_ranker.pt"
        util_cfg = default_util_cfg
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            util_from_ckpt = ckpt.get("utility_weights", ckpt.get("utility_cfg", {}))
            util_cfg = dict(default_util_cfg)
            util_cfg.update(util_from_ckpt)  # ckpt overrides weights, default keeps caps

        n, acc = eval_one_dataset(dataset_name, ds_cfg, util_cfg)
        total_n += n
        weighted_acc += acc * n

    print("\n--- OVERALL ---")
    print("Total N =", total_n)
    print("Overall (weighted) match utility-best expert:", weighted_acc / max(1, total_n))


if __name__ == "__main__":
    main()
