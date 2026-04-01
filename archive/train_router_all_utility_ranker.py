# train_router_all_utility_ranker.py
# ---------------------------------------------------------------------
# Utility-regression + ranking router training (closer to Oracle argmax).
#
# Key fixes vs your current version:
#  1) **Canonical expert order is enforced from router_config.json**
#     (cfg["training"]["allowed_experts"]) and used consistently.
#  2) **No feature normalization mismatch**: we DO NOT normalize text features
#     unless we also export mean/std AND inference scripts apply them.
#     (Your previous "normalize feats in train only" can destroy inference.)
#  3) Embedders are taken strictly from dataset config (one or many).
#  4) The output head is named **head_experts** for backward compatibility
#     with inspect/eval scripts that expect head_experts.* keys.
#
# Model output: scores [B, E] (interpretable as predicted utilities).
# Compatible with:
# - inspect_router_picks.py  (argmax + softmax works on scores)
# - eval_router_all_offline_utility.py (argmax selection works)
# ---------------------------------------------------------------------

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit("Missing sentence-transformers. Install: pip install sentence-transformers")

CFG_PATH = Path("configs/router_config.json")
RESULTS_DIR = Path("results")

# Basic feature dim (must match inspect/eval scripts)
FEAT_DIM = 5


# -------------------------
# Repro
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# IO
# -------------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# -------------------------
# Features
# -------------------------
def basic_text_features(q: str) -> np.ndarray:
    ql = len(q)
    nw = len(q.split())
    has_yesno = int(any(x in q.lower() for x in ["yes or no", "yes/no", "true or false"]))
    wh = int(any(q.lower().startswith(x) for x in ["what", "who", "when", "where", "why", "how"]))
    has_number = int(any(c.isdigit() for c in q))
    return np.array([ql, nw, has_yesno, wh, has_number], dtype=np.float32)


# -------------------------
# Metrics / Utility
# -------------------------
def get_metric(v: Dict[str, Any], name: str) -> Optional[float]:
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
                    return None
                return float(x)
            except Exception:
                return None
    return None

def get_latency_cap_seconds(cfg: Dict[str, Any], dataset_name: str, expert: str) -> float:
    util = cfg.get("utility", {})
    caps = util.get("latency_caps_seconds", {}) or {}
    default_cap = float(caps.get("default", 0.0) or 0.0)

    by_ds = caps.get("by_dataset", {}) or {}
    ds_caps = by_ds.get(dataset_name, {}) or {}
    ds_default = float(ds_caps.get("default", default_cap) or default_cap)

    # expert override inside dataset
    return float(ds_caps.get(expert, ds_default) or ds_default)

def compute_utility(cfg: Dict[str, Any], dataset_name: str, expert: str, metrics: Dict[str, Any]) -> float:
    util = cfg.get("utility", {})
    # If latency/vram missing, fallback costs per expert (optional)
    a = float(util.get("alpha_f1", 1.0))
    b = float(util.get("beta_em", 0.5))
    g = float(util.get("gamma_loose_em", 0.0))
    lam = float(util.get("lambda_latency", 0.0))
    mu = float(util.get("mu_vram", 0.0))

    f1 = get_metric(metrics, "f1") or 0.0
    em = get_metric(metrics, "em") or 0.0
    loose = get_metric(metrics, "loose_em") or 0.0

    lat = get_metric(metrics, "latency")
    vram = get_metric(metrics, "vram_mb")

    # If latency/vram missing, fallback costs per expert (optional)
    expert_costs: Dict[str, Dict[str, float]] = (cfg.get("training", {}) or {}).get("expert_costs", {}) or {}
    
    if lat is None:
        lat = float(expert_costs.get(expert, {}).get("latency", 0.0))
    if vram is None:
        vram = float(expert_costs.get(expert, {}).get("vram_mb", 0.0))


    # cap latency (prevents stalls dominating utility)
    cap = get_latency_cap_seconds(cfg, dataset_name, expert)
    lat_used = min(max(0.0, float(lat)), float(cap)) if cap and cap > 0 else max(0.0, float(lat))
    vram_used = max(0.0, float(vram))

    acc = (a * f1) + (b * em) + (g * loose)
    cost = (lam * math.log1p(lat_used)) + (mu * math.log1p(vram_used))
    return acc - cost

def utils_to_soft_targets(utils: torch.Tensor, tau: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    u = utils - utils.max(dim=1, keepdim=True).values
    return F.softmax(u / max(float(tau), eps), dim=1)


# -------------------------
# Embeddings
# -------------------------
def _try_load_embedder(name: str) -> Optional[SentenceTransformer]:
    try:
        return SentenceTransformer(name)
    except Exception:
        return None


def load_embedder_list(requested: List[str]) -> Tuple[List[str], List[SentenceTransformer]]:
    used_names: List[str] = []
    used_models: List[SentenceTransformer] = []
    for n in requested:
        m = _try_load_embedder(n)
        if m is not None:
            used_names.append(n)
            used_models.append(m)
    if not used_models:
        fallback = "sentence-transformers/all-mpnet-base-v2"
        used_names = [fallback]
        used_models = [SentenceTransformer(fallback)]
    return used_names, used_models


def embed_dim(model: SentenceTransformer) -> int:
    v = model.encode("hello", normalize_embeddings=True)
    return int(np.asarray(v).shape[0])


# -------------------------
# Data
# -------------------------
@dataclass
class Example:
    q: str
    feats: np.ndarray
    embs: List[np.ndarray]
    utils: np.ndarray
    best_idx: int


class RouterDataset(Dataset):
    def __init__(self, examples: List[Example]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        embs_t = [torch.from_numpy(e.astype(np.float32)) for e in ex.embs]
        feats_t = torch.from_numpy(ex.feats.astype(np.float32))
        utils_t = torch.from_numpy(ex.utils.astype(np.float32))
        best_t = torch.tensor(ex.best_idx, dtype=torch.long)
        return embs_t, feats_t, utils_t, best_t


def collate_batch(batch):
    n_enc = len(batch[0][0])
    embs_stacked = []
    for i in range(n_enc):
        embs_stacked.append(torch.stack([b[0][i] for b in batch], dim=0))
    feats = torch.stack([b[1] for b in batch], dim=0)
    utils = torch.stack([b[2] for b in batch], dim=0)
    best = torch.stack([b[3] for b in batch], dim=0)
    return embs_stacked, feats, utils, best


def split_train_val(examples: List[Example], frac_train: float = 0.8) -> Tuple[List[Example], List[Example]]:
    random.shuffle(examples)
    n = len(examples)
    n_train = int(frac_train * n)
    return examples[:n_train], examples[n_train:]


def print_oracle_stats(examples: List[Example], expert_names: List[str], tag: str):
    uniq = {e: 0 for e in expert_names}
    for ex in examples:
        uniq[expert_names[ex.best_idx]] += 1
    n = max(1, len(examples))
    print(f"[{tag}] utility oracle unique-best (on {len(examples)} examples)")
    for e in sorted(uniq.keys(), key=lambda k: uniq[k], reverse=True):
        print(f"  {e:>8}: {uniq[e]:4d} ({uniq[e]/n*100:5.1f}%)")


# -------------------------
# Model
# -------------------------
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

        # IMPORTANT: keep name head_experts for compatibility with old loaders
        # (but these are utility "scores", not class logits)
        self.head_experts = nn.Linear(hidden_dim, out_dim)

    def forward(self, enc_embs: List[torch.Tensor], feats: torch.Tensor) -> torch.Tensor:
        proj_embs = [self.proj[i](enc_embs[i]) for i in range(self.n_enc)]
        ref = proj_embs[0]
        gate_w = F.softmax(self.gate(torch.cat([ref, feats], dim=1)), dim=1)
        mixed = (torch.stack(proj_embs, dim=1) * gate_w.unsqueeze(-1)).sum(dim=1)
        h = self.trunk(torch.cat([mixed, feats], dim=1))
        return self.head_experts(h)  # [B, E] scores


# -------------------------
# Loss helpers
# -------------------------
def normalize_utils_per_example(y_utils: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Per-example z-score so regression focuses on *relative* ordering.
    """
    mu = y_utils.mean(dim=1, keepdim=True)
    sd = y_utils.std(dim=1, keepdim=True).clamp(min=eps)
    return (y_utils - mu) / sd


def rank_hinge_loss(scores: torch.Tensor, best_idx: torch.Tensor, margin: float) -> torch.Tensor:
    """
    Pairwise hinge: score(best) >= score(j) + margin for all j != best
    scores: [B,E], best_idx: [B]
    """
    bsz, E = scores.shape
    s_best = scores.gather(1, best_idx.view(-1, 1))  # [B,1]
    diffs = s_best - scores  # [B,E]

    mask = torch.ones_like(diffs, dtype=torch.bool)
    mask.scatter_(1, best_idx.view(-1, 1), False)

    viol = F.relu(float(margin) - diffs)  # [B,E]
    viol = viol[mask].view(bsz, E - 1)
    return viol.mean()


def pick_distribution(scores_np: np.ndarray, expert_names: List[str]) -> Tuple[str, float]:
    idx = np.argmax(scores_np, axis=1)
    counts = np.bincount(idx, minlength=len(expert_names))
    top_i = int(np.argmax(counts))
    pct = float(counts[top_i] / max(1, len(idx)) * 100.0)
    return expert_names[top_i], pct


def entropy_mean_from_scores_np(scores_np: np.ndarray) -> float:
    if scores_np.shape[0] == 0:
        return 0.0
    p = torch.softmax(torch.from_numpy(scores_np), dim=1)
    ent = (-(p * (p + 1e-12).log()).sum(dim=1)).mean().item()
    return float(ent)


# -------------------------
# Train per dataset
# -------------------------
def train_one_dataset(dataset_name: str, ds_cfg: Dict[str, Any], cfg: Dict[str, Any]):
    router_train_out = Path(ds_cfg["router_train_out"])
    if not router_train_out.exists():
        raise SystemExit(f"[{dataset_name}] Missing router train file: {router_train_out} (run build_router_train_all.py)")

    rows = read_jsonl(router_train_out)
    if not rows:
        raise SystemExit(f"[{dataset_name}] Empty router train file: {router_train_out}")

    # ---- Canonical expert order: ALWAYS from config ----
    tr_cfg = cfg.get("training", {})
    allowed_experts_cfg = tr_cfg.get("allowed_experts", None)
    if not allowed_experts_cfg:
        raise SystemExit("configs/router_config.json must contain training.allowed_experts in canonical order.")

    jsonl_experts = list((rows[0].get("experts") or {}).keys())
    if not jsonl_experts:
        raise SystemExit(f"[{dataset_name}] First row has no experts block/keys.")

    # keep config order but only those that exist in jsonl
    expert_names = [e for e in allowed_experts_cfg if e in jsonl_experts]
    if len(expert_names) != len(allowed_experts_cfg):
        missing = [e for e in allowed_experts_cfg if e not in jsonl_experts]
        raise SystemExit(
            f"[{dataset_name}] JSONL experts do not match config. Missing in JSONL: {missing}. "
            f"JSONL has: {jsonl_experts}"
        )

    out_dim = len(expert_names)

    # Embedders from dataset config (one or many)
    requested = (
            ds_cfg.get("embed_models")
            or ds_cfg.get("embed_model")
            or ds_cfg.get("embedders")
            or ds_cfg.get("requested_embedders")
            or ["sentence-transformers/all-mpnet-base-v2"]
    )
    if isinstance(requested, str):
        requested = [requested]
    requested = [str(x) for x in requested]

    used_names, embedders = load_embedder_list(requested)
    embed_dims = [embed_dim(m) for m in embedders]

    util_cfg = cfg.get("utility", {})
    expert_costs: Dict[str, Dict[str, float]] = tr_cfg.get("expert_costs", {}) or {}

    examples: List[Example] = []
    for r in rows:
        q = r.get("question") or ""
        if not q:
            continue
        block = r.get("experts", {})
        if not isinstance(block, dict) or not block:
            continue
        if any(e not in block for e in expert_names):
            continue

        utils = np.array([compute_utility(cfg, dataset_name, e, block[e]) for e in expert_names], dtype=np.float32)

        best_idx = int(np.argmax(utils))

        feats = basic_text_features(q)

        embs = []
        for m in embedders:
            v = m.encode(q, normalize_embeddings=True)
            embs.append(np.asarray(v, dtype=np.float32))

        examples.append(Example(q=q, feats=feats, embs=embs, utils=utils, best_idx=best_idx))

    if len(examples) < 50:
        raise SystemExit(f"[{dataset_name}] Too few examples: {len(examples)}")

    train_ex, val_ex = split_train_val(examples, 0.8)

    # Training hparams from config (with safe defaults)
    seed = int(tr_cfg.get("seed", 1337))
    epochs = int(tr_cfg.get("epochs", 80))
    batch_size = int(tr_cfg.get("batch_size", 64))
    lr = float(tr_cfg.get("lr", 8e-4))
    weight_decay = float(tr_cfg.get("weight_decay", 1e-3))
    hidden_dim = int(tr_cfg.get("hidden_dim", 256))
    dropout = float(tr_cfg.get("dropout", 0.10))
    patience = int(tr_cfg.get("patience", 12))
    min_delta = float(tr_cfg.get("min_delta", 2e-4))
    tau_listwise = float(tr_cfg.get("tau", 0.7))

    # Loss weights / margins (keep these here; tune later if needed)
    REG_WEIGHT = 1.0
    RANK_WEIGHT = 1.0
    PAIR_MARGIN = 0.35
    LISTWISE_WEIGHT = 0.15
    LOGIT_TAU = 1.0

    # Balanced sampling (recommended)
    USE_BALANCED_SAMPLER = True

    # Model dims (keep stable)
    PROJ_DIM = 256

    print(f"\n[{dataset_name}] examples={len(examples)} train={len(train_ex)} val={len(val_ex)}")
    print(f"[{dataset_name}] expert_names (CANONICAL)={expert_names}")
    print(f"[{dataset_name}] embedders requested={requested}")
    print(f"[{dataset_name}] embedders used={used_names}")
    print(f"[{dataset_name}] loss: REG*{REG_WEIGHT} + RANK*{RANK_WEIGHT} (margin={PAIR_MARGIN}) + LISTWISE*{LISTWISE_WEIGHT}")
    print(
        f"[{dataset_name}] utility weights: "
        f"alpha={util_cfg.get('alpha_f1', 1.0)} beta={util_cfg.get('beta_em', 0.5)} gamma={util_cfg.get('gamma_loose_em', 0.0)} "
        f"lambda_lat={util_cfg.get('lambda_latency', 0.0)} mu_vram={util_cfg.get('mu_vram', 0.0)}"
    )
    print_oracle_stats(train_ex, expert_names, tag=f"{dataset_name}/train")

    train_ds = RouterDataset(train_ex)
    val_ds = RouterDataset(val_ex)

    if USE_BALANCED_SAMPLER:
        labels = np.array([ex.best_idx for ex in train_ex], dtype=np.int64)
        counts = np.bincount(labels, minlength=out_dim).astype(np.float32)
        counts = np.clip(counts, 1.0, None)
        inv = 1.0 / counts
        sample_w = inv[labels]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_w, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, collate_fn=collate_batch)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

    val_loader = DataLoader(val_ds, batch_size=max(64, batch_size), shuffle=False, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GatedMultiEmbedRouter(
        embed_dims=embed_dims,
        feat_dim=FEAT_DIM,
        proj_dim=PROJ_DIM,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        dropout=dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_match = -1.0
    best_mse = float("inf")
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_losses = []

        for enc_embs, feats, y_utils, y_best in train_loader:
            enc_embs = [t.to(device) for t in enc_embs]
            feats = feats.to(device)
            y_utils = y_utils.to(device)
            y_best = y_best.to(device)

            scores = model(enc_embs, feats)  # [B,E]

            # regression on per-example normalized utilities
            y_norm = normalize_utils_per_example(y_utils)
            reg = F.mse_loss(scores, y_norm)

            # pairwise ranking (best must beat others)
            rnk = rank_hinge_loss(scores, y_best, margin=PAIR_MARGIN)

            # small listwise term (helps shape distribution a bit)
            tgt_p = utils_to_soft_targets(y_utils, tau=tau_listwise)
            ce = -(tgt_p * F.log_softmax(scores / max(1e-6, float(LOGIT_TAU)), dim=1)).sum(dim=1).mean()

            loss = (REG_WEIGHT * reg) + (RANK_WEIGHT * rnk) + (LISTWISE_WEIGHT * ce)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(loss.item())

        # ---- validation: match oracle-best (argmax utility) ----
        model.eval()
        all_scores = []
        all_best = []
        all_utils = []
        with torch.no_grad():
            for enc_embs, feats, y_utils, y_best in val_loader:
                enc_embs = [t.to(device) for t in enc_embs]
                feats = feats.to(device)
                y_utils = y_utils.to(device)
                y_best = y_best.to(device)

                scores = model(enc_embs, feats)

                all_scores.append(scores.detach().cpu().numpy())
                all_best.append(y_best.detach().cpu().numpy())
                all_utils.append(y_utils.detach().cpu().numpy())

        scores_np = np.concatenate(all_scores, axis=0) if all_scores else np.zeros((0, out_dim), dtype=np.float32)
        best_np = np.concatenate(all_best, axis=0) if all_best else np.zeros((0,), dtype=np.int64)
        utils_np = np.concatenate(all_utils, axis=0) if all_utils else np.zeros((0, out_dim), dtype=np.float32)

        pred = scores_np.argmax(axis=1) if scores_np.shape[0] else np.array([], dtype=np.int64)
        val_match = float((pred == best_np).mean()) if pred.size else 0.0

        y_norm_np = (
            (utils_np - utils_np.mean(axis=1, keepdims=True)) / (utils_np.std(axis=1, keepdims=True) + 1e-6)
            if utils_np.shape[0]
            else utils_np
        )
        val_mse = float(np.mean((scores_np - y_norm_np) ** 2)) if scores_np.shape[0] else float("inf")

        top_e, top_pct = pick_distribution(scores_np, expert_names)
        ent = entropy_mean_from_scores_np(scores_np)
        tr = float(np.mean(tr_losses)) if tr_losses else float("nan")

        print(
            f"[{dataset_name}] epoch {epoch:02d} | train {tr:.4f} | "
            f"val_match={val_match:.3f} | val_mse={val_mse:.3f} | "
            f"val_top={top_e} {top_pct:.2f}% | val_entropy={ent:.3f}"
        )

        improved = (val_match > best_match + min_delta) or (
                abs(val_match - best_match) <= 1e-9 and val_mse < best_mse - 1e-4
        )

        if improved:
            best_match = val_match
            best_mse = val_mse
            bad = 0

            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            ckpt_path = RESULTS_DIR / f"router_{dataset_name}_utility_ranker.pt"

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "expert_names": expert_names,
                    "out_dim": out_dim,
                    "feat_dim": FEAT_DIM,
                    "proj_dim": PROJ_DIM,
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "embed_models_requested": requested,
                    "embed_models_used": used_names,
                    "embed_dims": embed_dims,
                    "utility_weights": util_cfg,
                    "utility_cfg": util_cfg,
                    "train_hparams": {
                        "MODE": "utility_regression_ranking",
                        "REG_WEIGHT": REG_WEIGHT,
                        "RANK_WEIGHT": RANK_WEIGHT,
                        "PAIR_MARGIN": PAIR_MARGIN,
                        "LISTWISE_WEIGHT": LISTWISE_WEIGHT,
                        "TAU_LISTWISE": tau_listwise,
                        "LOGIT_TAU": LOGIT_TAU,
                        "USE_BALANCED_SAMPLER": USE_BALANCED_SAMPLER,
                        "LR": lr,
                        "WEIGHT_DECAY": weight_decay,
                        "EPOCHS": epochs,
                        "BATCH_SIZE": batch_size,
                        "SEED": seed,
                        "PATIENCE": patience,
                        "MIN_DELTA": min_delta,
                    },
                    "best_val_match": best_match,
                    "best_val_mse": best_mse,
                },
                ckpt_path,
            )
        else:
            bad += 1
            if bad >= patience:
                print(f"[{dataset_name}] Early stopping.")
                break


def main():
    if not CFG_PATH.exists():
        raise SystemExit(f"Missing {CFG_PATH}")

    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8-sig"))

    tr_cfg = cfg.get("training", {})
    seed = int(tr_cfg.get("seed", 1337))
    set_seed(seed)

    if "datasets" not in cfg:
        raise SystemExit(f"Invalid config: expected key 'datasets' in {CFG_PATH}")

    for dataset_name, ds_cfg in cfg["datasets"].items():
        train_one_dataset(dataset_name, ds_cfg, cfg)


if __name__ == "__main__":
    main()
