# train_router_two_stage.py
# Implements:
#  - Gate Δ-to-gold/Δ-utility training (regression or classification)
#  - Threshold calibration on validation split (sweep) and saving best threshold
#  - Feature augmentation via precomputed JSONL feature files (retrieval preview + NO_RAG uncertainty probes)
#  - Selector margin-weighted training (top1-top2 margin inside family)
#  - Optional selector margin filtering still supported (existing args)
#
# Changes applied:
#  1) REMOVE SQuAD answerability head training (Ryo concern: not meaningful pre-retrieval)
#  2) PubMed policy controllable:
#       - default: FORCE PubMedQA policy to RAG (policy=True)
#       - --pubmed_policy none disables forcing so PubMed can be routed by gate/shared gate
#  3) Stage-2 input supports retrieval-preview features:
#       - if --feature_files given and --feature_keys omitted, infer keys automatically (stable sorted union)
#       - feature standardization supported; stats saved when --save_feature_stats
#  4) commonsenseqa only has base_only/base_rag, so pools are dynamic per dataset/row
#  5) --only combined_pubmed_csqa_gate trains ONE classifier gate: pubmed=RAG(1) vs csqa=NO(0)
#  6) --use_passage_embeddings appends mean embedding of retrieved <DOCUMENT> passages (from a chosen expert’s prediction_raw)
#
# Critical fix:
#  7) When --tradeoff_mode is enabled, training MUST use the exact same utility as eval_router_two_stage.py
#     i.e. router_config.json["utility"] with latency caps.

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

CFG_PATH = Path("configs/router_config.json")
PRED_DIR = Path("prediction")

CANON_RAG_EXPERTS = ["base_rag", "sft_rag", "raft_rag"]
CANON_NO_EXPERTS  = ["base_only", "sft_only"]

DATASETS = ["hotpotqa", "squad_v2", "pubmedqa_v2", "commonsenseqa"]
SPECIAL_ONLY = "combined_pubmed_csqa_gate"

_DOC_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.DOTALL)


# --------------------------
# Config
# --------------------------

def load_cfg() -> Dict[str, Any]:
    if not CFG_PATH.exists():
        raise SystemExit(f"Missing {CFG_PATH}")
    return json.loads(CFG_PATH.read_text(encoding="utf-8-sig"))


# --------------------------
# Utility / tradeoff (MUST match eval_router_two_stage.py)
# --------------------------

def tradeoff_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    u = (cfg.get("utility") or {})
    caps = (u.get("latency_caps_seconds") or {})
    return {
        "alpha_f1": float(u.get("alpha_f1", 1.0)),
        "beta_em": float(u.get("beta_em", 0.0)),
        "gamma_loose_em": float(u.get("gamma_loose_em", 0.0)),
        "lambda_latency": float(u.get("lambda_latency", 0.0)),
        "mu_vram": float(u.get("mu_vram", 0.0)),
        "latency_caps": caps,
    }


def get_latency_s(outcome: Dict[str, Any]) -> float:
    return float(outcome.get("latency", outcome.get("time", 0.0)) or 0.0)


def get_vram_gb(outcome: Dict[str, Any]) -> float:
    mb = float(outcome.get("vram_mb", outcome.get("peak_vram_mb", 0.0)) or 0.0)
    return mb / 1024.0


def _get_latency_cap_seconds(tcfg: Dict[str, Any], dataset: str, expert: Optional[str]) -> float:
    caps = tcfg.get("latency_caps") or {}
    default_cap = float(caps.get("default", 3.0))

    by_dataset = (caps.get("by_dataset") or {})
    ds_cfg = by_dataset.get(dataset) or {}

    cap = float(ds_cfg.get("default", default_cap))
    if expert and expert in ds_cfg:
        cap = float(ds_cfg[expert])
    return cap


def tradeoff_U(outcome: Dict[str, Any], tcfg: Dict[str, Any], dataset: str, expert: Optional[str]) -> float:
    f1 = float(outcome.get("f1", 0.0) or 0.0)
    em = float(outcome.get("em", 0.0) or 0.0)
    loose = float(outcome.get("loose_em", em) or em)

    Q = (tcfg["alpha_f1"] * f1) + (tcfg["beta_em"] * em) + (tcfg["gamma_loose_em"] * loose)

    L = get_latency_s(outcome)
    V = get_vram_gb(outcome)

    cap = _get_latency_cap_seconds(tcfg, dataset, expert)
    lat_ratio = L / max(1e-8, cap)
    lat_pen = tcfg["lambda_latency"] * (lat_ratio if lat_ratio <= 1.0 else (lat_ratio ** 2))

    vram_pen = tcfg["mu_vram"] * V
    return float(Q - lat_pen - vram_pen)


def latency_cap_seconds(cfg: Dict[str, Any], dataset: str, expert: str) -> float:
    u = cfg.get("utility", {}) or {}
    caps = u.get("latency_caps_seconds", {}) or {}
    default_cap = float(caps.get("default", 3.0))
    by_ds = (caps.get("by_dataset", {}) or {}).get(dataset, {}) or {}
    return float(by_ds.get(expert, by_ds.get("default", default_cap)))


def utility_value(cfg: Dict[str, Any], dataset: str, expert: str, outcome: Dict[str, Any]) -> float:
    """
    Legacy utility (also based on cfg["utility"]). Used when NOT in --tradeoff_mode.
    """
    u = cfg.get("utility", {}) or {}
    a = float(u.get("alpha_f1", 1.0))
    b = float(u.get("beta_em", 0.0))
    g = float(u.get("gamma_loose_em", 0.0))
    lam = float(u.get("lambda_latency", 0.0))
    mu = float(u.get("mu_vram", 0.0))

    f1 = float(outcome.get("f1", 0.0) or 0.0)
    em = float(outcome.get("em", 0.0) or 0.0)
    loose = float(outcome.get("loose_em", em) or em)

    lat = float(outcome.get("latency", outcome.get("time", 0.0)) or 0.0)
    cap = latency_cap_seconds(cfg, dataset, expert)
    lat = min(lat, cap)

    vram_mb = float(outcome.get("vram_mb", outcome.get("peak_vram_mb", 0.0)) or 0.0)
    return a * f1 + b * em + g * loose - lam * lat - mu * vram_mb


def score_for_targets(
        cfg: Dict[str, Any],
        dataset: str,
        expert: str,
        outcome: Dict[str, Any],
        *,
        use_tradeoff: bool,
        tcfg: Dict[str, Any],
) -> float:
    if use_tradeoff:
        return tradeoff_U(outcome, tcfg, dataset, expert)
    return utility_value(cfg, dataset, expert, outcome)


# --------------------------
# IO
# --------------------------

def read_router_train(dataset: str) -> List[Dict[str, Any]]:
    p = PRED_DIR / f"router_train_{dataset}.jsonl"
    if not p.exists():
        raise SystemExit(f"Missing router train file: {p} (run build_router_train.py)")
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --------------------------
# Pools (dynamic for commonsenseqa)
# --------------------------

def pools_for_dataset(dataset: str) -> Tuple[List[str], List[str]]:
    if dataset == "commonsenseqa":
        return ["base_rag"], ["base_only"]
    return CANON_RAG_EXPERTS, CANON_NO_EXPERTS


def pools_for_row(ex: Dict[str, Any], rag_pool: List[str], no_pool: List[str]) -> Tuple[List[str], List[str]]:
    keys = set(ex.keys())
    rp = [e for e in rag_pool if e in keys]
    npool = [e for e in no_pool if e in keys]
    return rp, npool


def _best_in_pool(
        cfg: Dict[str, Any],
        dataset: str,
        ex: Dict[str, Any],
        pool: List[str],
        *,
        use_tradeoff: bool,
        tcfg: Dict[str, Any],
) -> Tuple[str, float]:
    best_e = None
    best_u = -1e18
    for e in pool:
        u = score_for_targets(cfg, dataset, e, ex[e], use_tradeoff=use_tradeoff, tcfg=tcfg)
        if u > best_u:
            best_u = u
            best_e = e
    if best_e is None:
        return ("", -1e18)
    return best_e, float(best_u)


def _top2_margin_in_pool(
        cfg: Dict[str, Any],
        dataset: str,
        ex: Dict[str, Any],
        pool: List[str],
        *,
        use_tradeoff: bool,
        tcfg: Dict[str, Any],
) -> float:
    if not pool:
        return 0.0
    utils = sorted(
        [score_for_targets(cfg, dataset, e, ex[e], use_tradeoff=use_tradeoff, tcfg=tcfg) for e in pool],
        reverse=True,
    )
    if len(utils) < 2:
        return 0.0
    return float(utils[0] - utils[1])


# --------------------------
# Policy (PubMed controllable)
# --------------------------

def policy_for_dataset(dataset: str, *, pubmed_policy_mode: str = "none") -> Optional[bool]:
    # Ryo: always use RAG for these datasets
    if dataset in ("hotpotqa", "squad_v2"):
        return True

    # PubMed controlled separately (shared gate recommended)
    if dataset == "pubmedqa_v2":
        return True if pubmed_policy_mode == "forced" else None

    # commonsenseqa: do not force here (shared gate will handle)
    return None


# --------------------------
# Embedding
# --------------------------

class Embedder:
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )


# --------------------------
# Feature loading
# --------------------------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_feature_map(paths: List[Path]) -> Dict[str, Dict[str, float]]:
    fmap: Dict[str, Dict[str, float]] = {}
    for p in paths:
        if not p.exists():
            print(f"[WARN] feature file not found: {p}")
            continue
        for r in _read_jsonl(p):
            rid = r.get("id")
            if rid is None:
                continue
            rid = str(rid)

            # Prefer dataset-aware keying if possible (prevents cross-dataset id collisions)
            ds = r.get("dataset")
            if ds is not None:
                rid_key = f"{str(ds)}::{rid}"
            else:
                rid_key = rid

            feats = r.get("features")
            if feats is None:
                feats = {k: v for k, v in r.items() if k not in ("id", "dataset", "question", "features")}
            if not isinstance(feats, dict):
                continue

            fmap.setdefault(rid_key, {})
            for k, v in feats.items():
                try:
                    fmap[rid_key][str(k)] = float(v)
                except Exception:
                    continue
    return fmap


def infer_feature_keys_from_map(fmap: Dict[str, Dict[str, float]]) -> List[str]:
    keys = set()
    for _, feats in fmap.items():
        for k in feats.keys():
            keys.add(str(k))
    return sorted(keys)


def build_feature_matrix(rows: List[Dict[str, Any]], fmap: Dict[str, Dict[str, float]], feature_keys: List[str]) -> torch.Tensor:
    n = len(rows)
    d = len(feature_keys)
    Xf = torch.zeros((n, d), dtype=torch.float32)
    if d == 0:
        return Xf
    for i, r in enumerate(rows):
        rid = str(r.get("id", i))
        ds = str(r.get("dataset", ""))  # should exist in router_train rows
        rid_key = f"{ds}::{rid}" if ds else rid
        feats = fmap.get(rid_key, {})
        for j, k in enumerate(feature_keys):
            if k in feats:
                Xf[i, j] = float(feats[k])
    return Xf


def standardize_features(Xf: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if Xf.numel() == 0 or Xf.size(1) == 0:
        return Xf, {"mean": [], "std": []}
    mean = Xf.mean(dim=0, keepdim=True)
    std = Xf.std(dim=0, keepdim=True).clamp_min(eps)
    return (Xf - mean) / std, {"mean": mean.squeeze(0).tolist(), "std": std.squeeze(0).tolist()}


# --------------------------
# Passage embeddings
# --------------------------

def extract_docs_from_prediction_raw(pred_raw: str, *, max_docs: int, max_chars: int) -> List[str]:
    if not pred_raw:
        return []
    docs = _DOC_RE.findall(pred_raw)
    docs = [d.strip() for d in docs if d.strip()]
    docs = docs[:max_docs]
    if max_chars is not None and max_chars > 0:
        docs = [d[:max_chars] for d in docs]
    return docs


def build_passage_embedding_matrix(
        rows: List[Dict[str, Any]],
        embedder: Embedder,
        *,
        source_expert: str = "base_rag",
        max_docs: int = 5,
        max_chars: int = 1200,
        batch_size_docs: int = 64,
) -> torch.Tensor:
    """
    Returns [N, D] (CPU tensor). D = embedding dim. Mean over docs from <DOCUMENT> blocks.
    """
    dummy = embedder.encode(["dummy"], batch_size=1).float()
    d = int(dummy.shape[-1])

    flat_docs: List[str] = []
    offsets: List[Tuple[int, int]] = []

    for r in rows:
        ex = r["experts"]
        pred_raw = ex.get(source_expert, {}).get("prediction_raw", "") or ""
        docs = extract_docs_from_prediction_raw(pred_raw, max_docs=max_docs, max_chars=max_chars)
        s = len(flat_docs)
        flat_docs.extend(docs)
        e = len(flat_docs)
        offsets.append((s, e))

    if len(flat_docs) == 0:
        return torch.zeros((len(rows), d), dtype=torch.float32)

    E = embedder.encode(flat_docs, batch_size=batch_size_docs).float().cpu()  # [M, D]

    Xp = torch.zeros((len(rows), d), dtype=torch.float32)
    for i, (s, e) in enumerate(offsets):
        if e > s:
            Xp[i] = E[s:e].mean(dim=0)
    return Xp


# --------------------------
# Dataset wrappers
# --------------------------

class TensorDatasetXY(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return int(self.X.size(0))

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TensorDatasetSoft(Dataset):
    def __init__(self, X: torch.Tensor, y_soft: torch.Tensor, w: Optional[torch.Tensor] = None):
        self.X = X
        self.y = y_soft
        self.w = w

    def __len__(self):
        return int(self.X.size(0))

    def __getitem__(self, idx):
        if self.w is None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.y[idx], self.w[idx]


# --------------------------
# Model
# --------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------
# Train helpers
# --------------------------

def split_train_val_indices(n: int, val_ratio: float, seed: int, *, min_val: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    if n < 2:
        return np.arange(n), np.array([], dtype=np.int64)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_val = int(round(n * float(val_ratio)))
    n_val = max(int(min_val), n_val)
    n_val = min(n - 1, n_val)

    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    return tr_idx, val_idx


def class_weights_from_labels(y: torch.Tensor, num_classes: int, device: str) -> torch.Tensor:
    cnt = Counter(y.tolist())
    w = torch.zeros(num_classes, dtype=torch.float32)
    for i in range(num_classes):
        w[i] = 1.0 / max(1, cnt.get(i, 0))
    w = w / (w.mean() + 1e-12)
    return w.to(device)


def make_balanced_sampler_from_hard_labels(y: torch.Tensor) -> WeightedRandomSampler:
    labels_np = y.cpu().numpy()
    num_classes = int(labels_np.max()) + 1 if labels_np.size else 1
    counts = np.bincount(labels_np, minlength=num_classes).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    inv = 1.0 / counts
    sample_w = inv[labels_np]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_w, dtype=torch.double),
        num_samples=len(labels_np),
        replacement=True,
    )


def entropy_bonus(probs: torch.Tensor) -> torch.Tensor:
    p = probs.clamp_min(1e-8)
    return -(p * p.log()).sum(dim=1).mean()


def kl_to_prior(mean_probs: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    p = mean_probs.clamp_min(1e-8)
    q = prior.clamp_min(1e-8)
    return (p * (p.log() - q.log())).sum()


def soft_cross_entropy_per_example(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    return -(target_probs * logp).sum(dim=1)


# --------------------------
# Gate: Δ training + threshold calibration
# --------------------------

def build_gate_delta_targets(
        cfg: Dict[str, Any],
        dataset: str,
        rows: List[Dict[str, Any]],
        *,
        use_tradeoff: bool,
        tcfg: Dict[str, Any],
) -> torch.Tensor:
    """
    delta_i = bestRAG - bestNO
    """
    rag_pool_ds, no_pool_ds = pools_for_dataset(dataset)

    deltas: List[float] = []
    for r in rows:
        ex = r["experts"]
        rag_pool, no_pool = pools_for_row(ex, rag_pool_ds, no_pool_ds)

        if len(rag_pool) == 0 or len(no_pool) == 0:
            deltas.append(0.0)
            continue

        _, br_u = _best_in_pool(cfg, dataset, ex, rag_pool, use_tradeoff=use_tradeoff, tcfg=tcfg)
        _, bn_u = _best_in_pool(cfg, dataset, ex, no_pool,  use_tradeoff=use_tradeoff, tcfg=tcfg)
        deltas.append(float(br_u - bn_u))

    return torch.tensor(deltas, dtype=torch.float32)


def build_gate_deadzone_from_deltas(deltas: torch.Tensor, *, gate_delta: float) -> Tuple[List[int], torch.Tensor]:
    idx = torch.where(deltas.abs() >= float(gate_delta))[0].tolist()
    if len(idx) == 0:
        return [], torch.zeros((0,), dtype=torch.long)
    y = (deltas[idx] > 0.0).long()
    return idx, y


def train_gate_classifier(
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        *,
        device: str,
        lr: float,
        weight_decay: float,
        batch_size: int,
        epochs: int,
        patience: int,
        min_delta: float,
        seed: int,
        min_val: int,
) -> Tuple[nn.Module, float]:
    tr_idx, va_idx = split_train_val_indices(int(X.size(0)), 0.2, seed, min_val=min_val)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    w = class_weights_from_labels(y_tr, 2, device)
    crit = nn.CrossEntropyLoss(weight=w)

    sampler = make_balanced_sampler_from_hard_labels(y_tr)
    tr_loader = DataLoader(TensorDatasetXY(X_tr, y_tr), batch_size=batch_size, sampler=sampler)
    va_loader = DataLoader(TensorDatasetXY(X_va, y_va), batch_size=batch_size, shuffle=False)

    best_acc = -1.0
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in tqdm(tr_loader, desc="Batches[gate-cls]", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb.long())
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)

        train_loss = total_loss / max(1, len(tr_loader.dataset))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
        val_acc = correct / max(1, total)
        print(f"[gate-cls] epoch {ep:02d} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc + min_delta:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[gate-cls] early stop. best_val_acc={best_acc:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, float(best_acc)


def train_gate_delta_regressor(
        model: nn.Module,
        X: torch.Tensor,
        deltas: torch.Tensor,
        *,
        device: str,
        lr: float,
        weight_decay: float,
        batch_size: int,
        epochs: int,
        patience: int,
        min_delta: float,
        seed: int,
        min_val: int,
        huber_delta: float,
) -> Tuple[nn.Module, Dict[str, Any]]:
    tr_idx, va_idx = split_train_val_indices(int(X.size(0)), 0.2, seed, min_val=min_val)
    X_tr, d_tr = X[tr_idx], deltas[tr_idx]
    X_va, d_va = X[va_idx], deltas[va_idx]

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.HuberLoss(delta=float(huber_delta))

    tr_loader = DataLoader(TensorDatasetXY(X_tr, d_tr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDatasetXY(X_va, d_va), batch_size=batch_size, shuffle=False)

    best_v = 1e18
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, db in tqdm(tr_loader, desc="Batches[gate-delta]", leave=False):
            xb = xb.to(device)
            db = db.to(device).float()

            opt.zero_grad(set_to_none=True)
            pred = model(xb).squeeze(1)
            loss = crit(pred, db)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)

        train_loss = total_loss / max(1, len(tr_loader.dataset))

        model.eval()
        total_v = 0.0
        with torch.no_grad():
            for xb, db in va_loader:
                xb = xb.to(device)
                db = db.to(device).float()
                pred = model(xb).squeeze(1)
                v = crit(pred, db)
                total_v += float(v.item()) * xb.size(0)

        val_loss = total_v / max(1, len(va_loader.dataset))
        print(f"[gate-delta] epoch {ep:02d} | train_loss={train_loss:.4f} | val_huber={val_loss:.4f}")

        if val_loss < best_v - float(min_delta):
            best_v = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[gate-delta] early stop. best_val_huber={best_v:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pv = model(X_va.to(device)).squeeze(1).detach().cpu().numpy()
        dv = d_va.detach().cpu().numpy()
    yv = (dv > 0).astype(np.int64)

    thr_cands = np.unique(np.concatenate([np.quantile(pv, np.linspace(0.05, 0.95, 19)), np.array([0.0])]))
    best_acc = -1.0
    best_thr = 0.0
    for thr in thr_cands:
        pred = (pv > thr).astype(np.int64)
        acc = float((pred == yv).mean()) if yv.size else 0.0
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)

    info = {
        "best_val_huber": float(best_v),
        "best_val_sign_acc": float(best_acc),
        "best_threshold": float(best_thr),
        "val_size": int(len(va_idx)),
    }
    print(f"[gate-delta] calibrated thr={best_thr:.4f} | val_sign_acc={best_acc:.4f}")
    return model, info


# --------------------------
# Selector training
# --------------------------

def soft_targets_from_utils(utils: np.ndarray, tau: float) -> np.ndarray:
    u = utils - np.max(utils)
    p = np.exp(u / max(1e-8, float(tau)))
    p = p / max(1e-12, p.sum())
    return p.astype(np.float32)


def build_selector_soft_targets(
        cfg: Dict[str, Any],
        dataset: str,
        rows: List[Dict[str, Any]],
        idxs: List[int],
        experts_in_group: List[str],
        *,
        tau: float,
        use_tradeoff: bool,
        tcfg: Dict[str, Any],
) -> torch.Tensor:
    ys = []
    for i in idxs:
        ex = rows[i]["experts"]
        scores = np.array(
            [score_for_targets(cfg, dataset, e, ex[e], use_tradeoff=use_tradeoff, tcfg=tcfg) for e in experts_in_group],
            dtype=np.float32,
        )
        ys.append(soft_targets_from_utils(scores, tau=tau))
    if not ys:
        return torch.zeros((0, len(experts_in_group)), dtype=torch.float32)
    return torch.from_numpy(np.stack(ys, axis=0))


def hard_argmax_from_soft(y_soft: torch.Tensor) -> torch.Tensor:
    if y_soft.numel() == 0:
        return torch.zeros((0,), dtype=torch.long)
    return y_soft.argmax(dim=1).long()


def should_train_selector(n: int, *, min_train: int, min_val: int, name: str, ds: str) -> bool:
    if n < (min_train + min_val):
        print(f"[{ds}] skip {name}: n={n} < (min_train+min_val)={min_train+min_val}")
        return False
    return True


def filter_by_margin_window(
        cfg: Dict[str, Any],
        dataset: str,
        rows: List[Dict[str, Any]],
        idxs: List[int],
        pool: List[str],
        *,
        margin_min: float,
        margin_max: float,
        use_tradeoff: bool,
        tcfg: Dict[str, Any],
) -> List[int]:
    out: List[int] = []
    for i in idxs:
        ex = rows[i]["experts"]
        m = _top2_margin_in_pool(cfg, dataset, ex, pool, use_tradeoff=use_tradeoff, tcfg=tcfg)
        if m >= float(margin_min) and m <= float(margin_max):
            out.append(i)
    return out


def selector_margin_weights(
        cfg: Dict[str, Any],
        dataset: str,
        rows: List[Dict[str, Any]],
        idxs: List[int],
        pool: List[str],
        *,
        use_tradeoff: bool,
        tcfg: Dict[str, Any],
        margin_scale: float,
        weight_min: float,
        weight_max: float,
) -> torch.Tensor:
    w = []
    ms = float(margin_scale)
    for i in idxs:
        ex = rows[i]["experts"]
        m = _top2_margin_in_pool(cfg, dataset, ex, pool, use_tradeoff=use_tradeoff, tcfg=tcfg)
        ww = (float(m) / max(1e-8, ms)) if ms > 0 else 1.0
        ww = float(np.clip(ww, float(weight_min), float(weight_max)))
        w.append(ww)
    if not w:
        return torch.zeros((0,), dtype=torch.float32)
    return torch.tensor(w, dtype=torch.float32)


def train_selector_soft(
        model: nn.Module,
        X: torch.Tensor,
        y_soft: torch.Tensor,
        *,
        device: str,
        lr: float,
        weight_decay: float,
        batch_size: int,
        epochs: int,
        patience: int,
        min_delta: float,
        seed: int,
        min_val: int,
        balanced_sampler: bool,
        reg_type: str,
        reg_weight: float,
        prior: Optional[torch.Tensor],
        hard_labels_for_sampler: Optional[torch.Tensor],
        example_weights: Optional[torch.Tensor],
) -> Tuple[nn.Module, float]:
    tr_idx, va_idx = split_train_val_indices(int(X.size(0)), 0.2, seed, min_val=min_val)
    X_tr, y_tr = X[tr_idx], y_soft[tr_idx]
    X_va, y_va = X[va_idx], y_soft[va_idx]

    w_tr = None
    w_va = None
    if example_weights is not None and example_weights.numel() == X.size(0):
        w_tr = example_weights[tr_idx]
        w_va = example_weights[va_idx]

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if balanced_sampler:
        if hard_labels_for_sampler is None:
            raise ValueError("balanced_sampler=True but hard_labels_for_sampler is None")
        sampler = make_balanced_sampler_from_hard_labels(hard_labels_for_sampler[tr_idx])
        tr_loader = DataLoader(TensorDatasetSoft(X_tr, y_tr, w_tr), batch_size=batch_size, sampler=sampler)
    else:
        tr_loader = DataLoader(TensorDatasetSoft(X_tr, y_tr, w_tr), batch_size=batch_size, shuffle=True)

    va_loader = DataLoader(TensorDatasetSoft(X_va, y_va, w_va), batch_size=batch_size, shuffle=False)

    best_score = -1e18
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(tr_loader, desc="Batches[sel]", leave=False):
            if len(batch) == 2:
                xb, tb = batch
                wb = None
            else:
                xb, tb, wb = batch

            xb = xb.to(device)
            tb = tb.to(device)
            if wb is not None:
                wb = wb.to(device).float()

            opt.zero_grad(set_to_none=True)
            logits = model(xb)

            per_ex = soft_cross_entropy_per_example(logits, tb)
            loss = (per_ex * wb).mean() if wb is not None else per_ex.mean()

            if reg_weight > 0.0 and reg_type != "none":
                probs = F.softmax(logits, dim=1)
                if reg_type == "entropy":
                    loss = loss - float(reg_weight) * entropy_bonus(probs)
                elif reg_type == "kl":
                    if prior is None:
                        raise ValueError("reg_type=kl requires prior")
                    mean_p = probs.mean(dim=0)
                    loss = loss + float(reg_weight) * kl_to_prior(mean_p, prior.to(device))
                else:
                    raise ValueError(f"Unknown reg_type: {reg_type}")

            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)

        train_loss = total_loss / max(1, len(tr_loader.dataset))

        model.eval()
        total_vloss = 0.0
        with torch.no_grad():
            for batch in va_loader:
                if len(batch) == 2:
                    xb, tb = batch
                    wb = None
                else:
                    xb, tb, wb = batch
                xb = xb.to(device)
                tb = tb.to(device)
                if wb is not None:
                    wb = wb.to(device).float()

                logits = model(xb)
                per_ex = soft_cross_entropy_per_example(logits, tb)
                vloss = (per_ex * wb).mean() if wb is not None else per_ex.mean()
                total_vloss += float(vloss.item()) * xb.size(0)

        val_loss = total_vloss / max(1, len(va_loader.dataset))
        val_score = -val_loss

        print(f"[sel] epoch {ep:02d} | train_loss={train_loss:.4f} | val_softCE={val_loss:.4f}")

        if val_score > best_score + min_delta:
            best_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[sel] early stop. best_val_score={best_score:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, float(best_score)


# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None,
                    help="Train only one dataset (hotpotqa|squad_v2|pubmedqa_v2|commonsenseqa|combined_pubmed_csqa_gate)")
    ap.add_argument("--out_dir", type=str, default="results/two_stage_utility", help="Where to save trained models")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min_delta", type=float, default=2e-4)

    ap.add_argument("--gate_objective", type=str, default="delta_reg", choices=["delta_reg", "cls"])
    ap.add_argument("--gate_delta", type=float, default=0.02)
    ap.add_argument("--gate_huber_delta", type=float, default=0.1)

    ap.add_argument("--sel_tau", type=float, default=0.2)
    ap.add_argument("--min_train_selector", type=int, default=20)
    ap.add_argument("--min_val_selector", type=int, default=5)

    ap.add_argument("--selector_balanced_sampler", action="store_true")
    ap.add_argument("--sel_reg_type", type=str, default="none", choices=["none", "entropy", "kl"])
    ap.add_argument("--sel_reg_weight", type=float, default=0.0)
    ap.add_argument("--sel_prior_mode", type=str, default="balanced", choices=["balanced", "uniform", "empirical"])
    ap.add_argument("--sel_margin_min", type=float, default=0.0)
    ap.add_argument("--sel_margin_max", type=float, default=1e9)
    ap.add_argument("--print_target_stats", action="store_true")

    ap.add_argument("--sel_use_margin_weighting", action="store_true")
    ap.add_argument("--sel_margin_scale", type=float, default=0.05)
    ap.add_argument("--sel_weight_min", type=float, default=0.2)
    ap.add_argument("--sel_weight_max", type=float, default=1.0)

    ap.add_argument("--tradeoff_mode", action="store_true")

    ap.add_argument("--feature_files", type=str, default=None)
    ap.add_argument("--feature_keys", type=str, default=None)
    ap.add_argument("--standardize_features", action="store_true")
    ap.add_argument("--save_feature_stats", action="store_true")

    ap.add_argument("--pubmed_policy", type=str, default="none", choices=["forced", "none"],
                    help="forced: pubmedqa_v2 policy=True (always RAG). none: allow routing (policy=None).")

    ap.add_argument("--use_passage_embeddings", action="store_true",
                    help="Append mean embedding of retrieved <DOCUMENT> passages from prediction_raw.")
    ap.add_argument("--passage_source_expert", type=str, default="base_rag",
                    help="Which expert's prediction_raw to parse for <DOCUMENT> passages.")
    ap.add_argument("--passage_max_docs", type=int, default=5)
    ap.add_argument("--passage_max_chars", type=int, default=1200)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_cfg()
    tcfg = tradeoff_from_cfg(cfg)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    feature_paths: List[Path] = []
    if args.feature_files:
        feature_paths = [Path(x.strip()) for x in args.feature_files.split(",") if x.strip()]
    fmap: Dict[str, Dict[str, float]] = load_feature_map(feature_paths) if feature_paths else {}

    feature_keys: List[str] = []
    if args.feature_keys:
        feature_keys = [x.strip() for x in args.feature_keys.split(",") if x.strip()]
    elif fmap:
        feature_keys = infer_feature_keys_from_map(fmap)

    # ----------------------------
    # Shared gate mode
    # ----------------------------
    if args.only == SPECIAL_ONLY:
        rows_pub = read_router_train("pubmedqa_v2")
        rows_cs = read_router_train("commonsenseqa")
        rows = rows_pub + rows_cs

        questions = [r["question"] for r in rows]
        total = len(rows)

        y = torch.tensor([1] * len(rows_pub) + [0] * len(rows_cs), dtype=torch.long)

        embed_model = "sentence-transformers/all-mpnet-base-v2"
        embedder = Embedder(embed_model, device=args.device)
        Xq = embedder.encode(questions, batch_size=args.batch_size).float().cpu()
        in_dim_q = int(Xq.size(1))

        Xf = build_feature_matrix(rows, fmap, feature_keys) if feature_keys else torch.zeros((total, 0), dtype=torch.float32)
        feat_stats = None
        if args.standardize_features and Xf.size(1) > 0:
            Xf, feat_stats = standardize_features(Xf)

        X = Xq if Xf.size(1) == 0 else torch.cat([Xq, Xf], dim=1)

        if args.use_passage_embeddings:
            Xp = build_passage_embedding_matrix(
                rows, embedder,
                source_expert=args.passage_source_expert,
                max_docs=args.passage_max_docs,
                max_chars=args.passage_max_chars,
                batch_size_docs=args.batch_size,
            )
            X = torch.cat([X, Xp], dim=1)

        in_dim = int(X.size(1))

        model_dir = out_root / SPECIAL_ONLY
        model_dir.mkdir(parents=True, exist_ok=True)
        gate_path = model_dir / "gate.pt"

        print(f"\n=== {SPECIAL_ONLY} === total={total} (pubmed={len(rows_pub)} csqa={len(rows_cs)}) embed={embed_model}")
        if feature_keys:
            print(f"[{SPECIAL_ONLY}] features: k={len(feature_keys)} | base_dim={in_dim_q} feat_dim={len(feature_keys)} pass={'on' if args.use_passage_embeddings else 'off'} => in_dim={in_dim}")
        if args.use_passage_embeddings:
            print(f"[{SPECIAL_ONLY}] passage embeddings: enabled | src={args.passage_source_expert} max_docs={args.passage_max_docs} max_chars={args.passage_max_chars}")

        gate_model = MLP(in_dim, args.hidden_dim, args.dropout, out_dim=2)
        gate_model, best_acc = train_gate_classifier(
            gate_model, X, y,
            device=args.device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            seed=args.seed,
            min_val=args.min_val_selector,
        )

        ckpt = {
            "state_dict": gate_model.state_dict(),
            "in_dim": in_dim,
            "embed_model": embed_model,
            "feature_keys": feature_keys,
            "gate_objective": "cls",
            "calibrated_threshold": 0.5,
            "best_val_acc": float(best_acc),
            "trained_on": {"pubmedqa_v2": len(rows_pub), "commonsenseqa": len(rows_cs)},
            "used_passage_embeddings": bool(args.use_passage_embeddings),
            "passage_source_expert": str(args.passage_source_expert),
            "passage_max_docs": int(args.passage_max_docs),
            "passage_max_chars": int(args.passage_max_chars),
        }
        if args.standardize_features and feat_stats is not None:
            ckpt["feature_stats"] = feat_stats

        torch.save(ckpt, gate_path)
        print(f"[{SPECIAL_ONLY}] saved gate(cls) -> {gate_path} (best_val_acc={best_acc:.4f})")
        print(f"\nDONE. Models saved under: {out_root}")
        return

    # ----------------------------
    # Per-dataset training
    # ----------------------------
    for dataset in DATASETS:
        if args.only and dataset != args.only:
            continue

        rows = read_router_train(dataset)
        questions = [r["question"] for r in rows]
        total = len(rows)

        ds_cfg = (cfg.get("datasets", {}) or {}).get(dataset, {}) or {}
        embed_models = ds_cfg.get("embed_models") or ["sentence-transformers/all-mpnet-base-v2"]
        if isinstance(embed_models, str):
            embed_models = [embed_models]
        embed_model = str(embed_models[0])

        embedder = Embedder(embed_model, device=args.device)
        Xq = embedder.encode(questions, batch_size=args.batch_size).float().cpu()
        in_dim_q = int(Xq.size(1))

        Xf = build_feature_matrix(rows, fmap, feature_keys) if feature_keys else torch.zeros((total, 0), dtype=torch.float32)
        feat_stats = None
        if args.standardize_features and Xf.size(1) > 0:
            Xf, feat_stats = standardize_features(Xf)

        X = Xq if Xf.size(1) == 0 else torch.cat([Xq, Xf], dim=1)

        if args.use_passage_embeddings:
            Xp = build_passage_embedding_matrix(
                rows, embedder,
                source_expert=args.passage_source_expert,
                max_docs=args.passage_max_docs,
                max_chars=args.passage_max_chars,
                batch_size_docs=args.batch_size,
            )
            X = torch.cat([X, Xp], dim=1)

        in_dim = int(X.size(1))

        pol = policy_for_dataset(dataset, pubmed_policy_mode=args.pubmed_policy)
        rag_pool_ds, no_pool_ds = pools_for_dataset(dataset)

        print(f"\n=== {dataset} === total={total} policy={pol} embed={embed_model}")
        if args.tradeoff_mode:
            print(f"[{dataset}] tradeoff_mode=True (utility-based, matches eval)")
        if feature_keys:
            print(f"[{dataset}] features: k={len(feature_keys)} | base_dim={in_dim_q} feat_dim={len(feature_keys)} pass={'on' if args.use_passage_embeddings else 'off'} => in_dim={in_dim}")
        if args.use_passage_embeddings:
            print(f"[{dataset}] passage embeddings: enabled | src={args.passage_source_expert} max_docs={args.passage_max_docs} max_chars={args.passage_max_chars}")
        if dataset == "pubmedqa_v2":
            print(f"[{dataset}] pubmed_policy={args.pubmed_policy}")
        print(f"[{dataset}] pools | rag={rag_pool_ds} no={no_pool_ds}")

        model_dir = out_root / dataset
        model_dir.mkdir(parents=True, exist_ok=True)

        gate_path = model_dir / "gate.pt"

        # ---- Gate ----
        if pol is None:
            deltas = build_gate_delta_targets(cfg, dataset, rows, use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg)

            if args.gate_objective == "cls":
                idx_gate, y_gate = build_gate_deadzone_from_deltas(deltas, gate_delta=args.gate_delta)
                if len(idx_gate) < 200:
                    print(f"[{dataset}] gate-cls: too few examples after deadzone (kept={len(idx_gate)}). Lower gate_delta.")
                X_gate = X[idx_gate] if len(idx_gate) > 0 else X[:0]

                gate_model = MLP(in_dim, args.hidden_dim, args.dropout, out_dim=2)
                if y_gate.numel() > 0:
                    gate_model, best_acc = train_gate_classifier(
                        gate_model, X_gate, y_gate,
                        device=args.device,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        patience=args.patience,
                        min_delta=args.min_delta,
                        seed=args.seed,
                        min_val=args.min_val_selector,
                    )
                else:
                    best_acc = 0.0
                    gate_model.eval()

                ckpt = {
                    "state_dict": gate_model.state_dict(),
                    "in_dim": in_dim,
                    "embed_model": embed_model,
                    "feature_keys": feature_keys,
                    "gate_objective": "cls",
                    "gate_delta": float(args.gate_delta),
                    "best_val_acc": float(best_acc),
                    "tradeoff_mode": bool(args.tradeoff_mode),
                    "used_passage_embeddings": bool(args.use_passage_embeddings),
                    "passage_source_expert": str(args.passage_source_expert),
                    "passage_max_docs": int(args.passage_max_docs),
                    "passage_max_chars": int(args.passage_max_chars),
                }
                if args.standardize_features and feat_stats is not None:
                    ckpt["feature_stats"] = feat_stats

                torch.save(ckpt, gate_path)
                print(f"[{dataset}] saved gate(cls) -> {gate_path} (best_val_acc={best_acc:.4f})")

            else:
                gate_model = MLP(in_dim, args.hidden_dim, args.dropout, out_dim=1)
                gate_model, info = train_gate_delta_regressor(
                    gate_model, X, deltas,
                    device=args.device,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    patience=args.patience,
                    min_delta=args.min_delta,
                    seed=args.seed,
                    min_val=args.min_val_selector,
                    huber_delta=float(args.gate_huber_delta),
                )

                ckpt = {
                    "state_dict": gate_model.state_dict(),
                    "in_dim": in_dim,
                    "embed_model": embed_model,
                    "feature_keys": feature_keys,
                    "gate_objective": "delta_reg",
                    "gate_huber_delta": float(args.gate_huber_delta),
                    "calibrated_threshold": float(info["best_threshold"]),
                    "best_val_sign_acc": float(info["best_val_sign_acc"]),
                    "best_val_huber": float(info["best_val_huber"]),
                    "tradeoff_mode": bool(args.tradeoff_mode),
                    "used_passage_embeddings": bool(args.use_passage_embeddings),
                    "passage_source_expert": str(args.passage_source_expert),
                    "passage_max_docs": int(args.passage_max_docs),
                    "passage_max_chars": int(args.passage_max_chars),
                }
                if args.standardize_features and feat_stats is not None:
                    ckpt["feature_stats"] = feat_stats

                torch.save(ckpt, gate_path)
                print(f"[{dataset}] saved gate(delta_reg) -> {gate_path} (thr={info['best_threshold']:.4f} val_sign_acc={info['best_val_sign_acc']:.4f})")

        else:
            ckpt = {
                "forced_policy": bool(pol),
                "in_dim": in_dim,
                "embed_model": embed_model,
                "feature_keys": feature_keys,
                "tradeoff_mode": bool(args.tradeoff_mode),
                "used_passage_embeddings": bool(args.use_passage_embeddings),
                "passage_source_expert": str(args.passage_source_expert),
                "passage_max_docs": int(args.passage_max_docs),
                "passage_max_chars": int(args.passage_max_chars),
            }
            if args.standardize_features and feat_stats is not None:
                ckpt["feature_stats"] = feat_stats
            torch.save(ckpt, gate_path)
            print(f"[{dataset}] gate forced={pol}; wrote marker -> {gate_path}")

        # ---- Selectors ----
        sel_rag_path = model_dir / "selector_rag.pt"
        sel_no_path = model_dir / "selector_no_rag.pt"

        idx_rag: List[int] = []
        idx_no: List[int] = []

        if pol is True:
            idx_rag = list(range(total))
            idx_no = []
        elif pol is False:
            idx_rag = []
            idx_no = list(range(total))
        else:
            deltas = build_gate_delta_targets(cfg, dataset, rows, use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg)
            for i in range(total):
                d = float(deltas[i].item())
                if d >= float(args.gate_delta):
                    idx_rag.append(i)
                elif d <= -float(args.gate_delta):
                    idx_no.append(i)

        idx_rag_f = filter_by_margin_window(
            cfg, dataset, rows, idx_rag, rag_pool_ds,
            margin_min=args.sel_margin_min, margin_max=args.sel_margin_max,
            use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg,
        )
        idx_no_f = filter_by_margin_window(
            cfg, dataset, rows, idx_no, no_pool_ds,
            margin_min=args.sel_margin_min, margin_max=args.sel_margin_max,
            use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg,
        )

        print(f"[{dataset}] selector train counts | rag={len(idx_rag)}→{len(idx_rag_f)} no={len(idx_no)}→{len(idx_no_f)} "
              f"(gate_delta={args.gate_delta}, sel_margin=[{args.sel_margin_min},{args.sel_margin_max}])")

        def make_selector_prior(y_hard: torch.Tensor, num_classes: int) -> torch.Tensor:
            if args.sel_prior_mode in ("balanced", "uniform"):
                return torch.full((num_classes,), 1.0 / float(num_classes), dtype=torch.float32)
            counts = torch.bincount(y_hard, minlength=num_classes).float()
            if float(counts.sum().item()) <= 0:
                return torch.full((num_classes,), 1.0 / float(num_classes), dtype=torch.float32)
            return (counts / counts.sum()).clamp_min(1e-8)

        # RAG selector
        if should_train_selector(len(idx_rag_f), min_train=args.min_train_selector, min_val=args.min_val_selector, name="selector_rag", ds=dataset):
            Xg = X[idx_rag_f]
            y_soft = build_selector_soft_targets(
                cfg, dataset, rows, idx_rag_f, rag_pool_ds,
                tau=args.sel_tau, use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg
            )
            y_hard = hard_argmax_from_soft(y_soft)

            prior = None
            if args.sel_reg_type == "kl" and y_hard.numel() > 0:
                prior = make_selector_prior(y_hard, len(rag_pool_ds))

            ex_w = None
            if args.sel_use_margin_weighting:
                ex_w = selector_margin_weights(
                    cfg, dataset, rows, idx_rag_f, rag_pool_ds,
                    use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg,
                    margin_scale=float(args.sel_margin_scale),
                    weight_min=float(args.sel_weight_min),
                    weight_max=float(args.sel_weight_max),
                )

            dropout_used = 0.0 if len(idx_rag_f) < 200 else float(args.dropout)
            sel_model = MLP(in_dim, args.hidden_dim, dropout_used, out_dim=len(rag_pool_ds))

            sel_model, best_score = train_selector_soft(
                sel_model, Xg, y_soft,
                device=args.device,
                lr=args.lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                min_delta=args.min_delta,
                seed=args.seed,
                min_val=args.min_val_selector,
                balanced_sampler=bool(args.selector_balanced_sampler),
                reg_type=str(args.sel_reg_type),
                reg_weight=float(args.sel_reg_weight),
                prior=prior,
                hard_labels_for_sampler=y_hard,
                example_weights=ex_w,
            )

            ckpt = {
                "state_dict": sel_model.state_dict(),
                "in_dim": in_dim,
                "experts": rag_pool_ds,
                "embed_model": embed_model,
                "feature_keys": feature_keys,
                "sel_tau": float(args.sel_tau),
                "trained_on_gate_delta": float(args.gate_delta),
                "trained_on_sel_margin_min": float(args.sel_margin_min),
                "trained_on_sel_margin_max": float(args.sel_margin_max),
                "sel_use_margin_weighting": bool(args.sel_use_margin_weighting),
                "sel_margin_scale": float(args.sel_margin_scale),
                "sel_weight_min": float(args.sel_weight_min),
                "sel_weight_max": float(args.sel_weight_max),
                "reg_type": str(args.sel_reg_type),
                "reg_weight": float(args.sel_reg_weight),
                "prior_mode": str(args.sel_prior_mode),
                "prior": (prior.tolist() if prior is not None else None),
                "tradeoff_mode": bool(args.tradeoff_mode),
                "best_val_score": float(best_score),
                "used_passage_embeddings": bool(args.use_passage_embeddings),
                "passage_source_expert": str(args.passage_source_expert),
                "passage_max_docs": int(args.passage_max_docs),
                "passage_max_chars": int(args.passage_max_chars),
            }
            if args.standardize_features and feat_stats is not None:
                ckpt["feature_stats"] = feat_stats

            torch.save(ckpt, sel_rag_path)
            print(f"[{dataset}] saved selector_rag -> {sel_rag_path} (best_val_score={best_score:.4f})")
        else:
            print(f"[{dataset}] skip selector_rag (count={len(idx_rag_f)})")

        # NO selector
        if pol is None:
            if should_train_selector(len(idx_no_f), min_train=args.min_train_selector, min_val=args.min_val_selector, name="selector_no_rag", ds=dataset):
                Xg = X[idx_no_f]
                y_soft = build_selector_soft_targets(
                    cfg, dataset, rows, idx_no_f, no_pool_ds,
                    tau=args.sel_tau, use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg
                )
                y_hard = hard_argmax_from_soft(y_soft)

                prior = None
                if args.sel_reg_type == "kl" and y_hard.numel() > 0:
                    prior = make_selector_prior(y_hard, len(no_pool_ds))

                ex_w = None
                if args.sel_use_margin_weighting:
                    ex_w = selector_margin_weights(
                        cfg, dataset, rows, idx_no_f, no_pool_ds,
                        use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg,
                        margin_scale=float(args.sel_margin_scale),
                        weight_min=float(args.sel_weight_min),
                        weight_max=float(args.sel_weight_max),
                    )

                dropout_used = 0.0 if len(idx_no_f) < 200 else float(args.dropout)
                sel_model = MLP(in_dim, args.hidden_dim, dropout_used, out_dim=len(no_pool_ds))

                sel_model, best_score = train_selector_soft(
                    sel_model, Xg, y_soft,
                    device=args.device,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    patience=args.patience,
                    min_delta=args.min_delta,
                    seed=args.seed,
                    min_val=args.min_val_selector,
                    balanced_sampler=bool(args.selector_balanced_sampler),
                    reg_type=str(args.sel_reg_type),
                    reg_weight=float(args.sel_reg_weight),
                    prior=prior,
                    hard_labels_for_sampler=y_hard,
                    example_weights=ex_w,
                )

                ckpt = {
                    "state_dict": sel_model.state_dict(),
                    "in_dim": in_dim,
                    "experts": no_pool_ds,
                    "embed_model": embed_model,
                    "feature_keys": feature_keys,
                    "sel_tau": float(args.sel_tau),
                    "trained_on_gate_delta": float(args.gate_delta),
                    "trained_on_sel_margin_min": float(args.sel_margin_min),
                    "trained_on_sel_margin_max": float(args.sel_margin_max),
                    "sel_use_margin_weighting": bool(args.sel_use_margin_weighting),
                    "sel_margin_scale": float(args.sel_margin_scale),
                    "sel_weight_min": float(args.sel_weight_min),
                    "sel_weight_max": float(args.sel_weight_max),
                    "reg_type": str(args.sel_reg_type),
                    "reg_weight": float(args.sel_reg_weight),
                    "prior_mode": str(args.sel_prior_mode),
                    "prior": (prior.tolist() if prior is not None else None),
                    "tradeoff_mode": bool(args.tradeoff_mode),
                    "best_val_score": float(best_score),
                    "used_passage_embeddings": bool(args.use_passage_embeddings),
                    "passage_source_expert": str(args.passage_source_expert),
                    "passage_max_docs": int(args.passage_max_docs),
                    "passage_max_chars": int(args.passage_max_chars),
                }
                if args.standardize_features and feat_stats is not None:
                    ckpt["feature_stats"] = feat_stats
                torch.save(ckpt, sel_no_path)
                print(f"[{dataset}] saved selector_no_rag -> {sel_no_path} (best_val_score={best_score:.4f})")
            else:
                print(f"[{dataset}] skip selector_no_rag (count={len(idx_no_f)})")
        else:
            if pol is True:
                print(f"[{dataset}] policy=True -> skip selector_no_rag")

    print(f"\nDONE. Models saved under: {out_root}")


if __name__ == "__main__":
    main()