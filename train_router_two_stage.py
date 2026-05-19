# train_router_two_stage.py

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
CANON_NO_EXPERTS = ["base_only", "sft_only"]

DATASETS = ["hotpotqa", "squad_v2", "pubmedqa_v2", "commonsenseqa"]
SPECIAL_ONLY = "combined_pubmed_csqa_gate"
SELECTOR_CMP_PREFIX = "selcmp__"
_DOC_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.DOTALL)


# --------------------------
# Config
# --------------------------

def load_cfg() -> Dict[str, Any]:
    if not CFG_PATH.exists():
        raise SystemExit(f"Missing {CFG_PATH}")
    return json.loads(CFG_PATH.read_text(encoding="utf-8-sig"))


# --------------------------
# Utility / tradeoff
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
# Pools / policy
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
        return "", -1e18
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


def policy_for_dataset(dataset: str, *, pubmed_policy_mode: str = "none") -> Optional[bool]:
    #if dataset in ("hotpotqa", "squad_v2"):
    #   return True
    #if dataset == "pubmedqa_v2":
    #   return True if pubmed_policy_mode == "forced" else None
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
            rid_raw = r.get("orig_id", r.get("id"))
            if rid_raw is None:
                continue
            rid_raw = str(rid_raw)

            ds = r.get("dataset")
            rid_key = f"{str(ds)}::{rid_raw}" if ds is not None else rid_raw

            feats = r.get("features")
            if feats is None:
                feats = {k: v for k, v in r.items() if k not in ("id", "orig_id", "dataset", "question", "features")}
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


def split_feature_keys_for_models(feature_keys: List[str]) -> Tuple[List[str], List[str]]:
    gate_keys = [k for k in feature_keys if not str(k).startswith(SELECTOR_CMP_PREFIX)]
    selector_keys = list(feature_keys)
    return gate_keys, selector_keys


def build_feature_matrix(rows: List[Dict[str, Any]], fmap: Dict[str, Dict[str, float]], feature_keys: List[str]) -> torch.Tensor:
    n = len(rows)
    d = len(feature_keys)
    Xf = torch.zeros((n, d), dtype=torch.float32)
    if d == 0:
        return Xf
    for i, r in enumerate(rows):
        rid = str(r.get("id", i))
        ds = str(r.get("dataset", ""))
        rid_key = f"{ds}::{rid}" if ds else rid
        feats = fmap.get(rid_key, {})
        for j, k in enumerate(feature_keys):
            if k in feats:
                Xf[i, j] = float(feats[k])
    return Xf


def report_feature_coverage(tag: str, Xf: torch.Tensor) -> None:
    if Xf.numel() == 0 or Xf.size(1) == 0:
        print(f"[FEAT-CHECK][{tag}] no feature columns")
        return
    matched_rows = int((Xf.abs().sum(dim=1) > 0).sum().item())
    mean_active = float((Xf != 0).sum(dim=1).float().mean().item())
    print(f"[FEAT-CHECK][{tag}] matched_rows={matched_rows}/{Xf.size(0)} mean_active_feats={mean_active:.2f}")


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

    E = embedder.encode(flat_docs, batch_size=batch_size_docs).float().cpu()

    Xp = torch.zeros((len(rows), d), dtype=torch.float32)
    for i, (s, e) in enumerate(offsets):
        if e > s:
            Xp[i] = E[s:e].mean(dim=0)
    return Xp


# --------------------------
# Dataset wrappers
# --------------------------

class TensorDatasetXY(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, w: Optional[torch.Tensor] = None):
        self.X = X
        self.y = y
        self.w = w

    def __len__(self):
        return int(self.X.size(0))

    def __getitem__(self, idx):
        if self.w is None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.y[idx], self.w[idx]


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


def weighted_mean(x: torch.Tensor, w: Optional[torch.Tensor]) -> torch.Tensor:
    if w is None:
        return x.mean()
    return (x * w).sum() / w.sum().clamp_min(1e-8)


def build_gate_example_weights(
        deltas: torch.Tensor,
        *,
        power: float = 1.0,
        weight_min: float = 0.25,
        weight_max: float = 4.0,
) -> torch.Tensor:
    if deltas.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)

    mag = deltas.abs().float()
    scale = torch.median(mag).clamp_min(1e-6)
    w = (mag / scale).pow(float(power))
    w = w.clamp(float(weight_min), float(weight_max))
    w = w / w.mean().clamp_min(1e-6)
    return w.float()


def selector_hard_utility_score(
        model: nn.Module,
        X: torch.Tensor,
        U_targets: torch.Tensor,
        *,
        device: str,
        batch_size: int,
) -> float:
    if X.size(0) == 0:
        return 0.0

    loader = DataLoader(TensorDatasetSoft(X, U_targets), batch_size=batch_size, shuffle=False)
    model = model.to(device)
    model.eval()

    total_u = 0.0
    total_n = 0
    with torch.no_grad():
        for xb, ub in loader:
            xb = xb.to(device)
            ub = ub.to(device).float()
            logits = model(xb)
            hard_idx = torch.argmax(logits, dim=1)
            hard_u = ub.gather(1, hard_idx.unsqueeze(1)).squeeze(1)
            total_u += float(hard_u.sum().item())
            total_n += int(xb.size(0))
    return total_u / max(1, total_n)


def _quantiles_np(a: np.ndarray, qs: List[float]) -> List[float]:
    if a.size == 0:
        return [0.0 for _ in qs]
    return [float(np.quantile(a, q)) for q in qs]


def print_gate_target_diagnostics(dataset: str, deltas: torch.Tensor, gate_delta: float) -> None:
    if deltas.numel() == 0:
        print(f"[DIAG][{dataset}][gate] empty deltas")
        return

    arr = deltas.detach().cpu().numpy().astype(np.float64)
    abs_arr = np.abs(arr)

    q10, q25, q50, q75, q90 = _quantiles_np(arr, [0.10, 0.25, 0.50, 0.75, 0.90])

    frac_pos = float((arr > 0).mean())
    frac_neg = float((arr < 0).mean())
    frac_dead = float((abs_arr < float(gate_delta)).mean())
    frac_strong_rag = float((arr >= float(gate_delta)).mean())
    frac_strong_no = float((arr <= -float(gate_delta)).mean())

    print(
        f"[DIAG][{dataset}][gate] "
        f"delta_mean={arr.mean():.4f} delta_std={arr.std():.4f} "
        f"q10={q10:.4f} q25={q25:.4f} q50={q50:.4f} q75={q75:.4f} q90={q90:.4f}"
    )
    print(
        f"[DIAG][{dataset}][gate] "
        f"frac_pos={frac_pos:.4f} frac_neg={frac_neg:.4f} "
        f"frac_strong_rag={frac_strong_rag:.4f} frac_strong_no={frac_strong_no:.4f} "
        f"frac_deadzone={frac_dead:.4f} gate_delta={float(gate_delta):.4f}"
    )


def _eta_squared_feature(x: np.ndarray, y: np.ndarray, num_classes: int) -> float:
    mask = np.isfinite(x)
    x = x[mask]
    y = y[mask]

    if x.size < 2 or num_classes <= 1:
        return 0.0

    var_total = float(np.var(x))
    if (not np.isfinite(var_total)) or var_total < 1e-12:
        return 0.0

    grand_mean = float(np.mean(x))
    ss_between = 0.0
    for c in range(num_classes):
        xc = x[y == c]
        if xc.size == 0:
            continue
        mc = float(np.mean(xc))
        ss_between += float(xc.size) * ((mc - grand_mean) ** 2)

    ss_total = float(x.size) * var_total
    return float(ss_between / max(1e-12, ss_total))


def print_selector_domain_diagnostics(
        dataset: str,
        name: str,
        U_full: torch.Tensor,
        experts: List[str],
        Xf_full: torch.Tensor,
        feature_keys: List[str],
        *,
        top_k: int = 12,
) -> None:
    if U_full.numel() == 0:
        print(f"[DIAG][{dataset}][{name}] empty utility matrix")
        return

    avg_u = U_full.mean(dim=0).detach().cpu().numpy()
    y_oracle = U_full.argmax(dim=1).detach().cpu().numpy()
    cnt = Counter(y_oracle.tolist())
    dist = [(experts[i], int(cnt.get(i, 0))) for i in range(len(experts))]

    oracle_u = float(U_full.max(dim=1).values.mean().item())
    const_idx = int(np.argmax(avg_u))
    const_expert = experts[const_idx]
    const_u = float(avg_u[const_idx])
    headroom = float(oracle_u - const_u)
    avg_u_pairs = [(experts[i], round(float(avg_u[i]), 4)) for i in range(len(experts))]

    print(f"[DIAG][{dataset}][{name}] oracle_counts={dist}")
    print(f"[DIAG][{dataset}][{name}] avg_u_by_expert={avg_u_pairs}")
    print(
        f"[DIAG][{dataset}][{name}] "
        f"oracle_u={oracle_u:.4f} best_constant={const_expert}:{const_u:.4f} headroom={headroom:.4f}"
    )

    if U_full.size(1) > 1:
        sorted_u = torch.sort(U_full, dim=1, descending=True).values
        margins = (sorted_u[:, 0] - sorted_u[:, 1]).detach().cpu().numpy()
        mq25, mq50, mq75 = _quantiles_np(margins, [0.25, 0.50, 0.75])
        print(
            f"[DIAG][{dataset}][{name}] "
            f"oracle_margin_mean={float(margins.mean()):.4f} "
            f"p25={mq25:.4f} p50={mq50:.4f} p75={mq75:.4f}"
        )
    else:
        print(f"[DIAG][{dataset}][{name}] trivial_single_expert=True")

    if len(experts) <= 1:
        return

    if Xf_full.numel() == 0 or len(feature_keys) == 0:
        print(f"[DIAG][{dataset}][{name}] no feature block available for selector diagnostics")
        return

    selcmp_idx = [i for i, k in enumerate(feature_keys) if str(k).startswith(SELECTOR_CMP_PREFIX)]
    if len(selcmp_idx) == 0:
        print(f"[DIAG][{dataset}][{name}] no selcmp features found")
        return

    Xn = Xf_full.detach().cpu().numpy()
    rows = []
    for j in selcmp_idx:
        score = _eta_squared_feature(Xn[:, j], y_oracle, len(experts))
        if np.isfinite(score) and score > 0.0:
            rows.append((float(score), str(feature_keys[j])))

    rows.sort(key=lambda z: z[0], reverse=True)
    top = rows[:top_k]

    if len(top) == 0:
        print(f"[DIAG][{dataset}][{name}] selcmp_separation_top=[]")
    else:
        pretty = [(k, round(v, 4)) for v, k in top]
        print(f"[DIAG][{dataset}][{name}] selcmp_separation_top={pretty}")


# --------------------------
# Gate
# --------------------------

def build_gate_delta_targets(
        cfg: Dict[str, Any],
        dataset: str,
        rows: List[Dict[str, Any]],
        *,
        use_tradeoff: bool,
        tcfg: Dict[str, Any],
) -> torch.Tensor:
    rag_pool_ds, no_pool_ds = pools_for_dataset(dataset)

    deltas: List[float] = []
    for r in rows:
        ex = r["experts"]
        rag_pool, no_pool = pools_for_row(ex, rag_pool_ds, no_pool_ds)

        if len(rag_pool) == 0 or len(no_pool) == 0:
            deltas.append(0.0)
            continue

        _, br_u = _best_in_pool(cfg, dataset, ex, rag_pool, use_tradeoff=use_tradeoff, tcfg=tcfg)
        _, bn_u = _best_in_pool(cfg, dataset, ex, no_pool, use_tradeoff=use_tradeoff, tcfg=tcfg)
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
        example_weights: Optional[torch.Tensor] = None,
) -> Tuple[nn.Module, float]:
    tr_idx, va_idx = split_train_val_indices(int(X.size(0)), 0.2, seed, min_val=min_val)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]

    w_tr = None
    w_va = None
    if example_weights is not None and example_weights.numel() == X.size(0):
        w_tr = example_weights[tr_idx]
        w_va = example_weights[va_idx]

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_w = class_weights_from_labels(y_tr, 2, device)

    sampler = make_balanced_sampler_from_hard_labels(y_tr)
    tr_loader = DataLoader(TensorDatasetXY(X_tr, y_tr, w_tr), batch_size=batch_size, sampler=sampler)
    va_loader = DataLoader(TensorDatasetXY(X_va, y_va, w_va), batch_size=batch_size, shuffle=False)

    best_acc = -1.0
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(tr_loader, desc="Batches[gate-cls]", leave=False):
            if len(batch) == 2:
                xb, yb = batch
                wb = None
            else:
                xb, yb, wb = batch

            xb = xb.to(device)
            yb = yb.to(device)
            if wb is not None:
                wb = wb.to(device).float()

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            per_ex = F.cross_entropy(logits, yb.long(), weight=class_w, reduction="none")
            loss = weighted_mean(per_ex, wb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)

        train_loss = total_loss / max(1, len(tr_loader.dataset))

        model.eval()
        correct_sum = 0.0
        weight_sum = 0.0
        with torch.no_grad():
            for batch in va_loader:
                if len(batch) == 2:
                    xb, yb = batch
                    wb = None
                else:
                    xb, yb, wb = batch

                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb).argmax(dim=1)

                if wb is None:
                    correct_sum += float((pred == yb).sum().item())
                    weight_sum += float(yb.numel())
                else:
                    wb = wb.to(device).float()
                    correct_sum += float((((pred == yb).float()) * wb).sum().item())
                    weight_sum += float(wb.sum().item())

        val_acc = correct_sum / max(1e-8, weight_sum)
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
        example_weights: Optional[torch.Tensor] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    tr_idx, va_idx = split_train_val_indices(int(X.size(0)), 0.2, seed, min_val=min_val)
    X_tr, d_tr = X[tr_idx], deltas[tr_idx]
    X_va, d_va = X[va_idx], deltas[va_idx]

    w_tr = None
    w_va = None
    if example_weights is not None and example_weights.numel() == X.size(0):
        w_tr = example_weights[tr_idx]
        w_va = example_weights[va_idx]

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    tr_loader = DataLoader(TensorDatasetXY(X_tr, d_tr, w_tr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDatasetXY(X_va, d_va, w_va), batch_size=batch_size, shuffle=False)

    best_v = 1e18
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(tr_loader, desc="Batches[gate-delta]", leave=False):
            if len(batch) == 2:
                xb, db = batch
                wb = None
            else:
                xb, db, wb = batch

            xb = xb.to(device)
            db = db.to(device).float()
            if wb is not None:
                wb = wb.to(device).float()

            opt.zero_grad(set_to_none=True)
            pred = model(xb).squeeze(1)
            per_ex = F.huber_loss(pred, db, reduction="none", delta=float(huber_delta))
            loss = weighted_mean(per_ex, wb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)

        train_loss = total_loss / max(1, len(tr_loader.dataset))

        model.eval()
        total_v = 0.0
        denom_v = 0.0
        with torch.no_grad():
            for batch in va_loader:
                if len(batch) == 2:
                    xb, db = batch
                    wb = None
                else:
                    xb, db, wb = batch

                xb = xb.to(device)
                db = db.to(device).float()
                if wb is not None:
                    wb = wb.to(device).float()

                pred = model(xb).squeeze(1)
                per_ex = F.huber_loss(pred, db, reduction="none", delta=float(huber_delta))

                if wb is None:
                    total_v += float(per_ex.sum().item())
                    denom_v += float(per_ex.numel())
                else:
                    total_v += float((per_ex * wb).sum().item())
                    denom_v += float(wb.sum().item())

        val_loss = total_v / max(1e-8, denom_v)
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

    if w_va is not None:
        wv = w_va.detach().cpu().numpy().astype(np.float64)
    else:
        wv = np.ones_like(dv, dtype=np.float64)

    thr_cands = np.unique(np.concatenate([np.quantile(pv, np.linspace(0.05, 0.95, 19)), np.array([0.0])]))
    best_acc = -1.0
    best_thr = 0.0
    for thr in thr_cands:
        pred = (pv > thr).astype(np.int64)
        acc = float((((pred == yv).astype(np.float64)) * wv).sum() / max(1e-8, wv.sum()))
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
# Selector targets
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


def build_selector_delta_anchor_matrix(
        cfg: Dict[str, Any],
        dataset: str,
        rows: List[Dict[str, Any]],
        idxs: List[int],
        experts_in_group: List[str],
        anchor_expert: str,
        *,
        use_tradeoff: bool,
        tcfg: Dict[str, Any],
) -> torch.Tensor:
    ys = []
    for i in idxs:
        ex = rows[i]["experts"]
        anchor_u = score_for_targets(cfg, dataset, anchor_expert, ex[anchor_expert], use_tradeoff=use_tradeoff, tcfg=tcfg)
        deltas = []
        for e in experts_in_group:
            u = score_for_targets(cfg, dataset, e, ex[e], use_tradeoff=use_tradeoff, tcfg=tcfg)
            deltas.append(float(u - anchor_u))
        ys.append(np.array(deltas, dtype=np.float32))
    if not ys:
        return torch.zeros((0, len(experts_in_group)), dtype=torch.float32)
    return torch.from_numpy(np.stack(ys, axis=0))


def build_selector_utility_matrix(
        cfg: Dict[str, Any],
        dataset: str,
        rows: List[Dict[str, Any]],
        idxs: List[int],
        experts_in_group: List[str],
        *,
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
        ys.append(scores)
    if not ys:
        return torch.zeros((0, len(experts_in_group)), dtype=torch.float32)
    return torch.from_numpy(np.stack(ys, axis=0))


def compute_constant_fallback_expert(
        cfg: Dict[str, Any],
        dataset: str,
        rows: List[Dict[str, Any]],
        idxs: List[int],
        experts_in_group: List[str],
        *,
        use_tradeoff: bool,
        tcfg: Dict[str, Any],
) -> Tuple[str, Dict[str, float]]:
    if len(experts_in_group) == 0:
        return "", {}
    if len(idxs) == 0:
        return experts_in_group[0], {e: 0.0 for e in experts_in_group}

    U_all = build_selector_utility_matrix(cfg, dataset, rows, idxs, experts_in_group, use_tradeoff=use_tradeoff, tcfg=tcfg)
    avg_u = U_all.mean(dim=0)
    best_idx = int(avg_u.argmax().item())
    best_expert = experts_in_group[best_idx]
    avg_map = {experts_in_group[j]: float(avg_u[j].item()) for j in range(len(experts_in_group))}
    return best_expert, avg_map


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


# --------------------------
# Selector trainers
# --------------------------

def train_selector_soft(
        model: nn.Module,
        X: torch.Tensor,
        y_soft: torch.Tensor,
        U_eval: torch.Tensor,
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
) -> Tuple[nn.Module, Dict[str, Any]]:
    tr_idx, va_idx = split_train_val_indices(int(X.size(0)), 0.2, seed, min_val=min_val)
    X_tr, y_tr = X[tr_idx], y_soft[tr_idx]
    X_va, y_va = X[va_idx], y_soft[va_idx]
    U_va = U_eval[va_idx]

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

    best_metric = -1e18
    best_state = None
    bad = 0
    best_val_softce = 1e18
    best_val_argmax_u = -1e18

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(tr_loader, desc="Batches[sel-softCE]", leave=False):
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
            loss = weighted_mean(per_ex, wb)

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
        denom = 0.0
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

                if wb is None:
                    total_vloss += float(per_ex.sum().item())
                    denom += float(per_ex.numel())
                else:
                    total_vloss += float((per_ex * wb).sum().item())
                    denom += float(wb.sum().item())

        val_softce = total_vloss / max(1e-8, denom)
        val_argmax_u = selector_hard_utility_score(model, X_va, U_va, device=device, batch_size=batch_size)

        print(
            f"[sel-softCE] epoch {ep:02d} | train_loss={train_loss:.4f} "
            f"| val_softCE={val_softce:.4f} | val_argmax_u={val_argmax_u:.4f}"
        )

        if val_argmax_u > best_metric + min_delta:
            best_metric = val_argmax_u
            best_val_softce = val_softce
            best_val_argmax_u = val_argmax_u
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[sel-softCE] early stop. best_val_argmax_u={best_metric:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    if U_va.numel() == 0:
        best_const_idx = 0
        best_const_val_u = -1e18
    else:
        const_scores = [float(U_va[:, j].mean().item()) for j in range(U_va.size(1))]
        best_const_idx = int(np.argmax(const_scores))
        best_const_val_u = float(const_scores[best_const_idx])

    info = {
        "best_val_softce": float(best_val_softce),
        "best_val_argmax_u": float(best_val_argmax_u),
        "best_constant_val_u": float(best_const_val_u),
        "best_constant_expert_idx": int(best_const_idx),
        "val_size": int(len(va_idx)),
    }
    return model, info


def train_selector_hard_ce(
        model: nn.Module,
        X: torch.Tensor,
        y_hard: torch.Tensor,
        U_targets: torch.Tensor,
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
        example_weights: Optional[torch.Tensor],
) -> Tuple[nn.Module, Dict[str, Any]]:
    tr_idx, va_idx = split_train_val_indices(int(X.size(0)), 0.2, seed, min_val=min_val)
    X_tr, y_tr = X[tr_idx], y_hard[tr_idx]
    X_va, y_va = X[va_idx], y_hard[va_idx]
    U_va = U_targets[va_idx]

    w_tr = None
    w_va = None
    if example_weights is not None and example_weights.numel() == X.size(0):
        w_tr = example_weights[tr_idx]
        w_va = example_weights[va_idx]

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_w = class_weights_from_labels(y_tr, int(U_targets.size(1)), device)

    if balanced_sampler:
        sampler = make_balanced_sampler_from_hard_labels(y_tr)
        tr_loader = DataLoader(TensorDatasetXY(X_tr, y_tr, w_tr), batch_size=batch_size, sampler=sampler)
    else:
        tr_loader = DataLoader(TensorDatasetXY(X_tr, y_tr, w_tr), batch_size=batch_size, shuffle=True)

    va_loader = DataLoader(TensorDatasetXY(X_va, y_va, w_va), batch_size=batch_size, shuffle=False)

    best_metric = -1e18
    best_state = None
    bad = 0
    best_val_acc = -1.0
    best_val_argmax_u = -1e18

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(tr_loader, desc="Batches[sel-CE]", leave=False):
            if len(batch) == 2:
                xb, yb = batch
                wb = None
            else:
                xb, yb, wb = batch

            xb = xb.to(device)
            yb = yb.to(device).long()
            if wb is not None:
                wb = wb.to(device).float()

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            per_ex = F.cross_entropy(logits, yb, weight=class_w, reduction="none")
            loss = weighted_mean(per_ex, wb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)

        train_loss = total_loss / max(1, len(tr_loader.dataset))

        model.eval()
        total_correct = 0.0
        total_acc_w = 0.0
        total_hard_u = 0.0
        total_u_w = 0.0

        with torch.no_grad():
            ptr = 0
            for batch in va_loader:
                if len(batch) == 2:
                    xb, yb = batch
                    wb = None
                else:
                    xb, yb, wb = batch

                bs = xb.size(0)
                ub = U_va[ptr:ptr + bs]
                ptr += bs

                xb = xb.to(device)
                yb = yb.to(device).long()
                ub = ub.to(device).float()

                logits = model(xb)
                pred = torch.argmax(logits, dim=1)
                hard_u = ub.gather(1, pred.unsqueeze(1)).squeeze(1)

                if wb is None:
                    total_correct += float((pred == yb).sum().item())
                    total_acc_w += float(bs)
                    total_hard_u += float(hard_u.sum().item())
                    total_u_w += float(bs)
                else:
                    wb = wb.to(device).float()
                    total_correct += float((((pred == yb).float()) * wb).sum().item())
                    total_acc_w += float(wb.sum().item())
                    total_hard_u += float((hard_u * wb).sum().item())
                    total_u_w += float(wb.sum().item())

        val_acc = total_correct / max(1e-8, total_acc_w)
        val_argmax_u = total_hard_u / max(1e-8, total_u_w)

        print(
            f"[sel-CE] epoch {ep:02d} | train_loss={train_loss:.4f} "
            f"| val_acc={val_acc:.4f} | val_argmax_u={val_argmax_u:.4f}"
        )

        if val_argmax_u > best_metric + min_delta:
            best_metric = val_argmax_u
            best_val_acc = val_acc
            best_val_argmax_u = val_argmax_u
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[sel-CE] early stop. best_val_argmax_u={best_metric:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    if U_va.numel() == 0:
        best_const_idx = 0
        best_const_val_u = -1e18
    else:
        if w_va is not None:
            denom = w_va.sum().clamp_min(1e-8)
            const_scores = [float(((U_va[:, j] * w_va).sum() / denom).item()) for j in range(U_va.size(1))]
        else:
            const_scores = [float(U_va[:, j].mean().item()) for j in range(U_va.size(1))]
        best_const_idx = int(np.argmax(const_scores))
        best_const_val_u = float(const_scores[best_const_idx])

    info = {
        "best_val_acc": float(best_val_acc),
        "best_val_argmax_u": float(best_val_argmax_u),
        "best_constant_val_u": float(best_const_val_u),
        "best_constant_expert_idx": int(best_const_idx),
        "val_size": int(len(va_idx)),
    }
    return model, info


def train_selector_expected_utility(
        model: nn.Module,
        X: torch.Tensor,
        U_targets: torch.Tensor,
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
) -> Tuple[nn.Module, Dict[str, Any]]:
    tr_idx, va_idx = split_train_val_indices(int(X.size(0)), 0.2, seed, min_val=min_val)
    X_tr, U_tr = X[tr_idx], U_targets[tr_idx]
    X_va, U_va = X[va_idx], U_targets[va_idx]

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
        tr_loader = DataLoader(TensorDatasetSoft(X_tr, U_tr, w_tr), batch_size=batch_size, sampler=sampler)
    else:
        tr_loader = DataLoader(TensorDatasetSoft(X_tr, U_tr, w_tr), batch_size=batch_size, shuffle=True)

    va_loader = DataLoader(TensorDatasetSoft(X_va, U_va, w_va), batch_size=batch_size, shuffle=False)

    best_metric = -1e18
    best_state = None
    bad = 0
    best_val_exp_u = -1e18
    best_val_argmax_u = -1e18

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(tr_loader, desc="Batches[sel-EU]", leave=False):
            if len(batch) == 2:
                xb, ub = batch
                wb = None
            else:
                xb, ub, wb = batch

            xb = xb.to(device)
            ub = ub.to(device).float()
            if wb is not None:
                wb = wb.to(device).float()

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            probs = F.softmax(logits, dim=1)

            if ub.size(1) <= 1:
                ub_norm = torch.zeros_like(ub)
            else:
                ub_centered = ub - ub.mean(dim=1, keepdim=True)
                ub_std = ub_centered.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
                ub_norm = ub_centered / ub_std

            exp_u_norm = (probs * ub_norm).sum(dim=1)
            loss = weighted_mean(-exp_u_norm, wb)

            if reg_weight > 0.0 and reg_type != "none":
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
        total_val_exp_u = 0.0
        total_val_argmax_u = 0.0
        total_weight = 0.0

        with torch.no_grad():
            for batch in va_loader:
                if len(batch) == 2:
                    xb, ub = batch
                    wb = None
                else:
                    xb, ub, wb = batch

                xb = xb.to(device)
                ub = ub.to(device).float()
                if wb is not None:
                    wb = wb.to(device).float()

                logits = model(xb)
                probs = F.softmax(logits, dim=1)

                exp_u = (probs * ub).sum(dim=1)
                hard_idx = torch.argmax(logits, dim=1)
                hard_u = ub.gather(1, hard_idx.unsqueeze(1)).squeeze(1)

                if wb is not None:
                    total_val_exp_u += float((exp_u * wb).sum().item())
                    total_val_argmax_u += float((hard_u * wb).sum().item())
                    total_weight += float(wb.sum().item())
                else:
                    total_val_exp_u += float(exp_u.sum().item())
                    total_val_argmax_u += float(hard_u.sum().item())
                    total_weight += float(xb.size(0))

        val_exp_u = total_val_exp_u / max(1e-8, total_weight)
        val_argmax_u = total_val_argmax_u / max(1e-8, total_weight)

        print(
            f"[sel-EU] epoch {ep:02d} | train_loss={train_loss:.4f} "
            f"| val_exp_u={val_exp_u:.4f} | val_argmax_u={val_argmax_u:.4f}"
        )

        if val_argmax_u > best_metric + min_delta:
            best_metric = val_argmax_u
            best_val_exp_u = val_exp_u
            best_val_argmax_u = val_argmax_u
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[sel-EU] early stop. best_val_argmax_u={best_metric:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    if U_va.numel() == 0:
        best_const_idx = 0
        best_const_val_u = -1e18
    else:
        if w_va is not None:
            denom = w_va.sum().clamp_min(1e-8)
            const_scores = [float(((U_va[:, j] * w_va).sum() / denom).item()) for j in range(U_va.size(1))]
        else:
            const_scores = [float(U_va[:, j].mean().item()) for j in range(U_va.size(1))]
        best_const_idx = int(np.argmax(const_scores))
        best_const_val_u = float(const_scores[best_const_idx])

    info = {
        "best_val_exp_u": float(best_val_exp_u),
        "best_val_argmax_u": float(best_val_argmax_u),
        "best_constant_val_u": float(best_const_val_u),
        "best_constant_expert_idx": int(best_const_idx),
        "val_size": int(len(va_idx)),
    }
    return model, info


def train_selector_delta_anchor(
        model: nn.Module,
        X: torch.Tensor,
        D_targets: torch.Tensor,
        U_targets: torch.Tensor,
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
        example_weights: Optional[torch.Tensor],
) -> Tuple[nn.Module, Dict[str, Any]]:
    tr_idx, va_idx = split_train_val_indices(int(X.size(0)), 0.2, seed, min_val=min_val)
    X_tr, D_tr = X[tr_idx], D_targets[tr_idx]
    X_va, D_va = X[va_idx], D_targets[va_idx]
    U_va = U_targets[va_idx]

    w_tr = None
    w_va = None
    if example_weights is not None and example_weights.numel() == X.size(0):
        w_tr = example_weights[tr_idx]
        w_va = example_weights[va_idx]

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    tr_loader = DataLoader(TensorDatasetSoft(X_tr, D_tr, w_tr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDatasetSoft(X_va, D_va, w_va), batch_size=batch_size, shuffle=False)

    best_metric = -1e18
    best_state = None
    bad = 0
    best_val_huber = 1e18
    best_val_argmax_u = -1e18

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(tr_loader, desc="Batches[sel-delta-anchor]", leave=False):
            if len(batch) == 2:
                xb, db = batch
                wb = None
            else:
                xb, db, wb = batch

            xb = xb.to(device)
            db = db.to(device).float()
            if wb is not None:
                wb = wb.to(device).float()

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            per_ex = F.huber_loss(pred, db, reduction="none", delta=0.1).mean(dim=1)
            loss = weighted_mean(per_ex, wb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)

        train_loss = total_loss / max(1, len(tr_loader.dataset))

        model.eval()
        total_vloss = 0.0
        denom = 0.0
        total_hard_u = 0.0
        total_u_w = 0.0

        with torch.no_grad():
            ptr = 0
            for batch in va_loader:
                if len(batch) == 2:
                    xb, db = batch
                    wb = None
                else:
                    xb, db, wb = batch

                bs = xb.size(0)
                ub = U_va[ptr:ptr + bs]
                ptr += bs

                xb = xb.to(device)
                db = db.to(device).float()
                ub = ub.to(device).float()
                if wb is not None:
                    wb = wb.to(device).float()

                pred = model(xb)
                per_ex = F.huber_loss(pred, db, reduction="none", delta=0.1).mean(dim=1)
                hard_idx = torch.argmax(pred, dim=1)
                hard_u = ub.gather(1, hard_idx.unsqueeze(1)).squeeze(1)

                if wb is None:
                    total_vloss += float(per_ex.sum().item())
                    denom += float(per_ex.numel())
                    total_hard_u += float(hard_u.sum().item())
                    total_u_w += float(bs)
                else:
                    total_vloss += float((per_ex * wb).sum().item())
                    denom += float(wb.sum().item())
                    total_hard_u += float((hard_u * wb).sum().item())
                    total_u_w += float(wb.sum().item())

        val_huber = total_vloss / max(1e-8, denom)
        val_argmax_u = total_hard_u / max(1e-8, total_u_w)

        print(
            f"[sel-delta-anchor] epoch {ep:02d} | train_loss={train_loss:.4f} "
            f"| val_huber={val_huber:.4f} | val_argmax_u={val_argmax_u:.4f}"
        )

        if val_argmax_u > best_metric + min_delta:
            best_metric = val_argmax_u
            best_val_huber = val_huber
            best_val_argmax_u = val_argmax_u
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[sel-delta-anchor] early stop. best_val_argmax_u={best_metric:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    if U_va.numel() == 0:
        best_const_idx = 0
        best_const_val_u = -1e18
    else:
        if w_va is not None:
            denom = w_va.sum().clamp_min(1e-8)
            const_scores = [float(((U_va[:, j] * w_va).sum() / denom).item()) for j in range(U_va.size(1))]
        else:
            const_scores = [float(U_va[:, j].mean().item()) for j in range(U_va.size(1))]
        best_const_idx = int(np.argmax(const_scores))
        best_const_val_u = float(const_scores[best_const_idx])

    info = {
        "best_val_huber": float(best_val_huber),
        "best_val_argmax_u": float(best_val_argmax_u),
        "best_constant_val_u": float(best_const_val_u),
        "best_constant_expert_idx": int(best_const_idx),
        "val_size": int(len(va_idx)),
    }
    return model, info


# --------------------------
# Trivial selector saver
# --------------------------

def save_trivial_selector_ckpt(
        *,
        path: Path,
        in_dim: int,
        hidden_dim: int,
        dropout: float,
        experts: List[str],
        embed_model: str,
        feature_keys: List[str],
        feature_stats: Optional[Dict[str, Any]],
        selector_objective: str,
        tradeoff_mode: bool,
        gate_delta: float,
        sel_margin_min: float,
        sel_margin_max: float,
        sel_use_margin_weighting: bool,
        sel_margin_scale: float,
        sel_weight_min: float,
        sel_weight_max: float,
        reg_type: str,
        reg_weight: float,
        prior_mode: str,
        used_passage_embeddings: bool,
        passage_source_expert: str,
        passage_max_docs: int,
        passage_max_chars: int,
        constant_expert: str,
        avg_utility_by_expert: Dict[str, float],
        selector_constant_margin: float,
):
    model = MLP(in_dim, hidden_dim, dropout, out_dim=len(experts))
    ckpt = {
        "state_dict": model.state_dict(),
        "in_dim": in_dim,
        "experts": experts,
        "embed_model": embed_model,
        "feature_keys": feature_keys,
        "sel_tau": None,
        "selector_objective": selector_objective,
        "trained_on_gate_delta": float(gate_delta),
        "trained_on_sel_margin_min": float(sel_margin_min),
        "trained_on_sel_margin_max": float(sel_margin_max),
        "sel_use_margin_weighting": bool(sel_use_margin_weighting),
        "sel_margin_scale": float(sel_margin_scale),
        "sel_weight_min": float(sel_weight_min),
        "sel_weight_max": float(sel_weight_max),
        "reg_type": str(reg_type),
        "reg_weight": float(reg_weight),
        "prior_mode": str(prior_mode),
        "prior": None,
        "tradeoff_mode": bool(tradeoff_mode),
        "best_val_score": None,
        "fallback_best_constant_expert": str(constant_expert),
        "val_best_constant_expert": str(constant_expert),
        "use_constant_selector": True,
        "selector_constant_margin": float(selector_constant_margin),
        "best_constant_val_u": float(avg_utility_by_expert.get(constant_expert, 0.0)),
        "best_model_val_argmax_u": None,
        "best_model_val_exp_u": None,
        "full_model_argmax_u": float(avg_utility_by_expert.get(constant_expert, 0.0)),
        "full_constant_u": float(avg_utility_by_expert.get(constant_expert, 0.0)),
        "avg_utility_by_expert": avg_utility_by_expert,
        "used_passage_embeddings": bool(used_passage_embeddings),
        "passage_source_expert": str(passage_source_expert),
        "passage_max_docs": int(passage_max_docs),
        "passage_max_chars": int(passage_max_chars),
    }
    if feature_stats is not None:
        ckpt["feature_stats"] = feature_stats
    torch.save(ckpt, path)


# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None,
                    help="Train only one dataset (hotpotqa|squad_v2|pubmedqa_v2|commonsenseqa|combined_pubmed_csqa_gate)")
    ap.add_argument("--out_dir", type=str, default="results/two_stage_utility")
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

    ap.add_argument("--gate_use_delta_weighting", action="store_true")
    ap.add_argument("--gate_weight_power", type=float, default=1.0)
    ap.add_argument("--gate_weight_min", type=float, default=0.25)
    ap.add_argument("--gate_weight_max", type=float, default=4.0)

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

    ap.add_argument("--pubmed_policy", type=str, default="none", choices=["forced", "none"])
    ap.add_argument("--use_passage_embeddings", action="store_true")
    ap.add_argument("--passage_source_expert", type=str, default="base_rag")
    ap.add_argument("--passage_max_docs", type=int, default=5)
    ap.add_argument("--passage_max_chars", type=int, default=1200)

    ap.add_argument("--selector_objective", type=str, default="expected_u",
                    choices=["soft_ce", "expected_u", "hard_ce", "delta_anchor"])
    ap.add_argument("--selector_constant_margin", type=float, default=0.002)

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

    gate_feature_keys, selector_feature_keys = split_feature_keys_for_models(feature_keys)

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

        Xf_gate = build_feature_matrix(rows, fmap, gate_feature_keys) if gate_feature_keys else torch.zeros((total, 0), dtype=torch.float32)
        gate_feat_stats = None
        if args.standardize_features and Xf_gate.size(1) > 0:
            Xf_gate, gate_feat_stats = standardize_features(Xf_gate)

        X_gate_all = Xq if Xf_gate.size(1) == 0 else torch.cat([Xq, Xf_gate], dim=1)

        if args.use_passage_embeddings:
            Xp = build_passage_embedding_matrix(
                rows,
                embedder,
                source_expert=args.passage_source_expert,
                max_docs=args.passage_max_docs,
                max_chars=args.passage_max_chars,
                batch_size_docs=args.batch_size,
            )
            X_gate_all = torch.cat([X_gate_all, Xp], dim=1)

        gate_in_dim = int(X_gate_all.size(1))
        model_dir = out_root / SPECIAL_ONLY
        model_dir.mkdir(parents=True, exist_ok=True)
        gate_path = model_dir / "gate.pt"

        print(f"\n=== {SPECIAL_ONLY} === total={total} (pubmed={len(rows_pub)} csqa={len(rows_cs)}) embed={embed_model}")

        gate_model = MLP(gate_in_dim, args.hidden_dim, args.dropout, out_dim=2)
        gate_model, best_acc = train_gate_classifier(
            gate_model,
            X_gate_all,
            y,
            device=args.device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            seed=args.seed,
            min_val=args.min_val_selector,
            example_weights=None,
        )

        ckpt = {
            "state_dict": gate_model.state_dict(),
            "in_dim": gate_in_dim,
            "embed_model": embed_model,
            "feature_keys": gate_feature_keys,
            "gate_objective": "cls",
            "calibrated_threshold": 0.5,
            "best_val_acc": float(best_acc),
            "trained_on": {"pubmedqa_v2": len(rows_pub), "commonsenseqa": len(rows_cs)},
            "used_passage_embeddings": bool(args.use_passage_embeddings),
            "passage_source_expert": str(args.passage_source_expert),
            "passage_max_docs": int(args.passage_max_docs),
            "passage_max_chars": int(args.passage_max_chars),
        }
        if args.standardize_features and gate_feat_stats is not None:
            ckpt["feature_stats"] = gate_feat_stats

        torch.save(ckpt, gate_path)
        print(f"[{SPECIAL_ONLY}] saved gate(cls) -> {gate_path} (best_val_acc={best_acc:.4f})")
        print(f"\nDONE. Models saved under: {out_root}")
        return

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

        Xf_gate = build_feature_matrix(rows, fmap, gate_feature_keys) if gate_feature_keys else torch.zeros((total, 0), dtype=torch.float32)
        Xf_sel = build_feature_matrix(rows, fmap, selector_feature_keys) if selector_feature_keys else torch.zeros((total, 0), dtype=torch.float32)
        report_feature_coverage(f"{dataset}/gate_raw", Xf_gate)
        report_feature_coverage(f"{dataset}/selector_raw", Xf_sel)

        gate_feat_stats = None
        sel_feat_stats = None
        if args.standardize_features and Xf_gate.size(1) > 0:
            Xf_gate, gate_feat_stats = standardize_features(Xf_gate)
        if args.standardize_features and Xf_sel.size(1) > 0:
            Xf_sel, sel_feat_stats = standardize_features(Xf_sel)

        X_gate_all = Xq if Xf_gate.size(1) == 0 else torch.cat([Xq, Xf_gate], dim=1)
        X_sel_all = Xq if Xf_sel.size(1) == 0 else torch.cat([Xq, Xf_sel], dim=1)

        if args.use_passage_embeddings:
            Xp = build_passage_embedding_matrix(
                rows,
                embedder,
                source_expert=args.passage_source_expert,
                max_docs=args.passage_max_docs,
                max_chars=args.passage_max_chars,
                batch_size_docs=args.batch_size,
            )
            X_gate_all = torch.cat([X_gate_all, Xp], dim=1)
            X_sel_all = torch.cat([X_sel_all, Xp], dim=1)

        gate_in_dim = int(X_gate_all.size(1))
        sel_in_dim = int(X_sel_all.size(1))

        pol = policy_for_dataset(dataset, pubmed_policy_mode=args.pubmed_policy)
        rag_pool_ds, no_pool_ds = pools_for_dataset(dataset)

        print(f"\n=== {dataset} === total={total} policy={pol} embed={embed_model}")
        if args.tradeoff_mode:
            print(f"[{dataset}] tradeoff_mode=True (utility-based, matches eval)")
        if gate_feature_keys or selector_feature_keys:
            print(
                f"[{dataset}] features: gate_k={len(gate_feature_keys)} sel_k={len(selector_feature_keys)} "
                f"| base_dim={in_dim_q} pass={'on' if args.use_passage_embeddings else 'off'} "
                f"=> gate_in_dim={gate_in_dim} sel_in_dim={sel_in_dim}"
            )
        if dataset == "pubmedqa_v2":
            print(f"[{dataset}] pubmed_policy={args.pubmed_policy}")
        print(f"[{dataset}] pools | rag={rag_pool_ds} no={no_pool_ds}")

        model_dir = out_root / dataset
        model_dir.mkdir(parents=True, exist_ok=True)
        gate_path = model_dir / "gate.pt"

        if pol is None:
            deltas = build_gate_delta_targets(cfg, dataset, rows, use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg)

            if args.print_target_stats:
                print_gate_target_diagnostics(dataset, deltas, float(args.gate_delta))

            gate_example_weights = None
            if args.gate_use_delta_weighting:
                gate_example_weights = build_gate_example_weights(
                    deltas,
                    power=float(args.gate_weight_power),
                    weight_min=float(args.gate_weight_min),
                    weight_max=float(args.gate_weight_max),
                )

            if args.gate_objective == "cls":
                idx_gate, y_gate = build_gate_deadzone_from_deltas(deltas, gate_delta=args.gate_delta)
                if len(idx_gate) < 200:
                    print(f"[{dataset}] gate-cls: too few examples after deadzone (kept={len(idx_gate)}). Lower gate_delta.")
                X_gate = X_gate_all[idx_gate] if len(idx_gate) > 0 else X_gate_all[:0]
                gate_w_sel = gate_example_weights[idx_gate] if (gate_example_weights is not None and len(idx_gate) > 0) else None

                gate_model = MLP(gate_in_dim, args.hidden_dim, args.dropout, out_dim=2)
                if y_gate.numel() > 0:
                    gate_model, best_acc = train_gate_classifier(
                        gate_model,
                        X_gate,
                        y_gate,
                        device=args.device,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        patience=args.patience,
                        min_delta=args.min_delta,
                        seed=args.seed,
                        min_val=args.min_val_selector,
                        example_weights=gate_w_sel,
                    )
                else:
                    best_acc = 0.0
                    gate_model.eval()

                ckpt = {
                    "state_dict": gate_model.state_dict(),
                    "in_dim": gate_in_dim,
                    "embed_model": embed_model,
                    "feature_keys": gate_feature_keys,
                    "gate_objective": "cls",
                    "gate_delta": float(args.gate_delta),
                    "best_val_acc": float(best_acc),
                    "tradeoff_mode": bool(args.tradeoff_mode),
                    "used_passage_embeddings": bool(args.use_passage_embeddings),
                    "passage_source_expert": str(args.passage_source_expert),
                    "passage_max_docs": int(args.passage_max_docs),
                    "passage_max_chars": int(args.passage_max_chars),
                    "gate_use_delta_weighting": bool(args.gate_use_delta_weighting),
                    "gate_weight_power": float(args.gate_weight_power),
                    "gate_weight_min": float(args.gate_weight_min),
                    "gate_weight_max": float(args.gate_weight_max),
                }
                if args.standardize_features and gate_feat_stats is not None:
                    ckpt["feature_stats"] = gate_feat_stats

                torch.save(ckpt, gate_path)
                print(f"[{dataset}] saved gate(cls) -> {gate_path} (best_val_acc={best_acc:.4f})")

            else:
                gate_model = MLP(gate_in_dim, args.hidden_dim, args.dropout, out_dim=1)
                gate_model, info = train_gate_delta_regressor(
                    gate_model,
                    X_gate_all,
                    deltas,
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
                    example_weights=gate_example_weights,
                )

                ckpt = {
                    "state_dict": gate_model.state_dict(),
                    "in_dim": gate_in_dim,
                    "embed_model": embed_model,
                    "feature_keys": gate_feature_keys,
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
                    "gate_use_delta_weighting": bool(args.gate_use_delta_weighting),
                    "gate_weight_power": float(args.gate_weight_power),
                    "gate_weight_min": float(args.gate_weight_min),
                    "gate_weight_max": float(args.gate_weight_max),
                }
                if args.standardize_features and gate_feat_stats is not None:
                    ckpt["feature_stats"] = gate_feat_stats

                torch.save(ckpt, gate_path)
                print(
                    f"[{dataset}] saved gate(delta_reg) -> {gate_path} "
                    f"(thr={info['best_threshold']:.4f} val_sign_acc={info['best_val_sign_acc']:.4f})"
                )

        else:
            ckpt = {
                "forced_policy": bool(pol),
                "in_dim": gate_in_dim,
                "embed_model": embed_model,
                "feature_keys": gate_feature_keys,
                "tradeoff_mode": bool(args.tradeoff_mode),
                "used_passage_embeddings": bool(args.use_passage_embeddings),
                "passage_source_expert": str(args.passage_source_expert),
                "passage_max_docs": int(args.passage_max_docs),
                "passage_max_chars": int(args.passage_max_chars),
            }
            if args.standardize_features and gate_feat_stats is not None:
                ckpt["feature_stats"] = gate_feat_stats
            torch.save(ckpt, gate_path)
            print(f"[{dataset}] gate forced={pol}; wrote marker -> {gate_path}")

        sel_rag_path = model_dir / "selector_rag.pt"
        sel_no_path = model_dir / "selector_no_rag.pt"

        idx_rag: List[int] = []
        idx_no: List[int] = []

        if pol is True:
            idx_rag = list(range(total))
        elif pol is False:
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

        print(
            f"[{dataset}] selector train counts | rag={len(idx_rag)}→{len(idx_rag_f)} "
            f"no={len(idx_no)}→{len(idx_no_f)} "
            f"(gate_delta={args.gate_delta}, sel_margin=[{args.sel_margin_min},{args.sel_margin_max}])"
        )

        if args.print_target_stats:
            if len(idx_rag) > 0:
                U_diag_rag = build_selector_utility_matrix(
                    cfg, dataset, rows, idx_rag, rag_pool_ds,
                    use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg
                )
                print_selector_domain_diagnostics(
                    dataset, "selector_rag_full", U_diag_rag, rag_pool_ds, Xf_sel[idx_rag], selector_feature_keys
                )

            if pol is None and len(idx_no) > 0:
                U_diag_no = build_selector_utility_matrix(
                    cfg, dataset, rows, idx_no, no_pool_ds,
                    use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg
                )
                print_selector_domain_diagnostics(
                    dataset, "selector_no_rag_full", U_diag_no, no_pool_ds, Xf_sel[idx_no], selector_feature_keys
                )

        def make_selector_prior(y_hard: torch.Tensor, num_classes: int) -> torch.Tensor:
            if args.sel_prior_mode in ("balanced", "uniform"):
                return torch.full((num_classes,), 1.0 / float(num_classes), dtype=torch.float32)
            counts = torch.bincount(y_hard, minlength=num_classes).float()
            if float(counts.sum().item()) <= 0:
                return torch.full((num_classes,), 1.0 / float(num_classes), dtype=torch.float32)
            return (counts / counts.sum()).clamp_min(1e-8)

        # -----------------
        # RAG selector
        # -----------------
        if len(rag_pool_ds) <= 1:
            const_expert, avg_utility_by_expert = compute_constant_fallback_expert(
                cfg, dataset, rows, idx_rag, rag_pool_ds,
                use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg
            )
            print(f"[{dataset}] selector_rag is trivial (only one expert: {rag_pool_ds}); saving constant selector.")
            save_trivial_selector_ckpt(
                path=sel_rag_path,
                in_dim=sel_in_dim,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                experts=rag_pool_ds,
                embed_model=embed_model,
                feature_keys=selector_feature_keys,
                feature_stats=sel_feat_stats if args.standardize_features else None,
                selector_objective="constant",
                tradeoff_mode=bool(args.tradeoff_mode),
                gate_delta=float(args.gate_delta),
                sel_margin_min=float(args.sel_margin_min),
                sel_margin_max=float(args.sel_margin_max),
                sel_use_margin_weighting=bool(args.sel_use_margin_weighting),
                sel_margin_scale=float(args.sel_margin_scale),
                sel_weight_min=float(args.sel_weight_min),
                sel_weight_max=float(args.sel_weight_max),
                reg_type=str(args.sel_reg_type),
                reg_weight=float(args.sel_reg_weight),
                prior_mode=str(args.sel_prior_mode),
                used_passage_embeddings=bool(args.use_passage_embeddings),
                passage_source_expert=str(args.passage_source_expert),
                passage_max_docs=int(args.passage_max_docs),
                passage_max_chars=int(args.passage_max_chars),
                constant_expert=const_expert,
                avg_utility_by_expert=avg_utility_by_expert,
                selector_constant_margin=float(args.selector_constant_margin),
            )
            print(f"[{dataset}] saved trivial selector_rag -> {sel_rag_path}")

        elif should_train_selector(len(idx_rag_f), min_train=args.min_train_selector, min_val=args.min_val_selector, name="selector_rag", ds=dataset):
            Xg = X_sel_all[idx_rag_f]
            U_targets = build_selector_utility_matrix(
                cfg, dataset, rows, idx_rag_f, rag_pool_ds,
                use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg
            )
            y_hard = U_targets.argmax(dim=1).long()

            if args.print_target_stats and y_hard.numel() > 0:
                cnt = Counter(y_hard.tolist())
                dist = [(rag_pool_ds[k], v) for k, v in sorted(cnt.items(), key=lambda x: -x[1])]
                print(f"[{dataset}] selector_rag targets argmax: {dist} (total={y_hard.numel()})")
                top_frac = max(cnt.values()) / float(y_hard.numel())
                if top_frac >= 0.80:
                    print(f"[WARN][{dataset}] selector_rag filtered target distribution is highly skewed (top_frac={top_frac:.3f}).")

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
            sel_model = MLP(sel_in_dim, args.hidden_dim, dropout_used, out_dim=len(rag_pool_ds))

            fallback_best_constant_expert, avg_utility_by_expert = compute_constant_fallback_expert(
                cfg, dataset, rows, idx_rag, rag_pool_ds,
                use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg
            )

            use_constant_selector = False
            best_constant_val_u = None
            best_model_val_argmax_u = None
            best_model_val_exp_u = None
            val_best_constant_expert = fallback_best_constant_expert
            anchor_expert = None

            if args.selector_objective == "expected_u":
                sel_model, sel_info = train_selector_expected_utility(
                    sel_model, Xg, U_targets,
                    device=args.device, lr=args.lr, weight_decay=args.weight_decay,
                    batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
                    min_delta=args.min_delta, seed=args.seed, min_val=args.min_val_selector,
                    balanced_sampler=bool(args.selector_balanced_sampler),
                    reg_type=str(args.sel_reg_type), reg_weight=float(args.sel_reg_weight),
                    prior=prior, hard_labels_for_sampler=y_hard, example_weights=ex_w,
                )
                best_score = float(sel_info["best_val_argmax_u"])
                best_constant_val_u = float(sel_info["best_constant_val_u"])
                best_model_val_argmax_u = float(sel_info["best_val_argmax_u"])
                best_model_val_exp_u = float(sel_info["best_val_exp_u"])
                val_best_constant_expert = rag_pool_ds[int(sel_info["best_constant_expert_idx"])]

            elif args.selector_objective == "hard_ce":
                sel_model, sel_info = train_selector_hard_ce(
                    sel_model, Xg, y_hard, U_targets,
                    device=args.device, lr=args.lr, weight_decay=args.weight_decay,
                    batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
                    min_delta=args.min_delta, seed=args.seed, min_val=args.min_val_selector,
                    balanced_sampler=bool(args.selector_balanced_sampler),
                    example_weights=ex_w,
                )
                best_score = float(sel_info["best_val_argmax_u"])
                best_constant_val_u = float(sel_info["best_constant_val_u"])
                best_model_val_argmax_u = float(sel_info["best_val_argmax_u"])
                best_model_val_exp_u = None
                val_best_constant_expert = rag_pool_ds[int(sel_info["best_constant_expert_idx"])]

            elif args.selector_objective == "delta_anchor":
                anchor_expert = fallback_best_constant_expert
                D_targets = build_selector_delta_anchor_matrix(
                    cfg, dataset, rows, idx_rag_f, rag_pool_ds, anchor_expert,
                    use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg
                )

                if args.print_target_stats:
                    print(f"[{dataset}] selector_rag delta-anchor expert: {anchor_expert}")

                sel_model, sel_info = train_selector_delta_anchor(
                    sel_model, Xg, D_targets, U_targets,
                    device=args.device, lr=args.lr, weight_decay=args.weight_decay,
                    batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
                    min_delta=args.min_delta, seed=args.seed, min_val=args.min_val_selector,
                    example_weights=ex_w,
                )
                best_score = float(sel_info["best_val_argmax_u"])
                best_constant_val_u = float(sel_info["best_constant_val_u"])
                best_model_val_argmax_u = float(sel_info["best_val_argmax_u"])
                best_model_val_exp_u = None
                val_best_constant_expert = rag_pool_ds[int(sel_info["best_constant_expert_idx"])]

            else:
                y_soft = build_selector_soft_targets(
                    cfg, dataset, rows, idx_rag_f, rag_pool_ds,
                    tau=args.sel_tau, use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg
                )
                sel_model, sel_info = train_selector_soft(
                    sel_model, Xg, y_soft, U_targets,
                    device=args.device, lr=args.lr, weight_decay=args.weight_decay,
                    batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
                    min_delta=args.min_delta, seed=args.seed, min_val=args.min_val_selector,
                    balanced_sampler=bool(args.selector_balanced_sampler),
                    reg_type=str(args.sel_reg_type), reg_weight=float(args.sel_reg_weight),
                    prior=prior, hard_labels_for_sampler=y_hard, example_weights=ex_w,
                )
                best_score = float(sel_info["best_val_argmax_u"])
                best_constant_val_u = float(sel_info["best_constant_val_u"])
                best_model_val_argmax_u = float(sel_info["best_val_argmax_u"])
                best_model_val_exp_u = None
                val_best_constant_expert = rag_pool_ds[int(sel_info["best_constant_expert_idx"])]

            X_full = X_sel_all[idx_rag]
            U_full = build_selector_utility_matrix(
                cfg, dataset, rows, idx_rag, rag_pool_ds,
                use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg
            )
            full_model_argmax_u = selector_hard_utility_score(
                sel_model, X_full, U_full,
                device=args.device, batch_size=max(256, int(args.batch_size))
            )
            full_constant_u = float(avg_utility_by_expert.get(fallback_best_constant_expert, 0.0))
            use_constant_selector = bool(full_constant_u >= (full_model_argmax_u - float(args.selector_constant_margin)))

            if args.print_target_stats:
                val_msg = ""
                if best_model_val_argmax_u is not None and best_constant_val_u is not None:
                    val_msg = (
                        f" | val_model_argmax_u={best_model_val_argmax_u:.4f}"
                        f" | val_constant_u={best_constant_val_u:.4f}"
                        f" | val_const={val_best_constant_expert}"
                    )
                print(
                    f"[{dataset}] selector_rag deploy compare | full_model_argmax_u={full_model_argmax_u:.4f} "
                    f"| full_constant_u={full_constant_u:.4f} | use_constant={use_constant_selector} "
                    f"| full_const={fallback_best_constant_expert}{val_msg}"
                )

            ckpt = {
                "state_dict": sel_model.state_dict(),
                "in_dim": sel_in_dim,
                "experts": rag_pool_ds,
                "embed_model": embed_model,
                "feature_keys": selector_feature_keys,
                "sel_tau": float(args.sel_tau),
                "selector_objective": str(args.selector_objective),
                "anchor_expert": str(anchor_expert) if anchor_expert is not None else None,
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
                "fallback_best_constant_expert": str(fallback_best_constant_expert),
                "val_best_constant_expert": str(val_best_constant_expert),
                "use_constant_selector": bool(use_constant_selector),
                "selector_constant_margin": float(args.selector_constant_margin),
                "best_constant_val_u": best_constant_val_u,
                "best_model_val_argmax_u": best_model_val_argmax_u,
                "best_model_val_exp_u": best_model_val_exp_u,
                "full_model_argmax_u": float(full_model_argmax_u),
                "full_constant_u": float(full_constant_u),
                "avg_utility_by_expert": avg_utility_by_expert,
                "used_passage_embeddings": bool(args.use_passage_embeddings),
                "passage_source_expert": str(args.passage_source_expert),
                "passage_max_docs": int(args.passage_max_docs),
                "passage_max_chars": int(args.passage_max_chars),
            }
            if args.standardize_features and sel_feat_stats is not None:
                ckpt["feature_stats"] = sel_feat_stats

            torch.save(ckpt, sel_rag_path)
            print(f"[{dataset}] saved selector_rag -> {sel_rag_path} (best_val_score={best_score:.4f})")
        else:
            print(f"[{dataset}] skip selector_rag (count={len(idx_rag_f)})")

        # -----------------
        # NO-RAG selector
        # -----------------
        if pol is None:
            if len(no_pool_ds) <= 1:
                const_expert, avg_utility_by_expert = compute_constant_fallback_expert(
                    cfg, dataset, rows, idx_no, no_pool_ds,
                    use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg
                )
                print(f"[{dataset}] selector_no_rag is trivial (only one expert: {no_pool_ds}); saving constant selector.")
                save_trivial_selector_ckpt(
                    path=sel_no_path,
                    in_dim=sel_in_dim,
                    hidden_dim=args.hidden_dim,
                    dropout=args.dropout,
                    experts=no_pool_ds,
                    embed_model=embed_model,
                    feature_keys=selector_feature_keys,
                    feature_stats=sel_feat_stats if args.standardize_features else None,
                    selector_objective="constant",
                    tradeoff_mode=bool(args.tradeoff_mode),
                    gate_delta=float(args.gate_delta),
                    sel_margin_min=float(args.sel_margin_min),
                    sel_margin_max=float(args.sel_margin_max),
                    sel_use_margin_weighting=bool(args.sel_use_margin_weighting),
                    sel_margin_scale=float(args.sel_margin_scale),
                    sel_weight_min=float(args.sel_weight_min),
                    sel_weight_max=float(args.sel_weight_max),
                    reg_type=str(args.sel_reg_type),
                    reg_weight=float(args.sel_reg_weight),
                    prior_mode=str(args.sel_prior_mode),
                    used_passage_embeddings=bool(args.use_passage_embeddings),
                    passage_source_expert=str(args.passage_source_expert),
                    passage_max_docs=int(args.passage_max_docs),
                    passage_max_chars=int(args.passage_max_chars),
                    constant_expert=const_expert,
                    avg_utility_by_expert=avg_utility_by_expert,
                    selector_constant_margin=float(args.selector_constant_margin),
                )
                print(f"[{dataset}] saved trivial selector_no_rag -> {sel_no_path}")

            elif should_train_selector(len(idx_no_f), min_train=args.min_train_selector, min_val=args.min_val_selector, name="selector_no_rag", ds=dataset):
                Xg = X_sel_all[idx_no_f]
                U_targets = build_selector_utility_matrix(
                    cfg, dataset, rows, idx_no_f, no_pool_ds,
                    use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg
                )
                y_hard = U_targets.argmax(dim=1).long()

                if args.print_target_stats and y_hard.numel() > 0:
                    cnt = Counter(y_hard.tolist())
                    dist = [(no_pool_ds[k], v) for k, v in sorted(cnt.items(), key=lambda x: -x[1])]
                    print(f"[{dataset}] selector_no_rag targets argmax: {dist} (total={y_hard.numel()})")
                    top_frac = max(cnt.values()) / float(y_hard.numel())
                    if top_frac >= 0.80:
                        print(f"[WARN][{dataset}] selector_no_rag filtered target distribution is highly skewed (top_frac={top_frac:.3f}).")

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
                sel_model = MLP(sel_in_dim, args.hidden_dim, dropout_used, out_dim=len(no_pool_ds))

                fallback_best_constant_expert, avg_utility_by_expert = compute_constant_fallback_expert(
                    cfg, dataset, rows, idx_no, no_pool_ds,
                    use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg
                )

                use_constant_selector = False
                best_constant_val_u = None
                best_model_val_argmax_u = None
                best_model_val_exp_u = None
                val_best_constant_expert = fallback_best_constant_expert
                anchor_expert = None

                if args.selector_objective == "expected_u":
                    sel_model, sel_info = train_selector_expected_utility(
                        sel_model, Xg, U_targets,
                        device=args.device, lr=args.lr, weight_decay=args.weight_decay,
                        batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
                        min_delta=args.min_delta, seed=args.seed, min_val=args.min_val_selector,
                        balanced_sampler=bool(args.selector_balanced_sampler),
                        reg_type=str(args.sel_reg_type), reg_weight=float(args.sel_reg_weight),
                        prior=prior, hard_labels_for_sampler=y_hard, example_weights=ex_w,
                    )
                    best_score = float(sel_info["best_val_argmax_u"])
                    best_constant_val_u = float(sel_info["best_constant_val_u"])
                    best_model_val_argmax_u = float(sel_info["best_val_argmax_u"])
                    best_model_val_exp_u = float(sel_info["best_val_exp_u"])
                    val_best_constant_expert = no_pool_ds[int(sel_info["best_constant_expert_idx"])]

                elif args.selector_objective == "hard_ce":
                    sel_model, sel_info = train_selector_hard_ce(
                        sel_model, Xg, y_hard, U_targets,
                        device=args.device, lr=args.lr, weight_decay=args.weight_decay,
                        batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
                        min_delta=args.min_delta, seed=args.seed, min_val=args.min_val_selector,
                        balanced_sampler=bool(args.selector_balanced_sampler),
                        example_weights=ex_w,
                    )
                    best_score = float(sel_info["best_val_argmax_u"])
                    best_constant_val_u = float(sel_info["best_constant_val_u"])
                    best_model_val_argmax_u = float(sel_info["best_val_argmax_u"])
                    best_model_val_exp_u = None
                    val_best_constant_expert = no_pool_ds[int(sel_info["best_constant_expert_idx"])]

                elif args.selector_objective == "delta_anchor":
                    anchor_expert = fallback_best_constant_expert
                    D_targets = build_selector_delta_anchor_matrix(
                        cfg, dataset, rows, idx_no_f, no_pool_ds, anchor_expert,
                        use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg
                    )

                    if args.print_target_stats:
                        print(f"[{dataset}] selector_no_rag delta-anchor expert: {anchor_expert}")

                    sel_model, sel_info = train_selector_delta_anchor(
                        sel_model, Xg, D_targets, U_targets,
                        device=args.device, lr=args.lr, weight_decay=args.weight_decay,
                        batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
                        min_delta=args.min_delta, seed=args.seed, min_val=args.min_val_selector,
                        example_weights=ex_w,
                    )
                    best_score = float(sel_info["best_val_argmax_u"])
                    best_constant_val_u = float(sel_info["best_constant_val_u"])
                    best_model_val_argmax_u = float(sel_info["best_val_argmax_u"])
                    best_model_val_exp_u = None
                    val_best_constant_expert = no_pool_ds[int(sel_info["best_constant_expert_idx"])]

                else:
                    y_soft = build_selector_soft_targets(
                        cfg, dataset, rows, idx_no_f, no_pool_ds,
                        tau=args.sel_tau, use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg
                    )
                    sel_model, sel_info = train_selector_soft(
                        sel_model, Xg, y_soft, U_targets,
                        device=args.device, lr=args.lr, weight_decay=args.weight_decay,
                        batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
                        min_delta=args.min_delta, seed=args.seed, min_val=args.min_val_selector,
                        balanced_sampler=bool(args.selector_balanced_sampler),
                        reg_type=str(args.sel_reg_type), reg_weight=float(args.sel_reg_weight),
                        prior=prior, hard_labels_for_sampler=y_hard, example_weights=ex_w,
                    )
                    best_score = float(sel_info["best_val_argmax_u"])
                    best_constant_val_u = float(sel_info["best_constant_val_u"])
                    best_model_val_argmax_u = float(sel_info["best_val_argmax_u"])
                    best_model_val_exp_u = None
                    val_best_constant_expert = no_pool_ds[int(sel_info["best_constant_expert_idx"])]

                X_full = X_sel_all[idx_no]
                U_full = build_selector_utility_matrix(
                    cfg, dataset, rows, idx_no, no_pool_ds,
                    use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg
                )
                full_model_argmax_u = selector_hard_utility_score(
                    sel_model, X_full, U_full,
                    device=args.device, batch_size=max(256, int(args.batch_size))
                )
                full_constant_u = float(avg_utility_by_expert.get(fallback_best_constant_expert, 0.0))
                use_constant_selector = bool(full_constant_u >= (full_model_argmax_u - float(args.selector_constant_margin)))

                if args.print_target_stats:
                    val_msg = ""
                    if best_model_val_argmax_u is not None and best_constant_val_u is not None:
                        val_msg = (
                            f" | val_model_argmax_u={best_model_val_argmax_u:.4f}"
                            f" | val_constant_u={best_constant_val_u:.4f}"
                            f" | val_const={val_best_constant_expert}"
                        )
                    print(
                        f"[{dataset}] selector_no_rag deploy compare | full_model_argmax_u={full_model_argmax_u:.4f} "
                        f"| full_constant_u={full_constant_u:.4f} | use_constant={use_constant_selector} "
                        f"| full_const={fallback_best_constant_expert}{val_msg}"
                    )

                ckpt = {
                    "state_dict": sel_model.state_dict(),
                    "in_dim": sel_in_dim,
                    "experts": no_pool_ds,
                    "embed_model": embed_model,
                    "feature_keys": selector_feature_keys,
                    "sel_tau": float(args.sel_tau),
                    "selector_objective": str(args.selector_objective),
                    "anchor_expert": str(anchor_expert) if anchor_expert is not None else None,
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
                    "fallback_best_constant_expert": str(fallback_best_constant_expert),
                    "val_best_constant_expert": str(val_best_constant_expert),
                    "use_constant_selector": bool(use_constant_selector),
                    "selector_constant_margin": float(args.selector_constant_margin),
                    "best_constant_val_u": best_constant_val_u,
                    "best_model_val_argmax_u": best_model_val_argmax_u,
                    "best_model_val_exp_u": best_model_val_exp_u,
                    "full_model_argmax_u": float(full_model_argmax_u),
                    "full_constant_u": float(full_constant_u),
                    "avg_utility_by_expert": avg_utility_by_expert,
                    "used_passage_embeddings": bool(args.use_passage_embeddings),
                    "passage_source_expert": str(args.passage_source_expert),
                    "passage_max_docs": int(args.passage_max_docs),
                    "passage_max_chars": int(args.passage_max_chars),
                }
                if args.standardize_features and sel_feat_stats is not None:
                    ckpt["feature_stats"] = sel_feat_stats

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