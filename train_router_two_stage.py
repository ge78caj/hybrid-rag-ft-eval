# train_router_two_stage.py
import argparse
import json
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

RAG_EXPERTS = ["base_rag", "sft_rag", "raft_rag"]
NO_EXPERTS  = ["base_only", "sft_only"]

DATASETS = ["hotpotqa", "squad_v2", "pubmedqa_v2"]


# --------------------------
# Config + utility
# --------------------------

def load_cfg() -> Dict[str, Any]:
    if not CFG_PATH.exists():
        raise SystemExit(f"Missing {CFG_PATH}")
    return json.loads(CFG_PATH.read_text(encoding="utf-8-sig"))


def latency_cap_seconds(cfg: Dict[str, Any], dataset: str, expert: str) -> float:
    u = cfg.get("utility", {}) or {}
    caps = u.get("latency_caps_seconds", {}) or {}
    default_cap = float(caps.get("default", 3.0))
    by_ds = (caps.get("by_dataset", {}) or {}).get(dataset, {}) or {}
    return float(by_ds.get(expert, by_ds.get("default", default_cap)))


def utility_value(cfg: Dict[str, Any], dataset: str, expert: str, outcome: Dict[str, Any]) -> float:
    """
    Training-time utility (legacy).
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


# --------------------------
# Tradeoff-U (for Step 4 "make tradeoff real")
# --------------------------

def tradeoff_from_cfg(cfg: Dict[str, Any]) -> Dict[str, float]:
    t = (cfg.get("tradeoff") or {})
    return {
        "w_f1": float(t.get("w_f1", 1.0)),
        "w_em": float(t.get("w_em", 0.0)),
        "w_loose": float(t.get("w_loose", 0.0)),
        "lambda_cost": float(t.get("lambda_cost", 0.0)),
        "wL": float(t.get("wL", 0.7)),
        "wV": float(t.get("wV", 0.3)),
        "sla_latency_s": float(t.get("sla_latency_s", 3.0)),
        "vram_budget_gb": float(t.get("vram_budget_gb", 12.0)),
        "beta_lat": float(t.get("beta_lat", 0.0)),
        "gamma_lat": float(t.get("gamma_lat", 3.0)),
        "beta_vram": float(t.get("beta_vram", 0.0)),
        "gamma_vram": float(t.get("gamma_vram", 3.0)),
    }


def get_latency_s(outcome: Dict[str, Any]) -> float:
    return float(outcome.get("latency", outcome.get("time", 0.0)) or 0.0)


def get_vram_gb(outcome: Dict[str, Any]) -> float:
    mb = float(outcome.get("vram_mb", outcome.get("peak_vram_mb", 0.0)) or 0.0)
    return mb / 1024.0


def tradeoff_U(outcome: Dict[str, Any], tcfg: Dict[str, float]) -> float:
    f1 = float(outcome.get("f1", 0.0) or 0.0)
    em = float(outcome.get("em", 0.0) or 0.0)
    loose = float(outcome.get("loose_em", em) or em)
    Q = tcfg["w_f1"] * f1 + tcfg["w_em"] * em + tcfg["w_loose"] * loose

    L = get_latency_s(outcome)
    V = get_vram_gb(outcome)

    cost_norm = (tcfg["wL"] * (L / max(1e-8, tcfg["sla_latency_s"]))) + (tcfg["wV"] * (V / max(1e-8, tcfg["vram_budget_gb"])))
    U = Q - tcfg["lambda_cost"] * cost_norm

    # optional barrier penalties
    if tcfg["beta_lat"] > 0.0:
        U -= tcfg["beta_lat"] * float(np.exp(tcfg["gamma_lat"] * max(0.0, (L - tcfg["sla_latency_s"]) / max(1e-8, tcfg["sla_latency_s"]))))
    if tcfg["beta_vram"] > 0.0:
        U -= tcfg["beta_vram"] * float(np.exp(tcfg["gamma_vram"] * max(0.0, (V - tcfg["vram_budget_gb"]) / max(1e-8, tcfg["vram_budget_gb"]))))
    return float(U)


def score_for_targets(cfg: Dict[str, Any], dataset: str, expert: str, outcome: Dict[str, Any], *, use_tradeoff: bool, tcfg: Dict[str, float]) -> float:
    if use_tradeoff:
        return tradeoff_U(outcome, tcfg)
    return utility_value(cfg, dataset, expert, outcome)


def read_router_train(dataset: str) -> List[Dict[str, Any]]:
    p = PRED_DIR / f"router_train_{dataset}.jsonl"
    if not p.exists():
        raise SystemExit(f"Missing router train file: {p} (run build_router_train.py)")
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _best_in_pool(cfg: Dict[str, Any], dataset: str, ex: Dict[str, Any], pool: List[str], *, use_tradeoff: bool, tcfg: Dict[str, float]) -> Tuple[str, float]:
    best_e = None
    best_u = -1e18
    for e in pool:
        u = score_for_targets(cfg, dataset, e, ex[e], use_tradeoff=use_tradeoff, tcfg=tcfg)
        if u > best_u:
            best_u = u
            best_e = e
    return best_e, float(best_u)


def _top2_margin_in_pool(cfg: Dict[str, Any], dataset: str, ex: Dict[str, Any], pool: List[str], *, use_tradeoff: bool, tcfg: Dict[str, float]) -> float:
    utils = sorted([score_for_targets(cfg, dataset, e, ex[e], use_tradeoff=use_tradeoff, tcfg=tcfg) for e in pool], reverse=True)
    if len(utils) < 2:
        return 0.0
    return float(utils[0] - utils[1])


# --------------------------
# Policy
# --------------------------

def policy_for_dataset(dataset: str) -> Optional[bool]:
    if dataset == "pubmedqa_v2":
        return False
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
    def __init__(self, X: torch.Tensor, y_soft: torch.Tensor):
        self.X = X
        self.y = y_soft

    def __len__(self):
        return int(self.X.size(0))

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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

def split_train_val_indices(
        n: int,
        val_ratio: float,
        seed: int,
        *,
        min_val: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
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
        for xb, yb in tqdm(tr_loader, desc="Batches[gate]", leave=False):
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
        print(f"[gate] epoch {ep:02d} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc + min_delta:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[gate] early stop. best_val_acc={best_acc:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, float(best_acc)


def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    return -(target_probs * logp).sum(dim=1).mean()


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
) -> Tuple[nn.Module, float]:
    tr_idx, va_idx = split_train_val_indices(int(X.size(0)), 0.2, seed, min_val=min_val)
    X_tr, y_tr = X[tr_idx], y_soft[tr_idx]
    X_va, y_va = X[va_idx], y_soft[va_idx]

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if balanced_sampler:
        if hard_labels_for_sampler is None:
            raise ValueError("balanced_sampler=True but hard_labels_for_sampler is None")
        sampler = make_balanced_sampler_from_hard_labels(hard_labels_for_sampler[tr_idx])
        tr_loader = DataLoader(TensorDatasetSoft(X_tr, y_tr), batch_size=batch_size, sampler=sampler)
    else:
        tr_loader = DataLoader(TensorDatasetSoft(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    va_loader = DataLoader(TensorDatasetSoft(X_va, y_va), batch_size=batch_size, shuffle=False)

    best_score = -1e18
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, tb in tqdm(tr_loader, desc="Batches[sel]", leave=False):
            xb = xb.to(device)
            tb = tb.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = soft_cross_entropy(logits, tb)

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
            for xb, tb in va_loader:
                xb = xb.to(device)
                tb = tb.to(device)
                logits = model(xb)
                vloss = soft_cross_entropy(logits, tb)
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
# Labels: gate + selectors
# --------------------------

def build_gate_deadzone(
        cfg: Dict[str, Any],
        dataset: str,
        rows: List[Dict[str, Any]],
        *,
        gate_delta: float,
        use_tradeoff: bool,
        tcfg: Dict[str, float],
) -> Tuple[List[int], List[int], List[float]]:
    idx_used: List[int] = []
    y_gate: List[int] = []
    margins: List[float] = []

    for i, r in enumerate(rows):
        ex = r["experts"]
        _, br_u = _best_in_pool(cfg, dataset, ex, RAG_EXPERTS, use_tradeoff=use_tradeoff, tcfg=tcfg)
        _, bn_u = _best_in_pool(cfg, dataset, ex, NO_EXPERTS, use_tradeoff=use_tradeoff, tcfg=tcfg)
        m = float(br_u - bn_u)

        if abs(m) < float(gate_delta):
            continue

        idx_used.append(i)
        y_gate.append(1 if m > 0 else 0)
        margins.append(m)

    return idx_used, y_gate, margins


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
        tcfg: Dict[str, float],
) -> torch.Tensor:
    ys = []
    for i in idxs:
        ex = rows[i]["experts"]
        scores = np.array([score_for_targets(cfg, dataset, e, ex[e], use_tradeoff=use_tradeoff, tcfg=tcfg)
                           for e in experts_in_group], dtype=np.float32)
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
        tcfg: Dict[str, float],
) -> List[int]:
    out: List[int] = []
    for i in idxs:
        ex = rows[i]["experts"]
        m = _top2_margin_in_pool(cfg, dataset, ex, pool, use_tradeoff=use_tradeoff, tcfg=tcfg)
        if m >= float(margin_min) and m <= float(margin_max):
            out.append(i)
    return out


# --------------------------
# Answerability (SQuAD v2 only)
# --------------------------

def is_squad_no_answer_gold(row: Dict[str, Any]) -> bool:
    g = row.get("gold_answer", [])
    if isinstance(g, str):
        return g.strip() == "NO_ANSWER"
    if isinstance(g, list):
        return any((isinstance(x, str) and x.strip() == "NO_ANSWER") for x in g)
    return False


def train_answerability_head(
        *,
        X: torch.Tensor,
        y: torch.Tensor,
        in_dim: int,
        device: str,
        hidden_dim: int,
        dropout: float,
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
        prior_mode: str,
) -> Tuple[Dict[str, Any], float]:
    model = MLP(in_dim, hidden_dim, dropout, out_dim=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    tr_idx, va_idx = split_train_val_indices(int(X.size(0)), 0.2, seed, min_val=min_val)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]

    w = class_weights_from_labels(y_tr, 2, device)
    crit = nn.CrossEntropyLoss(weight=w)

    if prior_mode == "balanced":
        prior = torch.tensor([0.5, 0.5], dtype=torch.float32)
    else:
        frac_pos = float(y_tr.float().mean().item()) if y_tr.numel() else 0.5
        prior = torch.tensor([1.0 - frac_pos, frac_pos], dtype=torch.float32)

    if balanced_sampler:
        sampler = make_balanced_sampler_from_hard_labels(y_tr)
        tr_loader = DataLoader(TensorDatasetXY(X_tr, y_tr), batch_size=batch_size, sampler=sampler)
    else:
        tr_loader = DataLoader(TensorDatasetXY(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    va_loader = DataLoader(TensorDatasetXY(X_va, y_va), batch_size=batch_size, shuffle=False)

    best_acc = -1.0
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in tqdm(tr_loader, desc="Batches[ans]", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb.long())

            if reg_weight > 0.0 and reg_type != "none":
                probs = F.softmax(logits, dim=1)
                if reg_type == "entropy":
                    loss = loss - float(reg_weight) * entropy_bonus(probs)
                elif reg_type == "kl":
                    mean_p = probs.mean(dim=0)
                    loss = loss + float(reg_weight) * kl_to_prior(mean_p, prior.to(device))
                else:
                    raise ValueError(f"Unknown ans_reg_type: {reg_type}")

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
        print(f"[ans] epoch {ep:02d} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc + min_delta:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[ans] early stop. best_val_acc={best_acc:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt = {
        "state_dict": model.state_dict(),
        "in_dim": in_dim,
        "prior": prior.tolist(),
        "prior_mode": prior_mode,
        "reg_type": reg_type,
        "reg_weight": float(reg_weight),
    }
    return ckpt, float(best_acc)


# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None, help="Train only one dataset (hotpotqa|squad_v2|pubmedqa_v2)")
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

    ap.add_argument("--gate_delta", type=float, default=0.02, help="Gate deadzone: skip if abs(margin)<delta")
    ap.add_argument("--sel_tau", type=float, default=0.2, help="Softmax temperature for utility targets")

    ap.add_argument("--min_train_selector", type=int, default=20)
    ap.add_argument("--min_val_selector", type=int, default=5)

    # selector collapse prevention
    ap.add_argument("--selector_balanced_sampler", action="store_true", help="Balanced sampler for selector training")
    ap.add_argument("--sel_reg_type", type=str, default="none", choices=["none", "entropy", "kl"])
    ap.add_argument("--sel_reg_weight", type=float, default=0.0)
    ap.add_argument("--sel_prior_mode", type=str, default="balanced", choices=["balanced", "uniform", "empirical"],
                    help="Prior used for selector KL regularization. balanced/uniform -> uniform prior; empirical -> from hard targets.")
    ap.add_argument("--sel_margin_min", type=float, default=0.0, help="Selector margin window min (top1-top2)")
    ap.add_argument("--sel_margin_max", type=float, default=1e9, help="Selector margin window max (top1-top2)")
    ap.add_argument("--print_target_stats", action="store_true", help="Print argmax distribution of selector targets")

    # tradeoff-real training (Step 4)
    ap.add_argument("--tradeoff_mode", action="store_true",
                    help="Use tradeoff_U to compute selector/gate targets instead of legacy utility_value.")
    ap.add_argument("--lambda_cost", type=float, default=None,
                    help="Override cfg.tradeoff.lambda_cost during training when --tradeoff_mode is on.")

    # answerability (SQuAD v2 only)
    ap.add_argument("--train_answerability", action="store_true", help="Train answerability head for squad_v2")
    ap.add_argument("--ans_hidden_dim", type=int, default=256)
    ap.add_argument("--ans_dropout", type=float, default=0.10)
    ap.add_argument("--ans_balanced_sampler", action="store_true")
    ap.add_argument("--ans_reg_type", type=str, default="none", choices=["none", "entropy", "kl"])
    ap.add_argument("--ans_reg_weight", type=float, default=0.0)
    ap.add_argument("--ans_prior_mode", type=str, default="empirical", choices=["empirical", "balanced"])

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_cfg()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # tradeoff config
    tcfg = tradeoff_from_cfg(cfg)
    if args.lambda_cost is not None:
        tcfg["lambda_cost"] = float(args.lambda_cost)

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
        X = embedder.encode(questions, batch_size=args.batch_size).float().cpu()
        in_dim = int(X.size(1))

        pol = policy_for_dataset(dataset)
        print(f"\n=== {dataset} === total={total} policy={pol} embed={embed_model}")
        if args.tradeoff_mode:
            print(f"[{dataset}] tradeoff_mode=True | lambda_cost={tcfg['lambda_cost']}")

        model_dir = out_root / dataset
        model_dir.mkdir(parents=True, exist_ok=True)

        # -----------------
        # Answerability head (SQuAD v2 only)
        # -----------------
        if args.train_answerability and dataset == "squad_v2":
            y_ans = torch.tensor([0 if is_squad_no_answer_gold(r) else 1 for r in rows], dtype=torch.long)
            frac_ans = float(y_ans.float().mean().item())
            print(f"[squad_v2] answerability labels: answerable_frac={frac_ans:.3f} (1=answerable)")

            ans_ckpt, best_acc = train_answerability_head(
                X=X,
                y=y_ans,
                in_dim=in_dim,
                device=args.device,
                hidden_dim=args.ans_hidden_dim,
                dropout=args.ans_dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                min_delta=args.min_delta,
                seed=args.seed,
                min_val=args.min_val_selector,
                balanced_sampler=args.ans_balanced_sampler,
                reg_type=args.ans_reg_type,
                reg_weight=args.ans_reg_weight,
                prior_mode=args.ans_prior_mode,
            )
            ans_ckpt["embed_model"] = embed_model
            ans_path = model_dir / "answerability.pt"
            torch.save(ans_ckpt, ans_path)
            print(f"[squad_v2] saved answerability -> {ans_path} (best_val_acc={best_acc:.4f})")

        # -----------------
        # Gate
        # -----------------
        gate_path = model_dir / "gate.pt"

        if pol is None:
            idx_gate, y_gate_list, _ = build_gate_deadzone(
                cfg, dataset, rows,
                gate_delta=args.gate_delta,
                use_tradeoff=bool(args.tradeoff_mode),
                tcfg=tcfg,
            )

            if len(idx_gate) < 200:
                print(f"[{dataset}] gate: too few examples after deadzone (kept={len(idx_gate)}). "
                      f"Lower gate_delta or accept weak gate.")

            y_gate = torch.tensor(y_gate_list, dtype=torch.long)
            X_gate = X[idx_gate]

            frac_rag = float(y_gate.float().mean().item()) if y_gate.numel() else 0.0
            maj = max(frac_rag, 1 - frac_rag) if y_gate.numel() else 0.0
            print(f"[{dataset}] gate deadzone kept={len(idx_gate)}/{total} | frac_rag={frac_rag:.3f} | majority≈{maj:.3f}")

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

            torch.save(
                {
                    "state_dict": gate_model.state_dict(),
                    "in_dim": in_dim,
                    "embed_model": embed_model,
                    "gate_delta": float(args.gate_delta),
                    "best_val_acc": float(best_acc),
                    "tradeoff_mode": bool(args.tradeoff_mode),
                    "lambda_cost": float(tcfg["lambda_cost"]),
                },
                gate_path,
            )
            print(f"[{dataset}] saved gate -> {gate_path} (best_val_acc={best_acc:.4f})")
        else:
            torch.save({"forced_policy": bool(pol), "in_dim": in_dim, "embed_model": embed_model}, gate_path)
            print(f"[{dataset}] gate forced={pol}; wrote marker -> {gate_path}")

        # -----------------
        # Selector(s): soft targets
        # -----------------
        sel_rag_path = model_dir / "selector_rag.pt"
        sel_no_path  = model_dir / "selector_no_rag.pt"

        idx_rag: List[int] = []
        idx_no: List[int] = []

        if pol is True:
            idx_rag = list(range(total))
            idx_no = []
        elif pol is False:
            idx_rag = []
            idx_no = list(range(total))
        else:
            for i, r in enumerate(rows):
                ex = r["experts"]
                _, br_u = _best_in_pool(cfg, dataset, ex, RAG_EXPERTS, use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg)
                _, bn_u = _best_in_pool(cfg, dataset, ex, NO_EXPERTS, use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg)
                m = float(br_u - bn_u)
                if m >= float(args.gate_delta):
                    idx_rag.append(i)
                elif m <= -float(args.gate_delta):
                    idx_no.append(i)

        idx_rag_f = filter_by_margin_window(
            cfg, dataset, rows, idx_rag, RAG_EXPERTS,
            margin_min=args.sel_margin_min, margin_max=args.sel_margin_max,
            use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg,
        )
        idx_no_f = filter_by_margin_window(
            cfg, dataset, rows, idx_no, NO_EXPERTS,
            margin_min=args.sel_margin_min, margin_max=args.sel_margin_max,
            use_tradeoff=bool(args.tradeoff_mode), tcfg=tcfg,
        )

        print(f"[{dataset}] selector train counts | rag={len(idx_rag)}→{len(idx_rag_f)} no={len(idx_no)}→{len(idx_no_f)} "
              f"(gate_delta={args.gate_delta}, sel_margin=[{args.sel_margin_min},{args.sel_margin_max}])")

        # ---------- helper: prior ----------
        def make_selector_prior(y_hard: torch.Tensor, num_classes: int) -> torch.Tensor:
            if args.sel_prior_mode in ("balanced", "uniform"):
                return torch.full((num_classes,), 1.0 / float(num_classes), dtype=torch.float32)
            # empirical:
            counts = torch.bincount(y_hard, minlength=num_classes).float()
            if float(counts.sum().item()) <= 0:
                return torch.full((num_classes,), 1.0 / float(num_classes), dtype=torch.float32)
            return (counts / counts.sum()).clamp_min(1e-8)

        # ---------- RAG selector ----------
        if should_train_selector(len(idx_rag_f), min_train=args.min_train_selector, min_val=args.min_val_selector,
                                 name="selector_rag", ds=dataset):
            Xg = X[idx_rag_f]
            y_soft = build_selector_soft_targets(
                cfg, dataset, rows, idx_rag_f, RAG_EXPERTS,
                tau=args.sel_tau,
                use_tradeoff=bool(args.tradeoff_mode),
                tcfg=tcfg,
            )
            y_hard = hard_argmax_from_soft(y_soft)

            if args.print_target_stats and y_hard.numel() > 0:
                cnt = Counter(y_hard.tolist())
                dist = [(RAG_EXPERTS[k], v) for k, v in sorted(cnt.items(), key=lambda x: -x[1])]
                print(f"[{dataset}] selector_rag hard-target argmax distribution: {dist} (total={len(idx_rag_f)})")

            prior = None
            if args.sel_reg_type == "kl" and y_hard.numel() > 0:
                prior = make_selector_prior(y_hard, len(RAG_EXPERTS))

            dropout_used = 0.0 if len(idx_rag_f) < 200 else float(args.dropout)
            sel_model = MLP(in_dim, args.hidden_dim, dropout_used, out_dim=len(RAG_EXPERTS))

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
            )

            torch.save(
                {
                    "state_dict": sel_model.state_dict(),
                    "in_dim": in_dim,
                    "experts": RAG_EXPERTS,
                    "embed_model": embed_model,
                    "sel_tau": float(args.sel_tau),
                    "trained_on_gate_delta": float(args.gate_delta),
                    "trained_on_sel_margin_min": float(args.sel_margin_min),
                    "trained_on_sel_margin_max": float(args.sel_margin_max),
                    "reg_type": str(args.sel_reg_type),
                    "reg_weight": float(args.sel_reg_weight),
                    "prior_mode": str(args.sel_prior_mode),
                    "prior": (prior.tolist() if prior is not None else None),
                    "tradeoff_mode": bool(args.tradeoff_mode),
                    "lambda_cost": float(tcfg["lambda_cost"]),
                    "best_val_score": float(best_score),
                },
                sel_rag_path,
            )
            print(f"[{dataset}] saved selector_rag -> {sel_rag_path} (best_val_score={best_score:.4f})")
        else:
            print(f"[{dataset}] skip selector_rag (count={len(idx_rag_f)})")

        # ---------- NO selector ----------
        if should_train_selector(len(idx_no_f), min_train=args.min_train_selector, min_val=args.min_val_selector,
                                 name="selector_no_rag", ds=dataset):
            Xg = X[idx_no_f]
            y_soft = build_selector_soft_targets(
                cfg, dataset, rows, idx_no_f, NO_EXPERTS,
                tau=args.sel_tau,
                use_tradeoff=bool(args.tradeoff_mode),
                tcfg=tcfg,
            )
            y_hard = hard_argmax_from_soft(y_soft)

            if args.print_target_stats and y_hard.numel() > 0:
                cnt = Counter(y_hard.tolist())
                dist = [(NO_EXPERTS[k], v) for k, v in sorted(cnt.items(), key=lambda x: -x[1])]
                print(f"[{dataset}] selector_no_rag hard-target argmax distribution: {dist} (total={len(idx_no_f)})")

            prior = None
            if args.sel_reg_type == "kl" and y_hard.numel() > 0:
                prior = make_selector_prior(y_hard, len(NO_EXPERTS))

            dropout_used = 0.0 if len(idx_no_f) < 200 else float(args.dropout)
            sel_model = MLP(in_dim, args.hidden_dim, dropout_used, out_dim=len(NO_EXPERTS))

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
            )

            torch.save(
                {
                    "state_dict": sel_model.state_dict(),
                    "in_dim": in_dim,
                    "experts": NO_EXPERTS,
                    "embed_model": embed_model,
                    "sel_tau": float(args.sel_tau),
                    "trained_on_gate_delta": float(args.gate_delta),
                    "trained_on_sel_margin_min": float(args.sel_margin_min),
                    "trained_on_sel_margin_max": float(args.sel_margin_max),
                    "reg_type": str(args.sel_reg_type),
                    "reg_weight": float(args.sel_reg_weight),
                    "prior_mode": str(args.sel_prior_mode),
                    "prior": (prior.tolist() if prior is not None else None),
                    "tradeoff_mode": bool(args.tradeoff_mode),
                    "lambda_cost": float(tcfg["lambda_cost"]),
                    "best_val_score": float(best_score),
                },
                sel_no_path,
            )
            print(f"[{dataset}] saved selector_no_rag -> {sel_no_path} (best_val_score={best_score:.4f})")
        else:
            print(f"[{dataset}] skip selector_no_rag (count={len(idx_no_f)})")

    print(f"\nDONE. Models saved under: {out_root}")


if __name__ == "__main__":
    main()
