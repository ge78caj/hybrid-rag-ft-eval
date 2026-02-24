# eval_router_two_stage.py
#
# FIXES for Ryo requirements (Problems 2 & 3):
#
# (2) PubMed gate must actually make retrieval decision:
#     - Add --pubmed_policy {forced,none} (default=forced for backward compat)
#     - If pubmed_policy=none, PubMed is NOT forced to RAG in family decision.
#     - If --use_shared_gate is enabled (PubMed+CSQA shared gate), we also do NOT force PubMed.
#
# (3) Stage-2 passage embeddings:
#     - Keep --use_passage_embeddings; when enabled, we can build passage embeddings.
#     - Robust dim-handling: each checkpoint (gate / selector / shared gate) may have different in_dim.
#       We build base Xq + features Xf (+ optional passage Xp), and then FEED each model the slice it expects:
#         * if ckpt_in_dim == Dq + Df -> no passages
#         * if ckpt_in_dim == Dq + Df + Dq -> with passages
#       This prevents the “mat1 1x791 vs 1559x256” crash and allows mixed checkpoints in combined eval.
#     - If a checkpoint requires passages but --use_passage_embeddings is OFF, we raise a clear error.
#
# NOTE:
# - This file does NOT require changing other files, as we infer “passages used” from ckpt["in_dim"].
# - You must have trained checkpoints for each dataset you evaluate (gate + selector_rag at minimum).
#
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

CFG_PATH = Path("configs/router_config.json")
PRED_DIR = Path("prediction")

CANON_RAG_EXPERTS = ["base_rag", "sft_rag", "raft_rag"]
CANON_NO_EXPERTS  = ["base_only", "sft_only"]

DATASETS = ["hotpotqa", "squad_v2", "pubmedqa_v2", "commonsenseqa"]
SHARED_GATE_NAME = "combined_pubmed_csqa_gate"


# --------------------------
# IO / config
# --------------------------

def load_cfg() -> Dict[str, Any]:
    if not CFG_PATH.exists():
        raise SystemExit(f"Missing {CFG_PATH}")
    return json.loads(CFG_PATH.read_text(encoding="utf-8-sig"))


def read_router_train(dataset: str) -> List[Dict[str, Any]]:
    p = PRED_DIR / f"router_train_{dataset}.jsonl"
    if not p.exists():
        raise SystemExit(f"Missing router train file: {p}")
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _get_gold_field(row: Dict[str, Any]) -> Any:
    if "gold_answer" in row:
        return row.get("gold_answer")
    if "gold_answers" in row:
        return row.get("gold_answers")
    return None


def is_squad_no_answer_gold(row: Dict[str, Any]) -> bool:
    g = _get_gold_field(row)
    if g is None:
        return False
    if isinstance(g, str):
        return g.strip() == "NO_ANSWER"
    if isinstance(g, list):
        return any((isinstance(x, str) and x.strip() == "NO_ANSWER") for x in g)
    return False


# --------------------------
# Expert pools
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


# --------------------------
# Models
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
# Policy / shared gate applicability
# --------------------------

def should_use_shared_gate_for_dataset(ds: str) -> bool:
    return ds in ("pubmedqa_v2", "commonsenseqa")


def policy_for_dataset(dataset: str, *, pubmed_policy: str) -> Optional[bool]:
    # Ryo: always use RAG for these datasets
    if dataset in ("hotpotqa", "squad_v2"):
        return True

    # PubMed controlled separately (shared gate recommended)
    if dataset == "pubmedqa_v2":
        return True if pubmed_policy == "forced" else None

    return None


# --------------------------
# Feature loading + standardization
# --------------------------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
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
        keys.update(map(str, feats.keys()))
    return sorted(keys)


def build_feature_matrix(rows: List[Dict[str, Any]], fmap: Dict[str, Dict[str, float]], feature_keys: List[str]) -> torch.Tensor:
    n = len(rows)
    d = len(feature_keys)
    Xf = torch.zeros((n, d), dtype=torch.float32)
    if d == 0:
        return Xf
    for i, r in enumerate(rows):
        rid = str(r.get("id", i))
        ds = str(r.get("dataset", ""))  # should exist
        rid_key = f"{ds}::{rid}" if ds else rid
        feats = fmap.get(rid_key, {})
        for j, k in enumerate(feature_keys):
            if k in feats:
                Xf[i, j] = float(feats[k])
    return Xf


def apply_feature_standardization(Xf: torch.Tensor, stats: Dict[str, Any], eps: float = 1e-8) -> torch.Tensor:
    if Xf.numel() == 0 or Xf.size(1) == 0:
        return Xf
    mean = stats.get("mean", None)
    std = stats.get("std", None)
    if not mean or not std:
        return Xf
    mean_t = torch.tensor(mean, dtype=torch.float32, device=Xf.device).view(1, -1)
    std_t = torch.tensor(std, dtype=torch.float32, device=Xf.device).view(1, -1).clamp_min(eps)
    if mean_t.size(1) != Xf.size(1) or std_t.size(1) != Xf.size(1):
        print("[WARN] feature_stats dim mismatch; skipping standardization.")
        return Xf
    return (Xf - mean_t) / std_t


def standardize_features_fit(Xf: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if Xf.numel() == 0 or Xf.size(1) == 0:
        return Xf, {"mean": [], "std": []}
    mean = Xf.mean(dim=0, keepdim=True)
    std = Xf.std(dim=0, keepdim=True).clamp_min(eps)
    return (Xf - mean) / std, {"mean": mean.squeeze(0).tolist(), "std": std.squeeze(0).tolist()}


# --------------------------
# Passage parsing + embeddings (Problem 3)
# --------------------------

_DOC_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.DOTALL)

def extract_docs_from_prediction_raw(pred_raw: str, max_docs: int, max_chars: int) -> List[str]:
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
        device: str,
        *,
        source_expert: str = "base_rag",
        max_docs: int = 5,
        max_chars: int = 1200,
        batch_size_docs: int = 64,
) -> torch.Tensor:
    """
    Returns [N, Dq] mean(doc_embeds) for docs in experts[source_expert].prediction_raw.
    If no docs exist for a row => zeros. If no docs exist at all => zeros.
    """
    # Determine embedding dim Dq
    dummy = embedder.encode(["dummy"], batch_size=1).float().to(device)
    d = int(dummy.shape[-1])

    flat_docs: List[str] = []
    offsets: List[Tuple[int, int]] = []

    for r in rows:
        ex = r["experts"]
        pr = ""
        if source_expert in ex:
            pr = ex[source_expert].get("prediction_raw", "") or ""
        docs = extract_docs_from_prediction_raw(pr, max_docs=max_docs, max_chars=max_chars)
        s = len(flat_docs)
        flat_docs.extend(docs)
        e = len(flat_docs)
        offsets.append((s, e))

    if len(flat_docs) == 0:
        return torch.zeros((len(rows), d), dtype=torch.float32, device=device)

    E = embedder.encode(flat_docs, batch_size=batch_size_docs).float().to(device)

    Xp = torch.zeros((len(rows), d), dtype=torch.float32, device=device)
    for i, (s, e) in enumerate(offsets):
        if e > s:
            Xp[i] = E[s:e].mean(dim=0)
    return Xp


# --------------------------
# Checkpoint loading
# --------------------------

def load_gate(model_dir: Path, device: str, hidden_dim: int, dropout: float) -> Dict[str, Any]:
    p = model_dir / "gate.pt"
    if not p.exists():
        raise SystemExit(f"Missing gate checkpoint: {p}")
    ckpt = torch.load(p, map_location="cpu", weights_only=False)

    if "forced_policy" in ckpt:
        return {
            "forced_policy": ckpt["forced_policy"],
            "in_dim": int(ckpt.get("in_dim") or 0),
            "embed_model": ckpt.get("embed_model"),
            "feature_keys": ckpt.get("feature_keys", []),
            "feature_stats": ckpt.get("feature_stats", None),
            "model": None,
            "gate_objective": "forced",
            "calibrated_threshold": float(ckpt.get("calibrated_threshold", 0.5) or 0.5),
            "raw_ckpt": ckpt,
        }

    in_dim = int(ckpt["in_dim"])
    gate_objective = str(ckpt.get("gate_objective", "cls"))

    if gate_objective == "delta_reg":
        model = MLP(in_dim, hidden_dim, dropout, out_dim=1)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        return {
            "forced_policy": None,
            "in_dim": in_dim,
            "embed_model": ckpt.get("embed_model"),
            "feature_keys": ckpt.get("feature_keys", []),
            "feature_stats": ckpt.get("feature_stats", None),
            "model": model,
            "gate_objective": "delta_reg",
            "calibrated_threshold": float(ckpt.get("calibrated_threshold", 0.0) or 0.0),
            "raw_ckpt": ckpt,
        }

    model = MLP(in_dim, hidden_dim, dropout, out_dim=2)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return {
        "forced_policy": None,
        "in_dim": in_dim,
        "embed_model": ckpt.get("embed_model"),
        "feature_keys": ckpt.get("feature_keys", []),
        "feature_stats": ckpt.get("feature_stats", None),
        "model": model,
        "gate_objective": "cls",
        "calibrated_threshold": float(ckpt.get("calibrated_threshold", 0.5) or 0.5),
        "raw_ckpt": ckpt,
    }


def load_selector(model_dir: Path, which: str, device: str, hidden_dim: int, dropout: float) -> Optional[Dict[str, Any]]:
    p = model_dir / f"selector_{which}.pt"
    if not p.exists():
        return None
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    in_dim = int(ckpt["in_dim"])
    experts = list(ckpt["experts"])
    model = MLP(in_dim, hidden_dim, dropout, out_dim=len(experts))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return {
        "model": model,
        "experts": experts,
        "in_dim": in_dim,
        "embed_model": ckpt.get("embed_model"),
        "feature_keys": ckpt.get("feature_keys", []),
        "feature_stats": ckpt.get("feature_stats", None),
        "raw_ckpt": ckpt,
    }


# --------------------------
# Utility/tradeoff (from router_config.json utility)
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


def parse_csv_floats(s: Optional[str]) -> Optional[List[float]]:
    if not s:
        return None
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(float(x))
    return out


# --------------------------
# Dim-safe model input assembly (Problem 3 robustness)
# --------------------------

def assemble_model_inputs(
        *,
        Xq: torch.Tensor,   # [N, Dq]
        Xf: torch.Tensor,   # [N, Df] or [N,0]
        Xp: Optional[torch.Tensor],  # [N, Dq] or None
        expected_in_dim: int,
        model_name: str,
        require_passages_ok: bool,
) -> torch.Tensor:
    """
    Return [N, expected_in_dim] by concatenating the needed components.
    We assume passage embeddings (if used) have dim == Dq.
    """
    Dq = int(Xq.size(1))
    Df = int(Xf.size(1))
    base = Dq + Df
    with_pass = base + Dq

    if expected_in_dim == base:
        return torch.cat([Xq, Xf], dim=1) if Df > 0 else Xq

    if expected_in_dim == with_pass:
        if Xp is None:
            if not require_passages_ok:
                raise SystemExit(
                    f"[DIM] {model_name} expects passages (in_dim={expected_in_dim} == {base}+{Dq}), "
                    f"but passage embeddings are not available. Re-run eval with --use_passage_embeddings."
                )
            # Should not happen if require_passages_ok=True implies Xp provided; keep safe.
            raise SystemExit(f"[DIM] {model_name} expects passages but Xp is None.")
        return torch.cat([Xq, Xf, Xp], dim=1) if Df > 0 else torch.cat([Xq, Xp], dim=1)

    raise SystemExit(
        f"[DIM] {model_name} in_dim mismatch: expected {expected_in_dim}, but possible are "
        f"base={base} (no passages) or base+pass={with_pass} (with passages). "
        f"This usually means you changed feature_keys or embed dim between train/eval."
    )


# --------------------------
# Core evaluation per dataset
# --------------------------

def run_eval_dataset(
        *,
        cfg: Dict[str, Any],
        dataset: str,
        rows: List[Dict[str, Any]],
        model_root: Path,
        device: str,
        batch_size: int,
        hidden_dim: int,
        dropout: float,
        tradeoff_mode: bool,
        gate_threshold: Optional[float],
        oracle_policy_aligned: bool,
        oracle_gate: bool,
        oracle_selector: bool,
        feature_fmap: Dict[str, Dict[str, float]],
        feature_keys_cli: Optional[List[str]],
        standardize_features_flag: bool,
        use_shared_gate: bool,
        shared_gate: Optional[Dict[str, Any]],
        pubmed_policy: str,  # forced|none
        use_passage_embeddings: bool,
        passage_source_expert: str,
        passage_max_docs: int,
        passage_max_chars: int,
) -> Dict[str, Any]:

    model_dir = model_root / dataset

    pol = policy_for_dataset(dataset, pubmed_policy=pubmed_policy)
    # If shared gate is enabled for PubMed/CSQA, we do NOT force policy in family decision
    if use_shared_gate and shared_gate is not None and should_use_shared_gate_for_dataset(dataset):
        pol_for_family = None
    else:
        pol_for_family = pol

    rag_pool_ds, no_pool_ds = pools_for_dataset(dataset)

    gate = load_gate(model_dir, device, hidden_dim, dropout)
    sel_rag = load_selector(model_dir, "rag", device, hidden_dim, dropout)
    sel_no  = load_selector(model_dir, "no_rag", device, hidden_dim, dropout)

    # Minimal sanity: selector_rag must exist (all our routers can always pick RAG experts).
    if sel_rag is None:
        raise SystemExit(f"[{dataset}] missing selector_rag.pt in {model_dir}")

    # feature keys: prefer CLI > ckpt > infer
    ckpt_feature_keys = (
            (gate.get("feature_keys") or [])
            or (sel_rag.get("feature_keys") if sel_rag else [])
            or (sel_no.get("feature_keys") if sel_no else [])
            or []
    )
    if feature_keys_cli is not None:
        feature_keys = feature_keys_cli
    elif ckpt_feature_keys:
        feature_keys = list(ckpt_feature_keys)
    elif feature_fmap:
        feature_keys = infer_feature_keys_from_map(feature_fmap)
    else:
        feature_keys = []

    # embed model: prefer dataset ckpts
    embed_model = (
            gate.get("embed_model")
            or (sel_rag.get("embed_model") if sel_rag else None)
            or (sel_no.get("embed_model") if sel_no else None)
            or "sentence-transformers/all-mpnet-base-v2"
    )

    # embed questions
    questions = [r["question"] for r in rows]
    embedder = Embedder(embed_model, device=device)
    Xq = embedder.encode(questions, batch_size=batch_size).float()  # [N, Dq] on device

    # numeric features
    feat_stats_used = None
    if feature_keys:
        Xf = build_feature_matrix(rows, feature_fmap, feature_keys).to(device)
        if standardize_features_flag and Xf.size(1) > 0:
            stats = gate.get("feature_stats") or (sel_rag.get("feature_stats") if sel_rag else None) or (sel_no.get("feature_stats") if sel_no else None)
            if stats:
                Xf = apply_feature_standardization(Xf, stats)
                feat_stats_used = "ckpt"
            else:
                Xf, _ = standardize_features_fit(Xf)
                feat_stats_used = "fit_eval"
    else:
        Xf = torch.zeros((len(rows), 0), dtype=torch.float32, device=device)

    # Passage embeddings are built once per dataset IF enabled.
    # We will only feed them to models whose ckpt in_dim indicates they need them.
    Xp = None
    if use_passage_embeddings:
        Xp = build_passage_embedding_matrix(
            rows, embedder, device,
            source_expert=passage_source_expert,
            max_docs=passage_max_docs,
            max_chars=passage_max_chars,
            batch_size_docs=batch_size,
        )

    def default_gate_threshold(gate_obj: Dict[str, Any]) -> float:
        if gate_threshold is not None:
            return float(gate_threshold)
        return float(gate_obj.get("calibrated_threshold", 0.5) or 0.5)

    def oracle_pool_for_dataset() -> List[str]:
        # oracle_policy_aligned should respect CURRENT policy choice (pubmed_policy flag)
        if oracle_policy_aligned and pol is not None:
            return rag_pool_ds if pol is True else no_pool_ds
        return rag_pool_ds + no_pool_ds

    tcfg = tradeoff_from_cfg(cfg)
    pool = oracle_pool_for_dataset()

    def best_in_family(ex: Dict[str, Any], fam: str) -> str:
        rp, npool = pools_for_row(ex, rag_pool_ds, no_pool_ds)
        fam_pool = rp if fam == "rag" else npool
        if not fam_pool:
            fam_pool = rp + npool
        best_e, best_v = None, -1e18
        for e in fam_pool:
            v = tradeoff_U(ex[e], tcfg, dataset, e) if tradeoff_mode else float(ex[e].get("f1", 0.0))
            if v > best_v:
                best_v, best_e = v, e
        return str(best_e)

    def family_oracle_label(ex: Dict[str, Any]) -> str:
        rp, npool = pools_for_row(ex, rag_pool_ds, no_pool_ds)
        if rp and not npool:
            return "rag"
        if npool and not rp:
            return "no"
        if not rp and not npool:
            return "no"
        br = best_in_family(ex, "rag")
        bn = best_in_family(ex, "no")
        vr = tradeoff_U(ex[br], tcfg, dataset, br) if tradeoff_mode else float(ex[br].get("f1", 0.0))
        vn = tradeoff_U(ex[bn], tcfg, dataset, bn) if tradeoff_mode else float(ex[bn].get("f1", 0.0))
        return "rag" if vr > vn else "no"

    # Prepare per-model input matrices with correct dims (Problem 3 fix)
    used_shared = bool(use_shared_gate and shared_gate is not None and should_use_shared_gate_for_dataset(dataset))

    X_gate = None
    if gate.get("gate_objective") != "forced":
        X_gate = assemble_model_inputs(
            Xq=Xq, Xf=Xf, Xp=Xp,
            expected_in_dim=int(gate["in_dim"]),
            model_name=f"{dataset}/gate",
            require_passages_ok=bool(use_passage_embeddings),
        )

    X_sel_rag = assemble_model_inputs(
        Xq=Xq, Xf=Xf, Xp=Xp,
        expected_in_dim=int(sel_rag["in_dim"]),
        model_name=f"{dataset}/selector_rag",
        require_passages_ok=bool(use_passage_embeddings),
    )

    X_sel_no = None
    if sel_no is not None:
        X_sel_no = assemble_model_inputs(
            Xq=Xq, Xf=Xf, Xp=Xp,
            expected_in_dim=int(sel_no["in_dim"]),
            model_name=f"{dataset}/selector_no_rag",
            require_passages_ok=bool(use_passage_embeddings),
        )

    X_shared = None
    if used_shared:
        X_shared = assemble_model_inputs(
            Xq=Xq, Xf=Xf, Xp=Xp,
            expected_in_dim=int(shared_gate["in_dim"]),
            model_name=f"{SHARED_GATE_NAME}/gate(shared)",
            require_passages_ok=bool(use_passage_embeddings),
        )

    def gate_predict_family(i: int) -> str:
        # Shared gate override (PubMed/CSQA)
        if used_shared:
            sg = shared_gate
            thr = default_gate_threshold(sg)
            gmodel = sg["model"]
            xi = X_shared[i]
            with torch.no_grad():
                logits = gmodel(xi.unsqueeze(0))[0]
                probs = torch.softmax(logits, dim=0)
                p_rag = float(probs[1].item())
                return "rag" if p_rag >= thr else "no"

        # Forced policy for family decision
        if pol_for_family is True:
            return "rag"
        if pol_for_family is False:
            return "no"

        # Normal dataset gate
        if gate.get("gate_objective") == "forced":
            forced = gate.get("forced_policy")
            return "rag" if forced else "no"

        thr = default_gate_threshold(gate)
        gmodel = gate["model"]
        xi = X_gate[i]
        with torch.no_grad():
            if gate.get("gate_objective") == "delta_reg":
                pred_delta = float(gmodel(xi.unsqueeze(0)).squeeze(0).squeeze(-1).item())
                return "rag" if pred_delta > thr else "no"
            else:
                logits = gmodel(xi.unsqueeze(0))[0]
                probs = torch.softmax(logits, dim=0)
                p_rag = float(probs[1].item())
                return "rag" if p_rag >= thr else "no"

    chosen_f1 = chosen_em = chosen_u = 0.0
    oracle_f1 = oracle_em = oracle_u = 0.0
    pick_counts = Counter()
    gate_counts = Counter()

    gate_oracle_TN = gate_oracle_TP = gate_oracle_FP = gate_oracle_FN = 0
    ent_sum = 0.0
    maxp_sum = 0.0
    diag_steps = 0

    for i in tqdm(range(len(rows)), desc=f"Eval {dataset}"):
        r = rows[i]
        ex = r["experts"]

        rag_pool_row, no_pool_row = pools_for_row(ex, rag_pool_ds, no_pool_ds)

        # 1) family decision
        if oracle_gate:
            fam = family_oracle_label(ex)
        else:
            fam = gate_predict_family(i)

        gate_counts[fam] += 1

        # gate oracle agreement (meaningful only when family not forced)
        if pol_for_family is None and (not oracle_gate):
            fam_or = family_oracle_label(ex)
            if fam == "rag" and fam_or == "rag":
                gate_oracle_TP += 1
            elif fam == "no" and fam_or == "no":
                gate_oracle_TN += 1
            elif fam == "rag" and fam_or == "no":
                gate_oracle_FP += 1
            else:
                gate_oracle_FN += 1

        # 2) within-family selection
        if oracle_selector:
            chosen_expert = best_in_family(ex, fam)
        else:
            if fam == "rag":
                sm = sel_rag["model"]
                experts = sel_rag["experts"]
                xi = X_sel_rag[i]
                with torch.no_grad():
                    logits = sm(xi.unsqueeze(0))[0]
                    probs = torch.softmax(logits, dim=0)
                    cls = int(torch.argmax(probs).item())
                    chosen_expert = experts[cls]
                    p = probs.detach().cpu().numpy()
                    ent_sum += float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum())
                    maxp_sum += float(p.max())
                    diag_steps += 1
            else:
                # if selector_no_rag missing, fallback deterministically
                if sel_no is None or X_sel_no is None:
                    chosen_expert = best_in_family(ex, "no")
                else:
                    sm = sel_no["model"]
                    experts = sel_no["experts"]
                    xi = X_sel_no[i]
                    with torch.no_grad():
                        logits = sm(xi.unsqueeze(0))[0]
                        probs = torch.softmax(logits, dim=0)
                        cls = int(torch.argmax(probs).item())
                        chosen_expert = experts[cls]
                        p = probs.detach().cpu().numpy()
                        ent_sum += float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum())
                        maxp_sum += float(p.max())
                        diag_steps += 1

        # safety fallback if chosen expert missing in row
        if chosen_expert not in ex:
            fallback_pool = (rag_pool_row if fam == "rag" else no_pool_row) or (rag_pool_row + no_pool_row)
            chosen_expert = fallback_pool[0] if fallback_pool else list(ex.keys())[0]

        pick_counts[chosen_expert] += 1
        out = ex[chosen_expert]
        chosen_f1 += float(out.get("f1", 0.0))
        chosen_em += float(out.get("em", 0.0))
        chosen_u += float(tradeoff_U(out, tcfg, dataset, chosen_expert) if tradeoff_mode else float(out.get("f1", 0.0)))

        # ORACLE overall (best among pool + NO_ANSWER for SQuAD)
        best_expert = None
        best_val = -1e18
        pool_row = [e for e in pool if e in ex] or list(ex.keys())
        for e in pool_row:
            val = tradeoff_U(ex[e], tcfg, dataset, e) if tradeoff_mode else float(ex[e].get("f1", 0.0))
            if val > best_val:
                best_val = val
                best_expert = e

        if dataset == "squad_v2":
            ok = 1.0 if is_squad_no_answer_gold(r) else 0.0
            out_na = {"f1": ok, "em": ok, "loose_em": ok, "latency": 0.0, "vram_mb": 0.0}
            val_na = tradeoff_U(out_na, tcfg, dataset, None) if tradeoff_mode else ok
            if val_na > best_val:
                out_best = out_na
                best_expert_for_u = None
            else:
                out_best = ex[best_expert]
                best_expert_for_u = best_expert
        else:
            out_best = ex[best_expert]
            best_expert_for_u = best_expert

        oracle_f1 += float(out_best.get("f1", 0.0))
        oracle_em += float(out_best.get("em", 0.0))
        oracle_u += float(tradeoff_U(out_best, tcfg, dataset, best_expert_for_u) if tradeoff_mode else float(out_best.get("f1", 0.0)))

    n = len(rows)

    gate_agree = None
    if pol_for_family is None and (not oracle_gate):
        total_g = gate_oracle_TN + gate_oracle_TP + gate_oracle_FP + gate_oracle_FN
        if total_g > 0:
            gate_agree = {
                "acc": (gate_oracle_TN + gate_oracle_TP) / total_g,
                "conf": {"TN": gate_oracle_TN, "TP": gate_oracle_TP, "FP": gate_oracle_FP, "FN": gate_oracle_FN},
            }

    selector_diag = None
    if diag_steps > 0 and (not oracle_selector):
        selector_diag = {
            "mean_entropy": ent_sum / diag_steps,
            "mean_maxprob": maxp_sum / diag_steps,
            "steps": diag_steps,
        }

    thr_used = None
    if pol_for_family is None and (not oracle_gate):
        if used_shared:
            thr_used = float(gate_threshold if gate_threshold is not None else shared_gate.get("calibrated_threshold", 0.5))
        else:
            thr_used = float(gate_threshold if gate_threshold is not None else gate.get("calibrated_threshold", 0.5))

    # Report whether this dataset's ckpts actually used passages (by in_dim)
    Dq = int(Xq.size(1))
    Df = int(Xf.size(1))
    base = Dq + Df
    used_pass_gate = (gate.get("in_dim") == base + Dq) if gate.get("in_dim") else False
    used_pass_rag  = (sel_rag.get("in_dim") == base + Dq) if sel_rag else False
    used_pass_no   = (sel_no.get("in_dim") == base + Dq) if sel_no else False
    used_pass_any = bool(used_pass_gate or used_pass_rag or used_pass_no)

    return {
        "dataset": dataset,
        "N": n,
        "chosen_f1": chosen_f1 / n,
        "chosen_em": chosen_em / n,
        "chosen_u": chosen_u / n,
        "oracle_f1": oracle_f1 / n,
        "oracle_em": oracle_em / n,
        "oracle_u": oracle_u / n,
        "gate_counts": dict(gate_counts),
        "pick_counts": pick_counts.most_common(10),
        "gate_oracle_agreement": gate_agree,
        "selector_diag": selector_diag,
        "embed_model": embed_model,
        "policy": pol,
        "policy_for_family": pol_for_family,
        "pubmed_policy": pubmed_policy if dataset == "pubmedqa_v2" else None,
        "gate_threshold": thr_used,
        "gate_objective": (shared_gate.get("gate_objective") if used_shared else gate.get("gate_objective", "cls")),
        "oracle_pool": pool,
        "feature_keys_used": feature_keys,
        "feature_standardization": ("on" if standardize_features_flag else "off"),
        "feature_stats_used": feat_stats_used,
        "pools_used": {"rag": rag_pool_ds, "no": no_pool_ds},
        "used_shared_gate": used_shared,
        "passage_embeddings_enabled": bool(use_passage_embeddings),
        "passage_embeddings_used_by_ckpt": used_pass_any,
        "dims": {"Dq": Dq, "Df": Df, "base": base, "base_plus_pass": base + Dq},
        "ckpt_dims": {
            "gate_in_dim": int(gate.get("in_dim") or 0),
            "sel_rag_in_dim": int(sel_rag.get("in_dim") or 0) if sel_rag else 0,
            "sel_no_in_dim": int(sel_no.get("in_dim") or 0) if sel_no else 0,
            "shared_gate_in_dim": int(shared_gate.get("in_dim") or 0) if used_shared else 0,
        }
    }


def print_res(tag: str, res: Dict[str, Any], oracle_policy_aligned: bool, tradeoff_mode: bool, oracle_gate: bool, oracle_selector: bool):
    dataset = res["dataset"]
    print(
        f"\n--- {dataset} --- {tag} N={res['N']} "
        f"policy={res['policy']} policy_for_family={res['policy_for_family']} oracle_policy_aligned={bool(oracle_policy_aligned)} "
        f"tradeoff_mode={bool(tradeoff_mode)} "
        f"oracle_gate={bool(oracle_gate)} oracle_selector={bool(oracle_selector)}"
    )
    print(f"embed_model={res['embed_model']}")
    print(f"gate_objective={res.get('gate_objective')}")
    if dataset == "pubmedqa_v2":
        print(f"pubmed_policy={res.get('pubmed_policy')}")
    if res.get("gate_threshold") is not None:
        print(f"gate_threshold_used={res['gate_threshold']}")
    if res.get("used_shared_gate"):
        print(f"used_shared_gate=True ({SHARED_GATE_NAME})")

    if res.get("passage_embeddings_enabled"):
        print(f"passage_embeddings_enabled=True | used_by_ckpt={res.get('passage_embeddings_used_by_ckpt')}")
        print(f"dims={res.get('dims')} ckpt_dims={res.get('ckpt_dims')}")

    print(f"oracle_pool={res['oracle_pool']}")
    print(f"pools_used={res['pools_used']}")
    if res.get("feature_keys_used"):
        print(f"features_used: k={len(res['feature_keys_used'])}")
        if res.get("feature_stats_used"):
            print(f"feature_stats_used={res['feature_stats_used']}")

    print(f"Gate picks: {res['gate_counts']}")
    if res.get("gate_oracle_agreement") is not None:
        ga = res["gate_oracle_agreement"]
        print(f"Gate oracle agreement: acc={ga['acc']:.4f} conf={ga['conf']}")
    print(f"Chosen distribution (top): {res['pick_counts']}")
    if res.get("selector_diag") is not None:
        sd = res["selector_diag"]
        print(f"Selector diagnostics: mean_entropy={sd['mean_entropy']:.4f} | mean_maxprob={sd['mean_maxprob']:.4f} | steps={sd['steps']}")
    print(f"Two-stage chosen avg F1={res['chosen_f1']:.4f} | avg EM={res['chosen_em']:.4f} | avg U={res['chosen_u']:.4f}")
    print(f"Oracle     avg F1={res['oracle_f1']:.4f} | avg EM={res['oracle_em']:.4f} | avg U={res['oracle_u']:.4f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--model_dir", type=str, default="results/two_stage_utility")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.10)

    ap.add_argument("--oracle_policy_aligned", action="store_true")
    ap.add_argument("--tradeoff_mode", action="store_true")

    ap.add_argument("--gate_threshold", type=float, default=None)
    ap.add_argument("--sweep_gate_threshold", type=str, default=None)

    ap.add_argument("--oracle_gate", action="store_true")
    ap.add_argument("--oracle_selector", action="store_true")

    ap.add_argument("--feature_files", type=str, default=None)
    ap.add_argument("--feature_keys", type=str, default=None)
    ap.add_argument("--standardize_features", action="store_true")

    ap.add_argument("--eval_all_combined", action="store_true")

    # replace the existing --use_shared_gate argument with:
    ap.add_argument("--use_shared_gate", action="store_true", default=True,
                    help="Use shared gate for pubmedqa_v2 vs commonsenseqa (default: enabled)")
    ap.add_argument("--disable_shared_gate", action="store_false", dest="use_shared_gate",
                    help="Disable shared gate and use per-dataset gates/policies")
    # Problem (2): pubmed policy override (matches train)
    ap.add_argument("--pubmed_policy", type=str, default="none", choices=["forced", "none"],
                    help="forced: pubmedqa_v2 policy=True (always RAG). none: allow routing (policy=None).")

    # Problem (3): passage embeddings
    ap.add_argument("--use_passage_embeddings", action="store_true",
                    help="Allow passage embeddings. Required if any checkpoint expects them (in_dim includes passages).")
    ap.add_argument("--passage_source_expert", type=str, default="base_rag")
    ap.add_argument("--passage_max_docs", type=int, default=5)
    ap.add_argument("--passage_max_chars", type=int, default=1200)

    args = ap.parse_args()

    if args.oracle_gate and args.oracle_selector:
        raise SystemExit("Choose at most one of --oracle_gate or --oracle_selector (or neither).")

    cfg = load_cfg()
    model_root = Path(args.model_dir)

    if args.tradeoff_mode:
        tcfg0 = tradeoff_from_cfg(cfg)
        print("\n==================== OFFLINE TWO-STAGE EVAL (tradeoff-oracle) ====================\n")
        print(f"Tradeoff cfg: {tcfg0}")
    else:
        print("\n==================== OFFLINE TWO-STAGE EVAL ====================\n")

    thr_sweep = parse_csv_floats(args.sweep_gate_threshold)

    feature_paths: List[Path] = []
    if args.feature_files:
        feature_paths = [Path(x.strip()) for x in args.feature_files.split(",") if x.strip()]
    fmap = load_feature_map(feature_paths) if feature_paths else {}

    feature_keys_cli: Optional[List[str]] = None
    if args.feature_keys:
        feature_keys_cli = [x.strip() for x in args.feature_keys.split(",") if x.strip()]

    # load shared gate once
    shared_gate = None
    if args.use_shared_gate:
        shared_dir = model_root / SHARED_GATE_NAME
        shared_gate = load_gate(shared_dir, args.device, args.hidden_dim, args.dropout)
        if shared_gate.get("gate_objective") != "cls":
            raise SystemExit(f"[shared_gate] expected cls gate, got: {shared_gate.get('gate_objective')}")

    def eval_one_dataset(dataset: str, rows: List[Dict[str, Any]]):
        if thr_sweep is not None and (not args.oracle_gate):
            print(f"\n--- {dataset} --- sweep gate_threshold ---")
            for t in thr_sweep:
                res = run_eval_dataset(
                    cfg=cfg,
                    dataset=dataset,
                    rows=rows,
                    model_root=model_root,
                    device=args.device,
                    batch_size=args.batch_size,
                    hidden_dim=args.hidden_dim,
                    dropout=args.dropout,
                    tradeoff_mode=bool(args.tradeoff_mode),
                    gate_threshold=t,
                    oracle_policy_aligned=bool(args.oracle_policy_aligned),
                    oracle_gate=bool(args.oracle_gate),
                    oracle_selector=bool(args.oracle_selector),
                    feature_fmap=fmap,
                    feature_keys_cli=feature_keys_cli,
                    standardize_features_flag=bool(args.standardize_features),
                    use_shared_gate=bool(args.use_shared_gate),
                    shared_gate=shared_gate,
                    pubmed_policy=str(args.pubmed_policy),
                    use_passage_embeddings=bool(args.use_passage_embeddings),
                    passage_source_expert=str(args.passage_source_expert),
                    passage_max_docs=int(args.passage_max_docs),
                    passage_max_chars=int(args.passage_max_chars),
                )
                print(f"thr={t:<6.3f} | chosen_F1={res['chosen_f1']:.4f} oracle_F1={res['oracle_f1']:.4f} | gate={res['gate_counts']}")
            return

        res = run_eval_dataset(
            cfg=cfg,
            dataset=dataset,
            rows=rows,
            model_root=model_root,
            device=args.device,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            tradeoff_mode=bool(args.tradeoff_mode),
            gate_threshold=args.gate_threshold,
            oracle_policy_aligned=bool(args.oracle_policy_aligned),
            oracle_gate=bool(args.oracle_gate),
            oracle_selector=bool(args.oracle_selector),
            feature_fmap=fmap,
            feature_keys_cli=feature_keys_cli,
            standardize_features_flag=bool(args.standardize_features),
            use_shared_gate=bool(args.use_shared_gate),
            shared_gate=shared_gate,
            pubmed_policy=str(args.pubmed_policy),
            use_passage_embeddings=bool(args.use_passage_embeddings),
            passage_source_expert=str(args.passage_source_expert),
            passage_max_docs=int(args.passage_max_docs),
            passage_max_chars=int(args.passage_max_chars),
        )

        tag = ""
        if args.gate_threshold is not None:
            tag = f"(gate_threshold={args.gate_threshold:.3f})"
        else:
            if res["policy_for_family"] is None and (not args.oracle_gate):
                tag = f"(gate_threshold=auto:{res.get('gate_threshold')})"
        print_res(tag, res, args.oracle_policy_aligned, args.tradeoff_mode, args.oracle_gate, args.oracle_selector)

    if not args.eval_all_combined:
        for dataset in DATASETS:
            if args.only and dataset != args.only:
                continue
            rows = read_router_train(dataset)
            eval_one_dataset(dataset, rows)
        print("\nDONE.")
        return

    combined_N = 0
    combined_chosen_f1 = combined_chosen_em = combined_chosen_u = 0.0
    combined_oracle_f1 = combined_oracle_em = combined_oracle_u = 0.0
    per_ds_results: Dict[str, Dict[str, Any]] = {}

    for dataset in DATASETS:
        if args.only and dataset != args.only:
            continue
        rows = read_router_train(dataset)
        res = run_eval_dataset(
            cfg=cfg,
            dataset=dataset,
            rows=rows,
            model_root=model_root,
            device=args.device,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            tradeoff_mode=bool(args.tradeoff_mode),
            gate_threshold=args.gate_threshold,
            oracle_policy_aligned=bool(args.oracle_policy_aligned),
            oracle_gate=bool(args.oracle_gate),
            oracle_selector=bool(args.oracle_selector),
            feature_fmap=fmap,
            feature_keys_cli=feature_keys_cli,
            standardize_features_flag=bool(args.standardize_features),
            use_shared_gate=bool(args.use_shared_gate),
            shared_gate=shared_gate,
            pubmed_policy=str(args.pubmed_policy),
            use_passage_embeddings=bool(args.use_passage_embeddings),
            passage_source_expert=str(args.passage_source_expert),
            passage_max_docs=int(args.passage_max_docs),
            passage_max_chars=int(args.passage_max_chars),
        )
        per_ds_results[dataset] = res

        n = int(res["N"])
        combined_N += n
        combined_chosen_f1 += float(res["chosen_f1"]) * n
        combined_chosen_em += float(res["chosen_em"]) * n
        combined_chosen_u  += float(res["chosen_u"])  * n
        combined_oracle_f1 += float(res["oracle_f1"]) * n
        combined_oracle_em += float(res["oracle_em"]) * n
        combined_oracle_u  += float(res["oracle_u"])  * n

    print("\n==================== COMBINED (all datasets) ====================\n")
    if combined_N > 0:
        print(f"Combined N={combined_N}")
        print(f"Chosen avg F1={combined_chosen_f1/combined_N:.4f} | avg EM={combined_chosen_em/combined_N:.4f} | avg U={combined_chosen_u/combined_N:.4f}")
        print(f"Oracle avg F1={combined_oracle_f1/combined_N:.4f} | avg EM={combined_oracle_em/combined_N:.4f} | avg U={combined_oracle_u/combined_N:.4f}\n")

    for ds, res in per_ds_results.items():
        print_res("(combined run)", res, args.oracle_policy_aligned, args.tradeoff_mode, args.oracle_gate, args.oracle_selector)

    print("\nDONE.")


if __name__ == "__main__":
    main()