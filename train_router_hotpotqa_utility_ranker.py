import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit("Missing sentence-transformers. Install: pip install sentence-transformers")


CFG_PATH = Path("configs/router_config.json")
RESULTS_DIR = Path("results")
FEAT_DIM = 5


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def compute_utility(expert: str, metrics: Dict[str, Any], util_cfg: Dict[str, Any]) -> float:
    # Accuracy parts
    f1 = get_metric(metrics, "f1") or 0.0
    em = get_metric(metrics, "em") or 0.0
    loose = get_metric(metrics, "loose_em") or 0.0

    # Cost parts (optional; default 0 unless you enable in config)
    lat = get_metric(metrics, "latency") or 0.0
    vram = get_metric(metrics, "vram_mb") or 0.0

    alpha = float(util_cfg.get("alpha_f1", 1.0))
    beta = float(util_cfg.get("beta_em", 0.5))
    gamma = float(util_cfg.get("gamma_loose_em", 0.0))
    lam = float(util_cfg.get("lambda_latency", 0.0))
    mu = float(util_cfg.get("mu_vram", 0.0))

    return (alpha * f1) + (beta * em) + (gamma * loose) - (lam * lat) - (mu * vram)


def softmax_np(x: np.ndarray, tau: float) -> np.ndarray:
    z = x / max(1e-8, tau)
    z = z - np.max(z)
    ez = np.exp(z)
    return ez / np.sum(ez)


def stratified_split_by_winner(examples: List["Example"], seed: int, train_frac: float = 0.8) -> Tuple[List["Example"], List["Example"]]:
    rng = random.Random(seed)
    by_cls: Dict[int, List[Example]] = {}
    for ex in examples:
        cls = int(np.argmax(ex.expert_utils))
        by_cls.setdefault(cls, []).append(ex)

    train, val = [], []
    for cls, items in by_cls.items():
        rng.shuffle(items)
        k = int(train_frac * len(items))
        train.extend(items[:k])
        val.extend(items[k:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


@dataclass
class Example:
    q: str
    enc_embs: List[np.ndarray]       # list of embeddings, one per encoder
    feats: np.ndarray                # [FEAT_DIM]
    expert_utils: np.ndarray         # [E]


class RouterDataset(Dataset):
    def __init__(self, examples: List[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        # return numpy; collate_fn will stack properly
        return ex.enc_embs, ex.feats, ex.expert_utils


def collate_batch(batch):
    # batch: list of (enc_embs_list, feats, utils)
    n_enc = len(batch[0][0])
    encs = [[] for _ in range(n_enc)]
    feats = []
    utils = []
    for enc_list, f, u in batch:
        for i in range(n_enc):
            encs[i].append(torch.from_numpy(enc_list[i]).float())
        feats.append(torch.from_numpy(f).float())
        utils.append(torch.from_numpy(u).float())
    encs = [torch.stack(encs[i], dim=0) for i in range(n_enc)]      # each [B, dim_i]
    feats = torch.stack(feats, dim=0)                                # [B, FEAT_DIM]
    utils = torch.stack(utils, dim=0)                                # [B, E]
    return encs, feats, utils


class GatedMultiEmbedRouter(nn.Module):
    def __init__(self, embed_dims: List[int], feat_dim: int, proj_dim: int, hidden_dim: int, out_dim: int, dropout: float):
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

    def forward(self, enc_embs: List[torch.Tensor], feats: torch.Tensor) -> torch.Tensor:
        # enc_embs: list of [B, dim_i], feats: [B, feat_dim]
        proj_embs = [self.proj[i](enc_embs[i]) for i in range(self.n_enc)]  # list of [B, proj_dim]
        ref = proj_embs[0]
        gate_w = F.softmax(self.gate(torch.cat([ref, feats], dim=1)), dim=1)  # [B, n_enc]
        mixed = (torch.stack(proj_embs, dim=1) * gate_w.unsqueeze(-1)).sum(dim=1)  # [B, proj_dim]
        h = self.trunk(torch.cat([mixed, feats], dim=1))
        return self.head_experts(h)  # [B, out_dim]


def distill_loss(logits: torch.Tensor, utilities: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Utilities -> soft targets via softmax(utility/tau).
    Use KLDiv between log_softmax(logits/tau) and target probs. Multiply by tau^2.
    """
    with torch.no_grad():
        t = utilities / max(1e-8, tau)
        t = t - t.max(dim=1, keepdim=True).values
        target = torch.softmax(t, dim=1)

    logp = torch.log_softmax(logits / max(1e-8, tau), dim=1)
    return torch.nn.functional.kl_div(logp, target, reduction="batchmean") * (tau * tau)


def compute_example_weights(
        utilities: torch.Tensor,
        class_weight: torch.Tensor,
        use_margin_weighting: bool,
        margin_m0: float,
) -> torch.Tensor:
    """
    weights per example [B]
    - class_weight based on winner class frequency (provided)
    - optional margin weighting: upweight ambiguous cases (small margin between top1/top2)
    """
    B, E = utilities.shape
    winners = torch.argmax(utilities, dim=1)  # [B]
    w = class_weight[winners]                 # [B]

    if use_margin_weighting:
        top2 = torch.topk(utilities, k=2, dim=1).values  # [B,2]
        margin = (top2[:, 0] - top2[:, 1]).clamp(min=0.0)
        # small margin => harder => higher weight, bounded in [1,2]
        # w_margin = 1 + (1 - tanh(margin/m0))
        m0 = max(1e-6, float(margin_m0))
        w_margin = 1.0 + (1.0 - torch.tanh(margin / m0))
        w = w * w_margin

    return w


def train_one_dataset(dataset_name: str, ds_cfg: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    util_cfg = cfg.get("utility", {})
    tr_cfg = cfg.get("training", {})

    seed = int(tr_cfg.get("seed", 1337))
    set_seed(seed)

    router_train_path = Path(ds_cfg["router_train_out"])
    if not router_train_path.exists():
        raise SystemExit(f"[{dataset_name}] Missing {router_train_path}. Run build_router_train_all.py first.")

    rows = read_jsonl(router_train_path)
    if not rows:
        raise SystemExit(f"[{dataset_name}] No rows in {router_train_path}")

    allowed = tr_cfg.get("allowed_experts", [])
    if not allowed:
        raise SystemExit("training.allowed_experts missing in configs/router_config.json")

    # Ensure these experts exist in the file
    file_experts = list(rows[0].get("experts", {}).keys())
    expert_names = [e for e in allowed if e in file_experts]
    if len(expert_names) < 2:
        raise SystemExit(f"[{dataset_name}] Too few experts found. allowed={allowed} file={file_experts}")

    embed_names = ds_cfg.get("embed_models") or []
    if not embed_names:
        raise SystemExit(f"[{dataset_name}] configs missing embed_models list")

    embedders: List[SentenceTransformer] = []
    embed_dims: List[int] = []
    for n in embed_names:
        m = SentenceTransformer(n)
        embedders.append(m)
        # robust dim
        v = m.encode("hello", normalize_embeddings=True)
        embed_dims.append(int(np.asarray(v).shape[0]))

    # Build examples (precompute embeddings once)
    examples: List[Example] = []
    for r in rows:
        q = r.get("question") or ""
        if not q:
            continue
        block = r.get("experts", {})
        if any(e not in block for e in expert_names):
            continue

        utils = np.array([compute_utility(e, block[e], util_cfg) for e in expert_names], dtype=np.float32)
        feats = basic_text_features(q).astype(np.float32)
        encs = [np.asarray(m.encode(q, normalize_embeddings=True), dtype=np.float32) for m in embedders]

        examples.append(Example(q=q, enc_embs=encs, feats=feats, expert_utils=utils))

    if len(examples) < 50:
        raise SystemExit(f"[{dataset_name}] Too few examples: {len(examples)}")

    # Stratified split (by winner)
    train_ex, val_ex = stratified_split_by_winner(examples, seed=seed, train_frac=0.8)

    # Class weights (winner frequency)
    winners = np.array([int(np.argmax(ex.expert_utils)) for ex in train_ex], dtype=np.int64)
    counts = np.bincount(winners, minlength=len(expert_names)).astype(np.float32)
    counts[counts == 0] = 1.0
    power = float(tr_cfg.get("class_balance_power", 0.0))
    # weight per class: count^-power normalized
    cls_w = (counts ** (-power))
    cls_w = cls_w / cls_w.mean()
    class_weight = torch.from_numpy(cls_w.astype(np.float32))

    # Training hyperparams
    epochs = int(tr_cfg.get("epochs", 50))
    batch_size = int(tr_cfg.get("batch_size", 64))
    lr = float(tr_cfg.get("lr", 8e-4))
    wd = float(tr_cfg.get("weight_decay", 1e-3))
    hidden_dim = int(tr_cfg.get("hidden_dim", 256))
    dropout = float(tr_cfg.get("dropout", 0.1))
    tau = float(tr_cfg.get("tau", 0.7))
    patience = int(tr_cfg.get("patience", 12))
    min_delta = float(tr_cfg.get("min_delta", 2e-4))
    use_margin_weighting = bool(tr_cfg.get("use_margin_weighting", False))
    margin_m0 = float(tr_cfg.get("margin_m0", 0.25))

    # Model sizes
    proj_dim = 256  # fixed, stable

    train_ds = RouterDataset(train_ex)
    val_ds = RouterDataset(val_ex)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GatedMultiEmbedRouter(embed_dims, FEAT_DIM, proj_dim, hidden_dim, len(expert_names), dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_val = float("inf")
    bad = 0

    print(f"\n==================== TRAIN {dataset_name} ====================")
    print("Experts:", expert_names)
    print("Embedders:", embed_names)
    print("Embed dims:", embed_dims)
    print(f"Train={len(train_ex)}  Val={len(val_ex)}  tau={tau}  class_balance_power={power}  margin_weighting={use_margin_weighting}")

    for epoch in range(1, epochs + 1):
        model.train()
        tr_losses = []

        for encs, feats, utils in train_loader:
            encs = [t.to(device) for t in encs]
            feats = feats.to(device)
            utils = utils.to(device)

            logits = model(encs, feats)  # [B,E]

            base = distill_loss(logits, utils, tau=tau)

            # example weights
            w = compute_example_weights(utils, class_weight.to(device), use_margin_weighting, margin_m0)  # [B]
            # convert base (scalar batchmean) into per-example approx by redoing KL per example
            # pragmatic: weight by average of w in batch (stable + simple)
            loss = base * w.mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(float(loss.item()))

        model.eval()
        va_losses = []
        with torch.no_grad():
            for encs, feats, utils in val_loader:
                encs = [t.to(device) for t in encs]
                feats = feats.to(device)
                utils = utils.to(device)
                logits = model(encs, feats)
                base = distill_loss(logits, utils, tau=tau)
                w = compute_example_weights(utils, class_weight.to(device), use_margin_weighting, margin_m0)
                loss = base * w.mean()
                va_losses.append(float(loss.item()))

        tr = float(np.mean(tr_losses)) if tr_losses else 0.0
        va = float(np.mean(va_losses)) if va_losses else 0.0
        print(f"epoch {epoch:03d} | train {tr:.4f} | val {va:.4f}")

        if va < best_val - min_delta:
            best_val = va
            bad = 0
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            ckpt_path = RESULTS_DIR / f"router_{dataset_name}_utility_ranker.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "expert_names": expert_names,
                    "out_dim": len(expert_names),
                    "feat_dim": FEAT_DIM,
                    "embed_models_requested": embed_names,
                    "embed_models_used": embed_names,
                    "embed_dims": embed_dims,
                    "proj_dim": proj_dim,
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "utility_cfg": util_cfg,
                    "training_cfg": tr_cfg,
                },
                ckpt_path,
            )
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    print(f"[OK] Saved best checkpoint: results/router_{dataset_name}_utility_ranker.pt")


def main():
    if not CFG_PATH.exists():
        raise SystemExit(f"Missing {CFG_PATH}")

    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8-sig"))

    datasets = cfg.get("datasets", {})
    if not datasets:
        raise SystemExit("router_config.json missing 'datasets'")

    for dataset_name, ds_cfg in datasets.items():
        train_one_dataset(dataset_name, ds_cfg, cfg)


if __name__ == "__main__":
    main()
