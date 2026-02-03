import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sentence_transformers import SentenceTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


@dataclass
class UtilityWeights:
    alpha_f1: float
    beta_em: float
    gamma_loose_em: float
    lambda_latency: float
    mu_vram: float


def compute_utility(expert_row: Dict[str, Any], w: UtilityWeights) -> float:
    f1 = float(expert_row.get("f1", 0.0))
    em = float(expert_row.get("em", 0.0))
    loose_em = float(expert_row.get("loose_em", 0.0))
    lat = float(expert_row.get("latency", 0.0))
    vram = float(expert_row.get("vram_mb", 0.0))
    return (
            w.alpha_f1 * f1
            + w.beta_em * em
            + w.gamma_loose_em * loose_em
            - w.lambda_latency * lat
            - w.mu_vram * vram
    )


class RouterNet(nn.Module):
    """
    Dataset-aware utility scorer:
      input = [question_embedding, simple_features, dataset_onehot]
      output = score per expert (K)
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RouterTrainDataset(Dataset):
    def __init__(
            self,
            rows: List[Dict[str, Any]],
            dataset_to_idx: Dict[str, int],
            embed_by_dataset: Dict[str, SentenceTransformer],
            expert_names: List[str],
            util_w: UtilityWeights,
    ):
        self.rows = rows
        self.dataset_to_idx = dataset_to_idx
        self.embed_by_dataset = embed_by_dataset
        self.expert_names = expert_names
        self.util_w = util_w

        # Precompute embeddings for speed
        self.embeddings: List[np.ndarray] = []
        self.features: List[np.ndarray] = []
        self.targets: List[np.ndarray] = []
        self._prepare()

    def _simple_text_features(self, q: str) -> np.ndarray:
        q = q.strip()
        n_chars = len(q)
        n_words = len(q.split())
        has_qmark = 1.0 if "?" in q else 0.0
        lower = q.lower()
        yn = 1.0 if (lower.startswith("is ") or lower.startswith("are ") or lower.startswith("was ") or lower.startswith("were ") or lower.startswith("do ") or lower.startswith("does ")) else 0.0
        wh = 1.0 if (lower.startswith("what") or lower.startswith("who") or lower.startswith("when") or lower.startswith("where") or lower.startswith("why") or lower.startswith("how")) else 0.0
        return np.array([n_chars, n_words, has_qmark, yn, wh], dtype=np.float32)

    def _dataset_onehot(self, dataset: str) -> np.ndarray:
        k = len(self.dataset_to_idx)
        v = np.zeros((k,), dtype=np.float32)
        v[self.dataset_to_idx[dataset]] = 1.0
        return v

    def _prepare(self) -> None:
        for r in tqdm(self.rows, desc="prepare_router_dataset"):
            ds = str(r["dataset"])
            q = str(r["question"])

            # embedding
            embedder = self.embed_by_dataset[ds]
            emb = embedder.encode([q], normalize_embeddings=True)[0].astype(np.float32)

            # features
            tf = self._simple_text_features(q)
            onehot = self._dataset_onehot(ds)
            x = np.concatenate([emb, tf, onehot], axis=0)

            # targets = per-expert utilities
            experts = r["experts"]
            y = []
            for en in self.expert_names:
                if en not in experts:
                    y.append(0.0)
                else:
                    y.append(compute_utility(experts[en], self.util_w))
            y = np.array(y, dtype=np.float32)

            self.features.append(x)
            self.targets.append(y)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.features[idx]), torch.from_numpy(self.targets[idx])


def main():
    cfg = json.loads(Path("configs/router_config.json").read_text(encoding="utf-8"))

    util_w = UtilityWeights(**cfg["utility"])
    train_cfg = cfg["training"]

    set_seed(int(train_cfg["seed"]))

    # Load all router_train files across datasets
    all_rows: List[Dict[str, Any]] = []
    dataset_names = list(cfg["datasets"].keys())
    for ds in dataset_names:
        p = Path(cfg["datasets"][ds]["router_train_out"])
        if not p.exists():
            raise RuntimeError(f"Missing router train file: {p} (run build_router_train_all.py)")
        all_rows.extend(read_jsonl(p))

    # Determine expert set (union across datasets)
    expert_set = set()
    for r in all_rows:
        expert_set.update(r["experts"].keys())
    expert_names = sorted(expert_set)
    print("[INFO] Experts:", expert_names)

    dataset_to_idx = {ds: i for i, ds in enumerate(dataset_names)}

    # Load embedders (dataset-specific)
    embed_by_dataset: Dict[str, SentenceTransformer] = {}
    for ds in dataset_names:
        model_name = cfg["datasets"][ds]["embed_model"]
        print(f"[INFO] Loading embedder for {ds}: {model_name}")
        embed_by_dataset[ds] = SentenceTransformer(model_name)

    # Build dataset
    ds_train = RouterTrainDataset(
        rows=all_rows,
        dataset_to_idx=dataset_to_idx,
        embed_by_dataset=embed_by_dataset,
        expert_names=expert_names,
        util_w=util_w,
    )

    loader = DataLoader(ds_train, batch_size=int(train_cfg["batch_size"]), shuffle=True)

    in_dim = ds_train[0][0].shape[0]
    out_dim = len(expert_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RouterNet(
        in_dim=in_dim,
        hidden_dim=int(train_cfg["hidden_dim"]),
        out_dim=out_dim,
        dropout=float(train_cfg["dropout"]),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]))
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(int(train_cfg["epochs"])):
        total = 0.0
        n = 0
        for x, y in tqdm(loader, desc=f"train_epoch[{epoch+1}/{train_cfg['epochs']}]"):
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item()) * x.size(0)
            n += x.size(0)

        print(f"[EPOCH {epoch+1}] loss={total/max(n,1):.6f}")

    # Save model + metadata needed for inference
    out_path = Path(cfg["model_out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": in_dim,
            "hidden_dim": int(train_cfg["hidden_dim"]),
            "dropout": float(train_cfg["dropout"]),
            "expert_names": expert_names,
            "dataset_names": dataset_names,
            "dataset_to_idx": dataset_to_idx,
            "embed_models": {ds: cfg["datasets"][ds]["embed_model"] for ds in dataset_names},
            "utility": cfg["utility"],
        },
        out_path,
    )

    print(f"[OK] Saved router model -> {out_path}")


if __name__ == "__main__":
    main()
