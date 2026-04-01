# tmp_oracle_given_gate.py
import json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

CFG_PATH = Path("configs/router_config.json")
PRED_DIR = Path("prediction")

RAG = ["base_rag","sft_rag","raft_rag"]
NO  = ["base_only","sft_only"]
ALL = RAG + NO

def load_cfg():
    return json.loads(CFG_PATH.read_text(encoding="utf-8-sig"))

def latency_cap_seconds(cfg, dataset, expert):
    u = cfg.get("utility", {}) or {}
    caps = u.get("latency_caps_seconds", {}) or {}
    default_cap = float(caps.get("default", 3.0))
    by_ds = (caps.get("by_dataset", {}) or {}).get(dataset, {}) or {}
    return float(by_ds.get(expert, by_ds.get("default", default_cap)))

def utility_value(cfg, dataset, expert, outcome):
    u = cfg.get("utility", {}) or {}
    a = float(u.get("alpha_f1", 1.0))
    b = float(u.get("beta_em", 0.0))
    g = float(u.get("gamma_loose_em", 0.0))
    lam = float(u.get("lambda_latency", 0.0))
    mu = float(u.get("mu_vram", 0.0))

    f1 = float(outcome.get("f1", 0.0) or 0.0)
    em = float(outcome.get("em", 0.0) or 0.0)
    loose = float(outcome.get("loose_em", em) or em)

    lat = float(outcome.get("latency", 0.0) or 0.0)
    lat = min(lat, latency_cap_seconds(cfg, dataset, expert))

    vram = float(outcome.get("vram_mb", 0.0) or 0.0)
    return a*f1 + b*em + g*loose - lam*lat - mu*vram

def read_rows(ds):
    p = PRED_DIR / f"router_train_{ds}.jsonl"
    return [json.loads(l) for l in p.open("r",encoding="utf-8") if l.strip()]

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, dropout, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)

def load_gate(model_dir, device, hidden=256, dropout=0.1):
    ckpt = torch.load(model_dir / "gate.pt", map_location="cpu", weights_only=False)
    if "forced_policy" in ckpt:
        return {"forced": ckpt["forced_policy"], "embed_model": ckpt.get("embed_model"), "model": None}
    m = MLP(int(ckpt["in_dim"]), hidden, dropout, 2)
    m.load_state_dict(ckpt["state_dict"])
    m.to(device).eval()
    return {"forced": None, "embed_model": ckpt.get("embed_model"), "model": m}

def policy_for_dataset(ds):
    if ds == "pubmedqa_v2":
        return False
    return None

def best_in(cfg, ds, ex, pool):
    best_e, best_u = None, -1e18
    for e in pool:
        u = utility_value(cfg, ds, e, ex[e])
        if u > best_u:
            best_u, best_e = u, e
    return best_e, best_u

def main(model_root="results/two_stage_utility_v3_balanced", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg()

    for ds in ["hotpotqa","squad_v2","pubmedqa_v2"]:
        rows = read_rows(ds)
        pol = policy_for_dataset(ds)
        model_dir = Path(model_root) / ds
        gate = load_gate(model_dir, device)
        embed_model = gate["embed_model"] or "sentence-transformers/all-mpnet-base-v2"
        enc = SentenceTransformer(embed_model, device=device)

        questions = [r["question"] for r in rows]
        X = enc.encode(questions, batch_size=64, convert_to_tensor=True, normalize_embeddings=True)

        oracle_all = []
        oracle_given_gate = []
        gate_counts = Counter()

        for i, r in enumerate(rows):
            ex = r["experts"]
            # full oracle
            _, u_all = best_in(cfg, ds, ex, ALL)
            oracle_all.append(u_all)

            # gate decision
            if pol is True:
                use_rag = True
            elif pol is False:
                use_rag = False
            else:
                with torch.no_grad():
                    pred = int(gate["model"](X[i].unsqueeze(0)).argmax(dim=1).item())
                use_rag = (pred == 1)

            gate_counts["rag" if use_rag else "no"] += 1
            pool = RAG if use_rag else NO
            _, u_g = best_in(cfg, ds, ex, pool)
            oracle_given_gate.append(u_g)

        print(f"\n=== {ds} ===")
        print("gate picks:", dict(gate_counts))
        print("oracle_all avg util:", float(np.mean(oracle_all)))
        print("oracle_given_gate avg util:", float(np.mean(oracle_given_gate)))
        print("gap (all - given_gate):", float(np.mean(oracle_all) - np.mean(oracle_given_gate)))

if __name__ == "__main__":
    main()
