# analyze_router_decomposition.py
import json
from pathlib import Path
import numpy as np
import torch

CFG_PATH = Path("configs/router_config.json")
RESULTS_DIR = Path("results")

def read_jsonl(p: Path):
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                out.append(json.loads(line))
    return out

def get_metric(v, name):
    aliases = {
        "f1": ["f1","F1"],
        "em": ["em","exact_match","EM"],
        "loose_em": ["loose_em","looseEM","loose_exact_match","lem"],
        "latency": ["latency","time"],
        "vram_mb": ["vram_mb","peak_vram_mb"],
    }
    for k in aliases.get(name,[name]):
        if k in v:
            try: return float(v[k])
            except: return 0.0
    return 0.0

def compute_utility(metrics, util_cfg):
    f1 = get_metric(metrics, "f1")
    em = get_metric(metrics, "em")
    loose = get_metric(metrics, "loose_em")
    lat = get_metric(metrics, "latency")
    vram = get_metric(metrics, "vram_mb")

    a = float(util_cfg.get("alpha_f1", 1.0))
    b = float(util_cfg.get("beta_em", 0.5))
    g = float(util_cfg.get("gamma_loose_em", 0.0))
    lam = float(util_cfg.get("lambda_latency", 0.0))
    mu = float(util_cfg.get("mu_vram", 0.0))

    # match your current utility definition
    return (a*f1) + (b*em) + (g*loose) - (lam*np.log1p(max(0.0,lat))) - (mu*np.log1p(max(0.0,vram)))

def main():
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8-sig"))
    util_cfg = cfg.get("utility", {})

    print("\n================ ROUTER DECOMPOSITION (router vs experts) ================\n")

    for ds, ds_cfg in cfg["datasets"].items():
        train_path = Path(ds_cfg["router_train_out"])
        ckpt_path = RESULTS_DIR / f"router_{ds}_utility_ranker.pt"

        if not train_path.exists():
            print(f"[{ds}] missing {train_path}")
            continue
        if not ckpt_path.exists():
            print(f"[{ds}] missing {ckpt_path}")
            continue

        rows = read_jsonl(train_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        expert_names = ckpt["expert_names"]

        # sanity: ensure row expert keys align
        row_order = list(rows[0]["experts"].keys())
        if row_order != expert_names:
            print(f"\n[{ds}] WARNING expert order mismatch:")
            print(" JSONL:", row_order)
            print(" CKPT :", expert_names)

        # compute oracle + best-fixed baseline
        regrets = []
        match_router_oracle = 0
        oracle_utils = []
        router_utils = []

        fixed_sum = np.zeros(len(expert_names), dtype=np.float64)
        fixed_count = 0

        # also track F1/EM for router + oracle + fixed
        router_f1, router_em = [], []
        oracle_f1, oracle_em = [], []
        fixed_f1, fixed_em = [], []

        for r in rows:
            block = r["experts"]

            utils = np.array([compute_utility(block[e], util_cfg) for e in expert_names], dtype=np.float64)
            oracle_i = int(utils.argmax())
            oracle_u = float(utils[oracle_i])

            fixed_sum += utils
            fixed_count += 1

            # router pick from stored router_scores if you have it, otherwise cannot here.
            # So we approximate router pick by reusing your eval output? No — we do it properly:
            # We will load router picks by calling your router model in eval script, not here.
            # --> Instead: we read the chosen expert saved by eval (if present) OR skip.
            #
            # To keep Step 1 self-contained, we compute only Oracle + BestFixed here.
            oracle_utils.append(oracle_u)

            oracle_f1.append(get_metric(block[expert_names[oracle_i]], "f1"))
            oracle_em.append(get_metric(block[expert_names[oracle_i]], "em"))

        best_fixed_i = int(fixed_sum.argmax())
        best_fixed = expert_names[best_fixed_i]

        # compute fixed metrics
        for r in rows:
            block = r["experts"]
            fixed_f1.append(get_metric(block[best_fixed], "f1"))
            fixed_em.append(get_metric(block[best_fixed], "em"))

        print(f"\n--- {ds} ---")
        print(f"N={len(rows)}")
        print(f"Best-fixed expert: {best_fixed}")
        print(f"Best-fixed avg F1: {float(np.mean(fixed_f1)):.4f} | avg EM: {float(np.mean(fixed_em)):.4f}")
        print(f"Oracle avg F1    : {float(np.mean(oracle_f1)):.4f} | avg EM: {float(np.mean(oracle_em)):.4f}")
        print(f"Oracle avg utility: {float(np.mean(oracle_utils)):.4f}")

        print("NEXT: run eval_router_all_offline_utility.py to add Router numbers on top of this.\n")

if __name__ == "__main__":
    main()
