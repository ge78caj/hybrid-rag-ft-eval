# find_best_fixed_expert.py

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

CFG_PATH = Path("configs/router_config.json")
PRED_DIR = Path("prediction")
DATASETS = ["hotpotqa", "squad_v2", "pubmedqa_v2", "commonsenseqa"]


def load_cfg() -> Dict[str, Any]:
    if not CFG_PATH.exists():
        raise SystemExit(f"Missing {CFG_PATH}")
    return json.loads(CFG_PATH.read_text(encoding="utf-8-sig"))


def read_router_train(dataset: str) -> List[Dict[str, Any]]:
    p = PRED_DIR / f"router_train_{dataset}.jsonl"
    if not p.exists():
        raise SystemExit(f"Missing router train file: {p}")
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def _get_latency_cap_seconds(tcfg: Dict[str, Any], dataset: str, expert: str) -> float:
    caps = tcfg.get("latency_caps") or {}
    default_cap = float(caps.get("default", 3.0))
    by_dataset = (caps.get("by_dataset") or {})
    ds_cfg = by_dataset.get(dataset) or {}
    cap = float(ds_cfg.get("default", default_cap))
    if expert in ds_cfg:
        cap = float(ds_cfg[expert])
    return cap


def tradeoff_U(outcome: Dict[str, Any], tcfg: Dict[str, Any], dataset: str, expert: str) -> float:
    f1 = float(outcome.get("f1", 0.0) or 0.0)
    em = float(outcome.get("em", 0.0) or 0.0)
    loose = float(outcome.get("loose_em", em) or em)

    Q = (
            tcfg["alpha_f1"] * f1
            + tcfg["beta_em"] * em
            + tcfg["gamma_loose_em"] * loose
    )

    L = get_latency_s(outcome)
    V = get_vram_gb(outcome)

    cap = _get_latency_cap_seconds(tcfg, dataset, expert)
    lat_ratio = L / max(1e-8, cap)
    lat_pen = tcfg["lambda_latency"] * (lat_ratio if lat_ratio <= 1.0 else (lat_ratio ** 2))
    vram_pen = tcfg["mu_vram"] * V

    return float(Q - lat_pen - vram_pen)


def score_of(outcome: Dict[str, Any], dataset: str, expert: str, metric: str, tcfg: Dict[str, Any]) -> float:
    if metric == "utility":
        return tradeoff_U(outcome, tcfg, dataset, expert)
    if metric == "f1":
        return float(outcome.get("f1", 0.0) or 0.0)
    if metric == "em":
        return float(outcome.get("em", 0.0) or 0.0)
    raise ValueError(metric)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", type=str, default="utility", choices=["utility", "f1", "em"])
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    args = ap.parse_args()

    cfg = load_cfg()
    tcfg = tradeoff_from_cfg(cfg)

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]

    for dataset in datasets:
        rows = read_router_train(dataset)
        n = len(rows)

        # only keep experts that exist on every row
        expert_counts = {}
        for r in rows:
            for e in r["experts"].keys():
                expert_counts[e] = expert_counts.get(e, 0) + 1

        valid_experts = sorted([e for e, c in expert_counts.items() if c == n])
        if not valid_experts:
            print(f"\n=== {dataset} ===")
            print("No expert is available on every example.")
            continue

        results = []
        oracle_sum = 0.0

        for r in rows:
            ex = r["experts"]
            best = max(score_of(ex[e], dataset, e, args.metric, tcfg) for e in ex.keys())
            oracle_sum += best

        for e in valid_experts:
            f1_sum = 0.0
            em_sum = 0.0
            u_sum = 0.0
            score_sum = 0.0

            for r in rows:
                out = r["experts"][e]
                f1_sum += float(out.get("f1", 0.0) or 0.0)
                em_sum += float(out.get("em", 0.0) or 0.0)
                u_sum += tradeoff_U(out, tcfg, dataset, e)
                score_sum += score_of(out, dataset, e, args.metric, tcfg)

            results.append({
                "expert": e,
                "avg_f1": f1_sum / n,
                "avg_em": em_sum / n,
                "avg_u": u_sum / n,
                "avg_score": score_sum / n,
            })

        results.sort(key=lambda x: x["avg_score"], reverse=True)
        best = results[0]

        print(f"\n=== {dataset} ===")
        print(f"N={n} | metric={args.metric}")
        print(
            f"BEST FIXED EXPERT: {best['expert']} | "
            f"avg_f1={best['avg_f1']:.4f} | "
            f"avg_em={best['avg_em']:.4f} | "
            f"avg_u={best['avg_u']:.4f} | "
            f"avg_{args.metric}={best['avg_score']:.4f}"
        )
        print(f"Oracle avg_{args.metric}={oracle_sum / n:.4f}")
        print("Ranking:")
        for i, r in enumerate(results, start=1):
            print(
                f"  {i}. {r['expert']:<10} "
                f"avg_f1={r['avg_f1']:.4f} | "
                f"avg_em={r['avg_em']:.4f} | "
                f"avg_u={r['avg_u']:.4f}"
            )


if __name__ == "__main__":
    main()


