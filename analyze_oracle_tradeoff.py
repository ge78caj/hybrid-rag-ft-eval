import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple

CFG_PATH = Path("configs/router_config.json")
PRED_DIR = Path("prediction")

CANON_RAG_EXPERTS = ["base_rag", "sft_rag", "raft_rag"]
CANON_NO_EXPERTS = ["base_only", "sft_only"]


def load_cfg() -> Dict[str, Any]:
    if not CFG_PATH.exists():
        raise SystemExit(f"Missing {CFG_PATH}")
    return json.loads(CFG_PATH.read_text(encoding="utf-8-sig"))


def read_router_train(dataset: str) -> List[Dict[str, Any]]:
    p = PRED_DIR / f"router_train_{dataset}.jsonl"
    if not p.exists():
        raise SystemExit(f"Missing {p}")
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pools_for_dataset(dataset: str) -> Tuple[List[str], List[str]]:
    if dataset == "commonsenseqa":
        return ["base_rag"], ["base_only"]
    return CANON_RAG_EXPERTS, CANON_NO_EXPERTS


def pools_for_row(ex: Dict[str, Any], rag_pool: List[str], no_pool: List[str]) -> Tuple[List[str], List[str]]:
    keys = set(ex.keys())
    rp = [e for e in rag_pool if e in keys]
    npool = [e for e in no_pool if e in keys]
    return rp, npool


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


def parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def candidate_pool(dataset: str, row: Dict[str, Any], family: str) -> List[str]:
    rag_pool_ds, no_pool_ds = pools_for_dataset(dataset)
    ex = row["experts"]
    rag_pool, no_pool = pools_for_row(ex, rag_pool_ds, no_pool_ds)

    if family == "rag":
        return rag_pool
    if family == "no":
        return no_pool
    if family == "full":
        return rag_pool + no_pool

    raise ValueError(f"Unknown family: {family}")


def canonical_family_experts(dataset: str, family: str) -> List[str]:
    rag_pool_ds, no_pool_ds = pools_for_dataset(dataset)
    if family == "rag":
        return rag_pool_ds
    if family == "no":
        return no_pool_ds
    if family == "full":
        return rag_pool_ds + no_pool_ds
    raise ValueError(f"Unknown family: {family}")


def mean_or_zero(total: float, n: int) -> float:
    return total / n if n > 0 else 0.0


def summarize_dataset(cfg: Dict[str, Any], dataset: str, family: str, lambda_values: List[float]) -> None:
    rows = read_router_train(dataset)
    experts_all = canonical_family_experts(dataset, family)

    print(f"\n==================== {dataset} | family={family} ====================\n")
    print(f"rows={len(rows)}")

    for lam in lambda_values:
        tcfg = tradeoff_from_cfg(cfg)
        tcfg["lambda_latency"] = float(lam)

        win_counts = Counter()

        all_sum_u = defaultdict(float)
        all_sum_f1 = defaultdict(float)
        all_sum_em = defaultdict(float)
        all_sum_lat = defaultdict(float)
        all_n = Counter()

        win_sum_u = defaultdict(float)
        win_sum_f1 = defaultdict(float)
        win_sum_em = defaultdict(float)
        win_sum_lat = defaultdict(float)
        win_n = Counter()

        oracle_total_u = 0.0
        oracle_total_f1 = 0.0
        oracle_total_em = 0.0
        oracle_total_lat = 0.0
        oracle_margin_sum = 0.0
        considered = 0

        for r in rows:
            ex = r["experts"]
            pool = candidate_pool(dataset, r, family)
            if not pool:
                continue

            considered += 1

            scored = []
            for e in pool:
                out = ex[e]
                u = tradeoff_U(out, tcfg, dataset, e)
                f1 = float(out.get("f1", 0.0) or 0.0)
                em = float(out.get("em", 0.0) or 0.0)
                lat = get_latency_s(out)

                scored.append((e, u, f1, em, lat))

                all_sum_u[e] += u
                all_sum_f1[e] += f1
                all_sum_em[e] += em
                all_sum_lat[e] += lat
                all_n[e] += 1

            scored.sort(key=lambda x: x[1], reverse=True)
            best_e, best_u, best_f1, best_em, best_lat = scored[0]

            second_u = scored[1][1] if len(scored) > 1 else scored[0][1]
            oracle_margin_sum += (best_u - second_u)

            win_counts[best_e] += 1
            win_sum_u[best_e] += best_u
            win_sum_f1[best_e] += best_f1
            win_sum_em[best_e] += best_em
            win_sum_lat[best_e] += best_lat
            win_n[best_e] += 1

            oracle_total_u += best_u
            oracle_total_f1 += best_f1
            oracle_total_em += best_em
            oracle_total_lat += best_lat

        print(f"--- lambda_latency={lam:.4f} ---")
        print(
            f"considered={considered} | "
            f"oracle_avg_u={mean_or_zero(oracle_total_u, considered):.4f} | "
            f"oracle_avg_f1={mean_or_zero(oracle_total_f1, considered):.4f} | "
            f"oracle_avg_em={mean_or_zero(oracle_total_em, considered):.4f} | "
            f"oracle_avg_latency={mean_or_zero(oracle_total_lat, considered):.4f} | "
            f"oracle_margin_mean={mean_or_zero(oracle_margin_sum, considered):.4f}"
        )

        print(
            "expert".ljust(12),
            "win%".rjust(8),
            "wins".rjust(8),
            "all_U".rjust(10),
            "all_F1".rjust(10),
            "all_EM".rjust(10),
            "all_lat".rjust(10),
            "win_U".rjust(10),
            "win_F1".rjust(10),
            "win_EM".rjust(10),
            "win_lat".rjust(10),
        )

        for e in experts_all:
            if all_n[e] == 0:
                continue

            print(
                e.ljust(12),
                f"{100.0 * win_counts[e] / max(1, considered):7.2f}".rjust(8),
                f"{win_counts[e]:8d}",
                f"{mean_or_zero(all_sum_u[e], all_n[e]):10.4f}",
                f"{mean_or_zero(all_sum_f1[e], all_n[e]):10.4f}",
                f"{mean_or_zero(all_sum_em[e], all_n[e]):10.4f}",
                f"{mean_or_zero(all_sum_lat[e], all_n[e]):10.4f}",
                f"{mean_or_zero(win_sum_u[e], win_n[e]):10.4f}",
                f"{mean_or_zero(win_sum_f1[e], win_n[e]):10.4f}",
                f"{mean_or_zero(win_sum_em[e], win_n[e]):10.4f}",
                f"{mean_or_zero(win_sum_lat[e], win_n[e]):10.4f}",
            )

        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True,
                    choices=["hotpotqa", "squad_v2", "pubmedqa_v2", "commonsenseqa"])
    ap.add_argument("--family", type=str, required=True, choices=["rag", "no", "full"])
    ap.add_argument("--lambdas", type=str, default="0,0.002,0.005,0.01,0.02,0.05")
    args = ap.parse_args()

    cfg = load_cfg()
    lambda_values = parse_csv_floats(args.lambdas)
    summarize_dataset(cfg, args.dataset, args.family, lambda_values)


if __name__ == "__main__":
    main()