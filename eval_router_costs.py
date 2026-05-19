import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from train_router_two_stage import (
    DATASETS,
    Embedder,
    MLP,
    build_feature_matrix,
    build_passage_embedding_matrix,
    load_cfg,
    load_feature_map,
    pools_for_dataset,
    pools_for_row,
    read_router_train,
    score_for_targets,
    tradeoff_from_cfg,
)


def get_latency_s(outcome: Dict[str, Any]) -> float:
    return float(outcome.get("latency", outcome.get("time", 0.0)) or 0.0)


def get_vram_gb(outcome: Dict[str, Any]) -> float:
    mb = float(outcome.get("vram_mb", outcome.get("peak_vram_mb", 0.0)) or 0.0)
    return mb / 1024.0


def apply_feature_stats(Xf: torch.Tensor, stats: Optional[Dict[str, Any]]) -> torch.Tensor:
    if stats is None or Xf.numel() == 0 or Xf.size(1) == 0:
        return Xf
    mean = torch.tensor(stats.get("mean", []), dtype=torch.float32)
    std = torch.tensor(stats.get("std", []), dtype=torch.float32)
    if mean.numel() == 0 or std.numel() == 0:
        return Xf
    if mean.numel() != Xf.size(1) or std.numel() != Xf.size(1):
        raise ValueError(
            f"Feature stats dimension mismatch: Xf has {Xf.size(1)} cols, "
            f"mean has {mean.numel()}, std has {std.numel()}"
        )
    std = std.clamp_min(1e-8)
    return (Xf - mean.unsqueeze(0)) / std.unsqueeze(0)


def build_router_input(
        rows: List[Dict[str, Any]],
        fmap: Dict[str, Dict[str, float]],
        ckpt: Dict[str, Any],
        *,
        device: str,
        batch_size: int,
        embedder_cache: Dict[str, Embedder],
) -> torch.Tensor:
    embed_model = str(ckpt["embed_model"])
    if embed_model not in embedder_cache:
        embedder_cache[embed_model] = Embedder(embed_model, device=device)
    embedder = embedder_cache[embed_model]

    questions = [r["question"] for r in rows]
    Xq = embedder.encode(questions, batch_size=batch_size).float().cpu()

    feature_keys = ckpt.get("feature_keys") or []
    if feature_keys:
        Xf = build_feature_matrix(rows, fmap, feature_keys)
        Xf = apply_feature_stats(Xf, ckpt.get("feature_stats"))
        X = torch.cat([Xq, Xf], dim=1)
    else:
        X = Xq

    if ckpt.get("used_passage_embeddings", False):
        Xp = build_passage_embedding_matrix(
            rows,
            embedder,
            source_expert=str(ckpt.get("passage_source_expert", "base_rag")),
            max_docs=int(ckpt.get("passage_max_docs", 5)),
            max_chars=int(ckpt.get("passage_max_chars", 1200)),
            batch_size_docs=batch_size,
        )
        X = torch.cat([X, Xp], dim=1)

    expected_dim = int(ckpt["in_dim"])
    if int(X.size(1)) != expected_dim:
        raise ValueError(
            f"Input dim mismatch for checkpoint: built {int(X.size(1))}, expected {expected_dim}"
        )
    return X


class GateRunner:
    def __init__(self, ckpt: Dict[str, Any], *, hidden_dim: int, dropout: float, device: str):
        self.ckpt = ckpt
        self.device = device
        self.kind = "forced" if "forced_policy" in ckpt else str(ckpt.get("gate_objective", "cls"))

        if self.kind == "forced":
            self.forced_policy = bool(ckpt["forced_policy"])
            self.model = None
            self.threshold = 0.5
        elif self.kind == "cls":
            self.model = MLP(int(ckpt["in_dim"]), hidden_dim, dropout, out_dim=2).to(device)
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
            self.threshold = 0.5
        elif self.kind == "delta_reg":
            self.model = MLP(int(ckpt["in_dim"]), hidden_dim, dropout, out_dim=1).to(device)
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
            self.threshold = float(ckpt.get("calibrated_threshold", 0.0))
        else:
            raise ValueError(f"Unsupported gate objective: {self.kind}")

    @torch.no_grad()
    def predict(self, X: torch.Tensor, *, batch_size: int) -> List[bool]:
        if self.kind == "forced":
            return [self.forced_policy for _ in range(X.size(0))]

        out: List[bool] = []
        for start in range(0, X.size(0), batch_size):
            xb = X[start:start + batch_size].to(self.device)
            if self.kind == "cls":
                pred = self.model(xb).argmax(dim=1).detach().cpu().tolist()
                out.extend([bool(p == 1) for p in pred])  # 1 => rag
            else:
                pred = self.model(xb).squeeze(1).detach().cpu().numpy()
                out.extend([bool(v > self.threshold) for v in pred])
        return out


class SelectorRunner:
    def __init__(self, ckpt: Dict[str, Any], *, hidden_dim: int, dropout: float, device: str):
        self.ckpt = ckpt
        self.device = device
        self.experts = list(ckpt["experts"])
        self.use_constant = bool(ckpt.get("use_constant_selector", False))
        self.constant_expert = str(ckpt.get("fallback_best_constant_expert", self.experts[0]))
        self.is_hier = bool(ckpt.get("is_hierarchical", False))

        self.model = None
        self.stage1 = None
        self.stage2 = None

        if self.use_constant:
            return

        if self.is_hier:
            self.stage1 = MLP(int(ckpt["in_dim"]), hidden_dim, dropout, out_dim=2).to(device)
            self.stage2 = MLP(int(ckpt["in_dim"]), hidden_dim, dropout, out_dim=2).to(device)
            self.stage1.load_state_dict(ckpt["stage1_state_dict"])
            self.stage2.load_state_dict(ckpt["stage2_state_dict"])
            self.stage1.eval()
            self.stage2.eval()
        else:
            out_dim = len(self.experts)
            self.model = MLP(int(ckpt["in_dim"]), hidden_dim, dropout, out_dim=out_dim).to(device)
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()

    @torch.no_grad()
    def predict(self, X: torch.Tensor, *, batch_size: int) -> List[str]:
        if self.use_constant:
            return [self.constant_expert for _ in range(X.size(0))]

        if self.is_hier:
            base_idx = self.experts.index("base_rag")
            sft_idx = self.experts.index("sft_rag")
            raft_idx = self.experts.index("raft_rag")

            preds: List[str] = []
            for start in range(0, X.size(0), batch_size):
                xb = X[start:start + batch_size].to(self.device)

                logits1 = self.stage1(xb)
                pred1 = torch.argmax(logits1, dim=1)  # 0=base_rag, 1=other_rag

                batch_pred = torch.full((xb.size(0),), base_idx, dtype=torch.long, device=self.device)
                other_mask = (pred1 == 1)
                if other_mask.any():
                    logits2 = self.stage2(xb[other_mask])
                    pred2 = torch.argmax(logits2, dim=1)  # 0=sft_rag, 1=raft_rag
                    mapped = torch.where(
                        pred2 == 0,
                        torch.full_like(pred2, sft_idx),
                        torch.full_like(pred2, raft_idx),
                        )
                    batch_pred[other_mask] = mapped

                preds.extend([self.experts[i] for i in batch_pred.detach().cpu().tolist()])
            return preds

        preds: List[str] = []
        for start in range(0, X.size(0), batch_size):
            xb = X[start:start + batch_size].to(self.device)
            logits = self.model(xb)
            idx = torch.argmax(logits, dim=1).detach().cpu().tolist()
            preds.extend([self.experts[i] for i in idx])
        return preds


def best_expert_and_family(
        cfg: Dict[str, Any],
        dataset: str,
        ex: Dict[str, Any],
        rag_pool: List[str],
        no_pool: List[str],
        *,
        use_tradeoff: bool,
        tcfg: Dict[str, Any],
) -> Tuple[str, bool]:
    candidates = rag_pool + no_pool
    if not candidates:
        raise ValueError(f"No experts available for dataset={dataset}")

    best_e = None
    best_u = -1e18
    for e in candidates:
        u = score_for_targets(cfg, dataset, e, ex[e], use_tradeoff=use_tradeoff, tcfg=tcfg)
        if u > best_u:
            best_u = u
            best_e = e

    assert best_e is not None
    return best_e, (best_e in rag_pool)


def best_in_family(
        cfg: Dict[str, Any],
        dataset: str,
        ex: Dict[str, Any],
        pool: List[str],
        *,
        use_tradeoff: bool,
        tcfg: Dict[str, Any],
) -> str:
    best_e = None
    best_u = -1e18
    for e in pool:
        u = score_for_targets(cfg, dataset, e, ex[e], use_tradeoff=use_tradeoff, tcfg=tcfg)
        if u > best_u:
            best_u = u
            best_e = e
    if best_e is None:
        raise ValueError(f"Empty family pool for dataset={dataset}")
    return best_e


def family_oracle_label(
        cfg: Dict[str, Any],
        dataset: str,
        ex: Dict[str, Any],
        rag_pool: List[str],
        no_pool: List[str],
        *,
        use_tradeoff: bool,
        tcfg: Dict[str, Any],
) -> bool:
    rag_best = -1e18
    for e in rag_pool:
        rag_best = max(rag_best, score_for_targets(cfg, dataset, e, ex[e], use_tradeoff=use_tradeoff, tcfg=tcfg))

    no_best = -1e18
    for e in no_pool:
        no_best = max(no_best, score_for_targets(cfg, dataset, e, ex[e], use_tradeoff=use_tradeoff, tcfg=tcfg))

    if len(rag_pool) == 0:
        return False
    if len(no_pool) == 0:
        return True
    return bool(rag_best > no_best)


def eval_dataset(
        dataset: str,
        *,
        cfg: Dict[str, Any],
        tcfg: Dict[str, Any],
        model_root: Path,
        fmap: Dict[str, Dict[str, float]],
        device: str,
        hidden_dim: int,
        dropout: float,
        batch_size: int,
        use_tradeoff: bool,
        oracle_gate: bool,
        oracle_selector: bool,
        embedder_cache: Dict[str, Embedder],
) -> Dict[str, Any]:
    rows = read_router_train(dataset)

    gate_ckpt = torch.load(model_root / dataset / "gate.pt", map_location="cpu")
    sel_rag_ckpt = torch.load(model_root / dataset / "selector_rag.pt", map_location="cpu")
    sel_no_ckpt = torch.load(model_root / dataset / "selector_no_rag.pt", map_location="cpu")

    X_gate = build_router_input(rows, fmap, gate_ckpt, device=device, batch_size=batch_size, embedder_cache=embedder_cache)
    X_sel_rag = build_router_input(rows, fmap, sel_rag_ckpt, device=device, batch_size=batch_size, embedder_cache=embedder_cache)
    X_sel_no = build_router_input(rows, fmap, sel_no_ckpt, device=device, batch_size=batch_size, embedder_cache=embedder_cache)

    gate_runner = GateRunner(gate_ckpt, hidden_dim=hidden_dim, dropout=dropout, device=device)
    sel_rag_runner = SelectorRunner(sel_rag_ckpt, hidden_dim=hidden_dim, dropout=dropout, device=device)
    sel_no_runner = SelectorRunner(sel_no_ckpt, hidden_dim=hidden_dim, dropout=dropout, device=device)

    learned_gate_preds = gate_runner.predict(X_gate, batch_size=batch_size)
    learned_rag_preds = sel_rag_runner.predict(X_sel_rag, batch_size=batch_size)
    learned_no_preds = sel_no_runner.predict(X_sel_no, batch_size=batch_size)

    rag_pool_ds, no_pool_ds = pools_for_dataset(dataset)

    chosen_f1 = chosen_em = chosen_u = 0.0
    chosen_latency = chosen_vram = 0.0

    oracle_f1 = oracle_em = oracle_u = 0.0
    oracle_latency = oracle_vram = 0.0

    gate_match = 0
    conf = {"TN": 0, "TP": 0, "FP": 0, "FN": 0}
    chosen_counter = Counter()

    for i, row in enumerate(rows):
        ex = row["experts"]
        rag_pool, no_pool = pools_for_row(ex, rag_pool_ds, no_pool_ds)

        oracle_best_expert, oracle_best_is_rag = best_expert_and_family(
            cfg, dataset, ex, rag_pool, no_pool, use_tradeoff=use_tradeoff, tcfg=tcfg
        )
        gate_oracle_is_rag = family_oracle_label(
            cfg, dataset, ex, rag_pool, no_pool, use_tradeoff=use_tradeoff, tcfg=tcfg
        )

        learned_gate_is_rag = learned_gate_preds[i]
        if learned_gate_is_rag == gate_oracle_is_rag:
            gate_match += 1
        if (not gate_oracle_is_rag) and (not learned_gate_is_rag):
            conf["TN"] += 1
        elif gate_oracle_is_rag and learned_gate_is_rag:
            conf["TP"] += 1
        elif (not gate_oracle_is_rag) and learned_gate_is_rag:
            conf["FP"] += 1
        else:
            conf["FN"] += 1

        final_gate_is_rag = gate_oracle_is_rag if oracle_gate else learned_gate_is_rag

        if final_gate_is_rag:
            if oracle_selector:
                chosen_expert = best_in_family(
                    cfg, dataset, ex, rag_pool, use_tradeoff=use_tradeoff, tcfg=tcfg
                )
            else:
                chosen_expert = learned_rag_preds[i]
        else:
            if oracle_selector:
                chosen_expert = best_in_family(
                    cfg, dataset, ex, no_pool, use_tradeoff=use_tradeoff, tcfg=tcfg
                )
            else:
                chosen_expert = learned_no_preds[i]

        chosen_outcome = ex[chosen_expert]
        oracle_outcome = ex[oracle_best_expert]

        chosen_f1 += float(chosen_outcome.get("f1", 0.0) or 0.0)
        chosen_em += float(chosen_outcome.get("em", 0.0) or 0.0)
        chosen_u += score_for_targets(
            cfg, dataset, chosen_expert, chosen_outcome, use_tradeoff=use_tradeoff, tcfg=tcfg
        )
        chosen_latency += get_latency_s(chosen_outcome)
        chosen_vram += get_vram_gb(chosen_outcome)

        oracle_f1 += float(oracle_outcome.get("f1", 0.0) or 0.0)
        oracle_em += float(oracle_outcome.get("em", 0.0) or 0.0)
        oracle_u += score_for_targets(
            cfg, dataset, oracle_best_expert, oracle_outcome, use_tradeoff=use_tradeoff, tcfg=tcfg
        )
        oracle_latency += get_latency_s(oracle_outcome)
        oracle_vram += get_vram_gb(oracle_outcome)

        chosen_counter[chosen_expert] += 1

    n = len(rows)
    res = {
        "dataset": dataset,
        "n": n,
        "gate_acc": gate_match / max(1, n),
        "conf": conf,
        "chosen_distribution": dict(chosen_counter.most_common()),
        "avg_f1": chosen_f1 / max(1, n),
        "avg_em": chosen_em / max(1, n),
        "avg_u": chosen_u / max(1, n),
        "avg_latency": chosen_latency / max(1, n),
        "avg_vram": chosen_vram / max(1, n),
        "oracle_avg_f1": oracle_f1 / max(1, n),
        "oracle_avg_em": oracle_em / max(1, n),
        "oracle_avg_u": oracle_u / max(1, n),
        "oracle_avg_latency": oracle_latency / max(1, n),
        "oracle_avg_vram": oracle_vram / max(1, n),
    }
    return res


def print_result(res: Dict[str, Any], *, oracle_gate: bool, oracle_selector: bool) -> None:
    print()
    print(
        f"--- {res['dataset']} --- "
        f"N={res['n']} oracle_gate={oracle_gate} oracle_selector={oracle_selector}"
    )
    print(
        f"Gate oracle agreement: acc={res['gate_acc']:.4f} "
        f"conf={res['conf']}"
    )
    print(f"Chosen distribution (top): {list(res['chosen_distribution'].items())[:10]}")
    print(
        f"Two-stage chosen avg F1={res['avg_f1']:.4f} | avg EM={res['avg_em']:.4f} | avg U={res['avg_u']:.4f} "
        f"| avg Latency={res['avg_latency']:.4f}s | avg VRAM={res['avg_vram']:.4f}GB"
    )
    print(
        f"Oracle     avg F1={res['oracle_avg_f1']:.4f} | avg EM={res['oracle_avg_em']:.4f} | avg U={res['oracle_avg_u']:.4f} "
        f"| avg Latency={res['oracle_avg_latency']:.4f}s | avg VRAM={res['oracle_avg_vram']:.4f}GB"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--feature_files", type=str, default=None)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--tradeoff_mode", action="store_true")
    ap.add_argument("--oracle_gate", action="store_true")
    ap.add_argument("--oracle_selector", action="store_true")
    ap.add_argument("--eval_all_combined", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg()
    tcfg = tradeoff_from_cfg(cfg)

    feature_paths: List[Path] = []
    if args.feature_files:
        feature_paths = [Path(x.strip()) for x in args.feature_files.split(",") if x.strip()]
    fmap = load_feature_map(feature_paths) if feature_paths else {}

    model_root = Path(args.model_dir)
    embedder_cache: Dict[str, Embedder] = {}

    datasets = DATASETS if args.eval_all_combined else [args.only]
    if not datasets or datasets == [None]:
        raise SystemExit("Pass --only <dataset> or use --eval_all_combined")

    all_results: List[Dict[str, Any]] = []
    for dataset in datasets:
        res = eval_dataset(
            dataset,
            cfg=cfg,
            tcfg=tcfg,
            model_root=model_root,
            fmap=fmap,
            device=args.device,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            batch_size=args.batch_size,
            use_tradeoff=bool(args.tradeoff_mode),
            oracle_gate=bool(args.oracle_gate),
            oracle_selector=bool(args.oracle_selector),
            embedder_cache=embedder_cache,
        )
        all_results.append(res)
        print_result(res, oracle_gate=bool(args.oracle_gate), oracle_selector=bool(args.oracle_selector))

    if args.eval_all_combined:
        total_n = sum(r["n"] for r in all_results)
        combined = {
            "avg_f1": sum(r["avg_f1"] * r["n"] for r in all_results) / total_n,
            "avg_em": sum(r["avg_em"] * r["n"] for r in all_results) / total_n,
            "avg_u": sum(r["avg_u"] * r["n"] for r in all_results) / total_n,
            "avg_latency": sum(r["avg_latency"] * r["n"] for r in all_results) / total_n,
            "avg_vram": sum(r["avg_vram"] * r["n"] for r in all_results) / total_n,
            "oracle_avg_f1": sum(r["oracle_avg_f1"] * r["n"] for r in all_results) / total_n,
            "oracle_avg_em": sum(r["oracle_avg_em"] * r["n"] for r in all_results) / total_n,
            "oracle_avg_u": sum(r["oracle_avg_u"] * r["n"] for r in all_results) / total_n,
            "oracle_avg_latency": sum(r["oracle_avg_latency"] * r["n"] for r in all_results) / total_n,
            "oracle_avg_vram": sum(r["oracle_avg_vram"] * r["n"] for r in all_results) / total_n,
        }
        print()
        print("==================== COMBINED ====================")
        print(f"Combined N={total_n}")
        print(
            f"Chosen avg F1={combined['avg_f1']:.4f} | avg EM={combined['avg_em']:.4f} | avg U={combined['avg_u']:.4f} "
            f"| avg Latency={combined['avg_latency']:.4f}s | avg VRAM={combined['avg_vram']:.4f}GB"
        )
        print(
            f"Oracle avg F1={combined['oracle_avg_f1']:.4f} | avg EM={combined['oracle_avg_em']:.4f} | avg U={combined['oracle_avg_u']:.4f} "
            f"| avg Latency={combined['oracle_avg_latency']:.4f}s | avg VRAM={combined['oracle_avg_vram']:.4f}GB"
        )


if __name__ == "__main__":
    main()