# eval_router_two_stage.py
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

CFG_PATH = Path("configs/router_config.json")
PRED_DIR = Path("prediction")

RAG_EXPERTS = ["base_rag", "sft_rag", "raft_rag"]
NO_EXPERTS = ["base_only", "sft_only"]
DATASETS = ["hotpotqa", "squad_v2", "pubmedqa_v2"]


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


def policy_for_dataset(dataset: str) -> Optional[bool]:
    # In your setup, PubMedQA is forced no-RAG.
    if dataset == "pubmedqa_v2":
        return False
    return None


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


def load_gate(model_dir: Path, device: str, hidden_dim: int, dropout: float) -> Dict[str, Any]:
    p = model_dir / "gate.pt"
    if not p.exists():
        raise SystemExit(f"Missing gate checkpoint: {p}")
    ckpt = torch.load(p, map_location="cpu", weights_only=False)

    # forced policy marker used for pubmedqa_v2 in your trainer
    if "forced_policy" in ckpt:
        return {
            "forced_policy": ckpt["forced_policy"],
            "in_dim": ckpt.get("in_dim"),
            "embed_model": ckpt.get("embed_model"),
            "model": None,
            "gate_delta": ckpt.get("gate_delta"),
            "gate_objective": "forced",
            "calibrated_threshold": float(ckpt.get("calibrated_threshold", 0.0) or 0.0),
        }

    in_dim = int(ckpt["in_dim"])
    gate_objective = str(ckpt.get("gate_objective", "cls"))

    # NEW: delta regression gate (out_dim=1)
    if gate_objective == "delta_reg":
        model = MLP(in_dim, hidden_dim, dropout, out_dim=1)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        return {
            "forced_policy": None,
            "in_dim": in_dim,
            "embed_model": ckpt.get("embed_model"),
            "model": model,
            "gate_delta": ckpt.get("gate_delta"),
            "gate_objective": "delta_reg",
            "calibrated_threshold": float(ckpt.get("calibrated_threshold", 0.0) or 0.0),
        }

    # default: classifier gate (out_dim=2)
    model = MLP(in_dim, hidden_dim, dropout, out_dim=2)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return {
        "forced_policy": None,
        "in_dim": in_dim,
        "embed_model": ckpt.get("embed_model"),
        "model": model,
        "gate_delta": ckpt.get("gate_delta"),
        "gate_objective": "cls",
        "calibrated_threshold": float(ckpt.get("calibrated_threshold", 0.0) or 0.0),
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
    return {"model": model, "experts": experts, "in_dim": in_dim, "embed_model": ckpt.get("embed_model")}


def load_answerability(model_dir: Path, device: str, hidden_dim: int, dropout: float) -> Optional[Dict[str, Any]]:
    p = model_dir / "answerability.pt"
    if not p.exists():
        return None
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    in_dim = int(ckpt["in_dim"])
    model = MLP(in_dim, hidden_dim, dropout, out_dim=2)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    prior = ckpt.get("prior")
    return {
        "model": model,
        "in_dim": in_dim,
        "embed_model": ckpt.get("embed_model"),
        "prior": prior,
        "reg_type": ckpt.get("reg_type"),
        "reg_weight": ckpt.get("reg_weight"),
    }


# --------------------------
# Tradeoff utility (research)
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

    cost_norm = (tcfg["wL"] * (L / max(1e-8, tcfg["sla_latency_s"]))) + (
            tcfg["wV"] * (V / max(1e-8, tcfg["vram_budget_gb"]))
    )
    U = Q - tcfg["lambda_cost"] * cost_norm

    if tcfg["beta_lat"] > 0.0:
        U -= tcfg["beta_lat"] * float(
            np.exp(
                tcfg["gamma_lat"]
                * max(0.0, (L - tcfg["sla_latency_s"]) / max(1e-8, tcfg["sla_latency_s"]))
            )
        )
    if tcfg["beta_vram"] > 0.0:
        U -= tcfg["beta_vram"] * float(
            np.exp(
                tcfg["gamma_vram"]
                * max(0.0, (V - tcfg["vram_budget_gb"]) / max(1e-8, tcfg["vram_budget_gb"]))
            )
        )
    return float(U)


def no_answer_outcome(row: Dict[str, Any]) -> Dict[str, Any]:
    ok = 1.0 if is_squad_no_answer_gold(row) else 0.0
    return {"f1": ok, "em": ok, "loose_em": ok, "latency": 0.0, "vram_mb": 0.0}


def parse_csv_floats(s: Optional[str]) -> Optional[List[float]]:
    if not s:
        return None
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(float(x))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--model_dir", type=str, default="results/two_stage_utility")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.10)

    ap.add_argument(
        "--oracle_policy_aligned",
        action="store_true",
        help="If dataset has forced policy (True/False), oracle is restricted to that pool. "
             "NO_ANSWER is still considered on squad_v2.",
    )
    ap.add_argument(
        "--tradeoff_mode",
        action="store_true",
        help="Evaluate using tradeoff_U (quality - lambda*cost - penalties).",
    )
    ap.add_argument(
        "--sweep_lambda_cost",
        type=str,
        default=None,
        help='Comma-separated lambda_cost values, e.g. "0,0.02,0.05,0.1". Only in tradeoff_mode.',
    )

    # Answerability (SQuAD v2 only)
    ap.add_argument(
        "--use_answerability",
        action="store_true",
        help="Use the trained answerability head on squad_v2 to optionally abstain with NO_ANSWER.",
    )
    ap.add_argument(
        "--ans_threshold",
        type=float,
        default=0.5,
        help="Threshold on P(answerable) to decide abstain (predict NO_ANSWER). Used only with --use_answerability.",
    )

    # Gate threshold / sweep
    ap.add_argument(
        "--gate_threshold",
        type=float,
        default=None,
        help=(
            "Gate decision threshold. "
            "If gate_objective=cls: use p(rag)>=threshold. "
            "If gate_objective=delta_reg: use pred_delta>threshold. "
            "If not set: uses checkpoint calibrated_threshold if present, otherwise argmax/0."
        ),
    )
    ap.add_argument(
        "--sweep_gate_threshold",
        type=str,
        default=None,
        help='Comma-separated thresholds, e.g. "0.1,0.2,0.3,0.4,0.5".',
    )

    # decomposition modes
    ap.add_argument(
        "--oracle_gate",
        action="store_true",
        help="Use oracle family (RAG vs NO-RAG) for each example, then apply learned selector within that family.",
    )
    ap.add_argument(
        "--oracle_selector",
        action="store_true",
        help="Use learned gate family, but inside the chosen family pick the oracle best expert.",
    )

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
        print("\n==================== OFFLINE TWO-STAGE EVAL (utility-aligned) ====================\n")

    lambdas = parse_csv_floats(args.sweep_lambda_cost)
    thr_sweep = parse_csv_floats(args.sweep_gate_threshold)

    for dataset in DATASETS:
        if args.only and dataset != args.only:
            continue

        rows = read_router_train(dataset)
        questions = [r["question"] for r in rows]

        model_dir = model_root / dataset
        pol = policy_for_dataset(dataset)

        gate = load_gate(model_dir, args.device, args.hidden_dim, args.dropout)
        sel_rag = load_selector(model_dir, "rag", args.device, args.hidden_dim, args.dropout)
        sel_no = load_selector(model_dir, "no_rag", args.device, args.hidden_dim, args.dropout)

        ans_head = None
        if dataset == "squad_v2" and args.use_answerability:
            ans_head = load_answerability(model_dir, args.device, args.hidden_dim, args.dropout)
            if ans_head is None:
                print(f"[WARN] --use_answerability set but missing answerability.pt in {model_dir}. Will ignore answerability.")

        embed_model = (
                gate.get("embed_model")
                or (sel_rag or {}).get("embed_model")
                or (sel_no or {}).get("embed_model")
                or (ans_head or {}).get("embed_model")
                or "sentence-transformers/all-mpnet-base-v2"
        )

        embedder = Embedder(embed_model, device=args.device)
        X = embedder.encode(questions, batch_size=args.batch_size).float()

        if pol is True and sel_rag is None:
            raise SystemExit(f"[{dataset}] policy=True but missing selector_rag.pt in {model_dir}")
        if pol is False and sel_no is None:
            raise SystemExit(f"[{dataset}] policy=False but missing selector_no_rag.pt in {model_dir}")
        if pol is None and (sel_rag is None or sel_no is None):
            raise SystemExit(f"[{dataset}] policy=None requires both selectors in {model_dir}")

        def oracle_pool_for_dataset() -> List[str]:
            if args.oracle_policy_aligned and pol is not None:
                return RAG_EXPERTS if pol is True else NO_EXPERTS
            return RAG_EXPERTS + NO_EXPERTS

        def best_in_family(ex: Dict[str, Any], fam: str, tcfg: Dict[str, float]) -> str:
            pool = RAG_EXPERTS if fam == "rag" else NO_EXPERTS
            best_e, best_v = None, -1e18
            for e in pool:
                out = ex[e]
                v = tradeoff_U(out, tcfg) if args.tradeoff_mode else float(out.get("f1", 0.0))
                if v > best_v:
                    best_v, best_e = v, e
            return str(best_e)

        def family_oracle_label(ex: Dict[str, Any], tcfg: Dict[str, float]) -> str:
            best_rag = best_in_family(ex, "rag", tcfg)
            best_no = best_in_family(ex, "no", tcfg)
            out_r = ex[best_rag]
            out_n = ex[best_no]
            vr = tradeoff_U(out_r, tcfg) if args.tradeoff_mode else float(out_r.get("f1", 0.0))
            vn = tradeoff_U(out_n, tcfg) if args.tradeoff_mode else float(out_n.get("f1", 0.0))
            return "rag" if vr > vn else "no"

        def default_gate_threshold() -> float:
            if args.gate_threshold is not None:
                return float(args.gate_threshold)
            # fall back to checkpoint calibrated threshold if available (works for both gate types)
            return float(gate.get("calibrated_threshold", 0.0) or 0.0)

        def gate_predict_family(xi: torch.Tensor, gate_threshold: Optional[float]) -> str:
            # policy overrides
            if pol is True:
                return "rag"
            if pol is False:
                return "no"

            gmodel = gate["model"]
            if gmodel is None:
                # should not happen for pol None, but be safe
                return "no"

            thr = default_gate_threshold() if gate_threshold is None else float(gate_threshold)

            with torch.no_grad():
                if gate.get("gate_objective") == "delta_reg":
                    pred_delta = float(gmodel(xi.unsqueeze(0)).squeeze(0).squeeze(-1).item())
                    return "rag" if pred_delta > thr else "no"
                else:
                    logits = gmodel(xi.unsqueeze(0))[0]
                    probs = torch.softmax(logits, dim=0)  # [p(no), p(rag)]
                    p_rag = float(probs[1].item())
                    if gate_threshold is None and args.gate_threshold is None and "calibrated_threshold" not in gate:
                        # classic behavior: argmax if no thresholds available
                        return "rag" if int(torch.argmax(probs).item()) == 1 else "no"
                    return "rag" if p_rag >= thr else "no"

        def run_eval(lambda_override: Optional[float] = None, gate_threshold: Optional[float] = None) -> Dict[str, Any]:
            chosen_f1 = chosen_em = chosen_u = 0.0
            oracle_f1 = oracle_em = oracle_u = 0.0

            pick_counts = Counter()
            gate_counts = Counter()
            ans_counts = Counter()

            gate_oracle_TN = gate_oracle_TP = gate_oracle_FP = gate_oracle_FN = 0

            # selector diagnostics
            ent_sum = 0.0
            maxp_sum = 0.0
            diag_steps = 0

            # answerability confusion counts (SQuAD only)
            ans_tp = ans_fp = ans_tn = ans_fn = 0  # positive = "ANSWERABLE"

            tcfg = tradeoff_from_cfg(cfg)
            if lambda_override is not None:
                tcfg["lambda_cost"] = float(lambda_override)

            pool = oracle_pool_for_dataset()

            for i in tqdm(range(len(rows)), desc=f"Eval {dataset}"):
                r = rows[i]
                ex = r["experts"]
                xi = X[i].to(args.device)

                # 1) Answerability (optional, SQuAD v2 only)
                predicted_no_answer = False
                if dataset == "squad_v2" and ans_head is not None:
                    gold_no_answer = is_squad_no_answer_gold(r)
                    gold_answerable = (not gold_no_answer)

                    am = ans_head["model"]
                    with torch.no_grad():
                        logits = am(xi.unsqueeze(0))
                        probs = torch.softmax(logits, dim=1)[0]  # [p(not), p(ans)]
                        p_answerable = float(probs[1].item())

                    predicted_answerable = (p_answerable >= float(args.ans_threshold))
                    predicted_no_answer = (not predicted_answerable)
                    ans_counts["ANSWER" if predicted_answerable else "NO_ANSWER"] += 1

                    if predicted_answerable and gold_answerable:
                        ans_tp += 1
                    elif predicted_answerable and (not gold_answerable):
                        ans_fp += 1
                    elif (not predicted_answerable) and (not gold_answerable):
                        ans_tn += 1
                    else:
                        ans_fn += 1

                if predicted_no_answer:
                    out = no_answer_outcome(r)
                    chosen_f1 += float(out["f1"])
                    chosen_em += float(out["em"])
                    chosen_u += float(tradeoff_U(out, tcfg) if args.tradeoff_mode else float(out["f1"]))
                    pick_counts["NO_ANSWER"] += 1
                else:
                    # 2) Family decision
                    if args.oracle_gate:
                        fam = family_oracle_label(ex, tcfg)
                    else:
                        fam = gate_predict_family(xi, gate_threshold)

                    gate_counts[fam] += 1

                    # gate oracle agreement (only meaningful if not forced policy and not oracle_gate)
                    if pol is None and (not args.oracle_gate):
                        fam_or = family_oracle_label(ex, tcfg)
                        if fam == "rag" and fam_or == "rag":
                            gate_oracle_TP += 1
                        elif fam == "no" and fam_or == "no":
                            gate_oracle_TN += 1
                        elif fam == "rag" and fam_or == "no":
                            gate_oracle_FP += 1
                        else:
                            gate_oracle_FN += 1

                    # 3) Within-family selection
                    if args.oracle_selector:
                        chosen_expert = best_in_family(ex, fam, tcfg)
                    else:
                        if fam == "rag":
                            sm = sel_rag["model"]
                            experts = sel_rag["experts"]
                        else:
                            sm = sel_no["model"]
                            experts = sel_no["experts"]

                        with torch.no_grad():
                            logits = sm(xi.unsqueeze(0))[0]
                            probs = torch.softmax(logits, dim=0)
                            cls = int(torch.argmax(probs).item())
                            chosen_expert = experts[cls]

                            # diagnostics (only for selector-based choice)
                            p = probs.detach().cpu().numpy()
                            ent = float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum())
                            ent_sum += ent
                            maxp_sum += float(p.max())
                            diag_steps += 1

                    pick_counts[chosen_expert] += 1
                    out = ex[chosen_expert]
                    chosen_f1 += float(out.get("f1", 0.0))
                    chosen_em += float(out.get("em", 0.0))
                    chosen_u += float(tradeoff_U(out, tcfg) if args.tradeoff_mode else float(out.get("f1", 0.0)))

                # ORACLE overall (best among pool + NO_ANSWER for SQuAD)
                best_expert = None
                best_val = -1e18
                for e in pool:
                    out_e = ex[e]
                    val = tradeoff_U(out_e, tcfg) if args.tradeoff_mode else float(out_e.get("f1", 0.0))
                    if val > best_val:
                        best_val = val
                        best_expert = e

                if dataset == "squad_v2":
                    out_na = no_answer_outcome(r)
                    val_na = tradeoff_U(out_na, tcfg) if args.tradeoff_mode else float(out_na.get("f1", 0.0))
                    if val_na > best_val:
                        best_val = val_na
                        best_expert = "NO_ANSWER"

                out_best = no_answer_outcome(r) if best_expert == "NO_ANSWER" else ex[best_expert]
                oracle_f1 += float(out_best.get("f1", 0.0))
                oracle_em += float(out_best.get("em", 0.0))
                oracle_u += float(tradeoff_U(out_best, tcfg) if args.tradeoff_mode else float(out_best.get("f1", 0.0)))

            n = len(rows)

            ans_metrics = None
            if dataset == "squad_v2" and ans_head is not None:
                eps = 1e-12
                acc = (ans_tp + ans_tn) / max(1, (ans_tp + ans_tn + ans_fp + ans_fn))
                prec = ans_tp / max(eps, (ans_tp + ans_fp))
                rec = ans_tp / max(eps, (ans_tp + ans_fn))
                f1m = 2 * prec * rec / max(eps, (prec + rec))
                ans_metrics = {
                    "acc": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1m,
                    "tp": ans_tp,
                    "fp": ans_fp,
                    "tn": ans_tn,
                    "fn": ans_fn,
                    "threshold": float(args.ans_threshold),
                }

            gate_agree = None
            if pol is None and (not args.oracle_gate):
                total = gate_oracle_TN + gate_oracle_TP + gate_oracle_FP + gate_oracle_FN
                if total > 0:
                    gate_agree = {
                        "acc": (gate_oracle_TN + gate_oracle_TP) / total,
                        "conf": {"TN": gate_oracle_TN, "TP": gate_oracle_TP, "FP": gate_oracle_FP, "FN": gate_oracle_FN},
                    }

            selector_diag = None
            if diag_steps > 0 and (not args.oracle_selector):
                selector_diag = {
                    "mean_entropy": ent_sum / diag_steps,
                    "mean_maxprob": maxp_sum / diag_steps,
                    "steps": diag_steps,
                }

            thr_used = None
            if pol is None and (not args.oracle_gate):
                thr_used = default_gate_threshold() if gate_threshold is None else float(gate_threshold)

            return {
                "chosen_f1": chosen_f1 / n,
                "chosen_em": chosen_em / n,
                "chosen_u": chosen_u / n,
                "oracle_f1": oracle_f1 / n,
                "oracle_em": oracle_em / n,
                "oracle_u": oracle_u / n,
                "gate_counts": dict(gate_counts),
                "pick_counts": pick_counts.most_common(10),
                "ans_counts": dict(ans_counts),
                "ans_metrics": ans_metrics,
                "gate_oracle_agreement": gate_agree,
                "selector_diag": selector_diag,
                "embed_model": embed_model,
                "policy": pol,
                "lambda_cost": (lambda_override if lambda_override is not None else tradeoff_from_cfg(cfg)["lambda_cost"]),
                "gate_threshold": thr_used,
                "gate_objective": gate.get("gate_objective", "cls"),
                "oracle_pool": pool,
            }

        # ---------- printing ----------
        def print_res(tag: str, res: Dict[str, Any]):
            print(
                f"\n--- {dataset} --- {tag} N={len(rows)} "
                f"policy={pol} oracle_policy_aligned={bool(args.oracle_policy_aligned)} "
                f"tradeoff_mode={bool(args.tradeoff_mode)} "
                f"use_answerability={bool(args.use_answerability and dataset=='squad_v2')} "
                f"oracle_gate={bool(args.oracle_gate)} oracle_selector={bool(args.oracle_selector)}"
            )
            print(f"embed_model={res['embed_model']}")
            print(f"gate_objective={res.get('gate_objective')}")
            if res.get("gate_threshold") is not None:
                print(f"gate_threshold_used={res['gate_threshold']}")
            print(f"oracle_pool={res['oracle_pool']}")
            if dataset == "squad_v2" and res["ans_metrics"] is not None:
                print(f"Answerability picks: {res['ans_counts']}")
                am = res["ans_metrics"]
                print(
                    f"Answerability metrics: acc={am['acc']:.4f} "
                    f"prec={am['precision']:.4f} rec={am['recall']:.4f} f1={am['f1']:.4f} "
                    f"(thr={am['threshold']:.2f}) | tp={am['tp']} fp={am['fp']} tn={am['tn']} fn={am['fn']}"
                )
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

        # default run / sweeps
        if thr_sweep is not None and (pol is None) and (not args.oracle_gate):
            print(f"\n--- {dataset} --- sweep gate_threshold ---")
            for t in thr_sweep:
                res = run_eval(gate_threshold=t)
                print(f"thr={t:<6.3f} | chosen_F1={res['chosen_f1']:.4f} oracle_F1={res['oracle_f1']:.4f} | gate={res['gate_counts']}")
            continue

        if lambdas is None:
            res = run_eval(gate_threshold=args.gate_threshold)
            tag = ""
            if args.gate_threshold is not None:
                tag = f"(gate_threshold={args.gate_threshold:.3f})"
            else:
                if pol is None and (not args.oracle_gate):
                    tag = f"(gate_threshold=auto:{res.get('gate_threshold')})"
            print_res(tag, res)
        else:
            print(f"\n--- {dataset} --- sweep lambda_cost ---")
            for lam in lambdas:
                res = run_eval(lambda_override=lam, gate_threshold=args.gate_threshold)
                print(f"lambda={lam:<6} | chosen_U={res['chosen_u']:.4f} oracle_U={res['oracle_u']:.4f} | chosen_F1={res['chosen_f1']:.4f} oracle_F1={res['oracle_f1']:.4f}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
