# build_router_train_all.py
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from collections import Counter

from metrics import extract_prediction, normalize_answer  # <-- use normalizer from metrics

CFG_PATH = Path("configs/router_config.json")
PRED_DIR = Path("prediction")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_gold(gold_field: Any) -> List[str]:
    if gold_field is None:
        return []
    if isinstance(gold_field, list):
        return [str(g) for g in gold_field if g is not None]
    return [str(gold_field)]


def _get_question(row: Dict[str, Any]) -> str:
    if "question" in row and row["question"] is not None:
        return str(row["question"])
    return str(row.get("q", ""))


def _resolve_prediction_path(pattern_or_path: str) -> Optional[Path]:
    """
    Robust resolution for config entries.
    Supports:
      - exact path: prediction/xxx.jsonl or xxx.jsonl
      - glob pattern: prediction/*.jsonl or *_predictions.jsonl
    """
    s = (pattern_or_path or "").strip()
    if not s:
        return None

    p0 = Path(s)
    if p0.exists() and p0.is_file():
        return p0

    p1 = PRED_DIR / s
    if p1.exists() and p1.is_file():
        return p1

    if s.replace("\\", "/").startswith("prediction/"):
        s2 = s.replace("\\", "/")[len("prediction/") :]
        p2 = PRED_DIR / s2
        if p2.exists() and p2.is_file():
            return p2
        s = s2

    has_glob = any(ch in s for ch in ["*", "?", "["])
    candidates: List[Path] = []

    if has_glob:
        candidates.extend(sorted(Path(".").glob(s)))
        candidates.extend(sorted(PRED_DIR.glob(s)))
        candidates.extend(sorted(PRED_DIR.glob(Path(s).name)))

    if not candidates:
        candidates = sorted(PRED_DIR.glob(Path(s).name))

    candidates = [c for c in candidates if c.is_file() and c.suffix.lower() == ".jsonl"]
    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# -----------------------
# Multi-gold EM/F1 scorer
# -----------------------

def _f1_tokens(gold: str, pred: str) -> float:
    gold_toks = normalize_answer(gold).split()
    pred_toks = normalize_answer(pred).split()

    # both empty => perfect
    if len(gold_toks) == 0 and len(pred_toks) == 0:
        return 1.0
    # one empty => 0
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return 0.0

    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def _score_multi_gold(gold_answers: List[str], pred_text: str) -> Dict[str, float]:
    """
    Score ONE example where gold_answers can contain multiple acceptable strings.
    EM = max over golds
    F1 = max over golds
    loose_em = same as EM here
    """
    pred_norm = normalize_answer(pred_text or "")
    em_best = 0.0
    f1_best = 0.0

    for g in (gold_answers or [""]):
        g_norm = normalize_answer(g)
        em = 1.0 if g_norm == pred_norm else 0.0
        f1 = _f1_tokens(g, pred_text or "")
        if em > em_best:
            em_best = em
        if f1 > f1_best:
            f1_best = f1

    return {"em": em_best, "f1": f1_best, "loose_em": em_best}


# -----------------------
# PubMedQA label handling
# -----------------------

_LABELS = {"yes", "no", "maybe"}


def _normalize_label(s: str) -> str:
    return re.sub(r"[^a-z]", "", (s or "").strip().lower())


def _extract_yes_no_maybe(text: str) -> str:
    """
    Convert a free-form answer into {yes,no,maybe,unknown}.
    Robust token-based detection; prefer "<answer>:" if present.
    """
    tn = (text or "").strip().lower()

    if "<answer>:" in tn:
        after = tn.rsplit("<answer>:", 1)[1].strip()
        first = _normalize_label(after.split()[0]) if after else ""
        if first in _LABELS:
            return first

    tokens = re.findall(r"[a-z]+", tn)
    if "yes" in tokens:
        return "yes"
    if "no" in tokens:
        return "no"
    if "maybe" in tokens:
        return "maybe"

    return "unknown"


def _score_pubmedqa(gold_answers: List[str], pred_text: str) -> Dict[str, float]:
    """
    Score PubMedQA as label classification on {yes,no,maybe}.
    If gold isn't a clean label, fall back to multi-gold EM/F1.
    """
    gold_label = _extract_yes_no_maybe(gold_answers[0] if gold_answers else "")
    pred_label = _extract_yes_no_maybe(pred_text)

    if gold_label not in _LABELS:
        return _score_multi_gold(gold_answers, pred_text)

    em = 1.0 if pred_label == gold_label else 0.0
    return {"em": em, "f1": em, "loose_em": em}


# -----------------------
# HotpotQA yes/no handling
# -----------------------

_YESNO = {"yes", "no"}


def _extract_yes_no_hotpot(text: str) -> str:
    """
    Convert a free-form answer into {yes,no,unknown}.
    Handles "<ANSWER>:" style + common paraphrases.
    """
    tn = (text or "").strip().lower()

    # Prefer explicit answer tag
    if "<answer>:" in tn:
        after = tn.rsplit("<answer>:", 1)[1].strip()
        first = _normalize_label(after.split()[0]) if after else ""
        if first in _YESNO:
            return first
    if "<answer>" in tn:
        after = tn.rsplit("<answer>", 1)[1].strip().lstrip(":").strip()
        first = _normalize_label(after.split()[0]) if after else ""
        if first in _YESNO:
            return first

    tokens = re.findall(r"[a-z]+", tn)
    if "yes" in tokens:
        return "yes"
    if "no" in tokens:
        return "no"

    # Heuristic paraphrases for nationality/same-entity style yes/no
    if re.search(r"\b(different|not the same)\b", tn):
        return "no"
    if re.search(r"\b(same|of the same|both)\b", tn):
        return "yes"

    return "unknown"


def _score_hotpotqa_yesno(gold_answers: List[str], pred_text: str) -> Dict[str, float]:
    """
    If gold is yes/no, score as yes/no classification.
    Otherwise fall back to multi-gold F1/EM.
    """
    gold_label = _normalize_label(gold_answers[0] if gold_answers else "")
    if gold_label not in _YESNO:
        return _score_multi_gold(gold_answers, pred_text)

    pred_label = _extract_yes_no_hotpot(pred_text)
    if pred_label not in _YESNO:
        return _score_multi_gold(gold_answers, pred_text)

    em = 1.0 if pred_label == gold_label else 0.0
    return {"em": em, "f1": em, "loose_em": em}


def _score_generic(dataset_name: str, gold_answers: List[str], pred_text: str) -> Dict[str, float]:
    name = (dataset_name or "").lower()

    if "hotpotqa" in name:
        gl = _normalize_label(gold_answers[0] if gold_answers else "")
        if gl in _YESNO:
            return _score_hotpotqa_yesno(gold_answers, pred_text)
        return _score_multi_gold(gold_answers, pred_text)

    if "pubmedqa" in name:
        return _score_pubmedqa(gold_answers, pred_text)

    return _score_multi_gold(gold_answers, pred_text)


# -----------------------
# Canonical expert order
# -----------------------

def _get_canonical_expert_order(cfg: Dict[str, Any]) -> List[str]:
    tr = cfg.get("training", {}) or {}
    allowed = tr.get("allowed_experts", None)
    if not allowed or not isinstance(allowed, list):
        raise SystemExit(
            "configs/router_config.json must define training.allowed_experts "
            "as the canonical expert order."
        )
    return [str(x) for x in allowed]


# -----------------------
# Build router-train
# -----------------------

def build_router_train_for_dataset(dataset_name: str, ds_cfg: Dict[str, Any], canonical_order: List[str]) -> None:
    out_path = Path(ds_cfg["router_train_out"])

    experts_cfg: Dict[str, str] = ds_cfg.get("experts", {}) or {}
    if not experts_cfg:
        raise RuntimeError(f"[{dataset_name}] No experts configured in router_config.json")

    expert_names = [e for e in canonical_order if e in experts_cfg]
    if not expert_names:
        raise RuntimeError(
            f"[{dataset_name}] None of ds_cfg.experts are in training.allowed_experts.\n"
            f"allowed_experts={canonical_order}\n"
            f"ds_cfg.experts={list(experts_cfg.keys())}"
        )

    expert_maps: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for expert_name in expert_names:
        pattern = experts_cfg[expert_name]
        p = _resolve_prediction_path(pattern)
        if p is None:
            raise RuntimeError(f"[{dataset_name}] Missing prediction file for expert={expert_name}: {pattern}")

        rows = _read_jsonl(p)
        m: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            rid = r.get("id")
            if rid is None:
                rid = _get_question(r)
            m[str(rid)] = r

        expert_maps[expert_name] = m
        print(f"[{dataset_name}] expert={expert_name} -> {p} (rows={len(rows)})")

    all_ids: Optional[set] = None
    for m in expert_maps.values():
        ids = set(m.keys())
        all_ids = ids if all_ids is None else (all_ids & ids)

    if not all_ids:
        raise RuntimeError(f"[{dataset_name}] No overlapping ids across experts")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    any_expert = expert_names[0]

    with out_path.open("w", encoding="utf-8") as f:
        for rid in sorted(all_ids):
            base_row = expert_maps[any_expert][rid]
            question = _get_question(base_row)

            if "gold_answer" in base_row:
                gold = _normalize_gold(base_row["gold_answer"])
            elif "gold_answers" in base_row:
                gold = _normalize_gold(base_row["gold_answers"])
            else:
                continue

            if not question or not gold:
                continue

            expert_outcomes: Dict[str, Any] = {}
            for expert_name in expert_names:
                r = expert_maps[expert_name][rid]
                raw_pred = r.get("prediction", "")
                pred_extracted = extract_prediction(raw_pred)
                if pred_extracted == "":
                    pred_extracted = str(raw_pred).strip()

                scores = _score_generic(dataset_name, gold, pred_extracted)
                f1 = float(scores.get("f1", 0.0))
                em = float(scores.get("em", 0.0))
                loose_em = float(scores.get("loose_em", em))

                expert_outcomes[expert_name] = {
                    "prediction": pred_extracted,
                    "prediction_raw": raw_pred,
                    "f1": f1,
                    "em": em,
                    "loose_em": loose_em,
                    "latency": float(r.get("time", 0.0) or 0.0),
                    "vram_mb": float(r.get("peak_vram_mb", 0.0) or 0.0),
                    "use_rag": bool(r.get("use_rag", False)),
                    "ft_mode": str(r.get("ft_mode", "")),
                }

            record = {
                "id": rid,
                "dataset": dataset_name,
                "question": question,
                "gold_answers": gold,
                "experts": expert_outcomes,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[OK] Wrote {kept} router-train rows -> {out_path}")
    print(f"[OK] Expert order written: {expert_names}")


def main():
    if not CFG_PATH.exists():
        raise SystemExit(f"Missing {CFG_PATH}")

    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8-sig"))
    datasets = cfg.get("datasets", {}) or {}
    if not datasets:
        raise SystemExit("router_config.json missing 'datasets'")

    canonical_order = _get_canonical_expert_order(cfg)
    print("[INFO] Canonical expert order:", canonical_order)

    for dataset_name, ds_cfg in datasets.items():
        build_router_train_for_dataset(dataset_name, ds_cfg, canonical_order)


if __name__ == "__main__":
    main()
