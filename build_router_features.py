# build_router_features.py
import json
import re
import math
from itertools import combinations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

PRED_DIR = Path("prediction")

DATASETS = ["hotpotqa", "squad_v2", "pubmedqa_v2", "commonsenseqa"]

DOC_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.DOTALL | re.IGNORECASE)

SELECTOR_CMP_PREFIX = "selcmp__"

RAG_EXPERTS = ["base_rag", "sft_rag", "raft_rag"]
NO_EXPERTS = ["base_only", "sft_only"]

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "where", "why", "how", "what", "which", "who", "whom",
    "of", "to", "in", "on", "for", "with", "as", "by", "from", "at", "into", "about", "over", "after", "before", "between",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "done", "have", "has", "had",
    "this", "that", "these", "those", "it", "its", "their", "his", "her", "they", "them", "he", "she", "we", "you", "i",
}

PUBMED_LABELS = {"yes", "no", "maybe"}
YESNO = {"yes", "no"}


def tokenize(s: str) -> List[str]:
    s = (s or "").lower()
    toks = re.findall(r"[a-z0-9]+", s)
    return [t for t in toks if t and t not in STOPWORDS]


def raw_tokens(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def normalize_text(s: str) -> str:
    return " ".join(raw_tokens(s))


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def token_recall(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa:
        return 0.0
    return len(sa & sb) / max(1, len(sa))


def safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def safe_std(xs: List[float]) -> float:
    if not xs:
        return 0.0
    m = safe_mean(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return float(math.sqrt(v))


def top2_gap(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    ys = sorted(xs, reverse=True)
    return float(ys[0] - ys[1])


def wh_type(q: str) -> str:
    ql = (q or "").strip().lower()
    for w in ["who", "what", "when", "where", "why", "how", "which"]:
        if ql.startswith(w + " ") or ql == w:
            return w
    return "other"


def extract_docs_from_prediction_raw(pred_raw: str) -> List[str]:
    if not pred_raw:
        return []
    return [m.strip() for m in DOC_RE.findall(pred_raw) if m and m.strip()]


def question_stats(q: str) -> Dict[str, float]:
    q = q or ""
    words = re.findall(r"\S+", q.strip())
    n_words = len(words)
    n_chars = len(q)
    caps = sum(1 for c in q if c.isupper())
    letters = sum(1 for c in q if c.isalpha())
    cap_ratio = (caps / max(1, letters))
    ql = q.lower()
    wt = wh_type(q)
    return {
        "q_len_words": float(n_words),
        "q_len_chars": float(n_chars),
        "q_has_and": 1.0 if " and " in f" {ql} " else 0.0,
        "q_has_or": 1.0 if " or " in f" {ql} " else 0.0,
        "q_wh_who": 1.0 if wt == "who" else 0.0,
        "q_wh_what": 1.0 if wt == "what" else 0.0,
        "q_wh_when": 1.0 if wt == "when" else 0.0,
        "q_wh_where": 1.0 if wt == "where" else 0.0,
        "q_wh_why": 1.0 if wt == "why" else 0.0,
        "q_wh_how": 1.0 if wt == "how" else 0.0,
        "q_wh_which": 1.0 if wt == "which" else 0.0,
        "q_wh_other": 1.0 if wt == "other" else 0.0,
        "q_capital_ratio": float(cap_ratio),
    }


def ctx_stats(q: str, docs: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    q_toks = tokenize(q)
    overlaps: List[float] = []
    doc_lens: List[int] = []
    ctx_chars = 0
    ctx_words = 0

    for d in docs:
        d = d or ""
        ctx_chars += len(d)
        w = re.findall(r"\S+", d)
        ctx_words += len(w)
        doc_lens.append(len(w))
        overlaps.append(jaccard(q_toks, tokenize(d)))

    n_docs = len(docs)
    if n_docs == 0:
        rp = {
            "n_docs": 0.0,
            "ctx_chars": 0.0,
            "ctx_words": 0.0,
            "max_q_doc_jaccard": 0.0,
            "mean_q_doc_jaccard": 0.0,
            "max_doclen_words": 0.0,
            "mean_doclen_words": 0.0,
        }
        ru = {
            "std_q_doc_jaccard": 0.0,
            "frac_low_overlap": 1.0,
            "top_doc_dominance": 0.0,
        }
        return rp, ru

    mean_ol = sum(overlaps) / n_docs
    max_ol = max(overlaps)
    var = sum((x - mean_ol) ** 2 for x in overlaps) / n_docs
    std = math.sqrt(var)

    low = sum(1 for x in overlaps if x < 0.02) / n_docs
    dominance = max_ol / max(1e-8, mean_ol)

    rp = {
        "n_docs": float(n_docs),
        "ctx_chars": float(ctx_chars),
        "ctx_words": float(ctx_words),
        "max_q_doc_jaccard": float(max_ol),
        "mean_q_doc_jaccard": float(mean_ol),
        "max_doclen_words": float(max(doc_lens) if doc_lens else 0),
        "mean_doclen_words": float(sum(doc_lens) / n_docs if doc_lens else 0),
    }
    ru = {
        "std_q_doc_jaccard": float(std),
        "frac_low_overlap": float(low),
        "top_doc_dominance": float(dominance),
    }
    return rp, ru


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pick_rag_prediction_raw(experts: Dict[str, Any], ds: str) -> Optional[str]:
    ds = (ds or "").lower()
    pref = ["base_rag"] if "commonsense" in ds else ["base_rag", "raft_rag", "sft_rag"]
    for k in pref:
        if k in experts and isinstance(experts[k], dict):
            pr = experts[k].get("prediction_raw")
            if pr:
                return pr
    return None


def prediction_text_from_expert_row(expert_row: Dict[str, Any]) -> str:
    pred = expert_row.get("prediction", None)
    if pred is None or str(pred).strip() == "":
        pred = expert_row.get("prediction_raw", "")
    return str(pred or "").strip()


def extract_pubmed_label(text: str) -> str:
    t = (text or "").strip().lower()

    if "<answer>:" in t:
        after = t.rsplit("<answer>:", 1)[1].strip()
        first = normalize_text(after).split()
        if first and first[0] in PUBMED_LABELS:
            return first[0]

    toks = raw_tokens(t)
    for lab in ["yes", "no", "maybe"]:
        if lab in toks:
            return lab
    return "unknown"


def extract_yes_no_label(text: str) -> str:
    t = (text or "").strip().lower()

    if "<answer>:" in t:
        after = t.rsplit("<answer>:", 1)[1].strip()
        first = normalize_text(after).split()
        if first and first[0] in YESNO:
            return first[0]

    toks = raw_tokens(t)
    if "yes" in toks:
        return "yes"
    if "no" in toks:
        return "no"
    return "unknown"


def is_no_answer_like(text: str) -> bool:
    nt = normalize_text(text)
    if nt == "":
        return True

    small_set = {
        "no answer",
        "noanswer",
        "no_answer",
        "unanswerable",
        "cannot answer",
        "cant answer",
        "cannot be answered",
        "not answerable",
        "unknown",
        "not enough information",
        "insufficient information",
    }
    if nt in small_set:
        return True

    if "no answer" in nt:
        return True
    if "cannot answer" in nt:
        return True
    if "unanswerable" in nt:
        return True

    return False


def length_bucket(text: str) -> str:
    n = len(re.findall(r"\S+", (text or "").strip()))
    if n <= 0:
        return "empty"
    if n <= 3:
        return "short"
    if n <= 8:
        return "medium"
    return "long"


def infer_answer_type(dataset: str, pred_text: str) -> str:
    ds = (dataset or "").lower()
    if "pubmedqa" in ds:
        lab = extract_pubmed_label(pred_text)
        if lab in PUBMED_LABELS:
            return lab
        return "other"

    if "squad" in ds:
        if is_no_answer_like(pred_text):
            return "no_answer"
        yn = extract_yes_no_label(pred_text)
        if yn in YESNO:
            return yn
        return length_bucket(pred_text)

    if "hotpotqa" in ds:
        yn = extract_yes_no_label(pred_text)
        if yn in YESNO:
            return yn
        if is_no_answer_like(pred_text):
            return "no_answer"
        return length_bucket(pred_text)

    if is_no_answer_like(pred_text):
        return "no_answer"
    yn = extract_yes_no_label(pred_text)
    if yn in YESNO:
        return yn
    return length_bucket(pred_text)


def expert_proxy_bundle(dataset: str, q: str, expert_row: Dict[str, Any]) -> Dict[str, Any]:
    pred_text = prediction_text_from_expert_row(expert_row)
    pred_norm = normalize_text(pred_text)
    pred_toks = tokenize(pred_text)

    docs = extract_docs_from_prediction_raw(str(expert_row.get("prediction_raw", "") or ""))
    rp, ru = ctx_stats(q, docs)

    all_doc_tokens: List[str] = []
    doc_norms: List[str] = []
    for d in docs:
        all_doc_tokens.extend(tokenize(d))
        doc_norms.append(normalize_text(d))

    pred_in_docs_exact = 0.0
    if pred_norm and pred_norm not in {"yes", "no", "maybe"}:
        pred_in_docs_exact = 1.0 if any(pred_norm in dn for dn in doc_norms) else 0.0

    ans_type = infer_answer_type(dataset, pred_text)

    type_flags = {
        "type_no_answer": 1.0 if ans_type == "no_answer" else 0.0,
        "type_yes": 1.0 if ans_type == "yes" else 0.0,
        "type_no": 1.0 if ans_type == "no" else 0.0,
        "type_maybe": 1.0 if ans_type == "maybe" else 0.0,
        "type_short": 1.0 if ans_type == "short" else 0.0,
        "type_medium": 1.0 if ans_type == "medium" else 0.0,
        "type_long": 1.0 if ans_type == "long" else 0.0,
        "type_other": 1.0 if ans_type == "other" else 0.0,
    }

    metrics = {
        "n_docs": float(rp["n_docs"]),
        "ctx_words": float(rp["ctx_words"]),
        "mean_q_doc_jaccard": float(rp["mean_q_doc_jaccard"]),
        "max_q_doc_jaccard": float(rp["max_q_doc_jaccard"]),
        "std_q_doc_jaccard": float(ru["std_q_doc_jaccard"]),
        "top_doc_dominance": float(ru["top_doc_dominance"]),
        "pred_len_words": float(len(re.findall(r"\S+", pred_text))),
        "pred_q_jaccard": float(jaccard(tokenize(q), pred_toks)),
        "pred_doc_jaccard": float(jaccard(pred_toks, all_doc_tokens)) if all_doc_tokens else 0.0,
        "pred_doc_token_recall": float(token_recall(pred_toks, all_doc_tokens)) if all_doc_tokens else 0.0,
        "pred_in_docs_exact": float(pred_in_docs_exact),
        "pred_is_empty": 1.0 if pred_norm == "" else 0.0,
    }
    metrics.update(type_flags)

    return {
        "pred_text": pred_text,
        "pred_norm": pred_norm,
        "pred_tokens": pred_toks,
        "doc_tokens": all_doc_tokens,
        "answer_type": ans_type,
        "metrics": metrics,
    }


def add_family_comparative_features(
        out: Dict[str, float],
        family_name: str,
        expert_names: List[str],
        bundles: Dict[str, Dict[str, Any]],
) -> None:
    available = [e for e in expert_names if e in bundles]
    if len(available) <= 1:
        return

    scalar_metrics = [
        "pred_len_words",
        "pred_q_jaccard",
        "pred_doc_jaccard",
        "pred_doc_token_recall",
        "pred_in_docs_exact",
        "pred_is_empty",
        "type_no_answer",
        "type_yes",
        "type_no",
        "type_maybe",
        "type_short",
        "type_medium",
        "type_long",
        "type_other",
    ]

    if family_name == "rag":
        scalar_metrics += [
            "n_docs",
            "ctx_words",
            "mean_q_doc_jaccard",
            "max_q_doc_jaccard",
            "std_q_doc_jaccard",
            "top_doc_dominance",
        ]

    winner_metrics = [
        "pred_q_jaccard",
        "pred_doc_jaccard",
        "pred_doc_token_recall",
        "pred_in_docs_exact",
    ]
    if family_name == "rag":
        winner_metrics += ["mean_q_doc_jaccard", "max_q_doc_jaccard"]

    out[f"{SELECTOR_CMP_PREFIX}{family_name}__n_available_experts"] = float(len(available))

    # per-expert scalar values
    for e in available:
        for m in scalar_metrics:
            out[f"{SELECTOR_CMP_PREFIX}{family_name}__{m}__{e}"] = float(bundles[e]["metrics"].get(m, 0.0))

    # family spread / std / gap
    for m in scalar_metrics:
        vals = [float(bundles[e]["metrics"].get(m, 0.0)) for e in available]
        out[f"{SELECTOR_CMP_PREFIX}{family_name}__{m}__spread"] = float(max(vals) - min(vals))
        out[f"{SELECTOR_CMP_PREFIX}{family_name}__{m}__std"] = float(safe_std(vals))
        out[f"{SELECTOR_CMP_PREFIX}{family_name}__{m}__top1_gap"] = float(top2_gap(vals))

    # winner one-hots
    for m in winner_metrics:
        best_e = max(available, key=lambda e: float(bundles[e]["metrics"].get(m, 0.0)))
        for e in available:
            out[f"{SELECTOR_CMP_PREFIX}{family_name}__best_{m}__{e}"] = 1.0 if e == best_e else 0.0

    # exact normalized-answer agreement + answer-type agreement
    exact_agree: List[float] = []
    type_agree: List[float] = []
    pred_jaccs: List[float] = []
    doc_jaccs: List[float] = []

    for a, b in combinations(available, 2):
        na = bundles[a]["pred_norm"]
        nb = bundles[b]["pred_norm"]
        ta = bundles[a]["answer_type"]
        tb = bundles[b]["answer_type"]

        exact_agree.append(1.0 if na != "" and na == nb else 0.0)
        type_agree.append(1.0 if ta == tb else 0.0)
        pred_jaccs.append(float(jaccard(bundles[a]["pred_tokens"], bundles[b]["pred_tokens"])))

        if family_name == "rag":
            doc_jaccs.append(float(jaccard(bundles[a]["doc_tokens"], bundles[b]["doc_tokens"])))

    out[f"{SELECTOR_CMP_PREFIX}{family_name}__pred_exact_agree_mean"] = float(safe_mean(exact_agree))
    out[f"{SELECTOR_CMP_PREFIX}{family_name}__pred_exact_agree_min"] = float(min(exact_agree) if exact_agree else 0.0)
    out[f"{SELECTOR_CMP_PREFIX}{family_name}__pred_exact_agree_max"] = float(max(exact_agree) if exact_agree else 0.0)

    out[f"{SELECTOR_CMP_PREFIX}{family_name}__type_agree_mean"] = float(safe_mean(type_agree))
    out[f"{SELECTOR_CMP_PREFIX}{family_name}__type_agree_min"] = float(min(type_agree) if type_agree else 0.0)
    out[f"{SELECTOR_CMP_PREFIX}{family_name}__type_agree_max"] = float(max(type_agree) if type_agree else 0.0)

    out[f"{SELECTOR_CMP_PREFIX}{family_name}__pred_agree_mean"] = float(safe_mean(pred_jaccs))
    out[f"{SELECTOR_CMP_PREFIX}{family_name}__pred_agree_min"] = float(min(pred_jaccs) if pred_jaccs else 0.0)
    out[f"{SELECTOR_CMP_PREFIX}{family_name}__pred_agree_max"] = float(max(pred_jaccs) if pred_jaccs else 0.0)

    if family_name == "rag":
        out[f"{SELECTOR_CMP_PREFIX}{family_name}__doc_agree_mean"] = float(safe_mean(doc_jaccs))
        out[f"{SELECTOR_CMP_PREFIX}{family_name}__doc_agree_min"] = float(min(doc_jaccs) if doc_jaccs else 0.0)
        out[f"{SELECTOR_CMP_PREFIX}{family_name}__doc_agree_max"] = float(max(doc_jaccs) if doc_jaccs else 0.0)

    # type counts within family
    type_names = ["no_answer", "yes", "no", "maybe", "short", "medium", "long", "other"]
    for tname in type_names:
        cnt = sum(1 for e in available if bundles[e]["answer_type"] == tname)
        out[f"{SELECTOR_CMP_PREFIX}{family_name}__count_type_{tname}"] = float(cnt)


def build_selector_comparative_features(dataset: str, q: str, experts: Dict[str, Any]) -> Dict[str, float]:
    bundles: Dict[str, Dict[str, Any]] = {}
    for name, exrow in experts.items():
        if not isinstance(exrow, dict):
            continue
        bundles[name] = expert_proxy_bundle(dataset, q, exrow)

    out: Dict[str, float] = {}
    add_family_comparative_features(out, "rag", RAG_EXPERTS, bundles)
    add_family_comparative_features(out, "no", NO_EXPERTS, bundles)
    return out


def main():
    out_rp = PRED_DIR / "features_retrieval_preview.jsonl"
    out_unc = PRED_DIR / "features_uncertainty.jsonl"

    out_rp.parent.mkdir(parents=True, exist_ok=True)

    total_selcmp_keys = set()

    with out_rp.open("w", encoding="utf-8") as rp_f, out_unc.open("w", encoding="utf-8") as un_f:
        for ds in DATASETS:
            p = PRED_DIR / f"router_train_{ds}.jsonl"
            if not p.exists():
                raise SystemExit(f"Missing {p} (did you run build_router_train.py?)")

            rows = load_jsonl(p)

            for r in rows:
                ex = r.get("experts", {}) or {}
                pred_raw = pick_rag_prediction_raw(ex, ds)
                docs = extract_docs_from_prediction_raw(pred_raw or "")

                q = r.get("question", "") or ""
                rid = r.get("id")

                qf = question_stats(q)
                rp, ru = ctx_stats(q, docs)
                selcmp = build_selector_comparative_features(ds, q, ex)

                total_selcmp_keys.update([k for k in selcmp if str(k).startswith(SELECTOR_CMP_PREFIX)])

                key = f"{ds}::{rid}"

                rp_row = {
                    "id": key,
                    "dataset": ds,
                    "orig_id": rid,
                    "features": {**qf, **rp},
                }

                un_row = {
                    "id": key,
                    "dataset": ds,
                    "orig_id": rid,
                    "features": {**ru, **selcmp},
                }

                rp_f.write(json.dumps(rp_row, ensure_ascii=False) + "\n")
                un_f.write(json.dumps(un_row, ensure_ascii=False) + "\n")

            print(f"[OK] built features for {ds}: {len(rows)} rows")

    print(f"\nWROTE:\n- {out_rp}\n- {out_unc}")
    print(f"[OK] selector comparative keys found: {len(total_selcmp_keys)}")
    for k in sorted(total_selcmp_keys):
        print(k)


if __name__ == "__main__":
    main()