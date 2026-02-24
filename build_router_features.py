# build_router_features.py
import json
import re
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

PRED_DIR = Path("prediction")

# IMPORTANT: include commonsenseqa now
# Keep deterministic order (trainer/eval should use same dataset loop order).
DATASETS = ["hotpotqa", "squad_v2", "pubmedqa_v2", "commonsenseqa"]

DOC_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.DOTALL | re.IGNORECASE)

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "where", "why", "how", "what", "which", "who", "whom",
    "of", "to", "in", "on", "for", "with", "as", "by", "from", "at", "into", "about", "over", "after", "before", "between",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "done", "have", "has", "had",
    "this", "that", "these", "those", "it", "its", "their", "his", "her", "they", "them", "he", "she", "we", "you", "i",
}

def tokenize(s: str) -> List[str]:
    s = (s or "").lower()
    toks = re.findall(r"[a-z0-9]+", s)
    return [t for t in toks if t and t not in STOPWORDS]

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

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
    """
    returns:
      - retrieval_preview_feats (rp)
      - retrieval_uncertainty_feats (ru)
    """
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
    """
    Picks the most appropriate expert.prediction_raw to extract <DOCUMENT> context from.
    commonsenseqa only has base_only/base_rag.
    Others usually have base_rag/raft_rag/sft_rag.
    """
    ds = (ds or "").lower()

    if "commonsense" in ds:
        pref = ["base_rag"]
    else:
        pref = ["base_rag", "raft_rag", "sft_rag"]

    for k in pref:
        if k in experts and isinstance(experts[k], dict):
            pr = experts[k].get("prediction_raw")
            if pr:
                return pr
    return None

def main():
    out_rp = PRED_DIR / "features_retrieval_preview.jsonl"
    out_unc = PRED_DIR / "features_uncertainty.jsonl"

    out_rp.parent.mkdir(parents=True, exist_ok=True)

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

                key = f"{ds}::{rid}"
                rp_row = {"id": key, "dataset": ds, "orig_id": rid, "features": {**qf, **rp}}
                un_row = {"id": key, "dataset": ds, "orig_id": rid, "features": {**ru}}

                rp_f.write(json.dumps(rp_row, ensure_ascii=False) + "\n")
                un_f.write(json.dumps(un_row, ensure_ascii=False) + "\n")

            print(f"[OK] built features for {ds}: {len(rows)} rows")

    print(f"\nWROTE:\n- {out_rp}\n- {out_unc}")

if __name__ == "__main__":
    main()
