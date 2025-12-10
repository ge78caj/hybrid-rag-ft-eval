from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

from metrics import compute_em_f1, normalize_answer


# -------------------------------------------------------------------
# Config: adjust only if your paths / filenames differ
# -------------------------------------------------------------------
PRED_DIR = Path("prediction")

PRED_BASE_TRUE  = PRED_DIR / "normal_hotpotqa_True_predictions.jsonl"
PRED_BASE_FALSE = PRED_DIR / "normal_hotpotqa_False_predictions.jsonl"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def extract_pred_answer(raw_pred: str) -> str:
    """
    Extract the short answer string from the raw 'prediction' field.

    The raw string looks like a chat-style log, for example:

    user
    The following is a conversation ...
    ...
    model
    It’s uncertain, but likely both Americans.

    We keep only the text after the last 'model' line and normalize it.
    """
    if not isinstance(raw_pred, str):
        return ""

    parts = raw_pred.split("model", maxsplit=1)
    if len(parts) == 2:
        ans = parts[1]
    else:
        ans = raw_pred

    ans = ans.strip()
    return ans


def load_single_file(
        path: Path,
        expected_use_rag: bool,
) -> Tuple[List[Any], List[str], List[float], List[float]]:
    """
    Load one predictions file and filter by use_rag and ft_mode == 'normal'.

    Returns:
        gold_answers: list of gold answers (strings or list[str])
        predictions:  list of cleaned prediction strings
        latencies:    list of latency values in *seconds*
        vram_mb:      list of peak_vram_mb values (MB)
    """
    gold_answers: List[Any] = []
    predictions: List[str] = []
    latencies: List[float] = []
    vram_mb: List[float] = []

    if not path.exists():
        print(f"[WARN] File not found: {path}")
        return gold_answers, predictions, latencies, vram_mb

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Could not parse line in {path.name}: {line[:80]}...")
                continue

            # Filter to the correct variant
            if obj.get("use_rag") is not expected_use_rag:
                continue
            if obj.get("ft_mode") != "normal":
                continue

            raw_pred = obj.get("prediction", "")
            gold = obj.get("gold_answer", "")

            cleaned_pred = normalize_answer(extract_pred_answer(raw_pred))

            gold_answers.append(gold)
            predictions.append(cleaned_pred)

            # Latency in seconds
            t = obj.get("time", None)
            if isinstance(t, (int, float)):
                latencies.append(float(t))

            # Peak VRAM (MB)
            v = obj.get("peak_vram_mb", None)
            if isinstance(v, (int, float)):
                vram_mb.append(float(v))

    return gold_answers, predictions, latencies, vram_mb


def load_both_files(
        path_true: Path,
        path_false: Path,
        expected_use_rag: bool,
) -> Tuple[List[Any], List[str], List[float], List[float]]:
    """
    Convenience: combine *_True_predictions.jsonl and *_False_predictions.jsonl
    for a given expected_use_rag flag.
    """
    gold_all: List[Any] = []
    pred_all: List[str] = []
    lat_all: List[float] = []
    vram_all: List[float] = []

    for path in (path_true, path_false):
        g, p, l, v = load_single_file(path, expected_use_rag)
        gold_all.extend(g)
        pred_all.extend(p)
        lat_all.extend(l)
        vram_all.extend(v)

    return gold_all, pred_all, lat_all, vram_all


def summarize_latency(latencies_s: List[float]) -> Dict[str, float]:
    """
    Given a list of latencies in seconds, return avg and p95 in milliseconds.
    """
    if not latencies_s:
        return {"avg_ms": 0.0, "p95_ms": 0.0}

    lat_ms = sorted(t * 1000.0 for t in latencies_s)
    n = len(lat_ms)
    avg_ms = sum(lat_ms) / n

    # p95 index: 0-based
    p95_index = int(0.95 * (n - 1))
    p95_ms = lat_ms[p95_index]

    return {"avg_ms": avg_ms, "p95_ms": p95_ms}


def summarize_vram(vram_list: List[float]) -> Dict[str, float]:
    """
    Summarize peak_vram_mb values.
    """
    if not vram_list:
        return {"avg_mb": 0.0, "max_mb": 0.0}

    n = len(vram_list)
    avg_mb = sum(vram_list) / n
    max_mb = max(vram_list)
    return {"avg_mb": avg_mb, "max_mb": max_mb}


def print_report(
        name: str,
        gold_answers: List[Any],
        predictions: List[str],
        latencies_s: List[float],
        vram_mb: List[float],
) -> None:
    """
    Print EM/F1 plus latency + VRAM stats for one method.
    Handles gold answers that are either strings or list[str].
    """
    # Convert gold answers to single strings so metrics.compute_em_f1 is happy
    gold_single: List[str] = []
    for g in gold_answers:
        if isinstance(g, list):
            if len(g) > 0:
                gold_single.append(str(g[0]))
            else:
                gold_single.append("")
        else:
            gold_single.append(str(g))

    result = compute_em_f1(gold_single, predictions)
    em = result["em"] * 100.0
    f1 = result["f1"] * 100.0

    lat_stats = summarize_latency(latencies_s)
    vram_stats = summarize_vram(vram_mb)

    print(f"=== {name} ===")
    print(f"EM: {em:.2f}%")
    print(f"F1: {f1:.2f}%")
    print(f"Avg latency: {lat_stats['avg_ms']:.1f} ms")
    print(f"p95 latency: {lat_stats['p95_ms']:.1f} ms")
    print(f"Avg peak VRAM: {vram_stats['avg_mb']:.1f} MB")
    print(f"Max peak VRAM: {vram_stats['max_mb']:.1f} MB")
    print()


def main() -> None:
    # Base + RAG
    gold_rag, pred_rag, lat_rag, vram_rag = load_both_files(
        PRED_BASE_TRUE,
        PRED_BASE_FALSE,
        expected_use_rag=True,
    )
    print_report("base+RAG", gold_rag, pred_rag, lat_rag, vram_rag)

    # Base only (no RAG)
    gold_base, pred_base, lat_base, vram_base = load_both_files(
        PRED_BASE_TRUE,
        PRED_BASE_FALSE,
        expected_use_rag=False,
    )
    print_report("base_only", gold_base, pred_base, lat_base, vram_base)


if __name__ == "__main__":
    main()
