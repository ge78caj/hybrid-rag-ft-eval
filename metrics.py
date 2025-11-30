import json
import csv
import math
import os
from typing import List, Dict, Any, Optional


# ---------- Text normalization helpers ----------

def normalize_answer(s: str) -> str:
    """
    Normalize text for EM/F1:
    - lowercase
    - strip leading/trailing spaces
    - remove simple punctuation
    """
    import re
    s = s.lower().strip()
    # remove punctuation (keep letters, numbers, spaces)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    # collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s


# ---------- EM / F1 computation ----------

def _f1_single(gold: str, pred: str) -> float:
    """
    Compute token-level F1 for a single (gold, pred) pair.
    """
    gold_tokens = normalize_answer(gold).split()
    pred_tokens = normalize_answer(pred).split()

    if len(gold_tokens) == 0 and len(pred_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = set(gold_tokens) & set(pred_tokens)
    num_common = sum(
        min(gold_tokens.count(t), pred_tokens.count(t)) for t in common
    )

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def compute_em_f1(
        gold_answers: List[str],
        predictions: List[str],
) -> Dict[str, float]:
    """
    Compute EM and F1 over a list of examples.
    Assumes gold_answers[i] corresponds to predictions[i].
    """
    assert len(gold_answers) == len(predictions), "Lengths must match"

    total = len(gold_answers)
    em_count = 0
    f1_sum = 0.0

    for gold, pred in zip(gold_answers, predictions):
        gold_norm = normalize_answer(gold)
        pred_norm = normalize_answer(pred)

        if gold_norm == pred_norm:
            em_count += 1

        f1_sum += _f1_single(gold, pred)

    em = em_count / total if total > 0 else 0.0
    f1 = f1_sum / total if total > 0 else 0.0

    return {
        "em": em,
        "f1": f1,
    }


# ---------- Latency / throughput ----------

def compute_latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
    """
    Compute average and p95 latency from a list of per-query latencies (in ms).
    """
    if not latencies_ms:
        return {
            "latency_avg_ms": 0.0,
            "latency_p95_ms": 0.0,
            "throughput_qps": 0.0,
        }

    latencies_sorted = sorted(latencies_ms)
    n = len(latencies_sorted)

    latency_avg = sum(latencies_sorted) / n
    # p95 index: 0-based
    idx_p95 = min(n - 1, int(math.ceil(0.95 * n)) - 1)
    latency_p95 = latencies_sorted[idx_p95]

    total_time_sec = sum(latencies_sorted) / 1000.0
    throughput_qps = n / total_time_sec if total_time_sec > 0 else 0.0

    return {
        "latency_avg_ms": latency_avg,
        "latency_p95_ms": latency_p95,
        "throughput_qps": throughput_qps,
    }


# ---------- Storage helpers ----------

def get_total_size_mb(paths: List[str]) -> float:
    """
    Sum file sizes in MB for the given list of paths.
    Ignores paths that do not exist.
    """
    total_bytes = 0
    for p in paths:
        if os.path.exists(p):
            total_bytes += os.path.getsize(p)
    return total_bytes / (1024 * 1024)


# ---------- Main entry point: aggregate all metrics ----------

def compute_all_metrics(
        *,
        gold_answers: List[str],
        predictions: List[str],
        latencies_ms: List[float],
        peak_vram_mb: float,
        params_total_m: float,
        params_trainable_m: float,
        storage_model_mb: float,
        storage_adapters_mb: float,
        storage_index_mb: float,
        train_gpu_hours: float,
        num_gpus: int = 1,
        method_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute a unified metrics dict for one run of a given method on a given dataset.
    """
    metrics: Dict[str, Any] = {}

    # 1) Performance metrics
    perf = compute_em_f1(gold_answers, predictions)
    metrics.update(perf)

    # 2) Latency / throughput
    latency = compute_latency_stats(latencies_ms)
    metrics.update(latency)

    # 3) Resource usage
    metrics["peak_vram_mb"] = peak_vram_mb
    metrics["params_total_m"] = params_total_m
    metrics["params_trainable_m"] = params_trainable_m

    # 4) Storage
    metrics["storage_model_mb"] = storage_model_mb
    metrics["storage_adapters_mb"] = storage_adapters_mb
    metrics["storage_index_mb"] = storage_index_mb

    # 5) Cost proxies
    metrics["train_gpu_hours"] = train_gpu_hours
    metrics["num_gpus"] = num_gpus
    # inference cost proxy (approx)
    metrics["inference_gpu_seconds_per_query"] = (
            latency["latency_avg_ms"] / 1000.0
    )

    # 6) Identifiers
    if method_name is not None:
        metrics["method_name"] = method_name
    if dataset_name is not None:
        metrics["dataset_name"] = dataset_name

    # 7) Extra info (e.g. model id, config hash, etc.)
    if extra_info:
        metrics.update(extra_info)

    return metrics


# ---------- Saving helpers ----------

def save_metrics_json(metrics: Dict[str, Any], path: str) -> None:
    """
    Save metrics as a JSON file.
    """
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def save_metrics_csv(metrics: Dict[str, Any], path: str) -> None:
    """
    Save metrics as a single-row CSV file.
    """
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    fieldnames = list(metrics.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(metrics)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def extract_prediction(raw):
    return raw.split("model\n", 1)[1].strip() if "model\n" in raw else raw.strip()

def main():
    files = "prediction/normal_hotpotqa_True_predictions.jsonl"
    gold_answers, predictions, latencies, vram = [], [], [], []


    for sample in load_jsonl(files):
        gold_answers.append(sample["gold_answer"][0])
        predictions.append(extract_prediction(sample["prediction"]))
        latencies.append(sample["time"] * 1000.0)
        vram.append(sample["peak_vram_mb"])

    metrics = compute_all_metrics(
        gold_answers=gold_answers,
        predictions=predictions,
        latencies_ms=latencies,
        peak_vram_mb=max(vram) if vram else 0.0,
        params_total_m=0,          
        params_trainable_m=0,       
        storage_model_mb=0,        
        storage_adapters_mb=0,      
        storage_index_mb=0,         
        train_gpu_hours=0,            
        num_gpus=1,
        dataset_name="hotpotqa",
    )

    save_metrics_json(metrics, f"results/{files.split('/')[-1].split('.')[0]}.json")
    save_metrics_csv(metrics, f"results/{files.split('/')[-1].split('.')[0]}.csv")

if __name__ == "__main__":
    main()