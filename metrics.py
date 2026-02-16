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


def is_gold_no_answer(gold: str) -> bool:
    # gold はこのパイプラインではすでに文字列になってる前提
    return gold == "NO_ANSWER"


def is_pred_no_answer(pred: str) -> bool:
    pred_norm = normalize_answer(pred)
    pred_tokens = pred_norm.split()
    return (
        pred_norm == "" or
        pred_norm == "no answer" or
        pred_norm == "noanswer" or
        (len(pred_tokens) >= 2 and pred_tokens[0] == "no" and pred_tokens[1] == "answer")
    )


def compute_em_f1_on_gold_answerable_only(
    gold_answers: List[str],
    predictions: List[str],
) -> Dict[str, float]:
    """
    GoldがANSWERABLEの例だけに絞って EM/F1 を計算。
    ＝「答えるべき問題での純粋なQA性能」
    """
    assert len(gold_answers) == len(predictions), "Lengths must match"

    gold_sub, pred_sub = [], []
    for g, p in zip(gold_answers, predictions):
        if not is_gold_no_answer(g):
            gold_sub.append(g)
            pred_sub.append(p)

    if len(gold_sub) == 0:
        return {"em_answerable_only": 0.0, "f1_answerable_only": 0.0, "n_answerable": 0}

    out = compute_em_f1(gold_sub, pred_sub)
    return {
        "em_answerable_only": out["em"],
        "f1_answerable_only": out["f1"],
        "n_answerable": len(gold_sub),
    }


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


def _is_sublist(gold_tokens: List[str], pred_tokens: List[str]) -> bool:
    """
    Check if gold_tokens appears as a consecutive subsequence in pred_tokens.
    
    Examples:
        gold=["no"], pred=["the", "answer", "is", "no"] -> True
        gold=["no"], pred=["november"] -> False
        gold=["no", "way"], pred=["there", "is", "no", "way", "out"] -> True
        gold=["no", "way"], pred=["way", "to", "say", "no"] -> False
    """
    if not gold_tokens:
        return True
    if not pred_tokens:
        return False
    
    gold_len = len(gold_tokens)
    pred_len = len(pred_tokens)
    
    if gold_len > pred_len:
        return False
    
    # Sliding window to check if gold_tokens appears consecutively in pred_tokens
    for i in range(pred_len - gold_len + 1):
        if pred_tokens[i:i + gold_len] == gold_tokens:
            return True
    
    return False


def compute_partial_match_em(
        gold_answers: List[str],
        predictions: List[str],
) -> Dict[str, float]:
    """
    Compute Partial Match EM: checks if gold answer tokens appear as a 
    consecutive subsequence in prediction tokens (preserving order).
    
    This avoids false positives like "no" matching "november" by using 
    word boundaries (tokenization).
    """
    assert len(gold_answers) == len(predictions), "Lengths must match"

    total = len(gold_answers)
    partial_match_count = 0

    for gold, pred in zip(gold_answers, predictions):
        gold_tokens = normalize_answer(gold).split()
        pred_tokens = normalize_answer(pred).split()

        if _is_sublist(gold_tokens, pred_tokens):
            partial_match_count += 1

    partial_em = partial_match_count / total if total > 0 else 0.0

    return {
        "partial_em": partial_em,
    }


# ---------- Classification metrics ----------

def compute_classification_metrics(
        gold_labels: List[str],
        pred_labels: List[str],
        target_classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute classification metrics including confusion matrix,
    per-class precision/recall/f1, and overall metrics.
    
    Args:
        gold_labels: List of ground truth labels
        pred_labels: List of predicted labels
        target_classes: Optional list of class names. If None, inferred from data.
    
    Returns:
        Dict with accuracy, macro_f1, weighted_f1, per_class metrics, and confusion_matrix
    """
    assert len(gold_labels) == len(pred_labels), "Lengths must match"
    
    # Determine classes
    if target_classes is None:
        target_classes = sorted(list(set(gold_labels + pred_labels)))
    
    class_to_idx = {cls: idx for idx, cls in enumerate(target_classes)}
    n_classes = len(target_classes)
    
    # Initialize confusion matrix
    confusion_matrix = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    
    # Build confusion matrix
    for gold, pred in zip(gold_labels, pred_labels):
        gold_idx = class_to_idx.get(gold, -1)
        pred_idx = class_to_idx.get(pred, -1)
        if gold_idx >= 0 and pred_idx >= 0:
            confusion_matrix[gold_idx][pred_idx] += 1
    
    # Compute per-class metrics
    per_class_metrics = {}
    f1_scores = []
    supports = []
    
    for idx, cls in enumerate(target_classes):
        # True Positives, False Positives, False Negatives
        tp = confusion_matrix[idx][idx]
        fp = sum(confusion_matrix[i][idx] for i in range(n_classes) if i != idx)
        fn = sum(confusion_matrix[idx][i] for i in range(n_classes) if i != idx)
        support = sum(confusion_matrix[idx])
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        }
        
        f1_scores.append(f1)
        supports.append(support)
    
    # Overall metrics
    total_correct = sum(confusion_matrix[i][i] for i in range(n_classes))
    total_samples = sum(supports)
    accuracy = total_correct / total_samples
    
    # Macro F1: simple average
    macro_f1 = sum(f1_scores) / len(f1_scores)
    
    # Weighted F1: weighted by support
    weighted_f1 = sum(f1 * sup for f1, sup in zip(f1_scores, supports)) / total_samples
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class_metrics,
        "confusion_matrix": confusion_matrix
    }


def compute_no_answer_detection(
        gold_answers: List[str],
        predictions: List[str],
) -> Dict[str, Any]:
    """
    Evaluate NO_ANSWER detection as a binary classification task.
    Uses the same normalization as compute_em_f1 for consistency.
    
    Args:
        gold_answers: List of gold answer strings (or lists containing "NO_ANSWER")
        predictions: List of predicted answer strings
    
    Returns:
        Classification metrics for NO_ANSWER vs ANSWERABLE
    """
    # Convert to binary labels
    gold_labels = []
    pred_labels = []
    
    for gold, pred in zip(gold_answers, predictions):
        # Handle gold answer (might be list or string)
        if isinstance(gold, list):
            gold_str = gold[0] if gold else ""
        else:
            gold_str = gold
        
        # Normalize using the same function as EM/F1
        pred_norm = normalize_answer(pred)
        
        # Determine labels
        gold_label = "NO_ANSWER" if gold_str == "NO_ANSWER" else "ANSWERABLE"
        
        # Check if prediction is NO_ANSWER (using same normalization approach)
        # Empty after normalization or explicit "no answer" tokens
        pred_tokens = pred_norm.split()
        is_no_answer = (
            pred_norm == "" or 
            pred_norm == "no answer" or
            pred_norm == "noanswer" or
            (len(pred_tokens) >= 2 and pred_tokens[0] == "no" and pred_tokens[1] == "answer")
        )
        pred_label = "NO_ANSWER" if is_no_answer else "ANSWERABLE"
        
        gold_labels.append(gold_label)
        pred_labels.append(pred_label)
    
    return compute_classification_metrics(
        gold_labels=gold_labels,
        pred_labels=pred_labels,
        target_classes=["NO_ANSWER", "ANSWERABLE"]
    )


def compute_yes_no_maybe_classification(
        gold_answers: List[str],
        predictions: List[str],
) -> Dict[str, Any]:
    """
    Evaluate yes/no/maybe classification (for datasets like PubMedQA).
    Uses the same normalization as compute_em_f1 for consistency.
    
    Args:
        gold_answers: List of gold answer strings (or lists)
        predictions: List of predicted answer strings
    
    Returns:
        Classification metrics for yes/no/maybe
    """
    # Convert to labels
    gold_labels = []
    pred_labels = []
    
    for gold, pred in zip(gold_answers, predictions):
        # Handle gold answer (might be list or string)
        if isinstance(gold, list):
            gold_str = gold[0] if gold else ""
        else:
            gold_str = gold
        
        # Normalize using the same function as EM/F1
        gold_norm = normalize_answer(gold_str)
        pred_norm = normalize_answer(pred)
        pred_tokens = pred_norm.split()
        
        # Determine gold label (should be yes/no/maybe)
        gold_label = gold_norm if gold_norm in ["yes", "no", "maybe"] else "unknown"
        
        # Determine pred label: check exact match first, then token-level match
        if pred_norm in ["yes", "no", "maybe"]:
            # Exact match
            pred_label = pred_norm
        elif "yes" in pred_tokens:
            pred_label = "yes"
        elif "no" in pred_tokens:
            pred_label = "no"
        elif "maybe" in pred_tokens:
            pred_label = "maybe"
        else:
            pred_label = "unknown"
        
        gold_labels.append(gold_label)
        pred_labels.append(pred_label)
    
    return compute_classification_metrics(
        gold_labels=gold_labels,
        pred_labels=pred_labels,
        target_classes=["yes", "no"]
    )


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
    Automatically adds classification metrics based on dataset type.
    """
    metrics: Dict[str, Any] = {}

    # 1) Performance metrics
    perf = compute_em_f1(gold_answers, predictions)
    metrics.update(perf)
    
    # 1b) Partial Match EM
    partial = compute_partial_match_em(gold_answers, predictions)
    metrics.update(partial)
    
    # 1c) Classification metrics (dataset-specific)
    if dataset_name:
        if "squad" in dataset_name.lower():
            # SQuAD v2: NO_ANSWER detection
            classification = compute_no_answer_detection(gold_answers, predictions)
            metrics["classification"] = classification
            metrics.update(compute_em_f1_on_gold_answerable_only(gold_answers, predictions))

        elif "pubmedqa" in dataset_name.lower():
            # PubMedQA: yes/no/maybe classification
            classification = compute_yes_no_maybe_classification(gold_answers, predictions)
            metrics["classification"] = classification

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
    if "<ANSWER>:" in raw:
        return raw.rsplit("<ANSWER>:", 1)[1].strip()
    return raw.split("assistant\n\n", 1)[1].strip() if "assistant\n\n" in raw else raw.strip()


def main():
    """
    Update existing results files with classification metrics for SQuAD v2 and PubMedQA.
    """
    # SQuAD v2 files
    squad_files = [
        # "prediction/ft_squad_v23500_False_Llama-3.2-1B-Instruct_predictions_add_unique_prompt.jsonl",
        # "prediction/ft_squad_v23500_True_Llama-3.2-1B-Instruct_predictions_add_unique_prompt.jsonl",
        # "prediction/raft_squad_v23500_False_Llama-3.2-1B-Instruct_predictions_add_unique_prompt.jsonl",
        # "prediction/raft_squad_v23500_True_Llama-3.2-1B-Instruct_predictions_add_unique_prompt.jsonl",
        # "prediction/normal_squad_v23500_False_Llama-3.2-1B-Instruct_predictions_add_unique_prompt.jsonl",
        # "prediction/normal_squad_v23500_True_Llama-3.2-1B-Instruct_predictions_add_unique_prompt.jsonl",
    ]
    
    # PubMedQA files
    pubmedqa_files = [
        # "prediction/ft_pubmedqa_v23500_False_Llama-3.2-1B-Instruct_predictions_add_unique_prompt.jsonl",
        # "prediction/ft_pubmedqa_v23500_True_Llama-3.2-1B-Instruct_predictions_add_unique_prompt.jsonl",
        # "prediction/raft_pubmedqa_v23500_False_Llama-3.2-1B-Instruct_predictions_add_unique_prompt.jsonl",
        # "prediction/raft_pubmedqa_v23500_True_Llama-3.2-1B-Instruct_predictions_add_unique_prompt.jsonl",
        # "prediction/normal_pubmedqa_v23500_False_Llama-3.2-1B-Instruct_predictions_add_unique_prompt.jsonl",
        # "prediction/normal_pubmedqa_v23500_True_Llama-3.2-1B-Instruct_predictions_add_unique_prompt.jsonl",
    ]
    
    hotpotqa_files = [
        # "prediction/ft_hotpotqa3500_False_Llama-3.2-1B-Instruct_predictions_add_unique_prompt.jsonl",
        # "prediction/ft_hotpotqa3500_True_Llama-3.2-1B-Instruct_predictions_add_unique_prompt.jsonl",
        # "prediction/raft_hotpotqa3500_False_Llama-3.2-1B-Instruct_predictions_add_unique_prompt.jsonl",
        # "prediction/raft_hotpotqa3500_True_Llama-3.2-1B-Instruct_predictions_add_unique_prompt.jsonl",
        # "prediction/normal_hotpotqa3500_False_Llama-3.2-1B-Instruct_predictions_add_unique_prompt.jsonl",
        # "prediction/normal_hotpotqa3500_True_Llama-3.2-1B-Instruct_predictions_add_unique_prompt.jsonl",
    ]
    
    commonsenseqa_files = [
        "prediction/normal_commonsenseqa3500_False_Llama-3.2-1B-Instruct_predictions_add_unique_prompt.jsonl",
        "prediction/normal_commonsenseqa3500_True_Llama-3.2-1B-Instruct_predictions_add_unique_prompt.jsonl",
    ]
    
    all_files = squad_files + pubmedqa_files + hotpotqa_files + commonsenseqa_files
    
    for file in all_files:
        if not os.path.exists(file):
            print(f"Skipping {file} (not found)")
            continue
        
        print(f"Processing {file}...")
        
        # Determine dataset name
        if "squad" in file.lower():
            dataset_name = "squad_v2"
        elif "pubmedqa" in file.lower():
            dataset_name = "pubmedqa_v2"
        elif "commonsenseqa" in file.lower():
            dataset_name = "commonsenseqa"
        else:
            dataset_name = None
        
        # Load predictions
        gold_answers, predictions, latencies, vram = [], [], [], []
        for sample in load_jsonl(file):
            # Handle gold_answer (might be list or string)
            gold_ans = sample["gold_answer"]
            if isinstance(gold_ans, list):
                gold_ans = gold_ans[0]
            gold_answers.append(gold_ans)
            predictions.append(extract_prediction(sample["prediction"]))
            latencies.append(sample["time"] * 1000.0)
            vram.append(sample["peak_vram_mb"])
        
        # Determine result file paths
        base_name = os.path.basename(file).replace(".jsonl", "")
        
        # Find the correct output path (check multiple possible locations)
        possible_paths = [
            f"results/Llama-3.2-1B/{dataset_name}/{base_name}.json",
            f"results/{base_name}.json",
        ]
        
        result_json_path = None
        result_csv_path = None
        
        for path in possible_paths:
            if os.path.exists(path):
                result_json_path = path
                result_csv_path = path.replace(".json", ".csv")
                break
        
        if result_json_path is None:
            print(f"  Warning: No existing results file found for {file}")
            # Create new metrics
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
                dataset_name=dataset_name,
            )
            # Save to default location
            default_json = f"results/Llama-3.2-1B/{dataset_name}/{base_name}.json"
            default_csv = f"results/Llama-3.2-1B/{dataset_name}/{base_name}.csv"
            save_metrics_json(metrics, default_json)
            save_metrics_csv(metrics, default_csv)
            print(f"  Created new results at {default_json}")
        else:
            # Load existing metrics
            with open(result_json_path, "r", encoding="utf-8") as f:
                existing_metrics = json.load(f)
            
            # Compute classification metrics
            if dataset_name == "squad_v2":
                classification = compute_no_answer_detection(gold_answers, predictions)
            elif dataset_name == "pubmedqa_v2":
                classification = compute_yes_no_maybe_classification(gold_answers, predictions)
            else:
                classification = None
            
            # Add classification metrics to existing results
            if classification:
                existing_metrics["classification"] = classification
                if dataset_name == "squad_v2":
                    existing_metrics.update(
                        compute_em_f1_on_gold_answerable_only(gold_answers, predictions)
                    )
                # Save updated metrics
                save_metrics_json(existing_metrics, result_json_path)
                save_metrics_csv(existing_metrics, result_csv_path)
                print(f"  Updated {result_json_path} with classification metrics")
                print(f"    Accuracy: {classification['accuracy']:.4f}")
                print(f"    Macro F1: {classification['macro_f1']:.4f}")

if __name__ == "__main__":
    main()
