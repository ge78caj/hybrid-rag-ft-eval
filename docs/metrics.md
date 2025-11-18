# Metrics for Cost–Performance Evaluation

This document defines the metrics we will use to compare **RAG**, **FT (LoRA)**, **RAFT**, and our **multi-LoRA RAFT** method on QA tasks.

We group metrics into: **task performance**, **efficiency (runtime)**, **resource usage**, **cost proxies**, and (optionally) **retrieval quality**.

---

## 1. Task Performance (QA)

**Goal:** How good are the answers?

- **EM (Exact Match)**  
  Percentage of predictions that exactly match the gold answer, after:
    - lowercasing,
    - trimming whitespace,
    - removing simple punctuation.

- **F1**  
  Token-level F1 between prediction and gold answer.
    - Split both prediction and gold into tokens (e.g. by whitespace).
    - Compute precision, recall, and F1.
    - Average F1 over all examples.

*Usage:*  
For each dataset (HotpotQA, PopQA, etc.), we will report EM and F1. EM can be used as the main performance value on the Pareto plots.

Optional (HotpotQA only, if time allows):
- **Supporting fact accuracy**: percentage of examples where the predicted support sentences match the annotated support facts.

---

## 2. Efficiency (Runtime)

**Goal:** How fast is the system per query?

- **latency_avg_ms**  
  Average end-to-end latency per query in milliseconds, including:
    - retrieval (for RAG/RAFT/multi-LoRA),
    - model inference.

- **latency_p95_ms**  
  95th percentile latency per query.  
  This approximates “worst-case” latency seen by most users.

- **throughput_qps** (optional)  
  Questions per second (Q/s) when running batched evaluation on GPU.

**How to measure:**
- For each query:
    - record `start = time.time()`,
    - run the full pipeline,
    - record `end = time.time()`,
    - store `latency_ms = (end - start) * 1000`.
- After evaluation:
    - `latency_avg_ms` = mean of all latencies,
    - `latency_p95_ms` = 95th percentile of all latencies.
- For throughput (optional), measure total time for N queries and compute `qps = N / total_time_seconds`.

---

## 3. Resource Usage (Memory & Storage)

**Goal:** How heavy is the system in terms of VRAM and disk?

- **peak_vram_mb**  
  Maximum GPU memory (in MB) used during training or evaluation.

  *Measurement:*
    - Before run: `torch.cuda.reset_peak_memory_stats()`
    - After run: `peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)`

- **params_total_m**  
  Total number of parameters (in millions) in the base model.

- **params_trainable_m**  
  Number of *trainable* parameters (in millions) for the method:
    - full FT: often equals `params_total_m`,
    - LoRA/RAFT: much smaller (only adapter parameters).

- **storage_model_mb**  
  Size (in MB) of the base model checkpoints on disk.

- **storage_adapters_mb**  
  Total size (in MB) of all LoRA adapter files (for FT/RAFT/multi-LoRA).

- **storage_index_mb**  
  Total size (in MB) of the retriever index (for RAG/RAFT/multi-LoRA).

*Measurement example (per file):*
```python
import os
size_mb = os.path.getsize(path) / (1024 * 1024)
