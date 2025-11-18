# Multi-LoRA RAFT (Hamza)

## 1. Motivation

In our project we compare:
- RAG,
- FT (LoRA),
- RAFT-style fine-tuning,
- and a new **multi-LoRA RAFT** method.

Recent work shows:
- Retrieval-augmented fine-tuning (RAFT/CRAFT) can teach models to use retrieved evidence efficiently.
- Multiple LoRA adapters can be treated as a pool of “experts” and selected or composed at inference time (e.g., LoraRetriever, LAG, MoLE, RAMoLE).
- Retrieval and generation should be optimized together (e.g., RA-DIT, Self-RAG).

Our idea is to extend RAFT with **multiple LoRA experts**, each specialized on a different dataset / skill, and to **route questions** to the most suitable LoRA at inference time.

## 2. High-Level Idea

We use:
- One shared **base model** (e.g. Qwen or Gemma, 7–8B range).
- Multiple **LoRA adapters** trained with RAFT-style data:
    - `lora_hotpotqa` for multi-hop reasoning.
    - `lora_popqa` for long-tail factual questions.
    - (optional) `lora_pubmedqa` for domain-specific biomedical QA.

At inference, given a question:
1. We run retrieval (RAG) to obtain relevant passages.
2. We select the most appropriate LoRA adapter (router).
3. We run RAFT-style inference with the selected adapter.

We then compare this method against single-RAFT and other baselines in a cost–performance map.

## 3. Phase A – Training LoRA Experts (RAFT-style)

For each dataset \(D \in \{\text{HotpotQA}, \text{PopQA}, \text{PubMedQA}\}\):

1. **Build RAFT-style training examples**:
    - For each question, retrieve top-k passages.
    - Input format (simplified):
      ```
      Question: <question>
      Context:
      <retrieved passages>
 
      Answer:
      ```
    - Target is the gold answer.

2. **Train a LoRA adapter on D**:
    - Freeze the base model.
    - Attach a LoRA head (e.g., on attention projections).
    - Fine-tune only the LoRA parameters on the RAFT-style data.

This yields dataset-specific experts:
- `lora_hotpotqa` (multi-hop expert),
- `lora_popqa` (long-tail expert),
- `lora_pubmedqa` (optional domain expert).

## 4. Phase B – LoRA Router (Selecting the Expert)

We consider two levels of routing:

### 4.1 Trivial router (per-dataset baseline)

When evaluating on a **single dataset**:
- HotpotQA → always use `lora_hotpotqa`.
- PopQA → always use `lora_popqa`.
- PubMedQA → always use `lora_pubmedqa`.

This gives a simple baseline: “multi-LoRAs without any input-aware routing”.

### 4.2 Input-aware router (embedding-based, LoraRetriever-style)

To handle mixed or unknown inputs, we design a simple input-aware router:

1. Choose an encoder (e.g., the same embedding model used for retrieval).
2. For each LoRA \(L_i\), build a **prototype embedding**:
    - Either (a) mean embedding of its training questions, or
    - (b) embedding of a short textual description of its domain (e.g., "multi-hop reasoning over Wikipedia" for HotpotQA).
3. For a new question:
    - Encode the question into a vector \(q\).
    - Compute similarity between \(q\) and each LoRA prototype.
    - Select the LoRA whose prototype is most similar to \(q\).

This gives a lightweight, LoraRetriever/LAG-style routing mechanism without complex per-token gating.

## 5. Phase C – Inference Pipeline

Given a new question:

1. **Retrieval**:
    - Encode the question and retrieve top-k passages from the corpus (FAISS or similar).

2. **Routing**:
    - Use either the trivial router (per-dataset) or the embedding-based router to choose one LoRA expert:
        - e.g., `lora_hotpotqa` for multi-hop-like questions,
        - or `lora_popqa` for short factual questions.

3. **RAFT-style generation**:
    - Construct the input:
      ```
      Question: <question>
      Context:
      <retrieved passages>
 
      Answer:
      ```
    - Run the base model with the selected LoRA adapter to generate the answer.

4. **Metric logging**:
    - Record EM/F1, latency, VRAM, train GPU-hours, and storage sizes using our `metrics.py` tool.

## 6. Comparison to Baselines

We will compare:

1. **FT-only (LoRA)**:
    - Single LoRA trained on merged data without retrieval.

2. **RAG-only**:
    - Base model with retrieval, no fine-tuning.

3. **Single RAFT-LoRA**:
    - One RAFT-style LoRA trained on merged RAFT data.

4. **Multi-LoRA RAFT (ours)**:
    - Multiple RAFT-trained LoRAs (HotpotQA / PopQA / PubMedQA),
    - With either dataset-based or input-aware routing.

We will place all methods on a cost–performance map (e.g., EM vs train GPU-hours / latency) to see under which constraints the multi-LoRA RAFT approach is the best choice.
