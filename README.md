## Notebook Overview

### `RAFT_Dataset_Generation.ipynb`

This notebook is used to construct datasets for RAFT training.  
It organizes each benchmark dataset into the RAFT format, including the question, candidate documents, oracle context, and gold labels.  
It also supports generating CoT-style answers, building the RAG corpus, and creating embeddings when needed.  
The outputs of this notebook are the preprocessed datasets which are available [here](https://drive.google.com/drive/folders/1waIVN6rZ0XuYOo_SiqGDQZrc7LhG-pQS?usp=sharing)
These datasets are then used in the later stages of training, inference, and evaluation.

### `RAFT_Model_Finetune_and_Inference.ipynb`

This notebook is used to fine-tune the model on the constructed datasets and run eavaluation.  
It supports settings such as LoRA and quantization, and enables comparisons across multiple conditions, including RAFT training, standard fine-tuning, and the original pretrained model.  
It also compares answer quality with and without RAG.

### `calculate_rag_hit_ratio.ipynb`

This notebook calculates whether the documents retrieved by RAG contain the oracle context required to answer each question.  
It is used as a simple auxiliary evaluation of retrieval quality.

### Repository Structure

This repository has two main parts.

### 1. Notebook-based model and dataset pipeline
These notebooks are mainly used for dataset construction, fine-tuning, inference, and retrieval analysis:

- `RAFT_Dataset_Generation.ipynb`
- `RAFT_Model_Finetune_and_Inference.ipynb`
- `calculate_rag_hit_ratio.ipynb`

These are mainly intended for Google Colab and are used to build the expert predictions that later become inputs to the routing pipeline.

### 2. Python-based routing and evaluation pipeline
Once the expert predictions already exist in `prediction/`, the routing experiments can be reproduced with the main Python scripts:

- `build_router_features.py`  
  Regenerates the router feature files, including the selector-comparative `selcmp__` features.

- `train_router_two_stage.py`  
  Trains the current two-stage router: gate + selectors.

- `eval_router_two_stage.py`  
  Evaluates the trained router offline and reports routing behavior, utility, F1, EM, and oracle comparisons.

- `eval_router_costs.py`  
  Reports routed average latency and the corresponding oracle cost statistics.

- `find_best_fixed_expert.py`  
  Computes the best single fixed expert baseline for each dataset.

You do **not** need to run `build_router_train.py` unless the files  
`prediction/router_train_*.jsonl` are missing or outdated.

---

## Recommended Reproduction Order

For the routing part of the project, the recommended order is:

1. Regenerate router features
2. Train the router for each dataset
3. Evaluate the router
4. Run oracle ablations
5. Compare against the best fixed expert
6. Export routed latency / cost summaries

---

## Router Reproduction Commands

### Step 0: optional environment setup

```bash
git clone https://github.com/ge78caj/hybrid-rag-ft-eval.git
cd hybrid-rag-ft-eval
pip install torch torchvision torchaudio sentence-transformers transformers scikit-learn numpy pandas tqdm


##Train commands:
#Squad_V2:
python train_router_two_stage.py --only squad_v2 --out_dir results/router_alllearned_gatecls_d1e4 --epochs 40 --batch_size 128 --tradeoff_mode --feature_files "prediction/features_retrieval_preview.jsonl,prediction/features_uncertainty.jsonl" --standardize_features --gate_objective cls --gate_delta 0.0001 --selector_objective hard_ce --hidden_dim 512 --dropout 0.1 --selector_constant_margin 0.002 --print_target_stats
#HotpotQA:
python train_router_two_stage.py --only hotpotqa --out_dir results/router_alllearned_gatecls_d1e4 --epochs 40 --batch_size 128 --tradeoff_mode --feature_files "prediction/features_retrieval_preview.jsonl,prediction/features_uncertainty.jsonl" --standardize_features --gate_objective cls --gate_delta 0.0001 --selector_objective soft_ce --sel_use_margin_weighting --sel_margin_scale 0.2 --sel_weight_min 0.2 --sel_weight_max 2.0 --hidden_dim 512 --dropout 0.1 --selector_constant_margin 0.002 --print_target_stats
#PubmedQA_V2:
python train_router_two_stage.py --only pubmedqa_v2 --out_dir results/router_alllearned_gatecls_d1e4 --epochs 40 --batch_size 128 --tradeoff_mode --feature_files "prediction/features_retrieval_preview.jsonl,prediction/features_uncertainty.jsonl" --standardize_features --gate_objective cls --gate_delta 0.0001 --selector_objective hard_ce --hidden_dim 512 --dropout 0.1 --selector_constant_margin 0.002 --print_target_stats               
#CommonsenseQA:
python train_router_two_stage.py --only commonsenseqa --out_dir results/router_alllearned_gatecls_d1e4 --epochs 40 --batch_size 128 --tradeoff_mode --feature_files "prediction/features_retrieval_preview.jsonl,prediction/features_uncertainty.jsonl" --standardize_features --gate_objective cls --gate_delta 0.0001 --hidden_dim 512 --dropout 0.1 --selector_constant_margin 0.002 --print_target_stats                                                                                                                    

##Eval command:
python eval_router_two_stage.py --only "DATASET_NAME" --model_dir results/router_alllearned_gatecls_d1e4 --tradeoff_mode --oracle_policy_aligned --disable_shared_gate --pubmed_policy none --feature_files "prediction/features_retrieval_preview.jsonl,prediction/features_uncertainty.jsonl" --standardize_features --hidden_dim 512 --dropout 0.1    

