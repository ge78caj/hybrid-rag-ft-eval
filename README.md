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

## Notes on Execution

The three notebooks above are primarily intended to be run on Google Colab, and execution in other environments is not guaranteed.  
To reproduce the experiments, the paths inside the notebooks need to be adjusted to match the local environment.  
The notebooks also require credentials such as `wandb` and OpenAI API keys.  
In addition, dataset names, output paths, and training settings and so on are switched manually across cells, so the configuration should be updated depending on the target dataset and experiment.