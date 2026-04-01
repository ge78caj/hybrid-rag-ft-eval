# train_router_hotpotqa_mlp_embed.py
"""
Train an embedding-based multi-expert router on HotpotQA.

- Input: prediction/router_train_hotpotqa.jsonl
  (created by build_router_labels_hotpotqa.py)
- Features: question embeddings (SentenceTransformer).
- Labels: best_expert (one of 6 experts).
- Model: sklearn MLPClassifier (multi-class).
"""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit(
        "Please install sentence-transformers first:\n"
        "  pip install sentence-transformers"
    )

ROUTER_LABELS_PATH = Path("prediction/router_train_hotpotqa.jsonl")
MODEL_SAVE_PATH = Path("methods/multi_lora/router_hotpotqa_mlp_embed.joblib")


def load_router_data(path: Path) -> Tuple[List[str], List[str]]:
    """Load questions and best_expert labels from router_train_hotpotqa.jsonl."""
    questions: List[str] = []
    labels: List[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj["question"]
            best = obj["best_expert"]
            questions.append(q)
            labels.append(best)

    return questions, labels


def encode_questions(questions: List[str]) -> np.ndarray:
    """Encode questions into dense embeddings."""
    print(f"Encoding {len(questions)} questions into embeddings...")
    # You can change the model to something else if needed
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # (N, D) float32
    emb = model.encode(questions, batch_size=32, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)
    print(f"Embeddings shape: {emb.shape}")
    return emb


def train_mlp_router(X: np.ndarray, y: List[str]) -> None:
    """Train an MLP router and print performance on a held-out test set."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    clf = MLPClassifier(
        hidden_layer_sizes=(256,),
        activation="relu",
        solver="adam",
        max_iter=200,
        alpha=1e-4,
        random_state=42,
    )

    print("Training MLP router...")
    clf.fit(X_train, y_train)

    print("Evaluating on held-out test set...")
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # Save with joblib so we can later integrate into methods/multi_lora/router.py
    try:
        import joblib

        MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, MODEL_SAVE_PATH)
        print(f"Saved MLP router model to: {MODEL_SAVE_PATH}")
    except ImportError:
        print(
            "joblib not installed, skipping model save.\n"
            "Install with: pip install joblib"
        )


def main() -> None:
    if not ROUTER_LABELS_PATH.exists():
        raise SystemExit(
            f"Router labels not found at {ROUTER_LABELS_PATH}. "
            "Run build_router_labels_hotpotqa.py first."
        )

    questions, labels = load_router_data(ROUTER_LABELS_PATH)
    print(f"Loaded {len(questions)} examples with labels.")

    X = encode_questions(questions)
    train_mlp_router(X, labels)


if __name__ == "__main__":
    main()
