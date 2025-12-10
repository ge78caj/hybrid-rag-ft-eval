import json
from pathlib import Path
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

DATA_PATH = Path("prediction/router_train_hotpotqa.jsonl")
MODEL_PATH = Path("methods/multi_lora/router_hotpotqa_tfidf.joblib")


def load_router_data(path: Path):
    questions: List[str] = []
    labels: List[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            questions.append(obj["question"])
            labels.append(obj["best_expert"])

    return questions, labels


def main():
    print(f"Loading router training data from: {DATA_PATH}")
    X, y = load_router_data(DATA_PATH)
    print(f"Loaded {len(X)} examples.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training router model (TF-IDF + LinearSVC, multi-class)...")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),      # unigrams + bigrams
        min_df=2,                # ignore ultra-rare terms
        max_features=20000,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    clf = LinearSVC()
    clf.fit(X_train_tfidf, y_train)

    # Evaluate on held-out set
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_tfidf)

    print("\n=== Router performance on held-out test set ===")
    print(classification_report(y_test, y_pred))

    # Save vectorizer + classifier together
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "clf": clf}, MODEL_PATH)
    print(f"Saved trained router model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
