"""
Routing utilities for choosing an expert in the multi-LoRA / multi-expert setup.

Right now we support:

- A simple dataset-name based router (simple_router)
- A learned TF-IDF router for HotpotQA that decides between:
    * "base_only" (no RAG)
    * "base_rag"  (use RAG)

Later we can extend this to more experts (PubMed, RAFT LoRAs, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib


# ================================================================
# 1) Simple dataset-name router (kept for backwards compatibility)
# ================================================================
def simple_router(question: str, dataset_name: str) -> str:
    """
    Very simple router that picks a LoRA adapter based only on the dataset name.

    Parameters
    ----------
    question : str
        The input question (currently unused, but kept for future extensions).
    dataset_name : str
        Name of the dataset, e.g. "hotpotqa", "popqa".

    Returns
    -------
    str
        The identifier of the LoRA adapter to use, e.g. "lora_hotpotqa".
    """
    name = dataset_name.lower()

    if name == "hotpotqa":
        return "lora_hotpotqa"
    if name == "popqa":
        return "lora_popqa"

    # Fallback adapter if we don't know the dataset
    return "lora_default"


# ================================================================
# 2) Learned HotpotQA router (TF-IDF + LinearSVC)
# ================================================================

# Path to the joblib model trained by train_router_hotpotqa_tfidf.py
ROUTER_HOTPOTQA_PATH = Path(__file__).parent / "router_hotpotqa_tfidf.joblib"


class HotpotQaRouter:
    """
    Small wrapper around the TF-IDF + LinearSVC model.

    Uses:
        label 0 -> "base_only"  (no RAG)
        label 1 -> "base_rag"   (use RAG)
    """

    def __init__(self, model_path: Path = ROUTER_HOTPOTQA_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"HotpotQA router model not found at: {model_path}.\n"
                "Run train_router_hotpotqa_tfidf.py first."
            )
        self.model = joblib.load(model_path)

    def __call__(self, question: str) -> str:
        """
        Route a single HotpotQA question.

        Returns
        -------
        str
            One of:
                - "base_only"
                - "base_rag"
        """
        label = int(self.model.predict([question])[0])
        return "base_only" if label == 0 else "base_rag"


# Lazy global instance so we only load the model once
_HOTPOT_ROUTER: Optional[HotpotQaRouter] = None


def get_hotpotqa_router() -> HotpotQaRouter:
    global _HOTPOT_ROUTER
    if _HOTPOT_ROUTER is None:
        _HOTPOT_ROUTER = HotpotQaRouter()
    return _HOTPOT_ROUTER


def route_hotpotqa(question: str) -> str:
    """
    Convenience function:
        question -> 'base_only' or 'base_rag'.
    """
    router = get_hotpotqa_router()
    return router(question)
