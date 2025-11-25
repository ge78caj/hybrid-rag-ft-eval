"""
Simple routing utilities for choosing a LoRA adapter.

For now we only implement a dataset-based router.
Later we can add an embedding-based router here.
"""


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
