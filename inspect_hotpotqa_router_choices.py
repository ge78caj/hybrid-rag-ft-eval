# save this as: inspect_hotpotqa_router_choices.py
import json
from pathlib import Path

from methods.multi_lora.router import route_hotpotqa

LABEL_PATH = Path("prediction/router_train_hotpotqa.jsonl")

def main():
    n = 0
    n_base_rag = 0
    n_base_only = 0

    with LABEL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            q = obj["question"]
            choice = route_hotpotqa(q)

            n += 1
            if choice == "base_rag":
                n_base_rag += 1
            else:
                n_base_only += 1

    print(f"Total questions: {n}")
    print(f"base_rag  : {n_base_rag} ({100*n_base_rag/n:.1f}%)")
    print(f"base_only : {n_base_only} ({100*n_base_only/n:.1f}%)")

if __name__ == "__main__":
    main()
