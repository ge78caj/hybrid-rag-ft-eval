from methods.multi_lora.router import simple_router

def main():
    q1 = "Who was the wife of the founder of Microsoft?"
    q2 = "In which year did the first iPhone release?"

    print("HotpotQA example ->", simple_router(q1, "hotpotqa"))
    print("PopQA example    ->", simple_router(q2, "popqa"))
    print("Unknown dataset  ->", simple_router(q2, "some_other_dataset"))

if __name__ == "__main__":
    main()
