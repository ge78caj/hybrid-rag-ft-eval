from methods.multi_lora.router import route_hotpotqa

def main():
    qs = [
        "Who was the wife of the founder of Microsoft?",
        "Brown State Fishing Lake is in a country that has a population of how many inhabitants?",
    ]
    for q in qs:
        choice = route_hotpotqa(q)
        print(f"Q: {q}\n -> router choice: {choice}\n")

if __name__ == "__main__":
    main()
