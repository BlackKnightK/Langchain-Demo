from .agent import run_query

EXAMPLES = [
    "What is the weather in Boston today? Give me one clothing suggestion.",
    "I have 90 minutes tonight: reading, vocabulary review, and a 20-minute break. Make a plan.",
    "What is 18 * 30 + 12?",
    "Rewrite this politely: Professor, I cannot attend class tomorrow.",
]


def main() -> None:
    print("Campus Life Assistant")
    print("Examples:")
    for idx, example in enumerate(EXAMPLES, start=1):
        print(f"  {idx}. {example}")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        try:
            answer, tools = run_query(query)
            print(f"\nAssistant: {answer}")
            print("Tool chain:", " -> ".join(tools) if tools else "(none)", "\n")
        except Exception as exc:
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    main()
