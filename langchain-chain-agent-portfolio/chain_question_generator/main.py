import argparse
from .loaders import load_document
from .pipeline import generate_quiz, save_quiz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an English-learning quiz from a document.")
    parser.add_argument("--file", required=True, help="Input .txt/.md/.pdf/.docx file")
    parser.add_argument("--num-questions", type=int, default=5)
    parser.add_argument(
        "--types",
        nargs="+",
        default=["reading_comprehension", "vocabulary", "inference", "main_idea"],
        choices=["reading_comprehension", "vocabulary", "inference", "main_idea"],
    )
    parser.add_argument("--output", default="outputs/quiz.json", help="Output .json or .md file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = load_document(args.file)
    result = generate_quiz(text, num_questions=args.num_questions, question_types=tuple(args.types))
    output = save_quiz(result, args.output)
    print(f"Quiz saved to: {output}")
    print(f"Generated {len(result.questions)} questions; difficulty={result.overall_difficulty}")


if __name__ == "__main__":
    main()
