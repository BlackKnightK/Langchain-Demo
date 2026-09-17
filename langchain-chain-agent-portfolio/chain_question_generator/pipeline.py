from pathlib import Path
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import build_model
from .schemas import KeyNotes, QuizResult

QUIZ_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an English-learning assessment designer. Create questions only from the supplied source material. "
        "Do not invent facts. Keep answers unambiguous and explanations concise."
    ),
    (
        "human",
        """Create a quiz from the source below.

Requirements:
- Number of questions: {num_questions}
- Allowed question types: {question_types}
- Mix question types when possible.
- Vocabulary questions should test words in context.
- Reading and inference questions must be answerable from the source.
- For multiple-choice questions, provide exactly four options.
- Difficulty should reflect the source and reasoning required.

SOURCE:
{source_text}
"""
    ),
])

NOTES_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Extract compact study notes from an English passage. Preserve factual details that could support quiz questions."
    ),
    (
        "human",
        "Extract main ideas, important facts, and useful vocabulary from this chunk:\n\n{chunk}"
    ),
])


def _prepare_source(text: str, long_text_threshold: int = 16_000) -> str:
    if len(text) <= long_text_threshold:
        return text

    splitter = RecursiveCharacterTextSplitter(chunk_size=7_000, chunk_overlap=400)
    chunks = splitter.split_text(text)[:6]

    model = build_model(temperature=0)
    notes_model = model.with_structured_output(KeyNotes)
    notes_chain = NOTES_PROMPT | notes_model

    all_notes: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        notes = notes_chain.invoke({"chunk": chunk})
        all_notes.append(
            f"Chunk {i}\n"
            f"Main ideas: {'; '.join(notes.main_ideas)}\n"
            f"Important facts: {'; '.join(notes.important_facts)}\n"
            f"Vocabulary: {'; '.join(notes.useful_vocabulary)}"
        )

    return "\n\n".join(all_notes)


def generate_quiz(
    text: str,
    num_questions: int = 5,
    question_types: tuple[str, ...] = (
        "reading_comprehension",
        "vocabulary",
        "inference",
        "main_idea",
    ),
) -> QuizResult:
    if not 1 <= num_questions <= 20:
        raise ValueError("num_questions must be between 1 and 20")

    source_text = _prepare_source(text)
    model = build_model(temperature=0.2)
    structured_model = model.with_structured_output(QuizResult)
    chain = QUIZ_PROMPT | structured_model

    result = chain.invoke({
        "num_questions": num_questions,
        "question_types": ", ".join(question_types),
        "source_text": source_text,
    })

    if len(result.questions) != num_questions:
        # Keep the generated result usable while surfacing the mismatch.
        result.questions = result.questions[:num_questions]
    return result


def save_quiz(result: QuizResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".json":
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return path

    if path.suffix.lower() in {".md", ".markdown"}:
        lines = [
            f"# {result.title}",
            "",
            f"**Overall difficulty:** {result.overall_difficulty}",
            "",
            "## Article summary",
            "",
            result.article_summary,
            "",
            "## Questions",
            "",
        ]
        for idx, q in enumerate(result.questions, start=1):
            lines.extend([
                f"### {idx}. {q.question}",
                f"- Type: `{q.question_type}`",
                f"- Difficulty: `{q.difficulty}`",
            ])
            if q.options:
                lines.append("- Options:")
                lines.extend([f"  - {opt}" for opt in q.options])
            lines.extend([
                f"- Answer: **{q.answer}**",
                f"- Explanation: {q.explanation}",
                "",
            ])
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    raise ValueError("Output file must end in .json or .md")
