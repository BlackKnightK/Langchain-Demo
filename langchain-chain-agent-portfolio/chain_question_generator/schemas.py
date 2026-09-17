from typing import Literal
from pydantic import BaseModel, Field

QuestionType = Literal["reading_comprehension", "vocabulary", "inference", "main_idea"]
Difficulty = Literal["easy", "medium", "hard"]


class QuizQuestion(BaseModel):
    question_type: QuestionType
    question: str
    options: list[str] = Field(default_factory=list, description="Four options for multiple-choice questions; empty for open questions.")
    answer: str
    explanation: str
    difficulty: Difficulty


class QuizResult(BaseModel):
    title: str
    article_summary: str
    overall_difficulty: Difficulty
    questions: list[QuizQuestion]


class KeyNotes(BaseModel):
    main_ideas: list[str]
    important_facts: list[str]
    useful_vocabulary: list[str]
