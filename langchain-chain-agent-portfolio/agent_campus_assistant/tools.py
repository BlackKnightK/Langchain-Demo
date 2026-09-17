import ast
import operator
from typing import Any
from langchain.tools import tool
from tavily import TavilyClient

from .config import get_settings

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_calculate(expression: str) -> float:
    """Evaluate a basic arithmetic expression without using eval()."""
    tree = ast.parse(expression, mode="eval")

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Exponent is too large")
            return float(_ALLOWED_BINOPS[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
            return float(_ALLOWED_UNARYOPS[type(node.op)](visit(node.operand)))
        raise ValueError("Only basic arithmetic is allowed")

    result = visit(tree)
    if abs(result) > 1e15:
        raise ValueError("Result is too large")
    return result


def _search(query: str, max_results: int = 5) -> dict[str, Any]:
    settings = get_settings()
    if not settings.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY is required for web-search tools")

    client = TavilyClient(api_key=settings.tavily_api_key)
    data = client.search(
        query=query,
        search_depth="basic",
        max_results=max_results,
        include_answer=True,
        include_raw_content=False,
        include_images=False,
    )
    results = data.get("results") or []
    return {
        "answer": (data.get("answer") or "").strip(),
        "sources": [
            {"title": item.get("title", ""), "url": item.get("url", "")}
            for item in results[:3]
        ],
    }


@tool
def search_weather(city: str, date: str = "today") -> dict:
    """Search current or date-specific weather for a city. Use for weather, temperature, rain, or clothing advice."""
    return _search(f"{city} {date} weather high low temperature precipitation")


@tool
def search_holiday_schedule(year: int, holiday: str, country_or_region: str = "China") -> dict:
    """Search an official/public holiday schedule, including dates or make-up workdays when relevant."""
    return _search(f"{country_or_region} {year} {holiday} official holiday schedule dates")


@tool
def search_exam_deadline(exam: str, session: str = "next available") -> dict:
    """Search registration dates, deadlines, or official pages for exams such as IELTS, TOEFL, CET, GRE, or GMAT."""
    return _search(f"{exam} {session} registration deadline official")


@tool
def calculate(expression: str) -> dict:
    """Calculate a basic arithmetic expression deterministically. Use this instead of doing arithmetic mentally."""
    return {"expression": expression, "result": safe_calculate(expression)}


@tool
def build_study_plan(total_minutes: int, tasks: str) -> dict:
    """Create a simple time-block plan. `tasks` should be comma-separated; explicit durations like 'break 20 min' are preserved when possible."""
    import re

    if total_minutes <= 0 or total_minutes > 24 * 60:
        raise ValueError("total_minutes must be between 1 and 1440")

    parts = [x.strip() for x in re.split(r"[,;，；\n]+", tasks) if x.strip()]
    if not parts:
        raise ValueError("No tasks were provided")

    explicit: list[tuple[str, int | None]] = []
    used = 0
    for task in parts:
        match = re.search(r"(\d+)\s*(?:min|mins|minute|minutes|分钟)", task, flags=re.I)
        minutes = int(match.group(1)) if match else None
        if minutes:
            used += minutes
        explicit.append((task, minutes))

    if used > total_minutes:
        raise ValueError("Explicit task durations exceed total_minutes")

    unspecified = sum(1 for _, minutes in explicit if minutes is None)
    remaining = total_minutes - used
    default_slot = remaining // unspecified if unspecified else 0

    plan = []
    cursor = 0
    for index, (task, minutes) in enumerate(explicit):
        duration = minutes if minutes is not None else default_slot
        if minutes is None and index == len(explicit) - 1:
            duration = total_minutes - cursor
        end = min(total_minutes, cursor + max(duration, 0))
        plan.append({"task": task, "start_min": cursor, "end_min": end, "duration_min": end - cursor})
        cursor = end

    return {"total_minutes": total_minutes, "plan": plan}


TOOLS = [
    search_weather,
    search_holiday_schedule,
    search_exam_deadline,
    calculate,
    build_study_plan,
]
