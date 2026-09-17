from typing import Any
from langchain.agents import create_agent

from .config import build_model
from .tools import TOOLS

SYSTEM_PROMPT = """
You are a campus life assistant for students.

Use tools only when they improve correctness:
- Current or date-sensitive public information -> use the relevant search tool.
- Arithmetic -> use the calculator tool.
- Time-block planning -> use the study-plan tool.
- Rewriting, explanation, or ordinary conversation -> answer directly without forcing a tool call.

When a search tool returns sources, summarize the result and retain the useful source links in the answer.
If a tool fails or information is uncertain, say so clearly rather than inventing details.
Keep answers concise and practical.
""".strip()


def build_agent():
    return create_agent(
        model=build_model(temperature=0),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )


def _extract_tool_chain(messages: list[Any]) -> list[str]:
    chain: list[str] = []
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name:
                chain.append(name)
    return chain


def run_query(query: str) -> tuple[str, list[str]]:
    agent = build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = result["messages"]
    final_content = messages[-1].content
    if not isinstance(final_content, str):
        final_content = str(final_content)
    return final_content, _extract_tool_chain(messages)
