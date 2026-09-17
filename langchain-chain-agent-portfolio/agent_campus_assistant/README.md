# Campus Life Assistant

A LangChain agent that dynamically decides whether to call a tool.

Available tools:

- weather search
- holiday schedule search
- exam deadline search
- safe arithmetic calculator
- deterministic study-plan builder

Unlike the original learning prototype in `legacy/`, the current version uses LangChain's native agent/tool-calling flow rather than asking the model to emit custom `Action:` strings that are parsed with regular expressions.

```bash
python -m agent_campus_assistant.main
```
