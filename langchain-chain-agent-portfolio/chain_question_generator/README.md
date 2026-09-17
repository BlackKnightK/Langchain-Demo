# English Quiz Generator

A deterministic LangChain workflow that converts `.txt`, `.md`, `.pdf`, or `.docx` English reading material into a validated quiz.

The main point of this demo is **workflow composition**: document loading, optional long-text compression, prompt construction, model invocation, schema validation, and persistence happen in a fixed order.

```bash
python -m chain_question_generator.main \
  --file chain_question_generator/examples/sample_article.txt \
  --num-questions 5 \
  --output outputs/quiz.json
```

Use `--output outputs/quiz.md` for a human-readable Markdown version.
