# AGENTS.md

## Purpose

MLflow-Monitor is a baseline-aware model monitoring package for MLflow.

Prefer architectural correctness, explicit behavior, and small reviewable changes over speed or breadth.

## Working rules

- Work ticket-by-ticket.
- Keep diffs small and reviewable.
- Only add code that is necessary for the current ticket.
- Do not broaden scope beyond the requested ticket.
- Preserve existing behavior unless the task explicitly changes it.
- Prefer the simplest implementation that satisfies the current ticket.
- Avoid speculative abstractions or future-facing indirection unless required by the current ticket.
- Look for relevant local skills under `.codex/skills/` before starting implementation or review work.
- Use the local `tdd` skill for behavior-changing work; skip it for docs-only or formatting-only edits.
- Use the local `seeking-design-truth` skill for ticket-driven work, design questions, behavior clarification, and other tasks that need grounding in the repo's design docs before implementation or review.
- Use the local `commit-discipline` skill for commit follow-through after green ticket slices.
- Commit very frequently by default.
- Do not log PII.

## Review rules

- Default to a code review mindset when asked to review.
- Prioritize correctness issues, behavioral regressions, edge cases, and missing tests.
- Treat findings as the primary output; keep summaries brief and secondary.
- Order findings by severity and include file references when possible.
- Call out assumptions, unclear intent, or test gaps when they affect confidence.
- If no findings are discovered, say so explicitly and note any residual risks or coverage gaps.
- Do not focus on style-only feedback unless it affects correctness, maintainability, or project consistency.

## Documentation

- Use Google-style docstrings for public runtime modules, classes, and functions under `src/mlflow_monitor/`.
- Keep docstrings concise and focused on purpose, inputs, returns, and important failure behavior.

## Validation

Use Python 3.12+ and `uv`.

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run ruff format .
uv build
```

## Branch naming

- `MM-{number}/{ticket-number}-xxx` for ticket branches.
- `zs/` for hotfixes, revisions, or similar work.
