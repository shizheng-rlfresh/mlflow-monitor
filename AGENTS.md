# AGENTS.md

## Purpose
MLflow-Monitor is a baseline-aware model monitoring package for MLflow.

Prefer architectural correctness, explicit behavior, and small reviewable changes over speed or breadth.

## Non-negotiable rules
- The persistence gateway is the only MLflow touchpoint.
- Never mutate MLflow training runs.
- Monitoring state lives under `{namespace_prefix}/{subject_id}`. Default prefix is `mlflow_monitor`.
- If no timeline exists, the first run must include `baseline_source_run_id`.
- Do not infer the initial baseline automatically.
- After bootstrap, baseline is immutable and must be read from timeline state.
- Failed runs are excluded from default timeline / trajectory / anchor-window views unless explicitly requested.
- Promotion happens after close and does not invalidate closed state.
- Do not broaden scope beyond the requested ticket.
- Do not introduce speculative abstractions, plugin systems, or framework-style extensibility unless required by the current ticket.

## Workflow
- Work ticket-by-ticket.
- Keep diffs small and reviewable.
- Prefer the simplest implementation that satisfies the current ticket.
- Preserve existing semantics unless the task explicitly changes them.
- For behavior changes, add or update tests.

# Documentation
Use Google-style docstrings for public modules, classes, and functions.
Keep docstrings concise and focused on purpose, inputs, returns, and important failure behavior.
Use descriptive test names. Add test docstrings only for non-obvious scenarios or important invariants.

## Review guidelines
- Don't log PII.
- Branch names reveal if the PR is for a specific ticket
  - branch name as `MM-{number}/{ticket number}-xxx` is for ticket
  - branch name starting with `zs/` is for hotfix, revision, etc.
- tickets in `docs/v0/ticket_breakdown_v0.md`

## Validation
Use Python 3.12+ and `uv`.

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run ruff format .
uv build
```

## Source of truth docs
Before changing behavior or architecture, consult:
- `docs/v0/design_doc_v0.md`
- `docs/v0/ticket_breakdown_v0.md`

Consult these when relevant to the task:
- `docs/v0/workflow_v0.md`
- `docs/v0/recipe_v0.md`
- `docs/v0/mlflow_mapping_v0.md`
- `docs/v0/cast_v0.md`
- `docs/v0/mlflow_glossary_v0.md`