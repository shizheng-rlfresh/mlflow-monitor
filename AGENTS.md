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
- Commit frequently at coherent green, reviewable slices.
- Do not log PII.

## Private maintainer overlay

- If `.agents/v0-development.md` exists, read it completely before maintainer
  release or ticket work. It supplements this public file with private design
  authority, sequencing, and release rules.
- Its absence in a public clone is expected. Do not infer or request the private
  plan when it is unavailable.
- Never stage or publish files under `.agents/`.

## Code Review Rules

- Default to a code review mindset when asked to review.
- Prioritize correctness issues, behavioral regressions, edge cases, and missing tests.
- Treat findings as the primary output; keep summaries brief and secondary.
- Order findings by severity and include file references when possible.
- Call out assumptions, unclear intent, or test gaps when they affect confidence.
- If no findings are discovered, say so explicitly and note any residual risks or coverage gaps.
- Do not focus on style-only feedback unless it affects correctness, maintainability, or project consistency.

### Pull request context

- Read the pull request's `Review context` before reviewing the diff.
- Use only the public documents named there as ticket-specific design context.
- Treat `Required behavior` and `Out of scope` as review boundaries, but flag any
  conflict with repository guidance, shipped public behavior, code, or tests.
- If the review context is absent or incomplete, do not invent intent. Review the
  diff against the repository guidance, relevant public documentation, code, and
  tests, and state the resulting uncertainty.

### Monitoring boundaries

- Flag any change that mutates a Source Training Run or stores monitoring-owned
  state or artifacts on it. Source Training Runs are read-only; monitoring state
  belongs to the subject's monitoring experiment or a Monitoring Run.
- Flag any materialized domain value or public result that contains
  `monitoring_run_id` without its immutable `source_run_id`, or that accepts a
  conflicting pair. A Baseline Source Run is source-only and has no
  `monitoring_run_id`; low-level MLflow adapters may use `run_id` only at the
  upstream API boundary.

## Documentation

- Use Google-style docstrings for public runtime modules, classes, and functions under `src/mlflow_monitor/`.
- Keep docstrings concise and focused on purpose, inputs, returns, and important failure behavior.
- Developer documentation is published on Read the Docs from `docs/site/`; treat
  it as a maintained product surface.
- Every implementation ticket must assess its `docs/site/` impact. When an
  implementation adds, removes, or changes behavior that requires corresponding
  developer documentation, add, remove, or edit the matching `.rst` content in
  the same branch and pull request. The implementation is incomplete until the
  developer documentation matches the delivered behavior.

## Validation

Use Python 3.12+ and `uv`.

During TDD, run the narrowest relevant pytest target for red/green iteration. Before
each green ticket commit, run focused tests plus the applicable Ruff and Pyright
checks. Before closing a ticket, run the complete gate:

```bash
uv sync --extra dev --group doc
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run --group doc sphinx-build -W -b html docs/site/source docs/site/build/html
uv build
```

Release transitions and final acceptance require the complete gate with no
omissions. Use `uv run ruff format .` only as an intentional formatting action,
never as proof that formatting was already valid.

## Branch naming

- For ticketed work, use
  `ticket/<lowercase-ticket-id>-<short-description>`.
- For GitHub issues, use `issue/<github-issue-number>-<short-description>`.
- For critical maintenance branched from a release tag, use
  `hotfix/<release-version>-<short-description>`.
- For exploratory work, use `spike/<short-description>`.
- For other work, use `work/<short-description>`.
- Use one branch per ticket and normalize ticket IDs to lowercase.
- Use lowercase, hyphen-separated descriptions.
- Continue revisions on the original branch when possible.
- Do not include Plane IDs unless the user explicitly identifies Plane as the
  source of work.
- Branch a critical release correction from the relevant release tag and merge the
  correction forward into the active development line.
- Never move or reuse a published release tag.

## Commit messages

- Describe the change concisely in plain language.
- Use specialized engineering terminology only when it explains the change more precisely.
- Do not require ticket prefixes, category labels, or Conventional Commit syntax.

## Plane

- Use Plane only when the user explicitly requests it or identifies a Plane work item.
- Treat Plane as read-only unless the user explicitly requests a change.
