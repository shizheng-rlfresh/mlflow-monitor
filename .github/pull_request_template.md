## What changed

<!-- Summarize the concrete changes in this pull request. -->

## Why

<!-- Explain the problem, requirement, or ticket outcome this change addresses. -->

## Impact

<!-- Describe user-facing or developer-facing behavior, compatibility, and migration impact. -->

## Validation

<!-- Check only commands that were run successfully (do not change the commands). Explain any omissions below. -->

- [ ] `uv sync --extra dev --group doc`
- [ ] `uv run pytest`
- [ ] `uv run ruff check .`
- [ ] `uv run ruff format --check .`
- [ ] `uv run pyright`
- [ ] `uv build`
- [ ] `uv run --group doc sphinx-build -W -b html docs/site/source docs/site/build/html`
- [ ] Update `docs/v0/`

<!-- Validation omissions, if any: -->
