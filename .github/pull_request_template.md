## What changed

<!-- Summarize the concrete changes in this pull request. -->

## Why

<!-- Explain the problem, requirement, or ticket outcome this change addresses. -->

## Review context

***Note: Please ensure including the context for agent reviewers. If human reviewers are requested, this could be skipped.***

<!--
Give reviewers only the context needed for this diff.
- Link only the relevant public documents under docs/site/source/.
- Summarize the required behavior in reviewable, externally safe terms.
- State important behavior that is explicitly out of scope.
- Do not link or quote the private design record.
-->

- Relevant public docs:
- Required behavior:
- Out of scope:

## Impact

<!-- Describe user-facing or developer-facing behavior, compatibility, and migration impact. -->

## Validation

<!-- Check only commands that were run successfully (do not change the commands). Explain any omissions below. -->

- [ ] `uv sync --extra dev --group doc`
- [ ] `uv run pytest`
- [ ] `uv run ruff check .`
- [ ] `uv run ruff format --check .`
- [ ] `uv run pyright`
- [ ] Matching `docs/site/` content is updated, or no developer-doc change is required
- [ ] `uv run sphinx-build -W -b html docs/site/source docs/site/build/html`
- [ ] `uv build`

<!-- Validation omissions, if any: -->
