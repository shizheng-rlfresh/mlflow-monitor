# Contributing

MLflow-Monitor is currently a repo-first project.

## Setup Paths

Choose the environment that matches what you want to do:

- Run the demo:

  ```bash
  uv sync
  ```

- Run tests and validation:

  ```bash
  uv sync --extra dev
  ```

## Repo Validation

Run the standard checks with the `dev` extra installed:

```bash
uv run pytest
uv run ruff check .
uv build
```

## Demo

The demo is documented in [demo/README.md](demo/README.md).
It works with the core repo install; `pytest` and `ruff` still require the `dev` extra.
