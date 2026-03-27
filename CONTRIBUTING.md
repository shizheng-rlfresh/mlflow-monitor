# Contributing

MLflow-Monitor is currently a repo-first project.

## Setup Paths

Choose the environment that matches what you want to do:

- Run the demo:

  ```bash
  uv sync --extra demo
  ```

- Run tests and validation:

  ```bash
  uv sync --extra dev
  ```

- Work on both demo and development tasks:

  ```bash
  uv sync --extra dev --extra demo
  ```

## Repo Validation

Run the standard checks with the `dev` extra installed:

```bash
uv run pytest -m "not demo"
uv run ruff check .
uv build
```

## Demo Tests

Demo tests are opt-in and require both extras:

```bash
uv sync --extra dev --extra demo
uv run pytest -m demo
```

## Demo

The demo is documented in [demo/README.md](demo/README.md).
It assumes the `demo` extra is installed and does not install `pytest` or `ruff`.
