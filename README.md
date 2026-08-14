[![CI](https://github.com/shizheng-rlfresh/mlflow-monitor/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/shizheng-rlfresh/mlflow-monitor/actions/workflows/ci.yml) [![Documentation Status](https://app.readthedocs.org/projects/mlflow-monitor/badge/?version=latest&style=flat)](https://mlflow-monitor.readthedocs.io/en/latest/) [![Release](https://img.shields.io/github/v/release/shizheng-rlfresh/mlflow-monitor?display_name=tag)](https://github.com/shizheng-rlfresh/mlflow-monitor/releases/latest) [![Python](https://img.shields.io/badge/python-%3E%3D3.12-blue.svg)](https://www.python.org/) [![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

# MLflow-Monitor

ML monitoring as a first-class workflow, built on MLflow.

MLflow-Monitor reads existing MLflow training runs, checks whether comparison is
meaningful, and stores monitoring state in its own namespace. Training runs stay
read-only.

> **Development status:** `main` is the active development line for the v0 product
> scope and currently identifies as `0.2.0.dev0`. It is not the immutable MVP
> snapshot. For the reproducible `create -> prepare -> check` product and its
> executable demo, use the [`v0.1.0` MVP Release](https://github.com/shizheng-rlfresh/mlflow-monitor/releases/tag/v0.1.0).

## Choose a Track

| Track                                   | Use it for                                                                                                                | Starting point                                                                                                                                                                                                                                                |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MVP Release (`v0.1.0`)**      | A stable portfolio artifact, the shipped`create -> prepare -> check` workflow, and the executable fraud-monitoring demo | [Release](https://github.com/shizheng-rlfresh/mlflow-monitor/releases/tag/v0.1.0) · [README](https://github.com/shizheng-rlfresh/mlflow-monitor/blob/v0.1.0/README.md) · [Demo](https://github.com/shizheng-rlfresh/mlflow-monitor/blob/v0.1.0/demo/README.md) |
| **Active development (`main`)** | Following or contributing to the evolving v0 implementation                                                               | This README and the source at the current revision                                                                                                                                                                                                            |

The active development line remains green as capabilities are added, but it does
not carry the MVP Release's stability promise. Public documentation on `main`
describes only behavior implemented at that revision.

## What the MVP Ships

The MVP Release covers the first three monitoring lifecycle stages:

- first-run bootstrap with an explicit baseline
- later monitoring runs that reuse the pinned baseline
- comparability outcomes of `pass`, `warn`, and `fail` against real MLflow
- persisted monitoring runs with `outputs/result.json` artifacts
- read-only treatment of training experiments throughout

Analyze, close, metric diffs, findings, and explicit LKG management are outside the
MVP snapshot and are not implied by the `v0.1.0` release.

## Run the Stable MVP

```bash
git clone https://github.com/shizheng-rlfresh/mlflow-monitor.git
cd mlflow-monitor
git checkout v0.1.0
uv sync --no-dev
```

Then follow the tagged [MVP Demo walkthrough](https://github.com/shizheng-rlfresh/mlflow-monitor/blob/v0.1.0/demo/README.md).
The permanent in-repository release pointer is
[`docs/releases/v0.1.0-mvp.md`](docs/releases/v0.1.0-mvp.md).

## Use the Active Development Line

```bash
git clone https://github.com/shizheng-rlfresh/mlflow-monitor.git
cd mlflow-monitor
uv sync --no-dev
```

The Python API remains the primary programmatic entry point while v0 develops:

```python
from mlflow_monitor import monitor

result = monitor.run(
    subject_id="fraud_model",
    source_run_id="training_run_id",
    baseline_source_run_id="baseline_source_run_id",
)

print(result.lifecycle_status)
print(result.comparability_status)
```

`baseline_source_run_id` is required for the first Monitoring Run for a
`subject_id`. Later Monitoring Runs reuse the pinned Baseline Source Run.

The executable MVP Demo belongs to `v0.1.0`; it is not an evolving test surface
for `main`.

## Architecture and Worldview

The current [architecture](docs/site/source/architecture.md) and
[worldview](docs/site/source/worldview.md) pages are retained MVP-staged presentation
documents.

## Development Setup

Use Python 3.12 or newer and `uv`:

Install dependencies

```bash
uv sync --group doc
```

Run full suite of validations (see `pyproject.toml` for details)

```bash
uv run --no-sync poe validate
```

## License

Apache-2.0
