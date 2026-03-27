# MLflow-Monitor

ML monitoring as a first-class workflow, built on MLflow.

MLflow-Monitor treats monitoring as a structured, traceable process rather than ad hoc metric checks. It reads existing MLflow training runs, validates the conditions that make comparison meaningful, and stores monitoring state in its own namespace. Training runs stay read-only.

What that means in practice:

- Evidence-based comparability checks before any metric interpretation
- Baseline-pinned timelines with structured monitoring lifecycle
- Traceable state from evidence gathering through findings
- Monitoring history persisted in MLflow, separate from training

For the design philosophy and world model behind these choices, see [docs/worldview.md](docs/worldview.md).

<p align="center">
  <img src="assets/system_diagram.png" alt="MLflow-Monitor overview" width="500">
</p>

## Current Status

Early alpha. The shipped runtime covers the first three stages of the monitoring lifecycle: create, prepare, and check. This means:

- First-run bootstrap with an explicit baseline
- Later runs that reuse the pinned baseline
- Comparability verdicts of `pass`, `warn`, and `fail` against real MLflow
- Persisted monitoring runs with `outputs/result.json` artifacts
- Read-only treatment of training experiments throughout

The later lifecycle stages (analyze, close, diff, findings, LKG promotion) are designed but not yet in the runtime.

This is a repo-first alpha: clone the repository, sync the environment with `uv`, and run the demo or SDK from source.

## Architecture

For a closer look at how the system is structured, including the layering between orchestration, workflow, and the MLflow gateway, see [docs/architecture.md](docs/architecture.md).

## Try It

Clone the repo and sync the environment:

```bash
git clone https://github.com/shizheng-rlfresh/mlflow-monitor.git
cd mlflow-monitor
uv sync
```

Start a local MLflow UI in one terminal:

```bash
uv run mlflow ui --port 5000 --backend-store-uri sqlite:///$PWD/.mlflow-dev/mlflow.db
```

Then follow the walkthrough in [demo/README.md](demo/README.md) for the full setup and monitoring commands.

## Python SDK

```python
from mlflow_monitor import monitor

result = monitor.run(
    subject_id="fraud_model",
    source_run_id="training_run_id",
    baseline_source_run_id="baseline_run_id",
)

print(result.lifecycle_status)
print(result.comparability_status)
```

The `baseline_source_run_id` is required on the first run for a subject. Later runs reuse the pinned baseline automatically.

## Development Setup

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

## Why This Project Exists

MLflow does a good job tracking training runs, metrics, params, and artifacts. What it doesn't provide is a structured layer for what happens after training: deciding whether a new run is actually comparable to the one you trust, and building a traceable record of that evaluation over time.

In practice, teams fill this gap with ad hoc scripts, naming conventions, and manual checks. That works for a while. At scale, the naming conventions drift, the comparability logic scatters across notebooks and pipelines, and the monitoring history lives nowhere durable.

MLflow-Monitor exists to close that gap. It treats monitoring as a repeatable workflow with its own lifecycle, its own state, and its own namespace inside MLflow. The training side stays untouched. The monitoring side is structured enough to answer questions like: is this run comparable to the baseline? What happened on the last monitoring check? What changed?

## License

Apache-2.0