# MLflow-Monitor

ML monitoring as a first-class workflow, built on MLflow.

MLflow-Monitor reads existing MLflow training runs, checks whether comparison is meaningful, and stores monitoring state in its own namespace. Training runs stay read-only.

What that means in practice:

- Comparability checks before metric interpretation
- Baseline-pinned monitoring timelines
- Traceable monitoring state and history in MLflow
- Monitoring kept separate from training

For the design philosophy and world model behind these choices, see [docs/worldview.md](docs/site/source/worldview.md).

<p align="center">
  <img src="docs/site/source/_static/system_diagram_v3.jpg" alt="MLflow-Monitor overview" width="500">
</p>

## Why This Project Exists

MLflow tracks training runs well, but it does not provide a structured layer for deciding whether a new run is meaningfully comparable to a trusted baseline over time.

In practice, teams often fill that gap with ad hoc scripts, naming conventions, and manual checks. As systems grow, those checks become harder to trust, harder to trace, and harder to reproduce.

MLflow-Monitor exists to make that monitoring step explicit. It treats monitoring as a first-class workflow with its own lifecycle, state, and persistence inside MLflow, while keeping training runs read-only.

## Try It

Clone the repo and sync the environment:

```bash
git clone https://github.com/shizheng-rlfresh/mlflow-monitor.git
cd mlflow-monitor
uv sync
```

Then follow the walkthrough in [demo/README.md](demo/README.md) for the full setup and monitoring commands.

## How to Use

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


## Current Status

Early alpha. The shipped runtime covers the first three stages of the monitoring lifecycle: create, prepare, and check. This means:

- First-run bootstrap with an explicit baseline
- Later runs that reuse the pinned baseline
- Comparability verdicts of `pass`, `warn`, and `fail` against real MLflow
- Persisted monitoring runs with `outputs/result.json` artifacts
- Read-only treatment of training experiments throughout

The later lifecycle stages (analyze, close, diff, findings, LKG promotion) are designed but not yet in the runtime.

This is a repo-first alpha: clone the repository, sync the environment with `uv`, and run the demo or Python API from source.

## Architecture

For a closer look at how the system is structured, including the layering between orchestration, workflow, and the MLflow gateway, see [docs/architecture.md](docs/site/source/architecture.md).

## Development Setup

```bash
uv sync --extra dev
```

## License

Apache-2.0
