# MLflow-Monitor

Baseline-aware model monitoring on top of MLflow.

Early alpha for teams who want monitoring semantics without leaving MLflow.

MLflow-Monitor reads existing MLflow training runs, checks whether they are comparable to a trusted baseline, and stores monitoring state in a separate MLflow namespace. Training runs stay read-only; monitoring history gets its own timeline.

New here? Start with [docs/worldview.md](docs/worldview.md) for the design story, system philosophy, and world model behind MLflow-Monitor.

<p align="center">
  <img src="assets/system_diagram.png" alt="MLflow-Monitor overview" width="500">
</p>

## What It Adds Beyond MLflow

MLflow already tracks training runs, metrics, params, tags, and artifacts. MLflow-Monitor adds:

- baseline pinning for a monitored subject
- comparability checks before metric interpretation
- timeline-aware monitoring runs
- separate persisted monitoring state and result artifacts

This makes it possible to answer questions like:

- Is this run comparable to the baseline?
- Which run is the baseline for this subject?
- What happened on the last monitoring attempt?

## Architecture And Concepts

For a deeper explanation of the system:

- see [docs/architecture.md](docs/architecture.md) for structure and runtime boundaries
- see [docs/worldview.md](docs/worldview.md) for the flagship concept paper, design philosophy, and world model

## Current Status

The current shipped workflow covers synchronous monitoring through create, prepare, and check:

- first run bootstrap with an explicit baseline
- later runs that reuse the pinned baseline
- comparability outcomes of `pass`, `warn`, and `fail`
- persisted monitoring runs and `outputs/result.json` artifacts in MLflow

Later workflow stages are not part of the current runtime yet.

This is a repo-first alpha right now: the clearest way to use it today is to clone the repository, sync the environment with `uv`, and run the demo or SDK from source.

## Try It

The fastest way to see the system working is the repo-level fraud demo:

```bash
git clone https://github.com/shizheng-rlfresh/mlflow-monitor.git
cd mlflow-monitor
uv sync --extra demo
uv run mlflow ui --port 5000 --backend-store-uri sqlite:///$PWD/.mlflow-dev/mlflow.db
```

Then follow the walkthrough in [demo/README.md](demo/README.md) for the setup and monitoring commands.

## Packaging Status

The project already builds as a Python package, but the primary supported workflow today is still repo-first:

- clone the repo
- sync dependencies with `uv`
- run the demo locally
- use the Python SDK from the checked-out source tree

A cleaner public installation story can come later without changing the core monitoring model.

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

## License

Apache-2.0
