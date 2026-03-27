# Fraud Demo Walkthrough

This demo shows a small end-to-end monitoring flow on top of real MLflow training runs.

It seeds four fraud-model training runs, each with a real trained sklearn model, realistic metrics, dataset-related artifacts, and the metadata used by the current monitoring workflow.

## What You Will See

- one training experiment: `training/fraud_model`
- four training runs:
  - baseline
  - comparable candidate
  - environment-mismatch candidate
  - non-comparable candidate
- one monitoring experiment after you run the SDK flow:
  - `mlflow_monitor/fraud_model`
- monitoring outcomes that cover:
  - `pass`
  - `warn`
  - `fail`

## Setup

Create a local MLflow store at the repo root:

```bash
mkdir -p .mlflow-dev
export MLFLOW_TRACKING_URI=sqlite:///./.mlflow-dev/mlflow.db
```

Install the demo dependency and start the MLflow UI:

```bash
uv sync --extra demo
mlflow ui --port 5000 --backend-store-uri sqlite:///$PWD/.mlflow-dev/mlflow.db
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).
If that does not work in your browser, try [http://localhost:5000](http://localhost:5000).

## Seed The Training Runs

Run:

```bash
uv run demo/setup.py
```

The script prints the four training run IDs and tells you to run the monitoring step next.

Screenshot to add: training experiment overview after seeding

## Seeded Training Runs

The demo seeds these four roles:

- `fraud-model-baseline-v1`
  The trusted baseline run.
- `fraud-model-candidate-v2`
  A comparable candidate that should produce `pass`.
- `fraud-model-env-shift-v3`
  A candidate with environment metadata changes that should produce `warn`.
- `fraud-model-schema-shift-v4`
  A candidate with schema metadata changes that should produce `fail`.

Each training run includes:

- a real trained sklearn model artifact
- metrics such as `accuracy`, `auc`, `f1`, `precision`, and `recall`
- model parameters
- schema and environment tags used by the monitoring contract
- dataset-related artifacts under `data/`

Screenshot to add: one training run detail page showing metrics, params, model artifact, and data artifacts

## Run Monitoring

Run:

```bash
uv run demo/run_monitoring.py
```

This script resolves the seeded runs automatically and executes the monitoring flow in order:

1. comparable candidate with explicit baseline -> `pass`
2. environment-mismatch candidate -> `warn`
3. non-comparable candidate -> `fail`

Expected result for all three monitoring runs:

- `lifecycle_status = checked`

Expected comparability results:

- comparable candidate -> `pass`
- environment-mismatch candidate -> `warn`
- non-comparable candidate -> `fail`

Screenshot to add: monitoring experiment overview after running pass, warn, and fail

## What To Inspect In MLflow

### Training Experiment

Open `training/fraud_model` and verify:

- the four training runs exist
- each run has metrics and params
- each run has a `model/` artifact tree
- each run has `data/summary.json` and `data/sample_rows.json`

### Monitoring Experiment

After running `uv run demo/run_monitoring.py`, open `mlflow_monitor/fraud_model` and verify:

- monitoring runs are present
- the baseline is reused after the first run
- each monitoring run has `monitoring.lifecycle_status`
- each monitoring run has `monitoring.comparability_status`
- `outputs/result.json` exists on the monitoring run

Experiment tags are also important. They hold the timeline state, including:

- baseline run id
- latest monitoring run id
- next sequence index
- indexed monitoring run ids

Screenshot to add: monitoring run detail page showing tags and outputs/result.json

## What MLflow Gives You vs What MLflow-Monitor Adds

Raw MLflow gives you:

- training runs
- model artifacts
- metrics
- params
- tags

MLflow-Monitor adds:

- a baseline-aware workflow
- comparability outcomes
- a separate monitoring timeline
- durable monitoring result artifacts

That is why the demo uses two experiments: one for training history and one for monitoring history.

## Cleanup

The local `.mlflow-dev/` directory is disposable. To reset the demo:

```bash
rm -rf .mlflow-dev
```
