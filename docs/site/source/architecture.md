# Architecture

> **MVP-era architecture:** This document presents the architecture shipped with
> the [`v0.1.0` MVP Release](https://github.com/shizheng-rlfresh/mlflow-monitor/releases/tag/v0.1.0).
> It is retained for historical context during active v0 development and is not a
> specification of every behavior on `main`.

MLflow-Monitor keeps training history and monitoring history separate.

Training runs remain the source of truth for model artifacts and training metadata. Monitoring runs read that evidence, evaluate comparability, and persist their own state in a monitoring-owned experiment.

```mermaid
flowchart LR
    PythonAPI["monitor.run(...)"]
    PythonAPI --> Create

    subgraph Training["Training Side"]
        TrainExp[("MLflow: {Training Experiment}")]
        TrainRuns["Training runs\nmetrics, params, tags, artifacts"]
    end

    subgraph Workflow["Monitoring Lifecycle"]
        Create["Create (shipped)"] --> Prepare["Prepare (shipped)"] --> Check["Check (shipped)"] --> Analyze["Analyze (planned)"] --> Close["Close (planned)"]
    end

    subgraph Monitoring["Monitoring Side"]
        MonExp[("MLflow: mlflow_monitor/{subject_id}")]
        Timeline["Timeline state and index\nbaseline, latest run, sequence index"]
        MonRuns["Monitoring runs\nlifecycle, comparability, result artifact"]
    end

    Workflow -->|read-only evidence| TrainRuns
    Workflow -->|persistent state| MonRuns
    TrainExp --> TrainRuns
    MonExp --> Timeline
    MonExp --> MonRuns
    Timeline --> MonRuns
```

Stages with dashed borders are designed but not yet in the runtime.

## Runtime Model

The full monitoring lifecycle is create → prepare → check → analyze → close.

The current runtime ships the first three stages:

- Create or reuse a monitoring run for one source training run
- Prepare baseline and comparison context
- Execute the contract check
- Persist a terminal monitoring result

Analyze (diff computation and finding generation) and close (finalization and optional LKG promotion) are the next stages on the roadmap.

## Training Side

MLflow training experiments hold the original model-development history: metrics, params, tags, model artifacts, and optional dataset-related artifacts.

MLflow-Monitor reads from those runs but does not mutate them.

## Monitoring Side

MLflow-Monitor creates one monitoring experiment per subject. For example, `training/fraud_model` contains source training runs, and `mlflow_monitor/fraud_model` contains monitoring runs for that subject.

The monitoring experiment holds timeline-level projections: the baseline, the latest monitoring run id, the next sequence index, and indexed run references for timeline traversal. Each successful Prepare writes its immutable baseline claim on the Monitoring Run. The experiment baseline and allocation index are repairable projections: the gateway validates them against durable run-level claims and allocation identities, repairs uniquely recoverable partial writes, and fails closed on contradictions.

Each monitoring run holds its allocation identity (source run, recipe identity, and sequence index) and its evaluation state: lifecycle status, comparability status, baseline and other references, and the final `outputs/result.json` artifact. These exist at the run level because they are specific to one evaluation event.

## Why This Split Matters

A monitoring run can complete successfully and still report `fail` comparability. That is a valid and useful outcome, not a crash. Comparability success is distinct from workflow execution success.

This separation keeps training history immutable, gives monitoring its own durable memory, makes baseline selection explicit, and preserves a clean audit trail from evidence through verdict.
