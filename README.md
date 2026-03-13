# MLflow-Monitor

**Baseline-aware model monitoring for MLflow.**

MLflow-Monitor reads your existing MLflow training runs, checks whether they are actually comparable, computes structured diffs against trusted reference points, and returns actionable findings — without modifying your training runs or adding new infrastructure.

```python
from mlflow_monitor import monitor

# First monitoring run for a subject: baseline is required
result = monitor.run(
    subject_id="churn_model",
    source_run_id="run_103",
    baseline_source_run_id="run_101",
)

print(result.status)    # pass / warn / fail
print(result.findings)  # prioritized findings with evidence
```

## Why it exists

MLflow tracks experiments well, but it does not answer questions like:

- Is this run actually better than our trusted baseline?
- Are these two runs even comparable?
- What changed since the last good state?
- Where is the audit trail for that conclusion?

Teams usually answer these with ad hoc notebooks, one-off scripts, or memory. Baselines drift, invalid comparisons slip through, and metric deltas lose context.

MLflow-Monitor fills that gap.

## What it does

- **Checks comparability first**  
  Detects schema, feature, data-scope, and environment mismatches before comparing metrics.

- **Compares against the right references**  
  Every run can be compared against:
  - the **baseline** for long-term drift
  - the **previous run** for iteration-to-iteration changes
  - the **last known good (LKG)** for distance from trusted state

- **Produces structured diffs and findings**  
  Diffs are raw evidence. Findings are interpreted issues with severity, category, and guidance.

- **Maintains a monitoring timeline**  
  Monitoring runs are stored in a system-owned namespace, giving each subject a durable history and queryable trajectory.

- **Supports promotion workflows**  
  Promotion happens after monitoring closes, aligning monitoring with the practical question: *is this good enough to trust or promote?*

## How it works

![System Diagram](assets/system_diagram_v0.png)

MLflow-Monitor **reads from training experiments in MLflow** and **writes only to its own monitoring namespace**. Your training history remains untouched.

## Why this design

- **No new infrastructure**  
  Uses your existing MLflow instance as the persistence layer.

- **Training runs stay immutable**  
  MLflow-Monitor never writes to or mutates training runs.

- **Zero-config by default**  
  Ships with a default recipe and permissive contract/profile so you can start with what MLflow already has.

- **Deterministic system-owned storage**  
  Monitoring state lives under `{namespace_prefix}/{subject_id}` (`mlflow_monitor` by default).

- **Baseline is explicit and durable**  
  On the first run for a subject, the caller chooses the baseline. After that, the timeline owns it.

## Quick start

```python
from mlflow_monitor import monitor

# First run for a subject
result = monitor.run(
    subject_id="churn_model",
    source_run_id="run_103",
    baseline_source_run_id="run_101",
)

# Later runs reuse the pinned baseline from timeline state
result = monitor.run(
    subject_id="churn_model",
    source_run_id="run_104",
)

if result.comparability == "pass":
    for finding in result.findings:
        print(f"[{finding.severity}] {finding.summary}")
```

```bash
mlflow-monitor run \
  --subject churn_model \
  --source-run run_103 \
  --baseline-run run_101
```

## Core concepts

- **Subject** — the model or training line being monitored
- **Timeline** — the ordered history of monitoring runs for a subject
- **Baseline** — the pinned reference point for that timeline
- **LKG** — the most recent trusted monitoring state
- **Contract** — the rules that determine whether runs are comparable
- **Diff** — objective evidence of what changed
- **Finding** — prioritized interpretation of those diffs
- **Recipe** — optional configuration for monitoring behavior

## What’s in v0

- Single timeline per subject
- Baseline / previous / LKG comparison
- Contract-based comparability checks
- Structured diffs and findings
- LKG promotion
- Anchor-window trajectory queries
- Default recipe for low-setup usage
- SDK and CLI entry points
- Configurable monitoring namespace prefix

## License

Apache-2.0
