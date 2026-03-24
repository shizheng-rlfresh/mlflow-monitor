# MLflow API Guide for MLflow-Monitor Gateway

## MlflowClient: the correct API for library code

MLflow exposes two Python API surfaces:

1. **Fluent API** (`mlflow.start_run()`, `mlflow.log_metric()`, etc.) — manages a global "active run" and "active experiment" via thread-local state. Designed for training scripts that own the process. Not suitable for library code because it pollutes global state and can interfere with the user's own MLflow usage.

2. **MlflowClient** (`from mlflow import MlflowClient`) — requires explicit `run_id` and `experiment_id` on every call. No global state, no side effects. This is what MLflow-Monitor must use.

### Import path (current as of MLflow 3.10.x)

```python
from mlflow import MlflowClient

# NOT: from mlflow.tracking import MlflowClient  (deprecated alias)
# NOT: from mlflow.client import MlflowClient    (internal module)
```

### Initialization

```python
client = MlflowClient()                              # uses MLFLOW_TRACKING_URI or ./mlruns
client = MlflowClient("http://localhost:5000")        # explicit tracking server
client = MlflowClient(tracking_uri="databricks")      # Databricks-managed
```

When `tracking_uri` is not provided, the client falls back to:
1. `MLFLOW_TRACKING_URI` environment variable
2. `./mlruns` (local file store)

This means our gateway works with any backend without needing to know or care which one the user has configured.

### Why MlflowClient, not the fluent API

MLflow-Monitor reads from training experiments and writes to monitoring experiments in the same call. The fluent API has one "active experiment" — switching it back and forth would interfere with user code running in the same process. `MlflowClient` takes explicit experiment/run IDs on every call, so there are no side effects.

---

## Core operations for the gateway

### Experiment management

**Lookup by name:**
```python
experiment = client.get_experiment_by_name("my-training-experiment")
# Returns Experiment object or None
```

**Create experiment:**
```python
experiment_id = client.create_experiment(
    name="mlflow_monitor/churn_model",
    tags={"monitor.version": "0.1.0"}
)
# Raises MlflowException (RESOURCE_ALREADY_EXISTS) on duplicate name
```

**Get-or-create pattern (MLflow does not provide this natively):**
```python
def get_or_create_experiment(client, name, tags=None):
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    try:
        return client.create_experiment(name, tags=tags)
    except MlflowException:
        # Race condition: another process created it
        return client.get_experiment_by_name(name).experiment_id
```

**Search experiments:**
```python
experiments = client.search_experiments(
    filter_string="name LIKE 'mlflow_monitor/%'"
)
# Returns PagedList — check .token for pagination
```

Experiment names are case-sensitive and unique. Slash-based names like `mlflow_monitor/churn_model` are valid.

### Run management

**Create a run:**
```python
run = client.create_run(
    experiment_id=experiment_id,
    run_name="monitor-check-20260322",
    tags={"source_run_id": "abc123", "lifecycle_status": "created"}
)
run_id = run.info.run_id
```

**Terminate a run:**
```python
client.set_terminated(run_id, status="FINISHED")  # or "FAILED"
# Valid statuses: RUNNING, SCHEDULED, FINISHED, FAILED, KILLED
```

**Search runs within an experiment:**
```python
runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string='tags.role = "timeline_config"',
    order_by=["attributes.start_time DESC"],
    max_results=100
)
```

### Tag operations

**Set a tag on a run:**
```python
client.set_tag(run_id, "lifecycle_status", "checked")
client.set_tag(run_id, "comparability_status", "pass")
client.set_tag(run_id, "sequence_index", "5")
```

Tags are mutable (last write wins). Keys up to 255 bytes, values up to 5,000 bytes. Updates are individual REST calls with no cross-tag atomicity.

**Query by tag:**
```python
# Find sentinel run
runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string='tags.role = "timeline_config"'
)

# Find active LKG
runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string='tags.lkg_status = "active"'
)

# Find closed runs ordered by sequence index
runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string='tags.lifecycle_status = "closed"',
    order_by=["tags.sequence_index ASC"]
)
```

Filter syntax supports `AND` only — `OR` is not supported. Use multiple queries and merge results if needed.

**Tag namespace:** never use the `mlflow.` prefix (reserved for MLflow system tags). Use a distinctive prefix like `monitor.` or no prefix for our custom tags.

### Metric and param operations

**Log metric:**
```python
client.log_metric(run_id, "drift_score", 0.42)
```

**Log param:**
```python
client.log_param(run_id, "subject_id", "churn_model")
```

**Read metrics/params from a run:**
```python
run = client.get_run(run_id)
metrics = run.data.metrics     # dict[str, float] — last logged value per key only
params = run.data.params       # dict[str, str]
tags = run.data.tags           # dict[str, str]
```

For full metric history: `client.get_metric_history(run_id, "loss")` returns list of `Metric` objects.

### Artifact operations

**Log JSON artifact (no temp files needed):**
```python
client.log_dict(run_id, {"findings": [...], "diffs": [...]}, "monitoring_results.json")
```

**Log text artifact:**
```python
client.log_text(run_id, "some text content", "notes.txt")
```

**Download artifact:**
```python
local_path = client.download_artifacts(run_id, "monitoring_results.json")
# Returns local filesystem path, then parse with json.load()
```

### Batch logging

```python
from mlflow.entities import Metric, Param, RunTag
import time

client.log_batch(
    run_id=run_id,
    metrics=[Metric("accuracy", 0.95, int(time.time() * 1000), step=0)],
    params=[Param("subject_id", "churn_model")],
    tags=[RunTag("lifecycle_status", "created")]
)
# Max 1000 items total per call (max 100 params, max 100 tags)
# Not fully atomic — partial writes may persist on failure
```

---

## Search API details and limitations

**Filter syntax:**
- `metrics.<key>` — numeric comparators (`>`, `>=`, `<`, `<=`, `=`, `!=`)
- `params.<key>` — string comparators (`=`, `!=`, `LIKE`, `ILIKE`)
- `tags.<key>` — string comparators plus `IS NULL` / `IS NOT NULL`
- `attributes.<attr>` — run metadata (`status`, `start_time`, `run_name`, etc.)
- Join with `AND` only — **`OR` is not supported**

**Pagination is critical:**
`client.search_runs()` returns at most `max_results` items (default 1,000) and silently truncates. Always paginate:

```python
all_runs, page_token = [], None
while True:
    page = client.search_runs(
        experiment_ids=[eid],
        max_results=1000,
        page_token=page_token
    )
    all_runs.extend(page)
    if not page.token:
        break
    page_token = page.token
```

---

## Compatibility

Target `mlflow>=2.1,<4` for broad compatibility. Core `MlflowClient` tracking methods have been stable since MLflow 2.0 and work identically across:
- Local file store (`./mlruns/`)
- Local/remote tracking server (`mlflow server`)
- Databricks-managed MLflow

Current latest: MLflow 3.10.1 (March 2026). The 2.x → 3.x transition primarily affected model management, GenAI features, and some removed flavors — not the core tracking API used by our gateway.

---

## Gateway mapping summary

| Gateway operation | MLflow API |
|---|---|
| Look up training experiment | `client.get_experiment_by_name(name)` |
| Create monitoring experiment | `client.create_experiment(name, tags)` |
| Create monitoring run | `client.create_run(experiment_id, tags)` |
| Set run tags (lifecycle, comparability, etc.) | `client.set_tag(run_id, key, value)` |
| Log monitoring metrics | `client.log_metric(run_id, key, value)` |
| Log structured artifacts (diffs, findings) | `client.log_dict(run_id, dict, path)` |
| Read source run data | `client.get_run(run_id)` → `.data.metrics/.params/.tags` |
| Find sentinel run | `client.search_runs(filter="tags.role = 'timeline_config'")` |
| Find active LKG | `client.search_runs(filter="tags.lkg_status = 'active'")` |
| List timeline runs | `client.search_runs(filter=..., order_by=["tags.sequence_index ASC"])` |
| Terminate run | `client.set_terminated(run_id, status)` |
| Download artifacts | `client.download_artifacts(run_id, path)` |
| Enforce read-only on training | Application-layer check before any write |
