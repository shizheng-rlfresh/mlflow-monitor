# MLflow-Monitor MVP Design

## 1. Purpose

Two goals before resuming the M2 roadmap:

1. **Demo-ready artifact.** A closed loop: `monitor.run()` reads real MLflow training runs, executes create → prepare → check, writes monitoring results to a real MLflow experiment, returns a synchronous result. This is the artifact for going public.

2. **Reality check.** Validate that the gateway protocol and workflow abstractions hold up against real MLflow. Discover what works, what's too strict, and what needs to change before building more on top.

The MVP does not extend the M1 execution boundary. It replaces the persistence layer beneath it.

---

## 2. Scope

### In scope

1. `MonitorMLflowClient` — thin adapter over `MlflowClient` encapsulating all raw MLflow API usage.
2. `MLflowMonitoringGateway` — implements `MonitoringGateway` protocol, calls the adapter (never `MlflowClient` directly).
3. Updated `monitor.py` — accepts optional gateway parameter, defaults to MLflow gateway.
4. CLI entry point — `mlflow-monitor run --subject <id> --source-run <id>`.
5. Demo setup script — seeds a local MLflow instance with synthetic training runs.
6. Integration tests — create → prepare → check path against real local MLflow.
7. Local development playground — `mlflow ui` + demo seeder + run loop for visual feedback.

### Out of scope

1. Analyze, close, diff, finding, query, promotion (M2/M3).
2. Retry hardening and concurrency safety (current code handles these minimally — acceptable for MVP).
3. Flexible evidence mapping via recipe (post-MVP — for now, one fixed convention).
4. Recipe file-format loading (JSON/YAML).

### Design posture

Some of what we designed against the in-memory gateway may not survive contact with real MLflow. The MVP is explicitly a discovery exercise — if something is too strict or doesn't work at all, we change it. The contract evidence conventions, tag schema, and write patterns described below are our best starting point, not commitments.

---

## 3. Architecture

No layering changes to the core system. Two new modules in the MLflow integration layer:

```
monitor.py (MODIFIED: gateway injection, MLflow default)
    ↓
orchestration.py (unchanged)
    ↓
workflow.py (unchanged)
    ↓
gateway.py (unchanged — protocol + InMemoryMonitoringGateway)

mlflow_gateway.py (NEW — MLflowMonitoringGateway, calls adapter)
    ↓
mlflow_client.py (NEW — MonitorMLflowClient, thin adapter over MlflowClient)
    ↓
MlflowClient (from mlflow import MlflowClient)
```

The separation:

- **`mlflow_client.py`** — the only file that imports `MlflowClient`. Encapsulates pagination, get-or-create patterns, tag string serialization, error handling, and raw MLflow API quirks. Tested in isolation.
- **`mlflow_gateway.py`** — implements the `MonitoringGateway` protocol. Speaks domain language (timelines, sentinel runs, evidence). Calls the adapter, never `MlflowClient` directly.
- **`gateway.py`** — unchanged. Protocol definition + `InMemoryMonitoringGateway` for unit tests.

If MLflow changes an API or we need to handle a new quirk, we fix `mlflow_client.py` only. The gateway and everything above it stay clean.

---

## 4. MonitorMLflowClient (Adapter)

### 4.1 Responsibility

All raw `MlflowClient` usage lives here. The gateway never touches `MlflowClient` directly.

```python
from mlflow import MlflowClient

class MonitorMLflowClient:
    """Thin adapter over MlflowClient for mlflow-monitor.

    Encapsulates pagination, get-or-create patterns, tag conventions,
    string serialization, and MLflow API error handling.
    """

    def __init__(self, tracking_uri: str | None = None):
        self._client = MlflowClient(tracking_uri=tracking_uri)
```

### 4.2 Operations

**Experiment management:**

```python
def get_or_create_experiment(self, name: str) -> str:
    """Return experiment ID, creating the experiment if it doesn't exist.
    Handles race condition where concurrent processes both try to create."""

def get_experiment_id_by_name(self, name: str) -> str | None:
    """Return experiment ID for a name, or None."""
```

**Run lifecycle:**

```python
def create_run(self, experiment_id: str, tags: dict[str, str]) -> str:
    """Create a run and return its run_id."""

def terminate_run(self, run_id: str, status: str) -> None:
    """Terminate a run with FINISHED or FAILED status."""

def set_tags(self, run_id: str, tags: dict[str, str]) -> None:
    """Set multiple tags on a run."""

def get_run(self, run_id: str) -> Run | None:
    """Return a run or None if not found. Catches MlflowException on missing runs."""
```

**Search (pagination built in):**

```python
def find_runs(
    self, experiment_id: str, filter_string: str,
    order_by: list[str] | None = None, max_results: int | None = None,
) -> list[Run]:
    """Search runs with automatic pagination. Returns all matching runs."""

def find_single_run(self, experiment_id: str, filter_string: str) -> Run | None:
    """Find exactly one run matching filter, or None. Convenience for sentinel/LKG lookups."""
```

**Read helpers:**

```python
def get_run_metrics(self, run_id: str) -> dict[str, float]:
    """Return metrics dict for a run."""

def get_run_params(self, run_id: str) -> dict[str, str]:
    """Return params dict for a run."""

def get_run_tags(self, run_id: str) -> dict[str, str]:
    """Return tags dict for a run."""

def list_artifact_paths(self, run_id: str) -> list[str]:
    """Return list of artifact paths for a run."""
```

**Artifact operations:**

```python
def log_json_artifact(self, run_id: str, data: dict, path: str) -> None:
    """Log a dict as a JSON artifact on a run. Uses client.log_dict()."""
```

### 4.3 What the adapter owns

- Pagination on `search_runs()` (loop until `.token` is None)
- Get-or-create with race-condition try/except for experiments
- String ↔ int serialization for tags like `sequence_index`
- `MlflowException` handling and normalization
- The `from mlflow import MlflowClient` import — nowhere else in the codebase

---

## 5. MLflowMonitoringGateway

### 5.1 Structure

```python
class MLflowMonitoringGateway:
    def __init__(self, config: GatewayConfig, tracking_uri: str | None = None):
        self._config = config
        self._mlflow = MonitorMLflowClient(tracking_uri=tracking_uri)
```

Gateway calls `self._mlflow.*` for all MLflow operations. No `MlflowClient` import here.

### 5.2 Where we READ (training experiments — read only)

For MVP, one fixed convention per evidence type. If the data isn't there, the permissive default contract skips that dimension — no error, just no check.

| Evidence type | MLflow source | Convention | Fallback if absent |
|---|---|---|---|
| Metrics | `run.data.metrics` | Standard MLflow | Empty dict |
| Environment | `run.data.tags` | Tags like `python_version`, `sklearn_version` | Empty dict — env check skipped |
| Features | `run.data.params` or `run.data.tags` | Param or tag `feature_columns` (comma-separated) | Empty tuple — feature check skipped |
| Schema | `run.data.tags` | Tags prefixed with `schema.` (e.g., `schema.age=float64`) | Empty dict — schema check skipped |
| Data scope | `run.data.tags` | Tag `data_scope` (e.g., `"2024-Q1"`) | None — data scope check skipped |

These conventions are our starting point. We may adjust them during implementation.

**We never write to training experiments.** The gateway enforces this.

### 5.3 Where we WRITE (monitoring experiments)

| What we write | MLflow primitive | Why |
|---|---|---|
| Sentinel run (baseline pointer) | Tags on a dedicated run | Queryable: `tags.monitor.role = "timeline_config"` |
| Lifecycle status | Tag: `monitor.lifecycle_status` | Mutable, queryable |
| Comparability status | Tag: `monitor.comparability_status` | Queryable |
| Sequence index | Tag: `monitor.sequence_index` | Queryable for ordering (stored as string) |
| Source run ID | Tag: `monitor.source_run_id` | Traceability |
| Recipe ID + version | Tags: `monitor.recipe_id`, `monitor.recipe_version` | Traceability + idempotency |
| Idempotency key | Tag: `monitor.idempotency_key` | Duplicate detection via search |
| Contract check result | JSON artifact via `log_json_artifact()` | Too complex for tags |
| LKG status | Tag: `monitor.lkg_status` | Queryable: `active` / `superseded` |

Tags for queryable state, artifacts for structured payloads. All monitoring tags use `monitor.` prefix.

### 5.4 Protocol method → implementation sketch

The gateway speaks domain language and delegates MLflow mechanics to the adapter:

```python
def get_timeline_state(self, subject_id):
    exp_id = self._mlflow.get_experiment_id_by_name(
        f"{self._config.namespace_prefix}/{subject_id}"
    )
    if exp_id is None:
        return None
    sentinel = self._mlflow.find_single_run(
        exp_id, 'tags.monitor.role = "timeline_config"'
    )
    if sentinel is None:
        return None
    tags = self._mlflow.get_run_tags(sentinel.info.run_id)
    return TimelineState(
        timeline_id=exp_id,
        baseline_source_run_id=tags["monitor.baseline_source_run_id"],
    )

def resolve_active_lkg_run_id(self, subject_id):
    exp_id = self._resolve_monitoring_experiment_id(subject_id)
    if exp_id is None:
        return None
    run = self._mlflow.find_single_run(
        exp_id, 'tags.monitor.lkg_status = "active"'
    )
    return run.info.run_id if run else None
```

Full protocol mapping:

| Protocol method | Adapter calls used |
|---|---|
| `initialize_timeline` | `get_or_create_experiment`, `find_single_run`, `create_run`, `set_tags` |
| `get_timeline_state` | `get_experiment_id_by_name`, `find_single_run`, `get_run_tags` |
| `reserve_sequence_index` | `find_runs` (find max sequence_index), arithmetic |
| `get_or_create_idempotent_run_id` | `find_single_run` (by idempotency_key tag) |
| `upsert_monitoring_run` | `create_run` or `set_tags`, `log_json_artifact` |
| `get_monitoring_run` | `get_run`, `get_run_tags` |
| `list_timeline_runs` | `find_runs` (exclude sentinel, order by sequence_index) |
| `resolve_source_run_id` | `get_run` or `find_runs` |
| `get_missing_source_run_metrics` | `get_run_metrics` |
| `get_missing_source_run_artifacts` | `list_artifact_paths` |
| `get_source_run_contract_evidence` | `get_run_metrics`, `get_run_tags`, `get_run_params` |
| `resolve_active_lkg_run_id` | `find_single_run` |
| `set_active_lkg_run_id` | `find_single_run`, `set_tags` |
| `mutate_training_run` | Raise unconditionally (no adapter call) |

---

## 6. Monitor API Update

### SDK

```python
from mlflow_monitor import monitor

# Against real MLflow (default)
result = monitor.run(
    subject_id="churn_model",
    source_run_id="abc123",
    baseline_source_run_id="abc123",  # required on first run
)

# Against in-memory (for unit testing)
from mlflow_monitor.gateway import InMemoryMonitoringGateway, GatewayConfig
result = monitor.run(
    subject_id="churn_model",
    source_run_id="abc123",
    baseline_source_run_id="abc123",
    gateway=InMemoryMonitoringGateway(GatewayConfig()),
)
```

### CLI

```bash
mlflow-monitor run --subject churn_model --source-run abc123 --baseline abc123
mlflow-monitor run --subject churn_model --source-run def456
mlflow-monitor run --subject churn_model --source-run abc123 --tracking-uri http://localhost:5000
```

Outputs JSON to stdout. Non-zero exit on failure.

---

## 7. Local MLflow Setup for Development

You need a local MLflow instance to build and test against.

### Option A: File-based (simplest, no server needed)

```bash
# No setup needed — MlflowClient defaults to ./mlruns when no tracking URI is set.
# Just run the demo seeder and it writes to ./mlruns/ in the current directory.
python -m mlflow_monitor.demo.setup

# To browse results in the UI:
mlflow ui --port 5000
# Open http://localhost:5000
```

This is the fastest path for development. Integration tests use this approach with a temp directory.

### Option B: Local tracking server (closer to production)

```bash
# Start a local tracking server with SQLite backend
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db

# Point your code at it
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# Now all MlflowClient calls go through the server
python -m mlflow_monitor.demo.setup
mlflow-monitor run --subject churn_model --source-run <id> --baseline <id>
```

This is useful for verifying the gateway works through the REST API, not just the file store.

### Recommendation for development

Start with Option A (file-based). It's zero-setup and the fastest iteration loop. Switch to Option B when you want to validate REST API behavior or test closer to how real users run MLflow.

---

## 8. Local Development Playground

**Terminal 1:**
```bash
mlflow ui --port 5000
```

**Terminal 2:**
```bash
python -m mlflow_monitor.demo.setup
mlflow-monitor run --subject churn_model --source-run <run_id_1> --baseline <run_id_1>
mlflow-monitor run --subject churn_model --source-run <run_id_2>
```

Open `http://localhost:5000` and see:

- Training experiment with seeded runs (untouched)
- Monitoring experiment `mlflow_monitor/churn_model` with sentinel + monitoring runs
- Tags on monitoring runs showing lifecycle_status, comparability_status, sequence_index
- JSON artifact on monitoring runs containing contract check result

Convenience target:

```makefile
playground:
	mlflow ui --port 5000 &
	sleep 2
	python -m mlflow_monitor.demo.setup
	@echo "Open http://localhost:5000 to see results"
```

---

## 9. Demo Setup Script

`python -m mlflow_monitor.demo.setup` creates:

1. Training experiment `churn_model_training`
2. Run 1 (baseline): accuracy=0.85, f1=0.82, loss=0.35, with environment tags, feature_columns, schema tags, data_scope
3. Run 2 (comparable): same environment/schema/features, different metrics
4. Run 3 (env mismatch): different sklearn_version (triggers `warn`)
5. Run 4 (schema mismatch): different schema tags (triggers `fail`)
6. Prints run IDs for copy-paste into CLI commands

---

## 10. Integration Tests

```
tests/integration/
  conftest.py             # temp mlruns dir per session, cleanup
  test_mlflow_client.py   # MonitorMLflowClient adapter tests
  test_mlflow_gateway.py  # gateway protocol against real MLflow
  test_mlflow_e2e.py      # full monitor.run() against real MLflow
```

No running server needed — local file store only.

Key scenarios:

1. First-run bootstrap creates monitoring experiment + sentinel.
2. Second run finds existing timeline, reuses baseline.
3. Comparable run returns `pass`.
4. Environment mismatch returns `warn`.
5. Schema mismatch returns `fail`.
6. Idempotent rerun returns same result, no duplicate.
7. Monitoring experiment under correct namespace.
8. Training experiment never modified.

---

## 11. File Layout

```
src/mlflow_monitor/
  mlflow_client.py           # NEW — thin MlflowClient adapter
  mlflow_gateway.py          # NEW — gateway implementation (calls adapter)
  monitor.py                 # MODIFIED — gateway injection, MLflow default
  cli.py                     # NEW — thin CLI wrapper

src/mlflow_monitor/demo/
  __init__.py
  setup.py                   # NEW — synthetic training run seeder

tests/integration/
  conftest.py                # NEW — MLflow fixtures
  test_mlflow_client.py      # NEW — adapter unit tests
  test_mlflow_gateway.py     # NEW — gateway protocol tests
  test_mlflow_e2e.py         # NEW — end-to-end tests

pyproject.toml               # MODIFIED — CLI entry point
Makefile                     # NEW — playground convenience
```

Entry point in `pyproject.toml`:

```toml
[project.scripts]
mlflow-monitor = "mlflow_monitor.cli:main"
```

---

## 12. Implementation Order

Focused sequence, one week target:

1. **MonitorMLflowClient adapter** — all raw MLflow API usage in one place. ~1 day.
2. **MLflowMonitoringGateway** — implement protocol against adapter. ~2 days.
3. **Integration tests** — validate adapter and gateway against real MLflow. ~1 day.
4. **Monitor API update** — gateway parameter injection. ~0.5 day.
5. **Demo setup + playground** — seed data, Makefile, verify visual loop. ~0.5 day.
6. **CLI** — thin wrapper, JSON output. ~0.5 day.
7. **README + cleanup** — update Quick Start, verify all tests pass. ~0.5 day.

---

## 13. Risks and Known Limitations

1. **Tag string serialization.** `sequence_index` is int in-memory, string in MLflow. Adapter owns serialization.
2. **Search pagination.** `search_runs()` truncates at 1,000. Adapter paginates automatically.
3. **Sequence index races.** Max + 1 can collide under concurrency. Acceptable for MVP.
4. **Experiment name races.** Adapter handles get-or-create with try/except.
5. **Tag size.** Keys ≤ 255 bytes, values ≤ 5,000 bytes. Contract check results go in artifacts.
6. **Evidence conventions may need adjustment.** Starting point, not commitment.
7. **Retry/concurrency not hardened.** Post-MVP concern.

---

## 14. Success Criteria

MVP is done when:

1. `mlflow-monitor run --subject churn_model --source-run <id>` returns JSON with comparability status.
2. `mlflow_monitor/churn_model` experiment visible in MLflow UI with sentinel + monitoring runs.
3. Training experiments never modified.
4. Demo produces pass, warn, and fail results demonstrating all three comparability outcomes.
5. All 183 existing unit tests still pass.
6. Integration tests pass against local file store.
7. README reflects the real MLflow workflow.
