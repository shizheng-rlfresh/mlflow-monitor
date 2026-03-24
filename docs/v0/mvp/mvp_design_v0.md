# MLflow-Monitor MVP Design

## 1. Purpose

Two goals before resuming the M2 roadmap:

1. **Demo-ready artifact.** A closed loop: `monitor.run()` reads real MLflow training runs, executes create → prepare → check, writes monitoring results to a real MLflow experiment, returns a synchronous result. This is the artifact for going public.

2. **Reality check.** Validate that the gateway protocol and workflow abstractions hold up against real MLflow. Discover what works, what's too strict, and what needs to change before building more on top.

The MVP does not extend the M1 execution boundary. It keeps the M1 create -> prepare -> check slice intact, but it validates that slice against real MLflow and refits the gateway/orchestration boundary wherever the current in-memory assumptions do not hold.

---

## 2. Scope

### In scope

0. Naming cleanup (prerequisite) — rename monitoring-side `run_id` to `monitoring_run_id` across the M1 codebase before MLflow integration work begins. Training-side identifiers such as `source_run_id` and `baseline_source_run_id` stay unchanged. This avoids ambiguity once both monitoring and training run identifiers are real MLflow IDs.
1. `MonitorMLflowClient` — thin adapter over `MlflowClient` encapsulating all raw MLflow API usage.
2. `MLflowMonitoringGateway` — implements `MonitoringGateway` protocol, calls the adapter (never `MlflowClient` directly).
3. Updated `monitor.py` — accepts optional gateway parameter, defaults to MLflow gateway.
4. Optional CLI entry point — `mlflow-monitor run --subject <id> --source-run <id>` if time remains after the SDK path is complete.
5. Demo setup script — seeds a local MLflow instance with synthetic training runs.
6. Integration tests — create → prepare → check path against a dedicated local MLflow store.
7. Local development playground — dedicated local MLflow store directory with `mlflow ui` pointed at it for visual feedback.

### Out of scope

1. Analyze, close, diff, finding, query, promotion (M2/M3).
2. Retry hardening and concurrency safety (acceptable for MVP — single user, sequential runs).
3. Flexible evidence mapping via recipe (post-MVP — for now, one fixed convention).
4. Recipe file-format loading (JSON/YAML).
5. `latest` run selector — caller always provides explicit `source_run_id`.
6. Multiple recipes per subject — one experiment = one subject = one recipe in v0.

### Design posture

Some of what we designed against the in-memory gateway may not survive contact with real MLflow. The MVP is explicitly a discovery exercise — if something is too strict or doesn't work at all, we change it. This includes the M1 code (orchestration, gateway protocol, etc.) — not just the new adapter. The tag schema, evidence conventions, and write patterns described below are our best starting point, not commitments.

---

## 3. Architecture

Two new modules are added in the MLflow integration layer. Core layering is preserved, but the current gateway/orchestration contract will need refitting because real MLflow assigns monitoring run IDs at `create_run()` time rather than letting orchestration allocate them ahead of persistence.

```text
monitor.py (MODIFIED: gateway injection, MLflow default)
    ↓
orchestration.py (MODIFIED: gateway-owned create-or-reuse monitoring run flow)
    ↓
workflow.py (unchanged)
    ↓
gateway.py (MODIFIED — protocol refit + InMemoryMonitoringGateway update)

mlflow_gateway.py (NEW — MLflowMonitoringGateway, calls adapter)
    ↓
mlflow_client.py (NEW — MonitorMLflowClient, thin adapter over MlflowClient)
    ↓
MlflowClient (from mlflow import MlflowClient)
```

The separation:

- **`mlflow_client.py`** — the only file that imports `MlflowClient`. Encapsulates get-or-create patterns, tag read/write, run lifecycle, error handling, and MLflow API quirks. Tested in isolation.
- **`mlflow_gateway.py`** — implements the `MonitoringGateway` protocol. Speaks domain language (timelines, baselines, evidence). Calls the adapter, never `MlflowClient` directly.
- **`gateway.py`** — refit for MLflow-shaped allocation/replay semantics while still keeping `InMemoryMonitoringGateway` as the unit-test implementation.

When the MLflow gateway doesn't fit the existing M1 abstractions, we change whatever needs changing — the adapter, the gateway protocol, or orchestration. `workflow.py` is expected to remain unchanged unless a concrete MLflow constraint proves otherwise.

The key contract change is explicit:

- the current in-memory protocol assumes monitoring run IDs can be allocated before persistence
- real MLflow assigns monitoring run IDs only when the run is created
- the gateway must therefore own create-or-reuse of the monitoring run and return the `monitoring_run_id` plus the sequence/replay context orchestration needs
- orchestration still owns stage progression, failure normalization, and result assembly

---

## 4. MonitorMLflowClient (Adapter)

### 4.1 Responsibility

All raw `MlflowClient` usage lives here. The gateway never touches `MlflowClient` directly.

```python
from mlflow import MlflowClient

class MonitorMLflowClient:
    """Thin adapter over MlflowClient for mlflow-monitor.

    Encapsulates get-or-create patterns, tag read/write,
    string serialization, and MLflow API error handling.
    """

    def __init__(self, tracking_uri: str | None = None):
        self._client = MlflowClient(tracking_uri=tracking_uri)
```

No context manager — `MlflowClient` is stateless (each call is independent).

### 4.2 Operations

**Experiment management:**

```python
def get_or_create_experiment(self, name: str) -> str:
    """Return experiment ID, creating the experiment if it doesn't exist.
    Handles race condition where concurrent processes both try to create."""

def get_experiment_id_by_name(self, name: str) -> str | None:
    """Return experiment ID for a name, or None."""

def get_experiment_tags(self, experiment_id: str) -> dict[str, str]:
    """Return all tags on an experiment."""

def set_experiment_tag(self, experiment_id: str, key: str, value: str) -> None:
    """Set a single tag on an experiment."""
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

**Read helpers:**

```python
def get_run_metrics(self, run_id: str) -> dict[str, float]:
    """Return run.data.metrics for a run."""

def get_run_params(self, run_id: str) -> dict[str, str]:
    """Return run.data.params for a run."""

def get_run_tags(self, run_id: str) -> dict[str, str]:
    """Return run.data.tags for a run."""

def list_artifact_paths(self, run_id: str) -> list[str]:
    """Return list of artifact paths for a run."""
```

**Artifact operations:**

```python
def log_json_artifact(self, run_id: str, data: dict, path: str) -> None:
    """Log a dict as a JSON artifact on a run. Uses client.log_dict()."""
```

### 4.3 What the adapter owns

- Get-or-create with race-condition try/except for experiments
- Soft-delete recovery for monitoring experiments: if MLflow returns the subject's monitoring experiment in `deleted` state, the adapter restores it because the experiment is the timeline/bookkeeping record for that subject, not disposable cache state
- String ↔ int serialization for tags like `next_sequence_index`
- `MlflowException` handling and normalization
- The `from mlflow import MlflowClient` import — nowhere else in the codebase

Monitoring runs and their containing monitoring experiment are treated as
bookkeeping owned by MLflow-Monitor. For MVP, deleting a monitoring experiment
does not mean "start a fresh timeline"; the adapter restores the soft-deleted
experiment so the existing timeline state remains authoritative.

---

## 5. MLflow Tag Schema

### 5.1 Design principles

- **No `search_runs()`.** All lookups are direct via experiment-level tags + `get_run()`. Experiment tags serve as the index. `search_runs()` is intentionally out of the MVP design.
- **Run IDs indexed by sequence.** Each monitoring run ID is stored as `monitoring.run.{sequence_index}` on the experiment. Combined with `monitoring.next_sequence_index`, this enables listing all runs without search.
- **Tag prefix convention:** `training.*` for values pointing to training runs, `monitoring.*` for values about our monitoring runs. No outer `monitor.` prefix needed since the experiment `mlflow_monitor/{subject_id}` is already our namespace.
- **No sentinel run.** The experiment IS the timeline. Experiment tags hold all timeline-level state directly.
- **Intentional MVP simplicity.** Timeline traversal is O(n) direct lookup via experiment tags and `get_run()`. This is an explicit MVP tradeoff for clarity and simplicity, not a claim that the design is optimal at large scale.

### 5.2 Experiment-level tags (timeline state)

These tags live on the `mlflow_monitor/{subject_id}` experiment itself:

| Tag key | Value type | Purpose |
|---|---|---|
| `training.baseline_run_id` | training run ID | The pinned baseline reference |
| `monitoring.lkg_run_id` | monitoring run ID | Current LKG (absent if none promoted) |
| `monitoring.latest_run_id` | monitoring run ID | Most recent monitoring run (for previous-run lookup) |
| `monitoring.next_sequence_index` | integer as string | Next available sequence index |
| `monitoring.run.{index}` | monitoring run ID | Run ID at each sequence index (e.g., `monitoring.run.0`, `monitoring.run.1`) |
| `training.{source_run_id}.monitoring_run_id` | monitoring run ID | Idempotency: maps training run → its monitoring run |

**Idempotency check:** To check if training run `abc123` was already monitored, read experiment tags and look for key `training.abc123.monitoring_run_id`. If present, fetch that monitoring run via `client.get_run()` and verify `monitoring.recipe_id` and `monitoring.recipe_version` match the current request. For MVP with one recipe, the match is always true.

**Previous run lookup:** Read `monitoring.latest_run_id` from experiment tags. That's the previous monitoring run. After creating a new monitoring run, update this tag.

**Timeline traversal:** Read `monitoring.next_sequence_index` from experiment tags, then walk `monitoring.run.0` through `monitoring.run.{N-1}` to resolve ordered monitoring run IDs. Fetch each run directly with `get_run()`. This is the only MVP listing path.

### 5.3 Run-level tags (per monitoring run)

These tags live on each monitoring run within the experiment:

| Tag key | Value type | Purpose |
|---|---|---|
| `training.source_run_id` | training run ID | Which training run this monitoring run evaluated |
| `monitoring.sequence_index` | integer as string | Position in timeline |
| `monitoring.lifecycle_status` | status string | created / prepared / checked / failed |
| `monitoring.comparability_status` | status string | pass / warn / fail (absent if not yet checked) |
| `monitoring.recipe_id` | string | Which recipe was used |
| `monitoring.recipe_version` | string | Which version of the recipe |

### 5.4 Run-level artifacts (per monitoring run)

One artifact, written once at the end of the run:

| Artifact path | Content | Written when |
|---|---|---|
| `outputs/result.json` | Full `MonitorRunResult.to_dict()` — status, comparability, references, error if failed | Once, at end of run |

Tags are upserted at each stage (MLflow tags are mutable, last write wins). The artifact captures the final state only.

---

## 6. MLflowMonitoringGateway

### 6.1 Structure

```python
class MLflowMonitoringGateway:
    def __init__(self, config: GatewayConfig, tracking_uri: str | None = None):
        self._config = config
        self._mlflow = MonitorMLflowClient(tracking_uri=tracking_uri)
```

Gateway calls `self._mlflow.*` for all MLflow operations.

### 6.2 Where we READ (training experiments — read only)

For MVP, one fixed convention per evidence type. If the data isn't there, the permissive default contract skips that dimension.

| Evidence type | MLflow source | Fallback if absent |
|---|---|---|
| Metrics | `run.data.metrics` | Empty dict |
| Environment | `run.data.tags` (e.g., `python_version`, `sklearn_version`) | Empty dict — env check skipped |
| Features | `run.data.params` or `run.data.tags` (`feature_columns`, comma-separated) | Empty tuple — feature check skipped |
| Schema | `run.data.tags` prefixed with `schema.` | Empty dict — schema check skipped |
| Data scope | `run.data.tags` (`data_scope`) | None — data scope check skipped |

These are the standard `run.data.metrics`, `run.data.params`, and `run.data.tags` accessors. If users log via `mlflow.log_metric()`, `mlflow.log_param()`, `mlflow.set_tag()`, we can read it. Anything stored as artifacts is not read for MVP.

**We never write to training experiments.** The gateway enforces this.

### 6.3 Where we WRITE (monitoring experiments)

All writes go to `{namespace_prefix}/{subject_id}` experiments. Tags for queryable state, artifacts for structured payloads. See §5 for full tag schema.

### 6.4 Protocol method → implementation

| Protocol method | How it works (all direct lookups — no search_runs) |
|---|---|
| `initialize_timeline(subject_id, baseline_id)` | Get-or-create experiment. Set experiment tags: `training.baseline_run_id`, `monitoring.next_sequence_index = "0"`. |
| `get_timeline_state(subject_id)` | Get experiment by name. Read `training.baseline_run_id` from experiment tags. |
| `reserve_sequence_index(subject_id)` | Read `monitoring.next_sequence_index` from experiment tags. Increment and write back. Return the read value. |
| `get_or_create_idempotent_run_id(key, factory)` | In the current M1 contract this returns an existing monitoring run ID or creates a new one. For the MLflow-backed gateway, this contract will be refit into a gateway-owned create-or-reuse flow because MLflow assigns the monitoring run ID at run creation time. |
| `upsert_monitoring_run(...)` | Persist stage updates to an existing monitoring run. For a new MLflow-backed run, the gateway-owned create-or-reuse flow will create the run first, then `upsert_monitoring_run(...)` will handle later lifecycle/comparability/result updates. |
| `get_monitoring_run(subject_id, run_id)` | `client.get_run(run_id)`, read run tags, build `MonitoringRunRecord`. |
| `list_timeline_runs(subject_id, exclude_failed)` | Read `monitoring.next_sequence_index` from experiment tags. Loop from 0 to N-1, read `monitoring.run.{i}` to get each run ID. Call `get_run()` for each to build `MonitoringRunRecord`. Filter by lifecycle_status if `exclude_failed`. |
| `resolve_source_run_id(...)` | `client.get_run(source_run_id)` — direct lookup. No `latest` selector in MVP. |
| `get_missing_source_run_metrics(...)` | Read `run.data.metrics`, check required keys. |
| `get_missing_source_run_artifacts(...)` | `client.list_artifacts(run_id)`, check required paths. |
| `get_source_run_contract_evidence(run_id)` | Read `run.data.metrics`, `run.data.tags`, `run.data.params`, build `ContractEvidence`. |
| `resolve_active_lkg_run_id(subject_id)` | Read `monitoring.lkg_run_id` from experiment tags. Direct lookup. |
| `set_active_lkg_run_id(subject_id, run_id)` | Set `monitoring.lkg_run_id` experiment tag. |
| `mutate_training_run(...)` | Raise unconditionally. |

---

## 7. Monitor API Update

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

Outputs JSON to stdout. Non-zero exit on failure. Lightweight wrapper — does not affect post-MVP development.

CLI is optional for this week. The required MVP milestone is the SDK path against real MLflow; the CLI ships only if time remains after the SDK, demo, and tests are green.

---

## 8. Local MLflow Setup for Development

### Dedicated local store

Use a dedicated directory for MLflow data, separate from your project root:

```bash
# Create a dedicated store directory
mkdir -p .mlflow-dev

# Preferred: use a local SQLite backend
export MLFLOW_TRACKING_URI=sqlite:///./.mlflow-dev/mlflow.db

# File-store remains possible for temporary experiments, but MLflow now warns
# that the filesystem backend is deprecated.
export MLFLOW_TRACKING_URI=./.mlflow-dev/mlruns
```

### Development loop

**Terminal 1:**

```bash
mlflow ui --port 5000 --backend-store-uri sqlite:///.mlflow-dev/mlflow.db
```

**Terminal 2:**

```bash
python -m mlflow_monitor.demo.setup
mlflow-monitor run --subject churn_model --source-run <run_id_1> --baseline <run_id_1>
mlflow-monitor run --subject churn_model --source-run <run_id_2>
```

Open `http://localhost:5000` and see:

- Training experiment with seeded runs (untouched)
- Monitoring experiment `mlflow_monitor/churn_model` with monitoring runs
- Experiment tags showing baseline, latest monitoring run, sequence index
- Run tags showing lifecycle_status, comparability_status, source_run_id
- `outputs/result.json` artifact on monitoring runs containing the full run result

Integration tests use the same SQLite-backed approach with a temp directory per
test session.

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
  conftest.py             # dedicated temp mlflow store per session, cleanup
  test_mlflow_client.py   # MonitorMLflowClient adapter tests
  test_mlflow_gateway.py  # gateway protocol against real MLflow
  test_mlflow_e2e.py      # full monitor.run() against real MLflow
```

No running server needed — local SQLite store only.

Key scenarios:

1. First-run bootstrap creates monitoring experiment with correct experiment tags.
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
  cli.py                     # NEW — optional thin CLI wrapper if time remains

src/mlflow_monitor/demo/
  __init__.py
  setup.py                   # NEW — synthetic training run seeder

tests/integration/
  conftest.py                # NEW — MLflow fixtures
  test_mlflow_client.py      # NEW — adapter tests
  test_mlflow_gateway.py     # NEW — gateway protocol tests
  test_mlflow_e2e.py         # NEW — end-to-end tests

pyproject.toml               # MODIFIED — optional CLI entry point if shipped
```

Optional entry point in `pyproject.toml` if CLI ships this week:

```toml
[project.scripts]
mlflow-monitor = "mlflow_monitor.cli:main"
```

---

## 12. Implementation Order

Focused sequence, one week target. Adapter and gateway are built together since they co-evolve:

0. **Rename `run_id` -> `monitoring_run_id` in M1 codebase** — prerequisite before MLflow integration begins. Rename only monitoring-side identifiers. Keep training-side identifiers such as `source_run_id` and `baseline_source_run_id` unchanged. All existing tests must still pass after rename. ~0.5 day.
1. **Gateway/orchestration contract refit** — replace the preallocated monitoring-run-ID assumption with a gateway-owned create-or-reuse flow that supports MLflow-assigned monitoring run IDs and preserves replay behavior. ~1 day.
2. **MonitorMLflowClient + MLflowMonitoringGateway** — build together, read-side first, then write-side. The adapter shapes emerge from what the gateway needs. ~2 days.
3. **Integration tests** — validate adapter, direct-lookup indexing, and gateway behavior against real local MLflow. Run early. ~1 day.
4. **Monitor API update** — gateway parameter injection and MLflow default. Small change. ~0.5 day.
5. **Demo setup + playground** — seed data, verify visual loop in MLflow UI. ~0.5 day.
6. **CLI** — optional thin wrapper, JSON output, only if the SDK path, demo, and tests are already green. ~0.5 day.

---

## 13. Risks and Known Limitations

1. **Tag string serialization.** `sequence_index` is int in-memory, string in MLflow. Adapter owns serialization.
2. **Sequence index races.** Read + increment is not atomic. Acceptable only for the single-user, sequential MVP target.
3. **Experiment tag accumulation.** Indexed run tags (`monitoring.run.{i}`) and idempotency tags (`training.{id}.monitoring_run_id`) grow with each run. Fine for MVP scale. Total tag count per experiment may have practical limits at scale.
4. **Experiment name races.** `create_experiment` fails on duplicate. Adapter handles get-or-create with try/except.
5. **Tag size.** Keys ≤ 255 bytes, values ≤ 5,000 bytes. Structured outputs go in `outputs/result.json` artifact, not tags.
6. **Evidence conventions may need adjustment.** The read conventions are a starting point.
7. **O(n) timeline listing.** Timeline traversal reads experiment tags and then calls `get_run()` for each indexed run. Acceptable for MVP simplicity, but not positioned as a large-scale design.
8. **Orchestration refit required.** Current idempotency/rerun logic was designed for in-memory and must change to support MLflow-assigned monitoring run IDs cleanly.

---

## 14. Success Criteria

MVP is done when:

1. `monitor.run(...)` works against real MLflow and returns the canonical synchronous result with comparability status.
2. `mlflow_monitor/churn_model` experiment is visible in MLflow UI with monitoring runs and correct tags.
3. Training experiments never modified.
4. Demo produces pass, warn, and fail results demonstrating all three comparability outcomes.
5. All existing unit tests still pass against in-memory gateway.
6. Integration tests pass against a local SQLite store.
7. First-run bootstrap and later baseline reuse both work against real MLflow.
8. CLI is optional and is not required for MVP completion this week.
