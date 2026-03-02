# mlflow-monitor: Technical Spec

## API Contract

### Core Usage

```python
import mlflow_monitor as mm

# Setup: point at model, provide baseline
monitor = mm.Monitor(
    model="models:/my-model@champion",
    baseline="s3://bucket/training_data.parquet",
)

# Run default checks
result = monitor.check(new_data)

# Run with custom check
result = monitor.check(new_data, checks=[my_custom_check])
```

### Monitor Class

```python
class Monitor:
    def __init__(
        self,
        model: str,              # MLflow model URI (e.g. "models:/name@alias" or "models:/name/3")
        baseline: str | pd.DataFrame,  # Path, URI, or DataFrame
        checks: list | None = None,    # Default checks to always run (optional)
        config: str | None = None,     # Path to YAML config (optional)
    ):
        """
        Connects to MLflow, resolves model metadata (schema, features, version).
        Loads baseline data into a DataFrame.
        Stores reference profile for checks.
        """

    def check(
        self,
        current: pd.DataFrame | str,  # New data to check (DataFrame or path)
        checks: list | None = None,    # Override or extend default checks
        log_to_mlflow: bool = False,   # Log results as MLflow run
        experiment: str | None = None, # MLflow experiment name (if logging)
    ) -> CheckResult:
        """
        Run monitoring checks on current data against baseline.
        Returns structured result.
        """
```

### CheckResult Object

Inspired by sklearn's design: simple access to core answer, richer detail available.

```python
class CheckResult:
    # Summary
    passed: bool                      # Overall: did all checks pass?
    score: float                      # Aggregate score (0-1, higher = more drift)
    n_checks: int                     # Number of checks executed
    n_failed: int                     # Number that exceeded threshold

    # Detail
    checks: dict[str, SingleCheckResult]  # Per-check results keyed by name

    # Metadata
    model_uri: str
    model_version: str
    baseline_hash: str
    current_hash: str
    timestamp: datetime
    
    # Access patterns
    def to_dict(self) -> dict:        # For serialization / MLflow logging
    def summary(self) -> str:         # Human-readable summary
    def __repr__(self) -> str:        # Quick console output


class SingleCheckResult:
    name: str                         # Check name
    passed: bool                      # Did this check pass?
    value: float                      # Primary metric value
    threshold: float | None           # Threshold used
    details: dict                     # Anything else the check wants to return
                                      # (per-feature stats, p-values, etc.)
```

### Custom Check Interface

A check is any callable that follows this protocol:

```python
def my_check(baseline: pd.DataFrame, current: pd.DataFrame, model_info: ModelInfo) -> SingleCheckResult:
    """
    baseline: the reference data
    current: the new data to evaluate
    model_info: metadata from MLflow (features, schema, version, etc.)
    
    Returns: SingleCheckResult
    """
```

This means a check can be:
- A plain function
- A class with `__call__`
- A lambda (for simple cases)
- A wrapper around Evidently, NannyML, or any other library
- A wrapper around another MLflow model (e.g. a trained anomaly detector)

```python
# Example: custom check using Evidently under the hood
def evidently_drift_check(baseline, current, model_info):
    from evidently.metrics import DataDriftTable
    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=baseline, current_data=current)
    result = report.as_dict()
    
    share_drifted = result["metrics"][0]["result"]["share_of_drifted_columns"]
    return SingleCheckResult(
        name="evidently_data_drift",
        passed=share_drifted < 0.3,
        value=share_drifted,
        threshold=0.3,
        details=result,
    )

# Use it
result = monitor.check(new_data, checks=[evidently_drift_check])
```

### ModelInfo Object

Extracted from MLflow registry. Passed to custom checks so they can use model metadata.

```python
class ModelInfo:
    name: str                  # Registered model name
    version: str               # Model version number
    alias: str | None          # Alias (e.g. "champion") if used
    run_id: str                # Source run ID
    schema: dict | None        # Input/output schema if logged
    features: list[str] | None # Feature names from schema
    tags: dict                 # Model tags from registry
    uri: str                   # Original model URI
```

## Built-in Default Checks

Ship with a small set of useful-out-of-the-box checks. These are the same callables users write — no special status.

### `basic_stats`
Compares mean, std, min, max, null count per feature between baseline and current. Flags features where values deviate significantly. Simple, no scipy required.

### `ks_drift`
Per-feature Kolmogorov-Smirnov test for numerical features. Chi-squared for categoricals. Returns per-feature p-values and drift flags. Configurable threshold (default 0.05).

### `schema_check`
Verifies current data has expected columns, types match baseline. Catches upstream pipeline breakage (missing columns, type changes, new nulls).

These are importable:

```python
from mlflow_monitor.checks import basic_stats, ks_drift, schema_check

result = monitor.check(new_data, checks=[schema_check, basic_stats])
```

When no checks are specified, `monitor.check(new_data)` runs all three defaults.

## MLflow Integration Points

### Reading (from MLflow)

- `mlflow.MlflowClient().get_registered_model()` — model metadata
- `mlflow.MlflowClient().get_model_version()` — version details, run_id
- `mlflow.models.get_model_info()` — input/output schema
- Model tags and aliases via client API

### Writing (to MLflow) — optional

When `log_to_mlflow=True`:

- Creates a new MLflow run in specified experiment
- **Metrics:** `monitor.score`, `monitor.n_failed`, per-check `check.<name>.value`
- **Params:** `model_uri`, `model_version`, `baseline_hash`, `n_features`, `n_baseline_rows`, `n_current_rows`
- **Tags:** `monitoring.type=check`, `monitoring.passed=true|false`, `monitoring.model=<name>`
- **Artifacts:** `result.json` (full serialized CheckResult)

This means MLflow's built-in UI charting works over time — each `check()` call with logging becomes a run, and you get metric trajectories for free.

## Optional YAML Config

For repeatable setups (e.g. in a cron job or Airflow DAG):

```yaml
model: "models:/my-model@champion"
baseline: "s3://bucket/training_data.parquet"

checks:
  - mlflow_monitor.checks.schema_check
  - mlflow_monitor.checks.ks_drift
  - mypackage.checks:custom_check    # dotted path to user callable

settings:
  log_to_mlflow: true
  experiment: "Model X Monitoring"
  threshold: 0.05                    # default threshold for built-in checks
```

Load and run:

```python
monitor = mm.Monitor.from_config("monitor.yaml")
result = monitor.check(new_data)
```

Or from CLI (stretch goal):

```bash
mlflow-monitor check --config monitor.yaml --current new_data.parquet
```

## Baseline Loading

The baseline parameter accepts:

1. **pandas DataFrame** — used directly
2. **Local file path** — `.csv`, `.parquet`, `.json` (auto-detected by extension)
3. **S3/GCS/Azure URI** — delegated to `pandas.read_parquet()` / `read_csv()` with fsspec
4. **Any string** — attempted as pandas-readable path

The baseline is loaded once at `Monitor.__init__()` and held in memory. For large baselines, a future optimization could store a statistical profile instead of the full DataFrame — but start simple.

## Project Structure

```
mlflow-monitor/
├── pyproject.toml
├── README.md
├── src/
│   └── mlflow_monitor/
│       ├── __init__.py          # Public API: Monitor, CheckResult, etc.
│       ├── monitor.py           # Monitor class
│       ├── result.py            # CheckResult, SingleCheckResult
│       ├── model_info.py        # ModelInfo + MLflow registry extraction
│       ├── baseline.py          # Baseline loading logic
│       ├── config.py            # YAML config parsing
│       ├── checks/
│       │   ├── __init__.py      # Exports default checks
│       │   ├── basic_stats.py
│       │   ├── ks_drift.py
│       │   └── schema_check.py
│       └── logging.py           # MLflow logging logic
├── tests/
│   ├── test_monitor.py
│   ├── test_checks.py
│   ├── test_result.py
│   ├── test_baseline.py
│   └── test_model_info.py
└── demo/
    ├── demo_basic.py            # Simple usage example
    ├── demo_custom_check.py     # Custom check example
    └── demo_config.yaml         # Config-driven example
```

## Dependencies

### Required
- `mlflow` (>=2.0) — registry access, optional logging
- `pandas` — data handling
- `scipy` — built-in statistical tests (KS, chi-squared)
- `numpy` — numerical operations
- `pyyaml` — config file parsing
- `click` — CLI (stretch goal)

### Not Required
- No Evidently, NannyML, or any drift library — users bring their own if they want
- No Jinja2 — no HTML reports
- No visualization libraries — MLflow UI handles that

## Key Design Decisions

### Why not make checks classes with fit/transform?
Overengineering for the scope. A callable with a clear signature is simpler, easier to write, easier to test. If the community asks for stateful checks later, we can add a protocol. Start with functions.

### Why hold the full baseline in memory?
Simplicity. Profile-based comparison (storing summary statistics instead of raw data) is an optimization for large datasets. It also limits what custom checks can do — they might need the raw data. Start with full DataFrame, optimize later if it's a real problem.

### Why not integrate deeper with MLflow's evaluate()?
`mlflow.evaluate()` is point-in-time model evaluation on a single dataset. It doesn't compare two datasets. Forcing monitoring into the evaluate paradigm would be awkward. Our `check()` is a different operation — it compares baseline vs. current.

### Why optional MLflow logging instead of always-on?
Not every check call needs to be a run. During development and debugging, you just want the result object. Logging is for production pipelines. Let the user decide.
