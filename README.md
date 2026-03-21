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

print(result.lifecycle_status)      # created / prepared / checked / ...
print(result.comparability_status)  # pass / warn / fail / None
```

## License

Apache-2.0
