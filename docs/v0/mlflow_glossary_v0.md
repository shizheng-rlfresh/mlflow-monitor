# MLflow-Monitor v0: MLflow Glossary

## Purpose

This document defines how MLflow-Monitor uses MLflow terminology. MLflow has its own concepts (experiment, run, tag, etc.) and MLflow-Monitor introduces its own (subject, timeline, monitoring run, etc.). This glossary exists to prevent ambiguity during development.

This is an internal reference document, not user-facing.

---

## MLflow Concepts (As MLflow Defines Them)

**Experiment** — A named container for runs. Experiments group related runs together. Each experiment has a unique name and ID. In the MLflow UI, experiments appear as top-level entries in the sidebar.

**Run** — A single execution record inside an experiment. Each run has a unique run ID (UUID), a status (RUNNING, FINISHED, FAILED, KILLED), and a start/end time. Runs contain metrics, params, tags, and artifacts.

**Metric** — A numeric measurement logged during a run. Metrics can be logged multiple times (e.g. at each training epoch) and have a history. Examples: `accuracy`, `f1_score`, `loss`.

**Param (Parameter)** — A key-value string pair logged once per run. Params are typically hyperparameters or configuration values. Examples: `learning_rate`, `batch_size`, `model_type`.

**Tag** — A key-value string pair that can be set or updated on a run at any time. Tags are mutable — unlike params, they can be changed after initial logging. MLflow uses some system tags internally (prefixed with `mlflow.`). Examples: `mlflow.runName`, `mlflow.source.type`.

**Artifact** — A file or directory associated with a run. Artifacts can be any format — model binaries, CSV files, images, JSON files, etc. Artifacts are stored in the artifact store (local filesystem, S3, GCS, etc.) and referenced by path within the run.

**Model** — In MLflow's model registry, a named model with versions. Each version points to an artifact from a specific run. Models can be in stages like "Staging", "Production", "Archived". MLflow-Monitor does not directly interact with the model registry in v0 — it reads from experiments and runs.

**Tracking Server** — The backend service that stores experiment/run/metric/param/tag data. Can be local (file-based) or remote (server with database backend). MLflow-Monitor connects to whatever tracking server the user's MLflow client is configured to use.

---

## MLflow-Monitor Concepts and Their MLflow Mapping

**Subject** — MLflow-Monitor's term for the thing being monitored. A subject maps 1:1 to an MLflow training experiment. "Subject" and "experiment" refer to the same MLflow entity. The subject ID is derived from the experiment name.

**Monitoring run** — An MLflow run created by MLflow-Monitor inside the `{namespace_prefix}/{subject_id}` monitoring experiment. This is distinct from a training run. A monitoring run evaluates a training run — it does not train anything.

**Training run** — An MLflow run inside a training experiment, created by the user's training pipeline. MLflow-Monitor reads from training runs but never writes to them.

**Monitoring experiment** — An MLflow experiment created and owned by MLflow-Monitor under the `{namespace_prefix}/{subject_id}` namespace. Contains monitoring runs, the sentinel run, and all monitoring artifacts.

**Training experiment** — An MLflow experiment created and owned by the user's team. Contains training runs. MLflow-Monitor reads from training experiments but never writes to them.

**Sentinel run** — A special MLflow run inside a monitoring experiment with tag `role=timeline_config`. Holds baseline configuration. Created once per subject at timeline initialization.

**Timeline** — Conceptually, the ordered sequence of monitoring runs for a subject. In MLflow terms, this is the set of monitoring runs in the monitoring experiment, ordered by `sequence_index` tag.

**Baseline** — Conceptually, the pinned reference point. In MLflow terms, this is a `baseline_source_run_id` tag on the sentinel run, pointing to a training run.

**LKG** — Conceptually, the last promoted monitoring run. In MLflow terms, this is the single monitoring run with tag `lkg_status=active` in the monitoring experiment.

---

## Tag Namespace

MLflow-Monitor uses the following custom tags on monitoring runs. All are plain string key-value pairs.

| Tag Key | Set By | Mutability | Purpose |
|---|---|---|---|
| `source_run_id` | Gateway (prepare) | Write-once | Points to the source training run |
| `source_experiment` | Gateway (prepare) | Write-once | Name of the source training experiment |
| `sequence_index` | Gateway (prepare) | Write-once | Monotonic ordering within timeline |
| `lifecycle_status` | Workflow (each stage) | Updated per stage | Current lifecycle state |
| `comparability_status` | Workflow (check) | Write-once | pass / warn / fail |
| `lkg_status` | Gateway (promote) | Updated on promotion | active / superseded |
| `recipe_id` | Workflow (create) | Write-once | Recipe identity |
| `recipe_version` | Workflow (create) | Write-once | Recipe version |
| `contract_id` | Workflow (check) | Write-once | Contract identity |
| `contract_version` | Workflow (check) | Write-once | Contract version |
| `mapping_version` | Gateway (prepare) | Write-once | Schema version (v0) |
| `role` | Gateway (init) | Write-once | Sentinel run marker (`timeline_config`) |
| `baseline_source_run_id` | Gateway (init) | Write-once | On sentinel run only — points to baseline training run |

---

## Disambiguation Quick Reference

| When we say... | We mean... | MLflow primitive |
|---|---|---|
| Subject | The thing being monitored | An MLflow training experiment |
| Timeline | Ordered monitoring history | Set of monitoring runs in a monitoring experiment |
| Monitoring run | One monitoring evaluation | An MLflow run in `{prefix}/{subject_id}` |
| Training run | A run from the user's pipeline | An MLflow run in the training experiment |
| Baseline | The reference starting point | A tag on the sentinel run pointing to a training run |
| LKG | Last promoted good run | A monitoring run with `lkg_status=active` |
| Finding | An interpreted issue | A JSON artifact on a monitoring run |
| Diff | An objective change record | A JSON artifact on a monitoring run |