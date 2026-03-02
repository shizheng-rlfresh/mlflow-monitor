# mlflow-monitor: Plan

## What

An MLflow-native model monitoring framework. Point at a registered model, bring your data, run any check — built-in or custom.

## Why

MLflow has zero monitoring. Drift detection libraries (Evidently, NannyML, etc.) exist in isolation — none integrate natively with MLflow's registry. The only bridge today is copy-pasted glue code. This is the bridge, packaged properly.

## Core Idea

- **Model from registry** — schema, features, metadata come from MLflow
- **Baseline from user** — they tell us where their data lives, once
- **Checks are pluggable** — ship useful defaults, but the real value is running whatever logic the user defines (callable, pyfunc, another MLflow model)
- **Results are structured** — sklearn-style: simple by default, richer if you ask
- **Output is pluggable** — MLflow logging, JSON, extensible to dashboards (Grafana, etc.)

## What It Does NOT Do

No dashboards. No scheduling. No alerting. No prescribed algorithms. Stays in its lane.

## Milestones

**M1: Walking skeleton** — Monitor class, model info from registry, baseline loading, result objects. Can run a dummy check end-to-end.

**M2: Checks** — Built-in defaults (schema, basic stats, KS drift). Custom check interface (any callable).

**M3: Integration** — MLflow logging, YAML config, output connector interface.

**M4: Ship** — Demo, README, tests, packaging.

## MLflow 3 Compatibility

MLflow 3 went all-in on GenAI: tracing, LLM judges, prompt management, agent evaluation, AI Gateway. Classical ML monitoring was not addressed — no drift detection, no baseline comparison, no distributional checks. The gap got wider, not smaller.

This means our tool fills deliberately abandoned territory. We're not competing with MLflow's roadmap — we're covering what they chose not to.

Useful MLflow 3 features to leverage:
- `LoggedModel` entity — richer model metadata to extract
- Model Registry webhooks — could trigger monitoring checks on new version registration
- OpenTelemetry integration — potential future path for exporting monitoring metrics

Target compatibility: MLflow >=2.0, designed to work with both 2.x and 3.x.

## Open Questions

- How clean is MLflow's registry API for extracting schema/features? Spike first.
- Model comparison (A vs B on same data) — variant of check() or separate method?
- Baseline storage: full DataFrame in memory for now, profile-based later?
- Output connectors: how far to go in v1? MLflow + JSON might be enough.
- CLI: nice to have or essential?
