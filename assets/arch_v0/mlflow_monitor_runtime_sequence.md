# MLflow-Monitor Runtime Sequence

This diagram presents the intended end-to-end interaction model for one MLflow-Monitor run. It focuses on the formal runtime collaboration between interfaces, workflow logic, semantic layers, the gateway, and MLflow-backed storage.

```mermaid
sequenceDiagram
    autonumber
    actor Caller
    participant SDK as Python SDK / CLI
    participant Workflow as Workflow
    participant Recipe as Recipe Layer
    participant Domain as Domain Layer
    participant Gateway as Gateway
    participant Training as MLflow Training
    participant Monitoring as MLflow Monitoring

    Note over Caller,Monitoring: Intended final interaction model for MLflow-Monitor v0.
    Note over Gateway,Monitoring: Gateway is the only MLflow-facing boundary.
    Note over Gateway,Training: Training experiments are read-only inputs.

    Caller->>SDK: invoke monitor.run(subject_id, source_run_id, baseline_source_run_id?, recipe?)
    SDK->>Workflow: submit normalized monitoring request

    rect rgb(245, 245, 245)
        Workflow->>Recipe: resolve recipe or system default
        Recipe-->>Workflow: validated monitoring intent
    end

    rect rgb(245, 245, 245)
        Workflow->>Gateway: resolve source run and timeline context
        Gateway->>Training: read source run metrics, params, tags, artifacts
        Training-->>Gateway: source run evidence
        Gateway->>Monitoring: read baseline, previous, LKG, monitoring state
        Monitoring-->>Gateway: timeline context
        Gateway-->>Workflow: prepared runtime context
    end

    Workflow->>Domain: apply invariants and contract-check semantics
    Domain-->>Workflow: comparability decision rules

    alt Comparable path
        Workflow->>Domain: compute diffs against baseline / previous / LKG / custom reference
        Domain-->>Workflow: structured diffs
        Workflow->>Domain: derive findings and summary
        Domain-->>Workflow: findings and summary
    else Non-comparable path
        Workflow->>Domain: record compatibility-only outcome
        Domain-->>Workflow: compatibility outputs and non-comparable summary
    end

    Workflow->>Gateway: persist monitoring outputs
    Gateway->>Monitoring: write run metadata, check outputs, diffs, findings, summary
    Monitoring-->>Gateway: persistence acknowledgement
    Gateway-->>Workflow: persisted references

    Workflow-->>SDK: structured run result
    SDK-->>Caller: synchronous monitoring result

    opt Promotion enabled after close
        Workflow->>Domain: evaluate promotion decision
        Domain-->>Workflow: promote or hold
        Workflow->>Gateway: update LKG metadata
        Gateway->>Monitoring: set active / superseded LKG state
    end
```

**Current status:** foundational domain semantics, recipe validation, workflow lifecycle primitives, result envelope, and gateway abstraction exist today. End-to-end orchestration, analysis engines, promotion flow, and query APIs are still not implemented as a complete runtime path in the current repository.
