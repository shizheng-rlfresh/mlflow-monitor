# MLflow-Monitor System Architecture

This diagram presents MLflow-Monitor as a compact system architecture, emphasizing subsystem responsibilities, explicit interactions, and the boundary between core monitoring logic and MLflow-backed persistence. It is intended as a public-facing architecture view rather than an execution trace.

```mermaid
flowchart LR
    classDef implemented fill:#ebfbee,stroke:#2b8a3e,stroke-width:1.5px,color:#1e1e1e;
    classDef partial fill:#fff4e6,stroke:#e67700,stroke-width:1.5px,color:#1e1e1e;
    classDef planned fill:#f8f9fa,stroke:#868e96,stroke-width:1.5px,stroke-dasharray: 5 5,color:#1e1e1e;
    classDef unsettled fill:#fff5f5,stroke:#c92a2a,stroke-width:1.5px,stroke-dasharray: 4 4,color:#1e1e1e;
    classDef external fill:#e7f5ff,stroke:#1c7ed6,stroke-width:1.5px,color:#1e1e1e;

    subgraph Outside["Outside Scope"]
        Sched["Scheduling / Triggering"]
        Actions["Downstream Actions"]
        Online["Online Monitoring"]
    end

    subgraph Interfaces["Interfaces"]
        Caller["Caller"]
        SDK["Python SDK (partial)"]
        CLI["Future CLI (planned)"]
    end

    subgraph Recipe["Recipe"]
        RecipeCore["Recipe Schema + Default + Validation (implemented)"]
        RecipeCompile["Run Plan Compilation (unsettled)"]
    end

    subgraph Workflow["Workflow Core"]
        Orchestrator["Run Orchestration (planned)"]
        Stages["Prepare / Check / Analyze / Close / Promote (planned)"]
        Result["Result Contract (implemented)"]
    end

    subgraph Domain["Domain"]
        Entities["Run / Timeline / Baseline / LKG (implemented)"]
        Contracts["Contract + Check Result (implemented)"]
        Diffs["Diff (planned)"]
        Findings["Finding (planned)"]
        Invariants["Invariants (implemented)"]
    end

    subgraph Gateway["Gateway"]
        Resolve["Source Run Resolution (partial)"]
        Persist["Monitoring Persistence (partial)"]
        Sequence["Sequence Index + Idempotency (implemented)"]
        Query["LKG / Query Support (unsettled)"]
    end

    subgraph MLflow["MLflow"]
        Training[("Training Experiments")]
        Monitoring[("Monitoring Namespace")]
    end

    Caller -->|invoke monitor run| SDK
    CLI -->|wrap same run contract| SDK
    Sched -->|decides when to call| SDK
    SDK -->|submit request| Orchestrator
    Orchestrator -->|return result| Actions

    Orchestrator -->|resolve monitoring intent| RecipeCore
    RecipeCore -->|normalize config| RecipeCompile
    RecipeCompile -->|prepared run plan| Orchestrator

    Orchestrator -->|drive stage execution| Stages
    Stages -->|emit public envelope| Result

    Orchestrator -->|apply semantics| Entities
    Orchestrator -->|evaluate comparability| Contracts
    Orchestrator -->|enforce safety rules| Invariants
    Orchestrator -->|generate evidence| Diffs
    Orchestrator -->|generate actionability| Findings

    Orchestrator -->|resolve context / persist outputs| Resolve
    Orchestrator -->|persist lifecycle + artifacts| Persist
    Persist -->|assign ordering + idempotency| Sequence
    Persist -->|stored state for retrieval| Query

    Resolve -->|read-only source access| Training
    Persist -->|system-owned writes| Monitoring
    Query -->|read monitoring state| Monitoring

    Online -. outside v0 .-> Caller

    class RecipeCore,Result,Entities,Contracts,Invariants,Sequence implemented
    class SDK,Resolve,Persist partial
    class CLI,Orchestrator,Stages,Diffs,Findings planned
    class RecipeCompile,Query unsettled
    class Caller,Training,Monitoring,Sched,Actions,Online external
```

**Current status:** domain semantics, invariants, result contract foundations, and gateway abstraction are real today. Full workflow execution, run-plan compilation, diff/finding engines, persisted artifact contract, query contract, and promotion semantics remain partial, planned, or unsettled in the current repository.
