MLflow-Monitor MVP (v0.1.0)
============================

``v0.1.0`` is the immutable MVP release of MLflow-Monitor. It proved that
baseline-aware monitoring could run against real MLflow state while keeping
Source Training Runs read-only and storing monitoring-owned state separately.

The MVP remains the reproducible release for the original demo. Development on
``main`` targets ``0.2.0`` and does not change what ``v0.1.0`` shipped.

What the MVP shipped
--------------------

The executable MVP lifecycle ends after Check:

.. code-block:: text

   Create -> Prepare -> Check

The release includes:

* explicit Baseline Source Run bootstrap for the first Monitoring Run;
* immutable baseline reuse for later Monitoring Runs on the same subject;
* Contract-based ``pass``, ``warn``, and ``fail`` comparability outcomes;
* monitoring-owned MLflow experiments, Monitoring Runs, and
  ``outputs/result.json`` artifacts;
* idempotent Monitoring Run allocation and recovery of partial allocation
  projections; and
* read-only treatment of Source Training Runs.

Concepts that remained aspirational
-----------------------------------

The original MVP Worldview and Architecture also presented the broader product
direction. The following concepts were not executable in ``v0.1.0``:

* Analyze and Close lifecycle stages;
* metric Diffs and reference-comparison coverage;
* Finding-policy execution and materialized Findings;
* explicit LKG selection and conflict resolution; and
* deployment, Promotion, rollback, notification, and scheduling behavior.

Historical documents
--------------------

The original presentation documents are preserved as historical context. Their
broader concepts should not be read as claims about the executable MVP.

* :doc:`Original MVP Worldview <worldview_v0.1.0>`
* :doc:`Original MVP Architecture <architecture_v0.1.0>`

.. toctree::
   :hidden:

   worldview_v0.1.0
   architecture_v0.1.0

Release and demo
----------------

* `GitHub release <https://github.com/shizheng-rlfresh/mlflow-monitor/releases/tag/v0.1.0>`_
* `Tagged source <https://github.com/shizheng-rlfresh/mlflow-monitor/tree/v0.1.0>`_
* `Tagged README <https://github.com/shizheng-rlfresh/mlflow-monitor/blob/v0.1.0/README.md>`_
* `Executable MVP demo <https://github.com/shizheng-rlfresh/mlflow-monitor/blob/v0.1.0/demo/README.md>`_

