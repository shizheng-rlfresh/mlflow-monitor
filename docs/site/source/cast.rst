Core Actors and State Transitions
=================================

Timeline and LKG Selection
--------------------------

A ``Timeline`` is the ordered read model for a monitored subject. Its
``TimelineEntry`` values identify monitoring and source runs, their sequence,
and their terminal lifecycle and comparability states. Only closed and failed
Monitoring Runs appear as entries. The baseline source run remains optional until
the subject has been initialized. A Timeline does not own an active Contract or
an active last-known-good pointer.

An ``LKGSelection`` records an explicit user-owned trust decision for one
Timeline entry. Its supersession identifiers preserve replacement history
without embedding mutable selection state in the Timeline.

Domain
------

.. automodule:: mlflow_monitor.domain
   :members:
   :show-inheritance:

Invariant
---------

.. automodule:: mlflow_monitor.invariant
   :members:
   :show-inheritance:

Result Contract
----------------

.. automodule:: mlflow_monitor.result_contract
   :members:
   :show-inheritance:

Identity
---------

.. automodule:: mlflow_monitor.identity
   :members:
   :show-inheritance:

Finding Policy
--------------

.. automodule:: mlflow_monitor.finding_policy
   :members:
   :show-inheritance:
