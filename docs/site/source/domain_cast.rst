Domain Models
=============

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

.. automodule:: mlflow_monitor.domain.timeline
   :members:
   :show-inheritance:


Lifecycle
---------

.. automodule:: mlflow_monitor.domain.lifecycle
   :members:
   :show-inheritance:


Contract
--------

.. automodule:: mlflow_monitor.domain.contract
   :members:
   :show-inheritance:

Diff
----

.. automodule:: mlflow_monitor.domain.diff
   :members:
   :show-inheritance:

Reference
---------

.. automodule:: mlflow_monitor.domain.reference
   :members:
   :show-inheritance:

Finding
-------

.. automodule:: mlflow_monitor.domain.finding
   :members:
   :show-inheritance:




