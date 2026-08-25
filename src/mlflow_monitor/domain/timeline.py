"""Timeline domain models for mlflow-monitor v0."""

from dataclasses import dataclass

from .contract import ComparabilityStatus
from .lifecycle import LifecycleStatus


@dataclass(frozen=True, slots=True)
class TimelineEntry:
    """Compact summary of one Monitoring Run in a Timeline.

    Attributes:
        monitoring_run_id: Monitoring Run summarized by this entry.
        source_run_id: Source Training Run evaluated by the Monitoring Run.
        sequence_index: Stable allocation order within the Timeline.
        lifecycle_status: Terminal lifecycle state of the Monitoring Run.
        comparability_status: Contract-check outcome when available.
    """

    monitoring_run_id: str
    source_run_id: str
    sequence_index: int
    lifecycle_status: LifecycleStatus
    comparability_status: ComparabilityStatus | None

    def __post_init__(self) -> None:
        """Validate the immutable Timeline Entry shape."""
        for field_name in ("monitoring_run_id", "source_run_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"TimelineEntry requires a non-empty string for field {field_name!r}."
                )
        if (
            not isinstance(self.sequence_index, int)
            or isinstance(self.sequence_index, bool)
            or self.sequence_index < 0
        ):
            raise ValueError("TimelineEntry sequence_index must be a nonnegative integer.")
        if not isinstance(self.lifecycle_status, LifecycleStatus):
            raise ValueError("TimelineEntry lifecycle_status must be a LifecycleStatus.")
        if self.lifecycle_status not in (LifecycleStatus.CLOSED, LifecycleStatus.FAILED):
            raise ValueError("TimelineEntry lifecycle_status must be closed or failed.")
        if self.comparability_status is not None and not isinstance(
            self.comparability_status, ComparabilityStatus
        ):
            raise ValueError(
                "TimelineEntry comparability_status must be a ComparabilityStatus or None."
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize this Timeline Entry into a deterministic dictionary.

        Returns:
            JSON-compatible Timeline Entry content.
        """
        return {
            "monitoring_run_id": self.monitoring_run_id,
            "source_run_id": self.source_run_id,
            "sequence_index": self.sequence_index,
            "lifecycle_status": self.lifecycle_status.value,
            "comparability_status": (
                self.comparability_status.value if self.comparability_status is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class Timeline:
    """Ordered monitoring history for one Subject.

    Attributes:
        timeline_id: Stable Timeline identifier.
        subject_id: Subject whose monitoring history this Timeline describes.
        baseline_source_run_id: Pinned Baseline Source Run, or None before
            bootstrap succeeds.
        entries: Canonically ordered Monitoring Run summaries.

    Note:
        A Timeline with an empty baseline cannot have closed entries;
            only failed entries are allowed with an empty baseline.
    """

    timeline_id: str
    subject_id: str
    baseline_source_run_id: str | None
    entries: tuple[TimelineEntry, ...]

    def __post_init__(self) -> None:
        """Validate identity and defensively freeze ordered entries."""
        for field_name in ("timeline_id", "subject_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Timeline requires a non-empty string for field {field_name!r}.")
        if self.baseline_source_run_id is not None and (
            not isinstance(self.baseline_source_run_id, str)
            or not self.baseline_source_run_id.strip()
        ):
            raise ValueError("Timeline baseline_source_run_id must be a non-empty string or None.")
        if isinstance(self.entries, str):
            raise ValueError("Timeline entries must be a collection of TimelineEntry values.")
        try:
            supplied_entries = tuple(self.entries)
        except TypeError as exc:
            raise ValueError(
                "Timeline entries must be a collection of TimelineEntry values."
            ) from exc
        if any(not isinstance(entry, TimelineEntry) for entry in supplied_entries):
            raise ValueError("Timeline entries must contain only TimelineEntry values.")

        # Timeline without valid baseline cannot have closed entries,
        # only failed entries are allowed.
        if self.baseline_source_run_id is None and any(
            entry.lifecycle_status == LifecycleStatus.CLOSED for entry in supplied_entries
        ):
            raise ValueError(
                "Timeline cannot accept closed entries without a baseline_source_run_id."
            )

        entries = tuple(sorted(supplied_entries, key=lambda entry: entry.sequence_index))
        sequence_indexes = tuple(entry.sequence_index for entry in entries)
        if len(sequence_indexes) != len(set(sequence_indexes)):
            raise ValueError("Timeline entries must have unique sequence_index values.")
        monitoring_run_ids = tuple(entry.monitoring_run_id for entry in entries)
        if len(monitoring_run_ids) != len(set(monitoring_run_ids)):
            raise ValueError("Timeline entries must have unique monitoring_run_id values.")
        object.__setattr__(self, "entries", entries)

    def to_dict(self) -> dict[str, object]:
        """Serialize this Timeline into a deterministic dictionary.

        Returns:
            JSON-compatible Timeline content with ordered entries.
        """
        return {
            "timeline_id": self.timeline_id,
            "subject_id": self.subject_id,
            "baseline_source_run_id": self.baseline_source_run_id,
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclass(frozen=True, slots=True)
class LKGSelection:
    """Immutable user selection of one trusted Monitoring Run.

    Attributes:
        lkg_selection_id: Stable identity of this trust-selection event.
        timeline_id: Timeline whose trust state this selection updates.
        monitoring_run_id: Selected Monitoring Run.
        source_run_id: Source Training Run evaluated by the selected Monitoring Run.
        supersedes_lkg_selection_ids: Prior selection identities superseded by this
            selection.
    """

    lkg_selection_id: str
    timeline_id: str
    monitoring_run_id: str
    source_run_id: str
    supersedes_lkg_selection_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate identity and freeze supersession history canonically."""
        for field_name in (
            "lkg_selection_id",
            "timeline_id",
            "monitoring_run_id",
            "source_run_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"LKGSelection requires a non-empty string for field {field_name!r}."
                )

        supplied_ids = self.supersedes_lkg_selection_ids
        if isinstance(supplied_ids, str):
            raise ValueError("LKGSelection supersedes_lkg_selection_ids must be a collection.")
        try:
            supersedes_ids = tuple(supplied_ids)
        except TypeError as exc:
            raise ValueError(
                "LKGSelection supersedes_lkg_selection_ids must be a collection."
            ) from exc
        if any(
            not isinstance(selection_id, str) or not selection_id.strip()
            for selection_id in supersedes_ids
        ):
            raise ValueError(
                "LKGSelection supersedes_lkg_selection_ids must contain non-empty strings."
            )

        if self.lkg_selection_id in supersedes_ids:
            raise ValueError("LKGSelection cannot supersede itself.")

        object.__setattr__(
            self,
            "supersedes_lkg_selection_ids",
            tuple(sorted(set(supersedes_ids))),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize this LKG Selection into a deterministic dictionary.

        Returns:
            JSON-compatible LKG Selection content.
        """
        return {
            "lkg_selection_id": self.lkg_selection_id,
            "timeline_id": self.timeline_id,
            "monitoring_run_id": self.monitoring_run_id,
            "source_run_id": self.source_run_id,
            "supersedes_lkg_selection_ids": list(self.supersedes_lkg_selection_ids),
        }
