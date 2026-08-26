"""Utility functions for custom exception types in MLflow Monitor."""

from collections.abc import Mapping


def render_message_fields(
    fields: tuple[tuple[str, str | int | None], ...] | Mapping[str, str],
) -> str:
    """Render message fields for error messages."""
    if isinstance(fields, Mapping):
        return ", ".join(sorted(fields))
    return ", ".join(f"{name}={value!r}" for name, value in fields)
