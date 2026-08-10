"""Utility functions for mlflow-monitor."""

import json
from typing import Any


def canonical_json(data: dict[str, Any]) -> str:
    """Return a canonical JSON string representation of the given dictionary.

    Args:
        data: Dictionary to be converted to a canonical JSON string.

    Returns:
        Canonical JSON string representation of the input dictionary.
    """
    try:
        encoded = json.dumps(
            obj=data,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except ValueError as exc:
        raise ValueError(f"Infinite or NaN value detected in JSON artifact: {exc}") from exc
    except TypeError as exc:
        raise TypeError(f"Non-serializable value detected in JSON artifact: {exc}") from exc

    return encoded
