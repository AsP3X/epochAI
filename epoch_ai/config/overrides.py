"""Apply dotted-key overrides to nested config dicts before Pydantic validation."""

from __future__ import annotations

from typing import Any


def apply_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``base`` with dotted keys merged (e.g. ``walk_forward.step_size``)."""
    result = _deep_copy(base)
    for key, value in overrides.items():
        parts = key.split(".")
        if not parts or not parts[0]:
            raise ValueError(f"Invalid override key: {key!r}")
        target: dict[str, Any] = result
        for part in parts[:-1]:
            nested = target.get(part)
            if nested is None:
                nested = {}
                target[part] = nested
            if not isinstance(nested, dict):
                raise ValueError(f"Cannot override {key!r}: {part!r} is not a mapping.")
            target = nested
        target[parts[-1]] = value
    return result


def parse_set_args(items: list[str]) -> dict[str, Any]:
    """Parse ``--set key=value`` CLI tokens into an overrides dict."""
    import yaml

    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, _, raw = item.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"--set expects key=value, got {item!r}")
        overrides[key] = yaml.safe_load(raw.strip())
    return overrides


def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    return value
