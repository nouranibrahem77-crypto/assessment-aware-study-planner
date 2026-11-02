"""Deterministic presentation-level train, validation, and test assignment."""

from __future__ import annotations

from collections.abc import Iterable

from study_planner.data.feature_engineering.constants import SPLIT_PRESENTATIONS


def split_for_presentation(code_presentation: str) -> str | None:
    """Return the fixed split for a presentation; never randomly repartition."""
    for split_name, presentations in SPLIT_PRESENTATIONS.items():
        if code_presentation in presentations:
            return split_name
    return None


def validate_split_definition() -> None:
    """Fail if a presentation appears in more than one fixed split."""
    presentations = [
        presentation
        for values in SPLIT_PRESENTATIONS.values()
        for presentation in values
    ]
    if len(presentations) != len(set(presentations)):
        raise ValueError("A presentation is assigned to multiple splits")


def presentations_for_split(split_name: str) -> tuple[str, ...]:
    """Return presentations assigned to a named split."""
    try:
        return SPLIT_PRESENTATIONS[split_name]
    except KeyError as error:
        raise ValueError(f"Unknown split: {split_name}") from error


def validate_observed_presentations(presentations: Iterable[str]) -> None:
    """Ensure observed FFF presentations match the requested fixed partitions."""
    validate_split_definition()
    expected = {value for values in SPLIT_PRESENTATIONS.values() for value in values}
    observed = set(presentations)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ValueError(
            "FFF presentations do not match fixed split: " + "; ".join(details)
        )
