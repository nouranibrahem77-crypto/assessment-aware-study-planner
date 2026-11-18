"""Seeded experiment execution, result persistence, and metric summaries."""

from __future__ import annotations

import csv
import json
import math
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

DEFAULT_EXPERIMENT_SEEDS = (42, 123, 2025, 31415, 271828)


class ExperimentRunStatus(StrEnum):
    """Normalized outcome for one requested experiment run."""

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    INCOMPLETE = "incomplete"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ExperimentCase:
    """One comparable method/scenario/profile combination."""

    method: str
    scenario_id: str
    profile_id: str | None = None
    configuration: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.method or not self.scenario_id:
            raise ValueError("method and scenario_id are required")


@dataclass(frozen=True, slots=True)
class ExperimentOutcome:
    """Output returned by an experiment-specific case evaluator."""

    status: ExperimentRunStatus
    metrics: Mapping[str, Any]
    provenance: Mapping[str, Any] | None = None
    failure_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExperimentRunRecord:
    """Serializable result for one case and seed, including failures."""

    experiment_id: str
    method: str
    scenario_id: str
    profile_id: str | None
    seed: int
    status: ExperimentRunStatus
    metrics: Mapping[str, Any]
    configuration: Mapping[str, Any]
    provenance: Mapping[str, Any]
    failure_reasons: tuple[str, ...] = ()
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Seed-level aggregate for one metric and matched experiment group."""

    experiment_id: str
    method: str
    scenario_id: str
    metric: str
    valid_count: int
    mean: float | None
    standard_deviation: float | None
    feasible_count: int
    infeasible_count: int
    incomplete_count: int
    error_count: int


MetricRunner = Callable[[ExperimentCase, int], ExperimentOutcome]


def _jsonable(value: Any) -> Any:
    """Convert nested mappings/sequences to sorted JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("experiment data cannot contain non-finite numbers")
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON compatible")


def _validate_metrics(
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate numeric metrics and the ordered search expansion trace."""
    normalized: dict[str, Any] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("metric names must be non-empty strings")
        if name == "expanded_order":
            if not isinstance(value, tuple) or any(
                not isinstance(state, Hashable) for state in value
            ):
                raise ValueError("expanded_order must be a tuple of hashable states")
            normalized[name] = _jsonable(value)
            continue
        if value is None:
            normalized[name] = None
        elif isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"metric {name!r} must be numeric or None")
        elif not math.isfinite(float(value)):
            raise ValueError(f"metric {name!r} must be finite or None")
        else:
            normalized[name] = value
    return normalized


def run_experiment(
    experiment_id: str,
    cases: Sequence[ExperimentCase],
    evaluator: MetricRunner,
    *,
    seeds: Sequence[int] = DEFAULT_EXPERIMENT_SEEDS,
) -> tuple[ExperimentRunRecord, ...]:
    """Run every case/seed pair and retain incomplete and exceptional runs.

    Evaluator exceptions are recorded and do not prevent independent runs from
    continuing. A returned ``INCOMPLETE`` status is preserved as such and is
    never interpreted as infeasibility.
    """
    if not experiment_id:
        raise ValueError("experiment_id cannot be empty")
    if not cases:
        raise ValueError("at least one experiment case is required")
    if not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ):
        raise ValueError("seeds must be a non-empty sequence of integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")

    records: list[ExperimentRunRecord] = []
    for case in cases:
        for seed in seeds:
            try:
                outcome = evaluator(case, seed)
                if not isinstance(outcome, ExperimentOutcome):
                    raise TypeError("evaluator must return ExperimentOutcome")
                metrics = _validate_metrics(outcome.metrics)
                records.append(
                    ExperimentRunRecord(
                        experiment_id,
                        case.method,
                        case.scenario_id,
                        case.profile_id,
                        seed,
                        outcome.status,
                        metrics,
                        dict(case.configuration or {}),
                        dict(outcome.provenance or {}),
                        outcome.failure_reasons,
                    )
                )
            except Exception as error:
                records.append(
                    ExperimentRunRecord(
                        experiment_id,
                        case.method,
                        case.scenario_id,
                        case.profile_id,
                        seed,
                        ExperimentRunStatus.ERROR,
                        {},
                        dict(case.configuration or {}),
                        {},
                        error_type=type(error).__name__,
                        error_message=str(error),
                    )
                )
    return tuple(records)


def summarize_runs(
    records: Sequence[ExperimentRunRecord],
) -> tuple[MetricSummary, ...]:
    """Summarize each metric with sample SD and visible status counts."""
    grouped: dict[tuple[str, str, str], list[ExperimentRunRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.experiment_id, record.method, record.scenario_id)].append(
            record
        )

    summaries: list[MetricSummary] = []
    for (experiment_id, method, scenario_id), group in sorted(grouped.items()):
        statuses = Counter(record.status for record in group)
        metric_names = sorted(
            {
                name
                for record in group
                for name in record.metrics
                if name != "expanded_order"
            }
        )
        for metric in metric_names:
            values = [
                float(value)
                for record in group
                if (value := record.metrics.get(metric)) is not None
            ]
            summaries.append(
                MetricSummary(
                    experiment_id,
                    method,
                    scenario_id,
                    metric,
                    len(values),
                    fmean(values) if values else None,
                    stdev(values) if len(values) > 1 else None,
                    statuses[ExperimentRunStatus.FEASIBLE],
                    statuses[ExperimentRunStatus.INFEASIBLE],
                    statuses[ExperimentRunStatus.INCOMPLETE],
                    statuses[ExperimentRunStatus.ERROR],
                )
            )
    return tuple(summaries)


def write_experiment_results(
    records: Sequence[ExperimentRunRecord], output_dir: Path, experiment_id: str
) -> tuple[Path, Path, Path]:
    """Write run CSV, summary CSV, and full JSON records atomically."""
    if not records:
        raise ValueError("cannot write empty experiment results")
    if not experiment_id or any(character in experiment_id for character in "/\\"):
        raise ValueError("experiment_id must be a simple filename component")
    if any(record.experiment_id != experiment_id for record in records):
        raise ValueError("all records must match experiment_id")

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / f"{experiment_id}_runs.csv"
    summary_path = output_dir / f"{experiment_id}_summary.csv"
    json_path = output_dir / f"{experiment_id}_runs.json"
    metric_names = sorted({name for row in records for name in row.metrics})
    run_fields = [
        "experiment_id",
        "method",
        "scenario_id",
        "profile_id",
        "seed",
        "status",
        *metric_names,
        "configuration_json",
        "provenance_json",
        "failure_reasons_json",
        "error_type",
        "error_message",
    ]
    summary_fields = [
        "experiment_id",
        "method",
        "scenario_id",
        "metric",
        "valid_count",
        "mean",
        "standard_deviation",
        "feasible_count",
        "infeasible_count",
        "incomplete_count",
        "error_count",
    ]
    _write_csv(
        runs_path,
        run_fields,
        (
            {
                "experiment_id": row.experiment_id,
                "method": row.method,
                "scenario_id": row.scenario_id,
                "profile_id": row.profile_id,
                "seed": row.seed,
                "status": row.status.value,
                **{
                    name: _dump_json(value) if name == "expanded_order" else value
                    for name, value in row.metrics.items()
                },
                "configuration_json": _dump_json(row.configuration),
                "provenance_json": _dump_json(row.provenance),
                "failure_reasons_json": _dump_json(row.failure_reasons),
                "error_type": row.error_type,
                "error_message": row.error_message,
            }
            for row in records
        ),
    )
    summaries = summarize_runs(records)
    _write_csv(
        summary_path,
        summary_fields,
        ({field: getattr(row, field) for field in summary_fields} for row in summaries),
    )
    _write_json(
        json_path,
        {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "runs": [
                {
                    **{
                        name: getattr(row, name)
                        for name in (
                            "experiment_id",
                            "method",
                            "scenario_id",
                            "profile_id",
                            "seed",
                            "metrics",
                            "configuration",
                            "provenance",
                            "failure_reasons",
                            "error_type",
                            "error_message",
                        )
                    },
                    "status": row.status.value,
                }
                for row in records
            ],
            "summary": [
                {name: getattr(row, name) for name in summary_fields}
                for row in summaries
            ],
        },
    )
    return runs_path, summary_path, json_path


def _dump_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Any) -> None:
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        writer = csv.DictWriter(temporary, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(_jsonable(payload), temporary, indent=2, sort_keys=True)
        temporary.write("\n")
    temporary_path.replace(path)
