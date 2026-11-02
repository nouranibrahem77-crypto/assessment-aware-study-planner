"""Input schema, join-key, and FFF activity-map validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from study_planner.data.feature_engineering.constants import ACTIVITY_TYPE_TO_GROUP
from study_planner.data.feature_engineering.splitting import (
    split_for_presentation,
    validate_observed_presentations,
)
from study_planner.data.oulad import iter_csv_chunks, validate_dataset_files


class FeatureDataError(ValueError):
    """Raised when OULAD rows cannot be joined or interpreted safely."""


@dataclass(frozen=True)
class AssessmentDefinition:
    """Metadata for one FFF assessment."""

    code_module: str
    code_presentation: str
    id_assessment: str
    assessment_type: str
    deadline_day: int | None
    weight: float | None


@dataclass(frozen=True)
class DatasetCatalog:
    """Validated assessment and VLE lookup tables plus source profile counts."""

    assessments: dict[str, AssessmentDefinition]
    site_groups: dict[tuple[str, str, str], str]
    activity_type_counts: dict[str, int]
    source_row_counts: dict[str, int]
    fff_row_counts: dict[str, int]
    fff_missing_counts: dict[str, dict[str, int]]


def _missing(value: str | None) -> bool:
    return value is None or value.strip().casefold() in {
        "",
        "?",
        "na",
        "n/a",
        "null",
        "nan",
    }


def _parse_deadline(value: str | None, filename: str, assessment_id: str) -> int | None:
    if _missing(value):
        return None
    try:
        return int(value or "")
    except ValueError as error:
        raise FeatureDataError(
            f"Invalid deadline {value!r} for assessment {assessment_id} in {filename}"
        ) from error


def _parse_weight(value: str | None, assessment_id: str) -> float | None:
    if _missing(value):
        return None
    try:
        return float(value or "")
    except ValueError as error:
        raise FeatureDataError(
            f"Invalid weight {value!r} for assessment {assessment_id}"
        ) from error


def _update_missing_counts(counts: Counter[str], row: dict[str, str | None]) -> None:
    for column, value in row.items():
        if _missing(value):
            counts[column] += 1


def inspect_fff_inputs(files: dict[str, Path]) -> DatasetCatalog:
    """Validate OULAD schemas and construct unambiguous FFF join indexes."""
    validate_dataset_files(files)
    source_row_counts: Counter[str] = Counter()
    fff_row_counts: Counter[str] = Counter()
    fff_missing_counts: dict[str, Counter[str]] = {
        filename: Counter() for filename in files
    }

    course_keys: set[tuple[str, str]] = set()
    course_presentations: set[str] = set()
    for chunk in iter_csv_chunks(files["courses.csv"], 10_000):
        for row in chunk:
            source_row_counts["courses.csv"] += 1
            if row.get("code_module") != "FFF":
                continue
            fff_row_counts["courses.csv"] += 1
            _update_missing_counts(fff_missing_counts["courses.csv"], row)
            presentation = row.get("code_presentation") or ""
            key = ("FFF", presentation)
            if key in course_keys:
                raise FeatureDataError(f"Duplicate course presentation key: {key}")
            course_keys.add(key)
            course_presentations.add(presentation)
    validate_observed_presentations(course_presentations)

    assessment_by_id: dict[str, AssessmentDefinition] = {}
    all_assessment_ids: set[str] = set()
    for chunk in iter_csv_chunks(files["assessments.csv"], 10_000):
        for row in chunk:
            source_row_counts["assessments.csv"] += 1
            assessment_id = row.get("id_assessment") or ""
            if not assessment_id or assessment_id in all_assessment_ids:
                raise FeatureDataError(
                    f"Missing or duplicate assessment identifier: {assessment_id!r}"
                )
            all_assessment_ids.add(assessment_id)
            if row.get("code_module") != "FFF":
                continue
            fff_row_counts["assessments.csv"] += 1
            _update_missing_counts(fff_missing_counts["assessments.csv"], row)
            presentation = row.get("code_presentation") or ""
            if split_for_presentation(presentation) is None:
                raise FeatureDataError(
                    f"FFF assessment {assessment_id} has unexpected presentation "
                    f"{presentation!r}"
                )
            if ("FFF", presentation) not in course_keys:
                raise FeatureDataError(
                    f"Assessment {assessment_id} has no matching course presentation"
                )
            assessment_type = row.get("assessment_type") or ""
            if not assessment_type:
                raise FeatureDataError(f"Assessment {assessment_id} has no type")
            assessment_by_id[assessment_id] = AssessmentDefinition(
                code_module="FFF",
                code_presentation=presentation,
                id_assessment=assessment_id,
                assessment_type=assessment_type,
                deadline_day=_parse_deadline(
                    row.get("date"), "assessments.csv", assessment_id
                ),
                weight=_parse_weight(row.get("weight"), assessment_id),
            )

    site_groups: dict[tuple[str, str, str], str] = {}
    activity_type_counts: Counter[str] = Counter()
    for chunk in iter_csv_chunks(files["vle.csv"], 10_000):
        for row in chunk:
            source_row_counts["vle.csv"] += 1
            if row.get("code_module") != "FFF":
                continue
            fff_row_counts["vle.csv"] += 1
            _update_missing_counts(fff_missing_counts["vle.csv"], row)
            presentation = row.get("code_presentation") or ""
            if split_for_presentation(presentation) is None:
                raise FeatureDataError(
                    f"FFF VLE row has unexpected presentation {presentation!r}"
                )
            site_id = row.get("id_site") or ""
            activity_type = row.get("activity_type") or ""
            key = ("FFF", presentation, site_id)
            if not site_id or key in site_groups:
                raise FeatureDataError(f"Missing or duplicate VLE join key: {key}")
            if activity_type not in ACTIVITY_TYPE_TO_GROUP:
                raise FeatureDataError(
                    f"Unmapped FFF activity_type {activity_type!r}; update the "
                    "documented activity groups before feature construction"
                )
            site_groups[key] = ACTIVITY_TYPE_TO_GROUP[activity_type]
            activity_type_counts[activity_type] += 1

    return DatasetCatalog(
        assessments=assessment_by_id,
        site_groups=site_groups,
        activity_type_counts=dict(sorted(activity_type_counts.items())),
        source_row_counts=dict(sorted(source_row_counts.items())),
        fff_row_counts=dict(sorted(fff_row_counts.items())),
        fff_missing_counts={
            filename: dict(sorted(counts.items()))
            for filename, counts in sorted(fff_missing_counts.items())
        },
    )
