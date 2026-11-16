"""Deterministic selection and export of complete student experiment profiles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from study_planner.data.feature_engineering.constants import (
    ACTIVITY_GROUPS,
    IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
    click_feature_name,
)
from study_planner.nn.dataset import (
    DatasetSchemaError,
    prepare_inference_frame,
    prepare_previous_marks_inference_frame,
)

STRICT_FFF_FEATURE_NAMES = (
    "assessment_type_cma",
    "assessment_type_exam",
    "assessment_type_tma",
    "assessment_weight",
    "assessment_weight_available",
    "assessment_deadline_day",
    "weeks_since_course_start",
    *(
        click_feature_name(group, week)
        for group in ACTIVITY_GROUPS
        for week in range(1, 5)
    ),
)
ASSESSMENT_TYPE_COLUMNS = STRICT_FFF_FEATURE_NAMES[:3]


class ProfileSelectionError(ValueError):
    """Raised when a feature file cannot provide the requested profiles."""


@dataclass(frozen=True, slots=True)
class StudentProfile:
    """One student's feature rows for the requested assessment set."""

    student_id: str
    rows_by_assessment: tuple[tuple[str, dict[str, str]], ...]


@dataclass(frozen=True, slots=True)
class ProfileSelection:
    """Reproducible profile selection and source-file identity."""

    course_module: str
    presentation: str
    assessment_ids: tuple[str, ...]
    feature_mode: Literal["strict", "proxy"]
    seed: int
    source_sha256: str
    profiles: tuple[StudentProfile, ...]


def select_student_profiles(
    feature_csv: Path,
    *,
    course_module: str,
    presentation: str,
    assessment_ids: tuple[str, ...],
    feature_mode: Literal["strict", "proxy"] | None = None,
    count: int = 30,
    seed: int = 42,
) -> ProfileSelection:
    """Select ``count`` complete students using a stable seeded sample.

    Complete means that the CSV contains exactly one row for every requested
    assessment for that student. The function does not use target scores when
    selecting profiles, avoiding outcome-driven sample selection.
    """
    if not feature_csv.is_file():
        raise FileNotFoundError(feature_csv)
    if not course_module or not presentation:
        raise ProfileSelectionError("course and presentation must be specified")
    if (
        not assessment_ids
        or len(set(assessment_ids)) != len(assessment_ids)
        or any(not item for item in assessment_ids)
    ):
        raise ProfileSelectionError("assessment_ids must be unique and non-empty")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ProfileSelectionError("count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ProfileSelectionError("seed must be an integer")

    wanted = set(assessment_ids)
    by_student: dict[str, dict[str, dict[str, str]]] = {}
    with feature_csv.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {*IDENTIFIER_COLUMNS, TARGET_COLUMN}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ProfileSelectionError(
                "feature CSV is missing required identifiers or target_score"
            )
        feature_names = tuple(
            name
            for name in reader.fieldnames
            if name not in {*IDENTIFIER_COLUMNS, TARGET_COLUMN}
        )
        proxy_names = tuple(
            name
            for lag in range(1, 6)
            for name in (
                f"previous_assessment_score_lag_{lag}",
                f"previous_assessment_score_lag_{lag}_available",
            )
        )
        proxy_schema = (
            *STRICT_FFF_FEATURE_NAMES[:7],
            *proxy_names,
            *STRICT_FFF_FEATURE_NAMES[7:],
        )
        detected_mode = (
            "strict"
            if feature_names == STRICT_FFF_FEATURE_NAMES
            else "proxy"
            if feature_names == proxy_schema
            else None
        )
        if detected_mode is None:
            raise ProfileSelectionError(
                "feature columns do not match the strict or previous-marks schema"
            )
        if feature_mode is not None and feature_mode != detected_mode:
            raise ProfileSelectionError(
                f"requested {feature_mode} mode but CSV uses {detected_mode} schema"
            )
        for row in reader:
            if (
                row["code_module"] != course_module
                or row["code_presentation"] != presentation
                or row["id_assessment"] not in wanted
            ):
                continue
            student_id = row["id_student"]
            assessment_id = row["id_assessment"]
            if not student_id or not assessment_id:
                raise ProfileSelectionError(
                    "feature rows need student and assessment IDs"
                )
            rows = by_student.setdefault(student_id, {})
            if assessment_id in rows:
                raise ProfileSelectionError(
                    f"duplicate row for student {student_id}, "
                    f"assessment {assessment_id}"
                )
            rows[assessment_id] = dict(row)

    eligible_ids = sorted(
        student_id
        for student_id, rows in by_student.items()
        if set(rows) == wanted
    )
    valid_ids: list[str] = []
    for student_id in eligible_ids:
        try:
            for _assessment_id, row in by_student[student_id].items():
                target = float(row[TARGET_COLUMN])
                if not math.isfinite(target) or not 0 <= target <= 100:
                    raise ValueError("target_score must be finite and in [0, 100]")
                feature_frame = pd.DataFrame(
                    [{name: row[name] for name in feature_names}]
                )
                assessment_types = ASSESSMENT_TYPE_COLUMNS
                if detected_mode == "strict":
                    prepare_inference_frame(
                        feature_frame, feature_names, assessment_types
                    )
                else:
                    prepare_previous_marks_inference_frame(
                        feature_frame, feature_names, assessment_types
                    )
        except (KeyError, TypeError, ValueError, DatasetSchemaError) as error:
            raise ProfileSelectionError(
                f"invalid feature/target row for student {student_id}, "
                f"assessment {assessment_id}: {error}"
            ) from error
        valid_ids.append(student_id)
    eligible_ids = valid_ids
    if len(eligible_ids) < count:
        raise ProfileSelectionError(
            f"requested {count} profiles but found "
            f"{len(eligible_ids)} complete students"
        )
    seed_bytes = str(seed).encode("ascii")
    selected_ids = sorted(
        sorted(
            eligible_ids,
            key=lambda student_id: hashlib.sha256(
                seed_bytes + b"\0" + student_id.encode("utf-8")
            ).digest(),
        )[:count]
    )
    profiles = tuple(
        StudentProfile(
            student_id,
            tuple(
                (assessment_id, by_student[student_id][assessment_id])
                for assessment_id in assessment_ids
            ),
        )
        for student_id in selected_ids
    )
    with feature_csv.open("rb") as source_stream:
        digest = hashlib.file_digest(source_stream, "sha256").hexdigest()
    return ProfileSelection(
        course_module,
        presentation,
        assessment_ids,
        detected_mode,
        seed,
        digest,
        profiles,
    )


def write_profile_manifest(selection: ProfileSelection, output_path: Path) -> None:
    """Write a stable JSON manifest containing IDs and reproducibility inputs."""
    payload: dict[str, Any] = {
        "schema_version": 1,
        "course_module": selection.course_module,
        "presentation": selection.presentation,
        "assessment_ids": list(selection.assessment_ids),
        "feature_mode": selection.feature_mode,
        "selection_seed": selection.seed,
        "selection_method": (
            "take the count smallest SHA-256 ranks of seed and student ID; "
            "sort selected IDs"
        ),
        "source_csv_sha256": selection.source_sha256,
        "student_ids": [profile.student_id for profile in selection.profiles],
        "assessment_ids_by_student": {
            profile.student_id: [
                assessment_id for assessment_id, _ in profile.rows_by_assessment
            ]
            for profile in selection.profiles
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_path.parent, delete=False
    ) as temporary:
        json.dump(payload, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(output_path)


def main(argv: list[str] | None = None) -> int:
    """Command line entry point for creating a reproducible profile manifest."""
    parser = argparse.ArgumentParser(
        description="Select complete student profiles from an assessment feature CSV."
    )
    parser.add_argument("--feature-csv", type=Path, required=True)
    parser.add_argument("--course-module", required=True)
    parser.add_argument("--presentation", required=True)
    parser.add_argument("--assessment-ids", nargs="+", required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        selection = select_student_profiles(
            args.feature_csv,
            course_module=args.course_module,
            presentation=args.presentation,
            assessment_ids=tuple(args.assessment_ids),
            count=args.count,
            seed=args.seed,
        )
        write_profile_manifest(selection, args.output)
    except (OSError, ProfileSelectionError) as error:
        parser.error(str(error))
    print(
        f"Wrote {len(selection.profiles)} profiles to {args.output} "
        f"(seed={selection.seed}, source_sha256={selection.source_sha256})"
    )
    return 0
