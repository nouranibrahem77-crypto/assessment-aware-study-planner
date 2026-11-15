"""Deterministic selection and export of complete student experiment profiles."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    seed: int
    source_sha256: str
    profiles: tuple[StudentProfile, ...]


def select_student_profiles(
    feature_csv: Path,
    *,
    course_module: str,
    presentation: str,
    assessment_ids: tuple[str, ...],
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
        required = {"code_module", "code_presentation", "id_student", "id_assessment"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ProfileSelectionError(
                "feature CSV is missing required OULAD identity columns"
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
            rows = by_student.setdefault(student_id, {})
            if assessment_id in rows:
                raise ProfileSelectionError(
                    f"duplicate row for student {student_id}, assessment {assessment_id}"
                )
            rows[assessment_id] = dict(row)

    eligible_ids = sorted(
        student_id
        for student_id, rows in by_student.items()
        if set(rows) == wanted
    )
    if len(eligible_ids) < count:
        raise ProfileSelectionError(
            f"requested {count} profiles but found {len(eligible_ids)} complete students"
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
        "selection_seed": selection.seed,
        "selection_method": "seeded sample without replacement from complete students",
        "source_csv_sha256": selection.source_sha256,
        "student_ids": [profile.student_id for profile in selection.profiles],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_path.parent, delete=False
    ) as temporary:
        json.dump(payload, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(output_path)
