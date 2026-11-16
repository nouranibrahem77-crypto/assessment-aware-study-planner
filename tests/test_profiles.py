"""Tests for deterministic student-profile selection and validation."""

import csv
import json

import pytest

from study_planner.data.feature_engineering.constants import (
    IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
)
from study_planner.eval.profiles import (
    STRICT_FFF_FEATURE_NAMES,
    ProfileSelectionError,
    select_student_profiles,
    write_profile_manifest,
)

ASSESSMENT_IDS = ("34899", "34900", "34901")
COURSE = "FFF"
PRESENTATION = "2014J"


def _headers() -> list[str]:
    return [*IDENTIFIER_COLUMNS, *STRICT_FFF_FEATURE_NAMES, TARGET_COLUMN]


def _row(student_id: str, assessment_id: str) -> dict[str, str]:
    row = {name: "1" for name in _headers()}
    row.update(
        {
            "code_module": COURSE,
            "code_presentation": PRESENTATION,
            "id_student": student_id,
            "id_assessment": assessment_id,
            "assessment_type_cma": "0",
            "assessment_type_exam": "0",
            "assessment_type_tma": "1",
            "target_score": "50",
        }
    )
    return row


def _write_features(path, student_count: int = 32) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_headers())
        writer.writeheader()
        for student_number in range(student_count):
            for assessment_id in ASSESSMENT_IDS:
                writer.writerow(_row(f"{1000 + student_number}", assessment_id))


def _proxy_headers() -> list[str]:
    prior_features = tuple(
        name
        for lag in range(1, 6)
        for name in (
            f"previous_assessment_score_lag_{lag}",
            f"previous_assessment_score_lag_{lag}_available",
        )
    )
    return [
        *IDENTIFIER_COLUMNS,
        *STRICT_FFF_FEATURE_NAMES[:7],
        *prior_features,
        *STRICT_FFF_FEATURE_NAMES[7:],
        TARGET_COLUMN,
    ]


def _select(path, *, count: int = 30, seed: int = 42):
    return select_student_profiles(
        path,
        course_module=COURSE,
        presentation=PRESENTATION,
        assessment_ids=ASSESSMENT_IDS,
        count=count,
        seed=seed,
    )


def test_profile_selection_is_repeatable_and_manifest_records_provenance(
    tmp_path,
) -> None:
    feature_path = tmp_path / "fff_test.csv"
    _write_features(feature_path)

    first = _select(feature_path)
    second = _select(feature_path)
    manifest_path = tmp_path / "profiles.json"
    write_profile_manifest(first, manifest_path)

    assert [item.student_id for item in first.profiles] == [
        item.student_id for item in second.profiles
    ]
    assert len(first.profiles) == 30
    assert first.feature_mode == "strict"
    assert all(
        tuple(assessment for assessment, _ in profile.rows_by_assessment)
        == ASSESSMENT_IDS
        for profile in first.profiles
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["selection_seed"] == 42
    assert manifest["selection_method"]
    assert manifest["source_csv_sha256"] == first.source_sha256
    assert len(manifest["student_ids"]) == 30


def test_profile_selection_rejects_too_few_complete_students(tmp_path) -> None:
    feature_path = tmp_path / "fff_test.csv"
    _write_features(feature_path, student_count=2)

    with pytest.raises(ProfileSelectionError, match="found 2 complete students"):
        _select(feature_path, count=3)


def test_profile_selection_rejects_duplicate_student_assessment_rows(tmp_path) -> None:
    feature_path = tmp_path / "fff_test.csv"
    _write_features(feature_path, student_count=2)
    with feature_path.open("a", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=_headers()).writerow(
            _row("1000", ASSESSMENT_IDS[0])
        )

    with pytest.raises(ProfileSelectionError, match="duplicate row"):
        _select(feature_path, count=1)


def test_profile_selection_rejects_invalid_numeric_feature_and_target(tmp_path) -> None:
    feature_path = tmp_path / "fff_test.csv"
    _write_features(feature_path, student_count=2)
    with feature_path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["click_content_navigation_week_1"] = "not-a-number"
    with feature_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_headers())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ProfileSelectionError, match="invalid feature/target row"):
        _select(feature_path, count=1)


def test_profile_selection_rejects_out_of_range_target(tmp_path) -> None:
    feature_path = tmp_path / "fff_test.csv"
    _write_features(feature_path, student_count=2)
    with feature_path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["target_score"] = "101"
    with feature_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_headers())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ProfileSelectionError, match="target_score must be finite"):
        _select(feature_path, count=1)


def test_profile_selection_validates_previous_marks_proxy_schema(tmp_path) -> None:
    feature_path = tmp_path / "fff_proxy_test.csv"
    headers = _proxy_headers()
    with feature_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        for assessment_id in ASSESSMENT_IDS:
            row = _row("1000", assessment_id)
            for lag in range(1, 6):
                row[f"previous_assessment_score_lag_{lag}"] = "50"
                row[f"previous_assessment_score_lag_{lag}_available"] = "1"
            writer.writerow(row)

    selection = select_student_profiles(
        feature_path,
        course_module=COURSE,
        presentation=PRESENTATION,
        assessment_ids=ASSESSMENT_IDS,
        feature_mode="proxy",
        count=1,
    )

    assert selection.feature_mode == "proxy"
    assert len(selection.profiles) == 1
