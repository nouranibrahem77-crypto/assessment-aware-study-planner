"""Tests for OULAD ingestion and bounded chunk summaries."""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import pytest

from study_planner.data.oulad import (
    REQUIRED_COLUMNS,
    DataValidationError,
    extract_archive,
    locate_dataset_files,
    summarize_csv,
    validate_dataset_files,
    write_exploration_report,
)


def _write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def _valid_fixture(directory: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for filename, required in REQUIRED_COLUMNS.items():
        path = directory / filename
        headers = sorted(required)
        _write_csv(path, headers, [["value"] * len(headers)])
        files[filename] = path
    return files


def test_validates_required_files_and_columns(tmp_path: Path) -> None:
    files = _valid_fixture(tmp_path)

    validate_dataset_files(files)
    assert locate_dataset_files(tmp_path) == files


def test_reports_missing_files(tmp_path: Path) -> None:
    _write_csv(tmp_path / "courses.csv", sorted(REQUIRED_COLUMNS["courses.csv"]), [])

    with pytest.raises(DataValidationError, match="missing files"):
        locate_dataset_files(tmp_path)


def test_rejects_invalid_schema(tmp_path: Path) -> None:
    files = _valid_fixture(tmp_path)
    path = files["courses.csv"]
    _write_csv(path, ["code_module", "wrong_column"], [["AAA", "value"]])

    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_dataset_files(files)


def test_summarizes_missing_values_across_chunks(tmp_path: Path) -> None:
    path = tmp_path / "studentVle.csv"
    _write_csv(
        path,
        ["id_student", "sum_click"],
        [["1", "2"], ["", "3"], ["4", "?"], ["5", ""]],
    )

    summary = summarize_csv(path, chunk_size=2)

    assert summary.row_count == 4
    assert summary.missing_counts == {"id_student": 1, "sum_click": 2}


def test_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for filename, required in REQUIRED_COLUMNS.items():
            archive.writestr(filename, ",".join(sorted(required)) + "\n")
        archive.writestr("../outside.csv", "bad\n")

    with pytest.raises(DataValidationError, match="Unsafe path"):
        extract_archive(archive_path, tmp_path / "raw")


def test_report_recommends_synthetic_chronological_split(tmp_path: Path) -> None:
    courses = tmp_path / "courses.csv"
    _write_csv(
        courses,
        ["code_module", "code_presentation", "module_presentation_length"],
        [["AAA", "2013B", "240"], ["AAA", "2013J", "240"]],
    )
    assessments = tmp_path / "assessments.csv"
    _write_csv(
        assessments,
        [
            "code_module",
            "code_presentation",
            "id_assessment",
            "assessment_type",
            "date",
            "weight",
        ],
        [
            ["AAA", presentation, str(assessment_id), "TMA", str(10 * index), "10"]
            for presentation, start_id in [("2013B", 1), ("2013J", 4)]
            for index, assessment_id in enumerate(range(start_id, start_id + 3), 1)
        ],
    )
    student_assessment = tmp_path / "studentAssessment.csv"
    _write_csv(
        student_assessment,
        ["id_assessment", "id_student", "date_submitted", "is_banked", "score"],
        [[str(assessment_id), "1", "12", "0", "75"] for assessment_id in range(1, 7)],
    )
    student_vle = tmp_path / "studentVle.csv"
    _write_csv(
        student_vle,
        [
            "code_module",
            "code_presentation",
            "id_student",
            "id_site",
            "date",
            "sum_click",
        ],
        [["AAA", "2013B", "1", "1", "1", "2"]],
    )
    vle = tmp_path / "vle.csv"
    _write_csv(
        vle,
        [
            "id_site",
            "code_module",
            "code_presentation",
            "activity_type",
            "week_from",
            "week_to",
        ],
        [["1", "AAA", "2013B", "resource", "1", "2"]],
    )
    files = {
        "courses.csv": courses,
        "assessments.csv": assessments,
        "studentAssessment.csv": student_assessment,
        "studentVle.csv": student_vle,
        "vle.csv": vle,
    }

    report_path = tmp_path / "report.md"
    write_exploration_report(files, report_path, chunk_size=2)

    report = report_path.read_text(encoding="utf-8")
    assert "Recommend **AAA**" in report
    assert "2013B" in report
    assert "2013J" in report
