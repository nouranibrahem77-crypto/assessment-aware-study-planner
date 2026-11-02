"""Synthetic tests for FFF chronological features and persistence."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from study_planner.data.feature_engineering.constants import click_feature_name
from study_planner.data.feature_engineering.features import (
    build_features,
    click_window_index,
)
from study_planner.data.feature_engineering.pipeline import run_feature_pipeline
from study_planner.data.feature_engineering.splitting import (
    SPLIT_PRESENTATIONS,
    split_for_presentation,
)
from study_planner.data.feature_engineering.validation import (
    FeatureDataError,
    inspect_fff_inputs,
)
from study_planner.data.oulad import REQUIRED_COLUMNS


def _write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def _fixture(directory: Path, unknown_vle_site: bool = False) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    presentations = [
        presentation
        for values in SPLIT_PRESENTATIONS.values()
        for presentation in values
    ]
    files = {filename: directory / filename for filename in REQUIRED_COLUMNS}

    _write_csv(
        files["courses.csv"],
        ["code_module", "code_presentation", "module_presentation_length"],
        [["FFF", presentation, "240"] for presentation in presentations],
    )
    assessment_rows: list[list[str]] = []
    assessment_ids: dict[str, list[str]] = {}
    for presentation_index, presentation in enumerate(presentations):
        ids = [str(100 + presentation_index * 10 + index) for index in range(1, 4)]
        assessment_ids[presentation] = ids
        for assessment_id, deadline in zip(ids, (20, 40, 60), strict=True):
            assessment_rows.append(
                ["FFF", presentation, assessment_id, "TMA", str(deadline), "10"]
            )
    _write_csv(
        files["assessments.csv"],
        [
            "code_module",
            "code_presentation",
            "id_assessment",
            "assessment_type",
            "date",
            "weight",
        ],
        assessment_rows,
    )

    submissions: list[list[str]] = []
    for ids in assessment_ids.values():
        submissions.extend(
            [
                [ids[0], "student-1", "19", "0", "0"],
                [ids[1], "student-1", "39", "0", "80"],
                [ids[2], "student-1", "59", "0", "90"],
                [ids[2], "student-2", "59", "0", "50"],
            ]
        )
    _write_csv(
        files["studentAssessment.csv"],
        ["id_assessment", "id_student", "date_submitted", "is_banked", "score"],
        submissions,
    )

    vle_rows: list[list[str]] = []
    interaction_rows: list[list[str]] = []
    for presentation in presentations:
        sites = {"resource": "1", "quiz": "2", "forumng": "3"}
        for activity_type, site_id in sites.items():
            vle_rows.append([site_id, "FFF", presentation, activity_type, "1", "10"])
        ids = assessment_ids[presentation]
        interaction_rows.extend(
            [
                ["FFF", presentation, "student-1", "1", "11", "100"],
                ["FFF", presentation, "student-1", "1", "12", "3"],
                ["FFF", presentation, "student-1", "1", "19", "5"],
                ["FFF", presentation, "student-1", "1", "26", "7"],
                ["FFF", presentation, "student-1", "1", "33", "11"],
                ["FFF", presentation, "student-1", "1", "40", "200"],
                ["FFF", presentation, "student-1", "1", "41", "300"],
            ]
        )
        if unknown_vle_site:
            interaction_rows.append(
                ["FFF", presentation, "student-1", "999", "39", "2"]
            )
    _write_csv(
        files["studentVle.csv"],
        [
            "code_module",
            "code_presentation",
            "id_student",
            "id_site",
            "date",
            "sum_click",
        ],
        interaction_rows,
    )
    _write_csv(
        files["vle.csv"],
        [
            "id_site",
            "code_module",
            "code_presentation",
            "activity_type",
            "week_from",
            "week_to",
        ],
        vle_rows,
    )
    return files


def test_fixed_presentation_split_is_chronological() -> None:
    assert split_for_presentation("2013B") == "train"
    assert split_for_presentation("2013J") == "train"
    assert split_for_presentation("2014B") == "validation"
    assert split_for_presentation("2014J") == "test"
    assert split_for_presentation("2015B") is None


def test_click_windows_exclude_cutoff_and_future_days() -> None:
    assert click_window_index(40, 12) == 0
    assert click_window_index(40, 19) == 1
    assert click_window_index(40, 26) == 2
    assert click_window_index(40, 33) == 3
    assert click_window_index(40, 11) is None
    assert click_window_index(40, 40) is None
    assert click_window_index(40, 41) is None


def test_prior_score_missing_is_not_confused_with_zero(tmp_path: Path) -> None:
    files = _fixture(tmp_path)
    catalog = inspect_fff_inputs(files)
    result = build_features(files, catalog, chunk_size=2, score_mode="proxy")

    train_rows = result.rows_by_split["train"]
    student_one_second = next(
        row
        for row in train_rows
        if row.code_presentation == "2013B"
        and row.id_student == "student-1"
        and row.id_assessment == "102"
    )
    student_two_third = next(
        row
        for row in train_rows
        if row.code_presentation == "2013B"
        and row.id_student == "student-2"
        and row.id_assessment == "103"
    )

    assert student_one_second.features["previous_assessment_score_lag_1"] == 0.0
    assert student_one_second.features["previous_assessment_score_lag_1_available"] == 1
    assert student_two_third.features["previous_assessment_score_lag_1"] is None
    assert student_two_third.features["previous_assessment_score_lag_1_available"] == 0


def test_clicks_before_cutoff_aggregate_and_future_clicks_do_not(
    tmp_path: Path,
) -> None:
    files = _fixture(tmp_path)
    result = build_features(
        files, inspect_fff_inputs(files), chunk_size=2, score_mode="proxy"
    )
    target = next(
        row
        for row in result.rows_by_split["train"]
        if row.code_presentation == "2013B"
        and row.id_student == "student-1"
        and row.id_assessment == "102"
    )

    assert target.features[click_feature_name("content_navigation", 1)] == 3
    assert target.features[click_feature_name("content_navigation", 2)] == 5
    assert target.features[click_feature_name("content_navigation", 3)] == 7
    assert target.features[click_feature_name("content_navigation", 4)] == 11
    assert target.features[click_feature_name("assessment_activity", 4)] == 0
    assert target.features[click_feature_name("content_navigation", 4)] == 11


def test_unmatched_clicks_are_missing_but_no_events_are_zero(tmp_path: Path) -> None:
    files = _fixture(tmp_path, unknown_vle_site=True)
    result = build_features(
        files, inspect_fff_inputs(files), chunk_size=3, score_mode="proxy"
    )
    rows = result.rows_by_split["train"]
    affected = next(
        row
        for row in rows
        if row.code_presentation == "2013B"
        and row.id_student == "student-1"
        and row.id_assessment == "102"
    )
    no_events = next(
        row
        for row in rows
        if row.code_presentation == "2013B"
        and row.id_student == "student-2"
        and row.id_assessment == "103"
    )

    assert affected.features[click_feature_name("content_navigation", 4)] is None
    assert affected.features[click_feature_name("collaboration", 4)] is None
    assert no_events.features[click_feature_name("content_navigation", 1)] == 0


def test_duplicate_vle_join_key_is_rejected(tmp_path: Path) -> None:
    files = _fixture(tmp_path)
    path = files["vle.csv"]
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    _write_csv(path, rows[0], [*rows[1:], rows[1]])

    with pytest.raises(FeatureDataError, match="duplicate VLE join key"):
        inspect_fff_inputs(files)


def test_unknown_activity_type_fails_closed(tmp_path: Path) -> None:
    files = _fixture(tmp_path)
    path = files["vle.csv"]
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    rows[1][3] = "new_unreviewed_activity"
    _write_csv(path, rows[0], rows[1:])

    with pytest.raises(FeatureDataError, match="Unmapped FFF activity_type"):
        inspect_fff_inputs(files)


def test_duplicate_student_assessment_submission_fails_closed(
    tmp_path: Path,
) -> None:
    files = _fixture(tmp_path)
    path = files["studentAssessment.csv"]
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    _write_csv(path, rows[0], [*rows[1:], rows[1]])

    with pytest.raises(FeatureDataError, match="Duplicate student-assessment join key"):
        build_features(files, inspect_fff_inputs(files), chunk_size=2)


@pytest.mark.parametrize(
    ("date_submitted", "is_banked", "drop_reason"),
    [("41", "0", "late_submission"), ("39", "1", "banked_resit")],
)
def test_late_and_banked_assessment_scores_are_excluded(
    tmp_path: Path,
    date_submitted: str,
    is_banked: str,
    drop_reason: str,
) -> None:
    files = _fixture(tmp_path)
    path = files["studentAssessment.csv"]
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    for row in rows[1:]:
        if row[0] == "102" and row[1] == "student-1":
            row[2] = date_submitted
            row[3] = is_banked
    _write_csv(path, rows[0], rows[1:])

    result = build_features(
        files, inspect_fff_inputs(files), chunk_size=2, score_mode="proxy"
    )
    assert not any(
        row.id_assessment == "102" and row.id_student == "student-1"
        for row in result.rows_by_split["train"]
    )
    assert result.dropped_records["train"][drop_reason] == 1


def test_strict_mode_has_no_previous_score_features_and_keeps_fixed_rows(
    tmp_path: Path,
) -> None:
    files = _fixture(tmp_path)
    catalog = inspect_fff_inputs(files)
    strict = build_features(files, catalog, chunk_size=2, score_mode="strict")
    proxy = build_features(files, catalog, chunk_size=2, score_mode="proxy")

    assert strict.score_feature_mode == "strict"
    assert not any(
        name.startswith("previous_assessment_score") for name in strict.feature_names
    )
    assert strict.max_previous_score_lags == 0
    assert {key: len(value) for key, value in strict.rows_by_split.items()} == {
        key: len(value) for key, value in proxy.rows_by_split.items()
    }
    assert any(
        name.startswith("previous_assessment_score") for name in proxy.feature_names
    )


def test_prior_scores_never_cross_presentation_boundaries(tmp_path: Path) -> None:
    files = _fixture(tmp_path)
    result = build_features(
        files, inspect_fff_inputs(files), chunk_size=2, score_mode="proxy"
    )
    validation_first = next(
        row
        for row in result.rows_by_split["validation"]
        if row.code_presentation == "2014B"
        and row.id_student == "student-1"
        and row.id_assessment == "121"
    )

    assert validation_first.features["previous_assessment_score_lag_1"] is None
    assert validation_first.features["previous_assessment_score_lag_1_available"] == 0


def test_same_deadline_assessment_is_not_a_prior_score(tmp_path: Path) -> None:
    files = _fixture(tmp_path)
    path = files["assessments.csv"]
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    for row in rows[1:]:
        if row[2] == "103":
            row[4] = "40"
    _write_csv(path, rows[0], rows[1:])
    submissions_path = files["studentAssessment.csv"]
    with submissions_path.open(newline="", encoding="utf-8") as file:
        submissions = list(csv.reader(file))
    for row in submissions[1:]:
        if row[0] == "103" and row[1] == "student-1":
            row[2] = "39"
    _write_csv(submissions_path, submissions[0], submissions[1:])

    result = build_features(
        files, inspect_fff_inputs(files), chunk_size=2, score_mode="proxy"
    )
    same_cutoff = next(
        row
        for row in result.rows_by_split["train"]
        if row.code_presentation == "2013B"
        and row.id_assessment == "103"
        and row.id_student == "student-1"
    )

    assert same_cutoff.features["previous_assessment_score_lag_1"] == 0.0


def test_assessment_type_categories_are_derived_from_training_only(
    tmp_path: Path,
) -> None:
    files = _fixture(tmp_path)
    path = files["assessments.csv"]
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    for row in rows[1:]:
        if row[2] == "121":
            row[3] = "Exam"
    _write_csv(path, rows[0], rows[1:])

    result = build_features(files, inspect_fff_inputs(files), chunk_size=2)
    validation_row = next(
        row
        for row in result.rows_by_split["validation"]
        if row.id_assessment == "121" and row.id_student == "student-1"
    )

    assert result.assessment_type_categories == ("TMA",)
    assert "assessment_type_exam" not in validation_row.features


def test_target_score_changes_do_not_change_input_features(tmp_path: Path) -> None:
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    files_a = _fixture(source_a)
    files_b = _fixture(source_b)
    path = files_b["studentAssessment.csv"]
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    for row in rows[1:]:
        if row[0] == "102" and row[1] == "student-1":
            row[4] = "10"
    _write_csv(path, rows[0], rows[1:])

    result_a = build_features(files_a, inspect_fff_inputs(files_a), chunk_size=2)
    result_b = build_features(files_b, inspect_fff_inputs(files_b), chunk_size=2)
    row_a = next(
        row
        for row in result_a.rows_by_split["train"]
        if row.id_assessment == "102" and row.id_student == "student-1"
    )
    row_b = next(
        row
        for row in result_b.rows_by_split["train"]
        if row.id_assessment == "102" and row.id_student == "student-1"
    )

    assert row_a.target_score != row_b.target_score
    assert row_a.features == row_b.features


def test_feature_pipeline_is_repeatable_and_keeps_target_separate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw"
    _fixture(source)
    output_a = tmp_path / "processed-a"
    output_b = tmp_path / "processed-b"
    report_a = tmp_path / "report-a.md"
    report_b = tmp_path / "report-b.md"

    run_feature_pipeline(source, output_a, report_a, chunk_size=2)
    run_feature_pipeline(source, output_b, report_b, chunk_size=2)

    for filename in (
        "fff_train.csv",
        "fff_validation.csv",
        "fff_test.csv",
        "fff_preprocessing.json",
    ):
        assert (output_a / filename).read_bytes() == (output_b / filename).read_bytes()
    with (output_a / "fff_train.csv").open(newline="", encoding="utf-8") as file:
        headers = next(csv.reader(file))
    assert "target_score" in headers
    assert "date_submitted" not in headers
    assert "score" not in headers
