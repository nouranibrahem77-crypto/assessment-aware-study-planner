"""Focused coverage for the documented 30-profile coursework runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from study_planner.integration import coursework
from study_planner.integration.coursework import (
    execute_coursework_profiles,
    load_coursework_scenario,
    planner_result_record,
)
from study_planner.integration.planner import PlannerOutcome, PlannerStatus

ROOT = Path(__file__).resolve().parents[1]


def test_scenario_uses_documented_values_and_matching_model_features() -> None:
    scenario = load_coursework_scenario(ROOT)

    assert scenario.config.presentation_split.course_identity == "FFF"
    assert [rule.assessment_id for rule in scenario.config.assignment_rules] == [
        "34899",
        "34900",
        "34901",
    ]
    assert [rule.target_mark for rule in scenario.config.assignment_rules] == [
        55,
        55,
        55,
    ]
    assert scenario.config.slot_catalog.weekly_capacity == 3
    assert len(scenario.config.slot_catalog.slots) == 42
    assert scenario.config.click_mapping.scenario.value == "central"
    assert scenario.config.click_mapping.activity_group == "content_navigation"
    assert scenario.config.click_mapping.clicks_per_session == 62
    assert scenario.adapter.model_mode.value == "model_b_previous_marks_proxy"
    assert scenario.feature_directory == ROOT / "data/processed/proxy"
    assert len(scenario.student_ids) == 30


def test_model_and_feature_artifacts_match_and_preserve_missing_previous_scores() -> (
    None
):
    scenario = load_coursework_scenario(ROOT)
    inputs = coursework.load_application_inputs(
        scenario.feature_directory,
        scenario.config,
        student_identity=scenario.student_ids[0],
        adapter=scenario.adapter,
    )

    assert inputs.student_identity == scenario.student_ids[0]
    assert inputs.model_identity == scenario.config.model_identity
    previous_score_context = next(
        context
        for context in inputs.feature_contexts
        if context.assignment_identity == "34899"
    )
    assert (
        previous_score_context.features["previous_assessment_score_lag_1"].isna().all()
    )
    assert (
        previous_score_context.features["previous_assessment_score_lag_1_available"]
        .eq(0)
        .all()
    )


def test_execution_records_every_profile_and_required_result_fields(
    tmp_path: Path, monkeypatch
) -> None:
    scenario = load_coursework_scenario(ROOT)
    monkeypatch.setattr(coursework, "load_coursework_scenario", lambda _: scenario)
    monkeypatch.setattr(
        coursework,
        "run_one_profile",
        lambda _scenario, student_id: {
            "student_id": student_id,
            "planner_status": "feasible",
            "planner_outcome": "target_achieved",
            "schedule_verified": True,
            "verification_status": "feasible",
            "allocations": {"34899": {"weekly_sessions": [1], "total_sessions": 1}},
            "timetable": {"34899": [{"slot_id": "slot-1"}]},
            "predicted_marks": {
                "34899": {"predicted_mark": 56, "target_mark": 55, "target_met": True}
            },
            "target_status": "met",
            "search_complete": True,
            "failure_reasons": [],
        },
    )

    run_directory = tmp_path / "run"
    payload = execute_coursework_profiles(ROOT, run_directory)
    saved = json.loads(
        (run_directory / "coursework_planner_30_profiles.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["summary"]["profiles_processed"] == 30
    assert saved["profiles"] == payload["profiles"]
    assert {
        "student_id",
        "planner_outcome",
        "allocations",
        "timetable",
        "predicted_marks",
        "target_status",
    } <= set(saved["profiles"][0])
    assert len(list((run_directory / "profiles").glob("*.json"))) == 30
    assert (run_directory / "coursework_planner_30_profiles.csv").is_file()
    first_profile = next((run_directory / "profiles").glob("*.json"))
    assert (
        json.loads(first_profile.read_text(encoding="utf-8"))["student_id"]
        in scenario.student_ids
    )


def _completed_record(student_id: str) -> dict[str, object]:
    return {
        "student_id": student_id,
        "planner_status": "feasible",
        "planner_outcome": "target_achieved",
        "schedule_verified": True,
        "verification_status": "feasible",
        "allocations": {"34899": {"weekly_sessions": [1], "total_sessions": 1}},
        "timetable": {"34899": [{"slot_id": "slot-1"}]},
        "predicted_marks": {
            "34899": {"predicted_mark": 56, "target_mark": 55, "target_met": True}
        },
        "target_status": "met",
        "search_complete": False,
        "failure_reasons": [],
        "execution_metrics": {"profile_wall_seconds": 1.25},
    }


def test_resume_skips_valid_completed_profiles(tmp_path: Path, monkeypatch) -> None:
    scenario = load_coursework_scenario(ROOT)
    monkeypatch.setattr(coursework, "load_coursework_scenario", lambda _: scenario)
    calls: list[str] = []

    def run(_scenario, student_id):
        calls.append(student_id)
        return _completed_record(student_id)

    monkeypatch.setattr(coursework, "run_one_profile", run)
    run_directory = tmp_path / "resume"
    execute_coursework_profiles(ROOT, run_directory)
    assert calls == list(scenario.student_ids)

    calls.clear()
    payload = execute_coursework_profiles(ROOT, run_directory)
    assert calls == []
    assert [item["student_id"] for item in payload["profiles"]] == list(
        scenario.student_ids
    )
    first_checkpoint = next(
        (run_directory / "profiles").glob(f"{scenario.student_ids[0]}-*.json")
    )
    resumed = coursework._valid_profile_record(
        first_checkpoint, scenario.student_ids[0]
    )
    assert resumed is not None
    assert resumed["search_complete"] is False


def test_resume_recovers_after_interrupted_run(tmp_path: Path, monkeypatch) -> None:
    scenario = load_coursework_scenario(ROOT)
    monkeypatch.setattr(coursework, "load_coursework_scenario", lambda _: scenario)
    calls: list[str] = []

    def interrupted(_scenario, student_id):
        calls.append(student_id)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return _completed_record(student_id)

    monkeypatch.setattr(coursework, "run_one_profile", interrupted)
    run_directory = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        execute_coursework_profiles(ROOT, run_directory)
    assert list((run_directory / "profiles").glob("*.json"))

    calls.clear()
    monkeypatch.setattr(
        coursework,
        "run_one_profile",
        lambda _scenario, student_id: (
            calls.append(student_id) or _completed_record(student_id)
        ),
    )
    payload = execute_coursework_profiles(ROOT, run_directory)
    assert calls == list(scenario.student_ids[1:])
    assert payload["summary"]["profiles_processed"] == 30


def test_corrupt_and_incomplete_checkpoints_are_rerun_not_skipped(
    tmp_path: Path, monkeypatch
) -> None:
    scenario = load_coursework_scenario(ROOT)
    monkeypatch.setattr(coursework, "load_coursework_scenario", lambda _: scenario)
    run_directory = tmp_path / "corrupt"
    profile_directory = run_directory / "profiles"
    profile_directory.mkdir(parents=True)
    first, second = scenario.student_ids[:2]
    (profile_directory / f"{first}-broken.json").write_text("{bad", encoding="utf-8")
    incomplete = _completed_record(second)
    incomplete["planner_outcome"] = "incomplete"
    (profile_directory / f"{second}-incomplete.json").write_text(
        json.dumps(incomplete), encoding="utf-8"
    )
    calls: list[str] = []
    monkeypatch.setattr(
        coursework,
        "run_one_profile",
        lambda _scenario, student_id: (
            calls.append(student_id) or _completed_record(student_id)
        ),
    )

    payload = execute_coursework_profiles(ROOT, run_directory)
    assert calls[:2] == [first, second]
    assert len(payload["profiles"]) == 30
    assert (profile_directory / f"{first}-broken.json").read_text(
        encoding="utf-8"
    ) == "{bad"


def test_unverified_target_achieved_checkpoint_is_rejected_on_resume(
    tmp_path: Path, monkeypatch
) -> None:
    scenario = load_coursework_scenario(ROOT)
    monkeypatch.setattr(coursework, "load_coursework_scenario", lambda _: scenario)
    run_directory = tmp_path / "unverified"
    profile_directory = run_directory / "profiles"
    profile_directory.mkdir(parents=True)
    student_id = scenario.student_ids[0]
    unverified = _completed_record(student_id)
    unverified["schedule_verified"] = False
    (profile_directory / f"{student_id}-bad.json").write_text(
        json.dumps(unverified), encoding="utf-8"
    )
    calls: list[str] = []
    monkeypatch.setattr(
        coursework,
        "run_one_profile",
        lambda _scenario, profile_id: (
            calls.append(profile_id) or _completed_record(profile_id)
        ),
    )

    execute_coursework_profiles(ROOT, run_directory)
    assert calls[0] == student_id
    assert (
        coursework._valid_profile_record(
            profile_directory / f"{student_id}-bad.json", student_id
        )
        is None
    )


@pytest.mark.parametrize(
    ("outcome", "search_complete", "eligible"),
    [
        ("target_unreachable", True, True),
        ("target_unreachable", False, False),
        ("no_feasible_schedule", True, True),
        ("incomplete", True, False),
    ],
)
def test_non_target_checkpoint_outcomes_require_completed_search(
    outcome: str, search_complete: bool, eligible: bool
) -> None:
    record = _completed_record("student")
    record["planner_outcome"] = outcome
    record["search_complete"] = search_complete
    assert coursework._is_checkpoint_eligible(record, "student") is eligible


def test_combined_outputs_preserve_all_profiles_and_execution_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    scenario = load_coursework_scenario(ROOT)
    monkeypatch.setattr(coursework, "load_coursework_scenario", lambda _: scenario)
    monkeypatch.setattr(
        coursework,
        "run_one_profile",
        lambda _scenario, student_id: _completed_record(student_id),
    )
    run_directory = tmp_path / "combined"
    payload = execute_coursework_profiles(ROOT, run_directory)
    saved = json.loads(
        (run_directory / "coursework_planner_30_profiles.json").read_text(
            encoding="utf-8"
        )
    )
    with (run_directory / "coursework_planner_30_profiles.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))

    assert [item["student_id"] for item in saved["profiles"]] == list(
        scenario.student_ids
    )
    assert payload["summary"]["profiles_processed"] == len(rows) == 30
    assert saved["profiles"][0]["execution_metrics"]["profile_wall_seconds"] >= 0
    assert (
        rows[0]["allocations"] and rows[0]["timetable"] and rows[0]["predicted_marks"]
    )


def test_failed_and_unreachable_results_remain_distinguishable() -> None:
    unreachable = SimpleNamespace(
        status=PlannerStatus.INFEASIBLE,
        outcome=PlannerOutcome.TARGET_UNREACHABLE,
        whole_plan_verification=None,
        weekly_timetable=None,
        allocations=(),
        predicted_marks=(),
        search_complete=True,
        failure_reasons=("target unreachable",),
    )
    record = planner_result_record("40878", unreachable)
    error_record = coursework._error_record("42552", RuntimeError("missing features"))

    assert record["planner_outcome"] == "target_unreachable"
    assert record["schedule_verified"] is False
    assert record["verification_status"] is None
    assert error_record["planner_outcome"] == "execution_error"
    assert error_record["schedule_verified"] is False
