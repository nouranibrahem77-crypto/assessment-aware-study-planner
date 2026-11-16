"""Focused coverage for the documented 30-profile coursework runner."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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

    payload = execute_coursework_profiles(
        ROOT, tmp_path / "results.json", tmp_path / "results.csv"
    )
    saved = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))

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
    assert (tmp_path / "results.csv").is_file()


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
