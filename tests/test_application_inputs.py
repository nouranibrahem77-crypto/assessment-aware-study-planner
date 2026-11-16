"""Tests for authoritative and provenance-bound OULAD planning inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from study_planner.csp import (
    AssignmentEntry,
    ConstraintCheckStatus,
    CspAssignment,
    OrderingPolicySnapshot,
    SameAssignmentOrderingConstraint,
    SessionPair,
    SessionVariable,
    WeeklySlot,
    WholePlanVerificationStatus,
    verify_whole_plan,
)
from study_planner.integration import (
    AllocationBounds,
    ApplicationAssignmentRule,
    ApplicationInputConfig,
    ApplicationInputError,
    ApplicationSlot,
    AssignmentTarget,
    CatalogChronologicalOrderingPolicy,
    ModelArtifactIdentity,
    OuladPresentationSplit,
    OutOfSupportAction,
    PlannerStatus,
    SupportPolicy,
    SyntheticWeeklySlotPattern,
    TargetRule,
    WeeklyTimeWindow,
    build_application_inputs,
    build_application_planning_problem,
    build_synthetic_weekly_slot_catalog,
    integrated_uniform_cost_search,
    load_application_inputs,
    load_session_nn_adapter,
)
from study_planner.integration.application_inputs import _slot_click_window
from study_planner.nn.inference import PreviousMarksRegressionPredictor
from study_planner.nn.models import MLPRegressor
from study_planner.search.nn_adapter import (
    CandidateSessionAllocation,
    EvaluationStatus,
    NNModelMode,
)
from study_planner.search.session_mapping import (
    STRICT_FFF_FEATURE_NAMES,
    ClickScenario,
    JointSupportStatus,
    MarginalSupportStatus,
    SessionClickMappingConfig,
)

MODEL_B_FEATURE_NAMES = (
    *STRICT_FFF_FEATURE_NAMES[:7],
    *tuple(
        name
        for lag in range(1, 6)
        for name in (
            f"previous_assessment_score_lag_{lag}",
            f"previous_assessment_score_lag_{lag}_available",
        )
    ),
    *STRICT_FFF_FEATURE_NAMES[7:],
)
ASSIGNMENT_IDS = ("a1", "a2", "a3")
ASSESSMENT_IDS = ("34860", "34861", "34862")
DEADLINES = (19, 47, 89)
TARGETS = (70.0, 75.0, 80.0)


def _write_artifact(directory, mode: str) -> None:
    directory.mkdir()
    feature_names = (
        MODEL_B_FEATURE_NAMES if mode == "proxy" else STRICT_FFF_FEATURE_NAMES
    )
    (directory / "feature_schema.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "source_feature_columns": list(feature_names),
                "assessment_type_indicator_columns": list(STRICT_FFF_FEATURE_NAMES[:3]),
            }
        ),
        encoding="utf-8",
    )
    (directory / "experiment_config.json").write_text(
        json.dumps(
            {
                "feature_mode": mode,
                "selected_mlp_config": {"hidden_sizes": [2], "dropout": 0.0},
                "preprocessed_feature_count": len(feature_names),
                "model_files": ["mlp_seed_1.pt"],
            }
        ),
        encoding="utf-8",
    )
    type_columns = tuple(STRICT_FFF_FEATURE_NAMES[:3])
    numeric_names = tuple(name for name in feature_names if name not in type_columns)
    training_frame = pd.DataFrame(
        {
            "assessment_type": ["CMA", "EXAM", "TMA"],
            **{
                name: np.zeros(3, dtype=float)
                for name in numeric_names
                if not name.startswith("previous_assessment_score_lag_")
                or name.endswith("_available")
            },
            **{
                name: np.asarray([np.nan, 12.0, 18.0])
                for name in numeric_names
                if name.startswith("previous_assessment_score_lag_")
                and not name.endswith("_available")
            },
        }
    )
    preprocessor = ColumnTransformer(
        [
            (
                "assessment_type",
                OneHotEncoder(handle_unknown="ignore"),
                ["assessment_type"],
            ),
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median", keep_empty_features=True),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric_names),
            ),
        ]
    ).fit(training_frame)
    joblib.dump(preprocessor, directory / "preprocessor.joblib")
    model = MLPRegressor(len(feature_names), (2,), 0.0)
    torch.save(model.state_dict(), directory / "mlp_seed_1.pt")


def _feature_rows(mode: str = "strict") -> pd.DataFrame:
    feature_names = (
        MODEL_B_FEATURE_NAMES if mode == "proxy" else STRICT_FFF_FEATURE_NAMES
    )
    rows = []
    for assignment_index, (assessment_id, deadline) in enumerate(
        zip(ASSESSMENT_IDS, DEADLINES, strict=True)
    ):
        row: dict[str, object] = dict.fromkeys(feature_names, 0.0)
        row["assessment_type_tma"] = 1.0
        row["assessment_weight"] = 10.0
        row["assessment_weight_available"] = 1.0
        row["assessment_deadline_day"] = deadline
        row["weeks_since_course_start"] = deadline / 7.0
        if mode == "proxy":
            for lag in range(1, 6):
                row[f"previous_assessment_score_lag_{lag}"] = np.nan
                row[f"previous_assessment_score_lag_{lag}_available"] = 0.0
        row.update(
            {
                "code_module": "FFF",
                "code_presentation": "2014J",
                "id_student": "student-1",
                "id_assessment": assessment_id,
                "target_score": 42.0 + assignment_index,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _write_processed_features(directory, mode: str) -> None:
    directory.mkdir()
    feature_names = (
        MODEL_B_FEATURE_NAMES if mode == "proxy" else STRICT_FFF_FEATURE_NAMES
    )
    test_rows = _feature_rows(mode)
    split_rows = {
        "train": pd.concat(
            [
                test_rows.assign(code_presentation="2013B"),
                test_rows.assign(code_presentation="2013J"),
            ],
            ignore_index=True,
        ),
        "validation": test_rows.assign(code_presentation="2014B"),
        "test": test_rows,
    }
    filenames = {}
    for split_name, rows in split_rows.items():
        filename = f"{split_name}.csv"
        rows.loc[
            :,
            [
                "code_module",
                "code_presentation",
                "id_student",
                "id_assessment",
                *feature_names,
                "target_score",
            ],
        ].to_csv(directory / filename, index=False)
        filenames[split_name] = filename
    manifest = {
        "course_module": "FFF",
        "score_feature_mode": mode,
        "target": "target_score",
        "feature_names": list(feature_names),
        "split_files": filenames,
        "split_presentations": {
            "train": ["2013B", "2013J"],
            "validation": ["2014B"],
            "test": ["2014J"],
        },
    }
    (directory / "fff_preprocessing.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _config(adapter, *, mode: str = "strict") -> ApplicationInputConfig:
    catalog, eligible_by_assignment = build_synthetic_weekly_slot_catalog(
        identity="test-catalog",
        version="v1",
        weekly_capacity=2,
        deadline_policy_identity="approved-deadline-input",
        deadline_policy_version="v1",
        assignment_deadlines=tuple(
            zip(ASSIGNMENT_IDS, DEADLINES, strict=True)
        ),
        week_indices=tuple(range(13)),
        pattern=SyntheticWeeklySlotPattern(
            (1,), 9 * 60, 10 * 60, "approved-local-timezone"
        ),
    )
    eligible = dict(eligible_by_assignment)
    rules = tuple(
        ApplicationAssignmentRule(
            assignment_identity=assignment_id,
            assessment_id=assessment_id,
            target_mark=target,
            deadline_day=deadline,
            eligible_slot_ids=eligible[assignment_id],
            allocation_bounds=AllocationBounds(0, 3),
        )
        for index, (assignment_id, assessment_id, target, deadline) in enumerate(
            zip(ASSIGNMENT_IDS, ASSESSMENT_IDS, TARGETS, DEADLINES, strict=True)
        )
    )
    mapping = SessionClickMappingConfig(ClickScenario.CENTRAL)
    return ApplicationInputConfig(
        presentation_split=OuladPresentationSplit("FFF", ("2013B", "2013J"), "2014J"),
        assignment_rules=rules,
        target_rule=TargetRule(
            "explicit-targets", "v1", tuple(zip(ASSIGNMENT_IDS, TARGETS, strict=True))
        ),
        slot_catalog=catalog,
        support_policy=SupportPolicy(
            "marginal-training-range",
            "v1",
            "strict-train-min-max-v1",
            OutOfSupportAction.REPORT_ONLY,
        ),
        click_mapping=mapping,
        model_identity=ModelArtifactIdentity.from_adapter(adapter),
    )


def _adapter(tmp_path, mode: str = "strict"):
    directory = tmp_path / mode
    _write_artifact(directory, mode)
    nn_mode = (
        NNModelMode.MODEL_B_PREVIOUS_MARKS
        if mode == "proxy"
        else NNModelMode.MODEL_A_STRICT
    )
    return load_session_nn_adapter(
        directory, nn_mode, SessionClickMappingConfig(ClickScenario.CENTRAL)
    )


@pytest.mark.parametrize("mode", ["strict", "proxy"])
def test_builds_three_assignment_inputs_with_exact_model_schema(tmp_path, mode: str):
    adapter = _adapter(tmp_path, mode)
    config = _config(adapter, mode=mode)
    inputs = build_application_inputs(
        config,
        student_identity="student-1",
        feature_rows=_feature_rows(mode),
        adapter=adapter,
    )

    assert inputs.assignment_ids == ASSIGNMENT_IDS
    assert all(
        tuple(context.features.columns) == tuple(adapter.predictor.feature_names)
        for context in inputs.feature_contexts
    )
    assert all(
        "target_score" not in context.features for context in inputs.feature_contexts
    )
    assert inputs.config.targets == tuple(
        AssignmentTarget(assignment_id, target)
        for assignment_id, target in zip(ASSIGNMENT_IDS, TARGETS, strict=True)
    )
    assert inputs.model_identity == ModelArtifactIdentity.from_adapter(adapter)
    assert inputs.model_identity.mode is (
        NNModelMode.MODEL_B_PREVIOUS_MARKS
        if mode == "proxy"
        else NNModelMode.MODEL_A_STRICT
    )
    a1_mappings = dict(config.slot_catalog.windows_by_assignment)["a1"]
    assert a1_mappings
    assert set(slot for slot, _ in a1_mappings) == set(
        config.assignment_rules[0].eligible_slot_ids
    )


def test_accepts_weeks_feature_rounded_for_csv_serialization(tmp_path):
    adapter = _adapter(tmp_path)
    rows = _feature_rows()
    rows["weeks_since_course_start"] = rows["weeks_since_course_start"].round(11)

    inputs = build_application_inputs(
        _config(adapter),
        student_identity="student-1",
        feature_rows=rows,
        adapter=adapter,
    )

    assert inputs.assignment_ids == ASSIGNMENT_IDS


def test_rejects_wrong_split_duplicate_source_and_mismatched_deadline(tmp_path):
    adapter = _adapter(tmp_path)
    config = _config(adapter)
    rows = _feature_rows()
    wrong_presentation = rows.copy()
    wrong_presentation["code_presentation"] = "2013B"
    with pytest.raises(ApplicationInputError, match="found 0"):
        build_application_inputs(
            config,
            student_identity="student-1",
            feature_rows=wrong_presentation,
            adapter=adapter,
        )

    duplicated = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    with pytest.raises(ApplicationInputError, match="found 2"):
        build_application_inputs(
            config,
            student_identity="student-1",
            feature_rows=duplicated,
            adapter=adapter,
        )

    bad_deadline = rows.copy()
    bad_deadline.loc[0, "assessment_deadline_day"] = 20
    bad_deadline.loc[0, "weeks_since_course_start"] = 20 / 7
    with pytest.raises(ApplicationInputError, match="deadline does not match"):
        build_application_inputs(
            config,
            student_identity="student-1",
            feature_rows=bad_deadline,
            adapter=adapter,
        )


def test_rejects_bad_course_target_slots_support_and_model_identity(tmp_path):
    adapter = _adapter(tmp_path)
    config = _config(adapter)
    wrong_course = _feature_rows()
    wrong_course["code_module"] = "ABC"
    with pytest.raises(ApplicationInputError, match="found 0"):
        build_application_inputs(
            config,
            student_identity="student-1",
            feature_rows=wrong_course,
            adapter=adapter,
        )

    other_adapter = _adapter(tmp_path, "proxy")
    with pytest.raises(ApplicationInputError, match="model identity"):
        build_application_inputs(
            config,
            student_identity="student-1",
            feature_rows=_feature_rows(),
            adapter=other_adapter,
        )

    with pytest.raises(ApplicationInputError, match="unknown slots"):
        ApplicationInputConfig(
            config.presentation_split,
            (
                ApplicationAssignmentRule(
                    "a1", "34860", 70, 19, ("missing",), AllocationBounds(0, 3)
                ),
                *config.assignment_rules[1:],
            ),
            config.target_rule,
            config.slot_catalog,
            config.support_policy,
            config.click_mapping,
            config.model_identity,
        )

    with pytest.raises(ApplicationInputError, match="joint support is unassessed"):
        SupportPolicy(
            "claimed-joint-support",
            "v1",
            "unknown-joint-reference",
            OutOfSupportAction.REPORT_ONLY,
            JointSupportStatus.SUPPORTED,
        )


def test_validates_allocation_bounds_and_training_presentation_order(tmp_path):
    with pytest.raises(ApplicationInputError, match="minimum sessions"):
        AllocationBounds(4, 2)
    with pytest.raises(ApplicationInputError, match="training presentations"):
        OuladPresentationSplit("FFF", ("2014B",), "2014J")
    with pytest.raises(ApplicationInputError, match="later OULAD test split"):
        OuladPresentationSplit("FFF", ("2013B",), "2013J")
    adapter = _adapter(tmp_path)
    inputs = build_application_inputs(
        _config(adapter),
        student_identity="student-1",
        feature_rows=_feature_rows(),
        adapter=adapter,
    )
    assert all(
        rule.allocation_bounds.accepts(0) for rule in inputs.config.assignment_rules
    )
    assert not inputs.config.assignment_rules[0].allocation_bounds.accepts(4)


def test_validates_explicit_weekly_time_windows():
    with pytest.raises(ApplicationInputError, match="weekday"):
        WeeklyTimeWindow(0, 9 * 60, 10 * 60, "local")
    with pytest.raises(ApplicationInputError, match="positive duration"):
        WeeklyTimeWindow(1, 10 * 60, 10 * 60, "local")
    with pytest.raises(ApplicationInputError, match="timezone identity"):
        WeeklyTimeWindow(1, 9 * 60, 10 * 60, "")
    with pytest.raises(ApplicationInputError, match="positive duration"):
        SyntheticWeeklySlotPattern((1, 3), 10 * 60, 10 * 60, "UTC")


def _synthetic_catalog(
    *, deadlines=None
):
    if deadlines is None:
        deadlines = tuple(zip(ASSIGNMENT_IDS, (24, 52, 94), strict=True))
    return build_synthetic_weekly_slot_catalog(
        identity="TEST-weekly-slots",
        version="test-v1",
        weekly_capacity=3,
        deadline_policy_identity="TEST-course-day-cutoff",
        deadline_policy_version="test-v1",
        assignment_deadlines=deadlines,
        week_indices=tuple(range(14)),
        pattern=SyntheticWeeklySlotPattern(
            (1, 3, 5), 18 * 60, 19 * 60, "TEST/UTC", 1
        ),
    )


def test_synthetic_weekly_expansion_has_stable_unique_slot_ids():
    first, eligible = _synthetic_catalog()
    repeated, _ = _synthetic_catalog()

    assert len(first.slots) == 42
    assert first.slots == repeated.slots
    assert len({slot.slot.slot_id for slot in first.slots}) == 42
    assert {slot.slot.week_index for slot in first.slots} == set(range(14))
    assert first.slots[0].slot.slot_id == "TEST-weekly-slots-W00-D1-T1080-1140"
    mappings = dict(first.windows_by_assignment)
    for assignment_id, slot_ids in eligible:
        assignment_map = mappings[assignment_id]
        assert {slot_id for slot_id, _ in assignment_map} <= set(slot_ids)
        assert {window for _, window in assignment_map} <= {1, 2, 3, 4}


def test_synthetic_deadline_cutoff_is_strict_and_slots_are_eligible_per_assignment():
    pattern = SyntheticWeeklySlotPattern((1,), 23 * 60, 24 * 60, "TEST/UTC", 1)
    catalog, eligible = build_synthetic_weekly_slot_catalog(
        identity="TEST-boundary",
        version="v1",
        weekly_capacity=3,
        deadline_policy_identity="TEST-deadline",
        deadline_policy_version="v1",
        assignment_deadlines=(("a1", 1), ("a2", 7), ("a3", 15)),
        week_indices=(0, 1, 2),
        pattern=pattern,
    )
    slot_id = catalog.slots[0].slot.slot_id
    assert slot_id not in dict(eligible)["a1"]  # ends exactly at deadline day
    assert slot_id in dict(eligible)["a2"]
    assert dict(catalog.slots[0].relative_windows)["a2"] == 4
    assert "TEST-boundary-W02-D1-T1380-1440" not in dict(eligible)["a3"]

    _, real_deadline_eligibility = _synthetic_catalog()
    eligibility = dict(real_deadline_eligibility)
    assert eligibility["a1"] != eligibility["a2"]
    assert eligibility["a2"] != eligibility["a3"]


def test_wednesday_course_start_controls_week_order_and_deadline_windows():
    catalog, eligible_by_assignment = build_synthetic_weekly_slot_catalog(
        identity="TEST-wednesday-start",
        version="synthetic-v1",
        weekly_capacity=3,
        deadline_policy_identity="TEST-course-day-start",
        deadline_policy_version="v1",
        assignment_deadlines=(("a1", 28), ("a2", 28), ("a3", 28)),
        week_indices=(0, 1, 2, 3),
        pattern=SyntheticWeeklySlotPattern(
            (2, 3, 4), 9 * 60, 10 * 60, "TEST/local-clock", 3
        ),
    )

    slots_by_id = {item.slot.slot_id: item.slot for item in catalog.slots}
    assert {slot.week_index for slot in slots_by_id.values()} == {0, 1, 2, 3}
    week_zero_wednesday = slots_by_id["TEST-wednesday-start-W00-D3-T0540-0600"]
    week_zero_thursday = slots_by_id["TEST-wednesday-start-W00-D4-T0540-0600"]
    week_zero_tuesday = slots_by_id["TEST-wednesday-start-W00-D2-T0540-0600"]
    week_one_wednesday = slots_by_id["TEST-wednesday-start-W01-D3-T0540-0600"]
    ordering = CatalogChronologicalOrderingPolicy(catalog)

    assert ordering.is_ordered(week_zero_wednesday, week_zero_thursday)
    assert ordering.is_ordered(week_zero_thursday, week_zero_tuesday)
    assert ordering.is_ordered(week_zero_tuesday, week_one_wednesday)
    assert not ordering.is_ordered(week_one_wednesday, week_zero_tuesday)

    mappings = dict(dict(catalog.windows_by_assignment)["a1"])
    assert mappings["TEST-wednesday-start-W00-D3-T0540-0600"] == 1
    assert mappings["TEST-wednesday-start-W01-D3-T0540-0600"] == 2
    assert mappings["TEST-wednesday-start-W02-D3-T0540-0600"] == 3
    assert mappings["TEST-wednesday-start-W03-D3-T0540-0600"] == 4
    assert set(dict(eligible_by_assignment)["a1"]) == set(mappings)


def test_whole_slot_crossing_deadline_is_excluded_and_model_rejects_overnight_window():
    # The catalog model only permits same-day windows. Exercise the cutoff
    # helper with a structural crossing interval, then verify such an interval
    # cannot be admitted as an application catalog slot.
    crossing_slot = SimpleNamespace(
        slot=WeeklySlot("TEST-crossing-deadline", 0),
        time_window=SimpleNamespace(
            weekday_iso=3,
            start_minute_local=23 * 60,
            end_minute_local=25 * 60,
        ),
    )
    assert (
        _slot_click_window(
            crossing_slot, deadline_day=1, course_start_weekday_iso=3
        )
        is None
    )
    with pytest.raises(
        ApplicationInputError, match="end minute must be in \\[0, 1440\\]"
    ):
        WeeklyTimeWindow(3, 23 * 60, 25 * 60, "TEST/local-clock")


def test_slot_ending_exactly_at_deadline_is_strictly_excluded():
    catalog, eligible = build_synthetic_weekly_slot_catalog(
        identity="TEST-exact-deadline",
        version="synthetic-v1",
        weekly_capacity=1,
        deadline_policy_identity="TEST-course-day-start",
        deadline_policy_version="v1",
        assignment_deadlines=(("a1", 1), ("a2", 1), ("a3", 1)),
        week_indices=(0,),
        pattern=SyntheticWeeklySlotPattern(
            (3,), 23 * 60, 24 * 60, "TEST/local-clock", 3
        ),
    )

    slot_id = catalog.slots[0].slot.slot_id
    assert slot_id not in dict(eligible)["a1"]
    # NN window assignment is independent: this interval is in window 4 even
    # though its end equals the conservative deadline boundary.
    assert dict(catalog.slots[0].relative_windows)["a1"] == 4


def test_deadline_eligibility_is_independent_of_four_nn_click_windows():
    catalog, eligible = build_synthetic_weekly_slot_catalog(
        identity="TEST-deadline-vs-click-history",
        version="synthetic-v1",
        weekly_capacity=3,
        deadline_policy_identity="TEST-course-day-start",
        deadline_policy_version="v1",
        assignment_deadlines=(("a1", 42), ("a2", 42), ("a3", 42)),
        week_indices=(0, 5, 6, 7),
        pattern=SyntheticWeeklySlotPattern(
            (1,), 9 * 60, 10 * 60, "TEST/local-clock", 1
        ),
    )

    eligible_ids = set(dict(eligible)["a1"])
    mappings = dict(dict(catalog.windows_by_assignment)["a1"])
    old_slot = "TEST-deadline-vs-click-history-W00-D1-T0540-0600"
    recent_slot = "TEST-deadline-vs-click-history-W05-D1-T0540-0600"
    deadline_day_slot = "TEST-deadline-vs-click-history-W06-D1-T0540-0600"
    after_deadline_slot = "TEST-deadline-vs-click-history-W07-D1-T0540-0600"

    assert old_slot in eligible_ids
    assert old_slot not in mappings
    assert recent_slot in eligible_ids
    assert mappings[recent_slot] == 4
    assert deadline_day_slot not in eligible_ids
    assert after_deadline_slot not in eligible_ids


def test_application_config_accepts_deadline_eligible_slots_without_nn_windows(
    tmp_path,
):
    config = _config(_adapter(tmp_path))
    rule = next(
        item for item in config.assignment_rules if item.assignment_identity == "a2"
    )
    mapped = {
        slot_id
        for slot_id, _ in dict(config.slot_catalog.windows_by_assignment)["a2"]
    }

    assert set(rule.eligible_slot_ids) - mapped


def test_catalog_rejects_mixed_timezone_identities_without_conversion():
    catalog, _ = _synthetic_catalog()
    first, second, *rest = catalog.slots
    changed_timezone = replace(
        second,
        time_window=replace(
            second.time_window, timezone_identity="TEST/another-local-clock"
        ),
    )

    with pytest.raises(ApplicationInputError, match="same timezone identity"):
        replace(catalog, slots=(first, changed_timezone, *rest))


def test_assignment_specific_mappings_reject_missing_and_extra_slots(tmp_path):
    adapter = _adapter(tmp_path)
    config = _config(adapter)
    a1_rule = config.assignment_rules[0]
    missing_slot_id = a1_rule.eligible_slot_ids[0]
    slots_missing_mapping = tuple(
        replace(
            slot,
            relative_windows=tuple(
                (assignment_id, window)
                for assignment_id, window in slot.relative_windows
                if not (slot.slot.slot_id == missing_slot_id and assignment_id == "a1")
            ),
        )
        for slot in config.slot_catalog.slots
    )
    with pytest.raises(ApplicationInputError, match="incorrect click window"):
        replace(
            config,
            slot_catalog=replace(config.slot_catalog, slots=slots_missing_mapping),
        )

    extra_slot = next(
        slot
        for slot in config.slot_catalog.slots
        if slot.slot.slot_id not in a1_rule.eligible_slot_ids
        and slot.relative_windows
    )
    existing_window = extra_slot.relative_windows[0][1]
    slots_with_extra_mapping = tuple(
        replace(
            slot,
            relative_windows=(*slot.relative_windows, ("a1", existing_window)),
        )
        if slot.slot.slot_id == extra_slot.slot.slot_id
        else slot
        for slot in config.slot_catalog.slots
    )
    with pytest.raises(ApplicationInputError, match="incorrect click window"):
        replace(
            config,
            slot_catalog=replace(config.slot_catalog, slots=slots_with_extra_mapping),
        )


def test_catalog_boundary_identity_is_currently_restricted_to_synthetic_rule(
    tmp_path,
):
    """TEST-only check that records the current production-catalog limitation."""
    adapter = _adapter(tmp_path)
    config = _config(adapter)

    assert config.slot_catalog.deadline_boundary_identity == (
        "course-day-start-exclusive-v1"
    )
    with pytest.raises(ApplicationInputError, match="unsupported deadline boundary"):
        replace(
            config.slot_catalog,
            deadline_boundary_identity="TEST/alternate-boundary-identity",
        )


def test_catalog_rejects_duplicate_slot_ids_and_config_rejects_late_eligible_slot(
    tmp_path,
):
    adapter = _adapter(tmp_path)
    config = _config(adapter)
    duplicate = config.slot_catalog.slots[0]
    with pytest.raises(ApplicationInputError, match="IDs must be unique"):
        replace(config.slot_catalog, slots=(duplicate, duplicate))

    late = ApplicationSlot(
        WeeklySlot("late-slot", 20),
        WeeklyTimeWindow(1, 9 * 60, 10 * 60, "approved-local-timezone"),
        (("a1", 4),),
    )
    altered_catalog = replace(
        config.slot_catalog,
        slots=config.slot_catalog.slots + (late,),
    )
    altered_rules = tuple(
        replace(rule, eligible_slot_ids=rule.eligible_slot_ids + ("late-slot",))
        if rule.assignment_identity == "a1"
        else rule
        for rule in config.assignment_rules
    )
    with pytest.raises(ApplicationInputError, match="strictly before"):
        replace(config, slot_catalog=altered_catalog, assignment_rules=altered_rules)

    assignment_id = "a1"
    eligible_id = config.assignment_rules[0].eligible_slot_ids[0]
    altered_slots = tuple(
        replace(
            slot,
            relative_windows=tuple(
                (
                    identity,
                    4 if window != 4 else 3,
                )
                if identity == assignment_id
                else (identity, window)
                for identity, window in slot.relative_windows
            ),
        )
        if slot.slot.slot_id == eligible_id
        else slot
        for slot in config.slot_catalog.slots
    )
    altered_mapping_catalog = replace(config.slot_catalog, slots=altered_slots)
    with pytest.raises(ApplicationInputError, match="incorrect click window"):
        replace(config, slot_catalog=altered_mapping_catalog)


def test_catalog_ordering_policy_uses_weekday_and_clock_order():
    catalog, _ = _synthetic_catalog()
    policy = CatalogChronologicalOrderingPolicy(catalog)
    earlier = catalog.slots[0].slot
    later_same_week = catalog.slots[1].slot
    later_week = catalog.slots[3].slot

    pairs = policy.required_pairs(
        (SessionVariable("a1", 1), SessionVariable("a1", 0), SessionVariable("a2", 0))
    )
    assert len(pairs) == 1
    assert pairs[0].earlier.session_index == 0
    assert pairs[0].later.session_index == 1
    assert policy.is_ordered(earlier, later_same_week)
    assert policy.is_ordered(later_same_week, later_week)
    assert not policy.is_ordered(later_same_week, earlier)

    same_day_slots = (
        ApplicationSlot(
            WeeklySlot("clock-early", 0),
            WeeklyTimeWindow(1, 9 * 60, 10 * 60, "TEST/UTC"),
            (("a1", 1),),
        ),
        ApplicationSlot(
            WeeklySlot("clock-late", 0),
            WeeklyTimeWindow(1, 11 * 60, 12 * 60, "TEST/UTC"),
            (("a1", 1),),
        ),
    )
    clock_catalog = replace(catalog, slots=same_day_slots)
    clock_policy = CatalogChronologicalOrderingPolicy(clock_catalog)
    assert clock_policy.is_ordered(same_day_slots[0].slot, same_day_slots[1].slot)
    assert not clock_policy.is_ordered(same_day_slots[1].slot, same_day_slots[0].slot)


def test_catalog_chronological_ordering_integrates_with_csp_and_verification(
    tmp_path,
):
    """TEST/SYNTHETIC: exercise real catalog ordering end to end."""
    adapter = _adapter(tmp_path, mode="proxy")
    config = _config(adapter, mode="proxy")
    catalog, eligible_by_assignment = build_synthetic_weekly_slot_catalog(
        identity="TEST-ordering-catalog",
        version="synthetic-v1",
        weekly_capacity=config.slot_catalog.weekly_capacity,
        deadline_policy_identity="TEST-course-day-boundary",
        deadline_policy_version="synthetic-v1",
        assignment_deadlines=tuple(
            zip(ASSIGNMENT_IDS, DEADLINES, strict=True)
        ),
        week_indices=tuple(range(13)),
        pattern=SyntheticWeeklySlotPattern(
            (1, 3, 5), 9 * 60, 10 * 60, "TEST/UTC", 1
        ),
    )
    eligible = dict(eligible_by_assignment)
    assignment_rules = tuple(
        replace(
            rule,
            eligible_slot_ids=eligible[rule.assignment_identity],
            allocation_bounds=(
                AllocationBounds(2, 2)
                if rule.assignment_identity == "a1"
                else AllocationBounds(0, 0)
            ),
        )
        for rule in config.assignment_rules
    )
    config = replace(config, slot_catalog=catalog, assignment_rules=assignment_rules)
    inputs = build_application_inputs(
        config,
        student_identity="student-1",
        feature_rows=_feature_rows("proxy"),
        adapter=adapter,
    )
    ordering_policy = CatalogChronologicalOrderingPolicy(catalog)
    binding = build_application_planning_problem(
        inputs, adapter=adapter, ordering_policy=ordering_policy
    )

    state = binding.allocation_space.initial_state
    eligible_windows = dict(catalog.windows_by_assignment)["a1"]
    window_counts: dict[int, int] = {}
    for _, window in eligible_windows:
        window_counts[window] = window_counts.get(window, 0) + 1
    allocation_window = next(
        window for window, count in sorted(window_counts.items()) if count >= 2
    )
    action = next(
        action
        for action in binding.allocation_space.actions(state)
        if action.assignment_identity == "a1"
        and action.relative_window == allocation_window
    )
    state = binding.allocation_space.transition(state, action)
    state = binding.allocation_space.transition(state, action)
    candidate = binding.candidate_factory(state)

    problem = candidate.problem
    assert isinstance(problem.ordering_policy, OrderingPolicySnapshot)
    earlier_variable = SessionVariable("a1", 0)
    later_variable = SessionVariable("a1", 1)
    expected_pair = SessionPair(earlier_variable, later_variable)
    assert problem.ordering_policy.required_session_pairs == (expected_pair,)
    ordering_constraints = tuple(
        constraint
        for constraint in problem.constraints
        if isinstance(constraint, SameAssignmentOrderingConstraint)
    )
    assert len(ordering_constraints) == 1
    ordering_constraint = ordering_constraints[0]
    assert ordering_constraint.scope == (earlier_variable, later_variable)

    mapped_slots = dict(eligible_windows)
    assignment_domain = next(
        domain.values
        for domain in problem.domains
        if domain.variable == earlier_variable
    )
    allocation_slots = tuple(
        slot
        for slot in assignment_domain
        if mapped_slots[slot.slot_id] == allocation_window
    )
    chronological_pair = next(
        (earlier, later)
        for earlier in allocation_slots
        for later in allocation_slots
        if ordering_policy.is_ordered(earlier, later)
    )
    early_slot, late_slot = chronological_pair
    valid_schedule = CspAssignment(
        (
            AssignmentEntry(earlier_variable, early_slot),
            AssignmentEntry(later_variable, late_slot),
        )
    )
    assert ordering_constraint.is_satisfied(valid_schedule)
    valid_result = verify_whole_plan(
        problem, valid_schedule, candidate.verification_policy
    )
    assert valid_result.status is WholePlanVerificationStatus.FEASIBLE

    reversed_schedule = CspAssignment(
        (
            AssignmentEntry(earlier_variable, late_slot),
            AssignmentEntry(later_variable, early_slot),
        )
    )
    assert not ordering_constraint.is_satisfied(reversed_schedule)
    invalid_result = verify_whole_plan(
        problem, reversed_schedule, candidate.verification_policy
    )
    failed_evidence = {
        item.constraint_id: item.status for item in invalid_result.constraint_checks
    }
    assert invalid_result.status is WholePlanVerificationStatus.INFEASIBLE
    assert failed_evidence[ordering_constraint.constraint_id] is (
        ConstraintCheckStatus.FAILED
    )
    assert (
        f"constraint_failed:{ordering_constraint.constraint_id}"
        in invalid_result.rejection_reasons
    )


def test_requires_three_distinct_assessment_inputs(tmp_path):
    adapter = _adapter(tmp_path, mode="strict")
    config = _config(adapter)
    duplicate_rule = ApplicationAssignmentRule(
        "a2",
        "34860",
        75,
        47,
        config.assignment_rules[1].eligible_slot_ids,
        AllocationBounds(0, 3),
    )
    with pytest.raises(ApplicationInputError, match="assessment IDs must be unique"):
        ApplicationInputConfig(
            config.presentation_split,
            (config.assignment_rules[0], duplicate_rule, config.assignment_rules[2]),
            config.target_rule,
            config.slot_catalog,
            config.support_policy,
            config.click_mapping,
            config.model_identity,
        )


def test_loads_model_b_only_through_its_proxy_loader(tmp_path):
    adapter = _adapter(tmp_path, "proxy")
    assert isinstance(adapter.predictor, PreviousMarksRegressionPredictor)
    assert adapter.model_mode is NNModelMode.MODEL_B_PREVIOUS_MARKS
    inputs = build_application_inputs(
        _config(adapter, mode="proxy"),
        student_identity="student-1",
        feature_rows=_feature_rows("proxy"),
        adapter=adapter,
    )
    assert all(
        name.startswith("previous_assessment_score_")
        for name in inputs.feature_contexts[0].features.columns
        if name.startswith("previous_assessment_score_")
    )


@pytest.mark.parametrize("mode", ["strict", "proxy"])
def test_explicit_test_configuration_runs_through_search_csp_and_nn(
    tmp_path, mode: str
):
    """TEST CONFIG ONLY: proves public application-input-to-planner wiring."""

    @dataclass(frozen=True)
    class TestOrdering:
        policy_identity: str = "test-ordering-v1"

        def required_pairs(self, sessions: tuple[SessionVariable, ...]):
            by_assignment: dict[str, list[SessionVariable]] = {}
            for session in sessions:
                by_assignment.setdefault(session.assignment_id, []).append(session)
            return tuple(
                SessionPair(earlier, later)
                for values in by_assignment.values()
                for earlier, later in zip(values, values[1:], strict=False)
            )

        def is_ordered(self, earlier: WeeklySlot, later: WeeklySlot) -> bool:
            return earlier.week_index < later.week_index

    adapter = _adapter(tmp_path, mode)
    config = _config(adapter, mode=mode)
    zero_targets = TargetRule(
        "TEST-zero-targets",
        "test-v1",
        tuple((identity, 0.0) for identity in ASSIGNMENT_IDS),
    )
    test_rules = tuple(
        replace(rule, target_mark=0.0) for rule in config.assignment_rules
    )
    test_config = replace(config, assignment_rules=test_rules, target_rule=zero_targets)
    inputs = build_application_inputs(
        test_config,
        student_identity="student-1",
        feature_rows=_feature_rows(mode),
        adapter=adapter,
    )
    binding = build_application_planning_problem(
        inputs, adapter=adapter, ordering_policy=TestOrdering()
    )
    assert binding.allocation_space.initial_state.allocations == tuple(
        (identity, (0, 0, 0, 0)) for identity in ASSIGNMENT_IDS
    )
    candidate = binding.candidate_factory(binding.allocation_space.initial_state)
    assert candidate.problem.variables == ()
    assert candidate.problem.weekly_capacity == config.slot_catalog.weekly_capacity
    assert candidate.targets == test_config.targets
    provisional = adapter.evaluate_provisional(
        CandidateSessionAllocation((0, 0, 0, 0)), inputs.feature_contexts[0]
    )
    assert provisional.status is EvaluationStatus.PREDICTED, provisional.error_message

    result = integrated_uniform_cost_search(
        binding.allocation_space,
        binding.candidate_factory,
        adapter,
    )
    assert result.status is PlannerStatus.FEASIBLE
    assert (
        tuple(item.assignment_identity for item in result.allocations) == ASSIGNMENT_IDS
    )
    assert all(not any(item.weekly_sessions) for item in result.allocations)
    assert len(result.weekly_timetable) == 3
    assert all(not entry.sessions for entry in result.weekly_timetable)
    assert result.whole_plan_verification is not None
    assert (
        result.whole_plan_verification.whole_plan_identity
        == candidate.problem.whole_plan_identity
    )
    assert result.nn_fitted_artifact_digest == inputs.model_identity.artifact_digest


def test_application_candidate_uses_bounds_eligible_slots_and_catalog_capacity(
    tmp_path,
):
    adapter = _adapter(tmp_path)
    config = _config(adapter)
    bounded_rules = tuple(
        replace(rule, allocation_bounds=AllocationBounds(1, 1))
        for rule in config.assignment_rules
    )
    bounded = replace(config, assignment_rules=bounded_rules)
    inputs = build_application_inputs(
        bounded,
        student_identity="student-1",
        feature_rows=_feature_rows(),
        adapter=adapter,
    )

    @dataclass(frozen=True)
    class TestOrdering:
        policy_identity: str = "test-ordering-v1"

        def required_pairs(self, sessions):
            return ()

        def is_ordered(self, earlier, later):
            return earlier.week_index < later.week_index

    binding = build_application_planning_problem(
        inputs, adapter=adapter, ordering_policy=TestOrdering()
    )
    state = binding.allocation_space.initial_state
    successor = binding.allocation_space.transition(
        state, binding.allocation_space.actions(state)[0]
    )
    assert not binding.allocation_space.is_candidate_state(successor)
    assert binding.allocation_space.actions(successor)

    complete = successor
    while not binding.allocation_space.is_candidate_state(complete):
        complete = binding.allocation_space.transition(
            complete, binding.allocation_space.actions(complete)[0]
        )
    candidate = binding.candidate_factory(complete)
    assert candidate.problem.weekly_capacity == config.slot_catalog.weekly_capacity
    assert candidate.slot_relative_windows_by_assignment == (
        config.slot_catalog.windows_by_assignment
    )
    assert {
        window
        for _, windows in candidate.slot_relative_windows_by_assignment
        for _, window in windows
    } == {1, 2, 3, 4}
    for domain in candidate.problem.domains:
        assignment_rule = next(
            rule
            for rule in bounded_rules
            if rule.assignment_identity == domain.variable.assignment_id
        )
        assert {slot.slot_id for slot in domain.values} == set(
            assignment_rule.eligible_slot_ids
        )


def test_missing_production_inputs_fail_with_required_configuration(tmp_path):
    adapter = _adapter(tmp_path)
    with pytest.raises(
        ApplicationInputError, match="production planning requires"
    ) as exc:
        build_application_planning_problem(
            None, adapter=adapter, ordering_policy=object()
        )
    message = str(exc.value)
    assert "target marks" in message
    assert "eligible slot IDs" in message
    assert "exact loader-verified fitted model artifact" in message


@pytest.mark.parametrize("mode", ["strict", "proxy"])
def test_loads_application_inputs_from_validated_later_split(tmp_path, mode: str):
    adapter = _adapter(tmp_path, mode)
    config = _config(adapter, mode=mode)
    processed_directory = tmp_path / f"processed-{mode}"
    _write_processed_features(processed_directory, mode)

    inputs = load_application_inputs(
        processed_directory,
        config,
        student_identity="student-1",
        adapter=adapter,
    )

    assert inputs.assignment_ids == ASSIGNMENT_IDS
    assert all(
        context.features["assessment_deadline_day"].iloc[0] == deadline
        for context, deadline in zip(inputs.feature_contexts, DEADLINES, strict=True)
    )
    assert all(
        "target_score" not in context.features for context in inputs.feature_contexts
    )


def test_support_policy_reports_or_rejects_marginal_out_of_range():
    report_policy = SupportPolicy(
        "support", "v1", "train-reference", OutOfSupportAction.REPORT_ONLY
    )
    reject_policy = SupportPolicy(
        "support", "v1", "train-reference", OutOfSupportAction.REJECT
    )

    assert report_policy.permits(MarginalSupportStatus.OUT_OF_RANGE)
    assert not reject_policy.permits(MarginalSupportStatus.OUT_OF_RANGE)
    assert reject_policy.permits(MarginalSupportStatus.IN_RANGE)
