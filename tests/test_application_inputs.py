"""Tests for authoritative and provenance-bound OULAD planning inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from study_planner.csp import SessionPair, SessionVariable, WeeklySlot
from study_planner.integration import (
    AllocationBounds,
    ApplicationAssignmentRule,
    ApplicationInputConfig,
    ApplicationInputError,
    ApplicationSlot,
    AssignmentTarget,
    ModelArtifactIdentity,
    OuladPresentationSplit,
    OutOfSupportAction,
    PlannerStatus,
    SupportPolicy,
    TargetRule,
    WeeklySlotCatalog,
    WeeklyTimeWindow,
    build_application_inputs,
    build_application_planning_problem,
    integrated_uniform_cost_search,
    load_application_inputs,
    load_session_nn_adapter,
)
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
    rules = tuple(
        ApplicationAssignmentRule(
            assignment_identity=assignment_id,
            assessment_id=assessment_id,
            target_mark=target,
            deadline_day=deadline,
            eligible_slot_ids=(f"slot-{index}",),
            allocation_bounds=AllocationBounds(0, 3),
        )
        for index, (assignment_id, assessment_id, target, deadline) in enumerate(
            zip(ASSIGNMENT_IDS, ASSESSMENT_IDS, TARGETS, DEADLINES, strict=True)
        )
    )
    assignment_windows = {
        assignment_id: index + 1 for index, assignment_id in enumerate(ASSIGNMENT_IDS)
    }
    slots = tuple(
        ApplicationSlot(
            WeeklySlot(f"slot-{index}", index),
            WeeklyTimeWindow(1, 9 * 60, 10 * 60, "approved-local-timezone"),
            tuple(assignment_windows.items()),
        )
        for index in range(3)
    )
    mapping = SessionClickMappingConfig(ClickScenario.CENTRAL)
    return ApplicationInputConfig(
        presentation_split=OuladPresentationSplit("FFF", ("2013B", "2013J"), "2014J"),
        assignment_rules=rules,
        target_rule=TargetRule(
            "explicit-targets", "v1", tuple(zip(ASSIGNMENT_IDS, TARGETS, strict=True))
        ),
        slot_catalog=WeeklySlotCatalog(
            "test-catalog",
            "v1",
            weekly_capacity=2,
            deadline_policy_identity="approved-deadline-input",
            deadline_policy_version="v1",
            slots=slots,
        ),
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
    assert dict(config.slot_catalog.windows_by_assignment)["a1"] == tuple(
        (f"slot-{index}", 1) for index in range(3)
    )


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


def test_requires_three_distinct_assessment_inputs(tmp_path):
    adapter = _adapter(tmp_path, mode="strict")
    config = _config(adapter)
    duplicate_rule = ApplicationAssignmentRule(
        "a2", "34860", 75, 47, ("slot-1",), AllocationBounds(0, 3)
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
    assert (
        len(
            {
                windows[0][1]
                for _, windows in candidate.slot_relative_windows_by_assignment
            }
        )
        == 3
    )
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
