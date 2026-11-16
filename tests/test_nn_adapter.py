"""Tests for the Search-facing strict NN candidate adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

import study_planner.nn.inference as inference_module
import study_planner.search.nn_adapter as nn_adapter_module
from study_planner.csp import (
    AssignmentCandidateIdentity,
    AssignmentEntry,
    AssignmentPlan,
    CspAssignment,
    CspProblem,
    SessionPair,
    SessionVariable,
    VariableDomain,
    WeeklySlot,
    WholePlanManifest,
    WholePlanVerificationPolicy,
    WholePlanVerificationResult,
    WholePlanVerificationStatus,
    build_required_constraints,
    expected_session_variables,
    verify_whole_plan,
)
from study_planner.nn.inference import RegressionPredictor, artifact_digest_from_files
from study_planner.nn.models import MLPRegressor
from study_planner.search.nn_adapter import (
    CandidateEvaluationError,
    CandidateFeatureContext,
    CandidateIdentity,
    CandidateSessionAllocation,
    CspVerificationResult,
    CspVerificationStatus,
    EvaluationPhase,
    EvaluationStatus,
    ScheduleData,
    ScheduledSession,
    SessionNNAdapter,
)
from study_planner.search.session_mapping import (
    STRICT_FFF_FEATURE_NAMES,
    ClickScenario,
    JointSupportStatus,
    MarginalSupportStatus,
    SessionClickMappingConfig,
)


def strict_row(**overrides: object) -> pd.DataFrame:
    values: dict[str, object] = dict.fromkeys(STRICT_FFF_FEATURE_NAMES, 0.0)
    values.update(
        {
            "assessment_type_cma": 1.0,
            "assessment_weight": 10.0,
            "assessment_weight_available": 1.0,
            "assessment_deadline_day": 47.0,
            "weeks_since_course_start": 47 / 7,
        }
    )
    values.update(overrides)
    return pd.DataFrame([values], columns=STRICT_FFF_FEATURE_NAMES)


class FakePredictor:
    """Small deterministic stand-in for the RegressionPredictor interface."""

    feature_names = STRICT_FFF_FEATURE_NAMES
    assessment_type_columns = STRICT_FFF_FEATURE_NAMES[:3]

    def __init__(
        self, fail: bool = False, artifact_bytes: bytes = b"weights-a"
    ) -> None:
        self.fail = fail
        self.mutate_input = False
        self.output_offset = 0.0
        self.calls: list[pd.DataFrame] = []
        self.artifact_bytes = artifact_bytes
        _set_artifact_digest(self)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.fail:
            raise RuntimeError("inference unavailable")
        self.calls.append(features.copy(deep=True))
        if self.mutate_input:
            features.iloc[0, 0] = 0.0
        click_total = features.loc[:, list(STRICT_FFF_FEATURE_NAMES[7:])].sum(axis=1)
        return np.asarray(
            [60.0 + min(float(click_total.iloc[0]) / 1000, 20) + self.output_offset]
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


def _set_artifact_digest(predictor: FakePredictor) -> None:
    mode = (
        "proxy"
        if "previous_assessment_score_lag_1" in predictor.feature_names
        else "strict"
    )
    predictor.artifact_digest = artifact_digest_from_files(
        {
            "feature_schema.json": repr((mode, predictor.feature_names)).encode(),
            "experiment_config.json": f'{{"feature_mode":"{mode}"}}'.encode(),
            "preprocessor.joblib": b"controlled-preprocessor-bytes",
            "mlp_seed_42.pt": predictor.artifact_bytes,
        }
    )
    predictor._fixture_state_snapshot = _fake_fitted_state(predictor)


def _fake_fitted_state(predictor: FakePredictor) -> str:
    payload = repr(
        (
            tuple(predictor.feature_names),
            tuple(predictor.assessment_type_columns),
            predictor.artifact_bytes,
        )
    ).encode("utf-8")
    return artifact_digest_from_files({"controlled-state.bin": payload})


@pytest.fixture(autouse=True)
def _isolated_fake_predictor_trust(monkeypatch):
    """Trust controlled fakes only in this test module."""
    real_lookup = nn_adapter_module.trusted_artifact_provenance
    real_verify = nn_adapter_module.verify_trusted_artifact_integrity
    capabilities: dict[int, object] = {}

    def lookup(predictor):
        if type(predictor) is not FakePredictor:
            return real_lookup(predictor)
        key = id(predictor)
        if key not in capabilities:
            mode = (
                "proxy"
                if "previous_assessment_score_lag_1" in predictor.feature_names
                else "strict"
            )
            capabilities[key] = SimpleNamespace(
                digest=predictor.artifact_digest,
                mode=mode,
                fixture_state=predictor._fixture_state_snapshot,
            )
        return capabilities[key]

    def verify(predictor, expected=None):
        if type(predictor) is not FakePredictor:
            return real_verify(predictor, expected)
        capability = lookup(predictor)
        if expected is not None and capability is not expected:
            raise ValueError("test predictor capability mismatch")
        if (
            predictor.artifact_digest != capability.digest
            or _fake_fitted_state(predictor) != capability.fixture_state
        ):
            raise ValueError("controlled test predictor state was changed")
        return capability

    monkeypatch.setattr(nn_adapter_module, "trusted_artifact_provenance", lookup)
    monkeypatch.setattr(
        nn_adapter_module, "verify_trusted_artifact_integrity", verify
    )


def _write_controlled_artifact(directory, weight_value: float) -> None:
    directory.mkdir()
    (directory / "feature_schema.json").write_text(
        json.dumps(
            {
                "mode": "strict",
                "source_feature_columns": list(STRICT_FFF_FEATURE_NAMES),
                "assessment_type_indicator_columns": list(
                    STRICT_FFF_FEATURE_NAMES[:3]
                ),
            }
        ),
        encoding="utf-8",
    )
    (directory / "experiment_config.json").write_text(
        json.dumps(
            {
                "feature_mode": "strict",
                "selected_mlp_config": {"hidden_sizes": [2], "dropout": 0.0},
                "preprocessed_feature_count": len(STRICT_FFF_FEATURE_NAMES),
                "model_files": ["mlp_seed_1.pt"],
            }
        ),
        encoding="utf-8",
    )
    preprocessor = StandardScaler().fit(
        np.zeros((2, len(STRICT_FFF_FEATURE_NAMES)), dtype=np.float64)
    )
    joblib.dump(preprocessor, directory / "preprocessor.joblib")
    model = MLPRegressor(len(STRICT_FFF_FEATURE_NAMES), (2,), 0.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(weight_value)
    torch.save(model.state_dict(), directory / "mlp_seed_1.pt")


def _load_controlled_predictor(tmp_path, weight_value: float = 0.125):
    directory = tmp_path / f"controlled-{weight_value}"
    _write_controlled_artifact(directory, weight_value)
    return RegressionPredictor.from_artifact_dir(directory)


def setup_adapter(*, scenario: ClickScenario = ClickScenario.LOW, fail: bool = False):
    predictor = FakePredictor(fail=fail)
    adapter = SessionNNAdapter(
        predictor,
        SessionClickMappingConfig(scenario=scenario),
    )
    context = CandidateFeatureContext("student-1", "assignment-1", strict_row())
    return predictor, adapter, context


def verified_result(
    adapter: SessionNNAdapter,
    allocation: CandidateSessionAllocation,
    context: CandidateFeatureContext,
    schedule: ScheduleData,
    *,
    status: CspVerificationStatus = CspVerificationStatus.FEASIBLE,
) -> WholePlanVerificationResult:
    """Build actual whole-plan verification for adapter contract tests."""
    names = (context.assignment_identity, "assignment-2", "assignment-3")
    counts = (allocation.weekly_sessions, (0, 0, 0, 0), (0, 0, 0, 0))
    plans = tuple(
        AssignmentPlan(
            name,
            AssignmentCandidateIdentity(
                context.student_identity,
                "course-1",
                name,
                count,
                context.base_feature_context_identity
                if name == context.assignment_identity
                else f"base-{name}",
                adapter.mapping_scenario_identity,
            ),
            context.base_feature_context_identity
            if name == context.assignment_identity
            else f"base-{name}",
        )
        for name, count in zip(names, counts, strict=True)
    )
    manifest = WholePlanManifest(
        context.student_identity,
        "course-1",
        plans,
        adapter.mapping_scenario_identity,
    )
    effective_schedule = schedule
    if status is CspVerificationStatus.INFEASIBLE and not schedule.sessions:
        effective_schedule = ScheduleData((ScheduledSession("invalid-slot", 1),))
    variables = expected_session_variables(manifest)
    slots = tuple(
        sorted(
            {
                WeeklySlot(
                    session.slot_id,
                    0
                    if session.relative_window is None
                    else session.relative_window - 1,
                )
                for session in effective_schedule.sessions
            }
            or {WeeklySlot("unused", 0)}
        )
    )
    domains = tuple(VariableDomain(variable, slots) for variable in variables)
    eligible = tuple(
        (
            variable,
            frozenset()
            if status is CspVerificationStatus.INFEASIBLE
            and variable.assignment_id == context.assignment_identity
            and effective_schedule.sessions
            else frozenset(slots),
        )
        for variable in variables
    )

    @dataclass(frozen=True)
    class _Policy:
        policy_identity: str = "adapter-test-order-v1"

        def required_pairs(self, sessions: tuple[SessionVariable, ...]):
            grouped: dict[str, list[SessionVariable]] = {}
            for variable in sessions:
                grouped.setdefault(variable.assignment_id, []).append(variable)
            return tuple(
                SessionPair(first, second)
                for group in grouped.values()
                for first, second in zip(group, group[1:], strict=False)
            )

        def is_ordered(self, earlier: WeeklySlot, later: WeeklySlot) -> bool:
            return earlier.week_index < later.week_index

    ordering = _Policy()
    constraints = build_required_constraints(
        manifest=manifest,
        variables=variables,
        eligible_slots=eligible,
        ordering_policy=ordering,
        weekly_capacity=max(1, len(variables)),
    )
    problem = CspProblem(
        manifest,
        variables,
        domains,
        constraints,
        max(1, len(variables)),
        ordering,
    )
    assignment = CspAssignment(
        tuple(
            AssignmentEntry(
                SessionVariable(context.assignment_identity, index),
                WeeklySlot(
                    session.slot_id,
                    0
                    if session.relative_window is None
                    else session.relative_window - 1,
                ),
            )
            for index, session in enumerate(effective_schedule.sessions)
        )
    )
    policy = WholePlanVerificationPolicy(
        "adapter-test-v1",
        "test-deadline",
        problem.ordering_policy.configuration_identity,
        "test-slots",
        "test-capacity",
        "slot-id-week",
        problem.weekly_capacity,
    )
    expected_plan = manifest.canonical_identity
    expected_problem = problem.problem_identity
    result = verify_whole_plan(
        problem,
        assignment,
        policy,
        expected_whole_plan_identity=expected_plan,
        expected_problem_identity=expected_problem,
    )
    if status is CspVerificationStatus.INFEASIBLE:
        assert result.status is WholePlanVerificationStatus.INFEASIBLE
    return result


def evaluate_final(
    adapter,
    allocation,
    verification,
    context,
    *,
    expected_model_provenance_identity=None,
):
    return adapter.evaluate_final(
        allocation,
        verification,
        context,
        expected_whole_plan_identity=verification.whole_plan_identity,
        expected_problem_identity=verification.problem_identity,
        expected_policy_identity=verification.verification_policy_identity,
        expected_policy_version=verification.verification_policy_version,
        expected_model_provenance_identity=(
            adapter.model_provenance_identity
            if expected_model_provenance_identity is None
            else expected_model_provenance_identity
        ),
        slot_relative_windows={
            session.slot_id: session.week_index + 1
            for group in verification.schedule
            for session in group.sessions
        },
    )


def test_valid_provisional_evaluation_is_not_goal_test_eligible() -> None:
    predictor, adapter, context = setup_adapter()
    allocation = CandidateSessionAllocation((1, 0, 0, 0))

    result = adapter.evaluate_provisional(allocation, context)

    assert result.phase is EvaluationPhase.PROVISIONAL
    assert result.status is EvaluationStatus.PREDICTED
    assert result.prediction == pytest.approx(60.022)
    assert result.mapping.weekly_sessions == (1, 0, 0, 0)
    assert result.mapping.scenario is ClickScenario.LOW
    assert result.mapping.joint_support_status is JointSupportStatus.UNASSESSED
    assert result.mapping.marginal_status is MarginalSupportStatus.IN_RANGE
    assert not result.usable_for_goal_test(result.identity)
    assert len(predictor.calls) == 1


def test_unverified_schedule_data_cannot_produce_final_usable_result() -> None:
    _, adapter, context = setup_adapter()
    allocation = CandidateSessionAllocation((1, 1, 0, 0))
    unverified_schedule = ScheduleData(
        (ScheduledSession("slot-a", 1), ScheduledSession("slot-b", 2))
    )

    with pytest.raises(CandidateEvaluationError, match="WholePlanVerificationResult"):
        adapter.evaluate_final(
            allocation,
            unverified_schedule,  # type: ignore[arg-type]
            context,
            expected_whole_plan_identity="expected-plan",
            expected_problem_identity="expected-problem",
            expected_policy_identity="expected-policy",
            expected_policy_version="expected-version",
            expected_model_provenance_identity=adapter.model_provenance_identity,
            slot_relative_windows={},
        )
    with pytest.raises(TypeError, match="must come from a CSP solver"):
        CspVerificationResult()


def test_verified_csp_result_allows_final_evaluation_and_preserves_schedule() -> None:
    _, adapter, context = setup_adapter()
    allocation = CandidateSessionAllocation((1, 1, 0, 0))
    schedule = ScheduleData(
        (ScheduledSession("slot-a", 1), ScheduledSession("slot-b", 2))
    )
    verification = verified_result(adapter, allocation, context, schedule)

    result = evaluate_final(adapter, allocation, verification, context)

    assert result.phase is EvaluationPhase.FINAL_SCHEDULED
    assert result.status is EvaluationStatus.PREDICTED
    assert result.mapping.weekly_sessions == (1, 1, 0, 0)
    assert result.schedule_identity == (("slot-a", 1), ("slot-b", 2))
    assert result.csp_verification_identity == verification.verification_identity
    assert result.fitted_artifact_digest == adapter.fitted_artifact_digest
    assert result.model_provenance_identity == adapter.model_provenance_identity
    assert result.model_mode is adapter.model_mode
    assert result.feature_schema_identity == adapter.feature_schema_identity
    assert result.prediction_identity is not None
    assert result.usable_for_goal_test(
        result.identity,
        result.schedule_identity,
        result.csp_verification_identity,
        verification.whole_plan_identity,
        verification.problem_identity,
        model_provenance_identity=adapter.model_provenance_identity,
        fitted_artifact_digest=adapter.fitted_artifact_digest,
        model_mode=adapter.model_mode,
        feature_schema_identity=adapter.feature_schema_identity,
    )
    assert not result.usable_for_goal_test(
        result.identity,
        result.schedule_identity,
        result.csp_verification_identity,
        verification.whole_plan_identity,
        verification.problem_identity,
        model_provenance_identity=adapter.model_provenance_identity,
        fitted_artifact_digest="0" * 64,
        model_mode=adapter.model_mode,
        feature_schema_identity=adapter.feature_schema_identity,
    )
    assert not result.usable_for_goal_test(
        result.identity,
        result.schedule_identity,
        result.csp_verification_identity,
        verification.whole_plan_identity,
        verification.problem_identity,
        model_provenance_identity=adapter.model_provenance_identity,
        fitted_artifact_digest=adapter.fitted_artifact_digest,
        model_mode=None,
        feature_schema_identity=adapter.feature_schema_identity,
    )
    assert not result.usable_for_goal_test(
        result.identity,
        result.schedule_identity,
        result.csp_verification_identity,
        verification.whole_plan_identity,
        verification.problem_identity,
        model_provenance_identity=adapter.model_provenance_identity,
        fitted_artifact_digest=adapter.fitted_artifact_digest,
        model_mode=adapter.model_mode,
        feature_schema_identity="0" * 64,
    )
    assert not result.usable_for_goal_test(
        result.identity,
        result.schedule_identity,
        result.csp_verification_identity,
        verification.whole_plan_identity,
        verification.problem_identity,
        model_provenance_identity="0" * 64,
        fitted_artifact_digest=adapter.fitted_artifact_digest,
        model_mode=adapter.model_mode,
        feature_schema_identity=adapter.feature_schema_identity,
    )
    assert not result.usable_for_goal_test(
        result.identity, (("different-slot", 1),), result.csp_verification_identity
    )
    assert result.monotonicity_report is not None
    assert result.monotonicity_report.observations[0].schedule_identity == (
        ("slot-a", 1),
        ("slot-b", 2),
    )


def test_nn_ignores_deadline_eligible_slot_outside_click_windows() -> None:
    _, adapter, context = setup_adapter()
    allocation = CandidateSessionAllocation((1, 1, 0, 0))
    schedule = ScheduleData(
        (
            ScheduledSession("old-deadline-eligible-slot", None),
            ScheduledSession("recent-click-window-slot", 2),
        )
    )
    verification = verified_result(adapter, allocation, context, schedule)

    result = adapter.evaluate_final(
        allocation,
        verification,
        context,
        expected_whole_plan_identity=verification.whole_plan_identity,
        expected_problem_identity=verification.problem_identity,
        expected_policy_identity=verification.verification_policy_identity,
        expected_policy_version=verification.verification_policy_version,
        expected_model_provenance_identity=adapter.model_provenance_identity,
        slot_relative_windows={"recent-click-window-slot": 2},
    )

    assert result.status is EvaluationStatus.PREDICTED
    assert result.schedule_identity == (
        ("old-deadline-eligible-slot", None),
        ("recent-click-window-slot", 2),
    )
    assert result.mapping.weekly_sessions == (0, 1, 0, 0)


def test_infeasible_csp_result_is_rejected_for_final_prediction() -> None:
    _, adapter, context = setup_adapter()
    allocation = CandidateSessionAllocation((1, 0, 0, 0))
    verification = verified_result(
        adapter,
        allocation,
        context,
        ScheduleData(()),
        status=CspVerificationStatus.INFEASIBLE,
    )

    with pytest.raises(CandidateEvaluationError, match="did not establish feasibility"):
        evaluate_final(adapter, allocation, verification, context)


def test_schedule_and_candidate_identity_mismatches_are_rejected() -> None:
    _, adapter, context = setup_adapter()
    allocation = CandidateSessionAllocation((1, 0, 0, 0))
    schedule = ScheduleData((ScheduledSession("slot-a", 1),))
    verification = verified_result(adapter, allocation, context, schedule)
    other_allocation = CandidateSessionAllocation((0, 1, 0, 0))

    with pytest.raises(CandidateEvaluationError, match="candidate identity mismatch"):
        evaluate_final(adapter, other_allocation, verification, context)

    other_context = CandidateFeatureContext("student-2", "assignment-1", strict_row())
    with pytest.raises(CandidateEvaluationError, match="candidate identity mismatch"):
        evaluate_final(adapter, allocation, verification, other_context)


def test_result_cannot_be_reused_for_a_different_candidate() -> None:
    _, adapter, context = setup_adapter()
    allocation = CandidateSessionAllocation((1, 0, 0, 0))
    schedule = ScheduleData((ScheduledSession("slot-a", 1),))
    result = evaluate_final(
        adapter,
        allocation,
        verified_result(adapter, allocation, context, schedule),
        context,
    )
    other_identity = CandidateIdentity(
        student_identity="student-2",
        assignment_identity=result.identity.assignment_identity,
        canonical_allocation_identity=result.identity.canonical_allocation_identity,
        base_feature_context_identity=result.identity.base_feature_context_identity,
        mapping_scenario_identity=result.identity.mapping_scenario_identity,
    )

    with pytest.raises(CandidateEvaluationError, match="identity mismatch"):
        result.usable_for_goal_test(
            other_identity, result.schedule_identity, "csp-run-1"
        )
    other_allocation_identity = adapter.candidate_identity(
        CandidateSessionAllocation((0, 1, 0, 0)), context
    )
    with pytest.raises(CandidateEvaluationError, match="identity mismatch"):
        result.usable_for_goal_test(
            other_allocation_identity,
            result.schedule_identity,
            "csp-run-1",
        )


def test_mapping_scenario_identity_mismatch_rejects_csp_result() -> None:
    _, adapter, context = setup_adapter()
    allocation = CandidateSessionAllocation((1, 0, 0, 0))
    verification = verified_result(
        adapter,
        allocation,
        context,
        ScheduleData((ScheduledSession("slot-a", 1),)),
    )
    other_adapter = SessionNNAdapter(
        FakePredictor(),
        SessionClickMappingConfig(ClickScenario.HIGH),
    )

    with pytest.raises(CandidateEvaluationError, match="mapping scenario mismatch"):
        evaluate_final(other_adapter, allocation, verification, context)


def test_matching_context_compares_increasing_allocations_across_schedules() -> None:
    _, adapter, context = setup_adapter()
    zero = CandidateSessionAllocation((0, 0, 0, 0))
    one = CandidateSessionAllocation((1, 0, 0, 0))
    zero_schedule = ScheduleData(())
    one_schedule = ScheduleData((ScheduledSession("slot-a", 1),))

    evaluate_final(
        adapter, zero, verified_result(adapter, zero, context, zero_schedule), context
    )
    result = evaluate_final(
        adapter, one, verified_result(adapter, one, context, one_schedule), context
    )

    report = result.monotonicity_report
    assert report is not None
    assert report.comparisons == 1
    assert len(report.observations) == 2
    assert report.observations[0].context == report.observations[1].context
    assert (
        report.observations[0].schedule_identity
        != report.observations[1].schedule_identity
    )


def test_unrelated_contexts_are_never_compared() -> None:
    _, adapter, context = setup_adapter()
    allocation = CandidateSessionAllocation((0, 0, 0, 0))
    empty_schedule = ScheduleData(())
    evaluate_final(
        adapter,
        allocation,
        verified_result(adapter, allocation, context, empty_schedule),
        context,
    )
    other_context = CandidateFeatureContext("student-2", "assignment-1", strict_row())
    next_allocation = CandidateSessionAllocation((1, 0, 0, 0))
    result = evaluate_final(
        adapter,
        next_allocation,
        verified_result(
            adapter,
            next_allocation,
            other_context,
            ScheduleData((ScheduledSession("slot-a", 1),)),
        ),
        other_context,
    )

    assert result.monotonicity_report is not None
    assert result.monotonicity_report.comparisons == 0
    assert len(result.monotonicity_report.observations) == 1


def test_schema_mismatch_is_rejected_before_inference() -> None:
    predictor = FakePredictor()
    predictor.feature_names = predictor.feature_names[:-1]

    with pytest.raises(CandidateEvaluationError, match="schema/order"):
        SessionNNAdapter(predictor, SessionClickMappingConfig(ClickScenario.LOW))


def test_missing_or_malformed_artifact_digest_is_rejected() -> None:
    config = SessionClickMappingConfig(ClickScenario.LOW)
    for invalid_digest in (None, "artifact-label", "0" * 63, "g" * 64):
        predictor = FakePredictor()
        predictor.artifact_digest = invalid_digest
        with pytest.raises(
            CandidateEvaluationError,
            match="digest is malformed|state was changed",
        ):
            SessionNNAdapter(predictor, config)


def test_changed_loaded_artifact_digest_is_rejected_for_final_evaluation() -> None:
    predictor, adapter, context = setup_adapter()
    allocation = CandidateSessionAllocation((0, 0, 0, 0))
    verification = verified_result(adapter, allocation, context, ScheduleData(()))
    predictor.artifact_digest = "0" * 64

    with pytest.raises(CandidateEvaluationError, match="state was changed"):
        evaluate_final(adapter, allocation, verification, context)


def test_persisted_artifact_bytes_produce_stable_distinct_digests() -> None:
    first_predictor = FakePredictor(artifact_bytes=b"fitted weights checkpoint A")
    same_predictor = FakePredictor(artifact_bytes=b"fitted weights checkpoint A")
    different_predictor = FakePredictor(artifact_bytes=b"fitted weights checkpoint B")
    config = SessionClickMappingConfig(ClickScenario.LOW)
    first = SessionNNAdapter(first_predictor, config)
    same = SessionNNAdapter(same_predictor, config)
    different = SessionNNAdapter(different_predictor, config)

    assert first.fitted_artifact_digest == same.fitted_artifact_digest
    assert first.model_provenance_identity == same.model_provenance_identity
    assert first.fitted_artifact_digest != different.fitted_artifact_digest
    assert first.model_provenance_identity != different.model_provenance_identity
    assert first.model_mode is different.model_mode


def test_loaded_artifact_identity_and_predictions_are_stable(tmp_path) -> None:
    source = tmp_path / "model-a"
    clone = tmp_path / "model-a-copy"
    _write_controlled_artifact(source, 0.125)
    clone.mkdir()
    for artifact in source.iterdir():
        (clone / artifact.name).write_bytes(artifact.read_bytes())
    loaded = RegressionPredictor.from_artifact_dir(source)
    loaded_clone = RegressionPredictor.from_artifact_dir(clone)
    different = _load_controlled_predictor(tmp_path, 0.25)

    assert loaded.artifact_digest == loaded_clone.artifact_digest
    assert loaded.artifact_digest != different.artifact_digest
    adapter = SessionNNAdapter(loaded, SessionClickMappingConfig(ClickScenario.LOW))
    clone_adapter = SessionNNAdapter(
        loaded_clone, SessionClickMappingConfig(ClickScenario.LOW)
    )
    assert adapter.model_provenance_identity == clone_adapter.model_provenance_identity
    assert adapter.model_provenance_identity != SessionNNAdapter(
        different, SessionClickMappingConfig(ClickScenario.LOW)
    ).model_provenance_identity
    allocation = CandidateSessionAllocation((0, 0, 0, 0))
    context = CandidateFeatureContext("student-1", "assignment-1", strict_row())
    verification = verified_result(adapter, allocation, context, ScheduleData(()))
    first = evaluate_final(adapter, allocation, verification, context)
    second = evaluate_final(adapter, allocation, verification, context)
    assert first.prediction == second.prediction
    assert first.fitted_artifact_digest == loaded.artifact_digest
    assert first.model_provenance_identity == adapter.model_provenance_identity


def test_direct_predictor_construction_cannot_issue_loader_provenance(
    tmp_path,
) -> None:
    loaded = _load_controlled_predictor(tmp_path)
    direct = RegressionPredictor(
        loaded.preprocessor,
        loaded.feature_names,
        loaded.assessment_type_columns,
        loaded.models,
        "a" * 64,
    )

    assert inference_module.trusted_artifact_provenance(direct) is None
    assert not hasattr(inference_module, "_register_loaded_artifact")
    with pytest.raises(AttributeError):
        inference_module._register_loaded_artifact(  # type: ignore[attr-defined]
            direct, digest="a" * 64, mode="strict"
        )
    with pytest.raises(ValueError, match="no trusted artifact"):
        inference_module.verify_trusted_artifact_integrity(direct)
    with pytest.raises(CandidateEvaluationError, match="verified artifact loader"):
        SessionNNAdapter(direct, SessionClickMappingConfig(ClickScenario.LOW))


def test_loader_capability_cannot_be_reused_with_another_predictor(tmp_path) -> None:
    first = _load_controlled_predictor(tmp_path, 0.125)
    second = _load_controlled_predictor(tmp_path, 0.25)
    first_capability = inference_module.trusted_artifact_provenance(first)

    assert first_capability is not None
    assert inference_module.verify_trusted_artifact_integrity(
        first, first_capability
    ) is first_capability
    with pytest.raises(ValueError, match="no trusted artifact"):
        inference_module.verify_trusted_artifact_integrity(second, first_capability)


def test_custom_predictor_forged_digest_is_not_trusted() -> None:
    class ForgedPredictor(FakePredictor):
        pass

    predictor = ForgedPredictor()
    with pytest.raises(CandidateEvaluationError, match="verified artifact loader"):
        SessionNNAdapter(
            predictor, SessionClickMappingConfig(ClickScenario.LOW)
        )


@pytest.mark.parametrize("mutation", ["weight", "preprocessor", "models", "digest"])
def test_loaded_predictor_mutations_fail_closed(tmp_path, mutation: str) -> None:
    predictor = _load_controlled_predictor(tmp_path)
    if mutation == "weight":
        with torch.no_grad():
            predictor.models[0].layers[0].weight[0, 0].add_(1)
    elif mutation == "preprocessor":
        predictor.preprocessor.mean_[0] += 1
    elif mutation == "models":
        predictor.models.append(predictor.models[0])
    else:
        predictor.artifact_digest = "0" * 64
    with pytest.raises(ValueError, match="state|artifact"):
        predictor.predict(strict_row())


def test_mutation_after_prediction_invalidates_goal_authorization(tmp_path) -> None:
    predictor = _load_controlled_predictor(tmp_path)
    adapter = SessionNNAdapter(predictor, SessionClickMappingConfig(ClickScenario.LOW))
    allocation = CandidateSessionAllocation((0, 0, 0, 0))
    context = CandidateFeatureContext("student-1", "assignment-1", strict_row())
    verification = verified_result(adapter, allocation, context, ScheduleData(()))
    result = evaluate_final(adapter, allocation, verification, context)
    with torch.no_grad():
        predictor.models[0].layers[0].weight[0, 0].add_(1)
    assert not result.usable_for_goal_test(
        result.identity,
        result.schedule_identity,
        result.csp_verification_identity,
        result.whole_plan_identity,
        result.problem_identity,
        model_provenance_identity=result.model_provenance_identity,
        fitted_artifact_digest=result.fitted_artifact_digest,
        model_mode=result.model_mode,
        feature_schema_identity=result.feature_schema_identity,
    )


def test_model_provenance_binds_mode_and_schema() -> None:
    _, model_a, _ = setup_adapter()
    model_b_predictor = FakePredictor()
    model_b_predictor.feature_names = MODEL_B_FEATURE_NAMES
    _set_artifact_digest(model_b_predictor)
    model_b = SessionNNAdapter(
        model_b_predictor, SessionClickMappingConfig(ClickScenario.LOW)
    )

    assert model_a.model_mode is not model_b.model_mode
    assert model_a.feature_schema_identity != model_b.feature_schema_identity
    assert model_a.model_provenance_identity != model_b.model_provenance_identity
    _, adapter_a, context = setup_adapter()
    allocation = CandidateSessionAllocation((0, 0, 0, 0))
    verification = verified_result(adapter_a, allocation, context, ScheduleData(()))
    prediction_a = evaluate_final(adapter_a, allocation, verification, context)
    assert not prediction_a.usable_for_goal_test(
        prediction_a.identity,
        prediction_a.schedule_identity,
        prediction_a.csp_verification_identity,
        prediction_a.whole_plan_identity,
        prediction_a.problem_identity,
        model_provenance_identity=model_b.model_provenance_identity,
        fitted_artifact_digest=model_b.fitted_artifact_digest,
        model_mode=model_b.model_mode,
        feature_schema_identity=model_b.feature_schema_identity,
    )


def test_prediction_identity_binds_model_and_prediction_output() -> None:
    predictor, adapter, context = setup_adapter()
    allocation = CandidateSessionAllocation((1, 0, 0, 0))
    verification = verified_result(
        adapter,
        allocation,
        context,
        ScheduleData((ScheduledSession("slot-a", 1),)),
    )
    first = evaluate_final(adapter, allocation, verification, context)
    with pytest.raises(CandidateEvaluationError, match="model provenance identity"):
        evaluate_final(
            adapter,
            allocation,
            verification,
            context,
            expected_model_provenance_identity="0" * 64,
        )
    other_artifact = SessionNNAdapter(
        FakePredictor(artifact_bytes=b"different fitted checkpoint"),
        SessionClickMappingConfig(ClickScenario.LOW),
    )
    other = evaluate_final(other_artifact, allocation, verification, context)
    assert first.prediction_identity != other.prediction_identity
    assert not first.usable_for_goal_test(
        first.identity,
        first.schedule_identity,
        first.csp_verification_identity,
        first.whole_plan_identity,
        first.problem_identity,
        model_provenance_identity=other_artifact.model_provenance_identity,
        fitted_artifact_digest=other_artifact.fitted_artifact_digest,
        model_mode=other_artifact.model_mode,
        feature_schema_identity=other_artifact.feature_schema_identity,
    )

    predictor.output_offset = 1.0
    changed_output = evaluate_final(adapter, allocation, verification, context)
    assert changed_output.prediction != first.prediction
    assert changed_output.prediction_identity != first.prediction_identity

    stale = replace(first, prediction=first.prediction + 1.0)
    assert not stale.usable_for_goal_test(
        stale.identity,
        stale.schedule_identity,
        stale.csp_verification_identity,
        stale.whole_plan_identity,
        stale.problem_identity,
        model_provenance_identity=adapter.model_provenance_identity,
        fitted_artifact_digest=adapter.fitted_artifact_digest,
        model_mode=adapter.model_mode,
        feature_schema_identity=adapter.feature_schema_identity,
    )


def test_source_and_mapping_snapshots_are_immutable_from_consumers() -> None:
    predictor, adapter, context = setup_adapter()
    predictor.mutate_input = True
    before = context.features.copy(deep=True)
    allocation = CandidateSessionAllocation((1, 0, 0, 0))
    result = adapter.evaluate_provisional(allocation, context)
    exposed = result.mapping.candidate_features
    exposed.loc[0, "assessment_type_cma"] = 0

    pd.testing.assert_frame_equal(context.features, before)
    assert result.mapping.candidate_features.loc[0, "assessment_type_cma"] == 1
    assert isinstance(result.mapping.features.values, tuple)
    with pytest.raises((AttributeError, TypeError)):
        result.mapping.features.values[0] = 0  # type: ignore[index]


def test_missing_values_and_support_diagnostics_are_preserved() -> None:
    predictor, adapter, _ = setup_adapter()
    context = CandidateFeatureContext(
        "student-1",
        "assignment-1",
        strict_row(
            assessment_weight=float("nan"),
            assessment_type_cma=float("nan"),
            assessment_type_exam=float("nan"),
            assessment_type_tma=float("nan"),
        ),
    )
    result = adapter.evaluate_provisional(
        CandidateSessionAllocation((0, 0, 0, 0)), context
    )
    assert "assessment_weight" in result.mapping.missing_numeric_features
    assert pd.isna(predictor.calls[0].loc[0, "assessment_type_cma"])

    oos_context = CandidateFeatureContext(
        "student-1", "assignment-1", strict_row(click_content_navigation_week_1=3796)
    )
    oos_result = adapter.evaluate_provisional(
        CandidateSessionAllocation((1, 0, 0, 0)), oos_context
    )
    assert oos_result.mapping.support_diagnostics[0].candidate_value == 3818
    assert oos_result.mapping.support_diagnostics[0].marginal_status is (
        MarginalSupportStatus.OUT_OF_RANGE
    )


def test_inference_failure_is_structured_and_not_goal_eligible() -> None:
    _, adapter, context = setup_adapter(fail=True)
    allocation = CandidateSessionAllocation((1, 0, 0, 0))
    result = adapter.evaluate_provisional(allocation, context)

    assert result.status is EvaluationStatus.INFERENCE_FAILED
    assert result.prediction is None
    assert result.error_type == "RuntimeError"
    assert result.error_message == "inference unavailable"
    assert not result.usable_for_goal_test(result.identity)


def test_duplicate_slots_and_allocation_mismatch_are_rejected() -> None:
    _, adapter, context = setup_adapter()
    with pytest.raises(CandidateEvaluationError, match="only once"):
        ScheduleData((ScheduledSession("slot-a", 1), ScheduledSession("slot-a", 2)))

    allocation = CandidateSessionAllocation((2, 0, 0, 0))
    schedule = ScheduleData((ScheduledSession("slot-a", 1),))
    with pytest.raises(CandidateEvaluationError, match="did not establish feasibility"):
        evaluate_final(
            adapter,
            allocation,
            verified_result(adapter, allocation, context, schedule),
            context,
        )
