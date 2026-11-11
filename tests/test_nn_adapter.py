"""Tests for the Search-facing strict NN candidate adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.mutate_input = False
        self.calls: list[pd.DataFrame] = []

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.fail:
            raise RuntimeError("inference unavailable")
        self.calls.append(features.copy(deep=True))
        if self.mutate_input:
            features.iloc[0, 0] = 0.0
        click_total = features.loc[:, list(STRICT_FFF_FEATURE_NAMES[7:])].sum(axis=1)
        return np.asarray([60.0 + min(float(click_total.iloc[0]) / 1000, 20)])


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
) -> CspVerificationResult:
    """Use the future solver boundary factory to model trusted CSP output."""
    return CspVerificationResult._from_solver(
        candidate_identity=adapter.candidate_identity(allocation, context),
        status=status,
        schedule=schedule if status is CspVerificationStatus.FEASIBLE else None,
        verification_identity="csp-run-1",
        rejection_reasons=("deadline",)
        if status is CspVerificationStatus.INFEASIBLE
        else (),
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

    with pytest.raises(CandidateEvaluationError, match="typed CSP verification"):
        adapter.evaluate_final(allocation, unverified_schedule, context)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must come from a CSP solver"):
        CspVerificationResult()


def test_verified_csp_result_allows_final_evaluation_and_preserves_schedule() -> None:
    _, adapter, context = setup_adapter()
    allocation = CandidateSessionAllocation((1, 1, 0, 0))
    schedule = ScheduleData(
        (ScheduledSession("slot-a", 1), ScheduledSession("slot-b", 2))
    )
    verification = verified_result(adapter, allocation, context, schedule)

    result = adapter.evaluate_final(allocation, verification, context)

    assert result.phase is EvaluationPhase.FINAL_SCHEDULED
    assert result.status is EvaluationStatus.PREDICTED
    assert result.mapping.weekly_sessions == (1, 1, 0, 0)
    assert result.schedule_identity == (("slot-a", 1), ("slot-b", 2))
    assert result.csp_verification_identity == "csp-run-1"
    assert result.usable_for_goal_test(
        result.identity, result.schedule_identity, result.csp_verification_identity
    )
    assert not result.usable_for_goal_test(
        result.identity, (("different-slot", 1),), result.csp_verification_identity
    )
    assert result.monotonicity_report is not None
    assert result.monotonicity_report.observations[0].schedule_identity == (
        ("slot-a", 1),
        ("slot-b", 2),
    )


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
        adapter.evaluate_final(allocation, verification, context)


def test_schedule_and_candidate_identity_mismatches_are_rejected() -> None:
    _, adapter, context = setup_adapter()
    allocation = CandidateSessionAllocation((1, 0, 0, 0))
    schedule = ScheduleData((ScheduledSession("slot-a", 1),))
    verification = verified_result(adapter, allocation, context, schedule)
    other_allocation = CandidateSessionAllocation((0, 1, 0, 0))

    with pytest.raises(CandidateEvaluationError, match="candidate identity mismatch"):
        adapter.evaluate_final(other_allocation, verification, context)

    other_context = CandidateFeatureContext("student-2", "assignment-1", strict_row())
    with pytest.raises(CandidateEvaluationError, match="candidate identity mismatch"):
        adapter.evaluate_final(allocation, verification, other_context)


def test_result_cannot_be_reused_for_a_different_candidate() -> None:
    _, adapter, context = setup_adapter()
    allocation = CandidateSessionAllocation((1, 0, 0, 0))
    schedule = ScheduleData((ScheduledSession("slot-a", 1),))
    result = adapter.evaluate_final(
        allocation, verified_result(adapter, allocation, context, schedule), context
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
        FakePredictor(), SessionClickMappingConfig(ClickScenario.HIGH)
    )

    with pytest.raises(CandidateEvaluationError, match="candidate identity mismatch"):
        other_adapter.evaluate_final(allocation, verification, context)


def test_matching_context_compares_increasing_allocations_across_schedules() -> None:
    _, adapter, context = setup_adapter()
    zero = CandidateSessionAllocation((0, 0, 0, 0))
    one = CandidateSessionAllocation((1, 0, 0, 0))
    zero_schedule = ScheduleData(())
    one_schedule = ScheduleData((ScheduledSession("slot-a", 1),))

    adapter.evaluate_final(
        zero, verified_result(adapter, zero, context, zero_schedule), context
    )
    result = adapter.evaluate_final(
        one, verified_result(adapter, one, context, one_schedule), context
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
    adapter.evaluate_final(
        allocation,
        verified_result(adapter, allocation, context, empty_schedule),
        context,
    )
    other_context = CandidateFeatureContext("student-2", "assignment-1", strict_row())
    next_allocation = CandidateSessionAllocation((1, 0, 0, 0))
    result = adapter.evaluate_final(
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
    with pytest.raises(CandidateEvaluationError, match="session count"):
        adapter.evaluate_final(
            allocation,
            verified_result(adapter, allocation, context, schedule),
            context,
        )
