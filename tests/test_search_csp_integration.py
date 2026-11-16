"""End-to-end Search, CSP schedule bridge, and NN integration tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

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
    build_required_constraints,
    expected_session_variables,
    verify_whole_plan,
)
from study_planner.csp.backtracking import (
    BacktrackingOptions,
    BacktrackingValueOrder,
    BacktrackingVariableOrder,
)
from study_planner.eval.baseline import SimpleAverageMarkPredictor
from study_planner.integration import (
    AssignmentTarget,
    CspAlgorithm,
    OutOfSupportAction,
    PlannerOutcome,
    PlannerStatus,
    SupportPolicy,
    WholePlanEnumerationStatus,
    WholePlanPlanningCandidate,
    WholePlanScheduleEnumerator,
    integrated_greedy_best_first_search,
    integrated_uniform_cost_search,
)
from study_planner.nn.inference import artifact_digest_from_files
from study_planner.search.nn_adapter import (
    CandidateEvaluationError,
    CandidateFeatureContext,
    CandidateSessionAllocation,
    NNModelMode,
    ScheduleData,
    ScheduledSession,
    SessionNNAdapter,
)
from study_planner.search.session_mapping import (
    STRICT_FFF_FEATURE_NAMES,
    ClickScenario,
    MarginalSupportStatus,
    SessionClickMappingConfig,
)

_TYPE_COLUMNS = STRICT_FFF_FEATURE_NAMES[:3]
_METADATA = STRICT_FFF_FEATURE_NAMES[3:7]
_PRIOR_MARKS = tuple(
    name
    for lag in range(1, 6)
    for name in (
        f"previous_assessment_score_lag_{lag}",
        f"previous_assessment_score_lag_{lag}_available",
    )
)
_MODEL_B_FEATURES = (
    *_TYPE_COLUMNS,
    *_METADATA,
    *_PRIOR_MARKS,
    *STRICT_FFF_FEATURE_NAMES[7:],
)


def _row(feature_names: tuple[str, ...]) -> pd.DataFrame:
    values = dict.fromkeys(feature_names, 0.0)
    values["assessment_type_cma"] = 1.0
    values["assessment_weight"] = 10.0
    values["assessment_weight_available"] = 1.0
    values["assessment_deadline_day"] = 47.0
    values["weeks_since_course_start"] = 47 / 7
    for lag in range(1, 6):
        values[f"previous_assessment_score_lag_{lag}"] = np.nan
        values[f"previous_assessment_score_lag_{lag}_available"] = 0.0
    return pd.DataFrame([values], columns=feature_names)


class _Predictor:
    assessment_type_columns = _TYPE_COLUMNS

    def __init__(
        self,
        *,
        model_b: bool = False,
        score_by_window: bool = False,
        scores_by_weight: dict[float, float] | None = None,
    ) -> None:
        self.feature_names = _MODEL_B_FEATURES if model_b else STRICT_FFF_FEATURE_NAMES
        self.calls = 0
        self.score_by_window = score_by_window
        self.scores_by_weight = scores_by_weight
        mode = "proxy" if model_b else "strict"
        self.artifact_digest = artifact_digest_from_files(
            {
                "feature_schema.json": repr((mode, self.feature_names)).encode(),
                "experiment_config.json": f'{{"feature_mode":"{mode}"}}'.encode(),
                "preprocessor.joblib": b"integration-test-preprocessor",
                "mlp_seed_42.pt": b"integration-test-fitted-weights",
            }
        )

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self.calls += 1
        if self.score_by_window:
            week_two = features["click_content_navigation_week_2"].iloc[0]
            return np.asarray([80.0 if week_two > 0 else 60.0])
        if self.scores_by_weight is not None:
            weight = float(features["assessment_weight"].iloc[0])
            return np.asarray([self.scores_by_weight[weight]])
        return np.asarray([60.0])


@pytest.fixture(autouse=True)
def _isolated_integration_predictor_trust(monkeypatch):
    """Issue fixture-local capabilities only to this module's exact fake type."""
    real_lookup = nn_adapter_module.trusted_artifact_provenance
    real_verify = nn_adapter_module.verify_trusted_artifact_integrity
    capabilities: dict[int, object] = {}

    def lookup(predictor):
        if type(predictor) is not _Predictor:
            return real_lookup(predictor)
        key = id(predictor)
        capabilities.setdefault(
            key,
            SimpleNamespace(
                digest=predictor.artifact_digest,
                mode=(
                    "proxy"
                    if predictor.feature_names == _MODEL_B_FEATURES
                    else "strict"
                ),
                state=(predictor.feature_names, predictor.artifact_digest),
            ),
        )
        return capabilities[key]

    def verify(predictor, expected=None):
        if type(predictor) is not _Predictor:
            return real_verify(predictor, expected)
        capability = lookup(predictor)
        if expected is not None and expected is not capability:
            raise ValueError("test predictor capability mismatch")
        if (predictor.feature_names, predictor.artifact_digest) != capability.state:
            raise ValueError("controlled test predictor state was changed")
        return capability

    monkeypatch.setattr(nn_adapter_module, "trusted_artifact_provenance", lookup)
    monkeypatch.setattr(
        nn_adapter_module, "verify_trusted_artifact_integrity", verify
    )


@dataclass(frozen=True)
class _IncreasingWeeks:
    policy_identity: str = "integration-test-order-v1"

    def required_pairs(self, sessions: tuple[SessionVariable, ...]):
        by_assignment: dict[str, list[SessionVariable]] = {}
        for session in sessions:
            by_assignment.setdefault(session.assignment_id, []).append(session)
        return tuple(
            SessionPair(first, second)
            for group in by_assignment.values()
            for first, second in zip(group, group[1:], strict=False)
        )

    def is_ordered(self, earlier: WeeklySlot, later: WeeklySlot) -> bool:
        return earlier.week_index < later.week_index


class _AllocationSpace:
    def __init__(self, initial: str, edges: dict[str, tuple[str, ...]]) -> None:
        self.initial_state = initial
        self.edges = edges

    def actions(self, state: str) -> tuple[str, ...]:
        return self.edges.get(state, ())

    def transition(self, state: str, action: str) -> str:
        assert action in self.actions(state)
        return action

    def action_cost(self, state: str, action: str, next_state: str) -> float:
        del state, action, next_state
        return 1.0


def _candidate(
    adapter: SessionNNAdapter,
    state: str,
    *,
    count: int = 1,
    empty_domain: bool = False,
    include_second_slot: bool = True,
    target: float = 0.0,
    assignment_weights: tuple[float, float, float] | None = None,
) -> WholePlanPlanningCandidate:
    allocation = (count, 0, 0, 0)
    ids = ("a1", "a2", "a3")
    weights = assignment_weights or (10.0, 10.0, 10.0)
    contexts = []
    for assignment_id, weight in zip(ids, weights, strict=True):
        features = _row(adapter.predictor.feature_names)
        features.loc[0, "assessment_weight"] = weight
        contexts.append(CandidateFeatureContext("student-1", assignment_id, features))
    contexts = tuple(contexts)
    context_map = {context.assignment_identity: context for context in contexts}
    plans = tuple(
        AssignmentPlan(
            assignment_id,
            AssignmentCandidateIdentity(
                "student-1",
                "course-1",
                assignment_id,
                allocation if assignment_id == "a1" else (0, 0, 0, 0),
                context_map[assignment_id].base_feature_context_identity,
                adapter.mapping_scenario_identity,
            ),
            context_map[assignment_id].base_feature_context_identity,
        )
        for assignment_id in ids
    )
    manifest = WholePlanManifest(
        "student-1", "course-1", plans, adapter.mapping_scenario_identity
    )
    variables = expected_session_variables(manifest)
    slots = (WeeklySlot("slot-week-1", 0),)
    if include_second_slot:
        slots += (WeeklySlot("slot-week-2", 1),)
    domain_values = () if empty_domain else slots
    domains = tuple(VariableDomain(variable, domain_values) for variable in variables)
    eligible = tuple((variable, frozenset(slots)) for variable in variables)
    ordering = _IncreasingWeeks()
    constraints = build_required_constraints(
        manifest=manifest,
        variables=variables,
        eligible_slots=eligible,
        ordering_policy=ordering,
        weekly_capacity=max(1, count),
    )
    problem = CspProblem(
        manifest,
        variables,
        domains,
        constraints,
        max(1, count),
        ordering,
    )
    policy = WholePlanVerificationPolicy(
        "integration-policy-v1",
        "deadline-eligibility-v1",
        problem.ordering_policy.configuration_identity,
        "test-slot-catalog-v1",
        "capacity-source-v1",
        "slot-id-week-v1",
        problem.weekly_capacity,
    )
    targets = tuple(
        AssignmentTarget(assignment_id, target if assignment_id == "a1" else 0)
        for assignment_id in ids
    )
    window_map = tuple(
        (slot.slot_id, slot.week_index + 1)
        for slot in (domain_values if variables else ())
    )
    return WholePlanPlanningCandidate(problem, policy, contexts, targets, window_map)


def _adapter(
    *,
    model_b: bool = False,
    score_by_window: bool = False,
    scores_by_weight: dict[float, float] | None = None,
) -> SessionNNAdapter:
    return SessionNNAdapter(
        _Predictor(
            model_b=model_b,
            score_by_window=score_by_window,
            scores_by_weight=scores_by_weight,
        ),
        SessionClickMappingConfig(ClickScenario.LOW),
    )


def test_bridge_enumerates_distinct_schedules_for_all_solver_configurations() -> None:
    for algorithm in CspAlgorithm:
        adapter = _adapter()
        candidate = _candidate(adapter, "one", count=1)
        bridge = WholePlanScheduleEnumerator(
            candidate.problem,
            candidate.verification_policy,
            algorithm=algorithm,
        )
        identities = []
        while True:
            step = bridge.next_schedule()
            if step.status.value == "schedule":
                assert step.verification is not None
                identities.append(step.verification.schedule_identity)
                assert len(step.verification.schedule) == 3
                assert step.verification.schedule[1].sessions == ()
                continue
            assert step.outcome is not None
            assert (
                step.outcome.status
                is WholePlanEnumerationStatus.EXHAUSTED_WITH_FEASIBLE
            )
            break
        assert len(identities) == 2
        assert len(set(identities)) == 2


@pytest.mark.parametrize("model_b", [False, True])
def test_successful_planning_preserves_model_schema_and_zero_allocations(
    model_b: bool,
) -> None:
    adapter = _adapter(model_b=model_b)
    candidate = _candidate(adapter, "zero", count=0)
    space = _AllocationSpace("zero", {})

    result = integrated_uniform_cost_search(
        space, lambda _state: candidate, adapter, csp_algorithm=CspAlgorithm.BT_FC
    )
    repeated = integrated_uniform_cost_search(
        space, lambda _state: candidate, adapter, csp_algorithm=CspAlgorithm.BT_FC
    )

    assert result.status is PlannerStatus.FEASIBLE
    assert result.outcome is PlannerOutcome.TARGET_ACHIEVED
    assert not result.search_complete
    assert result.nn_model_mode is (
        NNModelMode.MODEL_B_PREVIOUS_MARKS if model_b else NNModelMode.MODEL_A_STRICT
    )
    assert result.nn_model_identity == adapter.model_provenance_identity
    assert result.nn_fitted_artifact_digest == adapter.fitted_artifact_digest
    assert result.nn_feature_schema_identity == adapter.feature_schema_identity
    assert len(result.allocations) == 3
    assert result.allocations[1].weekly_sessions == (0, 0, 0, 0)
    assert result.weekly_timetable is not None
    assert len(result.weekly_timetable) == 3
    assert result.weekly_timetable[1].sessions == ()
    assert len(result.predicted_marks) == 3
    assert all(
        mark.model_provenance_identity == adapter.model_provenance_identity
        and mark.fitted_artifact_digest == adapter.fitted_artifact_digest
        and mark.prediction_identity
        for mark in result.predicted_marks
    )
    assert result.search_statistics.candidate_proposals == 1
    assert result.search_statistics.final_predictions == 3
    assert repeated.status is result.status
    assert repeated.allocations == result.allocations
    assert repeated.weekly_timetable == result.weekly_timetable
    assert repeated.predicted_marks == result.predicted_marks
    assert repeated.whole_plan_verification is not None
    assert result.whole_plan_verification is not None
    assert (
        repeated.whole_plan_verification.verification_identity
        == result.whole_plan_verification.verification_identity
    )
    assert (
        repeated.search_statistics.csp.nodes_visited
        == result.search_statistics.csp.nodes_visited
    )


def test_simple_average_baseline_runs_through_search_nn_csp_planner_path() -> None:
    predictor = SimpleAverageMarkPredictor.fit([70.0, 80.0])
    adapter = predictor.create_session_adapter(
        SessionClickMappingConfig(ClickScenario.LOW)
    )
    candidate = _candidate(adapter, "average-baseline", count=0)

    result = integrated_uniform_cost_search(
        _AllocationSpace("average-baseline", {}),
        lambda _state: candidate,
        adapter,
        csp_algorithm=CspAlgorithm.BT_FC,
    )

    assert result.status is PlannerStatus.FEASIBLE
    assert result.prediction_family == "simple_average_mark"
    assert result.nn_model_mode is NNModelMode.SIMPLE_AVERAGE_BASELINE
    assert [mark.predicted_mark for mark in result.predicted_marks] == [75.0] * 3
    assert all(
        mark.fitted_artifact_digest == predictor.artifact_digest
        for mark in result.predicted_marks
    )

    nn_adapter = _adapter()
    nn_candidate = _candidate(nn_adapter, "average-baseline", count=0)
    nn_result = integrated_uniform_cost_search(
        _AllocationSpace("average-baseline", {}),
        lambda _state: nn_candidate,
        nn_adapter,
        csp_algorithm=CspAlgorithm.BT_FC,
    )
    assert nn_result.status is PlannerStatus.FEASIBLE
    assert result.allocations == nn_result.allocations
    assert result.weekly_timetable == nn_result.weekly_timetable


def test_rejected_allocation_is_followed_by_a_valid_candidate() -> None:
    adapter = _adapter()
    bad = _candidate(adapter, "bad", empty_domain=True)
    good = _candidate(adapter, "good", include_second_slot=False)
    candidates = {"bad": bad, "good": good}
    result = integrated_uniform_cost_search(
        _AllocationSpace("bad", {"bad": ("good",)}),
        candidates.__getitem__,
        adapter,
    )

    assert result.status is PlannerStatus.FEASIBLE
    assert result.search_statistics.candidate_proposals == 2
    assert result.search_statistics.search.nodes_generated == 1
    assert result.search_statistics.schedules_examined == 1


def test_target_failure_tries_next_schedule_before_searching_next_allocation() -> None:
    adapter = _adapter(score_by_window=True)
    zero = _candidate(adapter, "zero", count=0, target=70)
    one = _candidate(adapter, "one", count=1, target=70)
    candidates = {"zero": zero, "one": one}
    result = integrated_uniform_cost_search(
        _AllocationSpace("zero", {"zero": ("one",)}),
        candidates.__getitem__,
        adapter,
        csp_options=BacktrackingOptions(
            variable_order=BacktrackingVariableOrder.CANONICAL,
            value_order=BacktrackingValueOrder.CANONICAL,
        ),
    )

    assert result.status is PlannerStatus.FEASIBLE
    assert result.outcome is PlannerOutcome.TARGET_ACHIEVED
    assert result.allocations[0].weekly_sessions == (1, 0, 0, 0)
    assert result.weekly_timetable is not None
    assert result.weekly_timetable[0].sessions[0].week_index == 1
    assert result.search_statistics.target_failures == 2
    assert result.search_statistics.schedules_examined == 3
    assert result.search_statistics.final_predictions == 9


def test_best_verified_plan_uses_later_candidate_with_higher_total_marks() -> None:
    adapter = _adapter(score_by_window=True)
    first = _candidate(adapter, "zero", count=0, target=90)
    later = _candidate(adapter, "one", count=1, target=90)
    candidates = {"zero": first, "one": later}

    result = integrated_uniform_cost_search(
        _AllocationSpace("zero", {"zero": ("one",)}),
        candidates.__getitem__,
        adapter,
        csp_options=BacktrackingOptions(
            variable_order=BacktrackingVariableOrder.CANONICAL,
            value_order=BacktrackingValueOrder.CANONICAL,
        ),
    )

    assert result.status is PlannerStatus.INFEASIBLE
    assert result.outcome is PlannerOutcome.TARGET_UNREACHABLE
    assert result.search_complete
    assert result.allocations[0].weekly_sessions == (1, 0, 0, 0)
    assert result.weekly_timetable is not None
    assert result.weekly_timetable[0].sessions[0].week_index == 1
    assert tuple(mark.predicted_mark for mark in result.predicted_marks) == (
        80.0,
        60.0,
        60.0,
    )
    assert tuple(target.target_mark for target in result.targets) == (90, 0, 0)
    assert "all_verified_schedules_missed_nn_target" in result.failure_reasons
    assert result.best_plan_comparison_rule
    assert all(
        mark.model_provenance_identity == adapter.model_provenance_identity
        and mark.fitted_artifact_digest == adapter.fitted_artifact_digest
        for mark in result.predicted_marks
    )


def test_best_plan_reports_different_final_marks_for_each_assignment() -> None:
    adapter = _adapter(scores_by_weight={10.0: 45.0, 20.0: 70.0, 30.0: 85.0})
    candidate = _candidate(
        adapter,
        "different-marks",
        count=0,
        target=55,
        assignment_weights=(10.0, 20.0, 30.0),
    )

    result = integrated_uniform_cost_search(
        _AllocationSpace("different-marks", {}), lambda _state: candidate, adapter
    )

    assert result.outcome is PlannerOutcome.TARGET_UNREACHABLE
    assert tuple(
        (mark.assignment_identity, mark.predicted_mark, mark.target_mark)
        for mark in result.predicted_marks
    ) == (("a1", 45.0, 55), ("a2", 70.0, 0), ("a3", 85.0, 0))
    assert result.weekly_timetable is not None
    assert result.whole_plan_verification is not None


def test_exhaustive_infeasibility_and_csp_resource_limit_are_distinct() -> None:
    adapter = _adapter()
    impossible = _candidate(adapter, "impossible", empty_domain=True)
    exhausted = integrated_uniform_cost_search(
        _AllocationSpace("impossible", {}), lambda _state: impossible, adapter
    )
    assert exhausted.status is PlannerStatus.INFEASIBLE
    assert exhausted.outcome is PlannerOutcome.NO_FEASIBLE_SCHEDULE
    assert exhausted.search_complete
    assert exhausted.search_statistics.schedules_examined == 0
    assert exhausted.search_statistics.csp_schedules_enumerated == 0
    assert exhausted.search_statistics.verification_feasible_schedules == 0
    assert exhausted.predicted_marks == ()
    assert exhausted.allocations == ()
    assert exhausted.weekly_timetable is None
    assert len(exhausted.targets) == 3
    assert exhausted.failure_reasons == ("csp_exhausted_without_feasible_schedule",)
    assert len(exhausted.search_statistics.terminal_csp_outcomes) == 1
    assert (
        exhausted.search_statistics.terminal_csp_outcomes[0].status
        is WholePlanEnumerationStatus.EXHAUSTED_NO_FEASIBLE
    )

    limited = _candidate(adapter, "limited", count=1)
    incomplete = integrated_uniform_cost_search(
        _AllocationSpace("limited", {}),
        lambda _state: limited,
        adapter,
        csp_options=BacktrackingOptions(max_nodes=0),
    )
    assert incomplete.status is PlannerStatus.INCOMPLETE
    assert incomplete.outcome is PlannerOutcome.INCOMPLETE
    assert not incomplete.search_complete
    assert "csp_enumeration_incomplete" in incomplete.failure_reasons
    assert incomplete.search_statistics.terminal_csp_outcomes[0].status.value == (
        "incomplete"
    )

    search_limited = integrated_uniform_cost_search(
        _AllocationSpace("limited-first", {"limited-first": ("limited-next",)}),
        lambda _state: _candidate(adapter, "limited", target=90),
        adapter,
        max_candidates=1,
    )
    assert search_limited.outcome is PlannerOutcome.INCOMPLETE
    assert not search_limited.search_complete
    assert search_limited.predicted_marks
    assert "candidate_limit" in search_limited.failure_reasons


def test_support_rejected_candidate_precedes_another_candidates_csp_infeasibility():
    adapter = _adapter()
    candidate_a = _candidate(adapter, "candidate-a", count=1)
    candidate_b = _candidate(adapter, "candidate-b", count=1, empty_domain=True)

    class RejectProvisionalSupport:
        def permits(self, _status: MarginalSupportStatus) -> bool:
            return False

    candidate_a = replace(candidate_a, support_policy=RejectProvisionalSupport())
    candidates = {"candidate-a": candidate_a, "candidate-b": candidate_b}
    result = integrated_uniform_cost_search(
        _AllocationSpace("candidate-a", {"candidate-a": ("candidate-b",)}),
        candidates.__getitem__,
        adapter,
    )

    assert result.search_complete
    assert result.search_statistics.candidate_proposals == 2
    assert result.search_statistics.provisional_support_rejections == 1
    assert result.search_statistics.csp_schedules_enumerated == 0
    assert len(result.search_statistics.terminal_csp_outcomes) == 1
    assert result.search_statistics.terminal_csp_outcomes[0].status is (
        WholePlanEnumerationStatus.EXHAUSTED_NO_FEASIBLE
    )
    assert "provisional_mapping_out_of_support" in result.failure_reasons
    assert "csp_exhausted_without_feasible_schedule" in result.failure_reasons
    # Any support rejection with no supported final score makes the search-wide
    # outcome indeterminate as to supported-plan availability.
    assert result.outcome is PlannerOutcome.NO_SUPPORTED_PLAN


def test_operational_failure_remains_error_even_after_support_rejection(monkeypatch):
    adapter = _adapter()
    candidate = replace(
        _candidate(adapter, "error-after-support", count=1),
        support_policy=SupportPolicy(
            "test-support-policy",
            "1",
            "test-reference",
            OutOfSupportAction.REJECT,
        ),
    )

    def fail_provisional(*_args, **_kwargs):
        raise RuntimeError("synthetic inference failure")

    monkeypatch.setattr(SessionNNAdapter, "evaluate_provisional", fail_provisional)
    result = integrated_uniform_cost_search(
        _AllocationSpace("error-after-support", {}),
        lambda _state: candidate,
        adapter,
    )

    assert result.status is PlannerStatus.ERROR
    assert result.outcome is PlannerOutcome.ERROR
    assert not result.search_complete
    assert "operational_error:RuntimeError:synthetic inference failure" in (
        result.failure_reasons
    )


def test_all_verified_schedules_missing_target_has_specific_reason() -> None:
    adapter = _adapter()
    candidate = _candidate(adapter, "one", count=1, target=90)
    result = integrated_uniform_cost_search(
        _AllocationSpace("one", {}), lambda _state: candidate, adapter
    )

    assert result.status is PlannerStatus.INFEASIBLE
    assert result.outcome is PlannerOutcome.TARGET_UNREACHABLE
    assert result.search_complete
    assert tuple(mark.predicted_mark for mark in result.predicted_marks) == (
        60.0,
        60.0,
        60.0,
    )
    assert result.search_statistics.schedules_examined == 2
    assert result.search_statistics.target_failures == 2
    assert result.search_statistics.verification_feasible_schedules == 2
    assert result.search_statistics.final_scored_supported_schedules == 2
    assert result.failure_reasons == (
        "all_verified_schedules_missed_nn_target",
        "final_target_failed",
    )
    assert (
        result.search_statistics.terminal_csp_outcomes[0].status
        is WholePlanEnumerationStatus.EXHAUSTED_WITH_FEASIBLE
    )


def test_all_schedules_rejected_by_verification_has_specific_reason(
    monkeypatch,
) -> None:
    import study_planner.integration.schedule_bridge as bridge_module

    adapter = _adapter()
    candidate = _candidate(adapter, "one", count=1)
    rejected = verify_whole_plan(
        candidate.problem,
        CspAssignment(
            (AssignmentEntry(candidate.problem.variables[0], WeeklySlot("outside", 8)),)
        ),
        candidate.verification_policy,
        expected_whole_plan_identity=candidate.problem.whole_plan_identity,
        expected_problem_identity=candidate.problem.problem_identity,
    )
    monkeypatch.setattr(
        bridge_module, "verify_whole_plan", lambda *_args, **_kwargs: rejected
    )

    result = integrated_uniform_cost_search(
        _AllocationSpace("one", {}), lambda _state: candidate, adapter
    )

    assert result.status is PlannerStatus.INFEASIBLE
    assert result.search_statistics.schedules_examined == 2
    assert result.search_statistics.verification_rejections == 2
    assert result.search_statistics.verification_feasible_schedules == 0
    assert result.search_statistics.support_policy_rejections == 0
    assert result.outcome is PlannerOutcome.NO_FEASIBLE_SCHEDULE
    assert result.failure_reasons == (
        "all_schedules_rejected_by_verification",
        "assignment_contains_out_of_domain_values",
    )
    assert (
        result.search_statistics.terminal_csp_outcomes[0].status
        is WholePlanEnumerationStatus.EXHAUSTED_NO_FEASIBLE
    )


def test_verification_feasible_schedules_rejected_by_support_are_not_csp_infeasible(
    monkeypatch,
) -> None:
    adapter = _adapter()
    candidate = _candidate(adapter, "support-rejected", count=1)
    policy = SupportPolicy(
        "test-support-policy",
        "1",
        "test-reference",
        OutOfSupportAction.REJECT,
    )
    candidate = replace(candidate, support_policy=policy)
    original_evaluate_final = SessionNNAdapter.evaluate_final

    def out_of_support_final(self, *args, **kwargs):
        prediction = original_evaluate_final(self, *args, **kwargs)
        if self is adapter:
            prediction = replace(
                prediction,
                mapping=replace(
                    prediction.mapping,
                    marginal_status=MarginalSupportStatus.OUT_OF_RANGE,
                ),
            )
        return prediction

    monkeypatch.setattr(SessionNNAdapter, "evaluate_final", out_of_support_final)
    result = integrated_uniform_cost_search(
        _AllocationSpace("support-rejected", {}), lambda _state: candidate, adapter
    )

    assert result.outcome is PlannerOutcome.NO_SUPPORTED_PLAN
    assert result.status is PlannerStatus.INFEASIBLE
    assert result.search_complete
    assert result.search_statistics.csp_schedules_enumerated == 2
    assert result.search_statistics.verification_feasible_schedules == 2
    assert result.search_statistics.verification_rejections == 0
    assert result.search_statistics.final_support_rejections == 2
    assert result.search_statistics.provisional_support_rejections == 0
    assert result.search_statistics.support_policy_rejections == 2
    assert result.search_statistics.final_scored_supported_schedules == 0
    assert result.predicted_marks == ()
    assert result.weekly_timetable is None
    assert result.search_statistics.terminal_csp_outcomes[0].status is (
        WholePlanEnumerationStatus.EXHAUSTED_WITH_FEASIBLE
    )
    assert "final_schedule_out_of_support" in result.failure_reasons


def test_target_failure_reasons_are_candidate_local(monkeypatch) -> None:
    from study_planner.integration.planner import _Controller

    adapter = _adapter()
    first = _candidate(adapter, "first", count=0, target=90)
    second = _candidate(adapter, "second", count=1)
    second = replace(
        second,
        support_policy=SupportPolicy(
            "test-support-policy",
            "1",
            "test-reference",
            OutOfSupportAction.REJECT,
        ),
    )
    original_evaluate_final = SessionNNAdapter.evaluate_final
    reject_second = False

    def out_of_support_final(self, *args, **kwargs):
        prediction = original_evaluate_final(self, *args, **kwargs)
        if self is adapter and reject_second:
            prediction = replace(
                prediction,
                mapping=replace(
                    prediction.mapping,
                    marginal_status=MarginalSupportStatus.OUT_OF_RANGE,
                ),
            )
        return prediction

    monkeypatch.setattr(SessionNNAdapter, "evaluate_final", out_of_support_final)
    controller = _Controller(
        _AllocationSpace("first", {}),
        lambda _state: first,
        adapter,
        CspAlgorithm.BT,
        None,
        None,
    )
    controller._evaluate(first)
    assert controller.target_failures > 0
    assert controller.failure_reasons.count(
        "all_verified_schedules_missed_nn_target"
    ) == 1

    controller.failure_reasons.clear()
    reject_second = True
    controller._evaluate(second)
    assert controller.target_failures > 0
    assert "final_schedule_out_of_support" in controller.failure_reasons
    assert "all_verified_schedules_missed_nn_target" not in controller.failure_reasons


def test_support_rejection_suppresses_all_schedules_missed_target_reason(
    monkeypatch,
) -> None:
    adapter = _adapter()
    candidate = _candidate(adapter, "mixed-support-and-target-failures", target=90)
    candidate = replace(
        candidate,
        support_policy=SupportPolicy(
            "test-support-policy",
            "1",
            "test-reference",
            OutOfSupportAction.REJECT,
        ),
    )
    original_evaluate_final = SessionNNAdapter.evaluate_final
    final_evaluations = 0

    def reject_first_schedule(self, *args, **kwargs):
        nonlocal final_evaluations
        prediction = original_evaluate_final(self, *args, **kwargs)
        final_evaluations += 1
        if self is adapter and final_evaluations <= 3:
            prediction = replace(
                prediction,
                mapping=replace(
                    prediction.mapping,
                    marginal_status=MarginalSupportStatus.OUT_OF_RANGE,
                ),
            )
        return prediction

    monkeypatch.setattr(SessionNNAdapter, "evaluate_final", reject_first_schedule)
    result = integrated_uniform_cost_search(
        _AllocationSpace("mixed-support-and-target-failures", {}),
        lambda _state: candidate,
        adapter,
    )

    assert result.search_complete
    assert result.search_statistics.verification_feasible_schedules == 2
    assert result.search_statistics.final_support_rejections == 1
    assert result.search_statistics.target_failures == 1
    assert result.search_statistics.final_scored_supported_schedules == 1
    assert "final_schedule_out_of_support" in result.failure_reasons
    assert "final_target_failed" in result.failure_reasons
    assert "all_verified_schedules_missed_nn_target" not in result.failure_reasons


def test_greedy_integration_and_candidate_limit() -> None:
    adapter = _adapter()
    candidate = _candidate(adapter, "state")
    greedy = integrated_greedy_best_first_search(
        _AllocationSpace("state", {}), lambda _state: candidate, adapter, lambda _s: 0
    )
    assert greedy.status is PlannerStatus.FEASIBLE

    limited = integrated_uniform_cost_search(
        _AllocationSpace("state", {"state": ("next",)}),
        lambda _state: _candidate(adapter, "limited", target=90),
        adapter,
        max_candidates=1,
    )
    assert limited.status is PlannerStatus.INCOMPLETE


def test_greedy_rejected_initial_candidate_then_valid_alternative() -> None:
    adapter = _adapter()
    rejected = _candidate(adapter, "initial", target=90)
    accepted = _candidate(adapter, "alternative", count=0, target=0)
    candidates = {"initial": rejected, "alternative": accepted}
    factory_calls: list[str] = []

    def factory(state: str) -> WholePlanPlanningCandidate:
        factory_calls.append(state)
        return candidates[state]

    result = integrated_greedy_best_first_search(
        _AllocationSpace("initial", {"initial": ("alternative",)}),
        factory,
        adapter,
        lambda _state: 0,
    )

    assert result.status is PlannerStatus.FEASIBLE
    assert result.allocations[0].weekly_sessions == (0, 0, 0, 0)
    assert factory_calls == ["initial", "alternative"]
    assert result.search_statistics.candidate_proposals == 2
    assert result.search_statistics.target_failures == 2


def test_greedy_candidate_limit_counts_initial_state_once() -> None:
    adapter = _adapter()
    initial = _candidate(adapter, "initial", target=90)
    factory_calls: list[str] = []

    def factory(state: str) -> WholePlanPlanningCandidate:
        factory_calls.append(state)
        return initial

    result = integrated_greedy_best_first_search(
        _AllocationSpace("initial", {"initial": ("alternative",)}),
        factory,
        adapter,
        lambda _state: 0,
        max_candidates=1,
    )

    assert result.status is PlannerStatus.INCOMPLETE
    assert factory_calls == ["initial"]
    assert result.search_statistics.candidate_proposals == 1
    assert result.search_statistics.search.nodes_generated == 1
    assert result.search_statistics.target_failures == 2
    assert "candidate_limit" in result.failure_reasons


def test_greedy_successful_initial_candidate_has_one_goal_evaluation() -> None:
    adapter = _adapter()
    candidate = _candidate(adapter, "initial", target=0)
    factory_calls: list[str] = []

    result = integrated_greedy_best_first_search(
        _AllocationSpace("initial", {}),
        lambda state: factory_calls.append(state) or candidate,
        adapter,
        lambda _state: 0,
    )

    assert result.status is PlannerStatus.FEASIBLE
    assert factory_calls == ["initial"]
    assert result.search_statistics.candidate_proposals == 1
    assert result.search_statistics.provisional_predictions == 3
    assert result.search_statistics.final_predictions == 3


def test_greedy_results_and_counters_are_deterministic() -> None:
    def run():
        adapter = _adapter()
        rejected = _candidate(adapter, "initial", target=90)
        accepted = _candidate(adapter, "alternative", count=0, target=0)
        candidates = {"initial": rejected, "alternative": accepted}
        return integrated_greedy_best_first_search(
            _AllocationSpace("initial", {"initial": ("alternative",)}),
            candidates.__getitem__,
            adapter,
            lambda state: 0 if state == "initial" else 1,
        )

    first = run()
    second = run()
    assert first.status is second.status is PlannerStatus.FEASIBLE
    assert first.allocations == second.allocations
    assert first.weekly_timetable == second.weekly_timetable
    assert first.predicted_marks == second.predicted_marks
    assert first.search_statistics.candidate_proposals == 2
    assert second.search_statistics.candidate_proposals == 2
    assert first.search_statistics.final_predictions == 9
    assert second.search_statistics.final_predictions == 9


def test_final_adapter_rejects_mismatched_or_nonfeasible_whole_plan_evidence() -> None:
    adapter = _adapter()
    candidate = _candidate(adapter, "one", count=1)
    partial = verify_whole_plan(
        candidate.problem,
        CspAssignment(),
        candidate.verification_policy,
        expected_whole_plan_identity=candidate.problem.whole_plan_identity,
        expected_problem_identity=candidate.problem.problem_identity,
    )
    context = candidate.feature_contexts[0]
    allocation_identity = candidate.problem.manifest.assignment("a1").candidate_identity
    allocation = CandidateSessionAllocation(
        allocation_identity.canonical_allocation_identity
    )
    with pytest.raises(CandidateEvaluationError, match="did not establish feasibility"):
        adapter.evaluate_final(
            allocation,
            partial,
            context,
            expected_whole_plan_identity=candidate.problem.whole_plan_identity,
            expected_problem_identity=candidate.problem.problem_identity,
            expected_policy_identity=candidate.verification_policy.policy_identity,
            expected_policy_version=candidate.verification_policy.version,
            expected_model_provenance_identity=adapter.model_provenance_identity,
            slot_relative_windows=dict(candidate.slot_relative_windows),
        )

    infeasible = verify_whole_plan(
        candidate.problem,
        CspAssignment(
            (AssignmentEntry(candidate.problem.variables[0], WeeklySlot("outside", 8)),)
        ),
        candidate.verification_policy,
        expected_whole_plan_identity=candidate.problem.whole_plan_identity,
        expected_problem_identity=candidate.problem.problem_identity,
    )
    assert infeasible.status.value == "infeasible"
    with pytest.raises(CandidateEvaluationError, match="did not establish feasibility"):
        adapter.evaluate_final(
            allocation,
            infeasible,
            context,
            expected_whole_plan_identity=candidate.problem.whole_plan_identity,
            expected_problem_identity=candidate.problem.problem_identity,
            expected_policy_identity=candidate.verification_policy.policy_identity,
            expected_policy_version=candidate.verification_policy.version,
            expected_model_provenance_identity=adapter.model_provenance_identity,
            slot_relative_windows=dict(candidate.slot_relative_windows),
        )
    bridge = WholePlanScheduleEnumerator(
        candidate.problem, candidate.verification_policy
    )
    verification = bridge.next_schedule().verification
    assert verification is not None
    with pytest.raises(
        CandidateEvaluationError, match="whole-plan verification identity mismatch"
    ):
        adapter.evaluate_final(
            allocation,
            verification,
            context,
            expected_whole_plan_identity="wrong-plan",
            expected_problem_identity=candidate.problem.problem_identity,
            expected_policy_identity=candidate.verification_policy.policy_identity,
            expected_policy_version=candidate.verification_policy.version,
            expected_model_provenance_identity=adapter.model_provenance_identity,
            slot_relative_windows=dict(candidate.slot_relative_windows),
        )
    with pytest.raises(CandidateEvaluationError, match="WholePlanVerificationResult"):
        adapter.evaluate_final(
            allocation,
            ScheduleData((ScheduledSession("slot-week-1", 1),)),  # type: ignore[arg-type]
            context,
            expected_whole_plan_identity=candidate.problem.whole_plan_identity,
            expected_problem_identity=candidate.problem.problem_identity,
            expected_policy_identity=candidate.verification_policy.policy_identity,
            expected_policy_version=candidate.verification_policy.version,
            expected_model_provenance_identity=adapter.model_provenance_identity,
            slot_relative_windows=dict(candidate.slot_relative_windows),
        )
