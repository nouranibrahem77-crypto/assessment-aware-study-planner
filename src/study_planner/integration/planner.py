"""Deterministic Search–NN–CSP planning orchestration."""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, replace
from enum import Enum
from time import perf_counter
from typing import Generic, Protocol, TypeVar

from study_planner.csp.ac3 import Ac3Options
from study_planner.csp.backtracking import BacktrackingOptions
from study_planner.csp.core import CspProblem
from study_planner.csp.forward_checking import ForwardCheckingOptions
from study_planner.csp.verification import (
    AssignmentSchedule,
    WholePlanVerificationPolicy,
    WholePlanVerificationResult,
)
from study_planner.integration.schedule_bridge import (
    CspAlgorithm,
    CspEnumerationStatistics,
    ScheduleBridgeStatus,
    WholePlanEnumerationStatus,
    WholePlanScheduleEnumerator,
    WholePlanSearchOutcome,
)
from study_planner.search.core import SearchStatistics
from study_planner.search.greedy import greedy_best_first_search
from study_planner.search.nn_adapter import (
    CandidateFeatureContext,
    CandidateSessionAllocation,
    EvaluationStatus,
    NNModelMode,
    SessionNNAdapter,
)
from study_planner.search.session_mapping import MarginalSupportStatus
from study_planner.search.ucs import uniform_cost_search


class PlannerStatus(str, Enum):
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    INCOMPLETE = "incomplete"
    ERROR = "error"


StateT = TypeVar("StateT", bound=Hashable)
ActionT = TypeVar("ActionT")


class AllocationSpace(Protocol[StateT, ActionT]):
    """Finite allocation graph without a separate goal predicate."""

    @property
    def initial_state(self) -> StateT: ...

    def actions(self, state: StateT) -> Iterable[ActionT]: ...

    def transition(self, state: StateT, action: ActionT) -> StateT: ...

    def action_cost(
        self, state: StateT, action: ActionT, next_state: StateT
    ) -> float: ...


class MarginalSupportPolicy(Protocol):
    """Structural contract for an explicit application's support disposition."""

    def permits(self, status: MarginalSupportStatus) -> bool: ...


@dataclass(frozen=True, slots=True)
class AssignmentTarget:
    assignment_identity: str
    target_mark: float

    def __post_init__(self) -> None:
        if not self.assignment_identity:
            raise ValueError("assignment target identity cannot be empty")
        if (
            isinstance(self.target_mark, bool)
            or not isinstance(self.target_mark, int | float)
            or not math.isfinite(float(self.target_mark))
            or not 0 <= self.target_mark <= 100
        ):
            raise ValueError("target mark must be finite and in [0, 100]")


@dataclass(frozen=True, slots=True)
class WholePlanPlanningCandidate:
    """All per-proposal inputs required to verify and score a whole plan."""

    problem: CspProblem
    verification_policy: WholePlanVerificationPolicy
    feature_contexts: tuple[CandidateFeatureContext, ...]
    targets: tuple[AssignmentTarget, ...]
    slot_relative_windows: tuple[tuple[str, int], ...]
    slot_relative_windows_by_assignment: tuple[
        tuple[str, tuple[tuple[str, int], ...]], ...
    ] = ()
    support_policy: MarginalSupportPolicy | None = None

    def __post_init__(self) -> None:
        manifest = self.problem.manifest
        assignments = tuple(
            item.assignment_identity for item in manifest.canonical_assignments
        )
        if tuple(
            sorted(item.assignment_identity for item in self.feature_contexts)
        ) != (assignments):
            raise ValueError("feature contexts must cover all three assignments")
        if tuple(sorted(item.assignment_identity for item in self.targets)) != (
            assignments
        ):
            raise ValueError("targets must cover all three assignments")
        contexts = {item.assignment_identity: item for item in self.feature_contexts}
        for plan in manifest.canonical_assignments:
            context = contexts[plan.assignment_identity]
            identity = plan.candidate_identity
            if (
                context.student_identity != manifest.student_identity
                or context.assignment_identity != plan.assignment_identity
                or context.base_feature_context_identity
                != identity.base_feature_context_identity
            ):
                raise ValueError(
                    "feature context does not match the whole-plan manifest"
                )
        slot_ids = {
            slot.slot_id for domain in self.problem.domains for slot in domain.values
        }
        if self.slot_relative_windows_by_assignment:
            mapping_ids = tuple(
                assignment_id
                for assignment_id, _ in self.slot_relative_windows_by_assignment
            )
            if len(set(mapping_ids)) != 3 or set(mapping_ids) != set(assignments):
                raise ValueError(
                    "assignment-relative windows must cover all three assignments"
                )
            if self.slot_relative_windows:
                raise ValueError(
                    "use either shared or assignment-relative windows, not both"
                )
            for assignment_id, entries in self.slot_relative_windows_by_assignment:
                windows = dict(entries)
                if (
                    len(windows) != len(entries)
                    or not slot_ids.issubset(windows)
                    or any(
                        isinstance(window, bool)
                        or not isinstance(window, int)
                        or window not in {1, 2, 3, 4}
                        for window in windows.values()
                    )
                ):
                    raise ValueError(f"invalid relative-window map for {assignment_id}")
        else:
            windows = dict(self.slot_relative_windows)
            if (
                len(windows) != len(self.slot_relative_windows)
                or set(windows) != slot_ids
            ):
                raise ValueError("relative-window map must cover the CSP slot catalog")
            if any(
                isinstance(window, bool)
                or not isinstance(window, int)
                or window not in {1, 2, 3, 4}
                for window in windows.values()
            ):
                raise ValueError("slot relative windows must be from one through four")


@dataclass(frozen=True, slots=True)
class AssignmentAllocation:
    assignment_identity: str
    weekly_sessions: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class AssignmentMark:
    assignment_identity: str
    predicted_mark: float
    target_mark: float
    prediction_identity: str
    model_provenance_identity: str
    fitted_artifact_digest: str


@dataclass(frozen=True, slots=True)
class PlannerStatistics:
    search: SearchStatistics
    candidate_proposals: int
    provisional_predictions: int
    final_predictions: int
    schedules_examined: int
    verification_rejections: int
    target_failures: int
    csp: CspEnumerationStatistics
    terminal_csp_outcomes: tuple[WholePlanSearchOutcome, ...]
    orchestration_time_seconds: float


@dataclass(frozen=True, slots=True)
class FinalPlannerResult:
    status: PlannerStatus
    allocations: tuple[AssignmentAllocation, ...]
    weekly_timetable: tuple[AssignmentSchedule, ...] | None
    predicted_marks: tuple[AssignmentMark, ...]
    nn_model_identity: str
    nn_model_mode: NNModelMode
    nn_fitted_artifact_digest: str
    nn_feature_schema_identity: str
    whole_plan_verification: WholePlanVerificationResult | None
    search_statistics: PlannerStatistics
    failure_reasons: tuple[str, ...]
    solution_cost: float | None


class _Controller(Generic[StateT, ActionT]):
    def __init__(
        self,
        allocation_space: AllocationSpace,
        candidate_factory: Callable[[StateT], WholePlanPlanningCandidate],
        adapter: SessionNNAdapter,
        algorithm: CspAlgorithm,
        csp_options: BacktrackingOptions | ForwardCheckingOptions | Ac3Options | None,
        max_candidates: int | None,
    ) -> None:
        if max_candidates is not None and (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or max_candidates < 0
        ):
            raise ValueError("max_candidates must be a non-negative integer")
        self.space = allocation_space
        self.candidate_factory = candidate_factory
        self.adapter = adapter
        self.algorithm = algorithm
        self.csp_options = csp_options
        self.max_candidates = max_candidates
        self.started = perf_counter()
        self.candidate_proposals = 0
        self.provisional_predictions = 0
        self.final_predictions = 0
        self.schedules_examined = 0
        self.verification_rejections = 0
        self.target_failures = 0
        self.csp_stats = CspEnumerationStatistics(0, 0, 0, 0, 0, 0, 0.0, 0.0)
        self.last_candidate: WholePlanPlanningCandidate | None = None
        self.last_scored: (
            tuple[
                WholePlanPlanningCandidate,
                WholePlanVerificationResult,
                tuple[AssignmentMark, ...],
            ]
            | None
        ) = None
        self.found: FinalPlannerResult | None = None
        self.failure_reasons: list[str] = []
        self.terminal_csp_outcomes: list[WholePlanSearchOutcome] = []
        self.incomplete = False
        self.error = False

    def is_goal(self, state: StateT) -> bool:
        if self.incomplete or self.error:
            return False
        if (
            self.max_candidates is not None
            and self.candidate_proposals >= self.max_candidates
        ):
            self.incomplete = True
            self.failure_reasons.append("candidate_limit")
            return False
        self.candidate_proposals += 1
        try:
            candidate = self.candidate_factory(state)
            self.last_candidate = candidate
            if self._evaluate(candidate):
                return True
        except Exception as error:
            self.error = True
            self.failure_reasons.append(
                f"operational_error:{type(error).__name__}:{error}"
            )
        return False

    def _evaluate(self, candidate: WholePlanPlanningCandidate) -> bool:
        if not isinstance(candidate, WholePlanPlanningCandidate):
            raise ValueError("candidate factory returned an invalid plan candidate")
        if (
            candidate.problem.whole_plan_identity
            != candidate.problem.manifest.canonical_identity
        ):
            raise ValueError("candidate problem and manifest identities differ")
        manifest = candidate.problem.manifest
        contexts = {
            item.assignment_identity: item for item in candidate.feature_contexts
        }
        targets = {
            item.assignment_identity: item.target_mark for item in candidate.targets
        }
        for plan in manifest.canonical_assignments:
            provisional = self.adapter.evaluate_provisional(
                CandidateSessionAllocation(
                    plan.candidate_identity.canonical_allocation_identity
                ),
                contexts[plan.assignment_identity],
            )
            self.provisional_predictions += 1
            if provisional.status is EvaluationStatus.INFERENCE_FAILED:
                raise RuntimeError("provisional NN inference failed")
            if (
                candidate.support_policy is not None
                and not candidate.support_policy.permits(
                    provisional.mapping.marginal_status
                )
            ):
                self.failure_reasons.append("provisional_mapping_out_of_support")
                return False

        bridge = WholePlanScheduleEnumerator(
            candidate.problem,
            candidate.verification_policy,
            algorithm=self.algorithm,
            options=self.csp_options,
        )
        try:
            while True:
                step = bridge.next_schedule()
                if step.status is ScheduleBridgeStatus.SCHEDULE:
                    assert step.verification is not None
                    self.schedules_examined += 1
                    verification = step.verification
                    if verification.status.value != "feasible":
                        self.verification_rejections += 1
                        self.failure_reasons.extend(verification.rejection_reasons)
                        continue
                    marks: list[AssignmentMark] = []
                    predictions = []
                    windows_by_assignment = dict(
                        (identity, dict(entries))
                        for identity, entries in (
                            candidate.slot_relative_windows_by_assignment
                        )
                    )
                    shared_windows = dict(candidate.slot_relative_windows)
                    for plan in manifest.canonical_assignments:
                        windows = windows_by_assignment.get(
                            plan.assignment_identity, shared_windows
                        )
                        prediction = self.adapter.evaluate_final(
                            CandidateSessionAllocation(
                                plan.candidate_identity.canonical_allocation_identity
                            ),
                            verification,
                            contexts[plan.assignment_identity],
                            expected_whole_plan_identity=manifest.canonical_identity,
                            expected_problem_identity=candidate.problem.problem_identity,
                            expected_policy_identity=candidate.verification_policy.policy_identity,
                            expected_policy_version=candidate.verification_policy.version,
                            expected_model_provenance_identity=self.adapter.model_provenance_identity,
                            slot_relative_windows=windows,
                        )
                        self.final_predictions += 1
                        if prediction.status is EvaluationStatus.INFERENCE_FAILED:
                            raise RuntimeError("final NN inference failed")
                        assert prediction.prediction is not None
                        if (
                            prediction.prediction_identity is None
                            or prediction.model_provenance_identity
                            != self.adapter.model_provenance_identity
                            or prediction.fitted_artifact_digest
                            != self.adapter.fitted_artifact_digest
                        ):
                            raise RuntimeError(
                                "final NN model provenance is incomplete"
                            )
                        predictions.append(prediction)
                        marks.append(
                            AssignmentMark(
                                plan.assignment_identity,
                                prediction.prediction,
                                targets[plan.assignment_identity],
                                prediction.prediction_identity,
                                prediction.model_provenance_identity,
                                prediction.fitted_artifact_digest,
                            )
                        )
                    if candidate.support_policy is not None and not all(
                        candidate.support_policy.permits(
                            prediction.mapping.marginal_status
                        )
                        for prediction in predictions
                    ):
                        self.failure_reasons.append("final_schedule_out_of_support")
                        continue
                    marks_tuple = tuple(marks)
                    self.last_scored = (candidate, verification, marks_tuple)
                    if all(
                        mark.predicted_mark >= mark.target_mark for mark in marks_tuple
                    ):
                        if not all(
                            result.usable_for_goal_test(
                                result.identity,
                                result.schedule_identity,
                                verification.verification_identity,
                                manifest.canonical_identity,
                                candidate.problem.problem_identity,
                                model_provenance_identity=self.adapter.model_provenance_identity,
                                fitted_artifact_digest=self.adapter.fitted_artifact_digest,
                                model_mode=self.adapter.model_mode,
                                feature_schema_identity=self.adapter.feature_schema_identity,
                            )
                            for result in predictions
                        ):
                            raise RuntimeError(
                                "final prediction provenance is incomplete"
                            )
                        self.found = self._result(
                            PlannerStatus.FEASIBLE,
                            candidate,
                            verification,
                            marks_tuple,
                            None,
                        )
                        return True
                    self.target_failures += 1
                    self.failure_reasons.append("final_target_failed")
                    continue
                if step.status is ScheduleBridgeStatus.INCOMPLETE:
                    self.incomplete = True
                    assert step.outcome is not None
                    self.terminal_csp_outcomes.append(step.outcome)
                    self.failure_reasons.append("csp_enumeration_incomplete")
                    break
                assert step.outcome is not None
                self.terminal_csp_outcomes.append(step.outcome)
                if (
                    step.outcome.status
                    is WholePlanEnumerationStatus.EXHAUSTED_NO_FEASIBLE
                ):
                    self.failure_reasons.append(
                        "all_schedules_rejected_by_verification"
                        if step.outcome.rejected_schedules
                        else "csp_exhausted_without_feasible_schedule"
                    )
                elif (
                    step.outcome.status
                    is WholePlanEnumerationStatus.EXHAUSTED_WITH_FEASIBLE
                    and self.target_failures
                ):
                    self.failure_reasons.append(
                        "all_verified_schedules_missed_nn_target"
                    )
                break
        finally:
            self._add_csp_statistics(bridge.statistics)
        return False

    def _add_csp_statistics(self, value: CspEnumerationStatistics) -> None:
        current = self.csp_stats
        self.csp_stats = CspEnumerationStatistics(
            current.nodes_visited + value.nodes_visited,
            current.assignments_attempted + value.assignments_attempted,
            current.backtracks + value.backtracks,
            current.constraint_checks + value.constraint_checks,
            current.domain_prunings + value.domain_prunings,
            current.compatibility_checks + value.compatibility_checks,
            current.solver_time_seconds + value.solver_time_seconds,
            current.wall_time_seconds + value.wall_time_seconds,
        )

    def _result(
        self,
        status: PlannerStatus,
        candidate: WholePlanPlanningCandidate | None,
        verification: WholePlanVerificationResult | None,
        marks: tuple[AssignmentMark, ...],
        search: SearchStatistics | None,
    ) -> FinalPlannerResult:
        if not self.adapter.fitted_artifact_digest:
            raise RuntimeError("final planner provenance requires a fitted artifact")
        if candidate is None:
            candidate = self.last_candidate
        if verification is None and self.last_scored is not None:
            scored_candidate, scored_verification, scored_marks = self.last_scored
            candidate = scored_candidate
            verification = scored_verification
            if not marks:
                marks = scored_marks
        allocations = (
            ()
            if candidate is None
            else tuple(
                AssignmentAllocation(
                    plan.assignment_identity,
                    plan.candidate_identity.canonical_allocation_identity,
                )
                for plan in candidate.problem.manifest.canonical_assignments
            )
        )
        return FinalPlannerResult(
            status,
            allocations,
            None if verification is None else verification.schedule,
            marks,
            self.adapter.model_provenance_identity,
            self.adapter.model_mode,
            self.adapter.fitted_artifact_digest,
            self.adapter.feature_schema_identity,
            verification,
            PlannerStatistics(
                search or SearchStatistics(0, 0, 0.0),
                self.candidate_proposals,
                self.provisional_predictions,
                self.final_predictions,
                self.schedules_examined,
                self.verification_rejections,
                self.target_failures,
                self.csp_stats,
                tuple(self.terminal_csp_outcomes),
                perf_counter() - self.started,
            ),
            tuple(sorted(set(self.failure_reasons))),
            None,
        )


class _IntegrationSearchProblem(Generic[StateT, ActionT]):
    def __init__(self, controller: _Controller[StateT, ActionT]) -> None:
        self._controller = controller
        self._space = controller.space
        self._goal_results: dict[StateT, bool] = {}

    @property
    def initial_state(self) -> StateT:
        return self._space.initial_state

    def is_goal(self, state: StateT) -> bool:
        candidate_state = getattr(self._space, "is_candidate_state", None)
        if callable(candidate_state) and not candidate_state(state):
            return False
        if state not in self._goal_results:
            self._goal_results[state] = self._controller.is_goal(state)
        return self._goal_results[state]

    def actions(self, state: StateT) -> Iterable[ActionT]:
        if (
            self._controller.incomplete
            or self._controller.error
            or self._controller.found
        ):
            return ()
        return self._space.actions(state)

    def transition(self, state: StateT, action: ActionT) -> StateT:
        return self._space.transition(state, action)

    def action_cost(self, state: StateT, action: ActionT, next_state: StateT) -> float:
        return self._space.action_cost(state, action, next_state)


def _run(
    allocation_space: AllocationSpace,
    candidate_factory: Callable[[StateT], WholePlanPlanningCandidate],
    adapter: SessionNNAdapter,
    *,
    greedy_heuristic: Callable[[StateT], float] | None,
    csp_algorithm: CspAlgorithm,
    csp_options: BacktrackingOptions | ForwardCheckingOptions | Ac3Options | None,
    max_candidates: int | None,
) -> FinalPlannerResult:
    controller: _Controller = _Controller(
        allocation_space,
        candidate_factory,
        adapter,
        csp_algorithm,
        csp_options,
        max_candidates,
    )
    wrapped = _IntegrationSearchProblem(controller)
    search_result = (
        uniform_cost_search(wrapped)
        if greedy_heuristic is None
        else greedy_best_first_search(wrapped, greedy_heuristic)
    )
    if controller.found is not None:
        return replace(
            controller.found,
            search_statistics=PlannerStatistics(
                search_result.statistics,
                controller.candidate_proposals,
                controller.provisional_predictions,
                controller.final_predictions,
                controller.schedules_examined,
                controller.verification_rejections,
                controller.target_failures,
                controller.csp_stats,
                tuple(controller.terminal_csp_outcomes),
                perf_counter() - controller.started,
            ),
            solution_cost=search_result.solution_cost,
        )
    status = (
        PlannerStatus.ERROR
        if controller.error
        else PlannerStatus.INCOMPLETE
        if controller.incomplete
        else PlannerStatus.INFEASIBLE
    )
    return controller._result(status, None, None, (), search_result.statistics)


def integrated_uniform_cost_search(
    allocation_space: AllocationSpace,
    candidate_factory: Callable[[StateT], WholePlanPlanningCandidate],
    adapter: SessionNNAdapter,
    *,
    csp_algorithm: CspAlgorithm = CspAlgorithm.BT,
    csp_options: BacktrackingOptions
    | ForwardCheckingOptions
    | Ac3Options
    | None = None,
    max_candidates: int | None = None,
) -> FinalPlannerResult:
    """Search allocations by uniform cost and verify all schedules lazily."""
    return _run(
        allocation_space,
        candidate_factory,
        adapter,
        greedy_heuristic=None,
        csp_algorithm=csp_algorithm,
        csp_options=csp_options,
        max_candidates=max_candidates,
    )


def integrated_greedy_best_first_search(
    allocation_space: AllocationSpace,
    candidate_factory: Callable[[StateT], WholePlanPlanningCandidate],
    adapter: SessionNNAdapter,
    heuristic: Callable[[StateT], float],
    *,
    csp_algorithm: CspAlgorithm = CspAlgorithm.BT,
    csp_options: BacktrackingOptions
    | ForwardCheckingOptions
    | Ac3Options
    | None = None,
    max_candidates: int | None = None,
) -> FinalPlannerResult:
    """Search allocations by caller-defined deterministic Greedy priority."""
    return _run(
        allocation_space,
        candidate_factory,
        adapter,
        greedy_heuristic=heuristic,
        csp_algorithm=csp_algorithm,
        csp_options=csp_options,
        max_candidates=max_candidates,
    )
