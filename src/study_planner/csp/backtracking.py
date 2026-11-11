"""Baseline recursive Backtracking Search for immutable CSP problems."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from time import perf_counter

from study_planner.csp.core import (
    AssignmentEntry,
    CspAssignment,
    CspEvaluation,
    CspEvaluationStatus,
    CspInputError,
    CspProblem,
    CspSolverOutcome,
    CspSolverStatus,
    SessionVariable,
    WeeklySlot,
    evaluate_assignment,
)


class BacktrackingVariableOrder(str, Enum):
    """Deterministic variable-order policies supported by baseline BT."""

    MRV_DEGREE = "mrv_degree"
    CANONICAL = "canonical"


class BacktrackingValueOrder(str, Enum):
    """Deterministic value-order policies supported by baseline BT."""

    LEAST_CONSTRAINING = "least_constraining"
    CANONICAL = "canonical"


class BacktrackingTerminationReason(str, Enum):
    """Reason a first-solution search or enumeration step stopped."""

    SOLUTION_FOUND = "solution_found"
    EXHAUSTED = "exhausted"
    NODE_LIMIT = "node_limit"


class BacktrackingEnumerationStatus(str, Enum):
    """Result of requesting one item from a lazy solution enumerator."""

    SOLUTION_FOUND = "solution_found"
    EXHAUSTED = "exhausted"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class BacktrackingOptions:
    """Explicit deterministic ordering and optional per-enumerator node limit."""

    max_nodes: int | None = None
    variable_order: BacktrackingVariableOrder = BacktrackingVariableOrder.MRV_DEGREE
    value_order: BacktrackingValueOrder = BacktrackingValueOrder.LEAST_CONSTRAINING

    def __post_init__(self) -> None:
        if self.max_nodes is not None and (
            isinstance(self.max_nodes, bool)
            or not isinstance(self.max_nodes, int)
            or self.max_nodes < 0
        ):
            raise CspInputError("max_nodes must be a non-negative integer or None")
        if not isinstance(self.variable_order, BacktrackingVariableOrder):
            raise CspInputError("variable_order is not a supported policy")
        if not isinstance(self.value_order, BacktrackingValueOrder):
            raise CspInputError("value_order is not a supported policy")


@dataclass(frozen=True, slots=True)
class BacktrackingStatistics:
    """Cumulative work counters and elapsed time for one BT invocation.

    ``nodes_visited`` counts entered partial-assignment states, including root
    and rejected states. ``assignments_attempted`` counts each variable/slot
    extension before recursion; it can exceed visited child nodes when a node
    limit prevents entry. ``backtracks`` counts attempted extensions whose
    subtrees were fully explored and yielded no solution. ``constraint_checks``
    counts every call to a constraint's ``is_satisfied`` predicate, including
    node validation, dynamic-MRV probes, LCV probes, and final solver-outcome
    verification. ``maximum_depth`` is the largest number of assigned
    variables in an entered state. ``solver_time_seconds`` accumulates active
    time spent advancing the solver and performing final solver-boundary
    validation, excluding caller pauses between lazy requests.
    ``wall_time_seconds`` is elapsed time since enumerator creation and can
    include caller work and pauses.
    """

    nodes_visited: int
    assignments_attempted: int
    backtracks: int
    constraint_checks: int
    maximum_depth: int
    solver_time_seconds: float
    wall_time_seconds: float


@dataclass(frozen=True, slots=True)
class BacktrackingResult:
    """First-solution CSP outcome with BT-specific statistics and stop reason."""

    outcome: CspSolverOutcome
    statistics: BacktrackingStatistics
    termination_reason: BacktrackingTerminationReason

    @property
    def status(self) -> CspSolverStatus:
        """Expose the shared CSP solver status directly."""
        return self.outcome.status

    @property
    def evaluation(self) -> CspEvaluation | None:
        """Return the complete feasible evaluation, if one was found."""
        return self.outcome.evaluation


@dataclass(frozen=True, slots=True)
class BacktrackingEnumerationResult:
    """One schedule, or an explicit exhausted/incomplete enumeration state."""

    status: BacktrackingEnumerationStatus
    evaluation: CspEvaluation | None
    schedule_identity: str | None
    statistics: BacktrackingStatistics
    termination_reason: BacktrackingTerminationReason


class _NodeLimitReached(Exception):
    """Internal unwinding signal when the configured visit limit is reached."""


class BacktrackingEnumerator:
    """Stateful, deterministic, lazy enumeration of distinct CSP schedules.

    One instance retains its recursive generator between ``next_solution``
    calls. Its node limit applies cumulatively to the entire enumeration. Each
    variable is selected deterministically from the current partial assignment
    and assigned exactly once per path, while domain values are unique; this
    gives each complete schedule one path and prevents duplicate results.
    """

    def __init__(
        self,
        problem: CspProblem,
        options: BacktrackingOptions | None = None,
    ) -> None:
        if not isinstance(problem, CspProblem):
            raise CspInputError("backtracking requires a typed CSP problem")
        selected_options = BacktrackingOptions() if options is None else options
        if not isinstance(selected_options, BacktrackingOptions):
            raise CspInputError("options must be BacktrackingOptions")
        self._problem = problem
        self._options = selected_options
        self._started = perf_counter()
        self._solver_time_seconds = 0.0
        self._nodes_visited = 0
        self._assignments_attempted = 0
        self._backtracks = 0
        self._constraint_checks = 0
        self._maximum_depth = 0
        self._assigned: list[tuple[SessionVariable, WeeklySlot]] = []
        self._terminal: BacktrackingEnumerationResult | None = None
        self._solutions = self._visit()

    @property
    def problem_identity(self) -> str:
        """Return the immutable problem identity served by this enumerator."""
        return self._problem.problem_identity

    @property
    def whole_plan_identity(self) -> str:
        """Return the whole-plan identity served by this enumerator."""
        return self._problem.whole_plan_identity

    @property
    def statistics(self) -> BacktrackingStatistics:
        """Return cumulative statistics through the most recent operation."""
        return self._statistics()

    def next_solution(self) -> BacktrackingEnumerationResult:
        """Return the next distinct solution, or exhaustive/incomplete status."""
        if self._terminal is not None:
            return self._terminal
        started = perf_counter()
        terminal_status: BacktrackingEnumerationStatus | None = None
        terminal_reason: BacktrackingTerminationReason | None = None
        try:
            evaluation = next(self._solutions)
        except StopIteration:
            evaluation = None
            terminal_status = BacktrackingEnumerationStatus.EXHAUSTED
            terminal_reason = BacktrackingTerminationReason.EXHAUSTED
        except _NodeLimitReached:
            evaluation = None
            terminal_status = BacktrackingEnumerationStatus.INCOMPLETE
            terminal_reason = BacktrackingTerminationReason.NODE_LIMIT
        finally:
            self._solver_time_seconds += perf_counter() - started
        if terminal_status is not None and terminal_reason is not None:
            self._terminal = self._result(terminal_status, terminal_reason)
            return self._terminal
        assert evaluation is not None
        if (
            evaluation.status is not CspEvaluationStatus.FEASIBLE
            or evaluation.problem_identity != self._problem.problem_identity
            or evaluation.whole_plan_identity != self._problem.whole_plan_identity
        ):
            raise RuntimeError("enumerator produced an invalid solution evaluation")
        return self._result(
            BacktrackingEnumerationStatus.SOLUTION_FOUND,
            BacktrackingTerminationReason.SOLUTION_FOUND,
            evaluation=evaluation,
            schedule_identity=_schedule_identity(evaluation.assignment),
        )

    def _result(
        self,
        status: BacktrackingEnumerationStatus,
        reason: BacktrackingTerminationReason,
        *,
        evaluation: CspEvaluation | None = None,
        schedule_identity: str | None = None,
    ) -> BacktrackingEnumerationResult:
        return BacktrackingEnumerationResult(
            status=status,
            evaluation=evaluation,
            schedule_identity=schedule_identity,
            statistics=self._statistics(),
            termination_reason=reason,
        )

    def _statistics(self) -> BacktrackingStatistics:
        return BacktrackingStatistics(
            nodes_visited=self._nodes_visited,
            assignments_attempted=self._assignments_attempted,
            backtracks=self._backtracks,
            constraint_checks=self._constraint_checks,
            maximum_depth=self._maximum_depth,
            solver_time_seconds=self._solver_time_seconds,
            wall_time_seconds=perf_counter() - self._started,
        )

    def _record_active_time(self, elapsed_seconds: float) -> None:
        """Include synchronous solver-boundary work in active time."""
        self._solver_time_seconds += elapsed_seconds

    def _record_constraint_check(self) -> None:
        self._constraint_checks += 1

    def _validate(self, assignment: CspAssignment) -> CspEvaluation:
        return evaluate_assignment(
            self._problem,
            assignment,
            _on_constraint_check=self._record_constraint_check,
        )

    def _visit(self) -> Iterator[CspEvaluation]:
        if (
            self._options.max_nodes is not None
            and self._nodes_visited >= self._options.max_nodes
        ):
            raise _NodeLimitReached
        self._nodes_visited += 1
        self._maximum_depth = max(self._maximum_depth, len(self._assigned))
        current = CspAssignment(
            tuple(
                AssignmentEntry(variable, value) for variable, value in self._assigned
            )
        )
        validation = self._validate(current)
        if validation.status is CspEvaluationStatus.BRANCH_REJECTED:
            return
        if validation.status is CspEvaluationStatus.FEASIBLE:
            yield validation
            return

        unassigned = tuple(
            variable
            for variable in self._problem.variables
            if all(
                variable != assigned_variable for assigned_variable, _ in self._assigned
            )
        )
        if not unassigned:
            raise RuntimeError("validator returned INCOMPLETE for a complete state")
        variable = self._select_variable(unassigned)
        values = self._order_values(variable, unassigned)
        for value in values:
            self._assignments_attempted += 1
            self._assigned.append((variable, value))
            subtree_yielded_solution = False
            try:
                for solution in self._visit():
                    subtree_yielded_solution = True
                    yield solution
            except _NodeLimitReached:
                self._assigned.pop()
                raise
            self._assigned.pop()
            if not subtree_yielded_solution:
                self._backtracks += 1

    def _select_variable(
        self, unassigned: tuple[SessionVariable, ...]
    ) -> SessionVariable:
        if self._options.variable_order is BacktrackingVariableOrder.CANONICAL:
            return min(unassigned)

        legal_counts: dict[SessionVariable, int] = {}
        for variable in unassigned:
            legal_counts[variable] = sum(
                self._validate(
                    CspAssignment(
                        tuple(
                            AssignmentEntry(assigned_variable, assigned_value)
                            for assigned_variable, assigned_value in (
                                *self._assigned,
                                (variable, value),
                            )
                        )
                    )
                ).status
                is not CspEvaluationStatus.BRANCH_REJECTED
                for value in self._problem.domain_for(variable)
            )
        unassigned_set = set(unassigned)

        def priority(variable: SessionVariable) -> tuple[int, int, SessionVariable]:
            degree = sum(
                1
                for constraint in self._problem.constraints
                if variable in constraint.scope
                and any(
                    neighbor != variable and neighbor in unassigned_set
                    for neighbor in constraint.scope
                )
            )
            return (legal_counts[variable], -degree, variable)

        return min(unassigned, key=priority)

    def _order_values(
        self,
        variable: SessionVariable,
        unassigned: tuple[SessionVariable, ...],
    ) -> tuple[WeeklySlot, ...]:
        values = tuple(sorted(self._problem.domain_for(variable)))
        if (
            self._options.value_order is BacktrackingValueOrder.CANONICAL
            or len(values) < 2
        ):
            return values

        unassigned_set = set(unassigned)
        neighbors = {
            neighbor
            for constraint in self._problem.constraints
            if variable in constraint.scope
            for neighbor in constraint.scope
            if neighbor != variable and neighbor in unassigned_set
        }
        if not neighbors:
            return values

        ranked: list[tuple[int, WeeklySlot]] = []
        for value in values:
            support_count = 0
            for neighbor in sorted(neighbors):
                for neighbor_value in sorted(self._problem.domain_for(neighbor)):
                    probe = CspAssignment(
                        tuple(
                            AssignmentEntry(placed_variable, placed_value)
                            for placed_variable, placed_value in (
                                *self._assigned,
                                (variable, value),
                                (neighbor, neighbor_value),
                            )
                        )
                    )
                    result = self._validate(probe)
                    if result.status is not CspEvaluationStatus.BRANCH_REJECTED:
                        support_count += 1
            ranked.append((support_count, value))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return tuple(value for _, value in ranked)


def backtracking_enumerator(
    problem: CspProblem,
    options: BacktrackingOptions | None = None,
) -> BacktrackingEnumerator:
    """Create a deterministic lazy enumerator for all feasible schedules."""
    return BacktrackingEnumerator(problem, options)


def backtracking_search(
    problem: CspProblem,
    options: BacktrackingOptions | None = None,
) -> BacktrackingResult:
    """Return the first solution, or an exhaustive/incomplete solver outcome.

    This convenience API wraps ``BacktrackingEnumerator.next_solution``. Use
    the enumerator directly when a caller may need subsequent schedules after
    evaluating the first one.
    """
    enumerator = BacktrackingEnumerator(problem, options)
    step = enumerator.next_solution()
    if step.status is BacktrackingEnumerationStatus.SOLUTION_FOUND:
        assert step.evaluation is not None
        verification_started = perf_counter()
        outcome = CspSolverOutcome._from_solver(
            problem=problem,
            status=CspSolverStatus.FEASIBLE,
            exhaustive=False,
            evaluation=step.evaluation,
            _on_constraint_check=enumerator._record_constraint_check,
        )
        enumerator._record_active_time(perf_counter() - verification_started)
        reason = BacktrackingTerminationReason.SOLUTION_FOUND
    elif step.status is BacktrackingEnumerationStatus.INCOMPLETE:
        outcome = CspSolverOutcome._from_solver(
            problem=problem,
            status=CspSolverStatus.INCOMPLETE,
            exhaustive=False,
            reason="node_limit",
        )
        reason = BacktrackingTerminationReason.NODE_LIMIT
    else:
        outcome = CspSolverOutcome._from_solver(
            problem=problem,
            status=CspSolverStatus.INFEASIBLE,
            exhaustive=True,
            reason="all_branches_exhausted",
        )
        reason = BacktrackingTerminationReason.EXHAUSTED
    return BacktrackingResult(outcome, enumerator.statistics, reason)


def _schedule_identity(assignment: CspAssignment) -> str:
    """Hash a canonical assignment-to-slot schedule independent of entry order."""
    placements = sorted(
        (
            entry.variable.assignment_id,
            entry.variable.session_index,
            entry.value.slot_id,
            entry.value.week_index,
        )
        for entry in assignment.entries
    )
    encoded = json.dumps(placements, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
