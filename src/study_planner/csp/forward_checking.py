"""Forward Checking search for immutable CSP problems.

FC tests constraints whose scope has exactly one unassigned variable after an
extension. N-ary constraints with two or more unassigned variables are deferred;
the generic partial-assignment interface offers no support oracle for them.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from time import perf_counter

from study_planner.csp.backtracking import (
    BacktrackingEnumerationStatus,
    BacktrackingTerminationReason,
    _NodeLimitReached,
    _schedule_identity,
)
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


class ForwardCheckingVariableOrder(str, Enum):
    """Variable ordering choices for FC."""

    MRV_DEGREE = "mrv_degree"
    CANONICAL = "canonical"


class ForwardCheckingValueOrder(str, Enum):
    """Value ordering choices for FC."""

    LEAST_CONSTRAINING = "least_constraining"
    CANONICAL = "canonical"


@dataclass(frozen=True, slots=True)
class ForwardCheckingOptions:
    """Ordering policies and an optional cumulative node limit."""

    max_nodes: int | None = None
    variable_order: ForwardCheckingVariableOrder = (
        ForwardCheckingVariableOrder.MRV_DEGREE
    )
    value_order: ForwardCheckingValueOrder = (
        ForwardCheckingValueOrder.LEAST_CONSTRAINING
    )

    def __post_init__(self) -> None:
        if self.max_nodes is not None and (
            isinstance(self.max_nodes, bool)
            or not isinstance(self.max_nodes, int)
            or self.max_nodes < 0
        ):
            raise CspInputError("max_nodes must be a non-negative integer or None")
        if not isinstance(self.variable_order, ForwardCheckingVariableOrder):
            raise CspInputError("variable_order is not a supported policy")
        if not isinstance(self.value_order, ForwardCheckingValueOrder):
            raise CspInputError("value_order is not a supported policy")


@dataclass(frozen=True, slots=True)
class ForwardCheckingStatistics:
    """Cumulative search counters; predicate calls are constraint_checks."""

    nodes_visited: int
    assignments_attempted: int
    backtracks: int
    constraint_checks: int
    domain_prunings: int
    maximum_depth: int
    solver_time_seconds: float
    wall_time_seconds: float


@dataclass(frozen=True, slots=True)
class ForwardCheckingResult:
    """First feasible outcome or exhaustive/incomplete result."""

    outcome: CspSolverOutcome
    statistics: ForwardCheckingStatistics
    termination_reason: BacktrackingTerminationReason

    @property
    def status(self) -> CspSolverStatus:
        return self.outcome.status

    @property
    def evaluation(self) -> CspEvaluation | None:
        return self.outcome.evaluation


@dataclass(frozen=True, slots=True)
class ForwardCheckingEnumerationResult:
    """One schedule or the terminal state of a lazy FC enumeration."""

    status: BacktrackingEnumerationStatus
    evaluation: CspEvaluation | None
    schedule_identity: str | None
    statistics: ForwardCheckingStatistics
    termination_reason: BacktrackingTerminationReason


class ForwardCheckingEnumerator:
    """Lazy distinct schedule enumeration with private, reversible domains."""

    def __init__(
        self, problem: CspProblem, options: ForwardCheckingOptions | None = None
    ) -> None:
        if not isinstance(problem, CspProblem):
            raise CspInputError("forward checking requires a typed CSP problem")
        selected = ForwardCheckingOptions() if options is None else options
        if not isinstance(selected, ForwardCheckingOptions):
            raise CspInputError("options must be ForwardCheckingOptions")
        self._problem = problem
        self._options = selected
        self._domains = {v: list(problem.domain_for(v)) for v in problem.variables}
        # Most CSP work only concerns constraints that mention the variable
        # being assigned. Keep the original order while avoiding a full scan
        # of unrelated constraints in every propagation and value-order probe.
        self._constraints_by_variable = {
            variable: tuple(
                constraint
                for constraint in problem.constraints
                if variable in constraint.scope
            )
            for variable in problem.variables
        }
        self._assigned: list[tuple[SessionVariable, WeeklySlot]] = []
        self._started = perf_counter()
        self._solver_time = 0.0
        self._nodes = self._attempts = self._backtracks = self._checks = (
            self._prunings
        ) = self._depth = 0
        self._terminal: ForwardCheckingEnumerationResult | None = None
        self._solutions = self._visit()

    @property
    def problem_identity(self) -> str:
        return self._problem.problem_identity

    @property
    def whole_plan_identity(self) -> str:
        return self._problem.whole_plan_identity

    @property
    def statistics(self) -> ForwardCheckingStatistics:
        return self._statistics()

    def next_solution(self) -> ForwardCheckingEnumerationResult:
        if self._terminal is not None:
            return self._terminal
        started = perf_counter()
        status = reason = None
        try:
            evaluation = next(self._solutions)
        except StopIteration:
            evaluation = None
            status, reason = (
                BacktrackingEnumerationStatus.EXHAUSTED,
                BacktrackingTerminationReason.EXHAUSTED,
            )
        except _NodeLimitReached:
            evaluation = None
            status, reason = (
                BacktrackingEnumerationStatus.INCOMPLETE,
                BacktrackingTerminationReason.NODE_LIMIT,
            )
        finally:
            self._solver_time += perf_counter() - started
        if status is not None:
            self._terminal = ForwardCheckingEnumerationResult(
                status, None, None, self._statistics(), reason
            )
            return self._terminal
        assert evaluation is not None
        return ForwardCheckingEnumerationResult(
            BacktrackingEnumerationStatus.SOLUTION_FOUND,
            evaluation,
            _schedule_identity(evaluation.assignment),
            self._statistics(),
            BacktrackingTerminationReason.SOLUTION_FOUND,
        )

    def _statistics(self) -> ForwardCheckingStatistics:
        return ForwardCheckingStatistics(
            self._nodes,
            self._attempts,
            self._backtracks,
            self._checks,
            self._prunings,
            self._depth,
            self._solver_time,
            perf_counter() - self._started,
        )

    def _assignment(self) -> CspAssignment:
        return CspAssignment(tuple(AssignmentEntry(v, x) for v, x in self._assigned))

    def _check(self, constraint: object, assignment: CspAssignment) -> bool:
        self._checks += 1
        return constraint.is_satisfied(assignment)  # type: ignore[attr-defined]

    def _validate(self) -> CspEvaluation:
        return evaluate_assignment(
            self._problem,
            self._assignment(),
            _on_constraint_check=lambda: setattr(self, "_checks", self._checks + 1),
        )

    def _visit(self) -> Iterator[CspEvaluation]:
        if (
            self._options.max_nodes is not None
            and self._nodes >= self._options.max_nodes
        ):
            raise _NodeLimitReached
        self._nodes += 1
        self._depth = max(self._depth, len(self._assigned))
        validation = self._validate()
        if validation.status is CspEvaluationStatus.BRANCH_REJECTED:
            return
        if validation.status is CspEvaluationStatus.FEASIBLE:
            yield validation
            return
        unassigned = tuple(
            v for v in self._problem.variables if v not in dict(self._assigned)
        )
        variable = self._select_variable(unassigned)
        for value in self._ordered_values(variable, unassigned):
            self._attempts += 1
            self._assigned.append((variable, value))
            changed: list[tuple[SessionVariable, WeeklySlot]] = []
            assignment = self._assignment()
            assigned_set = set(assignment.placements)
            wiped_out = False
            for other in unassigned:
                if other == variable:
                    continue
                for constraint in self._constraints_by_variable[other]:
                    scope = constraint.scope
                    if other not in scope or any(
                        v not in assigned_set for v in scope if v != other
                    ):
                        continue
                    retained = []
                    for candidate in self._domains[other]:
                        probe = CspAssignment(
                            (*assignment.entries, AssignmentEntry(other, candidate))
                        )
                        if self._check(constraint, probe):
                            retained.append(candidate)
                        else:
                            changed.append((other, candidate))
                            self._prunings += 1
                    self._domains[other] = retained
                    if not retained:
                        wiped_out = True
                        break
                if wiped_out:
                    break
            yielded = False
            try:
                if not wiped_out:
                    for solution in self._visit():
                        yielded = True
                        yield solution
            except _NodeLimitReached:
                self._restore(changed)
                self._assigned.pop()
                raise
            self._restore(changed)
            self._assigned.pop()
            if not yielded:
                self._backtracks += 1

    def _restore(self, changed: list[tuple[SessionVariable, WeeklySlot]]) -> None:
        for variable, value in reversed(changed):
            self._domains[variable].append(value)
            self._domains[variable].sort()

    def _select_variable(
        self, unassigned: tuple[SessionVariable, ...]
    ) -> SessionVariable:
        if self._options.variable_order is ForwardCheckingVariableOrder.CANONICAL:
            return min(unassigned)
        return min(
            unassigned,
            key=lambda v: (
                len(self._domains[v]),
                -sum(
                    1
                    for c in self._constraints_by_variable[v]
                    if any(n != v and n in unassigned for n in c.scope)
                ),
                v,
            ),
        )

    def _ordered_values(
        self, variable: SessionVariable, unassigned: tuple[SessionVariable, ...]
    ) -> tuple[WeeklySlot, ...]:
        values = tuple(sorted(self._domains[variable]))
        if (
            self._options.value_order is ForwardCheckingValueOrder.CANONICAL
            or len(values) < 2
        ):
            return values
        ranked = []
        for value in values:
            score = 0
            probe = CspAssignment(
                (*self._assignment().entries, AssignmentEntry(variable, value))
            )
            for other in unassigned:
                if other == variable:
                    continue
                for candidate in self._domains[other]:
                    candidate_assignment = CspAssignment(
                        (*probe.entries, AssignmentEntry(other, candidate))
                    )
                    valid = True
                    for constraint in self._constraints_by_variable[variable]:
                        if (
                            variable in constraint.scope
                            and other in constraint.scope
                            and all(
                                v in candidate_assignment.placements
                                for v in constraint.scope
                            )
                        ):
                            valid &= self._check(constraint, candidate_assignment)
                    score += int(valid)
            ranked.append((score, value))
        return tuple(v for _, v in sorted(ranked, key=lambda item: (-item[0], item[1])))


def forward_checking_enumerator(
    problem: CspProblem, options: ForwardCheckingOptions | None = None
) -> ForwardCheckingEnumerator:
    """Create a deterministic lazy FC solution enumerator."""
    return ForwardCheckingEnumerator(problem, options)


def forward_checking_search(
    problem: CspProblem, options: ForwardCheckingOptions | None = None
) -> ForwardCheckingResult:
    """Return the first feasible schedule or exhaustive/incomplete outcome."""
    enumerator = ForwardCheckingEnumerator(problem, options)
    step = enumerator.next_solution()
    if step.status is BacktrackingEnumerationStatus.SOLUTION_FOUND:
        assert step.evaluation is not None
        started = perf_counter()
        outcome = CspSolverOutcome._from_solver(
            problem=problem,
            status=CspSolverStatus.FEASIBLE,
            exhaustive=False,
            evaluation=step.evaluation,
            _on_constraint_check=lambda: setattr(
                enumerator, "_checks", enumerator._checks + 1
            ),
        )
        enumerator._solver_time += perf_counter() - started
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
    return ForwardCheckingResult(outcome, enumerator.statistics, reason)
