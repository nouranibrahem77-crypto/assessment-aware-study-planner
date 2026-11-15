"""AC-3 propagation integrated with deterministic CSP backtracking.

AC-3 operates only on the problem's explicit binary-constraint relations.
N-ary constraints remain subject to ordinary partial-assignment validation and
the FC check used by this solver; AC-3 does not make them arc-consistent.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from time import perf_counter

from study_planner.csp.backtracking import (
    BacktrackingEnumerationStatus,
    BacktrackingTerminationReason,
    _NodeLimitReached,
    _schedule_identity,
)
from study_planner.csp.core import (
    AssignmentEntry,
    BinaryConstraint,
    CspAssignment,
    CspEvaluation,
    CspEvaluationStatus,
    CspInputError,
    CspProblem,
    CspSolverOutcome,
    CspSolverStatus,
    SessionVariable,
    VariableDomain,
    WeeklySlot,
    _track_binary_compatibility_calls,
    evaluate_assignment,
)


@dataclass(frozen=True, slots=True)
class Ac3Statistics:
    """Cumulative AC-3 search work and elapsed times.

    ``constraint_checks`` counts every ``is_satisfied`` and
    ``is_compatible`` predicate call. ``compatibility_checks`` is the latter
    subset. ``arc_revisions`` counts dequeued directed arcs, and
    ``arcs_enqueued`` counts initial and re-enqueued directed arcs.
    """

    nodes_visited: int
    assignments_attempted: int
    backtracks: int
    constraint_checks: int
    compatibility_checks: int
    domain_prunings: int
    arc_revisions: int
    arcs_enqueued: int
    maximum_depth: int
    solver_time_seconds: float
    wall_time_seconds: float


@dataclass(frozen=True, slots=True)
class Ac3PropagationResult:
    """Standalone binary arc-consistency result over private domains."""

    consistent: bool
    domains: tuple[VariableDomain, ...]
    compatibility_checks: int
    domain_prunings: int
    arc_revisions: int
    arcs_enqueued: int


@dataclass(frozen=True, slots=True)
class Ac3Result:
    """First feasible outcome or exhaustive/incomplete solver outcome."""

    outcome: CspSolverOutcome
    statistics: Ac3Statistics
    termination_reason: BacktrackingTerminationReason

    @property
    def status(self) -> CspSolverStatus:
        return self.outcome.status

    @property
    def evaluation(self) -> CspEvaluation | None:
        return self.outcome.evaluation


@dataclass(frozen=True, slots=True)
class Ac3EnumerationResult:
    """One feasible schedule or the terminal enumeration status."""

    status: BacktrackingEnumerationStatus
    evaluation: CspEvaluation | None
    schedule_identity: str | None
    statistics: Ac3Statistics
    termination_reason: BacktrackingTerminationReason


@dataclass(frozen=True, slots=True)
class Ac3Options:
    """Deterministic ordering and cumulative recursive-node limit."""

    max_nodes: int | None = None
    canonical_variable_order: bool = False

    def __post_init__(self) -> None:
        if self.max_nodes is not None and (
            isinstance(self.max_nodes, bool)
            or not isinstance(self.max_nodes, int)
            or self.max_nodes < 0
        ):
            raise CspInputError("max_nodes must be a non-negative integer or None")
        if not isinstance(self.canonical_variable_order, bool):
            raise CspInputError("canonical_variable_order must be boolean")


@dataclass(frozen=True, slots=True)
class _Arc:
    constraint: BinaryConstraint
    variable: SessionVariable
    neighbor: SessionVariable


def _all_arcs(problem: CspProblem) -> tuple[_Arc, ...]:
    arcs = [
        _Arc(constraint, first, second)
        for constraint in problem.binary_constraints
        for first, second in (
            (constraint.scope[0], constraint.scope[1]),
            (constraint.scope[1], constraint.scope[0]),
        )
    ]
    return tuple(
        sorted(
            arcs,
            key=lambda arc: (
                arc.constraint.constraint_id,
                arc.variable,
                arc.neighbor,
            ),
        )
    )


def ac3_propagate(
    problem: CspProblem,
    domains: tuple[VariableDomain, ...] | None = None,
) -> Ac3PropagationResult:
    """Enforce binary arc consistency without mutating ``problem`` or inputs.

    The optional domain tuple must cover exactly the problem variables and is
    copied before revision. N-ary constraints are deliberately not propagated.
    """
    if not isinstance(problem, CspProblem):
        raise CspInputError("AC-3 requires a typed CSP problem")
    source = problem.domains if domains is None else domains
    if not isinstance(source, tuple) or any(
        not isinstance(domain, VariableDomain) for domain in source
    ):
        raise CspInputError("AC-3 domains must be an immutable domain tuple")
    if len({domain.variable for domain in source}) != len(source) or {
        domain.variable for domain in source
    } != set(problem.variables):
        raise CspInputError("AC-3 domains must cover exactly the problem variables")
    if any(
        not set(domain.values).issubset(set(problem.domain_for(domain.variable)))
        for domain in source
    ):
        raise CspInputError("AC-3 domains cannot introduce out-of-problem values")
    working = {domain.variable: list(domain.values) for domain in source}
    checks = prunings = revisions = enqueued = 0

    def count_check() -> None:
        nonlocal checks
        checks += 1

    arcs = _all_arcs(problem)
    queue: deque[_Arc] = deque(arcs)
    enqueued += len(arcs)
    consistent = not any(not values for values in working.values())
    while consistent and queue:
        arc = queue.popleft()
        revisions += 1
        with _track_binary_compatibility_calls(count_check):
            retained = arc.constraint.revise(
                arc.variable,
                tuple(working[arc.variable]),
                tuple(working[arc.neighbor]),
            )
        if retained == tuple(working[arc.variable]):
            continue
        prunings += len(working[arc.variable]) - len(retained)
        working[arc.variable] = list(retained)
        if not retained:
            consistent = False
            break
        for neighbor_arc in arcs:
            if neighbor_arc.neighbor == arc.variable:
                queue.append(neighbor_arc)
                enqueued += 1

    result_domains = tuple(
        VariableDomain(variable, tuple(working[variable]))
        for variable in problem.variables
    )
    return Ac3PropagationResult(
        consistent,
        result_domains,
        checks,
        prunings,
        revisions,
        enqueued,
    )


class Ac3Enumerator:
    """Deterministic lazy enumeration using FC plus MAC-style AC-3."""

    def __init__(self, problem: CspProblem, options: Ac3Options | None = None) -> None:
        if not isinstance(problem, CspProblem):
            raise CspInputError("AC-3 search requires a typed CSP problem")
        selected = Ac3Options() if options is None else options
        if not isinstance(selected, Ac3Options):
            raise CspInputError("options must be Ac3Options")
        self._problem = problem
        self._options = selected
        self._domains = {v: list(problem.domain_for(v)) for v in problem.variables}
        self._assigned: list[tuple[SessionVariable, WeeklySlot]] = []
        self._started = perf_counter()
        self._solver_time = 0.0
        self._nodes = self._attempts = self._backtracks = 0
        self._checks = self._compatibility_checks = self._prunings = 0
        self._arc_revisions = self._arcs_enqueued = self._depth = 0
        self._terminal: Ac3EnumerationResult | None = None
        self._solutions = self._visit(root=True)

    @property
    def problem_identity(self) -> str:
        return self._problem.problem_identity

    @property
    def whole_plan_identity(self) -> str:
        return self._problem.whole_plan_identity

    @property
    def statistics(self) -> Ac3Statistics:
        return self._statistics()

    def _statistics(self) -> Ac3Statistics:
        return Ac3Statistics(
            self._nodes,
            self._attempts,
            self._backtracks,
            self._checks,
            self._compatibility_checks,
            self._prunings,
            self._arc_revisions,
            self._arcs_enqueued,
            self._depth,
            self._solver_time,
            perf_counter() - self._started,
        )

    def next_solution(self) -> Ac3EnumerationResult:
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
            self._terminal = Ac3EnumerationResult(
                status, None, None, self._statistics(), reason
            )
            return self._terminal
        assert evaluation is not None
        return Ac3EnumerationResult(
            BacktrackingEnumerationStatus.SOLUTION_FOUND,
            evaluation,
            _schedule_identity(evaluation.assignment),
            self._statistics(),
            BacktrackingTerminationReason.SOLUTION_FOUND,
        )

    def _assignment(self) -> CspAssignment:
        return CspAssignment(tuple(AssignmentEntry(v, x) for v, x in self._assigned))

    def _validate(self) -> CspEvaluation:
        def count_check() -> None:
            self._checks += 1

        with _track_binary_compatibility_calls(self._count_compatibility):
            return evaluate_assignment(
                self._problem, self._assignment(), _on_constraint_check=count_check
            )

    def _count_compatibility(self) -> None:
        self._checks += 1
        self._compatibility_checks += 1

    def _run_ac3(self) -> bool:
        arcs = _all_arcs(self._problem)
        queue: deque[_Arc] = deque(arcs)
        self._arcs_enqueued += len(arcs)
        assigned = dict(self._assigned)
        while queue:
            arc = queue.popleft()
            self._arc_revisions += 1
            relation = arc.constraint
            xi_assigned = arc.variable in assigned
            xi_values = (
                (assigned[arc.variable],)
                if xi_assigned
                else tuple(self._domains[arc.variable])
            )
            xj_values = (
                (assigned[arc.neighbor],)
                if arc.neighbor in assigned
                else tuple(self._domains[arc.neighbor])
            )
            if xi_assigned:
                with _track_binary_compatibility_calls(self._count_compatibility):
                    if not relation.has_support(arc.variable, xi_values[0], xj_values):
                        return False
                continue
            with _track_binary_compatibility_calls(self._count_compatibility):
                retained = relation.revise(arc.variable, xi_values, xj_values)
            if retained == xi_values:
                continue
            self._prunings += len(xi_values) - len(retained)
            self._domains[arc.variable] = list(retained)
            if not retained:
                return False
            for adjacent in arcs:
                if adjacent.neighbor == arc.variable:
                    queue.append(adjacent)
                    self._arcs_enqueued += 1
        return True

    def _forward_check(self, unassigned: tuple[SessionVariable, ...]) -> bool:
        assignment = self._assignment()
        assigned_set = set(assignment.placements)
        for other in unassigned:
            if other in assigned_set:
                continue
            for constraint in self._problem.constraints:
                scope = constraint.scope
                if other not in scope or any(
                    variable not in assigned_set
                    for variable in scope
                    if variable != other
                ):
                    continue
                retained: list[WeeklySlot] = []
                for candidate in self._domains[other]:
                    probe = CspAssignment(
                        (*assignment.entries, AssignmentEntry(other, candidate))
                    )
                    self._checks += 1
                    with _track_binary_compatibility_calls(self._count_compatibility):
                        satisfied = constraint.is_satisfied(probe)
                    if satisfied:
                        retained.append(candidate)
                    else:
                        self._prunings += 1
                self._domains[other] = retained
                if not retained:
                    return False
        return True

    def _visit(self, *, root: bool = False) -> Iterator[CspEvaluation]:
        if (
            self._options.max_nodes is not None
            and self._nodes >= self._options.max_nodes
        ):
            raise _NodeLimitReached
        self._nodes += 1
        self._depth = max(self._depth, len(self._assigned))
        if root and not self._run_ac3():
            return
        validation = self._validate()
        if validation.status is CspEvaluationStatus.BRANCH_REJECTED:
            return
        if validation.status is CspEvaluationStatus.FEASIBLE:
            yield validation
            return
        unassigned = tuple(
            variable
            for variable in self._problem.variables
            if variable not in dict(self._assigned)
        )
        variable = self._select_variable(unassigned)
        for value in tuple(sorted(self._domains[variable])):
            self._attempts += 1
            self._assigned.append((variable, value))
            changed: list[tuple[SessionVariable, WeeklySlot]] = []
            before = {
                current: tuple(values) for current, values in self._domains.items()
            }
            propagated = self._forward_check(unassigned)
            if propagated:
                propagated = self._run_ac3()
            for current, old_values in before.items():
                current_values = set(self._domains[current])
                changed.extend(
                    (current, old_value)
                    for old_value in old_values
                    if old_value not in current_values
                )
            yielded = False
            try:
                if propagated:
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
            if value not in self._domains[variable]:
                self._domains[variable].append(value)
                self._domains[variable].sort()

    def _select_variable(
        self, unassigned: tuple[SessionVariable, ...]
    ) -> SessionVariable:
        if self._options.canonical_variable_order:
            return min(unassigned)
        unassigned_set = set(unassigned)
        return min(
            unassigned,
            key=lambda variable: (
                len(self._domains[variable]),
                -sum(
                    1
                    for constraint in self._problem.constraints
                    if variable in constraint.scope
                    and any(
                        neighbor != variable and neighbor in unassigned_set
                        for neighbor in constraint.scope
                    )
                ),
                variable,
            ),
        )


def ac3_enumerator(
    problem: CspProblem, options: Ac3Options | None = None
) -> Ac3Enumerator:
    """Create a deterministic lazy enumerator using BT + FC + AC-3."""
    return Ac3Enumerator(problem, options)


def ac3_search(problem: CspProblem, options: Ac3Options | None = None) -> Ac3Result:
    """Return the first solution or an exhaustive/incomplete AC-3 outcome."""
    enumerator = Ac3Enumerator(problem, options)
    step = enumerator.next_solution()
    if step.status is BacktrackingEnumerationStatus.SOLUTION_FOUND:
        assert step.evaluation is not None
        started = perf_counter()

        def count_check() -> None:
            enumerator._checks += 1

        with _track_binary_compatibility_calls(enumerator._count_compatibility):
            outcome = CspSolverOutcome._from_solver(
                problem=problem,
                status=CspSolverStatus.FEASIBLE,
                exhaustive=False,
                evaluation=step.evaluation,
                _on_constraint_check=count_check,
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
    return Ac3Result(outcome, enumerator.statistics, reason)
