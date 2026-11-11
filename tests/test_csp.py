"""Behavioral regression tests for CSP contracts and constraint relations."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from study_planner.csp import (
    AssignmentCandidateIdentity,
    AssignmentEntry,
    AssignmentPlan,
    CspAssignment,
    CspEvaluation,
    CspEvaluationStatus,
    CspInputError,
    CspProblem,
    CspSolverOutcome,
    CspSolverStatus,
    DeadlineConstraint,
    SameAssignmentOrderingConstraint,
    SessionPair,
    SessionVariable,
    WeeklySlot,
    WholePlanManifest,
    build_required_constraints,
    candidate_identity_for,
    evaluate_assignment,
    expected_session_variables,
)
from study_planner.csp.core import OrderingPolicySnapshot, VariableDomain


@dataclass(frozen=True)
class AdjacentIncreasingWeekPolicy:
    """Test-only ordering semantics: adjacent session indices, earlier week."""

    policy_identity: str = "test-adjacent-increasing-week-v1"

    def required_pairs(
        self, sessions: tuple[SessionVariable, ...]
    ) -> tuple[SessionPair, ...]:
        pairs: list[SessionPair] = []
        by_assignment: dict[str, list[SessionVariable]] = {}
        for session in sessions:
            by_assignment.setdefault(session.assignment_id, []).append(session)
        for assignment_sessions in by_assignment.values():
            ordered = sorted(assignment_sessions, key=lambda item: item.session_index)
            pairs.extend(
                SessionPair(first, second)
                for first, second in zip(ordered, ordered[1:], strict=False)
            )
        return tuple(pairs)

    def is_ordered(self, earlier: WeeklySlot, later: WeeklySlot) -> bool:
        return earlier.week_index < later.week_index


@dataclass(frozen=True)
class EmptyOrderingPolicy(AdjacentIncreasingWeekPolicy):
    """Invalid for assignments with multiple sessions."""

    policy_identity: str = "test-empty-policy-v1"

    def required_pairs(
        self, sessions: tuple[SessionVariable, ...]
    ) -> tuple[SessionPair, ...]:
        return ()


@dataclass(frozen=True)
class DuplicateOrderingPolicy(AdjacentIncreasingWeekPolicy):
    """Invalid because it repeats a required relationship."""

    def required_pairs(
        self, sessions: tuple[SessionVariable, ...]
    ) -> tuple[SessionPair, ...]:
        pair = super().required_pairs(sessions)[0]
        return (pair, pair)


@dataclass(frozen=True)
class ReversedOrderingPolicy(AdjacentIncreasingWeekPolicy):
    """Invalid because it reverses the canonical session relationship."""

    def required_pairs(
        self, sessions: tuple[SessionVariable, ...]
    ) -> tuple[SessionPair, ...]:
        pair = super().required_pairs(sessions)[0]
        return (SessionPair(pair.later, pair.earlier),)


@dataclass(frozen=True)
class UnrelatedOrderingPolicy(AdjacentIncreasingWeekPolicy):
    """Invalid because it connects sessions from different assignments."""

    def required_pairs(
        self, sessions: tuple[SessionVariable, ...]
    ) -> tuple[SessionPair, ...]:
        return (SessionPair(sessions[0], sessions[-1]),)


@dataclass(frozen=True)
class MissingOrderingPolicyMethods:
    policy_identity: str = "test-incomplete-policy"


class MutableOrderingPolicy:
    """Mutation fixture proving problems capture ordering behavior at creation."""

    policy_identity = "mutable-order-policy-v1"

    def __init__(self) -> None:
        self.require_increasing_weeks = True

    def required_pairs(
        self, sessions: tuple[SessionVariable, ...]
    ) -> tuple[SessionPair, ...]:
        return AdjacentIncreasingWeekPolicy().required_pairs(sessions)

    def is_ordered(self, earlier: WeeklySlot, later: WeeklySlot) -> bool:
        if self.require_increasing_weeks:
            return earlier.week_index < later.week_index
        return earlier.week_index > later.week_index


def make_manifest(
    allocations: tuple[tuple[str, tuple[int, int, int, int], str], ...] = (
        ("a1", (1, 1, 0, 0), "features-a1"),
        ("a2", (0, 0, 0, 0), "features-a2"),
        ("a3", (1, 0, 0, 0), "features-a3"),
    ),
) -> WholePlanManifest:
    """Build a small whole-plan fixture; a2 intentionally has zero sessions."""
    assignment_plans = tuple(
        AssignmentPlan(
            assignment_id,
            AssignmentCandidateIdentity(
                student_identity="student-1",
                course_id="course-1",
                assignment_identity=assignment_id,
                canonical_allocation_identity=allocation,
                base_feature_context_identity=feature_id,
                mapping_scenario_identity="central",
            ),
            feature_id,
        )
        for assignment_id, allocation, feature_id in allocations
    )
    return WholePlanManifest(
        student_identity="student-1",
        course_identity="course-1",
        assignments=assignment_plans,
        mapping_scenario_identity="central",
    )


def build_problem(
    *,
    capacity: int = 2,
    policy: object | None = None,
    manifest: WholePlanManifest | None = None,
) -> CspProblem:
    manifest = manifest or make_manifest()
    ordering = policy or AdjacentIncreasingWeekPolicy()
    variables = expected_session_variables(manifest)
    slots = (
        WeeklySlot("w0-a", 0),
        WeeklySlot("w0-b", 0),
        WeeklySlot("w1-a", 1),
        WeeklySlot("w2-a", 2),
    )
    domains = tuple(VariableDomain(variable, slots) for variable in variables)
    eligible = tuple((variable, frozenset(slots)) for variable in variables)
    constraints = build_required_constraints(
        manifest=manifest,
        variables=variables,
        eligible_slots=eligible,
        ordering_policy=ordering,  # type: ignore[arg-type]
        weekly_capacity=capacity,
    )
    return CspProblem(
        manifest=manifest,
        variables=variables,
        domains=domains,
        constraints=constraints,
        weekly_capacity=capacity,
        ordering_policy=ordering,  # type: ignore[arg-type]
    )


def valid_complete_assignment(problem: CspProblem) -> CspAssignment:
    variables = {
        variable.assignment_id + str(variable.session_index): variable
        for variable in problem.variables
    }
    return CspAssignment(
        (
            AssignmentEntry(variables["a10"], WeeklySlot("w0-a", 0)),
            AssignmentEntry(variables["a11"], WeeklySlot("w1-a", 1)),
            AssignmentEntry(variables["a30"], WeeklySlot("w0-b", 0)),
        )
    )


def test_whole_plan_identity_is_canonical_and_retains_zero_session_entry() -> None:
    first = make_manifest()
    reordered = make_manifest(
        tuple(
            reversed(
                (
                    ("a1", (1, 1, 0, 0), "features-a1"),
                    ("a2", (0, 0, 0, 0), "features-a2"),
                    ("a3", (1, 0, 0, 0), "features-a3"),
                )
            )
        )
    )

    assert first.canonical_identity == reordered.canonical_identity
    assert build_problem().problem_identity == build_problem().problem_identity
    assert first.assignment("a2").session_count == 0
    assert len(expected_session_variables(first)) == 3
    assert (
        candidate_identity_for(first, "a1") == first.assignment("a1").candidate_identity
    )


def test_manifest_rejects_missing_duplicate_and_mismatched_assignment_entries() -> None:
    entries = (
        ("a1", (1, 0, 0, 0), "f1"),
        ("a2", (1, 0, 0, 0), "f2"),
        ("a3", (0, 0, 0, 0), "f3"),
    )
    with pytest.raises(CspInputError, match="three assignments"):
        make_manifest(entries[:2])
    with pytest.raises(CspInputError, match="unique"):
        make_manifest((entries[0], entries[1], entries[1]))
    with pytest.raises(CspInputError, match="do not match"):
        AssignmentPlan(
            "a1",
            AssignmentCandidateIdentity(
                "student-1", "course-1", "a2", (1, 0, 0, 0), "f1", "central"
            ),
            "f1",
        )


def test_manifest_identity_changes_with_student_course_allocation_or_context() -> None:
    original = make_manifest()
    changed_student = WholePlanManifest(
        "student-2",
        "course-1",
        tuple(
            AssignmentPlan(
                item.assignment_identity,
                AssignmentCandidateIdentity(
                    "student-2",
                    "course-1",
                    item.assignment_identity,
                    item.candidate_identity.canonical_allocation_identity,
                    item.base_feature_context_identity,
                    "central",
                ),
                item.base_feature_context_identity,
            )
            for item in original.assignments
        ),
        "central",
    )
    changed_course = WholePlanManifest(
        "student-1",
        "course-2",
        tuple(
            AssignmentPlan(
                item.assignment_identity,
                replace(item.candidate_identity, course_id="course-2"),
                item.base_feature_context_identity,
            )
            for item in original.assignments
        ),
        "central",
    )
    changed_entry = make_manifest(
        (
            ("a1", (2, 0, 0, 0), "features-a1"),
            ("a2", (0, 0, 0, 0), "features-a2"),
            ("a3", (1, 0, 0, 0), "features-a3"),
        )
    )
    changed_context = make_manifest(
        (
            ("a1", (1, 1, 0, 0), "new-features-a1"),
            ("a2", (0, 0, 0, 0), "features-a2"),
            ("a3", (1, 0, 0, 0), "features-a3"),
        )
    )
    assert (
        len(
            {
                original.canonical_identity,
                changed_student.canonical_identity,
                changed_course.canonical_identity,
                changed_entry.canonical_identity,
                changed_context.canonical_identity,
            }
        )
        == 5
    )
    candidate = original.assignment("a1").candidate_identity
    other_course_candidate = replace(candidate, course_id="course-2")
    assert candidate.canonical_identity != other_course_candidate.canonical_identity
    assert candidate.canonical_identity == replace(candidate).canonical_identity


def test_manifest_rejects_candidate_from_another_course() -> None:
    original = make_manifest()
    entries = tuple(
        AssignmentPlan(
            item.assignment_identity,
            replace(item.candidate_identity, course_id="course-elsewhere")
            if item.assignment_identity == "a2"
            else item.candidate_identity,
            item.base_feature_context_identity,
        )
        for item in original.assignments
    )
    with pytest.raises(CspInputError, match="another course"):
        WholePlanManifest("student-1", "course-1", entries, "central")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("student_identity", "student-other", "another student"),
        ("mapping_scenario_identity", "high", "another mapping scenario"),
    ],
)
def test_manifest_rejects_candidate_identity_context_mismatches(
    field: str, value: str, message: str
) -> None:
    original = make_manifest()
    entries = tuple(
        AssignmentPlan(
            item.assignment_identity,
            replace(item.candidate_identity, **{field: value})
            if item.assignment_identity == "a2"
            else item.candidate_identity,
            item.base_feature_context_identity,
        )
        for item in original.assignments
    )
    with pytest.raises(CspInputError, match=message):
        WholePlanManifest("student-1", "course-1", entries, "central")


def test_problem_snapshots_mutable_ordering_policy_and_identity() -> None:
    policy = MutableOrderingPolicy()
    problem = build_problem(policy=policy)
    identity = problem.problem_identity
    ordering = next(
        item
        for item in problem.constraints
        if isinstance(item, SameAssignmentOrderingConstraint)
    )
    earlier, later = ordering.scope
    early_slot = WeeklySlot("w0-a", 0)
    late_slot = WeeklySlot("w1-a", 1)
    assert isinstance(problem.ordering_policy, OrderingPolicySnapshot)
    assert ordering.is_compatible(earlier, early_slot, later, late_slot)

    policy.require_increasing_weeks = False

    assert problem.problem_identity == identity
    assert ordering.is_compatible(earlier, early_slot, later, late_slot)
    assert not ordering.is_compatible(earlier, late_slot, later, early_slot)


def test_problem_binds_session_variables_and_domains_to_complete_manifest() -> None:
    problem = build_problem()
    assert problem.whole_plan_identity == problem.manifest.canonical_identity
    assert {item.assignment_id for item in problem.variables} == {"a1", "a3"}
    assert len(problem.binary_constraints) == 4
    assert build_problem(capacity=1).problem_identity != problem.problem_identity
    narrowed = list(problem.domains)
    narrowed[0] = VariableDomain(narrowed[0].variable, narrowed[0].values[:-1])
    assert replace(problem, domains=tuple(narrowed)).problem_identity != (
        problem.problem_identity
    )
    reordered_domains = tuple(
        VariableDomain(domain.variable, tuple(reversed(domain.values)))
        for domain in reversed(problem.domains)
    )
    assert replace(problem, domains=reordered_domains).problem_identity == (
        problem.problem_identity
    )
    assert replace(
        problem, constraints=tuple(reversed(problem.constraints))
    ).problem_identity == (problem.problem_identity)
    with pytest.raises(CspInputError, match="manifest"):
        replace(problem, variables=problem.variables[:-1])
    with pytest.raises(CspInputError, match="domains"):
        replace(problem, domains=problem.domains[:-1])


def test_problem_rejects_constraint_scopes_outside_declared_variables() -> None:
    @dataclass(frozen=True)
    class UnrelatedConstraint:
        constraint_id: str = "unrelated"
        scope: tuple[SessionVariable, ...] = (SessionVariable("other", 0),)
        configuration_identity: str = "unrelated-v1"

        def is_satisfied(self, assignment: CspAssignment) -> bool:
            return True

    problem = build_problem()
    with pytest.raises(CspInputError, match="outside this problem"):
        replace(problem, constraints=problem.constraints + (UnrelatedConstraint(),))


def test_complete_valid_assignment_is_feasible_and_partial_is_incomplete() -> None:
    problem = build_problem()
    complete = evaluate_assignment(problem, valid_complete_assignment(problem))
    partial = evaluate_assignment(
        problem,
        CspAssignment((AssignmentEntry(problem.variables[0], WeeklySlot("w0-a", 0)),)),
    )
    assert complete.status is CspEvaluationStatus.FEASIBLE
    assert partial.status is CspEvaluationStatus.INCOMPLETE


def test_constraint_failure_rejects_only_branch_not_whole_plan() -> None:
    problem = build_problem()
    first, second = problem.variables[:2]
    branch = CspAssignment(
        (
            AssignmentEntry(first, WeeklySlot("w1-a", 1)),
            AssignmentEntry(second, WeeklySlot("w0-a", 0)),
        )
    )
    result = evaluate_assignment(problem, branch)
    assert result.status is CspEvaluationStatus.BRANCH_REJECTED
    assert result.whole_plan_identity == problem.whole_plan_identity


def test_solver_outcome_infeasible_requires_exhaustive_search() -> None:
    problem = build_problem()
    with pytest.raises(CspInputError, match="exhaustive"):
        CspSolverOutcome._from_solver(
            problem=problem,
            status=CspSolverStatus.INFEASIBLE,
            exhaustive=False,
        )
    incomplete = CspSolverOutcome._from_solver(
        problem=problem,
        status=CspSolverStatus.INCOMPLETE,
        exhaustive=False,
        reason="resource_limit",
    )
    infeasible = CspSolverOutcome._from_solver(
        problem=problem,
        status=CspSolverStatus.INFEASIBLE,
        exhaustive=True,
        reason="frontier_exhausted",
    )
    assert incomplete.status is CspSolverStatus.INCOMPLETE
    assert infeasible.status is CspSolverStatus.INFEASIBLE


def test_solver_result_cannot_reuse_an_evaluation_under_another_problem_identity() -> (
    None
):
    problem = build_problem()
    evaluation = evaluate_assignment(problem, valid_complete_assignment(problem))
    other_problem = build_problem(capacity=3, manifest=problem.manifest)
    with pytest.raises(CspInputError, match="complete evaluation"):
        CspSolverOutcome._from_solver(
            problem=other_problem,
            status=CspSolverStatus.FEASIBLE,
            exhaustive=False,
            evaluation=evaluation,
        )


def test_solver_outcome_rejects_mismatched_whole_plan_identity() -> None:
    problem = build_problem()
    evaluation = evaluate_assignment(problem, valid_complete_assignment(problem))
    forged = object.__new__(CspEvaluation)
    object.__setattr__(forged, "whole_plan_identity", "different-whole-plan")
    object.__setattr__(forged, "problem_identity", problem.problem_identity)
    object.__setattr__(forged, "status", CspEvaluationStatus.FEASIBLE)
    object.__setattr__(forged, "assignment", evaluation.assignment)
    object.__setattr__(forged, "failed_checks", ())

    with pytest.raises(CspInputError, match="complete evaluation"):
        CspSolverOutcome._from_solver(
            problem=problem,
            status=CspSolverStatus.FEASIBLE,
            exhaustive=False,
            evaluation=forged,
        )


def test_validator_factory_rejects_partial_and_inconsistent_assignments() -> None:
    problem = build_problem()
    partial = CspAssignment(
        (AssignmentEntry(problem.variables[0], WeeklySlot("w0-a", 0)),)
    )
    with pytest.raises(CspInputError, match="complete, consistent"):
        CspEvaluation._from_validator(
            problem=problem,
            status=CspEvaluationStatus.FEASIBLE,
            assignment=partial,
        )

    complete = valid_complete_assignment(problem)
    out_of_domain = CspAssignment(
        tuple(
            AssignmentEntry(
                entry.variable,
                WeeklySlot("not-in-domain", 99)
                if entry.variable == problem.variables[0]
                else entry.value,
            )
            for entry in complete.entries
        )
    )
    with pytest.raises(CspInputError, match="complete, consistent"):
        CspEvaluation._from_validator(
            problem=problem,
            status=CspEvaluationStatus.FEASIBLE,
            assignment=out_of_domain,
        )

    constraint_violation = CspAssignment(
        (
            AssignmentEntry(problem.variables[0], WeeklySlot("w0-a", 0)),
            AssignmentEntry(problem.variables[1], WeeklySlot("w0-a", 0)),
            AssignmentEntry(problem.variables[2], WeeklySlot("w0-b", 0)),
        )
    )
    with pytest.raises(CspInputError, match="complete, consistent"):
        CspEvaluation._from_validator(
            problem=problem,
            status=CspEvaluationStatus.FEASIBLE,
            assignment=constraint_violation,
        )


def test_solver_factory_rechecks_feasible_evaluation_completeness() -> None:
    problem = build_problem()
    partial = CspAssignment(
        (AssignmentEntry(problem.variables[0], WeeklySlot("w0-a", 0)),)
    )
    # Model a stale or incorrectly constructed internal record. The solver
    # boundary must revalidate it instead of trusting its status field.
    forged = object.__new__(CspEvaluation)
    object.__setattr__(forged, "whole_plan_identity", problem.whole_plan_identity)
    object.__setattr__(forged, "problem_identity", problem.problem_identity)
    object.__setattr__(forged, "status", CspEvaluationStatus.FEASIBLE)
    object.__setattr__(forged, "assignment", partial)
    object.__setattr__(forged, "failed_checks", ())
    with pytest.raises(CspInputError, match="complete evaluation"):
        CspSolverOutcome._from_solver(
            problem=problem,
            status=CspSolverStatus.FEASIBLE,
            exhaustive=False,
            evaluation=forged,
        )


def test_zero_variable_problem_is_feasible_with_three_zero_allocations() -> None:
    zero_manifest = make_manifest(
        (
            ("a1", (0, 0, 0, 0), "features-a1"),
            ("a2", (0, 0, 0, 0), "features-a2"),
            ("a3", (0, 0, 0, 0), "features-a3"),
        )
    )
    problem = build_problem(manifest=zero_manifest)
    assert problem.variables == ()
    assert problem.domains == ()
    assert evaluate_assignment(problem, CspAssignment()).status is (
        CspEvaluationStatus.FEASIBLE
    )


@pytest.mark.parametrize(
    "policy",
    [
        EmptyOrderingPolicy(),
        DuplicateOrderingPolicy(),
        ReversedOrderingPolicy(),
        UnrelatedOrderingPolicy(),
        MissingOrderingPolicyMethods(),
    ],
)
def test_invalid_or_incomplete_ordering_relationships_are_rejected(
    policy: object,
) -> None:
    with pytest.raises(CspInputError):
        build_problem(policy=policy)


def test_deadline_eligibility_must_cover_every_declared_session_once() -> None:
    manifest = make_manifest()
    variables = expected_session_variables(manifest)
    with pytest.raises(CspInputError, match="cover each session"):
        build_required_constraints(
            manifest=manifest,
            variables=variables,
            eligible_slots=(),
            ordering_policy=AdjacentIncreasingWeekPolicy(),
            weekly_capacity=2,
        )


def test_missing_ordering_constraint_is_rejected_by_problem_contract() -> None:
    problem = build_problem()
    without_ordering = tuple(
        item
        for item in problem.constraints
        if not item.constraint_id.startswith("same_assignment_order:")
    )
    with pytest.raises(CspInputError, match="ordering"):
        replace(problem, constraints=without_ordering)


def test_binary_support_and_revision_work_without_artificial_assignments() -> None:
    problem = build_problem()
    first, second = problem.variables[:2]
    unique = next(
        item
        for item in problem.binary_constraints
        if item.constraint_id.startswith("weekly_slot_unique:")
        and set(item.scope) == {first, problem.variables[2]}
    )
    kept = unique.revise(
        first,
        (WeeklySlot("w0-a", 0), WeeklySlot("w1-a", 1)),
        (WeeklySlot("w0-a", 0),),
    )
    assert kept == (WeeklySlot("w1-a", 1),)
    assert unique.has_support(first, WeeklySlot("w0-a", 0), (WeeklySlot("w1-a", 1),))


def test_ordering_support_and_revision_work_in_both_variable_directions() -> None:
    problem = build_problem()
    ordering = next(
        item
        for item in problem.binary_constraints
        if isinstance(item, SameAssignmentOrderingConstraint)
    )
    earlier, later = ordering.scope
    week_zero = WeeklySlot("w0-a", 0)
    week_one = WeeklySlot("w1-a", 1)

    assert ordering.has_support(earlier, week_zero, (week_one,))
    assert ordering.has_support(later, week_one, (week_zero,))
    assert ordering.revise(earlier, (week_zero, week_one), (week_one,)) == (week_zero,)
    assert ordering.revise(later, (week_zero, week_one), (week_zero,)) == (week_one,)
    assert ordering.revise(earlier, (week_zero,), ()) == ()
    assert ordering.revise(later, (week_one,), ()) == ()


def test_deadline_eligibility_is_explicit_and_partial_future_session_is_ignored() -> (
    None
):
    problem = build_problem()
    first = problem.variables[0]
    deadline = next(
        item
        for item in problem.constraints
        if isinstance(item, DeadlineConstraint) and item.variable == first
    )
    boundary_slot = WeeklySlot("outside-cutoff", 3)
    assert deadline.is_satisfied(CspAssignment())
    assert not deadline.is_satisfied(
        CspAssignment((AssignmentEntry(first, boundary_slot),))
    )


@pytest.mark.parametrize("capacity, expected", [(1, False), (2, True)])
def test_weekly_capacity_counts_sessions_across_assignments(
    capacity: int, expected: bool
) -> None:
    problem = build_problem(capacity=capacity)
    result = evaluate_assignment(problem, valid_complete_assignment(problem))
    assert (result.status is CspEvaluationStatus.FEASIBLE) is expected
    if not expected:
        assert "weekly_capacity" in result.failed_checks


def test_duplicate_slot_ids_cannot_map_to_different_weeks() -> None:
    problem = build_problem()
    domains = list(problem.domains)
    domains[0] = VariableDomain(domains[0].variable, (WeeklySlot("ambiguous", 0),))
    domains[1] = VariableDomain(domains[1].variable, (WeeklySlot("ambiguous", 1),))
    with pytest.raises(CspInputError, match="multiple weeks"):
        replace(problem, domains=tuple(domains))


def test_empty_domains_and_out_of_domain_assignments_reject_branches() -> None:
    problem = build_problem()
    domains = list(problem.domains)
    domains[0] = VariableDomain(domains[0].variable, ())
    empty_domain_problem = replace(problem, domains=tuple(domains))
    result = evaluate_assignment(empty_domain_problem, CspAssignment())
    assert result.status is CspEvaluationStatus.BRANCH_REJECTED
    assert result.failed_checks[0].startswith("empty_domain:")

    unknown = CspAssignment(
        (AssignmentEntry(problem.variables[0], WeeklySlot("unknown", 8)),)
    )
    assert (
        evaluate_assignment(problem, unknown).status
        is CspEvaluationStatus.BRANCH_REJECTED
    )


def test_conflicting_assignment_entries_are_rejected_at_construction() -> None:
    variable = SessionVariable("a1", 0)
    with pytest.raises(CspInputError, match="twice"):
        CspAssignment(
            (
                AssignmentEntry(variable, WeeklySlot("s1", 0)),
                AssignmentEntry(variable, WeeklySlot("s2", 0)),
            )
        )


def test_ordering_constraint_accepts_unassigned_partner_and_checks_full_pair() -> None:
    problem = build_problem()
    ordering = next(
        item
        for item in problem.constraints
        if isinstance(item, SameAssignmentOrderingConstraint)
    )
    earlier, later = ordering.scope
    assert ordering.is_satisfied(
        CspAssignment((AssignmentEntry(earlier, WeeklySlot("w0-a", 0)),))
    )
    assert not ordering.is_satisfied(
        CspAssignment(
            (
                AssignmentEntry(earlier, WeeklySlot("w1-a", 1)),
                AssignmentEntry(later, WeeklySlot("w0-a", 0)),
            )
        )
    )
