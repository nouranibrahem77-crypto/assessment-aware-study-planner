"""Tests for whole-plan schedule verification provenance."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from study_planner.csp import (
    AssignmentCandidateIdentity,
    AssignmentEntry,
    AssignmentPlan,
    AssignmentSchedule,
    ConstraintCheckStatus,
    CspAssignment,
    CspEvaluation,
    CspInputError,
    CspProblem,
    SessionPair,
    SessionVariable,
    VariableDomain,
    WeeklySlot,
    WholePlanManifest,
    WholePlanVerificationPolicy,
    WholePlanVerificationStatus,
    build_required_constraints,
    evaluate_assignment,
    expected_session_variables,
    verify_whole_plan,
)


@dataclass(frozen=True)
class _IncreasingWeeks:
    policy_identity: str = "test-increasing-weeks-v1"

    def required_pairs(
        self, sessions: tuple[SessionVariable, ...]
    ) -> tuple[SessionPair, ...]:
        by_assignment: dict[str, list[SessionVariable]] = {}
        for variable in sessions:
            by_assignment.setdefault(variable.assignment_id, []).append(variable)
        return tuple(
            SessionPair(first, second)
            for variables in by_assignment.values()
            for first, second in zip(variables, variables[1:], strict=False)
        )

    def is_ordered(self, earlier: WeeklySlot, later: WeeklySlot) -> bool:
        return earlier.week_index < later.week_index


def _manifest(
    counts: tuple[int, int, int] = (2, 0, 1),
) -> WholePlanManifest:
    plans = tuple(
        AssignmentPlan(
            f"a{index}",
            AssignmentCandidateIdentity(
                "student",
                "course",
                f"a{index}",
                (count, 0, 0, 0),
                f"features-{index}",
                "scenario",
            ),
            f"features-{index}",
        )
        for index, count in enumerate(counts, 1)
    )
    return WholePlanManifest("student", "course", plans, "scenario")


def _problem(
    *,
    counts: tuple[int, int, int] = (2, 0, 1),
    capacity: int = 2,
    ineligible: frozenset[WeeklySlot] = frozenset(),
) -> CspProblem:
    manifest = _manifest(counts)
    variables = expected_session_variables(manifest)
    slots = (
        WeeklySlot("w0-a", 0),
        WeeklySlot("w0-b", 0),
        WeeklySlot("w0-c", 0),
        WeeklySlot("w1-a", 1),
        WeeklySlot("w2-a", 2),
    )
    domains = tuple(VariableDomain(variable, slots) for variable in variables)
    constraints = build_required_constraints(
        manifest=manifest,
        variables=variables,
        eligible_slots=tuple(
            (variable, frozenset(slots) - ineligible) for variable in variables
        ),
        ordering_policy=_IncreasingWeeks(),
        weekly_capacity=capacity,
    )
    return CspProblem(
        manifest,
        variables,
        domains,
        constraints,
        capacity,
        _IncreasingWeeks(),
    )


def _policy(
    problem: CspProblem, *, capacity: int | None = None
) -> WholePlanVerificationPolicy:
    return WholePlanVerificationPolicy(
        version="whole-plan-verification-v1",
        deadline_semantics_identity="eligible-slots-v1",
        ordering_semantics_identity=problem.ordering_policy.configuration_identity,
        slot_catalog_identity="weekly-slot-catalog-v1",
        capacity_source_identity="request-capacity-v1",
        slot_identity_semantics="slot-id-and-week-v1",
        weekly_capacity=problem.weekly_capacity if capacity is None else capacity,
    )


def _valid_assignment(problem: CspProblem) -> CspAssignment:
    by_key = {
        (item.assignment_id, item.session_index): item
        for item in problem.variables
    }
    return CspAssignment(
        (
            AssignmentEntry(by_key[("a1", 0)], WeeklySlot("w0-a", 0)),
            AssignmentEntry(by_key[("a1", 1)], WeeklySlot("w1-a", 1)),
            AssignmentEntry(by_key[("a3", 0)], WeeklySlot("w0-b", 0)),
        )
    )


def test_valid_schedule_records_all_checks_and_binds_full_plan() -> None:
    problem = _problem()
    before_domains = problem.domains
    result = verify_whole_plan(problem, _valid_assignment(problem), _policy(problem))

    assert result.status is WholePlanVerificationStatus.FEASIBLE
    assert result.whole_plan_manifest is problem.manifest
    assert result.whole_plan_identity == problem.manifest.canonical_identity
    assert result.problem_identity == problem.problem_identity
    assert result.schedule_identity
    assert result.verification_identity
    assert result.verification_policy_identity == _policy(problem).policy_identity
    assert result.verification_policy_version == "whole-plan-verification-v1"
    assert len(result.schedule) == 3
    assert result.schedule[1].assignment_identity == "a2"
    assert result.schedule[1].sessions == ()
    assert all(
        check.status is ConstraintCheckStatus.PASSED
        for check in result.constraint_checks
    )
    assert {check.constraint_id for check in result.constraint_checks} >= {
        "assignment_coverage",
        "assignment_domains",
        "weekly_capacity",
    }
    assert problem.domains is before_domains


def test_invalid_schedule_reports_deadline_uniqueness_order_and_capacity() -> None:
    deadline_slot = WeeklySlot("late", 9)
    manifest = _manifest()
    variables = expected_session_variables(manifest)
    by_key = {
        (item.assignment_id, item.session_index): item
        for item in variables
    }
    # Domains intentionally do not contain this slot, so construct a separate
    # problem whose domain contains it while its supplied deadline set excludes it.
    slots = (WeeklySlot("w0-a", 0), WeeklySlot("w0-b", 0), deadline_slot)
    domains = tuple(VariableDomain(variable, slots) for variable in variables)
    constraints = build_required_constraints(
        manifest=manifest,
        variables=variables,
        eligible_slots=tuple(
            (variable, frozenset(slots) - {deadline_slot}) for variable in variables
        ),
        ordering_policy=_IncreasingWeeks(),
        weekly_capacity=1,
    )
    problem = CspProblem(
        manifest, variables, domains, constraints, 1, _IncreasingWeeks()
    )
    assignment = CspAssignment(
        (
            AssignmentEntry(by_key[("a1", 0)], deadline_slot),
            AssignmentEntry(by_key[("a1", 1)], WeeklySlot("w0-a", 0)),
            AssignmentEntry(by_key[("a3", 0)], WeeklySlot("w0-a", 0)),
        )
    )
    result = verify_whole_plan(problem, assignment, _policy(problem))
    statuses = {check.constraint_id: check.status for check in result.constraint_checks}
    assert result.status is WholePlanVerificationStatus.INFEASIBLE
    assert statuses["deadline_before_cutoff:a1:0"] is ConstraintCheckStatus.FAILED
    assert any(
        key.startswith("weekly_slot_unique:") and value is ConstraintCheckStatus.FAILED
        for key, value in statuses.items()
    )
    assert any(
        key.startswith("same_assignment_order:")
        and value is ConstraintCheckStatus.FAILED
        for key, value in statuses.items()
    )
    assert statuses["weekly_capacity"] is ConstraintCheckStatus.FAILED
    assert result.rejection_reasons


def test_partial_and_identity_mismatched_schedules_are_incomplete() -> None:
    problem = _problem()
    assignment = _valid_assignment(problem)
    partial = CspAssignment(assignment.entries[:-1])
    partial_result = verify_whole_plan(problem, partial, _policy(problem))
    assert partial_result.status is WholePlanVerificationStatus.INCOMPLETE
    assert "assignment_missing_variables" in partial_result.rejection_reasons
    assert any(
        check.status is ConstraintCheckStatus.NOT_RUN
        for check in partial_result.constraint_checks
    )

    mismatch = verify_whole_plan(
        problem,
        assignment,
        _policy(problem),
        expected_whole_plan_identity="wrong-plan",
        expected_problem_identity="wrong-problem",
    )
    assert mismatch.status is WholePlanVerificationStatus.INCOMPLETE
    assert "whole_plan_identity_mismatch" in mismatch.rejection_reasons
    assert "problem_identity_mismatch" in mismatch.rejection_reasons

    capacity_mismatch = verify_whole_plan(
        problem, assignment, _policy(problem, capacity=99)
    )
    assert capacity_mismatch.status is WholePlanVerificationStatus.INCOMPLETE
    assert "verification_capacity_mismatch" in capacity_mismatch.rejection_reasons


def test_verification_identity_binds_expected_identities_and_outcome() -> None:
    problem = _problem()
    assignment = _valid_assignment(problem)
    policy = _policy(problem)

    valid = verify_whole_plan(problem, assignment, policy)
    repeated = verify_whole_plan(problem, assignment, policy)
    wrong_plan = verify_whole_plan(
        problem,
        assignment,
        policy,
        expected_whole_plan_identity="incorrect-plan",
    )
    wrong_problem = verify_whole_plan(
        problem,
        assignment,
        policy,
        expected_problem_identity="incorrect-problem",
    )
    invalid_assignment = CspAssignment(
        tuple(
            AssignmentEntry(entry.variable, WeeklySlot("w0-a", 0))
            if entry.variable == problem.variables[-1]
            else entry
            for entry in assignment.entries
        )
    )
    infeasible = verify_whole_plan(problem, invalid_assignment, policy)

    assert valid.status is WholePlanVerificationStatus.FEASIBLE
    assert wrong_plan.status is WholePlanVerificationStatus.INCOMPLETE
    assert wrong_problem.status is WholePlanVerificationStatus.INCOMPLETE
    assert infeasible.status is WholePlanVerificationStatus.INFEASIBLE
    assert "whole_plan_identity_mismatch" in wrong_plan.rejection_reasons
    assert "problem_identity_mismatch" in wrong_problem.rejection_reasons
    assert wrong_plan.rejection_reasons != wrong_problem.rejection_reasons
    assert valid.verification_identity == repeated.verification_identity
    assert valid.verification_identity != wrong_plan.verification_identity
    assert valid.verification_identity != wrong_problem.verification_identity
    assert wrong_plan.verification_identity != wrong_problem.verification_identity
    assert valid.verification_identity != infeasible.verification_identity


def test_verification_identity_ignores_semantically_irrelevant_input_order() -> None:
    problem = _problem()
    assignment = _valid_assignment(problem)
    policy = _policy(problem)
    baseline = verify_whole_plan(problem, assignment, policy)

    reordered_problem = CspProblem(
        manifest=problem.manifest,
        variables=tuple(reversed(problem.variables)),
        domains=tuple(reversed(problem.domains)),
        constraints=tuple(reversed(problem.constraints)),
        weekly_capacity=problem.weekly_capacity,
        ordering_policy=problem.ordering_policy,
    )
    reordered_assignment = CspAssignment(tuple(reversed(assignment.entries)))
    reordered = verify_whole_plan(
        reordered_problem,
        reordered_assignment,
        _policy(reordered_problem),
    )

    assert reordered_problem.problem_identity == problem.problem_identity
    assert reordered.schedule == baseline.schedule
    assert reordered.schedule_identity == baseline.schedule_identity
    assert reordered.verification_identity == baseline.verification_identity


def test_zero_session_plan_is_verified_and_keeps_all_three_entries() -> None:
    problem = _problem(counts=(0, 0, 0))
    result = verify_whole_plan(problem, CspAssignment(), _policy(problem))

    assert problem.variables == ()
    assert problem.domains == ()
    assert result.status is WholePlanVerificationStatus.FEASIBLE
    assert result.schedule == tuple(
        AssignmentSchedule(f"a{index}", ()) for index in range(1, 4)
    )
    assert len(result.schedule) == 3
    assert all(
        check.status is ConstraintCheckStatus.PASSED
        for check in result.constraint_checks
    )


def test_out_of_domain_and_csp_evaluation_are_not_final_verification() -> None:
    problem = _problem()
    assignment = _valid_assignment(problem)
    variable = problem.variables[0]
    out_of_domain = CspAssignment(
        tuple(
            AssignmentEntry(entry.variable, WeeklySlot("outside", 8))
            if entry.variable == variable
            else entry
            for entry in assignment.entries
        )
    )
    invalid = verify_whole_plan(problem, out_of_domain, _policy(problem))
    assert invalid.status is WholePlanVerificationStatus.INFEASIBLE
    assert "assignment_contains_out_of_domain_values" in invalid.rejection_reasons

    evaluation = evaluate_assignment(problem, assignment)
    assert isinstance(evaluation, CspEvaluation)
    with pytest.raises(CspInputError, match="CSP assignment"):
        verify_whole_plan(problem, evaluation, _policy(problem))  # type: ignore[arg-type]
