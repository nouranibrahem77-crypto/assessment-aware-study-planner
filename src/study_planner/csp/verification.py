"""Whole-plan schedule verification and immutable audit provenance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum

from study_planner.csp.core import (
    CspAssignment,
    CspInputError,
    CspProblem,
    SessionVariable,
    WeeklySlot,
    WholePlanManifest,
)


class WholePlanVerificationStatus(str, Enum):
    """Outcome for one schedule, not exhaustive allocation search."""

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    INCOMPLETE = "incomplete"


class ConstraintCheckStatus(str, Enum):
    """Whether one declared verification predicate passed or ran."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"


@dataclass(frozen=True, slots=True)
class WholePlanVerificationPolicy:
    """Versioned identities for semantics used by one verification run."""

    version: str
    deadline_semantics_identity: str
    ordering_semantics_identity: str
    slot_catalog_identity: str
    capacity_source_identity: str
    slot_identity_semantics: str
    weekly_capacity: int
    policy_identity: str = field(init=False)

    def __post_init__(self) -> None:
        labels = (
            self.version,
            self.deadline_semantics_identity,
            self.ordering_semantics_identity,
            self.slot_catalog_identity,
            self.capacity_source_identity,
            self.slot_identity_semantics,
        )
        if any(not isinstance(value, str) or not value for value in labels):
            raise CspInputError("verification policy identities must be non-empty")
        if (
            isinstance(self.weekly_capacity, bool)
            or not isinstance(self.weekly_capacity, int)
            or self.weekly_capacity < 0
        ):
            raise CspInputError("verification policy capacity must be non-negative")
        payload = {
            "version": self.version,
            "deadline_semantics": self.deadline_semantics_identity,
            "ordering_semantics": self.ordering_semantics_identity,
            "slot_catalog": self.slot_catalog_identity,
            "capacity_source": self.capacity_source_identity,
            "weekly_capacity": self.weekly_capacity,
            "slot_identity_semantics": self.slot_identity_semantics,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        object.__setattr__(self, "policy_identity", hashlib.sha256(encoded).hexdigest())


@dataclass(frozen=True, slots=True, order=True)
class WholePlanSessionPlacement:
    """Canonical slot placement for one manifest session."""

    session_index: int
    slot_id: str
    week_index: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.session_index, bool)
            or not isinstance(self.session_index, int)
            or self.session_index < 0
        ):
            raise CspInputError("verified session index must be non-negative")
        if not isinstance(self.slot_id, str) or not self.slot_id:
            raise CspInputError("verified slot identity must be non-empty")
        if (
            isinstance(self.week_index, bool)
            or not isinstance(self.week_index, int)
            or self.week_index < 0
        ):
            raise CspInputError("verified week index must be non-negative")


@dataclass(frozen=True, slots=True)
class AssignmentSchedule:
    """Schedule projection retaining even zero-session assignments."""

    assignment_identity: str
    sessions: tuple[WholePlanSessionPlacement, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.assignment_identity, str) or not (
            self.assignment_identity
        ):
            raise CspInputError("schedule assignment identity must be non-empty")
        if not isinstance(self.sessions, tuple) or any(
            not isinstance(session, WholePlanSessionPlacement)
            for session in self.sessions
        ):
            raise CspInputError("assignment sessions must be an immutable tuple")
        indexes = tuple(session.session_index for session in self.sessions)
        if indexes != tuple(sorted(set(indexes))):
            raise CspInputError("assignment session indexes must be unique and sorted")


@dataclass(frozen=True, slots=True)
class ConstraintCheckEvidence:
    """Outcome and scope for one required or additional hard check."""

    constraint_id: str
    scope: tuple[SessionVariable, ...]
    status: ConstraintCheckStatus
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.constraint_id, str) or not self.constraint_id:
            raise CspInputError("verification check needs a constraint identity")
        if not isinstance(self.scope, tuple) or any(
            not isinstance(variable, SessionVariable) for variable in self.scope
        ):
            raise CspInputError("verification check scope must be immutable")
        if len(set(self.scope)) != len(self.scope):
            raise CspInputError("verification check scope cannot repeat variables")
        if not isinstance(self.status, ConstraintCheckStatus):
            raise CspInputError("verification check status is invalid")
        if self.status is ConstraintCheckStatus.FAILED:
            if not isinstance(self.failure_reason, str) or not self.failure_reason:
                raise CspInputError("failed checks need a reason")
        elif self.failure_reason is not None:
            raise CspInputError("only failed checks may carry a failure reason")


@dataclass(frozen=True, slots=True, init=False)
class WholePlanVerificationResult:
    """Provenance for verification of one concrete whole-plan schedule.

    ``INFEASIBLE`` means this complete schedule failed one or more checks. It
    does not establish that the whole allocation has no feasible schedule.
    ``INCOMPLETE`` means the schedule or its requested identity/policy could
    not be fully verified.
    """

    whole_plan_manifest: WholePlanManifest
    whole_plan_identity: str
    expected_whole_plan_identity: str | None
    problem_identity: str
    expected_problem_identity: str | None
    status: WholePlanVerificationStatus
    schedule: tuple[AssignmentSchedule, ...]
    schedule_identity: str
    verification_identity: str
    verification_policy_identity: str
    verification_policy_version: str
    constraint_checks: tuple[ConstraintCheckEvidence, ...]
    rejection_reasons: tuple[str, ...]

    def __init__(self) -> None:
        raise TypeError("WholePlanVerificationResult must come from its verifier")


def verify_whole_plan(
    problem: CspProblem,
    assignment: CspAssignment,
    policy: WholePlanVerificationPolicy,
    *,
    expected_whole_plan_identity: str | None = None,
    expected_problem_identity: str | None = None,
) -> WholePlanVerificationResult:
    """Verify one schedule against every declared CSP constraint and identity."""
    if not isinstance(problem, CspProblem):
        raise CspInputError("whole-plan verification requires a typed CSP problem")
    if not isinstance(assignment, CspAssignment):
        raise CspInputError("whole-plan verification requires a CSP assignment")
    if not isinstance(policy, WholePlanVerificationPolicy):
        raise CspInputError("whole-plan verification requires an explicit policy")
    for name, identity in (
        ("expected_whole_plan_identity", expected_whole_plan_identity),
        ("expected_problem_identity", expected_problem_identity),
    ):
        if identity is not None and (not isinstance(identity, str) or not identity):
            raise CspInputError(f"{name} must be a non-empty string or None")

    placements = assignment.placements
    expected_variables = set(problem.variables)
    placed_variables = set(placements)
    missing = expected_variables - placed_variables
    unknown = placed_variables - expected_variables
    domain_by_variable = {
        domain.variable: set(domain.values) for domain in problem.domains
    }
    out_of_domain = {
        variable
        for variable, value in placements.items()
        if (
            variable not in domain_by_variable
            or value not in domain_by_variable[variable]
        )
    }

    reasons: list[str] = []
    if missing:
        reasons.append("assignment_missing_variables")
    if unknown:
        reasons.append("assignment_contains_unknown_variables")
    if out_of_domain:
        reasons.append("assignment_contains_out_of_domain_values")
    if (
        expected_whole_plan_identity is not None
        and expected_whole_plan_identity != problem.whole_plan_identity
    ):
        reasons.append("whole_plan_identity_mismatch")
    if (
        expected_problem_identity is not None
        and expected_problem_identity != problem.problem_identity
    ):
        reasons.append("problem_identity_mismatch")
    if policy.weekly_capacity != problem.weekly_capacity:
        reasons.append("verification_capacity_mismatch")
    if (
        policy.ordering_semantics_identity
        != problem.ordering_policy.configuration_identity
    ):
        reasons.append("verification_ordering_policy_mismatch")

    checks: list[ConstraintCheckEvidence] = [
        ConstraintCheckEvidence(
            "assignment_coverage",
            tuple(sorted(problem.variables)),
            (
                ConstraintCheckStatus.PASSED
                if not missing and not unknown
                else ConstraintCheckStatus.FAILED
            ),
            "assignment_variable_coverage_mismatch" if missing or unknown else None,
        ),
        ConstraintCheckEvidence(
            "assignment_domains",
            tuple(sorted(problem.variables)),
            (
                ConstraintCheckStatus.FAILED
                if out_of_domain
                else ConstraintCheckStatus.PASSED
                if not missing and not unknown
                else ConstraintCheckStatus.NOT_RUN
            ),
            "assignment_value_outside_domain" if out_of_domain else None,
        ),
    ]
    for constraint in problem.constraints:
        in_scope = set(constraint.scope).issubset(placed_variables)
        valid_scope_values = not any(
            variable in out_of_domain for variable in constraint.scope
        )
        if not in_scope or not valid_scope_values:
            checks.append(
                ConstraintCheckEvidence(
                    constraint.constraint_id,
                    constraint.scope,
                    ConstraintCheckStatus.NOT_RUN,
                )
            )
            continue
        passed = constraint.is_satisfied(assignment)
        checks.append(
            ConstraintCheckEvidence(
                constraint.constraint_id,
                constraint.scope,
                ConstraintCheckStatus.PASSED
                if passed
                else ConstraintCheckStatus.FAILED,
                None if passed else "constraint_violation",
            )
        )
        if not passed:
            reasons.append(f"constraint_failed:{constraint.constraint_id}")

    if any(check.status is ConstraintCheckStatus.FAILED for check in checks) and not (
        missing or unknown or out_of_domain
    ):
        complete_schedule_failed = True
    else:
        complete_schedule_failed = bool(unknown or out_of_domain)

    identity_mismatch = any(
        reason.endswith("identity_mismatch") or reason.startswith("verification_")
        for reason in reasons
    )
    if missing or unknown or identity_mismatch:
        status = WholePlanVerificationStatus.INCOMPLETE
    elif complete_schedule_failed:
        status = WholePlanVerificationStatus.INFEASIBLE
    else:
        status = WholePlanVerificationStatus.FEASIBLE

    schedule = _partition_schedule(problem.manifest, placements)
    schedule_identity = _schedule_identity(schedule)
    rejection_reasons = tuple(sorted(set(reasons)))
    verification_identity = _verification_identity(
        whole_plan_identity=problem.whole_plan_identity,
        expected_whole_plan_identity=expected_whole_plan_identity,
        problem_identity=problem.problem_identity,
        expected_problem_identity=expected_problem_identity,
        schedule_identity=schedule_identity,
        policy_identity=policy.policy_identity,
        policy_version=policy.version,
        status=status,
        rejection_reasons=rejection_reasons,
        checks=tuple(checks),
    )
    return _create_result(
        manifest=problem.manifest,
        expected_whole_plan_identity=expected_whole_plan_identity,
        problem_identity=problem.problem_identity,
        expected_problem_identity=expected_problem_identity,
        status=status,
        schedule=schedule,
        schedule_identity=schedule_identity,
        verification_identity=verification_identity,
        policy=policy,
        checks=tuple(checks),
        rejection_reasons=rejection_reasons,
    )


def _partition_schedule(
    manifest: WholePlanManifest,
    placements: dict[SessionVariable, WeeklySlot],
) -> tuple[AssignmentSchedule, ...]:
    by_assignment: dict[str, list[WholePlanSessionPlacement]] = {
        item.assignment_identity: [] for item in manifest.canonical_assignments
    }
    for variable, slot in placements.items():
        if variable.assignment_id in by_assignment:
            by_assignment[variable.assignment_id].append(
                WholePlanSessionPlacement(
                    variable.session_index, slot.slot_id, slot.week_index
                )
            )
    return tuple(
        AssignmentSchedule(
            item.assignment_identity,
            tuple(sorted(by_assignment[item.assignment_identity])),
        )
        for item in manifest.canonical_assignments
    )


def _schedule_identity(schedule: tuple[AssignmentSchedule, ...]) -> str:
    payload = [
        (
            assignment.assignment_identity,
            [
                (session.session_index, session.slot_id, session.week_index)
                for session in assignment.sessions
            ],
        )
        for assignment in schedule
    ]
    encoded = json.dumps(payload, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _verification_identity(
    *,
    whole_plan_identity: str,
    expected_whole_plan_identity: str | None,
    problem_identity: str,
    expected_problem_identity: str | None,
    schedule_identity: str,
    policy_identity: str,
    policy_version: str,
    status: WholePlanVerificationStatus,
    rejection_reasons: tuple[str, ...],
    checks: tuple[ConstraintCheckEvidence, ...],
) -> str:
    payload = {
        "whole_plan_identity": whole_plan_identity,
        "expected_whole_plan_identity": expected_whole_plan_identity,
        "problem_identity": problem_identity,
        "expected_problem_identity": expected_problem_identity,
        "schedule_identity": schedule_identity,
        "policy_identity": policy_identity,
        "policy_version": policy_version,
        "status": status.value,
        "rejection_reasons": sorted(set(rejection_reasons)),
        "checks": [
            (
                check.constraint_id,
                tuple((v.assignment_id, v.session_index) for v in check.scope),
                check.status.value,
                check.failure_reason,
            )
            for check in sorted(checks, key=lambda item: item.constraint_id)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _create_result(
    *,
    manifest: WholePlanManifest,
    expected_whole_plan_identity: str | None,
    problem_identity: str,
    expected_problem_identity: str | None,
    status: WholePlanVerificationStatus,
    schedule: tuple[AssignmentSchedule, ...],
    schedule_identity: str,
    verification_identity: str,
    policy: WholePlanVerificationPolicy,
    checks: tuple[ConstraintCheckEvidence, ...],
    rejection_reasons: tuple[str, ...],
) -> WholePlanVerificationResult:
    if status is WholePlanVerificationStatus.FEASIBLE and (
        rejection_reasons
        or any(check.status is not ConstraintCheckStatus.PASSED for check in checks)
    ):
        raise CspInputError("feasible verification requires every check to pass")
    if status is WholePlanVerificationStatus.INFEASIBLE and not rejection_reasons:
        raise CspInputError("infeasible schedule verification needs rejection evidence")
    record = object.__new__(WholePlanVerificationResult)
    values = {
        "whole_plan_manifest": manifest,
        "whole_plan_identity": manifest.canonical_identity,
        "expected_whole_plan_identity": expected_whole_plan_identity,
        "problem_identity": problem_identity,
        "expected_problem_identity": expected_problem_identity,
        "status": status,
        "schedule": schedule,
        "schedule_identity": schedule_identity,
        "verification_identity": verification_identity,
        "verification_policy_identity": policy.policy_identity,
        "verification_policy_version": policy.version,
        "constraint_checks": checks,
        "rejection_reasons": rejection_reasons,
    }
    for name, value in values.items():
        object.__setattr__(record, name, value)
    return record


def validate_whole_plan_verification_result(
    result: WholePlanVerificationResult,
    *,
    expected_whole_plan_identity: str,
    expected_problem_identity: str,
    expected_policy_identity: str,
    expected_policy_version: str,
) -> None:
    """Validate a feasible verification's stored identities and evidence.

    This is an integrity and contract check, not a security boundary against
    deliberate Python object fabrication.
    """
    if not isinstance(result, WholePlanVerificationResult):
        raise CspInputError("a WholePlanVerificationResult is required")
    if (
        not isinstance(expected_whole_plan_identity, str)
        or not expected_whole_plan_identity
        or not isinstance(expected_problem_identity, str)
        or not expected_problem_identity
        or not isinstance(expected_policy_identity, str)
        or not expected_policy_identity
        or not isinstance(expected_policy_version, str)
        or not expected_policy_version
    ):
        raise CspInputError("expected verification identities must be non-empty")
    if result.status is not WholePlanVerificationStatus.FEASIBLE:
        raise CspInputError("only feasible whole-plan verification can be consumed")
    if (
        result.whole_plan_identity != expected_whole_plan_identity
        or result.expected_whole_plan_identity != expected_whole_plan_identity
        or result.whole_plan_manifest.canonical_identity != result.whole_plan_identity
    ):
        raise CspInputError("whole-plan verification identity mismatch")
    if (
        result.problem_identity != expected_problem_identity
        or result.expected_problem_identity != expected_problem_identity
    ):
        raise CspInputError("CSP problem verification identity mismatch")
    if (
        not result.verification_policy_identity
        or not result.verification_policy_version
    ):
        raise CspInputError("verification policy provenance is missing")
    if (
        result.verification_policy_identity != expected_policy_identity
        or result.verification_policy_version != expected_policy_version
    ):
        raise CspInputError("verification policy identity mismatch")
    if result.rejection_reasons or any(
        check.status is not ConstraintCheckStatus.PASSED
        for check in result.constraint_checks
    ):
        raise CspInputError("feasible verification contains failed or skipped checks")

    schedule = result.schedule
    assignments = result.whole_plan_manifest.canonical_assignments
    if tuple(item.assignment_identity for item in schedule) != tuple(
        item.assignment_identity for item in assignments
    ):
        raise CspInputError("whole-plan schedule assignment partition is incomplete")
    all_slots: set[str] = set()
    variables: list[SessionVariable] = []
    for plan, partition in zip(assignments, schedule, strict=True):
        expected_indexes = tuple(range(plan.session_count))
        actual_indexes = tuple(session.session_index for session in partition.sessions)
        if actual_indexes != expected_indexes:
            raise CspInputError("whole-plan schedule session coverage is incomplete")
        for session in partition.sessions:
            if session.slot_id in all_slots:
                raise CspInputError("whole-plan schedule reuses a weekly slot")
            all_slots.add(session.slot_id)
            variables.append(
                SessionVariable(plan.assignment_identity, session.session_index)
            )

    check_ids = tuple(check.constraint_id for check in result.constraint_checks)
    if len(check_ids) != len(set(check_ids)):
        raise CspInputError("verification constraint evidence contains duplicate IDs")
    expected_required = {
        "assignment_coverage",
        "assignment_domains",
        "weekly_capacity",
        *{
            f"deadline_before_cutoff:{variable.assignment_id}:{variable.session_index}"
            for variable in variables
        },
        *{
            f"weekly_slot_unique:{first.assignment_id}:{first.session_index}:"
            f"{second.assignment_id}:{second.session_index}"
            for index, first in enumerate(sorted(variables))
            for second in sorted(variables)[index + 1 :]
        },
    }
    if not expected_required.issubset(check_ids):
        raise CspInputError("required whole-plan verification evidence is missing")
    for plan in assignments:
        plan_variables = [
            variable
            for variable in variables
            if variable.assignment_id == plan.assignment_identity
        ]
        if len(plan_variables) > 1 and not any(
            check_id.startswith(f"same_assignment_order:{plan.assignment_identity}:")
            for check_id in check_ids
        ):
            raise CspInputError("same-assignment ordering evidence is missing")

    if result.schedule_identity != _schedule_identity(schedule):
        raise CspInputError("whole-plan schedule identity mismatch")
    expected_verification_identity = _verification_identity(
        whole_plan_identity=result.whole_plan_identity,
        expected_whole_plan_identity=result.expected_whole_plan_identity,
        problem_identity=result.problem_identity,
        expected_problem_identity=result.expected_problem_identity,
        schedule_identity=result.schedule_identity,
        policy_identity=result.verification_policy_identity,
        policy_version=result.verification_policy_version,
        status=result.status,
        rejection_reasons=result.rejection_reasons,
        checks=result.constraint_checks,
    )
    if result.verification_identity != expected_verification_identity:
        raise CspInputError("whole-plan verification provenance identity mismatch")
