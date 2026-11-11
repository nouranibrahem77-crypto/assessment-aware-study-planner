"""Typed immutable contracts for whole-plan CSP instances and outcomes."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class CspInputError(ValueError):
    """Raised when a CSP manifest, problem, or assignment is malformed."""


@dataclass(frozen=True, slots=True, order=True)
class AssignmentCandidateIdentity:
    """Assignment-scoped identity corresponding to the Search-NN adapter."""

    student_identity: str
    course_id: str
    assignment_identity: str
    canonical_allocation_identity: tuple[int, int, int, int]
    base_feature_context_identity: str
    mapping_scenario_identity: str
    canonical_identity: str = field(init=False)

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (
                self.student_identity,
                self.course_id,
                self.assignment_identity,
                self.base_feature_context_identity,
                self.mapping_scenario_identity,
            )
        ):
            raise CspInputError("candidate identity fields must be non-empty strings")
        allocation = self.canonical_allocation_identity
        if (
            not isinstance(allocation, tuple)
            or len(allocation) != 4
            or any(
                isinstance(count, bool) or not isinstance(count, int) or count < 0
                for count in allocation
            )
        ):
            raise CspInputError("candidate allocation must be four non-negative counts")
        payload = {
            "identity_version": 1,
            "student": self.student_identity,
            "course": self.course_id,
            "assignment": self.assignment_identity,
            "allocation": allocation,
            "feature_context": self.base_feature_context_identity,
            "mapping_scenario": self.mapping_scenario_identity,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        object.__setattr__(
            self, "canonical_identity", hashlib.sha256(encoded).hexdigest()
        )


@dataclass(frozen=True, slots=True)
class AssignmentPlan:
    """One assignment in the plan, including assignments with zero sessions."""

    assignment_identity: str
    candidate_identity: AssignmentCandidateIdentity
    base_feature_context_identity: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.assignment_identity, str)
            or not self.assignment_identity
        ):
            raise CspInputError("assignment identity must be a non-empty string")
        if not isinstance(self.candidate_identity, AssignmentCandidateIdentity):
            raise CspInputError("assignment plan needs a candidate identity")
        if self.candidate_identity.assignment_identity != self.assignment_identity:
            raise CspInputError("assignment and candidate identities do not match")
        if (
            not isinstance(self.base_feature_context_identity, str)
            or not self.base_feature_context_identity
            or self.candidate_identity.base_feature_context_identity
            != self.base_feature_context_identity
        ):
            raise CspInputError("assignment feature context identities do not match")

    @property
    def session_count(self) -> int:
        """Return the total allocated sessions for this assignment."""
        return sum(self.candidate_identity.canonical_allocation_identity)


@dataclass(frozen=True, slots=True)
class WholePlanManifest:
    """Canonical identity and complete three-assignment Search proposal."""

    student_identity: str
    course_identity: str
    assignments: tuple[AssignmentPlan, ...]
    mapping_scenario_identity: str
    canonical_identity: str = field(init=False)

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (
                self.student_identity,
                self.course_identity,
                self.mapping_scenario_identity,
            )
        ):
            raise CspInputError("whole-plan identity fields must be non-empty strings")
        if not isinstance(self.assignments, tuple) or any(
            not isinstance(assignment, AssignmentPlan)
            for assignment in self.assignments
        ):
            raise CspInputError("assignments must be an immutable tuple of plans")
        if len(self.assignments) != 3:
            raise CspInputError(
                "a P1 whole-plan manifest must contain three assignments"
            )
        ids = tuple(item.assignment_identity for item in self.assignments)
        if len(set(ids)) != 3:
            raise CspInputError("whole-plan assignment identities must be unique")
        for item in self.assignments:
            candidate = item.candidate_identity
            if candidate.student_identity != self.student_identity:
                raise CspInputError("assignment candidate belongs to another student")
            if candidate.course_id != self.course_id:
                raise CspInputError("assignment candidate belongs to another course")
            if candidate.assignment_identity != item.assignment_identity:
                raise CspInputError(
                    "assignment candidate belongs to another assignment"
                )
            if candidate.mapping_scenario_identity != self.mapping_scenario_identity:
                raise CspInputError(
                    "assignment candidate uses another mapping scenario"
                )
        ordered = sorted(self.assignments, key=lambda item: item.assignment_identity)
        payload = {
            "identity_version": 2,
            "student": self.student_identity,
            "course": self.course_id,
            "scenario": self.mapping_scenario_identity,
            "assignments": [
                {
                    "id": item.assignment_identity,
                    "allocation": item.candidate_identity.canonical_allocation_identity,
                    "features": item.base_feature_context_identity,
                }
                for item in ordered
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        object.__setattr__(
            self, "canonical_identity", hashlib.sha256(encoded).hexdigest()
        )

    @property
    def course_id(self) -> str:
        """Return the course identity under the explicit manifest field name."""
        return self.course_identity

    @property
    def canonical_assignments(self) -> tuple[AssignmentPlan, ...]:
        """Return assignments in stable identity order."""
        return tuple(
            sorted(self.assignments, key=lambda item: item.assignment_identity)
        )

    def assignment(self, assignment_identity: str) -> AssignmentPlan:
        """Return the unique manifest entry for an assignment."""
        for item in self.assignments:
            if item.assignment_identity == assignment_identity:
                return item
        raise CspInputError(f"assignment is not in manifest: {assignment_identity}")


@dataclass(frozen=True, slots=True, order=True)
class SessionVariable:
    """One study session belonging to an assignment in a whole-plan CSP."""

    assignment_id: str
    session_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.assignment_id, str) or not self.assignment_id:
            raise CspInputError("assignment_id must be a non-empty string")
        if (
            isinstance(self.session_index, bool)
            or not isinstance(self.session_index, int)
            or self.session_index < 0
        ):
            raise CspInputError("session_index must be a non-negative integer")


@dataclass(frozen=True, slots=True, order=True)
class WeeklySlot:
    """Opaque slot identity with a caller-supplied week ordinal."""

    slot_id: str
    week_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.slot_id, str) or not self.slot_id:
            raise CspInputError("slot_id must be a non-empty string")
        if (
            isinstance(self.week_index, bool)
            or not isinstance(self.week_index, int)
            or self.week_index < 0
        ):
            raise CspInputError("week_index must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class VariableDomain:
    """Finite immutable domain for one session variable."""

    variable: SessionVariable
    values: tuple[WeeklySlot, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.variable, SessionVariable):
            raise CspInputError("domain variable is invalid")
        if not isinstance(self.values, tuple) or any(
            not isinstance(value, WeeklySlot) for value in self.values
        ):
            raise CspInputError("domain values must be a tuple of WeeklySlot values")
        if len(set(self.values)) != len(self.values):
            raise CspInputError("a domain cannot contain duplicate slots")


@dataclass(frozen=True, slots=True, order=True)
class AssignmentEntry:
    """One variable-to-slot choice in a partial or complete assignment."""

    variable: SessionVariable
    value: WeeklySlot

    def __post_init__(self) -> None:
        if not isinstance(self.variable, SessionVariable):
            raise CspInputError("assignment entry variable is invalid")
        if not isinstance(self.value, WeeklySlot):
            raise CspInputError("assignment entry slot is invalid")


@dataclass(frozen=True, slots=True)
class CspAssignment:
    """Immutable set of placements; completeness is relative to a problem."""

    entries: tuple[AssignmentEntry, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple) or any(
            not isinstance(entry, AssignmentEntry) for entry in self.entries
        ):
            raise CspInputError("assignment entries must be an immutable tuple")
        variables = tuple(entry.variable for entry in self.entries)
        if len(set(variables)) != len(variables):
            raise CspInputError("an assignment cannot place a variable twice")

    @property
    def placements(self) -> dict[SessionVariable, WeeklySlot]:
        """Return a fresh lookup dictionary for constraint evaluation."""
        return {entry.variable: entry.value for entry in self.entries}


@dataclass(frozen=True, slots=True, order=True)
class SessionPair:
    """Ordered pair of sessions explicitly required by an ordering policy."""

    earlier: SessionVariable
    later: SessionVariable


class SessionOrderingPolicy(Protocol):
    """Project-approved ordering relation; no default semantics are assumed."""

    @property
    def policy_identity(self) -> str:
        """Stable identity/version for the configured semantics."""

    def required_pairs(
        self, sessions: tuple[SessionVariable, ...]
    ) -> tuple[SessionPair, ...]:
        """Derive every same-assignment relationship required by this policy."""

    def is_ordered(self, earlier: WeeklySlot, later: WeeklySlot) -> bool:
        """Check two slot values using this policy's explicit semantics."""


@dataclass(frozen=True, slots=True)
class OrderingPolicySnapshot:
    """Immutable finite snapshot of an ordering policy for one CSP problem."""

    policy_identity: str
    sessions: tuple[SessionVariable, ...]
    required_session_pairs: tuple[SessionPair, ...]
    slot_universe: frozenset[WeeklySlot]
    ordered_slot_pairs: frozenset[tuple[WeeklySlot, WeeklySlot]]
    configuration_identity: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.policy_identity, str) or not self.policy_identity:
            raise CspInputError("ordering snapshot needs a stable policy identity")
        if not isinstance(self.sessions, tuple) or any(
            not isinstance(item, SessionVariable) for item in self.sessions
        ):
            raise CspInputError("ordering snapshot sessions must be immutable")
        if not isinstance(self.required_session_pairs, tuple) or any(
            not isinstance(item, SessionPair) for item in self.required_session_pairs
        ):
            raise CspInputError("ordering snapshot pairs must be immutable")
        if not isinstance(self.slot_universe, frozenset) or any(
            not isinstance(item, WeeklySlot) for item in self.slot_universe
        ):
            raise CspInputError("ordering snapshot slot universe must be immutable")
        if not isinstance(self.ordered_slot_pairs, frozenset) or any(
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(not isinstance(slot, WeeklySlot) for slot in pair)
            for pair in self.ordered_slot_pairs
        ):
            raise CspInputError("ordering snapshot relation must be immutable")
        if any(
            first not in self.slot_universe or second not in self.slot_universe
            for first, second in self.ordered_slot_pairs
        ):
            raise CspInputError("ordering snapshot relation uses an unknown slot")
        session_set = set(self.sessions)
        if any(
            pair.earlier not in session_set
            or pair.later not in session_set
            or pair.earlier.assignment_id != pair.later.assignment_id
            or pair.earlier.session_index >= pair.later.session_index
            for pair in self.required_session_pairs
        ):
            raise CspInputError("ordering snapshot contains an invalid session pair")
        object.__setattr__(
            self,
            "required_session_pairs",
            tuple(
                sorted(
                    self.required_session_pairs,
                    key=lambda item: (item.earlier, item.later),
                )
            ),
        )
        payload = {
            "policy_identity": self.policy_identity,
            "sessions": [
                (item.assignment_id, item.session_index) for item in self.sessions
            ],
            "required_pairs": [
                (
                    (pair.earlier.assignment_id, pair.earlier.session_index),
                    (pair.later.assignment_id, pair.later.session_index),
                )
                for pair in sorted(
                    self.required_session_pairs,
                    key=lambda item: (item.earlier, item.later),
                )
            ],
            "slots": sorted(
                (slot.slot_id, slot.week_index) for slot in self.slot_universe
            ),
            "ordered_slot_pairs": sorted(
                (
                    (first.slot_id, first.week_index),
                    (second.slot_id, second.week_index),
                )
                for first, second in self.ordered_slot_pairs
            ),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        object.__setattr__(
            self, "configuration_identity", hashlib.sha256(encoded).hexdigest()
        )

    def required_pairs(
        self, sessions: tuple[SessionVariable, ...]
    ) -> tuple[SessionPair, ...]:
        """Return snapshotted required pairs for this exact session manifest."""
        if sessions != self.sessions:
            raise CspInputError("ordering snapshot belongs to another session manifest")
        return self.required_session_pairs

    def is_ordered(self, earlier: WeeklySlot, later: WeeklySlot) -> bool:
        """Apply the immutable relation captured over this problem's slot universe."""
        if earlier not in self.slot_universe or later not in self.slot_universe:
            raise CspInputError("ordering query uses a slot outside the snapshot")
        return (earlier, later) in self.ordered_slot_pairs


def _snapshot_ordering_policy(
    policy: SessionOrderingPolicy | OrderingPolicySnapshot,
    sessions: tuple[SessionVariable, ...],
    slots: tuple[WeeklySlot, ...],
) -> OrderingPolicySnapshot:
    """Capture all policy behavior over this finite problem before solving."""
    slot_universe = frozenset(slots)
    if isinstance(policy, OrderingPolicySnapshot):
        if policy.sessions != sessions or not slot_universe.issubset(
            policy.slot_universe
        ):
            raise CspInputError("ordering snapshot does not cover this CSP problem")
        return policy
    policy_identity = getattr(policy, "policy_identity", None)
    required_pairs_fn = getattr(policy, "required_pairs", None)
    is_ordered_fn = getattr(policy, "is_ordered", None)
    if (
        not isinstance(policy_identity, str)
        or not policy_identity
        or not callable(required_pairs_fn)
        or not callable(is_ordered_fn)
    ):
        raise CspInputError("ordering policy is not safely snapshot-capable")
    try:
        required_pairs = required_pairs_fn(sessions)
    except Exception as error:
        raise CspInputError("ordering policy failed to derive session pairs") from error
    if not isinstance(required_pairs, tuple) or any(
        not isinstance(pair, SessionPair) for pair in required_pairs
    ):
        raise CspInputError("ordering policy must return immutable SessionPair values")
    if len(set(required_pairs)) != len(required_pairs):
        raise CspInputError("ordering policy returned duplicate session pairs")
    variables = set(sessions)
    for pair in required_pairs:
        if (
            pair.earlier not in variables
            or pair.later not in variables
            or pair.earlier.assignment_id != pair.later.assignment_id
            or pair.earlier.session_index >= pair.later.session_index
        ):
            raise CspInputError("ordering policy returned reversed or unrelated pairs")
    paired = {
        variable for pair in required_pairs for variable in (pair.earlier, pair.later)
    }
    by_assignment: dict[str, list[SessionVariable]] = {}
    for variable in sessions:
        by_assignment.setdefault(variable.assignment_id, []).append(variable)
    if any(
        len(items) > 1 and not set(items).issubset(paired)
        for items in by_assignment.values()
    ):
        raise CspInputError("ordering policy omitted one or more sessions")

    ordered_slots = tuple(sorted(slot_universe))
    ordered_pairs: set[tuple[WeeklySlot, WeeklySlot]] = set()
    try:
        for earlier in ordered_slots:
            for later in ordered_slots:
                result = is_ordered_fn(earlier, later)
                if not isinstance(result, bool):
                    raise CspInputError("ordering policy must return boolean results")
                if result:
                    ordered_pairs.add((earlier, later))
    except CspInputError:
        raise
    except Exception as error:
        raise CspInputError(
            "ordering policy failed to define its slot relation"
        ) from error
    return OrderingPolicySnapshot(
        policy_identity=policy_identity,
        sessions=sessions,
        required_session_pairs=required_pairs,
        slot_universe=slot_universe,
        ordered_slot_pairs=frozenset(ordered_pairs),
    )


class CspConstraint(Protocol):
    """General constraint over an explicit finite variable scope."""

    @property
    def constraint_id(self) -> str:
        """Stable identifier unique within the CSP problem."""

    @property
    def configuration_identity(self) -> str:
        """Stable identity for this constraint's parameters and semantics."""

    @property
    def scope(self) -> tuple[SessionVariable, ...]:
        """Variables whose values can affect this constraint."""

    def is_satisfied(self, assignment: CspAssignment) -> bool:
        """Check assigned scope values, allowing unassigned future variables."""


class BinaryConstraint(ABC):
    """Binary constraint with direct AC-3 support and domain revision methods."""

    @property
    @abstractmethod
    def scope(self) -> tuple[SessionVariable, SessionVariable]:
        """Return exactly the two distinct variables in this relation."""

    @abstractmethod
    def is_compatible(
        self,
        first_variable: SessionVariable,
        first_value: WeeklySlot,
        second_variable: SessionVariable,
        second_value: WeeklySlot,
    ) -> bool:
        """Check a candidate value pair without constructing an assignment."""

    def has_support(
        self,
        variable: SessionVariable,
        value: WeeklySlot,
        neighbor_domain: tuple[WeeklySlot, ...],
    ) -> bool:
        """Return whether value has support in the opposite variable's domain."""
        neighbor = self._neighbor(variable)
        return any(
            self.is_compatible(variable, value, neighbor, candidate)
            for candidate in neighbor_domain
        )

    def revise(
        self,
        variable: SessionVariable,
        domain: tuple[WeeklySlot, ...],
        neighbor_domain: tuple[WeeklySlot, ...],
    ) -> tuple[WeeklySlot, ...]:
        """Keep only values with support in the neighboring domain."""
        self._neighbor(variable)
        return tuple(
            value
            for value in domain
            if self.has_support(variable, value, neighbor_domain)
        )

    def _neighbor(self, variable: SessionVariable) -> SessionVariable:
        if len(self.scope) != 2 or self.scope[0] == self.scope[1]:
            raise CspInputError(
                "binary constraint scope must contain two distinct variables"
            )
        if variable == self.scope[0]:
            return self.scope[1]
        if variable == self.scope[1]:
            return self.scope[0]
        raise CspInputError("variable is outside binary constraint scope")


class CspEvaluationStatus(str, Enum):
    """Local validation states; validation cannot establish global infeasibility."""

    BRANCH_REJECTED = "branch_rejected"
    INCOMPLETE = "incomplete"
    FEASIBLE = "feasible"


@dataclass(frozen=True, slots=True)
class _AssignmentValidationEvidence:
    """Private result of one exact assignment validation pass."""

    whole_plan_identity: str
    problem_identity: str
    assignment: CspAssignment
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True, init=False)
class CspEvaluation:
    """Intermediate branch/assignment validation, not a verification result."""

    whole_plan_identity: str
    problem_identity: str
    status: CspEvaluationStatus
    assignment: CspAssignment
    failed_checks: tuple[str, ...] = ()

    def __init__(self) -> None:
        raise TypeError("CspEvaluation records must come from the local validator")

    @classmethod
    def _from_validator(
        cls,
        *,
        problem: CspProblem,
        status: CspEvaluationStatus,
        assignment: CspAssignment,
        failed_checks: tuple[str, ...] = (),
        _evidence: _AssignmentValidationEvidence | None = None,
        _on_constraint_check: Callable[[], None] | None = None,
    ) -> CspEvaluation:
        """Create a locally validated record bound to the exact CSP problem."""
        if not isinstance(problem, CspProblem):
            raise CspInputError("evaluation requires a typed CSP problem")
        if not isinstance(status, CspEvaluationStatus):
            raise CspInputError("evaluation status is invalid")
        if not isinstance(assignment, CspAssignment):
            raise CspInputError("evaluation needs a typed assignment")
        if not isinstance(failed_checks, tuple) or any(
            not isinstance(item, str) or not item for item in failed_checks
        ):
            raise CspInputError("failed checks must be immutable identifiers")
        complete = len(assignment.placements) == len(problem.variables) and set(
            assignment.placements
        ) == set(problem.variables)
        if _evidence is None:
            actual_failures = _assignment_check_failures(
                problem, assignment, on_constraint_check=_on_constraint_check
            )
        else:
            if (
                not isinstance(_evidence, _AssignmentValidationEvidence)
                or _evidence.whole_plan_identity != problem.whole_plan_identity
                or _evidence.problem_identity != problem.problem_identity
                or _evidence.assignment != assignment
            ):
                raise CspInputError("validation evidence does not match its inputs")
            actual_failures = _evidence.failures
        if status is CspEvaluationStatus.FEASIBLE:
            if not complete or actual_failures or failed_checks:
                raise CspInputError(
                    "feasible evaluation requires a complete, consistent assignment"
                )
        elif status is CspEvaluationStatus.INCOMPLETE:
            if complete or actual_failures or failed_checks:
                raise CspInputError(
                    "incomplete evaluation must be a valid partial assignment"
                )
        elif status is CspEvaluationStatus.BRANCH_REJECTED:
            if not actual_failures or failed_checks != actual_failures:
                raise CspInputError(
                    "rejected branch checks must match its actual violations"
                )
        else:
            raise CspInputError("invalid CSP evaluation status")
        result = object.__new__(cls)
        object.__setattr__(result, "whole_plan_identity", problem.whole_plan_identity)
        object.__setattr__(result, "problem_identity", problem.problem_identity)
        object.__setattr__(result, "status", status)
        object.__setattr__(result, "assignment", assignment)
        object.__setattr__(result, "failed_checks", failed_checks)
        return result

    def __post_init__(self) -> None:
        if not self.whole_plan_identity:
            raise CspInputError("evaluation needs a whole-plan identity")
        if not isinstance(self.status, CspEvaluationStatus):
            raise CspInputError("evaluation status is invalid")
        if not isinstance(self.assignment, CspAssignment):
            raise CspInputError("evaluation needs a typed assignment")
        if not isinstance(self.failed_checks, tuple) or any(
            not isinstance(item, str) or not item for item in self.failed_checks
        ):
            raise CspInputError("failed checks must be immutable identifiers")
        if (
            self.status is CspEvaluationStatus.BRANCH_REJECTED
            and not self.failed_checks
        ):
            raise CspInputError("a rejected branch must identify failed checks")
        if (
            self.status is not CspEvaluationStatus.BRANCH_REJECTED
            and self.failed_checks
        ):
            raise CspInputError("only rejected branches may carry failed checks")


class CspSolverStatus(str, Enum):
    """Terminal solver outcomes; only exhaustive search may issue INFEASIBLE."""

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True, init=False)
class CspSolverOutcome:
    """Terminal result issued by a solver boundary, distinct from validation."""

    whole_plan_identity: str
    problem_identity: str
    status: CspSolverStatus
    evaluation: CspEvaluation | None
    exhaustive: bool
    reason: str | None

    def __init__(self) -> None:
        raise TypeError("solver outcomes must be created at the solver boundary")

    @classmethod
    def _from_solver(
        cls,
        *,
        problem: CspProblem,
        status: CspSolverStatus,
        exhaustive: bool,
        evaluation: CspEvaluation | None = None,
        reason: str | None = None,
        _on_constraint_check: Callable[[], None] | None = None,
    ) -> CspSolverOutcome:
        """Create a result; the factory is a convention, not a security boundary."""
        if not isinstance(problem, CspProblem):
            raise CspInputError("solver outcome requires a typed CSP problem")
        if not isinstance(status, CspSolverStatus) or not isinstance(exhaustive, bool):
            raise CspInputError("solver status and exhaustive flag are invalid")
        if status is CspSolverStatus.FEASIBLE:
            if (
                not isinstance(evaluation, CspEvaluation)
                or evaluation.status is not CspEvaluationStatus.FEASIBLE
                or evaluation.whole_plan_identity != problem.whole_plan_identity
                or evaluation.problem_identity != problem.problem_identity
                or evaluation.failed_checks
                or _assignment_check_failures(
                    problem,
                    evaluation.assignment,
                    require_complete=True,
                    on_constraint_check=_on_constraint_check,
                )
            ):
                raise CspInputError(
                    "feasible outcome needs a matching complete evaluation"
                )
        elif status is CspSolverStatus.INFEASIBLE:
            if (
                not exhaustive
                or evaluation is not None
                or not isinstance(reason, str)
                or not reason
            ):
                raise CspInputError("infeasibility requires exhaustive search")
        elif status is CspSolverStatus.INCOMPLETE:
            if (
                exhaustive
                or not isinstance(reason, str)
                or not reason
                or (
                    evaluation is not None
                    and (
                        not isinstance(evaluation, CspEvaluation)
                        or evaluation.status is CspEvaluationStatus.FEASIBLE
                        or evaluation.whole_plan_identity != problem.whole_plan_identity
                        or evaluation.problem_identity != problem.problem_identity
                        or (
                            evaluation.status is CspEvaluationStatus.INCOMPLETE
                            and _assignment_check_failures(
                                problem,
                                evaluation.assignment,
                                on_constraint_check=_on_constraint_check,
                            )
                        )
                    )
                )
            ):
                raise CspInputError(
                    "incomplete outcome needs a stop reason and non-exhaustive run"
                )
        else:
            raise CspInputError("invalid solver outcome status")
        result = object.__new__(cls)
        object.__setattr__(result, "whole_plan_identity", problem.whole_plan_identity)
        object.__setattr__(result, "problem_identity", problem.problem_identity)
        object.__setattr__(result, "status", status)
        object.__setattr__(result, "evaluation", evaluation)
        object.__setattr__(result, "exhaustive", exhaustive)
        object.__setattr__(result, "reason", reason)
        return result


@dataclass(frozen=True, slots=True)
class CspProblem:
    """Whole-plan CSP instance validated against its complete manifest."""

    manifest: WholePlanManifest
    variables: tuple[SessionVariable, ...]
    domains: tuple[VariableDomain, ...]
    constraints: tuple[CspConstraint, ...]
    weekly_capacity: int
    ordering_policy: SessionOrderingPolicy | OrderingPolicySnapshot
    problem_identity: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, WholePlanManifest):
            raise CspInputError("CSP problem requires a whole-plan manifest")
        if (
            isinstance(self.weekly_capacity, bool)
            or not isinstance(self.weekly_capacity, int)
            or self.weekly_capacity < 0
        ):
            raise CspInputError("weekly_capacity must be a non-negative integer")
        if not all(
            isinstance(values, tuple)
            for values in (self.variables, self.domains, self.constraints)
        ):
            raise CspInputError("variables, domains, and constraints must be tuples")
        if any(
            not isinstance(variable, SessionVariable) for variable in self.variables
        ):
            raise CspInputError("problem contains an invalid variable")
        if len(set(self.variables)) != len(self.variables):
            raise CspInputError("problem variables must be unique")
        expected = {
            SessionVariable(item.assignment_identity, index)
            for item in self.manifest.assignments
            for index in range(item.session_count)
        }
        if set(self.variables) != expected or len(self.variables) != len(expected):
            raise CspInputError(
                "variables do not match the whole-plan allocation manifest"
            )
        if any(not isinstance(domain, VariableDomain) for domain in self.domains):
            raise CspInputError("problem contains an invalid domain")
        domain_variables = tuple(domain.variable for domain in self.domains)
        if len(set(domain_variables)) != len(domain_variables):
            raise CspInputError("each variable must have exactly one domain")
        if set(domain_variables) != set(self.variables):
            raise CspInputError("domains must cover exactly the problem variables")
        self._validate_slot_catalog()
        ordered_variables = tuple(sorted(self.variables))
        slots = tuple(
            sorted({slot for domain in self.domains for slot in domain.values})
        )
        policy_snapshot = _snapshot_ordering_policy(
            self.ordering_policy, ordered_variables, slots
        )
        object.__setattr__(self, "ordering_policy", policy_snapshot)
        from study_planner.csp.constraints import SameAssignmentOrderingConstraint

        object.__setattr__(
            self,
            "constraints",
            tuple(
                SameAssignmentOrderingConstraint(item.pair, policy_snapshot)
                if isinstance(item, SameAssignmentOrderingConstraint)
                else item
                for item in self.constraints
            ),
        )
        self._validate_constraints()
        object.__setattr__(self, "problem_identity", self._canonical_problem_identity())

    @property
    def whole_plan_identity(self) -> str:
        """Stable canonical identity derived from the full manifest."""
        return self.manifest.canonical_identity

    def domain_for(self, variable: SessionVariable) -> tuple[WeeklySlot, ...]:
        """Return one variable's configured slot domain."""
        for domain in self.domains:
            if domain.variable == variable:
                return domain.values
        raise CspInputError(f"variable has no domain: {variable!r}")

    def _validate_slot_catalog(self) -> None:
        slot_weeks: dict[str, int] = {}
        for domain in self.domains:
            for slot in domain.values:
                previous = slot_weeks.setdefault(slot.slot_id, slot.week_index)
                if previous != slot.week_index:
                    raise CspInputError(
                        "a slot identity cannot belong to multiple weeks"
                    )

    def _validate_constraints(self) -> None:
        from study_planner.csp.constraints import (
            DeadlineConstraint,
            SameAssignmentOrderingConstraint,
            UniqueSlotConstraint,
            WeeklyCapacityConstraint,
        )

        ids = tuple(getattr(item, "constraint_id", None) for item in self.constraints)
        if any(not isinstance(item, str) or not item for item in ids):
            raise CspInputError("constraints need non-empty stable identifiers")
        if len(set(ids)) != len(ids):
            raise CspInputError("constraint identifiers must be unique")
        variables = set(self.variables)
        for constraint in self.constraints:
            scope = getattr(constraint, "scope", None)
            if not isinstance(scope, tuple) or any(
                not isinstance(variable, SessionVariable) for variable in scope
            ):
                raise CspInputError(
                    "constraint scope must be a tuple of session variables"
                )
            if len(set(scope)) != len(scope) or not set(scope).issubset(variables):
                raise CspInputError(
                    "constraint scope is duplicate or outside this problem"
                )
            if not scope and variables:
                raise CspInputError(
                    "non-empty CSP problems cannot have empty constraint scopes"
                )
            if not callable(getattr(constraint, "is_satisfied", None)):
                raise CspInputError("constraint must support partial assignment checks")
            config_id = getattr(constraint, "configuration_identity", None)
            if not isinstance(config_id, str) or not config_id:
                raise CspInputError("constraints need a stable configuration identity")
            if isinstance(constraint, BinaryConstraint) and (
                len(scope) != 2 or scope[0] == scope[1]
            ):
                raise CspInputError(
                    "binary constraint scope must contain two variables"
                )

        ordered_variables = tuple(sorted(self.variables))
        expected_deadline = {
            f"deadline_before_cutoff:{v.assignment_id}:{v.session_index}"
            for v in ordered_variables
        }
        expected_unique = {
            _pair_id("weekly_slot_unique", a, b)
            for i, a in enumerate(ordered_variables)
            for b in ordered_variables[i + 1 :]
        }
        expected_pairs = self.ordering_policy.required_pairs(ordered_variables)
        if not isinstance(expected_pairs, tuple) or any(
            not isinstance(pair, SessionPair) for pair in expected_pairs
        ):
            raise CspInputError(
                "ordering policy must return immutable SessionPair values"
            )
        if len(set(expected_pairs)) != len(expected_pairs):
            raise CspInputError("ordering policy returned duplicate session pairs")
        for pair in expected_pairs:
            if (
                pair.earlier not in variables
                or pair.later not in variables
                or pair.earlier.assignment_id != pair.later.assignment_id
                or pair.earlier == pair.later
                or pair.earlier.session_index >= pair.later.session_index
            ):
                raise CspInputError("ordering policy returned unrelated session pairs")
        paired = {
            variable
            for pair in expected_pairs
            for variable in (pair.earlier, pair.later)
        }
        for assignment in self.manifest.assignments:
            sessions = tuple(
                variable
                for variable in ordered_variables
                if variable.assignment_id == assignment.assignment_identity
            )
            if len(sessions) > 1 and not set(sessions).issubset(paired):
                raise CspInputError(
                    "ordering policy omitted one or more sessions in an assignment"
                )
        expected_order = {
            _pair_id("same_assignment_order", pair.earlier, pair.later)
            for pair in expected_pairs
        }
        actual = set(ids)
        for prefix, expected_ids in (
            ("deadline_before_cutoff:", expected_deadline),
            ("weekly_slot_unique:", expected_unique),
            ("same_assignment_order:", expected_order),
        ):
            actual_ids = {item for item in actual if item.startswith(prefix)}
            if actual_ids != expected_ids:
                category = (
                    "ordering"
                    if prefix.startswith("same_assignment_order")
                    else prefix[:-1]
                )
                raise CspInputError(f"{category} constraint coverage is incomplete")
        constraints_by_id = dict(zip(ids, self.constraints, strict=True))
        for variable in ordered_variables:
            item = constraints_by_id[
                f"deadline_before_cutoff:{variable.assignment_id}:{variable.session_index}"
            ]
            if not isinstance(item, DeadlineConstraint) or item.scope != (variable,):
                raise CspInputError("deadline constraint has an invalid scope")
        for i, first in enumerate(ordered_variables):
            for second in ordered_variables[i + 1 :]:
                item = constraints_by_id[_pair_id("weekly_slot_unique", first, second)]
                if not isinstance(item, UniqueSlotConstraint) or item.scope != (
                    first,
                    second,
                ):
                    raise CspInputError(
                        "unique-slot constraint has an invalid binary scope"
                    )
        for pair in expected_pairs:
            item = constraints_by_id[
                _pair_id("same_assignment_order", pair.earlier, pair.later)
            ]
            if (
                not isinstance(item, SameAssignmentOrderingConstraint)
                or item.scope != (pair.earlier, pair.later)
                or getattr(item, "pair", None) != pair
                or getattr(item, "policy", None) != self.ordering_policy
            ):
                raise CspInputError(
                    "ordering constraint does not match its policy pair"
                )
        capacity = [
            item for item in self.constraints if item.constraint_id == "weekly_capacity"
        ]
        if (
            len(capacity) != 1
            or not isinstance(capacity[0], WeeklyCapacityConstraint)
            or capacity[0].capacity != self.weekly_capacity
        ):
            raise CspInputError("one weekly-capacity constraint must match the problem")
        if capacity[0].scope != ordered_variables:
            raise CspInputError("weekly-capacity scope must cover the whole plan")

    @property
    def binary_constraints(self) -> tuple[BinaryConstraint, ...]:
        """Return relations available to a future AC-3 implementation."""
        return tuple(
            item for item in self.constraints if isinstance(item, BinaryConstraint)
        )

    def _canonical_problem_identity(self) -> str:
        """Fingerprint plan, domains, capacity, and all constraint policies."""
        payload = {
            "identity_version": 1,
            "whole_plan_identity": self.whole_plan_identity,
            "weekly_capacity": self.weekly_capacity,
            "ordering_policy": self.ordering_policy.configuration_identity,
            "domains": [
                {
                    "assignment": domain.variable.assignment_id,
                    "session": domain.variable.session_index,
                    "slots": sorted(
                        (slot.slot_id, slot.week_index) for slot in domain.values
                    ),
                }
                for domain in sorted(self.domains, key=lambda item: item.variable)
            ],
            "constraints": sorted(
                (
                    item.constraint_id,
                    item.configuration_identity,
                )
                for item in self.constraints
            ),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def _pair_id(prefix: str, earlier: SessionVariable, later: SessionVariable) -> str:
    """Build a stable identifier for a binary session relation."""
    return (
        f"{prefix}:{earlier.assignment_id}:{earlier.session_index}:"
        f"{later.assignment_id}:{later.session_index}"
    )


def _assignment_check_failures(
    problem: CspProblem,
    assignment: CspAssignment,
    *,
    require_complete: bool = False,
    on_constraint_check: Callable[[], None] | None = None,
) -> tuple[str, ...]:
    """Return all domain, manifest, and constraint violations for an assignment."""
    if not isinstance(problem, CspProblem) or not isinstance(assignment, CspAssignment):
        raise CspInputError("assignment validation requires typed CSP inputs")
    required = set(problem.variables)
    placements = assignment.placements
    failures: list[str] = []
    if any(
        variable not in required
        or variable.assignment_id
        not in {item.assignment_identity for item in problem.manifest.assignments}
        for variable in placements
    ):
        failures.append("whole_plan_manifest")
    for variable, value in placements.items():
        if variable not in required or value not in problem.domain_for(variable):
            failures.append(f"domain:{variable.assignment_id}:{variable.session_index}")
    if require_complete and set(placements) != required:
        failures.append("assignment_completeness")
    failures.extend(
        f"empty_domain:{domain.variable.assignment_id}:{domain.variable.session_index}"
        for domain in problem.domains
        if not domain.values
    )
    if failures:
        return tuple(dict.fromkeys(failures))
    failures.extend(
        constraint.constraint_id
        for constraint in problem.constraints
        if _constraint_fails(
            constraint, assignment, on_constraint_check=on_constraint_check
        )
    )
    return tuple(dict.fromkeys(failures))


def evaluate_assignment(
    problem: CspProblem,
    assignment: CspAssignment,
    *,
    _on_constraint_check: Callable[[], None] | None = None,
) -> CspEvaluation:
    """Validate a branch; this function never concludes global infeasibility."""
    failed_checks = _assignment_check_failures(
        problem, assignment, on_constraint_check=_on_constraint_check
    )
    evidence = _AssignmentValidationEvidence(
        whole_plan_identity=problem.whole_plan_identity,
        problem_identity=problem.problem_identity,
        assignment=assignment,
        failures=failed_checks,
    )
    if failed_checks:
        return CspEvaluation._from_validator(
            problem=problem,
            status=CspEvaluationStatus.BRANCH_REJECTED,
            assignment=assignment,
            failed_checks=failed_checks,
            _evidence=evidence,
        )
    if len(assignment.placements) != len(problem.variables):
        return CspEvaluation._from_validator(
            problem=problem,
            status=CspEvaluationStatus.INCOMPLETE,
            assignment=assignment,
            _evidence=evidence,
        )
    return CspEvaluation._from_validator(
        problem=problem,
        status=CspEvaluationStatus.FEASIBLE,
        assignment=assignment,
        _evidence=evidence,
    )


def _constraint_fails(
    constraint: CspConstraint,
    assignment: CspAssignment,
    *,
    on_constraint_check: Callable[[], None] | None,
) -> bool:
    """Invoke one predicate and notify an optional exact-call counter."""
    if on_constraint_check is not None:
        on_constraint_check()
    return not constraint.is_satisfied(assignment)
