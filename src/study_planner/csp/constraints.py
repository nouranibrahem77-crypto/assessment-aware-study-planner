"""Whole-plan constraints and reusable binary relations for CSP propagation."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations

from study_planner.csp.core import (
    AssignmentCandidateIdentity,
    BinaryConstraint,
    CspAssignment,
    CspConstraint,
    CspInputError,
    OrderingPolicySnapshot,
    SessionOrderingPolicy,
    SessionPair,
    SessionVariable,
    WeeklySlot,
    WholePlanManifest,
    _pair_id,
)


@dataclass(frozen=True, slots=True)
class DeadlineConstraint:
    """Unary deadline check against caller-precomputed eligible weekly slots.

    The eligibility set must come from an approved deadline/week convention.
    This constraint performs no conversion and does not infer boundary rules.
    """

    variable: SessionVariable
    eligible_slots: frozenset[WeeklySlot]

    def __post_init__(self) -> None:
        if not isinstance(self.variable, SessionVariable):
            raise CspInputError("deadline constraint variable is invalid")
        if not isinstance(self.eligible_slots, frozenset) or any(
            not isinstance(slot, WeeklySlot) for slot in self.eligible_slots
        ):
            raise CspInputError("eligible slots must be an immutable slot set")

    @property
    def constraint_id(self) -> str:
        return (
            f"deadline_before_cutoff:{self.variable.assignment_id}:"
            f"{self.variable.session_index}"
        )

    @property
    def configuration_identity(self) -> str:
        """Stable encoding of this session's supplied deadline eligibility."""
        return json.dumps(
            sorted((slot.slot_id, slot.week_index) for slot in self.eligible_slots),
            separators=(",", ":"),
        )

    @property
    def scope(self) -> tuple[SessionVariable, ...]:
        return (self.variable,)

    def is_satisfied(self, assignment: CspAssignment) -> bool:
        value = assignment.placements.get(self.variable)
        return value is None or value in self.eligible_slots


@dataclass(frozen=True, slots=True)
class UniqueSlotConstraint(BinaryConstraint):
    """Binary all-different relation for one pair of session variables."""

    first: SessionVariable
    second: SessionVariable

    def __post_init__(self) -> None:
        if not isinstance(self.first, SessionVariable) or not isinstance(
            self.second, SessionVariable
        ):
            raise CspInputError("unique-slot constraint variables are invalid")
        if self.first == self.second:
            raise CspInputError("unique-slot constraint needs distinct variables")

    @property
    def scope(self) -> tuple[SessionVariable, SessionVariable]:
        return (self.first, self.second)

    @property
    def constraint_id(self) -> str:
        return _pair_id("weekly_slot_unique", self.first, self.second)

    @property
    def configuration_identity(self) -> str:
        """Identify the fixed all-different relation and its variable pair."""
        return self.constraint_id

    def is_compatible(
        self,
        first_variable: SessionVariable,
        first_value: WeeklySlot,
        second_variable: SessionVariable,
        second_value: WeeklySlot,
    ) -> bool:
        if {first_variable, second_variable} != set(self.scope):
            raise CspInputError("variables do not match unique-slot constraint scope")
        return first_value.slot_id != second_value.slot_id

    def is_satisfied(self, assignment: CspAssignment) -> bool:
        placements = assignment.placements
        if self.first not in placements or self.second not in placements:
            return True
        return self.is_compatible(
            self.first,
            placements[self.first],
            self.second,
            placements[self.second],
        )


@dataclass(frozen=True, slots=True)
class SameAssignmentOrderingConstraint(BinaryConstraint):
    """Binary ordering relation for one policy-derived session pair."""

    pair: SessionPair
    policy: SessionOrderingPolicy | OrderingPolicySnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.pair, SessionPair):
            raise CspInputError("ordering constraint needs a typed session pair")
        if self.pair.earlier.assignment_id != self.pair.later.assignment_id:
            raise CspInputError("ordered sessions must belong to the same assignment")
        if not callable(getattr(self.policy, "is_ordered", None)):
            raise CspInputError("ordering policy must provide is_ordered")
        if not isinstance(getattr(self.policy, "policy_identity", None), str) or not (
            self.policy.policy_identity
        ):
            raise CspInputError("ordering policy needs a stable identity")

    @property
    def scope(self) -> tuple[SessionVariable, SessionVariable]:
        return (self.pair.earlier, self.pair.later)

    @property
    def constraint_id(self) -> str:
        return _pair_id("same_assignment_order", self.pair.earlier, self.pair.later)

    def is_compatible(
        self,
        first_variable: SessionVariable,
        first_value: WeeklySlot,
        second_variable: SessionVariable,
        second_value: WeeklySlot,
    ) -> bool:
        if first_variable == self.pair.earlier and second_variable == self.pair.later:
            return self.policy.is_ordered(first_value, second_value)
        if first_variable == self.pair.later and second_variable == self.pair.earlier:
            return self.policy.is_ordered(second_value, first_value)
        raise CspInputError("variables do not match ordering constraint scope")

    def is_satisfied(self, assignment: CspAssignment) -> bool:
        placements = assignment.placements
        if self.pair.earlier not in placements or self.pair.later not in placements:
            return True
        return self.is_compatible(
            self.pair.earlier,
            placements[self.pair.earlier],
            self.pair.later,
            placements[self.pair.later],
        )

    @property
    def configuration_identity(self) -> str:
        """Bind the pair relation to the selected ordering policy version."""
        policy_configuration = getattr(
            self.policy, "configuration_identity", self.policy.policy_identity
        )
        return f"{policy_configuration}:{self.constraint_id}"


@dataclass(frozen=True, slots=True)
class WeeklyCapacityConstraint:
    """Global maximum placements per week, configured by the problem input."""

    capacity: int
    scope: tuple[SessionVariable, ...]

    def __post_init__(self) -> None:
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int):
            raise CspInputError("weekly capacity must be a non-negative integer")
        if self.capacity < 0:
            raise CspInputError("weekly capacity must be a non-negative integer")
        if not isinstance(self.scope, tuple) or any(
            not isinstance(variable, SessionVariable) for variable in self.scope
        ):
            raise CspInputError(
                "weekly capacity scope must be an immutable variable tuple"
            )
        if len(set(self.scope)) != len(self.scope):
            raise CspInputError("weekly capacity scope cannot contain duplicates")

    @property
    def constraint_id(self) -> str:
        return "weekly_capacity"

    @property
    def configuration_identity(self) -> str:
        """Identify the configured maximum and complete variable scope."""
        scoped = ",".join(
            f"{variable.assignment_id}:{variable.session_index}"
            for variable in self.scope
        )
        return f"capacity={self.capacity};scope={scoped}"

    def is_satisfied(self, assignment: CspAssignment) -> bool:
        counts = Counter(
            entry.value.week_index
            for entry in assignment.entries
            if entry.variable in self.scope
        )
        return all(count <= self.capacity for count in counts.values())


def build_required_constraints(
    *,
    manifest: WholePlanManifest,
    variables: tuple[SessionVariable, ...],
    eligible_slots: tuple[tuple[SessionVariable, frozenset[WeeklySlot]], ...],
    ordering_policy: SessionOrderingPolicy,
    weekly_capacity: int,
) -> tuple[CspConstraint, ...]:
    """Build explicit P1 constraints from complete manifest and policy inputs."""
    if not isinstance(manifest, WholePlanManifest):
        raise CspInputError("required constraints need a whole-plan manifest")
    if (
        not callable(getattr(ordering_policy, "required_pairs", None))
        or not callable(getattr(ordering_policy, "is_ordered", None))
        or not getattr(ordering_policy, "policy_identity", None)
    ):
        raise CspInputError("a valid explicit ordering policy is required")
    if not isinstance(variables, tuple) or set(variables) != set(
        expected_session_variables(manifest)
    ):
        raise CspInputError("variables do not match the whole-plan allocation manifest")
    if not isinstance(eligible_slots, tuple) or any(
        not isinstance(item, tuple)
        or len(item) != 2
        or not isinstance(item[0], SessionVariable)
        or not isinstance(item[1], frozenset)
        for item in eligible_slots
    ):
        raise CspInputError(
            "deadline eligibility must be immutable variable-slot pairs"
        )
    eligibility = dict(eligible_slots)
    if len(eligibility) != len(eligible_slots) or set(eligibility) != set(variables):
        raise CspInputError("deadline eligibility must cover each session exactly once")
    ordered_variables = tuple(sorted(variables))
    expected_pairs = ordering_policy.required_pairs(ordered_variables)
    if not isinstance(expected_pairs, tuple) or any(
        not isinstance(pair, SessionPair) for pair in expected_pairs
    ):
        raise CspInputError("ordering policy must return a tuple of SessionPair values")
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
            raise CspInputError("ordering policy returned a reversed or unrelated pair")
    paired = {
        variable for pair in expected_pairs for variable in (pair.earlier, pair.later)
    }
    for assignment in manifest.assignments:
        sessions = tuple(
            variable
            for variable in ordered_variables
            if variable.assignment_id == assignment.assignment_identity
        )
        if len(sessions) > 1 and not set(sessions).issubset(paired):
            raise CspInputError(
                "ordering policy omitted one or more sessions in an assignment"
            )

    constraints: list[CspConstraint] = [
        DeadlineConstraint(variable, eligibility[variable])
        for variable in ordered_variables
    ]
    constraints.extend(
        UniqueSlotConstraint(first, second)
        for first, second in combinations(ordered_variables, 2)
    )
    constraints.extend(
        SameAssignmentOrderingConstraint(pair, ordering_policy)
        for pair in expected_pairs
    )
    constraints.append(WeeklyCapacityConstraint(weekly_capacity, ordered_variables))
    return tuple(constraints)


def expected_session_variables(
    manifest: WholePlanManifest,
) -> tuple[SessionVariable, ...]:
    """Expand every allocation, including zeros, into deterministic variables."""
    return tuple(
        SessionVariable(assignment.assignment_identity, index)
        for assignment in manifest.canonical_assignments
        for index in range(assignment.session_count)
    )


def candidate_identity_for(
    manifest: WholePlanManifest, assignment_identity: str
) -> AssignmentCandidateIdentity:
    """Project one assignment candidate identity from a whole-plan manifest."""
    return manifest.assignment(assignment_identity).candidate_identity
