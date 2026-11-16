"""Application-input adapters for the existing Search and CSP planner."""

from __future__ import annotations

from dataclasses import dataclass, field

from study_planner.csp.constraints import (
    build_required_constraints,
    expected_session_variables,
)
from study_planner.csp.core import (
    AssignmentCandidateIdentity,
    AssignmentPlan,
    CspProblem,
    SessionOrderingPolicy,
    SessionPair,
    SessionVariable,
    VariableDomain,
    WeeklySlot,
    WholePlanManifest,
)
from study_planner.csp.verification import WholePlanVerificationPolicy
from study_planner.integration.application_inputs import (
    ApplicationInputError,
    ApplicationSlot,
    ValidatedApplicationInputs,
    WeeklySlotCatalog,
)
from study_planner.integration.planner import WholePlanPlanningCandidate
from study_planner.search.nn_adapter import (
    SessionNNAdapter,
)

Allocation = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class CatalogChronologicalOrderingPolicy:
    """Order same-assignment sessions by their synthetic catalog chronology."""

    catalog: WeeklySlotCatalog

    @property
    def policy_identity(self) -> str:
        return (
            f"catalog-chronological-v1:{self.catalog.configuration_identity}"
        )

    def required_pairs(
        self, sessions: tuple[SessionVariable, ...]
    ) -> tuple[SessionPair, ...]:
        by_assignment: dict[str, list[SessionVariable]] = {}
        for session in sessions:
            by_assignment.setdefault(session.assignment_id, []).append(session)
        return tuple(
            SessionPair(earlier, later)
            for assignment_sessions in by_assignment.values()
            for earlier, later in zip(
                sorted(assignment_sessions),
                sorted(assignment_sessions)[1:],
                strict=False,
            )
        )

    def is_ordered(self, earlier: WeeklySlot, later: WeeklySlot) -> bool:
        by_id = {slot.slot.slot_id: slot for slot in self.catalog.slots}
        if earlier.slot_id not in by_id or later.slot_id not in by_id:
            raise ApplicationInputError(
                "ordering references a slot outside the catalog"
            )
        return self._chronology_key(by_id[earlier.slot_id]) < self._chronology_key(
            by_id[later.slot_id]
        )

    def _chronology_key(self, slot: ApplicationSlot) -> tuple[int, int, int]:
        return (
            slot.slot.week_index,
            (
                slot.time_window.weekday_iso
                - self.catalog.course_start_weekday_iso
            )
            % 7,
            slot.time_window.start_minute_local,
        )


@dataclass(frozen=True, slots=True, order=True)
class AddStudySession:
    """One Search action adding a session in a provisional NN click window."""

    assignment_identity: str
    relative_window: int


@dataclass(frozen=True, slots=True)
class ApplicationAllocationState:
    """Hashable allocation state with one four-window count tuple per assignment."""

    allocations: tuple[tuple[str, Allocation], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.allocations, tuple) or any(
            not isinstance(entry, tuple)
            or len(entry) != 2
            or not isinstance(entry[0], str)
            or not entry[0]
            or not isinstance(entry[1], tuple)
            or len(entry[1]) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in entry[1]
            )
            for entry in self.allocations
        ):
            raise ApplicationInputError("allocation state is malformed")
        identities = tuple(item[0] for item in self.allocations)
        if len(identities) != 3 or len(set(identities)) != 3:
            raise ApplicationInputError(
                "allocation state must preserve three unique assignments"
            )

    def for_assignment(self, assignment_identity: str) -> Allocation:
        """Return one assignment's four relative-window counts."""
        for identity, allocation in self.allocations:
            if identity == assignment_identity:
                return allocation
        raise ApplicationInputError(
            f"allocation state omits assignment {assignment_identity}"
        )


class ApplicationAllocationSpace:
    """Deterministic finite graph adding one eligible session per transition."""

    def __init__(self, inputs: ValidatedApplicationInputs) -> None:
        self._config = inputs.config
        self._rules = {
            rule.assignment_identity: rule for rule in inputs.config.assignment_rules
        }
        slot_windows = {
            slot.slot.slot_id: dict(slot.relative_windows)
            for slot in inputs.config.slot_catalog.slots
        }
        self._available_windows = {
            assignment_id: tuple(
                sorted(
                    {
                        slot_windows[slot_id][assignment_id]
                        for slot_id in rule.eligible_slot_ids
                        if assignment_id in slot_windows[slot_id]
                    }
                )
            )
            for assignment_id, rule in self._rules.items()
        }
        assignment_ids = tuple(sorted(self._rules))
        self._initial_state = ApplicationAllocationState(
            tuple((identity, (0, 0, 0, 0)) for identity in assignment_ids)
        )

    @property
    def initial_state(self) -> ApplicationAllocationState:
        return self._initial_state

    def is_candidate_state(self, state: ApplicationAllocationState) -> bool:
        """Only evaluate Search proposals within every configured inclusive bound."""
        self._validate_state(state)
        return all(
            rule.allocation_bounds.accepts(sum(state.for_assignment(identity)))
            for identity, rule in self._rules.items()
        )

    def actions(self, state: ApplicationAllocationState) -> tuple[AddStudySession, ...]:
        self._validate_state(state)
        actions: list[AddStudySession] = []
        for identity in sorted(self._rules):
            rule = self._rules[identity]
            allocation = state.for_assignment(identity)
            if sum(allocation) >= rule.allocation_bounds.maximum_sessions:
                continue
            actions.extend(
                AddStudySession(identity, window)
                for window in self._available_windows[identity]
            )
        return tuple(actions)

    def transition(
        self, state: ApplicationAllocationState, action: AddStudySession
    ) -> ApplicationAllocationState:
        if action not in self.actions(state):
            raise ApplicationInputError("session action is not legal in this state")
        allocations = dict(state.allocations)
        counts = list(allocations[action.assignment_identity])
        counts[action.relative_window - 1] += 1
        allocations[action.assignment_identity] = tuple(counts)  # type: ignore[assignment]
        return ApplicationAllocationState(tuple(sorted(allocations.items())))

    def action_cost(
        self,
        state: ApplicationAllocationState,
        action: AddStudySession,
        next_state: ApplicationAllocationState,
    ) -> float:
        del state, action, next_state
        return 1.0

    def _validate_state(self, state: ApplicationAllocationState) -> None:
        if not isinstance(state, ApplicationAllocationState):
            raise ApplicationInputError("Search state has an invalid type")
        if tuple(identity for identity, _ in state.allocations) != tuple(
            sorted(self._rules)
        ):
            raise ApplicationInputError("Search state assignment identities differ")
        for identity, allocation in state.allocations:
            if (
                sum(allocation)
                > self._rules[identity].allocation_bounds.maximum_sessions
            ):
                raise ApplicationInputError("allocation exceeds configured upper bound")


@dataclass(frozen=True, slots=True)
class ApplicationPlanningBinding:
    """Search state space and candidate factory created from validated inputs."""

    inputs: ValidatedApplicationInputs
    adapter: SessionNNAdapter
    allocation_space: ApplicationAllocationSpace
    ordering_policy: SessionOrderingPolicy
    _feature_context_identities: tuple[tuple[str, str], ...] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_feature_context_identities",
            tuple(
                sorted(
                    (
                        context.assignment_identity,
                        context.base_feature_context_identity,
                    )
                    for context in self.inputs.feature_contexts
                )
            ),
        )

    def candidate_factory(
        self, state: ApplicationAllocationState
    ) -> WholePlanPlanningCandidate:
        """Build a fully identified CSP problem for one bounded allocation."""
        current_context_identities = tuple(
            sorted(
                (
                    context.assignment_identity,
                    context.base_feature_context_identity,
                )
                for context in self.inputs.feature_contexts
            )
        )
        if current_context_identities != self._feature_context_identities:
            raise ApplicationInputError(
                "validated feature context changed after planning binding"
            )
        if not self.allocation_space.is_candidate_state(state):
            raise ApplicationInputError(
                "allocation does not satisfy configured session bounds"
            )
        config = self.inputs.config
        assignments = []
        contexts = {
            item.assignment_identity: item for item in self.inputs.feature_contexts
        }
        for identity in sorted(contexts):
            allocation = state.for_assignment(identity)
            context = contexts[identity]
            candidate_identity = AssignmentCandidateIdentity(
                self.inputs.student_identity,
                config.presentation_split.course_identity,
                identity,
                allocation,
                context.base_feature_context_identity,
                self.adapter.mapping_scenario_identity,
            )
            assignments.append(
                AssignmentPlan(
                    identity, candidate_identity, context.base_feature_context_identity
                )
            )
        manifest = WholePlanManifest(
            self.inputs.student_identity,
            config.presentation_split.course_identity,
            tuple(assignments),
            self.adapter.mapping_scenario_identity,
        )
        variables = expected_session_variables(manifest)
        slots_by_id = {
            item.slot.slot_id: item.slot for item in config.slot_catalog.slots
        }
        rules = {rule.assignment_identity: rule for rule in config.assignment_rules}
        domains = tuple(
            VariableDomain(
                variable,
                tuple(
                    sorted(
                        slots_by_id[slot_id]
                        for slot_id in rules[variable.assignment_id].eligible_slot_ids
                    )
                ),
            )
            for variable in variables
        )
        eligible = tuple(
            (
                variable,
                frozenset(
                    slots_by_id[slot_id]
                    for slot_id in rules[variable.assignment_id].eligible_slot_ids
                ),
            )
            for variable in variables
        )
        capacity = config.slot_catalog.weekly_capacity
        constraints = build_required_constraints(
            manifest=manifest,
            variables=variables,
            eligible_slots=eligible,
            ordering_policy=self.ordering_policy,
            weekly_capacity=capacity,
        )
        problem = CspProblem(
            manifest,
            variables,
            domains,
            constraints,
            capacity,
            self.ordering_policy,
        )
        catalog = config.slot_catalog
        verification_policy = WholePlanVerificationPolicy(
            "application-planning-v1",
            f"{catalog.deadline_policy_identity}:{catalog.deadline_policy_version}",
            problem.ordering_policy.configuration_identity,
            catalog.configuration_identity,
            f"{catalog.identity}:{catalog.version}:weekly-capacity",
            "weekly-slot-id-and-week-index-v1",
            capacity,
        )
        return WholePlanPlanningCandidate(
            problem=problem,
            verification_policy=verification_policy,
            feature_contexts=tuple(contexts[key] for key in sorted(contexts)),
            targets=config.targets,
            slot_relative_windows=(),
            slot_relative_windows_by_assignment=catalog.windows_by_assignment,
            support_policy=config.support_policy,
        )


__all__ = [
    "AddStudySession",
    "CatalogChronologicalOrderingPolicy",
    "ApplicationAllocationSpace",
    "ApplicationAllocationState",
    "ApplicationPlanningBinding",
]
