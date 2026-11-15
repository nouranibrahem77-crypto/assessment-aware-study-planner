"""Bridge lazy CSP enumerators to whole-plan verification records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from study_planner.csp.ac3 import Ac3Enumerator, Ac3Options, ac3_enumerator
from study_planner.csp.backtracking import (
    BacktrackingEnumerator,
    BacktrackingOptions,
    backtracking_enumerator,
)
from study_planner.csp.core import CspInputError, CspProblem
from study_planner.csp.forward_checking import (
    ForwardCheckingEnumerator,
    ForwardCheckingOptions,
    forward_checking_enumerator,
)
from study_planner.csp.verification import (
    WholePlanVerificationPolicy,
    WholePlanVerificationResult,
    WholePlanVerificationStatus,
    verify_whole_plan,
)


class CspAlgorithm(str, Enum):
    """Independently selectable CSP enumeration configuration."""

    BT = "bt"
    BT_FC = "bt_fc"
    BT_FC_AC3 = "bt_fc_ac3"


class ScheduleBridgeStatus(str, Enum):
    """Status of one bridge request."""

    SCHEDULE = "schedule"
    EXHAUSTED = "exhausted"
    INCOMPLETE = "incomplete"


class WholePlanEnumerationStatus(str, Enum):
    """Terminal allocation-level result from exhaustive CSP enumeration."""

    EXHAUSTED_WITH_FEASIBLE = "exhausted_with_feasible"
    EXHAUSTED_NO_FEASIBLE = "exhausted_no_feasible"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class CspEnumerationStatistics:
    nodes_visited: int
    assignments_attempted: int
    backtracks: int
    constraint_checks: int
    domain_prunings: int
    compatibility_checks: int
    solver_time_seconds: float
    wall_time_seconds: float


@dataclass(frozen=True, slots=True)
class WholePlanSearchOutcome:
    whole_plan_identity: str
    problem_identity: str
    status: WholePlanEnumerationStatus
    schedules_examined: int
    feasible_schedules: int
    rejected_schedules: int
    rejection_reasons: tuple[str, ...]
    statistics: CspEnumerationStatistics


@dataclass(frozen=True, slots=True)
class ScheduleBridgeStep:
    status: ScheduleBridgeStatus
    verification: WholePlanVerificationResult | None
    outcome: WholePlanSearchOutcome | None


_Options = BacktrackingOptions | ForwardCheckingOptions | Ac3Options | None
_Enumerator = BacktrackingEnumerator | ForwardCheckingEnumerator | Ac3Enumerator


def _make_enumerator(
    algorithm: CspAlgorithm, problem: CspProblem, options: _Options
) -> _Enumerator:
    if algorithm is CspAlgorithm.BT:
        if options is not None and not isinstance(options, BacktrackingOptions):
            raise CspInputError("BT requires BacktrackingOptions")
        return backtracking_enumerator(problem, options)
    if algorithm is CspAlgorithm.BT_FC:
        if options is not None and not isinstance(options, ForwardCheckingOptions):
            raise CspInputError("BT+FC requires ForwardCheckingOptions")
        return forward_checking_enumerator(problem, options)
    if algorithm is CspAlgorithm.BT_FC_AC3:
        if options is not None and not isinstance(options, Ac3Options):
            raise CspInputError("BT+FC+AC-3 requires Ac3Options")
        return ac3_enumerator(problem, options)
    raise CspInputError("unsupported CSP algorithm")


class WholePlanScheduleEnumerator:
    """Lazily verify every distinct schedule from a selected CSP algorithm.

    A rejected verification applies to that concrete schedule only. The
    allocation is classified as having no feasible schedule only after the
    underlying enumerator reports exhaustive completion.
    """

    def __init__(
        self,
        problem: CspProblem,
        policy: WholePlanVerificationPolicy,
        *,
        algorithm: CspAlgorithm = CspAlgorithm.BT,
        options: _Options = None,
    ) -> None:
        if not isinstance(problem, CspProblem):
            raise CspInputError("schedule bridge requires a typed CSP problem")
        if not isinstance(policy, WholePlanVerificationPolicy):
            raise CspInputError("schedule bridge requires a verification policy")
        if not isinstance(algorithm, CspAlgorithm):
            raise CspInputError("schedule bridge requires a CSP algorithm")
        self._problem = problem
        self._enumerator = _make_enumerator(algorithm, problem, options)
        self._policy = policy
        self._schedules = 0
        self._feasible = 0
        self._rejected = 0
        self._reasons: list[str] = []
        self._verification_incomplete = False
        self._terminal: WholePlanSearchOutcome | None = None

    @property
    def statistics(self) -> CspEnumerationStatistics:
        source = self._enumerator.statistics
        return CspEnumerationStatistics(
            nodes_visited=source.nodes_visited,
            assignments_attempted=source.assignments_attempted,
            backtracks=source.backtracks,
            constraint_checks=source.constraint_checks,
            domain_prunings=getattr(source, "domain_prunings", 0),
            compatibility_checks=getattr(source, "compatibility_checks", 0),
            solver_time_seconds=source.solver_time_seconds,
            wall_time_seconds=source.wall_time_seconds,
        )

    def next_schedule(self) -> ScheduleBridgeStep:
        if self._terminal is not None:
            return ScheduleBridgeStep(
                ScheduleBridgeStatus.INCOMPLETE
                if self._terminal.status is WholePlanEnumerationStatus.INCOMPLETE
                else ScheduleBridgeStatus.EXHAUSTED,
                None,
                self._terminal,
            )
        step = self._enumerator.next_solution()
        status = step.status.value
        if status == "solution_found":
            evaluation = step.evaluation
            if evaluation is None or (
                evaluation.problem_identity != self._problem.problem_identity
                or evaluation.whole_plan_identity != self._problem.whole_plan_identity
            ):
                raise RuntimeError("CSP enumerator returned mismatched provenance")
            self._schedules += 1
            verification = verify_whole_plan(
                self._problem,
                evaluation.assignment,
                self._policy,
                expected_whole_plan_identity=self._problem.whole_plan_identity,
                expected_problem_identity=self._problem.problem_identity,
            )
            if verification.status is WholePlanVerificationStatus.FEASIBLE:
                self._feasible += 1
            else:
                self._rejected += 1
                self._reasons.extend(verification.rejection_reasons)
                self._verification_incomplete |= (
                    verification.status is WholePlanVerificationStatus.INCOMPLETE
                )
            return ScheduleBridgeStep(ScheduleBridgeStatus.SCHEDULE, verification, None)
        if status == "incomplete":
            self._terminal = self._outcome(WholePlanEnumerationStatus.INCOMPLETE)
            return ScheduleBridgeStep(
                ScheduleBridgeStatus.INCOMPLETE, None, self._terminal
            )
        terminal_status = (
            WholePlanEnumerationStatus.INCOMPLETE
            if self._verification_incomplete
            else WholePlanEnumerationStatus.EXHAUSTED_WITH_FEASIBLE
            if self._feasible
            else WholePlanEnumerationStatus.EXHAUSTED_NO_FEASIBLE
        )
        self._terminal = self._outcome(terminal_status)
        return ScheduleBridgeStep(
            ScheduleBridgeStatus.INCOMPLETE
            if terminal_status is WholePlanEnumerationStatus.INCOMPLETE
            else ScheduleBridgeStatus.EXHAUSTED,
            None,
            self._terminal,
        )

    def _outcome(self, status: WholePlanEnumerationStatus) -> WholePlanSearchOutcome:
        return WholePlanSearchOutcome(
            whole_plan_identity=self._problem.whole_plan_identity,
            problem_identity=self._problem.problem_identity,
            status=status,
            schedules_examined=self._schedules,
            feasible_schedules=self._feasible,
            rejected_schedules=self._rejected,
            rejection_reasons=tuple(sorted(set(self._reasons))),
            statistics=self.statistics,
        )
