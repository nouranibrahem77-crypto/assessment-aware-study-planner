"""Typed boundary between Search candidates, CSP verification, and NN inference."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np
import pandas as pd

from study_planner.search.session_mapping import (
    STRICT_FFF_FEATURE_NAMES,
    ClickScenario,
    ClickSupportDiagnostic,
    JointSupportStatus,
    MappingValidationError,
    MarginalSupportStatus,
    MonotonicityContext,
    MonotonicityObservation,
    MonotonicityReport,
    SessionClickMappingConfig,
    SessionClickMappingResult,
    evaluate_monotonicity,
    map_sessions_to_clicks,
)


class CandidateEvaluationError(ValueError):
    """Raised for invalid adapter inputs or incompatible candidate identities."""


class EvaluationPhase(str, Enum):
    """Whether windows are provisional or derived from verified CSP output."""

    PROVISIONAL = "provisional"
    FINAL_SCHEDULED = "final_scheduled"


class EvaluationStatus(str, Enum):
    """Outcome of feature mapping and NN inference."""

    PREDICTED = "predicted"
    INFERENCE_FAILED = "inference_failed"


class CspVerificationStatus(str, Enum):
    """Feasibility outcome recorded by the CSP verification contract."""

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"


@dataclass(frozen=True, slots=True)
class CandidateSessionAllocation:
    """Candidate session counts in the four deadline-relative windows."""

    weekly_sessions: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        _validate_counts(self.weekly_sessions)


@dataclass(frozen=True, slots=True)
class CandidateIdentity:
    """Stable identity binding predictions to learner, plan, base row, and rule."""

    student_identity: str
    assignment_identity: str
    canonical_allocation_identity: tuple[int, int, int, int]
    base_feature_context_identity: str
    mapping_scenario_identity: str

    def __post_init__(self) -> None:
        if not all(
            (
                isinstance(self.student_identity, str) and self.student_identity,
                isinstance(self.assignment_identity, str) and self.assignment_identity,
                self.base_feature_context_identity,
                self.mapping_scenario_identity,
            )
        ):
            raise CandidateEvaluationError("candidate identity fields cannot be empty")
        _validate_counts(self.canonical_allocation_identity)


@dataclass(frozen=True, slots=True)
class ScheduleData:
    """Raw schedule data. It carries no feasibility or CSP provenance."""

    sessions: tuple[ScheduledSession, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.sessions, tuple):
            raise CandidateEvaluationError(
                "schedule sessions must be an immutable tuple"
            )
        if any(not isinstance(session, ScheduledSession) for session in self.sessions):
            raise CandidateEvaluationError(
                "schedule contains an invalid session record"
            )
        slot_ids = tuple(session.slot_id for session in self.sessions)
        try:
            if len(set(slot_ids)) != len(slot_ids):
                raise CandidateEvaluationError("a time slot may be used only once")
        except TypeError as error:
            raise CandidateEvaluationError(
                "all schedule slot IDs must be hashable"
            ) from error

    @property
    def weekly_sessions(self) -> tuple[int, int, int, int]:
        """Count schedule records in the mapper's four relative windows."""
        counts = [0, 0, 0, 0]
        for session in self.sessions:
            counts[session.relative_window - 1] += 1
        return tuple(counts)  # type: ignore[return-value]

    @property
    def schedule_identity(self) -> tuple[tuple[str, int], ...]:
        """Return the canonical schedule details used in audit records."""
        return tuple(
            (session.slot_id, session.relative_window) for session in self.sessions
        )


@dataclass(frozen=True, slots=True)
class ScheduledSession:
    """One session placed in an opaque CSP slot and relative feature window."""

    slot_id: str
    relative_window: int

    def __post_init__(self) -> None:
        if not isinstance(self.slot_id, str) or not self.slot_id:
            raise CandidateEvaluationError("slot_id must be a non-empty string")
        if (
            isinstance(self.relative_window, bool)
            or not isinstance(self.relative_window, int)
            or self.relative_window not in {1, 2, 3, 4}
        ):
            raise CandidateEvaluationError("relative_window must be from 1 through 4")


@dataclass(frozen=True, slots=True, init=False)
class CspVerificationResult:
    """Typed CSP solver result; only feasible records carry an evaluable plan.

    The constructor is intentionally unavailable. The future CSP implementation
    should create records through ``_from_solver`` after completing its checks.
    This is an API provenance convention, not a security boundary against
    arbitrary Python code or deliberate object fabrication.
    """

    candidate_identity: CandidateIdentity
    status: CspVerificationStatus
    schedule: ScheduleData | None
    verification_identity: str
    rejection_reasons: tuple[str, ...]

    def __init__(self) -> None:
        raise TypeError("CspVerificationResult records must come from a CSP solver")

    @classmethod
    def _from_solver(
        cls,
        *,
        candidate_identity: CandidateIdentity,
        status: CspVerificationStatus,
        schedule: ScheduleData | None,
        verification_identity: str,
        rejection_reasons: tuple[str, ...] = (),
    ) -> CspVerificationResult:
        """Create a record at the trusted CSP integration boundary."""
        if not verification_identity:
            raise CandidateEvaluationError("verification identity cannot be empty")
        if not isinstance(candidate_identity, CandidateIdentity):
            raise CandidateEvaluationError("CSP result needs a candidate identity")
        if not isinstance(status, CspVerificationStatus):
            raise CandidateEvaluationError("CSP result status is invalid")
        if schedule is not None and not isinstance(schedule, ScheduleData):
            raise CandidateEvaluationError("CSP result schedule data is invalid")
        if status is CspVerificationStatus.FEASIBLE and schedule is None:
            raise CandidateEvaluationError("feasible CSP result needs schedule data")
        if status is CspVerificationStatus.INFEASIBLE and schedule is not None:
            raise CandidateEvaluationError(
                "infeasible CSP result cannot carry a schedule"
            )
        if status is CspVerificationStatus.FEASIBLE and rejection_reasons:
            raise CandidateEvaluationError(
                "feasible CSP result cannot have rejection reasons"
            )
        record = object.__new__(cls)
        object.__setattr__(record, "candidate_identity", candidate_identity)
        object.__setattr__(record, "status", status)
        object.__setattr__(record, "schedule", schedule)
        object.__setattr__(record, "verification_identity", verification_identity)
        object.__setattr__(record, "rejection_reasons", tuple(rejection_reasons))
        return record


@dataclass(frozen=True, slots=True)
class CandidateFeatureContext:
    """One student-assignment feature row plus audit-only identities."""

    student_identity: str
    assignment_identity: str
    features: pd.DataFrame

    def __post_init__(self) -> None:
        if not all(
            (
                isinstance(self.student_identity, str) and self.student_identity,
                isinstance(self.assignment_identity, str) and self.assignment_identity,
            )
        ):
            raise CandidateEvaluationError(
                "feature context requires student and assignment identities"
            )
        if not isinstance(self.features, pd.DataFrame) or len(self.features) != 1:
            raise CandidateEvaluationError(
                "feature context must contain one pandas feature row"
            )

    @property
    def base_feature_context_identity(self) -> str:
        """Compute a stable content identity for the current immutable base row."""
        digest = hashlib.sha256()
        digest.update(repr(tuple(self.features.columns)).encode("utf-8"))
        try:
            hashed = pd.util.hash_pandas_object(
                self.features, index=True, categorize=True
            ).to_numpy(dtype=np.uint64)
        except TypeError as error:
            raise CandidateEvaluationError(
                "base feature context cannot be fingerprinted"
            ) from error
        digest.update(hashed.tobytes())
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    """Deeply immutable scalar snapshot of one validated strict feature row."""

    columns: tuple[str, ...]
    values: tuple[float | None, ...]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> FeatureSnapshot:
        """Capture a validated one-row numeric strict feature frame."""
        values: list[float | None] = []
        for value in frame.iloc[0].tolist():
            values.append(None if pd.isna(value) else float(value))
        return cls(tuple(str(column) for column in frame.columns), tuple(values))

    def to_frame(self) -> pd.DataFrame:
        """Materialize a caller-owned DataFrame copy from the immutable snapshot."""
        return pd.DataFrame(
            [[np.nan if value is None else value for value in self.values]],
            columns=self.columns,
        )


@dataclass(frozen=True, slots=True)
class MappingEvaluationSnapshot:
    """Immutable mapping features and support metadata retained with a result."""

    features: FeatureSnapshot
    weekly_sessions: tuple[int, int, int, int]
    scenario: ClickScenario
    activity_group: str
    clicks_per_session: int
    changed_features: tuple[str, ...]
    missing_numeric_features: tuple[str, ...]
    marginal_status: MarginalSupportStatus
    joint_support_status: JointSupportStatus
    support_diagnostics: tuple[ClickSupportDiagnostic, ...]

    @classmethod
    def from_mapping(
        cls, mapping: SessionClickMappingResult
    ) -> MappingEvaluationSnapshot:
        """Copy mapping output into immutable, DataFrame-free result fields."""
        return cls(
            features=FeatureSnapshot.from_frame(mapping.candidate_features),
            weekly_sessions=mapping.weekly_sessions,
            scenario=mapping.scenario,
            activity_group=mapping.activity_group,
            clicks_per_session=mapping.clicks_per_session,
            changed_features=mapping.changed_features,
            missing_numeric_features=mapping.missing_numeric_features,
            marginal_status=mapping.marginal_status,
            joint_support_status=mapping.joint_support_status,
            support_diagnostics=mapping.support_diagnostics,
        )

    @property
    def candidate_features(self) -> pd.DataFrame:
        """Return a new DataFrame, never the stored evaluation snapshot."""
        return self.features.to_frame()


class StrictPredictor(Protocol):
    """Subset of the existing RegressionPredictor interface used here."""

    feature_names: tuple[str, ...]
    assessment_type_columns: tuple[str, ...]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict marks from a strict-schema feature batch."""


@dataclass(frozen=True, slots=True)
class CandidatePredictionResult:
    """Prediction and immutable audit data bound to one candidate identity."""

    identity: CandidateIdentity
    phase: EvaluationPhase
    status: EvaluationStatus
    mapping: MappingEvaluationSnapshot
    prediction: float | None
    schedule_identity: tuple[tuple[str, int], ...] | None
    csp_verification_identity: str | None
    monotonicity_report: MonotonicityReport | None
    error_type: str | None = None
    error_message: str | None = None

    def assert_candidate(self, identity: CandidateIdentity) -> None:
        """Reject result reuse for a different candidate or mapping scenario."""
        if self.identity != identity:
            raise CandidateEvaluationError(
                "prediction result candidate identity mismatch"
            )

    def usable_for_goal_test(
        self,
        identity: CandidateIdentity,
        schedule_identity: tuple[tuple[str, int], ...] | None = None,
        csp_verification_identity: str | None = None,
    ) -> bool:
        """Check candidate, schedule, and verification identities before use."""
        self.assert_candidate(identity)
        return (
            self.phase is EvaluationPhase.FINAL_SCHEDULED
            and self.status is EvaluationStatus.PREDICTED
            and schedule_identity is not None
            and schedule_identity == self.schedule_identity
            and csp_verification_identity is not None
            and csp_verification_identity == self.csp_verification_identity
        )


class CandidateEvaluator(Protocol):
    """Search-facing evaluation contract without NN implementation details."""

    def evaluate_provisional(
        self,
        allocation: CandidateSessionAllocation,
        context: CandidateFeatureContext,
    ) -> CandidatePredictionResult:
        """Predict using a caller-supplied provisional weekly distribution."""

    def evaluate_final(
        self,
        allocation: CandidateSessionAllocation,
        verification: CspVerificationResult,
        context: CandidateFeatureContext,
    ) -> CandidatePredictionResult:
        """Predict only after a feasible, candidate-matched CSP result."""


def _validate_counts(counts: tuple[int, ...]) -> None:
    if not isinstance(counts, tuple) or len(counts) != 4:
        raise CandidateEvaluationError(
            "weekly_sessions must be an immutable four-item tuple"
        )
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in counts
    ):
        raise CandidateEvaluationError(
            "weekly session counts must be non-negative integers"
        )


@dataclass
class SessionNNAdapter:
    """Map Search candidates and invoke the existing fitted NN predictor."""

    predictor: StrictPredictor
    mapping_config: SessionClickMappingConfig
    _monotonicity_observations: dict[
        MonotonicityContext,
        dict[
            tuple[tuple[int, int, int, int], tuple[tuple[str, int], ...]],
            float,
        ],
    ] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if tuple(self.predictor.feature_names) != STRICT_FFF_FEATURE_NAMES:
            raise CandidateEvaluationError(
                "predictor feature schema/order differs from strict FFF mapping schema"
            )
        if (
            tuple(self.predictor.assessment_type_columns)
            != STRICT_FFF_FEATURE_NAMES[:3]
        ):
            raise CandidateEvaluationError(
                "predictor assessment type columns differ from strict FFF schema"
            )

    @property
    def mapping_scenario_identity(self) -> str:
        """Identify all settings that affect the synthetic click mapping."""
        return ":".join(
            (
                self.mapping_config.scenario.value,
                self.mapping_config.activity_group,
                str(self.mapping_config.clicks_per_session),
            )
        )

    def candidate_identity(
        self,
        allocation: CandidateSessionAllocation,
        context: CandidateFeatureContext,
    ) -> CandidateIdentity:
        """Build the canonical identity used by prediction and CSP contracts."""
        if not isinstance(allocation, CandidateSessionAllocation):
            raise CandidateEvaluationError("invalid candidate allocation type")
        if not isinstance(context, CandidateFeatureContext):
            raise CandidateEvaluationError("invalid candidate feature context")
        return CandidateIdentity(
            student_identity=context.student_identity,
            assignment_identity=context.assignment_identity,
            canonical_allocation_identity=allocation.weekly_sessions,
            base_feature_context_identity=context.base_feature_context_identity,
            mapping_scenario_identity=self.mapping_scenario_identity,
        )

    def evaluate_provisional(
        self,
        allocation: CandidateSessionAllocation,
        context: CandidateFeatureContext,
    ) -> CandidatePredictionResult:
        """Evaluate an allocation under its caller-supplied provisional split."""
        identity = self.candidate_identity(allocation, context)
        mapping = self._map(allocation.weekly_sessions, context)
        return self._predict(
            identity,
            EvaluationPhase.PROVISIONAL,
            mapping,
            schedule_identity=None,
            csp_verification_identity=None,
            monotonicity_report=None,
        )

    def evaluate_final(
        self,
        allocation: CandidateSessionAllocation,
        verification: CspVerificationResult,
        context: CandidateFeatureContext,
    ) -> CandidatePredictionResult:
        """Evaluate only a feasible CSP schedule bound to this candidate."""
        expected_identity = self.candidate_identity(allocation, context)
        if not isinstance(verification, CspVerificationResult):
            raise CandidateEvaluationError(
                "final evaluation requires a typed CSP verification result"
            )
        if verification.candidate_identity != expected_identity:
            raise CandidateEvaluationError(
                "CSP verification candidate identity mismatch"
            )
        if verification.status is not CspVerificationStatus.FEASIBLE:
            raise CandidateEvaluationError(
                "CSP verification did not establish feasibility"
            )
        schedule = verification.schedule
        if schedule is None:
            raise CandidateEvaluationError("feasible CSP result has no schedule")
        if sum(schedule.weekly_sessions) != sum(allocation.weekly_sessions):
            raise CandidateEvaluationError(
                "CSP schedule session count differs from candidate allocation"
            )

        mapping = self._map(schedule.weekly_sessions, context)
        comparison_context = MonotonicityContext(
            student_id=context.student_identity,
            assignment_id=context.assignment_identity,
            base_feature_context_id=expected_identity.base_feature_context_identity,
            mapping_scenario_id=self.mapping_scenario_identity,
        )
        observation_key = (schedule.weekly_sessions, schedule.schedule_identity)
        observations = self._monotonicity_observations.setdefault(
            comparison_context, {}
        )
        result = self._predict(
            expected_identity,
            EvaluationPhase.FINAL_SCHEDULED,
            mapping,
            schedule_identity=schedule.schedule_identity,
            csp_verification_identity=verification.verification_identity,
            monotonicity_report=None,
        )
        if (
            result.status is EvaluationStatus.PREDICTED
            and result.prediction is not None
        ):
            observations[observation_key] = result.prediction
            report = evaluate_monotonicity(
                tuple(
                    MonotonicityObservation(
                        weekly_sessions=counts,
                        predicted_mark=mark,
                        context=comparison_context,
                        schedule_identity=schedule_identity,
                    )
                    for (counts, schedule_identity), mark in observations.items()
                    if np.isfinite(mark)
                )
            )
            result = CandidatePredictionResult(
                identity=result.identity,
                phase=result.phase,
                status=result.status,
                mapping=result.mapping,
                prediction=result.prediction,
                schedule_identity=result.schedule_identity,
                csp_verification_identity=result.csp_verification_identity,
                monotonicity_report=report,
            )
        return result

    def _map(
        self,
        weekly_sessions: tuple[int, int, int, int],
        context: CandidateFeatureContext,
    ) -> SessionClickMappingResult:
        try:
            return map_sessions_to_clicks(
                context.features.copy(deep=True),
                {week: count for week, count in enumerate(weekly_sessions, 1)},
                self.mapping_config,
                self.predictor.feature_names,
            )
        except (MappingValidationError, TypeError, ValueError) as error:
            raise CandidateEvaluationError(str(error)) from error

    def _predict(
        self,
        identity: CandidateIdentity,
        phase: EvaluationPhase,
        mapping: SessionClickMappingResult,
        *,
        schedule_identity: tuple[tuple[str, int], ...] | None,
        csp_verification_identity: str | None,
        monotonicity_report: MonotonicityReport | None,
    ) -> CandidatePredictionResult:
        snapshot = MappingEvaluationSnapshot.from_mapping(mapping)
        try:
            output = np.asarray(
                self.predictor.predict(snapshot.candidate_features.copy(deep=True))
            )
            if output.size != 1:
                raise ValueError("predictor must return exactly one mark")
            prediction = float(output.reshape(-1)[0])
            if not np.isfinite(prediction) or not 0 <= prediction <= 100:
                raise ValueError("predictor returned a mark outside finite [0, 100]")
        except Exception as error:
            return CandidatePredictionResult(
                identity=identity,
                phase=phase,
                status=EvaluationStatus.INFERENCE_FAILED,
                mapping=snapshot,
                prediction=None,
                schedule_identity=schedule_identity,
                csp_verification_identity=csp_verification_identity,
                monotonicity_report=None,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        return CandidatePredictionResult(
            identity=identity,
            phase=phase,
            status=EvaluationStatus.PREDICTED,
            mapping=snapshot,
            prediction=prediction,
            schedule_identity=schedule_identity,
            csp_verification_identity=csp_verification_identity,
            monotonicity_report=monotonicity_report,
        )
