"""Typed boundary between Search candidates, CSP verification, and NN inference."""

from __future__ import annotations

import hashlib
import json
import re
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Protocol

import numpy as np
import pandas as pd

from study_planner.csp.verification import (
    WholePlanVerificationResult,
    WholePlanVerificationStatus,
    validate_whole_plan_verification_result,
)
from study_planner.nn.inference import (
    _TrustedArtifactProvenance,
    trusted_artifact_provenance,
    verify_trusted_artifact_integrity,
)
from study_planner.search.session_mapping import (
    STRICT_FFF_FEATURE_NAMES,
    ClickScenario,
    ClickSupportDiagnostic,
    IncrementalMonotonicityReporter,
    JointSupportStatus,
    MappingValidationError,
    MarginalSupportStatus,
    MonotonicityContext,
    MonotonicityObservation,
    MonotonicityReport,
    SessionClickMappingConfig,
    SessionClickMappingResult,
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


class NNModelMode(str, Enum):
    """Supported prediction model modes, kept distinct for result provenance."""

    MODEL_A_STRICT = "model_a_strict"
    MODEL_B_PREVIOUS_MARKS = "model_b_previous_marks_proxy"
    SIMPLE_AVERAGE_BASELINE = "simple_average_mark_baseline"


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
            if session.relative_window is not None:
                counts[session.relative_window - 1] += 1
        return tuple(counts)  # type: ignore[return-value]

    @property
    def schedule_identity(self) -> tuple[tuple[str, int | None], ...]:
        """Return the canonical schedule details used in audit records."""
        return tuple(
            (session.slot_id, session.relative_window) for session in self.sessions
        )


@dataclass(frozen=True, slots=True)
class ScheduledSession:
    """One CSP session and its optional NN click-history window."""

    slot_id: str
    relative_window: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.slot_id, str) or not self.slot_id:
            raise CandidateEvaluationError("slot_id must be a non-empty string")
        if self.relative_window is not None and (
            isinstance(self.relative_window, bool)
            or not isinstance(self.relative_window, int)
            or self.relative_window not in {1, 2, 3, 4}
        ):
            raise CandidateEvaluationError(
                "relative_window must be absent or from 1 through 4"
            )


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
    artifact_digest: str | None

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict marks from a strict-schema feature batch."""


def verify_predictor_integrity(
    predictor: object, expected_digest: str | None = None
) -> str:
    """Verify a fitted NN artifact or explicitly fitted average baseline."""
    if getattr(predictor, "prediction_family", None) == "simple_average_mark":
        verifier = getattr(predictor, "verify_integrity", None)
        digest = verifier() if callable(verifier) else None
        if (
            not isinstance(digest, str)
            or digest != getattr(predictor, "artifact_digest", None)
            or (expected_digest is not None and digest != expected_digest)
        ):
            raise ValueError("average predictor provenance is invalid or changed")
        return digest
    provenance = verify_trusted_artifact_integrity(predictor)
    if expected_digest is not None and provenance.digest != expected_digest:
        raise ValueError("loaded model digest differs from expected provenance")
    return provenance.digest


@dataclass(frozen=True, slots=True)
class CandidatePredictionResult:
    """Prediction and immutable audit data bound to one candidate identity."""

    identity: CandidateIdentity
    phase: EvaluationPhase
    status: EvaluationStatus
    mapping: MappingEvaluationSnapshot
    prediction: float | None
    schedule_identity: tuple[tuple[str, int | None], ...] | None
    csp_verification_identity: str | None
    monotonicity_report: MonotonicityReport | None
    error_type: str | None = None
    error_message: str | None = None
    whole_plan_identity: str | None = None
    problem_identity: str | None = None
    verification_policy_identity: str | None = None
    verification_policy_version: str | None = None
    fitted_artifact_digest: str | None = None
    model_provenance_identity: str | None = None
    model_mode: NNModelMode | None = None
    feature_schema_identity: str | None = None
    prediction_identity: str | None = None
    _predictor_reference: weakref.ReferenceType[object] | None = field(
        default=None, repr=False, compare=False
    )

    def assert_candidate(self, identity: CandidateIdentity) -> None:
        """Reject result reuse for a different candidate or mapping scenario."""
        if self.identity != identity:
            raise CandidateEvaluationError(
                "prediction result candidate identity mismatch"
            )

    def usable_for_goal_test(
        self,
        identity: CandidateIdentity,
        schedule_identity: tuple[tuple[str, int | None], ...] | None = None,
        csp_verification_identity: str | None = None,
        whole_plan_identity: str | None = None,
        problem_identity: str | None = None,
        model_provenance_identity: str | None = None,
        fitted_artifact_digest: str | None = None,
        model_mode: NNModelMode | None = None,
        feature_schema_identity: str | None = None,
    ) -> bool:
        """Check candidate, schedule, and verification identities before use."""
        self.assert_candidate(identity)
        predictor = (
            None if self._predictor_reference is None else self._predictor_reference()
        )
        if predictor is None:
            return False
        try:
            fitted_digest = verify_predictor_integrity(
                predictor, self.fitted_artifact_digest
            )
        except (TypeError, ValueError, RuntimeError):
            return False
        return (
            self.phase is EvaluationPhase.FINAL_SCHEDULED
            and self.status is EvaluationStatus.PREDICTED
            and schedule_identity is not None
            and schedule_identity == self.schedule_identity
            and csp_verification_identity is not None
            and csp_verification_identity == self.csp_verification_identity
            and whole_plan_identity is not None
            and whole_plan_identity == self.whole_plan_identity
            and problem_identity is not None
            and problem_identity == self.problem_identity
            and model_provenance_identity is not None
            and model_provenance_identity == self.model_provenance_identity
            and fitted_artifact_digest is not None
            and fitted_artifact_digest == self.fitted_artifact_digest
            and model_mode is not None
            and model_mode is self.model_mode
            and feature_schema_identity is not None
            and feature_schema_identity == self.feature_schema_identity
            and fitted_digest == self.fitted_artifact_digest
            and self.prediction_identity == _prediction_identity_for_result(self)
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
        verification: WholePlanVerificationResult,
        context: CandidateFeatureContext,
        *,
        expected_whole_plan_identity: str,
        expected_problem_identity: str,
        expected_policy_identity: str,
        expected_policy_version: str,
        expected_model_provenance_identity: str,
        slot_relative_windows: Mapping[str, int],
    ) -> CandidatePredictionResult:
        """Predict only after matching whole-plan verification provenance."""


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


def _is_previous_marks_schema(feature_names: tuple[str, ...]) -> bool:
    type_columns = STRICT_FFF_FEATURE_NAMES[:3]
    metadata = STRICT_FFF_FEATURE_NAMES[3:7]
    previous_marks = tuple(
        name
        for lag in range(1, 6)
        for name in (
            f"previous_assessment_score_lag_{lag}",
            f"previous_assessment_score_lag_{lag}_available",
        )
    )
    expected = (
        *type_columns,
        *metadata,
        *previous_marks,
        *STRICT_FFF_FEATURE_NAMES[7:],
    )
    return feature_names == expected


def _prediction_identity_for_result(result: CandidatePredictionResult) -> str:
    """Hash all immutable inputs and output that define a prediction record."""
    payload = {
        "candidate": {
            "student": result.identity.student_identity,
            "assignment": result.identity.assignment_identity,
            "allocation": result.identity.canonical_allocation_identity,
            "base_context": result.identity.base_feature_context_identity,
            "mapping_scenario": result.identity.mapping_scenario_identity,
        },
        "phase": result.phase.value,
        "schedule_identity": result.schedule_identity,
        "verification_identity": result.csp_verification_identity,
        "whole_plan_identity": result.whole_plan_identity,
        "problem_identity": result.problem_identity,
        "policy_identity": result.verification_policy_identity,
        "policy_version": result.verification_policy_version,
        "artifact_digest": result.fitted_artifact_digest,
        "model_provenance_identity": result.model_provenance_identity,
        "model_mode": None if result.model_mode is None else result.model_mode.value,
        "feature_schema_identity": result.feature_schema_identity,
        "status": result.status.value,
        "prediction": result.prediction,
        "error_type": result.error_type,
        "error_message": result.error_message,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SessionNNAdapter:
    """Map candidates through an explicitly identified fitted NN artifact.

    The loader-supplied artifact digest binds the exact persisted prediction
    inputs; schema and predictor class are not sufficient to identify weights.
    """

    predictor: StrictPredictor
    mapping_config: SessionClickMappingConfig
    model_mode: NNModelMode = field(init=False)
    fitted_artifact_digest: str = field(init=False)
    feature_schema_identity: str = field(init=False)
    model_provenance_identity: str = field(init=False)
    _artifact_provenance: _TrustedArtifactProvenance | None = field(
        init=False, repr=False
    )
    _monotonicity_observations: dict[
        MonotonicityContext, IncrementalMonotonicityReporter
    ] = field(default_factory=dict, init=False, repr=False)
    _final_prediction_cache: dict[tuple[str, FeatureSnapshot], float] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        feature_names = tuple(self.predictor.feature_names)
        family = getattr(self.predictor, "prediction_family", "neural_network")
        if family == "simple_average_mark":
            if feature_names != STRICT_FFF_FEATURE_NAMES:
                raise CandidateEvaluationError(
                    "the average-mark baseline requires the Model A strict schema"
                )
            model_mode = NNModelMode.SIMPLE_AVERAGE_BASELINE
            artifact_provenance = None
        elif feature_names == STRICT_FFF_FEATURE_NAMES:
            model_mode = NNModelMode.MODEL_A_STRICT
            artifact_provenance = trusted_artifact_provenance(self.predictor)
        elif _is_previous_marks_schema(feature_names):
            model_mode = NNModelMode.MODEL_B_PREVIOUS_MARKS
            artifact_provenance = trusted_artifact_provenance(self.predictor)
        else:
            raise CandidateEvaluationError(
                "predictor schema/order differs from Model A and Model B schemas"
            )
        if (
            tuple(self.predictor.assessment_type_columns)
            != STRICT_FFF_FEATURE_NAMES[:3]
        ):
            raise CandidateEvaluationError(
                "predictor assessment type columns differ from strict FFF schema"
            )
        object.__setattr__(self, "model_mode", model_mode)
        if artifact_provenance is None:
            if model_mode is not NNModelMode.SIMPLE_AVERAGE_BASELINE:
                raise CandidateEvaluationError(
                    "predictor was not created by the verified artifact loader"
                )
            try:
                artifact_digest = verify_predictor_integrity(self.predictor)
            except ValueError as error:
                raise CandidateEvaluationError(str(error)) from error
        else:
            try:
                verify_trusted_artifact_integrity(self.predictor, artifact_provenance)
            except ValueError as error:
                raise CandidateEvaluationError(str(error)) from error
            expected_mode = (
                "strict"
                if model_mode is NNModelMode.MODEL_A_STRICT
                else "proxy"
            )
            if artifact_provenance.mode != expected_mode:
                raise CandidateEvaluationError(
                    "artifact mode and predictor schema differ"
                )
            artifact_digest = artifact_provenance.digest
        if not isinstance(artifact_digest, str) or not re.fullmatch(
            r"[0-9a-f]{64}", artifact_digest
        ):
            raise CandidateEvaluationError("trusted artifact digest is malformed")
        schema_payload = {
            "feature_names": feature_names,
            "assessment_type_columns": tuple(self.predictor.assessment_type_columns),
        }
        schema_identity = hashlib.sha256(
            json.dumps(schema_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        provenance_payload = {
            "artifact_digest": artifact_digest,
            "feature_schema_identity": schema_identity,
            "mode": self.model_mode.value,
            "prediction_family": family,
        }
        provenance_identity = hashlib.sha256(
            json.dumps(
                provenance_payload, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        object.__setattr__(self, "fitted_artifact_digest", artifact_digest)
        object.__setattr__(self, "feature_schema_identity", schema_identity)
        object.__setattr__(self, "model_provenance_identity", provenance_identity)
        object.__setattr__(self, "_artifact_provenance", artifact_provenance)
        object.__setattr__(self, "prediction_family", family)

    def _verify_integrity(self) -> None:
        """Recheck the baseline digest or trusted NN artifact before use."""
        try:
            if self._artifact_provenance is None:
                verify_predictor_integrity(self.predictor, self.fitted_artifact_digest)
            else:
                verify_trusted_artifact_integrity(
                    self.predictor, self._artifact_provenance
                )
        except ValueError as error:
            raise CandidateEvaluationError(str(error)) from error

    @property
    def model_identity(self) -> str:
        """Compatibility alias for the complete model provenance identity."""
        return self.model_provenance_identity

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
        verification: WholePlanVerificationResult,
        context: CandidateFeatureContext,
        *,
        expected_whole_plan_identity: str,
        expected_problem_identity: str,
        expected_policy_identity: str,
        expected_policy_version: str,
        expected_model_provenance_identity: str,
        slot_relative_windows: Mapping[str, int],
    ) -> CandidatePredictionResult:
        """Evaluate only a complete, matching, feasible whole-plan schedule."""
        if expected_model_provenance_identity != self.model_provenance_identity:
            raise CandidateEvaluationError("model provenance identity mismatch")
        self._verify_integrity()
        expected_identity = self.candidate_identity(allocation, context)
        if not isinstance(verification, WholePlanVerificationResult):
            raise CandidateEvaluationError(
                "final evaluation requires a WholePlanVerificationResult"
            )
        if verification.status is not WholePlanVerificationStatus.FEASIBLE:
            raise CandidateEvaluationError(
                "CSP verification did not establish feasibility"
            )
        try:
            validate_whole_plan_verification_result(
                verification,
                expected_whole_plan_identity=expected_whole_plan_identity,
                expected_problem_identity=expected_problem_identity,
                expected_policy_identity=expected_policy_identity,
                expected_policy_version=expected_policy_version,
            )
        except ValueError as error:
            raise CandidateEvaluationError(str(error)) from error
        manifest = verification.whole_plan_manifest
        if (
            manifest.mapping_scenario_identity
            != expected_identity.mapping_scenario_identity
        ):
            raise CandidateEvaluationError("whole-plan mapping scenario mismatch")
        plan = manifest.assignment(context.assignment_identity)
        candidate = plan.candidate_identity
        if (
            candidate.student_identity != expected_identity.student_identity
            or candidate.assignment_identity != expected_identity.assignment_identity
            or candidate.canonical_allocation_identity
            != expected_identity.canonical_allocation_identity
            or candidate.base_feature_context_identity
            != expected_identity.base_feature_context_identity
            or candidate.mapping_scenario_identity
            != expected_identity.mapping_scenario_identity
        ):
            raise CandidateEvaluationError("whole-plan candidate identity mismatch")
        if not isinstance(slot_relative_windows, Mapping):
            raise CandidateEvaluationError("slot relative-window mapping is required")
        selected = next(
            item
            for item in verification.schedule
            if item.assignment_identity == context.assignment_identity
        )
        if len(selected.sessions) != sum(allocation.weekly_sessions):
            raise CandidateEvaluationError(
                "CSP schedule session count differs from candidate allocation"
            )
        try:
            schedule = ScheduleData(
                tuple(
                    ScheduledSession(
                        session.slot_id, slot_relative_windows.get(session.slot_id)
                    )
                    for session in selected.sessions
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CandidateEvaluationError(
                "verified schedule lacks a valid relative-window mapping"
            ) from error
        mapping = self._map(schedule.weekly_sessions, context)
        comparison_context = MonotonicityContext(
            student_id=context.student_identity,
            assignment_id=context.assignment_identity,
            base_feature_context_id=expected_identity.base_feature_context_identity,
            mapping_scenario_id=self.mapping_scenario_identity,
        )
        reporter = self._monotonicity_observations.setdefault(
            comparison_context, IncrementalMonotonicityReporter()
        )
        result = self._predict(
            expected_identity,
            EvaluationPhase.FINAL_SCHEDULED,
            mapping,
            schedule_identity=schedule.schedule_identity,
            csp_verification_identity=verification.verification_identity,
            monotonicity_report=None,
            whole_plan_identity=verification.whole_plan_identity,
            problem_identity=verification.problem_identity,
            verification_policy_identity=verification.verification_policy_identity,
            verification_policy_version=verification.verification_policy_version,
        )
        if (
            result.status is EvaluationStatus.PREDICTED
            and result.prediction is not None
        ):
            report = reporter.upsert(
                MonotonicityObservation(
                    weekly_sessions=schedule.weekly_sessions,
                    predicted_mark=result.prediction,
                    context=comparison_context,
                    schedule_identity=schedule.schedule_identity,
                )
            )
            result = replace(result, monotonicity_report=report)
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
        schedule_identity: tuple[tuple[str, int | None], ...] | None,
        csp_verification_identity: str | None,
        monotonicity_report: MonotonicityReport | None,
        whole_plan_identity: str | None = None,
        problem_identity: str | None = None,
        verification_policy_identity: str | None = None,
        verification_policy_version: str | None = None,
    ) -> CandidatePredictionResult:
        self._verify_integrity()
        snapshot = MappingEvaluationSnapshot.from_mapping(mapping)
        provenance = {
            "fitted_artifact_digest": self.fitted_artifact_digest,
            "model_provenance_identity": self.model_provenance_identity,
            "model_mode": self.model_mode,
            "feature_schema_identity": self.feature_schema_identity,
        }
        cache_key = (self.model_provenance_identity, snapshot.features)
        cache_enabled = isinstance(
            self._artifact_provenance, _TrustedArtifactProvenance
        )
        cached_prediction = (
            self._final_prediction_cache.get(cache_key)
            if phase is EvaluationPhase.FINAL_SCHEDULED and cache_enabled
            else None
        )
        try:
            if cached_prediction is None:
                output = np.asarray(
                    self.predictor.predict(snapshot.candidate_features.copy(deep=True))
                )
                if output.size != 1:
                    raise ValueError("predictor must return exactly one mark")
                prediction = float(output.reshape(-1)[0])
                if not np.isfinite(prediction) or not 0 <= prediction <= 100:
                    raise ValueError(
                        "predictor returned a mark outside finite [0, 100]"
                    )
            else:
                prediction = cached_prediction
        except Exception as error:
            result = CandidatePredictionResult(
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
                whole_plan_identity=whole_plan_identity,
                problem_identity=problem_identity,
                verification_policy_identity=verification_policy_identity,
                verification_policy_version=verification_policy_version,
                **provenance,
                _predictor_reference=weakref.ref(self.predictor),
            )
            return replace(
                result,
                prediction_identity=_prediction_identity_for_result(result),
            )
        result = CandidatePredictionResult(
            identity=identity,
            phase=phase,
            status=EvaluationStatus.PREDICTED,
            mapping=snapshot,
            prediction=prediction,
            schedule_identity=schedule_identity,
            csp_verification_identity=csp_verification_identity,
            monotonicity_report=monotonicity_report,
            whole_plan_identity=whole_plan_identity,
            problem_identity=problem_identity,
            verification_policy_identity=verification_policy_identity,
            verification_policy_version=verification_policy_version,
            **provenance,
            _predictor_reference=weakref.ref(self.predictor),
        )
        self._verify_integrity()
        if (
            phase is EvaluationPhase.FINAL_SCHEDULED
            and cache_enabled
            and cached_prediction is None
        ):
            self._final_prediction_cache[cache_key] = prediction
        return replace(
            result, prediction_identity=_prediction_identity_for_result(result)
        )


@dataclass(frozen=True)
class SessionAverageMarkAdapter(SessionNNAdapter):
    """Planner adapter for a fitted, training-only average-mark baseline."""

    def __post_init__(self) -> None:
        if getattr(self.predictor, "prediction_family", None) != "simple_average_mark":
            raise CandidateEvaluationError(
                "SessionAverageMarkAdapter requires the simple-average predictor"
            )
        super().__post_init__()
