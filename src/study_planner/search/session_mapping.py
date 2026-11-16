"""Synthetic session-to-click mapping and monotonicity audit utilities.

The click increments are scenario inputs derived from training-window click
totals. They are not measured session durations or causal effects.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from types import MappingProxyType

import numpy as np
import pandas as pd

from study_planner.data.feature_engineering.constants import (
    ACTIVITY_GROUPS,
    IDENTIFIER_COLUMNS,
    STRICT_FFF_FEATURE_NAMES,
    TARGET_COLUMN,
    click_feature_name,
)
from study_planner.nn.dataset import (
    DatasetSchemaError,
    prepare_previous_marks_inference_frame,
)


class ClickScenario(str, Enum):
    """Documented synthetic click-perturbation sensitivity scenarios."""

    LOW = "low"
    CENTRAL = "central"
    HIGH = "high"


SCENARIO_CLICKS: Mapping[ClickScenario, int] = MappingProxyType(
    {
        ClickScenario.LOW: 22,
        ClickScenario.CENTRAL: 62,
        ClickScenario.HIGH: 121,
    }
)
"""Training-anchored scenarios, not empirically measured session lengths."""


class MappingValidationError(ValueError):
    """Raised when a strict feature row or mapping configuration is invalid."""


@dataclass(frozen=True, slots=True)
class ClickSupportBoundary:
    """Training-derived marginal support for one strict click feature."""

    minimum: float
    p99: float
    maximum: float


class MarginalSupportStatus(str, Enum):
    """Per-feature status against observed training minimum and maximum."""

    IN_RANGE = "marginal_in_range"
    OUT_OF_RANGE = "marginal_out_of_range"
    UNASSESSED = "marginal_unassessed"


class JointSupportStatus(str, Enum):
    """Multivariate support status; only unassessed is currently defensible."""

    SUPPORTED = "joint_supported"
    OUT_OF_SUPPORT = "joint_out_of_support"
    UNASSESSED = "joint_unassessed"


# Source: docs/search_integration_readiness.md, strict train split only
# (25,229 rows; presentations 2013B/2013J). Min/max are hard observed-range
# flags; p99 is a warning threshold, never a clipping or rejection bound.
TRAINING_CLICK_SUPPORT: Mapping[str, ClickSupportBoundary] = MappingProxyType(
    {
        "click_content_navigation_week_1": ClickSupportBoundary(0, 342.72, 3796),
        "click_content_navigation_week_2": ClickSupportBoundary(0, 361, 4195),
        "click_content_navigation_week_3": ClickSupportBoundary(0, 343.72, 736),
        "click_content_navigation_week_4": ClickSupportBoundary(0, 501, 1546),
        "click_assessment_activity_week_1": ClickSupportBoundary(0, 429, 1676),
        "click_assessment_activity_week_2": ClickSupportBoundary(0, 502, 2255),
        "click_assessment_activity_week_3": ClickSupportBoundary(0, 497, 1224),
        "click_assessment_activity_week_4": ClickSupportBoundary(0, 602, 1828),
        "click_collaboration_week_1": ClickSupportBoundary(0, 137, 997),
        "click_collaboration_week_2": ClickSupportBoundary(0, 152, 1018),
        "click_collaboration_week_3": ClickSupportBoundary(0, 127.72, 786),
        "click_collaboration_week_4": ClickSupportBoundary(0, 146, 994),
    }
)

CLICK_FEATURE_NAMES: tuple[str, ...] = tuple(TRAINING_CLICK_SUPPORT)
ALLOWED_ACTIVITY_GROUPS = frozenset(ACTIVITY_GROUPS)
WINDOWS = frozenset({1, 2, 3, 4})
@dataclass(frozen=True, slots=True)
class SessionClickMappingConfig:
    """Select one synthetic click scenario and an existing activity group."""

    scenario: ClickScenario
    activity_group: str = "content_navigation"

    def __post_init__(self) -> None:
        if not isinstance(self.scenario, ClickScenario):
            raise MappingValidationError("scenario must be a ClickScenario")
        if (
            not isinstance(self.activity_group, str)
            or self.activity_group not in ALLOWED_ACTIVITY_GROUPS
        ):
            raise MappingValidationError(
                f"Unknown activity group: {self.activity_group!r}"
            )

    @property
    def clicks_per_session(self) -> int:
        """Return the configured synthetic increment, not a measured duration."""
        return SCENARIO_CLICKS[self.scenario]


@dataclass(frozen=True, slots=True)
class ClickSupportDiagnostic:
    """Support flags for one candidate click feature value."""

    feature_name: str
    candidate_value: float | None
    boundary: ClickSupportBoundary
    marginal_status: MarginalSupportStatus
    exceeds_p99: bool

    @property
    def within_observed_range(self) -> bool:
        """Whether this feature lies within its marginal training range."""
        return self.marginal_status is MarginalSupportStatus.IN_RANGE


@dataclass(frozen=True, slots=True)
class SessionClickMappingResult:
    """Mapped feature copy and auditable scenario/support metadata."""

    candidate_features: pd.DataFrame
    weekly_sessions: tuple[int, int, int, int]
    scenario: ClickScenario
    activity_group: str
    clicks_per_session: int
    changed_features: tuple[str, ...]
    missing_numeric_features: tuple[str, ...]
    marginal_status: MarginalSupportStatus
    joint_support_status: JointSupportStatus
    support_diagnostics: tuple[ClickSupportDiagnostic, ...]

    @property
    def out_of_support_features(self) -> tuple[str, ...]:
        """Return features outside their observed training min/max range."""
        return tuple(
            diagnostic.feature_name
            for diagnostic in self.support_diagnostics
            if diagnostic.marginal_status is MarginalSupportStatus.OUT_OF_RANGE
        )

    @property
    def has_out_of_support_values(self) -> bool:
        """Whether any candidate click value is outside observed support."""
        return bool(self.out_of_support_features)


@dataclass(frozen=True, slots=True)
class MonotonicityObservation:
    """Prediction associated with counts in relative click windows 1–4."""

    weekly_sessions: tuple[int, int, int, int]
    predicted_mark: float
    context: MonotonicityContext
    schedule_identity: tuple[Hashable, ...]


@dataclass(frozen=True, slots=True)
class MonotonicityContext:
    """Shared identity for observations eligible for monotonicity comparison."""

    student_id: str
    assignment_id: str
    base_feature_context_id: str
    mapping_scenario_id: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (
                self.student_id,
                self.assignment_id,
                self.base_feature_context_id,
                self.mapping_scenario_id,
            )
        ):
            raise ValueError("monotonicity context identity fields cannot be empty")


@dataclass(frozen=True, slots=True)
class MonotonicityViolation:
    """A one-session allocation increase that lowered predicted marks."""

    parent_sessions: tuple[int, int, int, int]
    child_sessions: tuple[int, int, int, int]
    changed_window: int
    parent_mark: float
    child_mark: float
    prediction_delta: float
    parent_schedule_identity: tuple[Hashable, ...]
    child_schedule_identity: tuple[Hashable, ...]


@dataclass(frozen=True, slots=True)
class MonotonicityReport:
    """Summary of all evaluated one-session monotonicity comparisons."""

    comparisons: int
    violations: tuple[MonotonicityViolation, ...]
    tolerance: float
    observations: tuple[MonotonicityObservation, ...]

    @property
    def violation_rate(self) -> float:
        """Return violations/comparisons, or zero when no pairs were compared."""
        return len(self.violations) / self.comparisons if self.comparisons else 0.0


def _validate_weekly_sessions(
    weekly_sessions: Mapping[int, int],
) -> tuple[int, int, int, int]:
    """Validate and canonicalize counts for the four relative click windows."""
    if not isinstance(weekly_sessions, Mapping):
        raise MappingValidationError("weekly_sessions must be a mapping")
    counts = [0, 0, 0, 0]
    for week, count in weekly_sessions.items():
        if isinstance(week, bool) or not isinstance(week, int) or week not in WINDOWS:
            raise MappingValidationError("week keys must be integers from 1 through 4")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise MappingValidationError("session counts must be non-negative integers")
        counts[week - 1] = count
    return tuple(counts)  # type: ignore[return-value]


def _validate_feature_row(
    features: pd.DataFrame, feature_names: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate all strict inputs and return schema and missing numerics."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if len(features) != 1:
        raise MappingValidationError("features must contain exactly one row")
    expected = tuple(feature_names)
    if tuple(features.columns) != expected:
        raise MappingValidationError(
            "feature columns/order must exactly match the supplied strict schema"
        )
    forbidden = set(IDENTIFIER_COLUMNS) | {TARGET_COLUMN}
    if forbidden.intersection(expected):
        raise MappingValidationError(
            "model inputs cannot contain identifiers or target"
        )
    if expected != STRICT_FFF_FEATURE_NAMES:
        try:
            prepare_previous_marks_inference_frame(
                features, expected, STRICT_FFF_FEATURE_NAMES[:3]
            )
        except (DatasetSchemaError, TypeError, ValueError) as error:
            raise MappingValidationError(
                "feature_names must match Model A strict or Model B proxy schema"
            ) from error

    type_columns = (
        "assessment_type_cma",
        "assessment_type_exam",
        "assessment_type_tma",
    )
    raw_types = features.loc[:, list(type_columns)]
    type_values = raw_types.apply(pd.to_numeric, errors="coerce")
    invalid_types = (raw_types.notna() & type_values.isna()).any().any()
    observed_types = type_values.to_numpy(dtype=np.float64)
    observed_types = observed_types[~np.isnan(observed_types)]
    if invalid_types or not np.isin(observed_types, (0, 1)).all():
        raise MappingValidationError("assessment type indicators must be numeric 0/1")
    missing_type_rows = type_values.isna().any(axis=1)
    complete_type_rows = type_values.loc[~missing_type_rows]
    if not (complete_type_rows.sum(axis=1) == 1).all():
        raise MappingValidationError(
            "complete assessment type rows need exactly one active indicator"
        )

    numeric_names = tuple(name for name in expected if name not in type_columns)
    raw_numeric = features.loc[:, list(numeric_names)]
    numeric_values_frame = raw_numeric.apply(pd.to_numeric, errors="coerce")
    invalid_numeric = (raw_numeric.notna() & numeric_values_frame.isna()).any().any()
    numeric_values = numeric_values_frame.to_numpy(dtype=np.float64)
    if invalid_numeric or np.isinf(numeric_values).any():
        raise MappingValidationError("numeric features must be numeric and finite")

    availability = numeric_values_frame["assessment_weight_available"].dropna()
    if not availability.isin([0, 1]).all():
        raise MappingValidationError("assessment_weight_available must be 0 or 1")
    click_values = numeric_values_frame.loc[:, list(CLICK_FEATURE_NAMES)]
    click_array = click_values.to_numpy(dtype=np.float64)
    if (click_array[~np.isnan(click_array)] < 0).any():
        raise MappingValidationError("click features cannot be negative")
    missing_numeric = tuple(
        name for name in numeric_names if numeric_values_frame[name].isna().any()
    )
    return expected, missing_numeric


def map_sessions_to_clicks(
    features: pd.DataFrame,
    weekly_sessions: Mapping[int, int],
    config: SessionClickMappingConfig,
    feature_names: Sequence[str],
) -> SessionClickMappingResult:
    """Apply explicit weekly session counts to a copied strict feature row.

    ``weekly_sessions`` uses the existing deadline-relative feature windows:
    keys 1–4 correspond to the four strict click columns, not calendar weeks.
    Every session adds the selected scenario increment to one configured
    activity group in its allocated window. This is a deterministic synthetic
    perturbation, not an observed session duration or causal effect.

    The exact ordered strict schema must be supplied, typically from
    ``RegressionPredictor.feature_names``. Supported missing numeric values
    are preserved, but a session cannot be added to a missing click total.
    Out-of-range candidates are returned unchanged with marginal diagnostics.
    Joint support remains explicitly unassessed; no clipping or repair occurs.
    """
    if not isinstance(config, SessionClickMappingConfig):
        raise MappingValidationError("config must be SessionClickMappingConfig")
    ordered_names, missing_numeric = _validate_feature_row(features, feature_names)
    counts = _validate_weekly_sessions(weekly_sessions)
    candidate = features.copy(deep=True)
    increment = config.clicks_per_session
    changed: list[str] = []

    for week, session_count in enumerate(counts, start=1):
        if session_count == 0:
            continue
        name = click_feature_name(config.activity_group, week)
        if pd.isna(candidate.iloc[0][name]):
            raise MappingValidationError(
                f"cannot add sessions to missing click feature {name}"
            )
        try:
            value = float(candidate.iloc[0][name]) + session_count * increment
        except OverflowError as error:
            raise MappingValidationError(
                f"mapped click value for {name} is not finite"
            ) from error
        if not math.isfinite(value):
            raise MappingValidationError(f"mapped click value for {name} is not finite")
        candidate.at[candidate.index[0], name] = value
        changed.append(name)

    diagnostics: list[ClickSupportDiagnostic] = []
    for name in CLICK_FEATURE_NAMES:
        raw_value = candidate.iloc[0][name]
        boundary = TRAINING_CLICK_SUPPORT[name]
        if pd.isna(raw_value):
            diagnostics.append(
                ClickSupportDiagnostic(
                    feature_name=name,
                    candidate_value=None,
                    boundary=boundary,
                    marginal_status=MarginalSupportStatus.UNASSESSED,
                    exceeds_p99=False,
                )
            )
            continue
        value = float(raw_value)
        in_range = boundary.minimum <= value <= boundary.maximum
        diagnostics.append(
            ClickSupportDiagnostic(
                feature_name=name,
                candidate_value=value,
                boundary=boundary,
                marginal_status=(
                    MarginalSupportStatus.IN_RANGE
                    if in_range
                    else MarginalSupportStatus.OUT_OF_RANGE
                ),
                exceeds_p99=value > boundary.p99,
            )
        )
    marginal_status = (
        MarginalSupportStatus.OUT_OF_RANGE
        if any(
            diagnostic.marginal_status is MarginalSupportStatus.OUT_OF_RANGE
            for diagnostic in diagnostics
        )
        else MarginalSupportStatus.UNASSESSED
        if any(
            diagnostic.marginal_status is MarginalSupportStatus.UNASSESSED
            for diagnostic in diagnostics
        )
        else MarginalSupportStatus.IN_RANGE
    )
    if tuple(candidate.columns) != ordered_names:
        raise RuntimeError("mapping unexpectedly changed feature schema/order")

    return SessionClickMappingResult(
        candidate_features=candidate,
        weekly_sessions=counts,
        scenario=config.scenario,
        activity_group=config.activity_group,
        clicks_per_session=increment,
        changed_features=tuple(changed),
        missing_numeric_features=missing_numeric,
        marginal_status=marginal_status,
        joint_support_status=JointSupportStatus.UNASSESSED,
        support_diagnostics=tuple(diagnostics),
    )


def _validate_mark(value: Real, name: str) -> float:
    """Validate a finite numeric predicted mark or tolerance."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        numeric = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def evaluate_monotonicity(
    observations: Sequence[MonotonicityObservation],
    tolerance: float = 0.0,
) -> MonotonicityReport:
    """Report every observed one-session increase that lowers predicted marks.

    Only pairs differing by exactly one session in one of the four windows
    are compared. A decrease larger than ``tolerance`` is a violation. The
    function reports model behavior under supplied synthetic inputs; it does
    not assert a causal or real-world monotonic relationship.
    """
    numeric_tolerance = _validate_mark(tolerance, "tolerance")
    if numeric_tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    seen_observations: set[
        tuple[MonotonicityContext, tuple[int, int, int, int], tuple[Hashable, ...]]
    ] = set()
    normalized: list[MonotonicityObservation] = []
    for observation in observations:
        if len(observation.weekly_sessions) != 4:
            raise ValueError("weekly_sessions must contain exactly four counts")
        counts = _validate_weekly_sessions(
            {week: count for week, count in enumerate(observation.weekly_sessions, 1)}
        )
        if not isinstance(observation.schedule_identity, tuple):
            raise ValueError("schedule_identity must be a canonical tuple")
        try:
            hash(observation.schedule_identity)
        except TypeError as error:
            raise ValueError("schedule_identity must be hashable") from error
        identity = (observation.context, counts, observation.schedule_identity)
        if identity in seen_observations:
            raise ValueError(
                "monotonicity observations must be unique within each context"
            )
        seen_observations.add(identity)
        normalized.append(
            MonotonicityObservation(
                weekly_sessions=counts,
                predicted_mark=_validate_mark(
                    observation.predicted_mark, "predicted_mark"
                ),
                context=observation.context,
                schedule_identity=observation.schedule_identity,
            )
        )

    violations: list[MonotonicityViolation] = []
    comparisons = 0
    for parent in normalized:
        for child in normalized:
            if parent.context != child.context:
                continue
            differences = tuple(
                child_count - parent_count
                for parent_count, child_count in zip(
                    parent.weekly_sessions, child.weekly_sessions, strict=True
                )
            )
            if any(difference < 0 for difference in differences):
                continue
            if sum(differences) != 1:
                continue
            comparisons += 1
            changed_window = differences.index(1) + 1
            delta = child.predicted_mark - parent.predicted_mark
            if delta < -numeric_tolerance:
                violations.append(
                    MonotonicityViolation(
                        parent_sessions=parent.weekly_sessions,
                        child_sessions=child.weekly_sessions,
                        changed_window=changed_window,
                        parent_mark=parent.predicted_mark,
                        child_mark=child.predicted_mark,
                        prediction_delta=delta,
                        parent_schedule_identity=parent.schedule_identity,
                        child_schedule_identity=child.schedule_identity,
                    )
                )

    return MonotonicityReport(
        comparisons=comparisons,
        violations=tuple(violations),
        tolerance=numeric_tolerance,
        observations=tuple(normalized),
    )
