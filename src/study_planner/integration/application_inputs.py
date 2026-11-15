"""Validated, provenance-bound inputs for one OULAD planning request."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from study_planner.csp.core import SessionOrderingPolicy, WeeklySlot
from study_planner.integration.planner import AssignmentTarget
from study_planner.nn.dataset import (
    SPLIT_PRESENTATIONS,
    load_previous_marks_feature_data,
    load_strict_feature_data,
)
from study_planner.nn.inference import (
    PreviousMarksRegressionPredictor,
    RegressionPredictor,
    verify_trusted_artifact_integrity,
)
from study_planner.search.nn_adapter import (
    CandidateFeatureContext,
    NNModelMode,
    SessionNNAdapter,
)
from study_planner.search.session_mapping import (
    JointSupportStatus,
    MappingValidationError,
    MarginalSupportStatus,
    SessionClickMappingConfig,
    map_sessions_to_clicks,
)

if TYPE_CHECKING:
    from study_planner.integration.application_planning import (
        ApplicationPlanningBinding,
    )


class ApplicationInputError(ValueError):
    """Raised when application configuration or source rows are inconsistent."""


class OutOfSupportAction(str, Enum):
    """Explicit treatment for marginal support warnings."""

    REPORT_ONLY = "report_only"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class OuladPresentationSplit:
    """The established one-course chronological OULAD model split."""

    course_identity: str
    training_presentations: tuple[str, ...]
    testing_presentation: str

    def __post_init__(self) -> None:
        if not isinstance(self.course_identity, str) or not self.course_identity:
            raise ApplicationInputError("course identity cannot be empty")
        if not isinstance(self.training_presentations, tuple):
            raise ApplicationInputError(
                "training presentations must be an immutable tuple"
            )
        train = tuple(self.training_presentations)
        allowed_train = SPLIT_PRESENTATIONS["train"]
        if (
            not train
            or len(set(train)) != len(train)
            or not set(train) <= allowed_train
        ):
            raise ApplicationInputError(
                "training presentations must be unique presentations in the "
                "established OULAD training split"
            )
        if self.testing_presentation not in SPLIT_PRESENTATIONS["test"]:
            raise ApplicationInputError(
                "testing presentation must be in the established later OULAD test split"
            )
        presentation_order = tuple(
            sorted(
                set().union(
                    *SPLIT_PRESENTATIONS.values(),
                )
            )
        )
        test_index = presentation_order.index(self.testing_presentation)
        if any(presentation_order.index(item) >= test_index for item in train):
            raise ApplicationInputError(
                "training presentations must precede the testing presentation"
            )


@dataclass(frozen=True, slots=True)
class AllocationBounds:
    """Caller-approved inclusive session-count bounds for one assignment."""

    minimum_sessions: int
    maximum_sessions: int

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_sessions", self.minimum_sessions),
            ("maximum_sessions", self.maximum_sessions),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ApplicationInputError(f"{name} must be a non-negative integer")
        if self.minimum_sessions > self.maximum_sessions:
            raise ApplicationInputError("minimum sessions cannot exceed maximum")

    def accepts(self, session_count: int) -> bool:
        """Whether a count lies inside this explicitly supplied bound."""
        return self.minimum_sessions <= session_count <= self.maximum_sessions


@dataclass(frozen=True, slots=True)
class ApplicationAssignmentRule:
    """Explicit target, deadline eligibility, and allocation rule for an assessment."""

    assignment_identity: str
    assessment_id: str
    target_mark: float
    deadline_day: int
    eligible_slot_ids: tuple[str, ...]
    allocation_bounds: AllocationBounds

    def __post_init__(self) -> None:
        if (
            not isinstance(self.assignment_identity, str)
            or not self.assignment_identity
        ):
            raise ApplicationInputError("assignment identity cannot be empty")
        if not isinstance(self.assessment_id, str) or not self.assessment_id:
            raise ApplicationInputError("assessment identity cannot be empty")
        if (
            isinstance(self.target_mark, bool)
            or not isinstance(self.target_mark, int | float)
            or not math.isfinite(float(self.target_mark))
            or not 0 <= self.target_mark <= 100
        ):
            raise ApplicationInputError("target mark must be finite and in [0, 100]")
        if (
            isinstance(self.deadline_day, bool)
            or not isinstance(self.deadline_day, int)
            or self.deadline_day < 0
        ):
            raise ApplicationInputError("deadline day must be a non-negative integer")
        if not isinstance(self.eligible_slot_ids, tuple):
            raise ApplicationInputError("eligible slot IDs must be an immutable tuple")
        if any(
            not isinstance(slot_id, str) or not slot_id
            for slot_id in self.eligible_slot_ids
        ):
            raise ApplicationInputError("eligible slot IDs must be non-empty strings")
        if len(set(self.eligible_slot_ids)) != len(self.eligible_slot_ids):
            raise ApplicationInputError("eligible slot IDs cannot contain duplicates")
        object.__setattr__(
            self, "eligible_slot_ids", tuple(sorted(self.eligible_slot_ids))
        )
        if not isinstance(self.allocation_bounds, AllocationBounds):
            raise ApplicationInputError("assignment allocation bounds are required")

    @property
    def weeks_since_course_start(self) -> float:
        """Use the existing OULAD feature convention: deadline day divided by 7."""
        return self.deadline_day / 7.0


@dataclass(frozen=True, slots=True)
class TargetRule:
    """Versioned, explicit target marks; no target is inferred from a label."""

    identity: str
    version: str
    targets: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not self.identity or not self.version:
            raise ApplicationInputError("target rule identity and version are required")
        if not isinstance(self.targets, tuple):
            raise ApplicationInputError("target rule entries must be immutable")
        if any(not isinstance(item, tuple) or len(item) != 2 for item in self.targets):
            raise ApplicationInputError(
                "target rule entries must be assignment/mark pairs"
            )
        assignment_ids = tuple(assignment_id for assignment_id, _ in self.targets)
        if len(assignment_ids) != 3 or len(set(assignment_ids)) != 3:
            raise ApplicationInputError(
                "target rule must cover exactly three assignments"
            )
        for assignment_id, target in self.targets:
            if not assignment_id:
                raise ApplicationInputError(
                    "target assignment identity cannot be empty"
                )
            if (
                isinstance(target, bool)
                or not isinstance(target, int | float)
                or not math.isfinite(float(target))
                or not 0 <= target <= 100
            ):
                raise ApplicationInputError(
                    "target marks must be finite and in [0, 100]"
                )
        object.__setattr__(
            self, "targets", tuple(sorted(self.targets, key=lambda item: item[0]))
        )


@dataclass(frozen=True, slots=True)
class WeeklyTimeWindow:
    """Explicit local clock interval for one weekly study slot.

    Weekday uses ISO numbering (Monday=1 through Sunday=7); clock bounds are
    minutes after local midnight. The timezone identity must name the source
    catalog's timezone; no daylight-saving or calendar conversion is inferred.
    """

    weekday_iso: int
    start_minute_local: int
    end_minute_local: int
    timezone_identity: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.weekday_iso, bool)
            or not isinstance(self.weekday_iso, int)
            or not 1 <= self.weekday_iso <= 7
        ):
            raise ApplicationInputError("weekday must use ISO values 1 through 7")
        for name, value in (
            ("start minute", self.start_minute_local),
            ("end minute", self.end_minute_local),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 1440
            ):
                raise ApplicationInputError(f"{name} must be in [0, 1440]")
        if self.end_minute_local <= self.start_minute_local:
            raise ApplicationInputError(
                "weekly time window must have positive duration"
            )
        if not isinstance(self.timezone_identity, str) or not self.timezone_identity:
            raise ApplicationInputError("weekly time window needs a timezone identity")


@dataclass(frozen=True, slots=True)
class ApplicationSlot:
    """One catalog slot with a local time window and NN-relative windows."""

    slot: WeeklySlot
    time_window: WeeklyTimeWindow
    relative_windows: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.slot, WeeklySlot):
            raise ApplicationInputError("slot catalog entries must use WeeklySlot")
        if not isinstance(self.time_window, WeeklyTimeWindow):
            raise ApplicationInputError("slot needs an explicit weekly time window")
        if not isinstance(self.relative_windows, tuple):
            raise ApplicationInputError("relative windows must be immutable")
        if any(
            not isinstance(item, tuple) or len(item) != 2
            for item in self.relative_windows
        ):
            raise ApplicationInputError("slot windows must be assignment/window pairs")
        assignment_ids = tuple(
            assignment_id for assignment_id, _ in self.relative_windows
        )
        if len(assignment_ids) != 3 or len(set(assignment_ids)) != 3:
            raise ApplicationInputError(
                "each slot needs one relative-window entry for all three assignments"
            )
        if any(
            not isinstance(assignment_id, str)
            or not assignment_id
            or isinstance(window, bool)
            or not isinstance(window, int)
            or window not in {1, 2, 3, 4}
            for assignment_id, window in self.relative_windows
        ):
            raise ApplicationInputError(
                "slot relative windows must map assignments to windows 1 through 4"
            )
        object.__setattr__(
            self,
            "relative_windows",
            tuple(sorted(self.relative_windows, key=lambda item: item[0])),
        )


@dataclass(frozen=True, slots=True)
class WeeklySlotCatalog:
    """Versioned slot universe, capacity, and explicit deadline policy identity."""

    identity: str
    version: str
    weekly_capacity: int
    deadline_policy_identity: str
    deadline_policy_version: str
    slots: tuple[ApplicationSlot, ...]

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.identity,
                self.version,
                self.deadline_policy_identity,
                self.deadline_policy_version,
            )
        ):
            raise ApplicationInputError(
                "slot catalog and deadline policy identities are required"
            )
        if (
            isinstance(self.weekly_capacity, bool)
            or not isinstance(self.weekly_capacity, int)
            or self.weekly_capacity < 0
        ):
            raise ApplicationInputError(
                "weekly capacity must be a non-negative integer"
            )
        if not isinstance(self.slots, tuple) or not self.slots:
            raise ApplicationInputError(
                "slot catalog must contain immutable slot entries"
            )
        if any(not isinstance(item, ApplicationSlot) for item in self.slots):
            raise ApplicationInputError("slot catalog contains an invalid entry")
        slot_ids = tuple(item.slot.slot_id for item in self.slots)
        if len(slot_ids) != len(set(slot_ids)):
            raise ApplicationInputError("slot catalog IDs must be unique")
        assignment_ids = set(item[0] for item in self.slots[0].relative_windows)
        if any(
            set(item[0] for item in slot.relative_windows) != assignment_ids
            for slot in self.slots
        ):
            raise ApplicationInputError("all slots must map the same three assignments")

    @property
    def configuration_identity(self) -> str:
        """Digest the explicit catalog values, not only its descriptive label."""
        payload = {
            "identity": self.identity,
            "version": self.version,
            "weekly_capacity": self.weekly_capacity,
            "deadline_policy_identity": self.deadline_policy_identity,
            "deadline_policy_version": self.deadline_policy_version,
            "slots": [
                {
                    "slot_id": item.slot.slot_id,
                    "week_index": item.slot.week_index,
                    "weekday_iso": item.time_window.weekday_iso,
                    "start_minute_local": item.time_window.start_minute_local,
                    "end_minute_local": item.time_window.end_minute_local,
                    "timezone_identity": item.time_window.timezone_identity,
                    "relative_windows": item.relative_windows,
                }
                for item in sorted(self.slots, key=lambda entry: entry.slot.slot_id)
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def windows_by_assignment(
        self,
    ) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
        """Return canonical assignment-specific slot-to-window mappings."""
        assignment_ids = sorted(
            self.slots[0].relative_windows[index][0] for index in range(3)
        )
        return tuple(
            (
                assignment_id,
                tuple(
                    sorted(
                        (
                            slot.slot.slot_id,
                            dict(slot.relative_windows)[assignment_id],
                        )
                        for slot in self.slots
                    )
                ),
            )
            for assignment_id in assignment_ids
        )


@dataclass(frozen=True, slots=True)
class SupportPolicy:
    """Explicit disposition for training-reference marginal support flags."""

    identity: str
    version: str
    reference_identity: str
    out_of_support_action: OutOfSupportAction
    joint_support_status: JointSupportStatus = JointSupportStatus.UNASSESSED

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (self.identity, self.version, self.reference_identity)
        ):
            raise ApplicationInputError("support policy identities are required")
        if not isinstance(self.out_of_support_action, OutOfSupportAction):
            raise ApplicationInputError("out-of-support action must be explicit")
        if self.joint_support_status is not JointSupportStatus.UNASSESSED:
            raise ApplicationInputError(
                "joint support is unassessed by the existing feature support contract"
            )

    def permits(self, status: MarginalSupportStatus) -> bool:
        """Apply the configured disposition to a marginal support result."""
        if not isinstance(status, MarginalSupportStatus):
            raise ApplicationInputError("marginal support status has an invalid type")
        if status is MarginalSupportStatus.OUT_OF_RANGE:
            return self.out_of_support_action is OutOfSupportAction.REPORT_ONLY
        return True


@dataclass(frozen=True, slots=True)
class ModelArtifactIdentity:
    """Identity copied from a loader-verified model adapter, never user-labelled."""

    mode: NNModelMode
    artifact_digest: str
    provenance_identity: str
    feature_schema_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.mode, NNModelMode):
            raise ApplicationInputError("model mode must be Model A or Model B")
        for name, value in (
            ("artifact digest", self.artifact_digest),
            ("model provenance identity", self.provenance_identity),
            ("feature schema identity", self.feature_schema_identity),
        ):
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
            ):
                raise ApplicationInputError(
                    f"{name} must be a verified SHA-256 identity"
                )

    @classmethod
    def from_adapter(cls, adapter: SessionNNAdapter) -> ModelArtifactIdentity:
        """Capture provenance only from an adapter that already verified its loader."""
        return cls(
            mode=adapter.model_mode,
            artifact_digest=adapter.fitted_artifact_digest,
            provenance_identity=adapter.model_provenance_identity,
            feature_schema_identity=adapter.feature_schema_identity,
        )

    def matches(self, adapter: SessionNNAdapter) -> bool:
        """Whether all model provenance components still match the adapter."""
        return self == ModelArtifactIdentity.from_adapter(adapter)


@dataclass(frozen=True, slots=True)
class ApplicationInputConfig:
    """All business-sensitive planner inputs supplied with explicit provenance."""

    presentation_split: OuladPresentationSplit
    assignment_rules: tuple[ApplicationAssignmentRule, ...]
    target_rule: TargetRule
    slot_catalog: WeeklySlotCatalog
    support_policy: SupportPolicy
    click_mapping: SessionClickMappingConfig
    model_identity: ModelArtifactIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.presentation_split, OuladPresentationSplit):
            raise ApplicationInputError("an OULAD presentation split is required")
        if (
            not isinstance(self.assignment_rules, tuple)
            or len(self.assignment_rules) != 3
        ):
            raise ApplicationInputError("exactly three assignment rules are required")
        if any(
            not isinstance(rule, ApplicationAssignmentRule)
            for rule in self.assignment_rules
        ):
            raise ApplicationInputError("assignment rule has an invalid type")
        if not isinstance(self.target_rule, TargetRule):
            raise ApplicationInputError("a versioned target rule is required")
        if not isinstance(self.slot_catalog, WeeklySlotCatalog):
            raise ApplicationInputError("a versioned weekly slot catalog is required")
        if not isinstance(self.support_policy, SupportPolicy):
            raise ApplicationInputError("an explicit support policy is required")
        if not isinstance(self.click_mapping, SessionClickMappingConfig):
            raise ApplicationInputError(
                "a documented generated-data mapping rule is required"
            )
        if not isinstance(self.model_identity, ModelArtifactIdentity):
            raise ApplicationInputError("loader-verified model identity is required")
        assignment_ids = tuple(
            rule.assignment_identity for rule in self.assignment_rules
        )
        assessment_ids = tuple(rule.assessment_id for rule in self.assignment_rules)
        if len(set(assignment_ids)) != 3:
            raise ApplicationInputError("assignment identities must be unique")
        if len(set(assessment_ids)) != 3:
            raise ApplicationInputError("assessment IDs must be unique")
        if tuple(sorted(assignment_ids)) != tuple(
            sorted(assignment_id for assignment_id, _ in self.target_rule.targets)
        ):
            raise ApplicationInputError("target rule must match the three assignments")
        known_slots = {slot.slot.slot_id for slot in self.slot_catalog.slots}
        mapped_ids = {
            assignment_id
            for slot in self.slot_catalog.slots
            for assignment_id, _ in slot.relative_windows
        }
        if mapped_ids != set(assignment_ids):
            raise ApplicationInputError(
                "slot windows must cover exactly the three configured assignments"
            )
        for rule in self.assignment_rules:
            unknown = set(rule.eligible_slot_ids) - known_slots
            if unknown:
                raise ApplicationInputError(
                    f"assignment {rule.assignment_identity} references unknown slots"
                )

    @property
    def targets(self) -> tuple[AssignmentTarget, ...]:
        """Return planner target records in canonical assignment order."""
        marks = dict(self.target_rule.targets)
        return tuple(
            AssignmentTarget(rule.assignment_identity, marks[rule.assignment_identity])
            for rule in sorted(
                self.assignment_rules, key=lambda item: item.assignment_identity
            )
        )


@dataclass(frozen=True, slots=True)
class ValidatedApplicationInputs:
    """Validated three-assignment inputs ready for a planner candidate factory."""

    config: ApplicationInputConfig
    student_identity: str
    feature_contexts: tuple[CandidateFeatureContext, ...]
    model_identity: ModelArtifactIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.config, ApplicationInputConfig):
            raise ApplicationInputError("validated input config has an invalid type")
        if not isinstance(self.student_identity, str) or not self.student_identity:
            raise ApplicationInputError("validated student identity cannot be empty")
        if (
            not isinstance(self.feature_contexts, tuple)
            or len(self.feature_contexts) != 3
            or any(
                not isinstance(item, CandidateFeatureContext)
                for item in self.feature_contexts
            )
        ):
            raise ApplicationInputError(
                "validated inputs must preserve three feature contexts"
            )
        assignment_ids = tuple(
            item.assignment_identity for item in self.feature_contexts
        )
        expected_ids = tuple(
            sorted(rule.assignment_identity for rule in self.config.assignment_rules)
        )
        if (
            tuple(sorted(assignment_ids)) != expected_ids
            or len(set(assignment_ids)) != 3
        ):
            raise ApplicationInputError(
                "validated feature contexts must cover the configured assignments"
            )
        if self.model_identity != self.config.model_identity:
            raise ApplicationInputError(
                "validated model identity differs from application configuration"
            )

    @property
    def assignment_ids(self) -> tuple[str, str, str]:
        """Return the canonical three assignment identities."""
        return tuple(context.assignment_identity for context in self.feature_contexts)  # type: ignore[return-value]


def load_session_nn_adapter(
    artifact_directory: Path,
    model_mode: NNModelMode,
    click_mapping: SessionClickMappingConfig,
) -> SessionNNAdapter:
    """Load the selected persisted Model A or Model B artifact and bind identity."""
    if model_mode is NNModelMode.MODEL_A_STRICT:
        predictor = RegressionPredictor.from_artifact_dir(artifact_directory)
    elif model_mode is NNModelMode.MODEL_B_PREVIOUS_MARKS:
        predictor = PreviousMarksRegressionPredictor.from_artifact_dir(
            artifact_directory
        )
    else:
        raise ApplicationInputError("unsupported fitted model mode")
    try:
        return SessionNNAdapter(predictor, click_mapping)
    except (TypeError, ValueError) as error:
        raise ApplicationInputError(
            f"model artifact is incompatible: {error}"
        ) from error


def build_application_inputs(
    config: ApplicationInputConfig,
    *,
    student_identity: str,
    feature_rows: pd.DataFrame,
    adapter: SessionNNAdapter,
) -> ValidatedApplicationInputs:
    """Select and validate one student's three later-presentation assessment rows.

    `feature_rows` may contain the historical target column for audit/evaluation;
    it is deliberately excluded from every `CandidateFeatureContext`.
    Deadline-eligible slot IDs and assignment-relative slot windows are explicit
    configured inputs because the available P1 contract does not define their
    calendar boundary conversion.
    """
    if not isinstance(config, ApplicationInputConfig):
        raise ApplicationInputError("application input config has an invalid type")
    if not isinstance(student_identity, str) or not student_identity:
        raise ApplicationInputError("student identity cannot be empty")
    if not isinstance(feature_rows, pd.DataFrame):
        raise ApplicationInputError("feature rows must be a pandas DataFrame")
    if not isinstance(adapter, SessionNNAdapter):
        raise ApplicationInputError("a loader-verified SessionNNAdapter is required")
    try:
        verify_trusted_artifact_integrity(adapter.predictor)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ApplicationInputError(
            f"fitted model integrity check failed: {error}"
        ) from error
    if not config.model_identity.matches(adapter):
        raise ApplicationInputError(
            "configured model identity does not match loaded model"
        )
    if config.click_mapping != adapter.mapping_config:
        raise ApplicationInputError("generated-data rule differs from adapter mapping")
    required_metadata = {
        "code_module",
        "code_presentation",
        "id_student",
        "id_assessment",
    }
    missing = required_metadata - set(feature_rows.columns)
    if missing:
        raise ApplicationInputError(
            f"feature rows are missing OULAD identity columns: {sorted(missing)}"
        )
    missing_features = set(adapter.predictor.feature_names) - set(feature_rows.columns)
    if missing_features:
        raise ApplicationInputError(
            f"feature rows are missing model columns: {sorted(missing_features)}"
        )

    presentation = config.presentation_split.testing_presentation
    rows = feature_rows.loc[
        (
            feature_rows["code_module"].astype(str)
            == config.presentation_split.course_identity
        )
        & (feature_rows["code_presentation"].astype(str) == presentation)
        & (feature_rows["id_student"].astype(str) == student_identity)
    ]
    contexts: list[CandidateFeatureContext] = []
    expected_features = tuple(adapter.predictor.feature_names)
    targets = dict(config.target_rule.targets)
    for rule in sorted(
        config.assignment_rules, key=lambda item: item.assignment_identity
    ):
        selected = rows.loc[rows["id_assessment"].astype(str) == rule.assessment_id]
        if len(selected) != 1:
            raise ApplicationInputError(
                f"expected exactly one source row for student {student_identity}, "
                f"assessment {rule.assessment_id} in {presentation}; "
                f"found {len(selected)}"
            )
        source = selected.iloc[0]
        features = (
            source.loc[list(expected_features)].to_frame().T.reset_index(drop=True)
        )
        try:
            map_sessions_to_clicks(
                features,
                {},
                config.click_mapping,
                expected_features,
            )
        except (MappingValidationError, TypeError, ValueError) as error:
            raise ApplicationInputError(
                f"assessment {rule.assessment_id} has invalid model features: {error}"
            ) from error
        deadline = _finite_numeric_feature(features, "assessment_deadline_day")
        weeks = _finite_numeric_feature(features, "weeks_since_course_start")
        if not deadline.is_integer() or int(deadline) != rule.deadline_day:
            raise ApplicationInputError(
                f"assessment {rule.assessment_id} deadline does not match "
                "configured day"
            )
        if not math.isclose(
            # Processed CSVs round the derived quotient; accept that serialization
            # noise while still requiring the configured deadline convention.
            weeks, rule.weeks_since_course_start, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ApplicationInputError(
                f"assessment {rule.assessment_id} weeks-since-start must equal "
                "deadline_day / 7"
            )
        if targets[rule.assignment_identity] != rule.target_mark:
            raise ApplicationInputError(
                "assignment target differs from the versioned target rule"
            )
        contexts.append(
            CandidateFeatureContext(
                student_identity,
                rule.assignment_identity,
                features.copy(deep=True),
            )
        )
    return ValidatedApplicationInputs(
        config=config,
        student_identity=student_identity,
        feature_contexts=tuple(contexts),
        model_identity=config.model_identity,
    )


def load_application_inputs(
    processed_directory: Path,
    config: ApplicationInputConfig,
    *,
    student_identity: str,
    adapter: SessionNNAdapter,
) -> ValidatedApplicationInputs:
    """Load validated later-presentation rows using the matching Model A/B schema."""
    if adapter.model_mode is NNModelMode.MODEL_A_STRICT:
        data = load_strict_feature_data(processed_directory)
    elif adapter.model_mode is NNModelMode.MODEL_B_PREVIOUS_MARKS:
        data = load_previous_marks_feature_data(processed_directory)
    else:
        raise ApplicationInputError("unsupported fitted model mode")
    if tuple(data.feature_names) != tuple(adapter.predictor.feature_names):
        raise ApplicationInputError(
            "processed feature manifest does not match the fitted model schema"
        )
    if config.presentation_split.course_identity != data.manifest.get("course_module"):
        raise ApplicationInputError("configured course does not match feature manifest")
    if set(config.presentation_split.training_presentations) != set(
        data.manifest.get("split_presentations", {}).get("train", ())
    ):
        raise ApplicationInputError(
            "configured training presentations do not match feature manifest"
        )
    if config.presentation_split.testing_presentation not in data.manifest.get(
        "split_presentations", {}
    ).get("test", ()):
        raise ApplicationInputError(
            "configured testing presentation does not match feature manifest"
        )

    split = data.splits["test"]
    model_features = pd.DataFrame(index=split.X.index)
    categories = tuple(
        name.removeprefix("assessment_type_").upper()
        for name in data.assessment_type_columns
    )
    for column, category in zip(data.assessment_type_columns, categories, strict=True):
        model_features[column] = (split.assessment_types == category).astype(float)
    for name in data.numeric_feature_names:
        model_features[name] = split.X[name]
    model_features = model_features.loc[:, list(data.feature_names)]
    feature_rows = pd.concat(
        [
            split.identifiers.reset_index(drop=True),
            model_features.reset_index(drop=True),
        ],
        axis=1,
    )
    feature_rows["target_score"] = split.y
    return build_application_inputs(
        config,
        student_identity=student_identity,
        feature_rows=feature_rows,
        adapter=adapter,
    )


def _finite_numeric_feature(features: pd.DataFrame, name: str) -> float:
    try:
        value = float(features.iloc[0][name])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ApplicationInputError(
            f"feature {name} must be finite numeric data"
        ) from error
    if not math.isfinite(value):
        raise ApplicationInputError(f"feature {name} must be finite numeric data")
    return value


def build_application_planning_problem(
    inputs: ValidatedApplicationInputs | None,
    *,
    adapter: SessionNNAdapter,
    ordering_policy: SessionOrderingPolicy,
) -> ApplicationPlanningBinding:
    """Bind validated inputs to the existing Search and CSP planning contracts.

    The caller must supply the ordering policy because P1 does not define its
    relation. Deadline-eligible slot IDs are consumed exactly as configured;
    this builder does not infer them from clock windows or week indices.
    """
    if inputs is None:
        raise ApplicationInputError(
            "production planning requires validated inputs: course and split; "
            "three assignment/assessment IDs, target marks, deadline days, "
            "eligible slot IDs, and allocation bounds; versioned target rule; "
            "versioned slot catalog with weekly capacity, slot IDs/week indices, "
            "local time windows/timezone, assignment-relative click windows, and "
            "deadline policy; support policy; click-mapping rule; student feature "
            "rows; and the exact loader-verified fitted model artifact"
        )
    if not isinstance(inputs, ValidatedApplicationInputs):
        raise ApplicationInputError("validated application inputs are required")
    if not isinstance(adapter, SessionNNAdapter):
        raise ApplicationInputError("a loader-verified model adapter is required")
    if not callable(getattr(ordering_policy, "required_pairs", None)) or not callable(
        getattr(ordering_policy, "is_ordered", None)
    ):
        raise ApplicationInputError(
            "an explicit same-assignment ordering policy is required"
        )
    try:
        verify_trusted_artifact_integrity(adapter.predictor)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ApplicationInputError(f"model integrity check failed: {error}") from error
    if not inputs.model_identity.matches(adapter):
        raise ApplicationInputError("model adapter differs from validated inputs")
    config = inputs.config
    if not isinstance(config, ApplicationInputConfig):
        raise ApplicationInputError("validated inputs contain an invalid config")
    if config.click_mapping != adapter.mapping_config:
        raise ApplicationInputError("validated click mapping differs from adapter")
    expected_assignment_ids = {
        rule.assignment_identity for rule in config.assignment_rules
    }
    contexts = inputs.feature_contexts
    if not isinstance(contexts, tuple) or len(contexts) != 3:
        raise ApplicationInputError(
            "validated inputs must retain all three assignment feature rows"
        )
    if {context.assignment_identity for context in contexts} != expected_assignment_ids:
        raise ApplicationInputError(
            "validated feature rows must cover the configured assignments"
        )
    rules = {rule.assignment_identity: rule for rule in config.assignment_rules}
    for context in contexts:
        if context.student_identity != inputs.student_identity:
            raise ApplicationInputError(
                "feature context belongs to a different student"
            )
        if tuple(context.features.columns) != tuple(adapter.predictor.feature_names):
            raise ApplicationInputError(
                "feature context columns differ from the fitted model schema"
            )
        rule = rules[context.assignment_identity]
        deadline = _finite_numeric_feature(context.features, "assessment_deadline_day")
        weeks = _finite_numeric_feature(context.features, "weeks_since_course_start")
        if not deadline.is_integer() or int(deadline) != rule.deadline_day:
            raise ApplicationInputError(
                "validated feature context deadline differs from configuration"
            )
        if not math.isclose(
            weeks, rule.weeks_since_course_start, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ApplicationInputError(
                "validated feature context week conversion differs from configuration"
            )

    from study_planner.integration.application_planning import (
        ApplicationAllocationSpace,
        ApplicationPlanningBinding,
    )

    space = ApplicationAllocationSpace(inputs)
    return ApplicationPlanningBinding(
        inputs=inputs,
        adapter=adapter,
        allocation_space=space,
        ordering_policy=ordering_policy,
    )
