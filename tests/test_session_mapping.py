"""Tests for synthetic session-to-click mapping and monotonicity reports."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import pytest

from study_planner.data.feature_engineering.constants import (
    ACTIVITY_GROUPS,
    click_feature_name,
)
from study_planner.nn.dataset import prepare_inference_frame
from study_planner.search.session_mapping import (
    TRAINING_CLICK_SUPPORT,
    ClickScenario,
    IncrementalMonotonicityReporter,
    JointSupportStatus,
    MappingValidationError,
    MarginalSupportStatus,
    MonotonicityContext,
    MonotonicityObservation,
    SessionClickMappingConfig,
    evaluate_monotonicity,
    map_sessions_to_clicks,
)

STRICT_FEATURE_NAMES = (
    "assessment_type_cma",
    "assessment_type_exam",
    "assessment_type_tma",
    "assessment_weight",
    "assessment_weight_available",
    "assessment_deadline_day",
    "weeks_since_course_start",
    *(
        click_feature_name(group, week)
        for group in ACTIVITY_GROUPS
        for week in range(1, 5)
    ),
)


def strict_row(**feature_values: object) -> pd.DataFrame:
    """Build one ordered strict-schema test row with optional click values."""
    values: dict[str, float] = {name: 0.0 for name in STRICT_FEATURE_NAMES}
    values.update(
        {
            "assessment_type_cma": 1.0,
            "assessment_weight": 10.0,
            "assessment_weight_available": 1.0,
            "assessment_deadline_day": 47.0,
            "weeks_since_course_start": 47.0 / 7,
        }
    )
    values.update(feature_values)
    return pd.DataFrame(
        [[values[name] for name in STRICT_FEATURE_NAMES]],
        columns=STRICT_FEATURE_NAMES,
    )


def apply_mapping(
    features: pd.DataFrame,
    weekly_sessions: Mapping[int, int],
    scenario: ClickScenario = ClickScenario.CENTRAL,
) -> pd.DataFrame:
    """Apply the default content-navigation scenario to a copied row."""
    result = map_sessions_to_clicks(
        features,
        weekly_sessions,
        SessionClickMappingConfig(scenario=scenario),
        STRICT_FEATURE_NAMES,
    )
    return result.candidate_features


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        (ClickScenario.LOW, 22),
        (ClickScenario.CENTRAL, 62),
        (ClickScenario.HIGH, 121),
    ],
)
def test_scenarios_apply_documented_synthetic_increment(
    scenario: ClickScenario, expected: int
) -> None:
    original = strict_row()

    result = map_sessions_to_clicks(
        original,
        {2: 1},
        SessionClickMappingConfig(scenario=scenario),
        STRICT_FEATURE_NAMES,
    )

    feature = "click_content_navigation_week_2"
    assert result.candidate_features.loc[0, feature] == expected
    assert result.clicks_per_session == expected
    assert result.scenario is scenario
    assert result.changed_features == (feature,)


def test_zero_sessions_returns_identical_copy_and_no_changes() -> None:
    original = strict_row(click_content_navigation_week_1=27)

    result = map_sessions_to_clicks(
        original,
        {},
        SessionClickMappingConfig(ClickScenario.CENTRAL),
        STRICT_FEATURE_NAMES,
    )

    pd.testing.assert_frame_equal(result.candidate_features, original)
    assert result.candidate_features is not original
    assert result.weekly_sessions == (0, 0, 0, 0)
    assert result.changed_features == ()


def test_multiple_sessions_are_added_to_explicit_relative_weeks() -> None:
    original = strict_row(
        click_content_navigation_week_1=10,
        click_content_navigation_week_4=20,
    )

    result = map_sessions_to_clicks(
        original,
        {1: 2, 4: 3},
        SessionClickMappingConfig(ClickScenario.LOW),
        STRICT_FEATURE_NAMES,
    )

    assert result.candidate_features.loc[0, "click_content_navigation_week_1"] == 54
    assert result.candidate_features.loc[0, "click_content_navigation_week_4"] == 86
    assert result.weekly_sessions == (2, 0, 0, 3)


def test_configured_activity_group_changes_only_that_category() -> None:
    original = strict_row()

    result = map_sessions_to_clicks(
        original,
        {3: 1},
        SessionClickMappingConfig(
            ClickScenario.CENTRAL,
            activity_group="collaboration",
        ),
        STRICT_FEATURE_NAMES,
    )

    assert result.candidate_features.loc[0, "click_collaboration_week_3"] == 62
    assert result.candidate_features.loc[0, "click_content_navigation_week_3"] == 0
    assert result.changed_features == ("click_collaboration_week_3",)


def test_mapping_is_deterministic_and_does_not_mutate_input() -> None:
    original = strict_row(click_content_navigation_week_2=5)
    before = original.copy(deep=True)
    config = SessionClickMappingConfig(ClickScenario.HIGH)

    first = map_sessions_to_clicks(original, {2: 2}, config, STRICT_FEATURE_NAMES)
    second = map_sessions_to_clicks(original, {2: 2}, config, STRICT_FEATURE_NAMES)

    pd.testing.assert_frame_equal(original, before)
    pd.testing.assert_frame_equal(first.candidate_features, second.candidate_features)
    assert first.support_diagnostics == second.support_diagnostics


def test_feature_schema_and_order_are_preserved() -> None:
    original = strict_row()

    result = map_sessions_to_clicks(
        original,
        {3: 1},
        SessionClickMappingConfig(ClickScenario.CENTRAL),
        STRICT_FEATURE_NAMES,
    )

    assert tuple(result.candidate_features.columns) == STRICT_FEATURE_NAMES
    assert "target_score" not in result.candidate_features
    assert "id_student" not in result.candidate_features


@pytest.mark.parametrize(
    "weekly_sessions",
    [
        {0: 1},
        {5: 1},
        {1: -1},
        {1: 1.5},
        {1: True},
    ],
)
def test_rejects_invalid_weekly_allocations(
    weekly_sessions: Mapping[int, int],
) -> None:
    with pytest.raises(MappingValidationError):
        apply_mapping(strict_row(), weekly_sessions)


def test_rejects_invalid_scenario_group_schema_and_feature_values() -> None:
    with pytest.raises(MappingValidationError, match="scenario"):
        SessionClickMappingConfig("low")  # type: ignore[arg-type]
    with pytest.raises(MappingValidationError, match="activity group"):
        SessionClickMappingConfig(ClickScenario.LOW, "unknown")

    original = strict_row()
    with pytest.raises(MappingValidationError, match="columns/order"):
        map_sessions_to_clicks(
            original.iloc[:, ::-1],
            {},
            SessionClickMappingConfig(ClickScenario.LOW),
            STRICT_FEATURE_NAMES,
        )
    for invalid_click in (float("inf"), "not-a-number"):
        with pytest.raises(MappingValidationError, match="numeric"):
            apply_mapping(
                strict_row(click_content_navigation_week_1=invalid_click),
                {},  # type: ignore[arg-type]
            )
    with pytest.raises(MappingValidationError, match="negative"):
        apply_mapping(strict_row(click_collaboration_week_1=-1), {})


def test_rejects_invalid_numeric_non_click_feature() -> None:
    with pytest.raises(MappingValidationError, match="numeric"):
        apply_mapping(strict_row(assessment_weight="invalid"), {})


@pytest.mark.parametrize(
    "type_values",
    [
        {"assessment_type_cma": 0, "assessment_type_exam": 0},
        {"assessment_type_cma": 1, "assessment_type_exam": 1},
        {"assessment_type_cma": "bad"},
    ],
)
def test_rejects_missing_or_invalid_one_hot_feature_values(
    type_values: dict[str, object],
) -> None:
    with pytest.raises(
        MappingValidationError, match="assessment type|active indicator"
    ):
        apply_mapping(strict_row(**type_values), {})


@pytest.mark.parametrize(
    "type_values",
    [
        {
            "assessment_type_cma": float("nan"),
            "assessment_type_exam": float("nan"),
            "assessment_type_tma": float("nan"),
        },
        {"assessment_type_cma": 1, "assessment_type_exam": float("nan")},
    ],
)
def test_preserves_missing_assessment_type_as_inference_missing_category(
    type_values: dict[str, object],
) -> None:
    original = strict_row(**type_values)
    before = original.copy(deep=True)

    result = map_sessions_to_clicks(
        original,
        {},
        SessionClickMappingConfig(ClickScenario.CENTRAL),
        STRICT_FEATURE_NAMES,
    )
    inference_frame = prepare_inference_frame(
        result.candidate_features,
        STRICT_FEATURE_NAMES,
        STRICT_FEATURE_NAMES[:3],
    )

    assert pd.isna(inference_frame.loc[0, "assessment_type"])
    pd.testing.assert_frame_equal(result.candidate_features, original)
    pd.testing.assert_frame_equal(original, before)


def test_preserves_supported_missing_numeric_values_and_reports_uncertainty() -> None:
    original = strict_row(
        assessment_weight=float("nan"),
        click_collaboration_week_1=float("nan"),
    )
    before = original.copy(deep=True)

    result = map_sessions_to_clicks(
        original,
        {2: 1},
        SessionClickMappingConfig(ClickScenario.LOW),
        STRICT_FEATURE_NAMES,
    )

    assert pd.isna(result.candidate_features.loc[0, "assessment_weight"])
    assert pd.isna(result.candidate_features.loc[0, "click_collaboration_week_1"])
    assert "assessment_weight" in result.missing_numeric_features
    assert "click_collaboration_week_1" in result.missing_numeric_features
    diagnostic = next(
        item
        for item in result.support_diagnostics
        if item.feature_name == "click_collaboration_week_1"
    )
    assert diagnostic.candidate_value is None
    assert diagnostic.marginal_status is MarginalSupportStatus.UNASSESSED
    assert result.joint_support_status is JointSupportStatus.UNASSESSED
    pd.testing.assert_frame_equal(original, before)


def test_rejects_adding_sessions_to_missing_target_click_total() -> None:
    original = strict_row(click_content_navigation_week_2=float("nan"))

    with pytest.raises(MappingValidationError, match="missing click feature"):
        apply_mapping(original, {2: 1})


def test_reports_out_of_support_without_clipping_candidate() -> None:
    boundary = TRAINING_CLICK_SUPPORT["click_content_navigation_week_1"]
    original = strict_row(click_content_navigation_week_1=boundary.maximum)

    result = map_sessions_to_clicks(
        original,
        {1: 1},
        SessionClickMappingConfig(ClickScenario.LOW),
        STRICT_FEATURE_NAMES,
    )

    diagnostic = next(
        item
        for item in result.support_diagnostics
        if item.feature_name == "click_content_navigation_week_1"
    )
    assert diagnostic.candidate_value == boundary.maximum + 22
    assert not diagnostic.within_observed_range
    assert diagnostic.marginal_status is MarginalSupportStatus.OUT_OF_RANGE
    assert result.has_out_of_support_values
    assert diagnostic.feature_name in result.out_of_support_features


def test_p99_is_warning_boundary_not_hard_support_cutoff() -> None:
    name = "click_content_navigation_week_1"
    boundary = TRAINING_CLICK_SUPPORT[name]
    original = strict_row(**{name: boundary.p99})

    result = map_sessions_to_clicks(
        original,
        {1: 1},
        SessionClickMappingConfig(ClickScenario.LOW),
        STRICT_FEATURE_NAMES,
    )

    diagnostic = next(
        item for item in result.support_diagnostics if item.feature_name == name
    )
    assert diagnostic.exceeds_p99
    assert diagnostic.within_observed_range


def test_joint_support_is_explicitly_unassessed_when_marginals_are_in_range() -> None:
    original = strict_row(
        click_content_navigation_week_1=100,
        click_assessment_activity_week_2=100,
        click_collaboration_week_4=100,
    )

    result = map_sessions_to_clicks(
        original,
        {},
        SessionClickMappingConfig(ClickScenario.CENTRAL),
        STRICT_FEATURE_NAMES,
    )

    assert result.marginal_status is MarginalSupportStatus.IN_RANGE
    assert result.joint_support_status is JointSupportStatus.UNASSESSED
    assert not result.has_out_of_support_values


def test_monotonicity_report_records_each_one_session_decrease() -> None:
    context = MonotonicityContext(
        student_id="student-1",
        assignment_id="assignment-1",
        base_feature_context_id="base-1",
        mapping_scenario_id="central:content_navigation:62",
    )
    report = evaluate_monotonicity(
        [
            MonotonicityObservation((0, 0, 0, 0), 70, context, ()),
            MonotonicityObservation((1, 0, 0, 0), 68, context, ("slot-1",)),
            MonotonicityObservation((0, 1, 0, 0), 72, context, ("slot-2",)),
            MonotonicityObservation((2, 0, 0, 0), 67, context, ("slot-1", "slot-3")),
        ]
    )

    assert report.comparisons == 3
    assert len(report.violations) == 2
    assert report.violation_rate == pytest.approx(2 / 3)
    assert report.violations[0].changed_window == 1
    assert report.violations[0].prediction_delta == -2
    assert report.violations[0].parent_schedule_identity == ()
    assert report.violations[0].child_schedule_identity == ("slot-1",)


def test_monotonicity_compares_only_within_matching_context() -> None:
    def context(
        student: str = "student-1",
        assignment: str = "assignment-1",
        base: str = "base-1",
        scenario: str = "central:content_navigation:62",
    ) -> MonotonicityContext:
        return MonotonicityContext(student, assignment, base, scenario)

    report = evaluate_monotonicity(
        [
            MonotonicityObservation((0, 0, 0, 0), 70, context(), ("slot-1",)),
            MonotonicityObservation(
                (1, 0, 0, 0), 65, context(base="base-2"), ("slot-2",)
            ),
            MonotonicityObservation(
                (0, 1, 0, 0), 65, context(student="student-2"), ("slot-3",)
            ),
            MonotonicityObservation(
                (0, 0, 1, 0), 65, context(assignment="assignment-2"), ("slot-4",)
            ),
            MonotonicityObservation(
                (0, 0, 0, 1),
                65,
                context(scenario="high:content_navigation:121"),
                ("slot-5",),
            ),
            MonotonicityObservation(
                (0, 0, 0, 1), 65, context(base="base-2"), ("slot-6",)
            ),
        ]
    )

    assert report.comparisons == 0
    assert report.violations == ()


def test_incremental_monotonicity_matches_batch_for_effort_and_context_cases() -> None:
    context = MonotonicityContext(
        "student-1", "assignment-1", "base-1", "central:content_navigation:62"
    )
    other_assignment = MonotonicityContext(
        "student-1", "assignment-2", "base-2", "central:content_navigation:62"
    )
    other_student = MonotonicityContext(
        "student-2", "assignment-1", "base-3", "central:content_navigation:62"
    )
    observations = (
        MonotonicityObservation((0, 0, 0, 0), 70, context, ("zero",)),
        MonotonicityObservation((1, 0, 0, 0), 68, context, ("one-a",)),
        MonotonicityObservation((0, 1, 0, 0), 72, context, ("one-b",)),
        MonotonicityObservation((2, 0, 0, 0), 67, context, ("two-a",)),
        MonotonicityObservation((1, 1, 0, 0), 60, context, ("two-mixed",)),
        # Equal effort and decreasing effort do not create one-session comparisons.
        MonotonicityObservation((1, 0, 0, 0), 50, context, ("one-equal-effort",)),
        MonotonicityObservation(
            (0, 0, 0, 0), 99, other_assignment, ("other-assignment",)
        ),
        MonotonicityObservation((0, 0, 1, 0), 99, other_student, ("other-student",)),
    )
    insertion_orders = (
        tuple(range(len(observations))),
        tuple(reversed(range(len(observations)))),
        (2, 5, 0, 7, 4, 1, 6, 3),
    )

    for order in insertion_orders:
        ordered = tuple(observations[index] for index in order)
        reporter = IncrementalMonotonicityReporter(tolerance=0.5)
        reports = [reporter.add(item) for item in ordered]

        assert reports[-1] == evaluate_monotonicity(ordered, tolerance=0.5)
        assert reporter.report() == reports[-1]


def test_incremental_monotonicity_empty_single_and_duplicate_inputs() -> None:
    context = MonotonicityContext(
        "student-1", "assignment-1", "base-1", "central:content_navigation:62"
    )
    observation = MonotonicityObservation((0, 0, 0, 0), 70, context, ("schedule",))
    reporter = IncrementalMonotonicityReporter()

    assert reporter.report() == evaluate_monotonicity(())
    assert reporter.add(observation) == evaluate_monotonicity((observation,))
    with pytest.raises(ValueError, match="unique"):
        reporter.add(observation)
    with pytest.raises(ValueError, match="unique"):
        evaluate_monotonicity((observation, observation))


def test_incremental_upsert_matches_insertion_ordered_mapping_semantics() -> None:
    context = MonotonicityContext(
        "student-1", "assignment-1", "base-1", "central:content_navigation:62"
    )
    first = MonotonicityObservation((0, 0, 0, 0), 70, context, ("zero",))
    replacement = MonotonicityObservation((0, 0, 0, 0), 65, context, ("zero",))
    child = MonotonicityObservation((1, 0, 0, 0), 60, context, ("one",))
    reporter = IncrementalMonotonicityReporter()

    reporter.upsert(first)
    reporter.upsert(child)
    actual = reporter.upsert(replacement)
    expected = evaluate_monotonicity((replacement, child))

    assert actual == expected
    assert actual.violations[0].prediction_delta == -5


def test_monotonicity_rejects_invalid_tolerance_or_prediction() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        evaluate_monotonicity([], tolerance=-1)
    context = MonotonicityContext(
        student_id="student-1",
        assignment_id="assignment-1",
        base_feature_context_id="base-1",
        mapping_scenario_id="central:content_navigation:62",
    )
    with pytest.raises(ValueError, match="predicted_mark"):
        evaluate_monotonicity(
            [MonotonicityObservation((0, 0, 0, 0), float("nan"), context, ())]
        )
