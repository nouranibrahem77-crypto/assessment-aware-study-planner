"""Build fixed-cutoff FFF assessment examples and chunked VLE features."""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from study_planner.data.feature_engineering.constants import (
    ACTIVITY_GROUPS,
    CLICK_WINDOW_OFFSETS,
    SPLIT_PRESENTATIONS,
    click_feature_name,
)
from study_planner.data.feature_engineering.splitting import split_for_presentation
from study_planner.data.feature_engineering.validation import (
    AssessmentDefinition,
    DatasetCatalog,
    FeatureDataError,
    _missing,
)
from study_planner.data.oulad import DEFAULT_CHUNK_SIZE, iter_csv_chunks

FeatureValue = int | float | None
ScoreFeatureMode = Literal["strict", "proxy"]


@dataclass
class FeatureRow:
    """One scored assessment example with identifiers separate from features."""

    code_module: str
    code_presentation: str
    id_student: str
    id_assessment: str
    target_score: float
    deadline_day: int
    features: dict[str, FeatureValue]
    click_missing_features: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class FeatureBuildResult:
    """Feature rows plus auditable source and exclusion counts."""

    rows_by_split: dict[str, list[FeatureRow]]
    feature_names: tuple[str, ...]
    assessment_type_categories: tuple[str, ...]
    max_previous_score_lags: int
    score_feature_mode: ScoreFeatureMode
    dropped_records: dict[str, dict[str, int]]
    source_row_counts: dict[str, int]
    fff_row_counts: dict[str, int]
    source_missing_counts: dict[str, dict[str, int]]
    click_rows_by_presentation: dict[str, int]
    unmatched_vle_rows: dict[str, int]
    invalid_click_rows: dict[str, int]


@dataclass(frozen=True)
class _PriorScore:
    assessment_id: str
    deadline_day: int
    date_submitted: int
    score: float


def _parse_integer(value: str | None) -> int | None:
    if _missing(value):
        return None
    try:
        return int(value or "")
    except ValueError:
        return None


def _parse_score(value: str | None) -> float | None:
    if _missing(value):
        return None
    try:
        score = float(value or "")
    except ValueError:
        return None
    if not 0 <= score <= 100:
        return None
    return score


def _max_training_prior_lags(
    assessments: dict[str, AssessmentDefinition],
) -> int:
    training_presentations = set(SPLIT_PRESENTATIONS["train"])
    training_assessments = [
        assessment
        for assessment in assessments.values()
        if assessment.code_presentation in training_presentations
        and assessment.deadline_day is not None
    ]
    return max(
        (
            sum(
                other.code_presentation == current.code_presentation
                and other.deadline_day is not None
                and other.deadline_day < current.deadline_day
                for other in training_assessments
            )
            for current in training_assessments
        ),
        default=0,
    )


def _feature_schema(
    assessments: dict[str, AssessmentDefinition], max_lags: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    train_presentations = set(SPLIT_PRESENTATIONS["train"])
    categories = tuple(
        sorted(
            {
                assessment.assessment_type
                for assessment in assessments.values()
                if assessment.code_presentation in train_presentations
            }
        )
    )
    names: list[str] = [
        f"assessment_type_{category.lower()}" for category in categories
    ]
    names.extend(
        [
            "assessment_weight",
            "assessment_weight_available",
            "assessment_deadline_day",
            "weeks_since_course_start",
        ]
    )
    for lag in range(1, max_lags + 1):
        names.extend(
            [
                f"previous_assessment_score_lag_{lag}",
                f"previous_assessment_score_lag_{lag}_available",
            ]
        )
    names.extend(
        click_feature_name(group, week)
        for group in ACTIVITY_GROUPS
        for week in range(1, len(CLICK_WINDOW_OFFSETS) + 1)
    )
    return tuple(names), categories


def click_window_index(deadline_day: int, activity_day: int) -> int | None:
    """Return the 0-based weekly window or None when the event is outside cutoff."""
    offset = activity_day - (deadline_day - 28)
    if not 0 <= offset < 28:
        return None
    return offset // 7


def _submission_is_valid(
    row: dict[str, str | None], assessment: AssessmentDefinition
) -> tuple[float | None, int | None, str | None]:
    """Validate a non-banked on-time score, returning one exclusion reason."""
    if row.get("is_banked") == "1":
        return None, None, "banked_resit"
    if row.get("is_banked") != "0":
        return None, None, "invalid_banked_flag"
    score = _parse_score(row.get("score"))
    if score is None:
        return None, None, "missing_or_invalid_score"
    submitted = _parse_integer(row.get("date_submitted"))
    if submitted is None:
        return None, None, "missing_or_invalid_submission_date"
    if assessment.deadline_day is None:
        return None, None, "missing_deadline"
    if submitted > assessment.deadline_day:
        return None, None, "late_submission"
    return score, submitted, None


def _make_feature_row(
    assessment: AssessmentDefinition,
    student_id: str,
    score: float,
    categories: tuple[str, ...],
    feature_names: tuple[str, ...],
    max_lags: int,
    prior_scores: dict[tuple[str, str, str], _PriorScore],
    assessments_by_presentation: dict[str, list[AssessmentDefinition]],
) -> FeatureRow:
    deadline = assessment.deadline_day
    if deadline is None:
        raise FeatureDataError(
            f"Cannot construct features without deadline for {assessment.id_assessment}"
        )

    values: dict[str, FeatureValue] = {
        name: int(assessment.assessment_type == category)
        for category in categories
        for name in (f"assessment_type_{category.lower()}",)
    }
    values["assessment_weight"] = assessment.weight
    values["assessment_weight_available"] = int(assessment.weight is not None)
    values["assessment_deadline_day"] = deadline
    values["weeks_since_course_start"] = deadline / 7.0

    preceding = sorted(
        (
            candidate
            for candidate in assessments_by_presentation[assessment.code_presentation]
            if candidate.deadline_day is not None and candidate.deadline_day < deadline
        ),
        key=lambda candidate: (candidate.deadline_day or 0, candidate.id_assessment),
        reverse=True,
    )[:max_lags]
    for lag in range(1, max_lags + 1):
        score_name = f"previous_assessment_score_lag_{lag}"
        available_name = f"{score_name}_available"
        previous: _PriorScore | None = None
        if lag <= len(preceding):
            prior_definition = preceding[lag - 1]
            previous = prior_scores.get(
                (
                    assessment.code_presentation,
                    student_id,
                    prior_definition.id_assessment,
                )
            )
        if previous is not None and previous.date_submitted < deadline:
            values[score_name] = previous.score
            values[available_name] = 1
        else:
            values[score_name] = None
            values[available_name] = 0

    for group in ACTIVITY_GROUPS:
        for week in range(1, len(CLICK_WINDOW_OFFSETS) + 1):
            values[click_feature_name(group, week)] = 0

    if set(values) != set(feature_names):
        raise FeatureDataError("Feature schema and constructed values differ")

    return FeatureRow(
        code_module=assessment.code_module,
        code_presentation=assessment.code_presentation,
        id_student=student_id,
        id_assessment=assessment.id_assessment,
        target_score=score,
        deadline_day=deadline,
        features={name: values[name] for name in feature_names},
    )


def build_features(
    files: dict[str, Path],
    catalog: DatasetCatalog,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    score_mode: ScoreFeatureMode = "strict",
) -> FeatureBuildResult:
    """Build fixed splits using strict or assumed-available prior scores."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if score_mode not in {"strict", "proxy"}:
        raise ValueError(f"Unknown score feature mode: {score_mode}")

    max_lags = (
        _max_training_prior_lags(catalog.assessments) if score_mode == "proxy" else 0
    )
    feature_names, categories = _feature_schema(catalog.assessments, max_lags)
    assessments_by_presentation: dict[str, list[AssessmentDefinition]] = defaultdict(
        list
    )
    for assessment in catalog.assessments.values():
        assessments_by_presentation[assessment.code_presentation].append(assessment)
    for assessment_list in assessments_by_presentation.values():
        assessment_list.sort(
            key=lambda assessment: (
                assessment.deadline_day
                if assessment.deadline_day is not None
                else float("inf"),
                assessment.id_assessment,
            )
        )

    dropped: dict[str, Counter[str]] = {
        split: Counter() for split in SPLIT_PRESENTATIONS
    }
    source_row_counts = Counter(catalog.source_row_counts)
    fff_row_counts = Counter(catalog.fff_row_counts)
    source_missing_counts: dict[str, Counter[str]] = {
        presentation: Counter()
        for presentations in SPLIT_PRESENTATIONS.values()
        for presentation in presentations
    }
    prior_scores: dict[tuple[str, str, str], _PriorScore] = {}
    targets: list[tuple[AssessmentDefinition, str, float, str]] = []
    seen_submission_keys: set[tuple[str, str, str]] = set()

    for chunk in iter_csv_chunks(files["studentAssessment.csv"], chunk_size):
        for row in chunk:
            source_row_counts["studentAssessment.csv"] += 1
            assessment_id = row.get("id_assessment") or ""
            assessment = catalog.assessments.get(assessment_id)
            if assessment is None:
                continue
            fff_row_counts["studentAssessment.csv"] += 1
            presentation = assessment.code_presentation
            split = split_for_presentation(presentation)
            if split is None:
                raise FeatureDataError(
                    f"Assessment {assessment_id} is not in a configured split"
                )
            source_missing_counts[presentation].update(
                column for column, value in row.items() if _missing(value)
            )
            student_id = row.get("id_student") or ""
            key = (presentation, student_id, assessment_id)
            if not student_id:
                dropped[split]["missing_student_identifier"] += 1
                continue
            if key in seen_submission_keys:
                raise FeatureDataError(
                    "Duplicate student-assessment join key: "
                    f"{presentation}, {student_id}, {assessment_id}"
                )
            seen_submission_keys.add(key)

            score, submitted, reason = _submission_is_valid(row, assessment)
            if reason is not None:
                dropped[split][reason] += 1
                continue
            assert score is not None and submitted is not None
            if assessment.deadline_day is None:
                raise FeatureDataError(
                    f"Eligible score has no deadline: {assessment_id}"
                )
            if score_mode == "proxy":
                prior_scores[key] = _PriorScore(
                    assessment_id=assessment_id,
                    deadline_day=assessment.deadline_day,
                    date_submitted=submitted,
                    score=score,
                )
            targets.append((assessment, student_id, score, split))

    rows_by_split: dict[str, list[FeatureRow]] = {
        split: [] for split in SPLIT_PRESENTATIONS
    }
    rows_by_student: dict[tuple[str, str], list[FeatureRow]] = defaultdict(list)
    prior_score_map = {
        (presentation, student, assessment_id): record
        for (presentation, student, assessment_id), record in prior_scores.items()
    }
    for assessment, student_id, score, split in targets:
        row = _make_feature_row(
            assessment=assessment,
            student_id=student_id,
            score=score,
            categories=categories,
            feature_names=feature_names,
            max_lags=max_lags,
            prior_scores=prior_score_map,
            assessments_by_presentation=assessments_by_presentation,
        )
        rows_by_split[split].append(row)
        rows_by_student[(assessment.code_presentation, student_id)].append(row)
    for rows in rows_by_student.values():
        rows.sort(key=lambda row: (row.deadline_day, row.id_assessment))
    deadlines_by_student = {
        student_key: [row.deadline_day for row in rows]
        for student_key, rows in rows_by_student.items()
    }

    click_rows_by_presentation: Counter[str] = Counter()
    unmatched_vle_rows: Counter[str] = Counter()
    invalid_click_rows: Counter[str] = Counter()
    for chunk in iter_csv_chunks(files["studentVle.csv"], chunk_size):
        for event in chunk:
            source_row_counts["studentVle.csv"] += 1
            if event.get("code_module") != "FFF":
                continue
            presentation = event.get("code_presentation") or ""
            if presentation not in source_missing_counts:
                continue
            fff_row_counts["studentVle.csv"] += 1
            click_rows_by_presentation[presentation] += 1
            source_missing_counts[presentation].update(
                column for column, value in event.items() if _missing(value)
            )
            student_id = event.get("id_student") or ""
            student_rows = rows_by_student.get((presentation, student_id), [])
            if not student_id or not student_rows:
                if not student_id:
                    invalid_click_rows[presentation] += 1
                continue

            activity_day = _parse_integer(event.get("date"))
            if activity_day is None:
                invalid_click_rows[presentation] += 1
                for target_row in student_rows:
                    target_row.click_missing_features.update(
                        click_feature_name(group, week)
                        for group in ACTIVITY_GROUPS
                        for week in range(1, 5)
                    )
                continue

            site_id = event.get("id_site") or ""
            group = catalog.site_groups.get(("FFF", presentation, site_id))
            clicks = _parse_integer(event.get("sum_click"))
            if clicks is None or clicks < 0:
                invalid_click_rows[presentation] += 1
            if group is None:
                unmatched_vle_rows[presentation] += 1
            deadlines = deadlines_by_student[(presentation, student_id)]
            first = bisect_right(deadlines, activity_day)
            last = bisect_right(deadlines, activity_day + 28)
            for target_row in student_rows[first:last]:
                week_index = click_window_index(target_row.deadline_day, activity_day)
                if week_index is None:
                    continue
                week = week_index + 1
                if group is None:
                    target_row.click_missing_features.update(
                        click_feature_name(name, week) for name in ACTIVITY_GROUPS
                    )
                elif clicks is None or clicks < 0:
                    target_row.click_missing_features.add(
                        click_feature_name(group, week)
                    )
                else:
                    feature_name = click_feature_name(group, week)
                    current = target_row.features[feature_name]
                    assert isinstance(current, int)
                    target_row.features[feature_name] = current + clicks

    for rows in rows_by_split.values():
        for row in rows:
            for name in row.click_missing_features:
                row.features[name] = None
    for rows in rows_by_split.values():
        rows.sort(
            key=lambda row: (
                row.code_presentation,
                row.deadline_day,
                row.id_assessment,
                row.id_student,
            )
        )

    return FeatureBuildResult(
        rows_by_split=rows_by_split,
        feature_names=feature_names,
        assessment_type_categories=categories,
        max_previous_score_lags=max_lags,
        score_feature_mode=score_mode,
        dropped_records={
            split: dict(sorted(counts.items())) for split, counts in dropped.items()
        },
        source_row_counts=dict(sorted(source_row_counts.items())),
        fff_row_counts=dict(sorted(fff_row_counts.items())),
        source_missing_counts={
            presentation: dict(sorted(counts.items()))
            for presentation, counts in sorted(source_missing_counts.items())
        },
        click_rows_by_presentation=dict(sorted(click_rows_by_presentation.items())),
        unmatched_vle_rows=dict(sorted(unmatched_vle_rows.items())),
        invalid_click_rows=dict(sorted(invalid_click_rows.items())),
    )
