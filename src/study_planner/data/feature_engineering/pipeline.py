"""Orchestrate FFF validation, features, persistence, and factual reporting."""

from __future__ import annotations

import argparse
import logging
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from study_planner.data.feature_engineering.constants import (
    SPLIT_PRESENTATIONS,
)
from study_planner.data.feature_engineering.features import (
    FeatureBuildResult,
    FeatureRow,
    ScoreFeatureMode,
    build_features,
)
from study_planner.data.feature_engineering.persistence import persist_feature_splits
from study_planner.data.feature_engineering.validation import (
    DatasetCatalog,
    inspect_fff_inputs,
)
from study_planner.data.oulad import DEFAULT_CHUNK_SIZE, locate_dataset_files

LOGGER = logging.getLogger(__name__)


def _score_bands(rows: list[FeatureRow]) -> dict[str, int]:
    bands = {"0-39": 0, "40-59": 0, "60-79": 0, "80-100": 0}
    for row in rows:
        if row.target_score < 40:
            bands["0-39"] += 1
        elif row.target_score < 60:
            bands["40-59"] += 1
        elif row.target_score < 80:
            bands["60-79"] += 1
        else:
            bands["80-100"] += 1
    return bands


def _feature_missing_counts(
    rows: list[FeatureRow], names: tuple[str, ...]
) -> Counter[str]:
    missing: Counter[str] = Counter()
    for row in rows:
        for name in names:
            if row.features[name] is None:
                missing[name] += 1
    return missing


def _format_counts(counts: dict[str, int] | Counter[str]) -> str:
    return (
        ", ".join(f"{name}: {count}" for name, count in sorted(counts.items()))
        or "none"
    )


def render_feature_report(
    result: FeatureBuildResult,
    catalog: DatasetCatalog,
    manifest: dict[str, Any],
) -> str:
    """Create an evidence-backed Markdown report from generated feature rows."""
    lines = [
        "# FFF feature engineering report",
        "",
        "This report describes generated OULAD FFF features. It is descriptive",
        "only: click activity is not interpreted as causing assessment scores.",
        "No demographic variables or inferred study sessions are used.",
        "",
        "## Prediction cutoff and leakage controls",
        "",
        "For each target assessment with deadline day D, the prediction cutoff is",
        "the start of day D. Clicks use only day offsets less than D and within",
        "[D-28, D); the four half-open weekly windows are reported in the manifest.",
        "Target submissions after their own deadline and banked results are dropped.",
        "The target score and submission date are stored separately from model",
        "features.",
        "",
        (
            "Strict mode excludes all previous-assessment-score features. "
            "No prior-score availability assumption is used."
            if result.score_feature_mode == "strict"
            else "Proxy mode includes only valid, non-banked scores submitted on "
            "time for strictly earlier assessments. OULAD has no score-release "
            "timestamps, so publication before the cutoff is unverified."
        ),
        "No scaling or imputation parameters are fit.",
        "Assessment type categories and the number of prior-score lag slots are",
        "derived only from the 2013B/2013J training assessment metadata.",
        "",
        "## Input row counts",
        "",
        "Counts cover complete source files; FFF counts cover the selected course.",
        "",
        "| Source file | All rows | FFF rows |",
        "| --- | ---: | ---: |",
    ]
    for filename in sorted(result.source_row_counts):
        lines.append(
            f"| `{filename}` | {result.source_row_counts[filename]} | "
            f"{result.fff_row_counts.get(filename, 0)} |"
        )
    lines.extend(
        [
            "",
            "## Activity type mapping",
            "",
            "Types below are resource counts from FFF `vle.csv`; every observed type",
            "was assigned to exactly one documented group.",
            "",
            "| `activity_type` | Group | VLE resource rows |",
            "| --- | --- | ---: |",
        ]
    )
    from study_planner.data.feature_engineering.constants import ACTIVITY_TYPE_TO_GROUP

    for activity_type, count in catalog.activity_type_counts.items():
        lines.append(
            f"| `{activity_type}` | `{ACTIVITY_TYPE_TO_GROUP[activity_type]}` | "
            f"{count} |"
        )

    lines.extend(
        [
            "",
            "## Fixed chronological splits",
            "",
            "Rows were never randomly redistributed.",
            "",
            "| Split | Presentations | Scored assessment rows | CMA | Exam | TMA | "
            "Score bands 0-39 / 40-59 / 60-79 / 80-100 |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    all_rows: list[FeatureRow] = []
    rows_by_type: dict[str, Counter[str]] = {}
    for split_name, presentations in SPLIT_PRESENTATIONS.items():
        rows = result.rows_by_split[split_name]
        all_rows.extend(rows)
        by_type: Counter[str] = Counter()
        assessment_by_id = catalog.assessments
        for row in rows:
            by_type[assessment_by_id[row.id_assessment].assessment_type] += 1
        rows_by_type[split_name] = by_type
        bands = _score_bands(rows)
        lines.append(
            f"| {split_name} | {', '.join(presentations)} | {len(rows)} | "
            f"{by_type['CMA']} | {by_type['Exam']} | {by_type['TMA']} | "
            f"{bands['0-39']} / {bands['40-59']} / {bands['60-79']} / "
            f"{bands['80-100']} |"
        )

    lines.extend(
        [
            "",
            "Assessment type distribution by split:",
            "",
            "| Split | Type counts |",
            "| --- | --- |",
        ]
    )
    for split_name, counts in rows_by_type.items():
        lines.append(f"| {split_name} | {_format_counts(counts)} |")

    lines.extend(
        [
            "",
            "## Dropped submission records",
            "",
            "Only records present in `studentAssessment.csv` can be counted here;",
            "students with no recorded submission have no score row to retain.",
            "",
            "| Split | Dropped rows by reason |",
            "| --- | --- |",
        ]
    )
    for split_name in SPLIT_PRESENTATIONS:
        lines.append(
            f"| {split_name} | {_format_counts(result.dropped_records[split_name])} |"
        )

    lines.extend(
        [
            "",
            "## Missing values and click integrity",
            "",
            "Input missing-cell counts are restricted to FFF rows in the configured",
            "presentations. Feature missing counts refer to retained scored rows.",
            "Click zeros mean no matching event in a complete window; unknown or",
            "incomplete windows remain blank rather than becoming zero.",
            "",
            "| Split | Missing feature cells by feature |",
            "| --- | --- |",
        ]
    )
    for split_name, rows in result.rows_by_split.items():
        missing = _feature_missing_counts(rows, result.feature_names)
        lines.append(f"| {split_name} | {_format_counts(missing)} |")

    lines.extend(
        [
            "",
            "FFF source missing cells by presentation (table.column: count):",
            "",
        ]
    )
    missing_items: list[str] = []
    for presentation, counts in result.source_missing_counts.items():
        for column, count in counts.items():
            if count:
                missing_items.append(f"{presentation}.{column}: {count}")
    for filename, counts in catalog.fff_missing_counts.items():
        for column, count in counts.items():
            if count:
                missing_items.append(f"FFF {filename}.{column}: {count}")
    lines.extend([f"- {item}" for item in missing_items] or ["- none"])

    lines.extend(
        [
            "",
            "| Presentation | FFF studentVle rows | Unmatched VLE joins | "
            "Invalid click rows |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for presentation in sorted(result.click_rows_by_presentation):
        lines.append(
            f"| {presentation} | {result.click_rows_by_presentation[presentation]} | "
            f"{result.unmatched_vle_rows.get(presentation, 0)} | "
            f"{result.invalid_click_rows.get(presentation, 0)} |"
        )

    if all_rows:
        scores = sorted(row.target_score for row in all_rows)
        lines.extend(
            [
                "",
                "Overall target score summary: "
                f"n={len(scores)}, min={scores[0]:g}, "
                f"median={statistics.median(scores):g}, "
                f"mean={statistics.fmean(scores):.3f}, max={scores[-1]:g}.",
            ]
        )

    train_rows = result.rows_by_split["train"]
    train_assessment_ids = {
        row.id_assessment for row in train_rows if row.code_presentation == "2013B"
    }
    train_counts = Counter(row.id_assessment for row in train_rows)
    demo_candidates = sorted(
        (
            assessment
            for assessment in catalog.assessments.values()
            if assessment.code_presentation == "2013B"
            and assessment.assessment_type == "TMA"
            and assessment.deadline_day is not None
            and assessment.id_assessment in train_assessment_ids
        ),
        key=lambda assessment: (assessment.deadline_day or 0, assessment.id_assessment),
    )[:3]
    lines.extend(
        [
            "",
            "## Three real assignments for a minimum demo",
            "",
            "These are scored, on-time TMA examples from the training presentation",
            "2013B. Their ascending deadlines demonstrate early, middle, and later",
            "prediction cutoffs; the label count is the retained target-row count.",
            "",
            "| Assessment ID | Type | Deadline day | Retained target rows | "
            "Justification |",
            "| ---: | --- | ---: | ---: | --- |",
        ]
    )
    for assessment in demo_candidates:
        count = train_counts[assessment.id_assessment]
        lines.append(
            f"| {assessment.id_assessment} | {assessment.assessment_type} | "
            f"{assessment.deadline_day} | {count} | TMA with an observed deadline "
            "and retained on-time scores; prediction inputs are cut off before "
            "that deadline. |"
        )
    if len(demo_candidates) < 3:
        lines.append(
            "\nFewer than three suitable 2013B TMAs were present; no additional "
            "assignment was inferred."
        )

    lines.extend(
        [
            "",
            "## Output and reproducibility",
            "",
            "Processed CSVs and `fff_preprocessing.json` are under `data/processed/`.",
            "The JSON records source SHA-256 digests, stable feature names, the fixed",
            "split, train-fitted type categories, and the training-derived lag count.",
            "Identifier columns are retained for audit/join purposes and are not",
            "listed as model input features. Input missing values are counted below",
            "without demographic fields:",
            "",
        ]
    )
    for filename, counts in catalog.fff_missing_counts.items():
        lines.append(f"- `{filename}`: {_format_counts(counts)}")
    lines.append(
        "- Score-release dates are absent; the prior-score availability proxy above "
        "must be reviewed before a real deployment."
    )
    lines.append("")
    return "\n".join(lines)


def run_feature_pipeline(
    source_dir: Path,
    output_dir: Path,
    report_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    score_mode: ScoreFeatureMode = "strict",
) -> FeatureBuildResult:
    """Validate raw OULAD, build fixed FFF splits, persist outputs and report."""
    files = locate_dataset_files(source_dir)
    catalog = inspect_fff_inputs(files)
    result = build_features(files, catalog, chunk_size, score_mode)
    if not result.rows_by_split["train"]:
        raise ValueError("No valid FFF training targets were built")
    manifest = persist_feature_splits(result, files, output_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_feature_report(result, catalog, manifest), encoding="utf-8"
    )
    LOGGER.info(
        "Wrote FFF feature rows: train=%d validation=%d test=%d",
        *(len(result.rows_by_split[name]) for name in SPLIT_PRESENTATIONS),
    )
    LOGGER.info("Feature files written under %s", output_dir)
    LOGGER.info("Feature report written to %s", report_path)
    return result


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def main(argv: list[str] | None = None) -> int:
    """CLI for reproducible FFF feature generation; it does not train models."""
    parser = argparse.ArgumentParser(
        description="Build leakage-aware OULAD FFF assessment features."
    )
    root = _project_root()
    parser.add_argument("--source-dir", type=Path, default=root / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "processed")
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "feature_engineering.md",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--score-mode",
        choices=("strict", "proxy"),
        default="strict",
        help=(
            "exclude prior scores (strict) or include the documented "
            "availability proxy"
        ),
    )
    args = parser.parse_args(argv)
    if args.chunk_size < 1:
        parser.error("--chunk-size must be at least 1")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run_feature_pipeline(
            args.source_dir,
            args.output_dir,
            args.report,
            args.chunk_size,
            args.score_mode,
        )
    except (OSError, ValueError) as error:
        LOGGER.error("Feature generation failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
