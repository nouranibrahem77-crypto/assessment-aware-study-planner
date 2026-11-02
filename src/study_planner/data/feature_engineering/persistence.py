"""Deterministic CSV and manifest persistence for generated feature splits."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from study_planner.data.feature_engineering.constants import (
    ACTIVITY_GROUPS,
    CLICK_WINDOW_OFFSETS,
    IDENTIFIER_COLUMNS,
    SPLIT_PRESENTATIONS,
    TARGET_COLUMN,
)
from study_planner.data.feature_engineering.features import (
    FeatureBuildResult,
    FeatureRow,
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for an input file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _format_value(value: int | float | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def _write_split(
    path: Path, rows: list[FeatureRow], feature_names: tuple[str, ...]
) -> None:
    """Write a CSV using stable column and row ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [*IDENTIFIER_COLUMNS, *feature_names, TARGET_COLUMN]
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        writer = csv.DictWriter(temporary, fieldnames=headers, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            values: dict[str, int | float | str | None] = {
                "code_module": row.code_module,
                "code_presentation": row.code_presentation,
                "id_student": row.id_student,
                "id_assessment": row.id_assessment,
                **row.features,
                TARGET_COLUMN: row.target_score,
            }
            writer.writerow({name: _format_value(values[name]) for name in headers})
    os.replace(temporary_path, path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically with sorted keys for reproducible diffs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(payload, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
    os.replace(temporary_path, path)


def persist_feature_splits(
    result: FeatureBuildResult,
    source_files: dict[str, Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Write per-split features and train-fitted preprocessing metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths: dict[str, str] = {}
    for split_name, rows in result.rows_by_split.items():
        filename = f"fff_{split_name}.csv"
        _write_split(output_dir / filename, rows, result.feature_names)
        split_paths[split_name] = filename

    manifest: dict[str, Any] = {
        "feature_schema_version": 1,
        "course_module": "FFF",
        "target": TARGET_COLUMN,
        "identifiers_not_features": list(IDENTIFIER_COLUMNS),
        "prediction_cutoff": (
            "Start of the published assessment deadline day (integer day offset D). "
            "Only VLE events with day < D are included."
        ),
        "split_presentations": {
            split: list(presentations)
            for split, presentations in SPLIT_PRESENTATIONS.items()
        },
        "feature_names": list(result.feature_names),
        "score_feature_mode": result.score_feature_mode,
        "assessment_type_categories_fitted_from_training_presentations": list(
            result.assessment_type_categories
        ),
        "previous_score_lags_fitted_from_training_schedule": (
            result.max_previous_score_lags
        ),
        "click_window_day_offsets_relative_to_deadline": [
            {"week": index + 1, "start_inclusive": start, "end_exclusive": end}
            for index, (start, end) in enumerate(CLICK_WINDOW_OFFSETS)
        ],
        "activity_groups": {
            group: list(types) for group, types in ACTIVITY_GROUPS.items()
        },
        "score_availability_assumption": (
            "Strict mode excludes all previous assessment score features."
            if result.score_feature_mode == "strict"
            else "Proxy mode includes only non-banked, valid prior scores submitted "
            "on or before their own strictly earlier deadline. OULAD does not "
            "provide score-release timestamps, so actual availability before the "
            "current cutoff cannot be verified."
        ),
        "missing_value_policy": (
            "Unavailable prior scores and incomplete click windows are blank, with "
            "prior-score availability indicators. A complete click window with no "
            "matching event is zero recorded clicks. No imputation or scaling is fit."
        ),
        "target_policy": (
            "Only scores in [0, 100] from non-banked submissions on or before the "
            "target deadline are retained. Submission dates and target scores are "
            "not input features."
        ),
        "split_files": split_paths,
        "row_counts": {
            split: len(rows) for split, rows in result.rows_by_split.items()
        },
        "source_sha256": {
            filename: sha256_file(path)
            for filename, path in sorted(source_files.items())
        },
    }
    _write_json(output_dir / "fff_preprocessing.json", manifest)
    return manifest
