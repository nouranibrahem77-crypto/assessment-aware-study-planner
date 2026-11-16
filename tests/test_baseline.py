"""Tests for the train-fitted Simple Average-Mark Predictor."""

import csv
import json

import pytest

from study_planner.data.feature_engineering.constants import (
    IDENTIFIER_COLUMNS,
    STRICT_FFF_FEATURE_NAMES,
    TARGET_COLUMN,
)
from study_planner.eval.baseline import SimpleAverageMarkPredictor


def test_baseline_fits_only_from_strict_training_split(tmp_path) -> None:
    feature_names = [*IDENTIFIER_COLUMNS, *STRICT_FFF_FEATURE_NAMES, TARGET_COLUMN]
    train_path = tmp_path / "fff_train.csv"
    with train_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=feature_names)
        writer.writeheader()
        writer.writerow(
            {
                "code_module": "FFF",
                "code_presentation": "2013B",
                "id_student": "101",
                "id_assessment": "10",
                "target_score": "60",
            }
        )
        writer.writerow(
            {
                "code_module": "FFF",
                "code_presentation": "2013J",
                "id_student": "102",
                "id_assessment": "11",
                "target_score": "80",
            }
        )
    (tmp_path / "fff_preprocessing.json").write_text(
        json.dumps(
            {
                "course_module": "FFF",
                "score_feature_mode": "strict",
                "feature_names": list(STRICT_FFF_FEATURE_NAMES),
                "split_files": {"train": train_path.name},
            }
        ),
        encoding="utf-8",
    )

    predictor = SimpleAverageMarkPredictor.fit_from_processed_directory(tmp_path)

    assert predictor.mean_mark == 70.0
    assert predictor.training_target_count == 2
    assert predictor.source_data_sha256 is not None
    assert len(predictor.source_data_sha256) == 64
    assert predictor.verify_integrity() == predictor.artifact_digest


def test_baseline_requires_strict_schema_and_validates_integrity() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        SimpleAverageMarkPredictor.fit([])

    predictor = SimpleAverageMarkPredictor.fit([70.0, 80.0])
    object.__setattr__(predictor, "mean_mark", 99.0)

    with pytest.raises(ValueError, match="changed"):
        predictor.verify_integrity()
