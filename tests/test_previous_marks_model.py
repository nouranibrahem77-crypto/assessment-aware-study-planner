"""Tests for the isolated, submission-proxy Previous Marks model path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("torch")

import joblib
import torch

from study_planner.nn.dataset import (
    DatasetSchemaError,
    FeatureSplit,
    load_previous_marks_feature_data,
    load_strict_feature_data,
    prepare_inference_frame,
    prepare_previous_marks_inference_frame,
)
from study_planner.nn.inference import (
    PreviousMarksRegressionPredictor,
    RegressionPredictor,
)
from study_planner.nn.models import MLPRegressor
from study_planner.nn.preprocessing import build_preprocessor
from study_planner.nn.previous_marks_training import (
    _coverage,
    _fit_preprocessing,
    _paired_split,
)

IDENTIFIERS = ("code_module", "code_presentation", "id_student", "id_assessment")
PRIOR_COLUMNS = tuple(
    name
    for lag in range(1, 6)
    for name in (
        f"previous_assessment_score_lag_{lag}",
        f"previous_assessment_score_lag_{lag}_available",
    )
)
FEATURES = (
    "assessment_type_cma",
    "assessment_type_exam",
    "assessment_type_tma",
    "assessment_weight",
    "assessment_weight_available",
    "assessment_deadline_day",
    "weeks_since_course_start",
    *PRIOR_COLUMNS,
    *(f"click_content_navigation_week_{week}" for week in range(1, 5)),
    *(f"click_assessment_activity_week_{week}" for week in range(1, 5)),
    *(f"click_collaboration_week_{week}" for week in range(1, 5)),
)


def _data_split(
    ids: list[tuple[str, str, str, str]],
    *,
    scores: list[float],
    types: list[str] | None = None,
) -> FeatureSplit:
    type_names = types or ["TMA"] * len(ids)
    identifiers = pd.DataFrame(ids, columns=IDENTIFIERS)
    x = pd.DataFrame(0.0, index=range(len(ids)), columns=FEATURES)
    for index, category in enumerate(type_names):
        x.loc[index, f"assessment_type_{category.lower()}"] = 1.0
    x["previous_assessment_score_lag_1"] = [12.0] * len(ids)
    x["previous_assessment_score_lag_1_available"] = [1] * len(ids)
    for lag in range(2, 6):
        x[f"previous_assessment_score_lag_{lag}"] = [np.nan] * len(ids)
        x[f"previous_assessment_score_lag_{lag}_available"] = [0] * len(ids)
    return FeatureSplit(
        X=x,
        y=np.asarray(scores, dtype=float),
        assessment_types=np.asarray(type_names, dtype=object),
        identifiers=identifiers,
    )


def _write_proxy_dataset(directory: Path) -> tuple[str, ...]:
    directory.mkdir()
    feature_names = FEATURES
    type_columns = (
        "assessment_type_cma",
        "assessment_type_exam",
        "assessment_type_tma",
    )
    split_rows = {
        "train": [("2013B", "train-1"), ("2013J", "train-2")],
        "validation": [("2014B", "val-1")],
        "test": [("2014J", "test-1")],
    }
    manifest: dict[str, object] = {
        "course_module": "FFF",
        "score_feature_mode": "proxy",
        "target": "target_score",
        "feature_names": list(feature_names),
        "identifiers_not_features": list(IDENTIFIERS),
        "prediction_cutoff": "start of deadline day",
        "split_files": {},
        "row_counts": {},
    }
    for split_name, split_records in split_rows.items():
        filename = f"fff_{split_name}.csv"
        manifest["split_files"][split_name] = filename  # type: ignore[index]
        rows = [
            {
                "code_module": "FFF",
                "code_presentation": presentation,
                "id_student": student_id,
                "id_assessment": f"{split_name}-{student_id}-assessment",
                **{name: 0 for name in feature_names},
                "assessment_type_tma": 1,
                **{f"previous_assessment_score_lag_{lag}": "" for lag in range(1, 6)},
                "target_score": 60,
            }
            for presentation, student_id in split_records
        ]
        pd.DataFrame(rows).to_csv(directory / filename, index=False)
        manifest["row_counts"][split_name] = len(rows)  # type: ignore[index]
    (directory / "fff_preprocessing.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return type_columns


def test_previous_marks_loader_is_separate_and_preserves_missing_history(
    tmp_path: Path,
) -> None:
    _write_proxy_dataset(tmp_path / "proxy")
    data = load_previous_marks_feature_data(tmp_path / "proxy")
    score = "previous_assessment_score_lag_1"
    available = f"{score}_available"

    assert data.feature_names == FEATURES
    assert pd.isna(data.splits["train"].X.loc[0, score])
    assert data.splits["train"].X.loc[0, available] == 0
    with pytest.raises(DatasetSchemaError, match="strict feature mode"):
        load_strict_feature_data(tmp_path / "proxy")


def test_strict_inference_does_not_accept_model_b_features() -> None:
    features = pd.DataFrame({name: [0.0] for name in FEATURES})
    features.loc[0, "assessment_type_tma"] = 1.0
    for lag in range(1, 6):
        features.loc[0, f"previous_assessment_score_lag_{lag}"] = np.nan
    with pytest.raises(DatasetSchemaError, match="Strict inputs"):
        prepare_inference_frame(features, FEATURES, FEATURES[:3])
    prepared = prepare_previous_marks_inference_frame(features, FEATURES, FEATURES[:3])
    assert pd.isna(prepared.loc[0, "previous_assessment_score_lag_1"])


def test_proxy_availability_indicators_are_validated(tmp_path: Path) -> None:
    _write_proxy_dataset(tmp_path / "proxy")
    path = tmp_path / "proxy" / "fff_train.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "previous_assessment_score_lag_1_available"] = 2
    frame.loc[0, "previous_assessment_score_lag_1"] = 23
    frame.to_csv(path, index=False)
    with pytest.raises(DatasetSchemaError, match="availability indicator"):
        load_previous_marks_feature_data(tmp_path / "proxy")


def test_paired_split_aligns_by_identity_and_rejects_target_mismatch() -> None:
    ids = [
        ("FFF", "2013B", "s1", "a1"),
        ("FFF", "2013B", "s2", "a1"),
    ]
    strict = _data_split(ids, scores=[10.0, 20.0])
    proxy = _data_split(ids[::-1], scores=[20.0, 10.0])
    strict_pair, proxy_pair = _paired_split(strict, proxy, "train")
    assert strict_pair.identifiers["id_student"].tolist() == ["s1", "s2"]
    assert proxy_pair.identifiers["id_student"].tolist() == ["s1", "s2"]
    assert np.array_equal(strict_pair.y, proxy_pair.y)

    proxy.y[0] = 19.0
    with pytest.raises(ValueError, match="Targets differ"):
        _paired_split(strict, proxy, "train")


def test_coverage_counts_rows_with_at_least_one_prior_mark() -> None:
    split = _data_split(
        [("FFF", "2013B", "s1", "a1"), ("FFF", "2013B", "s2", "a1")],
        scores=[10.0, 20.0],
    )
    split.X.loc[1, "previous_assessment_score_lag_1"] = np.nan
    split.X.loc[1, "previous_assessment_score_lag_1_available"] = 0
    coverage = _coverage(split)
    assert coverage["rows_with_any_prior_mark"] == 1
    assert coverage["rows_without_prior_mark"] == 1
    assert coverage["prior_mark_coverage"] == 0.5


def test_model_b_preprocessing_is_fit_only_on_paired_training_rows() -> None:
    train = _data_split(
        [
            ("FFF", "2013B", "s1", "a1"),
            ("FFF", "2013J", "s2", "a1"),
        ],
        scores=[10.0, 20.0],
    )
    validation = _data_split([("FFF", "2014B", "s3", "a1")], scores=[30.0])
    test = _data_split([("FFF", "2014J", "s4", "a1")], scores=[40.0])
    score_name = "previous_assessment_score_lag_1"
    train.X[score_name] = [10.0, 30.0]
    validation.X[score_name] = [1000.0]
    test.X[score_name] = [2000.0]

    def model_input_split(split: FeatureSplit) -> FeatureSplit:
        model_frame = split.X.drop(columns=list(FEATURES[:3])).copy()
        model_frame["assessment_type"] = split.assessment_types
        return FeatureSplit(
            X=model_frame,
            y=split.y,
            assessment_types=split.assessment_types,
            identifiers=split.identifiers,
        )

    train, validation, test = map(model_input_split, (train, validation, test))
    numeric_names = tuple(name for name in FEATURES if name not in FEATURES[:3])

    preprocessor, transformed = _fit_preprocessing(
        train, validation, test, numeric_names
    )

    pipeline = preprocessor.named_transformers_["numeric"]
    score_position = numeric_names.index(score_name)
    assert pipeline.named_steps["imputer"].statistics_[score_position] == 20.0
    assert transformed["train"].shape[0] == 2
    assert transformed["validation"].shape[0] == 1
    assert transformed["test"].shape[0] == 1


def test_proxy_predictor_artifacts_are_separate_from_strict(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "proxy_artifact"
    artifact_dir.mkdir()
    categories = ["TMA", "CMA"]
    numeric_names = tuple(name for name in FEATURES if name not in FEATURES[:3])
    fit_frame = pd.DataFrame({"assessment_type": categories})
    for name in numeric_names:
        fit_frame[name] = (
            [np.nan, 1.0] if "previous_assessment_score" in name else [1.0, 2.0]
        )
    preprocessor = build_preprocessor(numeric_names)
    transformed = preprocessor.fit_transform(fit_frame)
    joblib.dump(preprocessor, artifact_dir / "preprocessor.joblib")
    schema = {
        "mode": "proxy",
        "source_feature_columns": list(FEATURES),
        "assessment_type_indicator_columns": list(FEATURES[:3]),
    }
    config = {
        "feature_mode": "proxy",
        "selected_mlp_config": {"hidden_sizes": [4], "dropout": 0.0},
        "preprocessed_feature_count": transformed.shape[1],
        "model_files": ["model.pt"],
    }
    (artifact_dir / "feature_schema.json").write_text(
        json.dumps(schema), encoding="utf-8"
    )
    (artifact_dir / "experiment_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    model = MLPRegressor(transformed.shape[1], (4,), 0.0)
    torch.save(model.state_dict(), artifact_dir / "model.pt")

    with pytest.raises(ValueError, match="strict feature artifacts"):
        RegressionPredictor.from_artifact_dir(artifact_dir)
    predictor = PreviousMarksRegressionPredictor.from_artifact_dir(artifact_dir)
    batch = pd.DataFrame({name: [0.0] for name in FEATURES})
    batch.loc[0, "assessment_type_tma"] = 1
    for lag in range(1, 6):
        batch.loc[0, f"previous_assessment_score_lag_{lag}"] = np.nan
        batch.loc[0, f"previous_assessment_score_lag_{lag}_available"] = 0
    first = predictor.predict(batch)
    second = predictor.predict(batch)
    assert np.array_equal(first, second)
    assert first.shape == (1,)
