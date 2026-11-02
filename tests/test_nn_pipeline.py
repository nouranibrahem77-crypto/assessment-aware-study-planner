"""Focused tests for train-only preprocessing and persisted MLP inference."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("torch")

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder

from study_planner.nn.dataset import DatasetSchemaError, prepare_inference_frame
from study_planner.nn.inference import RegressionPredictor
from study_planner.nn.models import MLPRegressor
from study_planner.nn.preprocessing import build_preprocessor


def test_preprocessing_fit_uses_training_rows_only_and_ignores_unseen_type() -> None:
    train = pd.DataFrame(
        {
            "assessment_type": ["TMA", "CMA", "TMA"],
            "numeric": [1.0, 3.0, np.nan],
        }
    )
    validation = pd.DataFrame(
        {"assessment_type": ["Exam"], "numeric": [1000.0]}
    )
    preprocessor = build_preprocessor(("numeric",))
    train_values = preprocessor.fit_transform(train)
    validation_values = preprocessor.transform(validation)

    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    assert numeric_pipeline.named_steps["imputer"].statistics_[0] == 2.0
    assert numeric_pipeline.named_steps["scaler"].mean_[0] == 2.0
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    assert isinstance(encoder, OneHotEncoder)
    assert encoder.categories_[0].tolist() == ["CMA", "TMA"]
    assert validation_values.shape == train_values[:1].shape
    assert np.isfinite(train_values).all()
    assert np.isfinite(validation_values).all()
    assert validation_values[0, -2:].tolist() == [0.0, 0.0]


def test_inference_schema_rejects_identifiers_targets_and_missing_columns() -> None:
    expected = ("assessment_type_tma", "numeric")
    type_columns = ("assessment_type_tma",)
    with pytest.raises(DatasetSchemaError, match="identifiers or target"):
        prepare_inference_frame(
            pd.DataFrame(
                {
                    "assessment_type_tma": [1],
                    "numeric": [2.0],
                    "target_score": [85.0],
                }
            ),
            expected,
            type_columns,
        )
    with pytest.raises(DatasetSchemaError, match="schema mismatch"):
        prepare_inference_frame(
            pd.DataFrame({"assessment_type_tma": [1]}), expected, type_columns
        )


def test_inference_artifact_reload_is_repeatable_batched_and_clipped(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    feature_names = ("assessment_type_tma", "assessment_weight")
    type_columns = ("assessment_type_tma",)
    frame = pd.DataFrame(
        {
            "assessment_type": ["TMA", "TMA"],
            "assessment_weight": [10.0, 20.0],
        }
    )
    preprocessor = build_preprocessor(("assessment_weight",))
    transformed = preprocessor.fit_transform(frame)
    joblib.dump(preprocessor, artifact_dir / "preprocessor.joblib")
    (artifact_dir / "feature_schema.json").write_text(
        json.dumps(
            {
                "mode": "strict",
                "source_feature_columns": list(feature_names),
                "assessment_type_indicator_columns": list(type_columns),
            }
        ),
        encoding="utf-8",
    )
    model = MLPRegressor(transformed.shape[1], (4,), 0.0)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    model.layers[-1].bias.data.fill_(150.0)
    torch.save(model.state_dict(), artifact_dir / "mlp_seed_11.pt")
    (artifact_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "feature_mode": "strict",
                "selected_mlp_config": {
                    "hidden_sizes": [4],
                    "dropout": 0.0,
                },
                "preprocessed_feature_count": transformed.shape[1],
                "model_files": ["mlp_seed_11.pt"],
            }
        ),
        encoding="utf-8",
    )

    predictor = RegressionPredictor.from_artifact_dir(artifact_dir)
    batch = pd.DataFrame(
        {
            "assessment_type_tma": [1, 1, 1],
            "assessment_weight": [10.0, 15.0, 20.0],
        }
    )
    first = predictor.predict(batch)
    second = predictor.predict(batch)

    assert first.shape == (3,)
    assert np.array_equal(first, second)
    assert first.tolist() == [100.0, 100.0, 100.0]
