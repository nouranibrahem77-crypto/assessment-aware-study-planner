"""Schema-checked inference from persisted strict FFF MLP artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer

from study_planner.nn.dataset import (
    prepare_inference_frame,
    prepare_previous_marks_inference_frame,
)
from study_planner.nn.metrics import TARGET_SCALE, clip_predictions
from study_planner.nn.models import MLPRegressor


@dataclass
class RegressionPredictor:
    """Loaded preprocessing and seed ensemble for validated feature batches."""

    preprocessor: ColumnTransformer
    feature_names: tuple[str, ...]
    assessment_type_columns: tuple[str, ...]
    models: list[MLPRegressor]

    @classmethod
    def from_artifact_dir(cls, artifact_dir: Path) -> RegressionPredictor:
        """Load a saved strict preprocessing object and selected MLP ensemble."""
        return cls._from_artifact_dir(artifact_dir, expected_mode="strict")

    @classmethod
    def _from_artifact_dir(
        cls, artifact_dir: Path, *, expected_mode: str
    ) -> RegressionPredictor:
        schema_path = artifact_dir / "feature_schema.json"
        config_path = artifact_dir / "experiment_config.json"
        if not schema_path.is_file() or not config_path.is_file():
            raise FileNotFoundError("Model artifact schema/configuration is incomplete")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if (
            schema.get("mode") != expected_mode
            or config.get("feature_mode") != expected_mode
        ):
            raise ValueError(f"Expected {expected_mode} feature artifacts")
        feature_names = tuple(schema["source_feature_columns"])
        type_columns = tuple(schema["assessment_type_indicator_columns"])
        preprocessor = joblib.load(artifact_dir / "preprocessor.joblib")
        mlp_config = config["selected_mlp_config"]
        input_size = int(config["preprocessed_feature_count"])
        models: list[MLPRegressor] = []
        for filename in config["model_files"]:
            model = MLPRegressor(
                input_size=input_size,
                hidden_sizes=tuple(mlp_config["hidden_sizes"]),
                dropout=float(mlp_config["dropout"]),
            )
            state = torch.load(
                artifact_dir / filename,
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state)
            model.eval()
            models.append(model)
        if not models:
            raise ValueError("No selected MLP checkpoints are listed in artifacts")
        return cls(preprocessor, feature_names, type_columns, models)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict a batch, averaging selected seeds and clipping to [0, 100]."""
        frame = prepare_inference_frame(
            features, self.feature_names, self.assessment_type_columns
        )
        transformed = np.asarray(self.preprocessor.transform(frame), dtype=np.float32)
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            tensor = torch.from_numpy(transformed)
            for model in self.models:
                model.eval()
                predictions.append(model(tensor).cpu().numpy() * TARGET_SCALE)
        return clip_predictions(np.mean(predictions, axis=0))


class PreviousMarksRegressionPredictor(RegressionPredictor):
    """Inference adapter for the separate submission-proxy Model B artifacts."""

    @classmethod
    def from_artifact_dir(cls, artifact_dir: Path) -> PreviousMarksRegressionPredictor:
        """Load only artifacts explicitly tagged for previous-mark proxy mode."""
        predictor = cls._from_artifact_dir(artifact_dir, expected_mode="proxy")
        for lag in range(1, 6):
            score_name = f"previous_assessment_score_lag_{lag}"
            if score_name not in predictor.feature_names:
                raise ValueError(f"Previous Marks artifacts lack {score_name}")
            if f"{score_name}_available" not in predictor.feature_names:
                raise ValueError(
                    f"Previous Marks artifacts lack {score_name}_available"
                )
        return predictor

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict with Model B's schema while preserving missing prior marks."""
        frame = prepare_previous_marks_inference_frame(
            features, self.feature_names, self.assessment_type_columns
        )
        transformed = np.asarray(self.preprocessor.transform(frame), dtype=np.float32)
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            tensor = torch.from_numpy(transformed)
            for model in self.models:
                model.eval()
                predictions.append(model(tensor).cpu().numpy() * TARGET_SCALE)
        return clip_predictions(np.mean(predictions, axis=0))
