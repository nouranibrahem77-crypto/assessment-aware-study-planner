"""Schema-checked inference from persisted strict FFF MLP artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from weakref import WeakKeyDictionary

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


@dataclass(frozen=True, slots=True)
class _TrustedArtifactProvenance:
    digest: str
    mode: str
    fitted_state_digest: str
    preprocessor_identity: int
    model_collection_identity: int
    model_identities: tuple[int, ...]


_TRUSTED_ARTIFACTS: WeakKeyDictionary[RegressionPredictor, _TrustedArtifactProvenance]
_TRUSTED_ARTIFACTS = WeakKeyDictionary()
_TRUSTED_ARTIFACTS_LOCK = RLock()


def _hash_field(digest: object, name: str, value: bytes) -> None:
    """Add one unambiguous length-framed field to a hashlib object."""
    digest.update(struct.pack(">Q", len(name.encode("utf-8"))))
    digest.update(name.encode("utf-8"))
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


def _fitted_state_digest(predictor: RegressionPredictor) -> str:
    """Fingerprint all mutable fitted inference state and its public schema."""
    digest = hashlib.sha256(b"study-planner-loaded-state-v1\0")
    preprocessor_state = pickle.dumps(predictor.preprocessor, protocol=5)
    _hash_field(digest, "preprocessor", preprocessor_state)
    schema = json.dumps(
        {
            "feature_names": tuple(predictor.feature_names),
            "assessment_type_columns": tuple(predictor.assessment_type_columns),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    _hash_field(digest, "schema", schema)
    for model_index, model in enumerate(predictor.models):
        _hash_field(digest, f"model-{model_index}-type", repr(type(model)).encode())
        model_state = pickle.dumps(model, protocol=5)
        _hash_field(digest, f"model-{model_index}-module-state", model_state)
        for key, tensor in sorted(model.state_dict().items()):
            contiguous = tensor.detach().to(device="cpu").contiguous()
            metadata = json.dumps(
                {"dtype": str(contiguous.dtype), "shape": tuple(contiguous.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            _hash_field(digest, f"model-{model_index}-{key}-metadata", metadata)
            raw = contiguous.reshape(-1).view(torch.uint8).numpy().tobytes()
            _hash_field(digest, f"model-{model_index}-{key}-bytes", raw)
    return digest.hexdigest()


def trusted_artifact_provenance(
    predictor: object,
) -> _TrustedArtifactProvenance | None:
    """Return loader-issued provenance for this exact predictor instance."""
    if not isinstance(predictor, RegressionPredictor):
        return None
    with _TRUSTED_ARTIFACTS_LOCK:
        return _TRUSTED_ARTIFACTS.get(predictor)


def verify_trusted_artifact_integrity(
    predictor: object,
    expected: _TrustedArtifactProvenance | None = None,
) -> _TrustedArtifactProvenance:
    """Fail closed if a loaded predictor or its fitted state has changed."""
    provenance = trusted_artifact_provenance(predictor)
    if provenance is None or (expected is not None and provenance is not expected):
        raise ValueError("predictor has no trusted artifact-loading provenance")
    assert isinstance(predictor, RegressionPredictor)
    try:
        state_digest = _fitted_state_digest(predictor)
        changed = (
            predictor.artifact_digest != provenance.digest
            or id(predictor.preprocessor) != provenance.preprocessor_identity
            or id(predictor.models) != provenance.model_collection_identity
            or tuple(id(model) for model in predictor.models)
            != provenance.model_identities
            or state_digest != provenance.fitted_state_digest
        )
    except Exception as error:
        raise ValueError("loaded fitted state cannot be integrity-verified") from error
    if changed:
        raise ValueError("loaded fitted state no longer matches its artifact")
    return provenance


def artifact_digest_from_files(files: Mapping[str, bytes]) -> str:
    """Return a deterministic SHA-256 digest for named persisted artifact bytes."""
    if not files:
        raise ValueError("fitted artifact payload cannot be empty")
    digest = hashlib.sha256(b"study-planner-fitted-artifact-v1\0")
    for name in sorted(files):
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or not isinstance(files[name], bytes)
        ):
            raise ValueError("artifact payload must map simple filenames to bytes")
        name_bytes = name.encode("utf-8")
        content = files[name]
        digest.update(struct.pack(">Q", len(name_bytes)))
        digest.update(name_bytes)
        digest.update(struct.pack(">Q", len(content)))
        digest.update(content)
    return digest.hexdigest()


@dataclass(eq=False)
class RegressionPredictor:
    """Loaded preprocessing and seed ensemble for validated feature batches."""

    preprocessor: ColumnTransformer
    feature_names: tuple[str, ...]
    assessment_type_columns: tuple[str, ...]
    models: list[MLPRegressor]
    artifact_digest: str | None = None

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
        schema_bytes = schema_path.read_bytes()
        config_bytes = config_path.read_bytes()
        schema = json.loads(schema_bytes)
        config = json.loads(config_bytes)
        if (
            schema.get("mode") != expected_mode
            or config.get("feature_mode") != expected_mode
        ):
            raise ValueError(f"Expected {expected_mode} feature artifacts")
        feature_names = tuple(schema["source_feature_columns"])
        type_columns = tuple(schema["assessment_type_indicator_columns"])
        model_files = config["model_files"]
        reserved_names = {
            "feature_schema.json",
            "experiment_config.json",
            "preprocessor.joblib",
        }
        if (
            not isinstance(model_files, list)
            or not model_files
            or any(
                not isinstance(filename, str)
                or not filename
                or Path(filename).name != filename
                or filename in reserved_names
                for filename in model_files
            )
            or len(set(model_files)) != len(model_files)
        ):
            raise ValueError("Model artifact checkpoint list is invalid")
        artifact_bytes = {
            "feature_schema.json": schema_bytes,
            "experiment_config.json": config_bytes,
            "preprocessor.joblib": (artifact_dir / "preprocessor.joblib").read_bytes(),
        }
        for filename in model_files:
            artifact_bytes[filename] = (artifact_dir / filename).read_bytes()
        artifact_digest = artifact_digest_from_files(artifact_bytes)
        preprocessor = joblib.load(io.BytesIO(artifact_bytes["preprocessor.joblib"]))
        mlp_config = config["selected_mlp_config"]
        input_size = int(config["preprocessed_feature_count"])
        models: list[MLPRegressor] = []
        for filename in model_files:
            model = MLPRegressor(
                input_size=input_size,
                hidden_sizes=tuple(mlp_config["hidden_sizes"]),
                dropout=float(mlp_config["dropout"]),
            )
            state = torch.load(
                io.BytesIO(artifact_bytes[filename]),
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state)
            model.eval()
            models.append(model)
        if not models:
            raise ValueError("No selected MLP checkpoints are listed in artifacts")
        predictor = cls(
            preprocessor, feature_names, type_columns, models, artifact_digest
        )
        if expected_mode == "proxy":
            for lag in range(1, 6):
                score_name = f"previous_assessment_score_lag_{lag}"
                if score_name not in predictor.feature_names:
                    raise ValueError(f"Previous Marks artifacts lack {score_name}")
                if f"{score_name}_available" not in predictor.feature_names:
                    raise ValueError(
                        f"Previous Marks artifacts lack {score_name}_available"
                    )
        provenance = _TrustedArtifactProvenance(
            digest=artifact_digest,
            mode=expected_mode,
            fitted_state_digest=_fitted_state_digest(predictor),
            preprocessor_identity=id(predictor.preprocessor),
            model_collection_identity=id(predictor.models),
            model_identities=tuple(id(model) for model in predictor.models),
        )
        with _TRUSTED_ARTIFACTS_LOCK:
            _TRUSTED_ARTIFACTS[predictor] = provenance
        return predictor

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict a batch, averaging selected seeds and clipping to [0, 100]."""
        verify_trusted_artifact_integrity(self)
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
        verify_trusted_artifact_integrity(self)
        return clip_predictions(np.mean(predictions, axis=0))


class PreviousMarksRegressionPredictor(RegressionPredictor):
    """Inference adapter for the separate submission-proxy Model B artifacts."""

    @classmethod
    def from_artifact_dir(cls, artifact_dir: Path) -> PreviousMarksRegressionPredictor:
        """Load only artifacts explicitly tagged for previous-mark proxy mode."""
        return cls._from_artifact_dir(artifact_dir, expected_mode="proxy")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict with Model B's schema while preserving missing prior marks."""
        verify_trusted_artifact_integrity(self)
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
        verify_trusted_artifact_integrity(self)
        return clip_predictions(np.mean(predictions, axis=0))
