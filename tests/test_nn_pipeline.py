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

from study_planner.nn import dataset as dataset_module
from study_planner.nn import training as training_module
from study_planner.nn.dataset import DatasetSchemaError, prepare_inference_frame
from study_planner.nn.inference import RegressionPredictor
from study_planner.nn.models import MLPRegressor
from study_planner.nn.preprocessing import build_preprocessor
from study_planner.nn.training import ReportSchemaError, _render_report


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


def _report_results() -> dict[str, object]:
    """Provide a complete in-memory report fixture with every result shape."""
    averaged = {
        f"{metric}_{suffix}": value
        for metric in ("mae", "rmse", "r2")
        for suffix, value in (("mean", 1.0), ("std", 0.1))
    }
    overall = {"mae": 1.0, "rmse": 1.2, "r2": 0.5, "n": 2}
    assessed = {
        "overall": overall,
        "by_assessment_type": {"TMA": overall},
    }
    runs = [
        {
            "seed": 42,
            "epochs_run": 2,
            "best_epoch": 1,
            "best_validation_loss": 0.1,
            "duration_seconds": 0.2,
            "history": [
                {
                    "epoch": 1,
                    "train_loss": 0.2,
                    "validation_loss": 0.1,
                    "validation_mae": 1.0,
                }
            ],
        }
    ]
    return {
        "shapes": {key: (2, 1) for key in ("train", "validation", "test")},
        "input_feature_names": ["assessment_weight"],
        "transformed_feature_names": ["numeric__assessment_weight"],
        "ridge_alpha_grid": [0.1, 1.0],
        "training_config": {"batch_size": 2},
        "seeds": [42],
        "selected_config": {"name": "small", "hidden_sizes": [4], "dropout": 0.0},
        "selected_ridge_alpha": 1.0,
        "validation_metrics": {
            "Dummy mean": averaged,
            "Ridge": averaged,
            "MLP (selected config, seed mean)": averaged,
        },
        "search_results": [
            {
                "config": {"name": "small"},
                "validation_mae_mean": 1.0,
                "validation_mae_std": 0.0,
            }
        ],
        "test_metrics": {
            "Dummy mean": assessed,
            "Ridge": assessed,
            "MLP seed mean": {**averaged, "n": 2},
            "MLP seed ensemble": assessed,
        },
        "selected_runs": runs,
        "train_metrics": averaged,
        "experiment_duration_seconds": 0.5,
        "mlp_search_duration_seconds": 0.3,
    }


def test_report_renderer_handles_baseline_nested_and_seed_mean_metrics(
    tmp_path: Path,
) -> None:
    report = _render_report(_report_results(), tmp_path)

    assert "| Dummy mean | 1.000 | 1.200 | 0.5000 | 2 |" in report
    assert "| MLP seed mean | 1.000 ± 0.100 | 1.000 ± 0.100" in report
    assert "| MLP seed ensemble | 1.000 | 1.200 | 0.5000 | 2 |" in report
    assert "| Dummy mean | TMA | 2 | 1.000 | 1.200 | 0.5000 |" in report


def test_report_renderer_fails_actionably_for_missing_baseline_metric(
    tmp_path: Path,
) -> None:
    results = _report_results()
    del results["test_metrics"]["Dummy mean"]["overall"]["mae"]  # type: ignore[index]

    with pytest.raises(
        ReportSchemaError,
        match=r"test_metrics.*Dummy mean.*overall.*mae",
    ):
        _render_report(results, tmp_path)


def test_report_renderer_fails_actionably_for_incomplete_seed_mean(
    tmp_path: Path,
) -> None:
    results = _report_results()
    del results["test_metrics"]["MLP seed mean"]["r2_mean"]  # type: ignore[index]

    with pytest.raises(
        ReportSchemaError,
        match=r"test_metrics.*MLP seed mean.*r2_mean",
    ):
        _render_report(results, tmp_path)


def test_validation_only_execution_never_reads_test_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    read_files: list[str] = []
    original_read_csv = dataset_module.pd.read_csv

    def guarded_read_csv(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        filename = Path(path).name
        if filename == "fff_test.csv":
            pytest.fail("validation-only execution attempted to read the test split")
        read_files.append(filename)
        return original_read_csv(path, *args, **kwargs)

    def fake_fit_mlp(*args: object, **kwargs: object) -> training_module.MLPTrainingRun:
        config = args[0]
        seed = int(args[1])
        train_features = args[2]
        validation_features = args[4]
        validation_targets = args[5]
        validation_types = args[6]
        assert isinstance(config, training_module.MLPConfig)
        assert isinstance(train_features, np.ndarray)
        assert isinstance(validation_features, np.ndarray)
        model = MLPRegressor(
            train_features.shape[1], config.hidden_sizes, config.dropout
        )
        predictions = np.full(
            len(validation_targets), float(np.mean(validation_targets))
        )
        metrics = training_module.regression_metrics(
            validation_targets, predictions, validation_types
        )
        history = [
            {
                "epoch": 1,
                "train_loss": 0.1,
                "validation_loss": 0.1,
                "validation_mae": metrics["overall"]["mae"],
            }
        ]
        return training_module.MLPTrainingRun(
            config=config,
            seed=seed,
            state_dict=model.state_dict(),
            history=history,
            best_epoch=1,
            best_validation_loss=0.1,
            validation_metrics=metrics,
            duration_seconds=0.01,
        )

    monkeypatch.setattr(dataset_module.pd, "read_csv", guarded_read_csv)
    monkeypatch.setattr(training_module, "_fit_mlp", fake_fit_mlp)
    output_dir = tmp_path / "nn_experiment_60ep"
    report_path = tmp_path / "nn_extended_training.md"

    result = training_module.run_validation_only_experiment(
        processed_dir=project_root / "data/processed/strict",
        original_output_dir=project_root / "data/processed/nn_experiment",
        output_dir=output_dir,
        report_path=report_path,
        max_epochs=1,
    )

    assert read_files == ["fff_train.csv", "fff_validation.csv"]
    assert result["validation_only"] is True
    assert result["test_partition_accessed"] is False
    assert "test_metrics" not in result
    assert len(result["seeds"]) == 3
    assert (output_dir / "mlp_seed_42.pt").is_file()
    assert report_path.is_file()
