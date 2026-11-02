"""Train, select, evaluate, and persist strict FFF regression models."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from study_planner.nn.dataset import (
    StrictFeatureData,
    load_strict_feature_data,
)
from study_planner.nn.metrics import (
    PREDICTION_MAX,
    PREDICTION_MIN,
    TARGET_SCALE,
    clip_predictions,
    metric_json_safe,
    regression_metrics,
)
from study_planner.nn.models import DEFAULT_SEARCH_CONFIGS, MLPConfig, MLPRegressor
from study_planner.nn.preprocessing import build_preprocessor

LOGGER = logging.getLogger(__name__)
TRAINING_SEEDS: tuple[int, ...] = (11, 23, 42)
RIDGE_ALPHAS: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
MAX_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 6
BATCH_SIZE = 512
HUBER_DELTA = 1.0


@dataclass
class MLPTrainingRun:
    """Restored best checkpoint and validation evidence for one run."""

    config: MLPConfig
    seed: int
    state_dict: dict[str, Tensor]
    history: list[dict[str, float | int]]
    best_epoch: int
    best_validation_loss: float
    validation_metrics: dict[str, Any]
    duration_seconds: float


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _prepare_arrays(
    data: StrictFeatureData,
) -> tuple[Any, dict[str, np.ndarray]]:
    """Fit the sklearn preprocessor on training rows and transform other splits."""
    preprocessor = build_preprocessor(data.numeric_feature_names)
    train = data.splits["train"]
    train_matrix = preprocessor.fit_transform(train.X).astype(np.float32)
    transformed: dict[str, np.ndarray] = {"train": train_matrix}
    for split_name in ("validation",):
        transformed[split_name] = preprocessor.transform(
            data.splits[split_name].X
        ).astype(np.float32)
    return preprocessor, transformed


def _make_loss(config: MLPConfig) -> nn.Module:
    if config.loss_name == "huber":
        return nn.SmoothL1Loss(beta=HUBER_DELTA)
    if config.loss_name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unsupported loss: {config.loss_name}")


def _validation_loss(
    model: MLPRegressor,
    features: Tensor,
    targets: Tensor,
    loss_function: nn.Module,
) -> float:
    model.eval()
    with torch.no_grad():
        return float(loss_function(model(features), targets).item())


def _fit_mlp(
    config: MLPConfig,
    seed: int,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    validation_types: np.ndarray,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    batch_size: int = BATCH_SIZE,
) -> MLPTrainingRun:
    """Fit one seeded model, stop on validation loss, and restore its best state."""
    _set_seed(seed)
    started = time.perf_counter()
    model = MLPRegressor(
        input_size=train_features.shape[1],
        hidden_sizes=config.hidden_sizes,
        dropout=config.dropout,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = _make_loss(config)
    x_train = torch.from_numpy(train_features)
    y_train = torch.from_numpy((train_targets / TARGET_SCALE).astype(np.float32))
    x_validation = torch.from_numpy(validation_features)
    y_validation = torch.from_numpy(
        (validation_targets / TARGET_SCALE).astype(np.float32)
    )
    train_dataset = TensorDataset(x_train, y_train)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    best_state: dict[str, Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        row_count = 0
        for batch_features, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_features), batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss_sum += float(loss.item()) * len(batch_targets)
            row_count += len(batch_targets)

        validation_loss = _validation_loss(
            model, x_validation, y_validation, loss_function
        )
        validation_predictions = model_predictions(model, validation_features)
        validation_score = regression_metrics(
            validation_targets,
            validation_predictions,
            validation_types,
        )["overall"]["mae"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss_sum / row_count,
                "validation_loss": validation_loss,
                "validation_mae": float(validation_score),
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("MLP training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    validation_predictions = model_predictions(model, validation_features)
    return MLPTrainingRun(
        config=config,
        seed=seed,
        state_dict=best_state,
        history=history,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        validation_metrics=regression_metrics(
            validation_targets, validation_predictions, validation_types
        ),
        duration_seconds=time.perf_counter() - started,
    )


def model_predictions(model: MLPRegressor, features: np.ndarray) -> np.ndarray:
    """Return raw unconstrained model predictions for an input matrix."""
    model.eval()
    with torch.no_grad():
        return (
            model(torch.from_numpy(features.astype(np.float32))).cpu().numpy()
            * TARGET_SCALE
        )


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _render_report(results: dict[str, Any], output_dir: Path) -> str:
    """Render factual experiment findings and representative loss curves."""
    lines = [
        "# Strict FFF neural regression experiment",
        "",
        "## Data and protocol",
        "",
        "Strict OULAD FFF features only. The prediction cutoff is the start of",
        "the assessment deadline day; the strict feature manifest excludes all",
        "prior assessment scores. Identifiers are audit-only. No demographic",
        "variables or future assessment outcomes enter the model.",
        "",
        f"Train, validation, and test rows: {results['shapes']['train'][0]:,}, "
        f"{results['shapes']['validation'][0]:,}, {results['shapes']['test'][0]:,}.",
        f"Model inputs: {results['shapes']['train'][1]} strict source columns; "
        f"{len(results['transformed_feature_names'])} transformed columns.",
        "The assessment type is reconstructed from the strict one-hot columns and",
        "encoded with a training-fitted OneHotEncoder (`handle_unknown='ignore'`).",
        "Numeric columns use training-median imputation (with missingness flags)",
        "and training-fitted standard scaling. The preprocessor is fit on train",
        "only; the saved preprocessing object is reused for validation, test, and",
        "inference. Predictions from all models are clipped to [0, 100] before",
        "metrics and saved prediction outputs.",
        "",
        "### Input feature list",
        "",
    ]
    lines.extend(f"- `{name}`" for name in results["input_feature_names"])
    lines.extend(
        [
            "",
            "## Baselines and selected model",
            "",
            f"Ridge alpha was selected from `{list(results['ridge_alpha_grid'])}` "
            "using validation MAE. The MLP search had two documented candidates:",
            "`(32,)` with no dropout and `(64, 32)` with 0.1 dropout. Both used",
            "Adam, learning rate 0.001, weight decay 0.0001, Huber loss, batches of",
            f"{results['training_config']['batch_size']}, and early stopping on",
            "validation Huber loss (patience 6, maximum 40 epochs). Three seeds",
            f"were run: `{results['seeds']}`. Selection minimized mean validation",
            "MAE across seeds; test data did not participate in selection.",
            "MLP targets are divided by 100 for optimization and predictions are",
            "rescaled to score units before clipping and metric calculation.",
            "",
            f"Selected MLP: `{results['selected_config']['name']}`, hidden sizes "
            f"`{results['selected_config']['hidden_sizes']}`, dropout "
            f"{results['selected_config']['dropout']}, Huber loss. Ridge alpha: "
            f"{results['selected_ridge_alpha']}.",
            "",
            "Selected MLP train metrics (mean ± standard deviation across seeds):",
            "",
            "| MAE | RMSE | R² |",
            "| ---: | ---: | ---: |",
            f"| {results['train_metrics']['mae_mean']:.3f} ± "
            f"{results['train_metrics']['mae_std']:.3f} | "
            f"{results['train_metrics']['rmse_mean']:.3f} ± "
            f"{results['train_metrics']['rmse_std']:.3f} | "
            f"{results['train_metrics']['r2_mean']:.4f} ± "
            f"{results['train_metrics']['r2_std']:.4f} |",
            "",
            "MLP candidate validation results (selection used mean MAE):",
            "",
            "| Candidate | Mean validation MAE | Std. dev. |",
            "| --- | ---: | ---: |",
        ]
    )
    for candidate in results["search_results"]:
        lines.append(
            f"| {candidate['config']['name']} | "
            f"{candidate['validation_mae_mean']:.3f} | "
            f"{candidate['validation_mae_std']:.3f} |"
        )
    lines.extend(
        [
            "",
            "| Model | Validation MAE | Validation RMSE | Validation R² |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name in ("Dummy mean", "Ridge", "MLP (selected config, seed mean)"):
        metrics = results["validation_metrics"][name]
        lines.append(
            f"| {name} | {metrics['mae_mean']:.3f} ± {metrics['mae_std']:.3f} | "
            f"{metrics['rmse_mean']:.3f} ± {metrics['rmse_std']:.3f} | "
            f"{metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Final test results",
            "",
            "Reported only after configuration and Ridge alpha selection on",
            "validation. MLP seed mean ± standard deviation is across the three",
            "selected-configuration seed models. The inference artifact averages",
            "those three seed predictions and is reported separately.",
            "",
            "| Model | Test MAE | Test RMSE | Test R² | n |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in ("Dummy mean", "Ridge", "MLP seed mean", "MLP seed ensemble"):
        metrics = results["test_metrics"][name]
        if name == "MLP seed mean":
            row = (
                f"| {name} | {metrics['mae_mean']:.3f} ± {metrics['mae_std']:.3f} | "
                f"{metrics['rmse_mean']:.3f} ± {metrics['rmse_std']:.3f} | "
                f"{metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f} | "
                f"{metrics['n']} |"
            )
        else:
            row = (
                f"| {name} | {metrics['mae']:.3f} | {metrics['rmse']:.3f} | "
                f"{metrics['r2']:.4f} | {metrics['n']} |"
            )
        lines.append(row)
    lines.extend(
        [
            "",
            "### Test metrics by assessment type and model",
            "",
            "| Model | Type | n | MAE | RMSE | R² |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for model_name in ("Dummy mean", "Ridge", "MLP seed ensemble"):
        for category, metrics in results["test_metrics"][model_name][
            "by_assessment_type"
        ].items():
            lines.append(
                f"| {model_name} | {category} | {metrics['n']} | "
                f"{metrics['mae']:.3f} | {metrics['rmse']:.3f} | "
                f"{metrics['r2']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Training curves and run time",
            "",
            "Checkpoint restoration minimizes validation Huber loss. Full epoch",
            "curves for all candidates and seeds are saved in `training_curves.csv`.",
            "Selected configuration checkpoints:",
            f"Total experiment runtime including preprocessing and evaluation: "
            f"{results['experiment_duration_seconds']:.2f} seconds; MLP search: "
            f"{results['mlp_search_duration_seconds']:.2f} seconds.",
            "",
            "| Seed | Epochs run | Best epoch | Best validation Huber loss | "
            "Duration (s) |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in results["selected_runs"]:
        lines.append(
            f"| {run['seed']} | {run['epochs_run']} | {run['best_epoch']} | "
            f"{run['best_validation_loss']:.5f} | {run['duration_seconds']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Representative selected-model training curve (seed 11):",
            "",
            "| Epoch | Train Huber loss | Validation Huber loss | Validation MAE |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    representative = next(
        run for run in results["selected_runs"] if run["seed"] == results["seeds"][0]
    )
    history = representative["history"]
    wanted = {0, 1, 2, 4, representative["best_epoch"] - 1, len(history) - 1}
    for index in sorted(wanted):
        if 0 <= index < len(history):
            point = history[index]
            lines.append(
                f"| {point['epoch']} | {point['train_loss']:.5f} | "
                f"{point['validation_loss']:.5f} | {point['validation_mae']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "OULAD engagement is observational: these metrics do not show that",
            "clicks cause marks. The dataset has day-level activity timestamps, no",
            "grade-release times, and target scores are retrospective labels.",
            "Prediction clipping is a reporting/deployment policy, not a monotonic",
            "constraint. No study sessions or causal effects are inferred.",
            "",
            "## Artifacts",
            "",
            f"All experiment files are under `{output_dir}`: fitted preprocessor, "
            "Dummy/Ridge estimators, selected-seed MLP state dictionaries, strict",
            "feature schema, configuration and versions, evaluation JSON, test",
            "predictions, and full training curves.",
            "",
        ]
    )
    return "\n".join(lines)


def run_training_experiment(
    processed_dir: Path,
    output_dir: Path,
    report_path: Path,
    seeds: tuple[int, ...] = TRAINING_SEEDS,
    search_configs: tuple[MLPConfig, ...] = DEFAULT_SEARCH_CONFIGS,
    max_epochs: int = MAX_EPOCHS,
) -> dict[str, Any]:
    """Fit preprocessing/baselines, select MLP on validation, and evaluate test."""
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Provide one or more unique training seeds")
    if not search_configs:
        raise ValueError("At least one MLP search configuration is required")
    if len({config.name for config in search_configs}) != len(search_configs):
        raise ValueError("MLP search configuration names must be unique")
    if max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    experiment_started = time.perf_counter()
    data = load_strict_feature_data(processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor, transformed = _prepare_arrays(data)
    splits = data.splits
    x_train, x_validation = transformed["train"], transformed["validation"]
    train = splits["train"]
    validation = splits["validation"]

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(x_train, train.y)
    ridge_candidates: list[tuple[float, Ridge, float]] = []
    for alpha in RIDGE_ALPHAS:
        ridge_candidate = Ridge(alpha=alpha)
        ridge_candidate.fit(x_train, train.y)
        validation_pred = ridge_candidate.predict(x_validation)
        mae = float(
            regression_metrics(
                validation.y, validation_pred, validation.assessment_types
            )["overall"]["mae"]
        )
        ridge_candidates.append((mae, ridge_candidate, alpha))
    _, ridge, selected_ridge_alpha = min(
        ridge_candidates, key=lambda candidate: (candidate[0], candidate[2])
    )

    search_started = time.perf_counter()
    runs_by_config: dict[str, list[MLPTrainingRun]] = {}
    curves: list[dict[str, Any]] = []
    search_summary: list[dict[str, Any]] = []
    for config in search_configs:
        config_runs: list[MLPTrainingRun] = []
        for seed in seeds:
            run = _fit_mlp(
                config,
                seed,
                x_train,
                train.y,
                x_validation,
                validation.y,
                validation.assessment_types,
                max_epochs=max_epochs,
            )
            config_runs.append(run)
            curves.extend(
                {"config": config.name, "seed": seed, **point} for point in run.history
            )
        runs_by_config[config.name] = config_runs
        validation_maes = [
            float(run.validation_metrics["overall"]["mae"]) for run in config_runs
        ]
        search_summary.append(
            {
                "config": config.to_dict(),
                "validation_mae_mean": statistics.fmean(validation_maes),
                "validation_mae_std": (
                    statistics.stdev(validation_maes)
                    if len(validation_maes) > 1
                    else 0.0
                ),
                "validation_mae_by_seed": validation_maes,
                "best_epochs": [run.best_epoch for run in config_runs],
            }
        )
    selected_summary = min(
        search_summary,
        key=lambda item: (item["validation_mae_mean"], item["config"]["name"]),
    )
    selected_config = next(
        config
        for config in search_configs
        if config.name == selected_summary["config"]["name"]
    )
    selected_runs = runs_by_config[selected_config.name]
    search_duration = time.perf_counter() - search_started

    # Test rows are transformed and evaluated only after validation-based selection.
    test = splits["test"]
    x_test = preprocessor.transform(test.X).astype(np.float32)
    baseline_predictions: dict[str, np.ndarray] = {
        "Dummy mean": dummy.predict(x_test),
        "Ridge": ridge.predict(x_test),
    }
    selected_models: list[MLPRegressor] = []
    seed_test_metrics: list[dict[str, Any]] = []
    seed_validation_metrics: list[dict[str, Any]] = []
    seed_train_metrics: list[dict[str, Any]] = []
    for run in selected_runs:
        model = MLPRegressor(
            input_size=x_train.shape[1],
            hidden_sizes=selected_config.hidden_sizes,
            dropout=selected_config.dropout,
        )
        model.load_state_dict(run.state_dict)
        selected_models.append(model)
        seed_train_metrics.append(
            regression_metrics(
                train.y, model_predictions(model, x_train), train.assessment_types
            )
        )
        seed_validation_metrics.append(run.validation_metrics)
        seed_test_metrics.append(
            regression_metrics(
                test.y, model_predictions(model, x_test), test.assessment_types
            )
        )
    ensemble_raw = np.mean(
        [model_predictions(model, x_test) for model in selected_models], axis=0
    )
    baseline_test_metrics = {
        name: metric_json_safe(
            regression_metrics(test.y, predictions, test.assessment_types)
        )
        for name, predictions in baseline_predictions.items()
    }
    ensemble_test_metrics = metric_json_safe(
        regression_metrics(test.y, ensemble_raw, test.assessment_types)
    )
    seed_test_overall = [item["overall"] for item in seed_test_metrics]
    test_metrics: dict[str, Any] = {
        **baseline_test_metrics,
        "MLP seed mean": {
            "n": len(test.y),
            **{
                key: value
                for metric in ("mae", "rmse", "r2")
                for key, value in (
                    (
                        f"{metric}_mean",
                        statistics.fmean(
                            [float(item[metric]) for item in seed_test_overall]
                        ),
                    ),
                    (
                        f"{metric}_std",
                        _mean_std([float(item[metric]) for item in seed_test_overall])[
                            "std"
                        ],
                    ),
                )
            },
        },
        "MLP seed ensemble": ensemble_test_metrics,
    }

    validation_metrics: dict[str, Any] = {}
    for name, estimator in (("Dummy mean", dummy), ("Ridge", ridge)):
        metrics_by_seed = regression_metrics(
            validation.y,
            estimator.predict(x_validation),
            validation.assessment_types,
        )["overall"]
        validation_metrics[name] = {
            f"{metric}_mean": metrics_by_seed[metric]
            for metric in ("mae", "rmse", "r2")
        } | {f"{metric}_std": 0.0 for metric in ("mae", "rmse", "r2")}
    for name, metric in (("MLP (selected config, seed mean)", "overall"),):
        run_metrics = [run.validation_metrics[metric] for run in selected_runs]
        validation_metrics[name] = {
            "mae_mean": statistics.fmean(float(item["mae"]) for item in run_metrics),
            "mae_std": _mean_std([float(item["mae"]) for item in run_metrics])["std"],
            "rmse_mean": statistics.fmean(float(item["rmse"]) for item in run_metrics),
            "rmse_std": _mean_std([float(item["rmse"]) for item in run_metrics])["std"],
            "r2_mean": statistics.fmean(float(item["r2"]) for item in run_metrics),
            "r2_std": _mean_std([float(item["r2"]) for item in run_metrics])["std"],
        }
    train_run_metrics = [item["overall"] for item in seed_train_metrics]
    train_metrics = {
        f"{metric}_{suffix}": (
            statistics.fmean(float(item[metric]) for item in train_run_metrics)
            if suffix == "mean"
            else _mean_std([float(item[metric]) for item in train_run_metrics])["std"]
        )
        for metric in ("mae", "rmse", "r2")
        for suffix in ("mean", "std")
    }
    full_duration = time.perf_counter() - experiment_started

    evaluation = {
        "baseline_metrics": baseline_test_metrics,
        "train_metrics": {"MLP (selected config, seed mean)": train_metrics},
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "selected_config": selected_config.to_dict(),
        "search_results": search_summary,
        "selected_ridge_alpha": selected_ridge_alpha,
        "seeds": list(seeds),
        "training_duration_seconds": {
            "mlp_search_total": search_duration,
            "full_experiment_total": full_duration,
            "selected_runs": [
                {
                    "seed": run.seed,
                    "duration_seconds": run.duration_seconds,
                    "epochs_run": len(run.history),
                    "best_epoch": run.best_epoch,
                    "best_validation_loss": run.best_validation_loss,
                    "train_metrics": metric_json_safe(train_metric),
                    "validation_metrics": metric_json_safe(validation_metric),
                    "test_metrics": metric_json_safe(test_metric),
                }
                for run, train_metric, validation_metric, test_metric in zip(
                    selected_runs,
                    seed_train_metrics,
                    seed_validation_metrics,
                    seed_test_metrics,
                    strict=True,
                )
            ],
        },
    }

    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    joblib.dump(dummy, output_dir / "dummy_mean.joblib")
    joblib.dump(ridge, output_dir / "ridge.joblib")
    model_files: list[str] = []
    selected_run_payloads: list[dict[str, Any]] = []
    for run in selected_runs:
        filename = f"mlp_seed_{run.seed}.pt"
        torch.save(run.state_dict, output_dir / filename)
        model_files.append(filename)
        selected_run_payloads.append(
            {
                "seed": run.seed,
                "epochs_run": len(run.history),
                "best_epoch": run.best_epoch,
                "best_validation_loss": run.best_validation_loss,
                "duration_seconds": run.duration_seconds,
                "history": run.history,
            }
        )

    _json_write(
        output_dir / "feature_schema.json",
        {
            "mode": "strict",
            "source_feature_columns": list(data.feature_names),
            "numeric_feature_columns": list(data.numeric_feature_names),
            "assessment_type_indicator_columns": list(data.assessment_type_columns),
            "model_categorical_columns": ["assessment_type"],
            "model_input_columns": ["assessment_type", *data.numeric_feature_names],
            "transformed_feature_names": list(preprocessor.get_feature_names_out()),
            "identifiers_excluded": list(data.manifest["identifiers_not_features"]),
            "target": data.manifest["target"],
            "prediction_bounds": [PREDICTION_MIN, PREDICTION_MAX],
            "cutoff": data.manifest["prediction_cutoff"],
        },
    )
    experiment_config = {
        "feature_mode": "strict",
        "selected_mlp_config": selected_config.to_dict(),
        "search_configs": [config.to_dict() for config in search_configs],
        "selected_ridge_alpha": selected_ridge_alpha,
        "ridge_alpha_grid": list(RIDGE_ALPHAS),
        "seed_list": list(seeds),
        "selection_metric": "mean validation MAE across configured seeds",
        "early_stopping_metric": "validation Huber loss",
        "loss": selected_config.loss_name,
        "target_scale_for_optimization": TARGET_SCALE,
        "batch_size": BATCH_SIZE,
        "max_epochs": max_epochs,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "device": "cpu",
        "prediction_policy": (
            "clip raw predictions to [0, 100] before metrics/inference"
        ),
        "model_files": model_files,
        "preprocessed_feature_count": int(x_train.shape[1]),
        "versions": {
            "python": __import__("platform").python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "joblib": joblib.__version__,
        },
    }
    _json_write(output_dir / "experiment_config.json", experiment_config)
    _json_write(output_dir / "evaluation.json", metric_json_safe(evaluation))
    with (output_dir / "training_curves.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "config",
                "seed",
                "epoch",
                "train_loss",
                "validation_loss",
                "validation_mae",
            ),
        )
        writer.writeheader()
        writer.writerows(curves)

    prediction_frame = pd.DataFrame(
        {
            **{
                column: test.identifiers[column].astype(str).to_numpy()
                for column in test.identifiers.columns
            },
            "assessment_type": test.assessment_types,
            "target_score": test.y,
            "dummy_prediction": clip_predictions(baseline_predictions["Dummy mean"]),
            "ridge_prediction": clip_predictions(baseline_predictions["Ridge"]),
            "mlp_ensemble_prediction": clip_predictions(ensemble_raw),
        }
    )
    prediction_frame.to_csv(output_dir / "test_predictions.csv", index=False)

    report_results = {
        "shapes": {
            name: (len(split.y), len(data.feature_names))
            for name, split in splits.items()
        },
        "input_feature_names": list(data.feature_names),
        "transformed_feature_names": list(preprocessor.get_feature_names_out()),
        "ridge_alpha_grid": RIDGE_ALPHAS,
        "training_config": {"batch_size": BATCH_SIZE},
        "seeds": list(seeds),
        "selected_config": selected_config.to_dict(),
        "selected_ridge_alpha": selected_ridge_alpha,
        "validation_metrics": validation_metrics,
        "search_results": search_summary,
        "test_metrics": test_metrics,
        "selected_runs": selected_run_payloads,
        "train_metrics": train_metrics,
        "experiment_duration_seconds": full_duration,
        "mlp_search_duration_seconds": search_duration,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(report_results, output_dir), encoding="utf-8")
    LOGGER.info("Selected MLP configuration: %s", selected_config.name)
    LOGGER.info("Wrote model artifacts under %s", output_dir)
    LOGGER.info("Wrote training report to %s", report_path)
    return evaluation


def main(argv: list[str] | None = None) -> int:
    """CLI for strict-feature baseline/MLP training and evaluation."""
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir", type=Path, default=root / "data" / "processed"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "data" / "processed" / "nn_experiment"
    )
    parser.add_argument(
        "--report", type=Path, default=root / "reports" / "nn_training.md"
    )
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(TRAINING_SEEDS))
    args = parser.parse_args(argv)
    if args.max_epochs < 1:
        parser.error("--max-epochs must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run_training_experiment(
            args.processed_dir,
            args.output_dir,
            args.report,
            tuple(args.seeds),
            max_epochs=args.max_epochs,
        )
    except (OSError, ValueError, RuntimeError) as error:
        LOGGER.error("Training failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
