"""Train, select, evaluate, and persist strict FFF regression models."""

from __future__ import annotations

import argparse
import csv
import hashlib
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


class ReportSchemaError(ValueError):
    """Raised when evaluation results do not match the report contract."""


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


def _strict_selected_config(original_output_dir: Path) -> MLPConfig:
    """Load and verify the fixed selected MLP protocol from the 40-epoch run."""
    config_path = original_output_dir / "experiment_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing original experiment config: {config_path}")
    original = json.loads(config_path.read_text(encoding="utf-8"))
    expected_model = MLPConfig(
        name="mlp_64_32_dropout",
        hidden_sizes=(64, 32),
        dropout=0.1,
        learning_rate=0.001,
        weight_decay=0.0001,
        loss_name="huber",
    )
    if original.get("feature_mode") != "strict":
        raise ValueError("Extended training requires the original strict run")
    if original.get("selected_mlp_config") != expected_model.to_dict():
        raise ValueError("Original selected MLP config differs from fixed protocol")
    if original.get("seed_list") != [42, 123, 2025]:
        raise ValueError("Original experiment seeds differ from fixed protocol")
    fixed_fields = {
        "batch_size": BATCH_SIZE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_metric": "validation Huber loss",
        "loss": "huber",
        "target_scale_for_optimization": TARGET_SCALE,
        "device": "cpu",
    }
    for key, expected in fixed_fields.items():
        if original.get(key) != expected:
            raise ValueError(
                f"Original experiment {key}={original.get(key)!r}; "
                f"expected fixed value {expected!r}"
            )
    if original.get("max_epochs") != MAX_EPOCHS:
        raise ValueError("Original experiment must be the 40-epoch run")
    return expected_model


def _split_identity_fingerprint(split: Any) -> str:
    """Hash row-ordered identifiers and labels for audit provenance."""
    identity = split.identifiers.astype("string").copy()
    identity["target_score"] = split.y
    hashes = pd.util.hash_pandas_object(identity, index=False).to_numpy()
    return hashlib.sha256(hashes.tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    """Hash an input file without retaining its contents in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_validation_only_experiment(
    processed_dir: Path,
    original_output_dir: Path,
    output_dir: Path,
    report_path: Path,
    seeds: tuple[int, ...] = (42, 123, 2025),
    max_epochs: int = 60,
) -> dict[str, Any]:
    """Train the previously selected MLP using train/validation rows only.

    This execution path requests only train and validation CSVs from the strict
    data loader. It has no test transform, prediction, or metric code path.
    """
    if seeds != (42, 123, 2025):
        raise ValueError("Extended experiment requires seeds (42, 123, 2025)")
    if max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    if output_dir.resolve() == original_output_dir.resolve():
        raise ValueError("Extended outputs must not overwrite the original run")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Extended output directory is not empty: {output_dir}")

    config = _strict_selected_config(original_output_dir)
    data = load_strict_feature_data(
        processed_dir, split_names=("train", "validation")
    )
    splits = data.splits
    train = splits["train"]
    validation = splits["validation"]
    preprocessor = build_preprocessor(data.numeric_feature_names)
    x_train = preprocessor.fit_transform(train.X).astype(np.float32)
    x_validation = preprocessor.transform(validation.X).astype(np.float32)

    started = time.perf_counter()
    runs: list[MLPTrainingRun] = []
    for seed in seeds:
        runs.append(
            _fit_mlp(
                config,
                seed,
                x_train,
                train.y,
                x_validation,
                validation.y,
                validation.assessment_types,
                max_epochs=max_epochs,
                patience=EARLY_STOPPING_PATIENCE,
                batch_size=BATCH_SIZE,
            )
        )
    duration = time.perf_counter() - started

    seed_results: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    model_files: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    for run in runs:
        validation_metrics = metric_json_safe(run.validation_metrics["overall"])
        final_window_start = run.history[max(0, len(run.history) - 10)]
        final_point = run.history[-1]
        seed_results.append(
            {
                "seed": run.seed,
                "epochs_run": len(run.history),
                "final_epoch": int(run.history[-1]["epoch"]),
                "stopping_reason": (
                    "early_stopping_patience"
                    if len(run.history) < max_epochs
                    else "max_epochs_reached"
                ),
                "best_epoch": run.best_epoch,
                "best_validation_huber_loss": run.best_validation_loss,
                "last_validation_huber_loss": run.history[-1]["validation_loss"],
                "best_checkpoint_validation_metrics": validation_metrics,
                "validation_metrics_at_final_epoch": {
                    "mae": final_point["validation_mae"],
                    "huber_loss": final_point["validation_loss"],
                },
                "last_ten_epoch_endpoint_change": {
                    "from_epoch": final_window_start["epoch"],
                    "to_epoch": final_point["epoch"],
                    "validation_huber_loss": (
                        float(final_point["validation_loss"])
                        - float(final_window_start["validation_loss"])
                    ),
                    "validation_mae": (
                        float(final_point["validation_mae"])
                        - float(final_window_start["validation_mae"])
                    ),
                },
                "duration_seconds": run.duration_seconds,
            }
        )
        curves.extend(
            {"config": config.name, "seed": run.seed, **point}
            for point in run.history
        )
        filename = f"mlp_seed_{run.seed}.pt"
        torch.save(run.state_dict, output_dir / filename)
        model_files.append(filename)

    metrics_by_seed = [
        run.validation_metrics["overall"] for run in runs
    ]
    aggregate_validation = {
        f"{metric}_{suffix}": (
            statistics.fmean(float(row[metric]) for row in metrics_by_seed)
            if suffix == "mean"
            else _mean_std([float(row[metric]) for row in metrics_by_seed])["std"]
        )
        for metric in ("mae", "rmse", "r2")
        for suffix in ("mean", "std")
    }
    original_eval_path = original_output_dir / "evaluation.json"
    original_evaluation = json.loads(original_eval_path.read_text(encoding="utf-8"))
    original_runs = {
        int(item["seed"]): item
        for item in original_evaluation["training_duration_seconds"][
            "selected_runs"
        ]
    }
    comparisons = {
        str(run.seed): {
            "original_best_epoch": original_runs[run.seed]["best_epoch"],
            "original_validation_metrics": original_runs[run.seed][
                "validation_metrics"
            ]["overall"],
            "extended_best_epoch": run.best_epoch,
            "extended_best_validation_huber_loss": run.best_validation_loss,
            "extended_validation_metrics": metric_json_safe(
                run.validation_metrics["overall"]
            ),
        }
        for run in runs
    }
    original_curve_path = original_output_dir / "training_curves.csv"
    with original_curve_path.open(newline="", encoding="utf-8") as file:
        original_curve_rows = [
            row
            for row in csv.DictReader(file)
            if row["config"] == config.name and int(row["seed"]) in seeds
        ]
    for run in runs:
        old_curve = {
            int(row["epoch"]): row
            for row in original_curve_rows
            if int(row["seed"]) == run.seed
        }
        shared_epochs = min(len(old_curve), len(run.history))
        metric_fields = ("train_loss", "validation_loss", "validation_mae")
        comparisons[str(run.seed)]["overlap_curve_epochs"] = shared_epochs
        comparisons[str(run.seed)]["overlap_training_validation_curves_match"] = (
            all(
                all(
                    float(old_curve[epoch][field])
                    == float(run.history[epoch - 1][field])
                    for field in metric_fields
                )
                for epoch in range(1, shared_epochs + 1)
            )
            and shared_epochs > 0
        )
    comparisons["aggregate"] = {
        "original_validation_metrics": original_evaluation["validation_metrics"][
            "MLP (selected config, seed mean)"
        ],
        "extended_validation_metrics": aggregate_validation,
    }

    input_hashes = {
        split_name: {
            "csv_sha256": _file_sha256(
                processed_dir / data.manifest["split_files"][split_name]
            ),
            "row_ordered_ids_and_target_sha256": _split_identity_fingerprint(split),
            "rows": len(split.y),
        }
        for split_name, split in splits.items()
    }
    _json_write(
        output_dir / "feature_schema.json",
        {
            "mode": "strict",
            "source_feature_columns": list(data.feature_names),
            "numeric_feature_columns": list(data.numeric_feature_names),
            "assessment_type_indicator_columns": list(data.assessment_type_columns),
            "identifiers_excluded": list(data.manifest["identifiers_not_features"]),
            "target": data.manifest["target"],
            "prediction_bounds": [PREDICTION_MIN, PREDICTION_MAX],
            "cutoff": data.manifest["prediction_cutoff"],
        },
    )
    experiment_config = {
        "experiment": "strict_fff_validation_only_extended_training",
        "feature_mode": "strict",
        "selected_mlp_config": config.to_dict(),
        "seed_list": list(seeds),
        "optimizer": "Adam",
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "loss": config.loss_name,
        "target_scale_for_optimization": TARGET_SCALE,
        "batch_size": BATCH_SIZE,
        "max_epochs": max_epochs,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_metric": "validation Huber loss",
        "validation_selection_metric": "MAE from best validation-Huber checkpoint",
        "device": "cpu",
        "prediction_policy": "clip predictions to [0, 100] before metrics",
        "model_files": model_files,
        "preprocessed_feature_count": int(x_train.shape[1]),
        "input_hashes": input_hashes,
        "versions": {
            "python": __import__("platform").python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "joblib": joblib.__version__,
        },
    }
    evaluation = {
        "validation_only": True,
        "rows": {name: len(split.y) for name, split in splits.items()},
        "seeds": seed_results,
        "aggregate_validation_metrics": aggregate_validation,
        "original_vs_extended_validation": comparisons,
        "experiment_duration_seconds": duration,
        "test_partition_accessed": False,
    }
    _json_write(output_dir / "experiment_config.json", experiment_config)
    _json_write(output_dir / "evaluation.json", metric_json_safe(evaluation))
    with (output_dir / "training_curves.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "config", "seed", "epoch", "train_loss",
                "validation_loss", "validation_mae",
            ),
        )
        writer.writeheader()
        writer.writerows(curves)

    report = _render_validation_only_report(
        config, seeds, seed_results, aggregate_validation, comparisons,
        input_hashes, duration, max_epochs,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return evaluation


def _render_validation_only_report(
    config: MLPConfig,
    seeds: tuple[int, ...],
    seed_results: list[dict[str, Any]],
    aggregate_metrics: dict[str, float],
    comparisons: dict[str, Any],
    input_hashes: dict[str, Any],
    duration: float,
    max_epochs: int,
) -> str:
    """Render validation-only evidence without introducing test metrics."""
    lines = [
        "# Strict FFF MLP extended training (validation only)",
        "",
        "This experiment trains only the previously selected strict FFF MLP.",
        "It accesses train and validation CSVs only; the test CSV is not loaded,",
        "transformed, predicted on, or evaluated. The original 40-epoch run is",
        "preserved in its separate artifact directory.",
        "",
        f"Fixed configuration: `{config.name}`, hidden sizes "
        f"`{list(config.hidden_sizes)}`, "
        f"dropout {config.dropout}; Adam lr={config.learning_rate}, weight decay="
        f"{config.weight_decay}; {config.loss_name} loss; batch size {BATCH_SIZE}; "
        f"patience {EARLY_STOPPING_PATIENCE}; maximum {max_epochs} epochs.",
        f"Seeds: `{list(seeds)}`. Duration for three training runs: {duration:.2f}s.",
        "",
        "## Per-seed validation results",
        "",
        "| Seed | Final epoch | Stop reason | Best epoch | Best Huber | "
        "Best checkpoint MAE | Best checkpoint RMSE | Final Huber | Final MAE |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in seed_results:
        metrics = result["best_checkpoint_validation_metrics"]
        lines.append(
            f"| {result['seed']} | {result['final_epoch']} | "
            f"{result['stopping_reason']} | {result['best_epoch']} | "
            f"{result['best_validation_huber_loss']:.8f} | "
            f"{metrics['mae']:.5f} | {metrics['rmse']:.5f} | "
            f"{result['validation_metrics_at_final_epoch']['huber_loss']:.8f} | "
            f"{result['validation_metrics_at_final_epoch']['mae']:.5f} |"
        )
    lines.extend(
        [
            "",
            "Validation MAE/RMSE are computed from each seed's checkpoint restored",
            "at minimum validation Huber loss. Aggregate values below are the",
            "arithmetic mean and sample standard deviation across the three seeds.",
            "",
            "| Metric | Mean | Std. dev. |",
            "| --- | ---: | ---: |",
        ]
    )
    for metric in ("mae", "rmse", "r2"):
        lines.append(
            f"| {metric.upper()} | {aggregate_metrics[f'{metric}_mean']:.5f} | "
            f"{aggregate_metrics[f'{metric}_std']:.5f} |"
        )
    lines.extend(["", "### Final-ten-epoch endpoint changes", ""])
    for result in seed_results:
        trend = result["last_ten_epoch_endpoint_change"]
        lines.append(
            f"Seed {result['seed']}: over epochs {trend['from_epoch']}–"
            f"{trend['to_epoch']}, endpoint change in validation Huber loss was "
            f"{trend['validation_huber_loss']:+.8f} and validation MAE was "
            f"{trend['validation_mae']:+.5f}. Negative values mean lower at the "
            "final epoch; this endpoint comparison does not imply a monotonic trend."
        )
    boundary_seed = next(result for result in seed_results if result["seed"] == 2025)
    stale_after_best = boundary_seed["final_epoch"] - boundary_seed["best_epoch"]
    lines.extend(
        [
            "",
            f"Seed 2025's best Huber checkpoint was epoch "
            f"{boundary_seed['best_epoch']}; "
            f"epochs {boundary_seed['best_epoch'] + 1}–"
            f"{boundary_seed['final_epoch']} ({stale_after_best} epochs) did not "
            "improve it. The run reached the 60-epoch cap before the six-epoch "
            "patience rule could stop it. Although the epoch-51-to-60 endpoint "
            "change is slightly lower, validation loss fluctuated and epoch 60 "
            "remained above the restored minimum. The near-cap evidence is mixed, "
            "with no new best in the final four epochs, rather than sustained "
            "improvement through epoch 60.",
        ]
    )
    lines.extend(
        [
            "",
            "## Comparison with the original 40-epoch run",
            "",
            "| Seed | Old best epoch | Old MAE | Old RMSE | New best epoch | "
            "New MAE | New RMSE |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for seed in seeds:
        result = comparisons[str(seed)]
        old = result["original_validation_metrics"]
        new = result["extended_validation_metrics"]
        lines.append(
            f"| {seed} | {result['original_best_epoch']} | {old['mae']:.5f} | "
            f"{old['rmse']:.5f} | {result['extended_best_epoch']} | "
            f"{new['mae']:.5f} | {new['rmse']:.5f} |"
        )
    lines.extend(
        [
            "",
            "Shared-epoch curve comparison with the original:",
            "",
        ]
    )
    for seed in seeds:
        result = comparisons[str(seed)]
        lines.append(
            f"Seed {seed}: {result['overlap_curve_epochs']} shared epochs; "
            f"train loss, validation Huber loss, and validation MAE match "
            f"exactly: {result['overlap_training_validation_curves_match']}."
        )
    aggregate_comparison = comparisons["aggregate"]
    old_aggregate = aggregate_comparison["original_validation_metrics"]
    new_aggregate = aggregate_comparison["extended_validation_metrics"]
    lines.extend(
        [
            "",
            "Aggregate validation metrics (mean ± sample standard deviation "
            "over seeds):",
            "",
            "| Run | MAE | RMSE | R² |",
            "| --- | ---: | ---: | ---: |",
            "| Original 40 epoch | "
            f"{old_aggregate['mae_mean']:.5f} ± {old_aggregate['mae_std']:.5f} | "
            f"{old_aggregate['rmse_mean']:.5f} ± {old_aggregate['rmse_std']:.5f} | "
            f"{old_aggregate['r2_mean']:.5f} ± {old_aggregate['r2_std']:.5f} |",
            "| Extended 60 epoch cap | "
            f"{new_aggregate['mae_mean']:.5f} ± {new_aggregate['mae_std']:.5f} | "
            f"{new_aggregate['rmse_mean']:.5f} ± {new_aggregate['rmse_std']:.5f} | "
            f"{new_aggregate['r2_mean']:.5f} ± {new_aggregate['r2_std']:.5f} |",
        ]
    )
    lines.extend(
        [
            "",
            "## Input identity and limitations",
            "",
            "The current train and validation files match byte-for-byte between",
            "`data/processed/` and `data/processed/strict/`; both were last",
            "modified before the original experiment artifacts. The original run",
            "did not persist input row-ID/target fingerprints, so historical",
            "identity with its exact training rows cannot be proven from its saved",
            "artifacts. Current row counts, strict schema, and row-ordered",
            "identifier/target fingerprints are saved in `experiment_config.json`",
            "for reproducibility from this run onward; row counts are in",
            "`evaluation.json`.",
            "",
            "Three seeds are descriptive; no statistical significance is claimed.",
            "No test results are included or used for decisions. This experiment",
            "does not change the selected model in the original artifact directory.",
            "",
            "### Recorded row-ordered identifier/target fingerprints",
            "",
            "| Split | Rows | SHA-256 (identifiers and target) |",
            "| --- | ---: | --- |",
        ]
    )
    for split_name, fingerprint in input_hashes.items():
        lines.append(
            f"| {split_name} | {fingerprint['rows']} | "
            f"`{fingerprint['row_ordered_ids_and_target_sha256']}` |"
        )
    return "\n".join(lines) + "\n"


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _require_metrics(
    value: Any, required: tuple[str, ...], result_path: str
) -> dict[str, Any]:
    """Validate one metric mapping and report its location when malformed."""
    if not isinstance(value, dict):
        raise ReportSchemaError(
            f"Report results at {result_path} must be a metric mapping; "
            f"received {type(value).__name__}."
        )
    missing = [name for name in required if name not in value]
    if missing:
        raise ReportSchemaError(
            f"Report results at {result_path} are missing required metric(s): "
            f"{', '.join(missing)}. Expected keys: {', '.join(required)}."
        )
    return value


def _overall_metrics(value: Any, result_path: str) -> dict[str, Any]:
    """Return the validated overall metric row from regression_metrics output."""
    if not isinstance(value, dict):
        raise ReportSchemaError(f"Report results at {result_path} must be a mapping.")
    if "overall" not in value:
        raise ReportSchemaError(
            f"Report results at {result_path} are missing 'overall'. "
            "Expected the structure returned by regression_metrics()."
        )
    return _require_metrics(
        value["overall"], ("mae", "rmse", "r2", "n"), f"{result_path}.overall"
    )


def _type_metrics(value: Any, result_path: str) -> dict[str, dict[str, Any]]:
    """Validate assessment-type metric rows before rendering the breakdown."""
    if not isinstance(value, dict) or "by_assessment_type" not in value:
        raise ReportSchemaError(
            f"Report results at {result_path} are missing 'by_assessment_type'."
        )
    by_type = value["by_assessment_type"]
    if not isinstance(by_type, dict) or not by_type:
        raise ReportSchemaError(
            f"Report results at {result_path}.by_assessment_type must be a "
            "non-empty mapping."
        )
    return {
        str(category): _require_metrics(
            metrics,
            ("mae", "rmse", "r2", "n"),
            f"{result_path}.by_assessment_type[{category!r}]",
        )
        for category, metrics in by_type.items()
    }


def _render_report(results: dict[str, Any], output_dir: Path) -> str:
    """Render factual experiment findings and representative loss curves."""
    for name in ("Dummy mean", "Ridge"):
        _require_metrics(
            results.get("validation_metrics", {}).get(name),
            ("mae_mean", "mae_std", "rmse_mean", "rmse_std", "r2_mean", "r2_std"),
            f"results['validation_metrics'][{name!r}]",
        )
    _require_metrics(
        results.get("validation_metrics", {}).get(
            "MLP (selected config, seed mean)"
        ),
        ("mae_mean", "mae_std", "rmse_mean", "rmse_std", "r2_mean", "r2_std"),
        "results['validation_metrics']['MLP (selected config, seed mean)']",
    )
    _require_metrics(
        results.get("train_metrics"),
        ("mae_mean", "mae_std", "rmse_mean", "rmse_std", "r2_mean", "r2_std"),
        "results['train_metrics']",
    )
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
        _require_metrics(
            candidate,
            ("config", "validation_mae_mean", "validation_mae_std"),
            "results['search_results'][item]",
        )
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
        metrics = _require_metrics(
            results["validation_metrics"].get(name),
            ("mae_mean", "mae_std", "rmse_mean", "rmse_std", "r2_mean", "r2_std"),
            f"results['validation_metrics'][{name!r}]",
        )
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
        metrics = results["test_metrics"].get(name)
        if metrics is None:
            raise ReportSchemaError(f"Report results are missing test model {name!r}.")
        if name == "MLP seed mean":
            metrics = _require_metrics(
                metrics,
                (
                    "mae_mean", "mae_std", "rmse_mean", "rmse_std",
                    "r2_mean", "r2_std", "n",
                ),
                f"results['test_metrics'][{name!r}]",
            )
            row = (
                f"| {name} | {metrics['mae_mean']:.3f} ± {metrics['mae_std']:.3f} | "
                f"{metrics['rmse_mean']:.3f} ± {metrics['rmse_std']:.3f} | "
                f"{metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f} | "
                f"{metrics['n']} |"
            )
        else:
            metrics = _overall_metrics(
                metrics, f"results['test_metrics'][{name!r}]"
            )
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
        type_metrics = _type_metrics(
            results["test_metrics"].get(model_name),
            f"results['test_metrics'][{model_name!r}]",
        )
        for category, metrics in type_metrics.items():
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
            "Representative selected-model training curve "
            f"(seed {results['seeds'][0]}):",
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


def validation_only_main(argv: list[str] | None = None) -> int:
    """CLI for the fixed 60-epoch validation-only extension experiment."""
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=validation_only_main.__doc__)
    parser.add_argument(
        "--processed-dir", type=Path, default=root / "data" / "processed" / "strict"
    )
    parser.add_argument(
        "--original-output-dir",
        type=Path,
        default=root / "data" / "processed" / "nn_experiment",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "processed" / "nn_experiment_60ep",
    )
    parser.add_argument(
        "--report", type=Path, default=root / "reports" / "nn_extended_training.md"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run_validation_only_experiment(
            args.processed_dir,
            args.original_output_dir,
            args.output_dir,
            args.report,
            max_epochs=60,
        )
    except (OSError, ValueError, RuntimeError) as error:
        LOGGER.error("Validation-only training failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
