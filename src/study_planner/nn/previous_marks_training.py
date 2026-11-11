"""Train and compare the separate submission-proxy Previous Marks MLP."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch

from study_planner.nn.dataset import (
    FeatureSplit,
    StrictFeatureData,
    load_previous_marks_feature_data,
    load_strict_feature_data,
)
from study_planner.nn.inference import RegressionPredictor
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
from study_planner.nn.training import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    MAX_EPOCHS,
    MLPTrainingRun,
    _fit_mlp,
    model_predictions,
)

LOGGER = logging.getLogger(__name__)
MODEL_B_SEEDS = (42, 123, 2025)
IDENTITY_COLUMNS = (
    "code_module",
    "code_presentation",
    "id_student",
    "id_assessment",
)


def _row_key(frame: pd.DataFrame) -> list[tuple[str, ...]]:
    return list(
        frame.loc[:, list(IDENTITY_COLUMNS)]
        .astype("string")
        .itertuples(index=False, name=None)
    )


def _paired_split(
    strict: FeatureSplit, proxy: FeatureSplit, split_name: str
) -> tuple[FeatureSplit, FeatureSplit]:
    """Align both models on the same identifier/target rows, or fail closed."""
    strict_keys = _row_key(strict.identifiers)
    proxy_keys = _row_key(proxy.identifiers)
    if len(strict_keys) != len(set(strict_keys)) or len(proxy_keys) != len(
        set(proxy_keys)
    ):
        raise ValueError(f"Duplicate assessment identity in {split_name} split")
    proxy_positions = {key: index for index, key in enumerate(proxy_keys)}
    common_strict_positions = [
        index for index, key in enumerate(strict_keys) if key in proxy_positions
    ]
    if not common_strict_positions:
        raise ValueError(f"No paired rows in {split_name} split")
    strict_positions = np.asarray(common_strict_positions, dtype=int)
    proxy_positions_array = np.asarray(
        [proxy_positions[strict_keys[index]] for index in common_strict_positions],
        dtype=int,
    )
    strict_y = strict.y[strict_positions]
    proxy_y = proxy.y[proxy_positions_array]
    if not np.array_equal(strict_y, proxy_y):
        raise ValueError(f"Targets differ for paired rows in {split_name} split")
    strict_types = strict.assessment_types[strict_positions]
    proxy_types = proxy.assessment_types[proxy_positions_array]
    if not np.array_equal(strict_types, proxy_types):
        raise ValueError(
            f"Assessment types differ for paired rows in {split_name} split"
        )

    def subset(split: FeatureSplit, positions: np.ndarray) -> FeatureSplit:
        return FeatureSplit(
            X=split.X.iloc[positions].reset_index(drop=True),
            y=split.y[positions].copy(),
            assessment_types=split.assessment_types[positions].copy(),
            identifiers=split.identifiers.iloc[positions].reset_index(drop=True),
        )

    return subset(strict, strict_positions), subset(proxy, proxy_positions_array)


def _paired_data(
    strict_data: StrictFeatureData, proxy_data: StrictFeatureData
) -> tuple[dict[str, FeatureSplit], dict[str, FeatureSplit], dict[str, float]]:
    strict_splits: dict[str, FeatureSplit] = {}
    proxy_splits: dict[str, FeatureSplit] = {}
    pair_coverage: dict[str, float] = {}
    for split_name in ("train", "validation", "test"):
        strict_splits[split_name], proxy_splits[split_name] = _paired_split(
            strict_data.splits[split_name], proxy_data.splits[split_name], split_name
        )
        source_count = len(strict_data.splits[split_name].y)
        pair_coverage[split_name] = (
            len(strict_splits[split_name].y) / source_count if source_count else 0.0
        )
    return strict_splits, proxy_splits, pair_coverage


def _source_feature_frame(
    split: FeatureSplit, feature_names: tuple[str, ...]
) -> pd.DataFrame:
    """Restore strict source one-hot columns expected by RegressionPredictor."""
    frame = split.X.drop(columns=["assessment_type"]).copy()
    for name in feature_names:
        if name.startswith("assessment_type_"):
            category = name.removeprefix("assessment_type_").upper()
            frame[name] = (split.assessment_types == category).astype(int)
    return frame.reindex(columns=feature_names)


def _coverage(split: FeatureSplit) -> dict[str, Any]:
    indicators = [
        name
        for name in split.X.columns
        if name.startswith("previous_assessment_score_lag_")
        and name.endswith("_available")
    ]
    if not indicators:
        raise ValueError("Proxy split has no previous-score availability indicators")
    observed = split.X.loc[:, indicators].fillna(0).to_numpy(dtype=float)
    rows_with_history = int((observed.sum(axis=1) > 0).sum())
    return {
        "rows": len(split.y),
        "rows_with_any_prior_mark": rows_with_history,
        "rows_without_prior_mark": len(split.y) - rows_with_history,
        "prior_mark_coverage": rows_with_history / len(split.y)
        if len(split.y)
        else 0.0,
        "missing_history_rate": (
            1.0 - rows_with_history / len(split.y) if len(split.y) else 0.0
        ),
        "missing_by_lag": {
            name.removesuffix("_available"): int(
                (split.X[name].fillna(0).to_numpy(dtype=float) == 0).sum()
            )
            for name in indicators
        },
    }


def _metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    assessment_types: np.ndarray,
) -> dict[str, Any]:
    return metric_json_safe(regression_metrics(targets, predictions, assessment_types))


def _fit_preprocessing(
    train: FeatureSplit,
    validation: FeatureSplit,
    test: FeatureSplit,
    numeric_feature_names: tuple[str, ...],
) -> tuple[Any, dict[str, np.ndarray]]:
    """Fit Model B preprocessing on paired training rows only."""
    preprocessor = build_preprocessor(numeric_feature_names)
    transformed = {
        "train": preprocessor.fit_transform(train.X).astype(np.float32),
        "validation": preprocessor.transform(validation.X).astype(np.float32),
        "test": preprocessor.transform(test.X).astype(np.float32),
    }
    return preprocessor, transformed


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(metric_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _render_comparison_report(results: dict[str, Any]) -> str:
    lines = [
        "# Previous Marks MLP comparison (submission-based proxy)",
        "",
        "## Scientific boundary",
        "",
        "OULAD does not contain grade-release timestamps. Model B uses a prior score",
        "only when its recorded submission was on or before its own earlier scheduled",
        "deadline. This submission-based rule does not prove the mark was released",
        "or visible to the student by the target prediction cutoff. Model B is a",
        "proxy sensitivity experiment, not a leakage-free or causal model.",
        "Model A's strict schema and artifacts are unchanged.",
        "",
        "## Protocol",
        "",
        "Paired row counts: "
        + f"{json.dumps(results['paired_row_counts'], sort_keys=True)}.",
        "Paired coverage of Model A rows: "
        + f"{json.dumps(results['paired_population_coverage'], sort_keys=True)}.",
        "Both models use the same target, fixed presentation splits, and identical",
        "student/presentation/assessment rows within each split. Preprocessors are fit",
        "independently on their respective training rows only. Model B selects its MLP",
        "configuration using validation MAE across the configured seeds; test metrics",
        "were calculated after that selection. Model A predictions come from the",
        "existing saved strict predictor.",
        "",
        "Seeds: " + ", ".join(map(str, results["seeds"])) + ".",
        "Selected Model B config: `" + results["selected_config"]["name"] + "`.",
        "",
        "## Prior-mark coverage",
        "",
        "| Split | Paired rows | With any prior mark | Without prior mark | "
        "Coverage | Missing-history rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_name, coverage in results["coverage"].items():
        lines.append(
            f"| {split_name} | {coverage['rows']} | "
            f"{coverage['rows_with_any_prior_mark']} | "
            f"{coverage['rows_without_prior_mark']} | "
            f"{coverage['prior_mark_coverage']:.3%} | "
            f"{coverage['missing_history_rate']:.3%} |"
        )
    for split_name in ("validation", "test"):
        lines.extend(
            [
                "",
                f"## Paired {split_name} metrics",
                "",
                "| Model | Type | n | MAE | RMSE | R² |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for model_name in ("Model A — Strict MLP", "Model B — Previous Marks MLP"):
            metrics = results["paired_metrics"][split_name][model_name]
            overall = metrics["overall"]
            lines.append(
                f"| {model_name} | Overall | {overall['n']} | {overall['mae']:.4f} | "
                f"{overall['rmse']:.4f} | {overall['r2']:.4f} |"
            )
            for category, values in metrics["by_assessment_type"].items():
                lines.append(
                    f"| {model_name} | {category} | {values['n']} | "
                    f"{values['mae']:.4f} | "
                    f"{values['rmse']:.4f} | {values['r2']:.4f} |"
                )
    lines.extend(
        [
            "",
            "## Limitations and recommendation",
            "",
            "These results quantify predictive performance under the stated proxy",
            "feature construction. They do not establish that marks were visible to",
            "students at the time or that prior marks cause improved prediction. Use",
            "grade-release timestamps or another authoritative availability record",
            "before describing this feature set as historically available or",
            "deployment-ready.",
            "Coverage reflects whether at least one proxy-eligible prior mark exists;",
            "missing marks remain missing and use availability indicators.",
            "",
            "Model B artifacts are stored separately under "
            "`data/processed/nn_experiment_previous_marks/`.",
            "",
        ]
    )
    return "\n".join(lines)


def run_previous_marks_experiment(
    strict_processed_dir: Path,
    proxy_processed_dir: Path,
    model_a_artifact_dir: Path,
    output_dir: Path,
    report_path: Path,
    *,
    seeds: tuple[int, ...] = MODEL_B_SEEDS,
    search_configs: tuple[MLPConfig, ...] = DEFAULT_SEARCH_CONFIGS,
    max_epochs: int = MAX_EPOCHS,
) -> dict[str, Any]:
    """Train Model B and compare to the untouched Model A on paired rows."""
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Provide one or more unique training seeds")
    if max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    if not search_configs or len({item.name for item in search_configs}) != len(
        search_configs
    ):
        raise ValueError("Provide uniquely named MLP search configurations")
    if output_dir.resolve() == model_a_artifact_dir.resolve():
        raise ValueError("Model B output must not overwrite Model A artifacts")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Model B output directory is not empty: {output_dir}")

    strict_data = load_strict_feature_data(strict_processed_dir)
    proxy_data = load_previous_marks_feature_data(proxy_processed_dir)
    strict_splits, proxy_splits, pair_coverage = _paired_data(strict_data, proxy_data)
    predictor_a = RegressionPredictor.from_artifact_dir(model_a_artifact_dir)
    if tuple(predictor_a.feature_names) != tuple(strict_data.feature_names):
        raise ValueError("Model A artifact schema differs from current strict schema")

    train = proxy_splits["train"]
    validation = proxy_splits["validation"]
    test = proxy_splits["test"]
    preprocessor, transformed = _fit_preprocessing(
        train,
        validation,
        test,
        proxy_data.numeric_feature_names,
    )
    x_train = transformed["train"]
    x_validation = transformed["validation"]
    started = time.perf_counter()
    runs_by_config: dict[str, list[MLPTrainingRun]] = {}
    curves: list[dict[str, Any]] = []
    search_results: list[dict[str, Any]] = []
    for config in search_configs:
        runs: list[MLPTrainingRun] = []
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
                patience=EARLY_STOPPING_PATIENCE,
                batch_size=BATCH_SIZE,
            )
            runs.append(run)
            curves.extend(
                {"config": config.name, "seed": seed, **point} for point in run.history
            )
        runs_by_config[config.name] = runs
        maes = [float(run.validation_metrics["overall"]["mae"]) for run in runs]
        search_results.append(
            {
                "config": config.to_dict(),
                "validation_mae_mean": statistics.fmean(maes),
                "validation_mae_std": statistics.stdev(maes) if len(maes) > 1 else 0.0,
            }
        )
    selected = min(
        search_results,
        key=lambda item: (item["validation_mae_mean"], item["config"]["name"]),
    )
    selected_config = next(
        config for config in search_configs if config.name == selected["config"]["name"]
    )
    selected_runs = runs_by_config[selected_config.name]
    duration = time.perf_counter() - started

    test_transform = transformed["test"]
    models: list[MLPRegressor] = []
    selected_metrics: list[dict[str, Any]] = []
    for run in selected_runs:
        model = MLPRegressor(
            input_size=x_train.shape[1],
            hidden_sizes=selected_config.hidden_sizes,
            dropout=selected_config.dropout,
        )
        model.load_state_dict(run.state_dict)
        model.eval()
        models.append(model)
        selected_metrics.append(
            _metrics(
                test.y, model_predictions(model, test_transform), test.assessment_types
            )
        )
    b_validation_predictions = np.mean(
        [model_predictions(model, x_validation) for model in models], axis=0
    )
    b_test_predictions = np.mean(
        [model_predictions(model, test_transform) for model in models], axis=0
    )
    a_validation_predictions = predictor_a.predict(
        _source_feature_frame(strict_splits["validation"], strict_data.feature_names)
    )
    a_test_predictions = predictor_a.predict(
        _source_feature_frame(strict_splits["test"], strict_data.feature_names)
    )
    paired_metrics = {
        "validation": {
            "Model A — Strict MLP": _metrics(
                validation.y, a_validation_predictions, validation.assessment_types
            ),
            "Model B — Previous Marks MLP": _metrics(
                validation.y, b_validation_predictions, validation.assessment_types
            ),
        },
        "test": {
            "Model A — Strict MLP": _metrics(
                test.y, a_test_predictions, test.assessment_types
            ),
            "Model B — Previous Marks MLP": _metrics(
                test.y, b_test_predictions, test.assessment_types
            ),
        },
    }
    coverage = {name: _coverage(split) for name, split in proxy_splits.items()}
    paired_row_counts = {name: len(split.y) for name, split in proxy_splits.items()}
    results: dict[str, Any] = {
        "scientific_status": "submission_based_proxy_sensitivity_experiment",
        "grade_release_timestamps_available": False,
        "proxy_assumption": (
            "Prior score is treated as a candidate input when non-banked, valid, "
            "submitted on time for a strictly earlier assessment in the same "
            "student and presentation; score release is unverified."
        ),
        "seeds": list(seeds),
        "selected_config": selected_config.to_dict(),
        "search_results": search_results,
        "paired_row_counts": paired_row_counts,
        "paired_population_coverage": pair_coverage,
        "coverage": coverage,
        "paired_metrics": paired_metrics,
        "selected_seed_test_metrics": selected_metrics,
        "training_duration_seconds": duration,
        "max_epochs": max_epochs,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    model_files: list[str] = []
    for run in selected_runs:
        filename = f"mlp_seed_{run.seed}.pt"
        torch.save(run.state_dict, output_dir / filename)
        model_files.append(filename)
    schema_payload = {
        "mode": "proxy",
        "source_feature_columns": list(proxy_data.feature_names),
        "numeric_feature_columns": list(proxy_data.numeric_feature_names),
        "assessment_type_indicator_columns": list(proxy_data.assessment_type_columns),
        "model_categorical_columns": ["assessment_type"],
        "model_input_columns": ["assessment_type", *proxy_data.numeric_feature_names],
        "transformed_feature_names": list(preprocessor.get_feature_names_out()),
        "identifiers_excluded": list(proxy_data.manifest["identifiers_not_features"]),
        "target": proxy_data.manifest["target"],
        "prediction_bounds": [PREDICTION_MIN, PREDICTION_MAX],
        "cutoff": proxy_data.manifest["prediction_cutoff"],
        "score_availability_assumption": proxy_data.manifest[
            "score_availability_assumption"
        ],
    }
    (output_dir / "feature_schema.json").write_text(
        json.dumps(schema_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config_payload = {
        "feature_mode": "proxy",
        "model_name": "Previous Marks MLP",
        "selected_mlp_config": selected_config.to_dict(),
        "search_configs": [config.to_dict() for config in search_configs],
        "seed_list": list(seeds),
        "selection_metric": "mean validation MAE across configured seeds",
        "early_stopping_metric": "validation Huber loss",
        "loss": selected_config.loss_name,
        "target_scale_for_optimization": TARGET_SCALE,
        "batch_size": BATCH_SIZE,
        "max_epochs": max_epochs,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "device": "cpu",
        "prediction_policy": "clip predictions to [0, 100]",
        "model_files": model_files,
        "preprocessed_feature_count": int(x_train.shape[1]),
        "proxy_assumption": results["proxy_assumption"],
        "versions": {
            "python": __import__("platform").python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "joblib": joblib.__version__,
        },
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_json(output_dir / "evaluation.json", results)
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
            "model_a_strict_prediction": clip_predictions(a_test_predictions),
            "model_b_previous_marks_prediction": clip_predictions(b_test_predictions),
        }
    )
    prediction_frame.to_csv(output_dir / "paired_test_predictions.csv", index=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_comparison_report(results), encoding="utf-8")
    LOGGER.info("Wrote Model B artifacts to %s", output_dir)
    LOGGER.info("Wrote paired report to %s", report_path)
    return results


def main(argv: list[str] | None = None) -> int:
    """Train Model B from existing strict and proxy feature directories."""
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-processed-dir",
        type=Path,
        default=root / "data" / "processed",
    )
    parser.add_argument(
        "--proxy-processed-dir",
        type=Path,
        default=root / "data" / "processed" / "proxy",
    )
    parser.add_argument(
        "--model-a-artifact-dir",
        type=Path,
        default=root / "data" / "processed" / "nn_experiment",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "processed" / "nn_experiment_previous_marks",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "previous_marks_mlp_comparison.md",
    )
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run_previous_marks_experiment(
            args.strict_processed_dir,
            args.proxy_processed_dir,
            args.model_a_artifact_dir,
            args.output_dir,
            args.report,
            max_epochs=args.max_epochs,
        )
    except (OSError, ValueError, RuntimeError) as error:
        LOGGER.error("Previous Marks experiment failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
