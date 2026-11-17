"""Run the synthetic NN versus training-only average planner experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import pandas as pd
from run_demo_experiments import _prediction_scenarios, _synthetic_config

from study_planner.eval.experiment_runner import (
    DEFAULT_EXPERIMENT_SEEDS,
    ExperimentRunRecord,
)
from study_planner.eval.experiments import (
    run_prediction_comparison,
    write_experiment_bundle,
    write_experiment_plot,
)
from study_planner.integration.application_inputs import (
    load_session_nn_adapter,
    load_simple_average_adapter,
)
from study_planner.search.nn_adapter import NNModelMode, SessionNNAdapter
from study_planner.search.session_mapping import (
    ClickScenario,
    SessionClickMappingConfig,
)


def _sha256(path: Path) -> str:
    """Hash a file without loading the full dataset into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_cohort_summary(
    records: tuple[ExperimentRunRecord, ...], path: Path
) -> None:
    """Write overall method means and sample standard deviations."""
    methods = sorted({record.method for record in records})
    metric_names = sorted(
        {name for record in records for name in record.metrics}
    )
    rows: list[dict[str, Any]] = []
    for method in methods:
        method_records = [record for record in records if record.method == method]
        statuses = Counter(record.status.value for record in method_records)
        for metric in metric_names:
            values = [
                float(value)
                for record in method_records
                if (value := record.metrics.get(metric)) is not None
            ]
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "run_count": len(values),
                    "mean": fmean(values) if values else None,
                    "standard_deviation": stdev(values) if len(values) > 1 else None,
                    "feasible_runs": statuses["feasible"],
                    "infeasible_runs": statuses["infeasible"],
                    "incomplete_runs": statuses["incomplete"],
                    "error_runs": statuses["error"],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "method",
            "metric",
            "run_count",
            "mean",
            "standard_deviation",
            "feasible_runs",
            "infeasible_runs",
            "incomplete_runs",
            "error_runs",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path,
    records: tuple[ExperimentRunRecord, ...],
    *,
    profile_count: int,
    seeds: tuple[int, ...],
    configuration: dict[str, Any],
    nn_adapter: SessionNNAdapter,
    average_adapter: SessionNNAdapter,
    profile_manifest_hash: str,
    feature_csv_hash: str,
) -> None:
    """Document matched inputs and the training-only baseline methodology."""
    outcome_counts = {
        method: Counter(
            record.provenance.get("planner_outcome", "unknown")
            for record in records
            if record.method == method
        )
        for method in ("nn", "average")
    }
    method_records = {
        method: [record for record in records if record.method == method]
        for method in ("nn", "average")
    }
    lines = [
        "# Experiment 3: NN versus Simple Average-Mark Planner",
        "",
        "**Configuration status:** synthetic coursework demonstration only.",
        "",
        f"Profiles: {profile_count}; seeds: `{list(seeds)}`.",
        f"Runs: {len(records)} ({profile_count} profiles × 2 predictors × "
        f"{len(seeds)} seeds).",
        "Each seed invokes the integrated planner independently. Fixed inputs make",
        "algorithmic outcomes deterministic; runtime is measured on each invocation.",
        "Seeds label five separate executions, not randomized planner behavior.",
        "Per-profile seed standard deviations are therefore zero for deterministic",
        "metrics; cohort standard deviations reflect differences across profiles,",
        "with each profile represented once per seed.",
        "",
        "## Matched configuration",
        "",
        f"- Course: {configuration['course_module']}; test presentation: "
        f"{configuration['test_presentation']}.",
        f"- Assignments: {', '.join(configuration['assessment_ids'])}.",
        f"- Per-assignment synthetic target: {configuration['target_mark']}.",
        f"- Weekly capacity: {configuration['weekly_capacity']}; "
        f"catalog slots: {configuration['slot_count']}.",
        f"- Click mapping: {configuration['click_scenario']} / "
        f"{configuration['activity_group']} / "
        f"{configuration['clicks_per_session']} clicks per hypothetical session.",
        f"- Support policy: {configuration['support_policy']}.",
        f"- Profile manifest SHA-256: `{profile_manifest_hash}`.",
        f"- Feature CSV SHA-256: `{feature_csv_hash}`.",
        "Both predictors use the same feature rows, target rule, assignments,",
        "slot catalog, support policy, click mapping, Search, and CSP mode. Only",
        "the fitted predictor differs.",
        "",
        "## Predictor and baseline methodology",
        "",
        f"- NN: strict Model A artifact `{nn_adapter.fitted_artifact_digest}`.",
        "  It is loaded and integrity-verified; no retraining was performed.",
        "- Simple Average-Mark: predicts the training-target mean "
        f"({average_adapter.predictor.mean_mark:.6f}) for each assignment.",
        f"  Fitted on {average_adapter.predictor.training_target_count} FFF "
        "training targets from the existing strict training split.",
        "  Training-target digest: "
        f"`{average_adapter.predictor.training_targets_sha256}`.",
        "  Source training-file digest: "
        f"`{average_adapter.predictor.source_data_sha256}`.",
        "  Validation/test scores and profile outcomes are excluded from fitting.",
        "- Both are wrapped by the same SessionNNAdapter interface and passed to",
        "  integrated Uniform Cost Search with Backtracking + Forward Checking",
        "  + AC-3 verification.",
        "",
        "## Results",
        "",
        "| Predictor | Target achieved | Mean predicted mark ± SD | "
        "Mean session cost ± SD |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in ("nn", "average"):
        rows = method_records[method]
        marks = [float(record.metrics["mean_predicted_mark"]) for record in rows]
        costs = [float(record.metrics["solution_cost"]) for record in rows]
        lines.append(
            f"| {method} | {outcome_counts[method]['target_achieved']} / "
            f"{len(rows)} | {fmean(marks):.3f} ± {stdev(marks):.3f} | "
            f"{fmean(costs):.3f} ± {stdev(costs):.3f} |"
        )
    lines.extend(
        (
        "",
        "Raw per-profile, per-predictor, per-seed records are in",
        "`experiment3_runs.csv` and `experiment3_runs.json`. The per-profile",
        "mean/sample-standard-deviation table is `experiment3_summary.csv`; the",
        "cohort-wide method mean/sample-standard-deviation table is",
        "`experiment3_method_summary.csv`. Plots compare predicted marks, target",
        "achievement, and solution cost.",
        "",
        "All target, slot, and click values are documented synthetic scenario",
        "assumptions. OULAD associations are observational and do not establish",
        "that study sessions cause marks to change.",
        "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Run all matched Experiment 3 profile/predictor/seed combinations."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-directory", type=Path, default=root / "data/processed/strict"
    )
    parser.add_argument(
        "--feature-csv",
        type=Path,
        default=root / "data/processed/strict/fff_test.csv",
    )
    parser.add_argument(
        "--profile-manifest",
        type=Path,
        default=root / "reports/experiment_results/profiles_fff_2014J.json",
    )
    parser.add_argument(
        "--artifact-directory", type=Path, default=root / "data/processed/nn_demo"
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=root / "reports/experiment_results/experiment3_nn_vs_average",
    )
    args = parser.parse_args()
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error(
            f"refusing to overwrite existing Experiment 3 results: "
            f"{args.output_directory}"
        )

    manifest = json.loads(args.profile_manifest.read_text(encoding="utf-8"))
    student_ids = tuple(str(value) for value in manifest["student_ids"])
    if len(student_ids) != 30 or len(set(student_ids)) != 30:
        parser.error("the shared profile manifest must contain 30 unique students")
    if manifest.get("feature_mode") != "strict":
        parser.error("Experiment 3 requires the shared strict-feature profile manifest")
    feature_csv_hash = _sha256(args.feature_csv)
    if feature_csv_hash != manifest.get("source_csv_sha256"):
        parser.error("feature CSV hash does not match the selected profile manifest")

    feature_rows = pd.read_csv(
        args.feature_csv,
        dtype={"id_student": "string", "id_assessment": "string"},
    )
    assessment_ids = tuple(str(value) for value in manifest["assessment_ids"])
    selected = feature_rows.loc[
        feature_rows["id_student"].isin(student_ids)
        & feature_rows["id_assessment"].isin(assessment_ids)
    ]
    counts = selected.groupby("id_student")["id_assessment"].agg(
        lambda values: (len(values), values.nunique())
    )
    if set(counts.index) != set(student_ids) or any(
        count != (3, 3) for count in counts
    ):
        parser.error("each selected profile must contain exactly the three assignments")

    mapping = SessionClickMappingConfig(ClickScenario.LOW, "content_navigation")
    try:
        nn_adapter = load_session_nn_adapter(
            args.artifact_directory, NNModelMode.MODEL_A_STRICT, mapping
        )
    except (OSError, ValueError) as error:
        parser.error(
            "the existing strict NN artifact is missing or incompatible; "
            f"no training was started: {error}"
        )
    average_adapter = load_simple_average_adapter(args.processed_directory, mapping)
    adapters = {"nn": nn_adapter, "average": average_adapter}
    planner_results = {}
    scenarios = _prediction_scenarios(
        feature_rows, student_ids, adapters, planner_results
    )
    records = run_prediction_comparison(scenarios, experiment_id="experiment3")

    output = args.output_directory
    write_experiment_bundle(
        records,
        output,
        "experiment3",
        "mean_predicted_mark",
        group_by_scenario=False,
    )
    write_experiment_plot(
        records,
        output,
        "experiment3_target_achievement",
        "all_targets_met",
        group_by_scenario=False,
    )
    write_experiment_plot(
        records,
        output,
        "experiment3_solution_cost",
        "solution_cost",
        group_by_scenario=False,
    )
    _write_cohort_summary(records, output / "experiment3_method_summary.csv")

    scenario_config = _synthetic_config(nn_adapter)
    configuration = {
        "course_module": scenario_config.presentation_split.course_identity,
        "test_presentation": scenario_config.presentation_split.testing_presentation,
        "assessment_ids": [
            rule.assessment_id for rule in scenario_config.assignment_rules
        ],
        "target_mark": scenario_config.assignment_rules[0].target_mark,
        "weekly_capacity": scenario_config.slot_catalog.weekly_capacity,
        "slot_count": len(scenario_config.slot_catalog.slots),
        "click_scenario": scenario_config.click_mapping.scenario.value,
        "activity_group": scenario_config.click_mapping.activity_group,
        "clicks_per_session": scenario_config.click_mapping.clicks_per_session,
        "support_policy": scenario_config.support_policy.out_of_support_action.value,
        "configuration_status": "SYNTHETIC_DEMO_ONLY_NOT_PRODUCTION",
    }
    (output / "experiment3_config.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output / "experiment3_report.md",
        records,
        profile_count=len(student_ids),
        seeds=DEFAULT_EXPERIMENT_SEEDS,
        configuration=configuration,
        nn_adapter=nn_adapter,
        average_adapter=average_adapter,
        profile_manifest_hash=_sha256(args.profile_manifest),
        feature_csv_hash=feature_csv_hash,
    )
    print(f"Experiment 3 results: {output}")
    print(f"Run records: {len(records)}")
    print(f"NN artifact digest: {nn_adapter.fitted_artifact_digest}")
    print(
        "Average baseline: "
        f"mean={average_adapter.predictor.mean_mark:.6f}, "
        f"training_targets={average_adapter.predictor.training_target_count}"
    )
    return 1 if any(record.error_type for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
