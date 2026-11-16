"""Profile-based monotonicity experiment for session-to-mark predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import pandas as pd

from study_planner.data.feature_engineering.constants import (
    ACTIVITY_GROUPS,
    IDENTIFIER_COLUMNS,
    STRICT_FFF_FEATURE_NAMES,
    TARGET_COLUMN,
)
from study_planner.eval.experiment_runner import DEFAULT_EXPERIMENT_SEEDS
from study_planner.nn.dataset import DatasetSchemaError, prepare_inference_frame
from study_planner.search.nn_adapter import (
    CandidateFeatureContext,
    CandidateSessionAllocation,
    EvaluationStatus,
    NNModelMode,
    SessionNNAdapter,
)
from study_planner.search.session_mapping import (
    ClickScenario,
    MarginalSupportStatus,
    MonotonicityContext,
    MonotonicityObservation,
    MonotonicityViolation,
    SessionClickMappingConfig,
    evaluate_monotonicity,
)


@dataclass(frozen=True, slots=True)
class MonotonicityRun:
    """One profile/model/seed audit of neighboring session allocations."""

    student_id: str
    seed: int
    status: str
    model_family: str
    model_identity: str
    fitted_model_digest: str
    max_sessions: int
    allocations_evaluated: int
    supported_allocations: int
    out_of_support_allocations: int
    unassessed_support_allocations: int
    inference_failures: int
    comparisons: int
    violations: tuple[MonotonicityViolation, ...]
    error_message: str | None = None

    @property
    def violation_rate(self) -> float | None:
        """Return violation/comparison ratio, or None if there were no pairs."""
        if self.comparisons == 0:
            return None
        return len(self.violations) / self.comparisons


@dataclass(frozen=True, slots=True)
class MonotonicitySummary:
    """Aggregate across profile-level monotonicity runs for a seed group."""

    seed: int | None
    profile_runs: int
    complete_runs: int
    error_runs: int
    comparisons: int
    violations: int
    pooled_violation_rate: float | None
    mean_profile_violation_rate: float | None
    profile_violation_rate_standard_deviation: float | None


@dataclass(frozen=True, slots=True)
class MonotonicityExperimentResult:
    """Provenance and outputs for one complete monotonicity experiment."""

    model_family: str
    model_identity: str
    fitted_model_digest: str
    feature_csv_sha256: str
    profile_manifest_sha256: str
    profile_manifest_path: str
    mapping_scenario: str
    max_sessions: int
    seeds: tuple[int, ...]
    runs: tuple[MonotonicityRun, ...]
    summaries: tuple[MonotonicitySummary, ...]


def _allocations(max_sessions: int) -> tuple[tuple[int, int, int, int], ...]:
    """Enumerate all four-window allocations with total count at most the cap."""
    if isinstance(max_sessions, bool) or not isinstance(max_sessions, int):
        raise ValueError("max_sessions must be an integer")
    if not 1 <= max_sessions <= 6:
        raise ValueError("max_sessions must be between 1 and 6")
    return tuple(
        tuple(int(count) for count in values)
        for values in itertools.product(range(max_sessions + 1), repeat=4)
        if sum(values) <= max_sessions
    )


def _load_profile_rows(
    feature_csv: Path, profile_manifest_path: Path
) -> tuple[pd.DataFrame, dict[str, Any], str, str]:
    """Validate profile manifest against the strict feature rows and hashes."""
    manifest_bytes = profile_manifest_path.read_bytes()
    profile_manifest = json.loads(manifest_bytes)
    with feature_csv.open("rb") as source_stream:
        feature_digest = hashlib.file_digest(source_stream, "sha256").hexdigest()
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if profile_manifest.get("feature_mode") != "strict":
        raise ValueError("monotonicity experiment requires strict Model A profiles")
    if profile_manifest.get("source_csv_sha256") != feature_digest:
        raise ValueError("profile manifest source hash differs from feature CSV")
    student_ids = profile_manifest.get("student_ids")
    assessment_ids = profile_manifest.get("assessment_ids")
    if (
        not isinstance(student_ids, list)
        or len(student_ids) < 1
        or len(set(student_ids)) != len(student_ids)
        or not isinstance(assessment_ids, list)
        or len(assessment_ids) != 3
        or len(set(assessment_ids)) != 3
    ):
        raise ValueError("profile manifest needs unique students and three assessments")

    frame = pd.read_csv(
        feature_csv,
        dtype={name: "string" for name in IDENTIFIER_COLUMNS},
        keep_default_na=True,
    )
    expected_columns = {*IDENTIFIER_COLUMNS, *STRICT_FFF_FEATURE_NAMES, TARGET_COLUMN}
    if set(frame.columns) != expected_columns:
        raise DatasetSchemaError("feature CSV columns do not match the strict schema")
    selected = frame.loc[
        frame["id_student"].isin(student_ids)
        & frame["id_assessment"].isin(assessment_ids)
        & (frame["code_module"].astype(str) == str(profile_manifest["course_module"]))
        & (
            frame["code_presentation"].astype(str)
            == str(profile_manifest["presentation"])
        )
    ].copy()
    if (
        selected["id_student"].isna().any()
        or selected["id_assessment"].isna().any()
        or selected.duplicated(["id_student", "id_assessment"]).any()
    ):
        raise ValueError("selected feature rows have missing or duplicate identities")
    expected_pairs = {
        (student_id, assessment_id)
        for student_id in student_ids
        for assessment_id in assessment_ids
    }
    actual_pairs = set(
        zip(
            selected["id_student"].astype(str),
            selected["id_assessment"].astype(str),
            strict=True,
        )
    )
    if actual_pairs != expected_pairs:
        raise ValueError("selected profiles do not have exactly three assessment rows")
    features = selected.loc[:, list(STRICT_FFF_FEATURE_NAMES)]
    try:
        prepare_inference_frame(
            features,
            STRICT_FFF_FEATURE_NAMES,
            STRICT_FFF_FEATURE_NAMES[:3],
        )
    except (DatasetSchemaError, TypeError, ValueError) as error:
        raise ValueError(
            f"selected strict profile features are invalid: {error}"
        ) from error
    return selected, profile_manifest, feature_digest, manifest_digest


def _summarize(
    runs: Sequence[MonotonicityRun], seeds: Sequence[int]
) -> tuple[MonotonicitySummary, ...]:
    summaries: list[MonotonicitySummary] = []
    for seed in (*seeds, None):
        group = [run for run in runs if seed is None or run.seed == seed]
        if not group:
            continue
        rates = [
            run.violation_rate
            for run in group
            if run.violation_rate is not None
        ]
        comparisons = sum(run.comparisons for run in group)
        violations = sum(len(run.violations) for run in group)
        summaries.append(
            MonotonicitySummary(
                seed,
                len(group),
                sum(run.status == "complete" for run in group),
                sum(run.status == "error" for run in group),
                comparisons,
                violations,
                violations / comparisons if comparisons else None,
                fmean(rates) if rates else None,
                stdev(rates) if len(rates) > 1 else None,
            )
        )
    return tuple(summaries)


def run_monotonicity_experiment(
    feature_csv: Path,
    profile_manifest_path: Path,
    adapter: SessionNNAdapter,
    *,
    max_sessions: int = 2,
    seeds: Sequence[int] = DEFAULT_EXPERIMENT_SEEDS,
) -> MonotonicityExperimentResult:
    """Evaluate one-session neighbors for every profile, assessment, and seed.

    Only predictions whose click mapping is marginally in training support are
    included in monotonicity comparisons. Out-of-support and unassessed values
    remain visible as excluded allocation counts. The seed is recorded for
    reproducibility; inference itself is deterministic.
    """
    if adapter.model_mode not in {
        NNModelMode.MODEL_A_STRICT,
        NNModelMode.SIMPLE_AVERAGE_BASELINE,
    }:
        raise ValueError("monotonicity experiment requires a strict feature model")
    if not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ):
        raise ValueError("seeds must be a non-empty sequence of integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    allocations = _allocations(max_sessions)
    selected, profile_manifest, feature_digest, manifest_digest = _load_profile_rows(
        feature_csv, profile_manifest_path
    )
    assessment_ids = tuple(profile_manifest["assessment_ids"])
    student_ids = tuple(profile_manifest["student_ids"])
    scenario = adapter.mapping_scenario_identity
    records: list[MonotonicityRun] = []

    for seed in seeds:
        for student_id in student_ids:
            profile_rows = selected.loc[selected["id_student"] == student_id]
            observations: list[MonotonicityObservation] = []
            supported = 0
            out_of_support = 0
            unassessed = 0
            inference_failures = 0
            errors: list[str] = []
            for assessment_id in assessment_ids:
                row = profile_rows.loc[profile_rows["id_assessment"] == assessment_id]
                feature_context = CandidateFeatureContext(
                    student_id,
                    assessment_id,
                    row.loc[:, list(STRICT_FFF_FEATURE_NAMES)].reset_index(drop=True),
                )
                monotonicity_context = MonotonicityContext(
                    student_id,
                    assessment_id,
                    feature_context.base_feature_context_identity,
                    adapter.mapping_scenario_identity,
                )
                for allocation in allocations:
                    try:
                        prediction = adapter.evaluate_provisional(
                            CandidateSessionAllocation(allocation), feature_context
                        )
                    except (TypeError, ValueError, RuntimeError) as error:
                        inference_failures += 1
                        errors.append(f"{assessment_id}:{type(error).__name__}:{error}")
                        continue
                    if prediction.status is EvaluationStatus.INFERENCE_FAILED:
                        inference_failures += 1
                        errors.append(
                            f"{assessment_id}:{prediction.error_type}:"
                            f"{prediction.error_message}"
                        )
                        continue
                    if (
                        prediction.mapping.marginal_status
                        is MarginalSupportStatus.OUT_OF_RANGE
                    ):
                        out_of_support += 1
                        continue
                    if (
                        prediction.mapping.marginal_status
                        is MarginalSupportStatus.UNASSESSED
                    ):
                        unassessed += 1
                        continue
                    if prediction.prediction is None:
                        inference_failures += 1
                        errors.append(f"{assessment_id}:missing_prediction")
                        continue
                    supported += 1
                    schedule_identity = tuple(
                        (f"window-{index + 1}", count)
                        for index, count in enumerate(allocation)
                    )
                    observations.append(
                        MonotonicityObservation(
                            allocation,
                            prediction.prediction,
                            monotonicity_context,
                            schedule_identity,
                        )
                    )

            try:
                report = evaluate_monotonicity(observations)
                status = "complete" if not inference_failures else "error"
                error_message = "; ".join(errors) or None
                comparisons = report.comparisons
                violations = report.violations
            except (TypeError, ValueError) as error:
                status = "error"
                error_message = f"{type(error).__name__}:{error}"
                comparisons = 0
                violations = ()
            records.append(
                MonotonicityRun(
                    student_id,
                    seed,
                    status,
                    adapter.prediction_family,
                    adapter.model_provenance_identity,
                    adapter.fitted_artifact_digest,
                    max_sessions,
                    len(assessment_ids) * len(allocations),
                    supported,
                    out_of_support,
                    unassessed,
                    inference_failures,
                    comparisons,
                    violations,
                    error_message,
                )
            )
    return MonotonicityExperimentResult(
        adapter.prediction_family,
        adapter.model_provenance_identity,
        adapter.fitted_artifact_digest,
        feature_digest,
        manifest_digest,
        str(profile_manifest_path),
        scenario,
        max_sessions,
        tuple(seeds),
        tuple(records),
        _summarize(records, seeds),
    )


def write_monotonicity_results(
    result: MonotonicityExperimentResult, output_dir: Path
) -> tuple[Path, Path, Path, Path]:
    """Write per-profile CSV/JSON, seed summary CSV, and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = output_dir / "monotonicity_runs.csv"
    runs_json = output_dir / "monotonicity_runs.json"
    summary_csv = output_dir / "monotonicity_summary.csv"
    report_path = output_dir / "monotonicity_report.md"
    run_fields = (
        "student_id",
        "seed",
        "status",
        "model_family",
        "model_identity",
        "fitted_model_digest",
        "max_sessions",
        "allocations_evaluated",
        "supported_allocations",
        "out_of_support_allocations",
        "unassessed_support_allocations",
        "inference_failures",
        "comparisons",
        "violations",
        "violation_rate",
        "violation_details_json",
        "error_message",
    )
    with _temporary_text(runs_csv, newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=run_fields)
        writer.writeheader()
        for run in result.runs:
            writer.writerow(
                {
                    "student_id": run.student_id,
                    "seed": run.seed,
                    "status": run.status,
                    "model_family": run.model_family,
                    "model_identity": run.model_identity,
                    "fitted_model_digest": run.fitted_model_digest,
                    "max_sessions": run.max_sessions,
                    "allocations_evaluated": run.allocations_evaluated,
                    "supported_allocations": run.supported_allocations,
                    "out_of_support_allocations": run.out_of_support_allocations,
                    "unassessed_support_allocations": (
                        run.unassessed_support_allocations
                    ),
                    "inference_failures": run.inference_failures,
                    "comparisons": run.comparisons,
                    "violations": len(run.violations),
                    "violation_rate": run.violation_rate,
                    "violation_details_json": _json(
                        [
                            {
                                "parent_sessions": item.parent_sessions,
                                "child_sessions": item.child_sessions,
                                "changed_window": item.changed_window,
                                "parent_mark": item.parent_mark,
                                "child_mark": item.child_mark,
                                "prediction_delta": item.prediction_delta,
                            }
                            for item in run.violations
                        ]
                    ),
                    "error_message": run.error_message,
                }
            )
    with _temporary_text(summary_csv, newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=tuple(MonotonicitySummary.__dataclass_fields__),
        )
        writer.writeheader()
        for summary in result.summaries:
            writer.writerow(
                {
                    name: getattr(summary, name)
                    for name in MonotonicitySummary.__dataclass_fields__
                }
            )
    _write_json(
        runs_json,
        {
            "schema_version": 1,
            "model_family": result.model_family,
            "model_identity": result.model_identity,
            "fitted_model_digest": result.fitted_model_digest,
            "feature_csv_sha256": result.feature_csv_sha256,
            "profile_manifest_sha256": result.profile_manifest_sha256,
            "profile_manifest_path": result.profile_manifest_path,
            "mapping_scenario": result.mapping_scenario,
            "max_sessions": result.max_sessions,
            "seeds": list(result.seeds),
            "runs": [
                {
                    name: getattr(run, name)
                    for name in MonotonicityRun.__dataclass_fields__
                    if name != "violations"
                }
                | {
                    "violations": [
                        {
                            name: getattr(violation, name)
                            for name in MonotonicityViolation.__dataclass_fields__
                        }
                        for violation in run.violations
                    ]
                }
                for run in result.runs
            ],
            "summaries": [
                {
                    name: getattr(summary, name)
                    for name in MonotonicitySummary.__dataclass_fields__
                }
                for summary in result.summaries
            ],
        },
    )
    report_path.write_text(_report_markdown(result), encoding="utf-8")
    return runs_csv, runs_json, summary_csv, report_path


def main(argv: list[str] | None = None) -> int:
    """CLI for the selected-profile monotonicity experiment."""
    parser = argparse.ArgumentParser(
        description=(
            "Measure predicted-mark changes between one-session-neighbor "
            "allocations for the shared student profiles."
        )
    )
    parser.add_argument(
        "--feature-csv", type=Path, default=Path("data/processed/strict/fff_test.csv")
    )
    parser.add_argument(
        "--profile-manifest",
        type=Path,
        default=Path("reports/experiment_results/profiles_fff_2014J.json"),
    )
    parser.add_argument(
        "--processed-directory", type=Path, default=Path("data/processed/strict")
    )
    parser.add_argument("--predictor", choices=("nn", "average"), default="nn")
    parser.add_argument("--artifact-directory", type=Path)
    parser.add_argument(
        "--click-scenario",
        choices=tuple(scenario.value for scenario in ClickScenario),
        default=ClickScenario.LOW.value,
    )
    parser.add_argument(
        "--activity-group", choices=tuple(ACTIVITY_GROUPS), default="content_navigation"
    )
    parser.add_argument("--max-sessions", type=int, default=2)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_EXPERIMENT_SEEDS
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/experiment_results")
    )
    args = parser.parse_args(argv)
    mapping = SessionClickMappingConfig(
        ClickScenario(args.click_scenario), args.activity_group
    )
    try:
        from study_planner.integration.application_inputs import (
            load_session_nn_adapter,
            load_simple_average_adapter,
        )

        if args.predictor == "nn":
            if args.artifact_directory is None:
                parser.error("--artifact-directory is required for --predictor nn")
            adapter = load_session_nn_adapter(
                args.artifact_directory, NNModelMode.MODEL_A_STRICT, mapping
            )
        else:
            adapter = load_simple_average_adapter(args.processed_directory, mapping)
        result = run_monotonicity_experiment(
            args.feature_csv,
            args.profile_manifest,
            adapter,
            max_sessions=args.max_sessions,
            seeds=args.seeds,
        )
        outputs = write_monotonicity_results(result, args.output_dir)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    overall = next(summary for summary in result.summaries if summary.seed is None)
    print(
        f"Completed {len(result.runs)} profile-seed monotonicity runs; "
        f"{overall.violations} violations across {overall.comparisons} "
        "supported comparisons."
    )
    for output in outputs:
        print(output)
    return 0


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _temporary_text(path: Path, *, newline: str | None = None):
    temporary = tempfile.NamedTemporaryFile(
        "w", newline=newline, encoding="utf-8", dir=path.parent, delete=False
    )
    temporary_path = Path(temporary.name)
    original_close = temporary.close

    class _AtomicText:
        def __enter__(self):
            return temporary

        def __exit__(self, exc_type, exc_value, traceback):
            original_close()
            if exc_type is None:
                temporary_path.replace(path)
            else:
                temporary_path.unlink(missing_ok=True)

    return _AtomicText()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(payload, temporary, indent=2, sort_keys=True, allow_nan=False)
        temporary.write("\n")
    temporary_path.replace(path)


def _report_markdown(result: MonotonicityExperimentResult) -> str:
    lines = [
        "# Monotonicity Experiment",
        "",
        f"- Model family: `{result.model_family}`",
        f"- Model identity: `{result.model_identity}`",
        f"- Model digest: `{result.fitted_model_digest}`",
        f"- Profile manifest SHA-256: `{result.profile_manifest_sha256}`",
        f"- Feature CSV SHA-256: `{result.feature_csv_sha256}`",
        f"- Click mapping: `{result.mapping_scenario}`",
        f"- Maximum total sessions per assignment: {result.max_sessions}",
        f"- Seeds: {', '.join(str(seed) for seed in result.seeds)}",
        "",
        "A violation is a one-session allocation increase that lowers the "
        "predicted mark. Only predictions whose marginal mapping support is "
        "in range are included in comparisons. Out-of-range and unassessed "
        "allocations are reported separately.",
        "",
        "| Seed | Profile runs | Comparisons | Violations | Pooled rate | "
        "Mean profile rate | Profile-rate SD | Errors |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in result.summaries:
        seed_label = "overall" if summary.seed is None else str(summary.seed)
        pooled = (
            "—"
            if summary.pooled_violation_rate is None
            else f"{summary.pooled_violation_rate:.4f}"
        )
        mean = (
            "—"
            if summary.mean_profile_violation_rate is None
            else f"{summary.mean_profile_violation_rate:.4f}"
        )
        deviation = (
            "—"
            if summary.profile_violation_rate_standard_deviation is None
            else f"{summary.profile_violation_rate_standard_deviation:.4f}"
        )
        lines.append(
            f"| {seed_label} | {summary.profile_runs} | {summary.comparisons} | "
            f"{summary.violations} | {pooled} | {mean} | {deviation} | "
            f"{summary.error_runs} |"
        )
    lines.extend(
        [
            "",
            "Predictions under synthetic session-to-click perturbations describe "
            "model behavior and do not establish a causal effect of study time.",
            "",
        ]
    )
    return "\n".join(lines)
