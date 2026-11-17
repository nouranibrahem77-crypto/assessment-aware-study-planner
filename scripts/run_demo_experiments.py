"""Run explicitly synthetic demo experiments; outputs are never production results."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from study_planner.csp.constraints import (
    build_required_constraints,
    expected_session_variables,
)
from study_planner.csp.core import (
    AssignmentCandidateIdentity,
    AssignmentPlan,
    CspProblem,
    SessionPair,
    SessionVariable,
    VariableDomain,
    WeeklySlot,
    WholePlanManifest,
)
from study_planner.csp.verification import WholePlanVerificationPolicy
from study_planner.eval.experiment_runner import (
    DEFAULT_EXPERIMENT_SEEDS,
    ExperimentRunRecord,
)
from study_planner.eval.experiments import (
    CspScenario,
    PredictionScenario,
    SearchScenario,
    run_csp_comparison,
    run_prediction_comparison,
    run_search_comparison,
    write_experiment_bundle,
)
from study_planner.integration.application_inputs import (
    AllocationBounds,
    ApplicationAssignmentRule,
    ApplicationInputConfig,
    ModelArtifactIdentity,
    OuladPresentationSplit,
    OutOfSupportAction,
    SupportPolicy,
    SyntheticWeeklySlotPattern,
    TargetRule,
    build_application_inputs,
    build_synthetic_weekly_slot_catalog,
    load_session_nn_adapter,
    load_simple_average_adapter,
)
from study_planner.integration.planner import (
    FinalPlannerResult,
    integrated_uniform_cost_search,
)
from study_planner.integration.schedule_bridge import CspAlgorithm
from study_planner.nn.models import DEFAULT_SEARCH_CONFIGS
from study_planner.nn.training import run_training_experiment
from study_planner.search.nn_adapter import NNModelMode, SessionNNAdapter
from study_planner.search.session_mapping import (
    ClickScenario,
    SessionClickMappingConfig,
)

LOGGER = logging.getLogger("study_planner.demo_experiments")
ASSIGNMENTS = ("a_34899", "a_34900", "a_34901")
ASSESSMENTS = ("34899", "34900", "34901")
DEADLINES = (24, 52, 94)
DEMO_TARGET_MARK = 60.0
DEMO_SLOT_WEEKS = (3, 7, 13)


def _synthetic_demo_catalog():
    catalog, eligible_by_assignment = build_synthetic_weekly_slot_catalog(
        identity="synthetic-demo-slots",
        version="v1",
        weekly_capacity=2,
        deadline_policy_identity="synthetic-demo-deadline",
        deadline_policy_version="v1",
        assignment_deadlines=tuple(zip(ASSIGNMENTS, DEADLINES, strict=True)),
        week_indices=DEMO_SLOT_WEEKS,
        pattern=SyntheticWeeklySlotPattern(
            weekdays_iso=(1,),
            start_minute_local=9 * 60,
            end_minute_local=10 * 60,
            timezone_identity="UTC",
        ),
    )
    slots_by_week = {slot.slot.week_index: slot for slot in catalog.slots}
    selected_slots = tuple(
        (identity, (slots_by_week[week].slot.slot_id,))
        for identity, week in zip(ASSIGNMENTS, DEMO_SLOT_WEEKS, strict=True)
    )
    available = dict(eligible_by_assignment)
    if any(
        slot_id not in available[identity]
        for identity, slot_ids in selected_slots
        for slot_id in slot_ids
    ):
        raise RuntimeError("synthetic demo slot is not deadline eligible")
    return catalog, selected_slots


@dataclass(frozen=True, slots=True)
class _DemoSearchProblem:
    """Small graph where UCS and greedy can return different path costs."""

    initial_state: str = "start"

    def is_goal(self, state: str) -> bool:
        return state == "goal"

    def actions(self, state: str) -> tuple[str, ...]:
        return {
            "start": ("cheap-1", "tempting"),
            "cheap-1": ("cheap-2",),
            "cheap-2": ("goal",),
            "tempting": ("goal",),
        }.get(state, ())

    def transition(self, state: str, action: str) -> str:
        del state
        return action

    def action_cost(self, state: str, action: str, next_state: str) -> float:
        del state, next_state
        return 5.0 if action == "tempting" else 1.0


def _search_scenarios() -> tuple[SearchScenario, ...]:
    """Return the shared synthetic Search cases and consistent A* heuristic."""
    greedy_values = {
        "start": 0.0,
        "cheap-1": 4.0,
        "cheap-2": 5.0,
        "tempting": 1.0,
        "goal": 0.0,
    }
    # Exact remaining path costs for this finite graph; therefore h(u) <=
    # cost(u, v) + h(v) for every edge and h(goal) = 0.
    astar_values = {
        "start": 3.0,
        "cheap-1": 2.0,
        "cheap-2": 1.0,
        "tempting": 1.0,
        "goal": 0.0,
    }
    return tuple(
        SearchScenario(
            f"capacity-{capacity}",
            _DemoSearchProblem(),
            greedy_values.__getitem__,
            {"configuration_status": "SYNTHETIC_DEMO_ONLY", "capacity": capacity},
            astar_values.__getitem__,
        )
        for capacity in (2, 3, 4)
    )


class _DemoOrdering:
    """Synthetic policy ordering sessions by ascending week index."""

    policy_identity = "synthetic-demo-ordering-v1"

    def required_pairs(
        self, sessions: tuple[SessionVariable, ...]
    ) -> tuple[SessionPair, ...]:
        by_assignment: dict[str, list[SessionVariable]] = {}
        for session in sessions:
            by_assignment.setdefault(session.assignment_id, []).append(session)
        return tuple(
            SessionPair(first, second)
            for group in by_assignment.values()
            for first, second in zip(sorted(group), sorted(group)[1:], strict=False)
        )

    def is_ordered(self, earlier: WeeklySlot, later: WeeklySlot) -> bool:
        return earlier.week_index < later.week_index


def _csp_scenarios() -> tuple[CspScenario, ...]:
    scenarios = []
    for slots_per_week in (2, 3, 4):
        assignments = tuple(
            AssignmentPlan(
                identity,
                AssignmentCandidateIdentity(
                    "demo-student",
                    "FFF",
                    identity,
                    (2, 0, 0, 0) if index == 0 else (1, 0, 0, 0),
                    f"demo-features-{identity}",
                    "synthetic-demo-mapping-v1",
                ),
                f"demo-features-{identity}",
            )
            for index, identity in enumerate(ASSIGNMENTS)
        )
        manifest = WholePlanManifest(
            "demo-student",
            "FFF",
            assignments,
            "synthetic-demo-mapping-v1",
        )
        variables = expected_session_variables(manifest)
        slots = tuple(
            WeeklySlot(f"demo-week-{week}-slot-{index + 1}", week)
            for week in (0, 1)
            for index in range(slots_per_week)
        )
        domains = tuple(VariableDomain(variable, slots) for variable in variables)
        eligible = tuple((variable, frozenset(slots)) for variable in variables)
        ordering = _DemoOrdering()
        constraints = build_required_constraints(
            manifest=manifest,
            variables=variables,
            eligible_slots=eligible,
            ordering_policy=ordering,
            weekly_capacity=slots_per_week,
        )
        problem = CspProblem(
            manifest, variables, domains, constraints, slots_per_week, ordering
        )
        policy = WholePlanVerificationPolicy(
            "synthetic-demo-v1",
            "synthetic-demo-deadline-v1",
            problem.ordering_policy.configuration_identity,
            f"synthetic-demo-{slots_per_week}-slots-per-week-v1",
            f"synthetic-demo-{slots_per_week}-slots-per-week",
            "synthetic-demo-slot-id-v1",
            slots_per_week,
        )
        scenarios.append(
            CspScenario(
                f"slots-per-week-{slots_per_week}",
                problem,
                policy,
                {
                    "configuration_status": "SYNTHETIC_DEMO_ONLY",
                    "physical_slots_per_week": slots_per_week,
                    "weekly_capacity_from_physical_slots": slots_per_week,
                },
            )
        )
    return tuple(scenarios)


def _replace_csp_status_rows(
    report_path: Path, records: tuple[ExperimentRunRecord, ...]
) -> None:
    """Refresh only the CSP status rows in the combined synthetic report."""
    if not report_path.is_file():
        return
    methods = sorted({record.method for record in records})
    if methods != ["bt", "bt_fc", "bt_fc_ac3"]:
        raise ValueError("CSP report update requires all three methods")
    status_order = ("feasible", "infeasible", "incomplete", "error")
    replacement = []
    for method in methods:
        counts = Counter(
            record.status.value for record in records if record.method == method
        )
        replacement.append(
            f"| csp | {method} | {sum(counts.values())} | "
            + " | ".join(str(counts[status]) for status in status_order)
            + " |"
        )
    lines = report_path.read_text(encoding="utf-8").splitlines()
    csp_indexes = [
        index for index, line in enumerate(lines) if line.startswith("| csp | ")
    ]
    if not csp_indexes:
        raise ValueError("combined report has no CSP status rows to update")
    first_csp = csp_indexes[0]
    filtered = [line for line in lines if not line.startswith("| csp | ")]
    filtered[first_csp:first_csp] = replacement
    report_path.write_text("\n".join(filtered) + "\n", encoding="utf-8")


def _synthetic_config(adapter: SessionNNAdapter) -> ApplicationInputConfig:
    catalog, eligible_by_assignment = _synthetic_demo_catalog()
    eligible = dict(eligible_by_assignment)
    rules = tuple(
        ApplicationAssignmentRule(
            identity,
            assessment_id,
            DEMO_TARGET_MARK,
            deadline,
            eligible[identity],
            AllocationBounds(0, 1),
        )
        for identity, assessment_id, deadline in zip(
            ASSIGNMENTS, ASSESSMENTS, DEADLINES, strict=True
        )
    )
    targets = tuple((identity, DEMO_TARGET_MARK) for identity in ASSIGNMENTS)
    return ApplicationInputConfig(
        OuladPresentationSplit("FFF", ("2013B", "2013J"), "2014J"),
        rules,
        TargetRule("synthetic-demo-targets", "v1", targets),
        catalog,
        SupportPolicy(
            "synthetic-demo-support",
            "v1",
            "strict-training-min-max",
            OutOfSupportAction.REPORT_ONLY,
        ),
        SessionClickMappingConfig(ClickScenario.LOW, "content_navigation"),
        ModelArtifactIdentity.from_adapter(adapter),
    )


def _prediction_scenarios(
    feature_rows: pd.DataFrame,
    student_ids: tuple[str, ...],
    adapters: dict[str, SessionNNAdapter],
    planner_results: dict[tuple[str, str], FinalPlannerResult],
) -> tuple[PredictionScenario, ...]:
    scenarios = []
    for student_id in student_ids:
        def run(
            method: str,
            _seed: int,
            *,
            identity: str = student_id,
        ) -> FinalPlannerResult:
            adapter = adapters[method]
            config = _synthetic_config(adapter)
            validated = build_application_inputs(
                config,
                student_identity=identity,
                feature_rows=feature_rows,
                adapter=adapter,
            )
            rules = _DemoOrdering()
            from study_planner.integration.application_inputs import (
                build_application_planning_problem,
            )

            binding = build_application_planning_problem(
                validated, adapter=adapter, ordering_policy=rules
            )
            result = integrated_uniform_cost_search(
                binding.allocation_space,
                binding.candidate_factory,
                adapter,
                csp_algorithm=CspAlgorithm.BT_FC_AC3,
            )
            planner_results[(identity, method)] = result
            return result

        scenarios.append(
            PredictionScenario(
                f"profile-{student_id}",
                student_id,
                {
                    "configuration_status": "SYNTHETIC_DEMO_ONLY",
                    "independent_execution_per_seed": True,
                    "deterministic_algorithmic_inputs": True,
                },
                run,
            )
        )
    return tuple(scenarios)


def _write_prediction_plans(
    output_directory: Path,
    records: Sequence[ExperimentRunRecord],
    planner_results: Mapping[tuple[str, str], FinalPlannerResult],
    student_ids: tuple[str, ...],
) -> tuple[Path, Path]:
    """Persist one detailed planner result per profile and predictor."""
    catalog, _eligible_by_assignment = _synthetic_demo_catalog()
    slot_catalog = [
        {
            "slot_id": slot.slot.slot_id,
            "week_index": slot.slot.week_index,
            "weekday_iso": slot.time_window.weekday_iso,
            "local_start": "09:00",
            "local_end": "10:00",
            "timezone": slot.time_window.timezone_identity,
            "relative_windows": dict(slot.relative_windows),
        }
        for slot in catalog.slots
    ]
    slots_by_id = {slot["slot_id"]: slot for slot in slot_catalog}
    details: list[dict[str, object]] = []
    for student_id in student_ids:
        for method in ("nn", "average"):
            run_record = next(
                record
                for record in records
                if record.profile_id == student_id
                and record.method == method
                and record.seed == DEFAULT_EXPERIMENT_SEEDS[0]
            )
            result = planner_results.get((student_id, method))
            if result is None:
                details.append(
                    {
                        "configuration_status": "SYNTHETIC_DEMO_ONLY_NOT_PRODUCTION",
                        "student_id": student_id,
                        "method": method,
                        "seed": run_record.seed,
                        "status": run_record.status.value,
                        "outcome": None,
                        "search_complete": None,
                        "allocations": [],
                        "weekly_timetable": None,
                        "predicted_marks": [],
                        "failure_reasons": list(run_record.failure_reasons),
                        "error_type": run_record.error_type,
                        "error_message": run_record.error_message,
                    }
                )
                continue

            details.append(
                {
                    "configuration_status": "SYNTHETIC_DEMO_ONLY_NOT_PRODUCTION",
                    "student_id": student_id,
                    "method": method,
                    "seed": run_record.seed,
                    "status": result.status.value,
                    "outcome": result.outcome.value,
                    "search_complete": result.search_complete,
                    "model_identity": result.nn_model_identity,
                    "model_mode": result.nn_model_mode.value,
                    "artifact_digest": result.nn_fitted_artifact_digest,
                    "allocations": [
                        {
                            "assignment_identity": item.assignment_identity,
                            "weekly_sessions": list(item.weekly_sessions),
                            "session_count": sum(item.weekly_sessions),
                        }
                        for item in result.allocations
                    ],
                    "weekly_timetable": (
                        None
                        if result.weekly_timetable is None
                        else [
                            {
                                "assignment_identity": schedule.assignment_identity,
                                "sessions": [
                                    {
                                        "session_index": session.session_index,
                                        **slots_by_id[session.slot_id],
                                    }
                                    for session in schedule.sessions
                                ],
                            }
                            for schedule in result.weekly_timetable
                        ]
                    ),
                    "predicted_marks": [
                        {
                            "assignment_identity": item.assignment_identity,
                            "predicted_mark": item.predicted_mark,
                            "target_mark": item.target_mark,
                        }
                        for item in result.predicted_marks
                    ],
                    "failure_reasons": list(result.failure_reasons),
                    "error_type": None,
                    "error_message": None,
                }
            )

    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "prediction_plans.json"
    csv_path = output_directory / "prediction_plans.csv"
    json_path.write_text(
        json.dumps(
            {
                "configuration_status": "SYNTHETIC_DEMO_ONLY_NOT_PRODUCTION",
                "profile_count": len(student_ids),
                "methods": ["nn", "average"],
                "run_seeds": list(DEFAULT_EXPERIMENT_SEEDS),
                "plans_are_deterministic_across_seeds": True,
                "slot_catalog": slot_catalog,
                "plans": details,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    csv_fields = (
        "student_id",
        "configuration_status",
        "method",
        "seed",
        "status",
        "outcome",
        "search_complete",
        "model_identity",
        "model_mode",
        "artifact_digest",
        "allocations_json",
        "weekly_timetable_json",
        "predicted_marks_json",
        "failure_reasons_json",
        "error_type",
        "error_message",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for detail in details:
            writer.writerow(
                {
                    **{
                        field: detail.get(field)
                        for field in csv_fields
                        if not field.endswith("_json")
                    },
                    "allocations_json": json.dumps(detail["allocations"]),
                    "weekly_timetable_json": json.dumps(
                        detail["weekly_timetable"]
                    ),
                    "predicted_marks_json": json.dumps(detail["predicted_marks"]),
                    "failure_reasons_json": json.dumps(detail["failure_reasons"]),
                }
            )
    return csv_path, json_path


def _write_demo_config(path: Path, profile_manifest: Path) -> None:
    catalog, eligible_by_assignment = _synthetic_demo_catalog()
    eligible = dict(eligible_by_assignment)
    payload = {
        "configuration_status": "SYNTHETIC_DEMO_ONLY_NOT_PRODUCTION",
        "profile_manifest": str(profile_manifest),
        "course": "FFF",
        "presentation": "2014J",
        "assignments": [
            {
                "assignment_identity": identity,
                "assessment_id": assessment_id,
                "target_mark": DEMO_TARGET_MARK,
                "deadline_day": deadline,
                "eligible_slots": list(eligible[identity]),
                "allocation_bounds": {"minimum_sessions": 0, "maximum_sessions": 1},
            }
            for identity, assessment_id, deadline in zip(
                ASSIGNMENTS, ASSESSMENTS, DEADLINES, strict=True
            )
        ],
        "slot_catalog": {
            "identity": catalog.identity,
            "version": catalog.version,
            "slots": [
                {
                    "slot_id": slot.slot.slot_id,
                    "week_index": slot.slot.week_index,
                    "weekday_iso": slot.time_window.weekday_iso,
                    "local_time": "09:00-10:00",
                    "timezone": slot.time_window.timezone_identity,
                    "relative_windows": dict(slot.relative_windows),
                }
                for slot in catalog.slots
            ],
            "weekly_capacity": catalog.weekly_capacity,
        },
        "target_rule": f"synthetic per-assignment target of {DEMO_TARGET_MARK:g}",
        "deadline_policy": "explicit slot lists; no calendar derivation",
        "ordering_policy": "ascending week index",
        "support_policy": "report-only marginal out-of-support status",
        "click_mapping": {
            "scenario": "low",
            "activity_group": "content_navigation",
            "clicks_per_session": 22,
            "window_rule": (
                "computed from each synthetic slot's week, weekday, "
                "and assignment deadline"
            ),
        },
        "seed_policy": list(DEFAULT_EXPERIMENT_SEEDS),
        "prediction_seed_execution": (
            "Execute the integrated planner independently for each profile, "
            "predictor, and protocol seed. The algorithms are deterministic, "
            "so outcomes may match; elapsed time is measured for every execution."
        ),
        "limitations": [
            "Synthetic planning policy values are for pipeline demonstration only.",
            "OULAD historical rows and features remain the empirical student inputs.",
            "The experiment does not estimate a causal effect of study sessions.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-directory", type=Path, default=root / "data/processed/strict"
    )
    parser.add_argument(
        "--feature-csv", type=Path, default=root / "data/processed/strict/fff_test.csv"
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
        default=root / "reports/experiment_results/demo",
    )
    parser.add_argument("--train-demo-nn", action="store_true")
    parser.add_argument(
        "--csp-only",
        action="store_true",
        help="run Experiment 2 only, preserving Search and prediction outputs",
    )
    parser.add_argument("--max-epochs", type=int, default=20)
    args = parser.parse_args(argv)

    if args.csp_only:
        if args.train_demo_nn:
            parser.error("--csp-only cannot be combined with --train-demo-nn")
        args.output_directory.mkdir(parents=True, exist_ok=True)
        csp_records = run_csp_comparison(_csp_scenarios())
        paths = write_experiment_bundle(
            csp_records, args.output_directory, "csp", "nodes_visited"
        )
        _replace_csp_status_rows(
            args.output_directory / "experiment_report.md", csp_records
        )
        print("Experiment 2 outputs:")
        for path in paths:
            print(path)
        return int(any(record.error_type for record in csp_records))

    if args.train_demo_nn:
        if args.artifact_directory.exists() and any(args.artifact_directory.iterdir()):
            parser.error(
                f"refusing to overwrite non-empty NN artifact directory: "
                f"{args.artifact_directory}"
            )
        run_training_experiment(
            args.processed_directory,
            args.artifact_directory,
            args.output_directory / "nn_demo_training.md",
            seeds=(42,),
            search_configs=(DEFAULT_SEARCH_CONFIGS[0],),
            max_epochs=args.max_epochs,
        )
    if not args.artifact_directory.is_dir():
        parser.error(
            "Model A Strict demo artifact is missing; rerun with --train-demo-nn"
        )

    manifest = json.loads(args.profile_manifest.read_text(encoding="utf-8"))
    student_ids = tuple(str(value) for value in manifest["student_ids"])
    if len(student_ids) != 30:
        parser.error("the shared profile manifest must contain exactly 30 students")
    feature_rows = pd.read_csv(args.feature_csv)
    mapping = SessionClickMappingConfig(ClickScenario.LOW, "content_navigation")
    adapters = {
        "nn": load_session_nn_adapter(
            args.artifact_directory, NNModelMode.MODEL_A_STRICT, mapping
        ),
        "average": load_simple_average_adapter(args.processed_directory, mapping),
    }
    _write_demo_config(
        args.output_directory / "demo_experiment_config.json",
        args.profile_manifest,
    )

    search_scenarios = _search_scenarios()
    output = args.output_directory
    search_records = run_search_comparison(search_scenarios)
    write_experiment_bundle(
        search_records,
        output,
        "search",
        "nodes_expanded",
    )
    csp_records = run_csp_comparison(_csp_scenarios())
    write_experiment_bundle(csp_records, output, "csp", "nodes_visited")
    planner_results: dict[tuple[str, str], FinalPlannerResult] = {}
    prediction_records = run_prediction_comparison(
        _prediction_scenarios(feature_rows, student_ids, adapters, planner_results)
    )
    write_experiment_bundle(
        prediction_records,
        output,
        "prediction",
        "mean_predicted_mark",
        group_by_scenario=False,
    )
    _write_prediction_plans(output, prediction_records, planner_results, student_ids)
    _write_demo_report(
        output / "experiment_report.md",
        {
            "search": search_records,
            "csp": csp_records,
            "prediction": prediction_records,
        },
        adapters,
        args.profile_manifest,
    )
    LOGGER.info("Synthetic demo experiment outputs written to %s", output)
    return 0


def _write_demo_report(
    path: Path,
    records_by_experiment: Mapping[str, Sequence[ExperimentRunRecord]],
    adapters: dict[str, SessionNNAdapter],
    profile_manifest: Path,
) -> None:
    """Write an aggregate report directly from the run records."""
    lines = [
        "# Synthetic Demo Experiment Report",
        "",
        "**Configuration status:** SYNTHETIC DEMO ONLY — not production evidence.",
        "",
        f"Profile manifest: `{profile_manifest}` (30 selected FFF 2014J students).",
        f"Experiment seeds: `{list(DEFAULT_EXPERIMENT_SEEDS)}`.",
        f"Demo NN artifact digest: `{adapters['nn'].fitted_artifact_digest}`.",
        "Average baseline artifact identity: "
        f"`{adapters['average'].model_provenance_identity}`.",
        "",
        "The Search and CSP runs execute the project algorithms on explicit small",
        "synthetic problems. Prediction runs use the actual strict profile rows,",
        "one demo-trained Model A artifact, and the training-only average baseline",
        f"with synthetic target marks of {DEMO_TARGET_MARK:g}, a three-slot demo",
        "catalog, low content-navigation click mapping, and report-only support.",
        "Prediction behavior is deterministic: each profile/predictor/seed is",
        "executed independently. Identical outcomes across seeds are expected;",
        "elapsed time is measured separately for each execution.",
        "",
        "## Run status counts",
        "",
        "| Experiment | Method | Runs | Feasible | Infeasible | Incomplete | Errors |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    status_order = (
        "feasible",
        "infeasible",
        "incomplete",
        "error",
    )
    for experiment_id, records in records_by_experiment.items():
        methods = sorted({record.method for record in records})
        for method in methods:
            counts = Counter(
                record.status.value
                for record in records
                if record.method == method
            )
            lines.append(
                f"| {experiment_id} | {method} | {sum(counts.values())} | "
                + " | ".join(str(counts[status]) for status in status_order)
                + " |"
            )
    lines.extend(
        (
            "",
            "Run-level and mean/sample-standard-deviation metric tables are in the",
            "matching CSV/JSON files. SVG comparisons are generated from those same",
            "run records. Per-profile allocations, verified timetables, and predicted",
            "marks are in `prediction_plans.csv` and `prediction_plans.json`. The",
            "weekly slot catalog in those files gives week, weekday, and UTC times.",
            "These synthetic results demonstrate the experiment pipeline; they must",
            "not be used as approved planner settings or causal evidence.",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
