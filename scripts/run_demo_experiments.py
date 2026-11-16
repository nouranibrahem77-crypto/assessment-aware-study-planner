"""Run explicitly synthetic demo experiments; outputs are never production results."""

from __future__ import annotations

import argparse
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
    ApplicationSlot,
    ModelArtifactIdentity,
    OuladPresentationSplit,
    OutOfSupportAction,
    SupportPolicy,
    TargetRule,
    WeeklySlotCatalog,
    WeeklyTimeWindow,
    build_application_inputs,
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


def _demo_slots(*, shared_week: bool) -> tuple[WeeklySlot, ...]:
    return tuple(
        WeeklySlot(f"demo-slot-{index + 1}", 0 if shared_week else index)
        for index in range(3)
    )


def _csp_scenarios() -> tuple[CspScenario, ...]:
    scenarios = []
    for capacity in (2, 3, 4):
        assignments = tuple(
            AssignmentPlan(
                identity,
                AssignmentCandidateIdentity(
                    "demo-student",
                    "FFF",
                    identity,
                    (1, 0, 0, 0),
                    f"demo-features-{identity}",
                    "synthetic-demo-mapping-v1",
                ),
                f"demo-features-{identity}",
            )
            for identity in ASSIGNMENTS
        )
        manifest = WholePlanManifest(
            "demo-student",
            "FFF",
            assignments,
            "synthetic-demo-mapping-v1",
        )
        variables = expected_session_variables(manifest)
        slots = _demo_slots(shared_week=True)
        domains = tuple(VariableDomain(variable, slots) for variable in variables)
        eligible = tuple((variable, frozenset(slots)) for variable in variables)
        ordering = _DemoOrdering()
        constraints = build_required_constraints(
            manifest=manifest,
            variables=variables,
            eligible_slots=eligible,
            ordering_policy=ordering,
            weekly_capacity=capacity,
        )
        problem = CspProblem(
            manifest, variables, domains, constraints, capacity, ordering
        )
        policy = WholePlanVerificationPolicy(
            "synthetic-demo-v1",
            "synthetic-demo-deadline-v1",
            problem.ordering_policy.configuration_identity,
            "synthetic-demo-three-slots-v1",
            f"synthetic-demo-capacity-{capacity}",
            "synthetic-demo-slot-id-v1",
            capacity,
        )
        scenarios.append(
            CspScenario(
                f"weekly-capacity-{capacity}",
                problem,
                policy,
                {"configuration_status": "SYNTHETIC_DEMO_ONLY", "capacity": capacity},
            )
        )
    return tuple(scenarios)


def _synthetic_config(adapter: SessionNNAdapter) -> ApplicationInputConfig:
    slots = tuple(
        ApplicationSlot(
            slot,
            WeeklyTimeWindow(
                weekday_iso=index * 2 + 1,
                start_minute_local=9 * 60,
                end_minute_local=10 * 60,
                timezone_identity="UTC",
            ),
            tuple((identity, index + 1) for identity in ASSIGNMENTS),
        )
        for index, slot in enumerate(_demo_slots(shared_week=False))
    )
    catalog = WeeklySlotCatalog(
        "synthetic-demo-slots",
        "v1",
        2,
        "synthetic-demo-deadline",
        "v1",
        slots,
    )
    rules = tuple(
        ApplicationAssignmentRule(
            identity,
            assessment_id,
            DEMO_TARGET_MARK,
            deadline,
            (slots[index].slot.slot_id,),
            AllocationBounds(0, 1),
        )
        for index, (identity, assessment_id, deadline) in enumerate(
            zip(ASSIGNMENTS, ASSESSMENTS, DEADLINES, strict=True)
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
) -> tuple[PredictionScenario, ...]:
    scenarios = []
    for student_id in student_ids:
        cached_results = {}

        def run(
            method: str,
            _seed: int,
            *,
            identity: str = student_id,
            cache: dict[str, FinalPlannerResult] = cached_results,
        ) -> FinalPlannerResult:
            if method in cache:
                return cache[method]
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
            cache[method] = result
            return result

        scenarios.append(
            PredictionScenario(
                f"profile-{student_id}",
                student_id,
                {
                    "configuration_status": "SYNTHETIC_DEMO_ONLY",
                    "deterministic_seed_reuse": True,
                    "elapsed_time_reused_from_first_seed": True,
                },
                run,
            )
        )
    return tuple(scenarios)


def _write_demo_config(path: Path, profile_manifest: Path) -> None:
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
                "eligible_slots": [f"demo-slot-{index + 1}"],
                "allocation_bounds": {"minimum_sessions": 0, "maximum_sessions": 1},
            }
            for index, (identity, assessment_id, deadline) in enumerate(
                zip(ASSIGNMENTS, ASSESSMENTS, DEADLINES, strict=True)
            )
        ],
        "slot_catalog": {
            "slot_ids": ["demo-slot-1", "demo-slot-2", "demo-slot-3"],
            "weeks": [0, 1, 2],
            "weekday_iso": [1, 3, 5],
            "local_time": "09:00-10:00",
            "timezone": "UTC",
            "weekly_capacity": 2,
        },
        "target_rule": f"synthetic per-assignment target of {DEMO_TARGET_MARK:g}",
        "deadline_policy": "explicit slot lists; no calendar derivation",
        "ordering_policy": "ascending week index",
        "support_policy": "report-only marginal out-of-support status",
        "click_mapping": {
            "scenario": "low",
            "activity_group": "content_navigation",
            "clicks_per_session": 22,
            "window_rule": "demo-slot index maps to relative click window 1, 2, or 3",
        },
        "seed_policy": list(DEFAULT_EXPERIMENT_SEEDS),
        "prediction_seed_execution": (
            "The demo planner is deterministic; cache one evaluation per profile "
            "and predictor, then record that outcome for each protocol seed. "
            "Elapsed time is the first-seed elapsed time repeated in those rows."
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
    parser.add_argument("--max-epochs", type=int, default=20)
    args = parser.parse_args(argv)

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

    search_scenarios = tuple(
        SearchScenario(
            f"capacity-{capacity}",
            _DemoSearchProblem(),
            lambda state: {
                "start": 0.0,
                "cheap-1": 4.0,
                "cheap-2": 5.0,
                "tempting": 1.0,
                "goal": 0.0,
            }[state],
            {"configuration_status": "SYNTHETIC_DEMO_ONLY", "capacity": capacity},
        )
        for capacity in (2, 3, 4)
    )
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
    prediction_records = run_prediction_comparison(
        _prediction_scenarios(feature_rows, student_ids, adapters)
    )
    write_experiment_bundle(
        prediction_records,
        output,
        "prediction",
        "mean_predicted_mark",
        group_by_scenario=False,
    )
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
        "Prediction behavior is deterministic: each profile/predictor is evaluated",
        "once and its outcome is retained under each protocol seed. Elapsed time is",
        "the first evaluation time repeated in the seed rows.",
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
            "run records. These results demonstrate the experiment pipeline; they",
            "must not be used to make claims about approved planner settings or a",
            "causal effect of study sessions on marks.",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
