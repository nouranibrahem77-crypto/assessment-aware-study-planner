"""Run the documented synthetic P1 coursework scenario for student profiles."""

from __future__ import annotations

import csv
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from study_planner.csp.verification import AssignmentSchedule
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
    build_application_planning_problem,
    build_synthetic_weekly_slot_catalog,
    load_application_inputs,
    load_session_nn_adapter,
)
from study_planner.integration.application_planning import (
    CatalogChronologicalOrderingPolicy,
)
from study_planner.integration.planner import (
    FinalPlannerResult,
    integrated_uniform_cost_search,
)
from study_planner.integration.schedule_bridge import CspAlgorithm
from study_planner.search.nn_adapter import (
    NNModelMode,
    SessionNNAdapter,
)
from study_planner.search.session_mapping import (
    ClickScenario,
    SessionClickMappingConfig,
)

DEFAULT_RUNS_DIRECTORY = Path("reports/coursework_runs")
PROFILE_DIRECTORY_NAME = "profiles"
EXPECTED_BOUNDARY_IDENTITY = "course-day-start-exclusive-v1"
EXPECTED_SLOT_ID_FORMULA = (
    "P1-SYNTH-W{week:02d}-D{ISO_weekday}-T{start_minute:04d}-{end_minute:04d}"
)
EXPECTED_COURSE_WEEK_ANCHOR = (
    "Synthetic week 0 begins Monday 00:00 UTC; this is not an OULAD calendar date."
)


class CourseworkConfigurationError(ValueError):
    """Raised when documented coursework configuration or local inputs are absent."""


@dataclass(frozen=True, slots=True)
class CourseworkScenario:
    """Typed scenario inputs shared across the 30 independent student runs."""

    config: ApplicationInputConfig
    adapter: SessionNNAdapter
    feature_directory: Path
    student_ids: tuple[str, ...]
    ordering_policy: CatalogChronologicalOrderingPolicy


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise CourseworkConfigurationError(
            f"cannot read configuration {path}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise CourseworkConfigurationError(
            f"invalid JSON in {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise CourseworkConfigurationError(
            f"configuration root must be an object: {path}"
        )
    return value


def _required_mapping(parent: dict[str, Any], key: str, source: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise CourseworkConfigurationError(f"missing object {source}.{key}")
    return value


def _required_path(root: Path, relative: str, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise CourseworkConfigurationError(f"missing {label} path")
    path = root / relative
    if not path.exists():
        raise CourseworkConfigurationError(f"{label} does not exist: {path}")
    return path


def load_coursework_scenario(project_root: Path) -> CourseworkScenario:
    """Build typed application inputs from the existing documented P1 scenario."""
    root = project_root.resolve()
    template_path = root / "configs/p1_production_template.json"
    draft_path = root / "configs/p1_production_draft.json"
    template = _read_json(template_path)
    draft = _read_json(draft_path)

    verified_data = _required_mapping(template, "verified_data", str(template_path))
    candidates = _required_mapping(
        template, "assessment_set_candidate_pending_approval", str(template_path)
    )
    demo = _required_mapping(draft, "APPROVED_DEMO_DECISIONS", str(draft_path))
    generated = _required_mapping(draft, "PROPOSED_GENERATED", str(draft_path))
    slot_proposal = _required_mapping(
        generated, "slot_catalog_proposal", "PROPOSED_GENERATED"
    )
    mapping_proposal = _required_mapping(
        generated, "session_click_mapping", "PROPOSED_GENERATED"
    )
    support_proposal = _required_mapping(
        generated, "support_policy", "PROPOSED_GENERATED"
    )
    verified = _required_mapping(draft, "VERIFIED", str(draft_path))
    model_record = _required_mapping(verified, "verified_model_artifact", "VERIFIED")

    demo_course = _required_mapping(demo, "course_and_split", "APPROVED_DEMO_DECISIONS")
    if verified_data.get("course_identity") != demo_course.get("course"):
        raise CourseworkConfigurationError(
            "template and documented coursework course identities differ"
        )
    if verified_data.get("testing_presentation") != demo_course.get("testing"):
        raise CourseworkConfigurationError(
            "template and documented coursework test presentations differ"
        )
    candidate_ids = tuple(str(item) for item in demo["candidate_assessment_ids"]["ids"])
    assignment_records = candidates.get("assignments")
    if not isinstance(assignment_records, list) or len(assignment_records) != 3:
        raise CourseworkConfigurationError(
            "the template must define three candidate assessments"
        )
    records_by_id = {str(item["assessment_id"]): item for item in assignment_records}
    if set(records_by_id) != set(candidate_ids):
        raise CourseworkConfigurationError(
            "draft candidate IDs do not match template assessment records"
        )

    target_records = demo["target_marks"].get("per_assessment")
    if not isinstance(target_records, list) or len(target_records) != 3:
        raise CourseworkConfigurationError(
            "the coursework scenario needs three per-assessment targets"
        )
    targets = {
        str(item["assessment_id"]): float(item["mark"]) for item in target_records
    }
    if set(targets) != set(candidate_ids):
        raise CourseworkConfigurationError(
            "target IDs do not match the three configured assessments"
        )

    capacity = int(demo["weekly_slot_capacity"]["sessions_per_week"])
    if capacity != int(slot_proposal["weekly_capacity"]):
        raise CourseworkConfigurationError(
            "scenario and slot-catalog capacities differ"
        )
    if (
        mapping_proposal.get("scenario") != "central"
        or int(mapping_proposal.get("clicks_per_hypothetical_session", -1)) != 62
        or mapping_proposal.get("activity_group") != "content_navigation"
    ):
        raise CourseworkConfigurationError(
            "configured mapping is not the documented central content-navigation "
            "scenario"
        )

    boundary_identity = str(slot_proposal["deadline_boundary_identity"])
    if boundary_identity != EXPECTED_BOUNDARY_IDENTITY:
        raise CourseworkConfigurationError(
            f"unsupported configured deadline boundary: {boundary_identity}"
        )
    if slot_proposal.get("slot_id_formula") != EXPECTED_SLOT_ID_FORMULA:
        raise CourseworkConfigurationError(
            "slot ID formula is not supported by the existing catalog builder"
        )
    if slot_proposal.get("course_week_anchor") != EXPECTED_COURSE_WEEK_ANCHOR:
        raise CourseworkConfigurationError(
            "configured synthetic course-week anchor is missing or unsupported"
        )

    time_windows = slot_proposal.get("weekly_time_windows")
    week_indices = slot_proposal.get("week_indices")
    if (
        not isinstance(time_windows, list)
        or not time_windows
        or not isinstance(week_indices, list)
    ):
        raise CourseworkConfigurationError(
            "slot proposal must include weekly windows and week indices"
        )
    first_window = time_windows[0]
    window_signature = (
        int(first_window["start_minute_local"]),
        int(first_window["end_minute_local"]),
        str(first_window["timezone_identity"]),
    )
    if any(
        (
            int(window["start_minute_local"]),
            int(window["end_minute_local"]),
            str(window["timezone_identity"]),
        )
        != window_signature
        for window in time_windows
    ):
        raise CourseworkConfigurationError(
            "weekly slot windows cannot be represented by the existing pattern builder"
        )
    if len(time_windows) != capacity:
        raise CourseworkConfigurationError(
            "the documented weekly catalog must contain one slot per capacity unit"
        )

    assignment_deadlines = tuple(
        (assessment_id, int(records_by_id[assessment_id]["deadline_day"]))
        for assessment_id in candidate_ids
    )
    catalog, eligible_by_assignment = build_synthetic_weekly_slot_catalog(
        identity=str(slot_proposal["catalog_identity"]),
        version=str(slot_proposal["version"]),
        weekly_capacity=capacity,
        deadline_policy_identity=boundary_identity,
        deadline_policy_version="v1",
        assignment_deadlines=assignment_deadlines,
        week_indices=tuple(int(value) for value in week_indices),
        pattern=SyntheticWeeklySlotPattern(
            weekdays_iso=tuple(int(window["weekday_iso"]) for window in time_windows),
            start_minute_local=window_signature[0],
            end_minute_local=window_signature[1],
            timezone_identity=window_signature[2],
            course_start_weekday_iso=1,
        ),
    )
    if (
        len(catalog.slots) != 42
        or len({slot.slot.slot_id for slot in catalog.slots}) != 42
    ):
        raise CourseworkConfigurationError(
            "existing slot rules did not produce the documented 42-slot catalog"
        )
    configured_eligible = slot_proposal.get("deadline_eligible_slot_ids")
    if not isinstance(configured_eligible, dict) or {
        key: tuple(value) for key, value in configured_eligible.items()
    } != dict(eligible_by_assignment):
        raise CourseworkConfigurationError(
            "derived deadline-eligible IDs differ from the existing documented catalog"
        )

    click_mapping = SessionClickMappingConfig(
        ClickScenario(mapping_proposal["scenario"]), mapping_proposal["activity_group"]
    )
    adapter = load_session_nn_adapter(
        _required_path(root, str(model_record["relative_path"]), "Model B artifact"),
        NNModelMode.MODEL_B_PREVIOUS_MARKS,
        click_mapping,
    )
    if adapter.model_mode is not NNModelMode.MODEL_B_PREVIOUS_MARKS:
        raise CourseworkConfigurationError("loaded NN artifact is not Model B Proxy")
    for field, actual in (
        ("artifact_digest_sha256", adapter.fitted_artifact_digest),
        ("model_provenance_identity", adapter.model_provenance_identity),
        ("feature_schema_identity", adapter.feature_schema_identity),
    ):
        if model_record.get(field) != actual:
            raise CourseworkConfigurationError(
                f"loaded model {field} does not match the documented artifact"
            )

    feature_directory = _required_path(
        root, str(model_record["feature_directory"]), "matching proxy feature directory"
    )
    if feature_directory != _required_path(
        root,
        str(demo["matching_feature_directory"]),
        "configured proxy feature directory",
    ):
        raise CourseworkConfigurationError(
            "Model B artifact and feature-data paths do not match"
        )
    if (
        not (feature_directory / "fff_test.csv").is_file()
        or not (feature_directory / "fff_preprocessing.json").is_file()
    ):
        raise CourseworkConfigurationError(
            "proxy test features or preprocessing manifest are missing"
        )

    profile_section = _required_mapping(template, "demo_profiles", str(template_path))
    student_ids = tuple(
        str(value) for value in profile_section["first_30_candidate_student_ids"]
    )
    if len(student_ids) != 30 or len(set(student_ids)) != 30:
        raise CourseworkConfigurationError(
            "template must provide exactly 30 unique profile IDs"
        )
    if profile_section["assessment_ids_used_for_intersection"] != list(candidate_ids):
        raise CourseworkConfigurationError(
            "30-profile cohort does not match configured assessments"
        )

    assignment_rules = tuple(
        ApplicationAssignmentRule(
            assignment_identity=assessment_id,
            assessment_id=assessment_id,
            target_mark=targets[assessment_id],
            deadline_day=int(records_by_id[assessment_id]["deadline_day"]),
            eligible_slot_ids=tuple(dict(eligible_by_assignment)[assessment_id]),
            allocation_bounds=AllocationBounds(
                0, len(dict(eligible_by_assignment)[assessment_id])
            ),
        )
        for assessment_id in candidate_ids
    )
    target_rule = TargetRule(
        identity="APPROVED_DEMO_DECISIONS.target_marks",
        version=str(slot_proposal["version"]),
        targets=tuple(
            (assessment_id, targets[assessment_id]) for assessment_id in candidate_ids
        ),
    )
    reference_identity = str(support_proposal["reference_identity"])
    support_version = reference_identity.rsplit("-", maxsplit=1)[-1]
    support_policy = SupportPolicy(
        identity=reference_identity,
        version=support_version,
        reference_identity=reference_identity,
        out_of_support_action=OutOfSupportAction(
            support_proposal["out_of_support_action"].lower()
        ),
    )
    config = ApplicationInputConfig(
        presentation_split=OuladPresentationSplit(
            course_identity=str(demo_course["course"]),
            training_presentations=tuple(demo_course["training"]),
            testing_presentation=str(demo_course["testing"]),
        ),
        assignment_rules=assignment_rules,
        target_rule=target_rule,
        slot_catalog=catalog,
        support_policy=support_policy,
        click_mapping=click_mapping,
        model_identity=ModelArtifactIdentity.from_adapter(adapter),
    )
    ordering_policy = CatalogChronologicalOrderingPolicy(catalog)
    return CourseworkScenario(
        config, adapter, feature_directory, student_ids, ordering_policy
    )


def _schedule_records(
    schedules: tuple[AssignmentSchedule, ...] | None,
) -> dict[str, list[dict[str, int | str]]]:
    if schedules is None:
        return {}
    return {
        schedule.assignment_identity: [
            {
                "session_index": session.session_index,
                "slot_id": session.slot_id,
                "week_index": session.week_index,
            }
            for session in schedule.sessions
        ]
        for schedule in schedules
    }


def planner_result_record(
    student_id: str, result: FinalPlannerResult
) -> dict[str, Any]:
    """Serialize a planner result without conflating reachability and verification."""
    verification = result.whole_plan_verification
    verification_status = None if verification is None else verification.status.value
    schedule_verified = (
        verification_status == "feasible" and result.weekly_timetable is not None
    )
    allocations = {
        allocation.assignment_identity: {
            "weekly_sessions": list(allocation.weekly_sessions),
            "total_sessions": sum(allocation.weekly_sessions),
        }
        for allocation in result.allocations
    }
    marks = {
        mark.assignment_identity: {
            "predicted_mark": mark.predicted_mark,
            "target_mark": mark.target_mark,
            "target_met": mark.predicted_mark >= mark.target_mark,
        }
        for mark in result.predicted_marks
    }
    target_status = (
        "met"
        if marks and all(value["target_met"] for value in marks.values())
        else "not_met"
        if marks
        else "unknown"
    )
    statistics = getattr(result, "search_statistics", None)
    execution_metrics = (
        {
            "search_nodes_expanded": statistics.search.nodes_expanded,
            "search_nodes_generated": statistics.search.nodes_generated,
            "search_seconds": statistics.search.execution_time_seconds,
            "orchestration_seconds": statistics.orchestration_time_seconds,
            "csp_schedules_examined": statistics.schedules_examined,
            "csp_feasible_schedules": statistics.verification_feasible_schedules,
            "csp_verification_rejections": statistics.verification_rejections,
            "csp_solver_seconds": sum(
                outcome.statistics.solver_time_seconds
                for outcome in statistics.terminal_csp_outcomes
            ),
            "provisional_predictions": statistics.provisional_predictions,
            "final_predictions": statistics.final_predictions,
        }
        if statistics is not None
        else {}
    )
    return {
        "student_id": student_id,
        "planner_status": result.status.value,
        "planner_outcome": result.outcome.value,
        "schedule_verified": schedule_verified,
        "verification_status": verification_status,
        "allocations": allocations,
        "timetable": _schedule_records(result.weekly_timetable),
        "predicted_marks": marks,
        "target_status": target_status,
        "search_complete": result.search_complete,
        "failure_reasons": list(result.failure_reasons),
        "execution_metrics": execution_metrics,
    }


def run_one_profile(scenario: CourseworkScenario, student_id: str) -> dict[str, Any]:
    """Run the existing integrated planner for one validated OULAD profile."""
    inputs = load_application_inputs(
        scenario.feature_directory,
        scenario.config,
        student_identity=student_id,
        adapter=scenario.adapter,
    )
    binding = build_application_planning_problem(
        inputs,
        adapter=scenario.adapter,
        ordering_policy=scenario.ordering_policy,
    )
    result = integrated_uniform_cost_search(
        binding.allocation_space,
        binding.candidate_factory,
        scenario.adapter,
        csp_algorithm=CspAlgorithm.BT_FC,
    )
    return planner_result_record(student_id, result)


def _error_record(student_id: str, error: Exception) -> dict[str, Any]:
    return {
        "student_id": student_id,
        "planner_status": "error",
        "planner_outcome": "execution_error",
        "schedule_verified": False,
        "verification_status": None,
        "allocations": {},
        "timetable": {},
        "predicted_marks": {},
        "target_status": "unknown",
        "search_complete": False,
        "failure_reasons": [f"{type(error).__name__}: {error}"],
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, int]:
    outcomes = [record["planner_outcome"] for record in records]
    return {
        "profiles_requested": len(records),
        "profiles_processed": len(records),
        "successful_plans": sum(
            record["planner_outcome"] == "target_achieved"
            and record["schedule_verified"]
            for record in records
        ),
        "unreachable_targets": outcomes.count("target_unreachable"),
        "no_feasible_schedule": outcomes.count("no_feasible_schedule"),
        "incomplete_runs": outcomes.count("incomplete"),
        "execution_failures": outcomes.count("execution_error")
        + outcomes.count("error"),
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "student_id",
        "planner_status",
        "planner_outcome",
        "schedule_verified",
        "verification_status",
        "allocations",
        "timetable",
        "predicted_marks",
        "target_status",
        "search_complete",
        "failure_reasons",
        "execution_metrics",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: json.dumps(record[key], sort_keys=True)
                    if isinstance(record[key], dict | list)
                    else record[key]
                    for key in columns
                }
            )


def _atomic_create_json(path: Path, payload: dict[str, Any]) -> None:
    """Create a JSON file atomically without replacing any existing artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _is_checkpoint_eligible(payload: object, student_id: str) -> bool:
    """Return whether a result is complete and safe to checkpoint or resume."""
    if not isinstance(payload, dict) or payload.get("student_id") != student_id:
        return False
    required = {
        "planner_status",
        "planner_outcome",
        "schedule_verified",
        "verification_status",
        "allocations",
        "timetable",
        "predicted_marks",
        "target_status",
        "search_complete",
        "failure_reasons",
        "execution_metrics",
    }
    if not required <= payload.keys():
        return False
    outcome = payload["planner_outcome"]
    target_achieved = outcome == "target_achieved"
    if (
        not isinstance(payload["search_complete"], bool)
        or not isinstance(payload["planner_status"], str)
        or not isinstance(outcome, str)
        or not isinstance(payload["schedule_verified"], bool)
        or not isinstance(payload["target_status"], str)
        or payload["planner_status"] in {"error", "incomplete"}
        or outcome in {"error", "execution_error", "incomplete"}
        or (target_achieved and payload["planner_status"] != "feasible")
        or (target_achieved and payload["target_status"] != "met")
        or (target_achieved and payload["schedule_verified"] is not True)
        or (target_achieved and payload["verification_status"] != "feasible")
        or (
            target_achieved
            and not all(
                payload[field]
                for field in ("allocations", "timetable", "predicted_marks")
            )
        )
        or (not target_achieved and payload["search_complete"] is not True)
        or (
            payload["verification_status"] is not None
            and not isinstance(payload["verification_status"], str)
        )
        or not isinstance(payload["allocations"], dict)
        or not isinstance(payload["timetable"], dict)
        or not isinstance(payload["predicted_marks"], dict)
        or not isinstance(payload["failure_reasons"], list)
        or not all(isinstance(reason, str) for reason in payload["failure_reasons"])
        or not isinstance(payload["execution_metrics"], dict)
        or not all(
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in payload["execution_metrics"].values()
        )
    ):
        return False
    return True


def _valid_profile_record(path: Path, student_id: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if _is_checkpoint_eligible(payload, student_id) else None


def _find_completed_profile(directory: Path, student_id: str) -> dict[str, Any] | None:
    for path in sorted(directory.glob(f"{student_id}-*.json")):
        record = _valid_profile_record(path, student_id)
        if record is not None:
            return record
    return None


def _write_csv_new(path: Path, records: list[dict[str, Any]]) -> None:
    """Write a combined CSV through an exclusive temporary path."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_csv(temporary, records)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def execute_coursework_profiles(
    project_root: Path,
    run_directory: Path | None = None,
) -> dict[str, Any]:
    """Run or resume profiles in an isolated directory of immutable checkpoints."""
    scenario = load_coursework_scenario(project_root)
    if run_directory is None:
        DEFAULT_RUNS_DIRECTORY.mkdir(parents=True, exist_ok=True)
        run_directory = DEFAULT_RUNS_DIRECTORY / (
            f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        )
        run_directory.mkdir(exist_ok=False)
    else:
        run_directory.mkdir(parents=True, exist_ok=True)
    profile_directory = run_directory / PROFILE_DIRECTORY_NAME
    profile_directory.mkdir(exist_ok=True)
    manifest_path = run_directory / "run_manifest.json"
    manifest = {
        "scenario": "documented_synthetic_p1_coursework",
        "student_ids": list(scenario.student_ids),
        "course": scenario.config.presentation_split.course_identity,
        "assignments": [
            rule.assessment_id for rule in scenario.config.assignment_rules
        ],
        "target_marks": {
            rule.assessment_id: rule.target_mark
            for rule in scenario.config.assignment_rules
        },
        "weekly_capacity": scenario.config.slot_catalog.weekly_capacity,
        "slot_count": len(scenario.config.slot_catalog.slots),
        "model_mode": scenario.adapter.model_mode.value,
        "artifact_digest_sha256": scenario.adapter.fitted_artifact_digest,
    }
    if manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CourseworkConfigurationError("run manifest is unreadable") from error
        if existing_manifest != manifest:
            raise CourseworkConfigurationError(
                "run directory belongs to a different scenario"
            )
    else:
        _atomic_create_json(manifest_path, manifest)

    records: list[dict[str, Any]] = []
    for student_id in scenario.student_ids:
        completed = _find_completed_profile(profile_directory, student_id)
        if completed is not None:
            records.append(completed)
            print(f"Skipping completed student {student_id}.", flush=True)
            continue
        print(f"Running planner for student {student_id}...", flush=True)
        started = time.perf_counter()
        try:
            record = run_one_profile(scenario, student_id)
        except Exception as error:  # preserve one visible record per profile
            record = _error_record(student_id, error)
        record.setdefault("execution_metrics", {})
        record["execution_metrics"]["profile_wall_seconds"] = (
            time.perf_counter() - started
        )
        records.append(record)
        if _is_checkpoint_eligible(record, student_id):
            checkpoint = profile_directory / f"{student_id}-{uuid.uuid4().hex}.json"
            _atomic_create_json(checkpoint, record)
        print(
            f"Finished student {student_id}: {record['planner_outcome']}",
            flush=True,
        )
    completed_records = [
        _find_completed_profile(profile_directory, student_id)
        for student_id in scenario.student_ids
    ]
    all_profiles_completed = all(record is not None for record in completed_records)
    final_records = (
        [record for record in completed_records if record is not None]
        if all_profiles_completed
        else records
    )
    payload = {
        "scenario": "documented_synthetic_p1_coursework",
        "synthetic_settings": {
            "course": scenario.config.presentation_split.course_identity,
            "assignments": [
                rule.assessment_id for rule in scenario.config.assignment_rules
            ],
            "target_marks": {
                rule.assessment_id: rule.target_mark
                for rule in scenario.config.assignment_rules
            },
            "weekly_capacity": scenario.config.slot_catalog.weekly_capacity,
            "slot_count": len(scenario.config.slot_catalog.slots),
            "session_click_scenario": scenario.config.click_mapping.scenario.value,
            "session_activity_group": scenario.config.click_mapping.activity_group,
            "clicks_per_hypothetical_session": (
                scenario.config.click_mapping.clicks_per_session
            ),
            "model_mode": scenario.adapter.model_mode.value,
            "artifact_digest_sha256": scenario.adapter.fitted_artifact_digest,
            "feature_directory": str(scenario.feature_directory),
        },
        "summary": _summary(final_records),
        "profiles": final_records,
    }
    json_output = run_directory / "coursework_planner_30_profiles.json"
    csv_output = run_directory / "coursework_planner_30_profiles.csv"
    if all_profiles_completed and not json_output.exists():
        _atomic_create_json(json_output, payload)
    if all_profiles_completed and not csv_output.exists():
        _write_csv_new(csv_output, final_records)
    return payload


__all__ = [
    "CourseworkConfigurationError",
    "CourseworkScenario",
    "execute_coursework_profiles",
    "load_coursework_scenario",
    "planner_result_record",
    "run_one_profile",
]
