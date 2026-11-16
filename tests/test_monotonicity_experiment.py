"""Tests for the selected-profile monotonicity experiment."""

import csv
import hashlib
import json

import pytest

from study_planner.data.feature_engineering.constants import (
    IDENTIFIER_COLUMNS,
    STRICT_FFF_FEATURE_NAMES,
    TARGET_COLUMN,
)
from study_planner.eval.baseline import SimpleAverageMarkPredictor
from study_planner.eval.monotonicity_experiment import (
    run_monotonicity_experiment,
    write_monotonicity_results,
)
from study_planner.search.session_mapping import (
    ClickScenario,
    SessionClickMappingConfig,
)


def _fixture_files(tmp_path):
    feature_path = tmp_path / "fff_test.csv"
    student_id = "1001"
    assessment_ids = ("34899", "34900", "34901")
    headers = [*IDENTIFIER_COLUMNS, *STRICT_FFF_FEATURE_NAMES, TARGET_COLUMN]
    with feature_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        for assessment_id in assessment_ids:
            row = {name: "0" for name in headers}
            row.update(
                {
                    "code_module": "FFF",
                    "code_presentation": "2014J",
                    "id_student": student_id,
                    "id_assessment": assessment_id,
                    "assessment_type_cma": "0",
                    "assessment_type_exam": "0",
                    "assessment_type_tma": "1",
                    "assessment_weight": "10",
                    "assessment_weight_available": "1",
                    "assessment_deadline_day": "24",
                    "weeks_since_course_start": str(24 / 7),
                    "target_score": "60",
                }
            )
            writer.writerow(row)
    source_digest = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "course_module": "FFF",
                "presentation": "2014J",
                "assessment_ids": list(assessment_ids),
                "feature_mode": "strict",
                "selection_seed": 42,
                "source_csv_sha256": source_digest,
                "student_ids": [student_id],
            }
        ),
        encoding="utf-8",
    )
    return feature_path, profile_path


def test_monotonicity_experiment_runs_matched_one_session_neighbors(tmp_path) -> None:
    feature_path, profile_path = _fixture_files(tmp_path)
    adapter = SimpleAverageMarkPredictor.fit([60.0, 80.0]).create_session_adapter(
        SessionClickMappingConfig(ClickScenario.LOW)
    )

    result = run_monotonicity_experiment(
        feature_path,
        profile_path,
        adapter,
        max_sessions=1,
        seeds=(42,),
    )

    run = result.runs[0]
    assert run.status == "complete"
    assert run.allocations_evaluated == 15
    assert run.supported_allocations == 15
    assert run.comparisons == 12
    assert run.violations == ()
    assert result.summaries[0].pooled_violation_rate == 0.0


def test_monotonicity_experiment_writes_reproducible_outputs(tmp_path) -> None:
    feature_path, profile_path = _fixture_files(tmp_path)
    adapter = SimpleAverageMarkPredictor.fit([75.0]).create_session_adapter(
        SessionClickMappingConfig(ClickScenario.LOW)
    )
    result = run_monotonicity_experiment(
        feature_path, profile_path, adapter, max_sessions=1, seeds=(42,)
    )

    outputs = write_monotonicity_results(result, tmp_path / "results")

    assert all(path.is_file() for path in outputs)
    report = outputs[-1].read_text(encoding="utf-8")
    assert "one-session allocation increase" in report
    payload = json.loads(outputs[1].read_text(encoding="utf-8"))
    assert payload["model_family"] == "simple_average_mark"
    assert len(payload["runs"]) == 1


def test_monotonicity_experiment_rejects_profile_source_hash_mismatch(tmp_path) -> None:
    feature_path, profile_path = _fixture_files(tmp_path)
    manifest = json.loads(profile_path.read_text(encoding="utf-8"))
    manifest["source_csv_sha256"] = "0" * 64
    profile_path.write_text(json.dumps(manifest), encoding="utf-8")
    adapter = SimpleAverageMarkPredictor.fit([70.0]).create_session_adapter(
        SessionClickMappingConfig(ClickScenario.LOW)
    )

    with pytest.raises(ValueError, match="source hash differs"):
        run_monotonicity_experiment(feature_path, profile_path, adapter, seeds=(42,))
