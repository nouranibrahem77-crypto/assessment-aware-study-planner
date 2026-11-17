# Project Compliance Audit

Audit date: 2026-11-17
Scope: current repository files, local data and generated outputs, tests, and
Git history. The named professor PDF was not found in the repository or the
available `/home/nouran` project files during this audit. The checklist below
uses the exact requirements supplied for this audit; PDF-only wording remains
unverified. All statuses concern evidence in the repository, not approval of
synthetic configuration as production policy.

## Status key

- **COMPLETE** — the required implementation or deliverable is present and
  directly evidenced.
- **PARTIALLY COMPLETE** — implementation or evidence exists, but a material
  part of the stated requirement remains incomplete or limited to synthetic
  scenarios.
- **MISSING** — no adequate deliverable was found.
- **NOT VERIFIED** — evidence is insufficient to establish the requirement.

## Requirements audit

| Requirement | Status | Evidence and current finding | Remaining work |
| --- | --- | --- | --- |
| Modular source: data, NN, search, CSP, integration, eval, tests | COMPLETE | `src/study_planner/{data,nn,search,csp,integration,eval}/`; `tests/` | None identified for this requirement. |
| Reproducible OULAD preparation with fixed seed | COMPLETE | `src/study_planner/data/oulad.py`, `src/study_planner/data/feature_engineering/`, `scripts/make_data.py`, `scripts/make_features.py`; fixed chronological partitions and source hashes in generated `data/processed/fff_preprocessing.json`; stable profile selection seed recorded by `src/study_planner/eval/profiles.py` and `reports/experiment_results/profiles_fff_2014J.json`. Raw OULAD files are present locally but ignored by Git. | Include/acquire the official source data and run documented preparation in the submission environment; preserve generated manifest/hashes. |
| Complete data dictionary covering real and generated features | COMPLETE | `docs/data_dictionary.md` defines all five OULAD input tables, identifiers, generated strict/proxy features, windows, target, split and missingness conventions. | None identified. |
| Trained NN and baseline comparison | COMPLETE | `reports/nn_training.md`, local strict-model artifacts under `data/processed/nn_experiment*/`, `src/study_planner/eval/baseline.py`; planner-compatible NN versus Simple Average-Mark results in `reports/experiment_results/experiment3_nn_vs_average/`. Training artifacts are local and ignored by Git; experiment 3 uses actual OULAD profile rows with synthetic planning rules. | Package or regenerate the intended NN artifacts for a clean submission checkout and retain the artifact identity. Do not present synthetic planning assumptions as real policy. |
| UCS and Greedy Best-First, counters, known-answer tests; A* optional | COMPLETE | `src/study_planner/search/ucs.py`, `greedy.py`, shared counters in `core.py`; known-answer coverage in `tests/test_search.py` and `tests/test_search_csp_integration.py`. A* is implemented in `src/study_planner/search/astar.py` and included in Experiment 1, but it is not needed to satisfy the optional extension. | None for stated requirement. |
| Backtracking, Forward Checking, AC-3 separately switchable with counters | COMPLETE | `src/study_planner/csp/{backtracking,forward_checking,ac3,core}.py`; selection via `CspAlgorithm` in `src/study_planner/integration/schedule_bridge.py`; correctness/counter tests in `tests/test_csp*.py`; comparative counters in `reports/experiment_results/demo/csp_summary.csv`. | None for implementation. Coursework interpretation of experiment settings remains synthetic as noted below. |
| Integrated `python run_demo.py` | COMPLETE | Root `run_demo.py`; README gives exact scenario commands. `tests/test_run_demo.py` covers normal, edge, impossible, NN-provisional-accept/CSP-infeasible, and verification-rejected cases. The command takes a named scenario, e.g. `python run_demo.py NORMAL`, and uses TEST configuration. | Run the documented command in the final submission environment with the required local OULAD features and model artifact available. |
| README with pinned requirements and exact run commands | PARTIALLY COMPLETE | `README.md` has pinned dependencies referenced from `pyproject.toml`, data, train, lint/test, demo and experiment commands. Its opening still says Search/integration are future work, contradicting the current code and later demo section. README also explicitly says the integrated planner has no NN-disabled mode. | Correct the stale opening/status text and reconcile the required NN-on/off presentation demonstration with the actual available interface. |
| At least 5 normal, 3 edge, and 2 impossible tests | PARTIALLY COMPLETE | Broad tests exist across `tests/`; `tests/test_run_demo.py` explicitly verifies normal and edge behavior, exhaustive impossible-target reporting, NN-accepted/CSP-infeasible behavior, and verification rejection. The repository does not identify or demonstrate a count of five normal, three edge, and two impossible cases as separate rubric categories. | Map existing tests to the required categories and add only enough missing cases to reach each minimum; document the mapping. |
| Experiment 1: UCS vs Greedy vs A*; ≥5 runs/seeds, mean/SD, table and plot | COMPLETE | `scripts/run_experiment1.py`; `reports/experiment_results/experiment1_astar/search_runs.{csv,json}`, `search_summary.csv`, and SVG comparison plots. Run records use seeds 42, 123, 2025, 31415, 271828 and include all three methods. This is a small synthetic graph comparison, not a production planner benchmark. | Preserve the synthetic-scenario label in the report and use the resulting table/plots as algorithm evidence only. |
| Experiment 2: BT vs BT+FC vs BT+FC+AC-3 at 2, 3, 4 slots per week; ≥5 runs/seeds, mean/SD, table and plot | COMPLETE | Isolated command: `python scripts/run_demo_experiments.py --csp-only`. Raw records: `reports/experiment_results/demo/csp_runs.csv` and `reports/experiment_results/demo/csp_runs.json` contain 45 records (3 methods × 3 scenarios × 5 seeds): `bt`, `bt_fc`, `bt_fc_ac3`; `slots-per-week-2`, `slots-per-week-3`, `slots-per-week-4`; seeds 42, 123, 2025, 31415, 271828; all 45 outcomes are feasible. Each raw configuration records `physical_slots_per_week` as 2/3/4 and `weekly_capacity_from_physical_slots` equal to that count. Summary table: `reports/experiment_results/demo/csp_summary.csv`; the JSON summary contains 108 metric rows with means and standard deviations. Plot: `reports/experiment_results/demo/csp_comparison.svg`. Combined status rows: `reports/experiment_results/demo/experiment_report.md`. These are explicitly synthetic results, not real-world timetable evidence. Visited nodes are equal for BT+FC and BT+FC+AC-3 in all three scenarios (23/23, 157/157, 597/597); no claim is made that AC-3 reduced visited nodes beyond FC. | None for the stated experiment comparison. Retain the synthetic-setting label and measured AC-3 node-count result in submitted materials. |
| Experiment 3: NN vs Simple Average-Mark inside planner; ≥5 runs/seeds, mean/SD, table and plot | COMPLETE | `scripts/run_experiment3.py`; `reports/experiment_results/experiment3_nn_vs_average/experiment3_report.md`, config, per-run CSV/JSON, summary/method-summary tables and SVG plots. It records 30 OULAD profiles × 2 predictors × 5 seeds (300 rows), with means and standard deviations. The report states the seeds invoke deterministic matched runs and all target/slot/click rules are synthetic. This experiment output is distinct from the separate coursework 30-profile execution. | Preserve the synthetic planning-settings label; do not cite it as proof that the coursework 30-profile run is complete. |
| Final report, maximum 12 pages | MISSING | No final report PDF was found in the repository or searched project files. Technical Markdown reports are not a page-verified final report. | Produce the required concise report using the professor's required sections and verified experiment outputs; render and verify no more than 12 pages. |
| 10-minute presentation and 5-minute live demo: Search trace, CSP pruning with/without FC/AC-3, NN on/off | PARTIALLY COMPLETE | No presentation deck was found. Demo output/test in `tests/test_run_demo.py` prints the actual solution path, not an expansion-by-expansion Search trace. CSP comparison counters/plots exist in the demo experiment outputs. `README.md` states the integrated planner requires an NN adapter and has no NN-disabled mode. | Prepare the 10-minute presentation and 5-minute live demo. Provide the required Search trace and show CSP work with each configuration. Resolve or demonstrate a rubric-compliant NN-off comparison; current integrated demo cannot run with NN disabled. |
| Individual contribution statement (≤200 words per person) and Git log | PARTIALLY COMPLETE | Git history is available (`git log`); no individual contribution statements were found. The current worktree also includes changes not represented by committed history, so the existing log may not capture the full final contribution record. | Obtain one ≤200-word statement per contributor and include it in the submission. Export/verify the final Git log after the required implementation changes are recorded. |
| P1: OULAD dataset | COMPLETE | `data/raw/{courses,assessments,studentAssessment,studentVle,vle}.csv` are locally present; acquisition and validation are implemented in `src/study_planner/data/oulad.py`. Data are ignored by Git. | Ensure the submission can reproduce acquisition/preparation or otherwise provide the required dataset access. |
| P1: three upcoming assignments | PARTIALLY COMPLETE | Integration contracts enforce three assignment contexts (`src/study_planner/integration/application_inputs.py`, `application_planning.py`); demos/profile outputs use assessments 34899, 34900, 34901 in FFF 2014J. The demo target, deadlines, eligible slots, and calendar are explicitly synthetic/TEST. | Confirm the selected assessments are the required upcoming assignments and supply approved assignment targets and timetable/deadline settings. |
| P1: NN predicts marks from previous marks and recent clicks | PARTIALLY COMPLETE | Model A Strict uses assessment metadata and clicks but excludes prior marks; Model B Previous-Marks Proxy adds prior-score lags/availability features. `docs/data_dictionary.md` and `reports/feature_engineering_proxy.md` state OULAD has no release timestamp, so visibility is unverified. | State which model fulfills the requirement and retain the availability limitation. Do not describe the proxy as verified student-visible prior marks. |
| P1: Search proposes allocations | COMPLETE | `src/study_planner/integration/application_planning.py`, `planner.py`, and `run_demo.py`; integration tests cover allocation behavior. | None for implementation. |
| P1: CSP verifies timetables | COMPLETE | `src/study_planner/integration/planner.py`, `schedule_bridge.py`, `src/study_planner/csp/verification.py`; integration and verification tests cover complete schedules. | Use approved timetable inputs for any result described as production. |
| P1: Search continues after CSP rejection | COMPLETE | `src/study_planner/integration/planner.py`; continued-search tests in `tests/test_search_csp_integration.py` and `tests/test_run_demo.py`. The injected verification-rejection test is separately labeled and is not confused with CSP infeasibility. | None for implementation. |
| P1: best achievable marks when target is unreachable | COMPLETE | `run_demo.py IMPOSSIBLE`; assertions in `tests/test_run_demo.py` verify exhaustive search, best verified marks, and shortfall reporting. | Keep the result labeled with its synthetic TEST targets/settings. |
| P1: outputs for at least 30 student profiles | PARTIALLY COMPLETE | `reports/experiment_results/demo/prediction_plans.csv/.json` and Experiment 3 contain 30 OULAD profiles under synthetic settings. Separately, `reports/coursework_runs/20261117-203921-d566b9d2/run_manifest.json` lists 30 IDs, but only three per-profile result JSON files exist in that run directory; it is not complete coursework execution evidence. | Complete and verify the required coursework run for all 30 profiles with the approved settings; do not state that this has already completed. |
| P1: monotonicity test and violation reporting | COMPLETE | `scripts/run_monotonicity.py`; `reports/experiment_results/demo/monotonicity/monotonicity_report.md`, runs and summary CSV/JSON. Summary records five seeds, 30 profiles per seed, 9,000 comparisons and 2,400 violations overall. Report metadata records the historical manifest SHA-256 `ac6f645af10d5d6ed4085ec2e1946bda7bfa87b7266144592e28941a70f4373f`; provenance clarification documents equivalent JSON content/current line-ending hash. These are synthetic mapping audits. | Preserve the provenance note and describe the results as model-response diagnostics under simulated inputs. |
| P1: cause-versus-correlation limitations | COMPLETE | `docs/search_design.md`, `docs/experiment_protocol.md`, `reports/experiment_results/demo/experiment_report.md`, and monotonicity report state that OULAD is observational and synthetic clicks do not establish session effects. | Carry the limitation into the final report and presentation. |
| P1: demo where NN says enough but CSP rejects schedule | COMPLETE | Root `run_demo.py` scenario `NN_ACCEPTS_CSP_INFEASIBLE`; `tests/test_run_demo.py` asserts provisional NN targets pass and real CSP exhausts with no feasible schedule for both model modes. This is an explicitly TEST-configured demonstration, not production evidence. | Show the named TEST case in the live demo and keep the settings label visible. |

## Data and result interpretation

- The repository contains genuine OULAD source tables and derived feature
  artifacts locally. The selected profile rows used in the demo experiments
  come from the OULAD test presentation. Data and trained model artifacts are
  ignored by Git; a clean checkout needs the documented acquisition,
  preparation, and training path or separately supplied artifacts.
- Search/CSP benchmark graphs, target values, slot calendars, capacities,
  deadline/ordering rules, and click increments in the demo bundles are
  synthetic scenario settings. The README and experiment reports identify
  them as TEST/demo settings. They are not approved production policy.
- Experiment 2 uses four session variables across two weeks, with 2, 3, or 4
  physical slots per week. The weekly-capacity value is set from that physical
  slot count; there is no additional independent limit. All 45 recorded
  method/scenario/seed runs are feasible. BT+FC and BT+FC+AC-3 visited the same
  number of nodes for each scenario; AC-3 added compatibility checks but did
  not reduce visited nodes beyond FC in these results.
- Experiment 3's 300 rows represent repeated deterministic planner invocations
  over 30 profiles and two predictors with five recorded protocol seeds. The
  seed labels do not introduce randomized planner behavior. The separate
  coursework run manifest has 30 intended IDs but only three profile result
  files, so the 30-profile coursework execution remains incomplete.
- The monotonicity summary contains 150 profile-seed runs and 9,000 adjacent
  comparisons, with 2,400 recorded decreases. Its historical and current
  manifests differ by line endings but parse to identical profile data, as
  documented in `reports/experiment_results/demo/monotonicity/monotonicity_report.md`.

## Exact remaining mandatory deliverables

1. **Complete the 30-profile coursework execution** with approved inputs and
   retain a result for every required profile. Evidence of the incomplete run:
   `reports/coursework_runs/20261117-203921-d566b9d2/run_manifest.json` lists
   30 profiles while the run directory currently contains only three profile
   result files.
2. **Approve and use the required assignment and timetable settings** for any
   final planner run: the current scenarios use generated targets, deadlines,
   slot calendars, capacities, click mapping, and support settings. Evidence:
   `reports/experiment_results/demo/demo_experiment_config.json` and
   `reports/experiment_results/demo/experiment_report.md` label them synthetic.
3. **Finish documentation and test-category reconciliation.** Update the stale
   planner-status statement in `README.md`; map tests to at least 5 normal, 3
   edge, and 2 impossible cases and add only missing cases. Evidence:
   `README.md` opening and `tests/test_run_demo.py`.
4. **Resolve the NN-off live-demo requirement.** Current integrated execution
   requires an NN adapter. Evidence: `README.md` integration-demo section and
   `run_demo.py`.
5. **Prepare the 10-minute presentation and 5-minute live-demo material**
   showing Search trace and CSP pruning with BT, BT+FC, and BT+FC+AC-3.
   Evidence: no slide deck is present; current Search demo reports a solution
   path, not an expansion trace; CSP comparison evidence is in
   `reports/experiment_results/demo/csp_summary.csv` and
   `csp_comparison.svg`.
6. **Produce the final report and verify the 12-page maximum.** Evidence: no
   final report PDF is present in the repository.
7. **Collect individual contribution statements** of at most 200 words per
   contributor and export/verify the final Git log after final work is
   recorded. Evidence: Git history exists, but no contribution statement was
   found and the current working-tree state may not be represented in that
   history.
8. **Confirm the source of the named official PDF and reconcile any additional
   wording** beyond the requirements supplied for this audit. The PDF was not
   found in the repository or searched project files, so its full text could
   not be independently checked.

## Optional extension

A* is implemented and included in Experiment 1, but the supplied requirement
calls it optional. It is not counted as a remaining mandatory deliverable.
