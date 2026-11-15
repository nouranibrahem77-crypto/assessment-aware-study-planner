# Project compliance and completion audit

Audit date: 2026-11-15  
Scope: repository state, the official P1 requirements supplied in the project
request, available project documents, locally present data and artifacts,
implementation contracts, tests, and Git history. The application-input phase
is implemented in the working tree and its focused tests have been run; final
verification results are recorded below.

## Audit limitation and headline

The official `IS Projects.pdf` is not present in the repository or tracked Git
history. This update treats the official P1 requirements explicitly supplied
by the project owner as authoritative for this audit. The PDF is still needed
to confirm that this list is complete and resolve conditional wording such as
whether A* is mandatory.

The repository has a broad, tested algorithmic foundation, local OULAD data,
three-seed neural-network reports, a tested Search–NN–CSP integration API,
and a typed validated application-input layer. It still lacks approved
production values for targets, calendar slots, and deadline eligibility, the
candidate/allocation-space wiring from those inputs, an executable end-to-end
demo, the requested P1 planner experiments, and final presentation/report
deliverables. `README.md` still says Search and integration are future work,
which no longer matches the implementation.

Status meanings:

1. **IMPLEMENTED AND VERIFIED** — implementation and relevant test or measured
   output evidence are present.
2. **IMPLEMENTED BUT NOT VERIFIED** — implementation exists, but the relevant
   real-input or required experiment evidence does not.
3. **PARTIALLY IMPLEMENTED** — only some of the requirement or an abstract
   contract exists.
4. **NOT IMPLEMENTED** — no adequate implementation or deliverable was found.

## Requirement-by-requirement audit

### A. Application inputs

| Requirement | Status | Existing evidence | Missing work and smallest next action |
| --- | --- | --- | --- |
| Previous marks and recent clicks | **PARTIALLY IMPLEMENTED** | `STRICT_FFF_FEATURE_NAMES` contains the three activity groups over four assessment-relative windows; Model B adds prior-assessment score/availability lags. `build_application_inputs` validates the selected artifact's feature schema and `load_application_inputs` uses the corresponding validated strict/proxy data loader. | Model A Strict intentionally has no previous-mark features, while Model B is a submission-based proxy. Select the intended P1 mode explicitly and verify real later-presentation profiles have the required historical inputs; do not merge the schemas. |
| Target rules | **IMPLEMENTED BUT NOT VERIFIED** | `src/study_planner/integration/application_inputs.py` requires a versioned `TargetRule`, exactly three explicit marks, and matches them to assignment identities. `tests/test_application_inputs.py` exercises mismatches. | Actual approved target values/rule source are not supplied, and no real target provider is wired to the planner. Obtain/version approved values and connect them to candidate construction. |
| Assessment deadlines and week conversion | **PARTIALLY IMPLEMENTED** | The input layer binds assessment IDs to supplied deadline days, checks source-row equality and `deadline_day / 7`, and the candidate factory uses each rule's eligible slot IDs as per-session CSP domains. | Approved deadline source, deadline-to-eligible-slot derivation/boundary, and real values remain user-supplied. Provide the approved eligibility list/policy; the implementation does not derive it from local clock fields or click windows. |
| Authoritative slot catalog and time windows | **PARTIALLY IMPLEMENTED** | `WeeklySlotCatalog` validates versioned slot identities, weekly capacity, timezone-qualified local weekday/time windows, and per-assignment click-window maps. The factory uses configured slots and capacity in CSP and passes distinct assignment mappings to final evaluation. | Production catalog and source are not supplied; all current catalog values are test configuration. Supply/version the authoritative catalog before production use. |
| Allocation bounds | **IMPLEMENTED BUT NOT VERIFIED** | `AllocationBounds` values now control Search candidate eligibility and upper-bound action generation in `application_planning.py`; candidate creation rejects states outside bounds. | Production values remain caller-supplied and have only explicit test coverage. Provide approved per-assignment min/max values. |
| Support policy | **PARTIALLY IMPLEMENTED** | `SupportPolicy` is applied to provisional and final mapping marginal statuses; joint support remains `UNASSESSED`, consistent with `session_mapping.py`. Tests cover explicit policy and invalid joint claims. | Approved support-reference identity and out-of-support disposition remain user-supplied. Supply the documented reference and select REPORT_ONLY or REJECT for production. |
| Exact fitted model artifact identity | **IMPLEMENTED AND VERIFIED** | `src/study_planner/nn/inference.py` hashes loaded schema/config/preprocessor/checkpoint bytes and binds integrity to the loaded predictor; `src/study_planner/search/nn_adapter.py` uses it in prediction and planner provenance. `tests/test_nn_adapter.py` covers loader identity and state mutation. | For demo/reproduction, identify the exact local or regenerated artifact directory in application configuration and preserve its digest in run output. Current model files are local ignored artifacts, not tracked deliverables. |

### B. End-to-end planner

| Requirement | Status | Existing evidence | Missing work and smallest next action |
| --- | --- | --- | --- |
| Real application inputs | **PARTIALLY IMPLEMENTED** | `application_inputs.py` validates later-presentation feature rows and exact Model A/B artifact identity. `application_planning.py` converts validated inputs into a deterministic one-session-per-action Search space and a candidate factory that applies allocation bounds, configured eligible-slot domains, weekly capacity, assignment-specific windows, support disposition, and whole-plan CSP verification. `tests/test_application_inputs.py` exercises the public end-to-end path in explicit TEST configuration for Model A and B, including zero-session assignments. | This is still not a production P1 run. User must provide approved targets/rule version; each assignment and assessment ID; deadline day/policy version/eligible slot IDs; min/max session bounds; versioned slot IDs/week indexes/local time windows/timezone/capacity/assignment-relative click windows; support policy/reference/action; mapping scenario/group/increment; course/train/test split and student feature rows; exact fitted artifact directory; and explicit same-assignment ordering policy. |
| Three assignment allocations | **IMPLEMENTED BUT NOT VERIFIED** | The planner enforces exactly three contexts/targets and returns three allocations. `tests/test_search_csp_integration.py` exercises three-assignment synthetic candidates, including zero-session entries. | Run the planner from real application inputs and save a reproducible output fixture or demonstration transcript. |
| Weekly timetable | **IMPLEMENTED BUT NOT VERIFIED** | `FinalPlannerResult.weekly_timetable` and the schedule bridge preserve three assignment schedules; integration tests verify schedules and empty assignments. | Verify conversion against the authoritative slot catalog using the real input provider. |
| Predicted marks | **IMPLEMENTED BUT NOT VERIFIED** | `AssignmentMark` records per-assignment targets, prediction identities, fitted artifact digest, and model provenance. The NN adapter has artifact-backed tests. | Exercise the complete planner with the persisted fitted Model A artifact and real rows/targets; report the resulting marks and context. |
| Search, NN, and CSP working together | **IMPLEMENTED BUT NOT VERIFIED** | `src/study_planner/integration/planner.py` and `schedule_bridge.py` orchestrate UCS/Greedy, lazy BT/FC/AC-3 schedules, whole-plan verification, and final NN evaluation. `tests/test_search_csp_integration.py` tests the combined flow with controlled predictors and synthetic CSP candidates. | Add one real-input end-to-end run that loads the actual artifact and goes through the public planner path. |

### C. P1 deliverables

| Requirement | Status | Existing evidence | Missing work and smallest next action |
| --- | --- | --- | --- |
| At least 30 student profiles | **PARTIALLY IMPLEMENTED** | `configs/p1_production_template.json` lists 30 deterministic later-test student IDs. For the candidate FFF 2014J assessments 34899/34900/34901, 914 students have one row each; all 30 listed candidates have all strict feature cells and all 12 click values present. | The assessment set and student selection still need project-owner approval, and no production planner run is recorded. Confirm the candidate set/student and run the planner. |
| Monotonicity experiment | **PARTIALLY IMPLEMENTED** | `src/study_planner/search/session_mapping.py` and `nn_adapter.py` detect/report monotonicity observations; unit tests exercise reporting. | No P1 experiment runner or report evaluates a declared set of allocations/scenarios and summarizes monotonicity violations. Define the protocol and run/report it on the selected profiles. |
| NN-sufficient but CSP-rejected demonstration | **PARTIALLY IMPLEMENTED** | Integration code continues after CSP-rejected candidates; synthetic tests cover rejected allocation followed by a valid one and invalid verification cases. | No dedicated reproducible scenario demonstrates provisional NN target sufficiency followed by exhaustive CSP rejection. Add it to the demo after real application rules are fixed, and report the separate NN/CSP outcomes. |

### D. Official experiments

No P1 experiment runner or experiment-specific output files were found. Existing
neural-network training reports are not substitutes for planner experiments.

| Requirement | Status | Existing evidence | Missing work and smallest next action |
| --- | --- | --- | --- |
| Experiment 1: UCS vs Greedy | **PARTIALLY IMPLEMENTED** | `src/study_planner/search/ucs.py`, `greedy.py`, and integration functions implement both. `tests/test_search.py` and integration tests verify behavior. | No benchmark scenarios, comparison runner, or comparative result table/plot. Build a deterministic scenario suite and collect agreed search/cost/time/planning metrics. |
| Experiment 1: A* if required | **NOT IMPLEMENTED** | `docs/search_design.md` calls A* optional; no A* source module was found. The official PDF is unavailable, so its mandatory status is unknown. | Check the official PDF first. If required, implement and test A*; otherwise explicitly mark it optional in the final report. |
| Experiment 2: BT vs BT+FC vs BT+FC+AC-3 at capacities 2, 3, 4 | **PARTIALLY IMPLEMENTED** | `src/study_planner/csp/backtracking.py`, `forward_checking.py`, and `ac3.py` provide the three configurations; `tests/test_csp*.py` covers correctness/equivalence on small CSPs. | No controlled capacity 2/3/4 performance experiment or comparative output. Create matched CSP instances and report nodes, checks/prunings, runtime, and outcomes per configuration/capacity. |
| Experiment 3: NN vs Simple Average Predictor inside planner | **NOT IMPLEMENTED** | `src/study_planner/nn/training.py` reports a Dummy mean baseline on held-out prediction rows. It is not a planner evaluator and does not run through Search/CSP. | Implement a planner-compatible average baseline with explicit baseline identity, then run the same planner scenarios/targets with both evaluators. |
| At least five seeds | **NOT IMPLEMENTED** for listed P1 experiments | `reports/nn_training.md`, `reports/nn_extended_training.md`, and `reports/previous_marks_mlp_comparison.md` use seeds 42, 123, and 2025 (three seeds). No planner-experiment seed runner exists. | Confirm whether seeds apply to all experiments in the official PDF; parameterize and run at least five as required. |
| Mean and standard deviation | **PARTIALLY IMPLEMENTED** | Neural training reports contain means and standard deviations across three seeds; `reports/nn_extended_training.md` reports validation aggregates. | Calculate the requested aggregates for each planner/search/CSP experiment across the required seeds. Do not report an SD from a single deterministic run. |
| Required experiment tables and plots | **PARTIALLY IMPLEMENTED** | NN reports include tables and local training curves; `reports/figures/` contains only `.gitkeep`. | Create comparison tables and requested plots for the P1 planner experiments, then integrate them into the final report. Exact required plot list awaits the PDF. |

### E. Final deliverables

| Requirement | Status | Existing evidence | Missing work and smallest next action |
| --- | --- | --- | --- |
| `run_demo.py` | **NOT IMPLEMENTED** | No `run_demo.py` or equivalent application entry point exists. `pyproject.toml` exposes data and training commands only. | Add a small reproducible CLI that loads the selected student/application inputs and fitted artifact, runs the planner, and prints a typed result. |
| Normal, edge, and impossible cases | **PARTIALLY IMPLEMENTED** | Unit/integration tests exercise normal, zero-session, empty-domain, rejected schedule, and resource-limit cases. | Expose three named runnable demo scenarios and capture their expected outcomes/output. |
| README and reproducibility | **PARTIALLY IMPLEMENTED** | `README.md` documents OULAD acquisition and NN training setup; `scripts/make_data.py` and `scripts/make_features.py` support data generation. | README is stale: it says planner Search/integration are future work. Add demo, experiment, artifact, seed, and output reproduction commands after the CLI/runner exist. |
| Final report, maximum 12 pages | **NOT IMPLEMENTED** | Several detailed Markdown technical reports exist under `reports/`; none is an integrated final report/PDF with verified page count. | Produce a concise report from the official rubric and experiment outputs; render and verify it is at most 12 pages. |
| Presentation (10 minutes + 5-minute demo) | **NOT IMPLEMENTED** | No slides or presentation script/deck found. | Build a 10-minute narrative plus a rehearsed 5-minute demo using the final runnable scenarios. |
| Individual contribution statements | **NOT IMPLEMENTED** | No contribution statement files found. | Obtain each team member’s contribution text and include it in the required submission format. This requires team input. |
| Git log | **PARTIALLY IMPLEMENTED** | Git history exists; latest commit is `5a29003` (`checkpoint: Model B and CSP foundation`, 2026-11-11), preceded by Search/NN/CSP feature commits. | Current CSP/integration/provenance implementation has substantial modified and untracked files (see Git-state section) and is absent from committed history. After team review, create the required final history/log and confirm whether submission needs commits or only an exported log. |

## Data, model artifacts, outputs, and structure

- The repository contains source packages under `src/study_planner/`, tests
  under `tests/`, two utility scripts under `scripts/`, docs under `docs/`, and
  Markdown reports under `reports/`. There is no application/demo package,
  experiment runner directory, project config content, final slide deck, or
  final report PDF.
- Raw OULAD CSVs and derived strict/proxy feature CSVs exist locally. They are
  ignored by `.gitignore` (`data/raw/*`, `data/processed/*`) and are not tracked.
  `README.md` provides source acquisition and feature-generation instructions.
- Local ignored fitted outputs exist under `data/processed/nn_experiment/`,
  `nn_experiment_60ep/`, and `nn_experiment_previous_marks/`; they include
  preprocessor/config/schema/evaluation files and three MLP checkpoints each.
  Strict Model A inference artifacts are tested; Model B is explicitly a
  submission-based availability proxy. Artifact files are not tracked, so a
  clean checkout needs the documented regeneration path or a separately
  supplied artifact package.
- `configs/p1_production_template.json` records verified local Model A Strict
  options (`data/processed/nn_experiment` and
  `data/processed/nn_experiment_60ep`) and Model B proxy path
  (`data/processed/nn_experiment_previous_marks`) with SHA-256 artifact,
  provenance, and schema identities measured through the current artifact
  loader. It selects no production model. All artifact directories are ignored
  by Git.
- Existing measured outputs are NN training/evaluation and feature-engineering
  reports. They do not contain the requested three planner experiments. No
  generated plots are present in `reports/figures/`.
- `docs/project_completion_status.md` now records the input-to-planner wiring
  and lists missing production values and experiments. `README.md` still
  contradicts the implementation status by describing planner search and
  integration as future work.

## Repository verification and Git state

Application-input-to-planner tests: **15 passed**. Full suite: **229 passed**, with three
existing sklearn warnings from test scalers fitted without feature names.
Ruff passed, and `git diff --check` passed. The worktree was already non-clean
before this phase, with CSP, Search/NN integration, and provenance files
modified or untracked; this audit does not attribute those pre-existing changes
to the application-input work.

The working tree is not clean. It contains modified CSP, Search/NN integration,
and documentation files, plus untracked FC/AC-3/verification/integration
modules and tests. These are existing working-tree changes, not commits made by
this audit. `git log` currently ends at `5a29003` on 2026-11-11; therefore that
log does not represent all current working-tree implementation. Do not treat
the committed history as proof the complete implementation is recorded.

## Shortest reliable completion order and estimates

Estimates are engineering hours, excluding waiting for rubric/team decisions or
external artifact access. Several tasks can be narrowed after reading the
official PDF. The order below minimizes rework by resolving inputs before
building the demo and experiments.

| Order | Task / smallest concrete result | Estimate |
| ---: | --- | ---: |
| 1 | Obtain/review official `IS Projects.pdf`; resolve whether A* and any unstated deliverables are mandatory. | 0.5–1 h |
| 2 | Approve and version actual target values, deadline boundary, slot catalog/time windows, allocation bounds, support reference, and selected model artifact. | 1–2 h plus decision time |
| 3 | Supply approved production values, configure the new allocation-space/candidate factory, and select at least 30 deterministic learners using held-out later-presentation rows. | 1–2 h wiring/configuration, plus decision time |
| 4 | Implement a planner-compatible Simple Average baseline and matched evaluator interface without weakening final provenance checks. | 2–4 h |
| 5 | Add `run_demo.py` with normal, edge, impossible, and NN-sufficient/CSP-rejected cases; exercise genuine artifact-backed planning. | 3–5 h |
| 6 | Implement and run Experiment 1 (UCS vs Greedy; add A* only if the PDF requires it), with at least five seeds/scenarios and search/planning metrics. | 3–5 h |
| 7 | Implement and run Experiment 2 (BT, BT+FC, BT+FC+AC-3; capacities 2, 3, 4) with matched instances and at least five seeds where randomness applies. | 3–5 h |
| 8 | Implement and run Experiment 3 (NN vs Simple Average in the same planner/scenarios), recording outcome, target, schedule, and resource statistics. | 3–5 h |
| 9 | Run the monotonicity protocol and summarize violations/support behavior over selected profiles and declared scenarios. | 2–3 h |
| 10 | Produce mean/SD tables, required plots, and integrated report; render and check the 12-page limit. | 3–5 h |
| 11 | Update README with environment, data/artifact regeneration, demo, experiments, seeds, and output commands; run clean-checkout reproducibility checks. | 1–2 h |
| 12 | Prepare/rehearse 10-minute presentation and 5-minute demo; collect individual contribution statements; export required Git log after approved implementation history is finalized. | 2–4 h |
| 13 | Final rubric cross-check, full tests/lint, verify artifacts are reproducible and outputs complete. | 1–2 h |

**Estimated remaining engineering effort: approximately 27.5–48 hours, plus
approval and artifact/data availability time.** This is
not a commitment; obtaining the official PDF may remove or add deliverables.
Experiments share input and runner work, so parallelization may reduce elapsed
calendar time but not the listed engineering effort substantially.

## Mandatory versus optional

- The official PDF is unavailable, so exact official mandatory/optional status
  cannot be certified. Items explicitly called required in the user-provided
  completion checklist are treated as mandatory for this plan pending PDF
  reconciliation.
- `docs/search_design.md` describes UCS and Greedy as required and A* as
  optional. Treat A* as optional only if the official PDF agrees.
- BT, BT+FC, and BT+FC+AC-3 are implemented and the internal integration
  readiness document describes them as required. The capacity comparison is
  not yet run.
- Model B / Previous Marks is a separately documented proxy sensitivity
  experiment, not a substitute for the listed planner-vs-average experiment.
  Whether Model B is an official deliverable must be checked against the PDF.
- Robustness extensions beyond the listed normal/edge/impossible cases, causal
  claims about clicks, and additional NN architectures are optional unless the
  official rubric says otherwise. Do not add them before required experiments
  and deliverables are complete.

## Immediate next action

Obtain the official `IS Projects.pdf` if it is available, approve/fill the
REQUIRED fields in `configs/p1_production_template.json`, and select one exact
Model A/B artifact and demo student. The input-to-planner factory is already
implemented; those approved values unblock the real demo and experiments.
