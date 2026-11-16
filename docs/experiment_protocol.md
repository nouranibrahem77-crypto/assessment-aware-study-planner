# Experiment Protocol

**Status:** Contract draft for implementation review. Production inputs that
need owner approval are explicitly marked pending below.

This document defines the shared execution and reporting rules for the three
planner experiments. It does not supply production targets, slot policies, or
other business configuration that has not been approved.

## Shared protocol

### Student profiles

- Use one versioned manifest of 30 complete FFF 2014J student profiles for
  experiments that evaluate students.
- A profile is complete for selection when it has exactly one row for each of
  the three selected assignments. The selected model mode's existing schema
  validator determines whether each row's field values and missingness are
  acceptable before it is used in a run.
- Store the manifest at
  `reports/experiment_results/profiles_fff_2014J.json`. It records the ordered
  assessment IDs, selected student IDs, selection seed, selection method, and
  SHA-256 of the source feature CSV.
- The currently documented assessment IDs 34899, 34900, and 34901 are a
  candidate set pending owner approval. Do not run production comparisons
  against this set until the assessment selection is approved.
- Search-only and CSP-only synthetic benchmark runs do not need student rows;
  where the experiment compares planner outcomes or prediction models, use the
  same profile manifest for every method being compared.

### Seeds and reproducibility

- Use five fixed run seeds: **42, 123, 2025, 31415, and 271828**.
- Record every seed in every per-run result. Do not omit failed or incomplete
  runs from the run-level output.
- The profile manifest uses seed 42 for stable profile selection. The five
  experiment seeds govern only stochastic experiment behavior; they must not
  silently select a different set of students.
- Search, CSP, feature mapping, and tie-breaking remain deterministic for a
  fixed configuration. If a method has no stochastic behavior, its five
  seeded runs may return identical algorithmic results; record each run rather
  than adding artificial randomness.
- Preserve the source feature manifest hash, profile-manifest hash, model
  artifact digest, and experiment configuration identity in each run where
  applicable.

### Shared inputs and comparison rules

- Compare methods on identical profile IDs, assignment set, feature rows,
  target marks, deadline policy, eligible slot IDs, slot catalog, weekly
  capacity, allocation bounds, support disposition, click-mapping scenario,
  and resource limits.
- Use Model A Strict for the primary NN planner comparison. Model B may be
  reported as a separate sensitivity analysis only; its submission-based
  prior-mark availability remains an unverified proxy because OULAD has no
  grade-release timestamps.
- The complete production values and approval references remain pending in
  `configs/p1_production_template.json`. Until approved, results must be
  labelled synthetic or test-configuration results and must not be presented
  as production outcomes.
- Predictions describe model behavior under synthetic session-to-click
  perturbations. They do not establish that study sessions cause higher marks.

## Experiment 1 — Search comparison

**Question:** For the same finite allocation problem, how do UCS and Greedy
Best-First Search compare in search effort and returned allocation cost?

- **Methods:** Uniform Cost Search (UCS) and Greedy Best-First Search.
  A* is not included in the primary comparison unless the official assignment
  rubric confirms that it is required; the available compliance audit notes
  that this requirement is unresolved.
- **Inputs:** Identical deterministic allocation spaces, initial states,
  action costs, candidate evaluator, targets, CSP policy, and resource limits.
  Use the approved production scenarios when available; before then, label the
  scenarios as synthetic/test configurations.
- **Metrics:** status; nodes expanded; nodes generated; solution cost when a
  solution is found; candidate proposals; CSP nodes visited; schedules
  examined; target failures; verification rejections; elapsed search time.
- **Comparison:** Report paired results by scenario and seed. Compare success
  status, nodes expanded, and solution cost. UCS is the cost-optimal reference
  when it completes; Greedy cost is descriptive and is not assumed optimal.
- **Outputs:**
  `reports/experiment_results/search_runs.csv`,
  `reports/experiment_results/search_summary.csv`, and
  `reports/experiment_results/search_comparison.png`.

## Experiment 2 — CSP comparison

**Question:** How do propagation strategies affect search effort for matched
weekly scheduling capacities?

- **Methods:** Backtracking (BT), Backtracking with Forward Checking (BT+FC),
  and Backtracking with Forward Checking plus AC-3 (BT+FC+AC-3).
- **Inputs:** Identical CSP variables, domains, constraints, assignment
  manifests, ordering/value-ordering policies, and node limits. Run each
  matched instance at weekly capacities **2, 3, and 4**. Keep all other
  scheduling rules fixed within a comparison.
- **Metrics:** terminal status; whether the result is exhaustive; feasible
  schedule count; nodes visited; assignments attempted; backtracks;
  constraint checks; domain prunings; compatibility checks; solver time; wall
  time.
- **Comparison:** Pair methods on the same instance, capacity, and seed.
  First compare feasibility and completeness; then compare work counters and
  runtime. Do not treat an incomplete run as an infeasible result.
- **Outputs:**
  `reports/experiment_results/csp_runs.csv`,
  `reports/experiment_results/csp_summary.csv`, and
  `reports/experiment_results/csp_comparison.png`.

## Experiment 3 — Prediction model comparison

**Question:** How does the fitted NN compare with a Simple Average-Mark
Predictor when each is used to evaluate the same planner scenarios?

- **Methods:** Model A Strict NN and the planner-compatible Simple
  Average-Mark Predictor. The baseline mean must be fitted from training
  partition target marks only; validation/test targets must not be used to fit
  it.
- **Inputs:** The same 30 profiles, assignments, candidate allocation space,
  target marks, schedule catalog, CSP settings, click mapping, support policy,
  search algorithm, and resource limits. Match each run by profile and seed.
- **Metrics:** planner status; predicted mark and target per assignment;
  number of target-satisfying assignments; whether all assignment targets are
  met; solution cost/session count; search nodes; schedules examined; CSP
  work counters; elapsed time. Record model identity, feature schema, and NN
  artifact digest where applicable.
- **Comparison:** Report paired NN and average-baseline outcomes for each
  profile and seed. Compare feasibility/target satisfaction, allocation cost,
  and search/CSP work. Report predictive error metrics only when ground-truth
  held-out target marks are available and evaluated separately from the
  planner's target rule.
- **Outputs:**
  `reports/experiment_results/prediction_runs.csv`,
  `reports/experiment_results/prediction_summary.csv`, and
  `reports/experiment_results/prediction_comparison.png`.

## Run status, failures, and incomplete results

- Emit one run record for every requested experiment/scenario/method/profile
  and seed combination, including errors and resource-limited outcomes.
- Use these run statuses: `feasible`, `infeasible`, `incomplete`, and `error`.
  Preserve the planner/CSP terminal status and failure reasons in additional
  fields rather than collapsing them into the run status.
- A resource limit, timeout, or interrupted enumeration is `incomplete`, never
  `infeasible`. Retain the counters and any incumbent/best-so-far result the
  API supplies.
- A thrown exception or invalid output is `error`; record exception type and
  message. Continue other independent runs, and include failed runs in the
  denominator metadata for the experiment.
- Only aggregate numeric metrics over runs where that metric is defined. For
  each metric, report the valid sample count alongside the mean and sample
  standard deviation (`ddof=1`). If fewer than two values are defined, write
  the standard deviation as null; if none are defined, write both aggregates
  as null. Never encode undefined values as zero.
- Report completion counts by status next to aggregate metrics so excluded
  incomplete/error values remain visible.

## Result files and documentation

Store generated deliverables under `reports/experiment_results/`:

- `profiles_fff_2014J.json` — reproducible shared profile manifest.
- `*_runs.csv` — one row per requested run, including configuration, method,
  seed, status, metrics, provenance, and error details where applicable.
- `*_summary.csv` — grouped sample count, mean, standard deviation, and status
  counts for each reported metric.
- `*_comparison.png` — plots generated from the saved run-level data.
- `experiment_report.md` — protocols followed, approved/pending inputs,
  aggregate tables, plots, exceptions/incomplete counts, limitations, and exact
  reproduction commands.
- `results_manifest.json` — output schema version, command/config identity,
  profile/source hashes, seed list, model digest(s), and generated file hashes.

CSV is the tabular source of record for run-level and summary tables. JSON
manifests are the source of record for configuration, provenance, and profile
selection. Plots and Markdown reports must be reproducible from these saved
files and must not contain manually transcribed metrics.

## Reproduction commands

The runner command names and argument syntax will be added here when the
experiment runner is implemented. The profile manifest can currently be
generated with:

```powershell
python scripts/make_profiles.py `
  --feature-csv data/processed/strict/fff_test.csv `
  --course-module FFF `
  --presentation 2014J `
  --assessment-ids <approved-id-1> <approved-id-2> <approved-id-3> `
  --count 30 `
  --seed 42 `
  --output reports/experiment_results/profiles_fff_2014J.json
```

Replace the assessment placeholders only after that assessment set is
approved. The profile-selection command requires the feature data to have
already been generated.

## Pending decisions before production runs

- Approval of course presentation and three-assessment set.
- Target marks and target-rule identity/version.
- Deadline and slot-eligibility policies, slot catalog, capacity, and
  same-assignment ordering policy.
- Allocation bounds, support reference/disposition, and click-mapping policy.
- Student selection approval and exact NN artifact/mode.
- Whether A* is an official requirement and any additional rubric-required
  metrics or plots.
