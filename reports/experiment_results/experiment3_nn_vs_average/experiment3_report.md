# Experiment 3: NN versus Simple Average-Mark Planner

**Configuration status:** synthetic coursework demonstration only.

Profiles: 30; seeds: `[42, 123, 2025, 31415, 271828]`.
Runs: 300 (30 profiles × 2 predictors × 5 seeds).
Each seed invokes the integrated planner independently. Fixed inputs make
algorithmic outcomes deterministic; runtime is measured on each invocation.
Seeds label five separate executions, not randomized planner behavior.
Per-profile seed standard deviations are therefore zero for deterministic
metrics; cohort standard deviations reflect differences across profiles,
with each profile represented once per seed.

## Matched configuration

- Course: FFF; test presentation: 2014J.
- Assignments: 34899, 34900, 34901.
- Per-assignment synthetic target: 60.0.
- Weekly capacity: 2; catalog slots: 3.
- Click mapping: low / content_navigation / 22 clicks per hypothetical session.
- Support policy: report_only.
- Profile manifest SHA-256: `20b9338e28c19af9ef3fb0d9cff1fe128d52584717182a044c260abcf4190ad1`.
- Feature CSV SHA-256: `40aaa5f029e0c002b75e06e0f009fe315d5765c83d01e621604bfc52b61d280f`.
Both predictors use the same feature rows, target rule, assignments,
slot catalog, support policy, click mapping, Search, and CSP mode. Only
the fitted predictor differs.

## Predictor and baseline methodology

- NN: strict Model A artifact `3e9080e6aee5b2cc42419316111d4fc62d2be7f92d10478a66b568ba538bca12`.
  It is loaded and integrity-verified; no retraining was performed.
- Simple Average-Mark: predicts the training-target mean (78.760950) for each assignment.
  Fitted on 25229 FFF training targets from the existing strict training split.
  Training-target digest: `212fbeaaec3856be06fd8e605337fc5697cc7f5c5f0c2b20e214b19d13228acd`.
  Source training-file digest: `ab1bf7bb93463b220763ae7de2f5928cf779751effb5f9711e6be1783767575d`.
  Validation/test scores and profile outcomes are excluded from fitting.
- Both are wrapped by the same SessionNNAdapter interface and passed to
  integrated Uniform Cost Search with Backtracking + Forward Checking
  + AC-3 verification.

## Results

| Predictor | Target achieved | Mean predicted mark ± SD | Mean session cost ± SD |
| --- | ---: | ---: | ---: |
| NN | 150 / 150 | 76.698 ± 4.306 | 0.000 ± 0.000 |
| Simple Average-Mark | 150 / 150 | 78.761 ± 0.000 | 0.000 ± 0.000 |

Raw per-profile, per-predictor, per-seed records are in
`experiment3_runs.csv` and `experiment3_runs.json`. The per-profile
mean/sample-standard-deviation table is `experiment3_summary.csv`; the
cohort-wide method mean/sample-standard-deviation table is
`experiment3_method_summary.csv`. Plots compare predicted marks, target
achievement, and solution cost.

All target, slot, and click values are documented synthetic scenario
assumptions. OULAD associations are observational and do not establish
that study sessions cause marks to change.
