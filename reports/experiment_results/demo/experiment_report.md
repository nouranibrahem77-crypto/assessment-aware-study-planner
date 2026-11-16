# Synthetic Demo Experiment Report

**Configuration status:** SYNTHETIC DEMO ONLY — not production evidence.

Profile manifest: `C:\Users\janam\OneDrive\Desktop\3 sem 1\Intelligent systems\project 1\assessment-aware-study-planner\reports\experiment_results\profiles_fff_2014J.json` (30 selected FFF 2014J students).
Experiment seeds: `[42, 123, 2025, 31415, 271828]`.
Demo NN artifact digest: `ab6d3a07b9fc4768e2e5341583f9a8e49c8e03e0520b26cf3daa9d7372e03e9a`.
Average baseline artifact identity: `0b3325d145d56a8aad8a48df2203b6db4d47bdf70b4e8256db00e312d09457a0`.

The Search and CSP runs execute the project algorithms on explicit small
synthetic problems. Prediction runs use the actual strict profile rows,
one demo-trained Model A artifact, and the training-only average baseline
with synthetic target marks of 60, a three-slot demo
catalog, low content-navigation click mapping, and report-only support.
Prediction behavior is deterministic: each profile/predictor is evaluated
once and its outcome is retained under each protocol seed. Elapsed time is
the first evaluation time repeated in the seed rows.

## Run status counts

| Experiment | Method | Runs | Feasible | Infeasible | Incomplete | Errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| search | greedy_best_first | 15 | 15 | 0 | 0 | 0 |
| search | ucs | 15 | 15 | 0 | 0 | 0 |
| csp | bt | 15 | 10 | 5 | 0 | 0 |
| csp | bt_fc | 15 | 10 | 5 | 0 | 0 |
| csp | bt_fc_ac3 | 15 | 10 | 5 | 0 | 0 |
| prediction | average | 150 | 150 | 0 | 0 | 0 |
| prediction | nn | 150 | 150 | 0 | 0 | 0 |

Run-level and mean/sample-standard-deviation metric tables are in the
matching CSV/JSON files. SVG comparisons are generated from those same
run records. These results demonstrate the experiment pipeline; they
must not be used to make claims about approved planner settings or a
causal effect of study sessions on marks.
