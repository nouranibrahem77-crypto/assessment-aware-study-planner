# Assessment-Aware Study Planner

## Final Project Report

**Project:** Intelligent Systems Project

**Report date:** 17 November 2026

## Problem

The planner addresses allocation of study sessions for three upcoming assessments. It predicts marks from historical OULAD features, proposes session allocations across eligible weeks, and checks whether each allocation can be placed in a timetable that respects the configured slot, deadline, ordering, and capacity constraints. A plan meets its goal only when a concrete schedule passes CSP verification and final predicted marks meet every configured target. Prediction alone cannot establish timetable feasibility. [1–4]

The inputs combine retrospective OULAD student features with explicit planning settings. Demo and experiment targets, slot calendars, capacities, and session-to-click changes are synthetic configurations, not observed student schedules or approved course policy. If Search completes without finding a target-reaching plan, the planner can return its best final-scored, supported, verified plan. A run stopped by a resource limit is incomplete and does not establish that a target is unreachable. The predicted relationship between study activity and marks is observational/simulated, not causal. [1, 2, 4, 11]

## Design Choices

Prediction, allocation search, and timetable scheduling are separate components: the NN estimates marks, Search proposes session counts and weeks, and CSP determines whether sessions can occupy eligible concrete slots. This distinguishes a promising predicted mark from a usable verified plan. Model A Strict is the primary mode and excludes previous marks. Model B is a separate proxy sensitivity comparison because its submission-based prior marks may not have been released by the next prediction cutoff. [1–3, 5]

UCS assigns unit cost to each added session, making path cost equal to plan size; it prioritizes smaller allocations and, under finite bounds and correct goal evaluation, finds a minimum-session goal. Greedy Best-First is also available in integrated planning; A* is used in the separate search comparison rather than integrated orchestration. CSP exposes BT, BT+FC, and BT+FC+AC-3 separately. FC prunes values when constraints have one unassigned variable; AC-3 propagates explicit binary constraints and does not make higher-scope constraints arc-consistent. No method is assumed to dominate runtime on every problem. [1, 4, 6–8]

The chronological split (2013B/2013J train, 2014B validation, 2014J test) and deadline-day feature cutoff constrain temporal leakage. Strict features use clicks before the start of the target deadline day and exclude prior scores. Training configurations record seeds and artifact settings; inference binds predictions to the loaded model, schema, candidate, and CSP verification identity. These choices support repeatable evaluation and prevent unverified predictions from satisfying the planner goal. [2–4, 6]

## Neural Network (NN)

Two MLP modes predict `target_score`, a mark in [0, 100]. Model A Strict uses assessment metadata and recent clicks, with 19 source features (17 model inputs; 18 transformed inputs). Model B Previous-Marks Proxy adds five prior-score lags and five availability indicators, with 29 source features (27 model inputs; 33 transformed inputs). The fixed chronological row counts are 25,229 train, 9,013 validation, and 14,241 test. Both saved experiments record seeds 42, 123, and 2025; test metrics below are three-seed ensemble results, not the mean of separate seed metrics. [2–5]

The selected MLP uses 64- and 32-unit hidden layers, 0.1 dropout, and Huber loss; configuration selection uses validation MAE, leaving test rows for final evaluation. [3–5]

| Test evaluation (n = 14,241) | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Model A Strict, seed ensemble | 10.061613 | 12.834290 | 0.074467 |
| Model A Dummy mean baseline | 10.488579 | 13.405465 | -0.009746 |
| Model A Ridge baseline | 10.240101 | 13.070490 | 0.040087 |
| Model B Previous-Marks Proxy, paired ensemble | 8.608332 | 11.328663 | 0.278883 |

Model A’s ensemble improves on its Dummy and Ridge baselines by these regression metrics. Model B’s test comparison uses the same paired rows but remains a proxy sensitivity result: a prior mark appears in 12,469 of 14,241 test rows (87.557%), while OULAD provides no grade-release timestamps to confirm it was visible by prediction time. MAE, RMSE, and R² are appropriate for numeric marks; they are not classification accuracy. Predictive association does not establish that clicks or prior marks cause later results. [3–5]

## Integration

The application selects Model A or Model B and binds it to the session-feature adapter. For each UCS allocation, the adapter maps proposed sessions to candidate click features and obtains provisional NN predictions before CSP verification. CSP then enumerates schedules and verifies each concrete timetable. Only a schedule that passes verification receives final NN predictions. Target misses or rejected schedules lead to another CSP schedule; once that allocation’s schedules are exhausted, Search can try another allocation. [6, 7]

```text
OULAD features + configured scenario -> UCS allocation -> provisional NN prediction
    -> CSP schedule enumeration and verification -> final NN prediction
Rejected schedule: next CSP schedule | CSP exhausted: return to UCS
```

CSP uses eligible slot domains and checks deadline eligibility, unique slot use, same-assignment ordering, and weekly capacity. BT, FC, and AC-3 are separate options. A target-achieving result may stop Search early. An unreachable-target result requires completed Search; node or candidate limits produce an incomplete result. When Search completes without meeting targets, the planner can return the best final-scored, supported, CSP-verified plan found. [1, 6–8]

## Limitations

OULAD is retrospective and observational, with day-level activity timestamps; it cannot establish that study clicks cause marks or generalization to other institutions or cohorts. Model B’s prior-score availability is unverified because release timestamps are absent. Session-to-click increments (22, 62, or 121 configured clicks per hypothetical session) are training-anchored synthetic assumptions, not measured session durations or effects. [3, 5, 12]

Planner experiments use synthetic targets, slot calendars/capacities, and click mappings even when profiles or features come from OULAD. Their results demonstrate behavior under those settings, not real student behavior, production policy, deployment, or intervention effectiveness. Search/CSP resource limits can leave a run incomplete; such a run cannot support an unreachable-target claim. The separate 30-profile coursework execution is also incomplete: its manifest lists 30 intended IDs but the recorded run contains only three per-profile result files. [9–13]

## Conclusions

The project implements OULAD feature preparation, two distinct mark-prediction modes, UCS allocation search, and CSP timetable verification. Model A’s saved ensemble improves on its Dummy and Ridge baselines on the chronological test split. Model B has better paired test metrics under its submission-based prior-mark proxy, with the grade-release limitation above. [2–5]

Experiment 1’s small synthetic search cases returned solution cost 3 for UCS and A*, versus 6 for Greedy. Experiment 2 recorded 45 feasible runs across BT, BT+FC, BT+FC+AC-3, and 2/3/4 physical slots per week; FC visited fewer nodes than BT in these scenarios, while AC-3 visited the same number as FC. Experiment 3 met its synthetic 60-mark target in all 150 runs per predictor, with mean predicted marks of 76.698 +/- 4.306 for the NN and 78.761 +/- 0.000 for Average-Mark. The target was met at zero sessions by both predictors, so this experiment does not show an NN planning advantage. These results demonstrate implemented component behavior under specified data and scenario settings; they do not establish causal study effects, real-world validation, or production deployment. [9–11]

The planner distinguishes target achievement with a verified schedule, target unreachability after exhaustive Search, and incomplete execution when a limit prevents completion. The cited 30-profile experiments are not evidence that the separate coursework batch completed. [6, 13]

## References

1. `docs/search_design.md` — project problem formulation, design and outcome semantics.
2. `reports/feature_engineering_strict.md` — OULAD strict features, chronological splits and cutoff.
3. `reports/nn_training.md` — Model A protocol, baselines, metrics and limitations.
4. `data/processed/nn_experiment/evaluation.json` — saved Model A evaluation metrics.
5. `reports/previous_marks_mlp_comparison.md`; `data/processed/nn_experiment_previous_marks/evaluation.json` — paired Model B comparison, coverage and limitations.
6. `src/study_planner/integration/planner.py` — integrated prediction, schedule evaluation, Search continuation and completion states.
7. `src/study_planner/integration/application_inputs.py`; `src/study_planner/integration/application_planning.py` — validated inputs, predictor binding and CSP problem construction.
8. `src/study_planner/integration/schedule_bridge.py`; `src/study_planner/csp/backtracking.py`; `src/study_planner/csp/forward_checking.py`; `src/study_planner/csp/ac3.py` — schedule enumeration and CSP methods.
9. `reports/experiment_results/experiment1_astar/search_summary.csv` — Experiment 1 run summaries.
10. `reports/experiment_results/demo/experiment_report.md`; `reports/experiment_results/demo/csp_summary.csv` — synthetic demo and Experiment 2 findings.
11. `reports/experiment_results/experiment3_nn_vs_average/experiment3_report.md`; `reports/experiment_results/experiment3_nn_vs_average/experiment3_method_summary.csv` — Experiment 3 protocol and results.
12. `reports/oulad_exploration.md` — dataset interpretation and generalization limitations.
13. `docs/project_compliance_audit.md`; `reports/coursework_runs/20261117-203921-d566b9d2/run_manifest.json` — coursework-run status and evidence.
