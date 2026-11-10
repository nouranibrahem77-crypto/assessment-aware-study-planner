# P1 Search integration readiness audit

## Scope and official requirements

This audit uses the verified P1 requirements recorded in
`docs/search_design.md` and `docs/effort_score_mapping_audit.md`. Officially,
Search starts with zero sessions and adds one session per action; the NN
predicts assessment marks from previous marks and recent online-learning
clicks; documented generated rules provide session click increments, slot
capacity, and target marks; CSP schedules candidate sessions; monotonicity is
tested and violations reported; and the effort-mark relationship is
observational, not causal. UCS and Greedy are required; CSP includes
Backtracking, Forward Checking, and AC-3. Search must continue after rejected
candidate plans.

This document distinguishes evidence from proposed engineering choices and
remaining decisions. It makes no source or model changes.

## Existing feature and model evidence

The strict feature manifest is `data/processed/strict/fff_preprocessing.json`.
It identifies course FFF, strict mode, feature schema version 1, 25,229
training rows (2013B and 2013J), and a prediction cutoff at the start of
deadline day `D`. The click windows are `[D-28,D-21)`, `[D-21,D-14)`,
`[D-14,D-7)`, and `[D-7,D)`. Only events with day `< D` are included.

The exact 12 click features are:

| Activity group | Existing columns | Interpretation in pipeline |
| --- | --- | --- |
| `content_navigation` | `click_content_navigation_week_1` … `week_4` | Sum of clicks on navigation/material resource types. Most plausible proxy category for a synthetic study interaction, but not a measure of studying. |
| `assessment_activity` | `click_assessment_activity_week_1` … `week_4` | Questionnaire and quiz clicks; assessment interactions, not a generic study session. |
| `collaboration` | `click_collaboration_week_1` … `week_4` | Forum, wiki, and synchronous-collaboration clicks; possible group-study proxy, but not generic individual study. |

Each value is an aggregate of observed `studentVle.sum_click`; the feature
builder has no session boundary, duration, effort field, or session-to-click
rule. For a session assigned to relative week `w` and group `g`, the only
schema-preserving feature change is to `click_{g}_week_{w}`. Assessment type,
weight/availability, deadline, and weeks-since-start remain fixed. A session
after the target cutoff must not affect any feature.

`RegressionPredictor.predict(DataFrame)` validates the exact saved strict
feature columns, rejects identifiers and `target_score`, applies the saved
preprocessor and MLP seed ensemble, and clips output scores to `[0,100]`.
Strict inference does not receive session count and does not guarantee
monotonicity. The original schema/artifacts can remain unchanged by applying a
mapping to a copied row whose columns still match this strict schema.

## Training-only distributions and scenario anchors

The following summaries were calculated from
`data/processed/strict/fff_train.csv` only (25,229 rows). Validation and test
were not inspected for selecting these anchors. Percentiles use linear
interpolation. Counts and distributions are observational feature totals,
not observed session lengths.

| Feature (all four windows summarized per feature) | p25 | Median | p75 | p95 | p99 | Max | Zero fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Content navigation W1 | 1 | 27 | 85 | 214 | 342.72 | 3,796 | 21.67% |
| Content navigation W2 | 3 | 33 | 88 | 215.60 | 361 | 4,195 | 19.01% |
| Content navigation W3 | 8 | 42 | 95 | 213 | 343.72 | 736 | 14.36% |
| Content navigation W4 | 38 | 87 | 155 | 312 | 501 | 1,546 | 5.84% |
| Assessment activity W1 | 0 | 0 | 17 | 198 | 429 | 1,676 | 64.61% |
| Assessment activity W2 | 0 | 0 | 18 | 215 | 502 | 2,255 | 64.47% |
| Assessment activity W3 | 0 | 0 | 28 | 220 | 497 | 1,224 | 62.41% |
| Assessment activity W4 | 0 | 0 | 37 | 328.60 | 602 | 1,828 | 58.13% |
| Collaboration W1 | 0 | 0 | 10 | 58 | 137 | 997 | 50.66% |
| Collaboration W2 | 0 | 2 | 14 | 66 | 152 | 1,018 | 45.88% |
| Collaboration W3 | 0 | 3 | 15 | 60 | 127.72 | 786 | 41.59% |
| Collaboration W4 | 0 | 7 | 22 | 71 | 146 | 994 | 28.78% |

### Proposed sensitivity scenarios, not empirical session lengths

For an initial, explicitly synthetic experiment, use `content_navigation` as
the primary generic session category because it most directly represents
access to learning materials. It still does not establish study intent. Use
training-only positive weekly totals pooled across the four windows as
scenario anchors: p25 = 22, median = 62, and p75 = 121 clicks. These are
**click perturbation scenario values**, not claims that one real session
generates those clicks. The empirical positive-total anchors were computed
from 85,559 positive content-navigation window values. Do not tune these
values against validation/test scores.

For robustness, repeat with group-specific positive-total quantile anchors:

| Category | Positive training window totals (`n`) | p25 / median / p75 candidate increments |
| --- | ---: | ---: |
| Content navigation | 85,559 | 22 / 62 / 121 |
| Assessment activity | 37,939 | 11 / 62 / 145 |
| Collaboration | 58,804 | 5 / 12 / 29 |

The alternative groups are separate sensitivity scenarios, not evidence that
they represent an ordinary study session. Assessment activity could model an
assessment-practice session; collaboration could model a group-study session.
The session-to-category rule remains a domain assumption. A configuration
should record category, increment rule/value, week mapping, provenance, and
version. Values must be finite and non-negative. Do not sum all categories or
choose a category after seeing which makes the target reachable.

These anchors summarize complete observed weekly totals. They can be large for
a single synthetic session and cannot be interpreted as an empirically
measured per-session distribution. That is why low/central/high perturbations
are a sensitivity analysis, not a fitted or causal mapping.

## Support policy

Use the strict training split only to fit the support reference. Proposed
minimum policy:

1. For each click feature, retain the training minimum, maximum, and selected
   quantile envelope (the table above gives key summaries). Flag any candidate
   value outside the observed min/max as **outside marginal support**. A
   narrower quantile envelope can flag extrapolation risk but must not be
   called a hard impossibility boundary.
2. Also assess joint support across the 12 click dimensions. Per-feature bounds
   do not detect implausible combinations. Implement a documented
   train-reference distance/density check only after choosing and validating
   its metric and threshold; otherwise report joint support as unassessed.
3. Include assessment type and relative window context in support summaries
   where training sample sizes permit. Do not let validation/test determine
   support cutoffs.
4. Never clip a synthetic feature to a bound. Preserve the requested generated
   increment, record the raw candidate and support flags, and either mark the
   prediction exploratory or reject it according to a separately approved
   policy. The current requirement does not say whether out-of-support
   candidates are pruned.

Because repeated sessions add increments, an in-range one-session perturbation
may become out of support after later actions. Support must be checked for each
candidate, not just at configuration setup. A support flag is not a causal
validity test.

## Search and CSP timing alternatives

Session count alone does not identify any of the four NN weeks. CSP's final
schedule can change which deadline-relative click window receives the
synthetic increment. Three designs were considered:

| Design | Operation | Benefits | Risks / trade-offs |
| --- | --- | --- | --- |
| **A. Assumed weekly distribution during Search** | A fixed documented rule distributes each assignment's sessions over relative weeks before NN scoring; CSP schedules afterward. | Keeps NN-before-CSP order and makes every Search state immediately scoreable. | The assumed week distribution may disagree with CSP's actual schedule. Prediction and final schedule become inconsistent; changing distribution rules can change reachability. |
| **B. Schedule before NN evaluation** | For each allocation, CSP finds a concrete schedule; map actual session weeks to NN windows, then predict. | Features correspond to the candidate schedule; no provisional week allocation. | Reverses the specified Search → NN → CSP validation flow; expensive CSP calls for non-goal states; if schedule infeasible Search must continue before any NN score. Multiple feasible schedules can yield different marks and need a selection rule. |
| **C. Two-stage prediction and scheduling** | Stage 1 uses a declared provisional week-allocation rule to generate NN inputs and identify target candidates. CSP then schedules those candidates. Stage 2 remaps each feasible schedule's actual weeks and re-predicts; accept only if the final schedule-specific prediction still meets target. If CSP rejects or final prediction fails, record the reason and resume Search. | Preserves NN proposal then CSP verification, while ensuring accepted plan's final NN inputs match its scheduled weeks. Makes provisional/final disagreement visible. | Adds two predictions for candidates, requires a deterministic provisional rule and a policy for multiple CSP schedules. A provisional predicted goal may fail at stage 2; only final re-predicted candidates count as solutions. |

**Recommendation: C.** It best preserves the official interaction in which Search
proposes, NN predicts, and CSP verifies, while preventing an accepted plan from
being reported on feature weeks inconsistent with its schedule. The
provisional distribution is a documented simulation assumption; the final
accepted score must use CSP's actual weeks. Search must resume after either
CSP infeasibility or failure of the schedule-specific re-prediction. For
Greedy, its priority is based on the provisional prediction; this does not
make Greedy optimal. For UCS, the first final feasible goal remains minimum
sessions only if all lower-cost states and schedule-specific goals are
evaluated consistently.

An unresolved CSP detail is whether it returns one deterministic feasible
schedule or all feasible schedules. If multiple schedules exist, define
whether to accept the first canonical schedule or evaluate all and use a
documented selection rule; do not select the schedule that maximizes marks
without declaring that optimization objective.

## Previous-score availability and leakage audit

The source has `studentAssessment.date_submitted`, an integer day offset for
submission. It is not a score publication/release timestamp. The feature
builder's proxy path requires a valid, non-banked on-time score for an earlier
assessment and checks that its submission day is before the current deadline.
That establishes submission timing, not when a learner or planner could see
the mark. `assessments.date` is the assessment deadline, not score release.
The inspected OULAD schema and data dictionary provide no grade-release field.

Earlier scores can be reconstructed retrospectively by student, presentation,
and assessment from `studentAssessment.csv`, as proxy feature engineering
already does. They cannot currently be declared safely available at a later
prediction cutoff. Do not infer release from the submission timestamp or
assessment ordering. The current strict model excludes all prior-score
features (`previous_score_lags_fitted_from_training_presentations = 0`), and
its saved strict schema/inference validation reject them. Preserve that
contract. Prior-score inputs require an authoritative release timestamp or a
defensible, separately justified availability rule plus a leakage audit before
feature/model/schema changes.

## Monotonicity audit protocol

For fixed observed assessment context and fixed mapping configuration, compare
each state with its one-session successor. The successor relation must hold
all other allocation dimensions fixed and add only the configured feature
increment. Compare raw predicted score outputs (the existing predictor's
documented `[0,100]` output is retained as-is); do not clip feature values,
sort scores, or repair decreases. Define a numeric tolerance before the run;
a violation is a decrease larger than that tolerance. Record each violation
with context identifiers in audit-only output, parent/child allocations,
feature/group/window changed, increment, support flags, both scores, and delta.
Report tested pairs, violating pairs, violation rate, and delta summaries by
assignment, increment scenario, group, and window. Report support violations
separately; do not quietly discard them from the monotonicity denominator.

This measures whether the trained predictor responds monotonically to the
specified synthetic feature changes. It does not show that study sessions,
clicks, or effort cause higher marks. No monotonic constraint or post-hoc
correction should be applied to the existing MLP.

## Implementation dependencies and order

1. Freeze documented generated-rule configuration interfaces for click
   increments, session-to-window mapping, capacity, and target marks. Record
   source/provenance, version, deterministic behavior, and scenario labels;
   keep numeric choices configurable.
2. Define the session/week representation and provisional week-distribution
   rule, plus CSP's deterministic output and multiple-schedule policy.
3. Implement a pure mapping adapter that copies a validated strict row,
   changes only one allowed click feature per action, preserves schema, and
   returns provenance. It must not write datasets or artifacts.
4. Build training-only support summaries/checks and an explicit
   out-of-support policy. Keep validation for model/rule sensitivity analysis
   and test sealed until final evaluation.
5. Implement monotonicity pair generation/reporting over increasing
   allocations; retain all violations and support flags.
6. Integrate the two-stage Search → provisional NN → CSP → final-week mapping
   and NN re-evaluation loop. Rejected or failed candidates return to Search.
7. Implement/verify CSP Backtracking, Forward Checking, and AC-3; then UCS and
   Greedy against the stable evaluator contract. A* remains optional.
8. Separately resolve the previous-score availability conflict. Do not add
   prior-score columns or retrain/change artifacts before a defensible cutoff
   and leakage audit are approved.

## Unresolved assumptions

- Official generation rules and provenance for slot capacity and target mark,
  plus whether target criteria are per assignment or aggregate.
- Whether positive-total quantiles are acceptable synthetic click perturbation
  scenarios, and which activity category is approved for each session type.
- How an absolute planning week maps to the four deadline-relative windows,
  especially near the deadline and for missing deadlines.
- Support metric/threshold and whether out-of-support candidates are rejected
  or retained only as exploratory results.
- Monotonicity tolerance, context population, and whether a single session
  successor or every coordinate-wise comparable allocation is audited.
- Whether CSP exposes one schedule or multiple schedules and how schedules
  are ordered/selected without optimizing on predicted marks implicitly.
- Whether previous marks are available to the planner at the cutoff. Existing
  submission dates do not answer this; score-release metadata is absent.
- Whether the instructor accepts the strict-model previous-score discrepancy
  as a documented limitation pending leakage-safe evidence.

## Readiness conclusion

The current strict click schema can represent deterministic synthetic
increments without changing the model architecture or saved schema. Training
data supports transparent sensitivity anchors and support checks, but does not
identify per-session click lengths or causal score effects. The two-stage
architecture aligns predictions with final CSP weeks. Implementation should
wait for rule provenance, support policy, CSP scheduling semantics, and a
decision on the unresolved previous-mark input requirement. No source code,
datasets, model artifacts, or experiment results were changed for this audit.
