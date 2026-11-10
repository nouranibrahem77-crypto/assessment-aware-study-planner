# Study effort-to-score mapping feasibility audit

## Executive finding

The official P1 rule calls for generated session-to-click increments,
generated capacity/target rules, and monotonicity testing with violations
reported. These rules make a deterministic simulation possible; they do not
turn inputs into observed effort or establish a causal effort-mark
relationship. The strict FFF MLP can score synthetic feature rows matching its
schema, but has no session input and does not guarantee monotonic predictions.

This blocks interpreting Search outputs as evidence about real study effects,
but not implementing a deterministic simulation once its rules are supplied.
Results must be described as the NN prediction under a documented synthetic
click perturbation, not as the effect of undertaking a session.

## Official requirements and design status

**Official P1 requirements:** the NN uses previous marks and recent online
learning clicks; each Search action adds one session; a documented generated
rule supplies its click increment, slot capacity, and target mark; monotonicity
is tested with violations reported; and the effort-mark relationship is
observational, not causal.

**Proposed decisions in this audit:** isolate click generation in a
deterministic, versioned mapping component; modify only copied strict feature
rows; apply a training-derived support check; and report monotonicity deltas
without forcing the existing MLP to be monotonic.

**Unresolved:** numeric rule values, support/rejection policy, monotonicity
tolerance/order, and leakage-safe availability of previous marks. No numeric
session length is selected here.

## What the repository supports

The strict feature builder creates, for assessment deadline day `D`, three
activity-group totals in four half-open windows: `[D-28,D-21)`,
`[D-21,D-14)`, `[D-14,D-7)`, and `[D-7,D)`. Groups are
`content_navigation`, `assessment_activity`, and `collaboration`; each value
is the sum of observed `studentVle.sum_click` for mapped VLE types. The cutoff
is the start of day `D`, so a hypothetical session can only be placed in a
window strictly before `D` to affect these features. The other strict inputs
are assessment type, weight/availability, and deadline-derived values; these
do not change when a session is added. Strict mode excludes all prior-score
features.

In principle, an externally specified hypothetical session could be assigned
to one of the four windows and a VLE activity group, then represented as an
increment to that group's click total. That is only a data transformation,
not evidence that a session generates a particular number or type of clicks.
The pipeline aggregates existing events; it does not contain session
boundaries, session counts, duration, study intent, or a mapping from a
planned session to clicks. Nor does it expose a session-aware prediction API.
`RegressionPredictor.predict` validates the saved strict feature schema,
applies the train-fitted preprocessor and MLP ensemble, and returns a clipped
score estimate. Supplying altered click totals is syntactically possible for a
valid feature row but would be a synthetic intervention outside the model's
observed-data validation.

### Proposed Session-to-Clicks Mapping component

The component takes a copy of an observed strict feature row, an added
assignment/week session, and a versioned rule configuration. It deterministically
returns a synthetic candidate row and an audit record. For each added session,
the configuration specifies a click-length rule, maps the planned week to one
of the four deadline-relative windows, and selects an appropriate existing
activity group. The session must precede the prediction cutoff. No numeric
click length is specified here; the rule must be documented and approved
before implementation.

The component changes only the selected click feature in a copied candidate.
It does not mutate historical data, labels, stored feature rows, or model
artifacts. It preserves the strict feature schema and passes a schema-validated
copy to the existing predictor. A documented support policy, based on training
observations and without test access, flags values outside observed ranges and
unsupported combinations. Out-of-support states must not be presented as
validated estimates; whether to prune them or report exploratory predictions
is unresolved.

The existing strict model schema excludes previous assessment marks, despite
the verified P1 requirement that previous marks be inputs. Proxy mode includes
prior scores, but OULAD has no score-release timestamps; a prior submission
before the target deadline does not prove that the score was available then.
Do not add prior-score features to the model without a defensible availability
cutoff and a separate leakage audit. This document does not authorize changing
the current strict model or its artifacts.

Therefore:

- Potentially changed features: only the `click_<group>_week_<1..4>` fields
  selected by the hypothetical session's assumed time window and activity
  group.
- Unchanged features: type indicators, assessment weight and its availability,
  deadline day, and weeks since course start.
- Unavailable by design: direct session count, duration, assignment/week plan,
  or explicit effort measure.
- Prediction cutoff: sessions on or after `D` cannot affect any input for that
  target without violating the established cutoff.

The strict feature and NN reports say click activity is observational and do
not establish that clicks cause marks. The dataset audit also notes that event
capture completeness is not independently measured. Retrospective labels are
valid assessment outcomes for predictive evaluation, but they do not reveal
the counterfactual score under an added session.

## Candidate approaches

| Approach | Scientific assumptions | Data requirements | Leakage and validity risks | Advantages / disadvantages | P1 fit | Complexity |
| --- | --- | --- | --- | --- | --- | --- |
| **A. Documented synthetic effort-to-click mapping** | A session receives a generated increment in a configured category/window. This is a simulation rule, not an observed session effect. | Versioned deterministic configuration for click-length rule, week-to-window mapping, activity category, support policy, capacity and target rules. Numeric values are not selected here. | Synthetic vectors can leave training support; tuning rules on validation/test leaks evaluation data. Never mutate historical observations or present output deltas as causal. | Directly satisfies the generated-rule mechanism and preserves schema/artifacts; reproducible and auditable. Predictions remain sensitive to assumptions. | Compatible with P1 as an explicitly labeled simulation. | Medium: mapping, config, audit, support checks, tests. |
| **B. Separate monotonic effort-response component calibrated on validation** | A separate component imposes non-decreasing marks as sessions increase. This is a simulation constraint; current validation records do not identify session response. | Session exposure labels and independent calibration/evaluation evidence are required for empirical calibration. Existing validation has no session counts. | Same-split calibration/evaluation overfits. Monotonicity cannot remove confounding; constraining the model can hide violations. | Can provide a distinct comparator, but must not replace required violation reporting for the existing MLP under synthetic increments. | Optional comparator, not required for the specified mapping + NN flow. | High: separate model and leakage-safe calibration. |
| **C. Keep existing NN as baseline; use a separately documented effort model for planning** | Existing NN scores mapped synthetic rows; the mapping is a separate generated-rule component, not a learned session effect. | Same rule config and support policy as A, plus current baseline. Additional session data are needed for empirical validation, not for a labeled simulation. | Do not call synthetic predictions causal; flag support violations, preserve source rows/artifacts, and keep previous-score leakage unresolved. | Separates trained prediction from generated inputs, but inherits arbitrary mapping and possible NN monotonicity violations. | Recommended architecture for P1's deterministic simulation requirements, subject to transparent caveats. | Medium/high: orchestration, provenance, support and monotonicity reports. |

## Monotonicity and synthetic augmentation

P1 requires monotonicity testing and violation reporting, not forced
monotonicity. For fixed observed context, compare predicted marks across
increasing session allocations using the same configuration. Predefine the
comparison order and tolerance. Record each decrease with assignment,
allocation pair, affected category/window, increment, and prediction delta;
summarize violation counts/rates by assignment and configuration. Do not
repair, reorder, clip away, or suppress violations. The unconstrained MLP may
violate monotonicity; clipping to `[0,100]` does not prevent this. Passing a
software check shows behavior only on tested inputs, not real-world monotonic
effort-mark effects.

Mapping operates on copied candidate rows only. It must not augment or
overwrite historical observations. It supports the statement “under this
generated rule, the NN output changes by …,” not proof that sessions produce
those clicks or cause marks to change. Do not retrain the existing model on
synthetic rows under this design.

## Correct role of the current NN and baseline

Use the MLP as a baseline predictor for assessment marks conditional on its
strict, cutoff-valid observed feature vector, and as the P1 simulation's
predictor on copied rows produced by the documented mapping. Its existing three-seed
ensemble, train-fitted preprocessing, fixed chronological split, and held-out
test results support that predictive role. Existing report metrics (including
test MAE/RMSE) characterize observed-row prediction, not counterfactual
session allocation. The MLP is still not an effort model: the mapping
specifies synthetic input perturbations, not a learned response to effort.
Report the prediction delta under the mapping without claiming that a real
session raises a score.

For P1, use the separate deterministic Session-to-Clicks Mapping followed by
the unchanged strict `RegressionPredictor`. The mapping returns a copied
synthetic row and provenance/support metadata; the MLP returns the simulated
mark estimate. Search may use that estimate under the documented rules, while
reports state it is observational and non-causal. Keep generated capacity and
target rules separate and versioned. Do not retrain or change original NN
architecture, schema, or artifacts. Out-of-support cases must be flagged, not
silently extrapolated.

There is a separate specification discrepancy: P1 requires previous marks as
NN inputs, but current strict feature generation, dataset validation, and
inference exclude prior scores. Proxy mode contains prior scores, but no
release timestamps establish their availability at later prediction cutoffs.
Preserve the strict model. Prior marks must not be added until an availability
cutoff is defensible and a leakage audit is complete; otherwise report this
input requirement as unmet.

## Required evidence and experiments before planning use

1. Define session semantics (duration, week/time, assignment, repeatability),
   target semantics, prediction horizon, and the output required from the
   effort model. Do not infer these from click aggregates.
2. Obtain session-level exposure data linked to learners and assessments, or
   design a controlled/randomized study of study-session assignments. Record
   actual participation and exposure timing before each assessment cutoff.
3. Define the estimand: predictive association versus causal effect of an
   assigned/attended session. For a causal claim, pre-specify confounders,
   treatment assignment/adherence, and identification strategy; historical
   OULAD click and score records alone do not establish this.
4. Keep learners/presentations grouped across train, calibration/validation,
   and final test. Fit any mapping/model only on training data, select it on
   validation, and use test once. Avoid per-row random splitting across the
   same student/presentation when dependence would leak information.
5. Compare the existing strict MLP baseline against appropriate simple
   baselines and the effort-aware candidate on the same eligible held-out
   examples. Report MAE/RMSE and calibration/uncertainty, subgroup and
   presentation performance, and session-response curves with sample support.
6. Evaluate incremental prediction value and response stability across
   session counts, timing windows, assignment types, and plausible
   specifications. Check overlap/support; flag out-of-distribution plans
   instead of silently extrapolating.
7. Pre-specify the monotonicity comparison order and tolerance. Test the
   unchanged MLP over mapped increasing allocations, retain and report every
   violation, and separately state that this is simulated prediction behavior,
   not evidence of causal effects.
8. Before Search integration, run sensitivity analyses over uncertainty and
   plausible effort-response alternatives. Report when target reachability
   changes across assumptions; do not present a single unsupported estimate
   as certain.

The current validation split may compare predictive models, but it contains
no session exposure label and cannot calibrate an effort-response curve.
Existing test results must remain untouched and cannot be repurposed to tune
the mapping.

## Recommendation and unresolved decisions

**Recommendation: Approach C using the required generated-rule simulation.**
Keep the strict MLP architecture, schema, and artifacts unchanged. Add a
separate deterministic, configurable Session-to-Clicks Mapping that changes
only copied candidate features, applies a training-derived support policy,
and emits provenance. Evaluate increasing allocations and report every
monotonicity violation; do not force the MLP to be monotonic. Label outputs as
observational predictions under synthetic inputs, not causal effects. Approach
B is not calibratable from current validation data, which contains no session
counts. The previous-mark input requirement also cannot be met by the current
strict model until an availability cutoff and leakage audit justify it.

Unresolved decisions:

- What click-length, assignment-week to relative-window, and
  activity-category rules should be configured? No numeric click length is
  selected in this audit.
- What formulas and provenance define generated capacity and target marks?
- Which training-derived support method and out-of-support policy should be
  used?
- What monotonicity comparison order and numerical tolerance are required?
- Are targets per assignment or course-level, and how are predictions
  aggregated and uncertainty handled?
- What evidence establishes prior-mark availability before each cutoff? If
  unavailable, how should the unmet P1 input requirement be reported?
- What out-of-support behavior, uncertainty threshold, and no-solution
  reporting rule should Search use?

The mapping is reproducible once its rule values and support policy are
specified, but its predictions are not causal estimates. The prior-mark
requirement remains unmet by the current strict NN until availability and
leakage are resolved.
