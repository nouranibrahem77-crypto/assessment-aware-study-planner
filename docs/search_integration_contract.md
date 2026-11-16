# Search–NN–CSP integration contract

## Status and scope

This document defines the proposed Search–NN–CSP control flow. The generic CSP
contracts, Backtracking, Forward Checking, AC-3, lazy schedule enumeration,
and Search–NN–CSP orchestration exist. The application-input adapter now builds
a deterministic allocation state space and CSP candidate factory from explicit
validated values. Production target, deadline eligibility, slot catalog,
allocation-bound, support, and ordering values must still come from an approved
application configuration.
The contract preserves the existing strict NN schema, inference interface,
architecture, and artifacts, and does not claim session effects are causal.

## Official requirements

- Search starts with zero sessions; each action adds one session to one
  assignment. The smallest target-meeting plan is required.
- The NN predicts marks from previous marks and recent online-learning clicks.
- Documented rules generate session click increments, slot capacity, and
  target marks. Search proposes; NN predicts; CSP verifies schedules; rejected
  candidates return to Search.
- The official example uses three assignments, target 55, three free slots per
  week, and a 4/5/3 session allocation. These are example values, not defaults.
- Required CSP experiment: compare BT, BT+FC, and BT+FC+AC-3 at 2, 3, and 4
  available slots per week.
- Sessions must precede their assignment deadlines, use unique weekly slots,
  and respect the same-assignment ordering constraint.
- Monotonicity is tested over increasing allocations and violations reported.
  The relationship is observational/simulated, not causal.

## Proposed component interfaces

The `study_planner.integration` package implements the bridge and orchestration
contracts below. The generic CSP foundation implements the whole-plan manifest,
problem, constraints, intermediate evaluation, BT, FC, AC-3, lazy enumeration,
and whole-plan schedule verification.

`SessionNNAdapter` requires a fitted-artifact SHA-256 digest produced by the
Model A or Model B artifact loader. The digest covers the exact schema,
experiment configuration, persisted preprocessor, and checkpoint bytes read
for inference. The loader issues a private capability for the exact predictor
instance only after loading the bytes it hashed. Adapters require that capability
and verify it against a fingerprint of the loaded preprocessor, feature schema,
model collection, and model tensor state before inference and final authorization.
Replacing the collection or mutating learned state invalidates the capability;
changing the public digest field cannot restore it. The complete model provenance
identity then hashes the artifact digest with the model mode and feature-schema
identity. Missing or malformed artifact digests are rejected; descriptive labels
and predictor class/schema are not accepted as artifact proof. Prediction records
bind their output to the candidate, schedule, verification, and model provenance
identities, and recheck the predictor capability when used for a goal test. Model A
strict and Model B previous-marks proxy remain separate schema/mode contracts.

| Component | Input | Output |
| --- | --- | --- |
| Search | Problem bounds, initial zero-session allocation, shared evaluator callbacks | Candidate allocation plus stable allocation key. UCS orders by path cost; Greedy orders by heuristic. Search continues after rejected candidates. |
| Provisional mapping | Immutable observed strict feature row, allocation, provisional-week rule, click scenario/config version | Copied feature row matching the strict schema; provenance and support flags. Only configured click columns may change. |
| NN inference | One or more strict-schema feature rows | Clipped score prediction(s) in `[0, 100]`, via existing `RegressionPredictor.predict`. |
| CSP scheduler | Whole-plan allocation for all three assignments, deadlines, weekly slot domains for the selected capacity, ordering policy | Lazy deterministic sequence of per-schedule `WholePlanVerificationResult` values, followed by a `WholePlanSearchOutcome` distinguishing exhaustive enumeration, no feasible schedule, limits, cancellation, and errors. |
| Final mapping | Feasible whole-plan verification result, expected whole-plan identity, one assignment projection, immutable observed strict row | Copied strict-schema row with click features from that assignment's actual scheduled windows and whole-plan verification provenance. |
| Result/audit recorder | Allocation/schedule keys, both predictions, CSP status, support and monotonicity records | Separate provisional/final prediction records, CSP rejection records, final-target failures, and terminal status. |

`build_application_planning_problem` in `study_planner.integration` is the
application adapter for these interfaces. Its Search state contains one
four-window allocation for each of the three assignments. Each action adds a
single session, costs one, and is available only below that assignment's
configured maximum. Search evaluates a state only when every configured
minimum/maximum bound is met. The candidate factory uses each assignment's
explicit eligible slot IDs as its CSP domain, the catalog's weekly capacity,
and its per-assignment slot-to-click-window mapping. It receives an explicit
same-assignment ordering policy from the caller; no ordering or deadline
conversion is inferred from the slot clock fields. The end-to-end tests label
their sample values as TEST configuration and they are not production defaults.

### Typed adapter and CSP verification boundary

`WholePlanIdentity` binds the student and course, all three assignment
allocations in canonical assignment-ID order, each assignment's base feature
context fingerprint, and the mapping-scenario identity. Each assignment entry
projects to an assignment candidate identity bound to student, course,
assignment, allocation, base-row fingerprint, and scenario. Candidate and
whole-plan identities remain separate, and the manifest rejects candidate
substitutions with a mismatched student, course, assignment, or scenario. The
full identity remains the authority for CSP matching; matching one assignment
projection alone does not prove that cross-assignment constraints were checked.

`ScheduleData` contains data only and is never sufficient for final prediction.
The standalone CSP verifier returns a `WholePlanVerificationResult` for each
complete three-assignment schedule supplied to it. It binds the manifest's
whole-plan identity, full
assignment-partitioned schedule details, schedule identity, verification
identity, `verification_policy_version`, feasible/rejected status for that
schedule, and immutable records for every applied constraint check. Required
check IDs are
`deadline_before_cutoff`, `weekly_slot_unique`, `same_assignment_order`, and
`weekly_capacity`; all must be checked and passed for a feasible status.
Additional hard constraints must also appear in the check records. The policy
version identifies deadline and ordering semantics, slot catalog/source,
capacity configuration/source identity, and slot-identity semantics.

A rejected schedule is not proof that its whole-plan allocation has no feasible
schedule. The enumerator returns a separate `WholePlanSearchOutcome` to report
whether it exhausted all schedules (with none or some feasible), stopped at a
limit/cancellation, or encountered an operational error. Only an exhaustive
outcome can establish that no feasible schedule exists for the allocation.

The generic CSP foundation now models a complete WholePlanManifest, including
all three assignment candidates and zero-session allocations. Its CspEvaluation
is bound to both the manifest identity and a CSP problem identity covering
capacity, domains, an immutable finite ordering-policy snapshot, and constraint
configuration. The problem captures required ordering pairs and the policy's
slot relation over its finite domain slots, then rebinds ordering constraints
to that snapshot. Mutation of the original policy cannot alter problem
behavior. The evaluation is an intermediate local-validation record only:
BRANCH_REJECTED rejects the current branch, INCOMPLETE leaves the decision
open, and FEASIBLE is created only after exact variable coverage, in-domain
values, and every constraint pass. The solver outcome independently repeats
these checks for its attached feasible evaluation. It cannot report whole-plan
infeasibility. A future solver-level outcome may report INFEASIBLE only after
exhaustive search; limits and cancellation remain INCOMPLETE.

Constraints expose explicit scopes. Binary constraints expose compatibility,
support, and revision operations for a future AC-3 solver without manufacturing
full assignments. BT exposes a deterministic lazy enumerator; a failed final NN
target evaluation can request the next distinct feasible schedule from the same
traversal. INCOMPLETE enumeration does not establish allocation infeasibility.
`WholePlanScheduleEnumerator` consumes the selected BT, BT+FC, or BT+FC+AC-3
enumerator, verifies each yielded assignment, and reports an allocation-level
terminal outcome only after exhaustive enumeration. `SessionNNAdapter` accepts
only a complete feasible `WholePlanVerificationResult`; it validates expected
manifest, problem, policy, schedule, and verification identities before
projecting one assignment's actual slot weeks into its four feature windows.
`CspEvaluation` remains an intermediate local validator record and must never
be treated as final verification provenance. Integrated UCS and Greedy
orchestration evaluate every allocation proposal, try each distinct verified
schedule after final-target misses, continue after exhaustive allocation-level
CSP rejection, and return `INCOMPLETE` when a configured candidate or CSP node
limit prevents an exhaustive decision. An `INCOMPLETE` enumerator is terminal
and cannot resume; restarting creates a new traversal from the first schedule.
The integration wrapper memoizes each state's goal result because Greedy's
generic search checks its initial state before queue insertion and checks it
again when popped. This keeps side-effecting candidate evaluation, CSP
enumeration, and counters at one invocation per state without changing Greedy's
heuristic ordering. Planner statistics retain terminal CSP enumeration
outcomes. Exhaustive empty CSP results, whole-plan verification rejection,
verified schedules missing NN targets, and resource-limited runs receive
separate deterministic failure reasons; an incomplete outcome is never
reported as exhaustive infeasibility.

`SessionNNAdapter.evaluate_final` takes a feasible whole-plan schedule result
and expected whole-plan, problem, policy, schedule, and verification
provenance. It verifies the assignment's `CandidateIdentity` against the
manifest and obtains that assignment's actual schedule projection. It rejects
raw schedules, incomplete/infeasible results, and identity/count mismatches.
No assignment-level final NN evaluation may occur before whole-plan feasibility
is established. Assignment-level final results remain bound to the same
whole-plan, schedule, and verification identities; the plan is eligible for
acceptance only after every required assignment has a successful final
prediction under the configured target rule. `CspVerificationResult` remains
an intermediate CSP evaluation and cannot authorize final prediction.

The solver factory remains an application-level provenance convention, not a
security boundary against malicious Python code or deliberate object
fabrication.

Prediction results retain an immutable scalar snapshot of the exact mapped
feature row and immutable support diagnostics; any DataFrame view is freshly
materialized for the caller. Monotonicity comparison context is keyed by
student, assignment, base feature context, and mapping scenario. Each
observation separately retains its allocation and actual schedule identity,
so comparable increasing allocations can use different schedules while
unrelated contexts remain separate. These comparisons describe model behavior
under the simulation inputs and do not establish causal effects.

The Session-to-Clicks Mapping must be deterministic for a fixed configuration
and input. The existing weekly columns are `click_<group>_week_<1..4>` for
`content_navigation`, `assessment_activity`, and `collaboration`. The four
windows are `[D-28,D-21)`, `[D-21,D-14)`, `[D-14,D-7)`, `[D-7,D)` before
deadline day `D`. The mapping modifies a copy only and never mutates the
observed row, source data, labels, or saved artifacts. The 22, 62, and 121
content-navigation click values documented in the readiness audit are
sensitivity scenarios derived from training-window totals; they are not
measured session durations or factual per-session click counts.

## Control flow

```mermaid
sequenceDiagram
    participant S as Search
    participant M as Mapping layer
    participant N as Strict NN
    participant C as CSP
    participant R as Audit recorder

    S->>S: Generate next allocation
    S->>M: allocation + observed row + provisional rule
    M-->>S: provisional strict-schema features + support flags
    S->>N: Predict provisional marks
    N-->>S: provisional prediction
    S->>R: Record provisional prediction separately
    alt Provisional goal not met
        S->>S: Continue Search
    else Provisional goal met
    S->>C: Verify complete three-assignment plan
    alt No feasible whole-plan schedule
        C-->>S: Exhaustive infeasible result + constraint checks
        S->>R: Record CSP rejections
        S->>S: Reject allocation; continue Search
    else Feasible whole-plan schedule returned
        loop Each distinct whole-plan schedule until success or exhaustion
            C-->>S: WholePlanVerificationResult + full schedule key
            loop Each required assignment
                S->>M: verified plan + identity + assignment projection + base row
                M-->>S: final strict-schema features + support flags
                S->>N: Predict assignment final mark
                N-->>S: final prediction
                S->>R: Record assignment result with whole-plan provenance
            end
            alt All required final marks meet the target rule
                S->>R: Record accepted plan
                S-->>S: Terminate with allocation and schedule
            else A required final mark fails
                S->>R: Record final-prediction failure
                S->>C: Request next distinct whole-plan schedule
            end
        end
        S->>S: If schedules exhausted, reject allocation; continue Search
    end
    end
```

Every Search proposal receives its provisional NN prediction before CSP,
preserving the specified Search → NN → CSP order. Under the proposed goal-test
interpretation, a proposal that does not meet the provisional target returns to
Search; a target-reaching proposal is passed to CSP for whole-plan feasibility
verification. This provisional result never accepts a plan. Since the
provisional weekly distribution may differ from feasible schedules, whether a
provisional miss may definitively prune a candidate remains an unresolved
Search-semantic decision.

## Detailed allocation lifecycle

1. Search produces a candidate allocation from the zero-session state. Each
   action adds exactly one session. UCS pops candidates in non-decreasing
   session cost; Greedy pops by heuristic priority.
2. The provisional mapping applies the configured deterministic weekly
   distribution to a copy of the observed strict feature row. It records the
   rule version, scenario (including low/central/high where selected),
   changed feature columns, and support status.
3. Strict NN inference produces provisional marks. Store these separately from
   all final predictions, keyed by allocation, context, and mapping/model
   versions.
4. For provisional target-reaching proposals, CSP verifies each complete
   schedule across all three assignments. It records
   checks for deadlines, unique weekly slots, same-assignment ordering, and
   weekly capacity under a versioned policy. If exhaustive CSP proves there is
   no feasible schedule, store its whole-plan rejection evidence and continue
   Search. An incomplete CSP result is not infeasibility.
5. Only after a feasible `WholePlanVerificationResult` exists may final NN
   evaluation begin. The adapter checks its whole-plan identity against the
   expected candidate, projects each assignment's `CandidateIdentity`, and
   obtains that assignment's verified schedule. It rebuilds final features
   from the immutable base row and actual scheduled windows; provisional
   increments are never reused.
6. Evaluate final predictions for all required assignments and bind every
   result to the same whole-plan identity, schedule identity, and verification
   identity. If the complete plan misses target, request the next distinct
   feasible whole-plan schedule. If schedules exhaust, record
   `final_target_failed` and resume Search.
7. Accept only when whole-plan feasibility is established and all required
   final predictions meet the documented target rule and support policy.

The final feature vector must reflect the *same allocation* as the schedule.
The provisional and final records remain distinct even if their vectors or
predictions happen to match.

## Search ordering and optimality

Both algorithms share the exact same state representation, provisional and
final candidate evaluator, strict NN interface, CSP interface, duplicate
handling, and acceptance/termination rules. Their frontier priorities differ.

**UCS** orders by cumulative path cost `g`. Each action has unit cost, so `g`
equals session count and UCS considers candidates in non-decreasing session
count. Continue until the first allocation with a feasible schedule and a
successful final prediction. Under finite bounds, non-negative unit costs, and
exhaustive schedule evaluation, the accepted plan has minimum session cost.

**Standard Greedy Best-First Search** orders by documented heuristic `h`
alone, with a deterministic tie-breaker. It does not force increasing-cost
layers and does not minimize session count. It may return a higher-cost
solution than UCS. This difference is intentional: UCS expands by accumulated
cost to preserve optimality; Greedy expands by estimated remaining score gap
to prioritize apparently promising candidates. Only a final
schedule-specific prediction can pass the goal test for either algorithm;
provisional marks never independently accept a plan.

## Fair algorithm comparison

Run both algorithms on the same problem instances, NN artifact, generated-rule
configuration, support policy, CSP ordering, and tie-breaking policy. Report
these metrics for each run and aggregate across the required runs:

- solution cost: total sessions in the accepted allocation;
- nodes expanded and nodes generated, using one shared counting convention;
- runtime under the same measurement boundary (including or excluding NN/CSP
  work consistently, with the choice stated);
- success/failure and terminal reason (accepted, exhausted, resource limit,
  or operational error);
- final predicted marks for the accepted schedule (never provisional marks);
- CSP feasibility and schedule outcome, including rejection counts/reasons.

Show provisional predictions separately for diagnosis. Do not compare one
algorithm's provisional score with another's final score. Report no-solution
best achievable marks from final predictions on feasible schedules only.

## Duplicate avoidance and audit records

Use stable keys and separate caches:

- `allocation_key`: canonical assignment/session-count representation plus
  problem identity. Do not merge distinct allocations because their scores
  match.
- Provisional prediction cache key: allocation key, base-row/context identity,
  provisional rule version, click-scenario version, support-policy version,
  and NN artifact/schema identity.
- `schedule_key`: allocation key plus canonical ordered assignment-to-slot
  mapping. Two enumerations of the same schedule share a key; distinct
  schedules must still be evaluated because their click windows may differ.
- Whole-plan verification cache key: whole-plan allocation identity, complete
  schedule identity, CSP policy/catalog/capacity versions, and CSP problem
  identity.
- Final prediction cache key: whole-plan verification identity plus selected
  assignment identity, base-row/context identity, mapping/scenario/support
  versions, and NN artifact/schema identity.
- CSP rejection key: allocation key plus CSP problem/configuration identity;
  retain structured reasons. A schedule-specific failure must not be confused
  with allocation infeasibility unless CSP has exhausted all schedules.

Record provisional and final predictions in different fields/records. Record
CSP rejection (`no_feasible_schedule` and its constraint reasons) separately
from final prediction failure (`target_not_met`, score, and target gap),
support rejection/flag, invalid NN input/output, and resource-limit stop. A
model/schema error is an operational failure, not evidence that a target is
unreachable.

For monotonicity, compare fixed-context allocation parent/child pairs under the
same configuration and actual schedule where possible. Store the two
predictions, changed group/window, increment scenario, tolerance, support
flags, and delta. Report decreases as violations; never alter features or
predictions to hide them. Provisional comparisons and final schedule-matched
comparisons should be labeled separately.

## Termination and no-solution behavior

Success terminates at the first accepted schedule in the algorithm's defined
candidate/schedule ordering. For an allocation, Search resumes only after CSP
has exhausted feasible schedules and all final predictions have failed, or
CSP has proven there is no feasible schedule.

Report **target unreachable at available capacity** only after the finite
allocation space is exhausted, CSP is exhaustive for each relevant allocation,
all required schedule-specific predictions are valid, and the support policy
has defined which candidates count. Report best achievable predicted marks
from final predictions for feasible schedules, with the corresponding
allocation/schedule and whether it was in support. Keep provisional-only
scores out of this statistic. If candidates were truncated by a resource limit,
report `incomplete` rather than a proven no-solution result. If whole-plan
verification found feasible schedules but SupportPolicy rejected all of them,
report `no_supported_plan`; this does not imply CSP infeasibility.
`no_feasible_schedule` is reserved for exhaustive results with no whole-plan
verification-feasible schedule and no support rejection that prevented a
supported plan from being evaluated. This is search-wide: if one candidate is
rejected by provisional support and another exhausts CSP without a feasible
schedule, the result is `no_supported_plan`, because the rejected candidate's
supported-plan feasibility was never evaluated. CSP enumeration,
verification-feasible schedules, verification rejections, support rejections,
and final-scored supported schedules are reported separately. Prediction
target failures remain separate from all of these counts.

The planner compares verified feasible schedules by maximizing the sum of
their final predicted assignment marks; ties maximize the mark tuple in
canonical assignment order, then minimize total sessions, then choose the
lexicographically smallest whole-plan and schedule identities. `target_achieved`
means the returned verified schedule meets all assignment targets;
`target_unreachable` requires exhaustive search and carries the best schedule
under that comparison; `no_feasible_schedule` means no whole-plan
verification-feasible schedule was found and carries no predictions or
timetable; `no_supported_plan` means support policy rejected candidate
evaluation before any supported final score was retained; `incomplete` reports
any best verified schedule found so far without claiming global optimality;
`error` represents an operational failure. A successful early stop is not an
exhaustive optimality claim.

## Testable acceptance criteria

- CSP evaluation covers all nine required combinations: BT, BT+FC, and
  BT+FC+AC-3 at weekly capacities 2, 3, and 4.
- Starting from zero, every generated successor adds exactly one session and
  has the expected unit path-cost increment.
- UCS candidate allocations are processed in nondecreasing session cost;
  standard Greedy is processed by `h` priority and may return a higher-cost
  solution. A deterministic tie-break is exercised for both.
- Both algorithms call the same candidate evaluator, mapping, NN and CSP
  interfaces, use the same state and termination contracts, and record the
  same comparison metrics.
- Provisional feature construction is deterministic, uses only the declared
  provisional distribution, preserves strict columns, changes only allowed
  click fields, and leaves the base row byte/value-identical.
- Final features are rebuilt from the immutable base and actual CSP schedule;
  their week mapping is correct and does not double-count provisional clicks.
- Provisional target success never bypasses CSP or final NN prediction.
- If a final prediction misses target, another schedule for the same allocation
  is tried before that allocation is rejected; schedule exhaustion resumes
  Search exactly once. Enumeration is lazy and may stop after a plan passes.
- No assignment-level final NN evaluation occurs before a feasible
  whole-plan-verification result exists. Every assignment result for an
  accepted plan matches the same whole-plan, schedule, and verification IDs.
- A feasible whole-plan CSP result records the versioned policy and every
  required/applied constraint check, with all required checks passed; an
  incomplete result is never treated as infeasible or feasible.
- CSP infeasibility records structured constraint reasons and is not recorded
  as a prediction failure. Final target failure is not recorded as CSP failure.
- Duplicate allocation/schedule enumeration does not repeat the corresponding
  CSP or NN call; distinct schedules are evaluated independently.
- Final acceptance requires CSP feasibility and the final, schedule-specific
  prediction to meet target. Provisional prediction alone can never accept.
- Exhaustion reports no solution and best final feasible prediction; a limit,
  schema error, or unscorable candidate is not reported as proven unreachable.
- Monotonicity reports preserve every violation and do not clip/repair
  synthetic inputs or prediction ordering. The 22/62/121 scenarios are labeled
  sensitivity cases, never measured session durations.
- Strict schema/model artifacts are unchanged; prior-score columns are not
  supplied unless an independently approved cutoff and leakage audit exist.

## Unresolved assumptions

- What deterministic provisional weekly distribution is approved, and how is
  it derived from an allocation with no selected weeks?
- What is the authoritative source and generator for weekly slot domains? The
  required capacity cases specify 2, 3, and 4 available slot values per week;
  whether that setting is uniform across weeks is not stated; exact slot
  identities, dates, and durations are not specified.
- How do day-offset deadlines map to weekly slot dates, and what exact
  before-deadline boundary applies? The four NN click windows do not define
  this scheduling rule.
- Does “sessions ordered” mean strictly increasing weeks, chronological slot
  order (possibly within a week), or another relation?
- The integration contract requires trying another feasible schedule when a
  final prediction misses target. Proposed behavior is deterministic lazy
  enumeration until acceptance or exhaustive schedule exhaustion; the official
  materials do not set the schedule ordering or resource-limit policy.
- Whether a provisional NN miss may definitively prune a Search proposal,
  given that final scheduled weeks can produce different features and marks.
- What whole-plan identity fields are available for course and base feature
  contexts in application inputs, and how will Search construct all three
  assignment projections before invoking CSP?
- What target rule and out-of-support policy govern
  final acceptance and best-achievable reporting?
- What is the monotonicity comparison tolerance and required population of
  contexts/schedules?
- How will previous marks be made available without leakage? The current strict
  schema excludes them, and submission time is not score-release time.

Until these choices are resolved, this contract defines safe control flow but
does not authorize numeric rule choices or changes to the NN.
