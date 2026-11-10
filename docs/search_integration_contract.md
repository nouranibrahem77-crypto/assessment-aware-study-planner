# Search–NN–CSP integration contract

## Status and scope

This document defines the proposed control flow before implementation. It
preserves the existing strict NN schema, inference interface, architecture,
and artifacts. It does not add code or claim session effects are causal.

## Official requirements

- Search starts with zero sessions; each action adds one session to one
  assignment. The smallest target-meeting plan is required.
- The NN predicts marks from previous marks and recent online-learning clicks.
- Documented rules generate session click increments, slot capacity, and
  target marks. Search proposes; NN predicts; CSP verifies schedules; rejected
  candidates return to Search.
- Sessions must precede their assignment deadlines, use unique weekly slots,
  and respect the same-assignment ordering constraint.
- Monotonicity is tested over increasing allocations and violations reported.
  The relationship is observational/simulated, not causal.

## Proposed component interfaces

These are design contracts, not implemented Python APIs.

| Component | Input | Output |
| --- | --- | --- |
| Search | Problem bounds, initial zero-session allocation, shared evaluator callbacks | Candidate allocation plus stable allocation key. UCS orders by path cost; Greedy orders by heuristic. Search continues after rejected candidates. |
| Provisional mapping | Immutable observed strict feature row, allocation, provisional-week rule, click scenario/config version | Copied feature row matching the strict schema; provenance and support flags. Only configured click columns may change. |
| NN inference | One or more strict-schema feature rows | Clipped score prediction(s) in `[0, 100]`, via existing `RegressionPredictor.predict`. |
| CSP scheduler | Candidate allocation, deadlines, slot domains, ordering rule | Ordered iterable of distinct feasible schedules, or an exhaustion result with rejection reasons. One-schedule-at-a-time iteration is preferred so final NN checks can stop on success. |
| Final mapping | Immutable observed strict row, allocation, one actual CSP schedule, click scenario/config version | Copied strict-schema row with click features determined by actual scheduled weeks, plus provenance/support flags. |
| Result/audit recorder | Allocation/schedule keys, both predictions, CSP status, support and monotonicity records | Separate provisional/final prediction records, CSP rejection records, final-target failures, and terminal status. |

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
    S->>C: Request feasible schedules for allocation
    alt No feasible schedule
        C-->>S: Exhausted + rejection reasons
        S->>R: Record CSP rejections
        S->>S: Reject allocation; continue Search
    else Feasible schedule returned
        loop Each distinct feasible schedule until success or exhaustion
            C-->>S: Schedule + schedule key
            S->>M: allocation + actual schedule + observed row
            M-->>S: final strict-schema features + support flags
            S->>N: Predict final marks
            N-->>S: final prediction
            S->>R: Record final prediction separately
            alt Final target met and support policy permits acceptance
                S->>R: Record accepted plan
                S-->>S: Terminate with allocation and schedule
            else Final target fails
                S->>R: Record final-prediction failure
                S->>C: Request next distinct schedule
            end
        end
        S->>S: If schedules exhausted, reject allocation; continue Search
    end
```

The provisional prediction is a screening/planning value only. It never
constitutes acceptance. CSP must be invoked for each allocation under this
contract; provisional success is not proof of schedule feasibility, and
provisional failure does not prove that every actual schedule would fail.
Therefore the implementation must not prune an allocation solely because its
provisional prediction misses target unless the official search semantics
explicitly authorize that pruning. The safe default is to schedule it and
perform final predictions, or document a sound bound that proves pruning safe.

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
4. CSP enumerates feasible schedules for this allocation. A schedule must
   satisfy deadlines, unique weekly slots, and same-assignment ordering. If
   none exist, store each rejection reason and continue Search.
5. For each distinct feasible schedule, the final mapping derives relative
   NN windows from the schedule's actual weeks and creates a fresh copied row
   from the same immutable observed base. It must not apply increments on top
   of the provisional row, which would double count sessions.
6. Run the existing NN on final features and record a final prediction.
   Evaluate target satisfaction on this prediction only. If it misses target,
   request another feasible schedule for the same allocation. If schedules
   exhaust, record `final_target_failed` and resume Search.
7. Accept only when the actual schedule is feasible and its final
   schedule-specific prediction meets the target under the documented target
   rule and support policy.

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
- Final prediction cache key: schedule key plus base-row/context identity,
  mapping/scenario/support versions, and NN artifact/schema identity.
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
scores out of this statistic. If candidates were unscorable, out of support
under a reject policy, or truncated by a resource limit, report an incomplete
or indeterminate result rather than a proven no-solution result. CSP rejection
counts and prediction target failures must remain separate in summaries.

## Testable acceptance criteria

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
  Search exactly once.
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
- Does CSP enumerate all feasible schedules, or return one canonical schedule?
  If it enumerates, what is its deterministic ordering and completeness proof?
- What target rule, generated capacity rule, and out-of-support policy govern
  final acceptance and best-achievable reporting?
- What is the monotonicity comparison tolerance and required population of
  contexts/schedules?
- How will previous marks be made available without leakage? The current strict
  schema excludes them, and submission time is not score-release time.

Until these choices are resolved, this contract defines safe control flow but
does not authorize numeric rule choices or changes to the NN.
