# P1 Constraint Satisfaction Problem Design

## Scope and source status

This document records requirements, design decisions, unresolved inputs, the
generic CSP foundation, and baseline Backtracking implementation. Forward
Checking and AC-3 are not implemented here. Search, NN, data, and model
artifacts are outside CSP implementation scope. Repository inspection found no standalone uploaded
assignment brief. The official requirements below are the P1 requirements
provided in the project instructions and recorded in the existing Search and
integration design documents. They are kept separate from proposed design
choices. The named PDF was not present in the workspace or a connected document
session during this revision; official facts follow the requirements and
example explicitly supplied with the request.

## Official assignment requirements

- Minimum scope: one course and three assignments.
- Search proposes an allocation; NN predicts marks; CSP verifies that the
  sessions can be scheduled. A rejected candidate returns to Search.
- Every session must occur before its assignment deadline.
- At most one session may occupy a weekly time slot.
- Sessions for the same assignment must be ordered.
- The official example describes three assignments, target mark 55, three free
  slots per week, and an allocation of 4, 5, and 3 sessions. These are example
  values, not constants for every planner run.
- The required CSP experiment compares BT, BT+FC, and BT+FC+AC-3 at weekly
  capacities of 2, 3, and 4 slots.
- Required CSP methods: Backtracking, Forward Checking, and AC-3.
- Required control-flow order is Search proposal → NN prediction → CSP
  verification. If CSP rejects a proposal, Search continues. Whole-plan
  verification must precede any final schedule-specific NN evaluation.
- At available-capacity exhaustion, report the target as unreachable and show
  the best achievable predicted marks, subject to complete, valid evaluations.

| Official example only | Value |
| --- | --- |
| Assignment count | 3 |
| Target mark | 55 |
| Available slots per week | 3 |
| Assignment session allocation | 4, 5, and 3 sessions |

These example values are not global defaults. The required capacity experiment
uses 2, 3, and 4 available slots per week.

## Proposed CSP formulation

### Implemented generic foundation and baseline BT

The current foundation represents one CSP problem for a complete
WholePlanManifest. The manifest binds the student, course, common mapping
scenario, and exactly three distinct assignment entries. Each immutable
AssignmentCandidateIdentity includes student, course, assignment, allocation,
base-feature-context, and mapping-scenario identities and has its own stable
digest. The manifest checks that each candidate agrees with the whole-plan
student, course, assignment, and scenario. Every entry remains present,
including allocations of zero sessions. Its canonical identity is a stable
digest of the canonicalized manifest fields. Session variables and domains
must match this manifest exactly.

The problem also has a separate deterministic problem identity that includes
the whole-plan identity, weekly capacity, domains, the finite ordering-policy
snapshot, and each constraint's configuration identity. Evaluations and solver
outcomes are bound to both identities so a result cannot be reused with a
different capacity, domain catalog, or declared constraint configuration.

At problem construction, the policy's required session pairs and its boolean
slot-order relation over the finite union of domain slots are captured in an
immutable `OrderingPolicySnapshot`. Ordering constraints are rebound to this
snapshot, so later mutation of the caller's policy cannot change problem
behavior without changing its already-computed identity. Opaque policies that
cannot provide stable identity, immutable session pairs, or boolean results
over the problem's finite slot set are rejected. This captures only the
explicitly supplied relation; it does not choose temporal ordering semantics.

CspEvaluation carries both the whole-plan identity and CSP problem identity
and is an intermediate local validator record. It can report
BRANCH_REJECTED, INCOMPLETE, or FEASIBLE. A rejected partial branch means only
that branch cannot be extended. A valid partial assignment is incomplete; a
complete valid assignment is locally feasible. The validator never reports
whole-plan infeasibility. Only a solver boundary may issue
CspSolverOutcome.INFEASIBLE, and it must attest exhaustive search. A resource
limit, cancellation, or other early stop is INCOMPLETE. The validator factory
requires exact coverage of the problem variables, values in their domains,
passing constraints, and the matching whole-plan and problem identities before
creating FEASIBLE. The solver factory independently repeats those checks for
an attached feasible evaluation. Partial, out-of-domain, or constraint-violating
assignments cannot become FEASIBLE through these APIs. The private factories
are application-level conventions, not security boundaries against deliberate
Python object fabrication.

This contract is not the final WholePlanVerificationResult used by the NN
adapter. That future result still needs the complete schedule identity,
verification identity, policy version, per-constraint check evidence, and
whole-plan identity. No assignment-level final NN evaluation may consume a
local CspEvaluation.

### Variables

For each candidate allocation, let `n[a]` be the number of sessions Search has
assigned to assignment `a`. Each required study session is a variable
`S[a,k]`, where `a` identifies its assignment and `k` distinguishes sessions
belonging to that assignment. Its value is one weekly time slot from the
supplied slot domain. The index distinguishes session variables; it does not
define an ordering relation, session duration, or spacing rule.

The CSP instance has `sum_a n[a]` variables. A zero-session allocation has no
variables and is feasible if the input itself is valid.

### Domains and their source

The input must provide an immutable weekly-slot catalog. A domain is a set of
slot identifiers available in a week; the catalog must supply the association
between slots and weeks. The initial domain for `S[a,k]` contains slots
allowed for assignment `a` by its deadline. Capacity `C` in an experiment is
the number of available slot values per week; unique-slot assignment limits
the number of sessions placed there to that capacity. For the required
experiment, run `C ∈ {2, 3, 4}`. The example's three slots/week is not a
universal setting. Use abstract slot identifiers until an authoritative slot
source is specified; do not invent calendar slots or durations.

Empty input domains are invalid configuration when required slot data is
missing. An empty *filtered variable domain* is a valid indication that this
candidate cannot be scheduled.

### Hard constraints

| Constraint | Required condition | Proposed representation |
| --- | --- | --- |
| Deadline | Every session occurs before its assignment deadline. | Remove deadline-ineligible slots from each session domain and validate again on complete assignments. Exact week conversion and boundary remain unresolved. |
| Unique slot | Only one session may use a slot. | Each slot value can be assigned to at most one session variable across the whole plan. |
| Same-assignment ordering | Sessions belonging to one assignment are ordered. | Apply the official ordering relation after its temporal meaning is specified; do not infer that it means distinct or strictly increasing weeks. |
| Weekly capacity | Experiment with 2, 3, and 4 available slots/week. | Provide capacity as a configurable problem input and check placements across the whole plan. Slot domains must also be built from the selected capacity configuration. |
| Other P1 hard constraints | No additional scheduling constraints are explicitly established in the available brief. | Do not add study-day spacing, session duration, workload balance, or assignment-specific preferences without an approved requirement. |

The implementation foundation also carries a configurable weekly-capacity
constraint. It is checked across sessions belonging to every assignment in
the whole-plan manifest. The selected capacity is an input, never a hardcoded
example value. The slot catalog and capacity must agree with the selected
experiment configuration; how that catalog is generated remains unresolved.

### Constraint propagation

**Forward Checking (FC).** After placing a session in a slot, remove that used
slot from the domains of all unassigned session variables. If any unassigned
variable has an empty domain, undo the placement and backtrack. Restore domain
changes exactly when returning from recursion. Additional forward propagation
for deadlines or ordering depends on the resolved input semantics and must be
recorded as part of the chosen method.

**AC-3.** The generic binary-constraint interface exposes variable scope,
pair compatibility, support checks, and domain revision directly. A future
solver can revise a session variable's weekly-slot domain without constructing
artificial full assignments, then enqueue adjacent arcs when a domain changes.
The foundation provides binary relations for unique-slot and policy-derived
same-assignment ordering checks. Deadline eligibility is unary; weekly
capacity is a whole-plan constraint. The precise ordering arcs and slot
catalog still depend on unresolved project inputs.

AC-3 applies to binary constraints only. Weekly capacity is checked as a
separate whole-plan constraint; no claim is made that AC-3 alone enforces this
global constraint.

**Ordering coverage.** The manifest determines each assignment's declared
session variables. An explicit approved ordering policy derives the required
session pairs and supplies the slot relation. The foundation rejects missing,
duplicate, reversed, or cross-assignment pairs and requires every session in
an assignment with multiple sessions to participate in a relationship. The
session index supplies canonical pair direction only; it does not define how
slots must be ordered. No week-level or within-week ordering rule is selected
by the foundation.

### Backtracking and deterministic ordering

Use recursive depth-first backtracking to place each session variable in an
available weekly slot. BT validates each partial assignment through the
existing validator and keeps mutable recursion state separate from the
immutable problem and its domains. A rejected partial branch is abandoned;
only complete assignments passing the validator return FEASIBLE. The first
solution API returns one result; the lazy enumerator can continue through
subsequent feasible schedules.

The implemented deterministic ordering is configurable. Its default is:

1. Choose the unassigned variable with the smallest current domain (MRV).
2. Break MRV ties by highest number of constraints with other unassigned
   variables (degree heuristic).
3. Break remaining ties by canonical `(assignment_id, session_index)` order.
4. Rank domain values by a pairwise least-constraining-value (LCV) probe against
   each unassigned neighbor, with canonical slot identity as the deterministic
   tie-break. For n-ary constraints this is a heuristic ordering measure, not
   full n-ary lookahead. Probes only order values; they do not remove values or
   prune branches. `canonical` variable/value policies are also available for
   controlled comparisons.

Variable ordering uses dynamic MRV without changing domains: for each
unassigned variable, test each original-domain value against the current
partial assignment and count values that remain consistent. Select the lowest
count, then highest degree among unassigned variables, then canonical variable
identity. These probes are observational checks only; they do not filter the
domain or establish FC.

No Forward Checking or AC-3 propagation is performed. With no node limit,
search reports INFEASIBLE only after all values at every reachable partial
assignment have been rejected or exhausted. `max_nodes` limits entered
recursive states; reaching it returns INCOMPLETE, including when the limit is
zero. It never converts an interrupted search to infeasibility.

The lazy `BacktrackingEnumerator` retains its traversal between requests and
returns one distinct schedule at a time in deterministic order. After the last
schedule it returns EXHAUSTED only after traversal finishes. A cumulative node
limit returns INCOMPLETE, even if earlier calls returned schedules; it never
establishes that no further schedule exists. INCOMPLETE is terminal for that
enumerator: repeated calls return the cached terminal result, and the same
instance cannot resume. Creating a new enumerator restarts from the first
solution. The first-solution helper wraps one enumerator request. A failed NN
evaluation can call `next_solution()` on the same non-incomplete enumerator to
continue without restarting.

BT statistics have these definitions: `nodes_visited` counts entered partial
assignment states, including root and rejected states; `assignments_attempted`
counts variable/slot extensions before recursion; `backtracks` counts attempted
extensions whose subtrees were proven solution-free; `constraint_checks` counts
every actual call to a constraint's `is_satisfied` predicate, including node
validation, dynamic-MRV probes, pairwise LCV probes, and independent final
solver-outcome validation. Validator evidence is reused when constructing its
evaluation to avoid repeating the same predicate pass; solver-boundary
validation remains an independent pass and is counted. `maximum_depth` is the
greatest number of assigned variables in an entered state.
`solver_time_seconds` is cumulative active time spent advancing search and
performing solver-boundary validation. `wall_time_seconds` is elapsed time
since enumerator creation and can include caller pauses or external NN work. A
node-limit attempt stopped before entering a state contributes an assignment
attempt but not a visited node.

## Search, NN, and CSP integration

1. UCS or Greedy emits a candidate allocation over the three assignments.
   Search remains responsible for its own expansion order and session cost.
2. The Search-facing NN adapter may calculate a provisional prediction using
   the documented provisional week distribution. This prediction is not a
   final goal result and does not prove schedule feasibility.
3. CSP receives a Search proposal that reaches the provisional NN target, plus
   the complete three-assignment allocation, weekly slot domains, deadlines,
   capacity setting, and ordering policy. A provisional non-goal returns to
   Search; whether a schedule-aware exception to that pruning is required is
   unresolved because final scheduled weeks can change NN features. CSP
   verifies the whole schedule before emitting a feasible result. That result
   contains the whole schedule plus assignment-level projections.
4. Only after whole-plan feasibility is established may the NN adapter evaluate
   any assignment's final prediction. It derives each assignment's actual
   weekly windows and evaluates against that assignment's immutable feature
   context. If any required final prediction misses target, request the next
   distinct whole-plan schedule. If CSP exhausts schedules, reject the
   allocation and resume the same Search frontier.
5. A plan is accepted only after successful final prediction for a
   CSP-verified schedule. If Search exhausts its finite allocation space and
   all relevant candidates/schedules were evaluated, report unreachable at
   available capacity and the best achievable final predicted marks.

### Whole-plan identity and assignment projections

**Proposed contract.** Define immutable `WholePlanIdentity` with:

- `student_identity` and the single-course `course_identity`;
- exactly three distinct `AssignmentPlanIdentity` entries in canonical
  assignment-ID order;
- each entry's `assignment_identity`, canonical allocation identity (the
  four deadline-relative session counts used by the current adapter), and
  base feature-context fingerprint;
- the common mapping-scenario identity.

The assignment projection is a pure lookup from one whole-plan entry to the
existing `CandidateIdentity` fields: student, assignment, that entry's
allocation tuple, that entry's feature-context fingerprint, and the common
mapping scenario. The CSP verifier and Search must compare the whole-plan
identity, not just the selected assignment projection. This preserves the
current per-assignment NN feature row while ensuring all three assignment
allocations are part of CSP's feasibility claim.

**Proposed `WholePlanVerificationResult`.** Replace the assignment-scoped
verification evidence used for final evaluation with one immutable record per
complete whole-plan schedule checked. It contains:

- `whole_plan_identity`;
- `status`: feasible or rejected for this complete schedule;
- a canonical `schedule_identity` and the complete schedule partitioned by
  assignment, retaining each session's canonical slot identity and week;
- a stable `verification_identity` and `verification_policy_version`;
- immutable `constraint_checks` for every applied constraint instance, each
  recording constraint ID, scope (for example, assignment/session, week, or
  slot), checked/passed/not-run outcome, and failure reason/details;
- structured rejection reasons when any required check fails.

At minimum the policy records checks for `deadline_before_cutoff`,
`weekly_slot_unique`, `same_assignment_order`, and `weekly_capacity`. Any
additional implemented hard constraint must also be named and recorded; no
unnamed constraint may silently affect feasibility. Policy version identifies
the deadline convention, ordering semantics, slot catalog/source version,
capacity configuration, and slot-identity semantics. For a `feasible` result,
every required check must be present and passed. A rejected schedule is not
proof that the whole plan has no feasible schedule.

The lazy schedule enumerator also returns a separate immutable
`WholePlanSearchOutcome` when it stops. Its status distinguishes exhaustive
enumeration with no feasible schedule, exhaustive enumeration after one or
more feasible schedules, a resource-limited/cancelled run, and an operational
error. It records the whole-plan identity, count and identities of yielded
schedules, and structured rejection details. A limit or error proves neither
feasibility nor infeasibility. Final-target failure across all yielded
schedules is distinct from CSP infeasibility.

The existing `CspVerificationResult._from_solver` and
`SessionNNAdapter.evaluate_final` are assignment-scoped contracts. They do not
yet implement this whole-plan result. The proposed adapter interface accepts a
`WholePlanVerificationResult`, expected `WholePlanIdentity`, the assignment
allocation, and its feature context. It rejects a non-feasible result or a
whole-plan identity mismatch, projects and compares the selected
`CandidateIdentity`, then obtains only that assignment's schedule projection.
It checks that the projection's session count matches that assignment's
allocation before mapping actual windows. A raw schedule or assignment-local
feasibility result is never sufficient.

The adapter may produce an assignment-level **final prediction** only from a
feasible whole-plan schedule result. Search may accept the plan only after all
required assignment predictions for the same whole-plan identity, schedule
identity, and verification identity satisfy the eventual target rule.
Provisional predictions remain separate and can never substitute for any
final result.

The solver factory remains an application-level provenance convention, not a
security boundary against deliberate Python object fabrication.

The implemented generic CspProblem and CspEvaluation are whole-plan scoped,
but CspEvaluation is only a local assignment validator. CspSolverOutcome
separates solver termination from branch validation and permits INFEASIBLE
only when its solver boundary marks the run exhaustive. These records do not
replace WholePlanVerificationResult and cannot yet authorize final NN
evaluation.

## Failure handling and termination

- Reject malformed input before solving: missing/duplicate assignment IDs,
  invalid session counts, malformed slot metadata, invalid capacity settings,
  or missing deadline/slot-domain inputs.
- If any variable domain becomes empty, backtrack. Exhaustive failure returns
  infeasible with implicated constraints where available.
- A zero-capacity week removes all its slots from eligible domains; if every
  candidate session loses all values, the candidate is infeasible.
- Duplicate complete schedules are keyed by the canonical full
  assignment-to-slot mapping and returned at most once.
- Yield feasible whole-plan verification results lazily in deterministic
  order, then return a separate `WholePlanSearchOutcome`. The
  integration layer evaluates all required final NN predictions for each
  schedule, stops on a target-reaching plan, and otherwise requests the next
  schedule. Exhaustion is needed before concluding that no schedule for this
  allocation meets target. Cancellation or a resource limit remains incomplete.
- Do not convert NN errors, schema errors, or support-policy exclusions into a
  CSP infeasibility result.

## Evaluation

Record per CSP invocation:

- **Nodes explored:** partial assignments selected for extension, including
  the root node; state the convention and use it consistently.
- **Backtracks:** assigned-variable branches abandoned because propagation
  produced a contradiction or all values failed.
- **Runtime:** monotonic wall-clock duration from validated problem entry to
  solver result; report separately from NN and Search runtime unless the
  experiment explicitly measures the integrated pipeline.
- **Feasibility outcome:** feasible schedule, exhaustive infeasible, limited,
  cancelled, or operational error.
- **Schedules found:** number of distinct feasible schedules yielded when
  enumeration is requested.

The required experiment has these nine configurations:

| Solver configuration | 2 slots/week | 3 slots/week | 4 slots/week |
| --- | --- | --- | --- |
| BT | Required | Required | Required |
| BT+FC | Required | Required | Required |
| BT+FC+AC-3 | Required | Required | Required |

Use the same CSP instances when comparing configurations at each capacity.
Report success/feasibility, nodes explored, backtracks, and runtime; identify
the exact counting and timing boundaries. Any aggregation across repeated runs
must state its run/seed policy. Separate exhaustive infeasibility from limits
and operational errors. Every reported feasible solution must pass an
independent final constraint check. CSP success by itself does not mean the
predicted-score target was met.

## Explicit unresolved decisions

- Authoritative source and generator for abstract weekly slot identifiers and
  week associations; no scheduling-slot catalog has been found in the
  repository. Exact calendar identities, times, and durations are unspecified.
- Whether the required capacity value is uniform across weeks and how the
  corresponding abstract slot domains are generated. The experiment defines
  capacity as 2, 3, or 4 available slot values per week; no other capacity
  semantics are established.
- Exact meaning of “sessions ordered”: strictly increasing distinct weeks,
  chronological order among slots (possibly same week), or another relation.
- Deadline comparison convention and how day-offset deadlines map to weekly
  slots; the repository's NN feature cutoff does not define CSP slot semantics.
- Whether slots need duration or overlap fields. Do not assume a session
  duration from the NN click scenarios.
- How the full three-assignment CSP verification identity and schedule project
  to the current assignment-scoped NN adapter contract.
- Resource-limit policy required before claiming target unreachability. The
  proposed behavior is deterministic lazy enumeration until a complete plan is
  accepted or all schedules are exhausted; the official materials available
  here do not prescribe the enumeration algorithm/order.
- Whether the assignment requires particular variable/value ordering, AC-3
  invocation frequency, or comparative performance thresholds beyond requiring
  Backtracking, Forward Checking, and AC-3.
- Target aggregation and support policy remain Search/NN integration decisions;
  they must not be inferred by CSP.
- The official previous-mark NN input requirement remains unresolved against
  the current strict NN schema, which excludes prior scores. CSP must not add
  such features.

The generic CSP contracts are implemented separately from the solver. No
Backtracking, Forward Checking, or AC-3 algorithm is implemented. No Search,
NN, dataset, or model artifact was changed.
