# P1 Production Configuration Decisions

**Status: illustrative demo draft, not production-ready.** The original template
remains unchanged. The classification and current values are in
`configs/p1_production_draft.json`.

## Specification source

The requested file `IS Projects(10).pdf` was not found in the repository,
project workspace, or connected document sessions. The entries below therefore
record the decisions and requirements stated in the latest task message and
cross-check them against repository code/docs; they do not claim a completed
PDF review. Reconcile this record with the PDF when it is made accessible.

## Decision table

| Item | Status | Current value / rationale | Still needed |
|---|---|---|---|
| Course and split | Approved for illustrative demo; data verified | FFF; train 2013B/2013J, validation 2014B, test 2014J | None for this demo scope |
| Candidate assessment set | Approved for illustrative demo, retain candidate status | FFF 2014J TMA IDs 34899/34900/34901; deadlines 24/52/94; they remain a data-derived candidate set | Assign business-facing identities; do not re-label as an officially selected assignment set |
| Targets | Approved illustrative generated setting | 55 per assessment, per the task decision; not OULAD-derived and not a universal target | Target-rule identity/version required by `TargetRule` |
| Weekly capacity | Approved illustrative generated setting | 3 sessions per week, per the task decision; not a measured learner capacity | Encode in `WeeklySlotCatalog`; do not infer per-assignment allocation bounds from it |
| Model and inputs | Approved explicit choice; artifact verified | Model B Proxy; artifact `data/processed/nn_experiment_previous_marks`; matching features `data/processed/proxy`; 29 source features | Preserve null scores and availability indicators. Submission timing does not prove mark-release timing |
| Search/CSP methods | Approved requirement from task message; supported by APIs | UCS and Greedy Best-First; Backtracking, Forward Checking, AC-3 | Select the algorithms for each run/experiment |
| Deadline, slot uniqueness, assignment ordering | Approved constraint intent from task message; partly represented | A session must precede its assignment deadline, no two sessions use one slot, same-assignment sessions are ordered | Build the catalog and eligible slot IDs; approve exact deadline boundary and ordering relation |
| NN inputs and click history | Approved requirement from task message; Model B schema supports it | Previous-score proxies plus availability flags, recent clicks across three groups/four windows, assessment type, weight, weeks since course start | Maintain separate Model B schema; do not add/impute absent marks |
| Click increment/group/scenario | Proposed generated choices, not approved requirements | `content_navigation`, central, 62 clicks is a training-anchored proposal in existing readiness docs/code; not a per-session measurement. The PDF was not checked | Confirm justification and mapping implementation; select/approve group, scenario and increment |
| Allocation bounds | Unresolved | No numeric bounds are specified here. Capacity 3 does not imply bounds 0–3 | Choose min/max independently and document rationale |
| Slot windows/catalog | Proposed generated choices, not actual availability | `configs/p1_production_draft.json` records a synthetic week 0 starting Monday 00:00 UTC, with Monday/Wednesday/Friday 18:00–19:00 UTC slots expanded across weeks 0–13 (42 stable IDs). It is a scenario, not learner availability. Weekly capacity remains the separately approved 3. | Approve/replace the synthetic weekday pattern, time windows, timezone identity, course-week anchor, and expansion range. Per-assignment bounds remain independently unresolved. |
| Deadline eligibility | Implemented proposal; boundary assumption documented | The professor requires a session before its assignment deadline but does not define deadline-day inclusivity. The implementation conservatively models deadline day `D` as 00:00 on course-relative day `D` and requires the whole slot to end strictly before `D 00:00`. Thus, a slot ending exactly at the boundary is ineligible. This is an implementation assumption, not a professor-specified rule. NN click-history windows are assigned independently. | Approve/replace the synthetic calendar anchor and timezone. The strict cutoff is explicitly an implementation assumption; it does not make four-window containment a CSP eligibility condition. |
| Same-assignment ordering | Implemented proposal; approval pending | `CatalogChronologicalOrderingPolicy` orders by course-week index, weekday offset from the explicit course-week anchor, and local start minute. | Approve or replace the ordering semantics along with the synthetic calendar. |
| Support policy | Proposed choice | Strict-training click min/max reference; `REPORT_ONLY`; joint support stays unassessed | Confirm reference identity/version and disposition; do not attribute `REPORT_ONLY` to the PDF |
| Demo profile | Proposed, not approved | Student 40878 is in the template’s verified candidate list and is the first numeric ID, a reproducible tie-break only | Approve 40878 or select another verified candidate. Its 34899 prior-score lags remain null with false availability indicators |
| Impossible result | Required behavior from task message; implementation gap | `FinalPlannerResult` can carry predictions on infeasible outcomes, but `planner.py` retains `last_scored`, not the best achievable plan | Define the comparison criterion, implement best-plan retention, and add regression coverage before claiming compliance |

The proposed click values are supported as training-distribution summaries, not
as causal/session measurements. `docs/search_integration_readiness.md` records
content-navigation positive-total anchors of 22/62/121. Nothing in this record
attributes 62, `content_navigation`, UTC, student 40878, or `REPORT_ONLY` to
the missing PDF.

## API and minimum configuration

The approved choices fit existing typed APIs: `TargetRule` carries marks and a
version; `WeeklySlotCatalog` carries weekly capacity and slots;
`SessionClickMappingConfig` selects one existing click group/scenario;
`ApplicationAssignmentRule` carries deadline, eligible slot IDs, and bounds;
`SupportPolicy` carries the support disposition; and
`build_application_planning_problem` accepts an explicit ordering policy.
`integrated_uniform_cost_search` and
`integrated_greedy_best_first_search` expose the Search choices, while
`CspAlgorithm` exposes BT, BT+FC, and BT+FC+AC-3. No architecture change is
needed to represent these values. There is no JSON-to-dataclass loader, so an
approved draft must be assembled into the typed objects through existing APIs.

The slot builder now creates the concrete proposal using the existing typed
APIs: `build_synthetic_weekly_slot_catalog` expands IDs, computes each
assignment's deadline-eligible IDs and independent four-window mappings, and
`CatalogChronologicalOrderingPolicy` supplies CSP ordering. `ApplicationInputConfig`
checks that CSP-eligible slots meet the strict deadline boundary and that every
NN mapping matches its four-window rule. A deadline-eligible slot outside the
four NN history windows remains in the CSP domain without a click-history
mapping. These generated choices remain a proposal until approved. Other
unresolved demo inputs remain:
configuration and target-rule identities/versions, independent per-assignment
allocation bounds, selected click group/scenario/increment, support reference
and disposition, and an approved profile. The synthetic slot catalog must stay
labeled hypothetical. Best-plan reporting is implemented and regression-tested.

A **later real-profile run** additionally needs approval of a specific profile
from the verified candidate list and use of that profile’s preserved proxy
rows. It must disclose that prior-score availability is unverified. Do not
describe the proposed UTC slots as that student’s availability.

## Exact next action

Make `IS Projects(10).pdf` accessible in the workspace and review it against
this record. Then approve a concrete synthetic slot/eligibility catalog and
independent allocation bounds; these are the minimum unresolved inputs needed
to construct `ApplicationInputConfig`. Before demonstrating the impossible
case as compliant, implement and test best-achievable-plan reporting. Do not
run the planner using the current unapproved proposals.
