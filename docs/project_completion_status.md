# Project completion status

## Whole-Plan Verification

**Status: implemented in the CSP layer.** Immutable verification binds a
complete three-assignment manifest and schedule to the CSP problem, expected
identities, policy identity/version, individual check evidence, outcome, and
canonical rejection reasons.

## Search–NN–CSP integration

**Status: implemented and locally verified.** The integration layer provides:

- A lazy bridge for BT, BT+FC, and BT+FC+AC-3. It verifies each yielded
  assignment, preserves zero-session entries, and distinguishes exhaustive
  enumeration from resource-limited enumeration.
- A final NN adapter boundary that requires feasible whole-plan verification
  and checks expected whole-plan, CSP problem, policy, schedule, and verification
  provenance before mapping actual assigned windows.
- UCS and Greedy orchestration that continues after CSP allocation rejection,
  tries later schedules after final NN target misses, and returns `INCOMPLETE`
  if candidate or CSP node limits prevent a conclusive run.
- Typed planner results with three allocations, the whole-plan timetable,
  per-assignment predictions and targets, NN identity/mode, verification
  provenance, counters, status, and failure reasons.

Model A strict features and Model B previous-marks proxy features retain their
separate schemas and existing predictor inference methods. Model B remains a
proxy mode and its prior-score availability caveat is unchanged. Final
provenance uses a SHA-256 digest over the exact persisted schema, configuration,
preprocessor, and checkpoint bytes consumed by the artifact loader. The adapter
combines this with model mode and feature-schema identity; each prediction
record has its own deterministic identity bound to its output and schedule
context. Missing artifact digests are rejected. Planner results report the
same complete model provenance identity carried by each prediction.

## Phase 1: Application Inputs

**Status: typed input layer and Search–CSP wiring implemented and tested;
production configuration values remain missing.**
`src/study_planner/integration/application_inputs.py` loads validated
later-presentation feature rows. `src/study_planner/integration/application_planning.py`
converts them into a deterministic allocation state space and a per-allocation
CSP candidate factory. It:

- Restricts the configured OULAD train/test presentations to the established
  earlier-train/later-test split and supports one course.
- Requires three assignment records and a versioned explicit target rule.
  Target values are supplied; the code does not derive a business target from
  historical `target_score` labels.
- Checks each row's assessment deadline and the existing
  `weeks_since_course_start = deadline_day / 7` feature convention.
- Requires a versioned slot catalog, weekly capacity, allocation bounds,
  explicit local weekday/time windows with timezone identity,
  deadline-eligibility policy identity, eligible slot IDs, and per-assignment
  relative-window mappings. The factory applies eligible slot IDs as CSP
  domains, uses catalog weekly capacity, and carries each assignment's window
  map to final NN evaluation. It checks the existing
  `weeks_since_course_start = deadline_day / 7` feature convention. It does not
  infer eligible slots from the clock window or deadline.
- Carries and applies a versioned marginal support disposition to provisional
  and final mappings. Joint support remains unassessed.
- Loads Model A Strict or Model B Previous Marks through their corresponding
  verified loaders and captures artifact digest, model provenance, schema, and
  mode in the input configuration.
- Excludes historical `target_score` from model feature contexts and validates
  strict/proxy schemas through the existing mapping and dataset loaders.

Focused tests cover both schemas through the public Search/CSP/NN path, split
and identity failures, three-assignment coverage including zero sessions,
target/deadline consistency, bounds, slot/time-window validation,
eligible-domain construction, support disposition, and model provenance.
Assignment-relative windows remain separate through the planner.

Production execution requires the user to provide: (1) course and approved
train/test presentations; (2) three assignment identities and assessment IDs;
(3) target marks and target-rule identity/version; (4) each deadline day,
deadline-policy identity/version, and eligible slot IDs; (5) min/max session
bounds per assignment; (6) versioned slot catalog identity/version, slot IDs
and week indexes, local weekday/start/end and timezone, weekly capacity, and
each assignment's click-window mapping; (7) support policy/reference identity
and REPORT_ONLY or REJECT disposition; (8) generated-click mapping
scenario/group/increment; (9) student identity and validated later-presentation
feature rows; (10) exact Model A or Model B artifact directory; and (11) an
explicit same-assignment ordering policy. This phase invents none of those
production values or ordering semantics.

## Exact next phase

`configs/p1_production_template.json` records the values verified from the
local OULAD manifests, assessment table, and loaded Model A/B artifacts. It is
deliberately marked incomplete. For the candidate 2014J TMA set 34899/34900/34901,
the later test split has 914 students with one row per assessment; the first 30
numeric student IDs are listed in the template. All 30 have all 19 strict
source features present on all three rows and all 12 click fields finite. The
proxy rows have all five availability indicators present; prior scores may be
missing by design, and the earliest TMA has no prior mark.

Next obtain approval for the candidate assessment set and supply its target
marks, slot/deadline eligibility rules, actual catalog/time windows/capacity,
allocation bounds, ordering policy, support disposition, generated-click
scenario/window rule, chosen student, and exact model mode/artifact. Then run
the planner on real inputs and execute the required experiments. Current
end-to-end tests use explicitly labeled TEST configuration.
