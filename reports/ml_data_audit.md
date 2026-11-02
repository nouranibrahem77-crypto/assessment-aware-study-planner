# FFF feature pipeline audit

## Scope and conclusion

Reviewed the feature builder, input validation, presentation splitting,
persistence, synthetic tests, generated CSVs/manifests, and the generated feature
reports. No model was trained. The pipeline enforces the deadline-day cutoff
for click features and contains no target score or target submission fields in
the feature matrix. The fixed presentation split is intact.

**The proxy configuration is not verified leakage-free:** OULAD has no grade
release timestamps. A score submitted on time for an earlier assessment may
still have been unpublished at the current prediction cutoff. The default is
therefore the strict configuration, which excludes all previous-score
features. The proxy configuration remains available for sensitivity analysis
with this assumption made explicit in its manifest.

## Findings

| Severity | Finding | Evidence and fix |
| --- | --- | --- |
| High, unresolved assumption | Prior-score publication time is absent from OULAD. | Proxy mode requires a valid, non-banked score for the same student and presentation, submitted on time for a strictly earlier assessment. That does not prove the grade was released by the cutoff. Added strict mode and made it the default; use proxy only with the assumption documented. |
| Pass | Click cutoff is exclusive at the start of deadline day D. | Click feature windows are `[D-28,D-21)`, `[D-21,D-14)`, `[D-14,D-7)`, and `[D-7,D)`. Indexing excludes event dates `>= D` and includes `D-28`. Unit tests cover day D and future days. |
| Pass | Target score and target submission are isolated from features. | `target_score` is persisted as a label after the feature columns; `date_submitted` is not persisted. A mutation test changes a target score while asserting all input features are identical. |
| Pass | Cross-presentation joins are scoped. | Assessment-score keys include presentation and student; VLE site keys include module, presentation, and site; target events are grouped by presentation and student. A test reuses a student ID across presentations and checks no prior-score carryover. |
| Pass | Join multiplication and duplicate submissions fail closed. | Course presentation, assessment ID, and VLE site keys are checked for duplicates; repeated student-assessment keys raise `FeatureDataError`. Duplicate VLE metadata and duplicate submission failure tests pass. No duplicates were encountered while generating the real artifacts. |
| Pass | Late and banked results are excluded. | Target/prior score eligibility rejects submissions after their assessment deadline and `is_banked=1`; failure-case tests verify both exclusions. |
| Pass | Missing scores and click counts are distinguishable from zero. | Missing prior scores remain blank with an availability indicator in proxy mode. A known window with no matching event is zero. Unknown site joins or invalid event counts mark affected click features missing. Real FFF data had zero unmatched VLE joins and zero invalid click rows. |
| Pass | Activity types are explicit and unknown types fail closed. | The 18 observed FFF `activity_type` values map to the three documented groups. Input validation rejects any unmapped FFF type; a failure-case test covers this. |
| Pass | Split boundaries and preprocessing fit scope are fixed. | Train = 2013B + 2013J; validation = 2014B; test = 2014J. Type vocabulary and proxy lag count derive from training schedule metadata only. There is no fitted imputation or scaling; no rows are randomly redistributed. |

## Verified data counts

Counts were read from the current local OULAD files and generated artifacts.
The strict and proxy configurations retain identical labels and split sizes.

| Source | All rows | FFF rows in selected presentations |
| --- | ---: | ---: |
| `courses.csv` | 22 | 4 |
| `assessments.csv` | 206 | 52 |
| `studentAssessment.csv` | 173,912 | 54,815 |
| `studentVle.csv` | 10,655,280 | 4,014,499 |
| `vle.csv` | 6,364 | 1,967 |

| Split | Presentation(s) | Scored rows | CMA | TMA | Strict feature columns | Proxy feature columns |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Train | 2013B, 2013J | 25,229 | 14,528 | 10,701 | 19 | 29 |
| Validation | 2014B | 9,013 | 5,144 | 3,869 | 19 | 29 |
| Test | 2014J | 14,241 | 8,260 | 5,981 | 19 | 29 |
| **Total** | — | **48,483** | **27,932** | **20,551** | — | — |

Feature-column counts exclude four identifiers and the `target_score` label.
Strict CSVs have 24 total columns; proxy CSVs have 34. Proxy adds five score
lags and five paired availability indicators.

| Dropped student-assessment records | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Banked (`is_banked=1`) | 196 | 156 | 268 |
| Late submission | 2,984 | 1,016 | 1,666 |
| Missing or invalid score | 26 | 11 | 9 |

The source has 46 blank FFF score cells across the four presentations; these
are included in the missing/invalid score exclusions above. FFF `vle.csv` has
1,141 missing `week_from` and 1,141 missing `week_to` values; those planned
resource-week fields are not used as features. No click feature cells were
missing in the real generated data. Proxy-mode unavailable prior-score cells
are intentionally blank and are counted per lag in
`reports/feature_engineering_proxy.md`.

## Configurations and generated files

- Strict, default: `data/processed/fff_{train,validation,test}.csv` and
  `data/processed/fff_preprocessing.json`.
- Proxy: `data/processed/proxy/fff_{train,validation,test}.csv` and its
  `fff_preprocessing.json`.
- Corresponding descriptions: `reports/feature_engineering.md` (strict) and
  `reports/feature_engineering_proxy.md` (proxy).

Both manifests contain the same split row counts and identify the selected
score mode. The processed outputs are ignored by Git. Neither configuration
fits a statistical imputer or scaler. Assessment type categories and proxy lag
schema are determined from training schedule metadata only.

## Remaining assumptions and limits

- Proxy prior scores cannot be confirmed as released before the cutoff. Strict
  mode removes this dependency; it does not establish that every other source
  field has a publication timestamp.
- OULAD `studentVle.date` is a day offset, not a timestamp. The implementation
  treats all activity on day D as unavailable at the start of D and all records
  on dates less than D as available before the cutoff.
- A missing event in `studentVle.csv` is treated as zero recorded clicks for a
  known resource/window. The source does not provide a separate completeness
  indicator for event capture.
- The target score is a retrospective label. This audit establishes that it is
  excluded from features, but does not claim the label would be available at
  prediction time.

The results support a **cutoff-enforced strict feature dataset**, with these
source-timing and event-completeness limitations. They do not justify calling
the proxy configuration leakage-free.
