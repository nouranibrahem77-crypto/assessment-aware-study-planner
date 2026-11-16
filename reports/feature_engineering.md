# FFF feature engineering report

This report describes generated OULAD FFF features. It is descriptive
only: click activity is not interpreted as causing assessment scores.
No demographic variables or inferred study sessions are used.

## Prediction cutoff and leakage controls

For each target assessment with deadline day D, the prediction cutoff is
the start of day D. Clicks use only day offsets less than D and within
[D-28, D); the four half-open weekly windows are reported in the manifest.
Target submissions after their own deadline and banked results are dropped.
The target score and submission date are stored separately from model
features.

Strict mode excludes all previous-assessment-score features. No prior-score availability assumption is used.
No scaling or imputation parameters are fit.
Assessment type categories and the number of prior-score lag slots are
derived only from the 2013B/2013J training assessment metadata.

## Input row counts

Counts cover complete source files; FFF counts cover the selected course.

| Source file | All rows | FFF rows |
| --- | ---: | ---: |
| `assessments.csv` | 206 | 52 |
| `courses.csv` | 22 | 4 |
| `studentAssessment.csv` | 173912 | 54815 |
| `studentVle.csv` | 10655280 | 4014499 |
| `vle.csv` | 6364 | 1967 |

## Activity type mapping

Types below are resource counts from FFF `vle.csv`; every observed type
was assigned to exactly one documented group.

| `activity_type` | Group | VLE resource rows |
| --- | --- | ---: |
| `dataplus` | `content_navigation` | 20 |
| `dualpane` | `content_navigation` | 17 |
| `folder` | `content_navigation` | 2 |
| `forumng` | `collaboration` | 25 |
| `glossary` | `content_navigation` | 4 |
| `homepage` | `content_navigation` | 4 |
| `htmlactivity` | `content_navigation` | 4 |
| `oucollaborate` | `collaboration` | 47 |
| `oucontent` | `content_navigation` | 419 |
| `ouelluminate` | `collaboration` | 16 |
| `ouwiki` | `collaboration` | 26 |
| `page` | `content_navigation` | 86 |
| `questionnaire` | `assessment_activity` | 57 |
| `quiz` | `assessment_activity` | 44 |
| `repeatactivity` | `content_navigation` | 5 |
| `resource` | `content_navigation` | 454 |
| `subpage` | `content_navigation` | 215 |
| `url` | `content_navigation` | 522 |

## Fixed chronological splits

Rows were never randomly redistributed.

| Split | Presentations | Scored assessment rows | CMA | Exam | TMA | Score bands 0-39 / 40-59 / 60-79 / 80-100 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| train | 2013B, 2013J | 25229 | 14528 | 0 | 10701 | 350 / 1936 / 8981 / 13962 |
| validation | 2014B | 9013 | 5144 | 0 | 3869 | 129 / 731 / 3469 / 4684 |
| test | 2014J | 14241 | 8260 | 0 | 5981 | 151 / 846 / 4838 / 8406 |

Assessment type distribution by split:

| Split | Type counts |
| --- | --- |
| train | CMA: 14528, TMA: 10701 |
| validation | CMA: 5144, TMA: 3869 |
| test | CMA: 8260, TMA: 5981 |

## Dropped submission records

Only records present in `studentAssessment.csv` can be counted here;
students with no recorded submission have no score row to retain.

| Split | Dropped rows by reason |
| --- | --- |
| train | banked_resit: 196, late_submission: 2984, missing_or_invalid_score: 26 |
| validation | banked_resit: 156, late_submission: 1016, missing_or_invalid_score: 11 |
| test | banked_resit: 268, late_submission: 1666, missing_or_invalid_score: 9 |

## Missing values and click integrity

Input missing-cell counts are restricted to FFF rows in the configured
presentations. Feature missing counts refer to retained scored rows.
Click zeros mean no matching event in a complete window; unknown or
incomplete windows remain blank rather than becoming zero.

| Split | Missing feature cells by feature |
| --- | --- |
| train | none |
| validation | none |
| test | none |

FFF source missing cells by presentation (table.column: count):

- 2013B.score: 14
- 2013J.score: 12
- 2014B.score: 11
- 2014J.score: 9
- FFF vle.csv.week_from: 1141
- FFF vle.csv.week_to: 1141

| Presentation | FFF studentVle rows | Unmatched VLE joins | Invalid click rows |
| --- | ---: | ---: | ---: |
| 2013B | 946765 | 0 | 0 |
| 2013J | 1172101 | 0 | 0 |
| 2014B | 685274 | 0 | 0 |
| 2014J | 1210359 | 0 | 0 |

Overall target score summary: n=48483, min=0, median=81, mean=78.991, max=100.

## Three real assignments for a minimum demo

These are scored, on-time TMA examples from the training presentation
2013B. Their ascending deadlines demonstrate early, middle, and later
prediction cutoffs; the label count is the retained target-row count.

| Assessment ID | Type | Deadline day | Retained target rows | Justification |
| ---: | --- | ---: | ---: | --- |
| 34860 | TMA | 19 | 1227 | TMA with an observed deadline and retained on-time scores; prediction inputs are cut off before that deadline. |
| 34861 | TMA | 47 | 1093 | TMA with an observed deadline and retained on-time scores; prediction inputs are cut off before that deadline. |
| 34862 | TMA | 89 | 778 | TMA with an observed deadline and retained on-time scores; prediction inputs are cut off before that deadline. |

## Output and reproducibility

Processed CSVs and `fff_preprocessing.json` are under `data/processed/`.
The JSON records source SHA-256 digests, stable feature names, the fixed
split, train-fitted type categories, and the training-derived lag count.
Identifier columns are retained for audit/join purposes and are not
listed as model input features. Input missing values are counted below
without demographic fields:

- `assessments.csv`: none
- `courses.csv`: none
- `studentAssessment.csv`: none
- `studentVle.csv`: none
- `vle.csv`: week_from: 1141, week_to: 1141
- Score-release dates are absent; the prior-score availability proxy above must be reviewed before a real deployment.
