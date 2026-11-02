# OULAD exploration report

Generated from the validated local OULAD CSV files. `studentVle.csv` and
the other tables were summarized in bounded chunks; no student-level
demographic attributes are used as model features or in split selection.

## Dataset inventory

| File | Rows | Missing cells by column |
| --- | ---: | --- |
| `courses.csv` | 22 | none |
| `assessments.csv` | 206 | date: 11 |
| `studentAssessment.csv` | 173912 | score: 173 |
| `studentVle.csv` | 10655280 | none |
| `vle.csv` | 6364 | week_from: 5243, week_to: 5243 |

## Course/presentation inventory

| Course | Presentation | Length (days) | Assessments | With deadline | Suitable (deadline + submissions) | Deadlines (days from start) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| AAA | 2013J | 268 | 6 | 5 | 5 | 19, 54, 117, 166, 215 |
| AAA | 2014J | 269 | 6 | 5 | 5 | 19, 54, 117, 166, 215 |
| BBB | 2013B | 240 | 12 | 11 | 11 | 19, 47, 54, 89, 89, 124, 124, 159, 159, 187, 187 |
| BBB | 2013J | 268 | 12 | 11 | 11 | 19, 47, 54, 96, 96, 131, 131, 166, 166, 208, 208 |
| BBB | 2014B | 234 | 12 | 11 | 11 | 12, 40, 47, 82, 82, 117, 117, 152, 152, 194, 194 |
| BBB | 2014J | 262 | 6 | 5 | 5 | 19, 54, 110, 152, 201 |
| CCC | 2014B | 241 | 10 | 8 | 8 | 18, 32, 67, 102, 137, 151, 200, 207 |
| CCC | 2014J | 269 | 10 | 8 | 8 | 18, 32, 67, 109, 144, 158, 207, 214 |
| DDD | 2013B | 240 | 14 | 14 | 14 | 23, 25, 51, 53, 79, 81, 114, 116, 149, 151, 170, 200, 206, 240 |
| DDD | 2013J | 261 | 7 | 7 | 7 | 25, 53, 88, 123, 165, 207, 261 |
| DDD | 2014B | 241 | 7 | 7 | 7 | 25, 53, 74, 116, 158, 200, 241 |
| DDD | 2014J | 262 | 7 | 6 | 6 | 20, 41, 62, 111, 146, 195 |
| EEE | 2013J | 268 | 5 | 5 | 4 | 33, 68, 124, 159, 235 |
| EEE | 2014B | 241 | 5 | 5 | 4 | 33, 68, 117, 152, 228 |
| EEE | 2014J | 269 | 5 | 5 | 4 | 33, 68, 131, 166, 235 |
| FFF | 2013B | 240 | 13 | 13 | 12 | 19, 47, 89, 131, 166, 222, 222, 222, 222, 222, 222, 222, 222 |
| FFF | 2013J | 268 | 13 | 13 | 12 | 19, 47, 96, 131, 173, 236, 236, 236, 236, 236, 236, 236, 236 |
| FFF | 2014B | 241 | 13 | 13 | 12 | 24, 52, 87, 129, 171, 227, 227, 227, 227, 227, 227, 227, 227 |
| FFF | 2014J | 269 | 13 | 13 | 12 | 24, 52, 94, 136, 199, 241, 241, 241, 241, 241, 241, 241, 241 |
| GGG | 2013J | 261 | 10 | 10 | 9 | 61, 124, 173, 229, 229, 229, 229, 229, 229, 229 |
| GGG | 2014B | 241 | 10 | 10 | 9 | 61, 117, 166, 222, 222, 222, 222, 222, 222, 222 |
| GGG | 2014J | 269 | 10 | 10 | 9 | 61, 124, 173, 229, 229, 229, 229, 229, 229, 229 |

## Chronological train/test options

Each option trains on all listed earlier presentations and tests on all
listed later presentations. Presentation codes are ordered by year and
then B (February) before J (October). A split is marked eligible when
both partitions contain at least three assessments with deadlines and
at least one recorded student assessment submission.

| Course | Train presentations | Test presentations | Train suitable assessments | Test suitable assessments | Eligible |
| --- | --- | --- | ---: | ---: | --- |
| AAA | 2013J | 2014J | 5 | 5 | yes |
| BBB | 2013B | 2013J, 2014B, 2014J | 11 | 27 | yes |
| BBB | 2013B, 2013J | 2014B, 2014J | 22 | 16 | yes |
| BBB | 2013B, 2013J, 2014B | 2014J | 33 | 5 | yes |
| CCC | 2014B | 2014J | 8 | 8 | yes |
| DDD | 2013B | 2013J, 2014B, 2014J | 14 | 20 | yes |
| DDD | 2013B, 2013J | 2014B, 2014J | 21 | 13 | yes |
| DDD | 2013B, 2013J, 2014B | 2014J | 28 | 6 | yes |
| EEE | 2013J | 2014B, 2014J | 4 | 8 | yes |
| EEE | 2013J, 2014B | 2014J | 8 | 4 | yes |
| FFF | 2013B | 2013J, 2014B, 2014J | 12 | 36 | yes |
| FFF | 2013B, 2013J | 2014B, 2014J | 24 | 24 | yes |
| FFF | 2013B, 2013J, 2014B | 2014J | 36 | 12 | yes |
| GGG | 2013J | 2014B, 2014J | 9 | 18 | yes |
| GGG | 2013J, 2014B | 2014J | 18 | 9 | yes |

## Evidence-based course recommendation

Recommend **FFF** for a presentation-level chronological experiment: use presentations 2013B for training and 2013J, 2014B, 2014J for testing. The listed evidence shows at least three deadline-bearing assessments with recorded submissions in each partition. This split keeps later course presentations out of training.

## Interpretation and limitations

OULAD deadlines are day offsets from presentation start, not calendar dates. An assessment is counted as suitable here only when its deadline is present and `studentAssessment.csv` contains at least one matching assessment ID. A later-presentation split is safer than a random row split for chronology, but it does not establish generalization to other institutions or cohorts.
