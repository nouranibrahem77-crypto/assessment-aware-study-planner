# Strict FFF neural regression experiment

## Data and protocol

Strict OULAD FFF features only. The prediction cutoff is the start of
the assessment deadline day; the strict feature manifest excludes all
prior assessment scores. Identifiers are audit-only. No demographic
variables or future assessment outcomes enter the model.

Train, validation, and test rows: 25,229, 9,013, 14,241.
Model inputs: 19 strict source columns; 18 transformed columns.
The assessment type is reconstructed from the strict one-hot columns and
encoded with a training-fitted OneHotEncoder (`handle_unknown='ignore'`).
Numeric columns use training-median imputation (with missingness flags)
and training-fitted standard scaling. The preprocessor is fit on train
only; the saved preprocessing object is reused for validation, test, and
inference. Predictions from all models are clipped to [0, 100] before
metrics and saved prediction outputs.

### Input feature list

- `assessment_type_cma`
- `assessment_type_exam`
- `assessment_type_tma`
- `assessment_weight`
- `assessment_weight_available`
- `assessment_deadline_day`
- `weeks_since_course_start`
- `click_content_navigation_week_1`
- `click_content_navigation_week_2`
- `click_content_navigation_week_3`
- `click_content_navigation_week_4`
- `click_assessment_activity_week_1`
- `click_assessment_activity_week_2`
- `click_assessment_activity_week_3`
- `click_assessment_activity_week_4`
- `click_collaboration_week_1`
- `click_collaboration_week_2`
- `click_collaboration_week_3`
- `click_collaboration_week_4`

## Baselines and selected model

Ridge alpha was selected from `[0.1, 1.0, 10.0, 100.0]` using validation MAE. The MLP search had two documented candidates:
`(32,)` with no dropout and `(64, 32)` with 0.1 dropout. Both used
Adam, learning rate 0.001, weight decay 0.0001, Huber loss, batches of
512, and early stopping on
validation Huber loss (patience 6, maximum 40 epochs). Three seeds
were run: `[42]`. Selection minimized mean validation
MAE across seeds; test data did not participate in selection.
MLP targets are divided by 100 for optimization and predictions are
rescaled to score units before clipping and metric calculation.

Selected MLP: `mlp_32`, hidden sizes `[32]`, dropout 0.0, Huber loss. Ridge alpha: 100.0.

Selected MLP train metrics (mean ± standard deviation across seeds):

| MAE | RMSE | R² |
| ---: | ---: | ---: |
| 10.288 ± 0.000 | 13.249 ± 0.000 | 0.0957 ± 0.0000 |

MLP candidate validation results (selection used mean MAE):

| Candidate | Mean validation MAE | Std. dev. |
| --- | ---: | ---: |
| mlp_32 | 10.450 | 0.000 |

| Model | Validation MAE | Validation RMSE | Validation R² |
| --- | ---: | ---: | ---: |
| Dummy mean | 10.648 ± 0.000 | 13.977 ± 0.000 | -0.0037 ± 0.0000 |
| Ridge | 10.424 ± 0.000 | 13.472 ± 0.000 | 0.0676 ± 0.0000 |
| MLP (selected config, seed mean) | 10.450 ± 0.000 | 13.324 ± 0.000 | 0.0879 ± 0.0000 |

## Final test results

Reported only after configuration and Ridge alpha selection on
validation. MLP seed mean ± standard deviation is across the three
selected-configuration seed models. The inference artifact averages
those three seed predictions and is reported separately.

| Model | Test MAE | Test RMSE | Test R² | n |
| --- | ---: | ---: | ---: | ---: |
| Dummy mean | 10.489 | 13.405 | -0.0097 | 14241 |
| Ridge | 10.240 | 13.070 | 0.0401 | 14241 |
| MLP seed mean | 10.382 ± 0.000 | 13.450 ± 0.000 | -0.0165 ± 0.0000 | 14241 |
| MLP seed ensemble | 10.382 | 13.450 | -0.0165 | 14241 |

### Test metrics by assessment type and model

| Model | Type | n | MAE | RMSE | R² |
| --- | --- | ---: | ---: | ---: | ---: |
| Dummy mean | CMA | 8260 | 10.127 | 12.387 | -0.0425 |
| Dummy mean | TMA | 5981 | 10.988 | 14.697 | -0.0005 |
| Ridge | CMA | 8260 | 9.475 | 11.935 | 0.0322 |
| Ridge | TMA | 5981 | 11.297 | 14.494 | 0.0270 |
| MLP seed ensemble | CMA | 8260 | 9.578 | 12.510 | -0.0633 |
| MLP seed ensemble | TMA | 5981 | 11.493 | 14.650 | 0.0059 |

## Training curves and run time

Checkpoint restoration minimizes validation Huber loss. Full epoch
curves for all candidates and seeds are saved in `training_curves.csv`.
Selected configuration checkpoints:
Total experiment runtime including preprocessing and evaluation: 6.94 seconds; MLP search: 6.59 seconds.

| Seed | Epochs run | Best epoch | Best validation Huber loss | Duration (s) |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 20 | 20 | 0.00888 | 4.15 |

Representative selected-model training curve (seed 42):

| Epoch | Train Huber loss | Validation Huber loss | Validation MAE |
| ---: | ---: | ---: | ---: |
| 1 | 0.24012 | 0.06970 | 33.020 |
| 2 | 0.03854 | 0.02053 | 15.302 |
| 3 | 0.01996 | 0.01593 | 13.499 |
| 5 | 0.01377 | 0.01238 | 12.020 |
| 20 | 0.00883 | 0.00888 | 10.450 |

## Limitations

OULAD engagement is observational: these metrics do not show that
clicks cause marks. The dataset has day-level activity timestamps, no
grade-release times, and target scores are retrospective labels.
Prediction clipping is a reporting/deployment policy, not a monotonic
constraint. No study sessions or causal effects are inferred.

## Artifacts

All experiment files are under `/home/nouran/projects/assessment-aware-study-planner/data/processed/nn_demo`: fitted preprocessor, Dummy/Ridge estimators, selected-seed MLP state dictionaries, strict
feature schema, configuration and versions, evaluation JSON, test
predictions, and full training curves.
