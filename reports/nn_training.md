# Strict FFF neural regression experiment

## Run status

**Training and evaluation did not run.** The current Python 3.12 environment has
none of NumPy, pandas, scikit-learn, joblib, or PyTorch installed. The pinned
CPU-only ML extra was added, but its installation timed out while downloading
the 16.7 MB NumPy wheel (about 2.1 MB had transferred). No baseline or MLP
metrics, selected hyperparameters, training duration, checkpoints, or model
artifacts are available. No results are estimated or fabricated.

The source files, strict manifest, and processed CSV headers were inspected;
the data rows below are verified from the existing strict artifacts.

## Verified input data

| Split | Presentation(s) | Rows | Strict input features | Target |
| --- | --- | ---: | ---: | --- |
| Train | 2013B, 2013J | 25,229 | 19 | `target_score` |
| Validation | 2014B | 9,013 | 19 | `target_score` |
| Test | 2014J | 14,241 | 19 | `target_score` |

The target is the continuous assessment score in [0, 100]. The feature manifest
is strict mode; it has no previous-assessment-score features. Four identifiers
and `target_score` are excluded from model inputs. No demographics are used.

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

## Implemented experiment protocol

The added pipeline fits a scikit-learn preprocessor on training rows only:
numeric median imputation with missingness indicators, standard scaling, and
assessment-type one-hot encoding with `handle_unknown="ignore"`. The source
assessment-type indicators are decoded as a categorical input; identifiers and
the target are schema-rejected. Validation and test are transformed with the
fitted training preprocessor.

The baselines are a training-mean `DummyRegressor` and Ridge regression. Ridge
alpha candidates are 0.1, 1, 10, and 100, selected by validation MAE. The small
MLP search compares hidden sizes `(32,)` with no dropout and `(64, 32)` with 0.1
dropout. The intended run uses Adam, learning rate 0.001, weight decay 0.0001,
Huber loss, batch size 512, at most 40 epochs, early stopping after six stale
validation-loss epochs, and deterministic CPU execution. Seeds are 11, 23, and
42. MLP configuration selection uses mean validation MAE across seeds; test
data is used only after selection. Raw model outputs are clipped to [0, 100]
for all reported metrics and inference. MLP targets are divided by 100 during
optimization and predictions are rescaled to score units before clipping.

## Metrics and model selection

| Requested result | Status |
| --- | --- |
| Dummy and Ridge baseline metrics | Not run; ML dependencies unavailable. |
| MLP train and validation metrics | Not run. |
| Final test MAE, RMSE, and R² | Not run; no test-driven selection occurred. |
| Per-assessment-type test metrics and sample counts | Not run. |
| Selected Ridge alpha and MLP configuration | None selected. |
| Training curves, duration, and three-seed mean/std | Not available; training did not start. |
| Model and preprocessing artifacts | Not created. Intended output: `data/processed/nn_experiment/`. |

To run after installing the optional dependencies, use:

```bash
python -m pip install -e '.[ml]' --extra-index-url https://download.pytorch.org/whl/cpu
train-fff-nn
```

## Limitations

OULAD engagement is observational. This experiment cannot establish that clicks
cause marks. Activity dates have day-level resolution, prior-score features are
excluded in strict mode, and target scores are retrospective labels. Prediction
clipping is an explicit output policy; the MLP has an unconstrained linear
output and no monotonic constraint. No study sessions, causal conversion, or
planner counterfactuals are implemented.

## Checks

- `./.venv/bin/python -m pytest`: 23 passed; the ML-specific test module was
  skipped because NumPy is unavailable.
- `./.venv/bin/ruff check .`: all checks passed.
