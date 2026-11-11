# Previous Marks MLP comparison (submission-based proxy)

## Scientific boundary

OULAD does not contain grade-release timestamps. Model B uses a prior score
only when its recorded submission was on or before its own earlier scheduled
deadline. This submission-based rule does not prove the mark was released
or visible to the student by the target prediction cutoff. Model B is a
proxy sensitivity experiment, not a leakage-free or causal model.
Model A's strict schema and artifacts are unchanged.

## Protocol

Paired row counts: {"test": 14241, "train": 25229, "validation": 9013}.
Paired coverage of Model A rows: {"test": 1.0, "train": 1.0, "validation": 1.0}.
Both models use the same target, fixed presentation splits, and identical
student/presentation/assessment rows within each split. Preprocessors are fit
independently on their respective training rows only. Model B selects its MLP
configuration using validation MAE across the configured seeds; test metrics
were calculated after that selection. Model A predictions come from the
existing saved strict predictor.

Seeds: 42, 123, 2025.
Selected Model B config: `mlp_64_32_dropout`.

## Prior-mark coverage

| Split | Paired rows | With any prior mark | Without prior mark | Coverage | Missing-history rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| test | 14241 | 12469 | 1772 | 87.557% | 12.443% |
| train | 25229 | 22013 | 3216 | 87.253% | 12.747% |
| validation | 9013 | 7808 | 1205 | 86.630% | 13.370% |

## Paired validation metrics

| Model | Type | n | MAE | RMSE | R² |
| --- | --- | ---: | ---: | ---: | ---: |
| Model A — Strict MLP | Overall | 9013 | 10.2193 | 13.0985 | 0.1185 |
| Model A — Strict MLP | CMA | 5144 | 9.2790 | 11.6061 | 0.0707 |
| Model A — Strict MLP | TMA | 3869 | 11.4695 | 14.8522 | 0.1353 |
| Model B — Previous Marks MLP | Overall | 9013 | 8.8668 | 11.5931 | 0.3095 |
| Model B — Previous Marks MLP | CMA | 5144 | 8.3846 | 10.6450 | 0.2182 |
| Model B — Previous Marks MLP | TMA | 3869 | 9.5080 | 12.7448 | 0.3633 |

## Paired test metrics

| Model | Type | n | MAE | RMSE | R² |
| --- | --- | ---: | ---: | ---: | ---: |
| Model A — Strict MLP | Overall | 14241 | 10.0616 | 12.8343 | 0.0745 |
| Model A — Strict MLP | CMA | 8260 | 9.3023 | 11.7800 | 0.0571 |
| Model A — Strict MLP | TMA | 5981 | 11.1102 | 14.1619 | 0.0710 |
| Model B — Previous Marks MLP | Overall | 14241 | 8.6083 | 11.3287 | 0.2789 |
| Model B — Previous Marks MLP | CMA | 8260 | 8.2633 | 10.6300 | 0.2322 |
| Model B — Previous Marks MLP | TMA | 5981 | 9.0848 | 12.2280 | 0.3074 |

## Limitations and recommendation

These results quantify predictive performance under the stated proxy
feature construction. They do not establish that marks were visible to
students at the time or that prior marks cause improved prediction. Use
grade-release timestamps or another authoritative availability record
before describing this feature set as historically available or
deployment-ready.
Coverage reflects whether at least one proxy-eligible prior mark exists;
missing marks remain missing and use availability indicators.

Model B artifacts are stored separately under `data/processed/nn_experiment_previous_marks/`.
