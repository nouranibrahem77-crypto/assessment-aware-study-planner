# Strict FFF MLP training stability audit

## Scope and evidence

This audit uses only `training_curves.csv`, `evaluation.json`,
`experiment_config.json`, `nn_training.md`, and the training/inference source
code. It does not read `test_predictions.csv`, compute test metrics, or rerun
training. The selected configuration is `mlp_64_32_dropout`: hidden sizes
`[64, 32]`, dropout 0.1, Adam learning rate 0.001, weight decay 0.0001, and
Huber loss. The configured cap is 40 epochs with early stopping after six
consecutive epochs without a validation Huber improvement.

## Seed 2025 at the epoch cap

The selected-configuration curve contains all 40 epochs for seed 2025. Its
lowest validation Huber loss in the saved curve is epoch 40: `0.00862918`.
The stopping code saves a checkpoint whenever validation Huber loss is a new
minimum, and therefore resets the stale-epoch counter at epoch 40. The run
ended because it reached `max_epochs=40`, not because six non-improving epochs
triggered early stopping. The source and curve support this explanation; they
do not establish what would happen after epoch 40.

| Epoch | Train Huber loss | Validation Huber loss | Validation MAE |
| ---: | ---: | ---: | ---: |
| 31 | 0.01115554 | 0.00883599 | 10.53655 |
| 32 | 0.01105101 | 0.00874216 | 10.39880 |
| 33 | 0.01103943 | 0.00870441 | 10.31961 |
| 34 | 0.01098158 | 0.00868598 | 10.32638 |
| 35 | 0.01090822 | 0.00870820 | 10.36750 |
| 36 | 0.01074207 | 0.00870241 | 10.37845 |
| 37 | 0.01080816 | 0.00880626 | 10.52365 |
| 38 | 0.01073812 | 0.00874669 | 10.42527 |
| 39 | 0.01059759 | 0.00870565 | 10.38885 |
| 40 | 0.01043539 | 0.00862918 | 10.26974 |

Train Huber loss fell from `0.01115554` to `0.01043539` across these epochs.
Validation Huber loss fluctuated, with a substantial temporary rise at epoch
37, but its epoch-40 value is the best observed and is `0.00020681` below
epoch 31. Validation MAE also fluctuated; it improved from `10.53655` at epoch
31 to `10.26974` at epoch 40, with the final epoch improving on epoch 39 by
`0.11910`. These observations indicate continued, noisy validation improvement
at the boundary, rather than a settled plateau or consistent deterioration.

## Checkpoint restoration and seed comparison

For the selected configuration, I independently compared each saved
validation-loss minimum in `training_curves.csv` with the recorded epoch and
best loss in `evaluation.json`:

| Seed | Recorded epochs | Curve minimum epoch | Recorded best epoch | Best validation Huber loss | Status |
| ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 23 | 17 | 17 | 0.00862656 | Match |
| 123 | 35 | 29 | 29 | 0.00858004 | Match |
| 2025 | 40 | 40 | 40 | 0.00862918 | Match |

The training source clones model weights whenever validation Huber loss
strictly improves, then loads the saved best state before producing each run's
validation metrics and returning the checkpoint. The persisted best epochs and
losses match the minima in the curves for all three selected-configuration
seeds. This verifies the restoration decision recorded by the implementation;
the audit did not independently load weights or compare checkpoint outputs.

Seed 42 stopped after six consecutive non-improving validation-loss epochs
following epoch 17. Seed 123 similarly stopped after epoch 35 with its best at
29. Seed 2025 had its best at the configured cap. The selected-configuration
best validation MAEs are `10.23505`, `10.21306`, and `10.26974` for seeds 42,
123, and 2025, respectively. Their observed range is about `0.05669` score
points. Best validation Huber losses are close (`0.00858004` to `0.00862918`),
though the seed-specific epoch of the minimum varies. This is a small observed
three-seed comparison, not a statistical significance claim.

## Ensemble membership

`experiment_config.json` lists exactly three selected model files:
`mlp_seed_42.pt`, `mlp_seed_123.pt`, and `mlp_seed_2025.pt`. These match the
configured seed list and the selected-run records. The training code creates
the ensemble from `selected_runs`, which belong to the selected configuration;
the inference loader iterates every file in `model_files`, predicts with each
loaded model, and takes the arithmetic mean. Thus the configured ensemble
contains exactly the three selected-configuration seed checkpoints. This is a
source/configuration audit; it does not evaluate their predictions.

## Reproducibility and stopping considerations

The run records Python, NumPy, pandas, scikit-learn, joblib, and PyTorch
versions. Training sets Python, NumPy, and PyTorch seeds; requests deterministic
PyTorch algorithms; uses one CPU thread; and seeds the shuffled DataLoader's
generator. Those are concrete reproducibility controls. They do not guarantee
bitwise identity across different library builds, CPU architectures, or
runtime environments. The persisted configuration does not contain a source
revision or a cryptographic hash of the processed data, so artifact provenance
could be made stronger in a later workflow change.

Early stopping monitors validation Huber loss, while candidate selection uses
mean validation MAE across seeds. This is an intentional two-metric protocol,
but it means the restored checkpoint is not necessarily the epoch with minimum
MAE. For seed 2025, epoch 40 is both the lowest saved Huber loss and lowest
validation MAE in that selected-configuration curve; no claim is made about
other unsaved epochs. A strict-improvement comparison also makes the saved
minimum dependent on the precision of the recorded loss values.

## Recommendation

Increasing the epoch cap is justified for a **validation-only follow-up**:
seed 2025 was still improving in both validation Huber loss and validation MAE
at epoch 40, and training loss was still falling. The evidence is not enough to
predict that later epochs will improve; the validation curves are noisy, and
seeds 42 and 123 had already triggered patience-based stopping. If extended,
retain the existing architecture, hyperparameters, splits, and selection
rules, raise only the maximum epoch budget modestly (for example, to 60), keep
patience unchanged, and review every seed's validation curve and restored
checkpoint before making a decision. Do not use test results to choose the
epoch budget. No models were retrained for this audit.

## Verified findings and assumptions

- **Verified:** the saved selected-configuration curves, run summaries, and
  configured model file list agree on three seeds and their best epochs.
- **Verified:** seed 2025's best recorded validation Huber loss occurs at the
  40-epoch boundary; its last epoch improves on epoch 39.
- **Verified:** source code saves a checkpoint on validation-Huber improvement
  and restores that state before returning each run.
- **Verified:** inference averages predictions from every checkpoint listed in
  the configuration, which lists the three selected-configuration seeds.
- **Assumption/limitation:** the training curves and configuration are trusted
  as outputs of the recorded run; this audit did not independently hash the
  source data or load model weights.
- **Not assessed:** test predictions, test metrics, statistical significance,
  and behavior beyond epoch 40.
