# Strict FFF MLP extended training (validation only)

This experiment trains only the previously selected strict FFF MLP.
It accesses train and validation CSVs only; the test CSV is not loaded,
transformed, predicted on, or evaluated. The original 40-epoch run is
preserved in its separate artifact directory.

Fixed configuration: `mlp_64_32_dropout`, hidden sizes `[64, 32]`, dropout 0.1; Adam lr=0.001, weight decay=0.0001; huber loss; batch size 512; patience 6; maximum 60 epochs.
Seeds: `[42, 123, 2025]`. Duration for three training runs: 63.44s.

## Per-seed validation results

| Seed | Final epoch | Stop reason | Best epoch | Best Huber | Best checkpoint MAE | Best checkpoint RMSE | Final Huber | Final MAE |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 23 | early_stopping_patience | 17 | 0.00862656 | 10.23505 | 13.13471 | 0.00866968 | 10.37444 |
| 123 | 35 | early_stopping_patience | 29 | 0.00858004 | 10.21306 | 13.09965 | 0.00862797 | 10.33436 |
| 2025 | 60 | max_epochs_reached | 56 | 0.00860733 | 10.25618 | 13.12046 | 0.00861372 | 10.24077 |

Validation MAE/RMSE are computed from each seed's checkpoint restored
at minimum validation Huber loss. Aggregate values below are the
arithmetic mean and sample standard deviation across the three seeds.

| Metric | Mean | Std. dev. |
| --- | ---: | ---: |
| MAE | 10.23476 | 0.02156 |
| RMSE | 13.11827 | 0.01763 |
| R2 | 0.11588 | 0.00238 |

### Final-ten-epoch endpoint changes

Seed 42: over epochs 14–23, endpoint change in validation Huber loss was -0.00000744 and validation MAE was +0.02084. Negative values mean lower at the final epoch; this endpoint comparison does not imply a monotonic trend.
Seed 123: over epochs 26–35, endpoint change in validation Huber loss was -0.00002860 and validation MAE was -0.01582. Negative values mean lower at the final epoch; this endpoint comparison does not imply a monotonic trend.
Seed 2025: over epochs 51–60, endpoint change in validation Huber loss was -0.00000216 and validation MAE was -0.02634. Negative values mean lower at the final epoch; this endpoint comparison does not imply a monotonic trend.

Seed 2025's best Huber checkpoint was epoch 56; epochs 57–60 (4 epochs) did not improve it. The run reached the 60-epoch cap before the six-epoch patience rule could stop it. Although the epoch-51-to-60 endpoint change is slightly lower, validation loss fluctuated and epoch 60 remained above the restored minimum. The near-cap evidence is mixed, with no new best in the final four epochs, rather than sustained improvement through epoch 60.

## Comparison with the original 40-epoch run

| Seed | Old best epoch | Old MAE | Old RMSE | New best epoch | New MAE | New RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 17 | 10.23505 | 13.13471 | 17 | 10.23505 | 13.13471 |
| 123 | 29 | 10.21306 | 13.09965 | 29 | 10.21306 | 13.09965 |
| 2025 | 40 | 10.26974 | 13.13711 | 56 | 10.25618 | 13.12046 |

Shared-epoch curve comparison with the original:

Seed 42: 23 shared epochs; train loss, validation Huber loss, and validation MAE match exactly: True.
Seed 123: 35 shared epochs; train loss, validation Huber loss, and validation MAE match exactly: True.
Seed 2025: 40 shared epochs; train loss, validation Huber loss, and validation MAE match exactly: True.

Aggregate validation metrics (mean ± sample standard deviation over seeds):

| Run | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Original 40 epoch | 10.23928 ± 0.02858 | 13.12382 ± 0.02097 | 0.11513 ± 0.00283 |
| Extended 60 epoch cap | 10.23476 ± 0.02156 | 13.11827 ± 0.01763 | 0.11588 ± 0.00238 |

## Input identity and limitations

The current train and validation files match byte-for-byte between
`data/processed/` and `data/processed/strict/`; both were last
modified before the original experiment artifacts. The original run
did not persist input row-ID/target fingerprints, so historical
identity with its exact training rows cannot be proven from its saved
artifacts. Current row counts, strict schema, and row-ordered
identifier/target fingerprints are saved in `experiment_config.json`
for reproducibility from this run onward; row counts are in
`evaluation.json`.

Three seeds are descriptive; no statistical significance is claimed.
No test results are included or used for decisions. This experiment
does not change the selected model in the original artifact directory.

### Recorded row-ordered identifier/target fingerprints

| Split | Rows | SHA-256 (identifiers and target) |
| --- | ---: | --- |
| train | 25229 | `a7d6287fd4643465756faf4be9d6cc491da5fbc102abf1dfa775afb15f0c87d8` |
| validation | 9013 | `bb49323f5c0e465d420a8240caf8534ee544a03f84cbe140e2a4c1e6197762ee` |
