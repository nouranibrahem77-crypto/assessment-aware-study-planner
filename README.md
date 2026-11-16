# Assessment-Aware Study Planner

A Python 3.11+ project for an Intelligent Systems university assignment. The
repository includes validated OULAD feature preparation and an initial strict
FFF regression training pipeline. Planner search and integration remain future
work.

## Requirements

- Ubuntu or Ubuntu on WSL
- Python 3.11 or newer
- `venv` support for the selected Python version

Check the Python version:

```bash
python3 --version
```

If Python 3.11 is not installed, install Python and its venv support using the
Ubuntu packages available for your release, then use `python3.11` in the
commands below. On WSL, run these commands in the Ubuntu terminal.

## Setup

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

For the strict FFF neural regression experiment, install the pinned CPU ML
dependencies from the PyTorch CPU wheel index:

```bash
python -m pip install -e '.[ml]' --extra-index-url https://download.pytorch.org/whl/cpu
train-fff-nn
```

The training command requires strict FFF files under `data/processed/` and
writes its model artifacts to `data/processed/nn_experiment/` plus
`reports/nn_training.md`. It uses the fixed chronological splits and does not
train on the proxy prior-score feature set.

For the separate Previous Marks sensitivity experiment, generate proxy features
and train Model B with:

```bash
make-fff-features --score-mode proxy --output-dir data/processed/proxy \
  --report reports/feature_engineering_proxy.md
train-previous-marks-nn
```

Model B artifacts are written under
`data/processed/nn_experiment_previous_marks/`. OULAD contains no grade-release
timestamps, so these features use an unverified submission-based availability
proxy; the experiment does not establish leakage-free or student-visible marks.

The development tools are pinned in `pyproject.toml` for repeatable installs.
To run the checks:

```bash
python -m pytest
ruff check .
```

## Integration demo

With the local OULAD processed features and fitted artifacts available, run one
explicitly TEST-configured Search–NN–CSP scenario from the repository root:

```bash
python run_demo.py NORMAL
python run_demo.py EDGE
python run_demo.py IMPOSSIBLE
python run_demo.py NN_ACCEPTS_CSP_INFEASIBLE
python run_demo.py VERIFICATION_REJECTED
```

The default uses the local Model A Strict artifact. To run against the separate
Model B proxy artifact and feature schema, append `--model proxy` to any command.
If the fitted artifact is in a different directory, pass it with
`--artifact-directory PATH`; the artifact must match the selected model schema.
All scenarios use labeled TEST targets, slot rules, and ordering. The
`NN_ACCEPTS_CSP_INFEASIBLE` case meets provisional targets but requests two
sessions from one eligible slot, so the real all-different CSP constraint
exhausts without a schedule. `VERIFICATION_REJECTED` separately injects one
invalid schedule at the verification boundary and lets the planner continue;
it is not a CSP-rejection demonstration. Both are TEST scenarios, not
production evidence.
The CSP-infeasible scenario uses a positive TEST target of 70 per assignment.
That shared threshold is based on current artifact-backed provisional results:
Model A Strict `(79.76, 75.78, 72.89)` and Model B Proxy
`(81.91, 81.42, 84.57)` for the configured TEST allocation. The runner checks
the current predictions against 70 and fails rather than claiming acceptance
if a future artifact does not meet it.
Each run prints the planner status, three assignments, allocations, timetable,
predicted marks when available, and the Search/CSP counters.

To generate per-student allocations, verified timetables, and predicted marks
for all 30 profiles under both predictors, run the synthetic experiment bundle:

```bash
python scripts/run_demo_experiments.py
```

The detailed plans are written to
`reports/experiment_results/demo/prediction_plans.csv` and
`reports/experiment_results/demo/prediction_plans.json`. These use synthetic
planning settings and are not production recommendations.

Results distinguish a verified target-reaching schedule, an exhaustively
unreachable target with the best verified final predictions, no feasible
schedule, and incomplete search. Incomplete results show a best verified plan
found so far, if one exists, and do not claim global optimality. Best-plan
comparison maximizes the sum of final assignment marks, then the canonical
per-assignment mark tuple, then minimizes total sessions, then uses the
lexicographically smallest whole-plan and schedule identities. Provisional NN
scores are never reported as achievable marks.
The `IMPOSSIBLE` TEST scenario has feasible schedules but uses TEST targets of
100, so it demonstrates exhaustive target-unreachable reporting. The separate
no-feasible-schedule case is covered by the integration regression tests.
Successful demos print the actual Search solution state/action path returned by
Search. Exhausted searches have no solution path; the displayed Search counters
are aggregate statistics, not an expansion-by-expansion trace. The integration
planner has no NN-disabled mode: every run requires an NN adapter for provisional
and final predictions. Model A Strict and Model B Proxy remain separate choices
via `--model strict` and `--model proxy`.

## Project layout

```text
src/study_planner/   Application package (data, nn, search, csp, integration, eval)
tests/               Automated tests
scripts/             Small project utilities
configs/             Future configuration files
docs/                Assignment and design notes
data/raw/            Original input data (not included)
data/processed/      Derived data (not included)
reports/figures/     Generated report figures
```

Keep source code independent of machine-specific paths. Use `pathlib` for file
paths and keep future data or model artifacts out of version control unless
they are small, intentional fixtures.

## OULAD data preparation

The data workflow downloads the ZIP linked by the [UCI OULAD dataset record](https://archive.ics.uci.edu/dataset/349/open%2Buniversity%2B), or accepts a local ZIP or extracted CSV directory. The official OU dataset page describes the tables and presentation codes; the UCI record links the downloadable archive and identifies its CC BY 4.0 license.

Activate the environment created above, then run one of:

```bash
python scripts/make_data.py --download
python scripts/make_data.py --archive /path/to/oulad.zip
python scripts/make_data.py --input-dir /path/to/extracted/oulad
```

Acquisition writes to `data/raw/` and refuses to replace existing CSVs or an
existing download unless `--overwrite` is supplied. The workflow validates the
five required schemas and writes `reports/oulad_exploration.md`. It processes
`studentVle.csv` in chunks. The report uses only course/presentation metadata,
assessment deadlines, and submission presence for split recommendations; it
does not use gender, age, region, disability, or deprivation attributes.

The CSVs are ignored by Git. The dataset creators request citation to Kuzilek,
Hlosta, and Zdrahal, “Open University Learning Analytics dataset,” *Scientific
Data* (2017), [doi:10.1038/sdata.2017.171](https://doi.org/10.1038/sdata.2017.171).
