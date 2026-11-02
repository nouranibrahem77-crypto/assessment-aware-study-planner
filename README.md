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

The development tools are pinned in `pyproject.toml` for repeatable installs.
To run the checks:

```bash
python -m pytest
ruff check .
```

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
