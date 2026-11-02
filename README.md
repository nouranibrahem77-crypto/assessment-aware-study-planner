# Assessment-Aware Study Planner

A Python 3.11+ project for an Intelligent Systems university assignment. The
repository starts with a modular package layout and development tooling; the
planning algorithms, datasets, and model training are intentionally not
implemented yet.

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
