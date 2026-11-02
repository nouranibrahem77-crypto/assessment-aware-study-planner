# Repository guidance

- Target Python 3.11 and newer. Add type hints to public functions and classes.
- Keep modules focused and use clear names; avoid unnecessary dependencies and
  abstractions.
- Use `pathlib` for filesystem paths and avoid machine-specific absolute paths.
- Keep project code under `src/study_planner/` and tests under `tests/`.
- Preserve reproducibility: pin development tools and make randomness explicit
  when algorithms are added.
- Do not commit downloaded datasets, generated model artifacts, or report output
  unless they are deliberately small fixtures or reviewed deliverables.
- Run `python -m pytest` and `ruff check .` after relevant changes.
