"""Acquire, validate, and explore the Open University Learning Analytics Dataset."""

from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
import stat
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# This is the Download link published on the UCI dataset record (DOI 10.24432/C5KK69).
OULAD_ARCHIVE_URL = (
    "https://archive.ics.uci.edu/static/public/349/"
    "open%2Buniversity%2Blearning%2Banalytics%2Bdataset.zip"
)
ARCHIVE_NAME = "oulad.zip"
DEFAULT_CHUNK_SIZE = 100_000
MISSING_VALUE_TOKENS = {"", "?", "na", "n/a", "null", "nan"}

REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "courses.csv": frozenset(
        {"code_module", "code_presentation", "module_presentation_length"}
    ),
    "assessments.csv": frozenset(
        {
            "code_module",
            "code_presentation",
            "id_assessment",
            "assessment_type",
            "date",
            "weight",
        }
    ),
    "studentAssessment.csv": frozenset(
        {"id_assessment", "id_student", "date_submitted", "is_banked", "score"}
    ),
    "studentVle.csv": frozenset(
        {
            "code_module",
            "code_presentation",
            "id_student",
            "id_site",
            "date",
            "sum_click",
        }
    ),
    "vle.csv": frozenset(
        {
            "id_site",
            "code_module",
            "code_presentation",
            "activity_type",
            "week_from",
            "week_to",
        }
    ),
}


class DataValidationError(ValueError):
    """Raised when an OULAD input is missing or does not match its schema."""


@dataclass(frozen=True)
class CsvSummary:
    """Row and blank-cell counts from a CSV processed in bounded chunks."""

    row_count: int
    missing_counts: dict[str, int]


@dataclass(frozen=True)
class Assessment:
    """Assessment facts used to evaluate presentation splits."""

    module: str
    presentation: str
    assessment_id: str
    assessment_type: str
    deadline: int | None
    submission_count: int

    @property
    def suitable(self) -> bool:
        """Whether this assessment has a published deadline and submissions."""
        return self.deadline is not None and self.submission_count > 0


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _is_blank(value: str | None) -> bool:
    return value is None or value.strip().casefold() in MISSING_VALUE_TOKENS


def _headers(path: Path) -> list[str]:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.reader(file)
            headers = next(reader, None)
    except (OSError, UnicodeError, csv.Error) as error:
        message = f"Cannot read CSV header in {path}: {error}"
        raise DataValidationError(message) from error

    if not headers or any(not header.strip() for header in headers):
        raise DataValidationError(f"{path} has an empty or missing header row")
    if len(headers) != len(set(headers)):
        raise DataValidationError(f"{path} has duplicate column names")
    return headers


def locate_dataset_files(input_dir: Path) -> dict[str, Path]:
    """Find the required CSV files in an extracted dataset directory."""
    if not input_dir.is_dir():
        raise DataValidationError(f"Dataset directory does not exist: {input_dir}")

    files: dict[str, Path] = {}
    missing: list[str] = []
    duplicates: list[str] = []
    for filename in REQUIRED_COLUMNS:
        matches = sorted(path for path in input_dir.rglob(filename) if path.is_file())
        if not matches:
            missing.append(filename)
        elif len(matches) > 1:
            duplicates.append(filename)
        else:
            files[filename] = matches[0]

    if missing or duplicates:
        issues: list[str] = []
        if missing:
            issues.append("missing files: " + ", ".join(missing))
        if duplicates:
            issues.append("ambiguous duplicate files: " + ", ".join(duplicates))
        issue_text = "; ".join(issues)
        raise DataValidationError(f"Invalid OULAD directory {input_dir}: {issue_text}")
    validate_dataset_files(files)
    return files


def validate_dataset_files(files: dict[str, Path]) -> None:
    """Check required filenames, readable CSV headers, and required columns."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(files))
    if missing:
        raise DataValidationError("Missing required files: " + ", ".join(missing))

    for filename, required_columns in REQUIRED_COLUMNS.items():
        path = files[filename]
        if not path.is_file():
            raise DataValidationError(f"Required file is missing: {path}")
        actual_columns = set(_headers(path))
        absent_columns = sorted(required_columns - actual_columns)
        if absent_columns:
            raise DataValidationError(
                f"{filename} is missing required columns: {', '.join(absent_columns)}"
            )


def iter_csv_chunks(
    path: Path, chunk_size: int
) -> Iterator[list[dict[str, str | None]]]:
    """Yield CSV rows in fixed-size batches without loading the file in memory."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            chunk: list[dict[str, str | None]] = []
            for row in reader:
                chunk.append(row)
                if len(chunk) == chunk_size:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk
    except (OSError, UnicodeError, csv.Error) as error:
        raise DataValidationError(f"Cannot read CSV {path}: {error}") from error


def summarize_csv(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> CsvSummary:
    """Count rows and blank cells by column, processing bounded-size chunks."""
    headers = _headers(path)
    missing_counts: Counter[str] = Counter({header: 0 for header in headers})
    row_count = 0
    for chunk in iter_csv_chunks(path, chunk_size):
        row_count += len(chunk)
        for row in chunk:
            for header in headers:
                if _is_blank(row.get(header)):
                    missing_counts[header] += 1
    return CsvSummary(row_count, dict(missing_counts))


def _read_rows(path: Path) -> list[dict[str, str | None]]:
    """Read a small metadata CSV into memory after schema validation."""
    return [row for chunk in iter_csv_chunks(path, DEFAULT_CHUNK_SIZE) for row in chunk]


def _presentation_key(value: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"(\d{4})([BJ])", value)
    if not match:
        return (9999, 2, value)
    year, season = match.groups()
    return (int(year), 0 if season == "B" else 1, value)


def _date(value: str | None) -> int | None:
    if _is_blank(value):
        return None
    try:
        return int(value)  # OULAD deadlines are days from the presentation start.
    except ValueError:
        return None


def _assessment_inventory(
    files: dict[str, Path], chunk_size: int
) -> tuple[list[Assessment], list[dict[str, str | None]]]:
    submission_counts: Counter[str] = Counter()
    for chunk in iter_csv_chunks(files["studentAssessment.csv"], chunk_size):
        for row in chunk:
            assessment_id = row.get("id_assessment")
            if not _is_blank(assessment_id):
                submission_counts[assessment_id or ""] += 1

    assessments: list[Assessment] = []
    for row in _read_rows(files["assessments.csv"]):
        assessment_id = row.get("id_assessment") or ""
        assessments.append(
            Assessment(
                module=row.get("code_module") or "",
                presentation=row.get("code_presentation") or "",
                assessment_id=assessment_id,
                assessment_type=row.get("assessment_type") or "",
                deadline=_date(row.get("date")),
                submission_count=submission_counts[assessment_id],
            )
        )
    return assessments, _read_rows(files["courses.csv"])


def _split_options(assessments: list[Assessment]) -> list[dict[str, object]]:
    by_module: dict[str, dict[str, list[Assessment]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for assessment in assessments:
        by_module[assessment.module][assessment.presentation].append(assessment)

    options: list[dict[str, object]] = []
    for module, presentations in sorted(by_module.items()):
        ordered = sorted(presentations, key=_presentation_key)
        for split_at in range(1, len(ordered)):
            train_presentations = ordered[:split_at]
            test_presentations = ordered[split_at:]
            train_assessments = [
                assessment
                for presentation in train_presentations
                for assessment in presentations[presentation]
                if assessment.suitable
            ]
            test_assessments = [
                assessment
                for presentation in test_presentations
                for assessment in presentations[presentation]
                if assessment.suitable
            ]
            options.append(
                {
                    "module": module,
                    "train_presentations": train_presentations,
                    "test_presentations": test_presentations,
                    "train_assessment_count": len(train_assessments),
                    "test_assessment_count": len(test_assessments),
                    "eligible": (
                        len(train_assessments) >= 3 and len(test_assessments) >= 3
                    ),
                }
            )
    return options


def _report_text(
    files: dict[str, Path], chunk_size: int
) -> tuple[str, dict[str, object]]:
    validate_dataset_files(files)
    summaries = {
        filename: summarize_csv(path, chunk_size) for filename, path in files.items()
    }
    assessments, course_rows = _assessment_inventory(files, chunk_size)
    options = _split_options(assessments)
    course_presentations: dict[str, set[str]] = defaultdict(set)
    presentation_lengths: dict[tuple[str, str], str] = {}
    for row in course_rows:
        module = row.get("code_module") or ""
        presentation = row.get("code_presentation") or ""
        course_presentations[module].add(presentation)
        presentation_lengths[(module, presentation)] = (
            row.get("module_presentation_length") or "unknown"
        )

    assessment_groups: dict[tuple[str, str], list[Assessment]] = defaultdict(list)
    for assessment in assessments:
        key = (assessment.module, assessment.presentation)
        assessment_groups[key].append(assessment)

    lines = [
        "# OULAD exploration report",
        "",
        "Generated from the validated local OULAD CSV files. `studentVle.csv` and",
        "the other tables were summarized in bounded chunks; no student-level",
        "demographic attributes are used as model features or in split selection.",
        "",
        "## Dataset inventory",
        "",
        "| File | Rows | Missing cells by column |",
        "| --- | ---: | --- |",
    ]
    for filename in REQUIRED_COLUMNS:
        summary = summaries[filename]
        missing_text = (
            ", ".join(
                f"{column}: {count}"
                for column, count in summary.missing_counts.items()
                if count
            )
            or "none"
        )
        lines.append(f"| `{filename}` | {summary.row_count} | {missing_text} |")

    lines.extend(
        [
            "",
            "## Course/presentation inventory",
            "",
            "| Course | Presentation | Length (days) | Assessments | With deadline | "
            "Suitable (deadline + submissions) | Deadlines (days from start) |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for module in sorted(course_presentations):
        for presentation in sorted(course_presentations[module], key=_presentation_key):
            group = assessment_groups[(module, presentation)]
            deadlines = sorted(
                assessment.deadline
                for assessment in group
                if assessment.deadline is not None
            )
            suitable_count = sum(assessment.suitable for assessment in group)
            deadline_text = ", ".join(map(str, deadlines)) if deadlines else "none"
            length = presentation_lengths[(module, presentation)]
            lines.append(
                f"| {module} | {presentation} | {length} | {len(group)} | "
                f"{len(deadlines)} | {suitable_count} | {deadline_text} |"
            )

    lines.extend(
        [
            "",
            "## Chronological train/test options",
            "",
            "Each option trains on all listed earlier presentations and tests on all",
            "listed later presentations. Presentation codes are ordered by year and",
            "then B (February) before J (October). A split is marked eligible when",
            "both partitions contain at least three assessments with deadlines and",
            "at least one recorded student assessment submission.",
            "",
            "| Course | Train presentations | Test presentations | "
            "Train suitable assessments | Test suitable assessments | Eligible |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for option in options:
        train = ", ".join(option["train_presentations"])  # type: ignore[arg-type]
        test = ", ".join(option["test_presentations"])  # type: ignore[arg-type]
        lines.append(
            f"| {option['module']} | {train} | {test} | "
            f"{option['train_assessment_count']} | {option['test_assessment_count']} | "
            f"{'yes' if option['eligible'] else 'no'} |"
        )

    eligible_options = [option for option in options if option["eligible"]]
    if eligible_options:
        selected = sorted(
            eligible_options,
            key=lambda option: (
                -int(option["test_assessment_count"]),
                -int(option["train_assessment_count"]),
                str(option["module"]),
                option["test_presentations"],
            ),
        )[0]
        lines.extend(
            [
                "",
                "## Evidence-based course recommendation",
                "",
                f"Recommend **{selected['module']}** for a presentation-level "
                "chronological experiment: use presentations "
                f"{', '.join(selected['train_presentations'])} for training and "
                f"{', '.join(selected['test_presentations'])} for testing. The "
                "listed evidence shows at least three deadline-bearing assessments "
                "with recorded submissions in each partition. This split keeps "
                "later course presentations out of training.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Evidence-based course recommendation",
                "",
                "No course/presentation split meets the stated minimum of three "
                "deadline-bearing assessments with recorded submissions in both "
                "partitions. No course recommendation is made from this dataset.",
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            "OULAD deadlines are day offsets from presentation start, not calendar "
            "dates. An assessment is counted as suitable here only when its deadline "
            "is present and `studentAssessment.csv` contains at least one matching "
            "assessment ID. A later-presentation split is safer than a random row "
            "split for chronology, but it does not establish generalization to "
            "other institutions or cohorts.",
            "",
        ]
    )
    details: dict[str, object] = {
        "summaries": summaries,
        "assessments": assessments,
        "course_rows": course_rows,
        "split_options": options,
    }
    return "\n".join(lines), details


def _download_archive(destination: Path, overwrite: bool) -> Path:
    """Stream the documented UCI archive into the raw-data directory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing archive {destination}; pass --overwrite."
        )
    temporary_path: Path | None = None
    try:
        with urllib.request.urlopen(OULAD_ARCHIVE_URL, timeout=60) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, prefix=".oulad-", delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                shutil.copyfileobj(response, temporary_file)
        if not zipfile.is_zipfile(temporary_path):
            raise DataValidationError("Downloaded UCI file is not a valid ZIP archive")
        temporary_path.replace(destination)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download OULAD from the verified UCI source: {error}. "
            "Use --archive with a local ZIP or --input-dir with extracted CSV files."
        ) from error
    except DataValidationError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination


def extract_archive(
    archive_path: Path, destination: Path, overwrite: bool = False
) -> None:
    """Safely extract the five required CSV files, flattening archive folders."""
    if not archive_path.is_file() or not zipfile.is_zipfile(archive_path):
        raise DataValidationError(f"Not a readable ZIP archive: {archive_path}")
    destination.mkdir(parents=True, exist_ok=True)
    expected = set(REQUIRED_COLUMNS)
    selected: dict[str, zipfile.ZipInfo] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise DataValidationError(
                    f"Unsafe path in ZIP archive: {member.filename}"
                )
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise DataValidationError(
                    f"Symbolic links are not allowed in ZIP archives: {member.filename}"
                )
            filename = member_path.name
            if not member.is_dir() and filename in expected:
                if filename in selected:
                    raise DataValidationError(
                        f"Archive contains duplicate required file {filename}"
                    )
                selected[filename] = member

        absent = sorted(expected - set(selected))
        if absent:
            raise DataValidationError(
                "Archive is missing required CSV files: " + ", ".join(absent)
            )
        existing = sorted(
            filename for filename in expected if (destination / filename).exists()
        )
        if existing and not overwrite:
            raise FileExistsError(
                "Refusing to overwrite existing data files: "
                + ", ".join(existing)
                + "; pass --overwrite."
            )

        for filename, member in selected.items():
            target = destination / filename
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            LOGGER.info("Extracted %s", target)


def write_exploration_report(
    files: dict[str, Path], report_path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> dict[str, object]:
    """Validate input files and write the generated Markdown report."""
    report, details = _report_text(files, chunk_size)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return details


def _build_parser() -> argparse.ArgumentParser:
    root = _project_root()
    parser = argparse.ArgumentParser(
        description="Acquire, validate, and explore OULAD CSV files."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--download",
        action="store_true",
        help="Download the UCI dataset archive into data/raw and extract it.",
    )
    source.add_argument(
        "--archive",
        type=Path,
        help="Use an existing local OULAD ZIP archive and extract it into data/raw.",
    )
    source.add_argument(
        "--input-dir",
        type=Path,
        help=(
            "Use existing extracted CSV files in this directory (searched recursively)."
        ),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=root / "data" / "raw", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "oulad_exploration.md",
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Rows per CSV processing batch (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing archive/CSV files during acquisition.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the OULAD acquisition and exploration command-line workflow."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.chunk_size < 1:
        parser.error("--chunk-size must be at least 1")

    try:
        if args.download:
            archive = _download_archive(args.data_dir / ARCHIVE_NAME, args.overwrite)
            extract_archive(archive, args.data_dir, args.overwrite)
            input_dir = args.data_dir
        elif args.archive:
            extract_archive(args.archive, args.data_dir, args.overwrite)
            input_dir = args.data_dir
        else:
            input_dir = args.input_dir or args.data_dir

        files = locate_dataset_files(input_dir)
        details = write_exploration_report(files, args.report, args.chunk_size)
    except (DataValidationError, FileExistsError, RuntimeError, OSError) as error:
        LOGGER.error("%s", error)
        return 1

    summaries = details["summaries"]
    assert isinstance(summaries, dict)
    LOGGER.info("Validated %d required OULAD files", len(files))
    LOGGER.info("Exploration report written to %s", args.report)
    for filename, summary in summaries.items():
        LOGGER.info("%s: %s rows", filename, summary.row_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
