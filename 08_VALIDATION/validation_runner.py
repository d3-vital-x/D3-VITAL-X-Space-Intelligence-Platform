
"""
D³ VITAL-X Space Intelligence Platform
Module 22: validation_runner.py

Purpose
-------
Run reproducible, diagnostic-only software and dataset checks.

Safety and scientific boundaries
--------------------------------
- Does not execute project modules being inspected.
- Does not make medical diagnoses.
- Does not claim scientific validity from software checks.
- Does not modify source datasets.
- Reports missing dependencies and optional modules explicitly.
- Produces JSON and CSV validation reports.

Standard-library only for the validation runner itself.
Optional packages are checked but are not required to import this module.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import platform
import sys
import time
import traceback

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional


MODULE_NAME = "validation_runner"
MODULE_NUMBER = 22
MODULE_VERSION = "1.0.0"

PROJECT_NAME = "D3-VITAL-X-Space-Intelligence-Platform"
REPORT_SCHEMA_VERSION = "1.0"

DEFAULT_REQUIRED_FILES = (
    "02_DASHBOARD/app.py",
    "02_DASHBOARD/pages/space_page.py",
    "02_DASHBOARD/pages/biomedical_page.py",
    "02_DASHBOARD/pages/live_page.py",
)

OPTIONAL_FILES = (
    "02_DASHBOARD/pages/result_view.py",
    "02_DASHBOARD/pages/export.py",
)

OPTIONAL_PACKAGES = (
    "numpy",
    "pandas",
    "streamlit",
    "PIL",
    "pydicom",
)

SUPPORTED_DATA_EXTENSIONS = {
    ".csv",
    ".json",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".dcm",
}


@dataclass
class ValidationResult:
    """One structured validation finding."""

    check_id: str
    category: str
    status: str
    message: str
    details: Optional[dict[str, Any]] = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ValidationRunner:
    """
    Diagnostic validation orchestrator.

    Parameters
    ----------
    project_root:
        Root of the project repository.
    output_dir:
        Directory for generated validation reports.
    dataset_path:
        Optional file or directory to inspect.
    strict:
        If True, missing required files contribute to a failing exit status.
    """

    def __init__(
        self,
        project_root: str | Path,
        output_dir: str | Path | None = None,
        dataset_path: str | Path | None = None,
        strict: bool = False,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.output_dir = (
            Path(output_dir).expanduser().resolve()
            if output_dir
            else self.project_root / "validation_reports"
        )
        self.dataset_path = (
            Path(dataset_path).expanduser().resolve()
            if dataset_path
            else None
        )
        self.strict = bool(strict)
        self.results: list[ValidationResult] = []
        self.started_at: str | None = None
        self.finished_at: str | None = None

    @staticmethod
    def utc_now() -> str:
        """Return a timezone-aware UTC timestamp."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def sha256_file(
        path: Path,
        chunk_size: int = 1024 * 1024,
    ) -> str:
        """Calculate a file hash without loading the whole file into memory."""
        digest = hashlib.sha256()

        with path.open("rb") as file_handle:
            while True:
                chunk = file_handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)

        return digest.hexdigest()

    def _record(
        self,
        check_id: str,
        category: str,
        status: str,
        message: str,
        details: Optional[dict[str, Any]] = None,
        duration_ms: float = 0.0,
    ) -> ValidationResult:
        result = ValidationResult(
            check_id=check_id,
            category=category,
            status=status,
            message=message,
            details=details or {},
            duration_ms=round(duration_ms, 3),
        )
        self.results.append(result)
        return result

    def _run_check(
        self,
        check_id: str,
        category: str,
        check_function: Callable[[], ValidationResult],
    ) -> None:
        """Run one check and convert unexpected errors into findings."""
        start = time.perf_counter()

        try:
            result = check_function()
            result.duration_ms = round(
                (time.perf_counter() - start) * 1000,
                3,
            )
            self.results.append(result)

        except Exception as exc:
            self._record(
                check_id=check_id,
                category=category,
                status="ERROR",
                message="Validation check raised an unexpected exception.",
                details={
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(limit=3),
                },
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    def check_project_root(self) -> ValidationResult:
        """Check that the configured project root exists."""
        if not self.project_root.exists():
            return ValidationResult(
                "PROJECT_ROOT",
                "structure",
                "FAIL",
                "Project root does not exist.",
                {"project_root": str(self.project_root)},
            )

        if not self.project_root.is_dir():
            return ValidationResult(
                "PROJECT_ROOT",
                "structure",
                "FAIL",
                "Project root is not a directory.",
                {"project_root": str(self.project_root)},
            )

        return ValidationResult(
            "PROJECT_ROOT",
            "structure",
            "PASS",
            "Project root exists.",
            {"project_root": str(self.project_root)},
        )

    def check_required_files(self) -> ValidationResult:
        """Check the expected public project entry points."""
        found: list[str] = []
        missing: list[str] = []

        for relative_path in DEFAULT_REQUIRED_FILES:
            candidate = self.project_root / relative_path
            if candidate.is_file():
                found.append(relative_path)
            else:
                missing.append(relative_path)

        if missing:
            return ValidationResult(
                "REQUIRED_FILES",
                "structure",
                "FAIL",
                "One or more required files are missing.",
                {
                    "found": found,
                    "missing": missing,
                    "note": (
                        "Paths reflect the default project layout and may "
                        "need adjustment if the repository structure differs."
                    ),
                },
            )

        return ValidationResult(
            "REQUIRED_FILES",
            "structure",
            "PASS",
            "All configured required files were found.",
            {"found": found, "missing": []},
        )

    def check_optional_files(self) -> ValidationResult:
        """Report optional files without treating absence as a failure."""
        found = []
        missing = []

        for relative_path in OPTIONAL_FILES:
            if (self.project_root / relative_path).is_file():
                found.append(relative_path)
            else:
                missing.append(relative_path)

        return ValidationResult(
            "OPTIONAL_FILES",
            "structure",
            "INFO",
            "Optional file inventory completed.",
            {"found": found, "missing": missing},
        )

    def check_python_syntax(self) -> ValidationResult:
        """
        Parse project Python files without importing or executing them.
        """
        dashboard_dir = self.project_root / "02_DASHBOARD"

        if not dashboard_dir.is_dir():
            return ValidationResult(
                "PYTHON_SYNTAX",
                "code",
                "SKIP",
                "Dashboard directory is unavailable.",
                {"path": str(dashboard_dir)},
            )

        python_files = sorted(
            path
            for path in dashboard_dir.rglob("*.py")
            if "__pycache__" not in path.parts
        )

        if not python_files:
            return ValidationResult(
                "PYTHON_SYNTAX",
                "code",
                "WARN",
                "No Python files were found in the dashboard directory.",
                {"path": str(dashboard_dir)},
            )

        valid: list[str] = []
        invalid: list[dict[str, str]] = []

        for path in python_files:
            try:
                source = path.read_text(encoding="utf-8")
                ast.parse(source, filename=str(path))
                valid.append(str(path.relative_to(self.project_root)))
            except (SyntaxError, UnicodeDecodeError, OSError) as exc:
                invalid.append({
                    "file": str(path.relative_to(self.project_root)),
                    "error": f"{type(exc).__name__}: {exc}",
                })

        if invalid:
            return ValidationResult(
                "PYTHON_SYNTAX",
                "code",
                "FAIL",
                "One or more Python files could not be parsed.",
                {"valid_files": valid, "invalid_files": invalid},
            )

        return ValidationResult(
            "PYTHON_SYNTAX",
            "code",
            "PASS",
            "All discovered dashboard Python files parsed successfully.",
            {
                "checked_count": len(valid),
                "files": valid,
                "note": (
                    "Syntax parsing does not prove runtime compatibility, "
                    "correct outputs, or scientific validity."
                ),
            },
        )

    def check_dependencies(self) -> ValidationResult:
        """Inventory optional runtime dependencies."""
        available: list[str] = []
        missing: list[str] = []

        for package in OPTIONAL_PACKAGES:
            try:
                spec = importlib.util.find_spec(package)
            except (ImportError, ValueError, ModuleNotFoundError):
                spec = None

            if spec is None:
                missing.append(package)
            else:
                available.append(package)

        status = "INFO" if not missing else "WARN"

        return ValidationResult(
            "DEPENDENCIES",
            "environment",
            status,
            "Optional dependency inventory completed.",
            {
                "available": available,
                "missing": missing,
                "note": (
                    "Package presence does not verify version compatibility "
                    "or successful execution of every feature."
                ),
            },
        )

    def check_runtime(self) -> ValidationResult:
        """Record the Python runtime information."""
        return ValidationResult(
            "RUNTIME",
            "environment",
            "INFO",
            "Runtime metadata recorded.",
            {
                "python_version": sys.version,
                "python_implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "executable": sys.executable,
            },
        )

    def check_dataset(self) -> ValidationResult:
        """Perform basic, non-destructive dataset inspection."""
        if self.dataset_path is None:
            return ValidationResult(
                "DATASET_INSPECTION",
                "data",
                "SKIP",
                "No dataset path was supplied.",
                {"expected_argument": "--dataset"},
            )

        path = self.dataset_path

        if not path.exists():
            return ValidationResult(
                "DATASET_INSPECTION",
                "data",
                "FAIL",
                "The supplied dataset path does not exist.",
                {"dataset_path": str(path)},
            )

        if path.is_dir():
            files = sorted(
                item for item in path.rglob("*") if item.is_file()
            )
            extensions: dict[str, int] = {}

            for item in files:
                extension = item.suffix.lower() or "[no extension]"
                extensions[extension] = extensions.get(extension, 0) + 1

            return ValidationResult(
                "DATASET_INSPECTION",
                "data",
                "INFO",
                "Dataset directory inventory completed.",
                {
                    "dataset_path": str(path),
                    "file_count": len(files),
                    "extensions": extensions,
                    "note": "Files were inventoried; contents were not validated.",
                },
            )

        extension = path.suffix.lower()
        details: dict[str, Any] = {
            "dataset_path": str(path),
            "extension": extension,
            "size_bytes": path.stat().st_size,
            "supported_extension": extension in SUPPORTED_DATA_EXTENSIONS,
        }

        if extension == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
                row_count = sum(1 for _ in reader)

            details.update({
                "header": header,
                "data_row_count": row_count,
                "header_present": bool(header),
                "duplicate_header_names": (
                    sorted({
                        name for name in (header or [])
                        if (header or []).count(name) > 1
                    })
                ),
            })

            if not header:
                return ValidationResult(
                    "DATASET_INSPECTION",
                    "data",
                    "FAIL",
                    "CSV file appears to have no header row.",
                    details,
                )

            return ValidationResult(
                "DATASET_INSPECTION",
                "data",
                "INFO",
                "CSV structural inspection completed.",
                {
                    **details,
                    "note": (
                        "This check does not validate scientific meaning, "
                        "units, missing-value policy, or statistical suitability."
                    ),
                },
            )

        if extension == ".json":
            with path.open("r", encoding="utf-8") as handle:
                json.load(handle)

            return ValidationResult(
                "DATASET_INSPECTION",
                "data",
                "PASS",
                "JSON syntax is valid.",
                {
                    **details,
                    "note": "Schema and scientific semantics were not checked.",
                },
            )

        if extension in {".png", ".jpg", ".jpeg", ".webp"}:
            with path.open("rb") as handle:
                signature = handle.read(16)

            details["signature_hex"] = signature.hex()

            return ValidationResult(
                "DATASET_INSPECTION",
                "data",
                "INFO",
                "Image file header bytes were recorded.",
                {
                    **details,
                    "note": (
                        "Image decoding and image-content validity were "
                        "not verified by this standard-library check."
                    ),
                },
            )

        if extension == ".dcm":
            with path.open("rb") as handle:
                handle.seek(128)
                marker = handle.read(4)

            details["dicom_preamble_marker_present"] = marker == b"DICM"

            return ValidationResult(
                "DATASET_INSPECTION",
                "data",
                "INFO",
                "DICOM preamble marker inspection completed.",
                {
                    **details,
                    "note": (
                        "Some valid DICOM files may lack the standard "
                        "preamble. This check does not decode pixels, inspect "
                        "patient metadata, or validate clinical meaning."
                    ),
                },
            )

        if extension == ".txt":
            with path.open("r", encoding="utf-8") as handle:
                sample = handle.read(4096)

            details["sample_characters_read"] = len(sample)

            return ValidationResult(
                "DATASET_INSPECTION",
                "data",
                "INFO",
                "Text file can be read as UTF-8.",
                details,
            )

        return ValidationResult(
            "DATASET_INSPECTION",
            "data",
            "WARN",
            "File extension is not supported by the basic inspector.",
            details,
        )

    def run(self) -> dict[str, Any]:
        """Execute the configured checks and return a report dictionary."""
        self.results = []
        self.started_at = self.utc_now()

        checks: tuple[tuple[str, str, Callable[[], ValidationResult]], ...] = (
            ("PROJECT_ROOT", "structure", self.check_project_root),
            ("REQUIRED_FILES", "structure", self.check_required_files),
            ("OPTIONAL_FILES", "structure", self.check_optional_files),
            ("PYTHON_SYNTAX", "code", self.check_python_syntax),
            ("DEPENDENCIES", "environment", self.check_dependencies),
            ("RUNTIME", "environment", self.check_runtime),
            ("DATASET_INSPECTION", "data", self.check_dataset),
        )

        for check_id, category, function in checks:
            self._run_check(check_id, category, function)

        self.finished_at = self.utc_now()

        counts: dict[str, int] = {}
        for result in self.results:
            counts[result.status] = counts.get(result.status, 0) + 1

        fail_count = counts.get("FAIL", 0) + counts.get("ERROR", 0)

        report_status = (
            "FAIL" if fail_count
            else "PASS_WITH_WARNINGS"
            if counts.get("WARN", 0) or counts.get("SKIP", 0)
            else "PASS"
        )

        return {
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "project_name": PROJECT_NAME,
            "module": MODULE_NAME,
            "module_number": MODULE_NUMBER,
            "module_version": MODULE_VERSION,
            "report_status": report_status,
            "strict_mode": self.strict,
            "started_at_utc": self.started_at,
            "finished_at_utc": self.finished_at,
            "project_root": str(self.project_root),
            "dataset_path": (
                str(self.dataset_path) if self.dataset_path else None
            ),
            "summary": {
                "total_checks": len(self.results),
                "status_counts": counts,
                "fail_or_error_count": fail_count,
            },
            "results": [result.to_dict() for result in self.results],
            "interpretation": {
                "diagnostic_only": True,
                "scientific_validity_established": False,
                "medical_diagnosis_performed": False,
                "note": (
                    "This report summarizes software and basic structural "
                    "checks only. It does not establish scientific validity, "
                    "clinical validity, or operational readiness."
                ),
            },
        }

    def save_report(
        self,
        report: dict[str, Any],
    ) -> dict[str, str]:
        """Write JSON and CSV reports without changing source inputs."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        json_path = self.output_dir / f"validation_report_{stamp}.json"
        csv_path = self.output_dir / f"validation_report_{stamp}.csv"

        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)

        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "check_id",
                "category",
                "status",
                "message",
                "duration_ms",
                "details_json",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

            for result in report.get("results", []):
                writer.writerow({
                    "check_id": result.get("check_id", ""),
                    "category": result.get("category", ""),
                    "status": result.get("status", ""),
                    "message": result.get("message", ""),
                    "duration_ms": result.get("duration_ms", 0),
                    "details_json": json.dumps(
                        result.get("details", {}),
                        ensure_ascii=False,
                    ),
                })

        return {
            "json_report": str(json_path),
            "csv_report": str(csv_path),
        }


def print_report_summary(report: dict[str, Any]) -> None:
    """Print a compact, readable console summary."""
    print("\n" + "=" * 68)
    print("D³ VITAL-X — VALIDATION REPORT")
    print("=" * 68)
    print("Module:", f"{MODULE_NUMBER}. {MODULE_NAME}")
    print("Report status:", report.get("report_status", "UNKNOWN"))
    print("Project root:", report.get("project_root", "N/A"))

    summary = report.get("summary", {})
    print("Total checks:", summary.get("total_checks", 0))
    print("Status counts:", json.dumps(
        summary.get("status_counts", {}),
        ensure_ascii=False,
    ))

    print("-" * 68)
    for result in report.get("results", []):
        print(
            f"[{result['status']:<5}] "
            f"{result['check_id']}: {result['message']}"
        )

    print("-" * 68)
    print("Scientific validity established: NO")
    print("Medical diagnosis performed: NO")
    print("=" * 68 + "\n")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run diagnostic-only validation checks for the "
            "D³ VITAL-X Space Intelligence Platform."
        )
    )

    parser.add_argument(
        "--project-root",
        default=None,
        help="Repository root. Defaults to this file's parent project root.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Report output directory. Defaults to PROJECT_ROOT/validation_reports.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional CSV, JSON, text, image, DICOM file, or directory to inspect.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 1 if any check has FAIL or ERROR status.",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)

    # Assumes this module is stored in PROJECT_ROOT/03_VALIDATION/.
    default_root = Path(__file__).resolve().parent.parent
    project_root = Path(args.project_root).expanduser() if args.project_root else default_root

    runner = ValidationRunner(
        project_root=project_root,
        output_dir=args.output_dir,
        dataset_path=args.dataset,
        strict=args.strict,
    )

    report = runner.run()
    report_paths = runner.save_report(report)
    print_report_summary(report)

    print("JSON report:", report_paths["json_report"])
    print("CSV report:", report_paths["csv_report"])

    has_failures = report["summary"]["fail_or_error_count"] > 0

    if args.strict and has_failures:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
