
"""
D³ VITAL-X — Public Report Export
Module 17: export.py

Purpose
-------
Export public analytics reports to JSON and CSV.

Features
--------
- JSON export for the complete structured report
- CSV export for tabular result summaries
- SHA-256 checksum for exported file bytes
- Safe filename validation
- Output-directory containment checks
- Optional overwrite protection

Scientific and privacy boundaries
---------------------------------
- Exporting does not validate scientific conclusions.
- SHA-256 checks file integrity only; it does not establish authenticity.
- Review report contents before public release.
- The protected intelligence core is not implemented here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile

from pathlib import Path
from typing import Any, Mapping, Optional


MODULE_NAME = "export"
MODULE_NUMBER = 17
MODULE_VERSION = "1.0.0"

DEFAULT_OUTPUT_DIR = "exports"

CSV_COLUMNS = [
    "report_title",
    "project",
    "result_name",
    "module",
    "module_version",
    "status",
    "sample_count",
    "aggregate_score",
    "candidate_flag",
    "overall_correlation",
    "entropy",
    "variance",
    "warnings",
    "errors",
    "interpretation",
    "result_fingerprint",
]


def _json_safe(value: Any) -> Any:
    """Convert common values to strict JSON-compatible data."""
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())

    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (ValueError, TypeError):
            pass

    return f"<unsupported:{type(value).__name__}>"


def _validate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the report before export."""
    if not isinstance(report, Mapping):
        raise ValueError("report must be a mapping.")

    normalized = _json_safe(report)

    if not isinstance(normalized, dict):
        raise ValueError("report could not be converted to a dictionary.")

    try:
        # Confirm strict JSON serialization before writing any file.
        json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Report contains values that cannot be serialized as strict JSON."
        ) from exc

    return normalized


def _validate_filename(filename: str) -> str:
    """Require a simple filename, not a path supplied by the caller."""
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must be a non-empty string.")

    if filename in {".", ".."}:
        raise ValueError("Invalid filename.")

    # Prevent directory traversal and absolute path input.
    if Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise ValueError(
            "filename must be a simple filename without directory components."
        )

    if "\x00" in filename:
        raise ValueError("filename contains an invalid null character.")

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", filename):
        raise ValueError(
            "filename may contain only letters, numbers, dots, "
            "underscores, and hyphens; it must start with a letter or number."
        )

    return filename


def _prepare_output_path(
    output_dir: str | Path,
    filename: str,
    *,
    overwrite: bool,
) -> Path:
    """
    Prepare an output path and reject existing targets unless overwrite=True.

    The destination directory is explicitly supplied by the caller.
    """
    safe_name = _validate_filename(filename)
    base_dir = Path(output_dir).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    target = (base_dir / safe_name).resolve()

    if target.parent != base_dir:
        raise ValueError("Target path must remain inside output_dir.")

    if target.exists() and not overwrite:
        raise FileExistsError(
            f"{target.name} already exists. Choose another filename "
            "or set overwrite=True."
        )

    if target.exists() and target.is_dir():
        raise IsADirectoryError(f"{target} is a directory.")

    return target


def _atomic_write(
    target: Path,
    content: bytes,
    *,
    overwrite: bool,
) -> None:
    """Write through a temporary file before publishing the final file."""
    temp_name: Optional[str] = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(target.parent),
            prefix=".export-",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_name = temp_file.name
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())

        if overwrite:
            os.replace(temp_name, target)
            temp_name = None
        else:
            # Exclusive creation avoids replacing an existing target.
            # Hard-link publication is atomic on common local filesystems.
            try:
                os.link(temp_name, target)
            except FileExistsError:
                raise FileExistsError(
                    f"{target.name} already exists; no file was overwritten."
                )
            except OSError as exc:
                # Do not silently fall back to a potentially overwriting move.
                raise OSError(
                    "Could not publish export without overwrite. "
                    "Use a supported local filesystem or choose another "
                    "export workflow."
                ) from exc

    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def _write_export(
    target: Path,
    content: bytes,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    """Write bytes and return export metadata."""
    _atomic_write(target, content, overwrite=overwrite)

    checksum = hashlib.sha256(content).hexdigest()

    return {
        "module": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "status": "exported",
        "format": target.suffix.lstrip(".").lower(),
        "filename": target.name,
        "path": str(target),
        "size_bytes": len(content),
        "sha256": checksum,
        "integrity_note": (
            "SHA-256 identifies the exported bytes; it does not prove "
            "source authenticity or scientific validity."
        ),
    }


def export_json(
    report: Mapping[str, Any],
    *,
    filename: str = "analysis_report.json",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    overwrite: bool = False,
    indent: int = 2,
) -> dict[str, Any]:
    """
    Export a complete report as UTF-8 JSON.

    Returns metadata describing the created file and its SHA-256 checksum.
    """
    if isinstance(indent, bool) or not isinstance(indent, int) or indent < 0:
        raise ValueError("indent must be a non-negative integer.")

    normalized = _validate_report(report)
    target = _prepare_output_path(
        output_dir,
        filename,
        overwrite=overwrite,
    )

    content = (
        json.dumps(
            normalized,
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    return _write_export(target, content, overwrite=overwrite)


def _result_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert report result summaries into CSV rows."""
    results = report.get("results", [])

    if not isinstance(results, list):
        raise ValueError("report['results'] must be a list for CSV export.")

    title = report.get("title", "")
    project = report.get("project", "")
    rows: list[dict[str, Any]] = []

    for index, result in enumerate(results):
        if not isinstance(result, Mapping):
            raise ValueError(
                f"report['results'][{index}] must be a mapping."
            )

        row = {
            "report_title": title,
            "project": project,
            "result_name": result.get("name", ""),
            "module": result.get("module", ""),
            "module_version": result.get("module_version", ""),
            "status": result.get("status", ""),
            "sample_count": result.get("sample_count", ""),
            "aggregate_score": result.get("aggregate_score", ""),
            "candidate_flag": result.get("candidate_flag", ""),
            "overall_correlation": result.get("overall_correlation", ""),
            "entropy": result.get("entropy", ""),
            "variance": result.get("variance", ""),
            "warnings": json.dumps(
                result.get("warnings", []),
                ensure_ascii=False,
                allow_nan=False,
            ),
            "errors": json.dumps(
                result.get("errors", []),
                ensure_ascii=False,
                allow_nan=False,
            ),
            "interpretation": result.get("interpretation", ""),
            "result_fingerprint": result.get("result_fingerprint", ""),
        }

        rows.append(
            {
                key: (
                    json.dumps(value, ensure_ascii=False, allow_nan=False)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
            }
        )

    return rows


def export_csv(
    report: Mapping[str, Any],
    *,
    filename: str = "analysis_summary.csv",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Export report result summaries as UTF-8 CSV.

    Nested warnings and errors are JSON-encoded within their CSV cells.
    This CSV is a summary, not a replacement for the complete JSON report.
    """
    normalized = _validate_report(report)
    target = _prepare_output_path(
        output_dir,
        filename,
        overwrite=overwrite,
    )

    rows = _result_rows(normalized)

    # utf-8-sig helps spreadsheet applications recognize UTF-8 text.
    import io

    text_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        text_buffer,
        fieldnames=CSV_COLUMNS,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)

    content = text_buffer.getvalue().encode("utf-8-sig")

    return _write_export(target, content, overwrite=overwrite)


def module_info() -> dict[str, Any]:
    """Return public module metadata."""
    return {
        "module": MODULE_NAME,
        "module_number": MODULE_NUMBER,
        "version": MODULE_VERSION,
        "purpose": "Export public analytics reports to JSON and CSV",
        "functions": [
            "export_json",
            "export_csv",
            "module_info",
            "run_contract_tests",
        ],
        "formats": ["JSON", "CSV"],
        "integrity": "SHA-256 checksum of exported bytes",
        "scientific_boundary": (
            "Export and integrity metadata do not establish scientific validity."
        ),
    }


def run_contract_tests() -> dict[str, bool]:
    """
    Check input validation and serialization behavior.

    Uses small structural contract objects only; no scientific observations
    are generated or evaluated.
    """
    checks: dict[str, bool] = {}

    try:
        _validate_filename("../unsafe.json")
        checks["path_traversal_filename_rejected"] = False
    except ValueError:
        checks["path_traversal_filename_rejected"] = True

    try:
        _validate_filename("report.json")
        checks["simple_filename_accepted"] = True
    except ValueError:
        checks["simple_filename_accepted"] = False

    try:
        _validate_report({"value": float("nan")})
        checks["nonfinite_json_value_normalized"] = True
    except (ValueError, TypeError):
        checks["nonfinite_json_value_normalized"] = False

    try:
        _result_rows(
            {
                "title": "Contract report",
                "project": "D3-VITAL-X",
                "results": [
                    {
                        "name": "contract",
                        "status": "checked",
                        "warnings": [],
                        "errors": [],
                    }
                ],
            }
        )
        checks["summary_csv_rows_built"] = True
    except (ValueError, TypeError):
        checks["summary_csv_rows_built"] = False

    checks["all_checks_passed"] = all(checks.values())
    return checks


if __name__ == "__main__":
    print(json.dumps(module_info(), indent=2, ensure_ascii=False))
    print(json.dumps(run_contract_tests(), indent=2, ensure_ascii=False))
