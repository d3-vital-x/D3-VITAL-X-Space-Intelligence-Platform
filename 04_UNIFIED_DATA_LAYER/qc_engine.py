
"""
D³ VITAL-X SPACE INTELLIGENCE PLATFORM
Module: qc_engine.py
Version: 0.1.0

Purpose:
    Run non-destructive structural and basic data-quality checks
    on UnifiedDataRecord objects and optional in-memory payloads.

Dependencies:
    data_schema.py

Safety boundary:
    - QC is not scientific validation.
    - Negative values are flagged, not automatically rejected.
    - No medical diagnosis or treatment recommendation.
    - Source payloads are never modified.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import importlib.util
import math


# ============================================================
# 1. Load the shared schema module
# ============================================================

PROJECT_ROOT = Path(
    "/content/D3-VITAL-X-Space-Intelligence-Platform"
)
MODULE_DIR = PROJECT_ROOT / "04_UNIFIED_DATA_LAYER"
SCHEMA_PATH = MODULE_DIR / "data_schema.py"


def _load_schema_module():
    """
    Load data_schema.py from the project folder.

    A file-based import is used so this module can also be
    imported when the project folder is not a Python package.
    """
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Required schema module not found: {SCHEMA_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "d3_vital_x_data_schema",
        SCHEMA_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load schema module: {SCHEMA_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_schema = _load_schema_module()

UnifiedDataRecord = _schema.UnifiedDataRecord
validate_schema = _schema.validate_schema

QC_VERSION = "0.1.0"


# ============================================================
# 2. Report construction
# ============================================================

def create_qc_report(dataset_id: Optional[str] = None) -> Dict[str, Any]:
    """Create an empty, machine-readable QC report."""
    return {
        "qc_version": QC_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "overall_status": "NOT_EVALUATED",
        "summary": {
            "total_checks": 0,
            "passed_checks": 0,
            "warning_count": 0,
            "error_count": 0,
        },
        "checks": [],
        "flags": [],
        "raw_data_modified": False,
        "scientific_validation_performed": False,
        "clinical_diagnosis_performed": False,
    }


def add_qc_check(
    report: Dict[str, Any],
    check_name: str,
    passed: bool,
    severity: str = "INFO",
    message: str = "",
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one QC result and update the report counters."""
    severity = severity.upper()

    if severity not in {"INFO", "WARNING", "ERROR"}:
        raise ValueError(f"Unsupported severity: {severity}")

    if not isinstance(check_name, str) or not check_name.strip():
        raise ValueError("check_name must be a non-empty string")

    details = dict(details or {})
    passed = bool(passed)

    check = {
        "check_name": check_name,
        "passed": passed,
        "severity": severity,
        "message": str(message),
        "details": details,
    }

    report["checks"].append(check)
    summary = report["summary"]
    summary["total_checks"] += 1

    if passed:
        summary["passed_checks"] += 1

    if severity == "WARNING":
        summary["warning_count"] += 1

    if severity == "ERROR":
        summary["error_count"] += 1

    if not passed or severity in {"WARNING", "ERROR"}:
        report["flags"].append({
            "check_name": check_name,
            "severity": severity,
            "message": str(message),
            "details": details,
        })


# ============================================================
# 3. Value inspection helpers
# ============================================================

def is_nan_value(value: Any) -> bool:
    """Return True for numeric NaN values."""
    if isinstance(value, bool):
        return False

    try:
        result = math.isnan(value)
        return isinstance(result, bool) and result
    except (TypeError, ValueError, OverflowError):
        try:
            return bool(math.isnan(value))
        except (TypeError, ValueError, OverflowError):
            return False


def is_infinite_value(value: Any) -> bool:
    """Return True for positive or negative infinity."""
    if isinstance(value, bool):
        return False

    try:
        result = math.isinf(value)
        return isinstance(result, bool) and result
    except (TypeError, ValueError, OverflowError):
        try:
            return bool(math.isinf(value))
        except (TypeError, ValueError, OverflowError):
            return False


def is_numeric_value(value: Any) -> bool:
    """Recognize ordinary Python numeric values, excluding bool."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    )


def inspect_values(data: Any, path: str = "root") -> Dict[str, Any]:
    """
    Recursively inspect dictionary/list/tuple payloads.

    This is a generic scan: it does not assume negative values
    are physically invalid, and it does not interpret units.
    """
    result = {
        "total_values": 0,
        "missing_values": 0,
        "nan_values": 0,
        "infinite_values": 0,
        "negative_values": 0,
        "non_numeric_values": 0,
        "negative_value_paths": [],
        "nan_value_paths": [],
        "infinite_value_paths": [],
    }

    def visit(value: Any, current_path: str) -> None:
        if isinstance(value, dict):
            if not value:
                result["total_values"] += 1
                result["non_numeric_values"] += 1
                return

            for key, item in value.items():
                visit(item, f"{current_path}.{key}")
            return

        if isinstance(value, (list, tuple)):
            if not value:
                result["total_values"] += 1
                result["non_numeric_values"] += 1
                return

            for index, item in enumerate(value):
                visit(item, f"{current_path}[{index}]")
            return

        result["total_values"] += 1

        if value is None:
            result["missing_values"] += 1
            return

        if is_nan_value(value):
            result["nan_values"] += 1
            result["nan_value_paths"].append(current_path)
            return

        if is_infinite_value(value):
            result["infinite_values"] += 1
            result["infinite_value_paths"].append(current_path)
            return

        if is_numeric_value(value):
            if value < 0:
                result["negative_values"] += 1
                result["negative_value_paths"].append(current_path)
        else:
            result["non_numeric_values"] += 1

    visit(data, path)
    return result


# ============================================================
# 4. Basic QC checks
# ============================================================

def _add_count_check(
    report: Dict[str, Any],
    name: str,
    count: int,
    issue_label: str,
    paths: Optional[list] = None,
) -> None:
    """Add a standard check for a count of detected issues."""
    passed = count == 0
    details: Dict[str, Any] = {"count": count}

    if paths is not None:
        details["paths"] = paths[:20]

    add_qc_check(
        report,
        name,
        passed,
        "INFO" if passed else "ERROR",
        (
            f"No {issue_label} detected."
            if passed
            else f"{count} {issue_label} detected."
        ),
        details,
    )


def run_missing_value_check(report, inspection) -> None:
    _add_count_check(
        report,
        "missing_values",
        inspection["missing_values"],
        "missing value(s)",
    )


def run_nan_check(report, inspection) -> None:
    _add_count_check(
        report,
        "nan_values",
        inspection["nan_values"],
        "NaN value(s)",
        inspection["nan_value_paths"],
    )


def run_infinite_check(report, inspection) -> None:
    _add_count_check(
        report,
        "infinite_values",
        inspection["infinite_values"],
        "infinite value(s)",
        inspection["infinite_value_paths"],
    )


def run_negative_value_check(report, inspection) -> None:
    count = inspection["negative_values"]

    add_qc_check(
        report,
        "negative_values",
        True,
        "WARNING" if count else "INFO",
        (
            "Negative values flagged for contextual review."
            if count
            else "No negative numeric values detected."
        ),
        {
            "count": count,
            "policy": "FLAG_ONLY",
            "paths": inspection["negative_value_paths"][:20],
        },
    )


def run_duplicate_check(report, data) -> None:
    """
    Count repeated scalar values in a top-level list or tuple.

    Repeated measurements are not automatically errors.
    Nested objects and dictionary records are not deduplicated here.
    """
    duplicate_count = 0
    sequence_length = 0

    if isinstance(data, (list, tuple)):
        sequence_length = len(data)
        scalar_keys = []

        for value in data:
            if value is None or isinstance(
                value, (str, int, float, bool)
            ):
                scalar_keys.append(repr(value))

        duplicate_count = len(scalar_keys) - len(set(scalar_keys))

    has_duplicates = duplicate_count > 0

    add_qc_check(
        report,
        "duplicate_values",
        True,
        "WARNING" if has_duplicates else "INFO",
        (
            "Repeated scalar values detected; review in context."
            if has_duplicates
            else "No repeated top-level scalar values detected."
        ),
        {
            "sequence_length": sequence_length,
            "duplicate_count": duplicate_count,
            "policy": "FLAG_ONLY",
        },
    )


def run_time_series_check(report, data) -> None:
    """
    Check a dictionary containing 'time' and 'signal' sequences.

    Time monotonicity is tested only when time values are numeric.
    """
    if not isinstance(data, dict):
        add_qc_check(
            report,
            "time_series_structure",
            True,
            "INFO",
            "Payload is not a dictionary time-series structure.",
        )
        return

    time_values = data.get("time")
    signal_values = data.get("signal")

    if time_values is None or signal_values is None:
        add_qc_check(
            report,
            "time_series_structure",
            True,
            "INFO",
            "Both 'time' and 'signal' fields are not present.",
        )
        return

    sequences_valid = (
        isinstance(time_values, (list, tuple))
        and isinstance(signal_values, (list, tuple))
    )

    lengths_match = (
        sequences_valid and len(time_values) == len(signal_values)
    )

    add_qc_check(
        report,
        "time_signal_length_match",
        lengths_match,
        "INFO" if lengths_match else "ERROR",
        (
            "Time and signal sequences have matching lengths."
            if lengths_match
            else "Time and signal sequences must be lists/tuples "
                 "of equal length."
        ),
        {
            "time_length": (
                len(time_values)
                if isinstance(time_values, (list, tuple))
                else None
            ),
            "signal_length": (
                len(signal_values)
                if isinstance(signal_values, (list, tuple))
                else None
            ),
        },
    )

    if not sequences_valid:
        return

    if not all(is_numeric_value(value) for value in time_values):
        add_qc_check(
            report,
            "time_numeric",
            False,
            "WARNING",
            "Time values are not all ordinary numeric values.",
        )
        return

    if len(time_values) > 1:
        non_monotonic_count = sum(
            1
            for previous, current in zip(
                time_values, time_values[1:]
            )
            if current <= previous
        )

        monotonic = non_monotonic_count == 0

        add_qc_check(
            report,
            "time_monotonicity",
            monotonic,
            "INFO" if monotonic else "WARNING",
            (
                "Time values are strictly increasing."
                if monotonic
                else "Time values are not strictly increasing."
            ),
            {"non_monotonic_count": non_monotonic_count},
        )


# ============================================================
# 5. Record and payload access
# ============================================================

def _record_to_dict(record: Any) -> Optional[Dict[str, Any]]:
    """Return a dictionary view of a supported record."""
    if isinstance(record, UnifiedDataRecord):
        return record.to_dict()

    if isinstance(record, dict):
        return record

    if hasattr(record, "to_dict") and callable(record.to_dict):
        try:
            converted = record.to_dict()
            if isinstance(converted, dict):
                return converted
        except Exception:
            return None

    return None


def _get_dataset_id(record_dict: Dict[str, Any]) -> Optional[str]:
    metadata = record_dict.get("metadata")

    if isinstance(metadata, dict):
        dataset_id = metadata.get("dataset_id")
        if dataset_id is not None:
            return str(dataset_id)

    return record_dict.get("record_id")


def _get_payload(record: Any, record_dict: Dict[str, Any], data: Any):
    """
    Prefer explicit payload argument; otherwise use legacy .data
    when present. payload_ref is only a reference and is not loaded.
    """
    if data is not None:
        return data, True

    if hasattr(record, "data"):
        return getattr(record, "data"), True

    if isinstance(record_dict, dict) and "data" in record_dict:
        return record_dict["data"], True

    return None, False


# ============================================================
# 6. Main QC runner
# ============================================================

def run_qc(
    record: Any,
    run_structural_validation: bool = True,
    data: Any = None,
) -> Dict[str, Any]:
    """
    Run QC checks on a schema record and optional payload.

    Parameters
    ----------
    record:
        UnifiedDataRecord or dictionary matching the schema.
    run_structural_validation:
        Whether to run data_schema.validate_schema().
    data:
        Optional in-memory payload. If omitted, a legacy .data
        attribute is used when available. File references are not
        automatically opened or downloaded.

    Returns
    -------
    dict
        A machine-readable QC report.
    """
    record_dict = _record_to_dict(record)

    if record_dict is None:
        report = create_qc_report()

        add_qc_check(
            report,
            "record_type",
            False,
            "ERROR",
            "Input is not a supported record or dictionary.",
        )
        report["overall_status"] = "REJECTED"
        return report

    report = create_qc_report(_get_dataset_id(record_dict))

    if run_structural_validation:
        validation_result = validate_schema(record_dict)
        validation_passed = bool(validation_result.get("valid", False))

        add_qc_check(
            report,
            "structural_validation",
            validation_passed,
            "INFO" if validation_passed else "ERROR",
            (
                "Structural schema validation passed."
                if validation_passed
                else "Structural schema validation failed."
            ),
            {
                "errors": validation_result.get("errors", []),
                "warnings": validation_result.get("warnings", []),
            },
        )

    payload, payload_available = _get_payload(
        record, record_dict, data
    )

    if payload_available:
        inspection = inspect_values(payload)

        run_missing_value_check(report, inspection)
        run_nan_check(report, inspection)
        run_infinite_check(report, inspection)
        run_negative_value_check(report, inspection)
        run_duplicate_check(report, payload)
        run_time_series_check(report, payload)

        add_qc_check(
            report,
            "payload_inspection",
            True,
            "INFO",
            "In-memory payload inspection completed.",
            {
                "total_values": inspection["total_values"],
                "non_numeric_values": inspection["non_numeric_values"],
            },
        )
    else:
        add_qc_check(
            report,
            "payload_inspection",
            True,
            "INFO",
            (
                "No in-memory payload supplied. "
                "Referenced file contents were not inspected."
            ),
        )

    report["raw_data_modified"] = False

    if report["summary"]["error_count"] > 0:
        report["overall_status"] = "REJECTED"
    elif report["summary"]["warning_count"] > 0:
        report["overall_status"] = "PASSED_WITH_WARNINGS"
    else:
        report["overall_status"] = "PASSED"

    return report


# ============================================================
# 7. Deterministic smoke test
# ============================================================

def self_check() -> Dict[str, Any]:
    """Run small tests using synthetic data only."""
    checks = {}

    sample_record = _schema.create_unified_record(
        source="qc_self_check",
        modality="time_series",
        data_type="synthetic",
        metadata={"dataset_id": "synthetic-qc-001"},
        payload_ref="example://synthetic-signal",
    )

    good_payload = {
        "time": [0, 1, 2, 3],
        "signal": [1.2, 1.4, 1.3, 1.5],
    }

    good_report = run_qc(sample_record, data=good_payload)
    checks["valid_record_accepted"] = (
        good_report["overall_status"] == "PASSED"
    )

    bad_payload = {
        "time": [0, 1, 2],
        "signal": [1.0, float("nan")],
    }

    bad_report = run_qc(sample_record, data=bad_payload)
    bad_check_names = {
        check["check_name"]
        for check in bad_report["checks"]
        if not check["passed"]
    }

    checks["length_mismatch_detected"] = (
        "time_signal_length_match" in bad_check_names
    )
    checks["nan_detected"] = "nan_values" in bad_check_names
    checks["raw_data_not_modified"] = (
        good_report["raw_data_modified"] is False
        and bad_report["raw_data_modified"] is False
    )
    checks["scientific_validation_not_claimed"] = (
        good_report["scientific_validation_performed"] is False
    )

    return {
        "qc_version": QC_VERSION,
        "passed": all(checks.values()),
        "checks": checks,
    }


if __name__ == "__main__":
    import json

    result = self_check()
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if not result["passed"]:
        raise SystemExit("QC engine self-check failed.")

    print("\nQC engine self-check passed.")
