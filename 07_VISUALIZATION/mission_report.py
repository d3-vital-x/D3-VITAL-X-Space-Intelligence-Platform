
"""
Module 40: mission_report.py

D³ VITAL-X Space Intelligence Platform

Purpose
-------
Generate structured, reproducible mission-processing reports.

Supported report content:
- Mission and science-product identification
- Input file metadata and optional SHA-256 checksum
- FITS validation and selected HDU information
- Instrument and observation metadata
- Quality-control warnings
- Visualization summary
- Processing settings
- Software and provenance metadata
- Calibration status

Design principles:
- Standard-library implementation
- JSON-safe output
- Explicit uncertainty and missing metadata
- No automatic scientific claim or anomaly confirmation
- Calibration is never assumed to be completed
- Suitable for JWST, Chandra and future mission adapters

This module produces processing reports.
It does not certify scientific validity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


MODULE_NAME = "mission_report"
MODULE_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0"

PROJECT_NAME = "D3-VITAL-X Space Intelligence Platform"

MAX_REPORT_BYTES = 10 * 1024 * 1024
MAX_HASH_FILE_BYTES = 2 * 1024 * 1024 * 1024
HASH_CHUNK_SIZE = 1024 * 1024


class MissionReportError(Exception):
    """Base exception for mission reporting."""


class MissionReportValidationError(MissionReportError):
    """Raised when report inputs are invalid."""


class MissionReportResourceError(MissionReportError):
    """Raised when a resource limit is exceeded."""


class MissionReportExportError(MissionReportError):
    """Raised when report export fails."""


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def _safe_text(value: Any, max_length: int = 500) -> Optional[str]:
    """Convert a value into bounded text."""

    if value is None:
        return None

    try:
        result = str(value).strip()
    except Exception:
        return None

    if not result:
        return None

    return result[:max_length]


def _safe_number(value: Any) -> Optional[float]:
    """Convert a finite numeric value into float."""

    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(number):
        return None

    return number


def _json_safe(value: Any, depth: int = 0) -> Any:
    """
    Convert supported objects into JSON-safe values.

    NumPy-like scalar values are handled through item().
    Arrays should be summarized by adapters before reporting.
    """

    if depth > 12:
        return "[maximum nesting depth reached]"

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        output = {}

        for key, item in list(value.items())[:500]:
            output[str(key)[:200]] = _json_safe(
                item,
                depth + 1
            )

        return output

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, depth + 1)
            for item in value[:1000]
        ]

    if hasattr(value, "item"):
        try:
            return _json_safe(value.item(), depth + 1)
        except Exception:
            pass

    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist(), depth + 1)
        except Exception:
            pass

    return _safe_text(value)


def _extract_mapping(
    source: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Return a safe copy of a dictionary-like input."""

    if source is None:
        return {}

    if not isinstance(source, dict):
        raise MissionReportValidationError(
            "Expected a dictionary input."
        )

    return _json_safe(source)


def calculate_sha256(
    file_path: str,
    max_bytes: int = MAX_HASH_FILE_BYTES
) -> Optional[str]:
    """
    Calculate SHA-256 using bounded streaming reads.

    Returns None when the file is unavailable.
    Raises a resource error when the file exceeds the limit.
    """

    path = Path(file_path)

    if not path.is_file():
        return None

    size = path.stat().st_size

    if size > max_bytes:
        raise MissionReportResourceError(
            "File exceeds SHA-256 processing limit."
        )

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(HASH_CHUNK_SIZE)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def _extract_warnings(
    qc_report: Dict[str, Any]
) -> list:
    """Collect warnings from common QC report structures."""

    warnings = []

    candidates = [
        qc_report.get("warnings"),
        qc_report.get("issues"),
        qc_report.get("messages"),
    ]

    for candidate in candidates:

        if not isinstance(candidate, (list, tuple)):
            continue

        for item in candidate[:200]:

            if isinstance(item, dict):
                message = (
                    item.get("message")
                    or item.get("description")
                    or item.get("code")
                )

                severity = item.get("severity", "unspecified")

                warnings.append({
                    "severity": _safe_text(severity, 50),
                    "message": _safe_text(message, 500)
                    or "Unspecified QC warning"
                })

            else:
                warnings.append({
                    "severity": "unspecified",
                    "message": _safe_text(item, 500)
                    or "Unspecified QC warning"
                })

    return warnings


class MissionReportBuilder:
    """
    Build mission-processing reports.

    Inputs are intentionally flexible dictionaries so that
    mission adapters and the science product router can supply
    their outputs without tight coupling.
    """

    def __init__(
        self,
        project_name: str = PROJECT_NAME,
        module_version: str = MODULE_VERSION
    ):
        self.project_name = _safe_text(project_name, 200)
        self.module_version = _safe_text(module_version, 50)

        if not self.project_name:
            raise MissionReportValidationError(
                "Project name cannot be empty."
            )

    def build(
        self,
        input_file: Optional[str] = None,
        mission: Optional[str] = None,
        product_type: Optional[str] = None,
        file_metadata: Optional[Dict[str, Any]] = None,
        fits_validation: Optional[Dict[str, Any]] = None,
        observation_metadata: Optional[Dict[str, Any]] = None,
        qc_report: Optional[Dict[str, Any]] = None,
        visualization: Optional[Dict[str, Any]] = None,
        processing_settings: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        calibration_completed: bool = False,
        calibration_details: Optional[str] = None,
        selected_hdu: Optional[Any] = None,
        plotted_point_count: Optional[int] = None,
        compute_checksum: bool = False
    ) -> Dict[str, Any]:
        """
        Construct a complete mission report.

        calibration_completed defaults to False.
        The builder does not infer calibration status.
        """

        if not isinstance(calibration_completed, bool):
            raise MissionReportValidationError(
                "calibration_completed must be boolean."
            )

        if plotted_point_count is not None:
            if (
                isinstance(plotted_point_count, bool)
                or not isinstance(plotted_point_count, int)
                or plotted_point_count < 0
            ):
                raise MissionReportValidationError(
                    "plotted_point_count must be a non-negative integer."
                )

        file_info = _extract_mapping(file_metadata)
        fits_info = _extract_mapping(fits_validation)
        observation = _extract_mapping(observation_metadata)
        qc = _extract_mapping(qc_report)
        visual = _extract_mapping(visualization)
        settings = _extract_mapping(processing_settings)
        source = _extract_mapping(provenance)

        filename = None
        file_size = None
        checksum = None

        if input_file is not None:
            path = Path(input_file)

            filename = path.name

            if path.is_file():
                file_size = path.stat().st_size

                if compute_checksum:
                    checksum = calculate_sha256(str(path))

        if file_info.get("filename") is not None:
            filename = _safe_text(
                file_info["filename"],
                300
            )

        if file_info.get("size_bytes") is not None:
            file_size = file_info["size_bytes"]

        if file_size is not None:
            try:
                file_size = int(file_size)
            except (TypeError, ValueError, OverflowError):
                file_size = None

        if selected_hdu is None:
            selected_hdu = fits_info.get("selected_hdu")

        if plotted_point_count is None:
            plotted_point_count = visual.get(
                "plotted_point_count"
            )

        if plotted_point_count is not None:
            try:
                plotted_point_count = int(plotted_point_count)
            except (TypeError, ValueError, OverflowError):
                plotted_point_count = None

        mission_name = _safe_text(
            mission or observation.get("mission"),
            100
        )

        product_name = _safe_text(
            product_type or observation.get("product_type"),
            100
        )

        warnings = _extract_warnings(qc)

        report = {
            "schema_version": SCHEMA_VERSION,

            "report": {
                "report_type": "mission_processing_report",
                "created_at_utc": _utc_now(),
                "status": "generated"
            },

            "project": {
                "name": self.project_name,
                "module": MODULE_NAME,
                "module_version": self.module_version
            },

            "mission": {
                "name": mission_name,
                "product_type": product_name,
                "instrument": _safe_text(
                    observation.get("instrument"),
                    100
                ),
                "detector": _safe_text(
                    observation.get("detector"),
                    100
                ),
                "observation_id": _safe_text(
                    observation.get("observation_id"),
                    150
                ),
                "proposal_id": _safe_text(
                    observation.get("proposal_id"),
                    100
                ),
                "target_name": _safe_text(
                    observation.get("target_name"),
                    200
                ),
                "observation_metadata": observation
            },

            "input": {
                "filename": filename,
                "size_bytes": file_size,
                "sha256": checksum,
                "checksum_status": (
                    "calculated"
                    if checksum
                    else "not_calculated"
                ),
                "file_metadata": file_info
            },

            "fits": {
                "is_valid": fits_info.get("is_valid"),
                "validation_status": _safe_text(
                    fits_info.get("validation_status"),
                    100
                ),
                "selected_hdu": _json_safe(selected_hdu),
                "hdu_count": fits_info.get("hdu_count"),
                "fits_metadata": fits_info
            },

            "quality_control": {
                "status": _safe_text(
                    qc.get("status"),
                    100
                ),
                "passed": qc.get("passed"),
                "warnings": warnings,
                "warning_count": len(warnings),
                "qc_summary": qc
            },

            "visualization": {
                "generated": bool(
                    visual.get("generated", False)
                ),
                "plot_type": _safe_text(
                    visual.get("plot_type"),
                    100
                ),
                "plotted_point_count": plotted_point_count,
                "output_format": _safe_text(
                    visual.get("output_format"),
                    50
                ),
                "visualization_summary": visual
            },

            "processing": {
                "settings": settings,
                "calibration": {
                    "completed": calibration_completed,
                    "details": _safe_text(
                        calibration_details,
                        500
                    )
                }
            },

            "provenance": {
                "generated_by": PROJECT_NAME,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "source": source
            },

            "scientific_scope": {
                "analysis_type": "exploratory_signal_processing",
                "scientific_validation": "not_certified",
                "anomaly_confirmation": False,
                "medical_diagnosis": False,
                "flight_certification": False
            }
        }

        return _json_safe(report)

    def save_json(
        self,
        report: Dict[str, Any],
        output_path: str,
        indent: int = 2
    ) -> str:
        """Save a report as a UTF-8 JSON file."""

        if not isinstance(report, dict):
            raise MissionReportValidationError(
                "Report must be a dictionary."
            )

        path = Path(output_path)

        try:
            serialized = json.dumps(
                report,
                indent=indent,
                ensure_ascii=False,
                allow_nan=False
            )

            encoded = serialized.encode("utf-8")

            if len(encoded) > MAX_REPORT_BYTES:
                raise MissionReportResourceError(
                    "Report exceeds maximum export size."
                )

            path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(path.parent),
                prefix=path.name + ".",
                suffix=".tmp",
                delete=False
            ) as temporary:

                temporary.write(serialized)
                temporary.flush()
                temp_path = temporary.name

            os.replace(temp_path, path)

            return str(path)

        except MissionReportError:
            raise

        except Exception as exc:
            raise MissionReportExportError(
                f"JSON export failed: {exc}"
            ) from exc

    @staticmethod
    def validate_report(
        report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform structural validation of a generated report."""

        required_sections = [
            "schema_version",
            "report",
            "project",
            "mission",
            "input",
            "fits",
            "quality_control",
            "visualization",
            "processing",
            "provenance",
            "scientific_scope"
        ]

        if not isinstance(report, dict):
            raise MissionReportValidationError(
                "Report must be a dictionary."
            )

        missing = [
            key for key in required_sections
            if key not in report
        ]

        try:
            json.dumps(
                report,
                allow_nan=False
            )
            json_safe = True
        except (TypeError, ValueError):
            json_safe = False

        return {
            "valid": (
                len(missing) == 0
                and json_safe
            ),
            "missing_sections": missing,
            "json_serializable": json_safe,
            "schema_version": report.get("schema_version")
        }


def build_mission_report(**kwargs) -> Dict[str, Any]:
    """Convenience function for report generation."""

    builder = MissionReportBuilder()
    return builder.build(**kwargs)


def save_mission_report(
    report: Dict[str, Any],
    output_path: str
) -> str:
    """Convenience function for JSON export."""

    builder = MissionReportBuilder()
    return builder.save_json(report, output_path)


def module_info() -> Dict[str, Any]:
    """Return module information."""

    return {
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "project": PROJECT_NAME,
        "report_format": "JSON",
        "checksum": "SHA-256",
        "scientific_scope": "exploratory_only"
    }


def run_contract_tests() -> Dict[str, Any]:
    """
    Run lightweight standard-library contract tests.

    These tests validate report construction and export.
    They do not test real JWST/Chandra FITS processing.
    """

    tests = []

    def record(name, function):
        try:
            function()
            tests.append({
                "name": name,
                "status": "PASS"
            })
        except Exception as exc:
            tests.append({
                "name": name,
                "status": "FAIL",
                "error": str(exc)
            })

    def test_basic_report():
        report = build_mission_report(
            mission="JWST",
            product_type="SPECTRUM"
        )

        assert report["mission"]["name"] == "JWST"
        assert report["mission"]["product_type"] == "SPECTRUM"
        assert MissionReportBuilder.validate_report(
            report
        )["valid"]

    def test_chandra_metadata():
        report = build_mission_report(
            mission="Chandra",
            observation_metadata={
                "instrument": "ACIS",
                "observation_id": "TEST-001"
            }
        )

        assert report["mission"]["instrument"] == "ACIS"
        assert report["mission"]["observation_id"] == "TEST-001"

    def test_qc_warnings():
        report = build_mission_report(
            qc_report={
                "status": "warning",
                "warnings": [
                    "Low signal-to-noise ratio",
                    {
                        "severity": "medium",
                        "message": "Incomplete metadata"
                    }
                ]
            }
        )

        assert report["quality_control"]["warning_count"] == 2

    def test_calibration_default():
        report = build_mission_report()

        assert report["processing"]["calibration"]["completed"] is False

    def test_visualization_summary():
        report = build_mission_report(
            visualization={
                "generated": True,
                "plot_type": "spectrum",
                "plotted_point_count": 1200,
                "output_format": "PNG"
            }
        )

        assert report["visualization"]["generated"] is True
        assert report["visualization"]["plotted_point_count"] == 1200

    def test_json_export():
        report = build_mission_report()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mission_report.json"

            saved = save_mission_report(
                report,
                str(path)
            )

            loaded = json.loads(
                Path(saved).read_text(encoding="utf-8")
            )

            assert loaded["schema_version"] == SCHEMA_VERSION

    def test_scientific_scope():
        report = build_mission_report()

        scope = report["scientific_scope"]

        assert scope["anomaly_confirmation"] is False
        assert scope["medical_diagnosis"] is False
        assert scope["flight_certification"] is False

    record("basic_report", test_basic_report)
    record("chandra_metadata", test_chandra_metadata)
    record("qc_warnings", test_qc_warnings)
    record("calibration_default", test_calibration_default)
    record("visualization_summary", test_visualization_summary)
    record("json_export", test_json_export)
    record("scientific_scope", test_scientific_scope)

    passed = sum(
        item["status"] == "PASS"
        for item in tests
    )

    failed = len(tests) - passed

    return {
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "total_tests": len(tests),
        "passed": passed,
        "failed": failed,
        "status": "PASS" if failed == 0 else "FAIL",
        "tests": tests
    }


def main():
    """Command-line contract-test entry point."""

    parser = argparse.ArgumentParser(
        description="D³ VITAL-X Mission Report Module"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run contract tests."
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Display module information."
    )

    args = parser.parse_args()

    if args.info:
        print(json.dumps(
            module_info(),
            indent=2,
            ensure_ascii=False
        ))
        return

    if args.test:
        result = run_contract_tests()

        print(json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        ))

        if result["failed"] > 0:
            raise SystemExit(1)

        return

    print(
        "D³ VITAL-X Mission Report Module "
        f"v{MODULE_VERSION}"
    )
    print("Use --test to run contract tests.")
    print("Use --info to display module information.")


if __name__ == "__main__":
    main()
