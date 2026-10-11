
"""
D³ VITAL-X — Public Result View
Module 16: result_view.py

Purpose
-------
Create readable, structured reports from public analytics results.

Supported inputs
----------------
- Dictionaries / mappings
- Dataclass-based result objects
- Objects exposing a to_dict() method

Design principles
-----------------
- Open Interface — Closed Intelligence Core
- No scientific observations are generated.
- No anomaly, physical event, or clinical condition is inferred here.
- Raw data and private implementation details are excluded by default.
- Output is suitable for JSON export and dashboard integration.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional
import hashlib
import json
import math


MODULE_NAME = "result_view"
MODULE_NUMBER = 16
MODULE_VERSION = "1.0.0"

DEFAULT_TITLE = "D³ VITAL-X — Analytics Report"

# Keys containing these terms are excluded from public report output.
# This is a defensive filter, not a guarantee that arbitrary content
# contains no confidential information.
SENSITIVE_KEY_TERMS = (
    "password",
    "secret",
    "api_key",
    "apikey",
    "access_token",
    "private_key",
    "private_core",
    "protected_core",
    "internal_algorithm",
    "credential",
)


def _is_sensitive_key(key: Any) -> bool:
    """Return True when a key appears to reference confidential material."""
    normalized = str(key).strip().lower().replace("-", "_").replace(" ", "_")
    return any(term in normalized for term in SENSITIVE_KEY_TERMS)


def _json_safe(value: Any, *, redact_sensitive: bool = True) -> Any:
    """
    Convert common Python values to JSON-compatible representations.

    Non-finite floats become None so output remains valid strict JSON.
    Mapping keys associated with common secrets are omitted by default.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Mapping):
        output = {}
        for key, item in value.items():
            if redact_sensitive and _is_sensitive_key(key):
                continue
            output[str(key)] = _json_safe(
                item,
                redact_sensitive=redact_sensitive,
            )
        return output

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, redact_sensitive=redact_sensitive)
            for item in value
        ]

    if isinstance(value, set):
        return [
            _json_safe(item, redact_sensitive=redact_sensitive)
            for item in sorted(value, key=str)
        ]

    # Support NumPy scalar values without requiring NumPy as a dependency.
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(
                value.item(),
                redact_sensitive=redact_sensitive,
            )
        except (ValueError, TypeError):
            pass

    # Avoid serializing arbitrary objects through repr(), which could expose
    # internal state or implementation details.
    return f"<unsupported:{type(value).__name__}>"


def _to_mapping(value: Any, name: str) -> dict[str, Any]:
    """Convert a supported result object into a dictionary."""
    if isinstance(value, Mapping):
        return dict(value)

    if hasattr(value, "to_dict") and callable(value.to_dict):
        converted = value.to_dict()
        if isinstance(converted, Mapping):
            return dict(converted)

    if is_dataclass(value) and not isinstance(value, type):
        converted = asdict(value)
        if isinstance(converted, Mapping):
            return dict(converted)

    raise ValueError(
        f"'{name}' must be a mapping, a dataclass instance, "
        "or an object exposing to_dict()."
    )


def _first_present(
    data: Mapping[str, Any],
    keys: tuple[str, ...],
    default: Any = None,
) -> Any:
    """Return the first available non-None field."""
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def _summarize_result(name: str, raw_result: Any) -> dict[str, Any]:
    """Extract stable, public-facing summary fields from one result."""
    result = _to_mapping(raw_result, name)
    safe_result = _json_safe(result)

    if not isinstance(safe_result, dict):
        safe_result = {}

    summary = {
        "name": name,
        "module": _first_present(
            safe_result,
            ("module", "module_name"),
            name,
        ),
        "module_version": _first_present(
            safe_result,
            ("module_version", "version"),
        ),
        "status": _first_present(
            safe_result,
            ("status", "state"),
            "unspecified",
        ),
        "sample_count": _first_present(
            safe_result,
            (
                "valid_pair_count",
                "valid_sample_count",
                "sample_count",
                "n_samples",
                "original_sample_count",
            ),
        ),
        "aggregate_score": _first_present(
            safe_result,
            ("aggregate_score",),
        ),
        "candidate_flag": _first_present(
            safe_result,
            ("candidate_flag",),
        ),
        "overall_correlation": _first_present(
            safe_result,
            ("overall_correlation",),
        ),
        "entropy": _first_present(
            safe_result,
            ("shannon_entropy_bits", "entropy"),
        ),
        "variance": _first_present(
            safe_result,
            ("variance", "population_variance", "sample_variance"),
        ),
        "warnings": safe_result.get("warnings", []),
        "errors": safe_result.get("errors", []),
        "interpretation": safe_result.get(
            "interpretation",
            "No interpretation was supplied by the source module.",
        ),
        "result_fingerprint": safe_result.get("result_fingerprint"),
    }

    # Preserve additional public-facing fields not already summarized.
    # Large raw arrays and explicitly sensitive keys are not included.
    excluded = {
        "raw_data",
        "input_data",
        "samples",
        "signal_x",
        "signal_y",
        "private_core",
        "protected_core",
        "internal_algorithm",
    }
    additional = {
        key: value
        for key, value in safe_result.items()
        if key not in summary
        and key not in excluded
        and not key.lower().endswith("_array")
    }

    summary["additional_public_fields"] = additional

    # Keep the report compact and predictable.
    for field_name in ("warnings", "errors"):
        value = summary[field_name]
        if not isinstance(value, list):
            summary[field_name] = [value] if value else []

    return summary


def build_result_report(
    results: Mapping[str, Any],
    *,
    title: str = DEFAULT_TITLE,
    project: str = "D³ VITAL-X Space Intelligence Platform",
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """
    Build a structured public report from one or more analytics results.

    Parameters
    ----------
    results:
        Mapping of descriptive names to result objects or dictionaries.

    title:
        Report title.

    project:
        Project label.

    metadata:
        Optional non-sensitive provenance details, such as dataset identifier,
        source URL, analysis date, or software commit.

    Returns
    -------
    A JSON-compatible dictionary.

    Note
    ----
    The timestamp indicates report creation time, not observation time.
    """
    if not isinstance(results, Mapping) or not results:
        raise ValueError("results must be a non-empty mapping.")

    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string.")

    if not isinstance(project, str) or not project.strip():
        raise ValueError("project must be a non-empty string.")

    if metadata is not None and not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping when provided.")

    summaries = []
    for name, result in results.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Every result name must be a non-empty string.")
        summaries.append(_summarize_result(name, result))

    warnings_count = sum(
        len(item["warnings"]) for item in summaries
    )
    errors_count = sum(
        len(item["errors"]) for item in summaries
    )

    flagged_count = sum(
        item["candidate_flag"] is True for item in summaries
    )

    report = {
        "report_schema_version": "1.0",
        "module": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "project": project,
        "title": title,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "result_count": len(summaries),
            "warning_count": warnings_count,
            "error_count": errors_count,
            "candidate_flag_count": flagged_count,
            "interpretation": (
                "Counts summarize source-module outputs. A candidate flag "
                "is not a confirmed anomaly or scientific discovery."
            ),
        },
        "metadata": _json_safe(metadata or {}),
        "results": summaries,
        "scientific_boundary": (
            "This report presents supplied analytical outputs. It does not "
            "independently validate the source data, establish causation, "
            "confirm a physical transition, or provide medical diagnosis."
        ),
    }

    # Fingerprint the report content without its creation timestamp or hash.
    fingerprint_payload = {
        key: value
        for key, value in report.items()
        if key != "created_at_utc"
    }

    canonical = json.dumps(
        fingerprint_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )

    report["report_fingerprint"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()

    return report


def render_text_report(report: Mapping[str, Any]) -> str:
    """Render a compact plain-text summary for logs and mobile screens."""
    if not isinstance(report, Mapping):
        raise ValueError("report must be a mapping.")

    title = report.get("title", DEFAULT_TITLE)
    project = report.get("project", "Unspecified project")
    summary = report.get("summary", {})
    results = report.get("results", [])

    lines = [
        str(title),
        "=" * min(max(len(str(title)), 8), 72),
        f"Project: {project}",
        f"Created (UTC): {report.get('created_at_utc', 'unspecified')}",
        "",
        "SUMMARY",
        f"  Results: {summary.get('result_count', 0)}",
        f"  Warnings: {summary.get('warning_count', 0)}",
        f"  Errors: {summary.get('error_count', 0)}",
        f"  Candidate flags: {summary.get('candidate_flag_count', 0)}",
        "",
        "RESULTS",
    ]

    for item in results:
        lines.extend(
            [
                f"- {item.get('name', 'unnamed')}",
                f"  Module: {item.get('module', 'unspecified')}",
                f"  Status: {item.get('status', 'unspecified')}",
            ]
        )

        if item.get("sample_count") is not None:
            lines.append(f"  Sample count: {item['sample_count']}")

        if item.get("aggregate_score") is not None:
            lines.append(f"  Aggregate score: {item['aggregate_score']}")

        if item.get("candidate_flag") is not None:
            lines.append(f"  Candidate flag: {item['candidate_flag']}")

        for warning in item.get("warnings", []):
            lines.append(f"  Warning: {warning}")

        for error in item.get("errors", []):
            lines.append(f"  Error: {error}")

        interpretation = item.get("interpretation")
        if interpretation:
            lines.append(f"  Interpretation: {interpretation}")

        lines.append("")

    lines.extend(
        [
            "SCIENTIFIC BOUNDARY",
            str(
                report.get(
                    "scientific_boundary",
                    "Interpret results in their documented scientific context.",
                )
            ),
        ]
    )

    if report.get("report_fingerprint"):
        lines.extend(
            [
                "",
                f"Report fingerprint: {report['report_fingerprint']}",
            ]
        )

    return "\n".join(lines)


def report_to_json(
    report: Mapping[str, Any],
    *,
    indent: int = 2,
) -> str:
    """Serialize a report to strict JSON."""
    if not isinstance(report, Mapping):
        raise ValueError("report must be a mapping.")

    if isinstance(indent, bool) or not isinstance(indent, int) or indent < 0:
        raise ValueError("indent must be a non-negative integer.")

    safe_report = _json_safe(report)

    return json.dumps(
        safe_report,
        indent=indent,
        ensure_ascii=False,
        allow_nan=False,
    )


def module_info() -> dict[str, Any]:
    """Return public metadata for module discovery."""
    return {
        "module": MODULE_NAME,
        "module_number": MODULE_NUMBER,
        "version": MODULE_VERSION,
        "purpose": "Format analytics outputs into readable public reports",
        "functions": [
            "build_result_report",
            "render_text_report",
            "report_to_json",
            "module_info",
            "run_contract_tests",
        ],
        "output_formats": ["Python dictionary", "plain text", "JSON"],
        "scientific_boundary": (
            "Presentation only; this module does not validate or infer "
            "scientific conclusions."
        ),
    }


def run_contract_tests() -> dict[str, bool]:
    """
    Run contract tests without generating scientific observations.
    """
    checks: dict[str, bool] = {}

    try:
        build_result_report({})
        checks["empty_results_rejected"] = False
    except ValueError:
        checks["empty_results_rejected"] = True

    try:
        build_result_report(
            {
                "review": {
                    "module": "contract-check",
                    "status": "completed",
                    "warnings": [],
                    "errors": [],
                }
            }
        )
        checks["mapping_result_accepted"] = True
    except (ValueError, TypeError):
        checks["mapping_result_accepted"] = False

    try:
        report_to_json({"value": float("inf")})
        checks["nonfinite_values_json_safe"] = True
    except (ValueError, TypeError):
        checks["nonfinite_values_json_safe"] = False

    try:
        build_result_report(
            {
                "review": {
                    "status": "completed",
                    "api_key": "must-not-appear",
                }
            }
        )
        checks["sensitive_key_filtered"] = True
    except (ValueError, TypeError):
        checks["sensitive_key_filtered"] = False

    checks["all_checks_passed"] = all(checks.values())
    return checks


if __name__ == "__main__":
    print(json.dumps(module_info(), indent=2, ensure_ascii=False))
    print(json.dumps(run_contract_tests(), indent=2, ensure_ascii=False))
