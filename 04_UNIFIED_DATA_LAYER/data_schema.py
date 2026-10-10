
"""
D³ VITAL-X SPACE INTELLIGENCE PLATFORM
Module: data_schema.py
Version: 0.1.0

Purpose:
    Define a consistent, serializable data-record contract for
    heterogeneous scientific and research inputs.

Architecture:
    Input Adapters -> Unified Data Layer -> Validation / QC
    -> Feature Engine Interface -> Analytics

Safety boundary:
    - No medical diagnosis.
    - No automatic scientific claim verification.
    - No hidden research-core implementation.
    - Raw source data is referenced, not silently modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Mapping
from uuid import uuid4
import json


# ============================================================
# 1. Schema constants
# ============================================================

SCHEMA_NAME = "d3_vital_x.unified_data_record"
SCHEMA_VERSION = "0.1.0"

SUPPORTED_MODALITIES = {
    "tabular",
    "time_series",
    "image",
    "dicom",
    "fits",
    "video",
    "text",
    "binary",
    "unknown",
}

SUPPORTED_DATA_TYPES = {
    "space",
    "biomedical",
    "seismic",
    "environmental",
    "synthetic",
    "general",
    "unknown",
}

SUPPORTED_CLAIM_LEVELS = {
    "C1",  # Directly measured / source-reported
    "C2",  # Computational observation
    "C3",  # Hypothesis or future interpretation
}

MAX_METADATA_DEPTH = 12


# ============================================================
# 2. Utility functions
# ============================================================

def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_json_compatible(value: Any) -> bool:
    """Check whether a value can be serialized to standard JSON."""
    try:
        json.dumps(value, allow_nan=False)
        return True
    except (TypeError, ValueError, OverflowError):
        return False


def _check_nested_depth(
    value: Any,
    depth: int = 0,
    max_depth: int = MAX_METADATA_DEPTH,
) -> bool:
    """Return False if nested metadata exceeds the permitted depth."""
    if depth > max_depth:
        return False

    if isinstance(value, dict):
        return all(
            _check_nested_depth(v, depth + 1, max_depth)
            for v in value.values()
        )

    if isinstance(value, (list, tuple)):
        return all(
            _check_nested_depth(v, depth + 1, max_depth)
            for v in value
        )

    return True


# ============================================================
# 3. Unified record model
# ============================================================

@dataclass
class UnifiedDataRecord:
    """
    Standard record shared across D³ VITAL-X public interfaces.

    payload_ref:
        Reference to a file, URI, dataset row, or other payload.
        The actual large/raw payload is not embedded by default.

    metadata:
        Source-specific, non-sensitive descriptive information.

    provenance:
        Traceability information such as source, processing step,
        software version, and reproducibility references.
    """

    record_id: str = field(default_factory=lambda: str(uuid4()))
    schema_name: str = SCHEMA_NAME
    schema_version: str = SCHEMA_VERSION

    source: str = "unknown"
    modality: str = "unknown"
    data_type: str = "unknown"

    observed_at: Optional[str] = None
    created_at: str = field(default_factory=_utc_now_iso)

    payload_ref: Optional[str] = None

    units: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    quality_flags: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    uncertainty: Dict[str, Any] = field(default_factory=dict)

    # Descriptive labels only; not medical diagnoses.
    labels: List[str] = field(default_factory=list)
    claim_level: str = "C1"

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary representation."""
        return asdict(self)

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize the record to strict JSON."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        validate: bool = True,
    ) -> "UnifiedDataRecord":
        """Build a record from a mapping."""
        if not isinstance(data, Mapping):
            raise TypeError("data must be a mapping/dictionary")

        allowed_fields = set(cls.__dataclass_fields__.keys())
        unknown_fields = set(data.keys()) - allowed_fields

        if unknown_fields:
            raise ValueError(
                "Unknown schema field(s): "
                + ", ".join(sorted(str(x) for x in unknown_fields))
            )

        record = cls(**dict(data))

        if validate:
            report = validate_schema(record)
            if not report["valid"]:
                raise ValueError(
                    "Invalid UnifiedDataRecord: "
                    + "; ".join(report["errors"])
                )

        return record


# ============================================================
# 4. Record factory
# ============================================================

def create_unified_record(
    source: str = "unknown",
    modality: str = "unknown",
    data_type: str = "unknown",
    observed_at: Optional[str] = None,
    payload_ref: Optional[str] = None,
    units: Optional[Dict[str, str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    quality_flags: Optional[List[str]] = None,
    provenance: Optional[Dict[str, Any]] = None,
    uncertainty: Optional[Dict[str, Any]] = None,
    labels: Optional[List[str]] = None,
    claim_level: str = "C1",
    record_id: Optional[str] = None,
) -> UnifiedDataRecord:
    """
    Create a standardized record with independent mutable fields.

    This factory creates a record only; it does not load, modify,
    classify, diagnose, or scientifically interpret the source data.
    """

    record = UnifiedDataRecord(
        record_id=record_id or str(uuid4()),
        source=source,
        modality=modality,
        data_type=data_type,
        observed_at=observed_at,
        payload_ref=payload_ref,
        units=dict(units or {}),
        metadata=dict(metadata or {}),
        quality_flags=list(quality_flags or []),
        provenance=dict(provenance or {}),
        uncertainty=dict(uncertainty or {}),
        labels=list(labels or []),
        claim_level=claim_level,
    )

    report = validate_schema(record)

    if not report["valid"]:
        raise ValueError(
            "Cannot create invalid record: "
            + "; ".join(report["errors"])
        )

    return record


# ============================================================
# 5. Schema validation
# ============================================================

def validate_schema(
    record: Any,
) -> Dict[str, Any]:
    """
    Validate a UnifiedDataRecord or dictionary.

    Returns:
        {
            "valid": bool,
            "errors": list[str],
            "warnings": list[str],
            "schema_version": str
        }

    This is structural validation, not scientific validation.
    """

    errors: List[str] = []
    warnings: List[str] = []

    if isinstance(record, UnifiedDataRecord):
        data = record.to_dict()
    elif isinstance(record, Mapping):
        data = dict(record)
    else:
        return {
            "valid": False,
            "errors": [
                "record must be a UnifiedDataRecord or dictionary"
            ],
            "warnings": [],
            "schema_version": SCHEMA_VERSION,
        }

    required_strings = (
        "record_id",
        "schema_name",
        "schema_version",
        "source",
        "modality",
        "data_type",
        "created_at",
        "claim_level",
    )

    for key in required_strings:
        if not _is_nonempty_string(data.get(key)):
            errors.append(f"{key} must be a non-empty string")

    if data.get("schema_name") != SCHEMA_NAME:
        errors.append(f"schema_name must equal {SCHEMA_NAME!r}")

    if data.get("schema_version") != SCHEMA_VERSION:
        warnings.append(
            "Record schema version differs from this module's version"
        )

    modality = data.get("modality")
    if isinstance(modality, str) and modality not in SUPPORTED_MODALITIES:
        errors.append(f"Unsupported modality: {modality!r}")

    data_type = data.get("data_type")
    if (
        isinstance(data_type, str)
        and data_type not in SUPPORTED_DATA_TYPES
    ):
        errors.append(f"Unsupported data_type: {data_type!r}")

    claim_level = data.get("claim_level")
    if (
        isinstance(claim_level, str)
        and claim_level not in SUPPORTED_CLAIM_LEVELS
    ):
        errors.append(
            f"claim_level must be one of "
            f"{sorted(SUPPORTED_CLAIM_LEVELS)}"
        )

    for key in ("created_at", "observed_at"):
        value = data.get(key)

        if value is None and key == "observed_at":
            continue

        if not _is_nonempty_string(value):
            errors.append(f"{key} must be an ISO-8601 string")
            continue

        try:
            # Accept Z suffix and timezone-aware or naive ISO strings.
            parsed = datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )
            if parsed.tzinfo is None:
                warnings.append(
                    f"{key} has no explicit timezone"
                )
        except ValueError:
            errors.append(f"{key} is not a valid ISO-8601 timestamp")

    payload_ref = data.get("payload_ref")
    if payload_ref is not None and not isinstance(payload_ref, str):
        errors.append("payload_ref must be a string or None")

    for key in (
        "units",
        "metadata",
        "provenance",
        "uncertainty",
    ):
        value = data.get(key)

        if not isinstance(value, dict):
            errors.append(f"{key} must be a dictionary")
            continue

        if not _is_json_compatible(value):
            errors.append(f"{key} must contain JSON-compatible values")

        if not _check_nested_depth(value):
            errors.append(
                f"{key} exceeds maximum nesting depth "
                f"({MAX_METADATA_DEPTH})"
            )

    units = data.get("units")
    if isinstance(units, dict):
        for key, value in units.items():
            if not isinstance(key, str) or not isinstance(value, str):
                errors.append("units keys and values must be strings")
                break

    for key in ("quality_flags", "labels"):
        value = data.get(key)

        if not isinstance(value, list):
            errors.append(f"{key} must be a list")
        elif not all(isinstance(item, str) for item in value):
            errors.append(f"every {key} item must be a string")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "schema_version": SCHEMA_VERSION,
    }


# ============================================================
# 6. Safe JSON round-trip helper
# ============================================================

def record_from_json(json_text: str) -> UnifiedDataRecord:
    """Parse JSON text and validate it before returning a record."""
    if not isinstance(json_text, str):
        raise TypeError("json_text must be a string")

    parsed = json.loads(json_text)

    if not isinstance(parsed, dict):
        raise ValueError("JSON root must be an object/dictionary")

    return UnifiedDataRecord.from_dict(parsed, validate=True)


# ============================================================
# 7. Module self-check
# ============================================================

def self_check() -> Dict[str, Any]:
    """Run a small deterministic structural smoke test."""
    sample = create_unified_record(
        source="self_check",
        modality="time_series",
        data_type="space",
        observed_at="2026-01-01T00:00:00Z",
        payload_ref="example://synthetic-signal",
        units={"time": "s", "signal": "arbitrary_unit"},
        metadata={"purpose": "schema smoke test"},
        provenance={"module": "data_schema.py"},
        claim_level="C2",
    )

    first_report = validate_schema(sample)
    restored = record_from_json(sample.to_json())
    second_report = validate_schema(restored)

    return {
        "valid": (
            first_report["valid"]
            and second_report["valid"]
        ),
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "round_trip_preserved_id": (
            sample.record_id == restored.record_id
        ),
        "errors": (
            first_report["errors"] + second_report["errors"]
        ),
        "warnings": (
            first_report["warnings"] + second_report["warnings"]
        ),
    }


if __name__ == "__main__":
    result = self_check()
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if not result["valid"] or not result["round_trip_preserved_id"]:
        raise SystemExit("Schema self-check failed.")

    print("\nSchema self-check passed.")
