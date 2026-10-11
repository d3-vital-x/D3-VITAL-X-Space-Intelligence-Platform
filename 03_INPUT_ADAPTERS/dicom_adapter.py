"""
D³ VITAL-X Space Intelligence Platform
Module 08: dicom_adapter.py
Version: 0.2.0

Biomedical Research Input Adapter.

Responsibilities:
- Technical DICOM inspection
- SHA-256 integrity tracking
- Controlled, privacy-conscious metadata extraction
- Basic structural validation
- Optional in-memory pixel loading
- Integration with the current UnifiedDataRecord schema
- Provenance and reproducibility metadata

Boundaries:
- The source DICOM file is never modified by this adapter.
- Pixel arrays are not embedded in UnifiedDataRecord.
- Patient identifiers are not copied into extracted metadata.
- Metadata filtering does not constitute full anonymization.
- No medical diagnosis or clinical interpretation is performed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Union


MODULE_NAME = "dicom_adapter"
MODULE_VERSION = "0.2.0"
SCHEMA_MODULE_NAME = "d3_vital_x_data_schema"

SourceType = Union[str, Path, bytes, bytearray]


# ---------------------------------------------------------
# 1. Time and integrity utilities
# ---------------------------------------------------------

def utc_timestamp() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def calculate_sha256(source: SourceType) -> str:
    """Calculate the SHA-256 hash of a file or raw bytes."""
    digest = hashlib.sha256()

    if isinstance(source, (bytes, bytearray)):
        digest.update(bytes(source))
        return digest.hexdigest()

    path = Path(source)

    if not path.is_file():
        raise FileNotFoundError(
            f"DICOM source is not a regular file: {path}"
        )

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def generate_dataset_id(
    raw_hash: str,
    prefix: str = "DICOM",
) -> str:
    """Generate an identifier from the content hash."""
    if not raw_hash or len(raw_hash) != 64:
        raise ValueError("A valid SHA-256 hash is required.")

    return f"{prefix}-{raw_hash[:16]}"


# ---------------------------------------------------------
# 2. Resolve and load the current unified schema
# ---------------------------------------------------------

def resolve_schema_path() -> Path:
    """Locate the current project data_schema.py."""
    if "__file__" in globals():
        current_file = Path(__file__).resolve()

        for parent in current_file.parents:
            candidate = (
                parent / "04_UNIFIED_DATA_LAYER" / "data_schema.py"
            )
            if candidate.is_file():
                return candidate

    fallback = (
        Path("/content/D3-VITAL-X-Space-Intelligence-Platform")
        / "04_UNIFIED_DATA_LAYER"
        / "data_schema.py"
    )

    if fallback.is_file():
        return fallback

    raise FileNotFoundError(
        "Unified schema not found. Expected "
        "04_UNIFIED_DATA_LAYER/data_schema.py."
    )


def load_schema_module():
    """Load the public schema and register it before execution."""
    schema_path = resolve_schema_path()

    existing = sys.modules.get(SCHEMA_MODULE_NAME)

    if existing is not None:
        existing_file = getattr(existing, "__file__", None)

        if (
            existing_file
            and Path(existing_file).resolve() == schema_path.resolve()
        ):
            return existing

    spec = importlib.util.spec_from_file_location(
        SCHEMA_MODULE_NAME,
        str(schema_path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Unable to load schema from {schema_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[SCHEMA_MODULE_NAME] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(SCHEMA_MODULE_NAME, None)
        raise

    if not hasattr(module, "UnifiedDataRecord"):
        raise AttributeError(
            "data_schema.py must define UnifiedDataRecord."
        )

    return module


# ---------------------------------------------------------
# 3. Optional dependency
# ---------------------------------------------------------

def ensure_pydicom():
    """Import pydicom or provide an actionable installation hint."""
    try:
        import pydicom
        return pydicom
    except ImportError as exc:
        raise ImportError(
            "pydicom is required. In Google Colab run: "
            "%pip install pydicom"
        ) from exc


# ---------------------------------------------------------
# 4. Source handling
# ---------------------------------------------------------

def _read_dataset(
    source: SourceType,
    stop_before_pixels: bool = True,
    force: bool = False,
):
    """
    Read a DICOM dataset.

    For path inputs, pydicom reads directly from the file.
    For bytes inputs, a BytesIO wrapper is used.
    """
    pydicom = ensure_pydicom()

    if isinstance(source, (str, Path)):
        path = Path(source)

        if not path.is_file():
            raise FileNotFoundError(
                f"DICOM file not found: {path}"
            )

        dataset = pydicom.dcmread(
            str(path),
            stop_before_pixels=stop_before_pixels,
            force=force,
        )

        return dataset

    if isinstance(source, (bytes, bytearray)):
        if not source:
            raise ValueError("DICOM byte source is empty.")

        return pydicom.dcmread(
            BytesIO(bytes(source)),
            stop_before_pixels=stop_before_pixels,
            force=force,
        )

    raise TypeError(
        "source must be a path, bytes, or bytearray."
    )


def _source_size(source: SourceType) -> int:
    """Return the byte size without parsing image pixels."""
    if isinstance(source, (bytes, bytearray)):
        return len(source)

    path = Path(source)

    if not path.is_file():
        raise FileNotFoundError(path)

    return path.stat().st_size


# ---------------------------------------------------------
# 5. Privacy-conscious technical metadata
# ---------------------------------------------------------

# Deliberately limited allowlist.
# Patient names, IDs, dates of birth, addresses, free-text descriptions,
# institution details and identifying UIDs are not copied into output.
SAFE_DICOM_FIELDS = {
    "Modality": "Modality",
    "Rows": "Rows",
    "Columns": "Columns",
    "SamplesPerPixel": "SamplesPerPixel",
    "PhotometricInterpretation": "PhotometricInterpretation",
    "BitsAllocated": "BitsAllocated",
    "BitsStored": "BitsStored",
    "HighBit": "HighBit",
    "PixelRepresentation": "PixelRepresentation",
    "NumberOfFrames": "NumberOfFrames",
    "ImageType": "ImageType",
}

EXCLUDED_DICOM_FIELDS = {
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientBirthTime",
    "PatientSex",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "OtherPatientIDs",
    "OtherPatientNames",
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "StudyDescription",
    "SeriesDescription",
    "AccessionNumber",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
}


def _json_safe_value(value: Any) -> Any:
    """Convert allowlisted values to JSON-friendly primitives."""
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]

    return str(value)


def extract_safe_metadata(dataset) -> Dict[str, Any]:
    """
    Extract a controlled subset of technical DICOM metadata.

    This is an output allowlist, not a claim that the source file
    or its pixels have been fully anonymized.
    """
    result = {}

    for output_name, dicom_name in SAFE_DICOM_FIELDS.items():
        if dicom_name in EXCLUDED_DICOM_FIELDS:
            continue

        try:
            if hasattr(dataset, dicom_name):
                result[output_name] = _json_safe_value(
                    getattr(dataset, dicom_name)
                )
        except Exception:
            # An unreadable optional field is omitted.
            continue

    return result


# ---------------------------------------------------------
# 6. DICOM inspection
# ---------------------------------------------------------

def inspect_dicom(
    source: SourceType,
    stop_before_pixels: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    """Inspect DICOM technical metadata and source-file integrity."""
    dataset = _read_dataset(
        source,
        stop_before_pixels=stop_before_pixels,
        force=force,
    )

    raw_hash = calculate_sha256(source)
    safe_metadata = extract_safe_metadata(dataset)

    return {
        "valid_dicom_read": True,
        "dataset_id": generate_dataset_id(raw_hash),
        "raw_sha256": raw_hash,
        "file_size_bytes": _source_size(source),
        "safe_metadata": safe_metadata,
        "pixel_data_requested": not stop_before_pixels,
        "force_read": force,
        "metadata_policy": "allowlist",
        "full_anonymization_verified": False,
        "inspected_at_utc": utc_timestamp(),
    }


# ---------------------------------------------------------
# 7. Structural validation
# ---------------------------------------------------------

def validate_dicom_structure(
    source: SourceType,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Perform basic technical DICOM checks.

    These checks do not establish diagnostic quality,
    scientific validity, or clinical suitability.
    """
    try:
        dataset = _read_dataset(
            source,
            stop_before_pixels=True,
            force=force,
        )

        has_sop_class = hasattr(dataset, "SOPClassUID")
        has_sop_instance = hasattr(dataset, "SOPInstanceUID")
        has_modality = hasattr(dataset, "Modality")

        warnings = []

        if not has_sop_class:
            warnings.append("SOPClassUID is missing.")

        if not has_sop_instance:
            warnings.append("SOPInstanceUID is missing.")

        if not has_modality:
            warnings.append("Modality is missing.")

        return {
            "valid": True,
            "readable": True,
            "has_sop_class_uid": has_sop_class,
            "has_sop_instance_uid": has_sop_instance,
            "has_modality": has_modality,
            "file_size_bytes": _source_size(source),
            "sha256": calculate_sha256(source),
            "warnings": warnings,
            "check_type": "technical_structure_only",
            "checked_at_utc": utc_timestamp(),
        }

    except Exception as exc:
        return {
            "valid": False,
            "readable": False,
            "warnings": [],
            "error_type": type(exc).__name__,
            "message": str(exc),
            "check_type": "technical_structure_only",
            "checked_at_utc": utc_timestamp(),
        }


# ---------------------------------------------------------
# 8. Optional in-memory pixel loading
# ---------------------------------------------------------

def load_dicom_pixels(
    source: SourceType,
    force: bool = False,
):
    """
    Decode the DICOM pixel array in memory.

    The original source file is not re-saved or modified.
    Decoding may require additional transfer-syntax dependencies.
    No normalization, smoothing, clipping or interpolation is applied.
    """
    dataset = _read_dataset(
        source,
        stop_before_pixels=False,
        force=force,
    )

    if "PixelData" not in dataset:
        raise ValueError(
            "DICOM dataset does not contain PixelData."
        )

    try:
        return dataset.pixel_array
    except Exception as exc:
        raise RuntimeError(
            "Pixel decoding failed. The file may require an additional "
            "pydicom pixel-data decoder or supported transfer syntax."
        ) from exc


# ---------------------------------------------------------
# 9. Convert to current UnifiedDataRecord
# ---------------------------------------------------------

def dicom_to_unified_record(
    source: SourceType,
    source_name: Optional[str] = None,
    source_uri: Optional[str] = None,
    anonymized: Optional[bool] = None,
    description: str = (
        "DICOM input for technical biomedical data inspection."
    ),
    license_info: str = "Not specified",
    extra_metadata: Optional[Dict[str, Any]] = None,
    force: bool = False,
):
    """
    Convert a DICOM file into the current UnifiedDataRecord.

    The record references the source through payload_ref.
    Pixel arrays are intentionally not embedded in the record.
    """
    schema = load_schema_module()

    dataset = _read_dataset(
        source,
        stop_before_pixels=True,
        force=force,
    )

    raw_hash = calculate_sha256(source)
    dataset_id = generate_dataset_id(raw_hash)
    safe_metadata = extract_safe_metadata(dataset)

    if source_name is None:
        if isinstance(source, (str, Path)):
            source_name = Path(source).name
        else:
            source_name = "dicom_input"

    if isinstance(source, (str, Path)):
        payload_ref = str(Path(source).resolve())
    else:
        # Bytes have no persistent file location.
        payload_ref = None

    rows = safe_metadata.get("Rows")
    columns = safe_metadata.get("Columns")

    dimensions = None
    if isinstance(rows, int) and isinstance(columns, int):
        dimensions = [rows, columns]

    structure_report = validate_dicom_structure(
        source,
        force=force,
    )

    metadata = {
        "adapter": MODULE_NAME,
        "adapter_version": MODULE_VERSION,
        "dataset_id": dataset_id,
        "source_name": source_name,
        "source_type": "DICOM",
        "description": description,
        "license_info": license_info,
        "anonymized_declared_by_caller": anonymized,
        "safe_dicom_metadata": safe_metadata,
        "dimensions": dimensions,
        "file_size_bytes": _source_size(source),
        "raw_sha256": raw_hash,
        "structural_validation": structure_report,
        "pixel_array_embedded": False,
        "scientific_processing_performed": False,
        "clinical_interpretation_performed": False,
        "full_anonymization_verified": False,
        "created_at": utc_timestamp(),
    }

    if extra_metadata is not None:
        if not isinstance(extra_metadata, dict):
            raise TypeError("extra_metadata must be a dictionary.")

        # Prevent caller metadata from overwriting core adapter fields.
        for key, value in extra_metadata.items():
            if key not in metadata:
                metadata[key] = value

    quality_flags = []

    if not structure_report["valid"]:
        quality_flags.append("dicom_structure_check_failed")

    if structure_report.get("warnings"):
        quality_flags.append("dicom_structure_warning")

    if payload_ref is None:
        quality_flags.append("source_bytes_not_persisted_by_adapter")

    RecordClass = schema.UnifiedDataRecord

    return RecordClass(
        record_id=dataset_id,
        schema_name=getattr(
            schema,
            "SCHEMA_NAME",
            "d3_vital_x.unified_data_record",
        ),
        schema_version=getattr(
            schema,
            "SCHEMA_VERSION",
            "0.1.0",
        ),
        source=str(source_name),
        modality="dicom",
        data_type="biomedical",
        observed_at=None,
        created_at=utc_timestamp(),
        payload_ref=payload_ref,
        units={},
        metadata=metadata,
        quality_flags=quality_flags,
        provenance={
            "adapter": MODULE_NAME,
            "adapter_version": MODULE_VERSION,
            "source_uri": source_uri,
            "source_file": payload_ref,
            "raw_sha256": raw_hash,
            "raw_data_modified": False,
            "raw_data_preserved": True,
            "safe_metadata_only": True,
        },
        uncertainty={},
        labels=[],
        claim_level="C1",
    )


# ---------------------------------------------------------
# 10. Safe record summary
# ---------------------------------------------------------

def summarize_dicom_record(record) -> Dict[str, Any]:
    """Return a JSON-friendly record summary without patient tags."""
    schema = load_schema_module()

    if not isinstance(record, schema.UnifiedDataRecord):
        raise TypeError(
            "record must be an instance of the current "
            "UnifiedDataRecord schema."
        )

    metadata = record.metadata or {}
    provenance = record.provenance or {}

    return {
        "record_id": record.record_id,
        "schema_name": record.schema_name,
        "schema_version": record.schema_version,
        "source": record.source,
        "modality": record.modality,
        "data_type": record.data_type,
        "payload_ref": record.payload_ref,
        "dimensions": metadata.get("dimensions"),
        "raw_sha256": provenance.get("raw_sha256"),
        "quality_flags": list(record.quality_flags or []),
        "claim_level": record.claim_level,
        "full_anonymization_verified": False,
        "generated_at": utc_timestamp(),
    }


def safe_metadata_json(
    source: SourceType,
    indent: int = 2,
) -> str:
    """Serialize a controlled DICOM inspection report as JSON."""
    report = inspect_dicom(
        source,
        stop_before_pixels=True,
    )

    return json.dumps(
        report,
        indent=indent,
        ensure_ascii=False,
        default=str,
    )


# ---------------------------------------------------------
# 11. Synthetic functional test
# ---------------------------------------------------------

def run_dicom_adapter_test() -> Dict[str, Any]:
    """
    Run a local synthetic DICOM test.

    Requires pydicom and NumPy. No external data or network is used.
    """
    import numpy as np
    import pydicom

    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import (
        ExplicitVRLittleEndian,
        SecondaryCaptureImageStorage,
        generate_uid,
    )

    # Synthetic test pixels; not a real patient image.
    expected_pixels = np.array(
        [[0, 100, 200], [300, 400, 500]],
        dtype=np.uint16,
    )

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dataset = FileDataset(
        None,
        {},
        file_meta=file_meta,
        preamble=b"\x00" * 128,
    )

    dataset.is_little_endian = True
    dataset.is_implicit_VR = False
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "OT"
    dataset.Rows = 2
    dataset.Columns = 3
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = expected_pixels.tobytes()

    buffer = BytesIO()
    dataset.save_as(buffer, enforce_file_format=True)
    raw_bytes = buffer.getvalue()

    original_hash = calculate_sha256(raw_bytes)

    inspection = inspect_dicom(raw_bytes)
    assert inspection["valid_dicom_read"] is True
    assert inspection["raw_sha256"] == original_hash
    assert inspection["safe_metadata"]["Rows"] == 2
    assert inspection["safe_metadata"]["Columns"] == 3

    validation = validate_dicom_structure(raw_bytes)
    assert validation["valid"] is True
    assert validation["readable"] is True

    decoded_pixels = load_dicom_pixels(raw_bytes)
    assert decoded_pixels.shape == (2, 3)
    assert np.array_equal(decoded_pixels, expected_pixels)

    record = dicom_to_unified_record(
        raw_bytes,
        source_name="synthetic_test.dcm",
        anonymized=True,
        description="Synthetic adapter test only.",
        license_info="Synthetic test fixture",
    )

    schema = load_schema_module()

    assert isinstance(record, schema.UnifiedDataRecord)
    assert record.modality == "dicom"
    assert record.data_type == "biomedical"
    assert record.payload_ref is None
    assert record.provenance["raw_sha256"] == original_hash
    assert record.metadata["pixel_array_embedded"] is False

    forbidden = {
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "PatientAddress",
        "PatientTelephoneNumbers",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
    }

    assert forbidden.isdisjoint(
        record.metadata["safe_dicom_metadata"].keys()
    )

    summary = summarize_dicom_record(record)

    return {
        "passed": True,
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "network_used": False,
        "synthetic_fixture_only": True,
        "source_hash_unchanged": (
            calculate_sha256(raw_bytes) == original_hash
        ),
        "decoded_pixel_shape": list(decoded_pixels.shape),
        "record_summary": summary,
    }


def module_info() -> Dict[str, Any]:
    """Return public adapter capabilities and boundaries."""
    return {
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "purpose": "Technical DICOM ingestion and inspection",
        "pixel_loading": "Optional; separate in-memory operation",
        "source_file_modified": False,
        "patient_identifying_tags_exported": False,
        "full_anonymization_guaranteed": False,
        "clinical_diagnosis": False,
        "scientific_interpretation": False,
        "proprietary_core_included": False,
    }


if __name__ == "__main__":
    print(json.dumps(
        run_dicom_adapter_test(),
        indent=2,
        ensure_ascii=False,
        default=str,
    ))
