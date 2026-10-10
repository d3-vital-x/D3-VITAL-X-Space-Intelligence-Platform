"""
D³ VITAL-X Image Input Adapter
Version: 0.2.0

Purpose:
- Inspect image-file technical metadata.
- Calculate SHA-256 for integrity and provenance tracking.
- Perform basic image-structure checks.
- Create a UnifiedDataRecord using the current data_schema.py.
- Optionally load image pixels into memory through a separate function.

Safety and scientific boundaries:
- The original image file is never modified.
- No pixel transformations are performed by this adapter.
- No medical diagnosis or scientific interpretation is performed.
- Pixel arrays are not embedded in the UnifiedDataRecord.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


MODULE_VERSION = "0.2.0"
SCHEMA_MODULE_NAME = "d3_vital_x_data_schema"

try:
    from PIL import Image

    PILLOW_AVAILABLE = True
except ImportError:
    Image = None
    PILLOW_AVAILABLE = False


# ---------------------------------------------------------
# 1. General utilities
# ---------------------------------------------------------

def utc_timestamp() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def ensure_image_dependency() -> None:
    """Raise an actionable error if Pillow is unavailable."""
    if not PILLOW_AVAILABLE:
        raise ImportError(
            "Pillow is required for image inspection. "
            "Install it using: %pip install Pillow"
        )


def calculate_sha256(file_path: str | Path) -> str:
    """Calculate the SHA-256 hash without modifying the source file."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a regular file: {path}")

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def generate_dataset_id(file_path: str | Path, raw_hash: str) -> str:
    """Generate a deterministic identifier from the filename and file hash."""
    path = Path(file_path)
    source = f"{path.name}|{raw_hash}"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

    return f"image_{digest}"


def resolve_schema_path() -> Path:
    """Find the project's current UnifiedDataRecord schema."""
    current_file = Path(__file__).resolve() if "__file__" in globals() else None

    candidates = []

    if current_file is not None:
        candidates.extend(
            parent / "04_UNIFIED_DATA_LAYER" / "data_schema.py"
            for parent in current_file.parents
        )

    candidates.append(
        Path("/content/D3-VITAL-X-Space-Intelligence-Platform")
        / "04_UNIFIED_DATA_LAYER"
        / "data_schema.py"
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Current data_schema.py was not found. Expected location: "
        "/content/D3-VITAL-X-Space-Intelligence-Platform/"
        "04_UNIFIED_DATA_LAYER/data_schema.py"
    )


def load_schema_module():
    """Load and register the current schema module safely."""
    schema_path = resolve_schema_path()

    existing = sys.modules.get(SCHEMA_MODULE_NAME)

    if existing is not None:
        existing_file = getattr(existing, "__file__", None)

        if existing_file and Path(existing_file).resolve() == schema_path.resolve():
            return existing

    spec = importlib.util.spec_from_file_location(
        SCHEMA_MODULE_NAME,
        str(schema_path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load schema module: {schema_path}")

    module = importlib.util.module_from_spec(spec)

    # Register before execution to support dataclasses and annotations.
    sys.modules[SCHEMA_MODULE_NAME] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(SCHEMA_MODULE_NAME, None)
        raise

    return module


# ---------------------------------------------------------
# 2. Image inspection
# ---------------------------------------------------------

def inspect_image(file_path: str | Path) -> Dict[str, Any]:
    """
    Inspect image metadata and return a technical report.

    This function does not intentionally alter or re-save the image.
    """
    ensure_image_dependency()

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a regular file: {path}")

    raw_hash_before = calculate_sha256(path)
    file_size = path.stat().st_size

    with Image.open(path) as image:
        image_format = image.format
        image_mode = image.mode
        bands = list(image.getbands())
        width, height = image.size

        # Read technical metadata while the image is open.
        exif_data = image.getexif()
        has_exif = bool(exif_data)

        is_animated = bool(getattr(image, "is_animated", False))
        frame_count = int(getattr(image, "n_frames", 1))

        metadata = {
            "file_name": path.name,
            "file_path": str(path.resolve()),
            "file_extension": path.suffix.lower(),
            "format": image_format,
            "mode": image_mode,
            "bands": bands,
            "channel_count": len(bands),
            "width": int(width),
            "height": int(height),
            "dimensions": [int(height), int(width)],
            "pixel_count": int(width) * int(height),
            "file_size_bytes": int(file_size),
            "raw_sha256": raw_hash_before,
            "has_exif": has_exif,
            "is_animated": is_animated,
            "frame_count": frame_count,
            "inspected_at": utc_timestamp(),
        }

    return metadata


# ---------------------------------------------------------
# 3. Optional pixel loading
# ---------------------------------------------------------

def load_image_array(
    file_path: str | Path,
    preserve_mode: bool = True,
):
    """
    Load pixels into memory as a NumPy array.

    The source image file is not re-saved or modified.
    If preserve_mode=False, convert the in-memory representation to RGB.
    """
    ensure_image_dependency()

    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "NumPy is required to load image arrays. "
            "Install it using: %pip install numpy"
        ) from exc

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    with Image.open(path) as image:
        if preserve_mode:
            return np.array(image)

        return np.array(image.convert("RGB"))


# ---------------------------------------------------------
# 4. Structural validation
# ---------------------------------------------------------

def validate_image_structure(
    image_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Check basic technical properties.

    This is structural QC, not scientific or clinical validation.
    """
    errors = []
    warnings = []

    width = image_metadata.get("width")
    height = image_metadata.get("height")
    channel_count = image_metadata.get("channel_count")
    frame_count = image_metadata.get("frame_count", 1)

    if not isinstance(width, int) or width <= 0:
        errors.append("Image width is missing or invalid.")

    if not isinstance(height, int) or height <= 0:
        errors.append("Image height is missing or invalid.")

    if not isinstance(channel_count, int) or channel_count <= 0:
        errors.append("Channel count is missing or invalid.")

    if image_metadata.get("is_animated"):
        warnings.append("Animated or multi-frame image detected.")

    if image_metadata.get("has_exif"):
        warnings.append(
            "EXIF metadata detected; inspect for sensitive metadata if needed."
        )

    if isinstance(frame_count, int) and frame_count > 1:
        warnings.append("Multiple image frames are present.")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "check_type": "technical_structure_only",
    }


# ---------------------------------------------------------
# 5. UnifiedDataRecord conversion
# ---------------------------------------------------------

def image_to_unified_record(
    file_path: str | Path,
    source_name: Optional[str] = None,
    description: str = "",
    data_type: str = "general",
    license_info: str = "Not specified",
    anonymized: Optional[bool] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
):
    """
    Convert image metadata into the current UnifiedDataRecord schema.

    Pixel data remains outside the record. Use load_image_array()
    separately if an in-memory pixel array is needed.
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    schema_module = load_schema_module()

    if not hasattr(schema_module, "UnifiedDataRecord"):
        raise AttributeError(
            "data_schema.py does not define UnifiedDataRecord."
        )

    RecordClass = schema_module.UnifiedDataRecord

    image_metadata = inspect_image(path)
    structure_report = validate_image_structure(image_metadata)

    raw_hash = image_metadata["raw_sha256"]
    dataset_id = generate_dataset_id(path, raw_hash)

    if source_name is None:
        source_name = path.name

    # Preserve core adapter fields against accidental replacement.
    metadata_extra = {
        "adapter": "image_adapter",
        "adapter_version": MODULE_VERSION,
        "dataset_id": dataset_id,
        "source_name": source_name,
        "description": description,
        "license_info": license_info,
        "anonymized": anonymized,
        "image_format": image_metadata["format"],
        "image_mode": image_metadata["mode"],
        "bands": image_metadata["bands"],
        "channel_count": image_metadata["channel_count"],
        "width": image_metadata["width"],
        "height": image_metadata["height"],
        "dimensions": image_metadata["dimensions"],
        "pixel_count": image_metadata["pixel_count"],
        "file_size_bytes": image_metadata["file_size_bytes"],
        "has_exif": image_metadata["has_exif"],
        "is_animated": image_metadata["is_animated"],
        "frame_count": image_metadata["frame_count"],
        "structure_report": structure_report,
        "raw_sha256": raw_hash,
        "raw_data_immutable": True,
        "scientific_processing_performed": False,
        "pixel_transformations_performed": False,
        "license_info": license_info,
        "anonymized": anonymized,
        "inspected_at": image_metadata["inspected_at"],
    }

    if extra_metadata is not None:
        # Only accept a dictionary of additional metadata.
        if not isinstance(extra_metadata, dict):
            raise TypeError("extra_metadata must be a dictionary.")

        # Additional metadata must not overwrite core provenance fields.
        protected_keys = set(metadata_extra)
        for key, value in extra_metadata.items():
            if key not in protected_keys:
                metadata_extra[key] = value

    record = RecordClass(
        record_id=dataset_id,
        schema_name=getattr(
            schema_module,
            "SCHEMA_NAME",
            "d3_vital_x.unified_data_record",
        ),
        schema_version=getattr(schema_module, "SCHEMA_VERSION", "0.1.0"),
        source=str(source_name),
        modality="image",
        data_type=data_type,
        observed_at=None,
        created_at=utc_timestamp(),
        payload_ref=str(path.resolve()),
        units={},
        metadata=metadata_extra,
        quality_flags=(
            ["image_structure_warning"]
            if structure_report["warnings"]
            else []
        ),
        provenance={
            "source_file": str(path.resolve()),
            "raw_sha256": raw_hash,
            "adapter": "image_adapter",
            "adapter_version": MODULE_VERSION,
            "raw_data_preserved": True,
            "pixel_transformations_performed": False,
            "scientific_processing_performed": False,
        },
        uncertainty={},
        labels=[],
        claim_level="C1",
    )

    return record


# ---------------------------------------------------------
# 6. Record summary
# ---------------------------------------------------------

def summarize_image_record(record) -> Dict[str, Any]:
    """Return a compact, JSON-friendly summary of a unified record."""
    metadata = record.metadata or {}
    structure_report = metadata.get("structure_report", {})

    return {
        "record_id": record.record_id,
        "schema_name": record.schema_name,
        "schema_version": record.schema_version,
        "source": record.source,
        "modality": record.modality,
        "data_type": record.data_type,
        "payload_ref": record.payload_ref,
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "channel_count": metadata.get("channel_count"),
        "format": metadata.get("image_format"),
        "structure_valid": structure_report.get("valid"),
        "raw_sha256": record.provenance.get("raw_sha256"),
        "raw_data_preserved": record.provenance.get("raw_data_preserved"),
        "claim_level": record.claim_level,
        "generated_at": utc_timestamp(),
    }


# ---------------------------------------------------------
# 7. Self-check
# ---------------------------------------------------------

def self_check() -> Dict[str, Any]:
    """
    Run basic adapter tests on a temporary synthetic image.

    The temporary test image is not a real scientific dataset.
    """
    if not PILLOW_AVAILABLE:
        return {
            "passed": False,
            "tests_run": 0,
            "error": "Pillow is not installed.",
            "install_hint": "%pip install Pillow",
        }

    import tempfile

    tests_run = 0

    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic_test.png"

            # Create a synthetic image only for this test.
            with Image.new("RGB", (8, 6), color=(20, 40, 60)) as image:
                image.save(path, format="PNG")

            original_hash = calculate_sha256(path)

            inspected = inspect_image(path)
            tests_run += 1
            assert inspected["width"] == 8
            assert inspected["height"] == 6
            assert inspected["channel_count"] == 3
            assert inspected["format"] == "PNG"

            report = validate_image_structure(inspected)
            tests_run += 1
            assert report["valid"] is True

            array = load_image_array(path)
            tests_run += 1
            assert tuple(array.shape) == (6, 8, 3)

            schema_path = resolve_schema_path()
            tests_run += 1
            assert schema_path.is_file()

            # Ensure inspection/loading did not change the file.
            final_hash = calculate_sha256(path)
            tests_run += 1
            assert original_hash == final_hash

            return {
                "passed": True,
                "tests_run": tests_run,
                "image_dimensions": [6, 8],
                "channel_count": 3,
                "source_hash_unchanged": True,
                "schema_path": str(schema_path),
            }

    except Exception as exc:
        return {
            "passed": False,
            "tests_run": tests_run,
            "error": f"{type(exc).__name__}: {exc}",
        }


if __name__ == "__main__":
    print("D³ VITAL-X Image Adapter", MODULE_VERSION)
    print(self_check())
