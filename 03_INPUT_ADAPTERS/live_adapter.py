
"""
D³ VITAL-X Space Intelligence Platform
Module 09: live_adapter.py
Version: 0.1.0

Purpose:
    Live video/frame and numerical matrix-stream adapter.

Responsibilities:
    - Accept individual frames or numerical matrices.
    - Decode image bytes using Pillow when available.
    - Preserve original input values without smoothing or clipping.
    - Validate shape, dtype, and finite numerical values.
    - Calculate SHA-256 hashes for reproducibility.
    - Create UnifiedDataRecord-compatible metadata.
    - Support frame-by-frame processing without retaining the
      entire video stream in memory.

Boundaries:
    - No scientific discovery or anomaly conclusion.
    - No medical diagnosis.
    - No spacecraft control.
    - No modification of source files.
    - No proprietary intelligence-core implementation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import sys

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Union
from uuid import uuid4

import numpy as np


MODULE_NAME = "live_adapter"
MODULE_VERSION = "0.1.0"

MAX_FRAME_PIXELS = 40_000_000
MAX_STREAM_FRAMES = 100_000

FrameInput = Union[
    np.ndarray,
    bytes,
    bytearray,
    memoryview,
]


# ============================================================
# 1. BASIC UTILITIES
# ============================================================

def utc_timestamp() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def calculate_sha256(
    data: Union[bytes, bytearray, memoryview, np.ndarray]
) -> str:
    """
    Calculate SHA-256 without changing the supplied input.

    Arrays are hashed using their C-order byte representation.
    Array metadata should also be recorded to distinguish shapes
    and data types that share identical byte sequences.
    """
    digest = hashlib.sha256()

    if isinstance(data, np.ndarray):
        if data.dtype.hasobject:
            raise TypeError(
                "Object-dtype arrays are not supported."
            )
        view = np.ascontiguousarray(data)
        digest.update(view.tobytes(order="C"))

    elif isinstance(data, (bytes, bytearray, memoryview)):
        digest.update(bytes(data))

    else:
        raise TypeError(
            "Expected bytes-like data or a NumPy array."
        )

    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Convert supported metadata values into JSON-safe values."""
    if isinstance(value, dict):
        return {
            str(k): _json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    if isinstance(value, np.generic):
        return _json_safe(value.item())

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (str, int, bool)) or value is None:
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    return str(value)


# ============================================================
# 2. FRAME DECODING
# ============================================================

def decode_frame(
    frame: FrameInput,
    *,
    color_mode: str = "RGB",
    copy_array: bool = True,
) -> np.ndarray:
    """
    Convert an image frame or numerical matrix to a NumPy array.

    Accepted:
        - NumPy arrays
        - Encoded image bytes
        - bytearray / memoryview containing encoded image data

    color_mode:
        RGB  : decode encoded images into RGB
        RGBA : decode encoded images into RGBA
        L    : decode encoded images into grayscale

    Existing NumPy arrays are NOT color-converted, normalized,
    clipped, or rescaled.
    """
    if color_mode not in {"RGB", "RGBA", "L"}:
        raise ValueError(
            "color_mode must be RGB, RGBA, or L."
        )

    if isinstance(frame, np.ndarray):
        array = np.array(frame, copy=copy_array)

    elif isinstance(frame, (bytes, bytearray, memoryview)):
        try:
            from PIL import Image, ImageOps
        except ImportError as exc:
            raise ImportError(
                "Pillow is required to decode image bytes."
            ) from exc

        raw_bytes = bytes(frame)

        with Image.open(io.BytesIO(raw_bytes)) as image:
            image.load()
            image = ImageOps.exif_transpose(image)

            width, height = image.size

            if width * height > MAX_FRAME_PIXELS:
                raise ValueError(
                    "Image exceeds MAX_FRAME_PIXELS."
                )

            converted = image.convert(color_mode)
            array = np.array(converted, copy=True)

    else:
        raise TypeError(
            "Unsupported frame type. Supply a NumPy array "
            "or encoded image bytes."
        )

    validate_frame(array)
    return array


# ============================================================
# 3. FRAME VALIDATION
# ============================================================

def validate_frame(
    frame: np.ndarray,
    *,
    max_pixels: int = MAX_FRAME_PIXELS,
) -> Dict[str, Any]:
    """
    Validate frame structure and numerical safety.

    This is technical validation, not scientific interpretation.
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a NumPy array.")

    if frame.size == 0:
        raise ValueError("Frame is empty.")

    if frame.dtype.hasobject:
        raise TypeError(
            "Object-dtype arrays are not supported."
        )

    if frame.ndim not in (2, 3):
        raise ValueError(
            "Expected a 2-D matrix or a 3-D image array."
        )

    height, width = frame.shape[:2]

    if height <= 0 or width <= 0:
        raise ValueError("Invalid frame dimensions.")

    if height * width > max_pixels:
        raise ValueError(
            "Frame exceeds the configured pixel limit."
        )

    if frame.ndim == 3 and frame.shape[2] not in (1, 2, 3, 4):
        raise ValueError(
            "Unsupported channel count."
        )

    if np.issubdtype(frame.dtype, np.number):
        finite_mask = np.isfinite(frame)
        finite_count = int(np.count_nonzero(finite_mask))
        total_count = int(frame.size)

        if np.issubdtype(frame.dtype, np.complexfloating):
            raise TypeError(
                "Complex-valued frames are not supported."
            )

        if finite_count == 0:
            raise ValueError(
                "Frame contains no finite numerical values."
            )

        nonfinite_count = total_count - finite_count

    elif frame.dtype == np.bool_:
        finite_count = int(frame.size)
        nonfinite_count = 0

    else:
        raise TypeError(
            f"Unsupported frame dtype: {frame.dtype}"
        )

    return {
        "valid": nonfinite_count == 0,
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "height": int(height),
        "width": int(width),
        "channels": (
            int(frame.shape[2])
            if frame.ndim == 3
            else 1
        ),
        "total_values": int(frame.size),
        "finite_values": finite_count,
        "nonfinite_values": nonfinite_count,
        "requires_qc_review": nonfinite_count > 0,
    }


# ============================================================
# 4. NUMERICAL FRAME SUMMARY
# ============================================================

def summarize_frame(frame: np.ndarray) -> Dict[str, Any]:
    """
    Return descriptive numerical statistics.

    Non-finite values are excluded from summary statistics but
    are counted explicitly. The original array is unchanged.
    """
    validation = validate_frame(frame)

    if not np.issubdtype(frame.dtype, np.number):
        raise TypeError(
            "Numerical summaries require a numeric array."
        )

    finite = np.asarray(frame)[np.isfinite(frame)]

    result: Dict[str, Any] = {
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "total_values": int(frame.size),
        "finite_values": int(finite.size),
        "nonfinite_values": validation["nonfinite_values"],
    }

    if finite.size:
        result.update({
            "minimum": float(np.min(finite)),
            "maximum": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "variance": float(np.var(finite)),
            "standard_deviation": float(np.std(finite)),
        })

    result["summary_type"] = "descriptive_only"
    return result


# ============================================================
# 5. UNIFIED DATA RECORD INTEGRATION
# ============================================================

def _load_unified_record_class():
    """
    Locate UnifiedDataRecord in the project's public schema.

    Supports common folder names used by project revisions.
    """
    for module_name in (
        "data_schema",
        "d3_vital_x_data_schema",
    ):
        module = sys.modules.get(module_name)

        if module is not None:
            record_class = getattr(
                module, "UnifiedDataRecord", None
            )
            if record_class is not None:
                return record_class

    current = Path(__file__).resolve() if "__file__" in globals() else None

    if current is not None:
        root = current.parent.parent

        candidates = [
            root / "04_UNIFIED_DATA" / "data_schema.py",
            root / "04_UNIFIED_DATA_LAYER" / "data_schema.py",
        ]

        for schema_path in candidates:
            if not schema_path.is_file():
                continue

            spec = importlib.util.spec_from_file_location(
                "d3_vital_x_data_schema",
                schema_path,
            )

            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            record_class = getattr(
                module, "UnifiedDataRecord", None
            )

            if record_class is not None:
                return record_class

    return None


def create_live_record(
    frame: np.ndarray,
    *,
    frame_index: int = 0,
    stream_id: Optional[str] = None,
    observed_at: Optional[str] = None,
    source_name: str = "live_stream",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Build a schema-compatible record without embedding frame data.

    payload_ref is a memory reference, not a persistent storage URL.
    The caller remains responsible for retaining the actual payload.
    """
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative.")

    validation = validate_frame(frame)
    summary = summarize_frame(frame)
    digest = calculate_sha256(frame)

    metadata: Dict[str, Any] = {
        "adapter": MODULE_NAME,
        "adapter_version": MODULE_VERSION,
        "stream_id": stream_id or "unspecified",
        "frame_index": int(frame_index),
        "frame_shape": list(frame.shape),
        "frame_dtype": str(frame.dtype),
        "frame_sha256": digest,
        "validation": validation,
        "summary": summary,
        "interpretation": "technical_metadata_only",
    }

    if extra_metadata:
        metadata["extra"] = _json_safe(extra_metadata)

    record_data = {
        "record_id": str(uuid4()),
        "source": source_name,
        "modality": "live_frame",
        "data_type": "numerical_matrix",
        "observed_at": observed_at or utc_timestamp(),
        "payload_ref": (
            f"memory://{stream_id or 'unspecified'}/"
            f"frame/{frame_index}"
        ),
        "units": {},
        "metadata": metadata,
        "quality_flags": (
            ["NONFINITE_VALUES_PRESENT"]
            if validation["nonfinite_values"]
            else []
        ),
        "provenance": {
            "adapter": MODULE_NAME,
            "adapter_version": MODULE_VERSION,
            "raw_sha256": digest,
            "created_at": utc_timestamp(),
            "source_name": source_name,
        },
        "uncertainty": {
            "status": "not_estimated_by_adapter"
        },
        "labels": [],
        "claim_level": "C1",
    }

    record_class = _load_unified_record_class()

    if record_class is not None:
        return record_class(**record_data)

    # Standalone testing fallback if the project schema is not
    # available in the current Colab environment.
    record_data["schema_name"] = "d3_vital_x_unified_data"
    record_data["schema_version"] = "0.1.0"
    record_data["created_at"] = utc_timestamp()
    record_data["fallback_schema"] = True

    return record_data


# ============================================================
# 6. SINGLE-FRAME ADAPTER
# ============================================================

def adapt_frame(
    frame: FrameInput,
    *,
    frame_index: int = 0,
    stream_id: Optional[str] = None,
    observed_at: Optional[str] = None,
    source_name: str = "live_stream",
    color_mode: str = "RGB",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Process one frame and return its matrix, record, and QC summary.

    The returned matrix is separate from the record metadata.
    No anomaly score is calculated here.
    """
    matrix = decode_frame(frame, color_mode=color_mode)

    record = create_live_record(
        matrix,
        frame_index=frame_index,
        stream_id=stream_id,
        observed_at=observed_at,
        source_name=source_name,
        extra_metadata=extra_metadata,
    )

    return {
        "success": True,
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "frame": matrix,
        "record": record,
        "validation": validate_frame(matrix),
        "summary": summarize_frame(matrix),
        "raw_sha256": calculate_sha256(matrix),
    }


# ============================================================
# 7. STREAM ADAPTER
# ============================================================

def adapt_stream(
    frames: Iterable[FrameInput],
    *,
    stream_id: Optional[str] = None,
    max_frames: int = MAX_STREAM_FRAMES,
) -> Iterator[Dict[str, Any]]:
    """
    Lazily adapt a sequence of frames.

    Only one frame is adapted per iteration, helping resource-limited
    devices avoid loading a whole video into memory.

    The iterable must yield decoded arrays or encoded image bytes.
    Video-file decoding should be performed by a separate capture
    layer that yields frames.
    """
    if max_frames <= 0:
        raise ValueError("max_frames must be positive.")

    for index, frame in enumerate(frames):
        if index >= max_frames:
            raise RuntimeError(
                "Stream frame limit reached. "
                "Increase max_frames explicitly if appropriate."
            )

        yield adapt_frame(
            frame,
            frame_index=index,
            stream_id=stream_id,
            source_name="live_stream",
        )


# ============================================================
# 8. MODULE METADATA
# ============================================================

def module_info() -> Dict[str, Any]:
    """Return public adapter capabilities and limitations."""
    return {
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "purpose": "Live video/frame and matrix-stream adaptation",
        "accepted_inputs": [
            "numpy.ndarray",
            "encoded_image_bytes",
            "bytearray",
            "memoryview",
        ],
        "video_file_decoding_included": False,
        "raw_input_mutation": False,
        "scientific_interpretation": False,
        "clinical_diagnosis": False,
        "spacecraft_control": False,
        "protected_core_included": False,
    }


# ============================================================
# 9. SELF-TEST
# ============================================================

def run_live_adapter_test() -> Dict[str, Any]:
    """
    Run deterministic local tests without a camera or network.

    Uses synthetic matrices only.
    """
    test_frame = np.arange(
        4 * 5 * 3, dtype=np.uint8
    ).reshape(4, 5, 3)

    original = test_frame.copy()
    original_hash = calculate_sha256(test_frame)

    result = adapt_frame(
        test_frame,
        frame_index=0,
        stream_id="synthetic-test",
    )

    assert result["success"] is True
    assert result["frame"].shape == (4, 5, 3)
    assert result["validation"]["valid"] is True
    assert result["raw_sha256"] == original_hash
    assert np.array_equal(test_frame, original)

    # A non-finite value must be flagged, not silently replaced.
    numeric_frame = np.array(
        [[1.0, 2.0], [np.nan, 4.0]],
        dtype=np.float64,
    )

    numeric_check = validate_frame(numeric_frame)
    assert numeric_check["nonfinite_values"] == 1
    assert numeric_check["requires_qc_review"] is True

    stream_results = list(
        adapt_stream(
            [test_frame, test_frame + 1],
            stream_id="synthetic-stream",
        )
    )

    assert len(stream_results) == 2
    assert stream_results[0]["record"] is not None
    assert stream_results[1]["record"] is not None

    # Reject unsupported dimensions.
    invalid_shape_rejected = False
    try:
        validate_frame(np.zeros((2, 2, 5)))
    except ValueError:
        invalid_shape_rejected = True

    assert invalid_shape_rejected

    return {
        "passed": True,
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "tests_run": 7,
        "network_used": False,
        "camera_required": False,
        "raw_input_unchanged": np.array_equal(
            test_frame, original
        ),
        "original_sha256": original_hash,
        "frame_shape": list(test_frame.shape),
        "stream_frames_processed": len(stream_results),
        "nonfinite_values_flagged": True,
        "invalid_channel_count_rejected": True,
    }


if __name__ == "__main__":
    print("D³ VITAL-X Live Adapter")
    print(json.dumps(
        run_live_adapter_test(),
        indent=2,
        ensure_ascii=False,
    ))
