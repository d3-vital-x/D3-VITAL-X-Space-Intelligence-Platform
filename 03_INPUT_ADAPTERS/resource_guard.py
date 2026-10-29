
"""
Module 34: resource_guard.py
D³ VITAL-X Space Intelligence Platform

Purpose
-------
- Validate input file size and supported file extensions.
- Estimate array memory before processing.
- Check dimensions and total element counts.
- Provide resource warnings and explicit rejection reasons.
- Support JWST, Chandra, FITS, CSV, and common image inputs.

Scope
-----
Resource validation only.
This module does not perform FITS calibration, scientific QC,
anomaly detection, or guarantee that a file is safe to process.
"""

from __future__ import annotations

import json
import math
import os

from pathlib import Path
from typing import Any, Optional


MODULE_NAME = "resource_guard"
MODULE_VERSION = "1.0.0"

DEFAULT_MAX_FILE_MB = 30
DEFAULT_WARNING_FILE_MB = 20
DEFAULT_MAX_ARRAY_MB = 256
DEFAULT_MAX_DIMENSIONS = 5
DEFAULT_MAX_ELEMENTS = 100_000_000

BYTES_PER_MB = 1024 * 1024

SUPPORTED_EXTENSIONS = (
    ".fits",
    ".fit",
    ".fts",
    ".fits.gz",
    ".fit.gz",
    ".fts.gz",
    ".csv",
    ".txt",
    ".json",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)

NUMERIC_DTYPE_BYTES = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "complex64": 8,
    "complex128": 16,
}


class ResourceGuardError(Exception):
    """Base exception for resource validation failures."""


class FileValidationError(ResourceGuardError):
    """Raised when a file fails input validation."""


class ResourceLimitError(ResourceGuardError):
    """Raised when a configured resource limit is exceeded."""


def _safe_positive_int(value: Any, name: str) -> int:
    """Validate a positive integer configuration value."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")

    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be a positive integer."
        ) from exc

    if result <= 0 or result != value:
        raise ValueError(
            f"{name} must be a positive integer."
        )

    return result


def _extension_of(path: Path) -> str:
    """Return a normalized extension, preserving .fits.gz."""

    name = path.name.lower()

    for extension in SUPPORTED_EXTENSIONS:
        if name.endswith(extension):
            return extension

    return path.suffix.lower()


class ResourceGuard:
    """
    Resource policy and input validation for astronomy products.

    Parameters
    ----------
    max_file_mb:
        Maximum on-disk size permitted for a single file.
    warning_file_mb:
        File size at which a non-blocking warning is generated.
    max_array_mb:
        Maximum estimated bytes for one materialized array.
    max_dimensions:
        Maximum permitted number of array dimensions.
    max_elements:
        Maximum permitted number of elements in an array.

    Notes
    -----
    Array memory estimates are not total process-memory estimates.
    FITS decompression, temporary arrays, table parsing, and plotting
    may require additional memory.
    """

    def __init__(
        self,
        max_file_mb: int = DEFAULT_MAX_FILE_MB,
        warning_file_mb: int = DEFAULT_WARNING_FILE_MB,
        max_array_mb: int = DEFAULT_MAX_ARRAY_MB,
        max_dimensions: int = DEFAULT_MAX_DIMENSIONS,
        max_elements: int = DEFAULT_MAX_ELEMENTS,
    ) -> None:

        self.max_file_mb = _safe_positive_int(
            max_file_mb, "max_file_mb"
        )
        self.warning_file_mb = _safe_positive_int(
            warning_file_mb, "warning_file_mb"
        )
        self.max_array_mb = _safe_positive_int(
            max_array_mb, "max_array_mb"
        )
        self.max_dimensions = _safe_positive_int(
            max_dimensions, "max_dimensions"
        )
        self.max_elements = _safe_positive_int(
            max_elements, "max_elements"
        )

        if self.warning_file_mb > self.max_file_mb:
            raise ValueError(
                "warning_file_mb cannot exceed max_file_mb."
            )

        self.max_file_bytes = (
            self.max_file_mb * BYTES_PER_MB
        )
        self.warning_file_bytes = (
            self.warning_file_mb * BYTES_PER_MB
        )
        self.max_array_bytes = (
            self.max_array_mb * BYTES_PER_MB
        )

    @staticmethod
    def module_info() -> dict[str, Any]:
        """Return module identity and validation capabilities."""

        return {
            "module": MODULE_NAME,
            "version": MODULE_VERSION,
            "purpose": "Input resource validation",
            "supported_extensions": list(
                SUPPORTED_EXTENSIONS
            ),
            "default_max_file_mb": DEFAULT_MAX_FILE_MB,
            "default_warning_file_mb": DEFAULT_WARNING_FILE_MB,
            "default_max_array_mb": DEFAULT_MAX_ARRAY_MB,
            "scientific_calibration": False,
            "anomaly_detection": False,
        }

    def validate_file(
        self,
        file_path: str | os.PathLike[str],
        expected_extensions: Optional[tuple[str, ...]] = None,
    ) -> dict[str, Any]:
        """
        Validate a local file's existence, size, and extension.

        expected_extensions can restrict the permitted extensions
        for a particular processing workflow.
        """

        try:
            path = Path(file_path).expanduser()
        except (TypeError, ValueError) as exc:
            raise FileValidationError(
                "Invalid file path."
            ) from exc

        if not path.exists():
            raise FileValidationError(
                "Input file does not exist."
            )

        if not path.is_file():
            raise FileValidationError(
                "Input path must refer to a regular file."
            )

        try:
            size_bytes = path.stat().st_size
        except OSError as exc:
            raise FileValidationError(
                "Unable to inspect input file."
            ) from exc

        if size_bytes <= 0:
            raise FileValidationError(
                "Input file is empty."
            )

        if size_bytes > self.max_file_bytes:
            raise ResourceLimitError(
                f"File exceeds the {self.max_file_mb} MB limit."
            )

        extension = _extension_of(path)

        allowed = (
            tuple(ext.lower() for ext in expected_extensions)
            if expected_extensions is not None
            else SUPPORTED_EXTENSIONS
        )

        if extension not in allowed:
            raise FileValidationError(
                f"Unsupported file extension: {extension or 'unknown'}."
            )

        warnings: list[str] = []

        if size_bytes > self.warning_file_bytes:
            warnings.append(
                f"File exceeds the {self.warning_file_mb} MB "
                "resource warning threshold."
            )

        return {
            "module": MODULE_NAME,
            "status": "PASS",
            "filename": path.name,
            "extension": extension,
            "size_bytes": size_bytes,
            "size_mb": round(
                size_bytes / BYTES_PER_MB, 4
            ),
            "max_file_mb": self.max_file_mb,
            "warnings": warnings,
        }

    def estimate_array_memory(
        self,
        shape: tuple[int, ...] | list[int],
        dtype: str = "float64",
        copies: int = 1,
    ) -> dict[str, Any]:
        """
        Estimate memory for a dense numeric array.

        This is a conservative planning aid, not a complete
        prediction of Python process RAM usage.
        """

        if not isinstance(shape, (tuple, list)):
            raise ValueError(
                "shape must be a tuple or list of dimensions."
            )

        if not shape:
            raise ValueError(
                "shape must contain at least one dimension."
            )

        if len(shape) > self.max_dimensions:
            raise ResourceLimitError(
                f"Array exceeds the {self.max_dimensions}-dimension limit."
            )

        dimensions: list[int] = []

        for dimension in shape:
            if isinstance(dimension, bool):
                raise ValueError(
                    "Dimensions must be positive integers."
                )

            if not isinstance(dimension, int) or dimension <= 0:
                raise ValueError(
                    "Dimensions must be positive integers."
                )

            dimensions.append(dimension)

        normalized_dtype = str(dtype).lower()

        if normalized_dtype not in NUMERIC_DTYPE_BYTES:
            raise ValueError(
                f"Unsupported numeric dtype: {normalized_dtype}."
            )

        copies = _safe_positive_int(copies, "copies")

        element_count = math.prod(dimensions)

        if element_count > self.max_elements:
            raise ResourceLimitError(
                "Array exceeds the configured element-count limit."
            )

        itemsize = NUMERIC_DTYPE_BYTES[normalized_dtype]
        array_bytes = element_count * itemsize
        estimated_total_bytes = array_bytes * copies

        if estimated_total_bytes > self.max_array_bytes:
            raise ResourceLimitError(
                "Estimated array memory exceeds the configured "
                f"{self.max_array_mb} MB limit."
            )

        return {
            "module": MODULE_NAME,
            "status": "PASS",
            "shape": dimensions,
            "dtype": normalized_dtype,
            "element_count": element_count,
            "itemsize_bytes": itemsize,
            "single_array_bytes": array_bytes,
            "estimated_total_bytes": estimated_total_bytes,
            "estimated_total_mb": round(
                estimated_total_bytes / BYTES_PER_MB, 4
            ),
            "copies_estimated": copies,
            "note": (
                "Estimate covers dense numeric array storage only; "
                "it does not include all process or plotting memory."
            ),
        }

    def validate_array_metadata(
        self,
        shape: tuple[int, ...] | list[int],
        dtype: str = "float64",
    ) -> dict[str, Any]:
        """
        Validate array dimensions and estimate one array's memory.

        Use this before explicitly materializing large arrays,
        when reliable shape and dtype metadata are available.
        """

        return self.estimate_array_memory(
            shape=shape,
            dtype=dtype,
            copies=1,
        )

    def validate_upload_bytes(
        self,
        filename: str,
        data: bytes,
    ) -> dict[str, Any]:
        """
        Validate uploaded bytes without writing them to disk.

        Intended for integration with Streamlit UploadedFile.getvalue().
        The caller already holds these bytes in memory.
        """

        if not isinstance(data, bytes):
            raise FileValidationError(
                "Upload content must be bytes."
            )

        if not data:
            raise FileValidationError(
                "Uploaded file is empty."
            )

        if len(data) > self.max_file_bytes:
            raise ResourceLimitError(
                f"Upload exceeds the {self.max_file_mb} MB limit."
            )

        safe_name = Path(
            str(filename).replace("\\", "/")
        ).name

        if not safe_name or safe_name in {".", ".."}:
            raise FileValidationError(
                "Invalid uploaded filename."
            )

        extension = _extension_of(Path(safe_name))

        if extension not in SUPPORTED_EXTENSIONS:
            raise FileValidationError(
                f"Unsupported upload extension: "
                f"{extension or 'unknown'}."
            )

        warnings: list[str] = []

        if len(data) > self.warning_file_bytes:
            warnings.append(
                f"Upload exceeds the {self.warning_file_mb} MB "
                "resource warning threshold."
            )

        return {
            "module": MODULE_NAME,
            "status": "PASS",
            "filename": safe_name,
            "extension": extension,
            "size_bytes": len(data),
            "size_mb": round(
                len(data) / BYTES_PER_MB, 4
            ),
            "max_file_mb": self.max_file_mb,
            "warnings": warnings,
            "note": (
                "Upload bytes are already resident in memory; "
                "this check does not reduce existing RAM usage."
            ),
        }


def module_info() -> dict[str, Any]:
    """Convenience entry point for dashboard health checks."""
    return ResourceGuard.module_info()


def run_contract_tests() -> dict[str, Any]:
    """
    Run offline contract tests without astronomy dependencies
    or network access.
    """

    import tempfile

    guard = ResourceGuard(
        max_file_mb=30,
        warning_file_mb=20,
        max_array_mb=16,
        max_dimensions=5,
        max_elements=10_000_000,
    )

    tests_passed = 0

    # Test 1: Valid supported file.
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "sample.csv"
        file_path.write_bytes(b"x,y\n1,2\n")

        result = guard.validate_file(file_path)
        assert result["status"] == "PASS"
        assert result["extension"] == ".csv"
        tests_passed += 1

        # Test 2: Empty file rejection.
        empty_path = Path(temp_dir) / "empty.csv"
        empty_path.write_bytes(b"")

        try:
            guard.validate_file(empty_path)
        except FileValidationError:
            tests_passed += 1
        else:
            raise AssertionError(
                "Empty file was not rejected."
            )

        # Test 3: Unsupported extension rejection.
        unknown_path = Path(temp_dir) / "sample.exe"
        unknown_path.write_bytes(b"test")

        try:
            guard.validate_file(unknown_path)
        except FileValidationError:
            tests_passed += 1
        else:
            raise AssertionError(
                "Unsupported extension was not rejected."
            )

    # Test 4: Valid array estimate.
    estimate = guard.estimate_array_memory(
        shape=(100, 100),
        dtype="float64",
    )
    assert estimate["element_count"] == 10_000
    assert estimate["single_array_bytes"] == 80_000
    tests_passed += 1

    # Test 5: Too many dimensions.
    try:
        guard.estimate_array_memory(
            shape=(2, 2, 2, 2, 2, 2),
            dtype="float32",
        )
    except ResourceLimitError:
        tests_passed += 1
    else:
        raise AssertionError(
            "Excessive dimensions were not rejected."
        )

    # Test 6: Array memory limit.
    try:
        guard.estimate_array_memory(
            shape=(2000, 2000),
            dtype="float64",
        )
    except ResourceLimitError:
        tests_passed += 1
    else:
        raise AssertionError(
            "Array memory limit was not enforced."
        )

    # Test 7: Upload byte validation.
    upload_result = guard.validate_upload_bytes(
        "test.fits",
        b"SIMPLE  =",
    )
    assert upload_result["status"] == "PASS"
    tests_passed += 1

    # Test 8: Unsupported upload extension.
    try:
        guard.validate_upload_bytes(
            "payload.exe",
            b"test",
        )
    except FileValidationError:
        tests_passed += 1
    else:
        raise AssertionError(
            "Unsupported upload was not rejected."
        )

    # Test 9: Oversized in-memory upload.
    tiny_guard = ResourceGuard(
        max_file_mb=1,
        warning_file_mb=1,
        max_array_mb=1,
    )

    try:
        tiny_guard.validate_upload_bytes(
            "large.csv",
            b"x" * (BYTES_PER_MB + 1),
        )
    except ResourceLimitError:
        tests_passed += 1
    else:
        raise AssertionError(
            "Oversized upload was not rejected."
        )

    return {
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "status": "PASS",
        "tests_passed": tests_passed,
        "tests_expected": 9,
        "network_used": False,
        "scope": "Offline resource-guard contract tests",
    }


if __name__ == "__main__":
    print(json.dumps(
        {
            "module_info": module_info(),
            "contract_tests": run_contract_tests(),
        },
        indent=2,
    ))
