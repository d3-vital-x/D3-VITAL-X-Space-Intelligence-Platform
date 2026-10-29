"""
Module 35: fits_inspector.py
D³ VITAL-X Space Intelligence Platform

FITS structural/metadata inspection only. This module does not calibrate
data, confirm scientific validity, or detect astronomical anomalies.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Optional

MODULE_NAME = "fits_inspector"
MODULE_VERSION = "1.0.1"

DEFAULT_MAX_FILE_MB = 30
DEFAULT_MAX_HDUS = 256
DEFAULT_MAX_HEADER_CARDS = 5000
DEFAULT_MAX_COLUMNS = 512
DEFAULT_MAX_DIMENSIONS = 5
DEFAULT_MAX_ELEMENTS = 100_000_000
BYTES_PER_MB = 1024 * 1024

FITS_EXTENSIONS = (
    ".fits.gz", ".fit.gz", ".fts.gz", ".fits", ".fit", ".fts"
)

MISSION_HEADER_KEYS = (
    "TELESCOP", "INSTRUME", "DETECTOR", "FILTER", "GRATING",
    "EXP_TYPE", "DATE-OBS", "TIME-OBS", "MJD-OBS", "EXPTIME",
    "TARGNAME", "OBJECT", "OBS_ID", "OBSERVTN", "PROPOSID",
    "PROGRAM", "CAL_VER", "DATAMODL", "BUNIT", "CTYPE1", "CTYPE2",
    "CTYPE3", "CUNIT1", "CUNIT2", "CUNIT3", "CRVAL1", "CRVAL2",
    "CRVAL3", "CDELT1", "CDELT2", "CDELT3", "CRPIX1", "CRPIX2",
    "CRPIX3", "WCSAXES",
)


class FITSInspectorError(Exception):
    """Base exception for FITS inspection errors."""


class FITSValidationError(FITSInspectorError):
    """Raised for missing, unsupported, or invalid FITS files."""


class FITSResourceLimitError(FITSInspectorError):
    """Raised when an inspection resource limit is exceeded."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    try:
        converted = value.item()
        if converted is not value:
            return _safe_value(converted)
    except (AttributeError, ValueError, TypeError):
        pass
    return str(value)


def _fits_extension(path: Path) -> Optional[str]:
    name = path.name.lower()
    for extension in FITS_EXTENSIONS:
        if name.endswith(extension):
            return extension
    return None


def _exception_name(exc: Exception) -> str:
    return type(exc).__name__


class FITSInspector:
    """Metadata-first FITS inspector with configurable safety limits."""

    def __init__(
        self,
        max_file_mb: int = DEFAULT_MAX_FILE_MB,
        max_hdus: int = DEFAULT_MAX_HDUS,
        max_header_cards: int = DEFAULT_MAX_HEADER_CARDS,
        max_columns: int = DEFAULT_MAX_COLUMNS,
        max_dimensions: int = DEFAULT_MAX_DIMENSIONS,
        max_elements: int = DEFAULT_MAX_ELEMENTS,
    ) -> None:
        self.max_file_mb = _positive_int(max_file_mb, "max_file_mb")
        self.max_hdus = _positive_int(max_hdus, "max_hdus")
        self.max_header_cards = _positive_int(
            max_header_cards, "max_header_cards"
        )
        self.max_columns = _positive_int(max_columns, "max_columns")
        self.max_dimensions = _positive_int(
            max_dimensions, "max_dimensions"
        )
        self.max_elements = _positive_int(max_elements, "max_elements")
        self.max_file_bytes = self.max_file_mb * BYTES_PER_MB

    @staticmethod
    def module_info() -> dict[str, Any]:
        return {
            "module": MODULE_NAME,
            "version": MODULE_VERSION,
            "purpose": "FITS structural and metadata inspection",
            "supported_extensions": list(FITS_EXTENSIONS),
            "default_max_file_mb": DEFAULT_MAX_FILE_MB,
            "metadata_first": True,
            "scientific_calibration": False,
            "anomaly_detection": False,
        }

    def inspect(
        self,
        file_path: str | os.PathLike[str],
        verify_fits: bool = True,
    ) -> dict[str, Any]:
        """Inspect a FITS file and return a JSON-compatible report."""
        try:
            path = Path(file_path).expanduser()
        except (TypeError, ValueError) as exc:
            raise FITSValidationError("Invalid FITS file path.") from exc

        if not path.exists():
            raise FITSValidationError("FITS file does not exist.")
        if not path.is_file():
            raise FITSValidationError("Input path must be a regular file.")

        extension = _fits_extension(path)
        if extension is None:
            raise FITSValidationError("Unsupported FITS filename extension.")

        try:
            file_size = path.stat().st_size
        except OSError as exc:
            raise FITSValidationError("Unable to inspect file metadata.") from exc

        if file_size <= 0:
            raise FITSValidationError("FITS file is empty.")
        if file_size > self.max_file_bytes:
            raise FITSResourceLimitError(
                f"File exceeds the {self.max_file_mb} MB limit."
            )

        try:
            from astropy.io import fits
        except ImportError as exc:
            raise FITSInspectorError(
                "Astropy is unavailable; install project requirements."
            ) from exc

        report: dict[str, Any] = {
            "module": MODULE_NAME,
            "version": MODULE_VERSION,
            "status": "INSPECTING",
            "filename": path.name,
            "extension": extension,
            "size_bytes": file_size,
            "size_mb": round(file_size / BYTES_PER_MB, 4),
            "hdu_count": 0,
            "hdus": [],
            "mission_metadata": {},
            "warnings": [],
            "errors": [],
            "scientific_calibration": False,
            "scientific_validity_confirmed": False,
        }

        # Gzip files cannot use memory mapping. Regular FITS files may
        # use it, but memory mapping does not guarantee zero RAM usage.
        use_memmap = not extension.endswith(".gz")

        try:
            with fits.open(
                path,
                mode="readonly",
                memmap=use_memmap,
                lazy_load_hdus=True,
                ignore_missing_end=False,
            ) as hdul:
                hdu_count = len(hdul)
                if hdu_count == 0:
                    raise FITSValidationError("No HDUs found.")
                if hdu_count > self.max_hdus:
                    raise FITSResourceLimitError(
                        f"FITS file has {hdu_count} HDUs; "
                        f"limit is {self.max_hdus}."
                    )
                report["hdu_count"] = hdu_count

                if verify_fits:
                    try:
                        hdul.verify("exception")
                    except Exception as exc:
                        raise FITSValidationError(
                            "FITS structural verification failed: "
                            + _exception_name(exc)
                        ) from exc

                for index, hdu in enumerate(hdul):
                    header = hdu.header
                    card_count = len(header.cards)
                    if card_count > self.max_header_cards:
                        raise FITSResourceLimitError(
                            f"HDU {index} header exceeds "
                            f"{self.max_header_cards} cards."
                        )

                    item: dict[str, Any] = {
                        "index": index,
                        "name": str(getattr(hdu, "name", "") or ""),
                        "type": type(hdu).__name__,
                        "header_card_count": card_count,
                        "data_present": False,
                        "shape": None,
                        "ndim": None,
                        "dtype": None,
                        "estimated_array_bytes": None,
                        "column_count": 0,
                        "columns": [],
                        "metadata": {},
                        "warnings": [],
                    }

                    # For image HDUs, shape and BITPIX come from header
                    # metadata, avoiding a deliberate full-array read.
                    shape = getattr(hdu, "shape", None)
                    if shape is not None:
                        shape = [int(dimension) for dimension in shape]
                        item["shape"] = shape
                        item["ndim"] = len(shape)
                        item["data_present"] = True

                        if len(shape) > self.max_dimensions:
                            raise FITSResourceLimitError(
                                f"HDU {index} exceeds dimensionality limit."
                            )
                        if any(dimension < 0 for dimension in shape):
                            raise FITSValidationError(
                                f"HDU {index} has an invalid shape."
                            )

                        element_count = math.prod(shape)
                        if element_count > self.max_elements:
                            raise FITSResourceLimitError(
                                f"HDU {index} exceeds element-count limit."
                            )

                        bitpix = header.get("BITPIX")
                        bitpix_types = {
                            8: ("uint8", 1),
                            16: ("int16", 2),
                            32: ("int32", 4),
                            64: ("int64", 8),
                            -32: ("float32", 4),
                            -64: ("float64", 8),
                        }
                        if bitpix in bitpix_types:
                            dtype_name, itemsize = bitpix_types[bitpix]
                            item["dtype"] = dtype_name
                            item["estimated_array_bytes"] = (
                                element_count * itemsize
                            )

                    # Table column definitions are inspected without
                    # converting the entire table into an array.
                    columns = getattr(hdu, "columns", None)
                    if columns is not None:
                        count = len(columns)
                        if count > self.max_columns:
                            raise FITSResourceLimitError(
                                f"HDU {index} has {count} columns; "
                                f"limit is {self.max_columns}."
                            )
                        item["column_count"] = count
                        item["data_present"] = item["data_present"] or count > 0
                        for column in columns:
                            item["columns"].append({
                                "name": str(column.name),
                                "format": _safe_value(
                                    getattr(column, "format", None)
                                ),
                                "unit": _safe_value(
                                    getattr(column, "unit", None)
                                ),
                                "description": _safe_value(
                                    getattr(column, "description", None)
                                ),
                            })

                    # Fallback for uncommon HDU types only.
                    if not item["data_present"] and columns is None:
                        try:
                            data = hdu.data
                            item["data_present"] = data is not None
                            if data is not None:
                                dtype = getattr(data, "dtype", None)
                                if dtype is not None:
                                    try:
                                        import numpy as np
                                        item["dtype"] = np.dtype(dtype).name
                                    except (TypeError, ValueError):
                                        item["dtype"] = str(dtype)
                        except Exception as exc:
                            item["warnings"].append(
                                "Could not determine data presence: "
                                + _exception_name(exc)
                            )

                    for key in MISSION_HEADER_KEYS:
                        try:
                            value = _safe_value(header[key])
                        except (KeyError, TypeError, ValueError):
                            value = None
                        if value is not None:
                            item["metadata"][key] = value

                    if not item["data_present"]:
                        item["warnings"].append(
                            "No data payload was detected for this HDU."
                        )

                    report["hdus"].append(item)
                    if index == 0:
                        report["mission_metadata"].update(item["metadata"])

                if not any(item["data_present"] for item in report["hdus"]):
                    report["warnings"].append(
                        "No HDU contains a detected data payload."
                    )
                report["status"] = "PASS"

        except (FITSValidationError, FITSResourceLimitError):
            raise
        except Exception as exc:
            raise FITSValidationError(
                "FITS inspection failed: " + _exception_name(exc)
            ) from exc

        report["warnings"].append(
            "Structural inspection does not confirm calibration, units, "
            "or scientific validity."
        )
        return report

    def inspect_summary(
        self, file_path: str | os.PathLike[str]
    ) -> dict[str, Any]:
        """Return a compact summary for dashboard display."""
        report = self.inspect(file_path)
        return {
            "module": report["module"],
            "version": report["version"],
            "status": report["status"],
            "filename": report["filename"],
            "size_mb": report["size_mb"],
            "hdu_count": report["hdu_count"],
            "mission_metadata": report["mission_metadata"],
            "hdus": [
                {
                    "index": hdu["index"],
                    "name": hdu["name"],
                    "type": hdu["type"],
                    "shape": hdu["shape"],
                    "dtype": hdu["dtype"],
                    "column_count": hdu["column_count"],
                    "columns": hdu["columns"],
                    "warnings": hdu["warnings"],
                }
                for hdu in report["hdus"]
            ],
            "warnings": report["warnings"],
            "scientific_validity_confirmed": False,
        }


def module_info() -> dict[str, Any]:
    """Convenience function for dashboard health checks."""
    return FITSInspector.module_info()


def run_contract_tests() -> dict[str, Any]:
    """Run offline contract tests with small synthetic FITS fixtures."""
    import tempfile
    import numpy as np

    try:
        from astropy.io import fits
    except ImportError as exc:
        raise FITSInspectorError(
            "Tests require Astropy and NumPy."
        ) from exc

    passed = 0
    inspector = FITSInspector(
        max_file_mb=30,
        max_hdus=16,
        max_header_cards=500,
        max_columns=64,
        max_dimensions=5,
        max_elements=1_000_000,
    )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        # Test 1: image shape, dtype, and mission metadata.
        image_path = root / "test_image.fits"
        primary = fits.PrimaryHDU(
            data=np.arange(100, dtype=np.float32).reshape(10, 10)
        )
        primary.header["TELESCOP"] = "TEST"
        primary.header["INSTRUME"] = "TESTCAM"
        primary.writeto(image_path)

        image_report = inspector.inspect(image_path)
        assert image_report["status"] == "PASS"
        assert image_report["hdu_count"] == 1
        assert image_report["hdus"][0]["shape"] == [10, 10]
        assert image_report["hdus"][0]["dtype"] == "float32"
        assert image_report["mission_metadata"]["TELESCOP"] == "TEST"
        passed += 1

        # Test 2: binary table column metadata.
        table_path = root / "test_table.fits"
        table_columns = [
            fits.Column(
                name="TIME", format="D", unit="s",
                array=np.array([1.0, 2.0, 3.0]),
            ),
            fits.Column(
                name="COUNTS", format="J",
                array=np.array([10, 12, 9], dtype=np.int32),
            ),
        ]
        table_hdu = fits.BinTableHDU.from_columns(table_columns)
        fits.HDUList([fits.PrimaryHDU(), table_hdu]).writeto(table_path)
        table_report = inspector.inspect(table_path)
        assert table_report["hdu_count"] == 2
        assert table_report["hdus"][1]["column_count"] == 2
        assert [
            col["name"] for col in table_report["hdus"][1]["columns"]
        ] == ["TIME", "COUNTS"]
        passed += 1

        # Test 3: unsupported extension.
        csv_path = root / "not_fits.csv"
        csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
        try:
            inspector.inspect(csv_path)
        except FITSValidationError:
            passed += 1
        else:
            raise AssertionError("Unsupported extension was not rejected.")

        # Test 4: missing file.
        try:
            inspector.inspect(root / "missing.fits")
        except FITSValidationError:
            passed += 1
        else:
            raise AssertionError("Missing file was not rejected.")

        # Test 5: file-size limit.
        small_limit = FITSInspector(max_file_mb=1)
        oversized = root / "oversized.fits"
        with oversized.open("wb") as stream:
            stream.truncate(BYTES_PER_MB + 1)
        try:
            small_limit.inspect(oversized)
        except FITSResourceLimitError:
            passed += 1
        else:
            raise AssertionError("Oversized file was not rejected.")

        # Test 6: HDU-count limit.
        one_hdu_limit = FITSInspector(max_hdus=1)
        multi_path = root / "multi_hdu.fits"
        fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.ones((2, 2), dtype=np.int16)),
        ]).writeto(multi_path)
        try:
            one_hdu_limit.inspect(multi_path)
        except FITSResourceLimitError:
            passed += 1
        else:
            raise AssertionError("HDU limit was not enforced.")

        # Test 7: compact summary contract.
        summary = inspector.inspect_summary(image_path)
        assert summary["status"] == "PASS"
        assert summary["scientific_validity_confirmed"] is False
        passed += 1

    return {
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "status": "PASS",
        "tests_passed": passed,
        "tests_expected": 7,
        "network_used": False,
        "scope": "Synthetic FITS structural contract tests",
        "note": (
            "Passing these tests does not establish compatibility "
            "with every JWST or Chandra product."
        ),
    }


if __name__ == "__main__":
    print(json.dumps({
        "module_info": module_info(),
        "contract_tests": run_contract_tests(),
    }, indent=2))
