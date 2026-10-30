"""
Module 37: chandra_adapter.py
D³ VITAL-X Space Intelligence Platform

Purpose
-------
- Read Chandra X-ray Observatory FITS products.
- Inspect FITS metadata and HDU structure.
- Read 2D images and exposure maps.
- Read event tables (EVENTS).
- Read OGIP-style PHA/SPECTRUM tables.
- Perform basic finite-value and shape QC.
- Provide bounded reads for mobile-friendly environments.

Scope
-----
Exploratory data ingestion and quality control only.
This module does not perform scientific discovery,
source detection, diagnosis, or anomaly confirmation.

Dependencies
------------
numpy
astropy >= 6.0, < 8.0

Public interface
----------------
ChandraAdapter
inspect_chandra()
read_chandra_image()
read_chandra_events()
read_chandra_spectrum()
read_chandra_exposure_map()
run_contract_tests()
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np


# ============================================================
# MODULE CONFIGURATION
# ============================================================

MODULE_NAME = "chandra_adapter"
MODULE_VERSION = "1.0.0"
OBSERVATORY = "Chandra X-ray Observatory"

DEFAULT_MAX_FILE_MB = 100.0
DEFAULT_MAX_ARRAY_MB = 256.0
DEFAULT_MAX_ELEMENTS = 50_000_000
DEFAULT_MAX_EVENT_ROWS = 1_000_000
DEFAULT_PREVIEW_SIZE = 512

FITS_EXTENSIONS = (
    ".fits",
    ".fit",
    ".fts",
    ".fits.gz",
    ".fit.gz",
    ".fts.gz",
)


# ============================================================
# EXCEPTIONS
# ============================================================

class ChandraAdapterError(Exception):
    """Base exception for Chandra adapter."""


class ChandraDependencyError(ChandraAdapterError):
    """Raised when Astropy is unavailable."""


class ChandraValidationError(ChandraAdapterError):
    """Raised when input validation fails."""


class ChandraResourceLimitError(ChandraAdapterError):
    """Raised when a configured resource limit is exceeded."""


# ============================================================
# DEPENDENCY LOADER
# ============================================================

def _get_fits():
    try:
        from astropy.io import fits
        return fits
    except ImportError as exc:
        raise ChandraDependencyError(
            "Astropy is required. Install it with: pip install astropy"
        ) from exc


# ============================================================
# GENERAL HELPERS
# ============================================================

def _json_safe(value: Any) -> Any:
    """Convert common NumPy and Python values to JSON-safe values."""

    if isinstance(value, dict):
        return {
            str(k): _json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())

    if isinstance(value, np.generic):
        return _json_safe(value.item())

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return float(value)

    if isinstance(value, (str, int, bool)) or value is None:
        return value

    return str(value)


def _validate_path(
    path: str,
    max_file_mb: float = DEFAULT_MAX_FILE_MB,
) -> Path:
    """Validate local FITS path and file size."""

    if not isinstance(path, (str, os.PathLike)):
        raise ChandraValidationError(
            "path must be a string or PathLike object."
        )

    file_path = Path(path).expanduser()

    if not file_path.exists():
        raise ChandraValidationError(
            f"File does not exist: {file_path}"
        )

    if not file_path.is_file():
        raise ChandraValidationError(
            "Input path must point to a file."
        )

    lower_name = file_path.name.lower()

    if not any(lower_name.endswith(ext) for ext in FITS_EXTENSIONS):
        raise ChandraValidationError(
            "Unsupported file extension. Expected a FITS file."
        )

    file_size_mb = file_path.stat().st_size / (1024 ** 2)

    if file_size_mb > float(max_file_mb):
        raise ChandraResourceLimitError(
            f"File size {file_size_mb:.2f} MB exceeds "
            f"limit {max_file_mb:.2f} MB."
        )

    return file_path


def _validate_numeric_array(
    array: Any,
    name: str,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
    max_array_mb: float = DEFAULT_MAX_ARRAY_MB,
) -> np.ndarray:
    """Validate a numeric array without silently accepting objects."""

    try:
        arr = np.asarray(array)
    except Exception as exc:
        raise ChandraValidationError(
            f"Unable to convert {name} to an array."
        ) from exc

    if arr.dtype.kind not in "iuf":
        raise ChandraValidationError(
            f"{name} must contain real numeric values."
        )

    if arr.size > int(max_elements):
        raise ChandraResourceLimitError(
            f"{name} contains too many elements: {arr.size}."
        )

    array_mb = arr.nbytes / (1024 ** 2)

    if array_mb > float(max_array_mb):
        raise ChandraResourceLimitError(
            f"{name} uses {array_mb:.2f} MB; "
            f"limit is {max_array_mb:.2f} MB."
        )

    return arr


def _finite_summary(array: np.ndarray) -> Dict[str, Any]:
    """Return finite-value quality metrics."""

    arr = np.asarray(array)
    finite_mask = np.isfinite(arr)
    finite_count = int(np.count_nonzero(finite_mask))
    total_count = int(arr.size)

    result = {
        "total_count": total_count,
        "finite_count": finite_count,
        "non_finite_count": total_count - finite_count,
        "finite_fraction": (
            finite_count / total_count
            if total_count else 0.0
        ),
    }

    if finite_count:
        finite_values = arr[finite_mask]

        result.update({
            "minimum": float(np.min(finite_values)),
            "maximum": float(np.max(finite_values)),
            "mean": float(np.mean(finite_values)),
            "std": float(np.std(finite_values)),
        })
    else:
        result.update({
            "minimum": None,
            "maximum": None,
            "mean": None,
            "std": None,
        })

    return result


def _header_summary(header) -> Dict[str, Any]:
    """Extract a compact set of useful FITS header fields."""

    keys = (
        "TELESCOP",
        "INSTRUME",
        "DETNAM",
        "GRATING",
        "OBS_ID",
        "OBJECT",
        "DATE-OBS",
        "DATE-END",
        "TIMESYS",
        "MJD-OBS",
        "EXPOSURE",
        "LIVETIME",
        "BUNIT",
        "CTYPE1",
        "CTYPE2",
        "CTYPE3",
        "CUNIT1",
        "CUNIT2",
        "CUNIT3",
        "CRVAL1",
        "CRVAL2",
        "CRVAL3",
        "CRPIX1",
        "CRPIX2",
        "CRPIX3",
        "CDELT1",
        "CDELT2",
        "CDELT3",
    )

    result = {}

    for key in keys:
        if key in header:
            result[key] = _json_safe(header[key])

    return result


def _open_fits(path: Path):
    """Open FITS read-only with lazy HDU loading."""

    fits = _get_fits()

    try:
        return fits.open(
            str(path),
            mode="readonly",
            memmap=not path.name.lower().endswith(".gz"),
            lazy_load_hdus=True,
        )
    except Exception as exc:
        raise ChandraValidationError(
            f"Unable to open FITS file: {exc}"
        ) from exc


def _hdu_name(hdu, index: int) -> str:
    name = getattr(hdu, "name", None)

    if name:
        return str(name).strip().upper()

    return "PRIMARY" if index == 0 else f"HDU_{index}"


def _find_table_hdu(hdul, preferred_name: str = "EVENTS"):
    """Find a binary or ASCII table HDU."""

    preferred_name = preferred_name.upper()

    for index, hdu in enumerate(hdul):
        data = getattr(hdu, "data", None)

        if data is None:
            continue

        if not hasattr(data, "dtype") or data.dtype.names is None:
            continue

        if _hdu_name(hdu, index) == preferred_name:
            return index, hdu

    for index, hdu in enumerate(hdul):
        data = getattr(hdu, "data", None)

        if data is None:
            continue

        if hasattr(data, "dtype") and data.dtype.names:
            return index, hdu

    return None, None


def _find_image_hdu(hdul, preferred_names: Sequence[str]):
    """Find a suitable 2D image HDU."""

    preferred = {
        name.upper()
        for name in preferred_names
    }

    for index, hdu in enumerate(hdul):
        data = getattr(hdu, "data", None)

        if data is None:
            continue

        if np.ndim(data) == 2:
            if _hdu_name(hdu, index) in preferred:
                return index, hdu

    for index, hdu in enumerate(hdul):
        data = getattr(hdu, "data", None)

        if data is not None and np.ndim(data) == 2:
            return index, hdu

    return None, None


# ============================================================
# MAIN ADAPTER
# ============================================================

class ChandraAdapter:
    """
    Bounded FITS reader for Chandra data products.

    Supported products:
    - Event tables
    - 2D images
    - Exposure maps
    - OGIP-style spectrum tables
    """

    def __init__(
        self,
        max_file_mb: float = DEFAULT_MAX_FILE_MB,
        max_array_mb: float = DEFAULT_MAX_ARRAY_MB,
        max_elements: int = DEFAULT_MAX_ELEMENTS,
        max_event_rows: int = DEFAULT_MAX_EVENT_ROWS,
        preview_size: int = DEFAULT_PREVIEW_SIZE,
    ):
        self.max_file_mb = float(max_file_mb)
        self.max_array_mb = float(max_array_mb)
        self.max_elements = int(max_elements)
        self.max_event_rows = int(max_event_rows)
        self.preview_size = int(preview_size)

        if self.max_file_mb <= 0:
            raise ChandraValidationError(
                "max_file_mb must be positive."
            )

        if self.max_array_mb <= 0:
            raise ChandraValidationError(
                "max_array_mb must be positive."
            )

        if self.max_elements < 1:
            raise ChandraValidationError(
                "max_elements must be at least 1."
            )

        if self.max_event_rows < 1:
            raise ChandraValidationError(
                "max_event_rows must be at least 1."
            )

        if self.preview_size < 1:
            raise ChandraValidationError(
                "preview_size must be at least 1."
            )

    def _path(self, path: str) -> Path:
        return _validate_path(
            path,
            max_file_mb=self.max_file_mb,
        )

    # --------------------------------------------------------
    # INSPECT
    # --------------------------------------------------------

    def inspect(self, path: str) -> Dict[str, Any]:
        """Inspect FITS HDUs and selected metadata."""

        file_path = self._path(path)

        with _open_fits(file_path) as hdul:
            hdu_list = []

            for index, hdu in enumerate(hdul):
                data = getattr(hdu, "data", None)

                shape = (
                    list(np.shape(data))
                    if data is not None else None
                )

                columns = []

                if (
                    data is not None
                    and hasattr(data, "dtype")
                    and data.dtype.names
                ):
                    columns = list(data.dtype.names)

                hdu_list.append({
                    "index": index,
                    "name": _hdu_name(hdu, index),
                    "type": type(hdu).__name__,
                    "shape": shape,
                    "columns": columns,
                    "header": _header_summary(hdu.header),
                })

            primary_header = _header_summary(hdul[0].header)

        return {
            "module": MODULE_NAME,
            "version": MODULE_VERSION,
            "observatory": OBSERVATORY,
            "file": file_path.name,
            "file_size_mb": round(
                file_path.stat().st_size / (1024 ** 2), 4
            ),
            "primary_header": primary_header,
            "hdu_count": len(hdu_list),
            "hdus": hdu_list,
            "status": "inspection_complete",
            "scope": "metadata_and_structure_only",
        }

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    def read_image(
        self,
        path: str,
        hdu_name: Optional[str] = None,
        preview: bool = True,
    ) -> Dict[str, Any]:
        """Read a 2D Chandra image."""

        file_path = self._path(path)

        preferred = (
            [hdu_name]
            if hdu_name
            else ["IMAGE", "PRIMARY", "EVENTS_IMAGE"]
        )

        with _open_fits(file_path) as hdul:
            index, hdu = _find_image_hdu(hdul, preferred)

            if hdu is None:
                raise ChandraValidationError(
                    "No 2D image HDU found."
                )

            data = _validate_numeric_array(
                hdu.data,
                "image",
                max_elements=self.max_elements,
                max_array_mb=self.max_array_mb,
            )

            image = np.array(data, dtype=np.float64, copy=True)
            header = _header_summary(hdu.header)
            selected_name = _hdu_name(hdu, index)

        finite = _finite_summary(image)

        result = {
            "module": MODULE_NAME,
            "product_type": "image",
            "file": file_path.name,
            "hdu_index": index,
            "hdu_name": selected_name,
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "image": image,
            "qc": finite,
            "header": header,
            "status": "read_complete",
        }

        if preview:
            stride_y = max(
                1,
                int(np.ceil(image.shape[0] / self.preview_size))
            )
            stride_x = max(
                1,
                int(np.ceil(image.shape[1] / self.preview_size))
            )

            result["preview"] = image[::stride_y, ::stride_x]
            result["preview_shape"] = list(
                result["preview"].shape
            )

        return result

    # --------------------------------------------------------
    # EXPOSURE MAP
    # --------------------------------------------------------

    def read_exposure_map(
        self,
        path: str,
        hdu_name: Optional[str] = None,
        preview: bool = True,
    ) -> Dict[str, Any]:
        """Read a 2D exposure map."""

        result = self.read_image(
            path=path,
            hdu_name=hdu_name or "EXPMAP",
            preview=preview,
        )

        result["product_type"] = "exposure_map"
        result["scope"] = (
            "exposure_map_reading; "
            "no exposure correction applied"
        )

        return result

    # --------------------------------------------------------
    # EVENT TABLE
    # --------------------------------------------------------

    def read_events(
        self,
        path: str,
        columns: Optional[Sequence[str]] = None,
        max_rows: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Read a bounded number of rows from an EVENTS table.

        Common Chandra event columns include:
        TIME, X, Y, DETX, DETY, ENERGY, PI, PHA, GRADE.
        """

        file_path = self._path(path)

        row_limit = (
            self.max_event_rows
            if max_rows is None
            else int(max_rows)
        )

        if row_limit < 1:
            raise ChandraValidationError(
                "max_rows must be at least 1."
            )

        if row_limit > self.max_event_rows:
            raise ChandraResourceLimitError(
                f"Requested {row_limit} rows exceeds "
                f"configured limit {self.max_event_rows}."
            )

        with _open_fits(file_path) as hdul:
            index, hdu = _find_table_hdu(
                hdul,
                preferred_name="EVENTS",
            )

            if hdu is None:
                raise ChandraValidationError(
                    "No EVENTS table found."
                )

            names = list(hdu.data.dtype.names or [])
            lookup = {name.upper(): name for name in names}

            if columns is None:
                selected = [
                    name for name in names
                    if name.upper() in {
                        "TIME", "X", "Y", "DETX", "DETY",
                        "ENERGY", "PI", "PHA", "GRADE",
                    }
                ]
            else:
                selected = []

                for requested in columns:
                    actual = lookup.get(str(requested).upper())

                    if actual is None:
                        raise ChandraValidationError(
                            f"Requested event column not found: "
                            f"{requested}"
                        )

                    selected.append(actual)

            if not selected:
                raise ChandraValidationError(
                    "No supported event columns found."
                )

            total_rows = len(hdu.data)
            rows_to_read = min(total_rows, row_limit)

            event_data = {}

            for name in selected:
                values = np.asarray(
                    hdu.data[name][:rows_to_read]
                )

                if values.dtype.kind not in "iuf":
                    continue

                values = _validate_numeric_array(
                    values,
                    name=f"event column {name}",
                    max_elements=self.max_elements,
                    max_array_mb=self.max_array_mb,
                )

                event_data[name] = np.array(
                    values,
                    copy=True,
                )

            header = _header_summary(hdu.header)
            selected_hdu = _hdu_name(hdu, index)

        if not event_data:
            raise ChandraValidationError(
                "No numeric event columns were available."
            )

        qc = {
            name: _finite_summary(values)
            for name, values in event_data.items()
        }

        return {
            "module": MODULE_NAME,
            "product_type": "events",
            "file": file_path.name,
            "hdu_index": index,
            "hdu_name": selected_hdu,
            "total_rows": int(total_rows),
            "rows_read": int(rows_to_read),
            "truncated": bool(rows_to_read < total_rows),
            "columns": list(event_data.keys()),
            "events": event_data,
            "qc": qc,
            "header": header,
            "status": "read_complete",
            "scope": "bounded_event_table_reading",
        }

    # --------------------------------------------------------
    # PHA / SPECTRUM TABLE
    # --------------------------------------------------------

    def read_spectrum(
        self,
        path: str,
        max_rows: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Read an OGIP-style PHA/SPECTRUM table.

        Common columns:
        CHANNEL, COUNTS, RATE, STAT_ERR.

        This method reads available columns. It does not
        apply response matrices, background subtraction,
        grouping, or spectral fitting.
        """

        file_path = self._path(path)

        row_limit = (
            self.max_event_rows
            if max_rows is None
            else int(max_rows)
        )

        if row_limit < 1:
            raise ChandraValidationError(
                "max_rows must be at least 1."
            )

        if row_limit > self.max_event_rows:
            raise ChandraResourceLimitError(
                "Requested spectrum rows exceed configured limit."
            )

        with _open_fits(file_path) as hdul:
            index, hdu = _find_table_hdu(
                hdul,
                preferred_name="SPECTRUM",
            )

            if hdu is None:
                raise ChandraValidationError(
                    "No SPECTRUM table found."
                )

            names = list(hdu.data.dtype.names or [])
            lookup = {name.upper(): name for name in names}

            aliases = {
                "channel": ("CHANNEL", "PI"),
                "counts": ("COUNTS",),
                "rate": ("RATE",),
                "stat_err": ("STAT_ERR", "ERROR"),
            }

            selected = {}

            for output_name, candidates in aliases.items():
                for candidate in candidates:
                    actual = lookup.get(candidate)

                    if actual is not None:
                        selected[output_name] = actual
                        break

            if "channel" not in selected:
                raise ChandraValidationError(
                    "Spectrum table requires CHANNEL or PI."
                )

            if not any(
                key in selected for key in ("counts", "rate")
            ):
                raise ChandraValidationError(
                    "Spectrum table requires COUNTS or RATE."
                )

            total_rows = len(hdu.data)
            rows_to_read = min(total_rows, row_limit)

            spectrum = {}

            for output_name, source_name in selected.items():
                values = np.asarray(
                    hdu.data[source_name][:rows_to_read]
                )

                if values.dtype.kind not in "iuf":
                    continue

                values = _validate_numeric_array(
                    values,
                    name=f"spectrum {output_name}",
                    max_elements=self.max_elements,
                    max_array_mb=self.max_array_mb,
                )

                spectrum[output_name] = np.array(
                    values,
                    copy=True,
                )

            header = _header_summary(hdu.header)
            selected_hdu = _hdu_name(hdu, index)

        return {
            "module": MODULE_NAME,
            "product_type": "spectrum",
            "file": file_path.name,
            "hdu_index": index,
            "hdu_name": selected_hdu,
            "total_rows": int(total_rows),
            "rows_read": int(rows_to_read),
            "truncated": bool(rows_to_read < total_rows),
            "spectrum": spectrum,
            "qc": {
                name: _finite_summary(values)
                for name, values in spectrum.items()
            },
            "header": header,
            "status": "read_complete",
            "scope": (
                "PHA_table_reading_only; "
                "no_response_or_background_correction"
            ),
        }


# ============================================================
# PUBLIC CONVENIENCE FUNCTIONS
# ============================================================

def module_info() -> Dict[str, Any]:
    return {
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "observatory": OBSERVATORY,
        "supported_products": [
            "FITS metadata",
            "2D image",
            "EVENTS table",
            "PHA/SPECTRUM table",
            "exposure map",
        ],
        "scope": "exploratory_data_ingestion_and_qc",
    }


def inspect_chandra(path: str, **kwargs) -> Dict[str, Any]:
    return ChandraAdapter(**kwargs).inspect(path)


def read_chandra_image(path: str, **kwargs) -> Dict[str, Any]:
    adapter_keys = {
        "max_file_mb", "max_array_mb", "max_elements",
        "max_event_rows", "preview_size",
    }

    adapter_config = {
        key: kwargs.pop(key)
        for key in list(kwargs)
        if key in adapter_keys
    }

    return ChandraAdapter(**adapter_config).read_image(
        path,
        **kwargs,
    )


def read_chandra_events(path: str, **kwargs) -> Dict[str, Any]:
    adapter_keys = {
        "max_file_mb", "max_array_mb", "max_elements",
        "max_event_rows", "preview_size",
    }

    adapter_config = {
        key: kwargs.pop(key)
        for key in list(kwargs)
        if key in adapter_keys
    }

    return ChandraAdapter(**adapter_config).read_events(
        path,
        **kwargs,
    )


def read_chandra_spectrum(path: str, **kwargs) -> Dict[str, Any]:
    adapter_keys = {
        "max_file_mb", "max_array_mb", "max_elements",
        "max_event_rows", "preview_size",
    }

    adapter_config = {
        key: kwargs.pop(key)
        for key in list(kwargs)
        if key in adapter_keys
    }

    return ChandraAdapter(**adapter_config).read_spectrum(
        path,
        **kwargs,
    )


def read_chandra_exposure_map(path: str, **kwargs) -> Dict[str, Any]:
    adapter_keys = {
        "max_file_mb", "max_array_mb", "max_elements",
        "max_event_rows", "preview_size",
    }

    adapter_config = {
        key: kwargs.pop(key)
        for key in list(kwargs)
        if key in adapter_keys
    }

    return ChandraAdapter(**adapter_config).read_exposure_map(
        path,
        **kwargs,
    )


# ============================================================
# CONTRACT TESTS
# ============================================================

def run_contract_tests() -> Dict[str, Any]:
    """
    Run seven local synthetic FITS contract tests.

    No network access and no external Chandra archive required.
    """

    fits = _get_fits()

    import tempfile

    results = []

    def record(name: str, passed: bool, detail: str = ""):
        results.append({
            "test": name,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        })

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        image_path = root / "test_image.fits"
        events_path = root / "test_events.fits"
        spectrum_path = root / "test_spectrum.fits"
        exposure_path = root / "test_exposure.fits"
        invalid_path = root / "invalid.txt"

        # 1. Synthetic image FITS
        image_data = np.arange(
            100,
            dtype=np.float32,
        ).reshape(10, 10)

        primary = fits.PrimaryHDU()
        primary.header["TELESCOP"] = "CHANDRA"
        primary.header["OBS_ID"] = "TEST001"

        image_hdu = fits.ImageHDU(
            data=image_data,
            name="IMAGE",
        )

        fits.HDUList([
            primary,
            image_hdu,
        ]).writeto(image_path)

        # 2. Synthetic EVENTS table
        event_columns = [
            fits.Column(
                name="TIME",
                format="D",
                array=np.arange(20, dtype=np.float64),
            ),
            fits.Column(
                name="X",
                format="E",
                array=np.linspace(1, 20, 20, dtype=np.float32),
            ),
            fits.Column(
                name="Y",
                format="E",
                array=np.linspace(2, 40, 20, dtype=np.float32),
            ),
            fits.Column(
                name="ENERGY",
                format="E",
                array=np.linspace(100, 2000, 20, dtype=np.float32),
            ),
        ]

        events_hdu = fits.BinTableHDU.from_columns(
            event_columns,
            name="EVENTS",
        )

        fits.HDUList([
            fits.PrimaryHDU(),
            events_hdu,
        ]).writeto(events_path)

        # 3. Synthetic PHA/SPECTRUM table
        spectrum_columns = [
            fits.Column(
                name="CHANNEL",
                format="J",
                array=np.arange(10, dtype=np.int32),
            ),
            fits.Column(
                name="COUNTS",
                format="J",
                array=np.arange(10, dtype=np.int32) * 3,
            ),
            fits.Column(
                name="STAT_ERR",
                format="E",
                array=np.sqrt(
                    np.arange(10, dtype=np.float32) * 3
                ),
            ),
        ]

        spectrum_hdu = fits.BinTableHDU.from_columns(
            spectrum_columns,
            name="SPECTRUM",
        )

        fits.HDUList([
            fits.PrimaryHDU(),
            spectrum_hdu,
        ]).writeto(spectrum_path)

        # 4. Synthetic exposure map
        exposure_data = np.full(
            (8, 8),
            1000.0,
            dtype=np.float32,
        )

        exposure_hdu = fits.ImageHDU(
            data=exposure_data,
            name="EXPMAP",
        )

        fits.HDUList([
            fits.PrimaryHDU(),
            exposure_hdu,
        ]).writeto(exposure_path)

        invalid_path.write_text(
            "not a FITS file",
            encoding="utf-8",
        )

        adapter = ChandraAdapter()

        # Test 1: Metadata inspection
        try:
            info = adapter.inspect(str(image_path))
            ok = (
                info["hdu_count"] == 2
                and info["primary_header"].get("TELESCOP")
                == "CHANDRA"
            )
            record("fits_inspection", ok)
        except Exception as exc:
            record("fits_inspection", False, str(exc))

        # Test 2: Image reading and finite QC
        try:
            result = adapter.read_image(str(image_path))
            ok = (
                result["shape"] == [10, 10]
                and result["qc"]["finite_count"] == 100
                and result["preview"].shape[0] <= 512
            )
            record("image_read_and_qc", ok)
        except Exception as exc:
            record("image_read_and_qc", False, str(exc))

        # Test 3: EVENTS table reading
        try:
            result = adapter.read_events(str(events_path))
            ok = (
                result["rows_read"] == 20
                and "TIME" in result["events"]
                and "ENERGY" in result["events"]
            )
            record("events_table_read", ok)
        except Exception as exc:
            record("events_table_read", False, str(exc))

        # Test 4: PHA/SPECTRUM table reading
        try:
            result = adapter.read_spectrum(str(spectrum_path))
            ok = (
                result["rows_read"] == 10
                and "channel" in result["spectrum"]
                and "counts" in result["spectrum"]
            )
            record("spectrum_table_read", ok)
        except Exception as exc:
            record("spectrum_table_read", False, str(exc))

        # Test 5: Exposure map reading
        try:
            result = adapter.read_exposure_map(
                str(exposure_path)
            )
            ok = (
                result["product_type"] == "exposure_map"
                and result["shape"] == [8, 8]
            )
            record("exposure_map_read", ok)
        except Exception as exc:
            record("exposure_map_read", False, str(exc))

        # Test 6: Invalid extension rejection
        try:
            adapter.inspect(str(invalid_path))
            record(
                "invalid_extension_rejection",
                False,
                "Invalid extension was not rejected.",
            )
        except ChandraValidationError:
            record("invalid_extension_rejection", True)
        except Exception as exc:
            record(
                "invalid_extension_rejection",
                False,
                str(exc),
            )

        # Test 7: Missing file rejection
        try:
            adapter.inspect(
                str(root / "missing_file.fits")
            )
            record(
                "missing_file_rejection",
                False,
                "Missing file was not rejected.",
            )
        except ChandraValidationError:
            record("missing_file_rejection", True)
        except Exception as exc:
            record(
                "missing_file_rejection",
                False,
                str(exc),
            )

    passed = sum(
        item["status"] == "PASS"
        for item in results
    )

    failed = len(results) - passed

    return {
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "test_type": "synthetic_contract_tests",
        "network_required": False,
        "total_tests": len(results),
        "passed": passed,
        "failed": failed,
        "overall_status": (
            "PASS" if failed == 0 else "FAIL"
        ),
        "results": results,
    }


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":
    report = run_contract_tests()

    print(
        json.dumps(
            _json_safe(report),
            indent=2,
            ensure_ascii=False,
        )
    )
