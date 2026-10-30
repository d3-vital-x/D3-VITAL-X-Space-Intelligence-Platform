
"""
Module 36: jwst_adapter.py
D³ VITAL-X Space Intelligence Platform

Purpose
-------
- Read calibrated JWST FITS image products.
- Extract one-dimensional JWST spectra.
- Inspect and preview spectral cubes.
- Preserve mission metadata and WCS information.
- Perform finite-value and structural QC.
- Generate bounded-size visualization previews.
- Normalize outputs for downstream D³ VITAL-X processing.

Supported products
------------------
1. Calibrated image
2. Extracted spectrum
3. Spectral cube

Scope
-----
This is a public data adapter, not a JWST calibration pipeline.
It does not perform anomaly detection or confirm scientific validity.
Original scientific arrays are not modified.
Preview sampling is kept separate from source data.
"""

from __future__ import annotations

import json
import logging
import math
import os

from pathlib import Path
from typing import Any, Optional

import numpy as np


MODULE_NAME = "jwst_adapter"
MODULE_VERSION = "1.0.0"

DEFAULT_MAX_FILE_MB = 30
DEFAULT_MAX_ARRAY_MB = 256
DEFAULT_MAX_ELEMENTS = 100_000_000
DEFAULT_MAX_DIMENSIONS = 5
DEFAULT_PREVIEW_SIZE = 512
DEFAULT_MAX_SPECTRUM_POINTS = 100_000

FITS_EXTENSIONS = (
    ".fits",
    ".fit",
    ".fts",
    ".fits.gz",
    ".fit.gz",
    ".fts.gz",
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------
# Exceptions
# ---------------------------------------------------------

class JWSTAdapterError(Exception):
    """Base exception for JWST adapter errors."""


class JWSTValidationError(JWSTAdapterError):
    """Raised for invalid or unsupported JWST data."""


class JWSTResourceLimitError(JWSTAdapterError):
    """Raised when configured resource limits are exceeded."""


class JWSTDependencyError(JWSTAdapterError):
    """Raised when required astronomy dependencies are missing."""


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def _safe_text(value: Any) -> Optional[str]:
    """Convert metadata values into safe plain text."""

    if value is None:
        return None

    try:
        result = str(value).strip()
    except Exception:
        return None

    if not result or result.lower() in {
        "none", "nan", "--", "unknown"
    }:
        return None

    return result


def _safe_float(value: Any) -> Optional[float]:
    """Return a finite float or None."""

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    return result if math.isfinite(result) else None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _fits_extension(path: Path) -> Optional[str]:
    name = path.name.lower()

    for extension in FITS_EXTENSIONS:
        if name.endswith(extension):
            return extension

    return None


def _require_astropy():
    """Import Astropy only when an astronomy operation is requested."""

    try:
        from astropy.io import fits
    except ImportError as exc:
        raise JWSTDependencyError(
            "Astropy is unavailable. Install project requirements."
        ) from exc

    return fits


def _open_fits(path: Path):
    """Open FITS using read-only access and bounded-resource policy."""

    fits = _require_astropy()

    return fits.open(
        path,
        mode="readonly",
        memmap=not path.name.lower().endswith(".gz"),
        lazy_load_hdus=True,
        ignore_missing_end=False,
    )


def _validate_path(
    file_path: str | os.PathLike[str],
    max_file_mb: int = DEFAULT_MAX_FILE_MB,
) -> Path:
    """Validate file existence, format and on-disk size."""

    try:
        path = Path(file_path).expanduser()
    except (TypeError, ValueError) as exc:
        raise JWSTValidationError(
            "Invalid input file path."
        ) from exc

    if not path.exists() or not path.is_file():
        raise JWSTValidationError(
            "JWST FITS file does not exist."
        )

    if _fits_extension(path) is None:
        raise JWSTValidationError(
            "Unsupported FITS file extension."
        )

    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        raise JWSTValidationError(
            "Unable to inspect input file."
        ) from exc

    if size_bytes <= 0:
        raise JWSTValidationError(
            "Input FITS file is empty."
        )

    if size_bytes > max_file_mb * 1024 * 1024:
        raise JWSTResourceLimitError(
            f"File exceeds the {max_file_mb} MB limit."
        )

    return path


def _validate_array(
    data: Any,
    max_array_mb: int = DEFAULT_MAX_ARRAY_MB,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
    max_dimensions: int = DEFAULT_MAX_DIMENSIONS,
) -> np.ndarray:
    """Validate array metadata before materializing a numeric array."""

    shape = tuple(int(x) for x in data.shape)

    if not shape:
        raise JWSTValidationError(
            "Empty or scalar FITS data is unsupported."
        )

    if len(shape) > max_dimensions:
        raise JWSTResourceLimitError(
            "Array exceeds the dimensionality limit."
        )

    elements = math.prod(shape)

    if elements <= 0 or elements > max_elements:
        raise JWSTResourceLimitError(
            "Array exceeds the configured element limit."
        )

    dtype = np.dtype(data.dtype)

    if dtype.kind not in "iuf":
        raise JWSTValidationError(
            "Unsupported non-numeric FITS array."
        )

    estimated_bytes = elements * dtype.itemsize

    if estimated_bytes > max_array_mb * 1024 * 1024:
        raise JWSTResourceLimitError(
            "Estimated array memory exceeds configured limit."
        )

    try:
        array = np.asarray(data)
    except (MemoryError, ValueError, OSError) as exc:
        raise JWSTResourceLimitError(
            "Unable to materialize the requested FITS array."
        ) from exc

    if array.dtype.kind not in "iuf":
        raise JWSTValidationError(
            "Unsupported array data type."
        )

    return array


def _finite_report(array: np.ndarray) -> dict[str, Any]:
    """Summarize finite and non-finite values."""

    total = int(array.size)

    if total == 0:
        return {
            "total_values": 0,
            "finite_values": 0,
            "non_finite_values": 0,
            "finite_fraction": None,
            "status": "EMPTY",
        }

    finite_mask = np.isfinite(array)
    finite_count = int(np.count_nonzero(finite_mask))
    non_finite_count = total - finite_count

    return {
        "total_values": total,
        "finite_values": finite_count,
        "non_finite_values": non_finite_count,
        "finite_fraction": round(
            finite_count / total, 8
        ),
        "status": (
            "PASS" if non_finite_count == 0
            else "WARNING"
        ),
    }


def _finite_statistics(array: np.ndarray) -> dict[str, Any]:
    """Calculate descriptive statistics using finite values only."""

    mask = np.isfinite(array)

    if not np.any(mask):
        return {
            "minimum": None,
            "maximum": None,
            "mean": None,
            "median": None,
            "standard_deviation": None,
        }

    values = array[mask].astype(np.float64, copy=False)

    return {
        "minimum": _safe_float(np.min(values)),
        "maximum": _safe_float(np.max(values)),
        "mean": _safe_float(np.mean(values)),
        "median": _safe_float(np.median(values)),
        "standard_deviation": _safe_float(
            np.std(values)
        ),
    }


def _metadata_from_header(header: Any) -> dict[str, Any]:
    """Extract commonly useful JWST FITS header metadata."""

    keys = (
        "TELESCOP",
        "INSTRUME",
        "DETECTOR",
        "FILTER",
        "GRATING",
        "EXP_TYPE",
        "DATE-OBS",
        "TIME-OBS",
        "MJD-OBS",
        "EXPTIME",
        "TARGNAME",
        "OBJECT",
        "OBS_ID",
        "OBSERVTN",
        "PROPOSID",
        "PROGRAM",
        "CAL_VER",
        "DATAMODL",
        "BUNIT",
        "CHANNEL",
        "BAND",
        "SUBARRAY",
    )

    result = {}

    for key in keys:
        value = _safe_text(header.get(key))

        if value is not None:
            result[key] = value

    return result


def _wcs_metadata(header: Any) -> dict[str, Any]:
    """Extract WCS axis information without altering source data."""

    keys = (
        "WCSAXES",
        "CTYPE1", "CTYPE2", "CTYPE3",
        "CUNIT1", "CUNIT2", "CUNIT3",
        "CRVAL1", "CRVAL2", "CRVAL3",
        "CRPIX1", "CRPIX2", "CRPIX3",
        "CDELT1", "CDELT2", "CDELT3",
        "CD1_1", "CD2_2",
        "PC1_1", "PC2_2",
    )

    result = {}

    for key in keys:
        value = _safe_text(header.get(key))

        if value is not None:
            result[key] = value

    result["available"] = bool(result)

    return result


def _preview_2d(
    array: np.ndarray,
    max_size: int = DEFAULT_PREVIEW_SIZE,
) -> np.ndarray:
    """
    Create a bounded 2D visualization preview.

    This is visualization-only. It is not used for scientific
    calculations or original-array replacement.
    """

    if array.ndim != 2:
        raise JWSTValidationError(
            "A two-dimensional array is required for image preview."
        )

    if max_size < 1:
        raise ValueError("max_size must be positive.")

    height, width = array.shape

    row_step = max(1, math.ceil(height / max_size))
    col_step = max(1, math.ceil(width / max_size))

    preview = array[::row_step, ::col_step]

    return np.array(preview, copy=True)


def _header_wavelength_axis(
    header: Any,
    length: int,
) -> Optional[np.ndarray]:
    """Build a linear wavelength axis when valid WCS cards exist."""

    ctype = str(header.get("CTYPE1", "")).upper()

    if not any(
        token in ctype
        for token in ("WAVE", "AWAV", "FREQ")
    ):
        return None

    crval = _safe_float(header.get("CRVAL1"))
    crpix = _safe_float(header.get("CRPIX1"))
    cdelt = _safe_float(
        header.get("CDELT1", header.get("CD1_1"))
    )

    if None in (crval, crpix, cdelt):
        return None

    pixels = np.arange(length, dtype=np.float64) + 1.0

    return crval + (pixels - crpix) * cdelt


def _find_column(names: list[str], candidates: tuple[str, ...]):
    """Find a column using case-insensitive candidate names."""

    lookup = {
        str(name).strip().upper(): name
        for name in names
    }

    for candidate in candidates:
        if candidate.upper() in lookup:
            return lookup[candidate.upper()]

    return None


# ---------------------------------------------------------
# JWST Adapter
# ---------------------------------------------------------

class JWSTAdapter:
    """
    Public JWST FITS adapter.

    Parameters
    ----------
    max_file_mb:
        Maximum permitted input file size.
    max_array_mb:
        Maximum estimated size of one materialized array.
    max_elements:
        Maximum array element count.
    max_dimensions:
        Maximum permitted array dimensions.
    preview_size:
        Maximum preview side length in pixels.
    """

    def __init__(
        self,
        max_file_mb: int = DEFAULT_MAX_FILE_MB,
        max_array_mb: int = DEFAULT_MAX_ARRAY_MB,
        max_elements: int = DEFAULT_MAX_ELEMENTS,
        max_dimensions: int = DEFAULT_MAX_DIMENSIONS,
        preview_size: int = DEFAULT_PREVIEW_SIZE,
    ) -> None:

        for name, value in {
            "max_file_mb": max_file_mb,
            "max_array_mb": max_array_mb,
            "max_elements": max_elements,
            "max_dimensions": max_dimensions,
            "preview_size": preview_size,
        }.items():

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be a positive integer."
                )

        self.max_file_mb = max_file_mb
        self.max_array_mb = max_array_mb
        self.max_elements = max_elements
        self.max_dimensions = max_dimensions
        self.preview_size = preview_size

    @staticmethod
    def module_info() -> dict[str, Any]:
        return {
            "module": MODULE_NAME,
            "version": MODULE_VERSION,
            "mission": "JWST",
            "archive": "MAST",
            "supported_products": [
                "calibrated_image",
                "extracted_spectrum",
                "spectral_cube",
            ],
            "supported_formats": list(FITS_EXTENSIONS),
            "scientific_calibration": False,
            "scientific_validity_confirmed": False,
            "anomaly_detection": False,
            "preview_is_scientific_data": False,
        }

    def inspect(
        self,
        file_path: str | os.PathLike[str],
    ) -> dict[str, Any]:
        """Return structural and JWST-specific metadata."""

        path = _validate_path(
            file_path,
            self.max_file_mb,
        )

        try:
            from fits_inspector import FITSInspector

            inspector = FITSInspector(
                max_file_mb=self.max_file_mb,
                max_dimensions=self.max_dimensions,
                max_elements=self.max_elements,
            )

            report = inspector.inspect(path)

        except ImportError:
            # Structural inspection remains available below.
            report = None

        except Exception as exc:
            raise JWSTValidationError(
                "FITS structural inspection failed: "
                + type(exc).__name__
            ) from exc

        with _open_fits(path) as hdul:
            if len(hdul) == 0:
                raise JWSTValidationError(
                    "No FITS HDUs found."
                )

            primary_header = hdul[0].header
            metadata = _metadata_from_header(primary_header)

            if not metadata.get("TELESCOP"):
                for hdu in hdul[1:]:
                    metadata.update(
                        _metadata_from_header(hdu.header)
                    )
                    if metadata.get("TELESCOP"):
                        break

            warnings = []

            if metadata.get("TELESCOP", "").upper() != "JWST":
                warnings.append(
                    "TELESCOP does not explicitly identify JWST."
                )

            if report is not None:
                warnings.extend(report.get("warnings", []))

            return {
                "module": MODULE_NAME,
                "version": MODULE_VERSION,
                "mission": "JWST",
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "hdu_count": len(hdul),
                "mission_metadata": metadata,
                "fits_inspection": report,
                "warnings": warnings,
                "scientific_calibration": False,
                "scientific_validity_confirmed": False,
            }

    def _select_image_hdu(self, hdul: Any, hdu_index: Optional[int]):
        """Select an image HDU without assuming HDU zero contains data."""

        if hdu_index is not None:
            if (
                isinstance(hdu_index, bool)
                or not isinstance(hdu_index, int)
                or hdu_index < 0
                or hdu_index >= len(hdul)
            ):
                raise JWSTValidationError(
                    "Invalid image HDU index."
                )

            hdu = hdul[hdu_index]

            if getattr(hdu, "data", None) is None:
                raise JWSTValidationError(
                    "Selected HDU contains no image data."
                )

            return hdu_index, hdu

        # Prefer a calibrated SCI extension.
        for index, hdu in enumerate(hdul):
            if str(getattr(hdu, "name", "")).upper() == "SCI":
                if getattr(hdu, "data", None) is not None:
                    return index, hdu

        # Otherwise select the first 2D image.
        for index, hdu in enumerate(hdul):
            shape = getattr(hdu, "shape", None)

            if shape is not None and len(shape) == 2:
                return index, hdu

        raise JWSTValidationError(
            "No suitable two-dimensional image HDU found."
        )

    def read_image(
        self,
        file_path: str | os.PathLike[str],
        hdu_index: Optional[int] = None,
    ) -> dict[str, Any]:
        """Read a calibrated JWST image and create a preview."""

        path = _validate_path(
            file_path,
            self.max_file_mb,
        )

        with _open_fits(path) as hdul:
            index, hdu = self._select_image_hdu(
                hdul, hdu_index
            )

            raw = getattr(hdu, "data", None)

            if raw is None:
                raise JWSTValidationError(
                    "Selected image HDU has no data."
                )

            data = _validate_array(
                raw,
                self.max_array_mb,
                self.max_elements,
                self.max_dimensions,
            )

            if data.ndim != 2:
                raise JWSTValidationError(
                    "Calibrated image must be two-dimensional."
                )

            header = hdu.header.copy()
            metadata = _metadata_from_header(hdul[0].header)
            metadata.update(_metadata_from_header(header))

            wcs = _wcs_metadata(header)

            qc = _finite_report(data)
            statistics = _finite_statistics(data)

            preview = _preview_2d(
                data,
                self.preview_size,
            )

            return {
                "module": MODULE_NAME,
                "version": MODULE_VERSION,
                "mission": "JWST",
                "product_type": "calibrated_image",
                "filename": path.name,
                "hdu_index": index,
                "hdu_name": str(
                    getattr(hdu, "name", "")
                ),
                "data": data,
                "shape": list(data.shape),
                "ndim": data.ndim,
                "dtype": str(data.dtype),
                "unit": _safe_text(header.get("BUNIT")),
                "metadata": metadata,
                "wcs": wcs,
                "qc": qc,
                "statistics": statistics,
                "preview": preview,
                "preview_shape": list(preview.shape),
                "preview_is_scientific_data": False,
                "warnings": (
                    [] if qc["status"] == "PASS"
                    else ["Non-finite image pixels were found."]
                ),
                "scientific_calibration": False,
                "scientific_validity_confirmed": False,
            }

    def read_spectrum(
        self,
        file_path: str | os.PathLike[str],
        hdu_index: Optional[int] = None,
        wavelength_column: Optional[str] = None,
        flux_column: Optional[str] = None,
        error_column: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Extract an X1D-like spectrum from a FITS binary table.

        Common column candidates include:
        WAVELENGTH, WAVE, LAMBDA, FLUX, NET, ERROR, ERR.
        """

        path = _validate_path(
            file_path,
            self.max_file_mb,
        )

        with _open_fits(path) as hdul:

            selected = None

            if hdu_index is not None:
                if (
                    isinstance(hdu_index, bool)
                    or not isinstance(hdu_index, int)
                    or hdu_index < 0
                    or hdu_index >= len(hdul)
                ):
                    raise JWSTValidationError(
                        "Invalid spectrum HDU index."
                    )

                selected = hdu_index

            else:
                # Prefer an extracted spectrum table.
                for index, hdu in enumerate(hdul):
                    names = getattr(
                        getattr(hdu, "columns", None),
                        "names",
                        None,
                    )

                    if names:
                        upper = {
                            str(name).upper()
                            for name in names
                        }

                        if (
                            "WAVELENGTH" in upper
                            or "WAVE" in upper
                            or "LAMBDA" in upper
                        ):
                            selected = index
                            break

            if selected is None:
                raise JWSTValidationError(
                    "No spectrum table with a recognizable "
                    "wavelength column was found."
                )

            hdu = hdul[selected]
            table = getattr(hdu, "data", None)

            if table is None or getattr(hdu, "columns", None) is None:
                raise JWSTValidationError(
                    "Selected HDU is not a populated spectrum table."
                )

            names = list(hdu.columns.names or [])

            wave_name = (
                wavelength_column
                or _find_column(
                    names,
                    ("WAVELENGTH", "WAVE", "LAMBDA"),
                )
            )

            flux_name = (
                flux_column
                or _find_column(
                    names,
                    ("FLUX", "NET", "SPEC"),
                )
            )

            err_name = (
                error_column
                or _find_column(
                    names,
                    ("ERROR", "ERR", "FLUX_ERROR"),
                )
            )

            if wave_name is None or flux_name is None:
                raise JWSTValidationError(
                    "Wavelength and flux columns could not "
                    "be identified. Supply explicit column names."
                )

            if wave_name not in names or flux_name not in names:
                raise JWSTValidationError(
                    "Requested spectrum column does not exist."
                )

            if err_name is not None and err_name not in names:
                raise JWSTValidationError(
                    "Requested error column does not exist."
                )

            row_count = len(table)

            if row_count <= 0:
                raise JWSTValidationError(
                    "Spectrum table is empty."
                )

            if row_count > self.max_elements:
                raise JWSTResourceLimitError(
                    "Spectrum row count exceeds configured limit."
                )

            try:
                wavelength = np.asarray(
                    table[wave_name],
                    dtype=np.float64,
                ).copy()

                flux = np.asarray(
                    table[flux_name],
                    dtype=np.float64,
                ).copy()

                error = (
                    np.asarray(
                        table[err_name],
                        dtype=np.float64,
                    ).copy()
                    if err_name is not None
                    else None
                )

            except (TypeError, ValueError, MemoryError) as exc:
                raise JWSTValidationError(
                    "Unable to extract numeric spectrum columns."
                ) from exc

            if wavelength.ndim != 1 or flux.ndim != 1:
                raise JWSTValidationError(
                    "Spectrum columns must be one-dimensional."
                )

            if wavelength.size != flux.size:
                raise JWSTValidationError(
                    "Wavelength and flux lengths do not match."
                )

            if error is not None and (
                error.ndim != 1 or error.size != flux.size
            ):
                raise JWSTValidationError(
                    "Error column length does not match flux."
                )

            wave_finite = np.isfinite(wavelength)
            flux_finite = np.isfinite(flux)

            valid_wave = wave_finite & (wavelength > 0)

            warnings = []

            if not np.all(wave_finite):
                warnings.append(
                    "Non-finite wavelength values were found."
                )

            if not np.all(flux_finite):
                warnings.append(
                    "Non-finite flux values were found."
                )

            if not np.all(valid_wave):
                warnings.append(
                    "Non-positive or invalid wavelength values were found."
                )

            if wavelength.size > 1:
                differences = np.diff(wavelength[wave_finite])

                if differences.size and not (
                    np.all(differences > 0)
                    or np.all(differences < 0)
                ):
                    warnings.append(
                        "Wavelength axis is not strictly monotonic."
                    )

            header = hdu.header.copy()
            metadata = _metadata_from_header(hdul[0].header)
            metadata.update(_metadata_from_header(header))

            qc = {
                "total_points": int(wavelength.size),
                "finite_wavelength_points": int(
                    np.count_nonzero(wave_finite)
                ),
                "finite_flux_points": int(
                    np.count_nonzero(flux_finite)
                ),
                "positive_wavelength_points": int(
                    np.count_nonzero(valid_wave)
                ),
                "status": (
                    "PASS" if not warnings else "WARNING"
                ),
            }

            return {
                "module": MODULE_NAME,
                "version": MODULE_VERSION,
                "mission": "JWST",
                "product_type": "extracted_spectrum",
                "filename": path.name,
                "hdu_index": selected,
                "wavelength": wavelength,
                "flux": flux,
                "error": error,
                "wavelength_column": str(wave_name),
                "flux_column": str(flux_name),
                "error_column": (
                    str(err_name) if err_name is not None
                    else None
                ),
                "wavelength_unit": _safe_text(
                    hdu.columns[wave_name].unit
                ),
                "flux_unit": _safe_text(
                    hdu.columns[flux_name].unit
                ),
                "metadata": metadata,
                "qc": qc,
                "flux_statistics": _finite_statistics(flux),
                "warnings": warnings,
                "scientific_calibration": False,
                "scientific_validity_confirmed": False,
            }

    def read_cube(
        self,
        file_path: str | os.PathLike[str],
        hdu_index: Optional[int] = None,
        slice_index: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Inspect a spectral cube and extract one selected 2D slice.

        A 3D cube is expected in NumPy order:
        (spectral_axis, y, x).

        Axis interpretation must be verified from the FITS header.
        """

        path = _validate_path(
            file_path,
            self.max_file_mb,
        )

        with _open_fits(path) as hdul:

            if hdu_index is None:
                selected = None

                for index, hdu in enumerate(hdul):
                    shape = getattr(hdu, "shape", None)

                    if shape is not None and len(shape) == 3:
                        selected = index
                        break

                if selected is None:
                    raise JWSTValidationError(
                        "No three-dimensional spectral cube found."
                    )

            else:
                if (
                    isinstance(hdu_index, bool)
                    or not isinstance(hdu_index, int)
                    or hdu_index < 0
                    or hdu_index >= len(hdul)
                ):
                    raise JWSTValidationError(
                        "Invalid cube HDU index."
                    )

                selected = hdu_index

            hdu = hdul[selected]
            raw = getattr(hdu, "data", None)

            if raw is None:
                raise JWSTValidationError(
                    "Selected cube HDU has no data."
                )

            shape = tuple(int(x) for x in raw.shape)

            if len(shape) != 3:
                raise JWSTValidationError(
                    "Spectral cube must have exactly three dimensions."
                )

            spectral_length = shape[0]

            if slice_index is None:
                slice_index = spectral_length // 2

            if (
                isinstance(slice_index, bool)
                or not isinstance(slice_index, int)
                or slice_index < 0
                or slice_index >= spectral_length
            ):
                raise JWSTValidationError(
                    "Requested spectral slice is out of range."
                )

            cube = _validate_array(
                raw,
                self.max_array_mb,
                self.max_elements,
                self.max_dimensions,
            )

            selected_slice = np.array(
                cube[slice_index, :, :],
                copy=True,
            )

            preview = _preview_2d(
                selected_slice,
                self.preview_size,
            )

            header = hdu.header.copy()

            metadata = _metadata_from_header(hdul[0].header)
            metadata.update(_metadata_from_header(header))

            wcs = _wcs_metadata(header)

            spectral_axis = _header_wavelength_axis(
                header,
                spectral_length,
            )

            qc = _finite_report(cube)

            warnings = []

            if spectral_axis is None:
                warnings.append(
                    "A reliable linear spectral axis could not "
                    "be constructed from the available header cards."
                )

            if qc["status"] != "PASS":
                warnings.append(
                    "Cube contains non-finite values."
                )

            return {
                "module": MODULE_NAME,
                "version": MODULE_VERSION,
                "mission": "JWST",
                "product_type": "spectral_cube",
                "filename": path.name,
                "hdu_index": selected,
                "data": cube,
                "shape": list(cube.shape),
                "ndim": cube.ndim,
                "dtype": str(cube.dtype),
                "metadata": metadata,
                "wcs": wcs,
                "spectral_axis": spectral_axis,
                "spectral_axis_unit": _safe_text(
                    header.get("CUNIT1")
                ),
                "selected_slice_index": slice_index,
                "selected_slice": selected_slice,
                "preview": preview,
                "preview_is_scientific_data": False,
                "qc": qc,
                "statistics": _finite_statistics(cube),
                "warnings": warnings,
                "scientific_calibration": False,
                "scientific_validity_confirmed": False,
            }


# ---------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------

def module_info() -> dict[str, Any]:
    return JWSTAdapter.module_info()


def inspect_jwst(
    file_path: str | os.PathLike[str],
) -> dict[str, Any]:
    return JWSTAdapter().inspect(file_path)


def read_jwst_image(
    file_path: str | os.PathLike[str],
    hdu_index: Optional[int] = None,
) -> dict[str, Any]:
    return JWSTAdapter().read_image(
        file_path,
        hdu_index=hdu_index,
    )


def read_jwst_spectrum(
    file_path: str | os.PathLike[str],
    hdu_index: Optional[int] = None,
) -> dict[str, Any]:
    return JWSTAdapter().read_spectrum(
        file_path,
        hdu_index=hdu_index,
    )


def read_jwst_cube(
    file_path: str | os.PathLike[str],
    hdu_index: Optional[int] = None,
    slice_index: Optional[int] = None,
) -> dict[str, Any]:
    return JWSTAdapter().read_cube(
        file_path,
        hdu_index=hdu_index,
        slice_index=slice_index,
    )


# ---------------------------------------------------------
# Offline contract tests
# ---------------------------------------------------------

def run_contract_tests() -> dict[str, Any]:
    """
    Test the adapter with small synthetic JWST-like FITS files.

    No network requests or archive downloads are performed.
    """

    import tempfile

    fits = _require_astropy()

    passed = 0

    adapter = JWSTAdapter(
        max_file_mb=5,
        max_array_mb=16,
        max_elements=1_000_000,
        max_dimensions=5,
        preview_size=32,
    )

    with tempfile.TemporaryDirectory() as directory:

        root = Path(directory)

        # Test 1: calibrated image
        image_path = root / "test_image.fits"

        primary = fits.PrimaryHDU(
            data=np.arange(
                400, dtype=np.float32
            ).reshape(20, 20)
        )

        primary.header["TELESCOP"] = "JWST"
        primary.header["INSTRUME"] = "NIRCAM"
        primary.header["FILTER"] = "F200W"
        primary.header["BUNIT"] = "MJy/sr"

        primary.writeto(image_path)

        image_result = adapter.read_image(image_path)

        assert image_result["product_type"] == "calibrated_image"
        assert image_result["shape"] == [20, 20]
        assert image_result["preview"].shape == (20, 20)
        assert image_result["metadata"]["TELESCOP"] == "JWST"
        assert image_result["qc"]["status"] == "PASS"

        passed += 1

        # Test 2: extracted spectrum
        spectrum_path = root / "test_x1d.fits"

        wavelength = np.linspace(1.0, 5.0, 100)
        flux = np.linspace(0.1, 1.0, 100)
        error = np.full(100, 0.02)

        columns = [
            fits.Column(
                name="WAVELENGTH",
                format="D",
                unit="um",
                array=wavelength,
            ),
            fits.Column(
                name="FLUX",
                format="D",
                unit="Jy",
                array=flux,
            ),
            fits.Column(
                name="ERROR",
                format="D",
                unit="Jy",
                array=error,
            ),
        ]

        primary = fits.PrimaryHDU()
        primary.header["TELESCOP"] = "JWST"

        table = fits.BinTableHDU.from_columns(
            columns,
            name="EXTRACT1D",
        )

        fits.HDUList(
            [primary, table]
        ).writeto(spectrum_path)

        spectrum_result = adapter.read_spectrum(
            spectrum_path
        )

        assert spectrum_result["product_type"] == "extracted_spectrum"
        assert spectrum_result["wavelength"].size == 100
        assert spectrum_result["flux"].size == 100
        assert spectrum_result["error"].size == 100
        assert spectrum_result["wavelength_unit"] == "um"

        passed += 1

        # Test 3: spectral cube
        cube_path = root / "test_cube.fits"

        cube = np.arange(
            8 * 10 * 12,
            dtype=np.float32,
        ).reshape(8, 10, 12)

        primary = fits.PrimaryHDU(data=cube)
        primary.header["TELESCOP"] = "JWST"
        primary.header["CTYPE1"] = "WAVE"
        primary.header["CUNIT1"] = "um"
        primary.header["CRVAL1"] = 1.0
        primary.header["CRPIX1"] = 1.0
        primary.header["CDELT1"] = 0.1

        primary.writeto(cube_path)

        cube_result = adapter.read_cube(
            cube_path,
            slice_index=3,
        )

        assert cube_result["product_type"] == "spectral_cube"
        assert cube_result["shape"] == [8, 10, 12]
        assert cube_result["selected_slice_index"] == 3
        assert cube_result["selected_slice"].shape == (10, 12)
        assert cube_result["spectral_axis"].size == 8

        passed += 1

        # Test 4: invalid extension
        invalid_path = root / "invalid.csv"
        invalid_path.write_text(
            "a,b\n1,2\n",
            encoding="utf-8",
        )

        try:
            adapter.inspect(invalid_path)
        except JWSTValidationError:
            passed += 1
        else:
            raise AssertionError(
                "Unsupported file extension was accepted."
            )

        # Test 5: missing file
        try:
            adapter.inspect(root / "missing.fits")
        except JWSTValidationError:
            passed += 1
        else:
            raise AssertionError(
                "Missing file was accepted."
            )

        # Test 6: invalid cube slice
        try:
            adapter.read_cube(
                cube_path,
                slice_index=100,
            )
        except JWSTValidationError:
            passed += 1
        else:
            raise AssertionError(
                "Out-of-range cube slice was accepted."
            )

        # Test 7: finite-value QC
        qc_array = np.array(
            [1.0, 2.0, np.nan, np.inf]
        )

        qc_result = _finite_report(qc_array)

        assert qc_result["finite_values"] == 2
        assert qc_result["non_finite_values"] == 2
        assert qc_result["status"] == "WARNING"

        passed += 1

    return {
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "status": "PASS",
        "tests_passed": passed,
        "tests_expected": 7,
        "network_used": False,
        "synthetic_data_used": True,
        "scope": "Synthetic JWST-like FITS contract tests",
        "scientific_validity_confirmed": False,
        "note": (
            "Passing synthetic contract tests does not establish "
            "compatibility with every JWST instrument or product."
        ),
    }


if __name__ == "__main__":

    print(
        json.dumps(
            {
                "module_info": module_info(),
                "contract_tests": run_contract_tests(),
            },
            indent=2,
            default=str,
        )
    )
