
"""
Module 33: mission_data_client.py
D³ VITAL-X Space Intelligence Platform

Purpose
-------
- Search public JWST observations through MAST.
- Normalize archive results into plain Python dictionaries.
- Download one explicitly selected MAST product.
- Enforce a configurable maximum download size.
- Provide a clear extension point for Chandra archive integration.

Scope
-----
This is an archive-access client, not a scientific calibration pipeline.
Chandra archive search/download is intentionally not claimed as implemented.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile

from pathlib import Path
from typing import Any, Optional


MODULE_NAME = "mission_data_client"
MODULE_VERSION = "1.0.0"

DEFAULT_MAX_DOWNLOAD_MB = 30
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RESULTS = 50

LOGGER = logging.getLogger(__name__)


class MissionDataError(Exception):
    """Base exception for mission data access errors."""


class UnsupportedMissionError(MissionDataError):
    """Raised when a mission backend is not implemented."""


class ArchiveUnavailableError(MissionDataError):
    """Raised when an archive cannot be reached or queried."""


class DownloadLimitError(MissionDataError):
    """Raised when a download exceeds the configured size limit."""


class InvalidProductError(MissionDataError):
    """Raised when a selected archive product is invalid."""


def _safe_int(value: Any, default: int = 0) -> int:
    """Convert a value to int without propagating conversion errors."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_text(value: Any) -> Optional[str]:
    """Convert archive metadata to a safe plain-text value."""
    if value is None:
        return None

    try:
        text = str(value).strip()
    except Exception:
        return None

    if not text or text.lower() in {"none", "nan", "--"}:
        return None

    return text


def _row_value(row: Any, key: str) -> Any:
    """Read a field from an Astropy Table row or mapping."""
    try:
        return row[key]
    except (KeyError, TypeError, IndexError, ValueError):
        return None


def _safe_filename(filename: str) -> str:
    """Keep only a filename, never a caller-supplied directory path."""
    name = str(filename).replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._+-]", "_", name)

    if name in {"", ".", ".."}:
        raise InvalidProductError("Invalid product filename.")

    return name


def _sha256_file(path: Path) -> str:
    """Calculate a file checksum using bounded-memory reads."""
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


class MissionDataClient:
    """
    Public archive client for D³ VITAL-X.

    Parameters
    ----------
    max_download_mb:
        Maximum permitted size of a single downloaded file.
    timeout_seconds:
        Network timeout used by supported archive clients.
    max_results:
        Maximum number of observations returned by a search.
    download_dir:
        Optional destination directory for downloaded products.

    Notes
    -----
    This limit applies to individual downloaded files. It is not
    a guarantee of total process RAM usage.
    """

    SUPPORTED_MISSIONS = ("JWST", "CHANDRA")

    def __init__(
        self,
        max_download_mb: int = DEFAULT_MAX_DOWNLOAD_MB,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_results: int = DEFAULT_MAX_RESULTS,
        download_dir: Optional[str] = None,
    ) -> None:

        if not 1 <= max_download_mb <= 30:
            raise ValueError(
                "max_download_mb must be between 1 and 30."
            )

        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive.")

        if max_results < 1:
            raise ValueError("max_results must be positive.")

        self.max_download_bytes = (
            max_download_mb * 1024 * 1024
        )
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results

        self.download_dir = (
            Path(download_dir).expanduser()
            if download_dir
            else Path(tempfile.gettempdir())
            / "d3_vital_x_mission_data"
        )

    @staticmethod
    def module_info() -> dict[str, Any]:
        """Return module identity and supported capabilities."""
        return {
            "module": MODULE_NAME,
            "version": MODULE_VERSION,
            "purpose": "Public astronomy archive access",
            "supported_missions": list(
                MissionDataClient.SUPPORTED_MISSIONS
            ),
            "implemented_backends": {
                "JWST": "MAST via astroquery",
                "CHANDRA": "Not yet implemented",
            },
            "scientific_calibration": False,
            "anomaly_detection": False,
        }

    def search_jwst(
        self,
        target: Optional[str] = None,
        proposal_id: Optional[str] = None,
        instrument: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Search public JWST observations in MAST.

        At least one search criterion is required.
        Returns normalized observation metadata, not data files.
        """

        criteria: dict[str, Any] = {
            "obs_collection": "JWST",
            "dataRights": "PUBLIC",
        }

        if target:
            target = str(target).strip()
            if not target:
                raise ValueError("target cannot be blank.")
            criteria["target_name"] = target

        if proposal_id:
            proposal_id = str(proposal_id).strip()
            if not proposal_id:
                raise ValueError("proposal_id cannot be blank.")
            criteria["proposal_id"] = proposal_id

        if instrument:
            instrument = str(instrument).strip()
            if not instrument:
                raise ValueError("instrument cannot be blank.")
            criteria["instrument_name"] = instrument

        if not any(
            key in criteria
            for key in (
                "target_name",
                "proposal_id",
                "instrument_name",
            )
        ):
            raise ValueError(
                "Provide target, proposal_id, or instrument."
            )

        result_limit = min(
            self.max_results,
            max(1, _safe_int(limit, self.max_results)),
        )

        try:
            from astroquery.mast import Observations

            observations = Observations.query_criteria(
                **criteria
            )

        except ImportError as exc:
            raise ArchiveUnavailableError(
                "astroquery is unavailable. Install the "
                "requirements.txt dependencies."
            ) from exc

        except Exception as exc:
            LOGGER.warning(
                "MAST observation query failed: %s",
                type(exc).__name__,
            )
            raise ArchiveUnavailableError(
                "MAST query failed. Check network access and "
                "search criteria, then try again."
            ) from exc

        results: list[dict[str, Any]] = []

        for row in observations[:result_limit]:
            results.append({
                "mission": "JWST",
                "archive": "MAST",
                "obsid": _safe_text(
                    _row_value(row, "obsid")
                ),
                "observation_id": _safe_text(
                    _row_value(row, "obs_id")
                ),
                "target": _safe_text(
                    _row_value(row, "target_name")
                ),
                "instrument": _safe_text(
                    _row_value(row, "instrument_name")
                ),
                "proposal_id": _safe_text(
                    _row_value(row, "proposal_id")
                ),
                "dataproduct_type": _safe_text(
                    _row_value(row, "dataproduct_type")
                ),
                "data_rights": _safe_text(
                    _row_value(row, "dataRights")
                ),
            })

        return results

    def list_jwst_products(
        self,
        obsid: str,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        List products associated with a MAST observation group ID.

        Important: obsid must be the MAST product-group identifier,
        not necessarily the mission-specific observation ID.
        """

        obsid = str(obsid).strip()

        if not obsid:
            raise ValueError("obsid cannot be blank.")

        result_limit = min(
            self.max_results,
            max(1, _safe_int(limit, self.max_results)),
        )

        try:
            from astroquery.mast import Observations

            products = Observations.get_product_list(obsid)

        except ImportError as exc:
            raise ArchiveUnavailableError(
                "astroquery is unavailable."
            ) from exc

        except Exception as exc:
            LOGGER.warning(
                "MAST product lookup failed: %s",
                type(exc).__name__,
            )
            raise ArchiveUnavailableError(
                "Could not retrieve MAST products for this obsid."
            ) from exc

        results: list[dict[str, Any]] = []

        for row in products[:result_limit]:
            results.append({
                "mission": "JWST",
                "archive": "MAST",
                "obsid": obsid,
                "data_uri": _safe_text(
                    _row_value(row, "dataURI")
                ),
                "filename": _safe_text(
                    _row_value(row, "productFilename")
                ),
                "product_type": _safe_text(
                    _row_value(row, "productType")
                ),
                "description": _safe_text(
                    _row_value(row, "description")
                ),
                "size_bytes": _safe_int(
                    _row_value(row, "size")
                ),
            })

        return results

    def download_jwst_product(
        self,
        data_uri: str,
        filename: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Download one explicitly selected MAST product.

        The returned path points to a completed local file.
        Products with known sizes above the configured limit
        are rejected before downloading.
        """

        data_uri = str(data_uri).strip()

        if (
            not data_uri.startswith("mast:")
            or len(data_uri) > 2048
            or "\x00" in data_uri
        ):
            raise InvalidProductError(
                "A valid MAST dataURI is required."
            )

        safe_name = _safe_filename(
            filename or data_uri.rsplit("/", 1)[-1]
        )

        self.download_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = self.download_dir / safe_name

        # Never trust an existing file without checking its size.
        if destination.exists():
            existing_size = destination.stat().st_size

            if existing_size > self.max_download_bytes:
                destination.unlink(missing_ok=True)
                raise DownloadLimitError(
                    "Existing file exceeds the configured limit."
                )

        try:
            from astroquery.mast import Observations

            status, message, _url = Observations.download_file(
                data_uri,
                local_path=str(destination),
                cache=False,
                verbose=False,
            )

        except ImportError as exc:
            raise ArchiveUnavailableError(
                "astroquery is unavailable."
            ) from exc

        except Exception as exc:
            destination.unlink(missing_ok=True)

            LOGGER.warning(
                "MAST download failed: %s",
                type(exc).__name__,
            )
            raise ArchiveUnavailableError(
                "MAST product download failed."
            ) from exc

        if str(status).upper() != "COMPLETE":
            destination.unlink(missing_ok=True)
            raise ArchiveUnavailableError(
                "MAST did not complete the download: "
                + str(message or status)
            )

        if not destination.is_file():
            raise ArchiveUnavailableError(
                "Download reported success, but no file was found."
            )

        size_bytes = destination.stat().st_size

        if size_bytes > self.max_download_bytes:
            destination.unlink(missing_ok=True)
            raise DownloadLimitError(
                "Downloaded file exceeds the configured "
                "size limit and has been deleted."
            )

        if size_bytes == 0:
            destination.unlink(missing_ok=True)
            raise InvalidProductError(
                "Downloaded file is empty."
            )

        return {
            "mission": "JWST",
            "archive": "MAST",
            "status": "COMPLETE",
            "filename": safe_name,
            "path": str(destination.resolve()),
            "size_bytes": size_bytes,
            "sha256": _sha256_file(destination),
            "calibrated": None,
            "note": (
                "Download verified for size and checksum only. "
                "Scientific FITS validation and calibration "
                "must be performed separately."
            ),
        }

    def search_chandra(
        self,
        target: Optional[str] = None,
        observation_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Chandra archive integration placeholder.

        Deliberately raises an explicit error instead of returning
        invented search results. A verified Chandra CDA backend
        will be implemented separately.
        """

        raise UnsupportedMissionError(
            "Chandra archive search is not implemented in "
            "Module 33 yet. Use the official Chandra Data "
            "Archive to obtain a product, then process its "
            "downloaded FITS file with the Chandra adapter."
        )

    def download_chandra_product(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Explicitly report that Chandra download is not available."""
        raise UnsupportedMissionError(
            "Chandra archive download is not implemented yet."
        )


def module_info() -> dict[str, Any]:
    """Convenience function for dashboard health checks."""
    return MissionDataClient.module_info()


def run_contract_tests() -> dict[str, Any]:
    """
    Run lightweight offline contract tests.
    No network requests or archive downloads are performed.
    """

    client = MissionDataClient()

    assert client.max_download_bytes == (
        DEFAULT_MAX_DOWNLOAD_MB * 1024 * 1024
    )

    assert module_info()["module"] == MODULE_NAME

    try:
        MissionDataClient(max_download_mb=31)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Maximum download size validation failed."
        )

    try:
        client.search_jwst()
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Empty JWST search should be rejected."
        )

    try:
        client.search_chandra(target="Example Target")
    except UnsupportedMissionError:
        pass
    else:
        raise AssertionError(
            "Unimplemented Chandra search must fail explicitly."
        )

    try:
        client.download_jwst_product("https://example.com/file.fits")
    except InvalidProductError:
        pass
    else:
        raise AssertionError(
            "Non-MAST URI should be rejected."
        )

    return {
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "status": "PASS",
        "network_used": False,
        "tests": 5,
        "scope": "Offline contract checks only",
    }


if __name__ == "__main__":
    import json

    print(json.dumps(
        {
            "module_info": module_info(),
            "contract_tests": run_contract_tests(),
        },
        indent=2,
    ))
