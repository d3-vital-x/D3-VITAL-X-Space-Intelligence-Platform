"""
Module 38: science_product_router.py
D³ VITAL-X Space Intelligence Platform

Purpose
-------
- Identify supported astronomical FITS science products.
- Detect the observatory/mission from FITS metadata.
- Classify products using HDU names, dimensions and columns.
- Route products to the appropriate public adapter.
- Support JWST and Chandra adapters.
- Reject ambiguous or unsupported routing requests.

Supported routing
-----------------
JWST:
    image
    spectrum
    cube

Chandra:
    image
    events
    spectrum
    exposure_map

Important
---------
- Local FITS files only. No network/archive download.
- No scientific discovery or anomaly confirmation.
- No automatic calibration or scientific correction.
- Product classification is a routing decision, not
  a scientific validation.

Dependencies
------------
numpy
astropy >= 6.0, < 8.0

Expected neighboring modules
----------------------------
jwst_adapter.py
chandra_adapter.py

Location
--------
03_INPUT_ADAPTERS/science_product_router.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


# ============================================================
# MODULE CONFIGURATION
# ============================================================

MODULE_NAME = "science_product_router"
MODULE_VERSION = "1.0.0"

DEFAULT_MAX_FILE_MB = 100.0

SUPPORTED_MISSIONS = ("JWST", "CHANDRA")

SUPPORTED_PRODUCTS = (
    "image",
    "spectrum",
    "cube",
    "events",
    "exposure_map",
)

MISSION_PRODUCTS = {
    "JWST": {
        "image",
        "spectrum",
        "cube",
    },
    "CHANDRA": {
        "image",
        "spectrum",
        "events",
        "exposure_map",
    },
}

PRODUCT_ALIASES = {
    "img": "image",
    "image": "image",
    "images": "image",
    "x1d": "spectrum",
    "pha": "spectrum",
    "spectrum": "spectrum",
    "spectra": "spectrum",
    "spec": "spectrum",
    "datacube": "cube",
    "data_cube": "cube",
    "cube": "cube",
    "event": "events",
    "events": "events",
    "event_list": "events",
    "eventlist": "events",
    "expmap": "exposure_map",
    "exposure": "exposure_map",
    "exposure_map": "exposure_map",
}


# ============================================================
# EXCEPTIONS
# ============================================================

class ScienceRouterError(Exception):
    """Base exception for science product routing."""


class ScienceRouterDependencyError(ScienceRouterError):
    """Raised when a required dependency is unavailable."""


class ScienceRouterValidationError(ScienceRouterError):
    """Raised for invalid or ambiguous routing requests."""


class ScienceRouterResourceError(ScienceRouterError):
    """Raised when a configured resource limit is exceeded."""


class ScienceRouterAdapterError(ScienceRouterError):
    """Raised when an adapter cannot be loaded or called."""


# ============================================================
# DEPENDENCY LOADER
# ============================================================

def _get_fits():
    try:
        from astropy.io import fits
        return fits

    except ImportError as exc:
        raise ScienceRouterDependencyError(
            "Astropy is required. Install with: pip install astropy"
        ) from exc


# ============================================================
# BASIC HELPERS
# ============================================================

def _normalise_mission(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    value = str(value).strip().upper()

    aliases = {
        "JWST": "JWST",
        "JAMES WEBB": "JWST",
        "JAMES WEBB SPACE TELESCOPE": "JWST",
        "CHANDRA": "CHANDRA",
        "CHANDRA X-RAY OBSERVATORY": "CHANDRA",
        "CXO": "CHANDRA",
    }

    mission = aliases.get(value)

    if mission is None:
        raise ScienceRouterValidationError(
            f"Unsupported mission: {value}. "
            f"Supported missions: {SUPPORTED_MISSIONS}"
        )

    return mission


def _normalise_product(
    value: Optional[str],
) -> Optional[str]:

    if value is None:
        return None

    key = str(value).strip().lower().replace("-", "_")

    product = PRODUCT_ALIASES.get(key)

    if product is None:
        raise ScienceRouterValidationError(
            f"Unsupported product type: {value}. "
            f"Supported products: {SUPPORTED_PRODUCTS}"
        )

    return product


def _json_safe(value: Any) -> Any:
    """Convert metadata values into JSON-compatible values."""

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, np.generic):
            return value.item()

    except ImportError:
        pass

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, float):
        if value != value or value in (
            float("inf"),
            float("-inf"),
        ):
            return None

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


# ============================================================
# FILE VALIDATION
# ============================================================

def _validate_file(
    path: str,
    max_file_mb: float = DEFAULT_MAX_FILE_MB,
) -> Path:

    if not isinstance(path, (str, os.PathLike)):
        raise ScienceRouterValidationError(
            "path must be a string or PathLike object."
        )

    file_path = Path(path).expanduser()

    if not file_path.exists():
        raise ScienceRouterValidationError(
            f"File does not exist: {file_path}"
        )

    if not file_path.is_file():
        raise ScienceRouterValidationError(
            "Input path must point to a file."
        )

    lower_name = file_path.name.lower()

    supported_extensions = (
        ".fits",
        ".fit",
        ".fts",
        ".fits.gz",
        ".fit.gz",
        ".fts.gz",
    )

    if not any(
        lower_name.endswith(ext)
        for ext in supported_extensions
    ):
        raise ScienceRouterValidationError(
            "Unsupported file extension. "
            "A FITS file is required."
        )

    size_mb = file_path.stat().st_size / (1024 ** 2)

    if size_mb > float(max_file_mb):
        raise ScienceRouterResourceError(
            f"File size {size_mb:.2f} MB exceeds "
            f"limit {max_file_mb:.2f} MB."
        )

    return file_path


# ============================================================
# FITS METADATA INSPECTION
# ============================================================

def _inspect_fits_structure(path: Path) -> Dict[str, Any]:
    """
    Read FITS headers, HDU names, dimensions and table columns.

    This function does not load full science arrays intentionally.
    """

    fits = _get_fits()

    try:
        with fits.open(
            str(path),
            mode="readonly",
            memmap=not path.name.lower().endswith(".gz"),
            lazy_load_hdus=True,
        ) as hdul:

            hdu_records = []
            mission_values = []

            for index, hdu in enumerate(hdul):
                header = hdu.header
                data = getattr(hdu, "data", None)

                hdu_name = (
                    str(getattr(hdu, "name", "") or "").strip().upper()
                )

                if not hdu_name:
                    hdu_name = (
                        "PRIMARY"
                        if index == 0
                        else f"HDU_{index}"
                    )

                shape = None
                columns = []

                if data is not None:
                    try:
                        shape = list(data.shape)
                    except Exception:
                        shape = None

                    dtype = getattr(data, "dtype", None)

                    if dtype is not None and dtype.names:
                        columns = [
                            str(name).upper()
                            for name in dtype.names
                        ]

                for key in (
                    "TELESCOP",
                    "OBSERVAT",
                    "MISSION",
                    "ORIGIN",
                ):
                    value = header.get(key)

                    if value:
                        mission_values.append(
                            str(value).strip().upper()
                        )

                hdu_records.append({
                    "index": index,
                    "name": hdu_name,
                    "shape": shape,
                    "columns": columns,
                    "header": {
                        key: _json_safe(header[key])
                        for key in (
                            "TELESCOP",
                            "INSTRUME",
                            "OBS_ID",
                            "OBJECT",
                            "DATE-OBS",
                            "EXPOSURE",
                        )
                        if key in header
                    },
                })

            primary_header = {
                key: _json_safe(hdul[0].header[key])
                for key in (
                    "TELESCOP",
                    "OBSERVAT",
                    "MISSION",
                    "INSTRUME",
                    "OBS_ID",
                    "OBJECT",
                )
                if key in hdul[0].header
            }

        return {
            "file": path.name,
            "file_size_mb": round(
                path.stat().st_size / (1024 ** 2),
                4,
            ),
            "primary_header": primary_header,
            "mission_values": mission_values,
            "hdu_count": len(hdu_records),
            "hdus": hdu_records,
        }

    except ScienceRouterError:
        raise

    except Exception as exc:
        raise ScienceRouterValidationError(
            f"Unable to inspect FITS structure: {exc}"
        ) from exc


# ============================================================
# MISSION DETECTION
# ============================================================

def _detect_mission(
    metadata: Dict[str, Any],
    explicit_mission: Optional[str] = None,
) -> str:
    """
    Detect JWST or Chandra from FITS metadata.

    Explicit mission selection is validated against recognizable
    metadata when the metadata itself identifies a supported mission.
    """

    requested = _normalise_mission(explicit_mission)

    values = metadata.get("mission_values", [])

    detected = set()

    for value in values:
        value = str(value).upper()

        if "CHANDRA" in value or value == "CXO":
            detected.add("CHANDRA")

        if "JWST" in value or "JAMES WEBB" in value:
            detected.add("JWST")

    filename = str(metadata.get("file", "")).lower()

    if filename.startswith(("jw", "jwst")):
        detected.add("JWST")

    if filename.startswith(("acis", "hrcf", "hrci", "chandra")):
        detected.add("CHANDRA")

    if len(detected) > 1:
        raise ScienceRouterValidationError(
            "Conflicting observatory identifiers found in FITS metadata."
        )

    metadata_mission = (
        next(iter(detected))
        if detected
        else None
    )

    if requested:
        if (
            metadata_mission is not None
            and requested != metadata_mission
        ):
            raise ScienceRouterValidationError(
                f"Requested mission {requested} conflicts with "
                f"metadata mission {metadata_mission}."
            )

        return requested

    if metadata_mission is None:
        raise ScienceRouterValidationError(
            "Unable to identify observatory. "
            "Provide an explicit mission only if independently verified."
        )

    return metadata_mission


# ============================================================
# PRODUCT CLASSIFICATION
# ============================================================

def _classify_product(
    metadata: Dict[str, Any],
    explicit_product: Optional[str] = None,
) -> str:
    """
    Classify the science product from HDU names and table columns.

    Priority:
        EVENTS
        SPECTRUM / X1D
        EXPMAP
        3D cube
        2D image
    """

    if explicit_product:
        return _normalise_product(explicit_product)

    hdus = metadata.get("hdus", [])

    names = {
        str(hdu.get("name", "")).upper()
        for hdu in hdus
    }

    all_columns = set()

    for hdu in hdus:
        all_columns.update(
            str(column).upper()
            for column in hdu.get("columns", [])
        )

    # Chandra event list
    if "EVENTS" in names:
        return "events"

    # Chandra PHA or JWST X1D spectrum
    if "SPECTRUM" in names or "X1D" in names:
        return "spectrum"

    if (
        {"WAVELENGTH", "FLUX"}.issubset(all_columns)
        or {"WAVE", "FLUX"}.issubset(all_columns)
    ):
        return "spectrum"

    if (
        "CHANNEL" in all_columns
        and (
            "COUNTS" in all_columns
            or "RATE" in all_columns
        )
    ):
        return "spectrum"

    # Chandra exposure map
    if "EXPMAP" in names or "EXPOSURE_MAP" in names:
        return "exposure_map"

    # Image or cube classification
    dimensions = []

    for hdu in hdus:
        shape = hdu.get("shape")

        if shape:
            try:
                dimensions.append(len(shape))
            except TypeError:
                pass

    if 3 in dimensions:
        return "cube"

    if 2 in dimensions:
        return "image"

    raise ScienceRouterValidationError(
        "Unable to classify science product from FITS structure. "
        "Provide product_type explicitly if the product is known."
    )


# ============================================================
# ADAPTER LOADER
# ============================================================

def _load_adapter_classes():
    """
    Load neighboring adapters.

    Supports both package-style and same-folder imports.
    """

    try:
        from .jwst_adapter import JWSTAdapter
        from .chandra_adapter import ChandraAdapter

        return {
            "JWST": JWSTAdapter,
            "CHANDRA": ChandraAdapter,
        }

    except ImportError:
        try:
            from jwst_adapter import JWSTAdapter
            from chandra_adapter import ChandraAdapter

            return {
                "JWST": JWSTAdapter,
                "CHANDRA": ChandraAdapter,
            }

        except ImportError as exc:
            raise ScienceRouterAdapterError(
                "Unable to import JWSTAdapter and ChandraAdapter. "
                "Check that jwst_adapter.py and chandra_adapter.py "
                "are available in the input adapters folder."
            ) from exc


# ============================================================
# MAIN ROUTER
# ============================================================

class ScienceProductRouter:
    """
    Route local JWST and Chandra FITS science products.

    Adapter instances may be injected for testing or controlled
    application integration.
    """

    def __init__(
        self,
        max_file_mb: float = DEFAULT_MAX_FILE_MB,
        adapters: Optional[Dict[str, Any]] = None,
    ):

        self.max_file_mb = float(max_file_mb)

        if self.max_file_mb <= 0:
            raise ScienceRouterValidationError(
                "max_file_mb must be positive."
            )

        self._injected_adapters = adapters or {}
        self._adapter_cache = {}

    def _get_adapter(self, mission: str):
        mission = _normalise_mission(mission)

        if mission in self._injected_adapters:
            return self._injected_adapters[mission]

        if mission in self._adapter_cache:
            return self._adapter_cache[mission]

        classes = _load_adapter_classes()

        adapter_class = classes[mission]

        try:
            adapter = adapter_class()
        except Exception as exc:
            raise ScienceRouterAdapterError(
                f"Unable to initialize {mission} adapter: {exc}"
            ) from exc

        self._adapter_cache[mission] = adapter

        return adapter

    def inspect_product(
        self,
        path: str,
        mission: Optional[str] = None,
        product_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Inspect and classify a science product without reading it."""

        file_path = _validate_file(
            path,
            max_file_mb=self.max_file_mb,
        )

        metadata = _inspect_fits_structure(file_path)

        detected_mission = _detect_mission(
            metadata,
            explicit_mission=mission,
        )

        detected_product = _classify_product(
            metadata,
            explicit_product=product_type,
        )

        if detected_product not in MISSION_PRODUCTS[detected_mission]:
            raise ScienceRouterValidationError(
                f"Product '{detected_product}' is not supported "
                f"by the {detected_mission} adapter."
            )

        return {
            "module": MODULE_NAME,
            "version": MODULE_VERSION,
            "file": file_path.name,
            "mission": detected_mission,
            "product_type": detected_product,
            "metadata": metadata,
            "routing_status": "ready",
            "scope": "metadata_based_product_classification",
        }

    def _dispatch(
        self,
        adapter: Any,
        mission: str,
        product_type: str,
        path: str,
        kwargs: Dict[str, Any],
    ):

        method_map = {
            "JWST": {
                "image": "read_image",
                "spectrum": "read_spectrum",
                "cube": "read_cube",
            },
            "CHANDRA": {
                "image": "read_image",
                "spectrum": "read_spectrum",
                "events": "read_events",
                "exposure_map": "read_exposure_map",
            },
        }

        method_name = method_map[mission][product_type]

        method = getattr(adapter, method_name, None)

        if method is None or not callable(method):
            raise ScienceRouterAdapterError(
                f"{mission} adapter does not expose "
                f"the required method: {method_name}"
            )

        try:
            return method(path, **kwargs)

        except Exception as exc:
            raise ScienceRouterAdapterError(
                f"{mission}.{method_name} failed: {exc}"
            ) from exc

    def route(
        self,
        path: str,
        mission: Optional[str] = None,
        product_type: Optional[str] = None,
        operation: str = "read",
        **adapter_kwargs,
    ) -> Dict[str, Any]:
        """
        Inspect and route a product to the appropriate adapter.

        operation:
            "inspect" - metadata/classification only
            "read"    - dispatch to the selected adapter
        """

        operation = str(operation).strip().lower()

        if operation not in ("inspect", "read"):
            raise ScienceRouterValidationError(
                "operation must be 'inspect' or 'read'."
            )

        routing = self.inspect_product(
            path=path,
            mission=mission,
            product_type=product_type,
        )

        if operation == "inspect":
            return routing

        adapter = self._get_adapter(
            routing["mission"]
        )

        result = self._dispatch(
            adapter=adapter,
            mission=routing["mission"],
            product_type=routing["product_type"],
            path=str(path),
            kwargs=adapter_kwargs,
        )

        if not isinstance(result, dict):
            raise ScienceRouterAdapterError(
                "Adapter must return a dictionary."
            )

        output = dict(result)

        output["routing"] = {
            "router": MODULE_NAME,
            "router_version": MODULE_VERSION,
            "mission": routing["mission"],
            "product_type": routing["product_type"],
            "adapter": (
                "JWSTAdapter"
                if routing["mission"] == "JWST"
                else "ChandraAdapter"
            ),
            "status": "dispatched",
        }

        return output


# ============================================================
# PUBLIC CONVENIENCE FUNCTIONS
# ============================================================

def module_info() -> Dict[str, Any]:
    return {
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "supported_missions": list(SUPPORTED_MISSIONS),
        "supported_products": {
            mission: sorted(products)
            for mission, products in MISSION_PRODUCTS.items()
        },
        "operations": [
            "inspect",
            "read",
        ],
        "network_required": False,
        "scope": "science_product_routing_only",
    }


def inspect_science_product(
    path: str,
    mission: Optional[str] = None,
    product_type: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:

    router = ScienceProductRouter(**kwargs)

    return router.inspect_product(
        path=path,
        mission=mission,
        product_type=product_type,
    )


def route_science_product(
    path: str,
    mission: Optional[str] = None,
    product_type: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:

    router_keys = {
        "max_file_mb",
        "adapters",
    }

    router_config = {
        key: kwargs.pop(key)
        for key in list(kwargs)
        if key in router_keys
    }

    router = ScienceProductRouter(**router_config)

    return router.route(
        path=path,
        mission=mission,
        product_type=product_type,
        **kwargs,
    )


# ============================================================
# CONTRACT TEST HELPERS
# ============================================================

class _FakeJWSTAdapter:
    """Test-only adapter stub."""

    def read_image(self, path, **kwargs):
        return {
            "product_type": "image",
            "adapter_test": "JWST",
        }

    def read_spectrum(self, path, **kwargs):
        return {
            "product_type": "spectrum",
            "adapter_test": "JWST",
        }

    def read_cube(self, path, **kwargs):
        return {
            "product_type": "cube",
            "adapter_test": "JWST",
        }


class _FakeChandraAdapter:
    """Test-only adapter stub."""

    def read_image(self, path, **kwargs):
        return {
            "product_type": "image",
            "adapter_test": "CHANDRA",
        }

    def read_spectrum(self, path, **kwargs):
        return {
            "product_type": "spectrum",
            "adapter_test": "CHANDRA",
        }

    def read_events(self, path, **kwargs):
        return {
            "product_type": "events",
            "adapter_test": "CHANDRA",
        }

    def read_exposure_map(self, path, **kwargs):
        return {
            "product_type": "exposure_map",
            "adapter_test": "CHANDRA",
        }


# ============================================================
# CONTRACT TESTS
# ============================================================

def run_contract_tests() -> Dict[str, Any]:
    """
    Seven local synthetic FITS contract tests.

    Fake adapters are injected, so tests do not depend on
    real JWST or Chandra adapter execution.

    Astropy is required to create and inspect synthetic FITS files.
    """

    fits = _get_fits()
    import tempfile
    import numpy as np

    results = []

    def record(name, passed, detail=""):
        results.append({
            "test": name,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        })

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        jwst_image = root / "jwst_image.fits"
        chandra_events = root / "chandra_events.fits"
        chandra_spectrum = root / "chandra_spectrum.fits"
        jwst_cube = root / "jwst_cube.fits"

        # JWST image
        primary = fits.PrimaryHDU()
        primary.header["TELESCOP"] = "JWST"

        image_hdu = fits.ImageHDU(
            data=np.ones((8, 8), dtype=np.float32),
            name="SCI",
        )

        fits.HDUList([
            primary,
            image_hdu,
        ]).writeto(jwst_image)

        # Chandra EVENTS
        primary = fits.PrimaryHDU()
        primary.header["TELESCOP"] = "CHANDRA"

        event_hdu = fits.BinTableHDU.from_columns([
            fits.Column(
                name="TIME",
                format="D",
                array=np.arange(10, dtype=np.float64),
            ),
            fits.Column(
                name="X",
                format="E",
                array=np.arange(10, dtype=np.float32),
            ),
            fits.Column(
                name="Y",
                format="E",
                array=np.arange(10, dtype=np.float32),
            ),
        ], name="EVENTS")

        fits.HDUList([
            primary,
            event_hdu,
        ]).writeto(chandra_events)

        # Chandra spectrum
        primary = fits.PrimaryHDU()
        primary.header["TELESCOP"] = "CHANDRA"

        spectrum_hdu = fits.BinTableHDU.from_columns([
            fits.Column(
                name="CHANNEL",
                format="J",
                array=np.arange(10, dtype=np.int32),
            ),
            fits.Column(
                name="COUNTS",
                format="J",
                array=np.arange(10, dtype=np.int32),
            ),
        ], name="SPECTRUM")

        fits.HDUList([
            primary,
            spectrum_hdu,
        ]).writeto(chandra_spectrum)

        # JWST cube
        primary = fits.PrimaryHDU()
        primary.header["TELESCOP"] = "JWST"

        cube_hdu = fits.ImageHDU(
            data=np.ones((4, 8, 8), dtype=np.float32),
            name="SCI",
        )

        fits.HDUList([
            primary,
            cube_hdu,
        ]).writeto(jwst_cube)

        fake_adapters = {
            "JWST": _FakeJWSTAdapter(),
            "CHANDRA": _FakeChandraAdapter(),
        }

        router = ScienceProductRouter(
            adapters=fake_adapters
        )

        # Test 1: JWST mission and image classification
        try:
            result = router.inspect_product(
                str(jwst_image)
            )

            ok = (
                result["mission"] == "JWST"
                and result["product_type"] == "image"
            )

            record("jwst_image_classification", ok)

        except Exception as exc:
            record("jwst_image_classification", False, str(exc))

        # Test 2: Chandra EVENTS classification
        try:
            result = router.inspect_product(
                str(chandra_events)
            )

            ok = (
                result["mission"] == "CHANDRA"
                and result["product_type"] == "events"
            )

            record("chandra_events_classification", ok)

        except Exception as exc:
            record("chandra_events_classification", False, str(exc))

        # Test 3: Chandra spectrum classification
        try:
            result = router.inspect_product(
                str(chandra_spectrum)
            )

            ok = (
                result["mission"] == "CHANDRA"
                and result["product_type"] == "spectrum"
            )

            record("chandra_spectrum_classification", ok)

        except Exception as exc:
            record("chandra_spectrum_classification", False, str(exc))

        # Test 4: JWST cube classification
        try:
            result = router.inspect_product(
                str(jwst_cube)
            )

            ok = (
                result["mission"] == "JWST"
                and result["product_type"] == "cube"
            )

            record("jwst_cube_classification", ok)

        except Exception as exc:
            record("jwst_cube_classification", False, str(exc))

        # Test 5: JWST spectrum dispatch with explicit product type
        try:
            result = router.route(
                str(jwst_image),
                mission="JWST",
                product_type="spectrum",
            )

            ok = (
                result["adapter_test"] == "JWST"
                and result["routing"]["status"] == "dispatched"
                and result["routing"]["product_type"] == "spectrum"
            )

            record("jwst_adapter_dispatch", ok)

        except Exception as exc:
            record("jwst_adapter_dispatch", False, str(exc))

        # Test 6: Chandra EVENTS dispatch
        try:
            result = router.route(
                str(chandra_events)
            )

            ok = (
                result["adapter_test"] == "CHANDRA"
                and result["routing"]["product_type"] == "events"
            )

            record("chandra_adapter_dispatch", ok)

        except Exception as exc:
            record("chandra_adapter_dispatch", False, str(exc))

        # Test 7: Reject unsupported JWST events route
        try:
            router.inspect_product(
                str(jwst_image),
                mission="JWST",
                product_type="events",
            )

            record(
                "unsupported_product_rejection",
                False,
                "Unsupported JWST events route was accepted.",
            )

        except ScienceRouterValidationError:
            record("unsupported_product_rejection", True)

        except Exception as exc:
            record(
                "unsupported_product_rejection",
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
