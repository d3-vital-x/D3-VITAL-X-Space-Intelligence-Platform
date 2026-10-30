"""
Module 39: space_visualization.py
D³ VITAL-X Space Intelligence Platform

Purpose
-------
- Visualize astronomical science products.
- Plot 2D FITS images and exposure maps.
- Plot JWST wavelength/flux spectra.
- Plot Chandra event distributions.
- Plot Chandra energy/channel histograms.
- Visualize 3D data cube slices.
- Display basic QC summaries.
- Export figures as PNG files.

Supported input formats
-----------------------
- NumPy arrays
- JWST adapter result dictionaries
- Chandra adapter result dictionaries
- Science product router result dictionaries

Scope
-----
Visualization only.
No source detection, scientific discovery,
diagnosis, or anomaly confirmation.

Dependencies
------------
numpy
matplotlib >= 3.8, < 4.0

Location
--------
06_VISUALIZATION/space_visualization.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


# ============================================================
# MODULE CONFIGURATION
# ============================================================

MODULE_NAME = "space_visualization"
MODULE_VERSION = "1.0.0"

DEFAULT_MAX_PLOT_ELEMENTS = 2_000_000
DEFAULT_MAX_PREVIEW_SIZE = 1024
DEFAULT_FIGURE_DPI = 120

SUPPORTED_FORMATS = (
    "image",
    "spectrum",
    "events",
    "cube",
    "exposure_map",
    "qc",
)


# ============================================================
# EXCEPTIONS
# ============================================================

class SpaceVisualizationError(Exception):
    """Base exception for visualization errors."""


class VisualizationDependencyError(SpaceVisualizationError):
    """Raised when Matplotlib is unavailable."""


class VisualizationValidationError(SpaceVisualizationError):
    """Raised for invalid visualization input."""


class VisualizationResourceError(SpaceVisualizationError):
    """Raised when configured plotting limits are exceeded."""


# ============================================================
# DEPENDENCY LOADER
# ============================================================

def _get_matplotlib():
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        return Figure, FigureCanvasAgg

    except ImportError as exc:
        raise VisualizationDependencyError(
            "Matplotlib is required. Install with: "
            "pip install matplotlib"
        ) from exc


# ============================================================
# BASIC HELPERS
# ============================================================

def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

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


def _validate_numeric_array(
    data: Any,
    name: str = "data",
    max_elements: int = DEFAULT_MAX_PLOT_ELEMENTS,
) -> np.ndarray:
    """Validate real numeric arrays and enforce element limits."""

    try:
        arr = np.asarray(data)
    except Exception as exc:
        raise VisualizationValidationError(
            f"Unable to convert {name} to a NumPy array."
        ) from exc

    if arr.dtype.kind not in "iuf":
        raise VisualizationValidationError(
            f"{name} must contain real numeric values."
        )

    if arr.size == 0:
        raise VisualizationValidationError(
            f"{name} is empty."
        )

    if arr.size > int(max_elements):
        raise VisualizationResourceError(
            f"{name} contains {arr.size} elements; "
            f"limit is {max_elements}."
        )

    return arr


def _finite_values(array: np.ndarray) -> np.ndarray:
    """Return finite values only."""

    arr = np.asarray(array)
    return arr[np.isfinite(arr)]


def _summary(array: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(array)
    finite = _finite_values(arr)

    result = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "total_count": int(arr.size),
        "finite_count": int(finite.size),
        "non_finite_count": int(arr.size - finite.size),
        "finite_fraction": (
            float(finite.size / arr.size)
            if arr.size else 0.0
        ),
    }

    if finite.size:
        result.update({
            "minimum": float(np.min(finite)),
            "maximum": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
        })
    else:
        result.update({
            "minimum": None,
            "maximum": None,
            "mean": None,
            "std": None,
        })

    return result


def _make_figure(
    figsize: Tuple[float, float] = (9, 6),
):
    Figure, FigureCanvasAgg = _get_matplotlib()

    fig = Figure(
        figsize=figsize,
        constrained_layout=True,
    )

    FigureCanvasAgg(fig)

    return fig


def _save_figure(
    fig,
    save_path: Optional[str],
    dpi: int = DEFAULT_FIGURE_DPI,
) -> Optional[str]:
    """Save a figure if an output path is supplied."""

    if save_path is None:
        return None

    path = Path(save_path).expanduser()

    if path.suffix.lower() != ".png":
        raise VisualizationValidationError(
            "Only PNG export is supported by this module."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        fig.savefig(
            str(path),
            dpi=int(dpi),
            format="png",
            bbox_inches="tight",
        )
    except Exception as exc:
        raise SpaceVisualizationError(
            f"Unable to save visualization: {exc}"
        ) from exc

    return str(path)


def _downsample_2d(
    array: np.ndarray,
    max_size: int = DEFAULT_MAX_PREVIEW_SIZE,
) -> np.ndarray:
    """Stride-based preview downsampling."""

    if array.ndim != 2:
        raise VisualizationValidationError(
            "Expected a 2D array."
        )

    if max_size < 1:
        raise VisualizationValidationError(
            "max_size must be at least 1."
        )

    stride_y = max(
        1,
        int(np.ceil(array.shape[0] / max_size)),
    )

    stride_x = max(
        1,
        int(np.ceil(array.shape[1] / max_size)),
    )

    return array[::stride_y, ::stride_x]


def _extract_array(
    data: Any,
    keys: Sequence[str],
) -> Optional[np.ndarray]:
    """Find the first matching array in a result dictionary."""

    if not isinstance(data, dict):
        return None

    for key in keys:
        if key not in data:
            continue

        value = data[key]

        if value is None:
            continue

        try:
            return np.asarray(value)
        except Exception:
            continue

    return None


def _product_type(data: Any) -> Optional[str]:
    if not isinstance(data, dict):
        return None

    routing = data.get("routing", {})

    if isinstance(routing, dict):
        value = routing.get("product_type")

        if value:
            return str(value).lower()

    value = data.get("product_type")

    if value:
        return str(value).lower()

    return None


# ============================================================
# VISUALIZATION CLASS
# ============================================================

class SpaceVisualizer:
    """
    Visualization utilities for astronomical science products.

    All plotting methods return a dictionary containing:
        figure
        axes
        summary
        saved_path
    """

    def __init__(
        self,
        max_plot_elements: int = DEFAULT_MAX_PLOT_ELEMENTS,
        max_preview_size: int = DEFAULT_MAX_PREVIEW_SIZE,
        dpi: int = DEFAULT_FIGURE_DPI,
    ):
        self.max_plot_elements = int(max_plot_elements)
        self.max_preview_size = int(max_preview_size)
        self.dpi = int(dpi)

        if self.max_plot_elements < 1:
            raise VisualizationValidationError(
                "max_plot_elements must be positive."
            )

        if self.max_preview_size < 1:
            raise VisualizationValidationError(
                "max_preview_size must be positive."
            )

        if self.dpi < 30 or self.dpi > 600:
            raise VisualizationValidationError(
                "dpi must be between 30 and 600."
            )

    # --------------------------------------------------------
    # 2D IMAGE
    # --------------------------------------------------------

    def plot_image(
        self,
        data: Any,
        title: str = "Astronomical Image",
        cmap: str = "gray",
        save_path: Optional[str] = None,
        max_size: Optional[int] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Visualize a 2D astronomical image."""

        if isinstance(data, dict):
            array = _extract_array(
                data,
                ("image", "data", "preview"),
            )
        else:
            array = np.asarray(data)

        if array is None:
            raise VisualizationValidationError(
                "No image array found."
            )

        array = _validate_numeric_array(
            array,
            name="image",
            max_elements=self.max_plot_elements,
        )

        if array.ndim != 2:
            raise VisualizationValidationError(
                "Image visualization requires a 2D array."
            )

        preview_limit = (
            self.max_preview_size
            if max_size is None
            else int(max_size)
        )

        preview = _downsample_2d(
            array,
            max_size=preview_limit,
        )

        finite = _finite_values(preview)

        if finite.size == 0:
            raise VisualizationValidationError(
                "Image contains no finite values."
            )

        fig = _make_figure()
        ax = fig.add_subplot(111)

        image = ax.imshow(
            preview,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="auto",
        )

        ax.set_title(str(title))
        ax.set_xlabel("X pixel")
        ax.set_ylabel("Y pixel")

        fig.colorbar(
            image,
            ax=ax,
            label="Pixel value",
        )

        saved = _save_figure(
            fig,
            save_path,
            dpi=self.dpi,
        )

        return {
            "module": MODULE_NAME,
            "visualization_type": "image",
            "figure": fig,
            "axes": ax,
            "summary": _summary(array),
            "preview_shape": list(preview.shape),
            "saved_path": saved,
            "status": "visualization_complete",
        }

    # --------------------------------------------------------
    # EXPOSURE MAP
    # --------------------------------------------------------

    def plot_exposure_map(
        self,
        data: Any,
        title: str = "Chandra Exposure Map",
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Visualize a 2D exposure map."""

        if isinstance(data, dict):
            array = _extract_array(
                data,
                ("image", "exposure_map", "data", "preview"),
            )
        else:
            array = np.asarray(data)

        if array is None:
            raise VisualizationValidationError(
                "No exposure map array found."
            )

        result = self.plot_image(
            array,
            title=title,
            cmap="viridis",
            save_path=save_path,
        )

        result["visualization_type"] = "exposure_map"

        return result

    # --------------------------------------------------------
    # SPECTRUM
    # --------------------------------------------------------

    def plot_spectrum(
        self,
        data: Any,
        x: Optional[Any] = None,
        y: Optional[Any] = None,
        title: str = "Astronomical Spectrum",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Plot a spectrum.

        Supports:
            {"spectrum": {"wavelength": ..., "flux": ...}}
            {"spectrum": {"channel": ..., "counts": ...}}
            {"wavelength": ..., "flux": ...}
            Explicit x and y arrays.
        """

        if x is None or y is None:
            source = data

            if isinstance(data, dict):
                source = data.get("spectrum", data)

            if not isinstance(source, dict):
                raise VisualizationValidationError(
                    "Spectrum must be a dictionary or use x/y arrays."
                )

            if x is None:
                x = next(
                    (
                        source[key]
                        for key in (
                            "wavelength",
                            "WAVELENGTH",
                            "wave",
                            "WAVE",
                            "channel",
                        )
                        if key in source
                    ),
                    None,
                )

            if y is None:
                y = next(
                    (
                        source[key]
                        for key in (
                            "flux",
                            "FLUX",
                            "counts",
                            "COUNTS",
                            "rate",
                            "RATE",
                        )
                        if key in source
                    ),
                    None,
                )

        if x is None or y is None:
            raise VisualizationValidationError(
                "Could not identify spectrum x/y arrays."
            )

        x_array = _validate_numeric_array(
            x,
            name="spectrum x",
            max_elements=self.max_plot_elements,
        ).reshape(-1)

        y_array = _validate_numeric_array(
            y,
            name="spectrum y",
            max_elements=self.max_plot_elements,
        ).reshape(-1)

        if x_array.size != y_array.size:
            raise VisualizationValidationError(
                "Spectrum x and y arrays must have equal lengths."
            )

        finite_mask = (
            np.isfinite(x_array)
            & np.isfinite(y_array)
        )

        if not np.any(finite_mask):
            raise VisualizationValidationError(
                "Spectrum contains no finite x/y pairs."
            )

        x_plot = x_array[finite_mask]
        y_plot = y_array[finite_mask]

        fig = _make_figure()
        ax = fig.add_subplot(111)

        ax.plot(
            x_plot,
            y_plot,
            linewidth=1.0,
        )

        ax.set_title(str(title))
        ax.set_xlabel(
            x_label or "Wavelength / Channel"
        )
        ax.set_ylabel(
            y_label or "Flux / Counts"
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        saved = _save_figure(
            fig,
            save_path,
            dpi=self.dpi,
        )

        return {
            "module": MODULE_NAME,
            "visualization_type": "spectrum",
            "figure": fig,
            "axes": ax,
            "summary": {
                "total_points": int(x_array.size),
                "finite_points": int(x_plot.size),
                "excluded_points": int(
                    x_array.size - x_plot.size
                ),
                "x_summary": _summary(x_plot),
                "y_summary": _summary(y_plot),
            },
            "saved_path": saved,
            "status": "visualization_complete",
        }

    # --------------------------------------------------------
    # EVENT DISTRIBUTION
    # --------------------------------------------------------

    def plot_events(
        self,
        data: Any,
        bins: int = 50,
        title: str = "Chandra Event Distribution",
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Visualize event positions and optional energy distribution.

        Expected input:
            {"events": {"X": ..., "Y": ..., "ENERGY": ...}}
        """

        if not isinstance(data, dict):
            raise VisualizationValidationError(
                "Event data must be a dictionary."
            )

        events = data.get("events", data)

        if not isinstance(events, dict):
            raise VisualizationValidationError(
                "Invalid event table structure."
            )

        lookup = {
            str(key).upper(): key
            for key in events
        }

        x_key = lookup.get("X")
        y_key = lookup.get("Y")
        energy_key = lookup.get("ENERGY")

        if x_key is None or y_key is None:
            raise VisualizationValidationError(
                "Event visualization requires X and Y columns."
            )

        x = _validate_numeric_array(
            events[x_key],
            name="event X",
            max_elements=self.max_plot_elements,
        ).reshape(-1)

        y = _validate_numeric_array(
            events[y_key],
            name="event Y",
            max_elements=self.max_plot_elements,
        ).reshape(-1)

        if x.size != y.size:
            raise VisualizationValidationError(
                "Event X and Y arrays must have equal lengths."
            )

        finite_mask = (
            np.isfinite(x)
            & np.isfinite(y)
        )

        x = x[finite_mask]
        y = y[finite_mask]

        if x.size == 0:
            raise VisualizationValidationError(
                "No finite event positions found."
            )

        if bins < 1 or bins > 500:
            raise VisualizationValidationError(
                "bins must be between 1 and 500."
            )

        fig = _make_figure(
            figsize=(10, 5),
        )

        if energy_key is not None:
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
        else:
            ax1 = fig.add_subplot(111)
            ax2 = None

        ax1.scatter(
            x,
            y,
            s=2,
            alpha=0.5,
        )

        ax1.set_title("Event Positions")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")

        energy_summary = None

        if ax2 is not None:
            energy = _validate_numeric_array(
                events[energy_key],
                name="event energy",
                max_elements=self.max_plot_elements,
            ).reshape(-1)

            energy = energy[np.isfinite(energy)]

            if energy.size:
                ax2.hist(
                    energy,
                    bins=int(bins),
                )

                ax2.set_title("Energy Distribution")
                ax2.set_xlabel("Energy")
                ax2.set_ylabel("Event count")

                energy_summary = _summary(energy)
            else:
                ax2.set_title("No finite energy values")

        fig.suptitle(str(title))

        saved = _save_figure(
            fig,
            save_path,
            dpi=self.dpi,
        )

        return {
            "module": MODULE_NAME,
            "visualization_type": "events",
            "figure": fig,
            "axes": (
                [ax1, ax2]
                if ax2 is not None
                else [ax1]
            ),
            "summary": {
                "event_positions": int(x.size),
                "x_summary": _summary(x),
                "y_summary": _summary(y),
                "energy_summary": energy_summary,
            },
            "saved_path": saved,
            "status": "visualization_complete",
        }

    # --------------------------------------------------------
    # 3D CUBE SLICE
    # --------------------------------------------------------

    def plot_cube_slice(
        self,
        data: Any,
        slice_index: int = 0,
        axis: int = 0,
        title: str = "Science Data Cube Slice",
        cmap: str = "viridis",
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Visualize one selected 2D slice of a 3D cube."""

        if isinstance(data, dict):
            cube = _extract_array(
                data,
                ("cube", "data", "science_cube"),
            )
        else:
            cube = np.asarray(data)

        if cube is None:
            raise VisualizationValidationError(
                "No cube array found."
            )

        cube = _validate_numeric_array(
            cube,
            name="cube",
            max_elements=self.max_plot_elements,
        )

        if cube.ndim != 3:
            raise VisualizationValidationError(
                "Cube visualization requires a 3D array."
            )

        if axis not in (0, 1, 2):
            raise VisualizationValidationError(
                "axis must be 0, 1 or 2."
            )

        if slice_index < 0 or slice_index >= cube.shape[axis]:
            raise VisualizationValidationError(
                "slice_index is outside the selected axis."
            )

        selected_slice = np.take(
            cube,
            indices=int(slice_index),
            axis=int(axis),
        )

        result = self.plot_image(
            selected_slice,
            title=(
                f"{title} | axis={axis}, "
                f"slice={slice_index}"
            ),
            cmap=cmap,
            save_path=save_path,
        )

        result["visualization_type"] = "cube_slice"
        result["cube_shape"] = list(cube.shape)
        result["slice_axis"] = int(axis)
        result["slice_index"] = int(slice_index)

        return result

    # --------------------------------------------------------
    # QC SUMMARY
    # --------------------------------------------------------

    def plot_qc_summary(
        self,
        qc: Dict[str, Any],
        title: str = "Data Quality Summary",
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Plot finite and non-finite QC counts."""

        if not isinstance(qc, dict):
            raise VisualizationValidationError(
                "QC input must be a dictionary."
            )

        labels = []
        values = []

        if (
            "finite_count" in qc
            and "non_finite_count" in qc
        ):
            labels = [
                "Finite",
                "Non-finite",
            ]

            values = [
                int(qc["finite_count"]),
                int(qc["non_finite_count"]),
            ]

        else:
            for key, value in qc.items():
                if not isinstance(value, dict):
                    continue

                if "finite_count" in value:
                    labels.append(str(key))
                    values.append(
                        int(value["finite_count"])
                    )

        if not values:
            raise VisualizationValidationError(
                "No supported QC counts found."
            )

        if any(value < 0 for value in values):
            raise VisualizationValidationError(
                "QC counts cannot be negative."
            )

        fig = _make_figure()
        ax = fig.add_subplot(111)

        ax.bar(
            labels,
            values,
        )

        ax.set_title(str(title))
        ax.set_ylabel("Count")

        saved = _save_figure(
            fig,
            save_path,
            dpi=self.dpi,
        )

        return {
            "module": MODULE_NAME,
            "visualization_type": "qc",
            "figure": fig,
            "axes": ax,
            "summary": {
                "labels": labels,
                "values": values,
            },
            "saved_path": saved,
            "status": "visualization_complete",
        }

    # --------------------------------------------------------
    # AUTOMATIC PRODUCT VISUALIZATION
    # --------------------------------------------------------

    def render_product(
        self,
        data: Any,
        product_type: Optional[str] = None,
        save_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Select a visualization based on product type.

        This does not automatically infer scientific meaning.
        """

        selected_type = (
            str(product_type).lower()
            if product_type
            else _product_type(data)
        )

        if selected_type is None:
            raise VisualizationValidationError(
                "product_type is required when it is not "
                "available in the input result."
            )

        if selected_type in ("image", "images"):
            return self.plot_image(
                data,
                save_path=save_path,
                **kwargs,
            )

        if selected_type in ("exposure_map", "expmap"):
            return self.plot_exposure_map(
                data,
                save_path=save_path,
                **kwargs,
            )

        if selected_type in ("spectrum", "spectra"):
            return self.plot_spectrum(
                data,
                save_path=save_path,
                **kwargs,
            )

        if selected_type in ("events", "event"):
            return self.plot_events(
                data,
                save_path=save_path,
                **kwargs,
            )

        if selected_type in ("cube", "datacube"):
            return self.plot_cube_slice(
                data,
                save_path=save_path,
                **kwargs,
            )

        if selected_type == "qc":
            return self.plot_qc_summary(
                data,
                save_path=save_path,
                **kwargs,
            )

        raise VisualizationValidationError(
            f"Unsupported visualization product: {selected_type}"
        )


# ============================================================
# PUBLIC CONVENIENCE FUNCTIONS
# ============================================================

def module_info() -> Dict[str, Any]:
    return {
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "supported_formats": list(SUPPORTED_FORMATS),
        "export_format": "PNG",
        "network_required": False,
        "scope": "astronomical_data_visualization_only",
    }


def visualize_image(data: Any, **kwargs) -> Dict[str, Any]:
    return SpaceVisualizer().plot_image(data, **kwargs)


def visualize_spectrum(data: Any, **kwargs) -> Dict[str, Any]:
    return SpaceVisualizer().plot_spectrum(data, **kwargs)


def visualize_events(data: Any, **kwargs) -> Dict[str, Any]:
    return SpaceVisualizer().plot_events(data, **kwargs)


def visualize_cube_slice(data: Any, **kwargs) -> Dict[str, Any]:
    return SpaceVisualizer().plot_cube_slice(data, **kwargs)


def visualize_exposure_map(data: Any, **kwargs) -> Dict[str, Any]:
    return SpaceVisualizer().plot_exposure_map(data, **kwargs)


def visualize_qc(qc: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    return SpaceVisualizer().plot_qc_summary(qc, **kwargs)


# ============================================================
# CONTRACT TESTS
# ============================================================

def run_contract_tests() -> Dict[str, Any]:
    """
    Seven synthetic visualization contract tests.

    Uses local NumPy arrays only.
    Does not require JWST/Chandra FITS files or network access.
    """

    import tempfile

    results = []

    def record(name, passed, detail=""):
        results.append({
            "test": name,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        })

    visualizer = SpaceVisualizer()

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        # Test 1: Image visualization
        try:
            image = np.arange(
                100,
                dtype=np.float32,
            ).reshape(10, 10)

            output = visualizer.plot_image(image)

            ok = (
                output["visualization_type"] == "image"
                and output["summary"]["finite_count"] == 100
            )

            record("image_visualization", ok)

        except Exception as exc:
            record("image_visualization", False, str(exc))

        # Test 2: Spectrum visualization
        try:
            wavelength = np.linspace(
                1.0,
                5.0,
                100,
            )

            flux = np.sin(wavelength) + 2.0

            output = visualizer.plot_spectrum({
                "spectrum": {
                    "wavelength": wavelength,
                    "flux": flux,
                }
            })

            ok = (
                output["visualization_type"] == "spectrum"
                and output["summary"]["finite_points"] == 100
            )

            record("spectrum_visualization", ok)

        except Exception as exc:
            record("spectrum_visualization", False, str(exc))

        # Test 3: Event visualization
        try:
            events = {
                "events": {
                    "X": np.arange(100, dtype=np.float32),
                    "Y": np.arange(100, dtype=np.float32) * 2,
                    "ENERGY": np.linspace(
                        100,
                        1000,
                        100,
                    ),
                }
            }

            output = visualizer.plot_events(events)

            ok = (
                output["visualization_type"] == "events"
                and output["summary"]["event_positions"] == 100
            )

            record("event_visualization", ok)

        except Exception as exc:
            record("event_visualization", False, str(exc))

        # Test 4: Cube slice visualization
        try:
            cube = np.ones(
                (4, 8, 8),
                dtype=np.float32,
            )

            output = visualizer.plot_cube_slice(
                cube,
                slice_index=2,
                axis=0,
            )

            ok = (
                output["visualization_type"] == "cube_slice"
                and output["slice_index"] == 2
                and output["summary"]["shape"] == [8, 8]
            )

            record("cube_slice_visualization", ok)

        except Exception as exc:
            record("cube_slice_visualization", False, str(exc))

        # Test 5: Exposure map visualization
        try:
            exposure = np.full(
                (8, 8),
                1000,
                dtype=np.float32,
            )

            output = visualizer.plot_exposure_map(exposure)

            ok = (
                output["visualization_type"] == "exposure_map"
                and output["summary"]["finite_count"] == 64
            )

            record("exposure_map_visualization", ok)

        except Exception as exc:
            record("exposure_map_visualization", False, str(exc))

        # Test 6: QC summary visualization
        try:
            qc = {
                "finite_count": 90,
                "non_finite_count": 10,
            }

            output = visualizer.plot_qc_summary(qc)

            ok = (
                output["visualization_type"] == "qc"
                and output["summary"]["values"] == [90, 10]
            )

            record("qc_visualization", ok)

        except Exception as exc:
            record("qc_visualization", False, str(exc))

        # Test 7: PNG export
        try:
            output_path = root / "test_image.png"

            output = visualizer.plot_image(
                np.ones((8, 8), dtype=np.float32),
                save_path=str(output_path),
            )

            ok = (
                output["saved_path"] == str(output_path)
                and output_path.exists()
                and output_path.stat().st_size > 0
            )

            record("png_export", ok)

        except Exception as exc:
            record("png_export", False, str(exc))

    passed = sum(
        item["status"] == "PASS"
        for item in results
    )

    failed = len(results) - passed

    return {
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "test_type": "synthetic_visualization_contract_tests",
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
