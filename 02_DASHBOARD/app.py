"""
D³ VITAL-X Space Intelligence Platform
Module 21: app.py

Version: 1.0.0

Main Streamlit dashboard entry point.

Features
--------
- Home, Data Upload, Data Quality, and Visualization pages.
- Integration interfaces for Modules 18–20:
  space_page.py, biomedical_page.py, and live_page.py.
- Dynamic public-page module discovery.
- File provenance and SHA-256 fingerprinting.
- Session-state preservation for uploaded files.
- Graceful handling of missing optional modules.

Scientific safeguards
---------------------
- Public framework only.
- No synthetic scientific observations are generated.
- No anomaly, physical event, or medical condition is confirmed.
- Uploaded snapshots are not represented as live feeds.
- The protected intelligence core is not imported or implemented.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


MODULE_NAME = "app"
MODULE_NUMBER = 21
MODULE_VERSION = "1.0.0"

APP_VERSION = "0.2.0-development"
PROJECT_STATUS = "ACTIVE DEVELOPMENT"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = Path(__file__).resolve().parent
PAGES_DIR = DASHBOARD_DIR / "pages"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="D³ VITAL-X Space Intelligence Platform",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PUBLIC PAGE MODULE REGISTRY
# ============================================================

PAGE_MODULES = {
    "🛰️ Space Mode": {
        "filename": "space_page.py",
        "module_name": "d3vitalx_space_page",
        "number": 18,
    },
    "🩻 Biomedical Mode": {
        "filename": "biomedical_page.py",
        "module_name": "d3vitalx_biomedical_page",
        "number": 19,
    },
    "📡 Live Mode": {
        "filename": "live_page.py",
        "module_name": "d3vitalx_live_page",
        "number": 20,
    },
}


# Optional public analytics files. Presence alone does not prove
# correctness, scientific validity, or complete integration.
OPTIONAL_ANALYTICS_FILES = (
    "metrics.py",
    "transition_analysis.py",
    "anomaly_scoring.py",
    "uncertainty.py",
    "explainability.py",
)

OPTIONAL_VISUALIZATION_FILES = (
    "entropy_map.py",
    "variance_map.py",
    "gradient_map.py",
    "csi_map.py",
    "anomaly_overlay.py",
    "timeseries_plot.py",
    "dashboard_metrics.py",
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def utc_timestamp() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def calculate_sha256(data: bytes) -> str:
    """Calculate SHA-256 for raw uploaded bytes."""
    return hashlib.sha256(data).hexdigest()


def format_bytes(size: int) -> str:
    """Format file size for display."""
    if size < 1024:
        return f"{size} B"

    if size < 1024**2:
        return f"{size / 1024:.2f} KB"

    if size < 1024**3:
        return f"{size / 1024**2:.2f} MB"

    return f"{size / 1024**3:.2f} GB"


def detect_input_type(filename: str) -> str:
    """Detect a supported public input type by extension."""
    suffix = Path(filename).suffix.lower()

    if suffix == ".csv":
        return "CSV"

    if suffix == ".json":
        return "JSON"

    if suffix in {
        ".png", ".jpg", ".jpeg", ".bmp",
        ".tif", ".tiff", ".webp",
    }:
        return "IMAGE"

    if suffix in {".txt", ".dat"}:
        return "TIME-SERIES"

    if suffix in {".dcm", ".dicom"}:
        return "DICOM"

    return "UNKNOWN"


def find_public_module(filename: str) -> Path | None:
    """
    Locate a public module without searching private-core directories.
    """
    excluded_parts = {
        "15_PRIVATE_CORE",
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "site-packages",
    }

    candidate_paths = [
        PAGES_DIR / filename,
        DASHBOARD_DIR / filename,
        PROJECT_ROOT / filename,
    ]

    for candidate in candidate_paths:
        if candidate.is_file():
            if not any(
                part in excluded_parts
                for part in candidate.parts
            ):
                return candidate

    # Search only known public project locations.
    public_roots = [
        PROJECT_ROOT / "02_DASHBOARD",
        PROJECT_ROOT / "06_ANALYTICS",
        PROJECT_ROOT / "07_VISUALIZATION",
    ]

    for root in public_roots:
        if not root.is_dir():
            continue

        for candidate in root.rglob(filename):
            if not candidate.is_file():
                continue

            if any(
                part in excluded_parts
                for part in candidate.parts
            ):
                continue

            return candidate

    return None


def load_public_module(
    module_name: str,
    filename: str,
) -> Any:
    """Load a module from a discovered public source file."""
    module_path = find_public_module(filename)

    if module_path is None:
        raise FileNotFoundError(
            f"Public module '{filename}' was not found."
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not create a loader for '{filename}'."
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return module


def _render_integrated_page(
    page_label: str,
    page_config: dict[str, Any],
) -> None:
    """
    Render a standalone page module inside the main app.

    Existing Modules 18–20 call st.set_page_config() inside
    render_page(). The main app owns page configuration, so this
    compatibility wrapper suppresses that redundant call while
    rendering an integrated page.
    """
    try:
        module = load_public_module(
            page_config["module_name"],
            page_config["filename"],
        )

    except Exception as exc:
        st.error(
            f"{page_label} could not be loaded."
        )
        st.caption(
            f"{type(exc).__name__}: {exc}"
        )
        st.info(
            "Check the module filename, dependencies, and project path."
        )
        return

    render_function = getattr(module, "render_page", None)

    if not callable(render_function):
        st.error(
            f"{page_config['filename']} does not expose a callable "
            "render_page() function."
        )
        return

    st.session_state["active_public_page"] = page_label

    original_set_page_config = st.set_page_config

    try:
        # Compatibility bridge for existing page modules that call
        # set_page_config() internally. The main app owns this setting.
        st.set_page_config = lambda *args, **kwargs: None

        render_function()

    except Exception as exc:
        st.error(
            f"An error occurred while rendering {page_label}."
        )
        st.caption(
            f"{type(exc).__name__}: {exc}"
        )

    finally:
        st.set_page_config = original_set_page_config


def _file_presence_status(
    filenames: tuple[str, ...],
) -> dict[str, Any]:
    """Report optional public files found in the repository."""
    found = []
    missing = []

    for filename in filenames:
        path = find_public_module(filename)

        if path is None:
            missing.append(filename)
        else:
            found.append(filename)

    return {
        "found_count": len(found),
        "expected_count": len(filenames),
        "found": found,
        "missing": missing,
        "complete": len(missing) == 0,
    }


def _initialize_session_state() -> None:
    """Initialize upload-related session state."""
    defaults = {
        "uploaded_data": None,
        "uploaded_name": None,
        "uploaded_type": None,
        "uploaded_hash": None,
        "uploaded_size": None,
        "uploaded_at_utc": None,
        "active_public_page": None,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def _store_uploaded_file(uploaded_file: Any) -> None:
    """Store uploaded bytes and their provenance in session state."""
    raw_bytes = uploaded_file.getvalue()

    st.session_state["uploaded_data"] = raw_bytes
    st.session_state["uploaded_name"] = Path(
        uploaded_file.name
    ).name
    st.session_state["uploaded_type"] = detect_input_type(
        uploaded_file.name
    )
    st.session_state["uploaded_hash"] = calculate_sha256(
        raw_bytes
    )
    st.session_state["uploaded_size"] = len(raw_bytes)
    st.session_state["uploaded_at_utc"] = utc_timestamp()


# ============================================================
# PAGE: HOME
# ============================================================

def render_home() -> None:
    st.header("Welcome to D³ VITAL-X")

    st.write(
        "A research-oriented public framework for scientific data "
        "ingestion, technical quality assessment, exploratory analysis, "
        "and visualization."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Input Layer", "Public")
    c2.metric("Page Modules", "18–20")
    c3.metric("App Module", "21")
    c4.metric("Status", "In Development")

    st.divider()

    st.subheader("Public Architecture")

    architecture = [
        "Input adapters",
        "Unified data representation",
        "Technical validation and quality control",
        "Feature and analytics interfaces",
        "Exploratory visualization",
        "Dashboard and report export",
        "Human review and provenance",
    ]

    for number, item in enumerate(architecture, start=1):
        st.write(f"**{number}.** {item}")

    st.divider()

    st.subheader("Available Research Modes")

    c1, c2, c3 = st.columns(3)

    c1.info(
        "**🛰️ Space Mode**\n\n"
        "Exploratory inspection of uploaded space-observation data."
    )

    c2.info(
        "**🩻 Biomedical Mode**\n\n"
        "Technical inspection of authorized biomedical research files."
    )

    c3.info(
        "**📡 Live Mode**\n\n"
        "Snapshot freshness and signal-monitoring interfaces."
    )

    st.warning(
        "Research prototype only. Not a medical diagnostic system, "
        "clinical decision tool, or flight-certified spacecraft system."
    )


# ============================================================
# PAGE: DATA UPLOAD
# ============================================================

def render_data_upload() -> None:
    st.header("📥 Data Upload")

    st.write(
        "Upload a file for technical inspection. The uploaded bytes "
        "are retained in this session for the public dashboard workflows."
    )

    uploaded_file = st.file_uploader(
        "Choose a public research file",
        type=[
            "csv", "json",
            "png", "jpg", "jpeg", "bmp",
            "tif", "tiff", "webp",
            "txt", "dat", "dcm", "dicom",
        ],
        accept_multiple_files=False,
        key="app_main_upload",
    )

    if uploaded_file is not None:
        _store_uploaded_file(uploaded_file)

    raw_bytes = st.session_state["uploaded_data"]

    if raw_bytes is None:
        st.info("Upload a file to begin.")
        return

    filename = st.session_state["uploaded_name"]
    input_type = st.session_state["uploaded_type"]

    st.success(f"Loaded: {filename}")

    c1, c2, c3 = st.columns(3)

    c1.metric("Input type", input_type)
    c2.metric(
        "File size",
        format_bytes(st.session_state["uploaded_size"]),
    )
    c3.metric(
        "SHA-256 prefix",
        st.session_state["uploaded_hash"][:12] + "…",
    )

    st.subheader("Provenance")

    st.json(
        {
            "filename": filename,
            "input_type": input_type,
            "file_size_bytes": len(raw_bytes),
            "sha256": st.session_state["uploaded_hash"],
            "received_at_utc": st.session_state["uploaded_at_utc"],
            "raw_bytes_changed_by_preview": False,
        }
    )

    if input_type == "CSV":
        try:
            frame = pd.read_csv(io.BytesIO(raw_bytes))

            st.subheader("CSV preview")
            st.dataframe(
                frame.head(100),
                use_container_width=True,
            )

            st.caption(
                f"Rows: {len(frame):,} | "
                f"Columns: {len(frame.columns):,}"
            )

        except Exception as exc:
            st.error(
                f"CSV preview failed: {type(exc).__name__}."
            )

    elif input_type == "JSON":
        try:
            parsed = json.loads(raw_bytes.decode("utf-8"))

            st.subheader("JSON preview")
            st.json(parsed)

        except Exception as exc:
            st.error(
                f"JSON preview failed: {type(exc).__name__}."
            )

    elif input_type == "IMAGE":
        try:
            from PIL import Image

            with Image.open(io.BytesIO(raw_bytes)) as image:
                st.subheader("Image preview")
                st.image(
                    image,
                    caption=filename,
                    use_container_width=True,
                )

                st.write(
                    {
                        "format": image.format,
                        "mode": image.mode,
                        "width_px": image.width,
                        "height_px": image.height,
                    }
                )

        except Exception as exc:
            st.error(
                f"Image preview failed: {type(exc).__name__}."
            )

    elif input_type == "TIME-SERIES":
        st.subheader("Text time-series preview")

        st.code(
            raw_bytes.decode("utf-8", errors="replace")[:5000],
            language="text",
        )

    elif input_type == "DICOM":
        st.info(
            "DICOM inspection is handled by Biomedical Mode. "
            "Use properly authorized and de-identified research data."
        )

    else:
        st.warning("Unsupported file type.")


# ============================================================
# PAGE: DATA QUALITY
# ============================================================

def render_data_quality() -> None:
    st.header("🔬 Data Quality & Integrity")

    raw_bytes = st.session_state["uploaded_data"]

    if raw_bytes is None:
        st.info("Upload a dataset from Data Upload first.")
        return

    st.subheader("File integrity")

    st.json(
        {
            "filename": st.session_state["uploaded_name"],
            "input_type": st.session_state["uploaded_type"],
            "size_bytes": st.session_state["uploaded_size"],
            "sha256": st.session_state["uploaded_hash"],
            "received_at_utc": st.session_state["uploaded_at_utc"],
        }
    )

    if st.session_state["uploaded_type"] != "CSV":
        st.info(
            "Detailed tabular quality checks are available for CSV inputs. "
            "Use Biomedical Mode for supported biomedical formats."
        )
        return

    try:
        frame = pd.read_csv(io.BytesIO(raw_bytes))

    except Exception as exc:
        st.error(
            f"Could not load CSV for quality checks: {type(exc).__name__}."
        )
        return

    total_cells = int(frame.shape[0] * frame.shape[1])
    missing_cells = int(frame.isna().sum().sum())
    duplicate_rows = int(frame.duplicated().sum())

    numeric_columns = []

    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            numeric_columns.append(str(column))

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", int(frame.shape[0]))
    c2.metric("Columns", int(frame.shape[1]))
    c3.metric("Numeric columns", len(numeric_columns))

    c4, c5, c6 = st.columns(3)
    c4.metric("Missing cells", missing_cells)
    c5.metric(
        "Missing fraction",
        f"{missing_cells / total_cells:.2%}" if total_cells else "0.00%",
    )
    c6.metric("Duplicate rows", duplicate_rows)

    quality_table = pd.DataFrame(
        {
            "dtype": frame.dtypes.astype(str),
            "missing_count": frame.isna().sum(),
            "missing_fraction": frame.isna().mean(),
        }
    )

    st.subheader("Per-column quality")
    st.dataframe(quality_table, use_container_width=True)

    if missing_cells:
        st.warning("Missing values detected; no rows were changed.")

    if duplicate_rows:
        st.warning("Duplicate rows detected; no rows were removed.")

    st.caption(
        "These checks describe technical data quality and do not validate "
        "scientific interpretation or clinical significance."
    )


# ============================================================
# PAGE: VISUALIZATION
# ============================================================

def render_visualization() -> None:
    st.header("📊 Exploratory Visualization")

    raw_bytes = st.session_state["uploaded_data"]

    if raw_bytes is None:
        st.info("Upload a file first.")
        return

    input_type = st.session_state["uploaded_type"]

    if input_type == "CSV":
        try:
            frame = pd.read_csv(io.BytesIO(raw_bytes))

        except Exception as exc:
            st.error(
                f"Could not read CSV: {type(exc).__name__}."
            )
            return

        numeric_columns = frame.select_dtypes(
            include="number"
        ).columns.tolist()

        if not numeric_columns:
            st.info("No numeric columns were detected.")
            return

        selected_column = st.selectbox(
            "Select numeric signal",
            options=numeric_columns,
            key="app_visual_signal",
        )

        values = pd.to_numeric(
            frame[selected_column],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)

        st.line_chart(values.reset_index(drop=True))

        st.caption(
            "Sample-index plot only. This is not a physical anomaly detector."
        )

    elif input_type == "IMAGE":
        try:
            from PIL import Image

            with Image.open(io.BytesIO(raw_bytes)) as image:
                st.image(
                    image,
                    caption="Uploaded image — visual inspection only",
                    use_container_width=True,
                )

        except Exception as exc:
            st.error(
                f"Image visualization failed: {type(exc).__name__}."
            )

    else:
        st.info(
            "This input type does not have a visualization workflow "
            "in the main dashboard yet."
        )


# ============================================================
# PAGE: ABOUT
# ============================================================

def render_about() -> None:
    st.header("ℹ️ About D³ VITAL-X")

    st.markdown(
        """
        **D³ VITAL-X Space Intelligence Platform**

        An independent research prototype for scientific data ingestion,
        technical validation, exploratory analysis, and visualization.

        **Public scope**
        - Data upload and provenance
        - Technical quality checks
        - Exploratory signal and image views
        - Research-oriented monitoring interfaces
        - Human review

        **Boundary**
        - No medical diagnosis or treatment recommendation
        - No automatic confirmation of a physical anomaly
        - No claim that uploaded snapshots are real-time feeds
        - Proprietary intelligence-core implementation is not included
        """
    )

    st.divider()

    st.subheader("Module file availability")

    st.json(
        {
            label: {
                "file": config["filename"],
                "path": str(
                    find_public_module(config["filename"]) or "NOT FOUND"
                ),
            }
            for label, config in PAGE_MODULES.items()
        }
    )

    st.subheader("Optional analytics file availability")

    st.json(
        _file_presence_status(OPTIONAL_ANALYTICS_FILES)
    )

    st.subheader("Optional visualization file availability")

    st.json(
        _file_presence_status(OPTIONAL_VISUALIZATION_FILES)
    )

    st.caption(
        "File presence checks do not certify that modules are compatible, "
        "tested, or scientifically validated."
    )


# ============================================================
# SESSION STATE
# ============================================================

_initialize_session_state()


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("🛰️ D³ VITAL-X Space Intelligence Platform")

st.markdown(
    """
    ### One AI Engine. Multiple Worlds. One Signal Language.

    **D³ VITAL-X BANGLADESH — Independent Research Prototype**
    """
)

st.info(
    "ACTIVE DEVELOPMENT — Public framework only. Some adapters, analytics, "
    "and external data-source integrations may remain under development."
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.header("🛰️ D³ VITAL-X")
    st.caption(f"App version: {APP_VERSION}")
    st.divider()

    page = st.radio(
        "Select Mode",
        options=[
            "🏠 Home",
            "📥 Data Upload",
            "🔬 Data Quality",
            "📊 Visualization",
            "🛰️ Space Mode",
            "🩻 Biomedical Mode",
            "📡 Live Mode",
            "ℹ️ About",
        ],
        key="app_navigation",
    )

    st.divider()
    st.subheader("System Status")

    st.success("Public dashboard: Loaded")
    st.success("Session upload interface: Available")

    analytics_status = _file_presence_status(
        OPTIONAL_ANALYTICS_FILES
    )

    if analytics_status["complete"]:
        st.info(
            "Optional analytics files: Present "
            "(compatibility not verified)"
        )
    else:
        st.warning(
            "Optional analytics files: Incomplete"
        )

    visualization_status = _file_presence_status(
        OPTIONAL_VISUALIZATION_FILES
    )

    if visualization_status["complete"]:
        st.info(
            "Optional visualization files: Present "
            "(compatibility not verified)"
        )
    else:
        st.warning(
            "Optional visualization files: Incomplete"
        )

    st.caption("D³ VITAL-X BANGLADESH")


# ============================================================
# ROUTER
# ============================================================

if page == "🏠 Home":
    render_home()

elif page == "📥 Data Upload":
    render_data_upload()

elif page == "🔬 Data Quality":
    render_data_quality()

elif page == "📊 Visualization":
    render_visualization()

elif page in PAGE_MODULES:
    _render_integrated_page(
        page,
        PAGE_MODULES[page],
    )

elif page == "ℹ️ About":
    render_about()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "D³ VITAL-X Space Intelligence Platform | "
    "Independent Research Prototype | "
    f"{PROJECT_STATUS}"
)
