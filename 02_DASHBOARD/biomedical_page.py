"""
D³ VITAL-X Space Intelligence Platform
Module 19: biomedical_page.py

Streamlit page for biomedical technical inspection and exploratory summaries.

Features
--------
- Upload an actual biomedical CSV, raster image, or DICOM file.
- Inspect CSV data quality, missing values, duplicate rows, and numeric summaries.
- Preview raster images and report basic image metadata.
- Inspect a conservative allowlist of DICOM technical metadata.
- Optionally attempt a DICOM pixel preview when decoding is supported.
- Download a strict UTF-8 JSON technical inspection report.

Scientific and privacy safeguards
---------------------------------
- No synthetic biomedical observations are generated.
- Outputs are technical/exploratory diagnostics, not medical advice.
- No disease diagnosis, treatment recommendation, or clinical anomaly declaration.
- Metadata filtering does not guarantee full DICOM or image anonymization.
- The protected intelligence core is not implemented here.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

MODULE_NAME = "biomedical_page"
MODULE_NUMBER = 19
MODULE_VERSION = "1.0.0"

IMAGE_EXTENSIONS = ("png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp")
DICOM_EXTENSIONS = ("dcm", "dicom")
SUPPORTED_EXTENSIONS = ("csv",) + IMAGE_EXTENSIONS + DICOM_EXTENSIONS

# Conservative technical allowlist. Direct patient identifiers, dates, free-text
# descriptions, and study/series/instance UIDs are intentionally excluded.
SAFE_DICOM_FIELDS = (
    "Modality",
    "Rows",
    "Columns",
    "SamplesPerPixel",
    "PhotometricInterpretation",
    "BitsAllocated",
    "BitsStored",
    "PixelRepresentation",
    "NumberOfFrames",
)


def _sha256(data: bytes) -> str:
    """Return the SHA-256 fingerprint of uploaded bytes."""
    return hashlib.sha256(data).hexdigest()


def _strict_json_bytes(data: Any) -> bytes:
    """Serialize strict UTF-8 JSON, replacing non-finite numbers with null."""

    def normalize(value: Any) -> Any:
        if value is None or isinstance(value, (str, bool, int)):
            return value

        if isinstance(value, float):
            return value if math.isfinite(value) else None

        if isinstance(value, dict):
            return {
                str(key): normalize(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]

        if isinstance(value, np.ndarray):
            return normalize(value.tolist())

        if hasattr(value, "item") and callable(value.item):
            try:
                return normalize(value.item())
            except (ValueError, TypeError):
                return str(value)

        return str(value)

    return (
        json.dumps(
            normalize(data),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Return column names with at least one finite numeric value."""
    result: list[str] = []

    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)

        if values.notna().any():
            result.append(str(column))

    return result


def _numeric_summary(
    frame: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Calculate descriptive numeric statistics without clinical interpretation."""
    summary: dict[str, dict[str, Any]] = {}

    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).dropna()

        if values.empty:
            continue

        summary[str(column)] = {
            "valid_count": int(values.size),
            "missing_or_nonfinite_count": int(len(frame) - values.size),
            "mean": float(values.mean()),
            "std_sample": (
                float(values.std(ddof=1))
                if values.size > 1
                else None
            ),
            "variance_sample": (
                float(values.var(ddof=1))
                if values.size > 1
                else None
            ),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }

    return summary


def _show_data_quality(frame: pd.DataFrame) -> dict[str, Any]:
    """Display CSV quality indicators and return them for the report."""
    total_cells = int(frame.shape[0] * frame.shape[1])
    missing_cells = int(frame.isna().sum().sum())

    quality = {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "numeric_columns": len(_numeric_columns(frame)),
        "missing_cells": missing_cells,
        "missing_cell_fraction": (
            missing_cells / total_cells if total_cells else 0.0
        ),
        "duplicate_rows": int(frame.duplicated().sum()),
    }

    st.subheader("Data quality overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", quality["rows"])
    c2.metric("Columns", quality["columns"])
    c3.metric("Numeric columns", quality["numeric_columns"])

    c4, c5, c6 = st.columns(3)
    c4.metric("Missing cells", quality["missing_cells"])
    c5.metric(
        "Missing-cell fraction",
        f"{quality['missing_cell_fraction']:.2%}",
    )
    c6.metric("Duplicate rows", quality["duplicate_rows"])

    if missing_cells:
        st.warning(
            "Missing values were found. Review the dataset and document an "
            "appropriate missing-data policy before downstream analysis."
        )

    if quality["duplicate_rows"]:
        st.warning(
            "Duplicate rows were detected; they have not been removed "
            "automatically."
        )

    return quality


def _safe_dicom_metadata(dataset: Any) -> dict[str, Any]:
    """Extract selected technical DICOM fields; never export patient tags."""
    metadata: dict[str, Any] = {}

    for field in SAFE_DICOM_FIELDS:
        try:
            value = getattr(dataset, field, None)

            if value is not None:
                metadata[field] = str(value)[:100]

        except Exception:
            continue

    return metadata


def _inspect_dicom(raw: bytes) -> tuple[dict[str, Any], Any]:
    """Read DICOM bytes in memory and return safe report metadata plus dataset."""
    try:
        import pydicom

    except ImportError as exc:
        raise ImportError(
            "DICOM inspection requires pydicom. Install it with "
            "`%pip install 'pydicom>=2.4,<4.0'`."
        ) from exc

    dataset = pydicom.dcmread(io.BytesIO(raw), force=False)

    report = {
        "format": "DICOM",
        "readable": True,
        "file_size_bytes": len(raw),
        "sha256": _sha256(raw),
        "safe_metadata": _safe_dicom_metadata(dataset),
        "pixel_data_present": hasattr(dataset, "PixelData"),
        "pixel_array_displayed": False,
        "patient_identifiers_exported": False,
        "full_anonymization_guaranteed": False,
    }

    return report, dataset


def _preview_pixel_array(
    pixel_array: np.ndarray,
) -> np.ndarray | None:
    """Create a display-only uint8 preview for supported grayscale/RGB arrays."""
    pixels = np.asarray(pixel_array)

    if pixels.size == 0 or not np.issubdtype(pixels.dtype, np.number):
        return None

    if pixels.ndim not in (2, 3):
        return None

    if pixels.ndim == 3 and pixels.shape[-1] not in (3, 4):
        return None

    if not np.isfinite(pixels).all():
        pixels = np.nan_to_num(
            pixels,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    if pixels.ndim == 3:
        if pixels.shape[-1] == 4:
            pixels = pixels[..., :3]

        if pixels.min() >= 0 and pixels.max() <= 255:
            return pixels.astype(np.uint8)

        low = float(pixels.min())
        high = float(pixels.max())

        if high <= low:
            return np.zeros(pixels.shape, dtype=np.uint8)

        scaled = (pixels - low) * (255.0 / (high - low))

        return np.clip(scaled, 0, 255).astype(np.uint8)

    low = float(pixels.min())
    high = float(pixels.max())

    if high <= low:
        return np.zeros(pixels.shape, dtype=np.uint8)

    scaled = (pixels - low) * (255.0 / (high - low))

    return np.clip(scaled, 0, 255).astype(np.uint8)


def module_info() -> dict[str, Any]:
    """Return the public module contract."""
    return {
        "module": MODULE_NAME,
        "module_number": MODULE_NUMBER,
        "version": MODULE_VERSION,
        "supported_inputs": ["CSV", "raster images", "DICOM"],
        "clinical_diagnosis": False,
        "treatment_recommendation": False,
        "clinical_anomaly_declaration": False,
        "synthetic_observations_generated": False,
        "protected_core_included": False,
    }


def run_contract_tests() -> dict[str, bool]:
    """Run deterministic helper/contract checks without real biomedical data."""
    fixture = b"D3-VITAL-X contract fixture; not scientific data"

    parsed = json.loads(
        _strict_json_bytes(
            {
                "finite": 1.5,
                "nonfinite": float("nan"),
            }
        ).decode("utf-8")
    )

    sample_frame = pd.DataFrame(
        {
            "signal": [1, 2, "3"],
            "label": ["a", "b", "c"],
        }
    )

    constant_preview = _preview_pixel_array(
        np.zeros((2, 2), dtype=np.float32)
    )

    checks = {
        "module_number_is_19": MODULE_NUMBER == 19,
        "module_version_is_defined": MODULE_VERSION == "1.0.0",
        "render_page_is_callable": callable(render_page),
        "sha256_is_64_hex_chars": len(_sha256(fixture)) == 64,
        "sha256_is_deterministic": _sha256(fixture) == _sha256(fixture),
        "strict_json_normalizes_nonfinite": parsed["nonfinite"] is None,
        "numeric_columns_detected": (
            _numeric_columns(sample_frame) == ["signal"]
        ),
        "constant_pixel_preview_is_safe": (
            constant_preview is not None
            and constant_preview.dtype == np.uint8
            and int(constant_preview.max()) == 0
        ),
        "clinical_boundary_is_disabled": (
            module_info()["clinical_diagnosis"] is False
        ),
        "patient_identifiers_not_in_allowlist": not any(
            field in SAFE_DICOM_FIELDS
            for field in (
                "PatientID",
                "PatientName",
                "PatientBirthDate",
                "StudyInstanceUID",
            )
        ),
    }

    checks["all_checks_passed"] = all(checks.values())

    return checks


def render_page() -> None:
    """Render the Biomedical Technical Inspection page."""
    st.set_page_config(
        page_title="D³ VITAL-X | Biomedical Research",
        page_icon="🩻",
        layout="wide",
    )

    st.title(
        f"🩻 Module {MODULE_NUMBER}: Biomedical Technical Inspection"
    )

    st.caption(
        "Technical inspection and exploratory summaries only. This page does not "
        "diagnose disease, recommend treatment, or confirm clinical anomalies."
    )

    with st.expander(
        "Scientific scope, privacy, and limitations",
        expanded=False,
    ):
        st.markdown(
            """
            - Upload only data you are authorized to inspect.
            - Prefer properly de-identified research data.
            - CSV statistics are descriptive and are not clinical thresholds.
            - Metadata filtering does not guarantee full DICOM/image anonymization.
            - Pixel previews are display aids, not clinical interpretations.
            - No anomaly, disease, or medical outcome is confirmed by this page.
            """
        )

    modality = st.sidebar.selectbox(
        "Select data modality",
        options=[
            "Automatic (by file extension)",
            "Tabular (CSV)",
            "Image (PNG/JPEG etc.)",
            "DICOM",
        ],
        key="biomedical_modality",
    )

    if modality == "Tabular (CSV)":
        allowed_types = ["csv"]

    elif modality == "Image (PNG/JPEG etc.)":
        allowed_types = list(IMAGE_EXTENSIONS)

    elif modality == "DICOM":
        allowed_types = list(DICOM_EXTENSIONS)

    else:
        allowed_types = list(SUPPORTED_EXTENSIONS)

    uploaded = st.file_uploader(
        "Upload a biomedical research file",
        type=allowed_types,
        accept_multiple_files=False,
        key="biomedical_input_upload",
        help="Supported formats: CSV, common raster images, and DICOM.",
    )

    if uploaded is None:
        st.info(
            "Upload an authorized file to begin. No scientific sample data "
            "is generated."
        )

        with st.expander("Module information and local contract tests"):
            st.json(module_info())

            checks = run_contract_tests()
            st.json(checks)

            if checks["all_checks_passed"]:
                st.success("Local contract tests passed.")
            else:
                st.error("One or more local contract tests failed.")

        return

    raw = uploaded.getvalue()
    filename = Path(uploaded.name).name
    suffix = Path(filename).suffix.lower()

    report: dict[str, Any] = {
        "project": "D³ VITAL-X Space Intelligence Platform",
        "page_module": MODULE_NAME,
        "page_module_version": MODULE_VERSION,
        "input": {
            "uploaded_filename": filename,
            "extension": suffix,
            "file_size_bytes": len(raw),
            "sha256": _sha256(raw),
        },
        "inspection": {},
        "scientific_boundary": (
            "Technical inspection and exploratory diagnostics only; not a diagnosis, "
            "treatment recommendation, clinical conclusion, or proof of discovery."
        ),
        "privacy_note": (
            "Review filenames and source files for sensitive information before sharing. "
            "DICOM metadata filtering does not guarantee full anonymization."
        ),
    }

    st.subheader("File integrity")

    c1, c2 = st.columns(2)
    c1.metric("File size", f"{len(raw):,} bytes")
    c2.metric("SHA-256 fingerprint", _sha256(raw)[:16] + "…")

    st.caption("The full SHA-256 fingerprint is included in the JSON report.")

    if suffix == ".csv":
        try:
            frame = pd.read_csv(io.BytesIO(raw))

        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            st.error(f"Could not read the CSV file: {type(exc).__name__}.")
            return

        except Exception as exc:
            st.error(
                f"Unexpected CSV parsing error: {type(exc).__name__}."
            )
            return

        if frame.empty or len(frame.columns) == 0:
            st.error("The uploaded CSV has no usable data rows or columns.")
            return

        st.subheader("Dataset preview")
        st.dataframe(frame.head(20), use_container_width=True)

        quality = _show_data_quality(frame)
        numeric_summary = _numeric_summary(frame)

        missing_by_column = {
            str(column): int(frame[column].isna().sum())
            for column in frame.columns
        }

        report["inspection"] = {
            "format": "CSV",
            "readable": True,
            "row_count": int(frame.shape[0]),
            "column_count": int(frame.shape[1]),
            "columns": [str(column) for column in frame.columns],
            "quality": quality,
            "missing_values_by_column": missing_by_column,
            "numeric_summary": numeric_summary,
        }

        st.subheader("Missing values by column")

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "column": column,
                        "missing_count": count,
                    }
                    for column, count in missing_by_column.items()
                ]
            ),
            use_container_width=True,
        )

        if numeric_summary:
            st.subheader("Descriptive numeric summaries")
            st.dataframe(
                pd.DataFrame(numeric_summary).T,
                use_container_width=True,
            )
            st.caption(
                "Descriptive statistics only; no clinical meaning is inferred."
            )

        else:
            st.info(
                "No usable numeric columns were found for descriptive summaries."
            )

    elif suffix in {f".{ext}" for ext in IMAGE_EXTENSIONS}:
        try:
            from PIL import Image, ImageOps

            with Image.open(io.BytesIO(raw)) as check_image:
                check_image.verify()

            with Image.open(io.BytesIO(raw)) as opened_image:
                image = ImageOps.exif_transpose(opened_image).copy()
                image_format = str(opened_image.format or "unknown")

        except Exception as exc:
            st.error(
                f"The image could not be safely read: {type(exc).__name__}."
            )
            return

        report["inspection"] = {
            "format": "RASTER_IMAGE",
            "readable": True,
            "image_format": image_format,
            "width_px": int(image.width),
            "height_px": int(image.height),
            "mode": str(image.mode),
            "pixel_analysis_performed": False,
            "exif_metadata_exported": False,
        }

        st.subheader("Image technical metadata")

        st.write(
            {
                "Format": image_format,
                "Width (px)": int(image.width),
                "Height (px)": int(image.height),
                "Color mode": str(image.mode),
            }
        )

        st.image(
            image,
            caption="Uploaded image — preview only",
            use_container_width=True,
        )

        st.caption(
            "EXIF metadata is not exported by this page; inspect source files "
            "before sharing."
        )

    elif suffix in {f".{ext}" for ext in DICOM_EXTENSIONS}:
        try:
            dicom_report, dataset = _inspect_dicom(raw)

        except ImportError as exc:
            st.error(str(exc))
            return

        except Exception as exc:
            st.error(
                "DICOM parsing failed. Confirm the file is valid and uses a "
                "supported transfer syntax. "
                f"Error type: {type(exc).__name__}."
            )
            return

        report["inspection"] = dicom_report

        st.subheader("DICOM technical metadata (allowlist)")
        st.json(dicom_report["safe_metadata"])

        st.caption(
            "Patient identifiers and free-text descriptions are not exported. "
            "This does not guarantee that the original file is fully de-identified."
        )

        if dicom_report["pixel_data_present"] and st.checkbox(
            "Attempt pixel preview (no clinical interpretation)",
            value=False,
            key="biomedical_dicom_preview",
        ):
            try:
                pixels = np.asarray(dataset.pixel_array)
                preview = _preview_pixel_array(pixels)

                report["inspection"]["pixel_shape"] = [
                    int(size) for size in pixels.shape
                ]
                report["inspection"]["pixel_dtype"] = str(pixels.dtype)

                if preview is None:
                    st.info(
                        f"Pixel data shape {tuple(pixels.shape)} is not supported "
                        "for this simple preview. Metadata inspection remains available."
                    )

                else:
                    st.image(
                        preview,
                        caption="DICOM pixel preview — no diagnosis",
                        use_container_width=True,
                    )
                    report["inspection"]["pixel_array_displayed"] = True

            except Exception as exc:
                st.warning(
                    "Pixel decoding was unavailable; metadata inspection remains "
                    "available. "
                    f"Error type: {type(exc).__name__}."
                )

    else:
        st.error("Unsupported file extension. Choose a listed file type.")
        return

    st.divider()
    st.subheader("Export inspection report")

    st.caption(
        "The report excludes raw image pixels and DICOM patient identifiers. "
        "Review the filename and report for sensitive information before sharing."
    )

    try:
        report_bytes = _strict_json_bytes(report)

        st.download_button(
            label="Download biomedical inspection report (JSON)",
            data=report_bytes,
            file_name="d3_vital_x_biomedical_inspection.json",
            mime="application/json",
            key="biomedical_report_download",
        )

    except (TypeError, ValueError) as exc:
        st.error(f"Could not prepare the report: {type(exc).__name__}.")

    with st.expander("Module information and local contract tests"):
        st.json(module_info())

        checks = run_contract_tests()
        st.json(checks)

        if checks["all_checks_passed"]:
            st.success("Local contract tests passed.")
        else:
            st.error("One or more local contract tests failed.")


if __name__ == "__main__":
    render_page()
