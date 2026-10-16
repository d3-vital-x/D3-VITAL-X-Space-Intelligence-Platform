
"""
D³ VITAL-X Space Intelligence Platform
Module 18: space_page.py
Version: 1.1.0

Exploratory analytics for uploaded space-observation datasets.

Features
--------
- Upload and inspect observational CSV data
- Inspect data quality and numeric columns
- Calculate entropy and variance
- Evaluate coupling between two signals
- Optional baseline-dependent anomaly-candidate evaluation
- Download a JSON analysis report

Scientific safeguards
---------------------
- No synthetic scientific observations are generated.
- Missing values are handled explicitly.
- Paired signals preserve sample alignment during coupling analysis.
- Baseline validation is performed before evaluation.
- Results are exploratory diagnostics, not confirmed discoveries.
- The protected intelligence core is not implemented here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# MODULE METADATA AND PATHS
# ---------------------------------------------------------

PAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PAGE_DIR.parent.parent

MODULE_NAME = "space_page"
MODULE_NUMBER = 18
MODULE_VERSION = "1.1.0"


# ---------------------------------------------------------
# ROBUST ANALYTICS MODULE DISCOVERY
# ---------------------------------------------------------

PAGE_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = None
ANALYTICS_DIR = None

for candidate in (PAGE_DIR, *PAGE_DIR.parents):
    possible_analytics = candidate / "06_ANALYTICS"
    required_file = possible_analytics / "entropy_variance.py"

    if required_file.is_file():
        PROJECT_ROOT = candidate
        ANALYTICS_DIR = possible_analytics
        break

if ANALYTICS_DIR is not None:
    analytics_path = str(ANALYTICS_DIR)

    # Prioritize the exact directory containing the modules.
    if analytics_path in sys.path:
        sys.path.remove(analytics_path)

    sys.path.insert(0, analytics_path)

else:
    # Keep the diagnostic explicit if deployment files are missing.
    st.error(
        "Analytics module directory could not be located."
    )
    st.code(
        f"Page directory: {PAGE_DIR}\n"
        f"Working directory: {Path.cwd()}\n"
        "Expected file: 06_ANALYTICS/entropy_variance.py"
    )
    st.stop()

# Diagnostic metadata for deployment verification.
st.session_state["space_analytics_path"] = str(ANALYTICS_DIR)


# ---------------------------------------------------------
# SERIALIZATION HELPERS
# ---------------------------------------------------------

def _to_plain_data(value: Any) -> Any:
    """Convert supported result objects into report-compatible data."""

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()

    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(value)

    if isinstance(value, dict):
        return value

    raise ValueError(
        "Result must be a dictionary, dataclass, or expose to_dict()."
    )


def _strict_json_bytes(data: Any) -> bytes:
    """Serialize a report as UTF-8 JSON without NaN or Infinity."""

    return (
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def _finite_number(value: Any, name: str) -> float:
    """Validate and convert a value to a finite floating-point number."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"'{name}' must be numeric, not boolean.")

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"'{name}' must be numeric.") from exc

    if not np.isfinite(number):
        raise ValueError(f"'{name}' must be finite.")

    return number


# ---------------------------------------------------------
# DATA HELPERS
# ---------------------------------------------------------

def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Identify columns containing at least one numeric value."""

    columns: list[str] = []

    for column in frame.columns:
        converted = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        if converted.notna().any():
            columns.append(column)

    return columns


def _get_numeric_series(
    frame: pd.DataFrame,
    column: str,
) -> np.ndarray:
    """Convert a selected column into a float64 array."""

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    ).to_numpy(dtype=np.float64)


def _show_data_quality(frame: pd.DataFrame) -> dict[str, Any]:
    """Display basic data-quality diagnostics."""

    total_cells = int(frame.shape[0] * frame.shape[1])
    missing_cells = int(frame.isna().sum().sum())

    quality = {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "numeric_columns": len(_numeric_columns(frame)),
        "missing_cells": missing_cells,
        "missing_cell_fraction": (
            missing_cells / total_cells
            if total_cells > 0
            else 0.0
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
            "Missing values were detected. Review the selected signals "
            "and justify the chosen missing-data policy."
        )

    if quality["duplicate_rows"]:
        st.warning(
            "Duplicate rows were detected. They have not been removed "
            "automatically."
        )

    return quality


# ---------------------------------------------------------
# BASELINE VALIDATION
# ---------------------------------------------------------

def _baseline_from_upload(uploaded_file: Any) -> dict[str, dict[str, float]]:
    """
    Parse and validate the baseline JSON.

    Required format:
    {
        "metric_name": {
            "mean": 1.0,
            "std": 0.2
        }
    }
    """

    try:
        raw = json.loads(
            uploaded_file.getvalue().decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Baseline must be valid UTF-8 JSON."
        ) from exc

    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            "Baseline JSON must contain a non-empty object."
        )

    validated: dict[str, dict[str, float]] = {}

    for name, stats in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "Every baseline metric must have a non-empty string name."
            )

        if not isinstance(stats, dict):
            raise ValueError(
                f"Baseline entry '{name}' must be an object."
            )

        if "mean" not in stats or "std" not in stats:
            raise ValueError(
                f"Baseline entry '{name}' requires mean and std."
            )

        mean = _finite_number(
            stats["mean"],
            f"{name}.mean",
        )

        std = _finite_number(
            stats["std"],
            f"{name}.std",
        )

        if std < 0:
            raise ValueError(
                f"Baseline standard deviation for '{name}' "
                "cannot be negative."
            )

        validated[name] = {
            "mean": mean,
            "std": std,
        }

    return validated


# ---------------------------------------------------------
# MAIN PAGE
# ---------------------------------------------------------

def render_page() -> None:
    """Render the Space Intelligence analytics page."""

    # Must be the first Streamlit command on this page.
    st.set_page_config(
        page_title="D³ VITAL-X | Space Analytics",
        page_icon="🛰️",
        layout="wide",
    )

    st.title("🛰️ D³ VITAL-X Space Intelligence")

    st.caption(
        "Exploratory analytics for uploaded space-observation data. "
        "A score is a diagnostic indicator, not a confirmed discovery."
    )

    with st.expander(
        "Scientific scope and limitations",
        expanded=False,
    ):
        st.markdown(
            """
            - Use observational data with documented provenance.
            - Entropy and variance depend on the signal and analysis settings.
            - Coupling measures association, not causation.
            - Equal-length signals are not necessarily physically aligned.
            - Baseline comparisons require compatible metrics and units.
            - Anomaly-candidate scores do not establish a physical event.
            - This page does not provide medical diagnoses.
            """
        )

    # -----------------------------------------------------
    # CSV UPLOAD
    # -----------------------------------------------------

    uploaded_csv = st.file_uploader(
        "Upload an observational CSV dataset",
        type=["csv"],
        help=(
            "Upload data you are authorized to analyze. "
            "Do not upload credentials or confidential information."
        ),
        key="space_csv_upload",
    )

    if uploaded_csv is None:
        st.info(
            "Upload a CSV file to begin. No synthetic scientific "
            "observations are generated."
        )
        return

    try:
        frame = pd.read_csv(uploaded_csv)
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        return

    if frame.empty:
        st.error("The uploaded CSV contains no data rows.")
        return

    if frame.shape[1] == 0:
        st.error("The uploaded CSV contains no columns.")
        return

    # Clear a previous report when the uploaded dataset changes.
    current_signature = (
        uploaded_csv.name,
        len(uploaded_csv.getvalue()),
    )

    if (
        st.session_state.get("space_dataset_signature")
        != current_signature
    ):
        st.session_state.pop("space_analysis_report", None)
        st.session_state.pop("space_baseline_result", None)
        st.session_state["space_dataset_signature"] = (
            current_signature
        )

    # -----------------------------------------------------
    # DATA PREVIEW AND QUALITY
    # -----------------------------------------------------

    st.subheader("Dataset preview")

    st.dataframe(
        frame.head(20),
        use_container_width=True,
    )

    quality = _show_data_quality(frame)
    numeric_columns = _numeric_columns(frame)

    if not numeric_columns:
        st.error(
            "No usable numeric columns were found. "
            "Check the CSV columns and data types."
        )
        return

    st.divider()
    st.subheader("Signal selection")

    selected_x = st.selectbox(
        "Primary signal",
        options=numeric_columns,
        key="space_primary_signal",
    )

    secondary_options = [
        column
        for column in numeric_columns
        if column != selected_x
    ]

    selected_y = None

    if secondary_options:
        selected_y = st.selectbox(
            "Second signal (for coupling analysis)",
            options=secondary_options,
            key="space_secondary_signal",
        )
    else:
        st.info(
            "At least two numeric columns are needed for coupling analysis."
        )

    entropy_bins = st.slider(
        "Entropy histogram bins",
        min_value=4,
        max_value=128,
        value=32,
        step=4,
        key="space_entropy_bins",
    )

    missing_policy_label = st.radio(
        "Missing / non-finite value handling",
        options=[
            "Reject selected signal if any value is missing",
            "Omit invalid values for each analysis",
        ],
        index=0,
        key="space_missing_policy",
    )

    missing_policy = (
        "omit"
        if missing_policy_label.startswith("Omit")
        else "reject"
    )

    # -----------------------------------------------------
    # RUN ANALYTICS
    # -----------------------------------------------------

    st.divider()
    st.subheader("Signal analytics")

    if st.button(
        "Run selected analytics",
        type="primary",
        key="space_run_analytics",
    ):
        # Prevent stale results being exported if this run fails.
        st.session_state.pop("space_analysis_report", None)
        st.session_state.pop("space_baseline_result", None)

        analysis_report: dict[str, Any] = {
            "project": "D³ VITAL-X Space Intelligence Platform",
            "page_module": MODULE_NAME,
            "module_number": MODULE_NUMBER,
            "page_module_version": MODULE_VERSION,
            "dataset": {
                "uploaded_filename": uploaded_csv.name,
                "row_count": int(frame.shape[0]),
                "column_count": int(frame.shape[1]),
                "quality": quality,
            },
            "analysis_settings": {
                "primary_signal": selected_x,
                "secondary_signal": selected_y,
                "entropy_bins": entropy_bins,
                "missing_policy": missing_policy,
            },
            "analyses": {},
            "scientific_boundary": (
                "Exploratory diagnostics only. Results do not prove "
                "an anomaly, causation, a physical event, or a clinical "
                "condition."
            ),
        }

        try:
            from entropy_variance import (
                calculate_entropy_variance,
            )

            x = _get_numeric_series(frame, selected_x)

            if missing_policy == "reject":
                if not np.isfinite(x).all():
                    raise ValueError(
                        f"Signal '{selected_x}' contains missing or "
                        "non-finite values. Inspect the data or choose "
                        "omission if scientifically justified."
                    )

            entropy_result = calculate_entropy_variance(
                x,
                entropy_bins=entropy_bins,
                missing_policy=missing_policy,
            )

            entropy_data = _to_plain_data(entropy_result)

            analysis_report["analyses"]["entropy_variance"] = (
                entropy_data
            )

            st.subheader("Entropy and variance")
            st.json(entropy_data)

            # ---------------------------------------------
            # COUPLING ANALYSIS
            # ---------------------------------------------

            if selected_y is not None:
                from coupling_transition import (
                    analyze_coupling_transition,
                )

                y = _get_numeric_series(frame, selected_y)

                if len(x) != len(y):
                    raise ValueError(
                        "Selected signals have different lengths."
                    )

                if missing_policy == "reject":
                    if not np.isfinite(y).all():
                        raise ValueError(
                            f"Signal '{selected_y}' contains missing "
                            "or non-finite values."
                        )

                    x_pair = x
                    y_pair = y

                else:
                    # Preserve original row alignment. Do not independently
                    # omit values from x and y before coupling analysis.
                    valid_pair_mask = (
                        np.isfinite(x) & np.isfinite(y)
                    )

                    x_pair = x[valid_pair_mask]
                    y_pair = y[valid_pair_mask]

                paired_count = len(x_pair)

                if paired_count < 2:
                    raise ValueError(
                        "At least two jointly valid paired samples "
                        "are required for coupling analysis."
                    )

                window_size = min(200, paired_count)

                step = max(
                    1,
                    min(50, max(1, paired_count // 4)),
                )

                min_samples = min(
                    10,
                    window_size,
                )

                coupling_result = analyze_coupling_transition(
                    x_pair,
                    y_pair,
                    window_size=window_size,
                    step=step,
                    min_samples=min_samples,
                    # Paired data have already been filtered jointly.
                    missing_policy="reject",
                )

                coupling_data = _to_plain_data(
                    coupling_result
                )

                analysis_report["analyses"][
                    "coupling_transition"
                ] = coupling_data

                analysis_report["analysis_settings"][
                    "coupling"
                ] = {
                    "original_sample_count": int(len(x)),
                    "jointly_valid_sample_count": int(paired_count),
                    "window_size": int(window_size),
                    "step": int(step),
                    "min_samples": int(min_samples),
                    "pairwise_alignment": (
                        "Original row order preserved; invalid pairs "
                        "omitted jointly when omission was selected."
                    ),
                }

                st.subheader("Coupling diagnostics")

                st.caption(
                    f"Jointly valid paired samples: {paired_count}"
                )

                st.json(coupling_data)

            # Save only after all requested analyses complete.
            st.session_state["space_analysis_report"] = (
                analysis_report
            )

            st.success(
                "Selected analytics completed. Review the diagnostics "
                "before interpreting the results."
            )

        except ImportError as exc:
            st.error(
                "A required analytics module could not be imported. "
                "Check the repository paths, module filenames, and "
                "dependencies."
            )
            st.code(str(exc))

        except (ValueError, TypeError, KeyError) as exc:
            st.error(f"Analysis could not be completed: {exc}")

        except Exception as exc:
            st.error(
                "An unexpected error occurred. Review the input data "
                "and analytics module interfaces."
            )
            st.code(f"{type(exc).__name__}: {exc}")

    # -----------------------------------------------------
    # OPTIONAL BASELINE EVALUATION
    # -----------------------------------------------------

    st.divider()
    st.subheader("Optional anomaly-candidate evaluation")

    st.write(
        "Upload a documented reference baseline in JSON format. "
        "The baseline metrics must match the observed metrics and use "
        "compatible units."
    )

    baseline_file = st.file_uploader(
        "Upload baseline JSON",
        type=["json"],
        key="space_baseline_upload",
    )

    st.caption(
        'Expected format: {"metric_name": {"mean": 1.0, "std": 0.2}}'
    )

    if st.button(
        "Evaluate against supplied baseline",
        key="space_evaluate_baseline",
    ):
        # Do not leave a previous baseline score visible after failure.
        st.session_state.pop("space_baseline_result", None)

        previous_report = st.session_state.get(
            "space_analysis_report"
        )

        if not previous_report:
            st.warning(
                "Run the selected analytics successfully first."
            )

        elif baseline_file is None:
            st.warning(
                "Upload a documented baseline JSON before evaluation."
            )

        else:
            try:
                from anomaly_evaluator import (
                    evaluate_anomaly_candidates,
                )

                baseline = _baseline_from_upload(
                    baseline_file
                )

                entropy_data = previous_report[
                    "analyses"
                ].get("entropy_variance", {})

                if not isinstance(entropy_data, dict):
                    raise ValueError(
                        "Entropy/variance results must be a dictionary."
                    )

                observed_metrics: dict[str, float] = {}

                for metric_name in (
                    "entropy_bits",
                    "variance",
                ):

                    if metric_name in entropy_data:
                        observed_metrics[metric_name] = (
                            _finite_number(
                                entropy_data[metric_name],
                                metric_name,
                            )
                        )

                if not observed_metrics:
                    raise ValueError(
                        "No recognized finite entropy/variance metrics "
                        "were found in the analysis result."
                    )

                matching_metrics = {
                    name: value
                    for name, value in observed_metrics.items()
                    if name in baseline
                }

                if not matching_metrics:
                    raise ValueError(
                        "No baseline metric names match the observed "
                        "metrics. Expected matching names include "
                        "'shannon_entropy_bits' or 'variance'."
                    )

                # Evaluate only metrics with explicit baseline entries.
                candidate_result = evaluate_anomaly_candidates(
                    matching_metrics,
                    baseline,
                    threshold=3.0,
                    min_evaluated_metrics=1,
                )

                candidate_data = _to_plain_data(
                    candidate_result
                )

                previous_report["analyses"][
                    "anomaly_evaluator"
                ] = candidate_data

                previous_report["baseline_metadata"] = {
                    "uploaded_filename": baseline_file.name,
                    "evaluated_metrics": list(
                        matching_metrics.keys()
                    ),
                    "screening_threshold": 3.0,
                    "scientific_interpretation": (
                        "Baseline-dependent screening only; not a "
                        "confirmed anomaly or discovery."
                    ),
                }

                st.session_state["space_analysis_report"] = (
                    previous_report
                )

                st.session_state["space_baseline_result"] = (
                    candidate_data
                )

                st.subheader("Anomaly-candidate result")
                st.json(candidate_data)

                st.warning(
                    "This is a baseline-dependent screening result, "
                    "not a confirmed physical anomaly or discovery."
                )

            except ImportError as exc:
                st.error(
                    "Could not import anomaly_evaluator. Check the "
                    "06_ANALYTICS folder and module filename."
                )
                st.code(str(exc))

            except (ValueError, TypeError, KeyError) as exc:
                st.error(f"Baseline evaluation failed: {exc}")

            except Exception as exc:
                st.error(
                    "Unexpected baseline evaluation error."
                )
                st.code(f"{type(exc).__name__}: {exc}")

    # -----------------------------------------------------
    # JSON REPORT EXPORT
    # -----------------------------------------------------

    st.divider()
    st.subheader("Export report")

    final_report = st.session_state.get(
        "space_analysis_report"
    )

    if final_report:
        try:
            report_bytes = _strict_json_bytes(
                final_report
            )

            st.download_button(
                label="Download analysis report (JSON)",
                data=report_bytes,
                file_name="d3_vital_x_space_report.json",
                mime="application/json",
                key="space_download_report",
            )

            st.caption(
                "The report contains analysis metadata and results. "
                "Review it for sensitive information before sharing."
            )

        except (TypeError, ValueError) as exc:
            st.error(
                f"Could not prepare the JSON report: {exc}"
            )

    else:
        st.info(
            "Run the selected analytics successfully to enable "
            "report export."
        )


# ---------------------------------------------------------
# STREAMLIT ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    render_page()
