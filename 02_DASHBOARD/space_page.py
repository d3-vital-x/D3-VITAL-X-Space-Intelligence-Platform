"""
D³ VITAL-X Space Intelligence Platform
Module 18: space_page.py

Streamlit page for exploratory space-data analytics.

Features
--------
- Upload an actual observational CSV dataset
- Inspect data quality and numeric columns
- Calculate entropy and variance for a selected signal
- Evaluate coupling between two aligned signals
- Optionally evaluate metrics against a user-provided baseline JSON
- Download a JSON report

Scientific safeguards
---------------------
- No synthetic scientific observations are generated.
- No anomaly is confirmed automatically.
- Baseline-based evaluation is optional and requires an explicit baseline.
- Outputs are exploratory diagnostics, not physical or medical conclusions.
- The protected intelligence core is not implemented here.
"""

from __future__ import annotations

import io
import json
import sys
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


# Resolve project modules dynamically to bypass numeric folder import issues.
PAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PAGE_DIR.parent

ANALYTICS_DIR = PROJECT_ROOT / "06_ANALYTICS"
VISUALIZATION_DIR = PROJECT_ROOT / "07_VISUALIZATION"

for module_dir in (ANALYTICS_DIR, VISUALIZATION_DIR):
    module_path = str(module_dir)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)


MODULE_NAME = "space_page"
MODULE_NUMBER = 18
MODULE_VERSION = "1.0.0"


def _to_plain_data(value: Any) -> Any:
    """Convert common result objects to JSON-compatible data."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()

    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(value)

    if isinstance(value, dict):
        return value

    raise ValueError(
        "The result must be a dictionary or expose to_dict()."
    )


def _strict_json_bytes(data: Any) -> bytes:
    """Serialize a report as strict UTF-8 JSON."""
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


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns that can be interpreted as numeric."""
    columns: list[str] = []

    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().any():
            columns.append(str(column))

    return columns


def _get_numeric_series(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Convert a selected column to floating-point values."""
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(
        dtype=np.float64
    )


def _show_data_quality(frame: pd.DataFrame) -> dict[str, Any]:
    """Display basic CSV data-quality information."""
    total_cells = int(frame.shape[0] * frame.shape[1])
    missing_cells = int(frame.isna().sum().sum())

    numeric_count = len(_numeric_columns(frame))

    quality = {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "numeric_columns": numeric_count,
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
            "Missing values were found. Review the selected signals and "
            "choose an appropriate missing-data policy before analysis."
        )

    if quality["duplicate_rows"]:
        st.warning(
            "Duplicate rows were detected. They have not been removed "
            "automatically."
        )

    return quality


def _baseline_from_upload(uploaded_file: Any) -> dict[str, Any]:
    """Parse and validate the uploaded baseline JSON structure."""
    try:
        raw = json.loads(uploaded_file.getvalue().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("The baseline upload is not valid UTF-8 JSON.") from exc

    if not isinstance(raw, dict) or not raw:
        raise ValueError("The baseline JSON must contain a non-empty object.")

    for name, stats in raw.items():
        if not isinstance(name, str) or not isinstance(stats, dict):
            raise ValueError(
                "Each baseline entry must map a metric name to an object."
            )
        if "mean" not in stats or "std" not in stats:
            raise ValueError(
                f"Baseline entry '{name}' must contain mean and std."
            )

    return raw


def render_page() -> None:
    """Render the Space Analytics page."""
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

    with st.expander("Scientific scope and limitations", expanded=False):
        st.markdown(
            """
            - Upload an actual observational dataset with documented provenance.
            - Entropy and variance depend on the data and analysis settings.
            - Coupling describes association, not causation.
            - Anomaly-candidate evaluation requires an independent, justified
              baseline supplied by the user.
            - This dashboard does not provide medical diagnosis or confirm
              physical events.
            """
        )

    uploaded_csv = st.file_uploader(
        "Upload a real observational CSV dataset",
        type=["csv"],
        help=(
            "Use data you are authorized to analyze. Do not upload credentials "
            "or confidential information."
        ),
    )

    if uploaded_csv is None:
        st.info(
            "Upload a CSV file to begin. No scientific sample data is "
            "generated or displayed when no dataset is supplied."
        )
        return

    try:
        frame = pd.read_csv(uploaded_csv)
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        return

    if frame.empty:
        st.error("The uploaded CSV has no data rows.")
        return

    st.subheader("Dataset preview")
    st.dataframe(frame.head(20), use_container_width=True)

    quality = _show_data_quality(frame)
    numeric_columns = _numeric_columns(frame)

    if not numeric_columns:
        st.error(
            "No usable numeric columns were found. Check the CSV format "
            "and the column data types."
        )
        return

    st.divider()
    st.subheader("Signal selection")

    selected_x = st.selectbox(
        "Primary signal",
        options=numeric_columns,
        key="space_primary_signal",
    )

    selected_y_options = [
        column for column in numeric_columns if column != selected_x
    ]

    selected_y = None
    if selected_y_options:
        selected_y = st.selectbox(
            "Second signal (for coupling analysis)",
            options=selected_y_options,
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
    )

    missing_policy_label = st.radio(
        "Missing / non-finite value handling",
        options=[
            "Reject selected signal if any value is missing",
            "Omit invalid values for each analysis",
        ],
        index=0,
    )

    missing_policy = (
        "omit"
        if missing_policy_label.startswith("Omit")
        else "reject"
    )

    analysis_report: dict[str, Any] = {
        "project": "D³ VITAL-X Space Intelligence Platform",
        "page_module": MODULE_NAME,
        "page_module_version": MODULE_VERSION,
        "dataset": {
            "uploaded_filename": uploaded_csv.name,
            "row_count": int(frame.shape[0]),
            "column_count": int(frame.shape[1]),
            "quality": quality,
        },
        "analyses": {},
        "scientific_boundary": (
            "Exploratory diagnostics only; not proof of an anomaly, "
            "causation, a physical event, or a clinical condition."
        ),
    }

    if st.button("Run selected analytics", type="primary"):
        try:
            # Dynamically load entropy_variance from 06_ANALYTICS
            ev_path = ANALYTICS_DIR / "entropy_variance.py"
            ev_spec = importlib.util.spec_from_file_location("entropy_variance", ev_path)
            entropy_variance = importlib.util.module_from_spec(ev_spec)
            ev_spec.loader.exec_module(entropy_variance)
            calculate_entropy_variance = entropy_variance.calculate_entropy_variance

            x = _get_numeric_series(frame, selected_x)

            if missing_policy == "reject" and not np.isfinite(x).all():
                raise ValueError(
                    f"Signal '{selected_x}' contains missing or non-finite "
                    "values. Choose omission only if scientifically justified."
                )

            entropy_result = calculate_entropy_variance(
                x,
                entropy_bins=entropy_bins,
                missing_policy=missing_policy,
            )

            entropy_data = _to_plain_data(entropy_result)
            analysis_report["analyses"]["entropy_variance"] = entropy_data

            st.success("Entropy and variance analysis completed.")
            st.json(entropy_data)

            if selected_y is not None:
                # Dynamically load coupling_transition from 06_ANALYTICS
                ct_path = ANALYTICS_DIR / "coupling_transition.py"
                ct_spec = importlib.util.spec_from_file_location("coupling_transition", ct_path)
                coupling_transition = importlib.util.module_from_spec(ct_spec)
                ct_spec.loader.exec_module(coupling_transition)
                analyze_coupling_transition = coupling_transition.analyze_coupling_transition

                y = _get_numeric_series(frame, selected_y)

                if len(x) != len(y):
                    raise ValueError("The selected signals are not aligned.")

                coupling_result = analyze_coupling_transition(
                    x,
                    y,
                    window_size=min(200, max(2, len(x))),
                    step=max(1, min(50, max(1, len(x) // 4))),
                    min_samples=max(2, min(10, len(x))),
                    missing_policy=missing_policy,
                )

                coupling_data = _to_plain_data(coupling_result)
                analysis_report["analyses"]["coupling_transition"] = (
                    coupling_data
                )

                st.subheader("Coupling diagnostics")
                st.json(coupling_data)

            st.session_state["space_analysis_report"] = analysis_report
            st.session_state["space_selected_x"] = selected_x
            st.session_state["space_selected_y"] = selected_y

        except ImportError as exc:
            st.error(
                "A required analytics module could not be imported. "
                "Check the repository paths and module filenames."
            )
            st.code(str(exc))
        except (ValueError, TypeError, KeyError) as exc:
            st.error(f"Analysis could not be completed: {exc}")
        except Exception as exc:
            st.error(
                "An unexpected error occurred. Review the uploaded data "
                "and module versions before retrying."
            )
            st.code(f"{type(exc).__name__}: {exc}")

    st.divider()
    st.subheader("Optional anomaly-candidate evaluation")

    st.write(
        "Upload a JSON baseline created from a documented reference dataset. "
        "Without an appropriate baseline, no anomaly-candidate score is shown."
    )

    baseline_file = st.file_uploader(
        "Upload baseline JSON",
        type=["json"],
        key="space_baseline_upload",
    )

    if baseline_file is not None:
        st.caption(
            "Expected format: metric names mapped to objects containing "
            "'mean' and 'std'. Baseline units must match the observed metrics."
        )

    if st.button("Evaluate against supplied baseline"):
        previous_report = st.session_state.get("space_analysis_report")

        if not previous_report:
            st.warning("Run the selected analytics first.")
        elif baseline_file is None:
            st.warning(
                "Upload a documented baseline JSON before running "
                "anomaly-candidate evaluation."
            )
        else:
            try:
                # Dynamically load anomaly_evaluator from 06_ANALYTICS
                ae_path = ANALYTICS_DIR / "anomaly_evaluator.py"
                ae_spec = importlib.util.spec_from_file_location("anomaly_evaluator", ae_path)
                anomaly_evaluator = importlib.util.module_from_spec(ae_spec)
                ae_spec.loader.exec_module(anomaly_evaluator)
                evaluate_anomaly_candidates = anomaly_evaluator.evaluate_anomaly_candidates

                baseline = _baseline_from_upload(baseline_file)

                # Use explicitly supplied observed metrics. The baseline
                # must use these exact metric names and matching units.
                entropy_data = previous_report["analyses"].get(
                    "entropy_variance", {}
                )

                observed_metrics = {}
                if "shannon_entropy_bits" in entropy_data:
                    observed_metrics["shannon_entropy_bits"] = (
                        entropy_data["shannon_entropy_bits"]
                    )
                if "variance" in entropy_data:
                    observed_metrics["variance"] = entropy_data["variance"]

                if not observed_metrics:
                    st.error(
                        "No recognized entropy/variance fields were found "
                        "in the previous analytics result."
                    )
                else:
                    candidate_result = evaluate_anomaly_candidates(
                        observed_metrics,
                        baseline,
                        threshold=3.0,
                        min_evaluated_metrics=1,
                    )

                    candidate_data = _to_plain_data(candidate_result)
                    previous_report["analyses"]["anomaly_evaluator"] = (
                        candidate_data
                    )
                    st.session_state["space_analysis_report"] = previous_report

                    st.subheader("Anomaly-candidate result")
                    st.json(candidate_data)

                    st.warning(
                        "This is a baseline-dependent screening result, "
                        "not a confirmed anomaly or discovery."
                    )

            except (ValueError, TypeError, KeyError) as exc:
                st.error(f"Baseline evaluation failed: {exc}")
            except ImportError as exc:
                st.error(f"Could not import the evaluator module: {exc}")

    st.divider()
    st.subheader("Export report")

    final_report = st.session_state.get("space_analysis_report")

    if final_report:
        try:
            report_bytes = _strict_json_bytes(final_report)

            st.download_button(
                label="Download analysis report (JSON)",
                data=report_bytes,
                file_name="d3_vital_x_space_report.json",
                mime="application/json",
            )

            st.caption(
                "The exported report contains analysis metadata and results. "
                "Review it for sensitive information before sharing publicly."
            )
        except (TypeError, ValueError) as exc:
            st.error(f"Could not prepare the report for download: {exc}")
    else:
        st.info("Run an analysis to enable report export.")


if __name__ == "__main__":
    render_page()
