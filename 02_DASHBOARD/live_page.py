"""
D³ VITAL-X Space Intelligence Platform
Module 20: live_page.py

Version: 1.0.0

Streamlit page for live-data status and observational signal monitoring.

Features
--------
- Upload an actual observational CSV or JSON snapshot.
- Inspect data freshness using a user-selected timestamp column.
- Review data quality, missing values, duplicates, and numeric signals.
- Display timestamp coverage and descriptive signal statistics.
- Flag stale or unverified data as a monitoring status, not an anomaly.
- Download a strict JSON monitoring report.

Scientific safeguards
---------------------
- No synthetic scientific observations are generated.
- File uploads are snapshots, not automatically live data feeds.
- No anomaly or physical event is automatically confirmed.
- Freshness thresholds are configurable monitoring settings.
- Correlation and descriptive statistics do not establish causation.
- The protected intelligence core is not implemented here.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


MODULE_NAME = "live_page"
MODULE_NUMBER = 20
MODULE_VERSION = "1.0.0"

SUPPORTED_EXTENSIONS = ("csv", "json")


def _sha256(raw: bytes) -> str:
    """Return a SHA-256 fingerprint for the uploaded file."""
    return hashlib.sha256(raw).hexdigest()


def _strict_json_bytes(data: Any) -> bytes:
    """Serialize data as strict UTF-8 JSON."""

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

        if isinstance(value, (pd.Timestamp, datetime)):
            if pd.isna(value):
                return None
            return value.isoformat()

        if isinstance(value, np.generic):
            return normalize(value.item())

        if pd.isna(value):
            return None

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


def _load_snapshot(filename: str, raw: bytes) -> pd.DataFrame:
    """Load a CSV or JSON snapshot into a DataFrame."""
    suffix = Path(filename).suffix.lower()

    try:
        if suffix == ".csv":
            frame = pd.read_csv(
                __import__("io").BytesIO(raw)
            )

        elif suffix == ".json":
            parsed = json.loads(raw.decode("utf-8"))

            if isinstance(parsed, list):
                frame = pd.DataFrame(parsed)

            elif isinstance(parsed, dict):
                # Support either a direct row-oriented object or
                # a wrapper containing a "data" list.
                if isinstance(parsed.get("data"), list):
                    frame = pd.DataFrame(parsed["data"])
                else:
                    frame = pd.DataFrame([parsed])

            else:
                raise ValueError(
                    "JSON must contain an object or a list of records."
                )

        else:
            raise ValueError(
                "Unsupported file type. Upload CSV or JSON."
            )

    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "The uploaded file is not valid UTF-8 CSV/JSON data."
        ) from exc

    except pd.errors.EmptyDataError as exc:
        raise ValueError("The uploaded CSV is empty.") from exc

    except pd.errors.ParserError as exc:
        raise ValueError("The CSV structure could not be parsed.") from exc

    if frame.empty or len(frame.columns) == 0:
        raise ValueError(
            "The uploaded snapshot has no usable rows or columns."
        )

    # Normalize column names for display while avoiding silent
    # replacement of distinct column names.
    frame.columns = [str(column) for column in frame.columns]

    if frame.columns.duplicated().any():
        raise ValueError(
            "Duplicate column names were found. Rename them before analysis."
        )

    return frame


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns containing at least one finite numeric value."""
    result: list[str] = []

    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)

        if values.notna().any():
            result.append(str(column))

    return result


def _quality_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize data quality without modifying the uploaded records."""
    total_cells = int(frame.shape[0] * frame.shape[1])
    missing_cells = int(frame.isna().sum().sum())

    nonfinite_by_column: dict[str, int] = {}

    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            values = pd.to_numeric(frame[column], errors="coerce")
            nonfinite_by_column[str(column)] = int(
                np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).sum()
            )

    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "numeric_columns": len(_numeric_columns(frame)),
        "missing_cells": missing_cells,
        "missing_cell_fraction": (
            float(missing_cells / total_cells)
            if total_cells
            else 0.0
        ),
        "duplicate_rows": int(frame.duplicated().sum()),
        "infinite_values_by_numeric_column": nonfinite_by_column,
    }


def _parse_timestamps(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Parse a selected timestamp column as UTC timestamps."""
    return pd.to_datetime(
        frame[column],
        errors="coerce",
        utc=True,
    )


def _freshness_summary(
    timestamps: pd.Series,
    stale_after_minutes: int,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """
    Calculate timestamp coverage and freshness.

    Freshness is only a timestamp-based monitoring indicator. It does not
    verify that a data source is connected or that observations are genuine.
    """
    now = now_utc or datetime.now(timezone.utc)

    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    now = now.astimezone(timezone.utc)

    valid = timestamps.dropna()

    result: dict[str, Any] = {
        "valid_timestamp_count": int(valid.size),
        "invalid_timestamp_count": int(timestamps.isna().sum()),
        "stale_after_minutes": int(stale_after_minutes),
        "latest_timestamp_utc": None,
        "age_minutes": None,
        "status": "UNVERIFIED",
        "future_timestamp_count": 0,
    }

    if valid.empty:
        result["status"] = "NO_VALID_TIMESTAMPS"
        return result

    latest = valid.max()
    age_minutes = (pd.Timestamp(now) - latest).total_seconds() / 60.0

    result["latest_timestamp_utc"] = latest.isoformat()
    result["age_minutes"] = float(age_minutes)
    result["future_timestamp_count"] = int(
        (valid > pd.Timestamp(now)).sum()
    )

    if result["future_timestamp_count"] > 0:
        result["status"] = "CHECK_FUTURE_TIMESTAMPS"

    elif age_minutes < 0:
        result["status"] = "CHECK_CLOCK_OR_TIMESTAMP"

    elif age_minutes > stale_after_minutes:
        result["status"] = "STALE"

    else:
        result["status"] = "RECENT_TIMESTAMP"

    return result


def _signal_summary(
    frame: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    """Return descriptive statistics for one numeric signal."""
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()

    if values.empty:
        raise ValueError(
            f"No finite numeric observations were found in '{column}'."
        )

    return {
        "column": str(column),
        "valid_count": int(values.size),
        "excluded_missing_or_nonfinite": int(len(frame) - values.size),
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


def module_info() -> dict[str, Any]:
    """Return public metadata for Module 20."""
    return {
        "module": MODULE_NAME,
        "module_number": MODULE_NUMBER,
        "version": MODULE_VERSION,
        "supported_inputs": ["CSV", "JSON"],
        "automatic_network_polling": False,
        "live_source_connection_verified": False,
        "synthetic_observations_generated": False,
        "automatic_anomaly_confirmation": False,
        "clinical_diagnosis": False,
        "protected_core_included": False,
    }


def run_contract_tests() -> dict[str, bool]:
    """Run deterministic checks without scientific observations."""
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    recent = pd.Series(
        pd.to_datetime(
            ["2026-01-01T11:55:00Z", "2026-01-01T11:58:00Z"],
            utc=True,
        )
    )

    old = pd.Series(
        pd.to_datetime(
            ["2026-01-01T10:00:00Z"],
            utc=True,
        )
    )

    normalized = json.loads(
        _strict_json_bytes(
            {
                "valid": 1.5,
                "nonfinite": float("nan"),
            }
        ).decode("utf-8")
    )

    checks = {
        "module_number_is_20": MODULE_NUMBER == 20,
        "module_version_is_defined": MODULE_VERSION == "1.0.0",
        "render_page_is_callable": callable(render_page),
        "sha256_is_64_hex_chars": len(_sha256(b"contract-test")) == 64,
        "strict_json_normalizes_nonfinite": (
            normalized["nonfinite"] is None
        ),
        "recent_timestamp_detected": (
            _freshness_summary(recent, 10, now)["status"]
            == "RECENT_TIMESTAMP"
        ),
        "stale_timestamp_detected": (
            _freshness_summary(old, 30, now)["status"] == "STALE"
        ),
        "empty_timestamp_detected": (
            _freshness_summary(
                pd.Series([], dtype="datetime64[ns, UTC]"),
                30,
                now,
            )["status"]
            == "NO_VALID_TIMESTAMPS"
        ),
        "automatic_polling_disabled": (
            module_info()["automatic_network_polling"] is False
        ),
        "automatic_anomaly_confirmation_disabled": (
            module_info()["automatic_anomaly_confirmation"] is False
        ),
    }

    checks["all_checks_passed"] = all(checks.values())

    return checks


def render_page() -> None:
    """Render the Live Data Status and Signal Monitoring page."""
    
    st.title(f"📡 Module {MODULE_NUMBER}: Live Data Monitoring")

    st.caption(
        "Snapshot-based monitoring of timestamps, data quality, and numeric "
        "signals. Uploading a file does not establish a real-time connection."
    )

    with st.expander(
        "Scope, freshness, and scientific limitations",
        expanded=False,
    ):
        st.markdown(
            """
            - Upload an actual observational CSV or JSON snapshot.
            - Timestamp freshness is not proof of a live network connection.
            - Incorrect clocks, timezone errors, and future timestamps require review.
            - Missing values and duplicate rows are reported, not automatically removed.
            - Signal summaries are descriptive and do not confirm anomalies.
            - No synthetic scientific observations are generated.
            """
        )

    uploaded_file = st.file_uploader(
        "Upload an observational snapshot (CSV or JSON)",
        type=list(SUPPORTED_EXTENSIONS),
        accept_multiple_files=False,
        key="live_snapshot_upload",
        help=(
            "Use an actual dataset you are authorized to analyze. "
            "The page does not automatically connect to a remote data source."
        ),
    )

    if uploaded_file is None:
        st.info(
            "Upload a real CSV or JSON snapshot to begin. No example "
            "observations are generated."
        )

        with st.expander("Module information and contract tests"):
            st.json(module_info())

            checks = run_contract_tests()
            st.json(checks)

            if checks["all_checks_passed"]:
                st.success("Contract tests passed.")
            else:
                st.error("One or more contract tests failed.")

        return

    raw = uploaded_file.getvalue()

    st.subheader("Snapshot provenance")

    c1, c2 = st.columns(2)
    c1.metric("File size", f"{len(raw):,} bytes")
    c2.metric("SHA-256", _sha256(raw)[:16] + "…")

    st.caption(
        "The uploaded filename and complete SHA-256 fingerprint are recorded "
        "in the downloadable report."
    )

    try:
        frame = _load_snapshot(uploaded_file.name, raw)

    except (ValueError, TypeError) as exc:
        st.error(f"Could not load snapshot: {exc}")
        return

    except Exception as exc:
        st.error(
            "Unexpected file-loading error. "
            f"Error type: {type(exc).__name__}."
        )
        return

    st.subheader("Snapshot preview")
    st.dataframe(frame.head(20), use_container_width=True)

    quality = _quality_summary(frame)

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

    if quality["missing_cells"]:
        st.warning(
            "Missing cells were detected. Review their distribution before "
            "using the dataset in further analysis."
        )

    if quality["duplicate_rows"]:
        st.warning(
            "Duplicate rows were detected. No rows were removed automatically."
        )

    numeric_columns = _numeric_columns(frame)

    st.divider()
    st.subheader("Timestamp and freshness monitoring")

    timestamp_column = st.selectbox(
        "Timestamp column",
        options=["(Not available / not selected)"] + list(frame.columns),
        key="live_timestamp_column",
    )

    stale_after_minutes = st.number_input(
        "Stale threshold (minutes)",
        min_value=1,
        max_value=525600,
        value=60,
        step=5,
        help=(
            "A monitoring threshold chosen for this review. "
            "It is not a scientific anomaly threshold."
        ),
    )

    freshness: dict[str, Any] = {
        "status": "UNVERIFIED",
        "valid_timestamp_count": 0,
        "invalid_timestamp_count": 0,
        "latest_timestamp_utc": None,
        "age_minutes": None,
        "stale_after_minutes": int(stale_after_minutes),
        "future_timestamp_count": 0,
    }

    if timestamp_column != "(Not available / not selected)":
        parsed_timestamps = _parse_timestamps(frame, timestamp_column)

        freshness = _freshness_summary(
            parsed_timestamps,
            int(stale_after_minutes),
        )

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Timestamp status",
            freshness["status"],
        )
        c2.metric(
            "Valid timestamps",
            freshness["valid_timestamp_count"],
        )
        c3.metric(
            "Invalid timestamps",
            freshness["invalid_timestamp_count"],
        )

        st.write(
            "Latest timestamp (UTC):",
            freshness["latest_timestamp_utc"] or "Unavailable",
        )

        age = freshness["age_minutes"]

        st.write(
            "Age of latest timestamp (minutes):",
            f"{age:.2f}" if age is not None else "Unavailable",
        )

        if freshness["status"] == "STALE":
            st.warning(
                "The latest valid timestamp exceeds the selected freshness "
                "threshold. This indicates stale timestamp data, not a confirmed anomaly."
            )

        elif freshness["status"] == "RECENT_TIMESTAMP":
            st.success(
                "The latest valid timestamp is within the selected threshold. "
                "This does not verify a live source connection."
            )

        elif freshness["status"] == "CHECK_FUTURE_TIMESTAMPS":
            st.warning(
                "Some timestamps are later than the current UTC time. "
                "Check timezone handling, clock settings, and source metadata."
            )

        elif freshness["status"] == "CHECK_CLOCK_OR_TIMESTAMP":
            st.warning(
                "The latest timestamp appears to be in the future. "
                "Review the timestamp and system clock."
            )

        else:
            st.warning(
                "Freshness could not be established from the selected timestamps."
            )

        st.caption(
            "Timestamps are interpreted as UTC by pandas. If the source contains "
            "timezone-naive local times, confirm and correct the timezone before "
            "interpreting freshness."
        )

    else:
        st.info(
            "No timestamp column selected. Data freshness remains unverified."
        )

    st.divider()
    st.subheader("Exploratory signal monitoring")

    selected_signal = None
    signal_result = None

    if numeric_columns:
        selected_signal = st.selectbox(
            "Numeric signal",
            options=numeric_columns,
            key="live_signal_column",
        )

        if st.button("Inspect selected signal", type="primary"):
            try:
                signal_result = _signal_summary(
                    frame,
                    selected_signal,
                )

                st.session_state["live_signal_result"] = signal_result

            except ValueError as exc:
                st.error(str(exc))

        previous_signal = st.session_state.get("live_signal_result")

        if (
            previous_signal
            and previous_signal.get("column") == selected_signal
        ):
            st.json(previous_signal)

    else:
        st.info(
            "No usable numeric columns were found. Timestamp and quality "
            "inspection remain available."
        )

    st.divider()
    st.subheader("Monitoring report export")

    report = {
        "project": "D³ VITAL-X Space Intelligence Platform",
        "page_module": MODULE_NAME,
        "page_module_number": MODULE_NUMBER,
        "page_module_version": MODULE_VERSION,
        "snapshot": {
            "uploaded_filename": Path(uploaded_file.name).name,
            "file_size_bytes": len(raw),
            "sha256": _sha256(raw),
            "row_count": int(frame.shape[0]),
            "column_count": int(frame.shape[1]),
        },
        "data_quality": quality,
        "freshness": freshness,
        "selected_signal": (
            st.session_state.get("live_signal_result")
            if selected_signal is not None
            and st.session_state.get("live_signal_result", {}).get("column")
            == selected_signal
            else None
        ),
        "source_connection": {
            "automatic_polling_enabled": False,
            "remote_connection_verified": False,
            "data_source_independently_verified": False,
        },
        "scientific_boundary": (
            "Snapshot-based exploratory monitoring only. Timestamp freshness "
            "does not establish a live source connection. No anomaly or physical "
            "event is automatically confirmed."
        ),
    }

    try:
        report_bytes = _strict_json_bytes(report)

        st.download_button(
            label="Download monitoring report (JSON)",
            data=report_bytes,
            file_name="d3_vital_x_live_monitoring_report.json",
            mime="application/json",
            key="live_report_download",
        )

    except (TypeError, ValueError) as exc:
        st.error(
            f"Could not prepare the monitoring report: {type(exc).__name__}."
        )

    with st.expander("Module information and contract tests"):
        st.json(module_info())

        checks = run_contract_tests()
        st.json(checks)

        if checks["all_checks_passed"]:
            st.success("Contract tests passed.")
        else:
            st.error("One or more contract tests failed.")


if __name__ == "__main__":
    render_page()
