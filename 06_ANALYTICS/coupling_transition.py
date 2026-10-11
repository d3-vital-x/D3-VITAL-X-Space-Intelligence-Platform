
"""
D³ VITAL-X — Coupling and Transition Diagnostics
Module 14: coupling_transition.py

Purpose
-------
Descriptive analysis of the relationship between two aligned numerical
signals, including window-wise correlation and changes in correlation.

Scientific boundary
-------------------
This module does not establish causation, a physical transition, a medical
condition, or a validated anomaly. Results depend on the supplied data,
window configuration, preprocessing, and sampling process.

Input data must come from the user's actual dataset. Contract checks do not
generate synthetic scientific observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional
import hashlib
import json
import math

import numpy as np


MODULE_NAME = "coupling_transition"
MODULE_NUMBER = 14
MODULE_VERSION = "1.0.0"

SUPPORTED_MISSING_POLICIES = {"reject", "omit"}


@dataclass
class WindowDiagnostic:
    """Descriptive statistics for one window of paired observations."""

    window_number: int
    start_index: int
    stop_index_exclusive: int
    sample_count: int
    correlation: Optional[float]
    covariance: Optional[float]
    correlation_change: Optional[float]
    status: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class CouplingTransitionResult:
    """Serializable result of a coupling/transition diagnostic run."""

    module: str
    module_version: str
    status: str
    original_sample_count: int
    valid_pair_count: int
    omitted_pair_count: int
    overall_correlation: Optional[float]
    overall_covariance: Optional[float]
    window_size: int
    step: int
    missing_policy: str
    windows: list[WindowDiagnostic]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    interpretation: str = (
        "Descriptive signal relationship only; not evidence of causation "
        "or a confirmed physical, clinical, or anomalous transition."
    )
    result_fingerprint: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible output."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize the result as JSON."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )


def _as_numeric_1d(values: Any, name: str) -> np.ndarray:
    """Validate and convert one input to a one-dimensional float array."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must contain numeric values."
        ) from exc

    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")

    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")

    return array


def _safe_float(value: Any) -> Optional[float]:
    """Convert finite numerical output to a JSON-safe Python float."""
    result = float(value)
    return result if math.isfinite(result) else None


def _correlation_and_covariance(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], list[str]]:
    """Compute Pearson correlation and sample covariance when defined."""
    warnings: list[str] = []

    if x.size < 2:
        return None, None, [
            "At least two paired observations are required."
        ]

    if np.ptp(x) == 0 or np.ptp(y) == 0:
        warnings.append(
            "Correlation is undefined because at least one signal is constant."
        )
        correlation = None
    else:
        correlation = _safe_float(np.corrcoef(x, y)[0, 1])

    covariance = _safe_float(np.cov(x, y, ddof=1)[0, 1])

    if covariance is None:
        warnings.append("Covariance could not be represented as a finite value.")

    return correlation, covariance, warnings


def analyze_coupling_transition(
    signal_x: Any,
    signal_y: Any,
    *,
    window_size: int = 100,
    step: int = 50,
    min_samples: int = 10,
    missing_policy: str = "reject",
) -> CouplingTransitionResult:
    """
    Analyze the descriptive relationship between two aligned signals.

    Parameters
    ----------
    signal_x, signal_y:
        One-dimensional numeric arrays from real observations. Values at
        matching positions must refer to corresponding observations.

    window_size:
        Number of valid paired observations per window.

    step:
        Number of valid paired observations between window starts.

    min_samples:
        Minimum observations required to report a window correlation.

    missing_policy:
        "reject" rejects NaN/Inf values.
        "omit" removes pairs for which either signal is non-finite.

    Notes
    -----
    Window indices refer to the retained-pair sequence after any omission.
    A change in correlation is a descriptive change, not a significance test.
    """

    errors: list[str] = []
    warnings: list[str] = []

    # Validate configuration before processing.
    for name, value in (
        ("window_size", window_size),
        ("step", step),
        ("min_samples", min_samples),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer.")

    if window_size < 2:
        raise ValueError("window_size must be at least 2.")

    if step < 1:
        raise ValueError("step must be at least 1.")

    if min_samples < 2:
        raise ValueError("min_samples must be at least 2.")

    if min_samples > window_size:
        raise ValueError("min_samples cannot exceed window_size.")

    if missing_policy not in SUPPORTED_MISSING_POLICIES:
        raise ValueError(
            "missing_policy must be either 'reject' or 'omit'."
        )

    # Validate aligned input signals.
    x = _as_numeric_1d(signal_x, "signal_x")
    y = _as_numeric_1d(signal_y, "signal_y")

    if x.size != y.size:
        raise ValueError(
            "signal_x and signal_y must have the same number of observations."
        )

    original_count = int(x.size)
    finite_pairs = np.isfinite(x) & np.isfinite(y)
    omitted_count = int(np.count_nonzero(~finite_pairs))

    if omitted_count and missing_policy == "reject":
        raise ValueError(
            f"Found {omitted_count} non-finite paired observation(s). "
            "Use missing_policy='omit' only when pairwise omission is justified."
        )

    if omitted_count:
        x = x[finite_pairs]
        y = y[finite_pairs]
        warnings.append(
            f"Omitted {omitted_count} pair(s) containing NaN or infinity."
        )

    valid_count = int(x.size)

    if valid_count < 2:
        raise ValueError(
            "At least two finite paired observations are required."
        )

    if valid_count < min_samples:
        warnings.append(
            "The complete dataset has fewer observations than min_samples; "
            "window-level correlations may be unavailable."
        )

    overall_corr, overall_cov, overall_warnings = (
        _correlation_and_covariance(x, y)
    )
    warnings.extend(overall_warnings)

    windows: list[WindowDiagnostic] = []
    previous_correlation: Optional[float] = None
    start = 0
    window_number = 1

    while start < valid_count:
        stop = min(start + window_size, valid_count)
        wx = x[start:stop]
        wy = y[start:stop]
        count = int(wx.size)
        window_warnings: list[str] = []

        # Do not report a correlation for a window below the configured
        # minimum, even if a numerical value could technically be computed.
        if count < min_samples:
            corr = None
            cov = None
            delta = None
            status = "insufficient_samples"
            window_warnings.append(
                f"Window has {count} observations; "
                f"{min_samples} are required."
            )
        else:
            corr, cov, stats_warnings = _correlation_and_covariance(wx, wy)
            window_warnings.extend(stats_warnings)

            if corr is None:
                delta = None
                status = "correlation_undefined"
            else:
                delta = (
                    _safe_float(corr - previous_correlation)
                    if previous_correlation is not None
                    else None
                )
                status = "descriptive_only"

                # Only compare against the previous window with a defined
                # correlation.
                previous_correlation = corr

        windows.append(
            WindowDiagnostic(
                window_number=window_number,
                start_index=start,
                stop_index_exclusive=stop,
                sample_count=count,
                correlation=corr,
                covariance=cov,
                correlation_change=delta,
                status=status,
                warnings=window_warnings,
            )
        )

        window_number += 1

        if stop == valid_count:
            break

        start += step

    if not windows:
        errors.append("No windows could be evaluated.")

    if valid_count < window_size:
        warnings.append(
            "The dataset is shorter than window_size; the available observations "
            "were evaluated as one partial window."
        )

    # Fingerprint the valid input and analysis configuration for reproducibility.
    digest = hashlib.sha256()
    digest.update(np.asarray(x, dtype="<f8").tobytes())
    digest.update(np.asarray(y, dtype="<f8").tobytes())
    config = {
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "window_size": int(window_size),
        "step": int(step),
        "min_samples": int(min_samples),
        "missing_policy": missing_policy,
    }
    digest.update(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )

    result = CouplingTransitionResult(
        module=MODULE_NAME,
        module_version=MODULE_VERSION,
        status="completed_with_warnings" if warnings else "completed",
        original_sample_count=original_count,
        valid_pair_count=valid_count,
        omitted_pair_count=omitted_count,
        overall_correlation=overall_corr,
        overall_covariance=overall_cov,
        window_size=int(window_size),
        step=int(step),
        missing_policy=missing_policy,
        windows=windows,
        warnings=warnings,
        errors=errors,
        result_fingerprint=digest.hexdigest(),
    )

    return result


def module_info() -> dict[str, Any]:
    """Expose public module metadata without revealing private algorithms."""
    return {
        "module": MODULE_NAME,
        "module_number": MODULE_NUMBER,
        "version": MODULE_VERSION,
        "purpose": "Descriptive coupling and window-wise transition diagnostics",
        "input": "Two aligned one-dimensional numeric signals",
        "outputs": [
            "overall_correlation",
            "overall_covariance",
            "window_correlation",
            "correlation_change",
            "warnings",
            "result_fingerprint",
        ],
        "scientific_boundary": (
            "No causal inference, confirmed-transition claim, or diagnosis."
        ),
    }


def run_contract_tests() -> dict[str, bool]:
    """
    Run input-contract checks without generating scientific observations.

    The tests below verify rejection of invalid input rather than generating
    synthetic signals to demonstrate a scientific result.
    """
    checks: dict[str, bool] = {}

    try:
        analyze_coupling_transition([], [])
        checks["empty_input_rejected"] = False
    except ValueError:
        checks["empty_input_rejected"] = True

    try:
        analyze_coupling_transition([1.0, 2.0], [1.0])
        checks["length_mismatch_rejected"] = False
    except ValueError:
        checks["length_mismatch_rejected"] = True

    try:
        analyze_coupling_transition(
            [1.0, 2.0],
            [1.0, 2.0],
            window_size=1,
        )
        checks["invalid_window_rejected"] = False
    except ValueError:
        checks["invalid_window_rejected"] = True

    try:
        analyze_coupling_transition(
            [1.0, float("nan")],
            [1.0, 2.0],
            missing_policy="reject",
        )
        checks["nonfinite_rejected_by_default"] = False
    except ValueError:
        checks["nonfinite_rejected_by_default"] = True

    checks["all_checks_passed"] = all(checks.values())
    return checks


if __name__ == "__main__":
    print(json.dumps(module_info(), indent=2, ensure_ascii=False))
    print(json.dumps(run_contract_tests(), indent=2, ensure_ascii=False))
