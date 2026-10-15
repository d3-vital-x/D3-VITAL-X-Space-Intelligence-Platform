
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
D³ VITAL-X Space Intelligence Platform
Module 27: time_reversal_test.py

Version: 1.0.0
Report schema: 1.0

Purpose
-------
Compare selected descriptive statistics of a numeric signal with the
same signal in reversed sample order.

This is a diagnostic validation utility, not a scientific proof engine.
It does not establish causality, physical validity, anomaly authenticity,
or medical diagnostic conclusions.

Design principles
-----------------
- Standard-library only.
- Read-only input handling.
- Deterministic calculations.
- JSON and CSV report outputs.
- Explicit scientific and medical boundaries.
- Built-in self-checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


MODULE_NAME = "time_reversal_test"
MODULE_VERSION = "1.0.0"
REPORT_SCHEMA_VERSION = "1.0"

DEFAULT_ENTROPY_BINS = 16

METRIC_NAMES = (
    "sample_count",
    "mean",
    "variance",
    "standard_deviation",
    "shannon_entropy_bits",
    "lag1_autocorrelation",
    "linear_trend_slope",
)


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(file_path: Path) -> str:
    """Calculate SHA-256 without modifying the source file."""
    digest = hashlib.sha256()

    with file_path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def validate_signal(
    values: Sequence[float],
    min_samples: int = 2,
) -> List[float]:
    """Validate and return a finite numeric copy of the signal."""
    if isinstance(values, (str, bytes)):
        raise TypeError("Signal must be a numeric sequence.")

    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Signal contains a non-numeric value.") from exc

    if len(result) < min_samples:
        raise ValueError(
            f"At least {min_samples} valid samples are required."
        )

    if not all(math.isfinite(value) for value in result):
        raise ValueError("Signal contains NaN or infinite values.")

    return result


def validate_positive_integer(value: int, name: str) -> int:
    """Validate a positive integer; reject booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def finite_or_none(value: Optional[float]) -> Optional[float]:
    """Convert a finite numeric value to float, otherwise return None."""
    if value is None:
        return None

    value = float(value)
    return value if math.isfinite(value) else None


# ---------------------------------------------------------------------
# Descriptive metrics
# ---------------------------------------------------------------------

def calculate_shannon_entropy(
    values: Sequence[float],
    bins: int = DEFAULT_ENTROPY_BINS,
) -> float:
    """
    Estimate Shannon entropy in bits using a fixed-width histogram.

    This is a histogram-based descriptive estimate, not differential
    entropy and not a universal measure of physical disorder.
    """
    signal = validate_signal(values, min_samples=1)
    validate_positive_integer(bins, "bins")

    minimum = min(signal)
    maximum = max(signal)

    if minimum == maximum:
        return 0.0

    width = (maximum - minimum) / bins
    counts = [0] * bins

    for value in signal:
        index = int((value - minimum) / width)

        # Include the maximum value in the final bin.
        if index >= bins:
            index = bins - 1
        elif index < 0:
            index = 0

        counts[index] += 1

    total = len(signal)
    entropy = 0.0

    for count in counts:
        if count == 0:
            continue

        probability = count / total
        entropy -= probability * math.log2(probability)

    return float(entropy)


def calculate_lag1_autocorrelation(
    values: Sequence[float],
) -> Optional[float]:
    """
    Calculate Pearson correlation between adjacent samples.

    Returns None when either adjacent series has zero variance.
    """
    signal = validate_signal(values, min_samples=3)

    left = signal[:-1]
    right = signal[1:]

    mean_left = statistics.mean(left)
    mean_right = statistics.mean(right)

    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]

    numerator = sum(
        x_value * y_value
        for x_value, y_value in zip(centered_left, centered_right)
    )

    sum_left_sq = sum(value * value for value in centered_left)
    sum_right_sq = sum(value * value for value in centered_right)

    denominator = math.sqrt(sum_left_sq * sum_right_sq)

    if denominator == 0.0:
        return None

    return finite_or_none(numerator / denominator)


def calculate_linear_trend_slope(
    values: Sequence[float],
) -> float:
    """
    Calculate ordinary least-squares slope against sample index.

    The x-axis is 0, 1, ..., n-1. This is not a physical-time slope
    unless the sample spacing and units are known.
    """
    signal = validate_signal(values, min_samples=2)

    n_samples = len(signal)
    mean_y = statistics.mean(signal)
    mean_x = (n_samples - 1) / 2.0

    numerator = sum(
        (index - mean_x) * (value - mean_y)
        for index, value in enumerate(signal)
    )

    denominator = sum(
        (index - mean_x) ** 2
        for index in range(n_samples)
    )

    if denominator == 0.0:
        raise ValueError("Cannot calculate trend slope for this signal.")

    return float(numerator / denominator)


def calculate_metrics(
    values: Sequence[float],
    entropy_bins: int = DEFAULT_ENTROPY_BINS,
) -> Dict[str, Optional[float]]:
    """Calculate the selected descriptive metrics."""
    signal = validate_signal(values, min_samples=2)

    variance = (
        float(statistics.variance(signal))
        if len(signal) >= 2
        else None
    )

    standard_deviation = (
        math.sqrt(variance)
        if variance is not None
        else None
    )

    autocorrelation = (
        calculate_lag1_autocorrelation(signal)
        if len(signal) >= 3
        else None
    )

    return {
        "sample_count": len(signal),
        "mean": float(statistics.mean(signal)),
        "variance": finite_or_none(variance),
        "standard_deviation": finite_or_none(standard_deviation),
        "shannon_entropy_bits": calculate_shannon_entropy(
            signal, bins=entropy_bins
        ),
        "lag1_autocorrelation": finite_or_none(autocorrelation),
        "linear_trend_slope": calculate_linear_trend_slope(signal),
    }


# ---------------------------------------------------------------------
# Time-reversal comparison
# ---------------------------------------------------------------------

def compare_metrics(
    original_metrics: Dict[str, Optional[float]],
    reversed_metrics: Dict[str, Optional[float]],
) -> Dict[str, Dict[str, Any]]:
    """
    Compare original and reversed metric values.

    Relative difference is omitted when the original value is zero
    or either value is undefined.
    """
    comparison: Dict[str, Dict[str, Any]] = {}

    for metric_name in METRIC_NAMES:
        original_value = original_metrics.get(metric_name)
        reversed_value = reversed_metrics.get(metric_name)

        if original_value is None or reversed_value is None:
            comparison[metric_name] = {
                "original": original_value,
                "reversed": reversed_value,
                "absolute_difference": None,
                "relative_difference": None,
                "comparison_status": "undefined",
            }
            continue

        original_number = float(original_value)
        reversed_number = float(reversed_value)

        absolute_difference = abs(reversed_number - original_number)

        if original_number != 0.0:
            relative_difference = (
                absolute_difference / abs(original_number)
            )
        else:
            relative_difference = None

        comparison[metric_name] = {
            "original": original_number,
            "reversed": reversed_number,
            "absolute_difference": finite_or_none(absolute_difference),
            "relative_difference": finite_or_none(relative_difference),
            "comparison_status": "computed",
        }

    return comparison


def run_time_reversal_test(
    values: Sequence[float],
    entropy_bins: int = DEFAULT_ENTROPY_BINS,
) -> Dict[str, Any]:
    """
    Reverse the sample order and compare descriptive metrics.

    No input array or source file is modified.
    """
    signal = validate_signal(values, min_samples=2)
    validate_positive_integer(entropy_bins, "entropy_bins")

    reversed_signal = list(reversed(signal))

    original_metrics = calculate_metrics(
        signal,
        entropy_bins=entropy_bins,
    )
    reversed_metrics = calculate_metrics(
        reversed_signal,
        entropy_bins=entropy_bins,
    )

    comparison = compare_metrics(
        original_metrics,
        reversed_metrics,
    )

    return {
        "test_name": "time_reversal_comparison",
        "sample_count": len(signal),
        "entropy_bins": entropy_bins,
        "original_metrics": original_metrics,
        "reversed_metrics": reversed_metrics,
        "metric_comparison": comparison,
        "interpretation_notes": [
            (
                "Mean, variance, standard deviation, and histogram entropy "
                "should be invariant under sample-order reversal when "
                "calculated with the same values and binning rule."
            ),
            (
                "Lag-1 Pearson autocorrelation is expected to remain "
                "unchanged under reversal for a finite sequence, subject "
                "to numerical precision and undefined cases."
            ),
            (
                "The linear trend slope against sample index is expected "
                "to change sign under reversal, subject to numerical "
                "precision."
            ),
            (
                "Differences can indicate implementation issues, numerical "
                "effects, or a mismatch in processing; they do not alone "
                "establish physical irreversibility or causality."
            ),
        ],
        "scientific_boundary": {
            "scientific_validity_established": False,
            "causality_established": False,
            "physical_irreversibility_established": False,
            "anomaly_confirmed": False,
            "medical_diagnosis": False,
            "human_review_required": True,
        },
    }


# ---------------------------------------------------------------------
# CSV input
# ---------------------------------------------------------------------

def load_numeric_csv(
    csv_path: Path,
    column: str,
) -> Tuple[List[float], Dict[str, Any]]:
    """
    Load one numeric CSV column.

    Blank, non-numeric, and non-finite cells are skipped and counted.
    The original CSV file is not modified.
    """
    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    if not column or not column.strip():
        raise ValueError("A non-empty CSV column name is required.")

    input_hash = sha256_file(csv_path)
    values: List[float] = []

    rows_seen = 0
    blank_cells = 0
    non_numeric_cells = 0
    non_finite_cells = 0

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file_obj:
        reader = csv.DictReader(file_obj)

        if not reader.fieldnames or column not in reader.fieldnames:
            available = reader.fieldnames or []
            raise ValueError(
                f"Column '{column}' not found. "
                f"Available columns: {available}"
            )

        for row in reader:
            rows_seen += 1
            raw_value = row.get(column)

            if raw_value is None or not raw_value.strip():
                blank_cells += 1
                continue

            try:
                value = float(raw_value)
            except (TypeError, ValueError, OverflowError):
                non_numeric_cells += 1
                continue

            if not math.isfinite(value):
                non_finite_cells += 1
                continue

            values.append(value)

    if len(values) < 2:
        raise ValueError(
            "The selected CSV column must contain at least "
            "two valid finite numeric samples."
        )

    metadata = {
        "input_file_name": csv_path.name,
        "input_file_sha256": input_hash,
        "selected_column": column,
        "rows_seen": rows_seen,
        "valid_numeric_samples": len(values),
        "blank_cells_skipped": blank_cells,
        "non_numeric_cells_skipped": non_numeric_cells,
        "non_finite_cells_skipped": non_finite_cells,
        "source_modified": False,
    }

    return values, metadata


# ---------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------

def build_report(
    result: Dict[str, Any],
    input_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a reproducibility-oriented report."""
    return {
        "module": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "created_utc": utc_now(),
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "input": input_metadata or {
            "input_type": "synthetic_self_check_or_direct_sequence",
            "source_modified": False,
        },
        "result": result,
        "report_boundary": {
            "purpose": "descriptive time-reversal diagnostic",
            "scientific_claim": False,
            "medical_diagnosis": False,
            "causal_inference": False,
            "requires_independent_review": True,
        },
    }


def save_report(
    report: Dict[str, Any],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Save JSON report and a flattened CSV metric comparison."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "time_reversal_report.json"
    csv_path = output_dir / "time_reversal_comparison.csv"

    with json_path.open("w", encoding="utf-8") as file_obj:
        json.dump(
            report,
            file_obj,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        file_obj.write("\n")

    result = report["result"]
    comparisons = result["metric_comparison"]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file_obj:
        fieldnames = [
            "metric",
            "original",
            "reversed",
            "absolute_difference",
            "relative_difference",
            "comparison_status",
        ]

        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()

        for metric_name, values in comparisons.items():
            writer.writerow({
                "metric": metric_name,
                "original": values["original"],
                "reversed": values["reversed"],
                "absolute_difference": values["absolute_difference"],
                "relative_difference": values["relative_difference"],
                "comparison_status": values["comparison_status"],
            })

    return json_path, csv_path


def print_summary(report: Dict[str, Any]) -> None:
    """Print a concise terminal summary."""
    result = report["result"]

    print("=" * 68)
    print("D³ VITAL-X — TIME REVERSAL TEST")
    print("=" * 68)
    print(f"Module version : {report['module_version']}")
    print(f"Created UTC    : {report['created_utc']}")
    print(f"Samples        : {result['sample_count']}")
    print(f"Entropy bins   : {result['entropy_bins']}")
    print("-" * 68)
    print(
        f"{'Metric':30s} {'Original':>12s} "
        f"{'Reversed':>12s} {'Abs. diff.':>12s}"
    )
    print("-" * 68)

    for metric_name, comparison in result["metric_comparison"].items():
        original = comparison["original"]
        reversed_value = comparison["reversed"]
        difference = comparison["absolute_difference"]

        original_text = (
            "undefined" if original is None else f"{original:.6g}"
        )
        reversed_text = (
            "undefined"
            if reversed_value is None
            else f"{reversed_value:.6g}"
        )
        difference_text = (
            "undefined" if difference is None else f"{difference:.6g}"
        )

        print(
            f"{metric_name:30s} {original_text:>12s} "
            f"{reversed_text:>12s} {difference_text:>12s}"
        )

    print("-" * 68)
    print("Scientific validity established: NO")
    print("Causality established           : NO")
    print("Medical diagnosis               : NO")
    print("Interpretation                  : Diagnostic only")
    print("=" * 68)


# ---------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------

def self_check() -> Dict[str, Any]:
    """Run deterministic checks on synthetic signals."""
    checks: List[Dict[str, Any]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({
            "name": name,
            "passed": bool(condition),
            "detail": detail,
        })
        if not condition:
            raise AssertionError(f"Self-check failed: {name} — {detail}")

    signal = [1.0, 2.0, 4.0, 3.0, 7.0, 6.0, 9.0]
    original_copy = list(signal)

    result = run_time_reversal_test(signal, entropy_bins=4)
    reversed_signal = list(reversed(signal))

    original_metrics = result["original_metrics"]
    reversed_metrics = result["reversed_metrics"]
    comparison = result["metric_comparison"]

    check(
        "input_not_modified",
        signal == original_copy,
        "Original in-memory input sequence remains unchanged.",
    )

    check(
        "reversed_order_correct",
        reversed_signal == [9.0, 6.0, 7.0, 3.0, 4.0, 2.0, 1.0],
        "Reversed sequence has the expected sample order.",
    )

    for metric_name in (
        "mean",
        "variance",
        "standard_deviation",
        "shannon_entropy_bits",
        "lag1_autocorrelation",
    ):
        first = original_metrics[metric_name]
        second = reversed_metrics[metric_name]

        if first is None or second is None:
            same = first is None and second is None
        else:
            same = math.isclose(
                float(first),
                float(second),
                rel_tol=1e-10,
                abs_tol=1e-10,
            )

        check(
            f"invariant_{metric_name}",
            same,
            f"{metric_name} remains invariant under reversal "
            "within numerical tolerance.",
        )

    slope_original = original_metrics["linear_trend_slope"]
    slope_reversed = reversed_metrics["linear_trend_slope"]

    check(
        "trend_slope_sign_reversal",
        math.isclose(
            float(slope_original),
            -float(slope_reversed),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ),
        "Linear trend slope changes sign under sample-order reversal.",
    )

    check(
        "comparison_metrics_present",
        set(METRIC_NAMES).issubset(comparison.keys()),
        "All declared metrics are present in the comparison report.",
    )

    constant_signal = [5.0, 5.0, 5.0, 5.0]
    constant_result = run_time_reversal_test(constant_signal)

    check(
        "constant_entropy_zero",
        math.isclose(
            constant_result["original_metrics"]["shannon_entropy_bits"],
            0.0,
            abs_tol=1e-12,
        ),
        "Constant signal has zero histogram entropy by this estimator.",
    )

    check(
        "constant_autocorrelation_undefined",
        constant_result["original_metrics"]["lag1_autocorrelation"] is None,
        "Lag-1 autocorrelation is undefined for a constant signal.",
    )

    invalid_signal_rejected = False
    try:
        run_time_reversal_test([1.0, float("nan"), 3.0])
    except ValueError:
        invalid_signal_rejected = True

    check(
        "non_finite_signal_rejected",
        invalid_signal_rejected,
        "NaN and infinite input values are rejected.",
    )

    short_signal_rejected = False
    try:
        run_time_reversal_test([1.0])
    except ValueError:
        short_signal_rejected = True

    check(
        "too_short_signal_rejected",
        short_signal_rejected,
        "Signals with fewer than two samples are rejected.",
    )

    # Ensure report is valid strict JSON (no NaN/Infinity tokens).
    json.dumps(result, allow_nan=False)

    check(
        "strict_json_serializable",
        True,
        "Result is serializable as strict JSON.",
    )

    return {
        "self_check": True,
        "module": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "all_checks_passed": all(item["passed"] for item in checks),
        "checks": checks,
        "scientific_boundary": {
            "scientific_validity_established": False,
            "causality_established": False,
            "medical_diagnosis": False,
        },
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare descriptive signal metrics before and after "
            "sample-order reversal. Diagnostic only."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to a numeric CSV file. Omit to run self-check.",
    )
    parser.add_argument(
        "--column",
        type=str,
        default=None,
        help="Numeric column name in the input CSV.",
    )
    parser.add_argument(
        "--entropy-bins",
        type=int,
        default=DEFAULT_ENTROPY_BINS,
        help=f"Histogram bins for entropy (default: {DEFAULT_ENTROPY_BINS}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation_reports") / "time_reversal",
        help="Directory for output reports.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run synthetic self-checks and exit.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        validate_positive_integer(args.entropy_bins, "entropy_bins")

        if args.self_check or args.input is None:
            result = self_check()

            print(json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ))

            if not result["all_checks_passed"]:
                return 1

            print(
                f"\nSELF-CHECK: PASS "
                f"({result['checks_passed']}/{result['checks_total']})"
            )
            return 0

        if not args.column:
            raise ValueError(
                "--column is required when --input is supplied."
            )

        values, input_metadata = load_numeric_csv(
            args.input,
            args.column,
        )

        result = run_time_reversal_test(
            values,
            entropy_bins=args.entropy_bins,
        )

        report = build_report(result, input_metadata)
        json_path, csv_path = save_report(report, args.output_dir)

        print_summary(report)
        print(f"JSON report : {json_path.resolve()}")
        print(f"CSV report  : {csv_path.resolve()}")

        return 0

    except (OSError, ValueError, TypeError, AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
