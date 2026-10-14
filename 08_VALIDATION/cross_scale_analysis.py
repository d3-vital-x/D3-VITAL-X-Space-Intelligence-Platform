"""
D³ VITAL-X Space Intelligence Platform
Module 26: cross_scale_analysis.py
Version: 1.0.0

Purpose
-------
Compare descriptive signal statistics across multiple window sizes.

Features
--------
- Multi-scale window analysis
- Mean, sample variance and standard deviation
- Histogram-based Shannon entropy (bits)
- Lag-1 autocorrelation
- Linear trend slope
- Cross-scale metric comparison
- Input SHA-256 fingerprint
- Deterministic analysis for fixed inputs and settings
- JSON and CSV report export
- Synthetic-data self-check
- Command-line interface

Scientific boundaries
---------------------
- Diagnostic analysis only.
- Does not establish scientific validity.
- Does not establish anomaly-detection accuracy.
- Does not infer causality or physical mechanisms.
- Does not perform medical diagnosis.
- Source data is never modified.
- Window sizes are measured in samples, not physical time,
  unless the user separately supplies a sampling interpretation.

Dependencies
------------
Python standard library only.

Example
-------
python cross_scale_analysis.py \
    --input signal.csv \
    --column signal \
    --scales 16,32,64,128 \
    --stride 8 \
    --output-dir validation_reports/cross_scale
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
from typing import Any, Optional, Sequence


MODULE_NAME = "cross_scale_analysis"
MODULE_NUMBER = 26
MODULE_VERSION = "1.0.0"
REPORT_SCHEMA_VERSION = "1.0"

DEFAULT_SCALES = (16, 32, 64, 128)
DEFAULT_STRIDE = 8
DEFAULT_ENTROPY_BINS = 16

SUPPORTED_METRICS = (
    "mean",
    "variance",
    "standard_deviation",
    "shannon_entropy_bits",
    "lag1_autocorrelation",
    "linear_trend_slope",
)


# ============================================================
# 1. General utilities
# ============================================================

def utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Calculate a file SHA-256 fingerprint incrementally."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _validate_integer(
    value: Any,
    name: str,
    minimum: int = 1,
) -> int:
    """Validate an integer while explicitly rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")

    if value < minimum:
        raise ValueError(
            f"{name} must be >= {minimum}."
        )

    return value


def _finite_float(value: Any, name: str = "value") -> float:
    """Convert a value to a finite float."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not boolean.")

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be numeric."
        ) from exc

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")

    return result


def validate_signal(
    values: Sequence[float],
    min_samples: int = 2,
) -> list[float]:
    """
    Validate a one-dimensional sequence of finite numeric values.

    Input values are copied into a new list. The original sequence
    is never modified.
    """
    _validate_integer(min_samples, "min_samples")

    if isinstance(values, (str, bytes)):
        raise ValueError("Signal must be a numeric sequence.")

    try:
        signal = [
            _finite_float(value, f"signal[{index}]")
            for index, value in enumerate(values)
        ]
    except TypeError as exc:
        raise ValueError(
            "Signal must be an iterable numeric sequence."
        ) from exc

    if len(signal) < min_samples:
        raise ValueError(
            f"At least {min_samples} valid samples are required."
        )

    return signal


def parse_scales(text: str) -> tuple[int, ...]:
    """Parse comma-separated window sizes."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Scales must not be empty.")

    try:
        scales = tuple(
            int(part.strip())
            for part in text.split(",")
        )
    except ValueError as exc:
        raise ValueError(
            "Scales must be comma-separated integers, "
            "for example: 16,32,64,128."
        ) from exc

    if not scales:
        raise ValueError("At least one scale is required.")

    if any(scale < 2 for scale in scales):
        raise ValueError(
            "Every scale must contain at least 2 samples."
        )

    if len(set(scales)) != len(scales):
        raise ValueError("Duplicate scales are not permitted.")

    return tuple(sorted(scales))


# ============================================================
# 2. Descriptive signal metrics
# ============================================================

def calculate_shannon_entropy(
    values: Sequence[float],
    bins: int = DEFAULT_ENTROPY_BINS,
) -> float:
    """
    Estimate histogram-based Shannon entropy in bits.

    The observed range is divided into equal-width bins.
    A constant signal has entropy 0 bits.

    Important:
    This is a discretization-dependent descriptive statistic,
    not a differential-entropy estimator.
    """
    bins = _validate_integer(bins, "bins")

    signal = validate_signal(values, min_samples=1)

    minimum = min(signal)
    maximum = max(signal)

    if minimum == maximum:
        return 0.0

    width = (maximum - minimum) / bins

    counts = [0] * bins

    for value in signal:
        index = int((value - minimum) / width)

        # Include the maximum value in the last bin.
        index = min(index, bins - 1)
        index = max(index, 0)

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
    Calculate lag-1 Pearson autocorrelation.

    Returns None when the statistic is undefined, including
    constant signals or insufficient samples.
    """
    signal = validate_signal(values, min_samples=2)

    left = signal[:-1]
    right = signal[1:]

    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)

    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right)
    )

    left_sum_squares = sum(
        (x - left_mean) ** 2
        for x in left
    )

    right_sum_squares = sum(
        (y - right_mean) ** 2
        for y in right
    )

    denominator = math.sqrt(
        left_sum_squares * right_sum_squares
    )

    if denominator == 0.0:
        return None

    return float(numerator / denominator)


def calculate_linear_trend_slope(
    values: Sequence[float],
) -> Optional[float]:
    """
    Calculate ordinary least-squares slope against sample index.

    The index is 0, 1, ..., n-1. The slope is therefore expressed
    in signal units per sample.

    This is not a physical-time derivative unless the sampling
    interval and units are known.
    """
    signal = validate_signal(values, min_samples=2)

    n = len(signal)
    mean_y = statistics.fmean(signal)
    mean_x = (n - 1) / 2.0

    numerator = sum(
        (index - mean_x) * (value - mean_y)
        for index, value in enumerate(signal)
    )

    denominator = sum(
        (index - mean_x) ** 2
        for index in range(n)
    )

    if denominator == 0.0:
        return None

    return float(numerator / denominator)


def calculate_metrics(
    values: Sequence[float],
    entropy_bins: int = DEFAULT_ENTROPY_BINS,
) -> dict[str, Optional[float]]:
    """Calculate the supported descriptive metrics for one window."""
    signal = validate_signal(values, min_samples=2)

    variance = statistics.variance(signal)

    return {
        "mean": float(statistics.fmean(signal)),
        "variance": float(variance),
        "standard_deviation": float(math.sqrt(variance)),
        "shannon_entropy_bits": calculate_shannon_entropy(
            signal,
            bins=entropy_bins,
        ),
        "lag1_autocorrelation": calculate_lag1_autocorrelation(
            signal,
        ),
        "linear_trend_slope": calculate_linear_trend_slope(
            signal,
        ),
    }


# ============================================================
# 3. Multi-scale window analysis
# ============================================================

def _summarize_metric(
    values: Sequence[Optional[float]],
) -> dict[str, Optional[float]]:
    """Summarize finite metric values without treating None as zero."""
    valid = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]

    if not valid:
        return {
            "valid_windows": 0,
            "mean": None,
            "standard_deviation": None,
            "minimum": None,
            "maximum": None,
        }

    return {
        "valid_windows": len(valid),
        "mean": float(statistics.fmean(valid)),
        "standard_deviation": (
            float(statistics.stdev(valid))
            if len(valid) >= 2
            else 0.0
        ),
        "minimum": float(min(valid)),
        "maximum": float(max(valid)),
    }


def analyze_scale(
    signal: Sequence[float],
    window_size: int,
    stride: int = DEFAULT_STRIDE,
    entropy_bins: int = DEFAULT_ENTROPY_BINS,
) -> dict[str, Any]:
    """
    Analyze consecutive windows at one scale.

    The final incomplete window is excluded.
    Overlapping windows are permitted when stride < window_size.

    Window-level observations may therefore be dependent.
    Their count must not be interpreted as an independent
    sample size.
    """
    window_size = _validate_integer(window_size, "window_size", 2)
    stride = _validate_integer(stride, "stride")
    entropy_bins = _validate_integer(entropy_bins, "entropy_bins")

    values = validate_signal(signal, min_samples=2)

    if window_size > len(values):
        raise ValueError(
            f"Window size {window_size} exceeds signal length "
            f"{len(values)}."
        )

    windows: list[dict[str, Any]] = []

    for start in range(
        0,
        len(values) - window_size + 1,
        stride,
    ):
        end = start + window_size
        window = values[start:end]

        metrics = calculate_metrics(
            window,
            entropy_bins=entropy_bins,
        )

        windows.append({
            "window_index": len(windows),
            "start_sample": start,
            "end_sample_exclusive": end,
            "metrics": metrics,
        })

    metric_summary = {}

    for metric_name in SUPPORTED_METRICS:
        metric_summary[metric_name] = _summarize_metric(
            [
                window["metrics"][metric_name]
                for window in windows
            ]
        )

    return {
        "window_size": window_size,
        "stride": stride,
        "window_count": len(windows),
        "overlapping_windows": stride < window_size,
        "metric_summary": metric_summary,
        "windows": windows,
    }


def analyze_cross_scale(
    values: Sequence[float],
    scales: Sequence[int] = DEFAULT_SCALES,
    stride: int = DEFAULT_STRIDE,
    entropy_bins: int = DEFAULT_ENTROPY_BINS,
    dataset_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run descriptive multi-scale analysis and compare scale summaries.

    Parameters
    ----------
    values:
        One-dimensional finite numeric signal.
    scales:
        Window sizes, measured in samples.
    stride:
        Step between successive window starts.
    entropy_bins:
        Number of equal-width bins used for Shannon entropy.
    dataset_id:
        Optional caller-supplied dataset identifier.

    Returns
    -------
    dict
        JSON-serializable report. No source data is modified.
    """
    signal = validate_signal(values, min_samples=2)

    stride = _validate_integer(stride, "stride")
    entropy_bins = _validate_integer(entropy_bins, "entropy_bins")

    scale_values = tuple(
        _validate_integer(scale, "scale", 2)
        for scale in scales
    )

    if not scale_values:
        raise ValueError("At least one scale is required.")

    if len(set(scale_values)) != len(scale_values):
        raise ValueError("Scales must be unique.")

    scale_values = tuple(sorted(scale_values))

    if any(scale > len(signal) for scale in scale_values):
        raise ValueError(
            "Every window size must be <= the signal length."
        )

    scale_results = [
        analyze_scale(
            signal=signal,
            window_size=scale,
            stride=stride,
            entropy_bins=entropy_bins,
        )
        for scale in scale_values
    ]

    cross_scale_summary: dict[str, Any] = {}

    for metric_name in SUPPORTED_METRICS:
        points = []

        for result in scale_results:
            summary = result["metric_summary"][metric_name]

            points.append({
                "window_size": result["window_size"],
                "mean": summary["mean"],
                "valid_windows": summary["valid_windows"],
            })

        valid_points = [
            point for point in points
            if point["mean"] is not None
        ]

        metric_means = [
            point["mean"]
            for point in valid_points
        ]

        if metric_means:
            mean_across_scales = statistics.fmean(metric_means)
            spread = max(metric_means) - min(metric_means)

            # Descriptive relative spread. Undefined when the
            # average is effectively zero.
            if abs(mean_across_scales) > 1e-15:
                relative_spread = (
                    spread / abs(mean_across_scales)
                )
            else:
                relative_spread = None
        else:
            mean_across_scales = None
            spread = None
            relative_spread = None

        cross_scale_summary[metric_name] = {
            "scale_points": points,
            "number_of_valid_scales": len(valid_points),
            "mean_of_scale_means": (
                float(mean_across_scales)
                if mean_across_scales is not None
                else None
            ),
            "absolute_spread_of_scale_means": (
                float(spread) if spread is not None else None
            ),
            "relative_spread_of_scale_means": (
                float(relative_spread)
                if relative_spread is not None
                else None
            ),
            "interpretation": (
                "DESCRIPTIVE_ONLY"
            ),
        }

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "module": MODULE_NAME,
        "module_number": MODULE_NUMBER,
        "module_version": MODULE_VERSION,
        "created_at_utc": utc_now(),
        "status": "COMPLETED",
        "dataset": {
            "dataset_id": dataset_id,
            "sample_count": len(signal),
            "source_data_modified": False,
        },
        "configuration": {
            "scales_samples": list(scale_values),
            "stride_samples": stride,
            "entropy_bins": entropy_bins,
            "metric_names": list(SUPPORTED_METRICS),
            "partial_windows": "EXCLUDED",
        },
        "scale_results": scale_results,
        "cross_scale_summary": cross_scale_summary,
        "scientific_boundary": {
            "analysis_type": "DESCRIPTIVE_DIAGNOSTIC",
            "scientific_validity_established": False,
            "anomaly_accuracy_established": False,
            "causality_established": False,
            "clinical_diagnosis_performed": False,
            "human_review_required": True,
            "overlapping_windows_may_be_dependent": True,
            "entropy_estimator": (
                "HISTOGRAM_BASED_SHANNON_ENTROPY_BITS"
            ),
        },
    }


# ============================================================
# 4. CSV input
# ============================================================

def load_numeric_csv(
    csv_path: str | Path,
    column: str,
) -> tuple[list[float], dict[str, Any]]:
    """
    Load finite numeric values from one CSV column.

    Blank and non-numeric rows are skipped and counted.
    Non-finite numeric values are skipped and counted.
    Original CSV content is never modified.
    """
    path = Path(csv_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Input CSV file not found: {path}"
        )

    if not isinstance(column, str) or not column.strip():
        raise ValueError("Column name must not be empty.")

    values: list[float] = []
    total_rows = 0
    blank_count = 0
    non_numeric_count = 0
    non_finite_count = 0

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            raise ValueError(
                "CSV does not contain a header row."
            )

        if column not in reader.fieldnames:
            raise ValueError(
                f"Column '{column}' not found. "
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:
            total_rows += 1
            raw_value = row.get(column)

            if raw_value is None or not raw_value.strip():
                blank_count += 1
                continue

            try:
                value = float(raw_value.strip())
            except (TypeError, ValueError, OverflowError):
                non_numeric_count += 1
                continue

            if not math.isfinite(value):
                non_finite_count += 1
                continue

            values.append(value)

    if len(values) < 2:
        raise ValueError(
            "Fewer than two valid finite numeric samples remain."
        )

    metadata = {
        "input_path": str(path),
        "input_filename": path.name,
        "input_sha256": sha256_file(path),
        "column": column,
        "total_data_rows": total_rows,
        "valid_numeric_samples": len(values),
        "blank_rows": blank_count,
        "non_numeric_rows": non_numeric_count,
        "non_finite_rows": non_finite_count,
        "source_modified": False,
    }

    return values, metadata


# ============================================================
# 5. Report generation and export
# ============================================================

def build_report(
    values: Sequence[float],
    scales: Sequence[int] = DEFAULT_SCALES,
    stride: int = DEFAULT_STRIDE,
    entropy_bins: int = DEFAULT_ENTROPY_BINS,
    dataset_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build the analysis report and attach input provenance."""
    metadata = dict(dataset_metadata or {})

    report = analyze_cross_scale(
        values=values,
        scales=scales,
        stride=stride,
        entropy_bins=entropy_bins,
        dataset_id=metadata.get("input_sha256"),
    )

    report["input_metadata"] = metadata

    report["execution_environment"] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    return report


def save_report(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """
    Save JSON and CSV summaries.

    Output files are written to the requested output directory.
    Input files are not modified.
    """
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)

    json_path = directory / "cross_scale_report.json"
    csv_path = directory / "cross_scale_summary.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            report,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        fieldnames = [
            "window_size",
            "window_count",
            "overlapping_windows",
            "metric",
            "valid_windows",
            "mean",
            "standard_deviation",
            "minimum",
            "maximum",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for scale in report["scale_results"]:
            for metric, summary in scale["metric_summary"].items():
                writer.writerow({
                    "window_size": scale["window_size"],
                    "window_count": scale["window_count"],
                    "overlapping_windows": (
                        scale["overlapping_windows"]
                    ),
                    "metric": metric,
                    "valid_windows": summary["valid_windows"],
                    "mean": summary["mean"],
                    "standard_deviation": (
                        summary["standard_deviation"]
                    ),
                    "minimum": summary["minimum"],
                    "maximum": summary["maximum"],
                })

    return {
        "json_report": str(json_path),
        "csv_report": str(csv_path),
    }


def print_summary(report: dict[str, Any]) -> None:
    """Print a compact summary without claiming scientific validation."""
    print("\n" + "=" * 68)
    print("D³ VITAL-X — CROSS-SCALE ANALYSIS")
    print("=" * 68)

    print("Module:", f"{MODULE_NUMBER}. {MODULE_NAME}")
    print("Status:", report.get("status", "UNKNOWN"))

    dataset = report.get("dataset", {})
    print("Samples:", dataset.get("sample_count", 0))

    configuration = report.get("configuration", {})
    print("Scales:", configuration.get("scales_samples", []))
    print("Stride:", configuration.get("stride_samples"))

    print("-" * 68)

    for metric, summary in report.get(
        "cross_scale_summary", {}
    ).items():
        print(f"\nMetric: {metric}")
        print(
            "  Valid scales:",
            summary["number_of_valid_scales"],
        )
        print(
            "  Mean of scale means:",
            summary["mean_of_scale_means"],
        )
        print(
            "  Absolute spread:",
            summary["absolute_spread_of_scale_means"],
        )
        print(
            "  Relative spread:",
            summary["relative_spread_of_scale_means"],
        )

    print("\nScientific validity established: NO")
    print("Anomaly-detection accuracy established: NO")
    print("Medical diagnosis performed: NO")
    print("=" * 68 + "\n")


# ============================================================
# 6. Deterministic synthetic self-check
# ============================================================

def self_check() -> dict[str, Any]:
    """
    Run deterministic checks using synthetic data only.

    No external dataset is required.
    """
    checks: dict[str, bool] = {}

    try:
        constant = [3.0] * 64

        checks["constant_entropy_zero"] = (
            calculate_shannon_entropy(constant) == 0.0
        )

        checks["constant_variance_zero"] = (
            statistics.variance(constant) == 0.0
        )

        checks["constant_autocorrelation_undefined"] = (
            calculate_lag1_autocorrelation(constant) is None
        )

        ramp = [float(index) for index in range(128)]

        checks["linear_ramp_slope_positive"] = (
            calculate_linear_trend_slope(ramp) is not None
            and calculate_linear_trend_slope(ramp) > 0.0
        )

        original = list(ramp)

        report = analyze_cross_scale(
            values=ramp,
            scales=(16, 32, 64),
            stride=8,
            entropy_bins=8,
            dataset_id="synthetic-self-check",
        )

        checks["expected_scales_present"] = (
            [
                item["window_size"]
                for item in report["scale_results"]
            ]
            == [16, 32, 64]
        )

        checks["all_scales_have_windows"] = all(
            item["window_count"] > 0
            for item in report["scale_results"]
        )

        checks["all_supported_metrics_present"] = all(
            metric in report["cross_scale_summary"]
            for metric in SUPPORTED_METRICS
        )

        checks["input_unchanged"] = ramp == original

        serialized = json.dumps(
            report,
            allow_nan=False,
        )

        restored = json.loads(serialized)

        checks["strict_json_round_trip"] = (
            restored["status"] == "COMPLETED"
        )

        try:
            analyze_cross_scale(
                values=[1.0, 2.0, 3.0],
                scales=(4,),
            )
            checks["oversized_window_rejected"] = False
        except ValueError:
            checks["oversized_window_rejected"] = True

        try:
            validate_signal([1.0, float("nan"), 2.0])
            checks["non_finite_input_rejected"] = False
        except ValueError:
            checks["non_finite_input_rejected"] = True

    except Exception as exc:
        return {
            "module": MODULE_NAME,
            "module_number": MODULE_NUMBER,
            "passed": False,
            "checks": checks,
            "error": f"{type(exc).__name__}: {exc}",
            "scientific_validity_established": False,
        }

    return {
        "module": MODULE_NAME,
        "module_number": MODULE_NUMBER,
        "module_version": MODULE_VERSION,
        "passed": all(checks.values()),
        "checks": checks,
        "check_count": len(checks),
        "failed_checks": [
            name for name, passed in checks.items()
            if not passed
        ],
        "scientific_validity_established": False,
        "synthetic_data_only": True,
    }


# ============================================================
# 7. Command-line interface
# ============================================================

def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic-only cross-scale signal analysis "
            "for the D³ VITAL-X Space Intelligence Platform."
        )
    )

    parser.add_argument(
        "--input",
        default=None,
        help="Input CSV file. Omit to run the synthetic self-check.",
    )

    parser.add_argument(
        "--column",
        default=None,
        help="Numeric CSV column to analyze.",
    )

    parser.add_argument(
        "--scales",
        default="16,32,64,128",
        help="Comma-separated window sizes in samples.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Step between window starts (default: 8).",
    )

    parser.add_argument(
        "--entropy-bins",
        type=int,
        default=DEFAULT_ENTROPY_BINS,
        help="Number of histogram bins (default: 16).",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for JSON and CSV reports.",
    )

    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run synthetic-data checks and exit.",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Execute the CLI workflow."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.stride < 1:
        parser.error("--stride must be >= 1.")

    if args.entropy_bins < 1:
        parser.error("--entropy-bins must be >= 1.")

    try:
        if args.self_check or args.input is None:
            result = self_check()

            print(
                json.dumps(
                    result,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )

            if not result["passed"]:
                return 1

            print("\nCross-scale self-check passed.")
            return 0

        if not args.column:
            parser.error("--column is required when --input is used.")

        scales = parse_scales(args.scales)

        values, metadata = load_numeric_csv(
            csv_path=args.input,
            column=args.column,
        )

        report = build_report(
            values=values,
            scales=scales,
            stride=args.stride,
            entropy_bins=args.entropy_bins,
            dataset_metadata=metadata,
        )

        if args.output_dir:
            output_dir = Path(args.output_dir).expanduser()
        else:
            project_root = Path(__file__).resolve().parent.parent
            output_dir = (
                project_root
                / "validation_reports"
                / "cross_scale"
            )

        paths = save_report(report, output_dir)
        print_summary(report)

        for label, path in paths.items():
            print(f"{label}: {path}")

        return 0

    except (OSError, ValueError, csv.Error, RuntimeError) as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
