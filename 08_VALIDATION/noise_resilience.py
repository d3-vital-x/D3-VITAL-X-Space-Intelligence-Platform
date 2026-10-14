
"""
D³ VITAL-X Space Intelligence Platform
Module 25: noise_resilience.py

Purpose
-------
Evaluate how descriptive signal statistics change under controlled
additive Gaussian noise.

Noise model
-----------
noisy_signal = original_signal + GaussianNoise

noise_standard_deviation = noise_level * original_sample_standard_deviation

Supported metrics
-----------------
- mean
- standard_deviation
- lag1_autocorrelation
- linear_trend_slope
- root_mean_square_energy

Outputs
-------
- JSON report
- CSV summary by noise level and metric

Scientific boundary
-------------------
This is a diagnostic robustness experiment. It does not establish
scientific validity, anomaly-detection accuracy, or clinical validity.

The experiment assumes that additive Gaussian noise is a useful stress
model for the dataset. Real measurement noise may be heteroscedastic,
correlated, non-Gaussian, or otherwise different.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence


MODULE_NAME = "noise_resilience"
MODULE_NUMBER = 25
MODULE_VERSION = "1.0.0"
REPORT_SCHEMA_VERSION = "1.0"

SUPPORTED_METRICS = (
    "mean",
    "standard_deviation",
    "lag1_autocorrelation",
    "linear_trend_slope",
    "root_mean_square_energy",
)

DEFAULT_NOISE_LEVELS = (0.0, 0.01, 0.05, 0.10, 0.20, 0.50)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Calculate SHA-256 incrementally."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_numeric_csv(
    csv_path: str | Path,
    column: str,
) -> tuple[list[float], dict[str, Any]]:
    """Read finite numeric values from one CSV column."""
    path = Path(csv_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    values: list[float] = []
    total_rows = 0
    blank_count = 0
    non_numeric_count = 0
    non_finite_count = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            raise ValueError("CSV does not contain a header row.")

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
            except (TypeError, ValueError):
                non_numeric_count += 1
                continue

            if not math.isfinite(value):
                non_finite_count += 1
                continue

            values.append(value)

    metadata = {
        "input_file": str(path),
        "input_sha256": sha256_file(path),
        "column": column,
        "total_data_rows": total_rows,
        "valid_numeric_count": len(values),
        "blank_count": blank_count,
        "non_numeric_count": non_numeric_count,
        "non_finite_count": non_finite_count,
        "excluded_count": (
            blank_count + non_numeric_count + non_finite_count
        ),
    }

    if len(values) < 3:
        raise ValueError(
            "At least 3 finite numeric observations are required."
        )

    return values, metadata


def sample_standard_deviation(values: Sequence[float]) -> float:
    """Calculate sample standard deviation."""
    if len(values) < 2:
        raise ValueError("At least 2 values are required.")

    return float(statistics.stdev(values))


def lag1_autocorrelation(values: Sequence[float]) -> float:
    """Calculate lag-1 autocorrelation using centered values."""
    if len(values) < 3:
        raise ValueError(
            "Lag-1 autocorrelation requires at least 3 observations."
        )

    center = statistics.fmean(values)
    numerator = sum(
        (values[i] - center) * (values[i + 1] - center)
        for i in range(len(values) - 1)
    )
    denominator = sum((value - center) ** 2 for value in values)

    if denominator <= 0:
        raise ValueError(
            "Lag-1 autocorrelation is undefined for a constant signal."
        )

    return numerator / denominator


def linear_trend_slope(values: Sequence[float]) -> float:
    """Calculate ordinary least-squares slope against sample index."""
    n = len(values)

    if n < 2:
        raise ValueError("Trend slope requires at least 2 observations.")

    x_mean = (n - 1) / 2.0
    y_mean = statistics.fmean(values)

    numerator = sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(values)
    )
    denominator = sum(
        (index - x_mean) ** 2 for index in range(n)
    )

    if denominator <= 0:
        raise ValueError("Trend slope is undefined.")

    return numerator / denominator


def root_mean_square_energy(values: Sequence[float]) -> float:
    """
    Calculate RMS amplitude.

    This is a descriptive signal-amplitude measure, not necessarily
    physical energy unless the input units and model justify that meaning.
    """
    if not values:
        raise ValueError("Cannot calculate RMS for empty data.")

    return math.sqrt(statistics.fmean(value * value for value in values))


def calculate_metric(
    values: Sequence[float],
    metric: str,
) -> float:
    """Calculate a supported statistic and ensure the result is finite."""
    if not values:
        raise ValueError("Metric input must not be empty.")

    if not all(math.isfinite(value) for value in values):
        raise ValueError("Metric input contains non-finite values.")

    if metric == "mean":
        result = statistics.fmean(values)
    elif metric == "standard_deviation":
        result = sample_standard_deviation(values)
    elif metric == "lag1_autocorrelation":
        result = lag1_autocorrelation(values)
    elif metric == "linear_trend_slope":
        result = linear_trend_slope(values)
    elif metric == "root_mean_square_energy":
        result = root_mean_square_energy(values)
    else:
        raise ValueError(
            f"Unsupported metric '{metric}'. "
            f"Choose from {', '.join(SUPPORTED_METRICS)}."
        )

    if not math.isfinite(result):
        raise ValueError(
            f"Metric '{metric}' returned a non-finite value."
        )

    return float(result)


def parse_noise_levels(raw_levels: str) -> list[float]:
    """Parse comma-separated non-negative noise levels."""
    try:
        levels = [float(item.strip()) for item in raw_levels.split(",")]
    except ValueError as exc:
        raise ValueError(
            "Noise levels must be comma-separated numeric values."
        ) from exc

    if not levels:
        raise ValueError("At least one noise level is required.")

    if any(not math.isfinite(level) or level < 0 for level in levels):
        raise ValueError(
            "Noise levels must be finite, non-negative numbers."
        )

    if len(set(levels)) != len(levels):
        raise ValueError("Duplicate noise levels are not allowed.")

    return sorted(levels)


def generate_noisy_signal(
    values: Sequence[float],
    noise_level: float,
    rng: random.Random,
) -> tuple[list[float], float]:
    """
    Add independent Gaussian noise to each observation.

    Noise SD = noise_level * original sample SD.
    """
    if noise_level < 0 or not math.isfinite(noise_level):
        raise ValueError("noise_level must be finite and non-negative.")

    original_sd = sample_standard_deviation(values)
    noise_sd = noise_level * original_sd

    noisy_values = [
        value + rng.gauss(0.0, noise_sd)
        for value in values
    ]

    if not all(math.isfinite(value) for value in noisy_values):
        raise ValueError("Generated noisy signal contains non-finite values.")

    return noisy_values, noise_sd


def run_noise_experiment(
    values: Sequence[float],
    noise_levels: Sequence[float] = DEFAULT_NOISE_LEVELS,
    metrics: Sequence[str] = SUPPORTED_METRICS,
    repetitions: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run repeated additive-noise experiments.

    For each noise level, each metric is calculated over multiple noisy
    realizations. Reports include the mean and sample SD of the metric
    across repetitions, plus change relative to the original statistic.
    """
    start_time = time.perf_counter()

    original = [float(value) for value in values]

    if len(original) < 3:
        raise ValueError("At least 3 observations are required.")

    if not all(math.isfinite(value) for value in original):
        raise ValueError("Input values must all be finite.")

    if isinstance(repetitions, bool) or repetitions < 1:
        raise ValueError("repetitions must be an integer of at least 1.")

    if isinstance(repetitions, float) and not repetitions.is_integer():
        raise ValueError("repetitions must be a whole number.")

    repetitions = int(repetitions)

    levels = [float(level) for level in noise_levels]

    if not levels:
        raise ValueError("At least one noise level is required.")

    if any(not math.isfinite(level) or level < 0 for level in levels):
        raise ValueError("Noise levels must be finite and non-negative.")

    if len(set(levels)) != len(levels):
        raise ValueError("Noise levels must be unique.")

    chosen_metrics = list(metrics)

    if not chosen_metrics:
        raise ValueError("At least one metric must be selected.")

    unknown_metrics = sorted(
        set(chosen_metrics) - set(SUPPORTED_METRICS)
    )

    if unknown_metrics:
        raise ValueError(
            f"Unsupported metrics: {unknown_metrics}. "
            f"Supported metrics: {list(SUPPORTED_METRICS)}"
        )

    # Calculate the baseline once. If a metric is undefined for the
    # original data, fail explicitly instead of silently skipping it.
    baseline: dict[str, float] = {
        metric: calculate_metric(original, metric)
        for metric in chosen_metrics
    }

    original_sd = sample_standard_deviation(original)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    for noise_level in sorted(levels):
        metric_samples: dict[str, list[float]] = {
            metric: [] for metric in chosen_metrics
        }
        noise_sd = noise_level * original_sd

        for _ in range(repetitions):
            noisy_values, _ = generate_noisy_signal(
                values=original,
                noise_level=noise_level,
                rng=rng,
            )

            for metric in chosen_metrics:
                metric_samples[metric].append(
                    calculate_metric(noisy_values, metric)
                )

        for metric in chosen_metrics:
            estimates = metric_samples[metric]
            average_estimate = statistics.fmean(estimates)
            spread = (
                statistics.stdev(estimates)
                if len(estimates) >= 2
                else 0.0
            )
            baseline_value = baseline[metric]
            difference = average_estimate - baseline_value

            # Relative change is not meaningful when baseline is zero
            # or numerically very close to zero.
            relative_change_percent = None
            tolerance = 1e-12 * max(1.0, abs(baseline_value))

            if abs(baseline_value) > tolerance:
                relative_change_percent = (
                    difference / abs(baseline_value)
                ) * 100.0

            rows.append({
                "noise_level": noise_level,
                "noise_sd": noise_sd,
                "metric": metric,
                "baseline_value": baseline_value,
                "noisy_metric_mean": average_estimate,
                "noisy_metric_std": spread,
                "absolute_change": difference,
                "absolute_change_magnitude": abs(difference),
                "relative_change_percent": relative_change_percent,
                "repetitions": repetitions,
            })

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "noise_model": "independent additive Gaussian noise",
        "noise_scale_definition": (
            "noise SD = noise_level * original sample standard deviation"
        ),
        "sample_size": len(original),
        "original_sample_standard_deviation": original_sd,
        "noise_levels": sorted(levels),
        "metrics": chosen_metrics,
        "repetitions_per_level": repetitions,
        "random_seed": seed,
        "baseline_statistics": baseline,
        "results": rows,
        "elapsed_ms": round(elapsed_ms, 3),
        "interpretation": (
            "Changes summarize sensitivity to the specified synthetic "
            "noise model. No universal pass/fail threshold is imposed."
        ),
    }


def build_report(
    values: Sequence[float],
    input_metadata: dict[str, Any],
    noise_levels: Sequence[float] = DEFAULT_NOISE_LEVELS,
    metrics: Sequence[str] = SUPPORTED_METRICS,
    repetitions: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """Build a structured, reproducible report."""
    experiment = run_noise_experiment(
        values=values,
        noise_levels=noise_levels,
        metrics=metrics,
        repetitions=repetitions,
        seed=seed,
    )

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "project": "D3-VITAL-X-Space-Intelligence-Platform",
        "module": MODULE_NAME,
        "module_number": MODULE_NUMBER,
        "module_version": MODULE_VERSION,
        "created_at_utc": utc_now(),
        "status": "COMPLETED",
        "input": input_metadata,
        "runtime": {
            "python_version": sys.version,
            "platform": platform.platform(),
        },
        "experiment": experiment,
        "scientific_boundary": {
            "diagnostic_only": True,
            "scientific_validity_established": False,
            "anomaly_detection_accuracy_established": False,
            "medical_diagnosis_performed": False,
            "limitations": [
                "Gaussian additive noise may not represent real measurement noise.",
                "Noise levels are defined relative to the input sample SD.",
                "Missing and non-numeric values are excluded before the experiment.",
                "The selected statistics are descriptive and may not represent the full system.",
                "No universal resilience threshold is defined by this module.",
            ],
        },
    }


def save_report(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Save JSON report and a flat CSV summary."""
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = target_dir / f"noise_resilience_{stamp}.json"
    csv_path = target_dir / f"noise_resilience_{stamp}.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    rows = report["experiment"]["results"]

    fieldnames = [
        "noise_level",
        "noise_sd",
        "metric",
        "baseline_value",
        "noisy_metric_mean",
        "noisy_metric_std",
        "absolute_change",
        "absolute_change_magnitude",
        "relative_change_percent",
        "repetitions",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "json_report": str(json_path),
        "csv_report": str(csv_path),
    }


def print_summary(report: dict[str, Any]) -> None:
    """Print a compact report summary."""
    experiment = report["experiment"]

    print("\n" + "=" * 68)
    print("D³ VITAL-X — NOISE RESILIENCE REPORT")
    print("=" * 68)
    print("Module:", f"{MODULE_NUMBER}. {MODULE_NAME}")
    print("Valid observations:", experiment["sample_size"])
    print("Noise levels:", experiment["noise_levels"])
    print("Metrics:", ", ".join(experiment["metrics"]))
    print("Repetitions per level:", experiment["repetitions_per_level"])
    print("Random seed:", experiment["random_seed"])
    print()

    for row in experiment["results"]:
        relative = row["relative_change_percent"]
        relative_text = (
            "N/A" if relative is None else f"{relative:.4f}%"
        )

        print(
            f"noise={row['noise_level']:.4f} | "
            f"{row['metric']} | "
            f"baseline={row['baseline_value']:.6g} | "
            f"noisy_mean={row['noisy_metric_mean']:.6g} | "
            f"abs_change={row['absolute_change_magnitude']:.6g} | "
            f"relative_change={relative_text}"
        )

    print()
    print("Scientific validity established: NO")
    print("Anomaly-detection accuracy established: NO")
    print("Medical diagnosis performed: NO")
    print("=" * 68 + "\n")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate descriptive statistic changes under controlled "
            "additive Gaussian noise."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--column",
        required=True,
        help="Numeric column name.",
    )
    parser.add_argument(
        "--noise-levels",
        default="0,0.01,0.05,0.1,0.2,0.5",
        help="Comma-separated noise/SD ratios.",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(SUPPORTED_METRICS),
        help="Comma-separated list of supported metrics.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=100,
        help="Noise realizations per level (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for reports.",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)

    try:
        values, metadata = load_numeric_csv(
            csv_path=args.input,
            column=args.column,
        )

        levels = parse_noise_levels(args.noise_levels)
        metrics = [
            item.strip()
            for item in args.metrics.split(",")
            if item.strip()
        ]

        if not metrics:
            raise ValueError("At least one metric must be selected.")

        report = build_report(
            values=values,
            input_metadata=metadata,
            noise_levels=levels,
            metrics=metrics,
            repetitions=args.repetitions,
            seed=args.seed,
        )

        if args.output_dir:
            output_dir = Path(args.output_dir).expanduser()
        else:
            # This module is expected inside PROJECT_ROOT/08_VALIDATION/.
            project_root = Path(__file__).resolve().parent.parent
            output_dir = project_root / "validation_reports" / "noise_resilience"

        paths = save_report(report, output_dir)
        print_summary(report)

        for label, path in paths.items():
            print(f"{label}: {path}")

        return 0

    except (OSError, ValueError, csv.Error) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
