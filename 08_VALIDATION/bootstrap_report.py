
"""
D³ VITAL-X Space Intelligence Platform
Module 23: bootstrap_report.py

Purpose
-------
Generate diagnostic bootstrap estimates and percentile confidence intervals
for a numeric column in a CSV dataset.

Features
--------
- Bootstrap resampling with replacement.
- Mean, median, variance, standard deviation, and minimum/maximum metrics.
- Configurable number of resamples, confidence level, and random seed.
- Optional NumPy acceleration is not required.
- JSON and CSV report export.
- Records excluded non-numeric or missing values.
- Does not modify the input dataset.

Scientific boundary
-------------------
Bootstrap intervals describe resampling uncertainty under the chosen
sampling procedure. They do not establish causality, scientific validity,
model accuracy, or clinical validity.
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
from typing import Any, Callable, Optional, Sequence


MODULE_NAME = "bootstrap_report"
MODULE_NUMBER = 23
MODULE_VERSION = "1.0.0"
REPORT_SCHEMA_VERSION = "1.0"

SUPPORTED_METRICS = (
    "mean",
    "median",
    "variance",
    "std",
    "minimum",
    "maximum",
)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 hash using bounded memory."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def quantile(values: Sequence[float], probability: float) -> float:
    """
    Calculate a linearly interpolated quantile.

    This uses the common (n - 1) interpolation convention.
    """
    if not values:
        raise ValueError("Cannot calculate a quantile from empty data.")

    if not 0.0 <= probability <= 1.0:
        raise ValueError("Probability must be between 0 and 1.")

    ordered = sorted(values)

    if len(ordered) == 1:
        return float(ordered[0])

    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return float(ordered[lower_index])

    fraction = position - lower_index
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]

    return float(
        lower_value + fraction * (upper_value - lower_value)
    )


def calculate_metric(values: Sequence[float], metric: str) -> float:
    """Calculate one supported descriptive statistic."""
    if not values:
        raise ValueError("Metric input must not be empty.")

    if metric == "mean":
        return float(statistics.fmean(values))

    if metric == "median":
        return float(statistics.median(values))

    if metric == "variance":
        # Sample variance; requires at least two observations.
        if len(values) < 2:
            raise ValueError(
                "Sample variance requires at least two observations."
            )
        return float(statistics.variance(values))

    if metric == "std":
        # Sample standard deviation; requires at least two observations.
        if len(values) < 2:
            raise ValueError(
                "Sample standard deviation requires at least two observations."
            )
        return float(statistics.stdev(values))

    if metric == "minimum":
        return float(min(values))

    if metric == "maximum":
        return float(max(values))

    raise ValueError(
        f"Unsupported metric: {metric}. "
        f"Choose from {', '.join(SUPPORTED_METRICS)}."
    )


def load_numeric_csv(
    csv_path: str | Path,
    column: str,
) -> tuple[list[float], dict[str, Any]]:
    """
    Load finite numeric values from a CSV column.

    Blank, non-numeric, NaN, and infinite values are excluded.
    The original file is never modified.
    """
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
                number = float(raw_value.strip())
            except (TypeError, ValueError):
                non_numeric_count += 1
                continue

            if not math.isfinite(number):
                non_finite_count += 1
                continue

            values.append(number)

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

    if not values:
        raise ValueError(
            f"No finite numeric values were found in column '{column}'."
        )

    return values, metadata


def bootstrap_statistic(
    values: Sequence[float],
    metric: str = "mean",
    resamples: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 42,
    batch_size: int = 1000,
) -> dict[str, Any]:
    """
    Estimate a statistic and its percentile bootstrap confidence interval.

    Sampling is with replacement and preserves the original sample size.

    Parameters
    ----------
    values:
        Finite numeric observations.
    metric:
        One of SUPPORTED_METRICS.
    resamples:
        Number of bootstrap resamples; minimum 100.
    confidence_level:
        Central interval probability, e.g. 0.95.
    seed:
        Random seed for repeatability.
    batch_size:
        Number of resamples between progress checks. No output is emitted.
    """
    start_time = time.perf_counter()

    clean_values = [float(value) for value in values]

    if not clean_values:
        raise ValueError("At least one numeric observation is required.")

    if not all(math.isfinite(value) for value in clean_values):
        raise ValueError("Input values must all be finite.")

    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported metric '{metric}'. "
            f"Choose from {', '.join(SUPPORTED_METRICS)}."
        )

    if isinstance(resamples, bool) or resamples < 100:
        raise ValueError("resamples must be an integer of at least 100.")

    if isinstance(resamples, float) and not resamples.is_integer():
        raise ValueError("resamples must be a whole number.")

    resamples = int(resamples)

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    if isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if metric in {"variance", "std"} and len(clean_values) < 2:
        raise ValueError(
            f"Metric '{metric}' requires at least two observations."
        )

    observed_statistic = calculate_metric(clean_values, metric)

    generator = random.Random(seed)
    sample_size = len(clean_values)
    bootstrap_statistics: list[float] = []

    # Generate one resample at a time to avoid a large resample matrix.
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)

        for _ in range(start, stop):
            sample = [
                clean_values[generator.randrange(sample_size)]
                for _ in range(sample_size)
            ]

            bootstrap_statistics.append(
                calculate_metric(sample, metric)
            )

    alpha = 1.0 - confidence_level

    lower_bound = quantile(
        bootstrap_statistics,
        alpha / 2.0,
    )
    upper_bound = quantile(
        bootstrap_statistics,
        1.0 - alpha / 2.0,
    )

    bootstrap_mean = statistics.fmean(bootstrap_statistics)
    bootstrap_sd = (
        statistics.stdev(bootstrap_statistics)
        if len(bootstrap_statistics) >= 2
        else 0.0
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "metric": metric,
        "sample_size": sample_size,
        "resamples": resamples,
        "confidence_level": confidence_level,
        "confidence_interval_method": "percentile bootstrap",
        "quantile_interpolation": "linear (n - 1 convention)",
        "observed_estimate": observed_statistic,
        "confidence_interval_lower": lower_bound,
        "confidence_interval_upper": upper_bound,
        "bootstrap_distribution_mean": bootstrap_mean,
        "bootstrap_distribution_std": bootstrap_sd,
        "random_seed": seed,
        "elapsed_ms": round(elapsed_ms, 3),
        "interpretation": (
            "Resampling-based uncertainty estimate; not proof of "
            "scientific validity, causality, or predictive accuracy."
        ),
    }


def build_report(
    values: Sequence[float],
    input_metadata: dict[str, Any],
    metric: str = "mean",
    resamples: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Build a structured bootstrap report."""
    result = bootstrap_statistic(
        values=values,
        metric=metric,
        resamples=resamples,
        confidence_level=confidence_level,
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
        "bootstrap": result,
        "scientific_boundary": {
            "diagnostic_only": True,
            "scientific_validity_established": False,
            "medical_diagnosis_performed": False,
            "causality_established": False,
            "limitations": [
                "The interval depends on the observed sample and resampling scheme.",
                "Ordinary bootstrap may be inappropriate for dependent time-series data.",
                "Small samples, outliers, selection bias, and missing data can affect results.",
                "The interval does not validate a model or scientific hypothesis.",
            ],
        },
    }


def save_report(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Save JSON and a compact CSV summary."""
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = target_dir / f"bootstrap_report_{timestamp}.json"
    csv_path = target_dir / f"bootstrap_report_{timestamp}.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    bootstrap = report["bootstrap"]
    input_metadata = report["input"]

    flat_row = {
        "module": report["module"],
        "module_number": report["module_number"],
        "created_at_utc": report["created_at_utc"],
        "input_file": input_metadata.get("input_file"),
        "input_sha256": input_metadata.get("input_sha256"),
        "column": input_metadata.get("column"),
        "metric": bootstrap["metric"],
        "sample_size": bootstrap["sample_size"],
        "resamples": bootstrap["resamples"],
        "confidence_level": bootstrap["confidence_level"],
        "observed_estimate": bootstrap["observed_estimate"],
        "confidence_interval_lower": bootstrap["confidence_interval_lower"],
        "confidence_interval_upper": bootstrap["confidence_interval_upper"],
        "bootstrap_distribution_mean": bootstrap["bootstrap_distribution_mean"],
        "bootstrap_distribution_std": bootstrap["bootstrap_distribution_std"],
        "random_seed": bootstrap["random_seed"],
        "status": report["status"],
    }

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_row.keys()))
        writer.writeheader()
        writer.writerow(flat_row)

    return {
        "json_report": str(json_path),
        "csv_report": str(csv_path),
    }


def print_summary(report: dict[str, Any]) -> None:
    """Print the main results to the terminal."""
    result = report["bootstrap"]
    metadata = report["input"]

    print("\n" + "=" * 66)
    print("D³ VITAL-X — BOOTSTRAP REPORT")
    print("=" * 66)
    print("Module:", f"{MODULE_NUMBER}. {MODULE_NAME}")
    print("Input column:", metadata.get("column"))
    print("Valid numeric observations:", result["sample_size"])
    print("Excluded input rows:", metadata.get("excluded_count", 0))
    print("Metric:", result["metric"])
    print("Resamples:", result["resamples"])
    print("Confidence level:", result["confidence_level"])
    print("Random seed:", result["random_seed"])
    print()
    print("Observed estimate:", result["observed_estimate"])
    print(
        "Confidence interval:",
        f"[{result['confidence_interval_lower']}, "
        f"{result['confidence_interval_upper']}]",
    )
    print()
    print("Scientific validity established: NO")
    print("Medical diagnosis performed: NO")
    print("=" * 66 + "\n")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a diagnostic percentile-bootstrap report "
            "from a numeric CSV column."
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
        help="Numeric column name to analyse.",
    )
    parser.add_argument(
        "--metric",
        choices=SUPPORTED_METRICS,
        default="mean",
        help="Statistic to bootstrap (default: mean).",
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=5000,
        help="Number of resamples, minimum 100 (default: 5000).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level between 0 and 1 (default: 0.95).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Report output directory.",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)

    try:
        input_path = Path(args.input).expanduser().resolve()

        values, metadata = load_numeric_csv(
            csv_path=input_path,
            column=args.column,
        )

        report = build_report(
            values=values,
            input_metadata=metadata,
            metric=args.metric,
            resamples=args.resamples,
            confidence_level=args.confidence,
            seed=args.seed,
        )

        if args.output_dir:
            output_dir = Path(args.output_dir).expanduser()
        else:
            # Assumes this file is inside PROJECT_ROOT/08_VALIDATION/.
            project_root = Path(__file__).resolve().parent.parent
            output_dir = project_root / "validation_reports" / "bootstrap"

        paths = save_report(report, output_dir)
        print_summary(report)

        print("JSON report:", paths["json_report"])
        print("CSV report:", paths["csv_report"])
        return 0

    except (OSError, ValueError, csv.Error) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
