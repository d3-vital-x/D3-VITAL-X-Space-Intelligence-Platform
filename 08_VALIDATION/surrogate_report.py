
"""
D³ VITAL-X Space Intelligence Platform
Module 24: surrogate_report.py

Purpose
-------
Compare an observed time-series statistic with a surrogate-data null
distribution.

Surrogate methods
-----------------
1. permutation:
   Randomly shuffles observations. Preserves the empirical values but
   destroys their original ordering.

2. phase_randomized:
   Randomizes Fourier phases while retaining Fourier amplitudes.
   Requires NumPy. Intended for approximately stationary, evenly sampled
   numeric series. It does not preserve the exact empirical distribution.

Supported statistics
--------------------
- lag1_autocorrelation
- skewness
- excess_kurtosis
- linear_trend_slope
- mean_abs_difference
- maximum_absolute_value

Important
---------
- A Monte Carlo p-value is an estimate under the selected null procedure.
- A surrogate test does not prove causality or scientific validity.
- The chosen statistic and surrogate method must match the scientific
  question and its null hypothesis.
- This module does not modify the input dataset.
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


MODULE_NAME = "surrogate_report"
MODULE_NUMBER = 24
MODULE_VERSION = "1.0.0"
REPORT_SCHEMA_VERSION = "1.0"

SURROGATE_METHODS = (
    "permutation",
    "phase_randomized",
)

SUPPORTED_STATISTICS = (
    "lag1_autocorrelation",
    "skewness",
    "excess_kurtosis",
    "linear_trend_slope",
    "mean_abs_difference",
    "maximum_absolute_value",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_numeric_csv(
    csv_path: str | Path,
    column: str,
) -> tuple[list[float], dict[str, Any]]:
    """Read finite numeric values from a CSV column."""
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
            except (ValueError, TypeError):
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

    if len(values) < 4:
        raise ValueError(
            "At least 4 finite numeric observations are required."
        )

    return values, metadata


def mean_value(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot calculate the mean of empty data.")
    return statistics.fmean(values)


def lag1_autocorrelation(values: Sequence[float]) -> float:
    """
    Compute lag-1 autocorrelation using centered adjacent observations.

    Returns NaN when the variance denominator is zero.
    """
    if len(values) < 3:
        raise ValueError("Lag-1 autocorrelation requires at least 3 values.")

    center = mean_value(values)
    left = [x - center for x in values[:-1]]
    right = [x - center for x in values[1:]]

    numerator = sum(a * b for a, b in zip(left, right))
    denominator = sum((x - center) ** 2 for x in values)

    if denominator == 0:
        return float("nan")

    return numerator / denominator


def skewness(values: Sequence[float]) -> float:
    """Calculate population-moment skewness."""
    if len(values) < 3:
        raise ValueError("Skewness requires at least 3 values.")

    center = mean_value(values)
    centered = [x - center for x in values]
    second_moment = statistics.fmean(x * x for x in centered)

    if second_moment <= 0:
        return float("nan")

    third_moment = statistics.fmean(x ** 3 for x in centered)
    return third_moment / (second_moment ** 1.5)


def excess_kurtosis(values: Sequence[float]) -> float:
    """Calculate excess kurtosis using population central moments."""
    if len(values) < 4:
        raise ValueError("Excess kurtosis requires at least 4 values.")

    center = mean_value(values)
    centered = [x - center for x in values]
    second_moment = statistics.fmean(x * x for x in centered)

    if second_moment <= 0:
        return float("nan")

    fourth_moment = statistics.fmean(x ** 4 for x in centered)
    return fourth_moment / (second_moment ** 2) - 3.0


def linear_trend_slope(values: Sequence[float]) -> float:
    """Ordinary least-squares slope against sample indices 0, 1, ..., n-1."""
    n = len(values)

    if n < 2:
        raise ValueError("Trend slope requires at least 2 values.")

    x_mean = (n - 1) / 2.0
    y_mean = mean_value(values)

    numerator = sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(values)
    )
    denominator = sum(
        (index - x_mean) ** 2 for index in range(n)
    )

    if denominator == 0:
        return float("nan")

    return numerator / denominator


def mean_abs_difference(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("Mean absolute difference needs at least 2 values.")

    return statistics.fmean(
        abs(values[i + 1] - values[i])
        for i in range(len(values) - 1)
    )


def maximum_absolute_value(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Input values must not be empty.")
    return max(abs(value) for value in values)


def calculate_statistic(
    values: Sequence[float],
    statistic_name: str,
) -> float:
    """Calculate a supported statistic and reject non-finite outputs."""
    if statistic_name == "lag1_autocorrelation":
        result = lag1_autocorrelation(values)
    elif statistic_name == "skewness":
        result = skewness(values)
    elif statistic_name == "excess_kurtosis":
        result = excess_kurtosis(values)
    elif statistic_name == "linear_trend_slope":
        result = linear_trend_slope(values)
    elif statistic_name == "mean_abs_difference":
        result = mean_abs_difference(values)
    elif statistic_name == "maximum_absolute_value":
        result = maximum_absolute_value(values)
    else:
        raise ValueError(
            f"Unsupported statistic '{statistic_name}'. "
            f"Choose from {', '.join(SUPPORTED_STATISTICS)}."
        )

    if not math.isfinite(result):
        raise ValueError(
            f"Statistic '{statistic_name}' is undefined for these data "
            "(for example, the input may be constant)."
        )

    return float(result)


def permutation_surrogate(
    values: Sequence[float],
    rng: random.Random,
) -> list[float]:
    """Shuffle values, preserving their empirical multiset."""
    surrogate = list(values)
    rng.shuffle(surrogate)
    return surrogate


def phase_randomized_surrogate(
    values: Sequence[float],
    rng: random.Random,
) -> list[float]:
    """
    Create a Fourier phase-randomized surrogate using NumPy.

    Fourier magnitudes are retained. DC and, for even-length input, the
    Nyquist component are kept real. The inverse transform is real-valued.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "phase_randomized requires NumPy. Install it with "
            "`pip install numpy` or select --method permutation."
        ) from exc

    array = np.asarray(values, dtype=float)

    if array.ndim != 1 or array.size < 4:
        raise ValueError(
            "Phase randomization requires a one-dimensional series "
            "with at least 4 observations."
        )

    spectrum = np.fft.rfft(array)
    randomized = spectrum.copy()

    # Preserve the zero-frequency component (mean).
    # Randomize only positive-frequency phases.
    last_randomizable = len(spectrum) - 1

    # For even n, the last rFFT coefficient is the Nyquist term and
    # must remain real. For odd n, the final coefficient can be randomized.
    if array.size % 2 == 0:
        last_randomizable -= 1

    for index in range(1, last_randomizable + 1):
        phase = rng.uniform(-math.pi, math.pi)
        randomized[index] = abs(spectrum[index]) * complex(
            math.cos(phase),
            math.sin(phase),
        )

    surrogate = np.fft.irfft(randomized, n=array.size)
    result = [float(value) for value in surrogate]

    if not all(math.isfinite(value) for value in result):
        raise ValueError("Generated surrogate contains non-finite values.")

    return result


def monte_carlo_p_value(
    observed: float,
    surrogate_statistics: Sequence[float],
    alternative: str = "two-sided",
) -> float:
    """
    Calculate a plus-one corrected Monte Carlo p-value.

    two-sided:
        Counts null statistics at least as far from zero as the observed
        statistic. This is appropriate only when zero is the meaningful
        reference for the selected statistic.

    greater:
        Tests whether the observed statistic is unusually large.

    less:
        Tests whether the observed statistic is unusually small.
    """
    if not surrogate_statistics:
        raise ValueError("Surrogate statistics must not be empty.")

    total = len(surrogate_statistics)

    if alternative == "greater":
        extreme = sum(value >= observed for value in surrogate_statistics)
    elif alternative == "less":
        extreme = sum(value <= observed for value in surrogate_statistics)
    elif alternative == "two-sided":
        extreme = sum(
            abs(value) >= abs(observed)
            for value in surrogate_statistics
        )
    else:
        raise ValueError(
            "alternative must be 'two-sided', 'greater', or 'less'."
        )

    return (extreme + 1) / (total + 1)


def run_surrogate_test(
    values: Sequence[float],
    method: str = "permutation",
    statistic_name: str = "lag1_autocorrelation",
    surrogates: int = 1000,
    seed: int = 42,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """
    Run a Monte Carlo surrogate test.

    For permutation surrogates, the empirical distribution is preserved
    while temporal ordering is randomized.

    For phase-randomized surrogates, the Fourier amplitude spectrum is
    retained while phases are randomized. The empirical distribution
    is generally not preserved.
    """
    start_time = time.perf_counter()

    data = [float(value) for value in values]

    if len(data) < 4:
        raise ValueError("At least 4 observations are required.")

    if not all(math.isfinite(value) for value in data):
        raise ValueError("All observations must be finite.")

    if method not in SURROGATE_METHODS:
        raise ValueError(
            f"Unsupported method '{method}'. "
            f"Choose from {', '.join(SURROGATE_METHODS)}."
        )

    if isinstance(surrogates, bool) or surrogates < 99:
        raise ValueError("surrogates must be an integer of at least 99.")

    if isinstance(surrogates, float) and not surrogates.is_integer():
        raise ValueError("surrogates must be a whole number.")

    surrogates = int(surrogates)

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(
            "alternative must be 'two-sided', 'greater', or 'less'."
        )

    observed = calculate_statistic(data, statistic_name)
    rng = random.Random(seed)
    null_statistics: list[float] = []

    generator = (
        permutation_surrogate
        if method == "permutation"
        else phase_randomized_surrogate
    )

    for _ in range(surrogates):
        surrogate = generator(data, rng)
        null_statistics.append(
            calculate_statistic(surrogate, statistic_name)
        )

    p_value = monte_carlo_p_value(
        observed=observed,
        surrogate_statistics=null_statistics,
        alternative=alternative,
    )

    null_mean = statistics.fmean(null_statistics)
    null_std = (
        statistics.stdev(null_statistics)
        if len(null_statistics) >= 2
        else 0.0
    )

    # This standardized distance is descriptive only. It is not a
    # substitute for the empirical Monte Carlo p-value.
    if null_std > 0:
        standardized_distance = (observed - null_mean) / null_std
    else:
        standardized_distance = None

    return {
        "method": method,
        "statistic": statistic_name,
        "alternative": alternative,
        "sample_size": len(data),
        "surrogate_count": surrogates,
        "random_seed": seed,
        "observed_statistic": observed,
        "null_distribution_mean": null_mean,
        "null_distribution_std": null_std,
        "null_distribution_min": min(null_statistics),
        "null_distribution_max": max(null_statistics),
        "monte_carlo_p_value": p_value,
        "standardized_distance_descriptive_only": standardized_distance,
        "elapsed_ms": round(
            (time.perf_counter() - start_time) * 1000.0,
            3,
        ),
        "surrogate_distribution": null_statistics,
    }


def build_report(
    values: Sequence[float],
    input_metadata: dict[str, Any],
    method: str = "permutation",
    statistic_name: str = "lag1_autocorrelation",
    surrogates: int = 1000,
    seed: int = 42,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Create a structured, machine-readable validation report."""
    result = run_surrogate_test(
        values=values,
        method=method,
        statistic_name=statistic_name,
        surrogates=surrogates,
        seed=seed,
        alternative=alternative,
    )

    # The full null distribution is available in the JSON report.
    # A compact summary is separately exported to CSV.
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
        "test": result,
        "interpretation_boundary": {
            "diagnostic_only": True,
            "scientific_validity_established": False,
            "causality_established": False,
            "medical_diagnosis_performed": False,
            "note": (
                "The p-value is conditional on the selected surrogate "
                "procedure, statistic, and null hypothesis. It is not "
                "the probability that the scientific hypothesis is true."
            ),
        },
    }


def save_report(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Save JSON report, full null distribution CSV, and summary CSV."""
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = target_dir / f"surrogate_report_{stamp}.json"
    summary_csv = target_dir / f"surrogate_summary_{stamp}.csv"
    distribution_csv = target_dir / f"surrogate_distribution_{stamp}.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    test = report["test"]

    summary = {
        "module": report["module"],
        "module_number": report["module_number"],
        "created_at_utc": report["created_at_utc"],
        "input_file": report["input"].get("input_file"),
        "input_sha256": report["input"].get("input_sha256"),
        "column": report["input"].get("column"),
        "method": test["method"],
        "statistic": test["statistic"],
        "alternative": test["alternative"],
        "sample_size": test["sample_size"],
        "surrogate_count": test["surrogate_count"],
        "random_seed": test["random_seed"],
        "observed_statistic": test["observed_statistic"],
        "null_distribution_mean": test["null_distribution_mean"],
        "null_distribution_std": test["null_distribution_std"],
        "monte_carlo_p_value": test["monte_carlo_p_value"],
        "status": report["status"],
    }

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    with distribution_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["surrogate_index", "statistic_value"])

        for index, value in enumerate(
            test["surrogate_distribution"],
            start=1,
        ):
            writer.writerow([index, value])

    return {
        "json_report": str(json_path),
        "summary_csv": str(summary_csv),
        "distribution_csv": str(distribution_csv),
    }


def print_summary(report: dict[str, Any]) -> None:
    test = report["test"]
    metadata = report["input"]

    print("\n" + "=" * 68)
    print("D³ VITAL-X — SURROGATE VALIDATION REPORT")
    print("=" * 68)
    print("Module:", f"{MODULE_NUMBER}. {MODULE_NAME}")
    print("Input column:", metadata.get("column"))
    print("Valid observations:", test["sample_size"])
    print("Method:", test["method"])
    print("Statistic:", test["statistic"])
    print("Alternative:", test["alternative"])
    print("Surrogate count:", test["surrogate_count"])
    print("Random seed:", test["random_seed"])
    print()
    print("Observed statistic:", test["observed_statistic"])
    print("Null mean:", test["null_distribution_mean"])
    print("Null standard deviation:", test["null_distribution_std"])
    print("Monte Carlo p-value:", test["monte_carlo_p_value"])
    print()
    print("Scientific validity established: NO")
    print("Causality established: NO")
    print("Medical diagnosis performed: NO")
    print("=" * 68 + "\n")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare an observed numeric time-series statistic "
            "with a surrogate-data null distribution."
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
        "--method",
        choices=SURROGATE_METHODS,
        default="permutation",
        help="Surrogate method (default: permutation).",
    )
    parser.add_argument(
        "--statistic",
        choices=SUPPORTED_STATISTICS,
        default="lag1_autocorrelation",
        help="Statistic to compare (default: lag1_autocorrelation).",
    )
    parser.add_argument(
        "--surrogates",
        type=int,
        default=1000,
        help="Number of surrogates; minimum 99 (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--alternative",
        choices=("two-sided", "greater", "less"),
        default="two-sided",
        help="Monte Carlo comparison direction.",
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

        report = build_report(
            values=values,
            input_metadata=metadata,
            method=args.method,
            statistic_name=args.statistic,
            surrogates=args.surrogates,
            seed=args.seed,
            alternative=args.alternative,
        )

        if args.output_dir:
            output_dir = Path(args.output_dir).expanduser()
        else:
            # Assumes this file lives in PROJECT_ROOT/08_VALIDATION/.
            project_root = Path(__file__).resolve().parent.parent
            output_dir = project_root / "validation_reports" / "surrogate"

        paths = save_report(report, output_dir)
        print_summary(report)

        for label, path in paths.items():
            print(f"{label}: {path}")

        return 0

    except (OSError, ValueError, csv.Error, RuntimeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
