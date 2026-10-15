
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
D³ VITAL-X Space Intelligence Platform
Module 28: reproducibility.py

Version: 1.0.0
Report schema: 1.0

Purpose
-------
Create a reproducibility audit report for a numeric CSV signal.

The module records input provenance, software environment, configuration,
SHA-256 hashes, repeated deterministic calculations, and seeded diagnostic
sampling checks.

Scientific boundary
-------------------
This module does not establish scientific validity, causal relationships,
physical explanations, anomaly authenticity, or medical diagnoses.

A successful repeatability check is not equivalent to independent
scientific replication.

Design principles
-----------------
- Python standard library only.
- Read-only input handling.
- Explicit random seed.
- Deterministic calculations for identical valid input.
- JSON and CSV reports.
- No automatic scientific or medical claims.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


MODULE_NAME = "reproducibility"
MODULE_VERSION = "1.0.0"
REPORT_SCHEMA_VERSION = "1.0"

DEFAULT_SEED = 20261015
DEFAULT_REPEATS = 5
DEFAULT_ENTROPY_BINS = 16


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def utc_now() -> str:
    """Return a UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(file_path: Path) -> str:
    """Calculate SHA-256 for a file without modifying it."""
    digest = hashlib.sha256()

    with Path(file_path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Calculate SHA-256 for a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_positive_integer(value: int, name: str) -> int:
    """Validate a positive integer; reject booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def validate_signal(
    values: Sequence[float],
    min_samples: int = 2,
) -> List[float]:
    """Return a validated finite numeric copy of a signal."""
    if isinstance(values, (str, bytes)):
        raise TypeError("Signal must be a numeric sequence.")

    try:
        signal = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Signal contains a non-numeric value.") from exc

    if len(signal) < min_samples:
        raise ValueError(
            f"At least {min_samples} valid samples are required."
        )

    if not all(math.isfinite(value) for value in signal):
        raise ValueError("Signal contains NaN or infinite values.")

    return signal


def json_fingerprint(data: Any) -> str:
    """
    Hash a JSON-compatible object using stable serialization.

    This provides a stable fingerprint for the serialized object, not
    a guarantee that all possible equivalent objects share one hash.
    """
    serialized = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return sha256_text(serialized)


# ---------------------------------------------------------------------
# Descriptive metrics
# ---------------------------------------------------------------------

def calculate_entropy_bits(
    values: Sequence[float],
    bins: int = DEFAULT_ENTROPY_BINS,
) -> float:
    """
    Estimate Shannon entropy from an equal-width histogram.

    This is a histogram-based estimate in bits, not differential entropy.
    """
    signal = validate_signal(values, min_samples=1)
    validate_positive_integer(bins, "bins")

    minimum = min(signal)
    maximum = max(signal)

    if minimum == maximum:
        return 0.0

    bin_width = (maximum - minimum) / bins
    counts = [0] * bins

    for value in signal:
        index = int((value - minimum) / bin_width)
        index = min(max(index, 0), bins - 1)
        counts[index] += 1

    total = len(signal)
    entropy = 0.0

    for count in counts:
        if count == 0:
            continue

        probability = count / total
        entropy -= probability * math.log2(probability)

    return float(entropy)


def calculate_metrics(
    values: Sequence[float],
    entropy_bins: int = DEFAULT_ENTROPY_BINS,
) -> Dict[str, Optional[float]]:
    """Calculate deterministic descriptive metrics."""
    signal = validate_signal(values, min_samples=2)

    mean_value = statistics.mean(signal)
    variance_value = statistics.variance(signal)
    standard_deviation = math.sqrt(variance_value)

    return {
        "sample_count": len(signal),
        "mean": float(mean_value),
        "variance": float(variance_value),
        "standard_deviation": float(standard_deviation),
        "minimum": float(min(signal)),
        "maximum": float(max(signal)),
        "shannon_entropy_bits": calculate_entropy_bits(
            signal,
            bins=entropy_bins,
        ),
    }


# ---------------------------------------------------------------------
# CSV input
# ---------------------------------------------------------------------

def load_numeric_csv(
    csv_path: Path,
    column: str,
) -> Tuple[List[float], Dict[str, Any]]:
    """
    Read a numeric CSV column and collect provenance metadata.

    Blank, non-numeric, and non-finite cells are skipped and counted.
    The original input file is never written to.
    """
    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    if not column or not column.strip():
        raise ValueError("A non-empty column name is required.")

    initial_hash = sha256_file(csv_path)

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
            raise ValueError(
                f"Column '{column}' not found. "
                f"Available columns: {reader.fieldnames or []}"
            )

        for row in reader:
            rows_seen += 1
            raw_value = row.get(column)

            if raw_value is None or not raw_value.strip():
                blank_cells += 1
                continue

            try:
                value = float(raw_value)
            except (ValueError, TypeError, OverflowError):
                non_numeric_cells += 1
                continue

            if not math.isfinite(value):
                non_finite_cells += 1
                continue

            values.append(value)

    if len(values) < 2:
        raise ValueError(
            "The selected column must contain at least two "
            "valid finite numeric samples."
        )

    final_hash = sha256_file(csv_path)

    metadata = {
        "input_file_name": csv_path.name,
        "input_file_sha256_before_analysis": initial_hash,
        "input_file_sha256_after_loading": final_hash,
        "input_hash_unchanged_during_loading": (
            initial_hash == final_hash
        ),
        "selected_column": column,
        "rows_seen": rows_seen,
        "valid_numeric_samples": len(values),
        "blank_cells_skipped": blank_cells,
        "non_numeric_cells_skipped": non_numeric_cells,
        "non_finite_cells_skipped": non_finite_cells,
        "source_modified_by_module": False,
    }

    if initial_hash != final_hash:
        raise RuntimeError(
            "Input file hash changed while loading. "
            "Check whether another process modified the file."
        )

    return values, metadata


# ---------------------------------------------------------------------
# Repeatability and seeded sampling checks
# ---------------------------------------------------------------------

def run_repeatability_check(
    values: Sequence[float],
    repeats: int = DEFAULT_REPEATS,
    seed: int = DEFAULT_SEED,
    entropy_bins: int = DEFAULT_ENTROPY_BINS,
) -> Dict[str, Any]:
    """
    Repeat deterministic calculations and compare their fingerprints.

    A local random.Random instance is used for a seeded permutation
    check. It does not alter Python's global random state.
    """
    signal = validate_signal(values, min_samples=2)
    validate_positive_integer(repeats, "repeats")
    validate_positive_integer(entropy_bins, "entropy_bins")

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")

    repeated_results: List[Dict[str, Optional[float]]] = []
    repeated_fingerprints: List[str] = []

    for _ in range(repeats):
        metrics = calculate_metrics(
            signal,
            entropy_bins=entropy_bins,
        )
        repeated_results.append(metrics)
        repeated_fingerprints.append(json_fingerprint(metrics))

    first_fingerprint = repeated_fingerprints[0]
    identical_results = all(
        fingerprint == first_fingerprint
        for fingerprint in repeated_fingerprints
    )

    # Seeded sampling is isolated from global random state.
    rng_a = random.Random(seed)
    rng_b = random.Random(seed)

    permutation_a = list(signal)
    permutation_b = list(signal)

    rng_a.shuffle(permutation_a)
    rng_b.shuffle(permutation_b)

    seeded_permutation_repeatable = permutation_a == permutation_b

    original_fingerprint = json_fingerprint(signal)

    return {
        "repeats_requested": repeats,
        "repeats_completed": len(repeated_results),
        "random_seed": seed,
        "entropy_bins": entropy_bins,
        "repeated_metrics": repeated_results,
        "result_fingerprints_sha256": repeated_fingerprints,
        "all_deterministic_results_identical": identical_results,
        "seeded_permutation_repeatable": seeded_permutation_repeatable,
        "input_sequence_fingerprint_sha256": original_fingerprint,
        "repeatability_status": (
            "PASS"
            if identical_results and seeded_permutation_repeatable
            else "FAIL"
        ),
        "interpretation": (
            "PASS indicates repeatability within this software run and "
            "configuration. It does not establish independent replication."
        ),
    }


# ---------------------------------------------------------------------
# Report construction and output
# ---------------------------------------------------------------------

def build_report(
    repeatability_result: Dict[str, Any],
    input_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the full reproducibility audit report."""
    report = {
        "module": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "created_utc": utc_now(),
        "runtime_environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "input_provenance": input_metadata,
        "repeatability": repeatability_result,
        "audit_summary": {
            "input_hash_unchanged_during_loading": input_metadata.get(
                "input_hash_unchanged_during_loading"
            ),
            "deterministic_results_identical": repeatability_result[
                "all_deterministic_results_identical"
            ],
            "seeded_permutation_repeatable": repeatability_result[
                "seeded_permutation_repeatable"
            ],
            "repeatability_passed": (
                repeatability_result["repeatability_status"] == "PASS"
            ),
        },
        "scientific_boundary": {
            "scientific_validity_established": False,
            "independent_replication_established": False,
            "causality_established": False,
            "anomaly_confirmed": False,
            "medical_diagnosis": False,
            "human_review_required": True,
        },
    }

    report["report_content_sha256"] = json_fingerprint(report)
    return report


def save_report(
    report: Dict[str, Any],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Write JSON report and a CSV summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "reproducibility_report.json"
    csv_path = output_dir / "reproducibility_summary.csv"

    with json_path.open("w", encoding="utf-8") as file_obj:
        json.dump(
            report,
            file_obj,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        file_obj.write("\n")

    repeatability = report["repeatability"]
    first_result = (
        repeatability["repeated_metrics"][0]
        if repeatability["repeated_metrics"]
        else {}
    )

    rows = [
        ("module", report["module"]),
        ("module_version", report["module_version"]),
        ("input_file_name", report["input_provenance"].get(
            "input_file_name", ""
        )),
        ("input_sha256", report["input_provenance"].get(
            "input_file_sha256_before_analysis", ""
        )),
        ("selected_column", report["input_provenance"].get(
            "selected_column", ""
        )),
        ("valid_numeric_samples", report["input_provenance"].get(
            "valid_numeric_samples", ""
        )),
        ("repeats_requested", repeatability["repeats_requested"]),
        ("random_seed", repeatability["random_seed"]),
        ("entropy_bins", repeatability["entropy_bins"]),
        ("repeatability_status", repeatability["repeatability_status"]),
        (
            "all_deterministic_results_identical",
            repeatability["all_deterministic_results_identical"],
        ),
        (
            "seeded_permutation_repeatable",
            repeatability["seeded_permutation_repeatable"],
        ),
        (
            "scientific_validity_established",
            report["scientific_boundary"]["scientific_validity_established"],
        ),
        (
            "independent_replication_established",
            report["scientific_boundary"][
                "independent_replication_established"
            ],
        ),
    ]

    for metric_name, value in first_result.items():
        rows.append((f"metric_{metric_name}", value))

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["field", "value"])
        writer.writerows(rows)

    return json_path, csv_path


def print_summary(report: Dict[str, Any]) -> None:
    """Print a concise human-readable summary."""
    repeatability = report["repeatability"]
    provenance = report["input_provenance"]

    print("=" * 68)
    print("D³ VITAL-X — REPRODUCIBILITY AUDIT")
    print("=" * 68)
    print(f"Module version     : {report['module_version']}")
    print(f"Created UTC        : {report['created_utc']}")
    print(f"Input file         : {provenance.get('input_file_name', 'N/A')}")
    print(f"Selected column    : {provenance.get('selected_column', 'N/A')}")
    print(f"Valid samples      : {provenance.get('valid_numeric_samples', 'N/A')}")
    print(f"Input SHA-256      : {provenance.get('input_file_sha256_before_analysis', 'N/A')}")
    print(f"Repeat count       : {repeatability['repeats_completed']}")
    print(f"Random seed        : {repeatability['random_seed']}")
    print(
        "Deterministic run  : "
        f"{repeatability['all_deterministic_results_identical']}"
    )
    print(
        "Seeded permutation : "
        f"{repeatability['seeded_permutation_repeatable']}"
    )
    print(f"Status             : {repeatability['repeatability_status']}")
    print("-" * 68)
    print("Scientific validity established : NO")
    print("Independent replication         : NOT ESTABLISHED")
    print("Causality established           : NO")
    print("Medical diagnosis               : NO")
    print("=" * 68)


# ---------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------

def self_check() -> Dict[str, Any]:
    """Run deterministic synthetic checks."""
    checks: List[Dict[str, Any]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({
            "name": name,
            "passed": bool(condition),
            "detail": detail,
        })
        if not condition:
            raise AssertionError(
                f"Self-check failed: {name} — {detail}"
            )

    signal = [1.0, 2.5, 4.0, 3.0, 6.5, 8.0, 5.5]
    original_copy = list(signal)

    result_a = run_repeatability_check(
        signal,
        repeats=5,
        seed=12345,
        entropy_bins=4,
    )
    result_b = run_repeatability_check(
        signal,
        repeats=5,
        seed=12345,
        entropy_bins=4,
    )

    check(
        "input_not_modified",
        signal == original_copy,
        "Input sequence remains unchanged.",
    )

    check(
        "repeated_metrics_identical",
        result_a["all_deterministic_results_identical"],
        "Repeated calculations produce identical fingerprints.",
    )

    check(
        "seeded_permutation_repeatable",
        result_a["seeded_permutation_repeatable"],
        "The same seed produces the same permutation in this runtime.",
    )

    check(
        "same_configuration_same_fingerprint",
        result_a["result_fingerprints_sha256"]
        == result_b["result_fingerprints_sha256"],
        "Identical input and configuration reproduce the same fingerprints.",
    )

    check(
        "repeat_count_correct",
        result_a["repeats_completed"] == 5,
        "All requested calculation repetitions completed.",
    )

    constant_metrics = calculate_metrics([7.0, 7.0, 7.0, 7.0])

    check(
        "constant_signal_variance_zero",
        math.isclose(
            constant_metrics["variance"],
            0.0,
            abs_tol=1e-12,
        ),
        "Constant signal has zero sample variance.",
    )

    check(
        "constant_signal_entropy_zero",
        math.isclose(
            constant_metrics["shannon_entropy_bits"],
            0.0,
            abs_tol=1e-12,
        ),
        "Constant signal has zero histogram entropy by this estimator.",
    )

    invalid_signal_rejected = False
    try:
        calculate_metrics([1.0, float("nan"), 3.0])
    except ValueError:
        invalid_signal_rejected = True

    check(
        "non_finite_values_rejected",
        invalid_signal_rejected,
        "NaN and infinite signal values are rejected.",
    )

    invalid_seed_rejected = False
    try:
        run_repeatability_check(signal, seed=True)
    except ValueError:
        invalid_seed_rejected = True

    check(
        "boolean_seed_rejected",
        invalid_seed_rejected,
        "Boolean values are not accepted as random seeds.",
    )

    strict_json = json.dumps(result_a, allow_nan=False)

    check(
        "strict_json_serialization",
        isinstance(strict_json, str),
        "Repeatability results serialize as strict JSON.",
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
            "independent_replication_established": False,
            "medical_diagnosis": False,
        },
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a repeatability and provenance report for a numeric "
            "CSV signal. Diagnostic only."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input CSV file. If omitted, self-check is run.",
    )
    parser.add_argument(
        "--column",
        type=str,
        default=None,
        help="Numeric column name in the CSV.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed for deterministic sampling (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Number of repeated calculations (default: {DEFAULT_REPEATS}).",
    )
    parser.add_argument(
        "--entropy-bins",
        type=int,
        default=DEFAULT_ENTROPY_BINS,
        help=f"Histogram bins (default: {DEFAULT_ENTROPY_BINS}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation_reports") / "reproducibility",
        help="Directory for generated reports.",
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
        validate_positive_integer(args.repeats, "repeats")
        validate_positive_integer(args.entropy_bins, "entropy_bins")

        if isinstance(args.seed, bool):
            raise ValueError("seed must be an integer.")

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

        repeatability_result = run_repeatability_check(
            values,
            repeats=args.repeats,
            seed=args.seed,
            entropy_bins=args.entropy_bins,
        )

        report = build_report(
            repeatability_result,
            input_metadata,
        )

        json_path, csv_path = save_report(
            report,
            args.output_dir,
        )

        print_summary(report)
        print(f"JSON report : {json_path.resolve()}")
        print(f"CSV report  : {csv_path.resolve()}")

        if repeatability_result["repeatability_status"] != "PASS":
            return 1

        return 0

    except (OSError, ValueError, TypeError, AssertionError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
