
# ============================================================
# D³ VITAL-X Space Intelligence Platform
# Module 13 — entropy_variance.py
# ============================================================
#
# ANALYTICS OUTPUT LAYER
#
# Purpose:
#   1. Shannon entropy from a numerical signal
#   2. Population and sample variance
#   3. Basic numerical-input quality diagnostics
#   4. Reproducible configuration and result fingerprint
#
# SECURITY / SCIENTIFIC BOUNDARIES:
#   - No proprietary intelligence algorithms
#   - No UTL/DVDH implementation
#   - No raw-data modification
#   - No dataset download
#   - No synthetic scientific data generation
#   - No medical diagnosis
#   - No automatic physical interpretation
#
# INPUT:
#   A real, one-dimensional numerical signal supplied by
#   an upstream adapter or analysis pipeline.
#
# IMPORTANT:
#   Entropy and variance depend on data preparation,
#   binning, units, sampling, and instrument characteristics.
#   These metrics alone do not establish an anomaly or cause.
#
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Sequence, Tuple
import hashlib
import json
import math

import numpy as np


# ============================================================
# 1. MODULE METADATA
# ============================================================

MODULE_NAME = "entropy_variance"
MODULE_NUMBER = 13
MODULE_VERSION = "1.0.0"

ENTROPY_METHOD = "SHANNON_HISTOGRAM"
DEFAULT_ENTROPY_BINS = 32

DEFAULT_VARIANCE_DDOF = 0

RAW_DATA_MODIFICATION_ALLOWED = False
SYNTHETIC_DATA_GENERATED = False

MEDICAL_DIAGNOSIS_SUPPORTED = False
AUTOMATIC_SCIENTIFIC_CLAIM_SUPPORTED = False


# ============================================================
# 2. RESULT SCHEMA
# ============================================================

@dataclass(frozen=True)
class EntropyVarianceResult:
    """
    Public result contract for numerical signal diagnostics.

    entropy_bits:
        Shannon entropy of the histogram probabilities.

    variance:
        Variance using the selected ddof.

    population_variance:
        Variance calculated with ddof=0.

    sample_variance:
        Variance calculated with ddof=1, or None if fewer
        than two valid samples are available.

    This result is descriptive, not a scientific diagnosis.
    """

    status: str

    sample_count_input: int
    sample_count_valid: int
    sample_count_excluded: int

    entropy_bits: Optional[float]
    entropy_bin_count: Optional[int]
    entropy_method: str

    variance: Optional[float]
    variance_ddof: int
    population_variance: Optional[float]
    sample_variance: Optional[float]

    mean: Optional[float]
    standard_deviation: Optional[float]
    minimum: Optional[float]
    maximum: Optional[float]

    signal_unit: Optional[str]
    variance_unit: Optional[str]

    warnings: Tuple[str, ...]
    errors: Tuple[str, ...]

    configuration: Dict[str, Any]
    result_fingerprint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible result dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize the result without non-standard NaN values."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )


# ============================================================
# 3. FINGERPRINT UTILITIES
# ============================================================

def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fingerprint(payload: Any) -> str:
    """
    Hash a canonical representation of the public result.

    This fingerprint supports integrity checks. It is not a
    digital signature or an independently trusted timestamp.
    """
    encoded = _canonical_json(payload).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


# ============================================================
# 4. VALIDATION HELPERS
# ============================================================

def _empty_result(
    *,
    status: str,
    input_count: int,
    excluded_count: int,
    ddof: int,
    bins: int,
    missing_policy: str,
    signal_unit: Optional[str],
    variance_unit: Optional[str],
    warnings: Sequence[str] = (),
    errors: Sequence[str] = (),
) -> EntropyVarianceResult:

    return EntropyVarianceResult(
        status=status,
        sample_count_input=input_count,
        sample_count_valid=0,
        sample_count_excluded=excluded_count,
        entropy_bits=None,
        entropy_bin_count=None,
        entropy_method=ENTROPY_METHOD,
        variance=None,
        variance_ddof=ddof,
        population_variance=None,
        sample_variance=None,
        mean=None,
        standard_deviation=None,
        minimum=None,
        maximum=None,
        signal_unit=signal_unit,
        variance_unit=variance_unit,
        warnings=tuple(warnings),
        errors=tuple(errors),
        configuration={
            "module": MODULE_NAME,
            "module_version": MODULE_VERSION,
            "entropy_bins": bins,
            "variance_ddof": ddof,
            "missing_policy": missing_policy,
        },
        result_fingerprint=None,
    )


def _validate_unit(
    value: Optional[str],
    field_name: str,
) -> Optional[str]:
    if value is None:
        return None

    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be a string or None."
        )

    cleaned = value.strip()

    if len(cleaned) > 64:
        raise ValueError(
            f"{field_name} must not exceed 64 characters."
        )

    return cleaned or None


# ============================================================
# 5. CORE ANALYSIS FUNCTION
# ============================================================

def calculate_entropy_variance(
    samples: Any,
    *,
    entropy_bins: int = DEFAULT_ENTROPY_BINS,
    variance_ddof: int = DEFAULT_VARIANCE_DDOF,
    missing_policy: str = "reject",
    signal_unit: Optional[str] = None,
    variance_unit: Optional[str] = None,
) -> EntropyVarianceResult:
    """
    Calculate Shannon histogram entropy and variance.

    Parameters
    ----------
    samples:
        A real, one-dimensional numerical signal supplied
        by an upstream data adapter or pipeline.

    entropy_bins:
        Positive integer number of histogram bins.
        Entropy is sensitive to this choice.

    variance_ddof:
        0 = population variance.
        1 = sample variance.

    missing_policy:
        "reject" rejects the signal if NaN/Inf values occur.
        "omit" excludes non-finite values and reports the count.

    signal_unit:
        Optional unit label for the input signal.

    variance_unit:
        Optional explicit unit for variance. If omitted and
        signal_unit is provided, the output uses "<unit>^2".

    Returns
    -------
    EntropyVarianceResult

    Notes
    -----
    - The input must be one-dimensional.
    - This function does not download data.
    - It does not modify the supplied source array.
    - It does not create synthetic scientific observations.
    - It does not infer physical meaning from the metrics.
    """

    # --------------------------------------------------------
    # Parameter validation
    # --------------------------------------------------------

    if (
        isinstance(entropy_bins, bool)
        or not isinstance(entropy_bins, (int, np.integer))
        or entropy_bins < 1
    ):
        raise ValueError(
            "entropy_bins must be a positive integer."
        )

    if (
        isinstance(variance_ddof, bool)
        or variance_ddof not in (0, 1)
    ):
        raise ValueError(
            "variance_ddof must be 0 or 1."
        )

    if missing_policy not in ("reject", "omit"):
        raise ValueError(
            "missing_policy must be 'reject' or 'omit'."
        )

    signal_unit = _validate_unit(
        signal_unit,
        "signal_unit",
    )

    variance_unit = _validate_unit(
        variance_unit,
        "variance_unit",
    )

    if variance_unit is None and signal_unit:
        variance_unit = f"{signal_unit}^2"

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if samples is None:
        return _empty_result(
            status="REJECTED",
            input_count=0,
            excluded_count=0,
            ddof=variance_ddof,
            bins=entropy_bins,
            missing_policy=missing_policy,
            signal_unit=signal_unit,
            variance_unit=variance_unit,
            errors=("Input signal is None.",),
        )

    try:
        values = np.asarray(samples, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return _empty_result(
            status="REJECTED",
            input_count=0,
            excluded_count=0,
            ddof=variance_ddof,
            bins=entropy_bins,
            missing_policy=missing_policy,
            signal_unit=signal_unit,
            variance_unit=variance_unit,
            errors=(
                "Input must contain convertible numerical values.",
            ),
        )

    if values.ndim != 1:
        return _empty_result(
            status="REJECTED",
            input_count=int(values.size),
            excluded_count=0,
            ddof=variance_ddof,
            bins=entropy_bins,
            missing_policy=missing_policy,
            signal_unit=signal_unit,
            variance_unit=variance_unit,
            errors=(
                "Input must be one-dimensional. "
                "Flattening images or multidimensional arrays "
                "is not performed automatically.",
            ),
        )

    input_count = int(values.size)

    if input_count == 0:
        return _empty_result(
            status="REJECTED",
            input_count=0,
            excluded_count=0,
            ddof=variance_ddof,
            bins=entropy_bins,
            missing_policy=missing_policy,
            signal_unit=signal_unit,
            variance_unit=variance_unit,
            errors=("Input signal is empty.",),
        )

    finite_mask = np.isfinite(values)

    excluded_count = int(
        input_count - np.count_nonzero(finite_mask)
    )

    # Reject non-finite values unless omission was requested.
    if excluded_count and missing_policy == "reject":
        return _empty_result(
            status="REJECTED",
            input_count=input_count,
            excluded_count=excluded_count,
            ddof=variance_ddof,
            bins=entropy_bins,
            missing_policy=missing_policy,
            signal_unit=signal_unit,
            variance_unit=variance_unit,
            errors=(
                "Non-finite values detected. "
                "Use missing_policy='omit' only when exclusion "
                "is scientifically justified and documented.",
            ),
        )

    # Use a new finite array; do not modify the input.
    valid_values = values[finite_mask].copy()

    valid_count = int(valid_values.size)

    warnings = []

    if excluded_count:
        warnings.append(
            f"Excluded {excluded_count} non-finite sample(s)."
        )

    if valid_count < input_count:
        warnings.append(
            "Metrics describe only the retained finite samples."
        )

    if valid_count < 2:
        return _empty_result(
            status="INSUFFICIENT_DATA",
            input_count=input_count,
            excluded_count=excluded_count,
            ddof=variance_ddof,
            bins=entropy_bins,
            missing_policy=missing_policy,
            signal_unit=signal_unit,
            variance_unit=variance_unit,
            warnings=warnings,
            errors=(
                "At least two finite samples are required "
                "for this analysis.",
            ),
        )

    # --------------------------------------------------------
    # Numerical statistics
    # --------------------------------------------------------

    try:
        mean_value = float(np.mean(valid_values))

        population_variance = float(
            np.var(valid_values, ddof=0)
        )

        sample_variance = float(
            np.var(valid_values, ddof=1)
        )

        selected_variance = float(
            np.var(valid_values, ddof=variance_ddof)
        )

        standard_deviation = float(
            np.sqrt(population_variance)
        )

        minimum = float(np.min(valid_values))
        maximum = float(np.max(valid_values))

    except (FloatingPointError, OverflowError, ValueError):
        return _empty_result(
            status="NUMERICAL_ERROR",
            input_count=input_count,
            excluded_count=excluded_count,
            ddof=variance_ddof,
            bins=entropy_bins,
            missing_policy=missing_policy,
            signal_unit=signal_unit,
            variance_unit=variance_unit,
            warnings=warnings,
            errors=(
                "Numerical statistics could not be computed.",
            ),
        )

    numeric_values = [
        mean_value,
        population_variance,
        sample_variance,
        selected_variance,
        standard_deviation,
        minimum,
        maximum,
    ]

    if not all(math.isfinite(v) for v in numeric_values):
        return _empty_result(
            status="NUMERICAL_ERROR",
            input_count=input_count,
            excluded_count=excluded_count,
            ddof=variance_ddof,
            bins=entropy_bins,
            missing_policy=missing_policy,
            signal_unit=signal_unit,
            variance_unit=variance_unit,
            warnings=warnings,
            errors=(
                "Non-finite statistics resulted from numerical overflow.",
            ),
        )

    # --------------------------------------------------------
    # Shannon entropy using a normalized histogram
    # --------------------------------------------------------

    try:
        counts, _ = np.histogram(
            valid_values,
            bins=int(entropy_bins),
        )

        occupied_counts = counts[counts > 0]

        probabilities = (
            occupied_counts.astype(np.float64)
            / float(valid_count)
        )

        entropy_bits = float(
            -np.sum(
                probabilities * np.log2(probabilities)
            )
        )

    except (FloatingPointError, OverflowError, ValueError):
        return _empty_result(
            status="NUMERICAL_ERROR",
            input_count=input_count,
            excluded_count=excluded_count,
            ddof=variance_ddof,
            bins=entropy_bins,
            missing_policy=missing_policy,
            signal_unit=signal_unit,
            variance_unit=variance_unit,
            warnings=warnings,
            errors=(
                "Histogram entropy could not be computed.",
            ),
        )

    if not math.isfinite(entropy_bits):
        return _empty_result(
            status="NUMERICAL_ERROR",
            input_count=input_count,
            excluded_count=excluded_count,
            ddof=variance_ddof,
            bins=entropy_bins,
            missing_policy=missing_policy,
            signal_unit=signal_unit,
            variance_unit=variance_unit,
            warnings=warnings,
            errors=("Entropy result is non-finite.",),
        )

    if np.all(valid_values == valid_values[0]):
        warnings.append(
            "Constant signal detected; histogram entropy and "
            "variance are zero for this input."
        )

    if valid_count < 10:
        warnings.append(
            "Small sample count: entropy estimates may be sensitive "
            "to binning and sample size."
        )

    warnings.append(
        "Entropy is histogram-bin dependent; record the bin count "
        "when comparing results."
    )

    warnings.append(
        "Descriptive metrics alone do not establish an anomaly, "
        "physical mechanism, or clinical finding."
    )

    # --------------------------------------------------------
    # Configuration and result fingerprint
    # --------------------------------------------------------

    configuration = {
        "module": MODULE_NAME,
        "module_number": MODULE_NUMBER,
        "module_version": MODULE_VERSION,
        "entropy_method": ENTROPY_METHOD,
        "entropy_bins": int(entropy_bins),
        "variance_ddof": int(variance_ddof),
        "missing_policy": missing_policy,
        "signal_unit": signal_unit,
        "variance_unit": variance_unit,
        "valid_sample_count": valid_count,
        "excluded_sample_count": excluded_count,
    }

    public_result = {
        "status": "OK",
        "sample_count_input": input_count,
        "sample_count_valid": valid_count,
        "sample_count_excluded": excluded_count,
        "entropy_bits": entropy_bits,
        "entropy_bin_count": int(entropy_bins),
        "entropy_method": ENTROPY_METHOD,
        "variance": selected_variance,
        "variance_ddof": int(variance_ddof),
        "population_variance": population_variance,
        "sample_variance": sample_variance,
        "mean": mean_value,
        "standard_deviation": standard_deviation,
        "minimum": minimum,
        "maximum": maximum,
        "signal_unit": signal_unit,
        "variance_unit": variance_unit,
        "warnings": warnings,
        "errors": [],
        "configuration": configuration,
    }

    result_hash = _fingerprint(public_result)

    return EntropyVarianceResult(
        status="OK",
        sample_count_input=input_count,
        sample_count_valid=valid_count,
        sample_count_excluded=excluded_count,
        entropy_bits=entropy_bits,
        entropy_bin_count=int(entropy_bins),
        entropy_method=ENTROPY_METHOD,
        variance=selected_variance,
        variance_ddof=int(variance_ddof),
        population_variance=population_variance,
        sample_variance=sample_variance,
        mean=mean_value,
        standard_deviation=standard_deviation,
        minimum=minimum,
        maximum=maximum,
        signal_unit=signal_unit,
        variance_unit=variance_unit,
        warnings=tuple(warnings),
        errors=(),
        configuration=configuration,
        result_fingerprint=result_hash,
    )


# ============================================================
# 6. PUBLIC CONTRACT
# ============================================================

def entropy_variance_contract() -> Dict[str, Any]:
    """Return the public contract for integration and review."""

    return {
        "module": MODULE_NAME,
        "module_number": MODULE_NUMBER,
        "module_version": MODULE_VERSION,
        "purpose": (
            "Descriptive Shannon histogram entropy and "
            "variance calculation for one-dimensional signals."
        ),
        "input": "One-dimensional numerical samples",
        "entropy_method": ENTROPY_METHOD,
        "default_entropy_bins": DEFAULT_ENTROPY_BINS,
        "default_variance_ddof": DEFAULT_VARIANCE_DDOF,
        "raw_data_modified": False,
        "synthetic_data_generated": False,
        "dataset_download_performed": False,
        "proprietary_engine_required": False,
        "medical_diagnosis_supported": False,
        "automatic_scientific_claim_supported": False,
    }


# ============================================================
# 7. CONTRACT-ONLY SELF-TEST
# ============================================================

def run_contract_tests() -> Dict[str, bool]:
    """
    Software-contract tests only.

    No fabricated scientific dataset is created.
    Real-data tests should use samples from a verified
    upstream dataset and retain its provenance.
    """

    results: Dict[str, bool] = {}

    # Empty-input validation: no scientific observations.
    empty_result = calculate_entropy_variance([])

    results["empty_input_rejected"] = (
        empty_result.status == "REJECTED"
    )

    # Multidimensional-input validation: no flattening.
    multidimensional_result = calculate_entropy_variance(
        np.empty((0, 0))
    )

    results["multidimensional_input_rejected"] = (
        multidimensional_result.status == "REJECTED"
    )

    # Parameter-validation tests.
    try:
        calculate_entropy_variance(
            [],
            entropy_bins=0,
        )
        results["invalid_bin_count_rejected"] = False
    except ValueError:
        results["invalid_bin_count_rejected"] = True

    try:
        calculate_entropy_variance(
            [],
            variance_ddof=2,
        )
        results["invalid_ddof_rejected"] = False
    except ValueError:
        results["invalid_ddof_rejected"] = True

    try:
        calculate_entropy_variance(
            [],
            missing_policy="unknown",
        )
        results["invalid_missing_policy_rejected"] = False
    except ValueError:
        results["invalid_missing_policy_rejected"] = True

    contract = entropy_variance_contract()

    results["no_synthetic_data_generation"] = (
        contract["synthetic_data_generated"] is False
    )

    results["no_raw_data_modification"] = (
        contract["raw_data_modified"] is False
    )

    results["medical_diagnosis_disabled"] = (
        contract["medical_diagnosis_supported"] is False
    )

    results["all_checks_passed"] = all(results.values())

    return results


# ============================================================
# 8. MODULE INFORMATION
# ============================================================

def module_info() -> Dict[str, Any]:
    return entropy_variance_contract()


# ============================================================
# 9. DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=" * 64)
    print("D³ VITAL-X — Module 13: entropy_variance.py")
    print("=" * 64)

    results = run_contract_tests()

    for name, passed in results.items():
        print(f"{name}: {passed}")

    print("-" * 64)

    if results["all_checks_passed"]:
        print("CONTRACT TESTS: PASS")
    else:
        print("CONTRACT TESTS: REVIEW REQUIRED")

    print(
        "No scientific dataset was generated or analyzed "
        "during these contract tests."
    )
    print("=" * 64)
