
"""
D³ VITAL-X — Anomaly Candidate Evaluator
Module 15: anomaly_evaluator.py

Purpose
-------
Evaluate supplied numerical metrics against an explicitly provided baseline.
This module produces descriptive anomaly-candidate scores only.

Scientific safeguards
---------------------
- No synthetic scientific observations are generated.
- No baseline is silently inferred from the evaluation sample.
- A score is not proof of an anomaly, transition, cause, or discovery.
- This module does not perform medical diagnosis.
- The protected intelligence core is not implemented or exposed here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional
import hashlib
import json
import math


MODULE_NAME = "anomaly_evaluator"
MODULE_NUMBER = 15
MODULE_VERSION = "1.0.0"


@dataclass
class MetricEvaluation:
    """Evaluation details for one named metric."""

    metric_name: str
    observed_value: Optional[float]
    baseline_mean: Optional[float]
    baseline_std: Optional[float]
    z_score: Optional[float]
    absolute_z_score: Optional[float]
    weight: float
    status: str
    explanation: str


@dataclass
class AnomalyEvaluationResult:
    """JSON-serializable output of the candidate evaluator."""

    module: str
    module_version: str
    status: str
    candidate_flag: Optional[bool]
    threshold: float
    aggregate_score: Optional[float]
    maximum_absolute_z: Optional[float]
    evaluated_metric_count: int
    skipped_metric_count: int
    metrics: list[MetricEvaluation]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    interpretation: str = (
        "A threshold flag identifies a statistical candidate under the "
        "provided baseline and configuration; it does not confirm a real "
        "anomaly, causal mechanism, physical event, or clinical condition."
    )
    result_fingerprint: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )


def _finite_number(value: Any, name: str) -> float:
    """Validate a finite numeric input; reject booleans."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not boolean.")

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc

    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")

    return number


def evaluate_anomaly_candidates(
    observed_metrics: Mapping[str, Any],
    baseline: Mapping[str, Mapping[str, Any]],
    *,
    weights: Optional[Mapping[str, Any]] = None,
    threshold: float = 3.0,
    min_evaluated_metrics: int = 1,
) -> AnomalyEvaluationResult:
    """
    Compare observed metrics with an explicitly supplied baseline.

    Parameters
    ----------
    observed_metrics:
        Mapping of metric names to observed numeric values.

    baseline:
        Mapping in this form:
        {
            "metric_name": {
                "mean": baseline_mean,
                "std": baseline_standard_deviation
            }
        }

        Baseline statistics must come from a documented reference dataset
        or other explicitly justified source. Do not silently estimate them
        from the observation being evaluated.

    weights:
        Optional non-negative metric weights. Weights are normalized over
        the metrics successfully evaluated. Unspecified weights default to 1.

    threshold:
        Candidate threshold applied to the maximum absolute z-score.
        The default 3.0 is a configurable screening convention, not a
        universal scientific significance threshold.

    min_evaluated_metrics:
        Minimum number of valid metrics required to produce a candidate flag.

    Scoring
    -------
    z = (observed_value - baseline_mean) / baseline_std

    aggregate_score = weighted RMS of valid z-scores.

    candidate_flag = True when the maximum absolute z-score meets or
    exceeds threshold, provided enough metrics were evaluated.

    Important
    ---------
    The maximum-z rule is a screening rule. It is not a multiple-testing
    correction, a calibrated probability, or a formal hypothesis test.
    """

    if not isinstance(observed_metrics, Mapping) or not observed_metrics:
        raise ValueError("observed_metrics must be a non-empty mapping.")

    if not isinstance(baseline, Mapping) or not baseline:
        raise ValueError("baseline must be a non-empty mapping.")

    threshold_value = _finite_number(threshold, "threshold")
    if threshold_value <= 0:
        raise ValueError("threshold must be greater than zero.")

    if (
        isinstance(min_evaluated_metrics, bool)
        or not isinstance(min_evaluated_metrics, int)
        or min_evaluated_metrics < 1
    ):
        raise ValueError("min_evaluated_metrics must be a positive integer.")

    if weights is not None and not isinstance(weights, Mapping):
        raise ValueError("weights must be a mapping when provided.")

    warnings: list[str] = []
    evaluations: list[MetricEvaluation] = []
    usable: list[tuple[float, float]] = []
    skipped_count = 0

    unknown_baseline_keys = sorted(
        str(key) for key in observed_metrics if key not in baseline
    )
    if unknown_baseline_keys:
        warnings.append(
            "Metrics without baseline entries were skipped: "
            + ", ".join(unknown_baseline_keys)
        )

    unused_baseline_keys = sorted(
        str(key) for key in baseline if key not in observed_metrics
    )
    if unused_baseline_keys:
        warnings.append(
            "Baseline entries not present in observed_metrics were unused: "
            + ", ".join(unused_baseline_keys)
        )

    for name, raw_observed in observed_metrics.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Every metric name must be a non-empty string.")

        if name not in baseline:
            skipped_count += 1
            evaluations.append(
                MetricEvaluation(
                    metric_name=name,
                    observed_value=None,
                    baseline_mean=None,
                    baseline_std=None,
                    z_score=None,
                    absolute_z_score=None,
                    weight=0.0,
                    status="missing_baseline",
                    explanation="No baseline statistics were supplied.",
                )
            )
            continue

        baseline_entry = baseline[name]
        if not isinstance(baseline_entry, Mapping):
            raise ValueError(
                f"Baseline for '{name}' must contain 'mean' and 'std'."
            )

        if "mean" not in baseline_entry or "std" not in baseline_entry:
            raise ValueError(
                f"Baseline for '{name}' must contain 'mean' and 'std'."
            )

        observed = _finite_number(raw_observed, f"observed_metrics[{name}]")
        mean = _finite_number(
            baseline_entry["mean"], f"baseline[{name}]['mean']"
        )
        std = _finite_number(
            baseline_entry["std"], f"baseline[{name}]['std']"
        )

        if std < 0:
            raise ValueError(
                f"Baseline standard deviation for '{name}' cannot be negative."
            )

        raw_weight = 1.0 if weights is None else weights.get(name, 1.0)
        weight = _finite_number(raw_weight, f"weights[{name}]")

        if weight < 0:
            raise ValueError(f"Weight for '{name}' cannot be negative.")

        if std == 0:
            skipped_count += 1
            warnings.append(
                f"Metric '{name}' was skipped because its baseline "
                "standard deviation is zero; no z-score is defined."
            )
            evaluations.append(
                MetricEvaluation(
                    metric_name=name,
                    observed_value=observed,
                    baseline_mean=mean,
                    baseline_std=std,
                    z_score=None,
                    absolute_z_score=None,
                    weight=weight,
                    status="zero_baseline_variance",
                    explanation=(
                        "A standardized deviation cannot be computed from "
                        "a zero-standard-deviation baseline."
                    ),
                )
            )
            continue

        z_score = (observed - mean) / std
        if not math.isfinite(z_score):
            raise ValueError(
                f"Computed z-score for '{name}' is not finite; "
                "check the scale of the inputs."
            )

        absolute_z = abs(z_score)

        evaluations.append(
            MetricEvaluation(
                metric_name=name,
                observed_value=observed,
                baseline_mean=mean,
                baseline_std=std,
                z_score=z_score,
                absolute_z_score=absolute_z,
                weight=weight,
                status="evaluated",
                explanation=(
                    "Standardized deviation from the supplied baseline; "
                    "not a calibrated probability."
                ),
            )
        )

        usable.append((z_score, weight))

    positive_weight_items = [
        (z, weight) for z, weight in usable if weight > 0
    ]

    evaluated_count = len(usable)

    if evaluated_count < min_evaluated_metrics:
        warnings.append(
            "Too few metrics could be evaluated to produce a candidate flag."
        )

    if not positive_weight_items:
        warnings.append(
            "No positive-weight metrics with valid baseline variance "
            "were available for the aggregate score."
        )
        aggregate_score = None
    else:
        total_weight = sum(weight for _, weight in positive_weight_items)
        weighted_square_sum = sum(
            weight * z * z for z, weight in positive_weight_items
        )
        aggregate_score = math.sqrt(weighted_square_sum / total_weight)

    max_abs_z = (
        max(abs(z) for z, _ in usable)
        if usable
        else None
    )

    if (
        evaluated_count < min_evaluated_metrics
        or max_abs_z is None
    ):
        candidate_flag: Optional[bool] = None
        status = "insufficient_evaluable_metrics"
    else:
        candidate_flag = max_abs_z >= threshold_value
        status = "candidate_flagged" if candidate_flag else "no_candidate_flag"

    # Include supplied inputs and configuration in the fingerprint.
    # Sorted keys make equivalent mapping orderings reproducible.
    fingerprint_payload = {
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "observed_metrics": dict(observed_metrics),
        "baseline": {
            key: dict(value) if isinstance(value, Mapping) else value
            for key, value in baseline.items()
        },
        "weights": dict(weights) if weights is not None else None,
        "threshold": threshold_value,
        "min_evaluated_metrics": min_evaluated_metrics,
    }

    try:
        canonical_payload = json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Inputs must be JSON-serializable to generate a reproducibility "
            "fingerprint."
        ) from exc

    fingerprint = hashlib.sha256(
        canonical_payload.encode("utf-8")
    ).hexdigest()

    return AnomalyEvaluationResult(
        module=MODULE_NAME,
        module_version=MODULE_VERSION,
        status=status,
        candidate_flag=candidate_flag,
        threshold=threshold_value,
        aggregate_score=aggregate_score,
        maximum_absolute_z=max_abs_z,
        evaluated_metric_count=evaluated_count,
        skipped_metric_count=skipped_count,
        metrics=evaluations,
        warnings=warnings,
        result_fingerprint=fingerprint,
    )


def module_info() -> dict[str, Any]:
    """Return public module metadata."""
    return {
        "module": MODULE_NAME,
        "module_number": MODULE_NUMBER,
        "version": MODULE_VERSION,
        "purpose": (
            "Evaluate numerical metrics against an explicit baseline "
            "and flag statistical candidates."
        ),
        "input": {
            "observed_metrics": "Mapping[str, finite numeric value]",
            "baseline": "Mapping[str, {'mean': number, 'std': non-negative number}]",
            "weights": "Optional mapping of metric names to non-negative weights",
        },
        "scoring": {
            "standardized_deviation": "(observed - baseline_mean) / baseline_std",
            "aggregate": "Weighted root-mean-square of valid z-scores",
            "candidate_rule": "Maximum absolute z-score >= threshold",
        },
        "scientific_boundary": (
            "Screening output only; not a calibrated probability, causal "
            "inference, confirmed physical event, or clinical diagnosis."
        ),
    }


def run_contract_tests() -> dict[str, bool]:
    """
    Validate input contracts without generating scientific observations.
    """
    checks: dict[str, bool] = {}

    try:
        evaluate_anomaly_candidates({}, {"metric": {"mean": 0, "std": 1}})
        checks["empty_observations_rejected"] = False
    except ValueError:
        checks["empty_observations_rejected"] = True

    try:
        evaluate_anomaly_candidates(
            {"metric": 1.0},
            {"metric": {"mean": 0.0, "std": 0.0}},
        )
        checks["zero_std_handled_without_false_zscore"] = True
    except ValueError:
        checks["zero_std_handled_without_false_zscore"] = False

    try:
        evaluate_anomaly_candidates(
            {"metric": 1.0},
            {"metric": {"mean": 0.0, "std": 1.0}},
            threshold=0,
        )
        checks["invalid_threshold_rejected"] = False
    except ValueError:
        checks["invalid_threshold_rejected"] = True

    try:
        evaluate_anomaly_candidates(
            {"metric": 1.0},
            {"metric": {"mean": 0.0, "std": 1.0}},
            weights={"metric": -1.0},
        )
        checks["negative_weight_rejected"] = False
    except ValueError:
        checks["negative_weight_rejected"] = True

    checks["all_checks_passed"] = all(checks.values())
    return checks


if __name__ == "__main__":
    print(json.dumps(module_info(), indent=2, ensure_ascii=False))
    print(json.dumps(run_contract_tests(), indent=2, ensure_ascii=False))
