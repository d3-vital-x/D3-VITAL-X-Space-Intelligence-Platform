
# ============================================================
# D³ VITAL-X Space Intelligence Platform
# Module 31 — engine_bridge.py
# ============================================================
#
# PUBLIC-TO-PRIVATE ENGINE BRIDGE
#
# Architecture:
#   Trusted Data Provider
#           |
#           v
#   Explicitly Injected Cython Function
#           |
#           v
#   Feature Validation / Allow-list
#           |
#           v
#   BlackBoxClient Response Contract
#
# Security principles:
#   - No dynamic import of the private core
#   - No raw counts or signal arrays in request metadata
#   - No synthetic fallback data
#   - No silent scientific fallback
#   - CONTRACT_ONLY never executes scientific analysis
#   - PRIVATE_ENGINE requires explicit dependencies
#   - Only requested, allow-listed, finite scalar features
#   - No medical diagnosis or flight certification
#
# IMPORTANT:
#   energy_raw is mapped to "variance" only when the caller
#   explicitly confirms that the private build uses variance.
#   The provided core computes np.var(seg), but this assertion
#   must match the exact compiled version being deployed.
#
# ============================================================

from __future__ import annotations

from typing import Any, Callable, Dict, Optional
import math


# ------------------------------------------------------------
# 1. Import public contract
# ------------------------------------------------------------

# Works when this file is imported as part of a package,
# or when 05_FEATURE_ENGINEERING is directly on sys.path.
if __package__:
    from .blackbox_client import (
        BlackBoxClient,
        BlackBoxRequest,
        ExecutionMode,
        ClientStatus,
        SAFE_FEATURES,
    )
else:
    from blackbox_client import (
        BlackBoxClient,
        BlackBoxRequest,
        ExecutionMode,
        ClientStatus,
        SAFE_FEATURES,
    )


# ------------------------------------------------------------
# 2. Module metadata
# ------------------------------------------------------------

MODULE_NAME = "engine_bridge"
MODULE_NUMBER = 31
MODULE_VERSION = "1.0.0"

PRIVATE_CORE_SOURCE_INCLUDED = False
DYNAMIC_PRIVATE_IMPORT_ENABLED = False
SYNTHETIC_FALLBACK_ENABLED = False
RAW_SIGNAL_IN_PUBLIC_METADATA_ALLOWED = False

# The current Cython function calculates only these features.
CORE_SUPPORTED_FEATURES = frozenset({
    "entropy",
    "variance",
})


# ------------------------------------------------------------
# 3. Exceptions
# ------------------------------------------------------------

class EngineBridgeError(RuntimeError):
    """Base exception for bridge configuration and execution errors."""


class CoreSchemaError(EngineBridgeError):
    """Raised when Cython output does not match the expected schema."""


class SignalInputError(EngineBridgeError):
    """Raised when a trusted signal provider returns invalid input."""


class FeatureConfigurationError(EngineBridgeError):
    """Raised when requested features are not supported or verified."""


# ------------------------------------------------------------
# 4. Configuration validation
# ------------------------------------------------------------

def _validate_positive_integer(value: Any, name: str) -> int:
    """Accept a positive integer but reject booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer.")

    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")

    return value


def _requested_core_features(
    public_request: Dict[str, Any],
    *,
    energy_semantics_verified: bool,
) -> set[str]:
    """Validate requested features against this core's capabilities."""
    requested = public_request.get("requested_features")

    if not isinstance(requested, list):
        raise FeatureConfigurationError(
            "Request requested_features must be a list."
        )

    if any(not isinstance(name, str) for name in requested):
        raise FeatureConfigurationError(
            "Every requested feature must be a string."
        )

    normalized = {name.strip().lower() for name in requested}

    unknown = normalized - SAFE_FEATURES
    if unknown:
        raise FeatureConfigurationError(
            "Request contains features outside the public allow-list."
        )

    unsupported = normalized - CORE_SUPPORTED_FEATURES
    if unsupported:
        raise FeatureConfigurationError(
            "The configured Cython core does not implement all "
            "requested features: " + ", ".join(sorted(unsupported))
        )

    if "variance" in normalized and not energy_semantics_verified:
        raise FeatureConfigurationError(
            "energy_raw-to-variance mapping is not explicitly verified."
        )

    return normalized


# ------------------------------------------------------------
# 5. Explicit Cython executor factory
# ------------------------------------------------------------

def make_cython_executor(
    cython_build_features: Callable[..., Dict[str, Any]],
    get_counts: Callable[[Dict[str, Any]], Any],
    *,
    window: int = 200,
    step: int = 50,
    energy_semantics_verified: bool = False,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Create an explicitly configured executor.

    cython_build_features:
        A trusted callable explicitly supplied by the application.
        For example, an already imported private build_features().

    get_counts:
        A trusted application callback that retrieves the actual
        signal using an authorized dataset reference/request ID.
        It must NOT read raw signal data from request metadata.

    energy_semantics_verified:
        Set True only after verifying that the exact deployed core's
        energy_raw output is np.var(seg) and that this definition is
        appropriate for the public feature contract.

    This function does not import the private core, read files,
    download datasets, or create fallback signals.
    """
    if not callable(cython_build_features):
        raise EngineBridgeError(
            "A configured Cython build_features callable is required."
        )

    if not callable(get_counts):
        raise EngineBridgeError(
            "A trusted signal-provider callable is required."
        )

    window = _validate_positive_integer(window, "window")
    step = _validate_positive_integer(step, "step")

    if not isinstance(energy_semantics_verified, bool):
        raise ValueError("energy_semantics_verified must be boolean.")

    def executor(
        public_request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run the configured core on a trusted real signal."""
        if not isinstance(public_request, dict):
            raise SignalInputError(
                "Executor input must be a public request dictionary."
            )

        requested = _requested_core_features(
            public_request,
            energy_semantics_verified=energy_semantics_verified,
        )

        # The signal provider receives identifiers and public
        # contract metadata only. It must resolve the real signal
        # through a trusted application-side data layer.
        #
        # Raw arrays are never accepted from request["metadata"].
        if "metadata" in public_request:
            metadata = public_request["metadata"]
            if not isinstance(metadata, dict):
                raise SignalInputError("Request metadata must be a dictionary.")

            forbidden_raw_keys = {
                "counts",
                "signal",
                "raw_signal",
                "raw_counts",
                "signal_samples",
                "samples",
                "raw_data",
            }
            if any(
                isinstance(key, str) and key.strip().lower() in forbidden_raw_keys
                for key in metadata
            ):
                raise SignalInputError(
                    "Raw signal fields are forbidden in public request metadata."
                )

        # Import NumPy only when actual execution is requested.
        try:
            import numpy as np
        except ImportError as exc:
            raise EngineBridgeError(
                "NumPy is required for PRIVATE_ENGINE execution."
            ) from exc

        # Trusted provider errors are allowed to propagate to the
        # BlackBoxClient boundary, which reports FAILED without
        # exposing internal exception details to the public user.
        counts = get_counts(public_request)

        try:
            signal = np.asarray(counts, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SignalInputError(
                "Trusted signal provider returned invalid numeric data."
            ) from exc

        if signal.ndim != 1:
            raise SignalInputError("Signal must be one-dimensional.")

        if signal.size < window:
            raise SignalInputError(
                f"Signal must contain at least {window} samples."
            )

        if not np.isfinite(signal).all():
            raise SignalInputError(
                "Signal contains NaN or infinite values."
            )

        if np.any(signal < 0):
            raise SignalInputError(
                "The configured count-data core requires non-negative values."
            )

        # No synthetic fallback is permitted if this call fails.
        raw = cython_build_features(
            signal,
            window=window,
            step=step,
        )

        if not isinstance(raw, dict):
            raise CoreSchemaError(
                "Cython core must return a dictionary."
            )

        if "entropy_raw" not in raw:
            raise CoreSchemaError(
                "Cython output is missing entropy_raw."
            )

        try:
            entropy_array = np.asarray(
                raw["entropy_raw"],
                dtype=np.float64,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise CoreSchemaError(
                "entropy_raw is not a valid numeric array."
            ) from exc

        if entropy_array.ndim != 1:
            raise CoreSchemaError(
                "entropy_raw must be a one-dimensional array."
            )

        if entropy_array.size == 0:
            raise CoreSchemaError(
                "Cython returned no entropy windows."
            )

        if not np.isfinite(entropy_array).all():
            raise CoreSchemaError(
                "entropy_raw contains non-finite values."
            )

        # The current public contract supports variance only when
        # its semantics have been explicitly verified.
        variance_array = None

        if "variance" in requested:
            if "energy_raw" not in raw:
                raise CoreSchemaError(
                    "Cython output is missing energy_raw."
                )

            try:
                variance_array = np.asarray(
                    raw["energy_raw"],
                    dtype=np.float64,
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise CoreSchemaError(
                    "energy_raw is not a valid numeric array."
                ) from exc

            if variance_array.ndim != 1:
                raise CoreSchemaError(
                    "energy_raw must be a one-dimensional array."
                )

            if variance_array.size != entropy_array.size:
                raise CoreSchemaError(
                    "Entropy and energy output lengths do not match."
                )

            if not np.isfinite(variance_array).all():
                raise CoreSchemaError(
                    "energy_raw contains non-finite values."
                )

            if np.any(variance_array < 0):
                raise CoreSchemaError(
                    "Variance output cannot be negative."
                )

        # Both arrays are expected to have one output per complete
        # window: floor((N - window) / step) + 1.
        expected_windows = (
            (signal.size - window) // step
        ) + 1

        if entropy_array.size != expected_windows:
            raise CoreSchemaError(
                "Entropy output length does not match the configured "
                "window/step calculation."
            )

        if variance_array is not None:
            if variance_array.size != expected_windows:
                raise CoreSchemaError(
                    "Energy output length does not match the configured "
                    "window/step calculation."
                )

        # Use the final complete window as a scalar summary.
        # This is not a full time-series export.
        features: Dict[str, float] = {}

        if "entropy" in requested:
            features["entropy"] = float(entropy_array[-1])

        if "variance" in requested:
            if variance_array is None:
                raise CoreSchemaError(
                    "Variance was requested but no verified variance output exists."
                )
            features["variance"] = float(variance_array[-1])

        # Final scalar and JSON-safety check.
        for name, value in features.items():
            if name not in SAFE_FEATURES:
                raise FeatureConfigurationError(
                    "An internal feature was outside the public allow-list."
                )
            if not math.isfinite(value):
                raise CoreSchemaError(
                    f"Feature {name} is not finite."
                )

        return {
            "status": ClientStatus.COMPLETED.value,
            "features": features,
            "uncertainty": {},
        }

    return executor


# ------------------------------------------------------------
# 6. Private-engine client factory
# ------------------------------------------------------------

def create_private_engine_client(
    *,
    target_id: str,
    cython_build_features: Callable[..., Dict[str, Any]],
    get_counts: Callable[[Dict[str, Any]], Any],
    window: int = 200,
    step: int = 50,
    energy_semantics_verified: bool = False,
) -> BlackBoxClient:
    """
    Create a PRIVATE_ENGINE client with explicit dependencies.

    This factory does not dynamically import d3_core.
    The caller must supply the configured Cython function and
    trusted signal provider from an authorized runtime.
    """
    if not isinstance(target_id, str) or not target_id.strip():
        raise ValueError("target_id must be explicitly configured.")

    executor = make_cython_executor(
        cython_build_features=cython_build_features,
        get_counts=get_counts,
        window=window,
        step=step,
        energy_semantics_verified=energy_semantics_verified,
    )

    return BlackBoxClient(
        target_id=target_id,
        execution_mode=ExecutionMode.PRIVATE_ENGINE,
        private_executor=executor,
    )


# ------------------------------------------------------------
# 7. Contract-only client factory
# ------------------------------------------------------------

def create_contract_only_engine_client(
    target_id: str,
) -> BlackBoxClient:
    """
    Create a client that validates contracts only.

    No Cython callable or signal provider is accepted.
    No scientific analysis can run in this mode.
    """
    if not isinstance(target_id, str) or not target_id.strip():
        raise ValueError("target_id must be explicitly configured.")

    return BlackBoxClient(
        target_id=target_id,
        execution_mode=ExecutionMode.CONTRACT_ONLY,
        private_executor=None,
    )


# ------------------------------------------------------------
# 8. Module contract
# ------------------------------------------------------------

def module_info() -> Dict[str, Any]:
    """Return public module capabilities and safety declarations."""
    return {
        "module_number": MODULE_NUMBER,
        "module_name": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "role": "EXPLICIT_PUBLIC_TO_PRIVATE_ENGINE_BRIDGE",
        "private_core_source_included": PRIVATE_CORE_SOURCE_INCLUDED,
        "dynamic_private_import_enabled": DYNAMIC_PRIVATE_IMPORT_ENABLED,
        "synthetic_fallback_enabled": SYNTHETIC_FALLBACK_ENABLED,
        "raw_signal_in_public_metadata_allowed": (
            RAW_SIGNAL_IN_PUBLIC_METADATA_ALLOWED
        ),
        "core_supported_features": sorted(CORE_SUPPORTED_FEATURES),
        "contract_only_runs_scientific_analysis": False,
        "private_engine_requires_injected_dependencies": True,
        "medical_diagnosis_supported": False,
        "flight_certification_supported": False,
    }


# ------------------------------------------------------------
# 9. Direct module test
# ------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=" * 64)
    print("D³ VITAL-X — Module 31: engine_bridge.py")
    print("=" * 64)
    print(json.dumps(module_info(), indent=2, ensure_ascii=False))
    print()
    print("BRIDGE MODULE IMPORT/CONTRACT CHECK: READY")
    print("No Cython core was imported or executed by this self-test.")
    print("No scientific dataset was created or analyzed.")
    print("=" * 64)
