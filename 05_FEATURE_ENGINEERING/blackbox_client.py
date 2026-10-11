
# ============================================================
# D³ VITAL-X Space Intelligence Platform
# Module 12 — blackbox_client.py
# ============================================================
#
# PUBLIC BLACK-BOX CLIENT / SECURITY BOUNDARY
#
# Principle:
#   OPEN INTERFACE — CLOSED INTELLIGENCE CORE
#
# This module provides:
#   1. Public request and response contracts
#   2. Allow-listed feature names
#   3. Request validation and metadata screening
#   4. SHA-256 fingerprints for public envelopes
#   5. Provenance and execution-status reporting
#   6. Optional, explicitly injected executor boundary
#
# This module does NOT include:
#   - UTL/DVDH implementation
#   - DSI, effective-mass or proprietary algorithms
#   - Model weights or private coefficients
#   - Private endpoint URLs or credentials
#   - Raw scientific arrays or image payloads
#   - Dynamic imports, eval() or exec()
#
# IMPORTANT:
#   The default mode does not execute a scientific engine.
#   No synthetic scientific dataset is created or analyzed.
#   Contract-only checks are software-interface checks.
#
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib
import json
import math
import re
import uuid


# ============================================================
# 1. MODULE METADATA
# ============================================================

MODULE_NAME = "blackbox_client"
MODULE_NUMBER = 12
MODULE_VERSION = "1.0.0"
PROTOCOL_VERSION = "1.0"

PROPRIETARY_ALGORITHMS_INCLUDED = False
PRIVATE_ENGINE_SOURCE_INCLUDED = False
SECRETS_INCLUDED = False

RAW_DATA_TRANSMISSION_ALLOWED = False
RAW_DATA_MODIFICATION_ALLOWED = False

MEDICAL_DIAGNOSIS_SUPPORTED = False
FLIGHT_CERTIFICATION_SUPPORTED = False

DEFAULT_EXECUTION_MODE = "CONTRACT_ONLY"


# ============================================================
# 2. PUBLIC STATUS ENUMS
# ============================================================

class ClientStatus(str, Enum):
    READY = "READY"
    DEFERRED = "DEFERRED"
    REJECTED = "REJECTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    NOT_CONFIGURED = "NOT_CONFIGURED"


class ExecutionMode(str, Enum):
    CONTRACT_ONLY = "CONTRACT_ONLY"
    PRIVATE_ENGINE = "PRIVATE_ENGINE"


# ============================================================
# 3. PUBLIC FEATURE ALLOW-LIST
# ============================================================

SAFE_FEATURES = frozenset({
    "entropy",
    "variance",
    "gradient",
    "coupling",
    "transition_index",
    "anomaly_score",
    "signal_quality",
    "confidence",
    "uncertainty",
})


# These keys must never be accepted in public metadata.
FORBIDDEN_KEYS = frozenset({
    "password",
    "passwd",
    "secret",
    "api_key",
    "apikey",
    "token",
    "access_token",
    "refresh_token",
    "authorization",
    "private_key",
    "model_weights",
    "weights",
    "mcmc_chain",
    "mcmc_parameters",
    "proprietary_parameters",
    "internal_coefficients",
    "private_coefficients",
    "raw_pixels",
    "raw_signal",
    "raw_counts",
    "raw_data",
    "raw_payload",
    "dicom_pixels",
    "pixel_array",
    "signal_samples",
    "private_endpoint",
})


FORBIDDEN_KEY_PARTS = (
    "password",
    "passwd",
    "secret",
    "api_key",
    "access_token",
    "refresh_token",
    "private_key",
    "model_weights",
    "mcmc_parameters",
    "proprietary_parameters",
    "private_coefficients",
    "raw_pixels",
    "raw_signal",
    "raw_counts",
    "raw_payload",
    "dicom_pixels",
    "pixel_array",
    "signal_samples",
    "private_endpoint",
)


# ============================================================
# 4. BASIC UTILITIES
# ============================================================

def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def generate_request_id() -> str:
    """Generate a non-sensitive request identifier."""
    return "bb-" + uuid.uuid4().hex


def canonical_json(payload: Any) -> str:
    """
    Create deterministic JSON for supported public metadata.

    Non-finite floats are rejected rather than serialized as
    non-standard JSON values.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def calculate_sha256(payload: Any) -> str:
    """
    Hash a JSON-compatible public object.

    This is an envelope fingerprint, not proof of authenticity,
    authorship, or a trusted timestamp.
    """
    encoded = canonical_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sanitize_identifier(
    value: Any,
    fallback: str = "unknown",
) -> str:
    """Normalize a public identifier; never execute its contents."""
    if value is None:
        return fallback

    text = str(value).strip()

    if not text:
        return fallback

    text = re.sub(r"[^A-Za-z0-9_.:@/-]", "_", text)
    return text[:128]


def normalize_feature_name(value: Any) -> str:
    return str(value).strip().lower()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[a-fA-F0-9]{64}", value) is not None
    )


def is_json_compatible(
    value: Any,
    depth: int = 0,
) -> bool:
    """
    Permit bounded JSON-compatible metadata only.

    Reject non-finite numbers, unsupported Python objects,
    and excessively nested structures.
    """
    if depth > 20:
        return False

    if value is None or isinstance(value, (str, bool, int)):
        return True

    if isinstance(value, float):
        return math.isfinite(value)

    if isinstance(value, list):
        return all(
            is_json_compatible(item, depth + 1)
            for item in value
        )

    if isinstance(value, dict):
        return all(
            isinstance(key, str)
            and is_json_compatible(item, depth + 1)
            for key, item in value.items()
        )

    return False


# ============================================================
# 5. PUBLIC REQUEST ENVELOPE
# ============================================================

@dataclass
class BlackBoxRequest:
    """
    Public request envelope.

    Contains identifiers, approved feature names, and metadata.
    It is not a container for scientific arrays or raw files.
    """

    request_id: str
    dataset_id: str
    target_id: str
    analysis_mode: str
    input_type: str

    requested_features: List[str] = field(
        default_factory=list
    )

    source_hash: Optional[str] = None

    provenance: Dict[str, Any] = field(
        default_factory=dict
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    human_review_required: bool = False

    created_at: str = field(
        default_factory=utc_timestamp
    )

    protocol_version: str = PROTOCOL_VERSION

    def normalize(self) -> "BlackBoxRequest":
        self.request_id = sanitize_identifier(
            self.request_id
        )
        self.dataset_id = sanitize_identifier(
            self.dataset_id
        )
        self.target_id = sanitize_identifier(
            self.target_id
        )

        self.analysis_mode = str(
            self.analysis_mode
        ).strip().upper()

        self.input_type = str(
            self.input_type
        ).strip().upper()

        if not isinstance(self.requested_features, list):
            raise ValueError(
                "requested_features must be a list."
            )

        self.requested_features = sorted({
            normalize_feature_name(item)
            for item in self.requested_features
            if isinstance(item, str) and item.strip()
        })

        return self

    def to_public_dict(self) -> Dict[str, Any]:
        """Return the public request contract."""
        return {
            "request_id": self.request_id,
            "dataset_id": self.dataset_id,
            "target_id": self.target_id,
            "analysis_mode": self.analysis_mode,
            "input_type": self.input_type,
            "requested_features": list(
                self.requested_features
            ),
            "source_hash": self.source_hash,
            "provenance": self.provenance,
            "metadata": self.metadata,
            "human_review_required": (
                self.human_review_required
            ),
            "created_at": self.created_at,
            "protocol_version": self.protocol_version,
        }


# ============================================================
# 6. PUBLIC RESPONSE ENVELOPE
# ============================================================

@dataclass
class BlackBoxResponse:
    """
    Public response envelope.

    A response may report status and allow-listed outputs.
    It must not disclose private implementation details.
    """

    request_id: str
    status: ClientStatus

    target_id: Optional[str] = None
    message: str = ""

    features: Dict[str, Any] = field(
        default_factory=dict
    )

    uncertainty: Dict[str, Any] = field(
        default_factory=dict
    )

    provenance: Dict[str, Any] = field(
        default_factory=dict
    )

    warnings: List[str] = field(
        default_factory=list
    )

    execution_metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=utc_timestamp
    )

    protocol_version: str = PROTOCOL_VERSION
    response_fingerprint: Optional[str] = None

    def finalize(self) -> "BlackBoxResponse":
        """Fingerprint the public response envelope."""
        payload = {
            "request_id": self.request_id,
            "status": self.status.value,
            "target_id": self.target_id,
            "features": self.features,
            "uncertainty": self.uncertainty,
            "warnings": self.warnings,
            "provenance": self.provenance,
            "execution_metadata": self.execution_metadata,
            "protocol_version": self.protocol_version,
        }

        self.response_fingerprint = calculate_sha256(payload)
        return self

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )


# ============================================================
# 7. BLACK-BOX CLIENT
# ============================================================

class BlackBoxClient:
    """
    Public interface to an optional external engine.

    Default behavior:
        - validate request metadata
        - build public contract
        - report DEFERRED
        - never run a scientific analysis

    PRIVATE_ENGINE mode requires an explicitly injected,
    trusted callable. No dynamic import is performed.
    """

    def __init__(
        self,
        target_id: Optional[str] = None,
        execution_mode: ExecutionMode = (
            ExecutionMode.CONTRACT_ONLY
        ),
        private_executor: Optional[Callable] = None,
    ):
        if not isinstance(execution_mode, ExecutionMode):
            raise ValueError("Invalid execution mode.")

        self.target_id = (
            sanitize_identifier(target_id)
            if target_id
            else None
        )

        self.execution_mode = execution_mode
        self._private_executor = private_executor

    # --------------------------------------------------------
    # Forbidden-key scanner
    # --------------------------------------------------------

    def _find_forbidden_keys(
        self,
        payload: Any,
        path: str = "",
    ) -> List[str]:
        found: List[str] = []

        if isinstance(payload, dict):
            for key, value in payload.items():
                key_lower = str(key).strip().lower()

                current_path = (
                    f"{path}.{key}" if path else str(key)
                )

                if (
                    key_lower in FORBIDDEN_KEYS
                    or any(
                        token in key_lower
                        for token in FORBIDDEN_KEY_PARTS
                    )
                ):
                    found.append(current_path)

                found.extend(
                    self._find_forbidden_keys(
                        value,
                        current_path,
                    )
                )

        elif isinstance(payload, list):
            for index, item in enumerate(payload):
                found.extend(
                    self._find_forbidden_keys(
                        item,
                        f"{path}[{index}]",
                    )
                )

        return sorted(set(found))

    # --------------------------------------------------------
    # Request validation
    # --------------------------------------------------------

    def validate_request(
        self,
        request: BlackBoxRequest,
    ) -> Tuple[bool, List[str]]:
        errors: List[str] = []

        if not isinstance(request, BlackBoxRequest):
            return False, ["Request must be BlackBoxRequest."]

        # Check required values before normalization.
        required_fields = {
            "request_id": request.request_id,
            "dataset_id": request.dataset_id,
            "target_id": request.target_id,
            "analysis_mode": request.analysis_mode,
            "input_type": request.input_type,
        }

        for name, value in required_fields.items():
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{name} is required.")

        if errors:
            return False, errors

        try:
            request.normalize()
        except (ValueError, TypeError):
            return False, [
                "Request fields could not be normalized."
            ]

        if request.protocol_version != PROTOCOL_VERSION:
            errors.append("Unsupported protocol version.")

        if not isinstance(request.provenance, dict):
            errors.append("provenance must be a dictionary.")

        if not isinstance(request.metadata, dict):
            errors.append("metadata must be a dictionary.")

        if not isinstance(request.human_review_required, bool):
            errors.append(
                "human_review_required must be boolean."
            )

        if not is_json_compatible(request.provenance):
            errors.append(
                "provenance contains unsupported metadata."
            )

        if not is_json_compatible(request.metadata):
            errors.append(
                "metadata contains unsupported values."
            )

        invalid_features = [
            feature
            for feature in request.requested_features
            if feature not in SAFE_FEATURES
        ]

        if invalid_features:
            errors.append(
                "Unsupported feature name(s): "
                + ", ".join(invalid_features)
            )

        if request.source_hash is not None:
            if not is_sha256(request.source_hash):
                errors.append(
                    "source_hash must be a 64-character SHA-256 hex digest."
                )

        if request.analysis_mode == "BIOMEDICAL":
            # This flag means human review is required.
            request.human_review_required = True

        forbidden = self._find_forbidden_keys({
            "provenance": request.provenance,
            "metadata": request.metadata,
        })

        if forbidden:
            errors.append(
                "Forbidden metadata key(s): "
                + ", ".join(forbidden)
            )

        return len(errors) == 0, errors

    # --------------------------------------------------------
    # Request builder
    # --------------------------------------------------------

    def build_request(
        self,
        *,
        dataset_id: str,
        analysis_mode: str,
        input_type: str,
        requested_features: Optional[List[str]] = None,
        source_hash: Optional[str] = None,
        provenance: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        human_review_required: bool = False,
        request_id: Optional[str] = None,
        target_id: Optional[str] = None,
    ) -> BlackBoxRequest:

        resolved_target = target_id or self.target_id

        if not resolved_target:
            raise ValueError(
                "target_id must be explicitly configured."
            )

        request = BlackBoxRequest(
            request_id=request_id or generate_request_id(),
            dataset_id=dataset_id,
            target_id=resolved_target,
            analysis_mode=analysis_mode,
            input_type=input_type,
            requested_features=requested_features or [],
            source_hash=source_hash,
            provenance=provenance or {},
            metadata=metadata or {},
            human_review_required=human_review_required,
        )

        valid, errors = self.validate_request(request)

        if not valid:
            raise ValueError(
                "Invalid black-box request: "
                + " | ".join(errors)
            )

        return request

    # --------------------------------------------------------
    # Contract preview
    # --------------------------------------------------------

    def preview_contract(
        self,
        request: BlackBoxRequest,
    ) -> Dict[str, Any]:

        valid, errors = self.validate_request(request)

        if not valid:
            return {
                "valid": False,
                "errors": errors,
                "raw_data_included": False,
                "private_implementation_exposed": False,
            }

        public_request = request.to_public_dict()

        return {
            "valid": True,
            "protocol_version": PROTOCOL_VERSION,
            "request": public_request,
            "request_fingerprint": calculate_sha256(
                public_request
            ),
            "raw_data_included": False,
            "private_implementation_exposed": False,
            "secrets_included": False,
        }

    # --------------------------------------------------------
    # Execution boundary
    # --------------------------------------------------------

    def execute(
        self,
        request: BlackBoxRequest,
    ) -> BlackBoxResponse:

        valid, errors = self.validate_request(request)

        if not valid:
            return BlackBoxResponse(
                request_id=sanitize_identifier(
                    getattr(request, "request_id", None)
                ),
                status=ClientStatus.REJECTED,
                target_id=getattr(request, "target_id", None),
                message="Request rejected by public validation.",
                warnings=errors,
                execution_metadata={
                    "execution_performed": False,
                    "private_engine_executed": False,
                },
            ).finalize()

        # Default public mode: no scientific engine execution.
        if self.execution_mode == ExecutionMode.CONTRACT_ONLY:
            return BlackBoxResponse(
                request_id=request.request_id,
                status=ClientStatus.DEFERRED,
                target_id=request.target_id,
                message=(
                    "Public contract validated. "
                    "No scientific engine was executed."
                ),
                provenance={
                    "source_hash": request.source_hash,
                    "protocol_version": PROTOCOL_VERSION,
                },
                warnings=[
                    "CONTRACT_ONLY_MODE",
                    "NO_SCIENTIFIC_ANALYSIS_PERFORMED",
                    "PRIVATE_IMPLEMENTATION_NOT_INCLUDED",
                ],
                execution_metadata={
                    "execution_performed": False,
                    "private_engine_executed": False,
                    "execution_mode": (
                        ExecutionMode.CONTRACT_ONLY.value
                    ),
                },
            ).finalize()

        # Private mode must be explicitly configured.
        if self.execution_mode == ExecutionMode.PRIVATE_ENGINE:
            if self._private_executor is None:
                return BlackBoxResponse(
                    request_id=request.request_id,
                    status=ClientStatus.NOT_CONFIGURED,
                    target_id=request.target_id,
                    message=(
                        "No private executor is configured "
                        "in this public environment."
                    ),
                    warnings=[
                        "PRIVATE_ENGINE_NOT_CONFIGURED",
                    ],
                    execution_metadata={
                        "execution_performed": False,
                        "private_engine_executed": False,
                    },
                ).finalize()

            if not callable(self._private_executor):
                return BlackBoxResponse(
                    request_id=request.request_id,
                    status=ClientStatus.FAILED,
                    target_id=request.target_id,
                    message="Configured executor is not callable.",
                    warnings=["INVALID_EXECUTOR"],
                    execution_metadata={
                        "execution_performed": False,
                    },
                ).finalize()

            try:
                # Only the public envelope is passed across
                # this explicitly configured boundary.
                result = self._private_executor(
                    request.to_public_dict()
                )
            except Exception:
                # Do not leak internal exception messages.
                return BlackBoxResponse(
                    request_id=request.request_id,
                    status=ClientStatus.FAILED,
                    target_id=request.target_id,
                    message="External execution failed.",
                    warnings=["EXECUTOR_ERROR_DETAILS_HIDDEN"],
                    execution_metadata={
                        "execution_performed": True,
                        "error_details_hidden": True,
                    },
                ).finalize()

            return self._sanitize_private_result(
                request,
                result,
            )

        return BlackBoxResponse(
            request_id=request.request_id,
            status=ClientStatus.FAILED,
            target_id=request.target_id,
            message="Unsupported execution mode.",
            warnings=["UNKNOWN_EXECUTION_MODE"],
            execution_metadata={
                "execution_performed": False,
            },
        ).finalize()

    # --------------------------------------------------------
    # Public result sanitization
    # --------------------------------------------------------

    def _sanitize_private_result(
        self,
        request: BlackBoxRequest,
        result: Any,
    ) -> BlackBoxResponse:
        """
        Filter an explicitly injected executor's result.

        Only allow-listed feature names and JSON-compatible
        values can be returned. This is not a substitute for
        sandboxing or auditing the private executor.
        """

        if not isinstance(result, dict):
            return BlackBoxResponse(
                request_id=request.request_id,
                status=ClientStatus.FAILED,
                target_id=request.target_id,
                message="Executor returned an invalid response.",
                warnings=["INVALID_EXECUTOR_RESPONSE"],
                execution_metadata={
                    "execution_performed": True,
                },
            ).finalize()

        raw_features = result.get("features", {})
        raw_uncertainty = result.get("uncertainty", {})

        if not isinstance(raw_features, dict):
            raw_features = {}

        if not isinstance(raw_uncertainty, dict):
            raw_uncertainty = {}

        # Reject sensitive keys anywhere in the public result.
        forbidden = self._find_forbidden_keys(result)

        if forbidden:
            return BlackBoxResponse(
                request_id=request.request_id,
                status=ClientStatus.FAILED,
                target_id=request.target_id,
                message=(
                    "Executor response failed public "
                    "boundary screening."
                ),
                warnings=["FORBIDDEN_RESULT_FIELDS_DETECTED"],
                execution_metadata={
                    "execution_performed": True,
                    "result_details_hidden": True,
                },
            ).finalize()

        safe_features: Dict[str, Any] = {}

        for name, value in raw_features.items():
            if not isinstance(name, str):
                continue

            normalized = normalize_feature_name(name)

            if (
                normalized in SAFE_FEATURES
                and is_json_compatible(value)
            ):
                safe_features[normalized] = value

        safe_uncertainty = (
            raw_uncertainty
            if is_json_compatible(raw_uncertainty)
            else {}
        )

        # Never allow an executor to claim that it did not run
        # while returning a completed result.
        raw_status = str(
            result.get("status", "COMPLETED")
        ).upper()

        try:
            status = ClientStatus(raw_status)
        except ValueError:
            status = ClientStatus.COMPLETED

        return BlackBoxResponse(
            request_id=request.request_id,
            status=status,
            target_id=request.target_id,
            message=(
                "Executor response filtered through "
                "the public feature contract."
            ),
            features=safe_features,
            uncertainty=safe_uncertainty,
            provenance={
                "source_hash": request.source_hash,
                "protocol_version": PROTOCOL_VERSION,
            },
            warnings=[
                "EXTERNAL_EXECUTOR_WAS_CONFIGURED",
                "PRIVATE_IMPLEMENTATION_NOT_INCLUDED_HERE",
                "HUMAN_REVIEW_REQUIRED_FOR_INTERPRETATION",
            ],
            execution_metadata={
                "execution_performed": True,
                "executor_result_filtered": True,
            },
        ).finalize()


# ============================================================
# 8. STANDARD PUBLIC CLIENT FACTORY
# ============================================================

def create_blackbox_client(
    target_id: Optional[str] = None,
) -> BlackBoxClient:
    """
    Create the safe public client.

    This factory never configures a private executor.
    """
    return BlackBoxClient(
        target_id=target_id,
        execution_mode=ExecutionMode.CONTRACT_ONLY,
        private_executor=None,
    )


# ============================================================
# 9. PUBLIC PROTOCOL CONTRACT
# ============================================================

def blackbox_protocol_contract() -> Dict[str, Any]:
    return {
        "module": MODULE_NAME,
        "module_number": MODULE_NUMBER,
        "module_version": MODULE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "purpose": (
            "Public request/response boundary for an "
            "optional external intelligence engine."
        ),
        "proprietary_algorithms_included": (
            PROPRIETARY_ALGORITHMS_INCLUDED
        ),
        "private_engine_source_included": (
            PRIVATE_ENGINE_SOURCE_INCLUDED
        ),
        "secrets_included": SECRETS_INCLUDED,
        "raw_data_transmission_allowed": (
            RAW_DATA_TRANSMISSION_ALLOWED
        ),
        "raw_data_modification_allowed": (
            RAW_DATA_MODIFICATION_ALLOWED
        ),
        "medical_diagnosis_supported": (
            MEDICAL_DIAGNOSIS_SUPPORTED
        ),
        "flight_certification_supported": (
            FLIGHT_CERTIFICATION_SUPPORTED
        ),
        "safe_features": sorted(SAFE_FEATURES),
        "default_execution_mode": DEFAULT_EXECUTION_MODE,
        "scientific_engine_included": False,
    }


# ============================================================
# 10. CONTRACT-ONLY SOFTWARE TESTS
# ============================================================

def run_blackbox_client_test() -> Dict[str, Any]:
    """
    Test interface behavior only.

    These checks use contract metadata; they do not fabricate,
    load, or analyze a scientific dataset.
    """

    results: Dict[str, bool] = {}

    client = create_blackbox_client(
        target_id="CONTRACT_TEST_TARGET"
    )

    results["client_created"] = isinstance(
        client, BlackBoxClient
    )

    # Contract-only metadata fixture, not scientific data.
    request = BlackBoxRequest(
        request_id="contract-test-001",
        dataset_id="CONTRACT_ONLY",
        target_id="CONTRACT_TEST_TARGET",
        analysis_mode="SPACE",
        input_type="TIME_SERIES",
        requested_features=[
            "entropy",
            "variance",
            "signal_quality",
        ],
        metadata={
            "test_scope": "interface_contract_only"
        },
    )

    valid, _ = client.validate_request(request)
    results["valid_contract_accepted"] = valid

    preview = client.preview_contract(request)
    results["contract_preview_valid"] = (
        preview.get("valid") is True
    )
    results["raw_data_excluded"] = (
        preview.get("raw_data_included") is False
    )
    results["private_source_hidden"] = (
        preview.get("private_implementation_exposed") is False
    )

    response = client.execute(request)
    results["default_execution_deferred"] = (
        response.status == ClientStatus.DEFERRED
    )
    results["scientific_execution_not_performed"] = (
        response.execution_metadata.get(
            "execution_performed"
        ) is False
    )

    # Forbidden feature request.
    invalid_feature_request = BlackBoxRequest(
        request_id="contract-test-002",
        dataset_id="CONTRACT_ONLY",
        target_id="CONTRACT_TEST_TARGET",
        analysis_mode="SPACE",
        input_type="TIME_SERIES",
        requested_features=[
            "internal_proprietary_feature"
        ],
    )

    valid, _ = client.validate_request(
        invalid_feature_request
    )
    results["unknown_feature_rejected"] = not valid

    # Secret/raw-payload metadata rejection.
    forbidden_metadata_request = BlackBoxRequest(
        request_id="contract-test-003",
        dataset_id="CONTRACT_ONLY",
        target_id="CONTRACT_TEST_TARGET",
        analysis_mode="SPACE",
        input_type="IMAGE",
        metadata={
            "api_key": "placeholder-not-a-real-key"
        },
    )

    valid, _ = client.validate_request(
        forbidden_metadata_request
    )
    results["forbidden_metadata_rejected"] = not valid

    # Biomedical mode must require human review.
    biomedical_request = BlackBoxRequest(
        request_id="contract-test-004",
        dataset_id="CONTRACT_ONLY",
        target_id="CONTRACT_TEST_TARGET",
        analysis_mode="BIOMEDICAL",
        input_type="DICOM",
        requested_features=["signal_quality"],
    )

    valid, _ = client.validate_request(biomedical_request)

    results["biomedical_contract_valid"] = valid
    results["biomedical_human_review_required"] = (
        biomedical_request.human_review_required is True
    )

    contract = blackbox_protocol_contract()

    results["private_source_not_included"] = (
        contract["private_engine_source_included"] is False
    )
    results["proprietary_algorithms_not_included"] = (
        contract["proprietary_algorithms_included"] is False
    )
    results["raw_data_transmission_disabled"] = (
        contract["raw_data_transmission_allowed"] is False
    )
    results["medical_diagnosis_not_supported"] = (
        contract["medical_diagnosis_supported"] is False
    )

    results["all_checks_passed"] = all(results.values())

    return results


# ============================================================
# 11. MODULE INFORMATION
# ============================================================

def module_info() -> Dict[str, Any]:
    return {
        "module_name": MODULE_NAME,
        "module_number": MODULE_NUMBER,
        "module_version": MODULE_VERSION,
        "role": "Public black-box boundary client",
        "default_execution_mode": DEFAULT_EXECUTION_MODE,
        "private_engine_source_included": False,
        "proprietary_algorithms_included": False,
        "secrets_included": False,
        "raw_data_transmission_allowed": False,
        "raw_data_modification_allowed": False,
        "scientific_analysis_performed_by_this_module": False,
        "medical_diagnosis_supported": False,
        "flight_certification_supported": False,
    }


# ============================================================
# 12. DIRECT MODULE TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 64)
    print("D³ VITAL-X — Module 12: blackbox_client.py")
    print("=" * 64)

    test_results = run_blackbox_client_test()

    for name, passed in test_results.items():
        print(f"{name}: {passed}")

    print("-" * 64)

    if test_results["all_checks_passed"]:
        print("CONTRACT TESTS: PASS")
    else:
        print("CONTRACT TESTS: REVIEW REQUIRED")

    print(
        "Note: contract tests do not validate scientific "
        "accuracy or execute the private intelligence core."
    )
    print("=" * 64)
