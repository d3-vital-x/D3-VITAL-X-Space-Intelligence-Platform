
# ============================================================
# D³ VITAL-X SPACE INTELLIGENCE PLATFORM
# Module 11 — engine_interface.py
#
# BLACK-BOX ENGINE INTERFACE
#
# Purpose:
# Define a controlled public contract between:
#
#     Real Data / Unified Data Layer
#                  |
#                  v
#         ENGINE INTERFACE
#                  |
#                  v
#        Configured Engine Provider
#
# Public:
# - Request and response schemas
# - Real dataset references and provenance
# - Input validation and SHA-256 verification metadata
# - Execution status and capability declarations
# - Safe error reporting and serialization
#
# Private:
# - Proprietary intelligence algorithms
# - Feature-generation equations
# - Internal model parameters and optimization
# - MCMC / parameter inference implementation
# - Private intelligence-core source code
#
# IMPORTANT:
# - This module does not download NASA data.
# - This module does not generate scientific results.
# - A configured provider is required for actual analysis.
# - A NULL provider never claims that analysis was performed.
#
# Safety:
# - No eval()
# - No exec()
# - No arbitrary code execution
# - No private source loading
# - No modification of raw input
# - No medical diagnosis
# - No flight certification
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple
import hashlib
import json
import math
import uuid


# ------------------------------------------------------------
# 1. Module metadata
# ------------------------------------------------------------

MODULE_NUMBER = 11
MODULE_NAME = "engine_interface"
MODULE_VERSION = "1.0.0"

INTERFACE_VERSION = "1.0"

PRIVATE_CORE_INCLUDED = False
RAW_DATA_MODIFICATION_ALLOWED = False
MEDICAL_DIAGNOSIS_SUPPORTED = False
FLIGHT_CERTIFICATION_SUPPORTED = False


# ------------------------------------------------------------
# 2. Utility functions
# ------------------------------------------------------------

def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def generate_request_id() -> str:
    """Generate a unique request identifier."""
    return "ENG-" + uuid.uuid4().hex[:16].upper()


def calculate_sha256_bytes(payload: bytes) -> str:
    """Calculate SHA-256 for bytes."""
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes.")
    return hashlib.sha256(payload).hexdigest()


def is_sha256(value: Any) -> bool:
    """Check whether a string is a 64-character SHA-256 hex digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(
            character in "0123456789abcdefABCDEF"
            for character in value
        )
    )


def _json_safe(value: Any) -> Any:
    """Convert supported objects into JSON-safe values."""
    if isinstance(value, Enum):
        return value.value

    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite numbers cannot be serialized.")

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    # Do not execute custom serialization methods or arbitrary code.
    raise TypeError(
        f"Unsupported JSON value type: {type(value).__name__}"
    )


def canonical_json(payload: Any) -> str:
    """Return deterministic JSON for supported data."""
    return json.dumps(
        _json_safe(payload),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def calculate_object_hash(payload: Any) -> str:
    """Calculate SHA-256 of a canonical JSON representation."""
    return calculate_sha256_bytes(
        canonical_json(payload).encode("utf-8")
    )


def _validate_string_list(value: Any, field_name: str) -> List[str]:
    """Validate a list of non-empty strings."""
    errors: List[str] = []

    if not isinstance(value, list):
        return [f"{field_name} must be a list."]

    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            errors.append(
                f"{field_name}[{index}] must be a non-empty string."
            )

    return errors


# ------------------------------------------------------------
# 3. Engine execution status
# ------------------------------------------------------------

class EngineStatus(str, Enum):
    """Public execution status labels."""

    READY = "READY"
    ACCEPTED = "ACCEPTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_CONFIGURED = "NOT_CONFIGURED"


# ------------------------------------------------------------
# 4. Analysis mode
# ------------------------------------------------------------

class AnalysisMode(str, Enum):
    """Interface-level request categories."""

    GENERAL = "GENERAL"
    SPACE = "SPACE"
    BIOMEDICAL_RESEARCH = "BIOMEDICAL_RESEARCH"
    TIME_SERIES = "TIME_SERIES"
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"
    NASA = "NASA"
    VALIDATION = "VALIDATION"


# ------------------------------------------------------------
# 5. Real dataset input reference
# ------------------------------------------------------------

@dataclass
class EngineInputReference:
    """
    Reference to a real dataset or UnifiedDataRecord.

    The interface passes references and metadata, not an
    embedded copy of the raw scientific dataset.

    file_path:
        Path to an existing local dataset file.

    source_uri:
        URI assigned to the dataset by its source system.

    raw_data_hash:
        SHA-256 digest supplied by the upstream adapter or
        computed by a trusted data-ingestion component.
    """

    dataset_id: str
    source_type: str

    file_path: Optional[str] = None
    source_uri: Optional[str] = None
    raw_data_hash: Optional[str] = None

    raw_data_immutable: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Validate reference structure without opening or downloading data."""
        errors: List[str] = []

        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            errors.append("dataset_id must be a non-empty string.")

        if not isinstance(self.source_type, str) or not self.source_type.strip():
            errors.append("source_type must be a non-empty string.")

        if not isinstance(self.raw_data_immutable, bool):
            errors.append("raw_data_immutable must be boolean.")
        elif self.raw_data_immutable is not True:
            errors.append("raw_data_immutable must be True.")

        if self.file_path is not None:
            if not isinstance(self.file_path, str) or not self.file_path.strip():
                errors.append("file_path must be a non-empty string or None.")

        if self.source_uri is not None:
            if not isinstance(self.source_uri, str) or not self.source_uri.strip():
                errors.append("source_uri must be a non-empty string or None.")

        if not self.file_path and not self.source_uri:
            errors.append(
                "Either file_path or source_uri must be provided."
            )

        if self.raw_data_hash is not None and not is_sha256(self.raw_data_hash):
            errors.append(
                "raw_data_hash must be a 64-character SHA-256 hex digest."
            )

        if not isinstance(self.metadata, dict):
            errors.append("metadata must be a dictionary.")
        else:
            try:
                canonical_json(self.metadata)
            except (TypeError, ValueError) as exc:
                errors.append(f"metadata is not JSON-safe: {exc}")

        return errors


# ------------------------------------------------------------
# 6. Engine request
# ------------------------------------------------------------

@dataclass
class EngineRequest:
    """Public request contract sent to a configured provider."""

    request_id: str
    interface_version: str
    mode: AnalysisMode
    input_reference: EngineInputReference

    requested_outputs: List[str] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_timestamp)

    def validate(self) -> List[str]:
        """Validate request fields and nested input reference."""
        errors: List[str] = []

        if not isinstance(self.request_id, str) or not self.request_id.strip():
            errors.append("request_id must be a non-empty string.")

        if (
            not isinstance(self.interface_version, str)
            or not self.interface_version.strip()
        ):
            errors.append("interface_version must be a non-empty string.")

        if not isinstance(self.mode, AnalysisMode):
            errors.append("mode must be an AnalysisMode.")

        if not isinstance(self.input_reference, EngineInputReference):
            errors.append("input_reference must be EngineInputReference.")
        else:
            errors.extend(
                f"input_reference: {message}"
                for message in self.input_reference.validate()
            )

        errors.extend(
            _validate_string_list(
                self.requested_outputs,
                "requested_outputs",
            )
        )

        for field_name, value in (
            ("options", self.options),
            ("provenance", self.provenance),
        ):
            if not isinstance(value, dict):
                errors.append(f"{field_name} must be a dictionary.")
            else:
                try:
                    canonical_json(value)
                except (TypeError, ValueError) as exc:
                    errors.append(f"{field_name} is not JSON-safe: {exc}")

        if not isinstance(self.created_at, str) or not self.created_at.strip():
            errors.append("created_at must be a non-empty string.")

        return errors

    def fingerprint(self) -> str:
        """
        Fingerprint the interface request.

        This identifies the request metadata; it does not prove
        that a dataset was downloaded, opened, or analyzed.
        """
        payload = {
            "interface_version": self.interface_version,
            "mode": self.mode.value,
            "input_reference": asdict(self.input_reference),
            "requested_outputs": self.requested_outputs,
            "options": self.options,
        }
        return calculate_object_hash(payload)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))

    def to_json(self, indent: int = 2) -> str:
        if not isinstance(indent, int) or indent < 0:
            raise ValueError("indent must be a non-negative integer.")

        return json.dumps(
            self.to_dict(),
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )


# ------------------------------------------------------------
# 7. Engine output
# ------------------------------------------------------------

@dataclass
class EngineOutput:
    """Container for a provider-returned computational output."""

    name: str
    value: Any

    unit: Optional[str] = None
    confidence: Optional[float] = None
    uncertainty: Optional[Any] = None

    interpretation_class: str = "COMPUTATIONAL_OUTPUT"
    human_review_required: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        errors: List[str] = []

        if not isinstance(self.name, str) or not self.name.strip():
            errors.append("Output name must be a non-empty string.")

        if self.confidence is not None:
            if (
                isinstance(self.confidence, bool)
                or not isinstance(self.confidence, (int, float))
                or not math.isfinite(float(self.confidence))
            ):
                errors.append("confidence must be a finite number.")
            elif not 0.0 <= float(self.confidence) <= 1.0:
                errors.append("confidence must be between 0 and 1.")

        if not isinstance(self.human_review_required, bool):
            errors.append("human_review_required must be boolean.")

        if not isinstance(self.interpretation_class, str):
            errors.append("interpretation_class must be a string.")

        if self.unit is not None and not isinstance(self.unit, str):
            errors.append("unit must be a string or None.")

        if not isinstance(self.metadata, dict):
            errors.append("metadata must be a dictionary.")

        try:
            canonical_json({
                "value": self.value,
                "uncertainty": self.uncertainty,
                "metadata": self.metadata,
            })
        except (TypeError, ValueError) as exc:
            errors.append(f"Output contains non-JSON-safe values: {exc}")

        return errors


# ------------------------------------------------------------
# 8. Engine response
# ------------------------------------------------------------

@dataclass
class EngineResponse:
    """Public response contract from an engine provider."""

    request_id: str
    interface_version: str
    status: EngineStatus

    outputs: List[EngineOutput] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    completed_at: Optional[str] = None

    def validate(self) -> List[str]:
        errors: List[str] = []

        if not isinstance(self.request_id, str) or not self.request_id.strip():
            errors.append("request_id must be a non-empty string.")

        if (
            not isinstance(self.interface_version, str)
            or not self.interface_version.strip()
        ):
            errors.append("interface_version must be a non-empty string.")

        if not isinstance(self.status, EngineStatus):
            errors.append("status must be EngineStatus.")

        if not isinstance(self.outputs, list):
            errors.append("outputs must be a list.")
        else:
            for index, output in enumerate(self.outputs):
                if not isinstance(output, EngineOutput):
                    errors.append(
                        f"outputs[{index}] must be EngineOutput."
                    )
                else:
                    errors.extend(
                        f"outputs[{index}]: {message}"
                        for message in output.validate()
                    )

        for field_name, value in (
            ("messages", self.messages),
            ("warnings", self.warnings),
            ("errors", self.errors),
        ):
            errors.extend(_validate_string_list(value, field_name))

        for field_name, value in (
            ("execution_metadata", self.execution_metadata),
            ("provenance", self.provenance),
        ):
            if not isinstance(value, dict):
                errors.append(f"{field_name} must be a dictionary.")
            else:
                try:
                    canonical_json(value)
                except (TypeError, ValueError) as exc:
                    errors.append(f"{field_name} is not JSON-safe: {exc}")

        if self.completed_at is not None and not isinstance(self.completed_at, str):
            errors.append("completed_at must be a string or None.")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))

    def to_json(self, indent: int = 2) -> str:
        if not isinstance(indent, int) or indent < 0:
            raise ValueError("indent must be a non-negative integer.")

        return json.dumps(
            self.to_dict(),
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )


# ------------------------------------------------------------
# 9. Public capability declaration
# ------------------------------------------------------------

@dataclass(frozen=True)
class EngineCapabilities:
    """
    Public capability declaration.

    Supported modes describe the interface vocabulary;
    they do not mean that a provider is deployed or operational.
    """

    interface_version: str = INTERFACE_VERSION
    black_box: bool = True
    proprietary_core_exposed: bool = False
    raw_data_modification: bool = False
    medical_diagnosis: bool = False
    flight_certification: bool = False

    supported_modes: Tuple[str, ...] = tuple(
        mode.value for mode in AnalysisMode
    )

    supported_output_types: Tuple[str, ...] = (
        "METRIC",
        "FEATURE",
        "ANOMALY_SCORE",
        "TRANSITION_INDICATOR",
        "QUALITY_INDICATOR",
        "UNCERTAINTY",
        "VISUALIZATION_REFERENCE",
        "EXPLANATION_REFERENCE",
    )


# ------------------------------------------------------------
# 10. Provider contract
# ------------------------------------------------------------

class EngineProvider:
    """
    Base provider contract.

    A production provider must implement execute() and connect
    to an authorized, tested analysis implementation.

    This public class contains no proprietary algorithm.
    """

    provider_name = "UNCONFIGURED"
    provider_version = "0.0"

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities()

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": EngineStatus.NOT_CONFIGURED.value,
            "provider": self.provider_name,
            "provider_version": self.provider_version,
            "available": False,
            "checked_at": utc_timestamp(),
        }

    def execute(self, request: EngineRequest) -> EngineResponse:
        raise NotImplementedError(
            "No engine implementation is configured."
        )


# ------------------------------------------------------------
# 11. Safe null provider
# ------------------------------------------------------------

class NullEngineProvider(EngineProvider):
    """
    Default public provider.

    It validates requests but does not perform scientific analysis.
    """

    provider_name = "PUBLIC_NULL_PROVIDER"
    provider_version = "1.0.0"

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": EngineStatus.NOT_CONFIGURED.value,
            "provider": self.provider_name,
            "provider_version": self.provider_version,
            "available": False,
            "private_core_executed": False,
            "checked_at": utc_timestamp(),
        }

    def execute(self, request: EngineRequest) -> EngineResponse:
        errors = request.validate()

        if errors:
            return EngineResponse(
                request_id=request.request_id,
                interface_version=request.interface_version,
                status=EngineStatus.REJECTED,
                errors=errors,
                warnings=[
                    "The request failed public interface validation."
                ],
                execution_metadata={
                    "execution_performed": False,
                    "private_core_executed": False,
                },
                provenance={
                    "interface": MODULE_NAME,
                    "module_version": MODULE_VERSION,
                },
                completed_at=utc_timestamp(),
            )

        return EngineResponse(
            request_id=request.request_id,
            interface_version=request.interface_version,
            status=EngineStatus.NOT_CONFIGURED,
            outputs=[],
            messages=[
                "The request structure is valid.",
                "No analysis was performed because no engine provider is configured.",
            ],
            warnings=[
                "This is not a scientific analysis result.",
                "No medical or mission-critical decision should be derived from this response.",
            ],
            execution_metadata={
                "provider": self.provider_name,
                "provider_version": self.provider_version,
                "execution_performed": False,
                "private_core_executed": False,
                "raw_data_modified": False,
            },
            provenance={
                "interface": MODULE_NAME,
                "module_version": MODULE_VERSION,
                "request_fingerprint": request.fingerprint(),
                "dataset_id": request.input_reference.dataset_id,
                "source_type": request.input_reference.source_type,
            },
            completed_at=utc_timestamp(),
        )


# ------------------------------------------------------------
# 12. Engine interface facade
# ------------------------------------------------------------

class EngineInterface:
    """Application-facing interface to a configured provider."""

    def __init__(
        self,
        provider: Optional[EngineProvider] = None,
    ) -> None:
        self.provider = (
            provider if provider is not None else NullEngineProvider()
        )

    def capabilities(self) -> EngineCapabilities:
        return self.provider.capabilities()

    def health_check(self) -> Dict[str, Any]:
        return self.provider.health_check()

    def create_request(
        self,
        input_reference: EngineInputReference,
        mode: AnalysisMode = AnalysisMode.GENERAL,
        requested_outputs: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> EngineRequest:
        request = EngineRequest(
            request_id=generate_request_id(),
            interface_version=INTERFACE_VERSION,
            mode=mode,
            input_reference=input_reference,
            requested_outputs=(
                list(requested_outputs)
                if requested_outputs is not None else []
            ),
            options=dict(options) if options is not None else {},
            provenance=dict(provenance) if provenance is not None else {},
        )

        errors = request.validate()
        if errors:
            raise ValueError("Invalid engine request: " + "; ".join(errors))

        return request

    def execute(self, request: EngineRequest) -> EngineResponse:
        """Validate the request before passing it to a provider."""
        request_errors = request.validate()

        if request_errors:
            return EngineResponse(
                request_id=request.request_id,
                interface_version=request.interface_version,
                status=EngineStatus.REJECTED,
                errors=request_errors,
                execution_metadata={
                    "execution_performed": False,
                    "private_core_executed": False,
                },
                completed_at=utc_timestamp(),
            )

        try:
            response = self.provider.execute(request)
        except NotImplementedError:
            response = EngineResponse(
                request_id=request.request_id,
                interface_version=INTERFACE_VERSION,
                status=EngineStatus.NOT_CONFIGURED,
                messages=["The provider has no execution implementation."],
                execution_metadata={
                    "execution_performed": False,
                    "private_core_executed": False,
                },
                completed_at=utc_timestamp(),
            )
        except Exception:
            # Do not expose internal exception details or secrets.
            response = EngineResponse(
                request_id=request.request_id,
                interface_version=INTERFACE_VERSION,
                status=EngineStatus.FAILED,
                errors=[
                    "Provider execution failed. Check protected server logs."
                ],
                execution_metadata={
                    "execution_performed": False,
                    "private_core_executed": False,
                },
                completed_at=utc_timestamp(),
            )

        if not isinstance(response, EngineResponse):
            return EngineResponse(
                request_id=request.request_id,
                interface_version=INTERFACE_VERSION,
                status=EngineStatus.FAILED,
                errors=[
                    "Provider returned an unsupported response object."
                ],
                completed_at=utc_timestamp(),
            )

        # Ensure the response corresponds to this request.
        if response.request_id != request.request_id:
            response.status = EngineStatus.FAILED
            response.outputs = []
            response.errors = [
                "Provider response request_id does not match the submitted request."
            ]
            response.execution_metadata = {
                "execution_performed": False,
                "response_correlation_failed": True,
            }

        return response


# ------------------------------------------------------------
# 13. Unified record -> input reference
# ------------------------------------------------------------

def _read_field(record: Any, field_name: str, default: Any = None) -> Any:
    """Read a field from a mapping or a normal record object."""
    if isinstance(record, Mapping):
        return record.get(field_name, default)

    # Deliberately do not call arbitrary methods.
    try:
        return getattr(record, field_name)
    except (AttributeError, TypeError):
        return default


def request_from_unified_record(
    record: Any,
    mode: AnalysisMode = AnalysisMode.GENERAL,
    requested_outputs: Optional[List[str]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> EngineRequest:
    """
    Build a request from an existing UnifiedDataRecord-like object.

    The function only reads declared metadata fields. It does not
    download, open, or copy raw scientific data.

    Adapt the field mapping if the project's actual data_schema.py
    uses different field names.
    """
    if record is None:
        raise TypeError("record must not be None.")

    record_id = _read_field(record, "record_id", "")
    source = _read_field(record, "source", "UNKNOWN")
    payload_ref = _read_field(record, "payload_ref", None)
    provenance = _read_field(record, "provenance", {})
    metadata = _read_field(record, "metadata", {})

    if not isinstance(metadata, Mapping):
        metadata = {}

    if not isinstance(provenance, Mapping):
        provenance = {}

    dataset_id = (
        metadata.get("dataset_id")
        or record_id
        or ""
    )

    source_type = (
        metadata.get("source_type")
        or source
        or "UNKNOWN"
    )

    file_path = metadata.get("file_path")
    source_uri = (
        metadata.get("source_uri")
        or payload_ref
    )

    raw_hash = (
        metadata.get("raw_data_hash")
        or provenance.get("raw_data_hash")
    )

    immutable = metadata.get("raw_data_immutable", True)

    input_reference = EngineInputReference(
        dataset_id=str(dataset_id),
        source_type=str(source_type),
        file_path=str(file_path) if file_path else None,
        source_uri=str(source_uri) if source_uri else None,
        raw_data_hash=raw_hash,
        raw_data_immutable=immutable,
        metadata={
            "record_id": str(record_id),
            "source": str(source),
            "modality": str(_read_field(record, "modality", "")),
            "data_type": str(_read_field(record, "data_type", "")),
            "schema_name": str(_read_field(record, "schema_name", "")),
            "schema_version": str(_read_field(record, "schema_version", "")),
        },
    )

    return EngineInterface().create_request(
        input_reference=input_reference,
        mode=mode,
        requested_outputs=requested_outputs,
        options=options,
        provenance={
            "created_by": MODULE_NAME,
            "created_at": utc_timestamp(),
            "upstream_provenance": _json_safe(dict(provenance)),
            "raw_data_modified": False,
        },
    )


# ------------------------------------------------------------
# 14. Response validation
# ------------------------------------------------------------

def validate_engine_response(
    response: EngineResponse,
) -> Dict[str, Any]:
    """Validate response structure at the public boundary."""
    if not isinstance(response, EngineResponse):
        return {
            "valid": False,
            "errors": ["response must be EngineResponse."],
            "warnings": [],
            "checked_at": utc_timestamp(),
        }

    errors = response.validate()
    warnings: List[str] = []

    if response.status == EngineStatus.COMPLETED:
        warnings.append(
            "COMPLETED indicates provider-reported completion only; "
            "scientific validity requires separate validation."
        )

    if not response.provenance:
        warnings.append("Response provenance is empty.")

    if response.status == EngineStatus.NOT_CONFIGURED:
        warnings.append("No scientific analysis was performed.")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "checked_at": utc_timestamp(),
    }


# ------------------------------------------------------------
# 15. Serialization helpers
# ------------------------------------------------------------

def request_to_dict(request: EngineRequest) -> Dict[str, Any]:
    return request.to_dict()


def request_to_json(request: EngineRequest, indent: int = 2) -> str:
    return request.to_json(indent=indent)


def response_to_dict(response: EngineResponse) -> Dict[str, Any]:
    return response.to_dict()


def response_to_json(response: EngineResponse, indent: int = 2) -> str:
    return response.to_json(indent=indent)


# ------------------------------------------------------------
# 16. Public interface contract
# ------------------------------------------------------------

def interface_contract() -> Dict[str, Any]:
    """Return a machine-readable description of the public contract."""
    return {
        "module_number": MODULE_NUMBER,
        "interface_name": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "interface_version": INTERFACE_VERSION,
        "architecture_role": "PUBLIC_TO_PRIVATE_BOUNDARY",
        "black_box": True,
        "private_core_included": PRIVATE_CORE_INCLUDED,
        "raw_data_modification_allowed": RAW_DATA_MODIFICATION_ALLOWED,
        "medical_diagnosis_supported": MEDICAL_DIAGNOSIS_SUPPORTED,
        "flight_certification_supported": FLIGHT_CERTIFICATION_SUPPORTED,
        "downloads_external_data": False,
        "runs_analysis_without_provider": False,
        "request_fields": [
            "request_id",
            "interface_version",
            "mode",
            "input_reference",
            "requested_outputs",
            "options",
            "provenance",
            "created_at",
        ],
        "input_reference_fields": [
            "dataset_id",
            "source_type",
            "file_path",
            "source_uri",
            "raw_data_hash",
            "raw_data_immutable",
            "metadata",
        ],
        "response_fields": [
            "request_id",
            "interface_version",
            "status",
            "outputs",
            "messages",
            "warnings",
            "errors",
            "execution_metadata",
            "provenance",
            "completed_at",
        ],
        "supported_modes": [mode.value for mode in AnalysisMode],
        "supported_statuses": [status.value for status in EngineStatus],
    }


# ------------------------------------------------------------
# 17. Interface-only self-test
# ------------------------------------------------------------

def run_interface_contract_test() -> Dict[str, Any]:
    """
    Test the interface contract without fabricating a scientific
    dataset and without claiming NASA data has been analyzed.

    This test checks enums, contract structure, serialization
    utilities, and rejection of an invalid empty input reference.
    """
    contract = interface_contract()

    assert contract["module_number"] == 11
    assert contract["private_core_included"] is False
    assert contract["raw_data_modification_allowed"] is False
    assert contract["downloads_external_data"] is False
    assert contract["runs_analysis_without_provider"] is False

    assert AnalysisMode.NASA.value == "NASA"
    assert EngineStatus.NOT_CONFIGURED.value == "NOT_CONFIGURED"

    invalid_reference = EngineInputReference(
        dataset_id="",
        source_type="",
    )
    invalid_errors = invalid_reference.validate()
    assert len(invalid_errors) >= 3

    # Verify deterministic hashing of a contract fragment.
    hash_one = calculate_object_hash({
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
    })
    hash_two = calculate_object_hash({
        "version": MODULE_VERSION,
        "module": MODULE_NAME,
    })
    assert hash_one == hash_two
    assert len(hash_one) == 64

    return {
        "passed": True,
        "test_type": "INTERFACE_CONTRACT_ONLY",
        "module_number": MODULE_NUMBER,
        "module": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "interface_version": INTERFACE_VERSION,
        "invalid_reference_rejected": bool(invalid_errors),
        "deterministic_hash_verified": hash_one == hash_two,
        "real_scientific_data_loaded": False,
        "scientific_analysis_performed": False,
        "private_core_executed": False,
        "message": (
            "Interface contract checks passed. "
            "No scientific dataset was fabricated or analyzed."
        ),
    }


# ------------------------------------------------------------
# 18. Real-data request test
# ------------------------------------------------------------

def run_real_data_request_test(
    input_reference: EngineInputReference,
    mode: AnalysisMode = AnalysisMode.NASA,
) -> Dict[str, Any]:
    """
    Validate a reference supplied by a real-data adapter.

    This function does not download the data and does not assert
    that the referenced file/URI exists. The upstream adapter must
    verify access and dataset integrity.

    With the default NULL provider, no scientific analysis occurs.
    """
    if not isinstance(input_reference, EngineInputReference):
        raise TypeError(
            "input_reference must be an EngineInputReference "
            "created from a real dataset."
        )

    errors = input_reference.validate()
    if errors:
        return {
            "passed": False,
            "test_type": "REAL_DATA_REFERENCE_VALIDATION",
            "errors": errors,
            "scientific_analysis_performed": False,
        }

    interface = EngineInterface()
    request = interface.create_request(
        input_reference=input_reference,
        mode=mode,
        requested_outputs=[
            "METRIC",
            "FEATURE",
            "QUALITY_INDICATOR",
            "UNCERTAINTY",
        ],
        options={
            "require_provenance": True,
            "preserve_raw_data": True,
        },
        provenance={
            "request_origin": "REAL_DATA_ADAPTER",
            "created_at": utc_timestamp(),
        },
    )

    response = interface.execute(request)
    validation = validate_engine_response(response)

    return {
        "passed": validation["valid"],
        "test_type": "REAL_DATA_INTERFACE_REQUEST",
        "dataset_id": input_reference.dataset_id,
        "source_type": input_reference.source_type,
        "request_id": request.request_id,
        "request_fingerprint": request.fingerprint(),
        "response_status": response.status.value,
        "response_validation": validation,
        "scientific_analysis_performed": (
            response.execution_metadata.get(
                "execution_performed", False
            )
        ),
        "provider_health": interface.health_check(),
        "message": (
            "Real-data reference was checked at the interface layer. "
            "Check response status before claiming analysis."
        ),
    }


# ------------------------------------------------------------
# 19. Module information
# ------------------------------------------------------------

def module_info() -> Dict[str, Any]:
    return {
        "module_number": MODULE_NUMBER,
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "interface_version": INTERFACE_VERSION,
        "role": "BLACK_BOX_ENGINE_INTERFACE",
        "private_core_included": PRIVATE_CORE_INCLUDED,
        "raw_data_modification_allowed": RAW_DATA_MODIFICATION_ALLOWED,
        "medical_diagnosis_supported": MEDICAL_DIAGNOSIS_SUPPORTED,
        "flight_certification_supported": FLIGHT_CERTIFICATION_SUPPORTED,
        "security_principles": [
            "No eval",
            "No exec",
            "No arbitrary code execution",
            "No private source loading",
            "No proprietary algorithm exposure",
            "No raw input modification",
            "No false analysis-completion claim",
        ],
    }


# ------------------------------------------------------------
# 20. Module entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    print(f"🟥 {MODULE_NAME}.py loaded (v{MODULE_VERSION})")
    print(f"🔢 Hackathon module number: {MODULE_NUMBER}")
    print(f"🔒 Interface version: {INTERFACE_VERSION}")
    print("🧠 Private intelligence core: NOT INCLUDED")
    print("🛡️ Raw-data modification: DISABLED")
    print("🌐 Automatic NASA download: NOT IMPLEMENTED")
    print("🚀 Black-box interface contract: READY")

    report = run_interface_contract_test()

    print("\n--- INTERFACE CONTRACT TEST ---")
    print(json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ))
