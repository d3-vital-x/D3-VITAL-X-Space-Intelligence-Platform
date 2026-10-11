
# ============================================================
# D³ VITAL-X SPACE INTELLIGENCE PLATFORM
# Module 10 — feature_schema.py
#
# PUBLIC FEATURE CONTRACT
#
# Purpose:
# Define a controlled, machine-readable schema for features
# exchanged between the Black-Box Engine Interface and public
# analytics, validation, and visualization layers.
#
# Public:
# - Feature identity and category
# - Value type and units
# - Quality and uncertainty metadata
# - Source references and provenance
# - Validation and deterministic fingerprints
# - Feature registry and batch serialization
#
# Private:
# - Proprietary feature-generation equations
# - Internal coefficients and parameters
# - Optimization and MCMC implementation
# - Intelligence-core algorithms
#
# Safety:
# - No eval()
# - No exec()
# - No arbitrary code execution
# - No medical diagnosis
# - No flight certification
# - No automatic scientific interpretation
# - No raw-data modification
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)
import hashlib
import json
import math
import uuid


# ------------------------------------------------------------
# 1. Module metadata
# ------------------------------------------------------------

MODULE_NUMBER = 10
MODULE_NAME = "feature_schema"
MODULE_VERSION = "1.0.0"
FEATURE_SCHEMA_VERSION = "1.0"

PROPRIETARY_ALGORITHMS_INCLUDED = False
RAW_DATA_MODIFICATION_ALLOWED = False
MEDICAL_DIAGNOSIS_SUPPORTED = False
FLIGHT_CERTIFICATION_SUPPORTED = False


# ------------------------------------------------------------
# 2. Utility functions
# ------------------------------------------------------------

def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def generate_feature_id() -> str:
    """Generate a unique feature instance ID."""
    return "FEAT-" + uuid.uuid4().hex[:16].upper()


def generate_batch_id() -> str:
    """Generate a unique feature batch ID."""
    return "BATCH-" + uuid.uuid4().hex[:16].upper()


def _json_default(value: Any) -> Any:
    """Convert supported non-JSON values into safe representations."""
    if isinstance(value, Enum):
        return value.value

    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass

    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return value.tolist()
        except Exception:
            pass

    return str(value)


def canonical_json(payload: Any) -> str:
    """
    Create a deterministic JSON representation.

    This is a serialization helper, not an algorithm executor.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
        allow_nan=False,
    )


def calculate_sha256(
    payload: Union[str, bytes, bytearray],
) -> str:
    """Calculate the SHA-256 digest of serialized data."""
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    elif isinstance(payload, bytearray):
        payload = bytes(payload)

    if not isinstance(payload, bytes):
        raise TypeError("payload must be str, bytes, or bytearray.")

    return hashlib.sha256(payload).hexdigest()


def object_hash(payload: Any) -> str:
    """Calculate a deterministic hash of JSON-serializable data."""
    return calculate_sha256(canonical_json(payload))


def is_finite_number(value: Any) -> bool:
    """Return True for finite integers/floats, excluding booleans."""
    if isinstance(value, bool):
        return False

    if not isinstance(value, (int, float)):
        return False

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _enum_value(value: Any) -> Any:
    """Return an enum's public value without changing other objects."""
    return value.value if isinstance(value, Enum) else value


# ------------------------------------------------------------
# 3. Feature categories
# ------------------------------------------------------------

class FeatureCategory(str, Enum):
    """Public categories describing feature types."""
    SIGNAL = "SIGNAL"
    STATISTICAL = "STATISTICAL"
    TEMPORAL = "TEMPORAL"
    SPECTRAL = "SPECTRAL"
    SPATIAL = "SPATIAL"
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"
    ENTROPY = "ENTROPY"
    VARIANCE = "VARIANCE"
    GRADIENT = "GRADIENT"
    COUPLING = "COUPLING"
    TRANSITION = "TRANSITION"
    ANOMALY = "ANOMALY"
    QUALITY = "QUALITY"
    UNCERTAINTY = "UNCERTAINTY"
    METADATA = "METADATA"
    CUSTOM = "CUSTOM"


# ------------------------------------------------------------
# 4. Feature value types
# ------------------------------------------------------------

class FeatureValueType(str, Enum):
    """Machine-readable value containers."""
    SCALAR = "SCALAR"
    VECTOR = "VECTOR"
    MATRIX = "MATRIX"
    SERIES = "SERIES"
    CATEGORICAL = "CATEGORICAL"
    BOOLEAN = "BOOLEAN"
    TEXT = "TEXT"
    REFERENCE = "REFERENCE"
    NULL = "NULL"


# ------------------------------------------------------------
# 5. Feature interpretation classes
# ------------------------------------------------------------

class FeatureInterpretation(str, Enum):
    """Conservative interface-level interpretation labels."""
    RAW_DERIVED = "RAW_DERIVED"
    COMPUTATIONAL = "COMPUTATIONAL"
    QUALITY_INDICATOR = "QUALITY_INDICATOR"
    ANOMALY_INDICATOR = "ANOMALY_INDICATOR"
    TRANSITION_INDICATOR = "TRANSITION_INDICATOR"
    UNCERTAINTY_INDICATOR = "UNCERTAINTY_INDICATOR"
    DESCRIPTIVE = "DESCRIPTIVE"
    REFERENCE_ONLY = "REFERENCE_ONLY"
    UNINTERPRETED = "UNINTERPRETED"


# ------------------------------------------------------------
# 6. Feature quality states
# ------------------------------------------------------------

class FeatureQuality(str, Enum):
    """Technical quality status of a feature."""
    UNKNOWN = "UNKNOWN"
    VALID = "VALID"
    QUESTIONABLE = "QUESTIONABLE"
    INVALID = "INVALID"
    NOT_AVAILABLE = "NOT_AVAILABLE"


# ------------------------------------------------------------
# 7. Claim classification
# ------------------------------------------------------------

class ClaimClass(str, Enum):
    """
    C1: Directly observed/measured information.
    C2: Computationally derived output.
    C3: Hypothesis or research interpretation.

    These labels classify claims; they do not establish truth.
    """
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"


# ------------------------------------------------------------
# 8. Feature definition
# ------------------------------------------------------------

@dataclass
class FeatureDefinition:
    """Public definition of a feature interface."""

    name: str
    category: FeatureCategory
    value_type: FeatureValueType

    interpretation: FeatureInterpretation = (
        FeatureInterpretation.UNINTERPRETED
    )
    unit: Optional[str] = None
    description: str = ""
    claim_class: ClaimClass = ClaimClass.C2
    human_review_required: bool = True
    nullable: bool = True
    version: str = FEATURE_SCHEMA_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Validate the feature definition."""
        errors: List[str] = []

        if not isinstance(self.name, str) or not self.name.strip():
            errors.append("name must be a non-empty string.")

        enum_fields = {
            "category": (self.category, FeatureCategory),
            "value_type": (self.value_type, FeatureValueType),
            "interpretation": (
                self.interpretation,
                FeatureInterpretation,
            ),
            "claim_class": (self.claim_class, ClaimClass),
        }

        for field_name, (value, expected_type) in enum_fields.items():
            if not isinstance(value, expected_type):
                errors.append(
                    f"{field_name} must be {expected_type.__name__}."
                )

        if not isinstance(self.human_review_required, bool):
            errors.append("human_review_required must be boolean.")

        if not isinstance(self.nullable, bool):
            errors.append("nullable must be boolean.")

        if self.unit is not None and not isinstance(self.unit, str):
            errors.append("unit must be a string or None.")

        if not isinstance(self.description, str):
            errors.append("description must be a string.")

        if not isinstance(self.version, str) or not self.version.strip():
            errors.append("version must be a non-empty string.")

        if not isinstance(self.metadata, dict):
            errors.append("metadata must be a dictionary.")

        return errors


# ------------------------------------------------------------
# 9. Feature value
# ------------------------------------------------------------

@dataclass
class FeatureValue:
    """Container for one feature value; does not generate features."""

    feature_id: str
    definition: FeatureDefinition
    value: Any

    quality: FeatureQuality = FeatureQuality.UNKNOWN
    confidence: Optional[float] = None
    uncertainty: Optional[Any] = None
    source_dataset_id: Optional[str] = None
    source_hash: Optional[str] = None
    timestamp: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Validate this feature container."""
        errors: List[str] = []

        if not isinstance(self.feature_id, str) or not self.feature_id.strip():
            errors.append("feature_id must be a non-empty string.")

        if not isinstance(self.definition, FeatureDefinition):
            errors.append("definition must be FeatureDefinition.")
            return errors

        errors.extend(
            f"definition: {message}"
            for message in self.definition.validate()
        )

        if not isinstance(self.quality, FeatureQuality):
            errors.append("quality must be FeatureQuality.")

        value_type = self.definition.value_type

        if self.value is None:
            if not self.definition.nullable:
                errors.append("Feature value cannot be null.")

        elif value_type == FeatureValueType.NULL:
            errors.append("NULL value type requires value=None.")

        elif value_type == FeatureValueType.SCALAR:
            if not is_finite_number(self.value):
                errors.append(
                    "SCALAR value must be a finite integer or float."
                )

        elif value_type == FeatureValueType.BOOLEAN:
            if not isinstance(self.value, bool):
                errors.append("BOOLEAN value must be bool.")

        elif value_type == FeatureValueType.TEXT:
            if not isinstance(self.value, str):
                errors.append("TEXT value must be a string.")

        elif value_type == FeatureValueType.CATEGORICAL:
            if not isinstance(self.value, (str, int, float, bool)):
                errors.append(
                    "CATEGORICAL value must be a string or primitive scalar."
                )
            elif isinstance(self.value, float) and not math.isfinite(self.value):
                errors.append("CATEGORICAL float value must be finite.")

        elif value_type == FeatureValueType.REFERENCE:
            if not isinstance(self.value, str) or not self.value.strip():
                errors.append(
                    "REFERENCE value must be a non-empty string."
                )

        elif value_type in (
            FeatureValueType.VECTOR,
            FeatureValueType.MATRIX,
            FeatureValueType.SERIES,
        ):
            if not isinstance(self.value, (list, tuple)):
                errors.append(
                    f"{value_type.value} value must be a list or tuple."
                )

        if self.confidence is not None:
            if not is_finite_number(self.confidence):
                errors.append("confidence must be a finite number.")
            elif not 0.0 <= float(self.confidence) <= 1.0:
                errors.append("confidence must be between 0 and 1.")

        if self.source_hash is not None:
            if not isinstance(self.source_hash, str):
                errors.append("source_hash must be a string.")
            elif len(self.source_hash) != 64:
                errors.append(
                    "source_hash must contain 64 hexadecimal characters."
                )
            elif any(
                character not in "0123456789abcdefABCDEF"
                for character in self.source_hash
            ):
                errors.append("source_hash contains invalid hexadecimal characters.")

        if self.source_dataset_id is not None:
            if not isinstance(self.source_dataset_id, str):
                errors.append("source_dataset_id must be a string or None.")

        if self.timestamp is not None and not isinstance(self.timestamp, str):
            errors.append("timestamp must be a string or None.")

        if not isinstance(self.provenance, dict):
            errors.append("provenance must be a dictionary.")

        if not isinstance(self.metadata, dict):
            errors.append("metadata must be a dictionary.")

        return errors

    def fingerprint(self) -> str:
        """
        Hash the feature container's public representation.

        This does not identify or fingerprint a proprietary algorithm.
        """
        payload = {
            "feature_id": self.feature_id,
            "definition": asdict(self.definition),
            "value": self.value,
            "quality": _enum_value(self.quality),
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "source_dataset_id": self.source_dataset_id,
            "source_hash": self.source_hash,
            "timestamp": self.timestamp,
        }
        return object_hash(payload)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return json.loads(canonical_json(asdict(self)))


# ------------------------------------------------------------
# 10. Feature batch
# ------------------------------------------------------------

@dataclass
class FeatureBatch:
    """Collection of features for one dataset or analysis request."""

    batch_id: str
    dataset_id: str
    features: List[FeatureValue] = field(default_factory=list)
    schema_version: str = FEATURE_SCHEMA_VERSION
    created_at: str = field(default_factory=utc_timestamp)
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Validate the complete feature batch."""
        errors: List[str] = []

        if not isinstance(self.batch_id, str) or not self.batch_id.strip():
            errors.append("batch_id must be a non-empty string.")

        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            errors.append("dataset_id must be a non-empty string.")

        if not isinstance(self.schema_version, str):
            errors.append("schema_version must be a string.")

        if not isinstance(self.created_at, str):
            errors.append("created_at must be a string.")

        if not isinstance(self.provenance, dict):
            errors.append("provenance must be a dictionary.")

        if not isinstance(self.metadata, dict):
            errors.append("metadata must be a dictionary.")

        if not isinstance(self.features, list):
            errors.append("features must be a list.")
            return errors

        seen_feature_ids = set()

        for index, feature in enumerate(self.features):
            if not isinstance(feature, FeatureValue):
                errors.append(
                    f"features[{index}] must be FeatureValue."
                )
                continue

            errors.extend(
                f"features[{index}]: {message}"
                for message in feature.validate()
            )

            if feature.feature_id in seen_feature_ids:
                errors.append(
                    f"Duplicate feature_id: {feature.feature_id}"
                )

            seen_feature_ids.add(feature.feature_id)

        return errors

    def feature_names(self) -> List[str]:
        """Return names of valid FeatureValue objects."""
        return [
            feature.definition.name
            for feature in self.features
            if isinstance(feature, FeatureValue)
        ]

    def get(self, name: str) -> Optional[FeatureValue]:
        """Return the first feature matching a name."""
        for feature in self.features:
            if (
                isinstance(feature, FeatureValue)
                and feature.definition.name == name
            ):
                return feature
        return None

    def fingerprint(self) -> str:
        """Calculate a deterministic hash of the batch contents."""
        payload = {
            "batch_id": self.batch_id,
            "dataset_id": self.dataset_id,
            "schema_version": self.schema_version,
            "features": [
                {
                    "feature_id": feature.feature_id,
                    "name": feature.definition.name,
                    "value": feature.value,
                    "quality": _enum_value(feature.quality),
                    "source_hash": feature.source_hash,
                }
                for feature in self.features
                if isinstance(feature, FeatureValue)
            ],
        }
        return object_hash(payload)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return json.loads(canonical_json(asdict(self)))

    def to_json(self, indent: int = 2) -> str:
        """Serialize the batch as JSON."""
        if not isinstance(indent, int) or indent < 0:
            raise ValueError("indent must be a non-negative integer.")

        return json.dumps(
            self.to_dict(),
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )


# ------------------------------------------------------------
# 11. Feature registry
# ------------------------------------------------------------

class FeatureRegistry:
    """
    Public registry of feature definitions.

    Registry entries define interface contracts, not algorithms.
    """

    def __init__(self) -> None:
        self._definitions: Dict[str, FeatureDefinition] = {}

    def register(self, definition: FeatureDefinition) -> None:
        """Validate and register a feature definition."""
        if not isinstance(definition, FeatureDefinition):
            raise TypeError("definition must be FeatureDefinition.")

        errors = definition.validate()
        if errors:
            raise ValueError(
                "Invalid feature definition: " + "; ".join(errors)
            )

        if definition.name in self._definitions:
            raise ValueError(
                f"Feature already registered: {definition.name}"
            )

        self._definitions[definition.name] = definition

    def get(self, name: str) -> Optional[FeatureDefinition]:
        return self._definitions.get(name)

    def contains(self, name: str) -> bool:
        return name in self._definitions

    def names(self) -> List[str]:
        return sorted(self._definitions.keys())

    def definitions(self) -> List[FeatureDefinition]:
        return [
            self._definitions[name]
            for name in sorted(self._definitions)
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "features": [
                json.loads(canonical_json(asdict(definition)))
                for definition in self.definitions()
            ],
        }


# ------------------------------------------------------------
# 12. Standard public feature definitions
# ------------------------------------------------------------

def create_standard_registry() -> FeatureRegistry:
    """
    Create standard public feature definitions.

    Names and descriptions are interface labels only.
    They do not define the method used to generate a feature.
    """
    registry = FeatureRegistry()

    standard_definitions = [
        FeatureDefinition(
            name="entropy",
            category=FeatureCategory.ENTROPY,
            value_type=FeatureValueType.SCALAR,
            interpretation=FeatureInterpretation.COMPUTATIONAL,
            unit="bits",
            description="Entropy-related computational feature.",
        ),
        FeatureDefinition(
            name="variance",
            category=FeatureCategory.VARIANCE,
            value_type=FeatureValueType.SCALAR,
            interpretation=FeatureInterpretation.COMPUTATIONAL,
            description="Variance-related computational feature.",
        ),
        FeatureDefinition(
            name="gradient",
            category=FeatureCategory.GRADIENT,
            value_type=FeatureValueType.SCALAR,
            interpretation=FeatureInterpretation.COMPUTATIONAL,
            description="Gradient-related computational feature.",
        ),
        FeatureDefinition(
            name="coupling",
            category=FeatureCategory.COUPLING,
            value_type=FeatureValueType.SCALAR,
            interpretation=FeatureInterpretation.COMPUTATIONAL,
            description="Coupling-related output supplied by an engine interface.",
        ),
        FeatureDefinition(
            name="transition_index",
            category=FeatureCategory.TRANSITION,
            value_type=FeatureValueType.SCALAR,
            interpretation=FeatureInterpretation.TRANSITION_INDICATOR,
            description="Transition-related computational indicator.",
        ),
        FeatureDefinition(
            name="anomaly_score",
            category=FeatureCategory.ANOMALY,
            value_type=FeatureValueType.SCALAR,
            interpretation=FeatureInterpretation.ANOMALY_INDICATOR,
            description=(
                "Computational anomaly-prioritization indicator; "
                "not a diagnosis or proof of a scientific anomaly."
            ),
            human_review_required=True,
        ),
        FeatureDefinition(
            name="signal_quality",
            category=FeatureCategory.QUALITY,
            value_type=FeatureValueType.SCALAR,
            interpretation=FeatureInterpretation.QUALITY_INDICATOR,
            description="Technical input or signal quality indicator.",
            human_review_required=True,
        ),
        FeatureDefinition(
            name="confidence",
            category=FeatureCategory.UNCERTAINTY,
            value_type=FeatureValueType.SCALAR,
            interpretation=FeatureInterpretation.UNCERTAINTY_INDICATOR,
            description="Confidence value supplied by a computational component.",
            human_review_required=True,
        ),
        FeatureDefinition(
            name="uncertainty",
            category=FeatureCategory.UNCERTAINTY,
            value_type=FeatureValueType.SCALAR,
            interpretation=FeatureInterpretation.UNCERTAINTY_INDICATOR,
            description="Uncertainty value supplied by a computational component.",
            human_review_required=True,
        ),
    ]

    for definition in standard_definitions:
        registry.register(definition)

    return registry


# ------------------------------------------------------------
# 13. Feature factory
# ------------------------------------------------------------

def create_feature(
    definition: FeatureDefinition,
    value: Any,
    source_dataset_id: Optional[str] = None,
    source_hash: Optional[str] = None,
    quality: FeatureQuality = FeatureQuality.UNKNOWN,
    confidence: Optional[float] = None,
    uncertainty: Optional[Any] = None,
    timestamp: Optional[str] = None,
    provenance: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> FeatureValue:
    """Create and validate a FeatureValue."""
    if not isinstance(definition, FeatureDefinition):
        raise TypeError("definition must be FeatureDefinition.")

    feature = FeatureValue(
        feature_id=generate_feature_id(),
        definition=definition,
        value=value,
        quality=quality,
        confidence=confidence,
        uncertainty=uncertainty,
        source_dataset_id=source_dataset_id,
        source_hash=source_hash,
        timestamp=timestamp or utc_timestamp(),
        provenance=dict(provenance) if provenance is not None else {},
        metadata=dict(metadata) if metadata is not None else {},
    )

    errors = feature.validate()
    if errors:
        raise ValueError("Invalid feature: " + "; ".join(errors))

    return feature


# ------------------------------------------------------------
# 14. Feature batch factory
# ------------------------------------------------------------

def create_feature_batch(
    dataset_id: str,
    features: Optional[Sequence[FeatureValue]] = None,
    provenance: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> FeatureBatch:
    """Create and validate a FeatureBatch."""
    batch = FeatureBatch(
        batch_id=generate_batch_id(),
        dataset_id=dataset_id,
        features=list(features) if features is not None else [],
        provenance=dict(provenance) if provenance is not None else {},
        metadata=dict(metadata) if metadata is not None else {},
    )

    errors = batch.validate()
    if errors:
        raise ValueError("Invalid feature batch: " + "; ".join(errors))

    return batch


# ------------------------------------------------------------
# 15. Public payload validation
# ------------------------------------------------------------

def validate_feature_payload(payload: Any) -> Dict[str, Any]:
    """Validate a FeatureValue or FeatureBatch."""
    if isinstance(payload, FeatureValue):
        errors = payload.validate()
        payload_type = "FeatureValue"

    elif isinstance(payload, FeatureBatch):
        errors = payload.validate()
        payload_type = "FeatureBatch"

    else:
        errors = [
            "Payload must be FeatureValue or FeatureBatch."
        ]
        payload_type = type(payload).__name__

    return {
        "valid": len(errors) == 0,
        "type": payload_type,
        "errors": errors,
        "checked_at": utc_timestamp(),
    }


# ------------------------------------------------------------
# 16. Public feature schema contract
# ------------------------------------------------------------

def feature_schema_contract() -> Dict[str, Any]:
    """Return the machine-readable public schema contract."""
    return {
        "module_number": MODULE_NUMBER,
        "schema_name": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "schema_version": FEATURE_SCHEMA_VERSION,
        "proprietary_algorithms_included": (
            PROPRIETARY_ALGORITHMS_INCLUDED
        ),
        "raw_data_modification_allowed": (
            RAW_DATA_MODIFICATION_ALLOWED
        ),
        "medical_diagnosis_supported": MEDICAL_DIAGNOSIS_SUPPORTED,
        "flight_certification_supported": FLIGHT_CERTIFICATION_SUPPORTED,
        "feature_categories": [
            item.value for item in FeatureCategory
        ],
        "value_types": [
            item.value for item in FeatureValueType
        ],
        "interpretation_classes": [
            item.value for item in FeatureInterpretation
        ],
        "quality_states": [
            item.value for item in FeatureQuality
        ],
        "claim_classes": [
            item.value for item in ClaimClass
        ],
        "required_feature_fields": [
            "feature_id",
            "definition",
            "value",
            "quality",
            "confidence",
            "uncertainty",
            "source_dataset_id",
            "source_hash",
            "timestamp",
            "provenance",
        ],
        "required_definition_fields": [
            "name",
            "category",
            "value_type",
            "interpretation",
            "unit",
            "description",
            "claim_class",
            "human_review_required",
            "nullable",
            "version",
        ],
    }


# ------------------------------------------------------------
# 17. Local self-test
# ------------------------------------------------------------

def run_feature_schema_test() -> Dict[str, Any]:
    """
    Run local tests using synthetic fixtures only.

    This test does not access external datasets, cameras,
    network services, medical records, or the intelligence core.
    """
    registry = create_standard_registry()

    assert registry.contains("entropy")
    assert registry.contains("variance")
    assert registry.contains("anomaly_score")

    entropy_definition = registry.get("entropy")
    anomaly_definition = registry.get("anomaly_score")

    assert entropy_definition is not None
    assert anomaly_definition is not None

    test_hash = "b" * 64

    entropy_feature = create_feature(
        definition=entropy_definition,
        value=5.25,
        source_dataset_id="TEST-DATASET-001",
        source_hash=test_hash,
        quality=FeatureQuality.VALID,
        confidence=0.95,
        uncertainty=0.10,
        provenance={
            "test_fixture": True,
            "generation_method": "SYNTHETIC_TEST_FIXTURE",
            "algorithm_exposed": False,
        },
    )

    anomaly_feature = create_feature(
        definition=anomaly_definition,
        value=0.72,
        source_dataset_id="TEST-DATASET-001",
        source_hash=test_hash,
        quality=FeatureQuality.VALID,
        confidence=0.88,
        uncertainty=0.06,
        provenance={
            "test_fixture": True,
            "interpretation": "PRIORITIZATION_ONLY",
            "medical_diagnosis": False,
        },
    )

    entropy_validation = validate_feature_payload(entropy_feature)
    assert entropy_validation["valid"] is True

    batch = create_feature_batch(
        dataset_id="TEST-DATASET-001",
        features=[entropy_feature, anomaly_feature],
        provenance={
            "test_fixture": True,
            "proprietary_algorithm_exposed": False,
        },
    )

    batch_validation = validate_feature_payload(batch)
    assert batch_validation["valid"] is True

    feature_fingerprint = entropy_feature.fingerprint()
    batch_fingerprint = batch.fingerprint()

    assert len(feature_fingerprint) == 64
    assert len(batch_fingerprint) == 64

    batch_json = batch.to_json()
    assert isinstance(batch_json, str)

    # Negative tests: invalid values must be rejected.
    invalid_scalar_rejected = False
    try:
        create_feature(
            definition=entropy_definition,
            value=float("nan"),
        )
    except ValueError:
        invalid_scalar_rejected = True

    assert invalid_scalar_rejected is True

    invalid_confidence_rejected = False
    try:
        create_feature(
            definition=entropy_definition,
            value=1.0,
            confidence=1.5,
        )
    except ValueError:
        invalid_confidence_rejected = True

    assert invalid_confidence_rejected is True

    contract = feature_schema_contract()

    assert contract["proprietary_algorithms_included"] is False
    assert contract["raw_data_modification_allowed"] is False
    assert contract["medical_diagnosis_supported"] is False
    assert contract["flight_certification_supported"] is False

    return {
        "passed": True,
        "module_number": MODULE_NUMBER,
        "module": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "schema_version": FEATURE_SCHEMA_VERSION,
        "registry_feature_count": len(registry.names()),
        "registry_features": registry.names(),
        "batch_feature_count": len(batch.features),
        "feature_fingerprint": feature_fingerprint,
        "batch_fingerprint": batch_fingerprint,
        "entropy_validation": entropy_validation,
        "batch_validation": batch_validation,
        "invalid_scalar_rejected": invalid_scalar_rejected,
        "invalid_confidence_rejected": invalid_confidence_rejected,
        "proprietary_algorithm_exposed": False,
        "raw_data_modified": False,
        "external_data_accessed": False,
        "message": (
            "Feature schema self-test passed. "
            "Public feature contracts and validation checks "
            "passed using synthetic fixtures."
        ),
    }


# ------------------------------------------------------------
# 18. Module information
# ------------------------------------------------------------

def module_info() -> Dict[str, Any]:
    """Return public module metadata."""
    return {
        "module_number": MODULE_NUMBER,
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "schema_version": FEATURE_SCHEMA_VERSION,
        "role": "PUBLIC_FEATURE_CONTRACT",
        "proprietary_algorithms_included": (
            PROPRIETARY_ALGORITHMS_INCLUDED
        ),
        "raw_data_modification_allowed": (
            RAW_DATA_MODIFICATION_ALLOWED
        ),
        "medical_diagnosis_supported": MEDICAL_DIAGNOSIS_SUPPORTED,
        "flight_certification_supported": FLIGHT_CERTIFICATION_SUPPORTED,
        "security_principles": [
            "No eval",
            "No exec",
            "No arbitrary code execution",
            "No proprietary algorithm exposure",
            "No feature-generation implementation",
            "Synthetic local self-test",
        ],
    }


# ------------------------------------------------------------
# 19. Module load message
# ------------------------------------------------------------

if __name__ == "__main__":
    print(f"🟥 {MODULE_NAME}.py loaded (v{MODULE_VERSION})")
    print(f"🔢 Module number: {MODULE_NUMBER}")
    print(f"🔒 Feature schema: {FEATURE_SCHEMA_VERSION}")
    print("🧠 Proprietary feature-generation logic: NOT INCLUDED")
    print("🛡️ Raw-data modification: DISABLED")
    print("🩻 Medical diagnosis: NOT SUPPORTED")
    print("🚀 Public feature contract: READY")

    test_report = run_feature_schema_test()

    print("\n--- SELF-TEST REPORT ---")
    print(json.dumps(
        test_report,
        indent=2,
        ensure_ascii=False,
        default=_json_default,
    ))
