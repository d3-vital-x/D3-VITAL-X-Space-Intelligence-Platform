
"""
D3 VITAL-X Space Intelligence Platform
Module 02: Central Configuration

Public configuration only.
No proprietary v10/v11 source code or private parameters.
"""

from pathlib import Path
from typing import Final
import os


# ============================================================
# 1. PROJECT IDENTITY
# ============================================================

PROJECT_NAME: Final[str] = "D3 VITAL-X Space Intelligence Platform"
PROJECT_VERSION: Final[str] = "0.1.0"
PROJECT_TAGLINE: Final[str] = (
    "One AI Engine. Multiple Worlds. One Signal Language."
)
PROJECT_STATUS: Final[str] = "ACTIVE DEVELOPMENT"
ORGANIZATION: Final[str] = "D3 VITAL-X BANGLADESH"


# ============================================================
# 2. PORTABLE PROJECT PATHS
# ============================================================

def _resolve_project_root() -> Path:
    """
    Resolve the repository root in GitHub and Google Colab.

    Optional override:
        D3_PROJECT_ROOT=/path/to/repository
    """
    override = os.getenv("D3_PROJECT_ROOT", "").strip()

    if override:
        return Path(override).expanduser().resolve()

    config_file = globals().get("__file__")

    if config_file:
        config_path = Path(config_file).resolve()

        if config_path.parent.name == "02_DASHBOARD":
            return config_path.parent.parent

        # Standalone Colab script or copied config.py
        return config_path.parent

    # Fallback for interactive notebook execution
    return Path.cwd().resolve()


PROJECT_ROOT: Final[Path] = _resolve_project_root()

DASHBOARD_DIR: Final[Path] = PROJECT_ROOT / "02_DASHBOARD"
INPUT_ADAPTERS_DIR: Final[Path] = PROJECT_ROOT / "03_INPUT_ADAPTERS"
UNIFIED_DATA_DIR: Final[Path] = PROJECT_ROOT / "04_UNIFIED_DATA_LAYER"
ENGINE_INTERFACE_DIR: Final[Path] = (
    PROJECT_ROOT / "05_FEATURE_ENGINE_INTERFACE"
)
ANALYTICS_DIR: Final[Path] = PROJECT_ROOT / "06_ANALYTICS"
VISUALIZATION_DIR: Final[Path] = PROJECT_ROOT / "07_VISUALIZATION"
VALIDATION_DIR: Final[Path] = PROJECT_ROOT / "08_VALIDATION"
LIVE_MODE_DIR: Final[Path] = PROJECT_ROOT / "09_LIVE_MODE"
EXPORT_DIR: Final[Path] = PROJECT_ROOT / "10_EXPORT"
DATA_DIR: Final[Path] = PROJECT_ROOT / "11_DATA"
TESTS_DIR: Final[Path] = PROJECT_ROOT / "13_TESTS"


# ============================================================
# 3. ARCHITECTURE BOUNDARY
# ============================================================

LOGICAL_COMPONENT_COUNT: Final[int] = 30
PUBLIC_COMPONENT_COUNT: Final[int] = 29
PROTECTED_COMPONENT_COUNT: Final[int] = 1

PUBLIC_INTERFACE_ENABLED: Final[bool] = True
PROTECTED_CORE_SOURCE_INCLUDED: Final[bool] = False

RAW_DATA_IMMUTABLE: Final[bool] = True

REMOTE_ENGINE_ENABLED: Final[bool] = (
    os.getenv("D3_REMOTE_ENGINE_ENABLED", "false")
    .strip()
    .lower()
    in {"1", "true", "yes"}
)


# ============================================================
# 4. DASHBOARD MODES
# ============================================================

SUPPORTED_MODES: Final[tuple[str, ...]] = (
    "HOME",
    "UPLOAD",
    "DATA_QUALITY",
    "VISUALIZATION",
    "SPACE",
    "BIOMEDICAL",
    "LIVE",
    "VALIDATION",
    "COMPARISON",
    "ABOUT",
)


# ============================================================
# 5. INPUT TYPES AND EXTENSIONS
# ============================================================

SUPPORTED_INPUT_TYPES: Final[tuple[str, ...]] = (
    "CSV",
    "IMAGE",
    "TIME_SERIES",
    "FITS",
    "DICOM",
    "VIDEO",
    "TEXT",
    "BINARY_DATA",
)

SUPPORTED_EXTENSIONS: Final[dict[str, tuple[str, ...]]] = {
    "CSV": (".csv",),

    "IMAGE": (
        ".png", ".jpg", ".jpeg", ".bmp",
        ".tif", ".tiff", ".webp",
    ),

    "TIME_SERIES": (".csv", ".txt", ".dat"),

    "FITS": (".fits", ".fit", ".fts"),

    "DICOM": (".dcm", ".dicom"),

    "VIDEO": (
        ".mp4", ".avi", ".mov", ".mkv", ".webm",
    ),

    "TEXT": (".txt", ".dat"),

    "BINARY_DATA": (".bin", ".dat"),
}

# Formats exposed by the general-purpose upload page.
# Specialized adapters may support additional formats.
DASHBOARD_UPLOAD_EXTENSIONS: Final[tuple[str, ...]] = (
    ".csv",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".txt",
    ".dat",
)


def get_input_type(filename: str) -> str | None:
    """Return a matching input category based on file extension."""
    suffix = Path(filename).suffix.lower()

    matches = [
        category
        for category, extensions in SUPPORTED_EXTENSIONS.items()
        if suffix in extensions
    ]

    # .dat can belong to multiple categories.
    # Do not guess when the extension is ambiguous.
    if len(matches) != 1:
        return None

    return matches[0]


# ============================================================
# 6. UPLOAD SIZE LIMIT
# ============================================================

def _read_upload_limit() -> int:
    raw_value = os.getenv("D3_MAX_UPLOAD_SIZE_MB", "200").strip()

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            "D3_MAX_UPLOAD_SIZE_MB must be a positive integer."
        ) from exc

    if value <= 0:
        raise ValueError(
            "D3_MAX_UPLOAD_SIZE_MB must be greater than zero."
        )

    return value


MAX_UPLOAD_SIZE_MB: Final[int] = _read_upload_limit()

# Binary-byte convention: 200 MiB = 209,715,200 bytes.
MAX_UPLOAD_SIZE_BYTES: Final[int] = (
    MAX_UPLOAD_SIZE_MB * 1024 * 1024
)

UPLOAD_SIZE_LABEL: Final[str] = (
    f"{MAX_UPLOAD_SIZE_MB} MiB maximum per file"
)

STREAMLIT_MAX_UPLOAD_SIZE_MB: Final[int] = MAX_UPLOAD_SIZE_MB


def validate_file_size(file_size_bytes: int) -> bool:
    """Check file size in bytes. This does not inspect file contents."""
    if isinstance(file_size_bytes, bool) or not isinstance(
        file_size_bytes, int
    ):
        raise TypeError("File size must be an integer in bytes.")

    if file_size_bytes < 0:
        raise ValueError("File size cannot be negative.")

    if file_size_bytes > MAX_UPLOAD_SIZE_BYTES:
        raise ValueError(
            f"File exceeds the configured "
            f"{MAX_UPLOAD_SIZE_MB} MiB limit."
        )

    return True


# ============================================================
# 7. SCIENTIFIC CLAIM CLASSIFICATION
# ============================================================

CLAIM_LEVELS: Final[dict[str, str]] = {
    "C1": "Directly Measured",
    "C2": "Computational Observation",
    "C3": "Hypothesis or Future Application",
}


# ============================================================
# 8. RESEARCH SAFETY BOUNDARIES
# ============================================================

CLINICAL_DIAGNOSIS_ENABLED: Final[bool] = False
TREATMENT_RECOMMENDATION_ENABLED: Final[bool] = False
FLIGHT_CERTIFICATION_CLAIM: Final[bool] = False
AUTOMATIC_DISCOVERY_CLAIM: Final[bool] = False

HUMAN_REVIEW_REQUIRED: Final[bool] = True
PRESERVE_RAW_INPUT: Final[bool] = True


# ============================================================
# 9. QUALITY CONTROL
# ============================================================

QC_ENABLED: Final[bool] = True
QC_REJECT_NEGATIVE_FILE_SIZE: Final[bool] = True
QC_RECORD_PROVENANCE: Final[bool] = True
QC_COMPUTE_SHA256: Final[bool] = True

QC_MODIFY_RAW_INPUT: Final[bool] = False


# ============================================================
# 10. ANALYTICS AND UNCERTAINTY
# ============================================================

PUBLIC_ANALYTICS_ENABLED: Final[bool] = True
UNCERTAINTY_REPORTING_ENABLED: Final[bool] = True
EXPLAINABILITY_ENABLED: Final[bool] = True

ANOMALY_SCORE_IS_DIAGNOSTIC: Final[bool] = False


# ============================================================
# 11. NASA, LIVE MODE AND REMOTE SERVICE
# ============================================================

LIVE_MODE_DEFAULT_ENABLED: Final[bool] = False
NASA_INTEGRATION_DEFAULT_ENABLED: Final[bool] = False

REMOTE_ENGINE_TIMEOUT_SECONDS: Final[int] = 30

REMOTE_ENGINE_API_URL: Final[str] = os.getenv(
    "D3_REMOTE_ENGINE_API_URL", ""
).strip()

# Never store API keys or private model parameters in this file.


# ============================================================
# 12. EXPORT SETTINGS
# ============================================================

SUPPORTED_EXPORT_FORMATS: Final[tuple[str, ...]] = (
    "CSV",
    "JSON",
)

EXPORT_INCLUDE_PROVENANCE: Final[bool] = True
EXPORT_INCLUDE_CLAIM_LEVEL: Final[bool] = True
EXPORT_INCLUDE_UNCERTAINTY: Final[bool] = True


# ============================================================
# 13. CONFIGURATION SELF-CHECK
# ============================================================

def validate_config() -> dict[str, object]:
    """Validate configuration invariants, not scientific performance."""

    errors: list[str] = []

    if not PROJECT_NAME.strip():
        errors.append("PROJECT_NAME must not be empty.")

    if MAX_UPLOAD_SIZE_MB <= 0:
        errors.append("Upload limit must be positive.")

    if LOGICAL_COMPONENT_COUNT != (
        PUBLIC_COMPONENT_COUNT + PROTECTED_COMPONENT_COUNT
    ):
        errors.append("Component counts do not add up.")

    if not PUBLIC_INTERFACE_ENABLED:
        errors.append("Public interface must be enabled.")

    if PROTECTED_CORE_SOURCE_INCLUDED:
        errors.append("Protected core source must not be included.")

    if CLINICAL_DIAGNOSIS_ENABLED:
        errors.append("Clinical diagnosis must remain disabled.")

    if TREATMENT_RECOMMENDATION_ENABLED:
        errors.append("Treatment recommendations must remain disabled.")

    if FLIGHT_CERTIFICATION_CLAIM:
        errors.append("Flight certification must not be claimed.")

    if QC_MODIFY_RAW_INPUT or not RAW_DATA_IMMUTABLE:
        errors.append("Raw input preservation invariant failed.")

    if not set(SUPPORTED_INPUT_TYPES).issuperset(
        SUPPORTED_EXTENSIONS.keys()
    ):
        errors.append("Input extension registry contains unknown types.")

    if not DASHBOARD_UPLOAD_EXTENSIONS:
        errors.append("Dashboard upload extension list is empty.")

    if REMOTE_ENGINE_TIMEOUT_SECONDS <= 0:
        errors.append("Remote engine timeout must be positive.")

    if errors:
        raise ValueError(
            "Configuration validation failed: "
            + " | ".join(errors)
        )

    return {
        "status": "PASS",
        "project_name": PROJECT_NAME,
        "version": PROJECT_VERSION,
        "project_root": str(PROJECT_ROOT),
        "max_upload_size_mb": MAX_UPLOAD_SIZE_MB,
        "max_upload_size_bytes": MAX_UPLOAD_SIZE_BYTES,
        "logical_components": LOGICAL_COMPONENT_COUNT,
        "public_components": PUBLIC_COMPONENT_COUNT,
        "protected_components": PROTECTED_COMPONENT_COUNT,
        "clinical_diagnosis_enabled": CLINICAL_DIAGNOSIS_ENABLED,
        "raw_data_immutable": RAW_DATA_IMMUTABLE,
        "human_review_required": HUMAN_REVIEW_REQUIRED,
    }


if __name__ == "__main__":
    for key, value in validate_config().items():
        print(f"{key}: {value}")
