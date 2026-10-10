"""
D³ VITAL-X Space Intelligence Platform
Module 07: nasa_adapter.py
Version: 0.2.0

Purpose:
- Ingest NASA and other authorized space-data resources.
- Download and preserve raw response bytes.
- Inspect file metadata and JSON structure.
- Create records compatible with the current UnifiedDataRecord schema.
- Track SHA-256 hashes and provenance.

Boundaries:
- No scientific interpretation.
- No smoothing, clipping, interpolation, or normalization.
- No clinical diagnosis.
- No proprietary intelligence core.
- API keys are not hard-coded or intentionally returned in URLs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import mimetypes
import os
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union


MODULE_NAME = "nasa_adapter"
MODULE_VERSION = "0.2.0"
DEFAULT_TIMEOUT = 30
NASA_API_BASE = "https://api.nasa.gov"
SCHEMA_MODULE_NAME = "d3_vital_x_data_schema"

SUPPORTED_EXTENSIONS = {
    ".csv": "csv",
    ".json": "json",
    ".txt": "text",
    ".xml": "xml",
    ".fits": "fits",
    ".fit": "fits",
    ".fts": "fits",
    ".hdf5": "hdf5",
    ".h5": "hdf5",
    ".nc": "netcdf",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tif": "image",
    ".tiff": "image",
    ".webp": "image",
    ".mp4": "video",
    ".avi": "video",
}

FORMAT_TO_MODALITY = {
    "csv": "tabular",
    "json": "text",
    "text": "text",
    "xml": "text",
    "fits": "fits",
    "hdf5": "binary",
    "netcdf": "binary",
    "image": "image",
    "video": "video",
    "unknown": "unknown",
}


# ---------------------------------------------------------
# 1. Time, paths, hashes
# ---------------------------------------------------------

def utc_timestamp() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def project_root() -> Path:
    """Locate the project root using this module's location."""
    if "__file__" in globals():
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (
                (parent / "04_UNIFIED_DATA_LAYER" / "data_schema.py")
                .is_file()
            ):
                return parent

    fallback = Path(
        "/content/D3-VITAL-X-Space-Intelligence-Platform"
    )

    if (fallback / "04_UNIFIED_DATA_LAYER" / "data_schema.py").is_file():
        return fallback

    raise FileNotFoundError(
        "Could not locate project root and data_schema.py."
    )


def calculate_sha256(
    source: Union[str, Path, bytes, bytearray]
) -> str:
    """Calculate SHA-256 for bytes or a regular file."""
    digest = hashlib.sha256()

    if isinstance(source, (bytes, bytearray)):
        digest.update(bytes(source))
        return digest.hexdigest()

    path = Path(source)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def generate_dataset_id(source_name: str, raw_hash: str) -> str:
    """Create a deterministic dataset identifier."""
    seed = f"{source_name}|{raw_hash}"
    short_hash = hashlib.sha256(
        seed.encode("utf-8")
    ).hexdigest()[:16]

    return f"NASA-{short_hash}"


def safe_filename(name: str) -> str:
    """Remove path separators and unsafe filename characters."""
    name = str(name).replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = name.strip("._")

    if name in ("", ".", ".."):
        return "nasa_resource"

    return name[:180]


def _redact_url(url: str) -> str:
    """Redact API-key-like query parameters before reporting a URL."""
    parts = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(
        parts.query,
        keep_blank_values=True,
    )

    safe_query = [
        (key, "[REDACTED]" if key.lower() in {
            "api_key", "apikey", "key", "token", "access_token"
        } else value)
        for key, value in query
    ]

    return urllib.parse.urlunsplit((
        parts.scheme,
        parts.netloc,
        parts.path,
        urllib.parse.urlencode(safe_query),
        parts.fragment,
    ))


# ---------------------------------------------------------
# 2. Load current Unified Data Layer schema
# ---------------------------------------------------------

def load_schema_module():
    """Load the current schema without assuming sys.path setup."""
    schema_path = (
        project_root()
        / "04_UNIFIED_DATA_LAYER"
        / "data_schema.py"
    )

    existing = sys.modules.get(SCHEMA_MODULE_NAME)

    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if (
            existing_file
            and Path(existing_file).resolve() == schema_path.resolve()
        ):
            return existing

    spec = importlib.util.spec_from_file_location(
        SCHEMA_MODULE_NAME,
        str(schema_path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load schema: {schema_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[SCHEMA_MODULE_NAME] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(SCHEMA_MODULE_NAME, None)
        raise

    if not hasattr(module, "UnifiedDataRecord"):
        raise AttributeError(
            "Current data_schema.py does not define UnifiedDataRecord."
        )

    return module


# ---------------------------------------------------------
# 3. Format detection
# ---------------------------------------------------------

def detect_format(
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
) -> str:
    """Detect a basic file format using extension and MIME metadata."""
    if filename:
        extension = Path(filename).suffix.lower()
        if extension in SUPPORTED_EXTENSIONS:
            return SUPPORTED_EXTENSIONS[extension]

    if content_type:
        mime = content_type.split(";", 1)[0].strip().lower()

        if "json" in mime:
            return "json"
        if "csv" in mime:
            return "csv"
        if "xml" in mime:
            return "xml"
        if "fits" in mime:
            return "fits"
        if mime.startswith("image/"):
            return "image"
        if mime.startswith("video/"):
            return "video"
        if mime.startswith("text/"):
            return "text"

    return "unknown"


def modality_for_format(detected_format: str) -> str:
    """Map a detected format to a supported schema modality."""
    return FORMAT_TO_MODALITY.get(detected_format, "unknown")


# ---------------------------------------------------------
# 4. NASA API-key and URL handling
# ---------------------------------------------------------

def get_nasa_api_key(
    explicit_key: Optional[str] = None,
) -> Optional[str]:
    """
    Read an API key from an explicit argument or NASA_API_KEY.
    Do not hard-code credentials in source code.
    """
    return explicit_key or os.environ.get("NASA_API_KEY")


def build_nasa_api_url(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Build an HTTPS URL.

    NASA API keys are added only to api.nasa.gov requests.
    Never send a NASA API key to an unrelated host.
    """
    if not endpoint or not str(endpoint).strip():
        raise ValueError("Endpoint cannot be empty.")

    if endpoint.startswith(("http://", "https://")):
        base_url = endpoint
    else:
        base_url = (
            f"{NASA_API_BASE}/{str(endpoint).lstrip('/')}"
        )

    parts = urllib.parse.urlsplit(base_url)

    if parts.scheme != "https":
        raise ValueError("Only HTTPS URLs are accepted.")

    if not parts.hostname:
        raise ValueError("URL hostname is missing.")

    query_pairs = urllib.parse.parse_qsl(
        parts.query,
        keep_blank_values=True,
    )

    if params:
        for key, value in params.items():
            if key == "api_key":
                raise ValueError(
                    "Pass API credentials through api_key, "
                    "not through params."
                )

            if isinstance(value, (list, tuple)):
                query_pairs.extend(
                    (key, str(item)) for item in value
                )
            elif value is not None:
                query_pairs.append((key, str(value)))

    key = get_nasa_api_key(api_key)

    if key and parts.hostname.lower() == "api.nasa.gov":
        query_pairs = [
            (name, value)
            for name, value in query_pairs
            if name.lower() != "api_key"
        ]
        query_pairs.append(("api_key", key))

    query = urllib.parse.urlencode(query_pairs)

    return urllib.parse.urlunsplit((
        parts.scheme,
        parts.netloc,
        parts.path,
        query,
        parts.fragment,
    ))


# ---------------------------------------------------------
# 5. Download raw resources
# ---------------------------------------------------------

def fetch_nasa_resource(
    url: str,
    output_dir: Union[str, Path],
    filename: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    api_key: Optional[str] = None,
    user_agent: str = "D3-VITAL-X-NASA-Adapter/0.2.0",
    max_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Download a resource and preserve the response body bytes.

    Download is written to a temporary file first; the final path
    is replaced only after a successful response.

    max_bytes, when set, limits the accepted download size.
    """
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero.")

    if max_bytes is not None and max_bytes <= 0:
        raise ValueError("max_bytes must be greater than zero.")

    final_url = build_nasa_api_url(url, api_key=api_key)
    parts = urllib.parse.urlsplit(final_url)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_name = Path(
        urllib.parse.unquote(parts.path)
    ).name or "nasa_resource"

    final_name = safe_filename(filename or original_name)
    output_path = output_dir / final_name

    # Do not overwrite an existing file silently.
    if output_path.exists():
        raise FileExistsError(
            f"Destination already exists: {output_path}"
        )

    request = urllib.request.Request(
        final_url,
        headers={
            "User-Agent": user_agent,
            "Accept": "*/*",
        },
    )

    started_at = utc_timestamp()
    temp_path = None

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:
            content_type = response.headers.get(
                "Content-Type", ""
            )
            status_code = getattr(response, "status", 200)
            final_response_url = response.geturl()

            declared_length = response.headers.get(
                "Content-Length"
            )

            if (
                max_bytes is not None
                and declared_length
                and int(declared_length) > max_bytes
            ):
                raise ValueError(
                    "Server-declared resource size exceeds max_bytes."
                )

            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=output_dir,
                prefix=".nasa_download_",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                total_bytes = 0

                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break

                    total_bytes += len(chunk)

                    if (
                        max_bytes is not None
                        and total_bytes > max_bytes
                    ):
                        raise ValueError(
                            "Downloaded resource exceeded max_bytes."
                        )

                    temp_file.write(chunk)

        # Atomic rename within the same output directory.
        os.replace(temp_path, output_path)
        temp_path = None

    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Resource request failed (HTTP {exc.code})."
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Resource connection failed: {exc.reason}"
        ) from exc

    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    raw_hash = calculate_sha256(output_path)

    return {
        "source_url": _redact_url(url),
        "resolved_url": _redact_url(final_response_url),
        "local_path": str(output_path.resolve()),
        "filename": final_name,
        "content_type": content_type,
        "format": detect_format(final_name, content_type),
        "http_status": status_code,
        "size_bytes": output_path.stat().st_size,
        "sha256": raw_hash,
        "download_started_at": started_at,
        "download_completed_at": utc_timestamp(),
        "raw_data_preserved": True,
    }


# ---------------------------------------------------------
# 6. Local file inspection
# ---------------------------------------------------------

def inspect_nasa_file(
    source: Union[str, Path],
) -> Dict[str, Any]:
    """Inspect a local file without changing it."""
    path = Path(source)

    if not path.is_file():
        raise FileNotFoundError(
            f"Source is not a regular file: {path}"
        )

    mime_type, _ = mimetypes.guess_type(str(path))

    return {
        "filename": path.name,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "extension": path.suffix.lower(),
        "format": detect_format(path.name, mime_type),
        "mime_type": mime_type,
        "sha256": calculate_sha256(path),
        "exists": True,
        "is_file": True,
        "inspected_at": utc_timestamp(),
    }


# ---------------------------------------------------------
# 7. JSON structural inspection
# ---------------------------------------------------------

def inspect_nasa_json(
    source: Union[str, Path],
) -> Dict[str, Any]:
    """Parse JSON for structural inspection only."""
    path = Path(source)

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8-sig") as file:
        payload = json.load(file)

    if isinstance(payload, dict):
        root_type = "object"
        key_count = len(payload)
        keys_preview = list(payload.keys())[:20]
    elif isinstance(payload, list):
        root_type = "array"
        key_count = None
        keys_preview = []
    else:
        root_type = type(payload).__name__
        key_count = None
        keys_preview = []

    return {
        "path": str(path.resolve()),
        "format": "json",
        "root_type": root_type,
        "key_count": key_count,
        "keys_preview": keys_preview,
        "valid_json": True,
        "inspected_at": utc_timestamp(),
    }


# ---------------------------------------------------------
# 8. Technical structural validation
# ---------------------------------------------------------

def validate_nasa_structure(
    source: Union[str, Path],
) -> Dict[str, Any]:
    """
    Validate basic file-level structure.

    This is not scientific, mission, biological, or medical validation.
    """
    path = Path(source)
    issues = []
    warnings = []

    if not path.exists():
        issues.append("Source file does not exist.")
    elif not path.is_file():
        issues.append("Source is not a regular file.")
    else:
        size_bytes = path.stat().st_size

        if size_bytes == 0:
            issues.append("File is empty.")

        detected_format = detect_format(filename=path.name)

        if detected_format == "unknown":
            warnings.append(
                "File format could not be identified by extension."
            )

        if size_bytes > 5 * 1024**3:
            warnings.append(
                "Large file detected; consider streaming or chunked access."
            )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "checked_at": utc_timestamp(),
    }


# ---------------------------------------------------------
# 9. Convert file into the current UnifiedDataRecord
# ---------------------------------------------------------

def nasa_to_unified_record(
    source: Union[str, Path],
    source_url: Optional[str] = None,
    dataset_id: Optional[str] = None,
    data_type: str = "space",
    description: str = "",
    license_info: str = "Not specified",
    anonymized: Optional[bool] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    validate: bool = True,
):
    """
    Build the current UnifiedDataRecord.

    The record references the original file through payload_ref.
    No pixel/spectral/scientific payload is parsed or transformed here.
    """
    path = Path(source)

    if not path.is_file():
        raise FileNotFoundError(
            f"Source is not a regular file: {path}"
        )

    schema = load_schema_module()

    raw_hash = calculate_sha256(path)
    detected_format = detect_format(filename=path.name)
    dataset_id = dataset_id or generate_dataset_id(
        path.name,
        raw_hash,
    )

    if data_type not in getattr(
        schema,
        "SUPPORTED_DATA_TYPES",
        ["space", "biomedical", "seismic", "environmental",
         "synthetic", "general", "unknown"],
    ):
        raise ValueError(
            f"Unsupported data_type: {data_type}"
        )

    modality = modality_for_format(detected_format)

    supported_modalities = getattr(
        schema,
        "SUPPORTED_MODALITIES",
        None,
    )

    if supported_modalities and modality not in supported_modalities:
        modality = "unknown"

    validation = (
        validate_nasa_structure(path)
        if validate
        else {
            "valid": None,
            "issues": [],
            "warnings": [],
        }
    )

    metadata = {
        "adapter": MODULE_NAME,
        "adapter_version": MODULE_VERSION,
        "dataset_id": dataset_id,
        "source_name": path.name,
        "source_type": "NASA_SPACE_RESOURCE",
        "source_url": _redact_url(source_url) if source_url else None,
        "description": description,
        "license_info": license_info,
        "anonymized": anonymized,
        "file_format": detected_format,
        "file_size_bytes": path.stat().st_size,
        "raw_sha256": raw_hash,
        "structural_validation": validation,
        "scientific_processing_performed": False,
        "raw_data_modified": False,
        "created_at": utc_timestamp(),
    }

    if extra_metadata is not None:
        if not isinstance(extra_metadata, dict):
            raise TypeError("extra_metadata must be a dictionary.")

        protected_keys = set(metadata)
        for key, value in extra_metadata.items():
            if key not in protected_keys:
                metadata[key] = value

    quality_flags = []

    if validation["valid"] is False:
        quality_flags.append("structural_validation_failed")

    if validation["warnings"]:
        quality_flags.append("structural_validation_warning")

    RecordClass = schema.UnifiedDataRecord

    return RecordClass(
        record_id=dataset_id,
        schema_name=getattr(
            schema,
            "SCHEMA_NAME",
            "d3_vital_x.unified_data_record",
        ),
        schema_version=getattr(
            schema,
            "SCHEMA_VERSION",
            "0.1.0",
        ),
        source=str(source_url or path.name),
        modality=modality,
        data_type=data_type,
        observed_at=None,
        created_at=utc_timestamp(),
        payload_ref=str(path.resolve()),
        units={},
        metadata=metadata,
        quality_flags=quality_flags,
        provenance={
            "adapter": MODULE_NAME,
            "adapter_version": MODULE_VERSION,
            "source_url": _redact_url(source_url) if source_url else None,
            "source_file": str(path.resolve()),
            "raw_sha256": raw_hash,
            "raw_data_preserved": True,
            "scientific_processing_performed": False,
        },
        uncertainty={},
        labels=[],
        claim_level="C1",
    )


# ---------------------------------------------------------
# 10. Record summary and provenance export
# ---------------------------------------------------------

def summarize_nasa_record(record) -> Dict[str, Any]:
    """Return a JSON-friendly summary of a unified record."""
    provenance = record.provenance or {}

    return {
        "record_id": record.record_id,
        "schema_name": record.schema_name,
        "schema_version": record.schema_version,
        "source": record.source,
        "modality": record.modality,
        "data_type": record.data_type,
        "payload_ref": record.payload_ref,
        "raw_sha256": provenance.get("raw_sha256"),
        "raw_data_preserved": provenance.get("raw_data_preserved"),
        "quality_flags": list(record.quality_flags or []),
        "claim_level": record.claim_level,
        "generated_at": utc_timestamp(),
    }


def nasa_provenance_json(record, indent: int = 2) -> str:
    """Serialize a record summary as JSON."""
    return json.dumps(
        summarize_nasa_record(record),
        indent=indent,
        ensure_ascii=False,
        default=str,
    )


# ---------------------------------------------------------
# 11. NASA JSON API helper
# ---------------------------------------------------------

def fetch_nasa_json(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_bytes: Optional[int] = 20 * 1024 * 1024,
) -> Dict[str, Any]:
    """
    Fetch a NASA API response to a local file.

    JSON parsing is separate; the response bytes are not re-serialized.
    """
    output_dir = Path(output_dir or (Path.cwd() / "nasa_raw"))
    output_dir.mkdir(parents=True, exist_ok=True)

    url = build_nasa_api_url(
        endpoint,
        params=params,
        api_key=api_key,
    )

    if filename is None:
        endpoint_name = (
            Path(urllib.parse.urlsplit(url).path).name
            or "nasa_api_response"
        )
        filename = safe_filename(f"{endpoint_name}.json")

    return fetch_nasa_resource(
        url=url,
        output_dir=output_dir,
        filename=filename,
        timeout=timeout,
        api_key=None,
        max_bytes=max_bytes,
    )


# ---------------------------------------------------------
# 12. Local self-test
# ---------------------------------------------------------

def run_nasa_adapter_test() -> Dict[str, Any]:
    """
    Run a local synthetic test without accessing the NASA network.
    """
    test_dir = Path(tempfile.mkdtemp(prefix="d3_nasa_test_"))
    test_file = test_dir / "synthetic_nasa_test.json"

    fixture = {
        "source": "SYNTHETIC_TEST_FIXTURE",
        "dataset_type": "synthetic",
        "values": [1, 2, 3, 4],
        "note": "Not NASA observational data.",
    }

    with test_file.open("w", encoding="utf-8") as file:
        json.dump(fixture, file, indent=2)

    original_hash = calculate_sha256(test_file)

    inspection = inspect_nasa_file(test_file)
    json_report = inspect_nasa_json(test_file)
    validation = validate_nasa_structure(test_file)

    record = nasa_to_unified_record(
        source=test_file,
        source_url="synthetic://NASA_TEST_FIXTURE",
        data_type="synthetic",
        description="Local adapter test fixture",
        license_info="Synthetic test data",
        anonymized=True,
        extra_metadata={
            "test_fixture": True,
            "external_network_used": False,
        },
    )

    summary = summarize_nasa_record(record)
    final_hash = calculate_sha256(test_file)

    assert inspection["exists"] is True
    assert inspection["format"] == "json"
    assert json_report["valid_json"] is True
    assert validation["valid"] is True
    assert record.payload_ref == str(test_file.resolve())
    assert original_hash == final_hash
    assert summary["raw_sha256"] == original_hash

    return {
        "passed": True,
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "tests_run": 7,
        "network_used": False,
        "raw_hash_unchanged": True,
        "inspection": inspection,
        "json_inspection": json_report,
        "validation": validation,
        "record_summary": summary,
        "test_directory": str(test_dir),
    }


def module_info() -> Dict[str, Any]:
    """Return public module metadata."""
    return {
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "purpose": "NASA and space-data ingestion",
        "supported_formats": sorted(SUPPORTED_EXTENSIONS),
        "scientific_interpretation": False,
        "clinical_diagnosis": False,
        "proprietary_core_included": False,
    }


if __name__ == "__main__":
    print("D³ VITAL-X NASA Adapter", MODULE_VERSION)
    print(run_nasa_adapter_test())
