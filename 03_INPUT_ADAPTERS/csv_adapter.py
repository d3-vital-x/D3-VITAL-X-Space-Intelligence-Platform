
"""
D³ VITAL-X SPACE INTELLIGENCE PLATFORM
Module 05: csv_adapter.py
Version: 0.1.0

Purpose:
    Read and inspect CSV files through a non-destructive adapter.
    Optionally create a UnifiedDataRecord using data_schema.py.

Features:
    - CSV header and duplicate-column inspection
    - Missing-value and numeric-column summaries
    - Conservative numeric conversion
    - SHA-256 file integrity fingerprint
    - Deterministic dataset identifier
    - Optional UnifiedDataRecord creation
    - Optional row limit for memory-conscious workflows

Safety:
    - Original file is never modified.
    - CSV parsing is not scientific validation.
    - No medical diagnosis or automated scientific conclusions.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List
import csv
import hashlib
import math
import importlib.util
import sys


MODULE_VERSION = "0.1.0"
DEFAULT_ENCODING = "utf-8-sig"
DEFAULT_DELIMITER = ","


# ============================================================
# 1. Project paths and schema loading
# ============================================================

def _find_project_root() -> Path:
    """Resolve the project root without requiring a fixed location."""
    candidates = []

    if "__file__" in globals():
        candidates.extend(Path(__file__).resolve().parents)

    candidates.append(
        Path("/content/D3-VITAL-X-Space-Intelligence-Platform")
    )

    for candidate in candidates:
        if (candidate / "04_UNIFIED_DATA_LAYER" / "data_schema.py").exists():
            return candidate

    # Useful for standalone inspection, even if schema is not installed.
    return Path("/content/D3-VITAL-X-Space-Intelligence-Platform")


PROJECT_ROOT = _find_project_root()
SCHEMA_PATH = (
    PROJECT_ROOT / "04_UNIFIED_DATA_LAYER" / "data_schema.py"
)


def _load_schema_module():
    """Load the project's shared schema module."""
    if not SCHEMA_PATH.exists():
        return None

    module_name = "d3_vital_x_shared_data_schema"

    spec = importlib.util.spec_from_file_location(
        module_name,
        SCHEMA_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load schema: {SCHEMA_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

    return module


# ============================================================
# 2. Basic utilities
# ============================================================

def utc_timestamp() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA-256 of a file without loading it all into memory."""
    path = Path(file_path)

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def safe_column_name(column_name: Any) -> str:
    """Normalize a CSV header field to a trimmed string."""
    if column_name is None:
        return ""
    return str(column_name).strip()


def detect_numeric(value: Any) -> bool:
    """Return True if a non-empty value parses as a finite float."""
    if value is None:
        return False

    text = str(value).strip()
    if not text:
        return False

    try:
        return math.isfinite(float(text))
    except (TypeError, ValueError, OverflowError):
        return False


def convert_value(value: Any) -> Any:
    """
    Convert finite numeric strings to int/float.
    Blank cells become None; other values remain trimmed strings.
    """
    if value is None:
        return None

    text = str(value).strip()
    if text == "":
        return None

    if not detect_numeric(text):
        return text

    number = float(text)

    if number.is_integer():
        return int(number)

    return number


def generate_dataset_id(file_path: Path, raw_hash: str) -> str:
    """
    Generate a deterministic ID from file name and file content hash.

    The same file name and content hash produce the same ID.
    """
    identifier_source = f"{Path(file_path).name}|{raw_hash}"
    digest = hashlib.sha256(
        identifier_source.encode("utf-8")
    ).hexdigest()[:16]

    return f"csv_{digest}"


def _validate_csv_options(
    encoding: str,
    delimiter: str,
    max_rows: Optional[int] = None,
) -> None:
    if not isinstance(encoding, str) or not encoding.strip():
        raise ValueError("encoding must be a non-empty string")

    if not isinstance(delimiter, str) or len(delimiter) != 1:
        raise ValueError("delimiter must be exactly one character")

    if max_rows is not None:
        if (
            isinstance(max_rows, bool)
            or not isinstance(max_rows, int)
            or max_rows < 0
        ):
            raise ValueError("max_rows must be a non-negative integer or None")


def _resolve_file(file_path: str | Path) -> Path:
    path = Path(file_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file, received: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, received: {path.name}")

    return path.resolve()


def _read_header(
    path: Path,
    encoding: str,
    delimiter: str,
) -> tuple[list[str], list[str]]:
    """Return raw and normalized header fields."""
    with path.open(
        mode="r",
        encoding=encoding,
        newline="",
    ) as file:
        reader = csv.reader(file, delimiter=delimiter)

        try:
            raw_header = next(reader)
        except StopIteration:
            raise ValueError("CSV file is empty; no header row found.")

    normalized = [safe_column_name(name) for name in raw_header]

    if not normalized:
        raise ValueError("CSV file contains an empty header row.")

    return raw_header, normalized


def _duplicate_column_names(columns: list[str]) -> list[str]:
    seen = set()
    duplicates = set()

    for column in columns:
        if column in seen:
            duplicates.add(column)
        seen.add(column)

    return sorted(duplicates)


# ============================================================
# 3. CSV inspection
# ============================================================

def inspect_csv(
    file_path: str | Path,
    encoding: str = DEFAULT_ENCODING,
    delimiter: str = DEFAULT_DELIMITER,
) -> Dict[str, Any]:
    """
    Inspect CSV structure without creating a data record.

    Duplicate headers are reported explicitly. Inspection uses
    positional CSV rows so duplicate headers do not silently
    overwrite one another during this stage.
    """
    _validate_csv_options(encoding, delimiter)
    path = _resolve_file(file_path)

    raw_header, columns = _read_header(path, encoding, delimiter)
    duplicates = _duplicate_column_names(columns)

    row_count = 0
    missing_by_column = {f"{i}:{name}": 0 for i, name in enumerate(columns)}
    non_empty_values = [[] for _ in columns]
    malformed_row_count = 0

    with path.open(
        mode="r",
        encoding=encoding,
        newline="",
    ) as file:
        reader = csv.reader(file, delimiter=delimiter)
        next(reader, None)

        for row in reader:
            row_count += 1

            if len(row) != len(columns):
                malformed_row_count += 1

            for index, column in enumerate(columns):
                value = row[index] if index < len(row) else None
                key = f"{index}:{column}"

                if value is None or str(value).strip() == "":
                    missing_by_column[key] += 1
                else:
                    non_empty_values[index].append(value)

    numeric_columns = []
    text_columns = []

    for index, column in enumerate(columns):
        values = non_empty_values[index]

        if values and all(detect_numeric(value) for value in values):
            numeric_columns.append(column)
        else:
            text_columns.append(column)

    return {
        "module_version": MODULE_VERSION,
        "file_name": path.name,
        "file_path": str(path),
        "file_size_bytes": path.stat().st_size,
        "sha256": calculate_sha256(path),
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
        "raw_columns": raw_header,
        "duplicate_columns": duplicates,
        "missing_values": missing_by_column,
        "numeric_columns": numeric_columns,
        "text_columns": text_columns,
        "malformed_row_count": malformed_row_count,
        "encoding": encoding,
        "delimiter": delimiter,
        "inspected_at": utc_timestamp(),
    }


# ============================================================
# 4. CSV loading
# ============================================================

def load_csv_data(
    file_path: str | Path,
    encoding: str = DEFAULT_ENCODING,
    delimiter: str = DEFAULT_DELIMITER,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Load CSV rows into a Python dictionary.

    max_rows=None loads all rows.
    max_rows=N loads at most N rows; source file remains unchanged.

    Duplicate or empty normalized headers are rejected to avoid
    ambiguous dictionary keys and silent data loss.
    """
    _validate_csv_options(encoding, delimiter, max_rows)
    path = _resolve_file(file_path)

    _, columns = _read_header(path, encoding, delimiter)

    if any(not name for name in columns):
        raise ValueError("CSV contains an empty column name.")

    duplicates = _duplicate_column_names(columns)
    if duplicates:
        raise ValueError(
            "Duplicate column names are ambiguous: "
            + ", ".join(duplicates)
        )

    rows: List[Dict[str, Any]] = []
    malformed_row_count = 0

    with path.open(
        mode="r",
        encoding=encoding,
        newline="",
    ) as file:
        reader = csv.reader(file, delimiter=delimiter)
        next(reader, None)

        for row in reader:
            if max_rows is not None and len(rows) >= max_rows:
                break

            if len(row) != len(columns):
                malformed_row_count += 1

            normalized_row = {}

            for index, column in enumerate(columns):
                raw_value = row[index] if index < len(row) else None
                normalized_row[column] = convert_value(raw_value)

            rows.append(normalized_row)

    return {
        "module_version": MODULE_VERSION,
        "file_name": path.name,
        "file_path": str(path),
        "sha256": calculate_sha256(path),
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "column_count": len(columns),
        "malformed_row_count": malformed_row_count,
        "is_partial_load": (
            max_rows is not None and len(rows) < _count_data_rows(
                path, encoding, delimiter
            )
        ),
        "max_rows": max_rows,
    }


def _count_data_rows(
    path: Path,
    encoding: str,
    delimiter: str,
) -> int:
    """Count data rows without storing them."""
    with path.open(
        mode="r",
        encoding=encoding,
        newline="",
    ) as file:
        reader = csv.reader(file, delimiter=delimiter)
        next(reader, None)
        return sum(1 for _ in reader)


# ============================================================
# 5. Optional UnifiedDataRecord creation
# ============================================================

def create_csv_record(
    file_path: str | Path,
    encoding: str = DEFAULT_ENCODING,
    delimiter: str = DEFAULT_DELIMITER,
    dataset_id: Optional[str] = None,
):
    """
    Create a UnifiedDataRecord describing the CSV source.

    The record contains source metadata and a payload reference;
    the full CSV table is returned separately by load_csv_data().
    """
    path = _resolve_file(file_path)
    raw_hash = calculate_sha256(path)
    generated_id = dataset_id or generate_dataset_id(path, raw_hash)
    inspection = inspect_csv(path, encoding, delimiter)

    schema = _load_schema_module()
    if schema is None:
        raise FileNotFoundError(
            "data_schema.py is required to create a UnifiedDataRecord. "
            f"Expected: {SCHEMA_PATH}"
        )

    return schema.create_unified_record(
        source="csv_adapter",
        modality="tabular",
        data_type="general",
        payload_ref=str(path),
        metadata={
            "dataset_id": generated_id,
            "file_name": path.name,
            "file_size_bytes": path.stat().st_size,
            "row_count": inspection["row_count"],
            "column_count": inspection["column_count"],
            "columns": inspection["columns"],
            "numeric_columns": inspection["numeric_columns"],
            "text_columns": inspection["text_columns"],
            "duplicate_columns": inspection["duplicate_columns"],
            "malformed_row_count": inspection["malformed_row_count"],
            "encoding": encoding,
            "delimiter": delimiter,
        },
        provenance={
            "adapter": "csv_adapter",
            "adapter_version": MODULE_VERSION,
            "sha256": raw_hash,
            "original_file_preserved": True,
        },
        claim_level="C1",
    )


# ============================================================
# 6. Deterministic smoke test
# ============================================================

def self_check() -> Dict[str, Any]:
    """Test the adapter using a temporary synthetic CSV."""
    import tempfile

    checks: Dict[str, bool] = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "synthetic_test.csv"
        original_text = (
            "time,signal,label\n"
            "0,1.25,A\n"
            "1,2.50,B\n"
            "2,,C\n"
        )
        path.write_text(original_text, encoding="utf-8")

        before_hash = calculate_sha256(path)

        inspection = inspect_csv(path)
        loaded = load_csv_data(path)

        checks["header_detected"] = (
            inspection["columns"] == ["time", "signal", "label"]
        )
        checks["row_count_correct"] = inspection["row_count"] == 3
        checks["missing_cell_detected"] = (
            inspection["missing_values"]["1:signal"] == 1
        )
        checks["numeric_conversion"] = (
            loaded["rows"][0]["time"] == 0
            and loaded["rows"][0]["signal"] == 1.25
        )
        checks["blank_converted_to_none"] = (
            loaded["rows"][2]["signal"] is None
        )
        checks["original_file_unchanged"] = (
            calculate_sha256(path) == before_hash
        )

        limited = load_csv_data(path, max_rows=1)
        checks["row_limit_respected"] = limited["row_count"] == 1
        checks["partial_load_marked"] = limited["is_partial_load"] is True

    return {
        "module_version": MODULE_VERSION,
        "passed": all(checks.values()),
        "checks": checks,
    }


if __name__ == "__main__":
    import json

    result = self_check()
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if not result["passed"]:
        raise SystemExit("CSV adapter self-check failed.")

    print("\nCSV adapter self-check passed.")
