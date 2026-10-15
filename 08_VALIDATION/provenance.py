
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
D³ VITAL-X Space Intelligence Platform
Module 29: provenance.py

Version: 1.0.0
Report schema: 1.0

Purpose
-------
Create a provenance manifest for one or more files.

The manifest records:
- File names and extensions
- File sizes and SHA-256 hashes
- Optional dataset/source identifiers
- Optional user-supplied metadata
- Runtime environment and UTC timestamp
- Optional parent-report hash
- A deterministic fingerprint of the manifest content

Scientific boundary
-------------------
Provenance records do not prove scientific validity, authorship,
ownership, authenticity of a source, causality, or medical conclusions.

This module reads input files and does not intentionally modify them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


MODULE_NAME = "provenance"
MODULE_VERSION = "1.0.0"
REPORT_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(file_path: Path) -> str:
    """Calculate SHA-256 for a file without changing it."""
    digest = hashlib.sha256()

    with Path(file_path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def sha256_text(value: str) -> str:
    """Calculate SHA-256 for a UTF-8 string."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_json_fingerprint(data: Any) -> str:
    """
    Fingerprint a JSON-compatible object using stable serialization.

    Dictionary keys are sorted and whitespace is normalized.
    """
    serialized = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return sha256_text(serialized)


def parse_metadata_items(items: Optional[Sequence[str]]) -> Dict[str, str]:
    """
    Parse repeated KEY=VALUE arguments into a dictionary.

    Example:
        --metadata instrument=JWST --metadata stage=level3
    """
    metadata: Dict[str, str] = {}

    for item in items or []:
        if "=" not in item:
            raise ValueError(
                f"Invalid metadata item '{item}'. Use KEY=VALUE."
            )

        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            raise ValueError("Metadata keys cannot be empty.")

        if key in metadata:
            raise ValueError(f"Duplicate metadata key: {key}")

        metadata[key] = value

    return metadata


# ---------------------------------------------------------------------
# File inventory
# ---------------------------------------------------------------------

def describe_file(file_path: Path) -> Dict[str, Any]:
    """
    Record basic file information and SHA-256.

    The resolved absolute path is deliberately not included, to reduce
    accidental disclosure of local directory names in public reports.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Input path not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Input path is not a regular file: {file_path}")

    size_before = file_path.stat().st_size
    hash_before = sha256_file(file_path)
    size_after = file_path.stat().st_size
    hash_after = sha256_file(file_path)

    if size_before != size_after or hash_before != hash_after:
        raise RuntimeError(
            f"File changed while being inventoried: {file_path.name}"
        )

    return {
        "file_name": file_path.name,
        "file_extension": file_path.suffix.lower(),
        "size_bytes": size_after,
        "sha256": hash_after,
        "hash_algorithm": "SHA-256",
        "read_check": "PASS",
        "file_changed_during_hashing": False,
    }


def inventory_files(
    input_paths: Sequence[Path],
) -> List[Dict[str, Any]]:
    """Create an inventory of unique input paths."""
    if not input_paths:
        raise ValueError("At least one --input file is required.")

    records: List[Dict[str, Any]] = []
    seen_paths = set()

    for input_path in input_paths:
        path = Path(input_path)

        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")

        resolved = path.resolve()

        if resolved in seen_paths:
            raise ValueError(
                f"Duplicate input file supplied: {path.name}"
            )

        seen_paths.add(resolved)
        records.append(describe_file(path))

    return records


# ---------------------------------------------------------------------
# Manifest creation
# ---------------------------------------------------------------------

def build_manifest(
    file_records: Sequence[Dict[str, Any]],
    dataset_id: Optional[str] = None,
    source_description: Optional[str] = None,
    source_url: Optional[str] = None,
    parent_report: Optional[Path] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Build a provenance manifest.

    source_url and source_description are user-supplied labels. This
    module does not visit or independently verify external URLs.
    """
    parent_record = None

    if parent_report is not None:
        parent_path = Path(parent_report)

        if not parent_path.is_file():
            raise FileNotFoundError(
                f"Parent report not found: {parent_path}"
            )

        parent_record = {
            "file_name": parent_path.name,
            "size_bytes": parent_path.stat().st_size,
            "sha256": sha256_file(parent_path),
            "hash_algorithm": "SHA-256",
        }

    manifest: Dict[str, Any] = {
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
        "dataset": {
            "dataset_id": dataset_id,
            "source_description": source_description,
            "source_url_user_supplied": source_url,
            "source_url_independently_verified": False,
        },
        "files": list(file_records),
        "file_count": len(file_records),
        "parent_report": parent_record,
        "user_metadata": dict(metadata or {}),
        "integrity_summary": {
            "all_files_hashed": all(
                bool(record.get("sha256")) for record in file_records
            ),
            "all_file_read_checks_passed": all(
                record.get("read_check") == "PASS"
                for record in file_records
            ),
            "source_files_modified_by_module": False,
        },
        "provenance_boundary": {
            "source_authenticity_established": False,
            "authorship_established": False,
            "ownership_established": False,
            "scientific_validity_established": False,
            "causality_established": False,
            "medical_diagnosis": False,
            "human_review_required": True,
        },
    }

    # Fingerprint excludes the fingerprint field itself.
    manifest["manifest_sha256"] = stable_json_fingerprint(manifest)

    return manifest


# ---------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------

def verify_manifest_structure(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify the manifest's recorded fingerprint and basic structure.

    This checks internal consistency only. It cannot establish that the
    recorded metadata or source claims are truthful.
    """
    recorded_hash = manifest.get("manifest_sha256")

    if not isinstance(recorded_hash, str) or not recorded_hash:
        return {
            "structure_valid": False,
            "fingerprint_matches": False,
            "status": "FAIL",
            "detail": "Missing manifest_sha256.",
        }

    content_without_hash = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_sha256"
    }

    computed_hash = stable_json_fingerprint(content_without_hash)
    fingerprint_matches = computed_hash == recorded_hash

    has_file_list = isinstance(manifest.get("files"), list)
    has_module = bool(manifest.get("module"))
    structure_valid = has_file_list and has_module

    return {
        "structure_valid": structure_valid,
        "recorded_fingerprint": recorded_hash,
        "computed_fingerprint": computed_hash,
        "fingerprint_matches": fingerprint_matches,
        "status": (
            "PASS"
            if structure_valid and fingerprint_matches
            else "FAIL"
        ),
        "detail": (
            "Internal manifest structure and fingerprint checked."
        ),
        "authenticity_verified": False,
    }


# ---------------------------------------------------------------------
# Output reports
# ---------------------------------------------------------------------

def save_manifest(
    manifest: Dict[str, Any],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Save the manifest as JSON and a file inventory as CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "provenance_manifest.json"
    csv_path = output_dir / "provenance_files.csv"

    with json_path.open("w", encoding="utf-8") as file_obj:
        json.dump(
            manifest,
            file_obj,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        file_obj.write("\n")

    fieldnames = [
        "file_name",
        "file_extension",
        "size_bytes",
        "sha256",
        "hash_algorithm",
        "read_check",
        "file_changed_during_hashing",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(manifest["files"])

    return json_path, csv_path


def print_summary(manifest: Dict[str, Any]) -> None:
    """Print a concise inventory summary."""
    integrity = manifest["integrity_summary"]

    print("=" * 68)
    print("D³ VITAL-X — PROVENANCE MANIFEST")
    print("=" * 68)
    print(f"Module version       : {manifest['module_version']}")
    print(f"Created UTC          : {manifest['created_utc']}")
    print(f"Dataset ID           : {manifest['dataset']['dataset_id'] or 'N/A'}")
    print(f"File count           : {manifest['file_count']}")
    print(f"All files hashed     : {integrity['all_files_hashed']}")
    print(
        "All read checks pass : "
        f"{integrity['all_file_read_checks_passed']}"
    )
    print(f"Manifest SHA-256     : {manifest['manifest_sha256']}")
    print("-" * 68)

    for index, record in enumerate(manifest["files"], start=1):
        print(
            f"{index}. {record['file_name']} "
            f"({record['size_bytes']} bytes)"
        )
        print(f"   SHA-256: {record['sha256']}")

    print("-" * 68)
    print("Source authenticity established : NO")
    print("Scientific validity established : NO")
    print("Medical diagnosis              : NO")
    print("=" * 68)


# ---------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------

def self_check() -> Dict[str, Any]:
    """Run deterministic checks using temporary synthetic files."""
    import tempfile

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

    with tempfile.TemporaryDirectory(
        prefix="d3_vital_x_provenance_"
    ) as temp_dir:
        temp_root = Path(temp_dir)
        sample_file = temp_root / "sample.txt"
        sample_file.write_text(
            "Synthetic provenance self-check.\n",
            encoding="utf-8",
        )

        before_hash = sha256_file(sample_file)
        before_size = sample_file.stat().st_size

        file_records = inventory_files([sample_file])

        check(
            "sha256_deterministic",
            sha256_file(sample_file) == before_hash,
            "Same file content produces the same SHA-256 hash.",
        )

        check(
            "file_size_recorded",
            file_records[0]["size_bytes"] == before_size,
            "File size is recorded correctly.",
        )

        check(
            "input_file_not_modified",
            sha256_file(sample_file) == before_hash
            and sample_file.stat().st_size == before_size,
            "Inventory operation leaves the source file unchanged.",
        )

        check(
            "filename_recorded",
            file_records[0]["file_name"] == "sample.txt",
            "File name is included in the inventory.",
        )

        manifest = build_manifest(
            file_records=file_records,
            dataset_id="synthetic-self-check",
            source_description="Synthetic test file",
            metadata={"test_mode": "true"},
        )

        verification = verify_manifest_structure(manifest)

        check(
            "manifest_fingerprint_matches",
            verification["fingerprint_matches"],
            "Manifest fingerprint matches its recorded value.",
        )

        check(
            "manifest_structure_valid",
            verification["structure_valid"],
            "Required manifest structure is present.",
        )

        check(
            "source_url_not_claimed_verified",
            manifest["dataset"]["source_url_independently_verified"]
            is False,
            "The manifest does not claim external URL verification.",
        )

        # Tampering with a copied manifest should invalidate its hash.
        altered = dict(manifest)
        altered["dataset"] = dict(manifest["dataset"])
        altered["dataset"]["dataset_id"] = "modified-after-hashing"

        altered_verification = verify_manifest_structure(altered)

        check(
            "tampering_detected",
            altered_verification["fingerprint_matches"] is False,
            "Changing manifest content invalidates the stored fingerprint.",
        )

        json_path, csv_path = save_manifest(
            manifest,
            temp_root / "output",
        )

        check(
            "json_report_created",
            json_path.is_file(),
            "JSON provenance report is created.",
        )

        check(
            "csv_inventory_created",
            csv_path.is_file(),
            "CSV file inventory is created.",
        )

        loaded_manifest = json.loads(
            json_path.read_text(encoding="utf-8")
        )

        check(
            "saved_json_valid",
            verify_manifest_structure(loaded_manifest)["status"] == "PASS",
            "Saved JSON can be parsed and its fingerprint verified.",
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
            "source_authenticity_established": False,
            "scientific_validity_established": False,
            "medical_diagnosis": False,
        },
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a file provenance manifest with SHA-256 hashes. "
            "The report does not independently verify source claims."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=None,
        help=(
            "Input file to inventory. Repeat this option for multiple files."
        ),
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="Optional dataset identifier supplied by the user.",
    )
    parser.add_argument(
        "--source-description",
        type=str,
        default=None,
        help="Optional source description supplied by the user.",
    )
    parser.add_argument(
        "--source-url",
        type=str,
        default=None,
        help=(
            "Optional source URL label. It is recorded but not visited "
            "or independently verified."
        ),
    )
    parser.add_argument(
        "--parent-report",
        type=Path,
        default=None,
        help="Optional existing report whose hash should be recorded.",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=None,
        help="Optional KEY=VALUE metadata. Repeat as needed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation_reports") / "provenance",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--verify-manifest",
        type=Path,
        default=None,
        help="Verify the fingerprint of an existing provenance JSON file.",
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
        if args.self_check:
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

        if args.verify_manifest is not None:
            manifest_path = args.verify_manifest

            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"Manifest not found: {manifest_path}"
                )

            with manifest_path.open(
                "r",
                encoding="utf-8",
            ) as file_obj:
                manifest = json.load(file_obj)

            verification = verify_manifest_structure(manifest)

            print(json.dumps(
                verification,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ))

            return 0 if verification["status"] == "PASS" else 1

        if not args.input:
            raise ValueError(
                "Provide at least one --input file or use --self-check."
            )

        metadata = parse_metadata_items(args.metadata)
        file_records = inventory_files(args.input)

        manifest = build_manifest(
            file_records=file_records,
            dataset_id=args.dataset_id,
            source_description=args.source_description,
            source_url=args.source_url,
            parent_report=args.parent_report,
            metadata=metadata,
        )

        json_path, csv_path = save_manifest(
            manifest,
            args.output_dir,
        )

        print_summary(manifest)
        print(f"JSON manifest : {json_path.resolve()}")
        print(f"CSV inventory : {csv_path.resolve()}")

        return 0

    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        AssertionError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
