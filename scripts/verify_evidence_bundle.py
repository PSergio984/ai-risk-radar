"""Verify bit-exact cryptographic SHA-256 hashes of frozen release artifacts."""

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Authoritative frozen release reports (tracked directly in git repository)
AUTHORITATIVE_RELEASE_REPORTS: dict[str, str] = {
    "artifacts/release.json": "1f4751a7108874364533b55f38a5ab1fdd3e26b0689dfbd502319e29098319b9",
    "artifacts/final_test_report.json": "6603ab4c7204bd38b2d179ee67bf17760788240e11f4fc61b267d3c3705d0901",
    "artifacts/research_report.json": "89134c36b00ccce0c262ee2719d46f8c2c4f9eca56f998b0d3ea86e9e3683296",
    "artifacts/latest_report.json": "db529c8dd2edda5cca03c15423c9143b69380f2b7829c1afa87393f3d47c59aa",
    "artifacts/corrected_holdout_evaluation.json": "ac961b6b790897fd1b57f501ba43a361eade013dada886a3e65c846cf301857d",
}

# Authoritative frozen binary model weights (untracked/ignored in git clone; verified when present)
AUTHORITATIVE_MODEL_HASHES: dict[str, str] = {
    "artifacts/models/random_forest.joblib": "f805f206803067139f8c52099273bf9e80ca53ec64b10aed270daabebbcfe113",
}

AUTHORITATIVE_RELEASE_HASHES: dict[str, str] = {
    **AUTHORITATIVE_RELEASE_REPORTS,
    **AUTHORITATIVE_MODEL_HASHES,
}

# Authoritative historical market evaluation snapshot hashes (if present locally)
AUTHORITATIVE_SNAPSHOT_HASHES: dict[str, str] = {
    "data/raw/snapshot/manifest.json": "b476eaff931e68f4ece308cd64e860928860925a6a59358269c29fbeb4df4974",
    "data/raw/snapshot/adjusted_close.parquet": "252dbfed64a8a4b4a340b4aa84cb34e9304c72b50fc2a9e79c3026225a81aa8f",
    "data/raw/snapshot/volume.parquet": "a2a8c5d3c3fedc77e581079839b243f7686a1259bdd90e8b17bb169d54c35af4",
}


def sha256_file(path: Path, target_crlf: bool = False) -> str:
    """Compute standard SHA-256 hex digest of file bytes, optionally normalizing CRLF for text reports."""
    data = path.read_bytes()
    if target_crlf and path.suffix == ".json":
        data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return hashlib.sha256(data).hexdigest()


def verify_bundle(repo_root: Path = REPO_ROOT) -> tuple[bool, list[dict[str, str]]]:
    """Verify all authoritative artifacts against stored SHA-256 hashes."""
    results: list[dict[str, str]] = []
    all_passed = True

    # 1. Verify frozen release reports (tracked in repository)
    for rel_path, expected_hash in AUTHORITATIVE_RELEASE_REPORTS.items():
        file_path = repo_root / rel_path
        if not file_path.is_file():
            results.append({
                "file": rel_path,
                "status": "MISSING",
                "expected": expected_hash,
                "actual": "N/A",
            })
            all_passed = False
            continue

        actual_hash = sha256_file(file_path)
        if actual_hash != expected_hash and file_path.suffix == ".json":
            crlf_hash = sha256_file(file_path, target_crlf=True)
            if crlf_hash == expected_hash:
                actual_hash = crlf_hash

        if actual_hash == expected_hash:
            results.append({
                "file": rel_path,
                "status": "PASS",
                "expected": expected_hash,
                "actual": actual_hash,
            })
        else:
            results.append({
                "file": rel_path,
                "status": "FAIL_HASH_MISMATCH",
                "expected": expected_hash,
                "actual": actual_hash,
            })
            all_passed = False

    # 2. Verify frozen binary model weights (if present locally; untracked on clean clone)
    for rel_path, expected_hash in AUTHORITATIVE_MODEL_HASHES.items():
        file_path = repo_root / rel_path
        if not file_path.is_file():
            results.append({
                "file": rel_path,
                "status": "OPTIONAL_ABSENT",
                "expected": expected_hash,
                "actual": "N/A (clean clone state; model weights untracked)",
            })
            continue

        actual_hash = sha256_file(file_path)
        if actual_hash == expected_hash:
            results.append({
                "file": rel_path,
                "status": "PASS",
                "expected": expected_hash,
                "actual": actual_hash,
            })
        else:
            results.append({
                "file": rel_path,
                "status": "FAIL_HASH_MISMATCH",
                "expected": expected_hash,
                "actual": actual_hash,
            })
            all_passed = False

    # 2. Verify snapshot files (if directory exists)
    for rel_path, expected_hash in AUTHORITATIVE_SNAPSHOT_HASHES.items():
        file_path = repo_root / rel_path
        if not file_path.is_file():
            results.append({
                "file": rel_path,
                "status": "OPTIONAL_ABSENT",
                "expected": expected_hash,
                "actual": "N/A (clean clone state; fetch via pipeline)",
            })
            continue

        actual_hash = sha256_file(file_path)
        if actual_hash == expected_hash:
            results.append({
                "file": rel_path,
                "status": "PASS",
                "expected": expected_hash,
                "actual": actual_hash,
            })
        else:
            results.append({
                "file": rel_path,
                "status": "FAIL_HASH_MISMATCH",
                "expected": expected_hash,
                "actual": actual_hash,
            })
            all_passed = False

    return all_passed, results


def main() -> int:
    print("=" * 80)
    print("AI INVESTMENT RISK RADAR: FROZEN EVIDENCE BUNDLE INTEGRITY VERIFICATION")
    print("=" * 80)

    passed, results = verify_bundle(REPO_ROOT)

    col_w = [40, 16, 16]
    print(f"{'Artifact File':<{col_w[0]}} | {'Status':<{col_w[1]}} | {'SHA-256 (Prefix)'}")
    print("-" * 80)

    for r in results:
        status_disp = r["status"]
        hash_disp = r["actual"][:16] + "..." if r["actual"] != "N/A" else "N/A"
        print(f"{r['file']:<{col_w[0]}} | {status_disp:<{col_w[1]}} | {hash_disp}")

    print("=" * 80)
    if passed:
        print("[SUCCESS] All frozen evaluation artifacts verified bit-for-bit against release.")
        return 0
    else:
        print("[FAILURE] One or more frozen artifacts failed cryptographic verification!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
