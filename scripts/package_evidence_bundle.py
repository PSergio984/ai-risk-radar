"""Assemble a reproducible local frozen evidence bundle with verified checksums."""

import json
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_evidence_bundle import (
    AUTHORITATIVE_RELEASE_HASHES,
    AUTHORITATIVE_SNAPSHOT_HASHES,
    sha256_file,
    verify_bundle,
)

OUTPUT_DIR = REPO_ROOT / ".scratch" / "evidence_bundle"
ARCHIVE_PATH = REPO_ROOT / ".scratch" / "ai_risk_radar_frozen_evidence.tar.gz"

LICENSING_TEXT = """# AI Investment Risk Radar: Evidence Bundle Licensing & Availability Guidance

## 1. Compliance with Financial Market Data Redistribution Restrictions
Under the Terms of Service of consumer market data providers (including Yahoo Finance / `yfinance`) and primary equity exchanges (such as the New York Stock Exchange `XNYS`), raw historical bar and tick price data are proprietary and licensed exclusively for direct individual/research consumption. **Bulk republication, redistribution, or hosting of raw multi-asset OHLCV market feeds on public cloud buckets or GitHub releases without an official exchange vendor redistribution agreement is strictly prohibited.**

To remain fully compliant with copyright and licensing laws while preserving absolute scientific reproducibility:
1. **No External Asset Uploads:** Raw market price datasets (`data/raw/`) and model weights are NOT published to public cloud hosting or GitHub releases.
2. **Local Client-Side Ingestion:** Users and peer reviewers acquire price data directly from primary API endpoints on their own behalf using the documented pipeline command:
   ```bash
   python -m ai_risk_radar latest --config config.json --data-mode refresh
   ```
3. **Deterministic Offline Demonstration Mode:** For immediate zero-network evaluation and automated test environments, the system provides a self-contained synthetic fixture mode:
   ```bash
   export AI_RISK_RADAR_NOTEBOOK_MODE=fixture
   pytest tests/test_fresh_audit_notebook_modes.py
   ```
4. **Cryptographic Immutability:** All evaluation conclusions are tethered to immutable SHA-256 digests. Any local reproduction can be verified bit-for-bit against the author's frozen release manifest using `scripts/verify_evidence_bundle.py`.

## 2. Review Bundle Scope
This bundle contains:
- Frozen Model Weights: `artifacts/models/random_forest.joblib`
- Release Manifest & Provenance: `artifacts/release.json`
- Single-Use Holdout Evaluation Report: `artifacts/final_test_report.json`
- In-Sample Research & Validation Report: `artifacts/research_report.json`
- Forward-Looking Latest Inference Report: `artifacts/latest_report.json`
- Cryptographic Checksums: `checksums.sha256`

All source bytes and cryptographic hashes match the submitted course evaluation baseline bit-for-bit.
"""


def package_bundle(repo_root: Path = REPO_ROOT) -> int:
    repo_root = Path(repo_root).resolve()
    output_dir = repo_root / ".scratch" / "evidence_bundle"
    archive_path = repo_root / ".scratch" / "ai_risk_radar_frozen_evidence.tar.gz"

    print("=" * 80)
    print("PACKAGING LOCAL REPRODUCIBLE FROZEN EVIDENCE BUNDLE")
    print("=" * 80)

    # 1. Verify before packaging
    passed, _results = verify_bundle(repo_root)
    if not passed:
        print("[ERROR] Verification failed! Cannot package untrusted or tampered artifacts.")
        return 1

    # 2. Prepare staging directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, object]] = []
    checksum_lines: list[str] = []

    # Copy release artifacts
    for rel_path in AUTHORITATIVE_RELEASE_HASHES:
        src = repo_root / rel_path
        dst = output_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        digest = sha256_file(src)
        size = src.stat().st_size
        manifest_entries.append({
            "path": rel_path,
            "sha256": digest,
            "bytes": size,
            "category": "frozen_release_artifact",
        })
        checksum_lines.append(f"{digest}  {rel_path}")

    # Copy snapshot files if present locally
    for rel_path in AUTHORITATIVE_SNAPSHOT_HASHES:
        src = repo_root / rel_path
        if src.is_file():
            dst = output_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            digest = sha256_file(src)
            size = src.stat().st_size
            manifest_entries.append({
                "path": rel_path,
                "sha256": digest,
                "bytes": size,
                "category": "historical_evaluation_snapshot",
            })
            checksum_lines.append(f"{digest}  {rel_path}")

    # 3. Write licensing and manifest documents
    (output_dir / "LICENSING_AND_DISTRIBUTION.md").write_text(LICENSING_TEXT, encoding="utf-8")
    (output_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    bundle_manifest = {
        "bundle_name": "ai_risk_radar_frozen_evidence",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_files": len(manifest_entries),
        "license_compliance": "Redistribution restricted under exchange / vendor terms; client-side ingestion and synthetic fixture supported",
        "artifacts": manifest_entries,
    }
    (output_dir / "evidence_manifest.json").write_text(
        json.dumps(bundle_manifest, indent=2), encoding="utf-8"
    )

    # 4. Create compressed tarball
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(output_dir, arcname="ai_risk_radar_evidence")

    # Clean up intermediate uncompressed staging folder
    if output_dir.exists():
        shutil.rmtree(output_dir)

    print(f"[SUCCESS] Compressed archive created: {archive_path} ({archive_path.stat().st_size:,} bytes)")
    print("[NOTICE] Prepared strictly for local review and audit inspection. No external asset upload.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(package_bundle())
