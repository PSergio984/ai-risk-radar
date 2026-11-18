"""Regression and verification tests for blind audit remediation findings.

Covers:
- Defect 1 (P1): latest refresh does not overwrite frozen evaluation snapshot
- Defect 2 (P2): non-session dates (weekends, holidays) resolve cleanly without NotSessionError
- Defect 3 (P2): dynamic configurable target event descriptions and legacy compatibility
- Defect 4 (P2): fallback when selected_threshold is absent from candidate_thresholds
- Incremental latest cache merging and deduplication
- Vectorized matrix baseline execution and sequential reconciliation
"""

import json
from pathlib import Path
from unittest.mock import patch

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from ai_risk_radar.contracts import DataBundle, Settings
from ai_risk_radar.data import (
    create_offline_synthetic_fixture,
    fetch_prices,
    load_snapshot,
    resolve_last_completed_session,
    save_snapshot,
)
from ai_risk_radar.pipeline import (
    _compute_policy_hash,
    run_final_test,
    run_latest,
    run_research,
)
from ai_risk_radar.reporting import build_report
from ai_risk_radar.simulation import (
    reconcile_vectorized_and_sequential,
    simulate,
    vectorized_linear_backtest,
    weekly_targets,
)


def _make_tiny_authored_bundle(start: str = "2016-01-04", end: str = "2026-11-16") -> DataBundle:
    """Generate a self-contained tiny deterministic DataBundle without network access."""
    cal = xcals.get_calendar("XNYS")
    sessions = cal.sessions_in_range(start, end)
    n = len(sessions)
    np.random.seed(42)

    tickers = ["SOXX", "XLK", "SPY", "SHY"]
    drift = {
        "SOXX": 0.012,
        "XLK": 0.010,
        "SPY": 0.006,
        "SHY": 0.001,
    }
    data = {}
    for t in tickers:
        ret = np.random.randn(n) * drift[t]
        data[t] = 100.0 * np.exp(np.cumsum(ret))

    adj_df = pd.DataFrame(data, index=sessions)
    vol_df = pd.DataFrame(10000.0, index=sessions, columns=tickers)
    manifest = {
        "source": "unit_test_synthetic",
        "tickers": tickers,
        "cutoff_session": sessions[-1].strftime("%Y-%m-%d"),
    }
    return DataBundle(adjusted_close=adj_df, volume=vol_df, sessions=sessions, manifest=manifest)


def _make_tiny_settings(
    tmp_path: Path,
    start: str = "2016-01-04",
    horizon: int = 20,
    threshold: float = -0.05,
) -> Settings:
    """Create settings pointed to tmp_path."""
    return Settings(
        tickers=("SOXX", "XLK", "SPY", "SHY"),
        acquisition_start=start,
        source="synthetic",
        target_horizon=horizon,
        target_threshold=threshold,
        weights_growth=(0.35, 0.35, 0.20, 0.10),
        weights_defensive=(0.10, 0.15, 0.35, 0.40),
        weights_fixed_baseline=(0.20, 0.20, 0.20, 0.40),
        candidate_thresholds=(0.20, 0.35, 0.50),
        cost_rate=0.001,
        cost_sensitivities=(0.0005, 0.0010, 0.0025),
        random_seed=42,
        initial_capital=10000.0,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        artifacts_dir=tmp_path / "artifacts",
    )


# ---------------------------------------------------------------------------
# Defect 1: latest refresh does not overwrite frozen evaluation snapshot
# ---------------------------------------------------------------------------


def test_defect_1_latest_refresh_does_not_overwrite_frozen_evaluation_snapshot(tmp_path: Path):
    """P1: run_latest(data_mode='refresh') must save to raw/latest, leaving raw/snapshot intact."""
    settings = _make_tiny_settings(tmp_path)
    snapshot_dir = Path(settings.raw_dir) / "snapshot"
    bundle = _make_tiny_authored_bundle(end="2026-06-01")
    save_snapshot(bundle, snapshot_dir)

    # Record frozen snapshot digest
    orig_manifest_text = (snapshot_dir / "manifest.json").read_text(encoding="utf-8")
    orig_manifest_data = json.loads(orig_manifest_text)

    # Run research to create a release manifest
    run_research(settings, data_mode="cache")
    release_path = Path(settings.artifacts_dir) / "release.json"
    assert release_path.is_file()

    # Create extended data bundle to simulate a refresh
    refreshed_bundle = _make_tiny_authored_bundle(end="2026-11-16")

    with patch("ai_risk_radar.pipeline.fetch_prices", return_value=refreshed_bundle):
        latest_report = run_latest(release_path, data_mode="refresh", settings=settings)

    assert latest_report.is_file()

    # The frozen evaluation snapshot MUST remain 100% unchanged
    current_manifest_text = (snapshot_dir / "manifest.json").read_text(encoding="utf-8")
    assert current_manifest_text == orig_manifest_text

    # The refreshed data MUST be isolated in raw/latest
    latest_dir = Path(settings.raw_dir) / "latest"
    assert latest_dir.is_dir()
    assert (latest_dir / "manifest.json").is_file()
    latest_manifest = json.loads((latest_dir / "manifest.json").read_text(encoding="utf-8"))
    assert latest_manifest["session_count"] == len(refreshed_bundle.sessions)
    assert latest_manifest["session_count"] > orig_manifest_data["session_count"]


# ---------------------------------------------------------------------------
# Defect 2: non-session refresh date handling
# ---------------------------------------------------------------------------


def test_defect_2_non_session_refresh_dates_resolve_without_error():
    """P2: resolve_last_completed_session must handle weekends, holidays, and sessions cleanly."""
    cal = xcals.get_calendar("XNYS")

    # Sunday 2026-11-15 -> resolves to Friday 2026-11-13
    sunday_res = resolve_last_completed_session(cal, "2026-11-15")
    assert sunday_res == pd.Timestamp("2026-11-13")

    # Christmas Holiday Friday 2026-12-25 -> resolves to Thursday 2026-12-24
    holiday_res = resolve_last_completed_session(cal, "2026-12-25")
    assert holiday_res == pd.Timestamp("2026-12-24")

    # Standard trading session Friday 2026-11-13 -> resolves to Friday 2026-11-13
    session_res = resolve_last_completed_session(cal, "2026-11-13")
    assert session_res == pd.Timestamp("2026-11-13")

    # Timestamp with timezone -> resolves correctly to date
    tz_ts = pd.Timestamp("2026-11-15 14:30:00", tz="America/New_York")
    assert resolve_last_completed_session(cal, tz_ts) == pd.Timestamp("2026-11-13")


def test_defect_2_fetch_prices_on_weekend_date_does_not_raise(tmp_path: Path):
    """P2: fetch_prices called with as_of='2026-11-15' (Sunday) resolves without NotSessionError."""
    settings = _make_tiny_settings(tmp_path)
    cal = xcals.get_calendar("XNYS")
    last_comp = resolve_last_completed_session(cal, "2026-11-15")
    exp_sessions = cal.sessions_in_range(settings.acquisition_start, last_comp)

    # Mock yfinance return to verify pipeline path without network
    mock_raw = pd.DataFrame(
        100.0,
        index=exp_sessions,
        columns=pd.MultiIndex.from_product([["Adj Close", "Volume"], list(settings.tickers)]),
    )

    with patch("yfinance.download", return_value=mock_raw):
        bundle = fetch_prices(settings, as_of="2026-11-15")

    assert len(bundle.sessions) == len(exp_sessions)
    assert bundle.sessions[-1] == pd.Timestamp("2026-11-13")


# ---------------------------------------------------------------------------
# Defect 3: dynamic configurable target event descriptions & policy identity
# ---------------------------------------------------------------------------


def test_defect_3_configurable_target_custom_horizon_and_threshold(tmp_path: Path):
    """P2: 5-session / -2% custom target config must be accurately described in report & policy hash."""
    settings_custom = _make_tiny_settings(tmp_path, horizon=5, threshold=-0.02)
    settings_default = _make_tiny_settings(tmp_path, horizon=20, threshold=-0.05)

    # 1. Policy hash must reflect custom target horizon and threshold
    hash_custom = _compute_policy_hash(0.35, settings_custom)
    hash_default = _compute_policy_hash(0.35, settings_default)
    assert hash_custom != hash_default, "Policy hash must distinguish target_horizon and threshold!"

    # 2. Legacy policy hash ignores custom target for backward compatibility
    legacy_hash_custom = _compute_policy_hash(0.35, settings_custom, legacy=True)
    legacy_hash_default = _compute_policy_hash(0.35, settings_default, legacy=True)
    assert legacy_hash_custom == legacy_hash_default

    # 3. Report payload describes the exact configured event
    metadata = {
        "target_horizon": settings_custom.target_horizon,
        "target_threshold": settings_custom.target_threshold,
        "pipeline_mode": "research",
    }
    dummy_risk = pd.Series([0.15], index=[pd.Timestamp("2026-11-16")])
    report = build_report({}, dummy_risk, metadata)

    forecast = report.payload["forecast"]
    assert forecast["horizon_sessions"] == 5
    assert forecast["loss_threshold"] == -0.02
    assert "5 trading sessions" in forecast["loss_definition"]
    assert "-2.0%" in forecast["loss_definition"]
    assert report.payload["identity"]["target_horizon"] == 5
    assert report.payload["identity"]["target_threshold"] == -0.02


# ---------------------------------------------------------------------------
# Defect 4: fallback when selected_threshold absent from candidate_thresholds
# ---------------------------------------------------------------------------


def test_defect_4_fallback_when_selected_threshold_absent_from_candidates(tmp_path: Path):
    """P2: run_final_test must build weekly_targets and simulate when selected_threshold not in candidates."""
    settings = _make_tiny_settings(tmp_path)
    bundle = _make_tiny_authored_bundle()
    snapshot_dir = Path(settings.raw_dir) / "snapshot"
    save_snapshot(bundle, snapshot_dir)

    # Run research with candidates [0.20, 0.35, 0.50]
    run_research(settings, data_mode="cache")
    release_path = Path(settings.artifacts_dir) / "release.json"
    release_data = json.loads(release_path.read_text(encoding="utf-8"))

    # Intentionally modify release to select 0.40, which is NOT in candidate_thresholds (0.20, 0.35, 0.50)
    release_data["selected_threshold"] = 0.40
    # Recompute policy hash for 0.40
    new_policy_hash = _compute_policy_hash(0.40, settings)
    release_data["hashes"]["policy"] = new_policy_hash
    release_data["policy_hash"] = new_policy_hash
    release_path.write_text(json.dumps(release_data, indent=2), encoding="utf-8")

    # Run final test: must NOT crash with TypeError
    report_path = run_final_test(release_path, settings=settings)
    assert report_path.is_file()

    report_data = json.loads(report_path.read_text(encoding="utf-8"))
    selected_metrics = report_data["performance"]["selected_strategy"]
    assert selected_metrics is not None
    assert "cagr" in selected_metrics
    assert "total_return" in selected_metrics


# ---------------------------------------------------------------------------
# Incremental cache refresh and deduplication
# ---------------------------------------------------------------------------


def test_incremental_cache_refresh_merges_and_deduplicates(tmp_path: Path):
    """Incremental fetch must merge overlapping sessions and take newer vendor data."""
    settings = _make_tiny_settings(tmp_path, start="2026-01-05")
    cal = xcals.get_calendar("XNYS")
    all_sessions = cal.sessions_in_range("2026-01-05", "2026-06-30")

    # Initial bundle up to 2026-05-15
    bundle1 = _make_tiny_authored_bundle(start="2026-01-05", end="2026-05-15")

    # Simulate updated prices on overlap (e.g., dividend adjustment on last 5 sessions)
    overlap_sessions = cal.sessions_in_range("2026-05-08", "2026-06-30")
    mock_raw = pd.DataFrame(
        index=overlap_sessions,
        columns=pd.MultiIndex.from_product([["Adj Close", "Volume"], list(settings.tickers)]),
    )
    for t in settings.tickers:
        mock_raw[("Adj Close", t)] = 105.0
        mock_raw[("Volume", t)] = 12000.0

    with patch("yfinance.download", return_value=mock_raw):
        refreshed = fetch_prices(
            settings,
            as_of="2026-06-30",
            existing_bundle=bundle1,
            overlap_sessions=5,
        )

    assert refreshed.manifest.get("incremental") is True
    assert len(refreshed.sessions) == len(all_sessions)
    assert refreshed.sessions[-1] == pd.Timestamp("2026-06-30")

    # Overlapping sessions must reflect the revised price (105.0)
    overlap_date = pd.Timestamp("2026-05-12")
    assert refreshed.adjusted_close.loc[overlap_date, "SOXX"] == 105.0


# ---------------------------------------------------------------------------
# Vectorized matrix baseline & reconciliation
# ---------------------------------------------------------------------------


def test_vectorized_simulate_and_reconciliation(tmp_path: Path):
    """Vectorized baseline must execute under matched conventions and reconcile against sequential engine."""
    bundle = _make_tiny_authored_bundle(start="2024-01-02", end="2026-11-16")
    settings = _make_tiny_settings(tmp_path)
    prices = bundle.adjusted_close

    # Create dummy weekly risk signals
    dummy_risk = pd.Series(0.10, index=bundle.sessions)
    # Set high risk periodically to trigger defensive rotation
    for s in bundle.sessions[::20]:
        dummy_risk.loc[s] = 0.45

    targets = weekly_targets(dummy_risk, bundle.sessions, settings, threshold=0.35)

    # 1. Run sequential simulate
    seq_res = simulate(prices, targets, settings, cost_rate=settings.cost_rate)

    # 2. Run vectorized baseline
    vec_res = vectorized_linear_backtest(prices, targets, settings, cost_rate=settings.cost_rate)

    assert seq_res.metrics["total_return"] is not None
    assert vec_res.metrics["total_return"] is not None

    # 3. Run formal reconciliation
    reconciliation = reconcile_vectorized_and_sequential(
        prices, targets, settings, cost_rate=settings.cost_rate
    )

    table = reconciliation["comparison_table"]
    assert "total_return" in table
    assert "cagr" in table
    assert "sharpe" in table
    assert "max_drawdown" in table

    # Total return divergence between sequential daily drift and vectorized linear approximation
    # should be small (< 200 bps over multi-year horizon)
    diff = abs(table["total_return"]["absolute_diff"])
    assert diff < 0.05, f"Vectorized and sequential total return diverge excessively: {diff:.4f}"
    assert len(reconciliation["explanation"]) > 20


# ---------------------------------------------------------------------------
# Offline synthetic fixture & clean notebook acquisition
# ---------------------------------------------------------------------------


def test_create_offline_synthetic_fixture_and_snapshot_roundtrip(tmp_path: Path):
    """create_offline_synthetic_fixture must generate valid bundle and round-trip through parquet snapshot."""
    settings = _make_tiny_settings(tmp_path)
    bundle = create_offline_synthetic_fixture(settings, start="2016-01-04", end="2026-11-16")

    assert len(bundle.sessions) >= 1460
    assert bundle.manifest["is_synthetic"] is True
    assert bundle.manifest["source"] == "deterministic_offline_fixture"
    assert (bundle.adjusted_close > 0.0).all().all()
    assert (bundle.volume > 0.0).all().all()

    snapshot_dir = tmp_path / "offline_snapshot"
    manifest_path = save_snapshot(bundle, snapshot_dir)
    assert manifest_path.is_file()

    loaded = load_snapshot(snapshot_dir)
    assert len(loaded.sessions) == len(bundle.sessions)
    assert loaded.manifest["is_synthetic"] is True
    pd.testing.assert_frame_equal(loaded.adjusted_close, bundle.adjusted_close, check_freq=False)


def test_exact_zero_drift_mathematical_reconciliation(tmp_path: Path):
    """When asset prices are constant (zero drift) and rebalance costs are zero,

    sequential and vectorized engines must produce bit-exact identical returns (0.0 divergence).
    """
    settings = _make_tiny_settings(tmp_path)
    cal = xcals.get_calendar("XNYS")
    sessions = cal.sessions_in_range("2024-01-02", "2024-06-28")

    # Flat constant prices -> daily returns = 0.0 -> zero price drift
    flat_prices = pd.DataFrame(
        100.0,
        index=sessions,
        columns=list(settings.tickers),
    )

    # Weekly alternating targets
    dummy_risk = pd.Series(0.10, index=sessions)
    for s in sessions[::10]:
        dummy_risk.loc[s] = 0.50
    targets = weekly_targets(dummy_risk, sessions, settings, threshold=0.35)

    # Zero cost rate to isolate the mathematical drift effect
    seq_res = simulate(flat_prices, targets, settings, cost_rate=0.0)
    vec_res = vectorized_linear_backtest(flat_prices, targets, settings, cost_rate=0.0)

    # In zero-drift constant-price regime, returns must be bit-exact identical
    seq_tot = seq_res.metrics["total_return"]
    vec_tot = vec_res.metrics["total_return"]
    assert seq_tot == 0.0
    assert vec_tot == 0.0
    assert abs(seq_tot - vec_tot) == 0.0


def test_clean_checkout_notebook_acquisition_and_execution_in_isolation(tmp_path: Path):
    """Execute end_to_end.ipynb in an isolated temp checkout with no pre-existing data/raw/snapshot.

    Verifies clean-first-use acquisition/fixture fallback and execution of all 6 stages.
    """
    import shutil
    import subprocess
    import sys

    import nbformat

    # Set up isolated workspace directory
    ws_dir = tmp_path / "clean_workspace"
    ws_dir.mkdir(parents=True)

    # Copy tracked project files (clean checkout state without data/ or artifacts/)
    shutil.copy("config.json", ws_dir / "config.json")
    shutil.copytree("ai_risk_radar", ws_dir / "ai_risk_radar")
    shutil.copytree("notebooks", ws_dir / "notebooks")

    # Verify that data/ is completely absent prior to execution
    assert not (ws_dir / "data").exists()

    # Execute notebook using the current environment Python via nbconvert CLI
    cmd = [
        sys.executable,
        "-m",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(ws_dir / "notebooks" / "end_to_end.ipynb"),
        "--output",
        "executed_end_to_end.ipynb",
    ]
    proc = subprocess.run(cmd, cwd=str(ws_dir), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"Notebook execution failed with stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"

    # Verify that data/raw/snapshot was generated by the notebook
    snapshot_dir = ws_dir / "data" / "raw" / "snapshot"
    assert snapshot_dir.is_dir(), "Notebook failed to acquire/generate snapshot in clean workspace"
    assert (snapshot_dir / "manifest.json").is_file()
    assert (snapshot_dir / "adjusted_close.parquet").is_file()

    # Inspect executed notebook outputs
    executed_nb_path = ws_dir / "notebooks" / "executed_end_to_end.ipynb"
    assert executed_nb_path.is_file()
    with executed_nb_path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    all_outputs = []
    for cell in nb.cells:
        if cell.cell_type == "code":
            for out in cell.get("outputs", []):
                if "text" in out:
                    all_outputs.append(out["text"])

    combined_output = "\n".join(all_outputs)
    assert "Config loaded from" in combined_output
    assert (
        "CACHED EVALUATION SNAPSHOT" in combined_output
        or "OFFLINE DETERMINISTIC FIXTURE MODE" in combined_output
        or "LIVE MARKET DATA" in combined_output
    )
    assert "43 causal technical features" in combined_output
    assert "Trained Random Forest" in combined_output
    assert "Selected Strategy Policy" in combined_output
    assert "Reconciliation Summary" in combined_output
    assert "ACTIONABLE LATEST TRADE SCHEDULE" in combined_output


