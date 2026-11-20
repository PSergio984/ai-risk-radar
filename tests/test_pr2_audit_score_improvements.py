"""Regression tests for PR2 audit findings and rubric score improvements.

Covers:
1. Notebook one-class prediction crash avoidance (using predict_risk / classes_ handling).
2. Non-round threshold collision avoidance (e.g. 0.354 vs 0.35).
3. Active data hash change detecting compound release hash divergence in run_latest.
4. Intervening NYSE exchange session staleness arithmetic across weekends/holidays.
5. Distinct fallback explanation for calibrated probability with selected_threshold=None.
6. HistGradientBoostingClassifier comparator in fit_models using purged temporal evaluation.
7. Causal regime and subgroup error analysis.
"""

import json
from dataclasses import replace
from pathlib import Path

import exchange_calendars as xcals
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ai_risk_radar.config import load_settings
from ai_risk_radar.contracts import BacktestResult, DataBundle, FeatureSet, ModelBundle, SplitSet
from ai_risk_radar.data import save_snapshot
from ai_risk_radar.models import fit_models, predict_risk
from ai_risk_radar.pipeline import (
    _compute_policy_hash,
    _compute_release_hash,
    _compute_sha256,
    format_threshold_key,
    parse_threshold_key,
    run_latest,
)
from ai_risk_radar.reporting import build_report
from ai_risk_radar.simulation import select_strategy

REPO_ROOT = Path(__file__).resolve().parents[1]


# ==============================================================================
# Bug 1: One-Class Prediction Crash Avoidance
# ==============================================================================

@pytest.mark.parametrize("class_label", [0.0, 1.0])
def test_one_class_estimator_predict_risk_safe(class_label):
    """Assert predict_risk does not raise IndexError [:, 1] on a one-class trained estimator."""
    frame = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0, 4.0]}, index=pd.date_range("2024-01-01", periods=4, freq="B"))
    rf = RandomForestClassifier(n_estimators=1, random_state=42)
    rf.fit(frame, [class_label] * 4)

    bundle = ModelBundle(
        estimator=rf,
        calibrator=None,
        feature_columns=("feat_a",),
        metadata={"probability_status": "unavailable", "reason": "Single class"},
    )

    preds = predict_risk(bundle, frame)
    assert len(preds) == len(frame)
    np.testing.assert_allclose(preds.values, class_label)


# ==============================================================================
# Bug 2: Threshold Formatting and Collision Avoidance
# ==============================================================================

def test_threshold_key_precision_no_collision():
    """Assert close float thresholds (e.g. 0.35 and 0.354) produce distinct keys and exact round-trip."""
    th1 = 0.35
    th2 = 0.354

    key1 = format_threshold_key(th1)
    key2 = format_threshold_key(th2)

    assert key1 != key2, f"Key collision detected: {key1} vs {key2}"
    assert parse_threshold_key(key1) == pytest.approx(th1, abs=1e-6)
    assert parse_threshold_key(key2) == pytest.approx(th2, abs=1e-6)

    # Standard thresholds keep their canonical 2-decimal format
    assert format_threshold_key(0.20) == "threshold_0.20"
    assert format_threshold_key(0.35) == "threshold_0.35"
    assert format_threshold_key(0.50) == "threshold_0.50"


def test_select_strategy_with_high_precision_thresholds():
    """Assert strategy selection preserves non-round float threshold exactly."""
    th_val = 0.354
    key = format_threshold_key(th_val)

    baseline = BacktestResult(
        daily=pd.DataFrame(),
        trades=pd.DataFrame(),
        metrics={"cagr": 0.05, "max_drawdown": -0.20, "turnover": 0.5},
    )
    candidates = {
        key: BacktestResult(
            daily=pd.DataFrame(),
            trades=pd.DataFrame(),
            metrics={"cagr": 0.12, "max_drawdown": -0.10, "turnover": 0.8},
        )
    }

    selected_key = select_strategy(candidates, baseline)
    assert selected_key == key
    parsed_th = parse_threshold_key(selected_key)
    assert parsed_th == pytest.approx(0.354, abs=1e-6)


# ==============================================================================
# Bug 3: Compound Release Hash Divergence Detection in run_latest
# ==============================================================================

def test_run_latest_detects_snapshot_manifest_divergence(tmp_path: Path):
    """When active snapshot manifest changes, run_latest must compute actual compound hash, not reuse old release hash."""
    settings = replace(
        load_settings(REPO_ROOT / "config.json"),
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        artifacts_dir=tmp_path / "artifacts",
        random_seed=42,
    )
    sessions = pd.bdate_range("2016-01-04", "2026-11-16")
    rng = np.random.default_rng(42)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(sessions), 4)), axis=0)),
        index=sessions,
        columns=settings.tickers,
    )
    bundle = DataBundle(
        adjusted_close=prices,
        volume=pd.DataFrame(10000, index=sessions, columns=settings.tickers),
        sessions=sessions,
        manifest={"source": "fixture_a", "cutoff_session": "2026-11-16"},
    )
    snapshot_dir = settings.raw_dir / "snapshot"
    save_snapshot(bundle, snapshot_dir)

    # Train dummy model
    rf = RandomForestClassifier(n_estimators=2, max_depth=1, random_state=42)
    rf.fit(pd.DataFrame({"ret_soxx_1": [-0.02, -0.01, 0.01, 0.02]}), [1, 1, 0, 0])
    model_path = settings.artifacts_dir / "models" / "random_forest.joblib"
    model_path.parent.mkdir(parents=True)
    joblib.dump(
        ModelBundle(rf, None, ("ret_soxx_1",), {"probability_status": "unavailable", "reason": "Test"}),
        model_path,
    )

    model_hash = _compute_sha256(model_path)
    old_data_hash = "deadbeef" * 8
    policy_hash = _compute_policy_hash(None, settings)
    old_compound_hash = _compute_release_hash(model_hash, old_data_hash, policy_hash)

    release_payload = {
        "release_version": "1.0.0",
        "selected_threshold": None,
        "target_horizon": 20,
        "target_threshold": -0.05,
        "boundaries": {"cutoff": "2026-11-16"},
        "hashes": {
            "model": model_hash,
            "data": old_data_hash,
            "policy": policy_hash,
            "release": old_compound_hash,
        },
    }
    release_path = settings.artifacts_dir / "release.json"
    release_path.write_text(json.dumps(release_payload), encoding="utf-8")

    report_path = run_latest(release_path, settings=settings, data_mode="cache")
    report_data = json.load(report_path.open("r", encoding="utf-8"))

    actual_data_hash = _compute_sha256(snapshot_dir / "manifest.json")
    assert actual_data_hash != old_data_hash

    # Must NOT report the stale compound hash
    assert report_data["identity"]["release_hash"] != old_compound_hash
    expected_new_compound = _compute_release_hash(model_hash, actual_data_hash, policy_hash)
    assert report_data["identity"]["release_hash"] == expected_new_compound


# ==============================================================================
# Bug 4: Exchange Calendar Trading Sessions Staleness
# ==============================================================================

def test_staleness_counts_exchange_sessions_not_calendar_days(tmp_path: Path):
    """Over a weekend or holiday, staleness_sessions must count exchange trading sessions, not calendar days."""
    cal = xcals.get_calendar("XNYS")

    # Friday 2026-11-13 close
    as_of_session = pd.Timestamp("2026-11-13")

    # Saturday 2026-11-14 (0 exchange sessions have closed since Friday close)
    sat_time = pd.Timestamp("2026-11-14 12:00:00", tz="America/New_York")
    from ai_risk_radar.data import resolve_last_completed_session
    sat_last = resolve_last_completed_session(cal, sat_time)
    assert sat_last == as_of_session
    # Intervening trading sessions between Friday and Saturday last completed: 0
    sat_staleness = max(0, len(cal.sessions_in_range(as_of_session, sat_last)) - 1) if as_of_session < sat_last else 0
    assert sat_staleness == 0, f"Saturday staleness should be 0, got {sat_staleness}"

    # Sunday 2026-11-15: still 0 exchange sessions closed
    sun_time = pd.Timestamp("2026-11-15 18:00:00", tz="America/New_York")
    sun_last = resolve_last_completed_session(cal, sun_time)
    assert sun_last == as_of_session
    sun_staleness = max(0, len(cal.sessions_in_range(as_of_session, sun_last)) - 1) if as_of_session < sun_last else 0
    assert sun_staleness == 0

    # Monday morning 2026-11-16 10:00 AM (before close: still 0 completed sessions since Friday)
    mon_morning = pd.Timestamp("2026-11-16 10:00:00", tz="America/New_York")
    mon_m_last = resolve_last_completed_session(cal, mon_morning)
    assert mon_m_last == as_of_session
    mon_m_staleness = max(0, len(cal.sessions_in_range(as_of_session, mon_m_last)) - 1) if as_of_session < mon_m_last else 0
    assert mon_m_staleness == 0

    # Monday evening 2026-11-16 17:00 (after close: exactly 1 trading session has closed)
    mon_eve = pd.Timestamp("2026-11-16 17:00:00", tz="America/New_York")
    mon_e_last = resolve_last_completed_session(cal, mon_eve)
    assert mon_e_last == pd.Timestamp("2026-11-16")
    mon_e_staleness = max(0, len(cal.sessions_in_range(as_of_session, mon_e_last)) - 1)
    assert mon_e_staleness == 1


# ==============================================================================
# Bug 5: Fallback Explanation Disambiguation
# ==============================================================================

def test_reporting_fallback_explanation_distinguishes_probability_vs_policy():
    """When probability is calibrated but selected_threshold is None, explanation must cite policy, not unavailable probability."""
    forecast_series = pd.Series([0.18], index=[pd.Timestamp("2026-11-16")])

    # Case A: Probability is calibrated, but no adaptive policy qualified (selected_threshold is None)
    metadata_policy_fallback = {
        "probability_status": "calibrated",
        "selected_threshold": None,
        "tickers": ["SOXX", "XLK", "SPY", "SHY"],
    }
    report_a = build_report({}, forecast_series, metadata_policy_fallback)
    desc_a = report_a.payload["allocation"]["trade_description"]
    assert "Probability unavailable" not in desc_a
    assert "policy" in desc_a.lower() or "qualifying" in desc_a.lower() or "threshold" in desc_a.lower()

    # Case B: Probability is genuinely unavailable
    metadata_prob_unavailable = {
        "probability_status": "unavailable",
        "selected_threshold": 0.35,
        "tickers": ["SOXX", "XLK", "SPY", "SHY"],
    }
    report_b = build_report({}, forecast_series, metadata_prob_unavailable)
    desc_b = report_b.payload["allocation"]["trade_description"]
    assert "Probability unavailable" in desc_b


# ==============================================================================
# Priority 3: HistGradientBoostingClassifier Comparator
# ==============================================================================

def test_fit_models_includes_hist_gradient_boosting_comparator():
    """fit_models includes HistGradientBoostingClassifier evaluated on purged temporal folds."""
    settings = load_settings(REPO_ROOT / "config.json")
    sessions = pd.bdate_range("2018-01-02", periods=500)
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "feat_1": rng.normal(0, 1, 500),
            "feat_2": rng.normal(0, 1, 500),
            "downside20": (rng.uniform(0, 1, 500) < 0.25).astype(float),
        },
        index=sessions,
    )
    features = FeatureSet(frame, ("feat_1", "feat_2"), ("downside20",))
    splits = SplitSet(sessions[:300], sessions[300:400], sessions[400:450], sessions[450:], {})

    models = fit_models(features, splits, settings)
    assert "hist_gradient_boosting" in models

    hgb_bundle = models["hist_gradient_boosting"]
    assert hgb_bundle.metadata["model_name"] == "hist_gradient_boosting"
    assert "cv_brier_score" in hgb_bundle.metadata
    assert hgb_bundle.metadata["cv_brier_score"] is not None

    preds = predict_risk(hgb_bundle, frame.loc[splits.validation])
    assert len(preds) == len(splits.validation)
    assert (preds >= 0.0).all() and (preds <= 1.0).all()


# ==============================================================================
# Priority 2: Regime and Subgroup Error Analysis
# ==============================================================================

def test_compute_regime_analysis():
    """Assert compute_regime_analysis provides annual, trend, and volatility breakdowns using causal features."""
    from ai_risk_radar.analysis import compute_regime_analysis

    settings = load_settings(REPO_ROOT / "config.json")
    sessions = pd.bdate_range("2020-01-02", periods=600)
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "ret_spy_63": rng.normal(0.02, 0.05, 600),
            "vol_spy_20": np.abs(rng.normal(0.15, 0.03, 600)),
            "downside20": (rng.uniform(0, 1, 600) < 0.20).astype(float),
        },
        index=sessions,
    )
    features = FeatureSet(frame, ("ret_spy_63", "vol_spy_20"), ("downside20",))
    splits = SplitSet(sessions[:350], sessions[350:450], sessions[450:550], sessions[550:], {})

    models = fit_models(features, splits, settings)
    rf_bundle = models["random_forest"]

    analysis = compute_regime_analysis(rf_bundle, features, splits)

    assert "partitions" in analysis
    assert set(analysis["partitions"].keys()) == {"train", "calibration", "validation"}
    assert "annual_breakdown" in analysis
    assert len(analysis["annual_breakdown"]) > 0
    assert "trend_regimes" in analysis
    assert "bull_market" in analysis["trend_regimes"]
    assert "bear_market" in analysis["trend_regimes"]
    assert "volatility_regimes" in analysis
    assert "low_volatility" in analysis["volatility_regimes"]
    assert "high_volatility" in analysis["volatility_regimes"]


# ==============================================================================
# Priority 1: Isolated End-to-End Notebook Execution (All Cells)
# ==============================================================================

def test_notebook_all_cells_execute_in_fixture_mode(monkeypatch, tmp_path: Path):
    """Assert all notebook code cells execute sequentially without error in offline fixture mode."""
    monkeypatch.setenv("AI_RISK_RADAR_NOTEBOOK_MODE", "fixture")
    nb = json.loads((REPO_ROOT / "notebooks" / "end_to_end.ipynb").read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    scope = {"__name__": "__main__"}

    for idx, cell in enumerate(code_cells):
        code = "".join(cell["source"])
        exec(compile(code, f"cell_{idx}.py", "exec"), scope)  # noqa: S102

    assert "bundle" in scope
    assert "features" in scope
    assert "models" in scope
    assert "regime_analysis" in scope
    assert "schedule_df" in scope
    assert len(scope["schedule_df"]) == 4


def test_research_notebook_all_cells_execute_and_have_saved_outputs(monkeypatch):
    """Assert all research.ipynb code cells have saved outputs and execute cleanly in fixture mode."""
    monkeypatch.setenv("AI_RISK_RADAR_NOTEBOOK_MODE", "fixture")
    nb = json.loads((REPO_ROOT / "notebooks" / "research.ipynb").read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 5
    for c in code_cells:
        assert len(c.get("outputs", [])) > 0, "Research notebook code cell missing saved outputs"

    scope = {"__name__": "__main__"}
    for idx, cell in enumerate(code_cells):
        code = "".join(cell["source"])
        exec(compile(code, f"research_cell_{idx}.py", "exec"), scope)  # noqa: S102

    assert "bundle" in scope
    assert "features" in scope
    assert "splits" in scope
    assert "corr" in scope


def test_app_fallback_banner_distinguishes_missing_prob_vs_policy():
    """Verify resolve_fallback_banner in app.py treats missing/NaN probability as unavailable error."""
    from app import resolve_fallback_banner

    # Case 1: probability is None despite calibrated status -> should trigger unavailable error
    view_none_prob = {
        "probability_status": "calibrated",
        "probability": None,
        "allocation": {"strategy_mode": "Fixed Baseline (Fallback)"},
    }
    banner = resolve_fallback_banner(view_none_prob)
    assert banner is not None
    assert banner[0] == "error"
    assert "Risk Probability Unavailable" in banner[1]
    assert "Model probability unavailable" in banner[1]

    # Case 2: probability is NaN -> should trigger unavailable error
    view_nan_prob = {
        "probability_status": "calibrated",
        "probability": float("nan"),
        "fallback_reason": None,
        "allocation": {"strategy_mode": "Fixed Baseline (Fallback)"},
    }
    banner = resolve_fallback_banner(view_nan_prob)
    assert banner is not None
    assert banner[0] == "error"
    assert "Risk Probability Unavailable" in banner[1]
    assert "None" not in banner[1]
    assert "Model probability unavailable" in banner[1]

    # Case 3: valid probability, but policy fallback -> should trigger adaptive policy info
    view_valid_prob_policy_fallback = {
        "probability_status": "calibrated",
        "probability": 0.15,
        "allocation": {"strategy_mode": "Fixed Baseline (Fallback)"},
    }
    banner = resolve_fallback_banner(view_valid_prob_policy_fallback)
    assert banner is not None
    assert banner[0] == "info"
    assert "Adaptive Policy Fallback" in banner[1]


def test_run_latest_cache_truncates_future_rows_and_enforces_as_of_cutoff(tmp_path: Path):
    """Verify run_latest with data_mode='cache' truncates future rows before feature building and prediction."""
    import dataclasses

    from ai_risk_radar.contracts import DataBundle, ModelBundle
    from ai_risk_radar.data import save_snapshot
    from ai_risk_radar.features import build_features
    from ai_risk_radar.models import ConstantPredictor
    from ai_risk_radar.pipeline import _compute_policy_hash, _compute_sha256, run_latest

    settings = load_settings("config.json")
    test_settings = dataclasses.replace(
        settings,
        raw_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
    )

    cal = xcals.get_calendar("XNYS")
    sessions = cal.sessions_in_range("2025-01-02", "2025-05-30")
    rng = np.random.default_rng(42)
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, (len(sessions), len(test_settings.tickers))), axis=0),
        index=sessions,
        columns=list(test_settings.tickers),
    )
    bundle = DataBundle(
        adjusted_close=prices,
        volume=prices * 1000.0,
        sessions=sessions,
        manifest={"source": "mock_test_cache", "cutoff_session": "2025-05-30"},
    )
    save_snapshot(bundle, test_settings.raw_dir / "snapshot")

    features = build_features(bundle, test_settings)
    model = ModelBundle(
        estimator=ConstantPredictor(0.25),
        calibrator=None,
        feature_columns=features.feature_columns,
        metadata={"probability_status": "calibrated"},
    )
    models_dir = test_settings.artifacts_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "random_forest.joblib"
    joblib.dump(model, model_path)

    release_payload = {
        "hashes": {
            "model": _compute_sha256(model_path),
            "policy": _compute_policy_hash(0.35, test_settings),
            "data": _compute_sha256(test_settings.raw_dir / "snapshot" / "manifest.json"),
        },
        "selected_threshold": 0.35,
    }
    release_path = test_settings.artifacts_dir / "release.json"
    release_path.write_text(json.dumps(release_payload, indent=2), encoding="utf-8")

    # Run latest with as_of='2025-04-30' on a cache extending through 2025-05-30
    report_path = run_latest(release_path, settings=test_settings, as_of="2025-04-30", data_mode="cache")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    # Assert cutoff session is enforced, no future rows are used, and staleness is 0
    assert report["data_status"]["as_of_session"] == "2025-04-30"
    assert report["data_status"]["cutoff_session"] == "2025-04-30"
    assert report["data_status"]["staleness_sessions"] == 0
    assert report["forecast"]["as_of_session"] == "2025-04-30"

    # Verify that as_of earlier than all sessions raises ValueError
    with pytest.raises(ValueError, match="No price data available on or before cutoff session"):
        run_latest(release_path, settings=test_settings, as_of="2024-01-01", data_mode="cache")


def test_format_threshold_key_adjacent_floats_and_validation():
    """Verify format_threshold_key preserves float identity without collision and rejects invalid inputs."""
    # Historical release compatibility
    assert format_threshold_key(0.35) == "threshold_0.35"
    assert format_threshold_key(0.20) == "threshold_0.20"
    assert format_threshold_key(0.50) == "threshold_0.50"

    # Exact float identity and round-trip parsing
    k1 = format_threshold_key(0.3500001)
    k2 = format_threshold_key(0.3500002)
    assert k1 != k2
    assert parse_threshold_key(k1) == 0.3500001
    assert parse_threshold_key(k2) == 0.3500002

    # Validation rejection for nonfinite or invalid inputs
    for invalid_val in [float("nan"), float("inf"), float("-inf"), None, "0.35"]:
        with pytest.raises(ValueError, match="Invalid threshold value"):
            format_threshold_key(invalid_val)  # type: ignore[arg-type]


def test_validate_sessions_and_prices_and_save_snapshot_reject_inf_nan(tmp_path: Path):
    """Verify validate_sessions_and_prices and save_snapshot strictly reject positive/negative infinity."""
    from ai_risk_radar.contracts import DataBundle
    from ai_risk_radar.data import save_snapshot, validate_sessions_and_prices

    cal = xcals.get_calendar("XNYS")
    sessions = cal.sessions_in_range("2025-01-02", "2025-01-10")
    prices = pd.DataFrame(
        100.0,
        index=sessions,
        columns=["SOXX", "XLK", "SPY", "SHY"],
    )
    volume = prices * 100.0

    # Clean data passes
    validate_sessions_and_prices(prices, volume, sessions)

    # Positive infinity in prices
    inf_prices = prices.copy()
    inf_prices.iloc[-1, 0] = float("inf")
    with pytest.raises(ValueError, match="Non-finite price"):
        validate_sessions_and_prices(inf_prices, volume, sessions)

    inf_bundle = DataBundle(inf_prices, volume, sessions, {"source": "test"})
    with pytest.raises(ValueError, match="Non-finite price"):
        save_snapshot(inf_bundle, tmp_path / "inf_snapshot")

    # Negative infinity in volume
    inf_vol = volume.copy()
    inf_vol.iloc[-1, 0] = float("inf")
    with pytest.raises(ValueError, match="Non-finite volume"):
        validate_sessions_and_prices(prices, inf_vol, sessions)


def test_notebook_cell13_unavailable_probability_and_weekly_timing():
    """Verify notebook cell 13 suppresses numeric probability on unavailable status and respects weekly schedule."""
    import contextlib
    import io

    from ai_risk_radar.contracts import DataBundle, ModelBundle
    from ai_risk_radar.models import ConstantPredictor

    settings = load_settings("config.json")
    cell_code = "".join(
        json.loads(Path("notebooks/end_to_end.ipynb").read_text(encoding="utf-8"))["cells"][13]["source"]
    )

    # Test Case 1: Mid-week cutoff (Wednesday 2025-05-28) with unavailable probability
    idx_midweek = pd.DatetimeIndex(["2025-05-28"])
    df_midweek = pd.DataFrame({"dummy": [1.0]}, index=idx_midweek)
    prices_midweek = pd.DataFrame([[100.0] * 4], index=idx_midweek, columns=settings.tickers)
    bundle_midweek = DataBundle(prices_midweek, prices_midweek, idx_midweek, {"source": "test"})
    rf_bundle_unavail = ModelBundle(
        ConstantPredictor(0.8),
        None,
        ("dummy",),
        {"probability_status": "unavailable", "reason": "Training data has only one class; probability unavailable"},
    )

    scope: dict[str, object] = {
        "settings": settings,
        "df": df_midweek,
        "bundle": bundle_midweek,
        "rf_bundle": rf_bundle_unavail,
        "selected_threshold": None,
        "pd": pd,
        "predict_risk": predict_risk,
        "xcals": xcals,
    }

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(cell_code, "cell13_midweek.py", "exec"), scope)  # noqa: S102

    output = buf.getvalue()
    # Downside Risk Probability must be Unavailable, not 80.0%
    assert "Unavailable (Training data has only one class; probability unavailable)" in output
    assert "80.0%" not in output
    assert "Risk Classification Level:          Unavailable" in output
    assert "Active Portfolio Policy Stance:     Defensive Baseline Fallback" in output
    # Scheduled Execution must be the following Monday 2025-06-02, not Thursday 2025-05-29
    assert "2025-06-02 (at Market Close" in output
    assert "2025-05-29" not in output
    # No immediate rebalance during unfinished week: total turnover must be 0
    assert "Total Gross Trade Volume:           $0.00" in output
    assert "Estimated Rebalance Cost (10 bps):  $0.00" in output

    # Test Case 2: Week-final session (Friday 2025-05-30) with calibrated probability
    idx_friday = pd.DatetimeIndex(["2025-05-30"])
    df_friday = pd.DataFrame({"dummy": [1.0]}, index=idx_friday)
    prices_friday = pd.DataFrame([[100.0] * 4], index=idx_friday, columns=settings.tickers)
    bundle_friday = DataBundle(prices_friday, prices_friday, idx_friday, {"source": "test"})
    rf_bundle_calibrated = ModelBundle(
        ConstantPredictor(0.40),
        None,
        ("dummy",),
        {"probability_status": "calibrated"},
    )

    scope_friday: dict[str, object] = {
        "settings": settings,
        "df": df_friday,
        "bundle": bundle_friday,
        "rf_bundle": rf_bundle_calibrated,
        "selected_threshold": 0.35,
        "pd": pd,
        "predict_risk": predict_risk,
        "xcals": xcals,
    }

    buf_friday = io.StringIO()
    with contextlib.redirect_stdout(buf_friday):
        exec(compile(cell_code, "cell13_friday.py", "exec"), scope_friday)  # noqa: S102

    output_friday = buf_friday.getvalue()
    assert "Downside Risk Probability (20-day): 40.0%" in output_friday
    assert "Risk Classification Level:          High" in output_friday
    assert "Active Portfolio Policy Stance:     Defensive" in output_friday
    assert "Scheduled Execution Session:        2025-06-02 (at Market Close)" in output_friday
    assert "Total Gross Trade Volume:           $3,000.00" in output_friday
    assert "Estimated Rebalance Cost (10 bps):  $3.00" in output_friday





