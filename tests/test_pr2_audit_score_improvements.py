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


