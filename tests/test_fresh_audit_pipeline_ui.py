"""Isolated regressions for provenance, fallback rendering, and frozen seed identity."""

import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from streamlit.testing.v1 import AppTest

from ai_risk_radar.config import load_settings
from ai_risk_radar.contracts import BacktestResult, DataBundle, ModelBundle
from ai_risk_radar.data import save_snapshot
from ai_risk_radar.pipeline import (
    _compute_policy_hash,
    _compute_release_hash,
    _compute_sha256,
    run_final_test,
    run_latest,
)
from ai_risk_radar.reporting import build_report, save_report

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def authored_release(tmp_path):
    """Build a tiny frozen estimator and authored prices, entirely under tmp_path."""
    settings = replace(
        load_settings(REPO_ROOT / "config.json"),
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        artifacts_dir=tmp_path / "artifacts",
        random_seed=7,
    )
    sessions = pd.bdate_range("2016-01-04", "2026-11-16")
    rng = np.random.default_rng(132)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, .01, (len(sessions), 4)), axis=0)),
        index=sessions,
        columns=settings.tickers,
    )
    bundle = DataBundle(
        adjusted_close=prices,
        volume=pd.DataFrame(10000, index=sessions, columns=settings.tickers),
        sessions=sessions,
        manifest={"source": "authored_regression_fixture", "cutoff_session": "2026-11-16"},
    )
    snapshot_dir = settings.raw_dir / "snapshot"
    save_snapshot(bundle, snapshot_dir)
    # No project fitting pipeline runs: four toy rows establish an authenticated seed.
    estimator = RandomForestClassifier(n_estimators=2, max_depth=1, random_state=7)
    estimator.fit(pd.DataFrame({"ret_soxx_1": [-.02, -.01, .01, .02]}), [1, 1, 0, 0])
    model_path = settings.artifacts_dir / "models" / "random_forest.joblib"
    model_path.parent.mkdir(parents=True)
    joblib.dump(ModelBundle(estimator, None, ("ret_soxx_1",), {
        "probability_status": "unavailable", "reason": "Authored fixture"
    }), model_path)
    model_hash = _compute_sha256(model_path)
    data_hash = _compute_sha256(snapshot_dir / "manifest.json")
    policy_hash = _compute_policy_hash(None, settings)
    release = {
        "release_version": "1.0.0",
        "selected_threshold": None,
        "target_horizon": 20,
        "target_threshold": -.05,
        "boundaries": {"cutoff": "2026-11-16"},
        "hashes": {
            "model": model_hash, "data": data_hash, "policy": policy_hash,
            "release": _compute_release_hash(model_hash, data_hash, policy_hash),
        },
    }
    release_path = settings.artifacts_dir / "release.json"
    release_path.write_text(json.dumps(release), encoding="utf-8")
    return settings, release_path, bundle


@pytest.mark.parametrize("manifest", [
    {"source": "deterministic_offline_fixture", "is_synthetic": True},
    {"source": "deterministic_offline_fixture"},
    {"source": "yfinance", "is_synthetic": True},
])
def test_latest_rejects_synthetic_cache_before_prediction(authored_release, monkeypatch, manifest):
    settings, release_path, bundle = authored_release
    save_snapshot(replace(bundle, manifest=manifest), settings.raw_dir / "snapshot")
    report_path = settings.artifacts_dir / "latest_report.json"
    report_path.write_text('{"previous": "valid report"}', encoding="utf-8")
    before = report_path.read_bytes()

    def forbidden_prediction(*args, **kwargs):
        pytest.fail("Synthetic input reached inference")

    monkeypatch.setattr("ai_risk_radar.pipeline.predict_risk", forbidden_prediction)
    with pytest.raises(ValueError, match="[Ss]ynthetic"):
        run_latest(release_path, settings=settings)
    assert report_path.read_bytes() == before


def test_latest_report_retains_cached_input_source(authored_release):
    settings, release_path, _ = authored_release
    report = json.loads(run_latest(release_path, settings=settings).read_text(encoding="utf-8"))
    assert report["data_status"]["source"] == "authored_regression_fixture"
    assert report["data_status"]["refresh_status"] == "cached"


def test_identical_nondefault_seed_reproduces_without_mutating_consumption(authored_release):
    settings, release_path, _ = authored_release
    final_path = run_final_test(release_path, settings=settings)
    release_before = release_path.read_bytes()
    final_before = final_path.read_bytes()
    reproduction = json.loads(run_final_test(release_path, settings=settings).read_text(encoding="utf-8"))
    assert reproduction["reproduction_status"] == "verified_match"
    assert release_path.read_bytes() == release_before
    assert final_path.read_bytes() == final_before


def test_seed_tampering_is_rejected_before_simulation(authored_release, monkeypatch):
    settings, release_path, _ = authored_release
    run_final_test(release_path, settings=settings)
    # An editable manifest seed must not override the seed in the hash-verified model.
    release = json.loads(release_path.read_text(encoding="utf-8"))
    release["random_seed"] = 42
    release_path.write_text(json.dumps(release), encoding="utf-8")

    def forbidden_simulation(*args, **kwargs):
        pytest.fail("Altered seed reached holdout simulation")

    monkeypatch.setattr("ai_risk_radar.pipeline.simulate", forbidden_simulation)
    with pytest.raises(ValueError, match="random_seed"):
        run_final_test(release_path, settings=replace(settings, random_seed=42))


@pytest.mark.parametrize("synthetic", [False, True])
def test_fallback_report_renders_compact_and_detailed(tmp_path, monkeypatch, synthetic):
    monkeypatch.chdir(tmp_path)
    forecast = pd.Series([.4], index=[pd.Timestamp("2026-11-16")])
    result = BacktestResult(pd.DataFrame(), pd.DataFrame(), {
        "total_return": .1, "cagr": .05, "max_drawdown": -.02,
        "sharpe": 1., "turnover": .5, "defensive_occupancy": 1.,
    })
    report = build_report({"selected": result, "fixed_defensive_weekly": result}, forecast, {
        "selected_threshold": None, "probability_status": "unavailable",
        "fallback_reason": "Insufficient calibration support", "model_hash": "authored-model",
        "release_hash": "authored-release", "is_synthetic": synthetic,
        "source": "deterministic_offline_fixture" if synthetic else "authored_regression_fixture",
    })
    save_report(report, tmp_path / "artifacts" / "latest_report.json")
    app = AppTest.from_file(str(REPO_ROOT / "app.py")).run()
    assert not app.exception
    if synthetic:
        assert any("synthetic" in warning.value.lower() for warning in app.warning)
    app.radio[0].set_value("Detailed").run()
    assert not app.exception
    assert app.dataframe[0].value.iloc[0]["Strategy"] == "Selected Strategy (Fixed Baseline Fallback)"
