"""Regression checks for coherent adjustments, completed sessions, and one-class training."""

from dataclasses import replace
from datetime import date
from pathlib import Path
from unittest.mock import patch

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ai_risk_radar.config import load_settings
from ai_risk_radar.contracts import DataBundle, FeatureSet, ModelBundle, SplitSet
from ai_risk_radar.data import fetch_prices, resolve_last_completed_session
from ai_risk_radar.models import fit_models, predict_risk
from ai_risk_radar.reporting import build_report


@pytest.fixture
def acquisition_case(tmp_path):
    settings = replace(
        load_settings(Path(__file__).parents[1] / "config.json"),
        acquisition_start="2025-01-02",
        raw_dir=tmp_path / "raw",
        artifacts_dir=tmp_path / "artifacts",
    )
    sessions = xcals.get_calendar("XNYS").sessions_in_range("2025-01-02", "2025-02-14")[:25]
    prices = pd.DataFrame(100.0, index=sessions[:20], columns=settings.tickers)
    volume = pd.DataFrame(1000.0, index=sessions[:20], columns=settings.tickers)
    cached = DataBundle(prices, volume, sessions[:20], {"source": "mock_vendor"})
    return settings, sessions, cached


def vendor_response(sessions, tickers, price):
    return pd.concat(
        {
            "Adj Close": pd.DataFrame(price, index=sessions, columns=tickers),
            "Volume": pd.DataFrame(1000.0, index=sessions, columns=tickers),
        },
        axis=1,
    )


def test_adjustment_revision_refreshes_complete_history_without_phantom_return(acquisition_case):
    settings, sessions, cached = acquisition_case
    overlap = vendor_response(sessions[10:], settings.tickers, 99.0)
    coherent = vendor_response(sessions, settings.tickers, 99.0)
    with patch("ai_risk_radar.data.yf.download", side_effect=[overlap, coherent]) as download:
        result = fetch_prices(settings, as_of=str(sessions[-1].date()), existing_bundle=cached)

    np.testing.assert_allclose(result.adjusted_close.to_numpy(), 99.0)
    np.testing.assert_allclose(result.adjusted_close.pct_change().iloc[1:], 0.0)
    assert download.call_args_list[1].kwargs["start"] == "2025-01-02"
    assert result.manifest["full_history_refresh"] is True


def test_unchanged_adjustment_overlap_keeps_incremental_acquisition(acquisition_case):
    settings, sessions, cached = acquisition_case
    response = vendor_response(sessions[10:], settings.tickers, 100.0)
    response.loc[sessions[20:], "Adj Close"] = 102.0
    with patch("ai_risk_radar.data.yf.download", return_value=response) as download:
        result = fetch_prices(settings, as_of=str(sessions[-1].date()), existing_bundle=cached)

    assert download.call_count == 1
    assert result.adjusted_close.iloc[0, 0] == 100.0
    assert result.adjusted_close.iloc[-1, 0] == 102.0
    assert result.manifest["full_history_refresh"] is False


@pytest.mark.parametrize("failure", ["empty", "incomplete"])
def test_adjustment_refresh_failure_does_not_reuse_inconsistent_cache(acquisition_case, failure):
    settings, sessions, cached = acquisition_case
    overlap = vendor_response(sessions[10:], settings.tickers, 99.0)
    fallback = (
        pd.DataFrame()
        if failure == "empty"
        else vendor_response(sessions[1:], settings.tickers, 99.0)
    )
    with (
        patch("ai_risk_radar.data.yf.download", side_effect=[overlap, fallback, fallback, fallback]),
        pytest.raises((ValueError, RuntimeError)),
    ):
        fetch_prices(settings, as_of=str(sessions[-1].date()), existing_bundle=cached)
    np.testing.assert_allclose(cached.adjusted_close.to_numpy(), 100.0)


def test_nonoverlapping_download_requires_coherent_full_history(acquisition_case):
    settings, sessions, cached = acquisition_case
    tail = vendor_response(sessions[20:], settings.tickers, 99.0)
    coherent = vendor_response(sessions, settings.tickers, 99.0)
    with patch("ai_risk_radar.data.yf.download", side_effect=[tail, coherent]):
        result = fetch_prices(settings, as_of=str(sessions[-1].date()), existing_bundle=cached)
    np.testing.assert_allclose(result.adjusted_close.to_numpy(), 99.0)


@pytest.mark.parametrize(
    ("as_of", "expected"),
    [
        ("2025-01-06T14:00:00Z", "2025-01-03"),
        ("2025-01-06 15:59:00", "2025-01-03"),
        ("2025-01-06T21:00:00Z", "2025-01-06"),
        ("2025-01-07T01:00:00Z", "2025-01-06"),
        (pd.Timestamp("2025-01-06"), "2025-01-03"),
        ("2025-11-28T17:59:00Z", "2025-11-26"),
        ("2025-11-28T18:00:00Z", "2025-11-28"),
        ("2025-01-06", "2025-01-06"),
        (date(2025, 1, 6), "2025-01-06"),
        ("2025-01-05", "2025-01-03"),
    ],
)
def test_timestamp_cutoffs_use_actual_close_while_dates_select_inclusive_session(as_of, expected):
    assert resolve_last_completed_session(xcals.get_calendar("XNYS"), as_of) == pd.Timestamp(expected)


@pytest.mark.parametrize("label", [0.0, 1.0])
def test_single_class_risk_prediction_maps_the_actual_class(label):
    frame = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]})
    estimator = RandomForestClassifier(n_estimators=1, random_state=42).fit(frame, [label] * 4)
    model = ModelBundle(estimator, None, ("f",), {"probability_status": "unavailable"})
    np.testing.assert_allclose(predict_risk(model, frame), label)


@pytest.mark.parametrize("label", [0.0, 1.0])
def test_single_class_training_keeps_calibration_unavailable_and_report_on_baseline(label):
    settings = load_settings(Path(__file__).parents[1] / "config.json")
    sessions = pd.bdate_range("2020-01-02", periods=200)
    frame = pd.DataFrame(
        {"f": np.arange(200, dtype=float), "downside20": label}, index=sessions
    )
    # Calibration has support in both classes; it cannot rescue a one-class trained estimator.
    frame.loc[sessions[120:180], "downside20"] = [0.0, 1.0] * 30
    features = FeatureSet(frame, ("f",), ("downside20",))
    splits = SplitSet(sessions[:120], sessions[120:180], sessions[180:190], sessions[190:], {})
    models = fit_models(features, splits, settings)
    forest = models["random_forest"]
    assert forest.metadata["probability_status"] == "unavailable"
    assert forest.calibrator is None
    np.testing.assert_allclose(predict_risk(forest, frame.iloc[-2:]), label)
    np.testing.assert_allclose(predict_risk(models["decision_tree"], frame.iloc[-2:]), label)
    report = build_report({}, predict_risk(forest, frame.iloc[-2:]), forest.metadata)
    assert report.payload["forecast"]["probability"] is None
    assert report.payload["allocation"]["weights"] == {
        "SOXX": 0.20, "XLK": 0.20, "SPY": 0.20, "SHY": 0.40
    }
