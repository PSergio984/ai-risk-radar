"""Cold bootstrap script for AI Investment Risk Radar.

Initializes deterministic offline synthetic fixture data, fits baseline models,
and generates a standalone fixture demonstration report.
Allows clean-checkout container startup and offline testing without requiring
proprietary external market downloads or pre-existing binary weights.
"""

import sys
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_risk_radar.config import load_settings
from ai_risk_radar.data import (
    create_offline_synthetic_fixture,
    load_snapshot,
    save_snapshot,
)
from ai_risk_radar.features import build_features
from ai_risk_radar.models import fit_models, predict_risk
from ai_risk_radar.reporting import build_report, save_report
from ai_risk_radar.simulation import (
    benchmark_targets,
    select_strategy,
    simulate,
    weekly_targets,
)
from ai_risk_radar.splits import make_splits


def bootstrap_fixture(repo_root: Path = REPO_ROOT) -> Path:
    """Generate deterministic offline fixture data and demonstration report."""
    print("=" * 72)
    print("AI INVESTMENT RISK RADAR: COLD BOOTSTRAP FIXTURE INITIALIZER")
    print("=" * 72)

    config_path = repo_root / "config.json"
    settings = load_settings(config_path)

    fixture_dir = repo_root / settings.raw_dir / "notebook_fixture"
    if not fixture_dir.is_dir() or not (fixture_dir / "manifest.json").is_file():
        print(f"[INFO] Generating synthetic fixture data at: {fixture_dir}")
        bundle = create_offline_synthetic_fixture(settings, random_seed=settings.random_seed)
        save_snapshot(bundle, fixture_dir)
    else:
        print(f"[INFO] Loading existing synthetic fixture from: {fixture_dir}")
        bundle = load_snapshot(fixture_dir)

    print(f"[INFO] Building causal features across {len(bundle.sessions)} trading sessions...")
    features = build_features(bundle, settings)
    splits = make_splits(features, settings)

    print("[INFO] Fitting models across Train and Calibration partitions...")
    models = fit_models(features, splits, settings)
    rf_bundle = models["random_forest"]

    df = features.frame
    val_prices = bundle.adjusted_close.loc[splits.validation]
    val_frame = df.loc[splits.validation]

    print("[INFO] Evaluating validation partition candidate strategies...")
    val_probs = predict_risk(rf_bundle, val_frame)

    cost_rate = getattr(settings, "cost_rate", 0.0010)
    candidates = {}
    for th in settings.candidate_thresholds:
        key = f"threshold_{float(th):.2f}"
        targets = weekly_targets(val_probs, val_prices, settings, threshold=th)
        candidates[key] = simulate(val_prices, targets, settings, cost_rate)

    benchmarks = benchmark_targets(bundle, settings, sessions=splits.validation)
    benchmark_results = {
        name: simulate(val_prices, targets, settings, cost_rate)
        for name, targets in benchmarks.items()
    }

    selected_key = select_strategy(candidates, benchmark_results["growth_buy_and_hold"])
    selected_th = float(selected_key.replace("threshold_", "")) if selected_key else None
    all_results = {**benchmark_results, **candidates}

    latest_session = df.index[-1]
    latest_frame = df.loc[[latest_session]]
    latest_p = predict_risk(rf_bundle, latest_frame)

    metadata = {
        "is_synthetic": True,
        "disclaimer": "Synthetic fixture for pipeline demonstration and offline testing. NOT real market data.",
        "source": "deterministic_offline_fixture",
        "tickers": list(settings.tickers),
        "selected_threshold": selected_th,
        "probability_status": rf_bundle.metadata.get("probability_status", "calibrated"),
        "fallback_reason": rf_bundle.metadata.get("reason"),
        "weights_growth": list(settings.weights_growth),
        "weights_defensive": list(settings.weights_defensive),
        "weights_fixed_baseline": list(settings.weights_fixed_baseline),
        "target_horizon": settings.target_horizon,
        "target_threshold": settings.target_threshold,
        "staleness_sessions": 0,
        "cutoff_session": latest_session.strftime("%Y-%m-%d"),
        "model_hash": "fixture_synthetic_demonstration_model",
        "release_hash": "fixture_synthetic_demonstration_release",
    }

    report = build_report(all_results, latest_p, metadata)
    artifacts_dir = repo_root / settings.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    fixture_report_path = artifacts_dir / "fixture_report.json"
    save_report(report, fixture_report_path)
    print(f"[SUCCESS] Cold bootstrap fixture report generated at: {fixture_report_path}")

    # If no latest_report.json exists, also provide it as latest_report.json
    latest_report_path = artifacts_dir / "latest_report.json"
    if not latest_report_path.is_file():
        save_report(report, latest_report_path)
        print(f"[INFO] Populated initial dashboard report at: {latest_report_path}")

    return fixture_report_path


if __name__ == "__main__":
    bootstrap_fixture()
