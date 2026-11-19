"""Substantive error, subgroup, and regime analysis across causal market partitions."""

from typing import Any

import numpy as np

from ai_risk_radar.contracts import FeatureSet, ModelBundle, SplitSet
from ai_risk_radar.models import classification_report, predict_risk


def compute_regime_analysis(
    bundle: ModelBundle,
    features: FeatureSet,
    splits: SplitSet,
) -> dict[str, Any]:
    """Compute deep error and regime breakdown using strictly causal features."""
    df = features.frame
    target_col = features.target_columns[0]

    # 1. Honest partition metrics (Train, Calibration, Validation)
    partition_metrics: dict[str, Any] = {}
    for part_name, sessions in [
        ("train", splits.train),
        ("calibration", splits.calibration),
        ("validation", splits.validation),
    ]:
        part_frame = df.loc[sessions]
        y_true = part_frame[target_col]
        y_prob = predict_risk(bundle, part_frame)
        report = classification_report(y_true, y_prob)
        partition_metrics[part_name] = {
            "sessions": len(sessions),
            "prevalence": float(y_true.mean()),
            "brier_score": report.get("brier_score"),
            "mean_predicted_probability": float(y_prob.mean()),
        }

    # 2. Validation partition deep error and regime analysis
    val_df = df.loc[splits.validation].copy()
    y_val = val_df[target_col]
    val_prob = predict_risk(bundle, val_df)
    val_df["pred_risk"] = val_prob
    val_df["error_sq"] = (y_val - val_prob) ** 2

    # Annual breakdown
    annual_breakdown: dict[int, Any] = {}
    for year, group in val_df.groupby(val_df.index.year):
        y_yr = group[target_col]
        p_yr = group["pred_risk"]
        annual_breakdown[int(year)] = {
            "session_count": len(group),
            "event_count": int(y_yr.sum()),
            "prevalence": float(y_yr.mean()),
            "brier_score": float(np.mean((y_yr - p_yr) ** 2)),
            "mean_pred_prob": float(p_yr.mean()),
        }

    # Causal Regime: S&P 500 Trend (Bull: ret_spy_63 >= 0 vs Bear: ret_spy_63 < 0)
    # Uses only past 63 sessions return available at session t close
    trend_regimes: dict[str, Any] = {}
    if "ret_spy_63" in val_df.columns:
        bull_mask = val_df["ret_spy_63"] >= 0.0
        bear_mask = ~bull_mask
        for regime_name, mask in [("bull_market", bull_mask), ("bear_market", bear_mask)]:
            reg_df = val_df[mask]
            if len(reg_df) > 0:
                y_reg = reg_df[target_col]
                p_reg = reg_df["pred_risk"]
                trend_regimes[regime_name] = {
                    "session_count": len(reg_df),
                    "event_count": int(y_reg.sum()),
                    "prevalence": float(y_reg.mean()),
                    "brier_score": float(np.mean((y_reg - p_reg) ** 2)),
                    "mean_pred_prob": float(p_reg.mean()),
                }

    # Causal Regime: Volatility Regime (Low Vol vs High Vol based on vol_spy_20 median from training)
    vol_regimes: dict[str, Any] = {}
    if "vol_spy_20" in val_df.columns:
        median_vol = float(df.loc[splits.train, "vol_spy_20"].median())
        low_vol_mask = val_df["vol_spy_20"] <= median_vol
        high_vol_mask = ~low_vol_mask
        for regime_name, mask in [("low_volatility", low_vol_mask), ("high_volatility", high_vol_mask)]:
            reg_df = val_df[mask]
            if len(reg_df) > 0:
                y_reg = reg_df[target_col]
                p_reg = reg_df["pred_risk"]
                vol_regimes[regime_name] = {
                    "session_count": len(reg_df),
                    "event_count": int(y_reg.sum()),
                    "prevalence": float(y_reg.mean()),
                    "brier_score": float(np.mean((y_reg - p_reg) ** 2)),
                    "mean_pred_prob": float(p_reg.mean()),
                }

    return {
        "partitions": partition_metrics,
        "annual_breakdown": annual_breakdown,
        "trend_regimes": trend_regimes,
        "volatility_regimes": vol_regimes,
    }
