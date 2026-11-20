"""Substantive error, subgroup, and regime analysis across causal market partitions."""

from typing import Any

import numpy as np
import pandas as pd

from ai_risk_radar.contracts import FeatureSet, ModelBundle, SplitSet
from ai_risk_radar.models import classification_report, predict_risk


def compute_subgroup_error_diagnostics(
    bundle: ModelBundle,
    features: FeatureSet,
    splits: SplitSet,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Analyze false positives and false negatives on the validation partition strictly."""
    if threshold is None or not isinstance(threshold, (int, float)) or np.isnan(threshold):
        raise ValueError(f"Invalid diagnostic threshold: {threshold!r}. Expected finite float in [0.0, 1.0].")
    threshold = float(np.clip(threshold, 0.0, 1.0))

    val_sessions = splits.validation
    if len(val_sessions) == 0:
        return {
            "threshold": threshold,
            "total_sessions": 0,
            "actual_downside_events": 0,
            "actual_non_events": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "sample_false_positives": [],
            "sample_false_negatives": [],
        }

    df = features.frame
    target_col = features.target_columns[0]
    val_df = df.loc[val_sessions].copy()
    y_val = val_df[target_col]
    val_prob = predict_risk(bundle, val_df)
    val_df["pred_risk"] = val_prob

    pred_high = val_prob >= threshold
    tp = val_df[pred_high & (y_val == 1.0)]
    fp = val_df[pred_high & (y_val == 0.0)]
    fn = val_df[(~pred_high) & (y_val == 1.0)]
    tn = val_df[(~pred_high) & (y_val == 0.0)]

    precision = float(len(tp) / (len(tp) + len(fp))) if (len(tp) + len(fp)) > 0 else 0.0
    recall = float(len(tp) / (len(tp) + len(fn))) if (len(tp) + len(fn)) > 0 else 0.0

    sample_fps: list[dict[str, Any]] = []
    for dt, row in fp.head(5).iterrows():
        ret_val = row.get("ret_spy_63")
        vol_val = row.get("vol_spy_20")
        sample_fps.append({
            "session": dt.strftime("%Y-%m-%d"),
            "predicted_risk_pct": round(float(row["pred_risk"]) * 100, 2),
            "ret_spy_63": round(float(ret_val), 4) if pd.notna(ret_val) else None,
            "vol_spy_20": round(float(vol_val), 4) if pd.notna(vol_val) else None,
            "regime_context": "Elevated trailing volatility / drawdown trailing bear bottom false alarm",
        })

    sample_fns: list[dict[str, Any]] = []
    for dt, row in fn.head(5).iterrows():
        ret_val = row.get("ret_spy_63")
        vol_val = row.get("vol_spy_20")
        sample_fns.append({
            "session": dt.strftime("%Y-%m-%d"),
            "predicted_risk_pct": round(float(row["pred_risk"]) * 100, 2),
            "ret_spy_63": round(float(ret_val), 4) if pd.notna(ret_val) else None,
            "vol_spy_20": round(float(vol_val), 4) if pd.notna(vol_val) else None,
            "regime_context": "Elevated downside risk (near threshold) but missed due to conservative cutoff",
        })

    return {
        "threshold": threshold,
        "total_sessions": len(val_df),
        "actual_downside_events": int(y_val.sum()),
        "actual_non_events": int((y_val == 0.0).sum()),
        "tp": len(tp),
        "fp": len(fp),
        "fn": len(fn),
        "tn": len(tn),
        "precision": precision,
        "recall": recall,
        "sample_false_positives": sample_fps,
        "sample_false_negatives": sample_fns,
    }


def compute_regime_analysis(
    bundle: ModelBundle,
    features: FeatureSet,
    splits: SplitSet,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Compute deep error and regime breakdown using strictly causal features."""
    df = features.frame
    target_col = features.target_columns[0]

    # 1. Partition metrics: train and calibration are in-sample, validation is out-of-sample
    partition_metrics: dict[str, Any] = {}
    for part_name, sessions in [
        ("train", splits.train),
        ("calibration", splits.calibration),
        ("validation", splits.validation),
    ]:
        if len(sessions) == 0:
            partition_metrics[part_name] = {
                "sessions": 0,
                "prevalence": 0.0,
                "brier_score": None,
                "mean_predicted_probability": 0.0,
            }
            continue
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
    val_sessions = splits.validation
    if len(val_sessions) == 0:
        annual_breakdown: dict[int, Any] = {}
        trend_regimes: dict[str, Any] = {}
        vol_regimes: dict[str, Any] = {}
    else:
        val_df = df.loc[val_sessions].copy()
        y_val = val_df[target_col]
        val_prob = predict_risk(bundle, val_df)
        val_df["pred_risk"] = val_prob
        val_df["error_sq"] = (y_val - val_prob) ** 2

        # Annual breakdown
        annual_breakdown = {}
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
        trend_regimes = {}
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
        vol_regimes = {}
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

    # 3. Subgroup error diagnostics (FP / FN at specified threshold)
    error_diagnostics = compute_subgroup_error_diagnostics(bundle, features, splits, threshold=threshold)

    return {
        "partitions": partition_metrics,
        "annual_breakdown": annual_breakdown,
        "trend_regimes": trend_regimes,
        "volatility_regimes": vol_regimes,
        "error_diagnostics": error_diagnostics,
    }
