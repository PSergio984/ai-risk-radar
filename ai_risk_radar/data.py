"""ETF price data acquisition, validation, and hashed snapshots."""

import datetime
import hashlib
import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import yfinance as yf

from ai_risk_radar.contracts import DataBundle, Settings


def _compute_sha256(file_path: Path) -> str:
    """Compute hex SHA256 digest of a file."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def validate_sessions_and_prices(
    adjusted_close: pd.DataFrame,
    volume: pd.DataFrame,
    expected_sessions: pd.DatetimeIndex | None = None,
) -> None:
    """Validate that sessions are strictly ascending, unique, complete, and have positive prices."""
    if not adjusted_close.index.is_monotonic_increasing or adjusted_close.index.has_duplicates:
        raise ValueError("Sessions must be unique and sorted ascending")

    if not volume.index.equals(adjusted_close.index):
        raise ValueError("Volume index must match adjusted_close index exactly")

    # Check for nonpositive or NaN values
    for col in adjusted_close.columns:
        series = adjusted_close[col]
        nan_mask = series.isna()
        if nan_mask.any():
            first_nan_date = adjusted_close.index[nan_mask][0]
            raise ValueError(
                f"Missing exchange session for {col} on {first_nan_date.strftime('%Y-%m-%d')}"
            )
        inf_mask = ~np.isfinite(series)
        if inf_mask.any():
            first_inf_date = adjusted_close.index[inf_mask][0]
            raise ValueError(
                f"Non-finite price for {col} on {first_inf_date.strftime('%Y-%m-%d')}"
            )
        nonpos_mask = series <= 0.0
        if nonpos_mask.any():
            first_nonpos_date = adjusted_close.index[nonpos_mask][0]
            raise ValueError(
                f"Nonpositive price for {col} on {first_nonpos_date.strftime('%Y-%m-%d')}"
            )

    for col in volume.columns:
        vol_series = volume[col]
        if vol_series.isna().any():
            first_nan = volume.index[vol_series.isna()][0]
            raise ValueError(
                f"Missing exchange session volume for {col} on {first_nan.strftime('%Y-%m-%d')}"
            )
        nonfinite_vol = ~np.isfinite(vol_series)
        if nonfinite_vol.any():
            first_inf = volume.index[nonfinite_vol][0]
            raise ValueError(
                f"Non-finite volume for {col} on {first_inf.strftime('%Y-%m-%d')}"
            )
        if (vol_series < 0.0).any():
            first_neg = volume.index[vol_series < 0.0][0]
            raise ValueError(
                f"Negative volume for {col} on {first_neg.strftime('%Y-%m-%d')}"
            )

    if expected_sessions is not None:
        missing_sessions = expected_sessions.difference(adjusted_close.index)
        if len(missing_sessions) > 0:
            missing_date = missing_sessions[0]
            first_col = adjusted_close.columns[0]
            raise ValueError(
                f"Missing exchange session for {first_col} on {missing_date.strftime('%Y-%m-%d')}"
            )


def save_snapshot(bundle: DataBundle, path: Path | str) -> Path:
    """Persist a DataBundle to parquet files and generate a hashed manifest."""
    validate_sessions_and_prices(bundle.adjusted_close, bundle.volume)
    snapshot_dir = Path(path)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    adj_close_path = snapshot_dir / "adjusted_close.parquet"
    volume_path = snapshot_dir / "volume.parquet"

    bundle.adjusted_close.to_parquet(adj_close_path)
    bundle.volume.to_parquet(volume_path)

    adj_hash = _compute_sha256(adj_close_path)
    vol_hash = _compute_sha256(volume_path)

    manifest = dict(bundle.manifest)
    manifest["hashes"] = {
        "adjusted_close.parquet": adj_hash,
        "volume.parquet": vol_hash,
    }
    manifest["session_count"] = len(bundle.sessions)
    manifest["start_session"] = bundle.sessions[0].strftime("%Y-%m-%d")
    manifest["end_session"] = bundle.sessions[-1].strftime("%Y-%m-%d")

    manifest_path = snapshot_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest_path


def load_snapshot(path: Path | str) -> DataBundle:
    """Load a snapshot from disk and verify SHA256 integrity."""
    snapshot_dir = Path(path)
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found in snapshot directory: {snapshot_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = manifest.get("hashes", {})

    adj_close_path = snapshot_dir / "adjusted_close.parquet"
    volume_path = snapshot_dir / "volume.parquet"

    if not adj_close_path.is_file() or not volume_path.is_file():
        raise FileNotFoundError(f"Missing parquet data files in snapshot: {snapshot_dir}")

    # Verify hashes
    actual_adj_hash = _compute_sha256(adj_close_path)
    expected_adj_hash = hashes.get("adjusted_close.parquet")
    if actual_adj_hash != expected_adj_hash:
        raise ValueError(
            f"Hash mismatch for adjusted_close.parquet: expected {expected_adj_hash}, got {actual_adj_hash}"
        )

    actual_vol_hash = _compute_sha256(volume_path)
    expected_vol_hash = hashes.get("volume.parquet")
    if actual_vol_hash != expected_vol_hash:
        raise ValueError(
            f"Hash mismatch for volume.parquet: expected {expected_vol_hash}, got {actual_vol_hash}"
        )

    adjusted_close = pd.read_parquet(adj_close_path)
    volume = pd.read_parquet(volume_path)

    # Normalize indices to tz-naive UTC calendar dates
    adjusted_close.index = pd.to_datetime(adjusted_close.index).normalize()
    volume.index = pd.to_datetime(volume.index).normalize()

    validate_sessions_and_prices(adjusted_close, volume)
    sessions = pd.DatetimeIndex(adjusted_close.index)

    return DataBundle(
        adjusted_close=adjusted_close,
        volume=volume,
        sessions=sessions,
        manifest=manifest,
    )


def resolve_last_completed_session(
    cal: xcals.ExchangeCalendar,
    as_of: pd.Timestamp | str | datetime.date | None = None,
) -> pd.Timestamp:
    """Resolve last completed trading session on XNYS calendar for any date or timestamp."""
    if as_of is None:
        ts = pd.Timestamp.now(tz="UTC").tz_convert("America/New_York")
        is_date_only = False
    elif isinstance(as_of, str):
        if not any(c in as_of for c in ("T", " ", ":")):
            is_date_only = True
            target_date = pd.Timestamp(as_of)
        else:
            is_date_only = False
            ts = pd.Timestamp(as_of)
    elif type(as_of) is datetime.date:
        is_date_only = True
        target_date = pd.Timestamp(as_of)
    else:
        is_date_only = False
        ts = pd.Timestamp(as_of)

    if is_date_only:
        if cal.is_session(target_date):
            return target_date
        return pd.Timestamp(cal.date_to_session(target_date, direction="previous"))

    if ts.tz is None:
        ts = ts.tz_localize("America/New_York")
    else:
        ts = ts.tz_convert("America/New_York")

    session_date = pd.Timestamp(ts.date())
    if cal.is_session(session_date):
        market_close = cal.session_close(session_date).tz_convert("America/New_York")
        if ts >= market_close:
            return session_date
        return pd.Timestamp(cal.previous_session(session_date))
    return pd.Timestamp(cal.date_to_session(session_date, direction="previous"))


def _download_price_data(
    tickers_list: list[str],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    max_retries: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download adjusted close and volume data from yfinance with retry handling."""
    raw_df = pd.DataFrame()
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            raw_df = yf.download(
                tickers=tickers_list,
                start=start_dt.strftime("%Y-%m-%d"),
                end=(end_dt + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                auto_adjust=False,
                actions=True,
                progress=False,
                threads=False,
            )
            if not raw_df.empty:
                break
        except (OSError, RuntimeError, ConnectionError, ValueError, TimeoutError) as e:
            last_err = e

    if raw_df.empty:
        raise RuntimeError(f"yfinance returned an empty dataset after {max_retries} attempts: {last_err}")

    # Extract Adj Close and Volume
    if isinstance(raw_df.columns, pd.MultiIndex):
        adj_close = raw_df["Adj Close"][tickers_list].copy()
        volume = raw_df["Volume"][tickers_list].copy()
    else:
        adj_close = pd.DataFrame({tickers_list[0]: raw_df["Adj Close"]})
        volume = pd.DataFrame({tickers_list[0]: raw_df["Volume"]})

    # Normalize index to tz-naive
    adj_close.index = pd.to_datetime(adj_close.index).tz_localize(None).normalize()
    volume.index = pd.to_datetime(volume.index).tz_localize(None).normalize()
    return adj_close, volume


def fetch_prices(
    settings: Settings,
    as_of: pd.Timestamp | str | None = None,
    existing_bundle: DataBundle | None = None,
    overlap_sessions: int = 10,
) -> DataBundle:
    """Fetch ETF prices from yfinance and align with XNYS exchange calendar.

    If existing_bundle is provided, performs an incremental fetch starting
    overlap_sessions before the end of the existing bundle to capture any vendor revisions,
    merging and deduplicating the data.
    """
    cal = xcals.get_calendar("XNYS")
    last_completed = resolve_last_completed_session(cal, as_of)

    start_dt = pd.Timestamp(settings.acquisition_start)
    expected_sessions = cal.sessions_in_range(start_dt, last_completed)

    tickers_list = list(settings.tickers)
    fetch_start_dt = start_dt
    cached_adj = None
    cached_vol = None

    if existing_bundle is not None and len(existing_bundle.sessions) > 0:
        cached_adj = existing_bundle.adjusted_close.copy()
        cached_vol = existing_bundle.volume.copy()
        cached_adj.index = pd.to_datetime(cached_adj.index).tz_localize(None).normalize()
        cached_vol.index = pd.to_datetime(cached_vol.index).tz_localize(None).normalize()
        existing_sessions = pd.to_datetime(existing_bundle.sessions).tz_localize(None).normalize()
        existing_last = existing_sessions[-1]

        if existing_last > last_completed:
            adj = cached_adj.reindex(expected_sessions)
            vol = cached_vol.reindex(expected_sessions)
            validate_sessions_and_prices(adj, vol, expected_sessions=expected_sessions)
            manifest = dict(existing_bundle.manifest)
            manifest["cutoff_session"] = last_completed.strftime("%Y-%m-%d")
            manifest["full_history_refresh"] = False
            return DataBundle(adjusted_close=adj, volume=vol, sessions=expected_sessions, manifest=manifest)

        overlap_candidates = cal.sessions_in_range(start_dt, existing_last)
        if len(overlap_candidates) > overlap_sessions:
            fetch_start_dt = overlap_candidates[-overlap_sessions]
        else:
            fetch_start_dt = start_dt

    adj_close, volume = _download_price_data(tickers_list, fetch_start_dt, last_completed)

    overlap_coherent = True
    if cached_adj is not None:
        # Check overlap sessions between cached_adj and newly fetched data
        overlap_idx = cached_adj.index.intersection(adj_close.index)
        if len(overlap_idx) == 0:
            overlap_coherent = False
        else:
            # Check price revision coherence across all tickers
            cached_overlap_vals = cached_adj.loc[overlap_idx, tickers_list].to_numpy()
            new_overlap_vals = adj_close.loc[overlap_idx, tickers_list].to_numpy()
            if not np.allclose(cached_overlap_vals, new_overlap_vals, rtol=1e-5, atol=1e-5):
                overlap_coherent = False

    if not overlap_coherent and fetch_start_dt != start_dt:
        # Adjustment revision detected or non-overlapping gap: download full coherent history
        adj_close, volume = _download_price_data(tickers_list, start_dt, last_completed)
        full_history_refresh = True
    else:
        full_history_refresh = (existing_bundle is None)

    # Merge with cached data if doing incremental refresh:
    # Newly fetched data takes precedence on overlapping sessions for revisions
    if cached_adj is not None and not full_history_refresh:
        adj_close = pd.concat([cached_adj.loc[cached_adj.index < adj_close.index[0]], adj_close])
    if cached_vol is not None and not full_history_refresh:
        volume = pd.concat([cached_vol.loc[cached_vol.index < volume.index[0]], volume])

    # Reindex to exact expected exchange sessions
    adj_close = adj_close.reindex(expected_sessions)
    volume = volume.reindex(expected_sessions)

    # Reject if any ticker has missing session or nonpositive price
    validate_sessions_and_prices(adj_close, volume, expected_sessions=expected_sessions)

    manifest = {
        "source": settings.source,
        "tickers": list(settings.tickers),
        "acquisition_start": settings.acquisition_start,
        "cutoff_session": last_completed.strftime("%Y-%m-%d"),
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "adjustment_settings": {"auto_adjust": False, "price_column": "Adj Close"},
        "incremental": existing_bundle is not None and not full_history_refresh,
        "full_history_refresh": full_history_refresh,
    }

    return DataBundle(
        adjusted_close=adj_close,
        volume=volume,
        sessions=expected_sessions,
        manifest=manifest,
    )


def create_offline_synthetic_fixture(
    settings: Settings,
    start: str = "2016-01-04",
    end: str = "2026-11-16",
    random_seed: int = 42,
) -> DataBundle:
    """Generate a deterministic synthetic DataBundle for offline testing and pipeline demonstration.

    Explicitly labeled as synthetic in its manifest. Never silently treated as real market prices.
    Generates enough sessions (>= 4 years) to satisfy train/calibration/validation/test split constraints.
    """
    cal = xcals.get_calendar("XNYS")
    sessions = cal.sessions_in_range(start, end)
    n = len(sessions)
    if n < 1460:
        raise ValueError(f"Synthetic fixture requires at least 1460 trading sessions, got {n}")

    rng = np.random.RandomState(random_seed)
    tickers = list(settings.tickers)
    drift = {
        "SOXX": 0.012,
        "XLK": 0.010,
        "SPY": 0.006,
        "SHY": 0.001,
    }

    data = {}
    for t in tickers:
        d = drift.get(t, 0.005)
        ret = rng.randn(n) * d
        data[t] = 100.0 * np.exp(np.cumsum(ret))

    adj_df = pd.DataFrame(data, index=sessions)
    vol_df = pd.DataFrame(10000.0, index=sessions, columns=tickers)

    manifest = {
        "source": "deterministic_offline_fixture",
        "tickers": tickers,
        "acquisition_start": start,
        "cutoff_session": sessions[-1].strftime("%Y-%m-%d"),
        "session_count": n,
        "is_synthetic": True,
        "disclaimer": "Synthetic fixture for pipeline demonstration and offline testing. NOT real market data.",
    }

    validate_sessions_and_prices(adj_df, vol_df, expected_sessions=sessions)
    return DataBundle(
        adjusted_close=adj_df,
        volume=vol_df,
        sessions=sessions,
        manifest=manifest,
    )
