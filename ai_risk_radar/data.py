"""ETF price data acquisition, validation, and hashed snapshots."""

import hashlib
import json
from pathlib import Path

import exchange_calendars as xcals
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
        nonpos_mask = series <= 0.0
        if nonpos_mask.any():
            first_nonpos_date = adjusted_close.index[nonpos_mask][0]
            raise ValueError(
                f"Nonpositive price for {col} on {first_nonpos_date.strftime('%Y-%m-%d')}"
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
    as_of: pd.Timestamp | str | None = None,
) -> pd.Timestamp:
    """Resolve last completed trading session on XNYS calendar for any date or timestamp."""
    if as_of is None:
        now_ny = pd.Timestamp.now(tz="UTC").tz_convert("America/New_York")
        today_date = pd.Timestamp(now_ny.date())
        if cal.is_session(today_date):
            if now_ny.hour >= 16:
                return today_date
            return pd.Timestamp(cal.previous_session(today_date))
        return pd.Timestamp(cal.date_to_session(today_date, direction="previous"))

    as_of_ts = pd.Timestamp(as_of)
    if as_of_ts.tz is not None:
        as_of_ts = as_of_ts.tz_convert("America/New_York").tz_localize(None)
    as_of_date = pd.Timestamp(as_of_ts.date())
    if cal.is_session(as_of_date):
        return as_of_date
    return pd.Timestamp(cal.date_to_session(as_of_date, direction="previous"))


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
        existing_last = existing_bundle.sessions[-1]
        if existing_last >= last_completed:
            adj = existing_bundle.adjusted_close.reindex(expected_sessions)
            vol = existing_bundle.volume.reindex(expected_sessions)
            validate_sessions_and_prices(adj, vol, expected_sessions=expected_sessions)
            manifest = dict(existing_bundle.manifest)
            manifest["cutoff_session"] = last_completed.strftime("%Y-%m-%d")
            return DataBundle(adjusted_close=adj, volume=vol, sessions=expected_sessions, manifest=manifest)

        overlap_candidates = cal.sessions_in_range(start_dt, existing_last)
        if len(overlap_candidates) > overlap_sessions:
            fetch_start_dt = overlap_candidates[-overlap_sessions]
        else:
            fetch_start_dt = start_dt

        cached_adj = existing_bundle.adjusted_close
        cached_vol = existing_bundle.volume

    max_retries = 3
    raw_df = pd.DataFrame()
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            raw_df = yf.download(
                tickers=tickers_list,
                start=fetch_start_dt.strftime("%Y-%m-%d"),
                end=(last_completed + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
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

    # Merge with cached data if doing incremental refresh:
    # Newly fetched data takes precedence on overlapping sessions for revisions
    if cached_adj is not None:
        adj_close = pd.concat([cached_adj.loc[cached_adj.index < adj_close.index[0]], adj_close])
    if cached_vol is not None:
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
        "incremental": existing_bundle is not None,
    }

    return DataBundle(
        adjusted_close=adj_close,
        volume=volume,
        sessions=expected_sessions,
        manifest=manifest,
    )
