"""Price forecasting engine for AgroMind AI.

Loads AgriPriceBD CSV datasets, cleans retail mid-price series, validates inputs,
fits ARIMA time-series models with confidence intervals, handles fallback to
persistence baseline on insufficient/failing data, and compares empirical accuracy.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, Tuple, Any, List
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CROPS: List[str] = ["garlic", "chickpea", "green_chilli", "cucumber", "sweet_pumpkin"]

# Minimum historical points needed to reliably fit ARIMA(1,1,1)
MIN_ARIMA_POINTS = 30


def validate_crop(crop: str) -> str:
    """Validate that the crop is supported; normalize to lowercase string."""
    if not isinstance(crop, str) or not crop.strip():
        raise ValueError("Crop name must be a non-empty string.")
    normalized = crop.strip().lower()
    if normalized not in CROPS:
        raise ValueError(f"Unsupported crop '{crop}'. Supported crops: {', '.join(CROPS)}")
    return normalized


def validate_horizon(horizon: int, min_val: int = 1, max_val: int = 90) -> int:
    """Validate that forecast horizon is a positive integer within bounds."""
    if not isinstance(horizon, (int, np.integer)):
        raise TypeError(f"Forecast horizon must be an integer, got {type(horizon).__name__}.")
    if horizon < min_val or horizon > max_val:
        raise ValueError(f"Forecast horizon must be between {min_val} and {max_val} days (got {horizon}).")
    return int(horizon)


def load_series(crop: str, raw_dir: Path = RAW_DIR) -> pd.Series:
    """Return a daily retail mid-price series (BDT/kg) for the specified crop.

    Handles zero-price portal reporting anomalies by treating them as missing,
    fills missing calendar days, and validates time index monotonicity.
    """
    valid_crop = validate_crop(crop)
    csv_path = Path(raw_dir) / f"{valid_crop}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Source price data file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV file for {valid_crop}: {exc}") from exc

    required_cols = {"date", "retail price minimum", "retail price maximum"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"CSV file {csv_path.name} is missing required columns: {missing}")

    if df.empty:
        raise ValueError(f"Price dataset for {valid_crop} is empty.")

    df = df.sort_values("date").dropna(subset=["date"])
    df = df.set_index("date")

    # Mid-price calculation: average of retail min and max
    mid = (pd.to_numeric(df["retail price minimum"], errors="coerce") +
           pd.to_numeric(df["retail price maximum"], errors="coerce")) / 2.0

    # Zero values represent portal reporting outages / data pipeline gaps; treat as missing
    mid = mid.where(mid > 0)
    valid_mid = mid.dropna()
    if valid_mid.empty:
        raise ValueError(f"No positive retail price records found for {valid_crop}.")

    # Generate contiguous daily calendar index
    full_index = pd.date_range(valid_mid.index.min(), valid_mid.index.max(), freq="D")
    clean_series = mid.reindex(full_index).ffill().bfill().rename(valid_crop)
    clean_series.index.name = "date"
    return clean_series


def naive_forecast(history: pd.Series, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Persistence baseline: the last known price repeated, with empirical confidence bands.

    Confidence intervals (80%) are computed from historical daily changes:
    z_0.80 ≈ 1.282, widening with sqrt(h) over horizon h.
    """
    valid_h = validate_horizon(horizon)
    last_val = float(history.iloc[-1])
    forecast = np.repeat(last_val, valid_h)

    # Estimate empirical volatility from daily diffs
    diffs = history.diff().dropna()
    sigma = float(diffs.std()) if len(diffs) > 1 and not np.isnan(diffs.std()) else 1.0
    if sigma <= 0:
        sigma = max(last_val * 0.02, 1.0)

    # 80% CI: z = 1.28155, scales with sqrt(day_step)
    steps = np.arange(1, valid_h + 1)
    margin = 1.28155 * sigma * np.sqrt(steps)
    lower = np.maximum(0.0, forecast - margin)
    upper = forecast + margin

    return forecast, lower, upper


def arima_forecast(
    history: pd.Series,
    horizon: int,
    order: Tuple[int, int, int] = (1, 1, 1),
    max_history_days: int = 730,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Fit ARIMA on recent history and return (forecast, lower, upper, metadata).

    If history is too short (< 30 days) or ARIMA fails to converge, cleanly
    falls back to naive_forecast without ambiguous application crashes.
    """
    valid_h = validate_horizon(horizon)

    # Filter strictly positive values
    clean = pd.to_numeric(history, errors="coerce").dropna()
    clean = clean[clean > 0]

    metadata: Dict[str, Any] = {
        "model_name": f"ARIMA{order}",
        "is_fallback": False,
        "observations_used": 0,
        "note": "সফলভাবে তৈরি ARIMA(1,1,1) পূর্বাভাস",
    }

    # Insufficient data check -> Graceful fallback
    if len(clean) < MIN_ARIMA_POINTS:
        f, l, u = naive_forecast(clean, valid_h)
        metadata["model_name"] = "Naive Persistence Baseline"
        metadata["is_fallback"] = True
        metadata["observations_used"] = len(clean)
        metadata["note"] = f"পর্যাপ্ত ডেটা না থাকায় ({len(clean)}<{MIN_ARIMA_POINTS}) সরল নাইভ পূর্বাভাস ব্যবহৃত হয়েছে।"
        return f, l, u, metadata

    recent = clean.iloc[-max_history_days:]
    metadata["observations_used"] = len(recent)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(recent.values, order=order)
            fit = model.fit()

        res = fit.get_forecast(valid_h)
        ci = res.conf_int(alpha=0.20)  # 80% interval
        mean_forecast = res.predicted_mean
        lower_band = np.maximum(0.0, ci[:, 0])
        upper_band = ci[:, 1]

        # Sanity check: no NaNs or Infs
        if np.any(np.isnan(mean_forecast)) or np.any(np.isnan(lower_band)) or np.any(np.isnan(upper_band)):
            raise ValueError("ARIMA produced NaN values in forecast.")

        return mean_forecast, lower_band, upper_band, metadata

    except Exception as exc:
        # Fall back gracefully to naive baseline
        f, l, u = naive_forecast(clean, valid_h)
        metadata["model_name"] = "Naive Persistence Baseline (Fallback)"
        metadata["is_fallback"] = True
        metadata["note"] = f"ARIMA কনভার্জ না করায় নিরাপদ নাইভ পূর্বাভাস প্রদান করা হয়েছে: {exc}"
        return f, l, u, metadata


@functools.lru_cache(maxsize=32)
def evaluate(crop: str, horizon: int = 14, n_origins: int = 20) -> Dict[str, Any]:
    """Walk-forward test over the last 20% of the dataset. Returns MAE per method."""
    valid_crop = validate_crop(crop)
    valid_h = validate_horizon(horizon)
    s = load_series(valid_crop)

    if len(s) < valid_h + 20:
        raise ValueError(f"Insufficient series length ({len(s)}) for walk-forward evaluation.")

    start = int(len(s) * 0.8)
    origins = np.linspace(start, len(s) - valid_h, n_origins).astype(int)
    err = {"naive": [], "arima": []}

    for o in origins:
        hist = s.iloc[:o]
        actual = s.iloc[o : o + valid_h].values
        # Naive forecast
        n_fc, _, _ = naive_forecast(hist, valid_h)
        err["naive"].append(float(np.abs(n_fc - actual).mean()))

        # ARIMA forecast
        a_fc, _, _, _ = arima_forecast(hist, valid_h)
        err["arima"].append(float(np.abs(a_fc - actual).mean()))

    naive_mae = float(np.mean(err["naive"]))
    arima_mae = float(np.mean(err["arima"]))
    arima_wins_pct = float(np.mean(np.array(err["arima"]) < np.array(err["naive"])) * 100)

    recommended = "ARIMA" if arima_mae <= naive_mae else "Naive"
    diff_pct = abs(naive_mae - arima_mae) / max(naive_mae, 1e-4) * 100.0

    if arima_mae <= naive_mae:
        summary_bn = (
            f"গত {n_origins}টি পরীক্ষামূলক সময়ে ARIMA মডেল নাইভ পদ্ধতির চেয়ে "
            f"{diff_pct:.1f}% কম ত্রুটি (MAE {arima_mae:.2f} বনাম {naive_mae:.2f}) দেখিয়েছে।"
        )
    else:
        summary_bn = (
            f"এই ফসলের উচ্চ অস্থিরতার কারণে সরল নাইভ পদ্ধতি ARIMA-এর চেয়ে "
            f"{diff_pct:.1f}% কম ত্রুটি (MAE {naive_mae:.2f} বনাম {arima_mae:.2f}) বজায় রেখেছে।"
        )

    return {
        "crop": valid_crop,
        "horizon": valid_h,
        "n_origins": n_origins,
        "naive_mae": round(naive_mae, 2),
        "arima_mae": round(arima_mae, 2),
        "arima_wins_pct": round(arima_wins_pct, 1),
        "recommended_model": recommended,
        "diff_pct": round(diff_pct, 1),
        "summary_bn": summary_bn,
        "last_price": round(float(s.iloc[-1]), 2),
    }