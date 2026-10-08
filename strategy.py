"""
Strategy module - pure functions only (no I/O, no state).

Indicators
    EMA   : standard exponential moving average, alpha = 2 / (n + 1),
            seeded with the SMA of the first n values (TradingView style).
    RSI   : Wilder's RSI, alpha = 1 / n, averages seeded with an SMA.
    ATR   : Wilder's Average True Range, alpha = 1 / n, seeded with an SMA.
    VWAP  : cumulative(typical_price * volume) / cumulative(volume),
            optionally reset at each UTC day (intraday session VWAP).

Signals (evaluated on the most recent row of the frame passed in, which the
bot guarantees is a *closed* candle) - "buy the pullback in an uptrend":
    LONG  : close > VWAP, EMA_fast > EMA_slow, RSI was < RSI_PULLBACK_LONG
            within the previous RSI_LOOKBACK bars and is now above it and rising.
    SHORT : close < VWAP, EMA_fast < EMA_slow, RSI was > RSI_PULLBACK_SHORT
            within the previous RSI_LOOKBACK bars and is now below it and falling.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import config

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


# ---------------------------------------------------------------------------
# Low-level smoothing helper
# ---------------------------------------------------------------------------
def _recursive_smooth(series: pd.Series, period: int, alpha: float) -> pd.Series:
    """Recursive exponential smoothing seeded with an SMA.

    out[seed] = mean(first `period` valid values)
    out[i]    = alpha * x[i] + (1 - alpha) * out[i-1]

    Leading NaNs are skipped; values before the seed are NaN.
    """
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    valid_idx = np.flatnonzero(~np.isnan(values))
    if period <= 0 or len(valid_idx) < period:
        return pd.Series(out, index=series.index)

    start = valid_idx[0]
    seed_end = start + period  # exclusive
    window = values[start:seed_end]
    if len(window) < period or np.isnan(window).any():
        return pd.Series(out, index=series.index)

    out[seed_end - 1] = window.mean()
    for i in range(seed_end, len(values)):
        x = values[i]
        # Carry the previous value forward if a data point is missing
        out[i] = out[i - 1] if np.isnan(x) else alpha * x + (1.0 - alpha) * out[i - 1]
    return pd.Series(out, index=series.index)


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def ema(series: pd.Series, period: int) -> pd.Series:
    return _recursive_smooth(series, period, 2.0 / (period + 1.0))


def wilder(series: pd.Series, period: int) -> pd.Series:
    return _recursive_smooth(series, period, 1.0 / period)


def rsi(close: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = wilder(gain, period)
    avg_loss = wilder(loss, period)

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        out = 100.0 - 100.0 / (1.0 + rs)
    # Edge cases: no losses -> 100, no movement at all -> 50
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    out[avg_gain.isna() | avg_loss.isna()] = np.nan
    return out


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = config.ATR_PERIOD) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1, skipna=True)  # first bar: prev_close is NaN -> TR = high - low
    return wilder(tr, period)


def vwap(df: pd.DataFrame, reset_daily: bool = config.VWAP_RESET_DAILY) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    if reset_daily:
        session = pd.to_datetime(df["timestamp"], utc=True).dt.date
        cum_pv = pv.groupby(session).cumsum()
        cum_vol = df["volume"].groupby(session).cumsum()
    else:
        cum_pv = pv.cumsum()
        cum_vol = df["volume"].cumsum()
    out = cum_pv / cum_vol.replace(0.0, np.nan)
    # Zero-volume opening bars of a session: fall back to typical price
    return out.fillna(typical)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of an OHLCV frame with EMA9, EMA21, RSI14, ATR14, VWAP columns.

    The frame must contain: timestamp (datetime or ms), open, high, low, close, volume.
    Column names for the indicators follow the configured periods, and fixed
    aliases (EMA_FAST, EMA_SLOW, RSI, ATR) are added for period-agnostic access.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS))
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV frame missing columns: {missing}")

    out = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(out["timestamp"]):
        out["timestamp"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True)

    out["EMA_FAST"] = ema(out["close"], config.EMA_FAST)
    out["EMA_SLOW"] = ema(out["close"], config.EMA_SLOW)
    out["RSI"] = rsi(out["close"], config.RSI_PERIOD)
    out["ATR"] = atr(out["high"], out["low"], out["close"], config.ATR_PERIOD)
    out["VWAP"] = vwap(out)

    # Human-readable names (e.g. EMA9, EMA21, RSI14, ATR14)
    out[f"EMA{config.EMA_FAST}"] = out["EMA_FAST"]
    out[f"EMA{config.EMA_SLOW}"] = out["EMA_SLOW"]
    out[f"RSI{config.RSI_PERIOD}"] = out["RSI"]
    out[f"ATR{config.ATR_PERIOD}"] = out["ATR"]
    return out


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
def evaluate_signal(df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate the entry rules on the last row and explain the outcome.

    Returns a dict with ``signal`` ("LONG", "SHORT" or None), ``trend``
    ("UP", "DOWN" or "MIXED"), the RSI values involved and a short
    human-readable ``reason`` (useful for logging why nothing fired).
    """
    lookback = config.RSI_LOOKBACK
    lo, hi = config.RSI_PULLBACK_LONG, config.RSI_PULLBACK_SHORT
    result: Dict[str, Any] = {"signal": None, "trend": None, "rsi": None,
                              "rsi_min": None, "rsi_max": None, "reason": ""}
    if df is None or len(df) < lookback + 2:
        result["reason"] = "not enough bars"
        return result
    needed = ["close", "VWAP", "EMA_FAST", "EMA_SLOW", "RSI"]
    if any(c not in df.columns for c in needed):
        result["reason"] = "indicators missing"
        return result

    tail = df.iloc[-(lookback + 1):]
    if tail[needed].isna().any().any():
        result["reason"] = "indicators warming up"
        return result

    last = tail.iloc[-1]
    rsi_now = float(last["RSI"])
    rsi_prev = float(tail["RSI"].iloc[-2])
    rsi_window = tail["RSI"].iloc[:-1]  # previous `lookback` bars
    rsi_min, rsi_max = float(rsi_window.min()), float(rsi_window.max())
    result.update(rsi=rsi_now, rsi_min=rsi_min, rsi_max=rsi_max)

    above_vwap = last["close"] > last["VWAP"]
    below_vwap = last["close"] < last["VWAP"]
    ema_up = last["EMA_FAST"] > last["EMA_SLOW"]
    ema_down = last["EMA_FAST"] < last["EMA_SLOW"]

    if above_vwap and ema_up:
        result["trend"] = "UP"
        dipped = rsi_min < lo
        if dipped and rsi_now > lo and rsi_now > rsi_prev:
            result["signal"] = "LONG"
            result["reason"] = f"uptrend pullback: RSI {rsi_min:.1f} -> {rsi_now:.1f} crossed {lo}"
        elif not dipped:
            result["reason"] = f"uptrend, waiting for pullback (RSI min {rsi_min:.1f} >= {lo})"
        else:
            result["reason"] = f"uptrend, RSI {rsi_now:.1f} still below {lo} or not rising yet"
    elif below_vwap and ema_down:
        result["trend"] = "DOWN"
        spiked = rsi_max > hi
        if spiked and rsi_now < hi and rsi_now < rsi_prev:
            result["signal"] = "SHORT"
            result["reason"] = f"downtrend pullback: RSI {rsi_max:.1f} -> {rsi_now:.1f} crossed {hi}"
        elif not spiked:
            result["reason"] = f"downtrend, waiting for pullback (RSI max {rsi_max:.1f} <= {hi})"
        else:
            result["reason"] = f"downtrend, RSI {rsi_now:.1f} still above {hi} or not falling yet"
    else:
        result["trend"] = "MIXED"
        result["reason"] = (
            f"no clear trend (close {'>' if above_vwap else '<'} VWAP, "
            f"EMA{config.EMA_FAST} {'>' if ema_up else '<'} EMA{config.EMA_SLOW})"
        )
    return result


def generate_signal(df: pd.DataFrame) -> Optional[str]:
    """Return "LONG", "SHORT" or None based on the last row of an indicator frame."""
    return evaluate_signal(df)["signal"]


def calculate_sl_tp(
    side: str,
    entry_price: float,
    atr_value: float,
    sl_multiplier: float = config.SL_ATR_MULTIPLIER,
    rr_ratio: float = config.RR_RATIO,
) -> Tuple[float, float]:
    """ATR-based stop-loss and take-profit.

    stop distance   = sl_multiplier * ATR
    target distance = rr_ratio * stop distance
    """
    if entry_price <= 0 or atr_value is None or not np.isfinite(atr_value) or atr_value <= 0:
        raise ValueError(f"invalid entry/ATR (entry={entry_price}, atr={atr_value})")
    stop_dist = sl_multiplier * atr_value
    target_dist = rr_ratio * stop_dist
    side = side.upper()
    if side == "LONG":
        return entry_price - stop_dist, entry_price + target_dist
    if side == "SHORT":
        return entry_price + stop_dist, entry_price - target_dist
    raise ValueError(f"invalid side {side!r}")


def round_trip_cost_pct(fee_rate: float = config.FEE_RATE,
                        max_slippage: float = config.SLIPPAGE_MAX) -> float:
    """Worst-case fees + slippage for entry and exit, as a fraction of price."""
    return 2.0 * (fee_rate + max_slippage)


def check_trade_costs(entry_price: float, tp: float,
                      multiple: float = getattr(config, "MIN_TP_COST_MULTIPLE", 0.0)) -> Tuple[bool, str]:
    """(ok, reason): is the take-profit distance large enough to beat costs?

    A trade is only worth taking when the target is at least ``multiple``
    times the round-trip cost; otherwise even a winner nets ~nothing.
    """
    if multiple <= 0 or entry_price <= 0:
        return True, ""
    tp_pct = abs(tp - entry_price) / entry_price
    need = multiple * round_trip_cost_pct()
    if tp_pct < need:
        return False, (f"TP distance {tp_pct * 100:.3f}% < {multiple:g}x round-trip "
                       f"costs ({need * 100:.3f}%) - ATR too small")
    return True, ""
