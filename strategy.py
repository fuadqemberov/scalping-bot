"""
Strategy module - pure functions only (no I/O, no state).

Indicators
    EMA   : standard exponential moving average, alpha = 2 / (n + 1),
            seeded with the SMA of the first n values (TradingView style).
    RSI   : Wilder's RSI, alpha = 1 / n, averages seeded with an SMA.
    ATR   : Wilder's Average True Range, alpha = 1 / n, seeded with an SMA.
    VWAP  : cumulative(typical_price * volume) / cumulative(volume),
            optionally reset at each UTC day (intraday session VWAP).

Strategies (config.STRATEGIES, first one to fire wins):
    trend_pullback   : described below
    squeeze_breakout : Bollinger Bands inside Keltner Channel for >= N bars,
                       then release with momentum + volume (+ EMA200 direction)
    bb_reversion     : ADX < 20 range; prior bar outside a band with extreme
                       RSI, signal bar closes back inside -> target middle band

trend_pullback (evaluated on the most recent row of the frame passed in,
which the bot guarantees is a *closed* candle) - "buy the pullback in an uptrend":
    LONG  : close > VWAP, EMA_fast > EMA_slow, RSI was < RSI_PULLBACK_LONG
            within the previous RSI_LOOKBACK bars and is now above it and rising.
    SHORT : close < VWAP, EMA_fast < EMA_slow, RSI was > RSI_PULLBACK_SHORT
            within the previous RSI_LOOKBACK bars and is now below it and falling.

Confirmation filters (config, each optional) then veto weak signals:
    * EMA200 trend: price on the right side of a sloping EMA200
    * ADX >= ADX_MIN: the market is trending, not chopping sideways
    * volume of the signal bar >= VOLUME_MIN_RATIO x 20-bar average
    * the signal candle closes in the trade direction
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


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Average Directional Index (trend strength, 0-100)."""
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    plus_dm[up.isna()] = np.nan
    minus_dm[down.isna()] = np.nan
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    tr[prev_close.isna()] = np.nan
    atr_s = wilder(tr, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * wilder(plus_dm, period) / atr_s
        minus_di = 100.0 * wilder(minus_dm, period) / atr_s
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    return wilder(dx, period)


def linreg_endpoint(series: pd.Series, period: int) -> pd.Series:
    """Rolling least-squares line fitted over `period` bars, value at the last bar."""
    x = np.arange(period, dtype=float)
    xc = x - x.mean()
    denom = float((xc ** 2).sum())

    def _fit(y: np.ndarray) -> float:
        ym = y.mean()
        return ym + float(np.dot(xc, y - ym)) / denom * xc[-1]

    return series.rolling(period, min_periods=period).apply(_fit, raw=True)


def squeeze_momentum(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """TTM-squeeze style momentum: close vs. the mid of the recent range,
    smoothed with a linear regression. > 0 and rising = bullish pressure."""
    hh = df["high"].rolling(period, min_periods=period).max()
    ll = df["low"].rolling(period, min_periods=period).min()
    sma = df["close"].rolling(period, min_periods=period).mean()
    return linreg_endpoint(df["close"] - ((hh + ll) / 2.0 + sma) / 2.0, period)


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

    # Confirmation filters
    if getattr(config, "EMA_TREND", 0) > 0:
        out["EMA_TREND"] = ema(out["close"], config.EMA_TREND)
        out["EMA_TREND_SLOPE"] = out["EMA_TREND"] - out["EMA_TREND"].shift(config.EMA_TREND_SLOPE_BARS)
    out["ADX"] = adx(out["high"], out["low"], out["close"], getattr(config, "ADX_PERIOD", 14))
    vol_n = getattr(config, "VOLUME_SMA_PERIOD", 20)
    # Average of the bars BEFORE the signal bar, so the signal bar is compared to its past
    out["VOL_SMA"] = out["volume"].shift(1).rolling(vol_n, min_periods=vol_n).mean()

    # Bollinger Bands, Keltner Channel, squeeze state and momentum
    bb_n, kc_n = getattr(config, "BB_PERIOD", 20), getattr(config, "KC_PERIOD", 20)
    mid = out["close"].rolling(bb_n, min_periods=bb_n).mean()
    sd = out["close"].rolling(bb_n, min_periods=bb_n).std(ddof=0)
    out["BB_MID"] = mid
    out["BB_UP"] = mid + getattr(config, "BB_STD", 2.0) * sd
    out["BB_LO"] = mid - getattr(config, "BB_STD", 2.0) * sd
    kc_mid = ema(out["close"], kc_n)
    kc_w = getattr(config, "KC_ATR_MULT", 1.5) * atr(out["high"], out["low"], out["close"], kc_n)
    out["KC_UP"] = kc_mid + kc_w
    out["KC_LO"] = kc_mid - kc_w
    out["SQZ_ON"] = (out["BB_UP"] < out["KC_UP"]) & (out["BB_LO"] > out["KC_LO"])
    out["MOM"] = squeeze_momentum(out, bb_n)

    # Human-readable names (e.g. EMA9, EMA21, RSI14, ATR14)
    out[f"EMA{config.EMA_FAST}"] = out["EMA_FAST"]
    out[f"EMA{config.EMA_SLOW}"] = out["EMA_SLOW"]
    out[f"RSI{config.RSI_PERIOD}"] = out["RSI"]
    out[f"ATR{config.ATR_PERIOD}"] = out["ATR"]
    return out


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
def evaluate_signal(df: pd.DataFrame, strategies: Optional[list] = None) -> Dict[str, Any]:
    """Run the enabled strategies (config.STRATEGIES) on the last closed bar.

    The first strategy that fires wins. The returned dict always has
    ``signal`` ("LONG"/"SHORT"/None), ``strategy``, ``reason``, ``trend`` and
    ``rsi`` (for logs and the dashboard); a fired signal may also carry
    ``sl``/``tp`` prices (relative to ``close``) or ``sl_mult``/``rr``.
    """
    names = list(strategies or getattr(config, "STRATEGIES", None) or ["trend_pullback"])
    reasons, filters_failed, base = [], [], None
    for name in names:
        fn = STRATEGY_FUNCS.get(name)
        if fn is None:
            reasons.append(f"{name}: unknown strategy")
            continue
        try:
            res = fn(df)
        except Exception as exc:  # one broken strategy must not stop the others
            res = {"signal": None, "reason": f"error {type(exc).__name__}: {exc}"}
        res["strategy"] = name
        if name == "trend_pullback":
            base = res
        if res.get("signal"):
            if base is None and "trend_pullback" in STRATEGY_FUNCS:
                base = trend_pullback(df)
            res.setdefault("trend", base.get("trend"))
            res.setdefault("rsi", base.get("rsi"))
            res["reason"] = f"[{name}] {res.get('reason', '')}"
            return res
        reasons.append(f"{name}: {res.get('reason', '')}")
        filters_failed += res.get("filters_failed") or []
    if base is None:
        base = trend_pullback(df)
    return {"signal": None, "strategy": None, "trend": base.get("trend"), "rsi": base.get("rsi"),
            "reason": " | ".join(reasons), "filters_failed": filters_failed}


def _last_rows_ok(df: pd.DataFrame, cols: list, n: int) -> Optional[str]:
    if df is None or len(df) < n:
        return "not enough bars"
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return f"indicators missing {missing}"
    if df[cols].iloc[-n:].isna().any().any():
        return "indicators warming up"
    return None


def squeeze_breakout(df: pd.DataFrame) -> Dict[str, Any]:
    """Volatility squeeze release with momentum, volume and trend confirmation.

    Squeeze = Bollinger Bands inside the Keltner Channel (volatility is
    compressed). When the bands expand back outside after >= SQZ_MIN_BARS,
    a move usually starts; trade it in the direction of the momentum.
    """
    min_bars, look = config.SQZ_MIN_BARS, config.SQZ_LOOKBACK
    n = min_bars + look + 2
    cols = ["close", "BB_MID", "SQZ_ON", "MOM", "VOL_SMA", "ATR"]
    bad = _last_rows_ok(df, cols, n)
    if bad:
        return {"signal": None, "reason": bad}
    tail = df.iloc[-n:]
    s = tail["SQZ_ON"].to_numpy(dtype=bool)
    last, prev = tail.iloc[-1], tail.iloc[-2]
    if s[-1]:
        run = int(np.argmax(~s[::-1])) if (~s).any() else len(s)
        return {"signal": None, "reason": f"squeeze on for {run} bars, waiting for release"}

    released = False
    for j in range(len(s) - 1, len(s) - 1 - look, -1):
        if not s[j] and s[j - 1]:
            run = 0
            k = j - 1
            while k >= 0 and s[k]:
                run, k = run + 1, k - 1
            released = run >= min_bars
            if not released:
                return {"signal": None, "reason": f"squeeze released after only {run} bars (< {min_bars})"}
            break
    if not released:
        return {"signal": None, "reason": "no recent squeeze release"}

    mom, mom_prev = float(last["MOM"]), float(prev["MOM"])
    if mom > 0 and mom > mom_prev and last["close"] > last["BB_MID"]:
        side = "LONG"
    elif mom < 0 and mom < mom_prev and last["close"] < last["BB_MID"]:
        side = "SHORT"
    else:
        return {"signal": None, "reason": f"squeeze released but momentum unclear ({mom_prev:.4g} -> {mom:.4g})"}

    failed = []
    if last["volume"] < config.SQZ_VOLUME_RATIO * last["VOL_SMA"]:
        failed.append(f"volume {last['volume'] / last['VOL_SMA']:.2f}x avg < {config.SQZ_VOLUME_RATIO:g}x")
    if config.SQZ_USE_TREND_FILTER and getattr(config, "EMA_TREND", 0) > 0:
        t = last.get("EMA_TREND")
        d = 1 if side == "LONG" else -1
        if t is None or pd.isna(t):
            failed.append(f"EMA{config.EMA_TREND} warming up")
        elif (last["close"] - t) * d <= 0:
            failed.append(f"against EMA{config.EMA_TREND} trend")
    if failed:
        return {"signal": None, "reason": f"{side} breakout filtered out: {'; '.join(failed)}",
                "filters_failed": failed}
    return {"signal": side, "sl_mult": config.SQZ_SL_ATR, "rr": config.SQZ_RR,
            "reason": f"squeeze release, momentum {mom_prev:.4g} -> {mom:.4g}, "
                      f"volume {last['volume'] / last['VOL_SMA']:.1f}x"}


def bb_reversion(df: pd.DataFrame) -> Dict[str, Any]:
    """Range-market mean reversion back into the Bollinger Bands.

    Previous bar closed outside a band with an extreme RSI, the signal bar
    closes back inside in the opposite colour, and ADX says there is no
    trend. Target = middle band, stop = beyond the swing extreme.
    """
    cols = ["open", "high", "low", "close", "BB_UP", "BB_LO", "BB_MID", "RSI", "ADX", "ATR"]
    bad = _last_rows_ok(df, cols, 2)
    if bad:
        return {"signal": None, "reason": bad}
    last, prev = df.iloc[-1], df.iloc[-2]
    if last["ADX"] >= config.BBR_ADX_MAX:
        return {"signal": None, "reason": f"ADX {last['ADX']:.1f} >= {config.BBR_ADX_MAX} (trending, no fade)"}

    close, buf = float(last["close"]), config.BBR_SL_ATR_BUFFER * float(last["ATR"])
    if (prev["close"] < prev["BB_LO"] and prev["RSI"] < config.BBR_RSI_LONG
            and close > last["BB_LO"] and close > last["open"]):
        side, tp = "LONG", float(last["BB_MID"])
        sl = min(float(prev["low"]), float(last["low"])) - buf
    elif (prev["close"] > prev["BB_UP"] and prev["RSI"] > config.BBR_RSI_SHORT
            and close < last["BB_UP"] and close < last["open"]):
        side, tp = "SHORT", float(last["BB_MID"])
        sl = max(float(prev["high"]), float(last["high"])) + buf
    else:
        return {"signal": None, "reason": f"range (ADX {last['ADX']:.1f}), no band re-entry"}

    risk, reward = abs(close - sl), (tp - close) * (1 if side == "LONG" else -1)
    if reward <= 0 or reward < config.BBR_MIN_RR * risk:
        return {"signal": None, "reason": f"{side} fade skipped: reward {reward:.4g} < "
                                          f"{config.BBR_MIN_RR:g} x risk {risk:.4g}",
                "filters_failed": ["reward/risk"]}
    return {"signal": side, "sl": sl, "tp": tp, "close": close,
            "reason": f"band re-entry, RSI {prev['RSI']:.1f}, ADX {last['ADX']:.1f}, RR {reward / risk:.2f}"}


def trend_pullback(df: pd.DataFrame) -> Dict[str, Any]:
    """Pullback in a trend: EMA9/21 + VWAP trend, RSI dip/spike and turn.

    Returns a dict with ``signal`` ("LONG", "SHORT" or None), ``trend``
    ("UP", "DOWN" or "MIXED"), the RSI values involved and a short
    human-readable ``reason`` (useful for logging why nothing fired).
    """
    lookback = config.RSI_LOOKBACK
    lo, hi = config.RSI_PULLBACK_LONG, config.RSI_PULLBACK_SHORT
    result: Dict[str, Any] = {"signal": None, "trend": None, "rsi": None,
                              "rsi_min": None, "rsi_max": None, "reason": "",
                              "filters_failed": []}
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

    if result["signal"]:
        failed = confirmation_failures(last, result["signal"])
        result["filters_failed"] = failed
        if failed:
            result["reason"] = f"{result['signal']} filtered out: {'; '.join(failed)}"
            result["signal"] = None
        else:
            result["sl_mult"] = config.SL_ATR_MULTIPLIER
            result["rr"] = config.RR_RATIO
    return result


STRATEGY_FUNCS = {
    "trend_pullback": trend_pullback,
    "squeeze_breakout": squeeze_breakout,
    "bb_reversion": bb_reversion,
}


def sl_tp_for_signal(diag: Dict[str, Any], side: str, entry_price: float,
                     atr_value: float) -> Tuple[float, float]:
    """Stop / target for a fired signal.

    Strategies that set explicit ``sl``/``tp`` prices (measured from the
    signal bar's ``close``) keep those distances from the actual entry price;
    the others use ATR multiples (``sl_mult`` x ATR, ``rr`` x stop).
    """
    if diag.get("sl") is not None and diag.get("tp") is not None:
        shift = entry_price - float(diag.get("close", entry_price))
        sl, tp = float(diag["sl"]) + shift, float(diag["tp"]) + shift
        ok = (sl < entry_price < tp) if side == "LONG" else (tp < entry_price < sl)
        if not ok:
            raise ValueError(f"invalid {side} levels sl={sl:.6g} entry={entry_price:.6g} tp={tp:.6g}")
        return sl, tp
    return calculate_sl_tp(side, entry_price, atr_value,
                           diag.get("sl_mult") or config.SL_ATR_MULTIPLIER,
                           diag.get("rr") or config.RR_RATIO)


def confirmation_failures(row: pd.Series, side: str) -> list:
    """Extra checks on the signal bar. Returns the list of failed checks (empty = OK)."""
    failed = []
    d = 1 if side == "LONG" else -1

    if getattr(config, "EMA_TREND", 0) > 0:
        trend, slope = row.get("EMA_TREND"), row.get("EMA_TREND_SLOPE")
        if trend is None or pd.isna(trend) or pd.isna(slope):
            failed.append(f"EMA{config.EMA_TREND} warming up")
        elif (row["close"] - trend) * d <= 0:
            failed.append(f"close {'below' if d > 0 else 'above'} EMA{config.EMA_TREND}")
        elif slope * d <= 0:
            failed.append(f"EMA{config.EMA_TREND} {'falling' if d > 0 else 'rising'}")

    adx_min = getattr(config, "ADX_MIN", 0)
    if adx_min > 0:
        a = row.get("ADX")
        if a is None or pd.isna(a):
            failed.append("ADX warming up")
        elif a < adx_min:
            failed.append(f"ADX {a:.1f} < {adx_min} (choppy)")

    vol_ratio = getattr(config, "VOLUME_MIN_RATIO", 0)
    if vol_ratio > 0:
        avg = row.get("VOL_SMA")
        if avg is None or pd.isna(avg) or avg <= 0:
            failed.append("volume average warming up")
        elif row["volume"] < vol_ratio * avg:
            failed.append(f"volume {row['volume'] / avg:.2f}x avg < {vol_ratio:g}x")

    if getattr(config, "REQUIRE_CONFIRM_CANDLE", False) and "open" in row:
        if (row["close"] - row["open"]) * d <= 0:
            failed.append(f"signal candle is not {'green' if d > 0 else 'red'}")
    return failed


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
