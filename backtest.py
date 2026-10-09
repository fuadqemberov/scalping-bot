"""
Backtest the live strategy on historical candles before paper/live trading.

Uses the same pieces as the bot: strategy.compute_indicators /
evaluate_signal (incl. the confirmation filters) / calculate_sl_tp /
check_trade_costs, RiskManager.size_order (leverage + max loss per trade)
and PaperEngine fees, slippage and break-even stop. All symbols are
replayed bar by bar on one clock so MAX_OPEN_POSITIONS and
MAX_SAME_SIDE_POSITIONS apply as they do live.

Execution model (conservative):
    * A signal on a closed bar enters at that bar's close (the live bot
      enters at the ticker price right after the candle closes).
    * SL / TP are checked on the following bars' high/low. If a bar touches
      both, the stop is assumed to have hit first. A bar that opens beyond
      the stop fills at its open (gap).
    * The break-even move is applied after the bar's SL/TP check, i.e. it
      protects from the next bar on.
    * The daily drawdown breaker is not simulated.

Usage:
    python backtest.py                      # current config.py settings, 14 days
    python backtest.py --days 30
    python backtest.py --compare            # current vs. previous (10x, no filters) vs. old 1m
    python backtest.py --set LEVERAGE=15 ADX_MIN=25 BREAKEVEN_TRIGGER_R=0
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List

import ccxt
import pandas as pd

import config
import strategy
from paper_engine import PaperEngine
from risk_manager import RiskManager

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


@dataclass
class Params:
    name: str
    overrides: Dict[str, Any] = field(default_factory=dict)   # config.NAME -> value

    def get(self, key: str) -> Any:
        return self.overrides.get(key, getattr(config, key))


# Settings before this change (5m, 10x, no loss cap / break-even / filters)
PREVIOUS = dict(LEVERAGE=10.0, MAX_LOSS_PER_TRADE=0.0, BREAKEVEN_TRIGGER_R=0.0,
                EMA_TREND=0, ADX_MIN=0, VOLUME_MIN_RATIO=0.0, REQUIRE_CONFIRM_CANDLE=False)
# The original 1m settings from the first trade log
LEGACY = dict(PREVIOUS, TIMEFRAME="1m", RR_RATIO=1.5, MIN_TP_COST_MULTIPLE=0.0,
              MAX_OPEN_POSITIONS=3, MAX_SAME_SIDE_POSITIONS=99, SL_COOLDOWN_BARS=0)


@contextlib.contextmanager
def patched_config(overrides: Dict[str, Any]) -> Iterator[None]:
    old = {k: getattr(config, k) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(config, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(config, k, v)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def connect() -> ccxt.Exchange:
    for ex_id in [config.EXCHANGE] + list(config.FALLBACK_EXCHANGES):
        try:
            ex = getattr(ccxt, ex_id)({"enableRateLimit": True, "timeout": config.REQUEST_TIMEOUT_MS})
            ex.load_markets()
            print(f"Using {ex_id} public data")
            return ex
        except Exception as exc:
            print(f"{ex_id} unavailable: {type(exc).__name__}: {str(exc)[:120]}")
    raise SystemExit("No exchange reachable")


def fetch_history(ex: ccxt.Exchange, symbol: str, timeframe: str, days: float) -> pd.DataFrame:
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    now = ex.milliseconds()
    since = now - int(days * 86_400_000)
    rows: List[list] = []
    while since < now:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + tf_ms
        if nxt <= since:
            break
        since = nxt
        time.sleep(ex.rateLimit / 1000)
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df = df[df["timestamp"] <= now - tf_ms]  # drop the still-forming candle
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
@dataclass
class Result:
    params: Params
    trades: List[dict] = field(default_factory=list)
    skipped: Dict[str, int] = field(default_factory=dict)


def run(params: Params, data: Dict[str, pd.DataFrame]) -> Result:
    with patched_config(params.overrides):
        return _run(params, data)


def _run(params: Params, data: Dict[str, pd.DataFrame]) -> Result:
    engine = PaperEngine()
    risk = RiskManager()
    res = Result(params)
    lookback = config.RSI_LOOKBACK

    frames = {s: strategy.compute_indicators(df).set_index("timestamp", drop=False) for s, df in data.items()}
    entered_at: Dict[str, int] = {}       # symbol -> bar index of entry
    cooldown_until: Dict[str, int] = {}   # symbol -> first bar index allowed
    clock = sorted(set().union(*[f.index for f in frames.values()]))
    idx_of = {s: {ts: i for i, ts in enumerate(f.index)} for s, f in frames.items()}

    def skip(reason: str) -> None:
        res.skipped[reason] = res.skipped.get(reason, 0) + 1

    def close(sym: str, px: float, reason: str, ts) -> None:
        t = engine.close_position(px, reason, sym)
        if t:
            t["bar_time"] = ts
            res.trades.append(t)
            entered_at.pop(sym, None)
            if reason == "STOP_LOSS" and config.SL_COOLDOWN_BARS > 0:
                cooldown_until[sym] = idx_of[sym][ts] + config.SL_COOLDOWN_BARS + 1

    for ts in clock:
        # 1) exits inside this bar, then the break-even move
        for sym, f in frames.items():
            i = idx_of[sym].get(ts)
            if i is None:
                continue
            pos = engine.get_position(sym)
            if not pos or entered_at.get(sym) == i:
                continue
            bar = f.loc[ts]
            d = 1 if pos["side"] == "LONG" else -1
            stop_reason = "BREAKEVEN_STOP" if pos.get("breakeven_moved") else "STOP_LOSS"
            worst, best = (bar["low"], bar["high"]) if d > 0 else (bar["high"], bar["low"])
            if (bar["open"] - pos["sl"]) * d <= 0:
                close(sym, bar["open"], stop_reason, ts)
            elif (worst - pos["sl"]) * d <= 0:
                close(sym, pos["sl"], stop_reason, ts)
            elif (best - pos["tp"]) * d >= 0:
                close(sym, pos["tp"], "TAKE_PROFIT", ts)
            else:
                engine.update_market_price(bar["close"], sym)
                engine.apply_breakeven(best, sym)

        # 2) signals on this closed bar
        for sym, f in frames.items():
            i = idx_of[sym].get(ts)
            if i is None or i < lookback + 2:
                continue
            diag = strategy.evaluate_signal(f.iloc[i - lookback - 1:i + 1])
            signal = diag["signal"]
            if not signal:
                if diag.get("filters_failed"):
                    skip("filtered: " + diag["filters_failed"][0].split(" ")[0])
                continue
            price = float(f["close"].iloc[i])
            pos = engine.get_position(sym)
            if pos and signal != pos["side"] and config.EXIT_ON_REVERSAL:
                close(sym, price, "REVERSAL", ts)
                pos = None
                if not config.ENTER_ON_REVERSAL:
                    continue
            if pos:
                continue
            if cooldown_until.get(sym, -1) > i:
                skip("cooldown after SL")
                continue
            open_pos = [engine.get_position(s) for s in frames if engine.has_position(s)]
            if len(open_pos) >= config.MAX_OPEN_POSITIONS:
                skip("max open positions")
                continue
            if sum(p["side"] == signal for p in open_pos) >= config.MAX_SAME_SIDE_POSITIONS:
                skip("same-side limit")
                continue
            try:
                sl, tp = strategy.calculate_sl_tp(signal, price, float(f["ATR"].iloc[i]),
                                                  config.SL_ATR_MULTIPLIER, config.RR_RATIO)
            except ValueError:
                continue
            ok, _ = strategy.check_trade_costs(price, tp, config.MIN_TP_COST_MULTIPLE)
            if not ok:
                skip("TP below costs")
                continue
            qty, lev, why = risk.size_order(engine.balance, engine.get_equity(), engine.get_used_margin(),
                                            price, sl)
            if qty <= 0:
                skip(why.split(" ")[0] + " size")
                continue
            if engine.open_position(signal, price, qty, sl, tp, symbol=sym, leverage=lev):
                entered_at[sym] = i

    # Close leftovers at the last price so PnL is complete
    for sym, f in frames.items():
        if engine.has_position(sym):
            close(sym, float(f["close"].iloc[-1]), "END_OF_TEST", f.index[-1])
    return res


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report(res: Result) -> None:
    p, t = res.params, pd.DataFrame(res.trades)
    keys = ["TIMEFRAME", "LEVERAGE", "RR_RATIO", "MAX_LOSS_PER_TRADE", "BREAKEVEN_TRIGGER_R",
            "EMA_TREND", "ADX_MIN", "VOLUME_MIN_RATIO", "REQUIRE_CONFIRM_CANDLE"]
    print(f"\n=== {p.name}: " + ", ".join(f"{k}={p.get(k)}" for k in keys) + " ===")
    if t.empty:
        print("No trades.")
        if res.skipped:
            print("Signals skipped:", ", ".join(f"{k} {v}" for k, v in sorted(res.skipped.items())))
        return
    wins = t[t.net_pnl > 0]
    losses = t[t.net_pnl <= 0]
    eq = config.INITIAL_BALANCE + t.net_pnl.cumsum()
    peak = eq.cummax().clip(lower=config.INITIAL_BALANCE)
    loss_sum = losses.net_pnl.sum()
    pf = wins.net_pnl.sum() / -loss_sum if loss_sum < 0 else float("inf")
    print(f"Trades {len(t)} | win rate {len(wins) / len(t) * 100:.1f}% | profit factor {pf:.2f} | "
          f"avg win {wins.net_pnl.mean() if len(wins) else 0:+.2f} | "
          f"avg loss {losses.net_pnl.mean() if len(losses) else 0:+.2f} | "
          f"worst {t.net_pnl.min():+.2f}")
    print(f"Gross {t.gross_pnl.sum():+.2f} | fees {-t.total_fees.sum():.2f} | "
          f"(slippage inside fills {t.slippage_cost.sum():.2f}) | NET {t.net_pnl.sum():+.2f} USDT "
          f"({t.net_pnl.sum() / config.INITIAL_BALANCE * 100:+.2f}%) | max drawdown "
          f"{((peak - eq) / peak).max() * 100:.2f}%")
    print(t.groupby("symbol").net_pnl.agg(trades="count", net="sum").round(2).to_string())
    print(t.exit_reason.value_counts().to_string())
    if res.skipped:
        print("Signals skipped:", ", ".join(f"{k} {v}" for k, v in sorted(res.skipped.items())))


def parse_sets(items: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        key = key.strip().upper()
        if not hasattr(config, key):
            raise SystemExit(f"Unknown config setting {key}")
        try:
            out[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key] = raw
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=float, default=14)
    ap.add_argument("--symbols", nargs="*", default=config.SYMBOLS)
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                    help="override config.py values for the main run, e.g. LEVERAGE=15")
    ap.add_argument("--compare", action="store_true",
                    help="also run the previous settings (10x, no filters) and the old 1m settings")
    ap.add_argument("--csv", help="write the trades of the main run to this CSV file")
    args = ap.parse_args()

    runs = [Params("current", parse_sets(args.set))]
    if args.compare:
        runs += [Params("previous (10x, no filters)", dict(PREVIOUS)), Params("old 1m", dict(LEGACY))]

    ex = connect()
    cache: Dict[str, Dict[str, pd.DataFrame]] = {}
    for p in runs:
        tf = p.get("TIMEFRAME")
        if tf not in cache:
            cache[tf] = {}
            for sym in args.symbols:
                if sym not in ex.markets:
                    print(f"{sym} not listed on {ex.id} - skipped")
                    continue
                print(f"Fetching {args.days:g} days of {tf} {sym} ...")
                cache[tf][sym] = fetch_history(ex, sym, tf, args.days)
        res = run(p, cache[tf])
        report(res)
        if args.csv and p is runs[0] and res.trades:
            pd.DataFrame(res.trades).to_csv(args.csv, index=False)
            print(f"Trades written to {args.csv}")


if __name__ == "__main__":
    main()
