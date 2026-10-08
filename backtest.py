"""
Backtest the live strategy on historical candles before paper/live trading.

Uses the same pieces as the bot: strategy.compute_indicators /
evaluate_signal / calculate_sl_tp / check_trade_costs, RiskManager sizing and
PaperEngine fees + slippage. All symbols are replayed bar by bar on one clock
so MAX_OPEN_POSITIONS and MAX_SAME_SIDE_POSITIONS apply as they do live.

Execution model (conservative):
    * A signal on a closed bar enters at that bar's close (the live bot
      enters at the ticker price right after the candle closes).
    * SL / TP are checked on the following bars' high/low. If a bar touches
      both, the stop is assumed to have hit first. A bar that opens beyond
      the stop fills at its open (gap).
    * The daily drawdown breaker is not simulated.

Usage:
    python backtest.py                      # current config.py settings, 14 days
    python backtest.py --days 30
    python backtest.py --compare            # current settings vs. the old 1m settings
    python backtest.py --timeframe 15m --rr 2.5
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import ccxt
import pandas as pd

import config
import strategy
from paper_engine import PaperEngine
from risk_manager import RiskManager

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("backtest")


@dataclass
class Params:
    name: str
    timeframe: str = config.TIMEFRAME
    sl_mult: float = config.SL_ATR_MULTIPLIER
    rr: float = config.RR_RATIO
    min_tp_cost: float = config.MIN_TP_COST_MULTIPLE
    max_open: int = config.MAX_OPEN_POSITIONS
    max_same_side: int = config.MAX_SAME_SIDE_POSITIONS
    cooldown_bars: int = config.SL_COOLDOWN_BARS


LEGACY = dict(timeframe="1m", sl_mult=1.5, rr=1.5, min_tp_cost=0.0,
              max_open=3, max_same_side=99, cooldown_bars=0)


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
    engine: PaperEngine
    trades: List[dict] = field(default_factory=list)
    skipped: Dict[str, int] = field(default_factory=dict)


def run(params: Params, data: Dict[str, pd.DataFrame]) -> Result:
    engine = PaperEngine()
    risk = RiskManager()
    res = Result(params, engine)
    lookback = config.RSI_LOOKBACK

    frames = {s: strategy.compute_indicators(df).set_index("timestamp", drop=False) for s, df in data.items()}
    positions_on: Dict[str, int] = {}     # symbol -> bar index of entry
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
            positions_on.pop(sym, None)
            if reason == "STOP_LOSS" and params.cooldown_bars > 0:
                cooldown_until[sym] = idx_of[sym][ts] + params.cooldown_bars + 1

    for ts in clock:
        # 1) exits inside this bar
        for sym, f in frames.items():
            if ts not in idx_of[sym]:
                continue
            pos = engine.get_position(sym)
            if not pos or positions_on.get(sym) == idx_of[sym][ts]:
                continue
            bar = f.loc[ts]
            if pos["side"] == "LONG":
                if bar["open"] <= pos["sl"]:
                    close(sym, bar["open"], "STOP_LOSS", ts)
                elif bar["low"] <= pos["sl"]:
                    close(sym, pos["sl"], "STOP_LOSS", ts)
                elif bar["high"] >= pos["tp"]:
                    close(sym, pos["tp"], "TAKE_PROFIT", ts)
            else:
                if bar["open"] >= pos["sl"]:
                    close(sym, bar["open"], "STOP_LOSS", ts)
                elif bar["high"] >= pos["sl"]:
                    close(sym, pos["sl"], "STOP_LOSS", ts)
                elif bar["low"] <= pos["tp"]:
                    close(sym, pos["tp"], "TAKE_PROFIT", ts)

        # 2) signals on this closed bar
        for sym, f in frames.items():
            i = idx_of[sym].get(ts)
            if i is None or i < lookback + 2:
                continue
            diag = strategy.evaluate_signal(f.iloc[i - lookback - 1:i + 1])
            signal = diag["signal"]
            if not signal:
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
            if len(open_pos) >= params.max_open:
                skip("max open positions")
                continue
            if sum(p["side"] == signal for p in open_pos) >= params.max_same_side:
                skip("same-side limit")
                continue
            atr_v = float(f["ATR"].iloc[i])
            try:
                sl, tp = strategy.calculate_sl_tp(signal, price, atr_v, params.sl_mult, params.rr)
            except ValueError:
                continue
            ok, _ = strategy.check_trade_costs(price, tp, params.min_tp_cost)
            if not ok:
                skip("TP below costs")
                continue
            if config.POSITION_SIZING_MODE == "margin":
                lev = config.LEVERAGE
                qty = risk.calculate_margin_size(engine.balance, price, engine.get_used_margin())
            else:
                lev = None
                qty = risk.calculate_position_size(engine.get_equity(), price, sl)
            if qty <= 0:
                skip("size zero")
                continue
            if engine.open_position(signal, price, qty, sl, tp, symbol=sym, leverage=lev):
                positions_on[sym] = i

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
    print(f"\n=== {p.name}: {p.timeframe}, SL {p.sl_mult}xATR, RR {p.rr}, "
          f"min TP {p.min_tp_cost}x costs, max open {p.max_open}, same-side {p.max_same_side}, "
          f"cooldown {p.cooldown_bars} bars ===")
    if t.empty:
        print("No trades.")
        return
    wins = t[t.net_pnl > 0]
    losses = t[t.net_pnl <= 0]
    eq = config.INITIAL_BALANCE + t.net_pnl.cumsum()
    peak = eq.cummax().clip(lower=config.INITIAL_BALANCE)
    pf = wins.net_pnl.sum() / -losses.net_pnl.sum() if len(losses) and losses.net_pnl.sum() < 0 else float("inf")
    tp_pct = ((t.tp - t.entry_price).abs() / t.entry_price * 100).median()
    sl_pct = ((t.sl - t.entry_price).abs() / t.entry_price * 100).median()
    print(f"Trades {len(t)} | win rate {len(wins) / len(t) * 100:.1f}% | profit factor {pf:.2f}")
    print(f"Gross {t.gross_pnl.sum():+.2f} | fees {-t.total_fees.sum():.2f} | "
          f"(slippage inside fills {t.slippage_cost.sum():.2f}) | NET {t.net_pnl.sum():+.2f} USDT "
          f"({t.net_pnl.sum() / config.INITIAL_BALANCE * 100:+.2f}%)")
    print(f"Max drawdown {((peak - eq) / peak).max() * 100:.2f}% | median SL {sl_pct:.3f}% | median TP {tp_pct:.3f}% "
          f"| round-trip cost {strategy.round_trip_cost_pct() * 100:.3f}% worst case")
    print(t.groupby("symbol").net_pnl.agg(trades="count", net="sum").round(2).to_string())
    print(t.exit_reason.value_counts().to_string())
    if res.skipped:
        print("Signals skipped:", ", ".join(f"{k} {v}" for k, v in sorted(res.skipped.items())))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=float, default=14)
    ap.add_argument("--symbols", nargs="*", default=config.SYMBOLS)
    ap.add_argument("--timeframe")
    ap.add_argument("--sl-mult", type=float)
    ap.add_argument("--rr", type=float)
    ap.add_argument("--min-tp-cost", type=float)
    ap.add_argument("--max-open", type=int)
    ap.add_argument("--max-same-side", type=int)
    ap.add_argument("--cooldown", type=int, dest="cooldown_bars")
    ap.add_argument("--compare", action="store_true", help="also run the old 1m settings")
    ap.add_argument("--csv", help="write the trades of the main run to this CSV file")
    args = ap.parse_args()

    overrides = {k: v for k in ("timeframe", "sl_mult", "rr", "min_tp_cost", "max_open",
                                "max_same_side", "cooldown_bars")
                 if (v := getattr(args, k)) is not None}
    runs = [Params("current", **overrides)]
    if args.compare:
        runs.append(Params("old 1m settings", **LEGACY))

    ex = connect()
    cache: Dict[str, Dict[str, pd.DataFrame]] = {}
    for p in runs:
        if p.timeframe not in cache:
            cache[p.timeframe] = {}
            for sym in args.symbols:
                if sym not in ex.markets:
                    print(f"{sym} not listed on {ex.id} - skipped")
                    continue
                print(f"Fetching {args.days:g} days of {p.timeframe} {sym} ...")
                cache[p.timeframe][sym] = fetch_history(ex, sym, p.timeframe, args.days)
        res = run(p, cache[p.timeframe])
        report(res)
        if args.csv and p is runs[0] and res.trades:
            pd.DataFrame(res.trades).to_csv(args.csv, index=False)
            print(f"Trades written to {args.csv}")


if __name__ == "__main__":
    main()
