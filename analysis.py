"""
Trade analysis - turns a trade history into diagnoses and concrete fixes.

Works on the engine's trade dicts (live bot / backtest) and on the CSV the
dashboard exports, old or new format. Pure functions, no I/O:

    res = analyze(trades)          # list of dicts or DataFrame
    print(report_text(res))

What it measures
    * Cost structure: how much of each loss / of the gross PnL went to fees +
      slippage, the *real* reward:risk after costs and the win rate needed to
      break even. (This is what sank the 1m and the 100x runs.)
    * Results in R (multiples of the initial stop risk): expectancy, profit
      factor, streaks.
    * MFE / MAE (max favourable / adverse excursion, in R): did losers go
      into profit first (-> break-even / partial exits), did winners nearly
      hit the stop (-> stop too tight), did losers never work at all
      (-> bad entries)?
    * Breakdowns by strategy, symbol, side, entry hour, exit reason.
    * Insights: rule-based, plain-language recommendations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

import config

# Dashboard CSV header -> canonical column
_CSV_MAP = {
    "#": "id", "Symbol": "symbol", "Side": "side", "Qty": "qty",
    "Entry Time": "entry_time", "Exit Time": "exit_time", "Duration": "duration",
    "Entry": "entry", "Exit": "exit", "SL": "sl", "TP": "tp",
    "Gross PnL": "gross", "Fees": "fees", "Slippage $": "slippage", "Net PnL": "net",
    "PnL %": "pnl_pct", "Reason": "reason", "Strategy": "strategy", "Balance": "balance",
    "Lev": "leverage", "MFE R": "mfe_r", "MAE R": "mae_r", "Stop %": "stop_pct",
    "Target %": "target_pct", "Hour": "ctx_hour_utc", "ADX": "ctx_adx",
    "ATR %": "ctx_atr_pct", "Vol x": "ctx_vol_ratio", "Signal": "ctx_reason",
}
# Engine trade dict key -> canonical column
_ENGINE_MAP = {
    "entry_price": "entry", "exit_price": "exit", "gross_pnl": "gross", "total_fees": "fees",
    "slippage_cost": "slippage", "net_pnl": "net", "exit_reason": "reason",
    "balance_after": "balance", "duration_sec": "duration_sec",
}

MIN_SAMPLE = 30   # below this many trades, results are mostly luck


def _duration_to_sec(v: Any) -> Optional[float]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    total, num = 0.0, ""
    for ch in str(v):
        if ch.isdigit() or ch == ".":
            num += ch
        elif ch in "hms" and num:
            total += float(num) * {"h": 3600, "m": 60, "s": 1}[ch]
            num = ""
    return total


def _to_time(col: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(col):
        return pd.to_datetime(col, utc=True)
    if col.map(lambda v: hasattr(v, "year")).all():          # datetime objects
        return pd.to_datetime(col, utc=True)
    out = pd.to_datetime(col, errors="coerce", utc=True, format="%m-%d %H:%M:%S")  # dashboard CSV
    if out.isna().mean() > 0.5:
        out = pd.to_datetime(col, errors="coerce", utc=True, format="ISO8601")
    return out


def normalize(trades: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
    """Canonical trade table with derived columns (R, stop %, cost %, ...)."""
    df = pd.DataFrame(trades).copy()
    if df.empty:
        return df
    df = df.rename(columns={**_CSV_MAP, **_ENGINE_MAP})
    df = df.loc[:, ~df.columns.duplicated()]
    for col in ("qty", "entry", "exit", "sl", "tp", "gross", "fees", "slippage", "net",
                "mfe_r", "mae_r", "stop_pct", "target_pct", "leverage"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "duration_sec" not in df:
        df["duration_sec"] = df["duration"].map(_duration_to_sec) if "duration" in df else np.nan
    if "strategy" not in df:
        df["strategy"] = "unknown"
    df["strategy"] = df["strategy"].fillna("unknown").replace("", "unknown")
    if "entry_time" in df:
        df["entry_time"] = _to_time(df["entry_time"])

    df["notional"] = df["qty"] * df["entry"]
    if "stop_pct" not in df or df["stop_pct"].isna().all():
        df["stop_pct"] = (df["sl"] - df["entry"]).abs() / df["entry"] * 100
    if "target_pct" not in df or df["target_pct"].isna().all():
        df["target_pct"] = (df["tp"] - df["entry"]).abs() / df["entry"] * 100
    df["risk_usd"] = df["qty"] * (df["sl"] - df["entry"]).abs()
    df["r"] = df["net"] / df["risk_usd"].replace(0, np.nan)
    slip = df["slippage"] if "slippage" in df else 0.0
    df["cost_usd"] = df["fees"] + slip
    df["cost_pct"] = df["cost_usd"] / df["notional"] * 100
    df["win"] = df["net"] > 0
    if "ctx_hour_utc" not in df and "entry_time" in df:
        df["ctx_hour_utc"] = df["entry_time"].dt.hour
    if "entry_time" in df:
        df = df.sort_values("entry_time").reset_index(drop=True)
    return df


def _group(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if key not in df or df[key].isna().all():
        return pd.DataFrame()
    g = df.groupby(key, dropna=True)
    out = pd.DataFrame({
        "trades": g.size(),
        "win %": g["win"].mean() * 100,
        "net $": g["net"].sum(),
        "avg R": g["r"].mean(),
        "fees $": g["fees"].sum(),
    })
    return out.sort_values("net $").round(2)


def _max_streak(wins: pd.Series, value: bool) -> int:
    best = cur = 0
    for w in wins:
        cur = cur + 1 if bool(w) == value else 0
        best = max(best, cur)
    return best


def analyze(trades: Union[pd.DataFrame, List[Dict[str, Any]]]) -> Dict[str, Any]:
    df = normalize(trades)
    if df.empty:
        return {"df": df, "summary": {"trades": 0}, "groups": {}, "insights": ["No trades yet."]}

    n = len(df)
    wins, losses = df[df["win"]], df[~df["win"]]
    gross_win, gross_loss = wins["net"].sum(), -losses["net"].sum()
    cost_rt = float(df["cost_pct"].median())          # measured round trip, % of notional
    stop = float(df["stop_pct"].median())
    target = float(df["target_pct"].median())
    eff_rr = (target - cost_rt) / (stop + cost_rt) if stop + cost_rt > 0 else np.nan
    be_wr = 1 / (1 + eff_rr) * 100 if eff_rr and eff_rr > 0 else 100.0

    s: Dict[str, Any] = {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / n * 100,
        "net": df["net"].sum(),
        "gross": df["gross"].sum(),
        "fees": df["fees"].sum(),
        "slippage": df["slippage"].sum() if "slippage" in df else 0.0,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("inf"),
        "avg_win": wins["net"].mean() if len(wins) else 0.0,
        "avg_loss": losses["net"].mean() if len(losses) else 0.0,
        "expectancy_usd": df["net"].mean(),
        "expectancy_r": df["r"].mean(),
        "median_stop_pct": stop,
        "median_target_pct": target,
        "round_trip_cost_pct": cost_rt,
        "effective_rr": eff_rr,
        "breakeven_win_rate": be_wr,
        "cost_share_of_losses": (losses["cost_usd"].sum() / gross_loss * 100) if gross_loss > 0 else 0.0,
        "max_loss_streak": _max_streak(df["win"], False),
        "max_win_streak": _max_streak(df["win"], True),
        "median_hold_min": df["duration_sec"].median() / 60 if df["duration_sec"].notna().any() else None,
    }
    has_mfe = "mfe_r" in df and df["mfe_r"].notna().any()
    if has_mfe:
        lz = losses.dropna(subset=["mfe_r"])
        wz = wins.dropna(subset=["mae_r"])
        s["losers_up_1r_first"] = (lz["mfe_r"] >= 1.0).mean() * 100 if len(lz) else None
        s["losers_up_half_r_first"] = (lz["mfe_r"] >= 0.5).mean() * 100 if len(lz) else None
        s["losers_never_worked"] = (lz["mfe_r"] < 0.2).mean() * 100 if len(lz) else None
        s["winners_near_stop"] = (wz["mae_r"] >= 0.7).mean() * 100 if len(wz) else None
        s["avg_mfe_r_losers"] = lz["mfe_r"].mean() if len(lz) else None
        s["avg_mae_r_winners"] = wz["mae_r"].mean() if len(wz) else None

    groups = {
        "strategy": _group(df, "strategy"),
        "symbol": _group(df, "symbol"),
        "side": _group(df, "side"),
        "exit reason": _group(df, "reason"),
        "hour (UTC)": _group(df, "ctx_hour_utc"),
    }
    return {"df": df, "summary": s, "groups": groups, "insights": _insights(df, s, groups, has_mfe)}


def _insights(df: pd.DataFrame, s: Dict[str, Any], groups: Dict[str, pd.DataFrame],
              has_mfe: bool) -> List[str]:
    out: List[str] = []
    n = s["trades"]
    if n < MIN_SAMPLE:
        out.append(f"⚠️ Only {n} trades - results are mostly luck below ~{MIN_SAMPLE}. "
                   f"Judge settings with `python backtest.py --days 30`, not with one day.")

    # 1) Cost structure
    if s["effective_rr"] == s["effective_rr"] and s["effective_rr"] < 1.0:
        out.append(
            f"❌ Costs dominate: median stop {s['median_stop_pct']:.2f}% vs round-trip costs "
            f"{s['round_trip_cost_pct']:.2f}%. After costs a full target wins "
            f"{s['median_target_pct'] - s['round_trip_cost_pct']:.2f}% but a stop loses "
            f"{s['median_stop_pct'] + s['round_trip_cost_pct']:.2f}% (real RR {s['effective_rr']:.2f}), "
            f"so you need a {s['breakeven_win_rate']:.0f}% win rate just to break even. "
            f"Use wider stops (MIN_STOP_COST_MULTIPLE) or a higher timeframe.")
    elif s["cost_share_of_losses"] > 25:
        out.append(f"⚠️ Fees + slippage are {s['cost_share_of_losses']:.0f}% of all losses. "
                   f"Fewer, larger-target trades would keep more of the edge.")
    if s["gross"] > 0 > s["net"]:
        out.append(f"❌ The signals made money before fees (gross {s['gross']:+.2f} $, already after "
                   f"{s['slippage']:.2f} $ slippage) but {s['fees']:.2f} $ fees turned it into "
                   f"{s['net']:+.2f} $.")
    if s["win_rate"] < s["breakeven_win_rate"] and n >= 10:
        out.append(f"❌ Win rate {s['win_rate']:.0f}% is below the {s['breakeven_win_rate']:.0f}% "
                   f"this stop/target/cost structure needs.")

    # 2) MFE / MAE
    if has_mfe:
        if (s.get("losers_up_1r_first") or 0) >= 20:
            out.append(f"💡 {s['losers_up_1r_first']:.0f}% of losers were +1R in profit before "
                       f"stopping out. Break-even / partial take-profit earlier would save them.")
        elif (s.get("losers_up_half_r_first") or 0) >= 30:
            out.append(f"💡 {s['losers_up_half_r_first']:.0f}% of losers were +0.5R first. "
                       f"Consider BREAKEVEN_TRIGGER_R = 0.75.")
        if (s.get("losers_never_worked") or 0) >= 60 and s["losses"] >= 5:
            out.append(f"❌ {s['losers_never_worked']:.0f}% of losers never got even +0.2R: the entry "
                       f"itself is wrong (late or counter-trend), not the stop. Tighten entry filters.")
        if (s.get("winners_near_stop") or 0) >= 30 and s["wins"] >= 5:
            out.append(f"⚠️ {s['winners_near_stop']:.0f}% of winners came within 0.3R of the stop - "
                       f"stops are close to noise level; a little wider would avoid many losers.")
    else:
        out.append("ℹ️ No MFE/MAE in this file (older export). New trades record it automatically.")

    # 3) Speed of losses
    fast = df[(~df["win"]) & (df["duration_sec"] < 5 * 60)]
    if s["losses"] >= 3 and len(fast) / s["losses"] >= 0.5:
        out.append(f"⚠️ {len(fast)} of {s['losses']} losers died within 5 minutes - stops sit "
                   f"inside normal 5m noise, or entries chase the move.")

    # 4) Breakdowns
    for key in ("strategy", "symbol", "side"):
        g = groups.get(key)
        if g is None or g.empty or len(g) < 2:
            continue
        worst = g.iloc[0]
        if worst["net $"] < 0 and worst["trades"] >= max(3, 0.25 * n):
            out.append(f"🔎 {key} '{g.index[0]}' lost {worst['net $']:.2f} $ over "
                       f"{int(worst['trades'])} trades (win {worst['win %']:.0f}%)."
                       + (" Consider removing it from STRATEGIES." if key == "strategy" else ""))
    side = groups.get("side")
    if side is not None and len(side) == 1 and n >= 4:
        out.append(f"🔎 All {n} trades were {side.index[0]} - one-sided exposure; with correlated "
                   f"coins this is a single bet repeated.")

    # 5) Breaker / liquidation
    reasons = df["reason"].value_counts()
    if reasons.get("DRAWDOWN_BREAKER", 0):
        out.append("❌ The daily drawdown breaker force-closed trades. Entries should stop before "
                   "the limit (CHECK_DAILY_RISK_BUDGET).")
    if reasons.get("LIQUIDATION", 0):
        out.append("❌ Liquidations happened - leverage too high for the stop distance.")
    if s["max_loss_streak"] >= 4:
        out.append(f"⚠️ Longest losing streak: {s['max_loss_streak']}. Make sure "
                   f"MAX_LOSS_PER_TRADE x streak stays well below the daily limit.")
    if s["net"] > 0 and s["profit_factor"] >= 1.3 and n >= MIN_SAMPLE:
        out.append(f"✅ Profitable: PF {s['profit_factor']:.2f}, expectancy {s['expectancy_r']:+.2f}R per trade.")
    return out


def report_text(res: Dict[str, Any]) -> str:
    s = res["summary"]
    if not s.get("trades"):
        return "No trades."
    lines = [
        f"Trades {s['trades']} | win {s['win_rate']:.1f}% ({s['wins']}W/{s['losses']}L) | "
        f"PF {s['profit_factor']:.2f} | net {s['net']:+.2f} $ | gross {s['gross']:+.2f} $ | "
        f"fees {s['fees']:.2f} $ | slippage {s['slippage']:.2f} $",
        f"Expectancy {s['expectancy_usd']:+.2f} $ = {s['expectancy_r']:+.2f}R per trade | "
        f"avg win {s['avg_win']:+.2f} $ | avg loss {s['avg_loss']:+.2f} $ | "
        f"streaks W{s['max_win_streak']}/L{s['max_loss_streak']}",
        f"Structure: stop {s['median_stop_pct']:.2f}% | target {s['median_target_pct']:.2f}% | "
        f"costs {s['round_trip_cost_pct']:.2f}% | real RR {s['effective_rr']:.2f} | "
        f"break-even win rate {s['breakeven_win_rate']:.0f}% | costs = "
        f"{s['cost_share_of_losses']:.0f}% of losses",
    ]
    if s.get("avg_mfe_r_losers") is not None or s.get("avg_mae_r_winners") is not None:
        f = lambda v: "n/a" if v is None else f"{v:.2f}R"
        lines.append(f"MFE/MAE: losers reached {f(s.get('avg_mfe_r_losers'))} on average before stopping, "
                     f"winners dipped {f(s.get('avg_mae_r_winners'))} against them")
    for name, g in res["groups"].items():
        if g is not None and not g.empty and len(g) > 1:
            lines += ["", f"By {name}:", g.to_string()]
    lines += ["", "Insights:"] + [f"  {i}" for i in res["insights"]]
    return "\n".join(lines)
