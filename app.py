"""
Streamlit dashboard for the autonomous paper-trading scalping bot.

Run with:  streamlit run app.py

The BotWorker is created once per Python process (st.cache_resource) and its
handle is also stored in st.session_state, so browser reruns, auto-refreshes
and multiple tabs all share the same single background bot.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import config
from bot_worker import BotWorker

st.set_page_config(page_title="Scalping Bot - Paper Trading", page_icon="📈", layout="wide")

# Full-width elements: newer Streamlit uses width="stretch" (use_container_width is deprecated)
_ST_VERSION = tuple(int(p) for p in st.__version__.split(".")[:2] if p.isdigit())
STRETCH = {"width": "stretch"} if _ST_VERSION >= (1, 50) else {"use_container_width": True}

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=config.UI_REFRESH_MS, key="dashboard_refresh")
except ImportError:  # dashboard still works, just without auto refresh
    st.warning("streamlit-autorefresh not installed - run `pip install streamlit-autorefresh` for live updates.")


# ---------------------------------------------------------------------------
# Single bot instance
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Starting trading bot...")
def get_worker() -> BotWorker:
    worker = BotWorker()
    worker.start()
    return worker


if "worker" not in st.session_state:
    st.session_state.worker = get_worker()
worker: BotWorker = st.session_state.worker
if not worker.is_alive():  # thread died unexpectedly (should not happen) -> restart
    worker.start()

snap = worker.state.snapshot()
stats = snap.get("stats") or worker.engine.get_stats()
positions: dict = snap.get("positions") or {}
trades = snap.get("trade_history") or []
symbols: list = snap.get("symbols") or [snap.get("symbol")]
markets: dict = snap.get("markets") or {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_money(v: float | None, signed: bool = False) -> str:
    if v is None:
        return "—"
    return f"{v:+,.2f} $" if signed else f"{v:,.2f} $"


def fmt_dt(v) -> str:
    if v is None:
        return "—"
    return pd.Timestamp(v).strftime("%Y-%m-%d %H:%M:%S UTC")


def fmt_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


STATUS_COLORS = {
    "RUNNING": "#16a34a",
    "PAUSED": "#f59e0b",
    "SUSPENDED": "#dc2626",
    "STARTING": "#3b82f6",
    "STOPPED": "#6b7280",
}


# ---------------------------------------------------------------------------
# Sidebar - controls and parameters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ Controls")
    if worker.is_paused():
        if st.button("▶️ Resume Trading", **STRETCH, type="primary"):
            worker.resume()
            st.rerun()
    else:
        if st.button("⏸️ Pause Trading", **STRETCH):
            worker.pause()
            st.rerun()

    st.markdown("---")
    st.subheader("🚨 Emergency")
    if st.button("🛑 KILL SWITCH", **STRETCH, type="primary",
                 help="Closes any open position at market immediately and pauses the bot."):
        result = worker.kill_switch()
        if result:
            st.toast(f"Position closed: net PnL {result['net_pnl']:+.2f} $", icon="🛑")
        else:
            st.toast("Bot paused (no open position)", icon="🛑")
        st.rerun()

    st.markdown("---")
    st.subheader("📊 Market")
    # Symbol shown on the chart; the bot itself trades all of them.
    default_sym = st.session_state.get("chart_symbol")
    if default_sym not in symbols:
        default_sym = next((s for s in symbols if s in positions), symbols[0])
    selected = st.selectbox("Chart symbol", symbols, index=symbols.index(default_sym), key="chart_symbol")
    st.markdown(
        f"**Scanning:** `{', '.join(symbols)}`  \n"
        f"**Exchange:** `{snap.get('exchange')}`  \n"
        f"**Timeframe:** `{snap.get('timeframe')}`  \n"
        f"**Open positions:** {len(positions)} / {config.MAX_OPEN_POSITIONS}"
    )

    st.subheader("⚙️ Risk Parameters")
    if config.POSITION_SIZING_MODE == "margin":
        sizing_txt = (
            f"- Order size: **{config.MARGIN_PER_TRADE * 100:.0f}%** of balance as margin "
            f"× **{config.LEVERAGE:g}x** = **{config.MARGIN_PER_TRADE * config.LEVERAGE * 100:.0f}%** notional\n"
        )
    else:
        sizing_txt = (
            f"- Risk / trade: **{config.RISK_PER_TRADE * 100:.2f}%** of equity\n"
            f"- Max leverage: **{config.MAX_LEVERAGE:.0f}×**\n"
        )
    st.markdown(
        sizing_txt
        + f"- Daily DD limit: **{config.DAILY_DRAWDOWN_LIMIT * 100:.1f}%** "
        f"(halt {config.SUSPENSION_HOURS}h)\n"
        f"- SL: **{config.SL_ATR_MULTIPLIER} × ATR{config.ATR_PERIOD}**\n"
        f"- TP: **{config.RR_RATIO} R** (min **{config.MIN_TP_COST_MULTIPLE:g}×** round-trip costs)\n"
        f"- Max loss / trade: **{config.MAX_LOSS_PER_TRADE * 100:.2f}%** of balance · "
        f"break-even at **{config.BREAKEVEN_TRIGGER_R:g}R**\n"
        f"- Max same-side positions: **{config.MAX_SAME_SIDE_POSITIONS}** · "
        f"SL cooldown: **{config.SL_COOLDOWN_BARS}** bars\n"
        f"- Fee: **{config.FEE_RATE * 100:.3f}%** / side\n"
        f"- Slippage: **{config.SLIPPAGE_MIN * 100:.2f}–{config.SLIPPAGE_MAX * 100:.2f}%**"
    )
    st.subheader("🧠 Strategy")
    st.markdown(
        f"- EMA {config.EMA_FAST} / {config.EMA_SLOW} + VWAP trend filter\n"
        f"- RSI{config.RSI_PERIOD} pullback in trend: dip < {config.RSI_PULLBACK_LONG} → long, "
        f"spike > {config.RSI_PULLBACK_SHORT} → short (lookback {config.RSI_LOOKBACK})\n"
        f"- Filters: EMA{config.EMA_TREND} trend, ADX ≥ {config.ADX_MIN}, "
        f"volume ≥ {config.VOLUME_MIN_RATIO:g}× avg, "
        f"confirm candle {'on' if config.REQUIRE_CONFIRM_CANDLE else 'off'}\n"
        f"- Loop every **{config.LOOP_INTERVAL}s**"
    )


market = markets.get(selected) or {}
candles: pd.DataFrame | None = market.get("candles")
last_price = market.get("last_price")
position = positions.get(selected)


# ---------------------------------------------------------------------------
# Header + status badge
# ---------------------------------------------------------------------------
status = snap.get("status", "STARTING")
color = STATUS_COLORS.get(status, "#6b7280")
st.title("📈 Autonomous Scalping Bot — Paper Trading")

extra = ""
if status == "SUSPENDED" and snap.get("resume_time"):
    extra = f" &nbsp;·&nbsp; resumes {fmt_dt(snap['resume_time'])}"
price_txt = f"{last_price:,.2f}" if last_price else "—"
st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:8px">
      <span style="background:{color};color:white;padding:6px 16px;border-radius:999px;
                   font-weight:700;letter-spacing:1px">● {status}</span>
      <span style="font-size:1.05rem"><b>{selected}</b> {price_txt}</span>
      <span style="color:#888">Last update: {fmt_dt(snap.get('last_update'))}
        · Iterations: {snap.get('iterations', 0)}{extra}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
if snap.get("last_error"):
    st.error(f"⚠️ {snap['last_error']} — retrying automatically.")


# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Balance", fmt_money(stats["balance"]))
k2.metric("Equity", fmt_money(stats["equity"]), f"{stats['total_return_pct']:+.2f}%")
k3.metric("Unrealized PnL", fmt_money(stats["unrealized_pnl"], signed=True))
k4.metric("Realized PnL", fmt_money(stats["realized_pnl"], signed=True),
          f"fees {stats['total_fees']:.2f} $", delta_color="off")
k5.metric("Win Rate", f"{stats['win_rate']:.1f}%", f"{stats['wins']}W / {stats['losses']}L", delta_color="off")
k6.metric("Total Trades", stats["total_trades"], f"daily {stats['daily_pnl']:+.2f} $")


# ---------------------------------------------------------------------------
# Chart + active position
# ---------------------------------------------------------------------------
chart_col, pos_col = st.columns([3, 1])

with chart_col:
    if candles is None or candles.empty:
        st.info("Waiting for market data...")
    else:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
                            vertical_spacing=0.03)
        x = candles["timestamp"]
        fig.add_trace(go.Candlestick(x=x, open=candles["open"], high=candles["high"],
                                     low=candles["low"], close=candles["close"], name="Price",
                                     increasing_line_color="#26a69a", decreasing_line_color="#ef5350"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=candles["EMA_FAST"], name=f"EMA{config.EMA_FAST}",
                                 line=dict(color="#f59e0b", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=candles["EMA_SLOW"], name=f"EMA{config.EMA_SLOW}",
                                 line=dict(color="#3b82f6", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=candles["VWAP"], name="VWAP",
                                 line=dict(color="#a855f7", width=1.5, dash="dot")), row=1, col=1)

        # Trade markers (only those inside the visible window)
        sym_trades = [t for t in trades if t.get("symbol") == selected]
        if sym_trades:
            tdf = pd.DataFrame(sym_trades)
            t0 = x.min()
            tdf = tdf[pd.to_datetime(tdf["exit_time"], utc=True) >= t0]
            for side, sym, col in (("LONG", "triangle-up", "#16a34a"), ("SHORT", "triangle-down", "#dc2626")):
                sub = tdf[(tdf["side"] == side) & (pd.to_datetime(tdf["entry_time"], utc=True) >= t0)]
                if not sub.empty:
                    fig.add_trace(go.Scatter(x=sub["entry_time"], y=sub["entry_price"], mode="markers",
                                             name=f"{side} entry", marker=dict(symbol=sym, size=13, color=col,
                                             line=dict(width=1, color="white"))), row=1, col=1)
            if not tdf.empty:
                fig.add_trace(go.Scatter(
                    x=tdf["exit_time"], y=tdf["exit_price"], mode="markers", name="Exit",
                    marker=dict(symbol="x", size=11, color=["#16a34a" if p > 0 else "#dc2626" for p in tdf["net_pnl"]]),
                    text=[f"{r} | {p:+.2f}$" for r, p in zip(tdf["exit_reason"], tdf["net_pnl"])],
                    hovertemplate="%{text}<extra>Exit</extra>"), row=1, col=1)

        # Active position: entry marker + SL/TP lines
        if position:
            pc = "#16a34a" if position["side"] == "LONG" else "#dc2626"
            fig.add_trace(go.Scatter(x=[position["entry_time"]], y=[position["entry_price"]], mode="markers",
                                     name="Open entry", marker=dict(symbol="star", size=15, color=pc)), row=1, col=1)
            fig.add_hline(y=position["sl"], line=dict(color="#dc2626", dash="dash"),
                          annotation_text="SL", row=1, col=1)
            fig.add_hline(y=position["tp"], line=dict(color="#16a34a", dash="dash"),
                          annotation_text="TP", row=1, col=1)
            fig.add_hline(y=position["entry_price"], line=dict(color="#9ca3af", dash="dot"),
                          annotation_text="Entry", row=1, col=1)

        # RSI panel
        fig.add_trace(go.Scatter(x=x, y=candles["RSI"], name=f"RSI{config.RSI_PERIOD}",
                                 line=dict(color="#06b6d4", width=1.3)), row=2, col=1)
        for lvl in (config.RSI_PULLBACK_LONG, config.RSI_PULLBACK_SHORT):
            fig.add_hline(y=lvl, line=dict(color="#6b7280", dash="dot", width=1), row=2, col=1)

        fig.update_layout(height=620, margin=dict(l=10, r=10, t=30, b=10), template="plotly_dark",
                          xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.04, x=0),
                          uirevision="keep")  # preserve zoom across refreshes
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        st.plotly_chart(fig, **STRETCH, key="main_chart")

        last = candles.iloc[-1]
        st.caption(
            f"EMA{config.EMA_FAST} {last['EMA_FAST']:.6g} · EMA{config.EMA_SLOW} {last['EMA_SLOW']:.6g} · "
            f"VWAP {last['VWAP']:.6g} · RSI {last['RSI']:.1f} · ATR {last['ATR']:.4g} · "
            f"Last {selected} signal: {market.get('last_signal') or '—'} ({fmt_dt(market.get('last_signal_time'))})"
        )
        diag = market.get("diagnostics")
        if diag and diag.get("reason"):
            st.caption(f"🔎 Last closed candle: **{diag.get('signal') or 'no signal'}** — {diag['reason']}")

    # Overview of every scanned symbol
    rows = []
    for sym in symbols:
        m = markets.get(sym) or {}
        d = m.get("diagnostics") or {}
        rows.append({
            "Symbol": sym,
            "Price": m.get("last_price"),
            "Trend": d.get("trend") or "—",
            "RSI": round(d["rsi"], 1) if d.get("rsi") is not None else None,
            "Status": d.get("reason") or (m.get("error") or "waiting for data"),
            "Last signal": f"{m.get('last_signal')} {pd.Timestamp(m['last_signal_time']).strftime('%H:%M')}"
                           if m.get("last_signal") else "—",
            "Position": positions[sym]["side"] if sym in positions else "",
        })
    st.dataframe(pd.DataFrame(rows), **STRETCH, hide_index=True)

with pos_col:
    st.subheader("🎯 Active Positions")
    if not positions:
        st.info("No open position — scanning for signals.")
    for sym, position in positions.items():
        cur = (markets.get(sym) or {}).get("last_price") or position.get("current_price") or position["entry_price"]
        direction = 1 if position["side"] == "LONG" else -1
        upnl = (cur - position["entry_price"]) * position["qty"] * direction
        upnl_pct = (cur / position["entry_price"] - 1) * 100 * direction
        side_color = "#16a34a" if position["side"] == "LONG" else "#dc2626"
        pnl_color = "#16a34a" if upnl >= 0 else "#dc2626"
        duration = (datetime.now(timezone.utc) - pd.Timestamp(position["entry_time"]).to_pydatetime()).total_seconds()
        st.markdown(
            f"""
            <div style="border:1px solid #333;border-radius:12px;padding:14px;line-height:1.9">
              <span style="background:{side_color};color:white;padding:3px 12px;border-radius:6px;
                           font-weight:700">{position['side']}</span>
              &nbsp;<b>{position['qty']:g}</b> {position['symbol'].split('/')[0]}<br>
              Entry: <b>{position['entry_price']:,.6g}</b><br>
              Current: <b>{cur:,.6g}</b><br>
              PnL: <b style="color:{pnl_color}">{upnl:+,.2f} &#36; ({upnl_pct:+.3f}%)</b><br>
              <span style="color:#dc2626">SL: {position['sl']:,.6g}</span><br>
              <span style="color:#16a34a">TP: {position['tp']:,.6g}</span><br>
              Notional: {position['entry_price'] * position['qty']:,.2f} &#36;<br>
              {f"Leverage: <b>{position['leverage']:g}x</b> · Margin: {position['margin']:,.2f} &#36;<br>"
               f"ROE: <b style='color:{pnl_color}'>{upnl / position['margin'] * 100:+.2f}%</b><br>"
               f"<span style='color:#f97316'>Liq: {position['liq_price']:,.6g}</span><br>"
               if position.get('margin') else ""}
              Entry fee: {position['entry_fee']:.4f} &#36;<br>
              Duration: <b>{fmt_duration(duration)}</b>
            </div><div style="height:8px"></div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("📉 Risk")
    st.markdown(
        # "$" is escaped because Streamlit markdown treats $...$ as LaTeX
        f"Daily PnL: **{stats['daily_pnl']:+.2f} \\$**  \n"
        f"Daily DD limit: **-{stats['day_start_equity'] * config.DAILY_DRAWDOWN_LIMIT:.2f} \\$**  \n"
        f"Max drawdown: **{stats['max_drawdown_pct']:.2f}%**  \n"
        f"Profit factor: **{stats['profit_factor']:.2f}**  \n"
        f"Slippage paid: **{stats['total_slippage']:.2f} \\$**"
    )


# ---------------------------------------------------------------------------
# Trade history
# ---------------------------------------------------------------------------
st.subheader("📜 Trade History")
if not trades:
    st.caption("No completed trades yet.")
else:
    hist = pd.DataFrame(trades).sort_values("exit_time", ascending=False)
    view = pd.DataFrame({
        "#": hist["id"],
        "Symbol": hist["symbol"],
        "Side": hist["side"],
        "Qty": hist["qty"],
        "Entry Time": pd.to_datetime(hist["entry_time"], utc=True).dt.strftime("%m-%d %H:%M:%S"),
        "Exit Time": pd.to_datetime(hist["exit_time"], utc=True).dt.strftime("%m-%d %H:%M:%S"),
        "Duration": hist["duration_sec"].apply(fmt_duration),
        "Entry": hist["entry_price"].map(lambda v: float(f"{v:.6g}")),
        "Exit": hist["exit_price"].map(lambda v: float(f"{v:.6g}")),
        "SL": hist["sl"].map(lambda v: float(f"{v:.6g}")),
        "TP": hist["tp"].map(lambda v: float(f"{v:.6g}")),
        "Gross PnL": hist["gross_pnl"].round(4),
        "Fees": hist["total_fees"].round(4),
        "Slippage $": hist["slippage_cost"].round(4),
        "Net PnL": hist["net_pnl"].round(4),
        "PnL %": hist["pnl_pct"].round(3),
        "Reason": hist["exit_reason"],
        "Balance": hist["balance_after"].round(2),
    })
    st.dataframe(view, **STRETCH, height=320, hide_index=True)
    st.download_button("⬇️ Download CSV", view.to_csv(index=False).encode(), "trade_history.csv", "text/csv")


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------
with st.expander("🪵 Bot Log (latest 100 lines)", expanded=False):
    logs = snap.get("logs") or []
    st.code("\n".join(reversed(logs[-100:])) or "No log entries yet.", language="text")

st.caption("Paper trading only — no real orders are ever sent. Public market data via ccxt. All times UTC.")
