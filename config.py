"""
Central configuration for the autonomous paper-trading scalping bot.

Every tunable parameter lives here so the rest of the code base never
contains "magic numbers". Edit values, then restart the app.
"""

# ---------------------------------------------------------------------------
# Market / data source
# ---------------------------------------------------------------------------
EXCHANGE = "binance"            # Any ccxt exchange id with public OHLCV
# Symbols scanned every iteration. Each one is evaluated independently and
# can hold its own position (capped by MAX_OPEN_POSITIONS below).
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
SYMBOL = SYMBOLS[0]             # Backwards-compatible alias (primary symbol)

# 1m is the shortest *sensible* timeframe for this strategy. Do NOT use "1s":
#   * only Binance spot offers 1s klines - the OKX/KuCoin fallbacks do not
#     (the bot refuses exchanges that don't support the timeframe);
#   * the loop runs every LOOP_INTERVAL (10s), so it would skip most 1s bars;
#   * a 1s ATR is a few dollars on BTC, i.e. far below the ~0.1-0.16%
#     round-trip fees+slippage - every trade would lose money by design;
#   * RSI14/EMA21 on 1s bars is mostly noise; VWAP over 300 bars = 5 minutes.
# If anything, moving UP to "3m"/"5m" makes the TP clear costs more often.
#
# Trade log analysis (58 trades on 1m): the median stop was 0.11 % and the
# median target 0.11 % of price, while fees + slippage cost 0.14 % per round
# trip. Gross PnL was +1 USDT, fees+slippage were -80 USDT: the edge was
# eaten entirely by costs. On 5m bars ATR is ~2-3x larger, so the same cost
# becomes a much smaller fraction of each target.
TIMEFRAME = "5m"
CANDLE_LIMIT = 300              # Candles fetched per iteration (OKX max is 300)
                                # EMA_TREND=200 needs >= 200 bars, keep this >= 300

# Binance blocks some regions (HTTP 451). If the primary exchange is
# unreachable at start-up, these public-data fallbacks are tried in order.
# Set to [] to disable the fallback behaviour entirely.
FALLBACK_EXCHANGES = ["okx", "kucoin", "binanceus"]

# Only act on fully closed candles (the last OHLCV row is still forming).
USE_CLOSED_CANDLES_ONLY = True

# ---------------------------------------------------------------------------
# Paper account
# ---------------------------------------------------------------------------
INITIAL_BALANCE = 2000.0        # USDT

# ---------------------------------------------------------------------------
# Execution simulation
# ---------------------------------------------------------------------------
SLIPPAGE_MIN = 0.0001           # 0.01 %
SLIPPAGE_MAX = 0.0003           # 0.03 %
FEE_RATE = 0.0005               # 0.05 % taker fee, charged on entry AND exit

# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------
# Position sizing mode:
#   "margin" : every order uses MARGIN_PER_TRADE of the wallet balance as
#              margin at LEVERAGE  -> notional = balance * 7% * 10 = 70 %
#              of the balance per position.
#   "risk"   : size so that a stop-out loses RISK_PER_TRADE of equity.
POSITION_SIZING_MODE = "margin"
MARGIN_PER_TRADE = 0.07         # 7 % of balance used as margin per order
# 100x -> margin 7 % x 100 = up to 700 % of balance notional per position.
# Leverage scales wins AND losses (and fees) by the same factor, so it is
# paired with MAX_LOSS_PER_TRADE below, which shrinks the order so that one
# stop-out still costs at most 1 % of the balance. At 100x the liquidation
# price is only ~0.47 % away: any setup whose stop is farther than that is
# skipped, and a fast wick can liquidate a position before its stop fills.
# Note: real exchanges cap leverage per symbol (e.g. Binance: BTC 125x, many
# altcoins 50-75x); this paper engine does not enforce those caps.
LEVERAGE = 100.0
# Hard cap: a stop-out (incl. worst-case fees + slippage) may lose at most
# this fraction of the balance. At 100x this cap - not the leverage - sets
# the position size (e.g. stop 0.37 % -> ~3,800 $ notional on 2,000 $).
# Set 0 to disable.
MAX_LOSS_PER_TRADE = 0.01
# Isolated-margin liquidation model: the position is liquidated when its loss
# eats the margin down to the maintenance level (~-0.47 % price move at 100x).
MAINTENANCE_MARGIN_RATE = 0.005  # 0.5 % of notional

RISK_PER_TRADE = 0.01           # Used only in "risk" mode
DAILY_DRAWDOWN_LIMIT = 0.03     # 3 % daily equity loss -> circuit breaker
SUSPENSION_HOURS = 24           # Trading halt duration after a breach
MAX_LEVERAGE = 100.0             # Cap on position notional / equity ("risk" mode)
MAX_OPEN_POSITIONS = 2          # Max simultaneous positions across all symbols
# BTC/ETH/SOL/BNB/XRP move together on short timeframes, so 3 longs at once
# are effectively one 3x-sized bet (the log shows 3 longs stopped out
# together at 17:19-17:24 and pairs at 01:35 and 05:01). Allow at most this
# many open positions in the same direction.
MAX_SAME_SIDE_POSITIONS = 1
# After a stop-loss on a symbol, ignore its signals for this many bars
# (stops BNB-style whipsaw: short stopped, long, short stopped again...).
SL_COOLDOWN_BARS = 3
# Fallbacks only - the real amount step / minimum come from the exchange's
# market metadata for each symbol.
QTY_PRECISION = 5               # Decimal places for order quantity
MIN_QTY = 0.00001               # Smallest tradable quantity

# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------
# Strategies evaluated on every closed candle, in priority order: the first
# one that fires on a symbol opens the trade. Remove a name to switch it off.
#   "trend_pullback"   : buy the dip / sell the rally inside a trend
#                        (EMA9/21 + VWAP + RSI pullback, EMA200/ADX/volume filters)
#   "squeeze_breakout" : volatility squeeze (Bollinger Bands inside Keltner
#                        Channel) releasing with momentum + volume - catches the
#                        start of a move after a quiet period
#   "bb_reversion"     : range-market mean reversion - price stretched outside
#                        the Bollinger Band with extreme RSI snaps back inside;
#                        only when ADX shows NO trend
# Compare them on real data first:  python backtest.py --compare-strategies
STRATEGIES = ["trend_pullback", "squeeze_breakout", "bb_reversion"]

# ---------------------------------------------------------------------------
# Strategy / indicators
# ---------------------------------------------------------------------------
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
ATR_PERIOD = 14

# Pullback-in-trend entry. The old 35/65 levels contradicted the trend
# filter: in a 1m uptrend (close > VWAP, EMA9 > EMA21) RSI almost never
# drops below 35, so LONG/SHORT fired roughly once every ~100 hours.
#   LONG : uptrend + RSI dipped below RSI_PULLBACK_LONG in the last
#          RSI_LOOKBACK bars and has now turned back up above it.
#   SHORT: downtrend + RSI rose above RSI_PULLBACK_SHORT and turned back down.
# 45/55 gives roughly 1 signal per symbol per hour on 1m data. Lower values
# (40/60) = fewer, deeper pullbacks; 50/50 = many more, noisier signals.
RSI_PULLBACK_LONG = 45
RSI_PULLBACK_SHORT = 55
RSI_LOOKBACK = 3                # Bars to look back for the pullback
# Old names kept so older code/dashboards keep working
RSI_OVERSOLD = RSI_PULLBACK_LONG
RSI_OVERBOUGHT = RSI_PULLBACK_SHORT

VWAP_RESET_DAILY = True         # Anchor VWAP to the UTC session (intraday)

# Extra signal confirmations (each can be switched off with 0 / False).
# They run AFTER the pullback rule above and only remove signals.
EMA_TREND = 200                 # Long-term trend: LONG only above a rising EMA200,
                                # SHORT only below a falling one (~17h on 5m). 0 = off
EMA_TREND_SLOPE_BARS = 10       # Bars used to measure the EMA200 slope
ADX_PERIOD = 14
ADX_MIN = 20                    # Skip choppy, trendless markets (ADX < 20). 0 = off
VOLUME_SMA_PERIOD = 20
VOLUME_MIN_RATIO = 1.0          # Signal bar volume >= 1.0 x 20-bar average. 0 = off
REQUIRE_CONFIRM_CANDLE = True   # LONG needs a green signal bar, SHORT a red one
# The filters above apply to "trend_pullback". The other strategies have
# their own rules below (EMA200 trend direction is also used by the squeeze).

# Bollinger Bands / Keltner Channel (shared by squeeze_breakout + bb_reversion)
BB_PERIOD = 20
BB_STD = 2.0
KC_PERIOD = 20
KC_ATR_MULT = 1.5

# squeeze_breakout
SQZ_MIN_BARS = 6                # Squeeze must have lasted >= 6 bars ...
SQZ_LOOKBACK = 3                # ... and released within the last 3 bars
SQZ_VOLUME_RATIO = 1.2          # Breakout bar volume >= 1.2 x 20-bar average
SQZ_USE_TREND_FILTER = True     # Only break out in the EMA200 direction
SQZ_SL_ATR = 1.5                # Stop = 1.5 x ATR
SQZ_RR = 2.0                    # Target = 2 x stop

# bb_reversion
BBR_ADX_MAX = 20                # Only trade ranges: ADX below this
BBR_RSI_LONG = 30               # Previous bar closed below the lower band with RSI < 30
BBR_RSI_SHORT = 70              # ... or above the upper band with RSI > 70
BBR_SL_ATR_BUFFER = 0.5         # Stop = swing extreme -/+ 0.5 x ATR
BBR_MIN_RR = 1.2                # Target is the middle band; skip if reward < 1.2 x risk

SL_ATR_MULTIPLIER = 1.5         # Stop distance = 1.5 x ATR
RR_RATIO = 2.0                  # Take-profit distance = RR x stop distance

# Break-even stop: once price has moved BREAKEVEN_TRIGGER_R x the initial
# stop distance in our favour, the stop is moved to entry + round-trip costs,
# so a trade that was already +1R can no longer turn into a full loss.
# Trade-off: some trades that would have dipped back and then hit the 2R
# target are closed at ~0 instead. Set 0 to disable.
BREAKEVEN_TRIGGER_R = 1.0

# Close the open position when an opposite signal appears, then (optionally)
# enter in the new direction on the same signal.
EXIT_ON_REVERSAL = True
ENTER_ON_REVERSAL = True

# Skip trades whose take-profit distance is smaller than this multiple of
# the worst-case round-trip cost (2 x (fee + max slippage) = 0.16 %).
# With 2.0 the target must be >= 0.32 % of price, so costs eat at most half
# of a winner. In the 1m log 20 of 36 TAKE_PROFIT trades still lost money
# because the target was smaller than the costs. Set 0 to disable.
MIN_TP_COST_MULTIPLE = 2.0

# ---------------------------------------------------------------------------
# Loop / UI
# ---------------------------------------------------------------------------
LOOP_INTERVAL = 10              # Seconds between bot iterations
# The loop also wakes up this many seconds after every candle close, so
# signals are evaluated right away instead of up to LOOP_INTERVAL later.
CANDLE_CLOSE_DELAY = 2.0

# Real-time price stream (ccxt WebSocket). Every tick is checked against
# SL / TP / break-even / liquidation instead of once per LOOP_INTERVAL, so
# stops fill near the stop price. The REST ticker check in the main loop
# keeps running as a backup if the stream disconnects.
USE_WEBSOCKET = True
WS_RECONNECT_SECONDS = 5        # Wait before reconnecting after a stream error
WS_STALE_SECONDS = 30           # Dashboard shows the stream as stale after this
MAX_BACKOFF_SECONDS = 60        # Cap for exponential back-off on errors
REQUEST_TIMEOUT_MS = 10000      # ccxt HTTP timeout
UI_REFRESH_MS = 2500            # Dashboard auto-refresh interval
LOG_BUFFER_SIZE = 500           # Log lines kept in memory for the dashboard
# Log one line per symbol per closed candle explaining why there was / wasn't
# a signal (trend state, RSI, what is still missing).
LOG_SIGNAL_DIAGNOSTICS = True

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FILE = "logs/bot.log"       # Relative to the project directory
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
