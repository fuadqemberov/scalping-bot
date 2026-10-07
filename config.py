"""
Central configuration for the autonomous paper-trading scalping bot.

Every tunable parameter lives here so the rest of the code base never
contains "magic numbers". Edit values, then restart the app.
"""

# ---------------------------------------------------------------------------
# Market / data source
# ---------------------------------------------------------------------------
EXCHANGE = "binance"            # Any ccxt exchange id with public OHLCV
SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"
CANDLE_LIMIT = 100              # Candles fetched per iteration

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
RISK_PER_TRADE = 0.01           # 1 % of equity risked per trade (incl. costs)
DAILY_DRAWDOWN_LIMIT = 0.03     # 3 % daily equity loss -> circuit breaker
SUSPENSION_HOURS = 24           # Trading halt duration after a breach
MAX_LEVERAGE = 10.0             # Cap on position notional / equity
QTY_PRECISION = 5               # Decimal places for order quantity (BTC)
MIN_QTY = 0.00001               # Smallest tradable quantity

# ---------------------------------------------------------------------------
# Strategy / indicators
# ---------------------------------------------------------------------------
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
ATR_PERIOD = 14

RSI_OVERSOLD = 35               # LONG: RSI dipped below this, now back above
RSI_OVERBOUGHT = 65             # SHORT: RSI spiked above this, now back below
RSI_LOOKBACK = 3                # Bars to look back for the RSI extreme

VWAP_RESET_DAILY = True         # Anchor VWAP to the UTC session (intraday)

SL_ATR_MULTIPLIER = 1.5         # Stop distance = 1.5 x ATR
RR_RATIO = 1.5                  # Take-profit distance = RR x stop distance

# Close the open position when an opposite signal appears, then (optionally)
# enter in the new direction on the same signal.
EXIT_ON_REVERSAL = True
ENTER_ON_REVERSAL = True

# Skip trades whose gross take-profit would not even cover round-trip
# fees + worst-case slippage (prevents "winning" trades that still lose money
# when ATR is tiny). Note: on quiet 1m BTC markets this filters out MOST
# setups, because ~0.16% round-trip costs often exceed 2.25 x ATR.
SKIP_IF_TP_BELOW_COSTS = False

# ---------------------------------------------------------------------------
# Loop / UI
# ---------------------------------------------------------------------------
LOOP_INTERVAL = 10              # Seconds between bot iterations
MAX_BACKOFF_SECONDS = 60        # Cap for exponential back-off on errors
REQUEST_TIMEOUT_MS = 10000      # ccxt HTTP timeout
UI_REFRESH_MS = 2500            # Dashboard auto-refresh interval
LOG_BUFFER_SIZE = 300           # Log lines kept in memory for the dashboard

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FILE = "logs/bot.log"       # Relative to the project directory
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
