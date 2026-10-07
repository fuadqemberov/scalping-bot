"""
BotWorker - the autonomous trading loop running in a daemon thread.

Each iteration (every LOOP_INTERVAL seconds):
    1. Fetch the latest ticker price -> mark-to-market + SL/TP check.
    2. Daily drawdown circuit breaker -> flatten + suspend 24h on breach.
    3. Fetch OHLCV candles, compute indicators, publish to SharedState.
    4. On each newly closed candle evaluate the strategy signal:
         - position open  : exit on an opposite signal (optionally reverse)
         - no position    : size the trade and open it
    Pause stops *new entries* only; an open position keeps being protected
    by its SL/TP. The kill switch flattens the position and pauses.

Run headless (without the dashboard):  python bot_worker.py
"""

from __future__ import annotations

import collections
import copy
import logging
import os
import threading
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Deque, Dict, Optional

import ccxt
import pandas as pd

import config
import strategy
from paper_engine import PaperEngine
from risk_manager import RiskManager

logger = logging.getLogger("bot")

STATUS_STARTING = "STARTING"
STATUS_RUNNING = "RUNNING"
STATUS_PAUSED = "PAUSED"
STATUS_SUSPENDED = "SUSPENDED"
STATUS_STOPPED = "STOPPED"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
class SharedStateLogHandler(logging.Handler):
    """Mirrors log records into SharedState so the dashboard can display them."""

    def __init__(self, shared_state: "SharedState") -> None:
        super().__init__()
        self.shared_state = shared_state

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.shared_state.add_log(self.format(record))
        except Exception:  # never let logging break the bot
            self.handleError(record)


_logging_configured = False
_logging_lock = threading.Lock()


def setup_logging() -> None:
    """Configure console + rotating file logging exactly once per process."""
    global _logging_configured
    with _logging_lock:
        if _logging_configured:
            return
        fmt = logging.Formatter(config.LOG_FORMAT, config.LOG_DATE_FORMAT)
        root = logging.getLogger()
        root.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))

        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)

        try:
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.LOG_FILE)
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except OSError as exc:
            root.warning("File logging disabled: %s", exc)

        # ccxt / urllib3 are chatty at DEBUG level
        logging.getLogger("ccxt").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        _logging_configured = True


# ---------------------------------------------------------------------------
# Shared state between the worker thread and the UI
# ---------------------------------------------------------------------------
class SharedState:
    """Thread-safe container for everything the dashboard needs to render."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {
            "status": STATUS_STARTING,
            "paused": False,
            "exchange": config.EXCHANGE,
            "symbol": config.SYMBOL,
            "timeframe": config.TIMEFRAME,
            "last_price": None,
            "last_price_time": None,
            "candles": None,            # DataFrame with indicators (incl. forming candle)
            "last_signal": None,
            "last_signal_time": None,
            "last_candle_time": None,
            "last_update": None,
            "last_error": None,
            "iterations": 0,
            "consecutive_errors": 0,
            "resume_time": None,
            "position": None,
            "stats": None,
            "trade_history": [],
        }
        self._logs: Deque[str] = collections.deque(maxlen=config.LOG_BUFFER_SIZE)

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            self._data.update(kwargs)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def snapshot(self) -> Dict[str, Any]:
        """Deep-ish copy safe to use outside the lock."""
        with self._lock:
            snap = {}
            for k, v in self._data.items():
                if isinstance(v, pd.DataFrame):
                    snap[k] = v.copy()
                else:
                    snap[k] = copy.deepcopy(v)
            snap["logs"] = list(self._logs)
            return snap

    def add_log(self, line: str) -> None:
        with self._lock:
            self._logs.append(line)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
class BotWorker:
    def __init__(
        self,
        shared_state: Optional[SharedState] = None,
        engine: Optional[PaperEngine] = None,
        risk_manager: Optional[RiskManager] = None,
    ) -> None:
        setup_logging()
        self.state = shared_state or SharedState()
        self.engine = engine or PaperEngine()
        self.risk = risk_manager or RiskManager()
        self.symbol = config.SYMBOL

        # Attach the UI log mirror once
        handler = SharedStateLogHandler(self.state)
        handler.setFormatter(logging.Formatter(config.LOG_FORMAT, config.LOG_DATE_FORMAT))
        logging.getLogger().addHandler(handler)
        self._ui_log_handler = handler

        self.exchange: Optional[ccxt.Exchange] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._paused = threading.Event()        # set => paused
        self._trade_lock = threading.Lock()     # serialises trading actions (loop vs. kill switch)
        self._last_processed_candle_ts: Optional[pd.Timestamp] = None
        self._consecutive_errors = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.info("BotWorker already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="BotWorker", daemon=True)
        self._thread.start()
        logger.info("BotWorker thread started")

    def stop(self, timeout: float = 15.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self.state.update(status=STATUS_STOPPED)
        logger.info("BotWorker stopped")

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def pause(self) -> None:
        self._paused.set()
        self.state.update(paused=True)
        self._publish_status()
        logger.info("Bot PAUSED - no new entries (open position still protected by SL/TP)")

    def resume(self) -> None:
        self._paused.clear()
        self.state.update(paused=False)
        self._publish_status()
        logger.info("Bot RESUMED")

    def is_paused(self) -> bool:
        return self._paused.is_set()

    def kill_switch(self) -> Optional[Dict[str, Any]]:
        """Emergency stop: close any open position at market and pause the bot."""
        logger.warning("EMERGENCY KILL SWITCH activated")
        self.pause()
        trade = None
        try:
            with self._trade_lock:
                if self.engine.has_position():
                    price = None
                    try:
                        price = self._fetch_price()
                    except Exception as exc:
                        logger.error("Kill switch: live price unavailable (%s), using last known", exc)
                    price = price or self.engine.last_price or self.state.get("last_price")
                    if price:
                        trade = self.engine.close_position(price, "KILL_SWITCH")
                    else:
                        logger.error("Kill switch: no price available - position NOT closed")
        except Exception:
            logger.exception("Kill switch failed")
        self._publish_account()
        return trade

    # ------------------------------------------------------------------
    # Exchange helpers
    # ------------------------------------------------------------------
    def _connect(self) -> bool:
        """Create a ccxt client, falling back to alternate public exchanges."""
        candidates = [config.EXCHANGE] + [e for e in config.FALLBACK_EXCHANGES if e != config.EXCHANGE]
        for ex_id in candidates:
            try:
                ex_class = getattr(ccxt, ex_id)
                ex = ex_class({"enableRateLimit": True, "timeout": config.REQUEST_TIMEOUT_MS})
                ex.load_markets()
                if self.symbol not in ex.markets:
                    logger.warning("%s does not list %s - skipping", ex_id, self.symbol)
                    continue
                self.exchange = ex
                self.state.update(exchange=ex_id)
                if ex_id != config.EXCHANGE:
                    logger.warning("Primary exchange '%s' unavailable - using fallback '%s'",
                                   config.EXCHANGE, ex_id)
                logger.info("Connected to %s public API (%s %s)", ex_id, self.symbol, config.TIMEFRAME)
                return True
            except AttributeError:
                logger.error("Unknown ccxt exchange id '%s'", ex_id)
            except Exception as exc:
                logger.error("Could not connect to %s: %s: %s", ex_id, type(exc).__name__, str(exc)[:200])
        return False

    def _fetch_price(self) -> Optional[float]:
        ticker = self.exchange.fetch_ticker(self.symbol)
        price = ticker.get("last") or ticker.get("close")
        if price is None and ticker.get("bid") and ticker.get("ask"):
            price = (ticker["bid"] + ticker["ask"]) / 2
        return float(price) if price else None

    def _fetch_candles(self) -> pd.DataFrame:
        raw = self.exchange.fetch_ohlcv(self.symbol, config.TIMEFRAME, limit=config.CANDLE_LIMIT)
        if not raw:
            return pd.DataFrame()
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        return df.dropna(subset=["open", "high", "low", "close"])

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _run(self) -> None:
        logger.info("Autonomous loop starting (interval %ss)", config.LOOP_INTERVAL)
        self._publish_status()
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                if self.exchange is None and not self._connect():
                    raise ConnectionError("No exchange reachable")
                self._iteration()
                self._consecutive_errors = 0
                self.state.update(last_error=None)
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as exc:
                self._handle_error(f"Network error: {type(exc).__name__}: {str(exc)[:200]}")
            except ccxt.ExchangeError as exc:
                self._handle_error(f"Exchange error: {str(exc)[:200]}")
            except ConnectionError as exc:
                self._handle_error(str(exc))
            except Exception as exc:  # catch-all: the loop must never die
                logger.exception("Unexpected error in iteration")
                self._handle_error(f"Unexpected error: {type(exc).__name__}: {exc}")
            finally:
                self.state.update(
                    iterations=self.state.get("iterations", 0) + 1,
                    consecutive_errors=self._consecutive_errors,
                    last_update=datetime.now(timezone.utc),
                )
                self._publish_status()

            # Exponential back-off on consecutive errors, otherwise normal cadence
            if self._consecutive_errors:
                delay = min(config.LOOP_INTERVAL * (2 ** (self._consecutive_errors - 1)),
                            config.MAX_BACKOFF_SECONDS)
            else:
                delay = max(0.0, config.LOOP_INTERVAL - (time.monotonic() - started))
            self._stop_event.wait(delay)
        logger.info("Autonomous loop exited")

    def _handle_error(self, msg: str) -> None:
        self._consecutive_errors += 1
        logger.error("%s (consecutive errors: %d)", msg, self._consecutive_errors)
        self.state.update(last_error=msg)
        # Force a reconnect after repeated failures
        if self._consecutive_errors >= 5:
            self.exchange = None

    def _iteration(self) -> None:
        # 1) Latest tick -> mark-to-market and SL/TP protection --------------
        price = self._fetch_price()
        if price is None or price <= 0:
            raise ccxt.ExchangeError("Ticker returned no valid price")
        self.state.update(last_price=price, last_price_time=datetime.now(timezone.utc))

        with self._trade_lock:
            closed = self.engine.check_sl_tp(price)
            if closed:
                logger.info("Exit by %s: net PnL %.4f USDT", closed["exit_reason"], closed["net_pnl"])

        # 2) Daily drawdown circuit breaker ----------------------------------
        if self.risk.check_daily_drawdown(self.engine) or self.risk.should_suspend():
            with self._trade_lock:
                if self.engine.has_position():
                    self.engine.close_position(price, "DRAWDOWN_BREAKER")
        suspended = self.risk.should_suspend()
        self.state.update(resume_time=self.risk.resume_time)

        # 3) Candles + indicators --------------------------------------------
        candles = self._fetch_candles()
        min_bars = max(config.EMA_SLOW, config.RSI_PERIOD + 1, config.ATR_PERIOD) + config.RSI_LOOKBACK + 2
        if candles.empty or len(candles) < min_bars:
            logger.warning("Not enough candle data (%d rows, need %d)", len(candles), min_bars)
            self._publish_account()
            return

        ind = strategy.compute_indicators(candles)
        self.state.update(candles=ind)

        # The last row is the still-forming candle - evaluate signals on closed bars
        closed_df = ind.iloc[:-1] if config.USE_CLOSED_CANDLES_ONLY else ind
        if closed_df.empty:
            self._publish_account()
            return
        candle_ts = closed_df["timestamp"].iloc[-1]
        self.state.update(last_candle_time=candle_ts.to_pydatetime())

        # 4) Signal handling - once per newly closed candle -------------------
        if candle_ts == self._last_processed_candle_ts:
            self._publish_account()
            return
        self._last_processed_candle_ts = candle_ts

        signal = strategy.generate_signal(closed_df)
        if signal:
            logger.info("Signal %s on candle %s (close %.2f, RSI %.1f)", signal,
                        candle_ts.strftime("%H:%M"), closed_df["close"].iloc[-1], closed_df["RSI"].iloc[-1])
            self.state.update(last_signal=signal, last_signal_time=datetime.now(timezone.utc))

        with self._trade_lock:
            position = self.engine.get_position()

            # Reversal exit
            if position and signal and signal != position["side"] and config.EXIT_ON_REVERSAL:
                self.engine.close_position(price, "REVERSAL")
                position = None
                if not config.ENTER_ON_REVERSAL:
                    signal = None

            # New entry
            if position is None and signal:
                if suspended:
                    logger.info("Signal %s ignored - trading suspended until %s", signal, self.risk.resume_time)
                elif self._paused.is_set():
                    logger.info("Signal %s ignored - bot paused", signal)
                else:
                    self._enter(signal, price, float(closed_df["ATR"].iloc[-1]))

        self._publish_account()

    def _enter(self, side: str, price: float, atr_value: float) -> None:
        try:
            sl, tp = strategy.calculate_sl_tp(side, price, atr_value)
        except ValueError as exc:
            logger.warning("Entry skipped: %s", exc)
            return

        equity = self.engine.get_equity()
        qty = self.risk.calculate_position_size(equity, price, sl)
        if qty <= 0:
            logger.warning("Entry skipped: position size is zero (equity %.2f, stop %.2f)", equity, abs(price - sl))
            return

        if config.SKIP_IF_TP_BELOW_COSTS:
            reward = abs(tp - price) * qty
            costs = self.risk.estimate_round_trip_cost(price, qty)
            if reward <= costs:
                logger.info("Entry skipped: TP reward %.4f <= round-trip costs %.4f (ATR too small)", reward, costs)
                return

        self.engine.open_position(side, price, qty, sl, tp)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------
    def _publish_account(self) -> None:
        try:
            self.state.update(
                position=self.engine.get_position(),
                stats=self.engine.get_stats(),
                trade_history=self.engine.get_trade_history(),
            )
        except Exception:
            logger.exception("Failed to publish account state")

    def _publish_status(self) -> None:
        if self._stop_event.is_set():
            status = STATUS_STOPPED
        elif self.risk.should_suspend():
            status = STATUS_SUSPENDED
        elif self._paused.is_set():
            status = STATUS_PAUSED
        else:
            status = STATUS_RUNNING
        self.state.update(status=status, paused=self._paused.is_set(), resume_time=self.risk.resume_time)


# ---------------------------------------------------------------------------
# Headless entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    worker = BotWorker()
    worker.start()
    try:
        while True:
            time.sleep(60)
            s = worker.engine.get_stats()
            logger.info("Equity %.2f | trades %d | win rate %.1f%%",
                        s["equity"], s["total_trades"], s["win_rate"])
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        worker.stop()
