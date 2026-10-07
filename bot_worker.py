"""
BotWorker - the autonomous trading loop running in a daemon thread.

Each iteration (every LOOP_INTERVAL seconds), for every symbol in
config.SYMBOLS:
    1. Fetch the latest ticker prices -> mark-to-market + SL/TP check.
    2. Daily drawdown circuit breaker -> flatten everything + suspend 24h.
    3. Fetch OHLCV candles, compute indicators, publish to SharedState.
    4. On each newly closed candle evaluate the strategy signal:
         - position open  : exit on an opposite signal (optionally reverse)
         - no position    : size the trade and open it (if MAX_OPEN_POSITIONS
                            is not reached)
    Pause stops *new entries* only; open positions keep being protected
    by their SL/TP. The kill switch flattens all positions and pauses.

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
from typing import Any, Deque, Dict, List, Optional, Tuple

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


def configured_symbols() -> List[str]:
    """Symbols to trade: config.SYMBOLS, falling back to the legacy SYMBOL."""
    symbols = list(getattr(config, "SYMBOLS", None) or [config.SYMBOL])
    seen, out = set(), []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


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
def _empty_market() -> Dict[str, Any]:
    return {
        "last_price": None,
        "last_price_time": None,
        "candles": None,            # DataFrame with indicators (incl. forming candle)
        "last_signal": None,
        "last_signal_time": None,
        "last_candle_time": None,
        "diagnostics": None,        # strategy.evaluate_signal() of the last closed candle
        "error": None,
    }


class SharedState:
    """Thread-safe container for everything the dashboard needs to render."""

    def __init__(self, symbols: Optional[List[str]] = None) -> None:
        self._lock = threading.Lock()
        symbols = symbols or configured_symbols()
        self._data: Dict[str, Any] = {
            "status": STATUS_STARTING,
            "paused": False,
            "exchange": config.EXCHANGE,
            "symbols": list(symbols),
            "symbol": symbols[0],       # primary symbol (backwards compatible)
            "timeframe": config.TIMEFRAME,
            "markets": {s: _empty_market() for s in symbols},
            "last_signal": None,        # most recent signal on any symbol
            "last_signal_symbol": None,
            "last_signal_time": None,
            "last_update": None,
            "last_error": None,
            "iterations": 0,
            "consecutive_errors": 0,
            "resume_time": None,
            "positions": {},            # symbol -> position dict
            "stats": None,
            "trade_history": [],
        }
        self._logs: Deque[str] = collections.deque(maxlen=config.LOG_BUFFER_SIZE)

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            self._data.update(kwargs)

    def update_market(self, symbol: str, **kwargs: Any) -> None:
        with self._lock:
            self._data["markets"].setdefault(symbol, _empty_market()).update(kwargs)

    def set_symbols(self, symbols: List[str]) -> None:
        with self._lock:
            self._data["symbols"] = list(symbols)
            if symbols:
                self._data["symbol"] = symbols[0]
            for s in symbols:
                self._data["markets"].setdefault(s, _empty_market())
            for s in list(self._data["markets"]):
                if s not in symbols:
                    del self._data["markets"][s]

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def snapshot(self) -> Dict[str, Any]:
        """Deep-ish copy safe to use outside the lock."""
        def _copy(v: Any) -> Any:
            if isinstance(v, pd.DataFrame):
                return v.copy()
            if isinstance(v, dict):
                return {k: _copy(x) for k, x in v.items()}
            return copy.deepcopy(v)

        with self._lock:
            snap = {k: _copy(v) for k, v in self._data.items()}
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
        self.symbols: List[str] = configured_symbols()
        self.symbol = self.symbols[0]   # primary symbol (backwards compatible)
        self.state = shared_state or SharedState(self.symbols)
        self.engine = engine or PaperEngine(symbol=self.symbol)
        self.risk = risk_manager or RiskManager()

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
        self._last_processed_candle_ts: Dict[str, pd.Timestamp] = {}
        self._last_prices: Dict[str, float] = {}
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
        logger.info("Bot PAUSED - no new entries (open positions still protected by SL/TP)")

    def resume(self) -> None:
        self._paused.clear()
        self.state.update(paused=False)
        self._publish_status()
        logger.info("Bot RESUMED")

    def is_paused(self) -> bool:
        return self._paused.is_set()

    def _open_symbols(self) -> List[str]:
        return [s for s in self.symbols if self.engine.has_position(s)]

    def kill_switch(self) -> List[Dict[str, Any]]:
        """Emergency stop: close every open position at market and pause the bot."""
        logger.warning("EMERGENCY KILL SWITCH activated")
        self.pause()
        closed: List[Dict[str, Any]] = []
        try:
            with self._trade_lock:
                for sym in self._open_symbols():
                    price = None
                    try:
                        price = self._fetch_price(sym)
                    except Exception as exc:
                        logger.error("Kill switch: live %s price unavailable (%s), using last known", sym, exc)
                    pos = self.engine.get_position(sym) or {}
                    price = price or self._last_prices.get(sym) or pos.get("current_price")
                    if price:
                        trade = self.engine.close_position(price, "KILL_SWITCH", sym)
                        if trade:
                            closed.append(trade)
                    else:
                        logger.error("Kill switch: no price for %s - position NOT closed", sym)
        except Exception:
            logger.exception("Kill switch failed")
        self._publish_account()
        return closed

    # ------------------------------------------------------------------
    # Exchange helpers
    # ------------------------------------------------------------------
    def _connect(self) -> bool:
        """Create a ccxt client, falling back to alternate public exchanges."""
        wanted = configured_symbols()
        candidates = [config.EXCHANGE] + [e for e in config.FALLBACK_EXCHANGES if e != config.EXCHANGE]
        for ex_id in candidates:
            try:
                ex_class = getattr(ccxt, ex_id)
                ex = ex_class({"enableRateLimit": True, "timeout": config.REQUEST_TIMEOUT_MS})
                ex.load_markets()
                tfs = getattr(ex, "timeframes", None) or {}
                if tfs and config.TIMEFRAME not in tfs:
                    logger.warning("%s does not support timeframe %s - skipping", ex_id, config.TIMEFRAME)
                    continue
                available = [s for s in wanted if s in ex.markets]
                missing = [s for s in wanted if s not in ex.markets]
                if not available:
                    logger.warning("%s lists none of %s - skipping", ex_id, wanted)
                    continue
                if missing:
                    logger.warning("%s does not list %s - those symbols are skipped", ex_id, missing)
                # Keep symbols that still have an open position so they stay protected
                keep = available + [s for s in self._open_symbols() if s not in available]
                self.symbols = keep
                self.symbol = keep[0]
                self.state.set_symbols(keep)
                self.exchange = ex
                self.state.update(exchange=ex_id)
                if ex_id != config.EXCHANGE:
                    logger.warning("Primary exchange '%s' unavailable - using fallback '%s'",
                                   config.EXCHANGE, ex_id)
                logger.info("Connected to %s public API (%s, %s)", ex_id, ", ".join(available), config.TIMEFRAME)
                return True
            except AttributeError:
                logger.error("Unknown ccxt exchange id '%s'", ex_id)
            except Exception as exc:
                logger.error("Could not connect to %s: %s: %s", ex_id, type(exc).__name__, str(exc)[:200])
        return False

    @staticmethod
    def _ticker_price(ticker: Dict[str, Any]) -> Optional[float]:
        price = ticker.get("last") or ticker.get("close")
        if price is None and ticker.get("bid") and ticker.get("ask"):
            price = (ticker["bid"] + ticker["ask"]) / 2
        return float(price) if price else None

    def _fetch_price(self, symbol: str) -> Optional[float]:
        return self._ticker_price(self.exchange.fetch_ticker(symbol))

    def _fetch_prices(self) -> Dict[str, float]:
        """Latest price for every symbol (one request when the exchange allows it)."""
        prices: Dict[str, float] = {}
        if len(self.symbols) > 1 and self.exchange.has.get("fetchTickers"):
            try:
                tickers = self.exchange.fetch_tickers(self.symbols)
                for sym in self.symbols:
                    t = tickers.get(sym)
                    p = self._ticker_price(t) if t else None
                    if p and p > 0:
                        prices[sym] = p
            except Exception as exc:
                logger.warning("fetch_tickers failed (%s) - falling back to single tickers", type(exc).__name__)
        for sym in self.symbols:
            if sym not in prices:
                try:
                    p = self._fetch_price(sym)
                    if p and p > 0:
                        prices[sym] = p
                except ccxt.BaseError as exc:
                    logger.warning("%s ticker failed: %s", sym, str(exc)[:150])
        return prices

    def _fetch_candles(self, symbol: str) -> pd.DataFrame:
        raw = self.exchange.fetch_ohlcv(symbol, config.TIMEFRAME, limit=config.CANDLE_LIMIT)
        if not raw:
            return pd.DataFrame()
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        return df.dropna(subset=["open", "high", "low", "close"])

    def _amount_rules(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """(qty_step, min_qty) from the exchange market metadata, if known."""
        try:
            market = self.exchange.market(symbol)
        except Exception:
            return None, None
        step = (market.get("precision") or {}).get("amount")
        if step is not None:
            # ccxt >= 4 uses TICK_SIZE (step); older/other modes use decimal places
            if getattr(self.exchange, "precisionMode", None) != getattr(ccxt, "TICK_SIZE", 4):
                step = 10 ** -int(step)
            step = float(step)
        min_qty = ((market.get("limits") or {}).get("amount") or {}).get("min")
        return step, (float(min_qty) if min_qty else None)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _run(self) -> None:
        logger.info("Autonomous loop starting (interval %ss, symbols %s)",
                    config.LOOP_INTERVAL, ", ".join(self.symbols))
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
        # 1) Latest ticks -> mark-to-market and SL/TP protection -------------
        prices = self._fetch_prices()
        if not prices:
            raise ccxt.ExchangeError("Tickers returned no valid price")
        now = datetime.now(timezone.utc)
        for sym, price in prices.items():
            self._last_prices[sym] = price
            self.state.update_market(sym, last_price=price, last_price_time=now)
            with self._trade_lock:
                closed = self.engine.check_sl_tp(price, sym)
            if closed:
                logger.info("%s exit by %s: net PnL %.4f USDT", sym, closed["exit_reason"], closed["net_pnl"])
        primary = self.symbols[0]
        self.state.update(last_price=prices.get(primary), last_price_time=now)

        # 2) Daily drawdown circuit breaker ----------------------------------
        if self.risk.check_daily_drawdown(self.engine) or self.risk.should_suspend():
            with self._trade_lock:
                for sym in self._open_symbols():
                    px = prices.get(sym) or self._last_prices.get(sym)
                    if px:
                        self.engine.close_position(px, "DRAWDOWN_BREAKER", sym)
        suspended = self.risk.should_suspend()
        self.state.update(resume_time=self.risk.resume_time)

        # 3+4) Candles, indicators and signals per symbol --------------------
        failures = 0
        for sym in list(self.symbols):
            if sym not in prices:
                failures += 1
                continue
            try:
                self._process_symbol(sym, prices[sym], suspended)
                self.state.update_market(sym, error=None)
            except ccxt.BaseError as exc:
                failures += 1
                msg = f"{type(exc).__name__}: {str(exc)[:150]}"
                logger.warning("%s skipped this iteration: %s", sym, msg)
                self.state.update_market(sym, error=msg)

        self._publish_account()
        if failures == len(self.symbols):
            raise ccxt.NetworkError("All symbols failed this iteration")

    def _process_symbol(self, sym: str, price: float, suspended: bool) -> None:
        candles = self._fetch_candles(sym)
        min_bars = max(config.EMA_SLOW, config.RSI_PERIOD + 1, config.ATR_PERIOD) + config.RSI_LOOKBACK + 2
        if candles.empty or len(candles) < min_bars:
            logger.warning("%s: not enough candle data (%d rows, need %d)", sym, len(candles), min_bars)
            return

        ind = strategy.compute_indicators(candles)
        self.state.update_market(sym, candles=ind)
        if sym == self.symbols[0]:
            self.state.update(candles=ind)  # backwards compatible primary chart

        # The last row is the still-forming candle - evaluate signals on closed bars
        closed_df = ind.iloc[:-1] if config.USE_CLOSED_CANDLES_ONLY else ind
        if closed_df.empty:
            return
        candle_ts = closed_df["timestamp"].iloc[-1]
        self.state.update_market(sym, last_candle_time=candle_ts.to_pydatetime())

        # Signal handling - once per newly closed candle
        if candle_ts == self._last_processed_candle_ts.get(sym):
            return
        self._last_processed_candle_ts[sym] = candle_ts

        diag = strategy.evaluate_signal(closed_df)
        signal = diag["signal"]
        self.state.update_market(sym, diagnostics=diag)
        if signal:
            now = datetime.now(timezone.utc)
            logger.info("Signal %s %s on candle %s (close %.6g, RSI %.1f) - %s", signal, sym,
                        candle_ts.strftime("%H:%M"), closed_df["close"].iloc[-1], diag["rsi"], diag["reason"])
            self.state.update_market(sym, last_signal=signal, last_signal_time=now)
            self.state.update(last_signal=signal, last_signal_symbol=sym, last_signal_time=now)
        elif getattr(config, "LOG_SIGNAL_DIAGNOSTICS", False):
            rsi_txt = f"{diag['rsi']:.1f}" if diag["rsi"] is not None else "n/a"
            logger.info("%s %s no signal | RSI %s | %s", sym, candle_ts.strftime("%H:%M"), rsi_txt, diag["reason"])

        with self._trade_lock:
            position = self.engine.get_position(sym)

            # Reversal exit
            if position and signal and signal != position["side"] and config.EXIT_ON_REVERSAL:
                self.engine.close_position(price, "REVERSAL", sym)
                position = None
                if not config.ENTER_ON_REVERSAL:
                    signal = None

            # New entry
            if position is None and signal:
                open_count = len(self._open_symbols())
                if suspended:
                    logger.info("Signal %s %s ignored - trading suspended until %s",
                                signal, sym, self.risk.resume_time)
                elif self._paused.is_set():
                    logger.info("Signal %s %s ignored - bot paused", signal, sym)
                elif open_count >= config.MAX_OPEN_POSITIONS:
                    logger.info("Signal %s %s ignored - MAX_OPEN_POSITIONS (%d) reached",
                                signal, sym, config.MAX_OPEN_POSITIONS)
                else:
                    self._enter(sym, signal, price, float(closed_df["ATR"].iloc[-1]))

    def _enter(self, sym: str, side: str, price: float, atr_value: float) -> None:
        try:
            sl, tp = strategy.calculate_sl_tp(side, price, atr_value)
        except ValueError as exc:
            logger.warning("%s entry skipped: %s", sym, exc)
            return

        equity = self.engine.get_equity()
        step, min_qty = self._amount_rules(sym)
        leverage = None
        if getattr(config, "POSITION_SIZING_MODE", "risk") == "margin":
            leverage = config.LEVERAGE
            qty = self.risk.calculate_margin_size(
                self.engine.balance, price, used_margin=self.engine.get_used_margin(),
                qty_step=step, min_qty=min_qty,
            )
            # The stop must trigger before liquidation, otherwise skip the trade
            mmr = config.MAINTENANCE_MARGIN_RATE
            liq_move = (1.0 / leverage - mmr - config.SLIPPAGE_MAX) * price
            if abs(price - sl) >= liq_move:
                logger.warning("%s entry skipped: stop %.6g is beyond the %gx liquidation distance %.6g",
                               sym, abs(price - sl), leverage, liq_move)
                return
        else:
            qty = self.risk.calculate_position_size(equity, price, sl, qty_step=step, min_qty=min_qty)
        if qty <= 0:
            logger.warning("%s entry skipped: position size is zero (equity %.2f, stop %.6g)",
                           sym, equity, abs(price - sl))
            return

        if config.SKIP_IF_TP_BELOW_COSTS:
            reward = abs(tp - price) * qty
            costs = self.risk.estimate_round_trip_cost(price, qty)
            if reward <= costs:
                logger.info("%s entry skipped: TP reward %.4f <= round-trip costs %.4f (ATR too small)",
                            sym, reward, costs)
                return

        self.engine.open_position(side, price, qty, sl, tp, symbol=sym, leverage=leverage)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------
    def _publish_account(self) -> None:
        try:
            positions = {s: self.engine.get_position(s) for s in self._open_symbols()}
            primary = positions.get(self.symbols[0]) if self.symbols else None
            self.state.update(
                positions=positions,
                position=primary or (next(iter(positions.values())) if positions else None),
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
            logger.info("Equity %.2f | trades %d | open %d | win rate %.1f%%",
                        s["equity"], s["total_trades"], len(worker._open_symbols()), s["win_rate"])
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        worker.stop()
