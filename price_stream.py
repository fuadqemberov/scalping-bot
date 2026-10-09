"""
PriceStream - real-time ticker prices over WebSocket (ccxt.pro, bundled with ccxt >= 4).

Runs its own asyncio loop in a daemon thread and calls ``on_tick(symbol,
price)`` for every ticker update (Binance pushes ~1 update/second per symbol).
The bot uses each tick to check SL / TP / break-even / liquidation, so stops
are hit within about a second instead of up to LOOP_INTERVAL (10 s) later.

The stream reconnects on its own after errors. It is an *addition* to the
REST polling in the main loop, never a replacement: if the stream is down,
the REST ticker check every LOOP_INTERVAL still protects open positions.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import config

logger = logging.getLogger("stream")

try:  # ccxt.pro ships inside the regular ccxt package since 2022
    import ccxt.pro as ccxtpro
except Exception:  # pragma: no cover - very old ccxt
    ccxtpro = None


def _ticker_price(t: Dict[str, Any]) -> Optional[float]:
    price = t.get("last") or t.get("close")
    if price is None and t.get("bid") and t.get("ask"):
        price = (t["bid"] + t["ask"]) / 2
    return float(price) if price else None


class PriceStream:
    def __init__(self, exchange_id: str, symbols: List[str],
                 on_tick: Callable[[str, float], None]) -> None:
        self.exchange_id = exchange_id
        self.symbols = list(symbols)
        self.on_tick = on_tick
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._prices: Dict[str, float] = {}
        self._times: Dict[str, float] = {}
        self.connected = False
        self.last_error: Optional[str] = None
        self.ticks = 0
        self.reconnects = 0

    # ------------------------------------------------------------------
    # Public API (thread-safe)
    # ------------------------------------------------------------------
    @staticmethod
    def available(exchange_id: str) -> bool:
        return ccxtpro is not None and hasattr(ccxtpro, exchange_id)

    def start(self) -> bool:
        if not self.available(self.exchange_id):
            logger.warning("WebSocket not available for %s - using REST polling only", self.exchange_id)
            return False
        if self._thread and self._thread.is_alive():
            return True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="PriceStream", daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        loop, task = self._loop, self._task
        if loop and task and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass
        if self._thread:
            self._thread.join(timeout=timeout)
        self.connected = False

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def latest(self, symbol: str, max_age: float = 5.0) -> Optional[float]:
        """Last streamed price if it is younger than ``max_age`` seconds."""
        with self._lock:
            t = self._times.get(symbol)
            if t is None or time.time() - t > max_age:
                return None
            return self._prices.get(symbol)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            last = max(self._times.values()) if self._times else None
        age = (time.time() - last) if last else None
        stale = age is None or age > getattr(config, "WS_STALE_SECONDS", 30)
        return {
            "enabled": True,
            "exchange": self.exchange_id,
            "connected": self.connected and not stale,
            "last_tick_age": age,
            "ticks": self.ticks,
            "reconnects": self.reconnects,
            "error": self.last_error,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            asyncio.set_event_loop(loop)
            self._task = loop.create_task(self._main())
            loop.run_until_complete(self._task)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Price stream thread crashed")
        finally:
            self.connected = False
            try:
                loop.close()
            except Exception:
                pass
            logger.info("Price stream stopped")

    def _handle(self, symbol: str, ticker: Dict[str, Any]) -> None:
        price = _ticker_price(ticker or {})
        if not price or price <= 0 or symbol not in self.symbols:
            return
        with self._lock:
            self._prices[symbol] = price
            self._times[symbol] = time.time()
        self.ticks += 1
        try:
            self.on_tick(symbol, price)
        except Exception:
            logger.exception("on_tick failed for %s", symbol)

    async def _watch_all(self, ex) -> None:
        if len(self.symbols) > 1 and ex.has.get("watchTickers"):
            while not self._stop.is_set():
                tickers = await ex.watch_tickers(self.symbols)
                self.connected, self.last_error = True, None
                for sym, t in (tickers or {}).items():
                    self._handle(sym, t)
        else:
            async def one(sym: str) -> None:
                while not self._stop.is_set():
                    t = await ex.watch_ticker(sym)
                    self.connected, self.last_error = True, None
                    self._handle(sym, t)
            await asyncio.gather(*(one(s) for s in self.symbols))

    async def _main(self) -> None:
        delay = getattr(config, "WS_RECONNECT_SECONDS", 5)
        while not self._stop.is_set():
            ex = getattr(ccxtpro, self.exchange_id)({"enableRateLimit": True,
                                                     "timeout": config.REQUEST_TIMEOUT_MS})
            try:
                logger.info("WebSocket connecting to %s (%s)", self.exchange_id, ", ".join(self.symbols))
                await self._watch_all(ex)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.connected = False
                self.last_error = f"{type(exc).__name__}: {str(exc)[:150]}"
                self.reconnects += 1
                logger.warning("WebSocket error (%s) - reconnecting in %ss; REST polling still active",
                               self.last_error, delay)
            finally:
                try:
                    await ex.close()
                except Exception:
                    pass
            if not self._stop.is_set():
                await asyncio.sleep(delay)
