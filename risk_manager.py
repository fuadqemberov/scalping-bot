"""
RiskManager - position sizing and the daily drawdown circuit breaker.

Position sizing is *cost-aware*: the per-unit loss at the stop includes the
stop distance plus worst-case round-trip slippage and both taker fees, so a
stopped-out trade loses at most RISK_PER_TRADE of equity (barring gaps).
"""

from __future__ import annotations

import logging
import math
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

import config

logger = logging.getLogger("risk")


class RiskManager:
    def __init__(
        self,
        risk_per_trade: float = config.RISK_PER_TRADE,
        daily_drawdown_limit: float = config.DAILY_DRAWDOWN_LIMIT,
        suspension_hours: float = config.SUSPENSION_HOURS,
        max_leverage: float = config.MAX_LEVERAGE,
        fee_rate: float = config.FEE_RATE,
        max_slippage: float = config.SLIPPAGE_MAX,
    ) -> None:
        self._lock = threading.Lock()
        self.risk_per_trade = risk_per_trade
        self.daily_drawdown_limit = daily_drawdown_limit
        self.suspension_hours = suspension_hours
        self.max_leverage = max_leverage
        self.fee_rate = fee_rate
        self.max_slippage = max_slippage
        self.suspended_until: Optional[datetime] = None
        self.last_breach_time: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------
    def calculate_position_size(
        self,
        equity: float,
        entry_price: float,
        sl_price: float,
        qty_step: Optional[float] = None,
        min_qty: Optional[float] = None,
    ) -> float:
        """Quantity such that a stop-out loses <= risk_per_trade * equity (incl. costs).

        ``qty_step`` / ``min_qty`` come from the exchange market of the symbol
        (e.g. 0.00001 BTC, 0.0001 ETH, 0.1 XRP); config values are fallbacks.
        Returns 0.0 if the trade cannot be sized sensibly.
        """
        try:
            if equity <= 0 or entry_price <= 0 or sl_price <= 0:
                return 0.0
            stop_dist = abs(entry_price - sl_price)
            if stop_dist <= 0:
                return 0.0

            risk_amount = equity * self.risk_per_trade
            # Worst-case per-unit costs: entry + exit slippage and fees
            cost_per_unit = 2 * entry_price * (self.max_slippage + self.fee_rate)
            qty = risk_amount / (stop_dist + cost_per_unit)

            # Leverage cap on notional exposure
            max_qty = equity * self.max_leverage / entry_price
            if qty > max_qty:
                logger.info("Size capped by MAX_LEVERAGE %.1fx (%.5f -> %.5f)",
                            self.max_leverage, qty, max_qty)
                qty = max_qty

            # Floor to exchange precision so we never exceed the risk budget
            step = qty_step if qty_step and qty_step > 0 else 10 ** -config.QTY_PRECISION
            qty = math.floor(qty / step + 1e-9) * step
            decimals = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
            qty = round(qty, decimals)
            floor_qty = min_qty if min_qty and min_qty > 0 else config.MIN_QTY
            return qty if qty >= floor_qty and qty > 0 else 0.0
        except Exception:  # defensive: sizing must never crash the loop
            logger.exception("Position sizing failed")
            return 0.0

    def estimate_round_trip_cost(self, entry_price: float, qty: float) -> float:
        """Worst-case fees + slippage for opening and closing a position."""
        return 2 * entry_price * qty * (self.max_slippage + self.fee_rate)

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------
    def check_daily_drawdown(self, engine) -> bool:
        """True if today's equity loss reached the limit. Triggers a suspension."""
        try:
            dd = engine.get_daily_drawdown_pct()
        except Exception:
            logger.exception("Could not read daily drawdown from engine")
            return False

        if dd >= self.daily_drawdown_limit:
            with self._lock:
                if self.suspended_until is None:
                    now = datetime.now(timezone.utc)
                    self.last_breach_time = now
                    self.suspended_until = now + timedelta(hours=self.suspension_hours)
                    logger.warning(
                        "DAILY DRAWDOWN BREACH %.2f%% >= %.2f%% - trading suspended until %s UTC",
                        dd * 100, self.daily_drawdown_limit * 100,
                        self.suspended_until.strftime("%Y-%m-%d %H:%M:%S"),
                    )
            return True
        return False

    def should_suspend(self) -> bool:
        """True while a circuit-breaker suspension is active (auto-clears on expiry)."""
        with self._lock:
            if self.suspended_until is None:
                return False
            if datetime.now(timezone.utc) >= self.suspended_until:
                logger.info("Suspension expired - trading may resume")
                self.suspended_until = None
                return False
            return True

    @property
    def resume_time(self) -> Optional[datetime]:
        with self._lock:
            return self.suspended_until

    def clear_suspension(self) -> None:
        """Manual override (not used by the bot automatically)."""
        with self._lock:
            self.suspended_until = None
        logger.warning("Suspension manually cleared")
