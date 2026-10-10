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
from typing import Optional, Tuple

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
            return self._round_qty(qty, qty_step, min_qty)
        except Exception:  # defensive: sizing must never crash the loop
            logger.exception("Position sizing failed")
            return 0.0

    def calculate_margin_size(
        self,
        balance: float,
        entry_price: float,
        used_margin: float = 0.0,
        margin_pct: float = config.MARGIN_PER_TRADE,
        leverage: float = config.LEVERAGE,
        qty_step: Optional[float] = None,
        min_qty: Optional[float] = None,
    ) -> float:
        """Fixed-margin sizing: margin = margin_pct * balance, notional = margin * leverage.

        Returns 0.0 if the free balance cannot cover the margin plus the
        entry fee, or the trade cannot be sized sensibly.
        """
        try:
            if balance <= 0 or entry_price <= 0 or margin_pct <= 0 or leverage <= 0:
                return 0.0
            margin = balance * margin_pct
            notional = margin * leverage
            entry_fee = notional * (self.fee_rate + self.max_slippage)
            free = balance - used_margin
            if margin + entry_fee > free:
                logger.info("Not enough free balance for margin %.2f (free %.2f, used %.2f)",
                            margin, free, used_margin)
                return 0.0
            return self._round_qty(notional / entry_price, qty_step, min_qty)
        except Exception:
            logger.exception("Margin sizing failed")
            return 0.0

    def size_order(
        self,
        balance: float,
        equity: float,
        used_margin: float,
        entry_price: float,
        sl_price: float,
        qty_step: Optional[float] = None,
        min_qty: Optional[float] = None,
    ) -> Tuple[float, Optional[float], str]:
        """Size an order per config. Returns (qty, leverage or None, reason if qty == 0).

        "margin" mode: leverage = highest value <= LEVERAGE that keeps the
        liquidation price LIQ_BUFFER_MULT x the stop distance away; notional =
        MARGIN_PER_TRADE x balance x leverage, capped so a stop-out (incl.
        costs) loses at most MAX_LOSS_PER_TRADE of the balance.
        """
        if getattr(config, "POSITION_SIZING_MODE", "risk") != "margin":
            qty = self.calculate_position_size(equity, entry_price, sl_price, qty_step, min_qty)
            return qty, None, "" if qty > 0 else "position size is zero"

        stop_dist = abs(entry_price - sl_price)
        if stop_dist <= 0 or balance <= 0:
            return 0.0, None, "invalid stop"
        stop_frac = stop_dist / entry_price
        cost_frac = 2 * (self.max_slippage + self.fee_rate)

        # Highest leverage (<= LEVERAGE) whose liquidation price stays at least
        # LIQ_BUFFER_MULT x the stop distance away (isolated margin model).
        buf = getattr(config, "LIQ_BUFFER_MULT", 1.0)
        lev_limit = 1.0 / (buf * stop_frac + config.MAINTENANCE_MARGIN_RATE + self.max_slippage)
        leverage = float(min(config.LEVERAGE, math.floor(lev_limit)))
        if leverage < 1:
            return 0.0, None, f"stop {stop_frac * 100:.2f}% too wide for any leverage"

        notional = balance * config.MARGIN_PER_TRADE * leverage
        max_loss = getattr(config, "MAX_LOSS_PER_TRADE", 0.0)
        if max_loss > 0:
            notional = min(notional, balance * max_loss / (stop_frac + cost_frac))
        margin = notional / leverage
        entry_fee = notional * (self.fee_rate + self.max_slippage)
        if margin + entry_fee > balance - used_margin:
            return 0.0, leverage, (f"not enough free balance for margin {margin:.2f} "
                                   f"(free {balance - used_margin:.2f})")
        qty = self._round_qty(notional / entry_price, qty_step, min_qty)
        return qty, leverage, "" if qty > 0 else "position size is zero"

    @staticmethod
    def _round_qty(qty: float, qty_step: Optional[float], min_qty: Optional[float]) -> float:
        """Floor qty to the market's amount step; 0.0 if below the minimum."""
        step = qty_step if qty_step and qty_step > 0 else 10 ** -config.QTY_PRECISION
        qty = math.floor(qty / step + 1e-9) * step
        decimals = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
        qty = round(qty, decimals)
        floor_qty = min_qty if min_qty and min_qty > 0 else config.MIN_QTY
        return qty if qty >= floor_qty and qty > 0 else 0.0

    def estimate_round_trip_cost(self, entry_price: float, qty: float) -> float:
        """Worst-case fees + slippage for opening and closing a position."""
        return 2 * entry_price * qty * (self.max_slippage + self.fee_rate)

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------
    def daily_risk_budget_ok(self, engine, extra_risk_frac: float) -> Tuple[bool, str]:
        """Would one more trade risking ``extra_risk_frac`` of the day-start equity
        (plus the stop risk of open positions) still fit inside the daily limit?"""
        try:
            start = engine.day_start_equity or 0.0
            if start <= 0:
                return True, ""
            dd = engine.get_daily_drawdown_pct()
            open_risk = 0.0
            with engine._lock:  # positions may be closed by the price-stream thread
                positions = [dict(p) for p in engine.positions.values()]
            for pos in positions:
                d = 1 if pos["side"] == "LONG" else -1
                # loss from the current price to the stop, plus the exit fee
                loss = max(0.0, (pos["current_price"] - pos["sl"]) * d) * pos["qty"]
                loss += pos["sl"] * pos["qty"] * (self.fee_rate + self.max_slippage)
                open_risk += loss / start
            total = dd + open_risk + extra_risk_frac
            if total > self.daily_drawdown_limit:
                return False, (f"daily risk budget: today {dd * 100:.2f}% + open {open_risk * 100:.2f}% "
                               f"+ new {extra_risk_frac * 100:.2f}% > {self.daily_drawdown_limit * 100:.1f}%")
        except Exception:
            logger.exception("Daily risk budget check failed")
        return True, ""

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
                    if getattr(config, "SUSPEND_UNTIL_NEXT_UTC_DAY", False):
                        self.suspended_until = datetime(now.year, now.month, now.day,
                                                        tzinfo=timezone.utc) + timedelta(days=1)
                    else:
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
