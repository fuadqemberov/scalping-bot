"""
PaperEngine - a thread-safe virtual trading account.

Accounting model (perpetual-futures style, shorting allowed):
    * ``balance``  = realised cash. Fees are deducted when they are paid
                     (entry fee at open, exit fee at close).
    * ``equity``   = balance + unrealised PnL of the open position.
    * Gross PnL    = (exit_fill - entry_fill) * qty * direction
    * Net PnL      = gross PnL - entry fee - exit fee
      (slippage is already embedded in the fill prices and is reported
       separately in dollars for transparency).

Only one position per symbol is allowed; the bot itself enforces a single
position overall.
"""

from __future__ import annotations

import copy
import logging
import random
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import config

logger = logging.getLogger("engine")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PaperEngine:
    """Virtual exchange account with slippage + fee simulation."""

    def __init__(
        self,
        initial_balance: float = config.INITIAL_BALANCE,
        fee_rate: float = config.FEE_RATE,
        slippage_min: float = config.SLIPPAGE_MIN,
        slippage_max: float = config.SLIPPAGE_MAX,
        symbol: str = config.SYMBOL,
    ) -> None:
        if initial_balance <= 0:
            raise ValueError("initial_balance must be positive")
        if not 0 <= slippage_min <= slippage_max:
            raise ValueError("slippage bounds must satisfy 0 <= min <= max")

        # RLock: public methods call each other while holding the lock.
        self._lock = threading.RLock()

        self.symbol = symbol
        self.initial_balance = float(initial_balance)
        self.fee_rate = float(fee_rate)
        self.slippage_min = float(slippage_min)
        self.slippage_max = float(slippage_max)

        self.balance: float = float(initial_balance)
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.last_price: Optional[float] = None

        # Equity curve / drawdown statistics
        self.peak_equity: float = float(initial_balance)
        self.max_drawdown_pct: float = 0.0

        # Daily tracking for the drawdown circuit breaker (UTC day)
        self._day = _utcnow().date()
        self.day_start_equity: float = float(initial_balance)
        self.daily_realized_pnl: float = 0.0

        self._trade_counter = 0

    # ------------------------------------------------------------------
    # Internal helpers (call with lock held)
    # ------------------------------------------------------------------
    def _random_slippage(self) -> float:
        return random.uniform(self.slippage_min, self.slippage_max)

    @staticmethod
    def _direction(side: str) -> int:
        return 1 if side == "LONG" else -1

    def _apply_slippage(self, price: float, side: str, is_entry: bool) -> float:
        """Return a fill price that is always *worse* than the reference.

        Buying (long entry / short exit) fills higher, selling (short entry /
        long exit) fills lower.
        """
        slip = self._random_slippage()
        buying = (side == "LONG") == is_entry
        return price * (1 + slip) if buying else price * (1 - slip)

    def _unrealized(self, pos: Dict[str, Any], price: float) -> float:
        return (price - pos["entry_price"]) * pos["qty"] * self._direction(pos["side"])

    def _equity_unlocked(self) -> float:
        upnl = sum(p["unrealized_pnl"] for p in self.positions.values())
        return self.balance + upnl

    def _roll_day_if_needed(self) -> None:
        """Reset daily counters at UTC midnight."""
        today = _utcnow().date()
        if today != self._day:
            self._day = today
            self.day_start_equity = self._equity_unlocked()
            self.daily_realized_pnl = 0.0
            logger.info("New UTC trading day %s - day start equity %.2f",
                        today, self.day_start_equity)

    def _update_drawdown_stats(self) -> None:
        eq = self._equity_unlocked()
        if eq > self.peak_equity:
            self.peak_equity = eq
        if self.peak_equity > 0:
            dd = (self.peak_equity - eq) / self.peak_equity
            self.max_drawdown_pct = max(self.max_drawdown_pct, dd)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def has_position(self, symbol: Optional[str] = None) -> bool:
        with self._lock:
            return (symbol or self.symbol) in self.positions

    def get_position(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return a copy of the open position (or None)."""
        with self._lock:
            pos = self.positions.get(symbol or self.symbol)
            return copy.deepcopy(pos) if pos else None

    def open_position(
        self,
        side: str,
        price: float,
        qty: float,
        sl: float,
        tp: float,
        symbol: Optional[str] = None,
        leverage: Optional[float] = None,
        strategy: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Open a market position with simulated slippage and entry fee.

        With ``leverage`` the position is treated as isolated margin:
        margin = notional / leverage and a liquidation price is computed.

        Returns a copy of the new position dict, or None if rejected.
        """
        symbol = symbol or self.symbol
        side = side.upper()
        try:
            if side not in ("LONG", "SHORT"):
                raise ValueError(f"invalid side {side!r}")
            if price <= 0 or qty <= 0:
                raise ValueError(f"price/qty must be positive (price={price}, qty={qty})")
            if side == "LONG" and not (sl < price < tp):
                raise ValueError(f"LONG requires sl < price < tp (sl={sl}, price={price}, tp={tp})")
            if side == "SHORT" and not (tp < price < sl):
                raise ValueError(f"SHORT requires tp < price < sl (tp={tp}, price={price}, sl={sl})")
        except ValueError as exc:
            logger.warning("Order rejected: %s", exc)
            return None

        with self._lock:
            self._roll_day_if_needed()
            if symbol in self.positions:
                logger.warning("Order rejected: position already open on %s", symbol)
                return None

            fill = self._apply_slippage(price, side, is_entry=True)
            # Re-anchor SL/TP to the actual fill so their distances stay as
            # planned. Previously they were measured from the pre-slippage
            # price, so entry slippage moved the fill toward the TP (shorts
            # fill lower, longs higher) and shrank the reward: the log's
            # average RR was ~1.0 instead of the configured 1.5.
            shift = fill - price
            sl, tp = sl + shift, tp + shift
            notional = fill * qty
            fee = notional * self.fee_rate
            if fee >= self.balance:
                logger.warning("Order rejected: insufficient balance for fees")
                return None

            margin = notional / leverage if leverage and leverage > 0 else None
            liq_price = None
            if margin is not None:
                if margin + fee > self.balance - self._used_margin_unlocked():
                    logger.warning("Order rejected: insufficient free balance for margin %.2f", margin)
                    return None
                # Isolated margin: liquidated when loss = margin - maintenance margin
                mmr = config.MAINTENANCE_MARGIN_RATE
                move = 1.0 / leverage - mmr
                liq_price = fill * (1 - move) if side == "LONG" else fill * (1 + move)

            self.balance -= fee
            self.daily_realized_pnl -= fee
            self._trade_counter += 1

            pos = {
                "id": self._trade_counter,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "requested_price": price,
                "entry_price": fill,
                "entry_slippage_cost": abs(fill - price) * qty,
                "entry_fee": fee,
                "sl": sl,
                "tp": tp,
                "entry_time": _utcnow(),
                "unrealized_pnl": 0.0,
                "current_price": price,
                "leverage": leverage,
                "margin": margin,
                "liq_price": liq_price,
                "initial_sl": sl,
                "risk_dist": abs(fill - sl),   # 1R in price units
                "breakeven_moved": False,
                "strategy": strategy,
            }
            pos["unrealized_pnl"] = self._unrealized(pos, price)
            self.positions[symbol] = pos
            self.last_price = price
            self._update_drawdown_stats()

            logger.info(
                "OPEN %s #%d %s [%s] qty=%g ref=%.6g fill=%.6g SL=%.6g TP=%.6g fee=%.4f%s",
                side, pos["id"], symbol, strategy or "-", qty, price, fill, sl, tp, fee,
                f" | {leverage:g}x margin={margin:.2f} notional={notional:.2f} liq={liq_price:.6g}"
                if margin is not None else "",
            )
            return copy.deepcopy(pos)

    def close_position(
        self, price: float, reason: str, symbol: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Close the open position at ``price`` (+ slippage). Returns the trade record."""
        symbol = symbol or self.symbol
        if price is None or price <= 0:
            logger.warning("close_position ignored: invalid price %s", price)
            return None

        with self._lock:
            self._roll_day_if_needed()
            pos = self.positions.get(symbol)
            if pos is None:
                return None

            side = pos["side"]
            fill = self._apply_slippage(price, side, is_entry=False)
            qty = pos["qty"]
            gross = (fill - pos["entry_price"]) * qty * self._direction(side)
            exit_fee = fill * qty * self.fee_rate
            net = gross - pos["entry_fee"] - exit_fee

            self.balance += gross - exit_fee
            self.daily_realized_pnl += gross - exit_fee
            del self.positions[symbol]
            self.last_price = price

            exit_time = _utcnow()
            entry_notional = pos["entry_price"] * qty
            trade = {
                "id": pos["id"],
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "entry_time": pos["entry_time"],
                "exit_time": exit_time,
                "duration_sec": (exit_time - pos["entry_time"]).total_seconds(),
                "entry_price": pos["entry_price"],
                "exit_price": fill,
                "exit_reference_price": price,
                "sl": pos.get("initial_sl", pos["sl"]),
                "final_sl": pos["sl"],
                "tp": pos["tp"],
                "gross_pnl": gross,
                "entry_fee": pos["entry_fee"],
                "exit_fee": exit_fee,
                "total_fees": pos["entry_fee"] + exit_fee,
                "slippage_cost": pos["entry_slippage_cost"] + abs(fill - price) * qty,
                "net_pnl": net,
                "pnl_pct": (net / entry_notional * 100) if entry_notional else 0.0,
                "exit_reason": reason,
                "strategy": pos.get("strategy"),
                "leverage": pos.get("leverage"),
                "margin": pos.get("margin"),
                "roe_pct": (net / pos["margin"] * 100) if pos.get("margin") else None,
                "balance_after": self.balance,
            }
            self.trade_history.append(trade)
            self._update_drawdown_stats()

            logger.info(
                "CLOSE %s #%d %s @ %.2f (ref %.2f) reason=%s net=%.4f balance=%.2f",
                side, pos["id"], symbol, fill, price, reason, net, self.balance,
            )
            return copy.deepcopy(trade)

    def update_market_price(self, price: float, symbol: Optional[str] = None) -> None:
        """Mark the open position to market (updates unrealised PnL)."""
        if price is None or price <= 0:
            return
        symbol = symbol or self.symbol
        with self._lock:
            self._roll_day_if_needed()
            self.last_price = price
            pos = self.positions.get(symbol)
            if pos:
                pos["current_price"] = price
                pos["unrealized_pnl"] = self._unrealized(pos, price)
            self._update_drawdown_stats()

    def check_sl_tp(self, current_price: float, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Close the position if SL or TP has been touched.

        * Stop-loss behaves like a stop-market order: if price gapped through
          the stop, the fill happens at the (worse) current price.
        * Take-profit is triggered at the TP level (no positive gap credit,
          which keeps the simulation conservative).
        Slippage and fees are applied in ``close_position``.
        """
        symbol = symbol or self.symbol
        with self._lock:
            self.update_market_price(current_price, symbol)
            pos = self.positions.get(symbol)
            if pos is None:
                return None
            side, sl, tp = pos["side"], pos["sl"], pos["tp"]
            liq = pos.get("liq_price")
            if liq is not None and ((side == "LONG" and current_price <= liq)
                                    or (side == "SHORT" and current_price >= liq)):
                return self._liquidate(pos, symbol)
            self.apply_breakeven(current_price, symbol)
            sl = pos["sl"]
            stop_reason = "BREAKEVEN_STOP" if pos.get("breakeven_moved") else "STOP_LOSS"
            if side == "LONG":
                if current_price <= sl:
                    return self.close_position(current_price, stop_reason, symbol)
                if current_price >= tp:
                    return self.close_position(tp, "TAKE_PROFIT", symbol)
            else:
                if current_price >= sl:
                    return self.close_position(current_price, stop_reason, symbol)
                if current_price <= tp:
                    return self.close_position(tp, "TAKE_PROFIT", symbol)
            return None

    def apply_breakeven(self, favorable_price: float, symbol: Optional[str] = None) -> bool:
        """Move the stop to entry + costs once price reached BREAKEVEN_TRIGGER_R.

        ``favorable_price`` is the best price seen (the ticker live, or the
        bar high/low in a backtest). Returns True if the stop was moved now.
        """
        trigger_r = getattr(config, "BREAKEVEN_TRIGGER_R", 0.0)
        symbol = symbol or self.symbol
        with self._lock:
            pos = self.positions.get(symbol)
            if not pos or trigger_r <= 0 or pos.get("breakeven_moved") or not pos.get("risk_dist"):
                return False
            d = self._direction(pos["side"])
            entry = pos["entry_price"]
            if (favorable_price - entry) * d < trigger_r * pos["risk_dist"]:
                return False
            # Cover both fees and the worst-case exit slippage
            offset = entry * (2 * self.fee_rate + self.slippage_max)
            new_sl = entry + d * offset
            # Only tighten, and never put the stop beyond the current price
            if (new_sl - pos["sl"]) * d <= 0 or (favorable_price - new_sl) * d <= 0:
                return False
            pos["sl"] = new_sl
            pos["breakeven_moved"] = True
            logger.info("BREAKEVEN %s #%d %s: SL %.6g -> %.6g (price %.6g reached %.1fR)",
                        pos["side"], pos["id"], symbol, pos["initial_sl"], new_sl,
                        favorable_price, trigger_r)
            return True

    def _used_margin_unlocked(self) -> float:
        return sum(p.get("margin") or 0.0 for p in self.positions.values())

    def get_used_margin(self) -> float:
        with self._lock:
            return self._used_margin_unlocked()

    def _liquidate(self, pos: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
        """Isolated-margin liquidation: the whole margin is lost (call with lock held)."""
        logger.warning("LIQUIDATION %s #%d %s at %.6g", pos["side"], pos["id"], symbol, pos["liq_price"])
        trade = self.close_position(pos["liq_price"], "LIQUIDATION", symbol)
        if trade and pos.get("margin"):
            # Loss can never exceed the isolated margin (+ fees already paid)
            floor = -(pos["margin"] + trade["entry_fee"] + trade["exit_fee"])
            if trade["net_pnl"] < floor:
                diff = floor - trade["net_pnl"]
                self.balance += diff
                self.daily_realized_pnl += diff
                self.trade_history[-1]["net_pnl"] = trade["net_pnl"] = floor
                self.trade_history[-1]["balance_after"] = trade["balance_after"] = self.balance
        return trade

    def get_equity(self) -> float:
        with self._lock:
            return self._equity_unlocked()

    def get_daily_pnl(self) -> float:
        """Equity change since the start of the current UTC day (incl. unrealised)."""
        with self._lock:
            self._roll_day_if_needed()
            return self._equity_unlocked() - self.day_start_equity

    def get_daily_drawdown_pct(self) -> float:
        """Today's loss as a positive fraction of day-start equity (0 if profitable)."""
        with self._lock:
            pnl = self.get_daily_pnl()
            if self.day_start_equity <= 0:
                return 0.0
            return max(0.0, -pnl / self.day_start_equity)

    def get_trade_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self.trade_history)

    def get_stats(self) -> Dict[str, Any]:
        """Aggregate account and performance statistics (snapshot copy)."""
        with self._lock:
            trades = self.trade_history
            wins = [t for t in trades if t["net_pnl"] > 0]
            losses = [t for t in trades if t["net_pnl"] <= 0]
            gross_win = sum(t["net_pnl"] for t in wins)
            gross_loss = -sum(t["net_pnl"] for t in losses)
            realized = sum(t["net_pnl"] for t in trades)
            # Fee of a still-open position is already paid -> part of realised cash flow
            open_fees = sum(p["entry_fee"] for p in self.positions.values())
            unrealized = sum(p["unrealized_pnl"] for p in self.positions.values())
            equity = self._equity_unlocked()
            n = len(trades)
            return {
                "initial_balance": self.initial_balance,
                "balance": self.balance,
                "equity": equity,
                "unrealized_pnl": unrealized,
                "realized_pnl": realized,
                "open_position_fees": open_fees,
                "total_return_pct": (equity / self.initial_balance - 1) * 100,
                "total_trades": n,
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": (len(wins) / n * 100) if n else 0.0,
                "avg_win": (gross_win / len(wins)) if wins else 0.0,
                "avg_loss": (-gross_loss / len(losses)) if losses else 0.0,
                "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0),
                "total_fees": sum(t["total_fees"] for t in trades) + open_fees,
                "total_slippage": sum(t["slippage_cost"] for t in trades),
                "used_margin": self._used_margin_unlocked(),
                "free_balance": self.balance - self._used_margin_unlocked(),
                "max_drawdown_pct": self.max_drawdown_pct * 100,
                "daily_pnl": equity - self.day_start_equity,
                "day_start_equity": self.day_start_equity,
                "daily_realized_pnl": self.daily_realized_pnl,
            }
