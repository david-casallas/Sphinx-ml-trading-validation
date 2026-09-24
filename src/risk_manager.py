"""
risk_manager.py

Strategy-level risk management for the USDJPY bot.

Responsibilities:
1. Track strategy equity = initial_capital + cumulative PnL from this bot's trades
2. Compute dynamic lot size from the MPO scaling formula (from the position-sizing
   simulation used during development; that notebook is not part of this repository)
3. Manage the drawdown-based circuit breaker (same rule as notebook 03, cell 12)
4. Persist state across restarts

The bot calls:
    - risk.compute_lot_size(current_price) before opening a trade
    - risk.can_open_trade(now) before opening a trade
    - risk.record_close(ticket, pnl_usd, reason, close_dt) when a trade closes
"""

from __future__ import annotations
from typing import Optional
from datetime import datetime, timedelta
from pathlib import Path
import json
import math
import logging


class RiskManager:
    """
    Manages strategy equity, dynamic sizing, and drawdown circuit breaker.
    
    Initial equity is configured via yaml (not read from broker balance).
    This is the strategy-level capital you allocate to this specific bot.
    """
    
    def __init__(
        self,
        # Capital and sizing
        initial_equity: float,
        base_mpo: float,
        leverage: float,
        contract_size: float,
        # Mult formula parameters
        base_for_mult: float = 1.25,
        max_n: float = 6.0,
        min_mult: float = 1.0,
        min_mult_downside: float = 0.3,
        downside_beta: float = 0.5,
        # Circuit breaker parameters
        dd_threshold: float = -0.05,
        cooldown_hours: float = 1.8,
        # Lot size constraints
        min_lot: float = 0.01,
        max_lot: float = 100.0,
        lot_step: float = 0.01,
        # State
        state_path: str = '../data/risk_state.json',
        logger: Optional[logging.Logger] = None,
    ):
        # Capital params (immutable from yaml)
        self.initial_equity = float(initial_equity)
        self.base_mpo = float(base_mpo)
        self.leverage = float(leverage)
        self.contract_size = float(contract_size)
        
        # Mult formula
        self.base_for_mult = float(base_for_mult)
        self.max_n = float(max_n)
        self.min_mult = float(min_mult)
        self.min_mult_downside = float(min_mult_downside)
        self.downside_beta = float(downside_beta)
        
        # Circuit breaker
        self.dd_threshold = float(dd_threshold)
        self.cooldown_hours = float(cooldown_hours)
        
        # Lot constraints
        self.min_lot = float(min_lot)
        self.max_lot = float(max_lot)
        self.lot_step = float(lot_step)
        
        # State path
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Logger
        self.log = logger or logging.getLogger('risk')
        
        # Mutable state (persisted)
        self.cumulative_pnl: float = 0.0
        self.peak_equity: float = self.initial_equity
        self.cooldown_until: Optional[datetime] = None
        self.n_trades_closed: int = 0
        
        # === RETURN-SPACE tracking for the circuit breaker ===
        # The notebook's circuit breaker triggers on cumulative SUMMED RETURN
        # (price-percent space, ~±0.6% per trade), NOT on money drawdown. Money DD
        # is amplified by lot*leverage and would trip the breaker after far fewer
        # losses than the model intends. To match the model exactly, we track a
        # separate cumulative-return series and trigger the breaker off THAT.
        # cum_return: sum of per-trade realized returns (fractions, e.g. -0.006)
        # ret_peak:   running max of cum_return
        # breaker DD = cum_return - ret_peak  (this is what must hit dd_threshold)
        self.cum_return: float = 0.0
        self.ret_peak: float = 0.0
        
        # Deal IDs we've already accounted for (avoid double-counting on restart)
        self.processed_deal_ids: set = set()
        
        self.load_state()
    
    # ============================================================
    # EQUITY ACCESSORS
    # ============================================================
    
    @property
    def current_equity(self) -> float:
        return self.initial_equity + self.cumulative_pnl
    
    @property
    def current_drawdown_pct(self) -> float:
        """
        MONEY drawdown as fraction (e.g. -0.05 = -5%). Always <= 0.
        Used for REPORTING only — NOT for the circuit breaker.
        """
        if self.peak_equity <= 0:
            return 0.0
        return (self.current_equity - self.peak_equity) / self.peak_equity
    
    @property
    def breaker_drawdown(self) -> float:
        """
        RETURN-SPACE drawdown (summed-reward space), matching the notebook's
        circuit breaker exactly. This is what triggers the cooldown. Independent
        of position sizing, so the bot pauses after the same sequence of adverse
        RETURNS as the backtest — not after a sizing-amplified money loss.
        """
        return self.cum_return - self.ret_peak
    
    @property
    def total_return_pct(self) -> float:
        """Total return since inception, as fraction."""
        if self.initial_equity <= 0:
            return 0.0
        return (self.current_equity - self.initial_equity) / self.initial_equity
    
    # ============================================================
    # MULT FORMULA (from the development position-sizing simulation, with discontinuity fix)
    # ============================================================
    
    def _compute_mult(self) -> float:
        """Computes the MPO multiplier from current equity."""
        eq = self.current_equity
        if eq <= 0:
            return 0.0
        
        if eq >= self.initial_equity:
            # Upside: scale up from 1.0 smoothly
            ratio = eq / self.initial_equity
            if ratio <= 1.0:
                return self.min_mult
            ratio_capped = min(ratio, self.max_n)
            mult = math.log(ratio_capped) / math.log(self.base_for_mult)
            return max(self.min_mult, 1.0 + mult)
        else:
            # Downside: shrink lot size when in loss
            fraction_lost = 1.0 - (eq / self.initial_equity)
            return max(self.min_mult_downside, 1.0 - self.downside_beta * fraction_lost)
    
    # ============================================================
    # LOT SIZE COMPUTATION
    # ============================================================
    
    def compute_lot_size(self, current_price: float) -> float:
        """
        Computes the lot size for a new trade based on current equity and price.
        
        Formula:
            MPO_current = base_mpo * Mult(equity)
            exposure_usd = MPO_current * leverage
            lot = exposure_usd / contract_size
        
        For USDJPY: contract_size = 100,000 (1 standard lot = 100k USD nominal)
        For indices: contract_size depends on broker (usually 1 unit per lot)
        
        Returns lot size rounded to lot_step, clamped to [min_lot, max_lot].
        """
        mult = self._compute_mult()
        mpo_current = self.base_mpo * mult
        exposure_usd = mpo_current * self.leverage
        
        lot_raw = exposure_usd / self.contract_size
        
        # Round down to nearest lot_step (don't over-expose)
        lot = math.floor(lot_raw / self.lot_step) * self.lot_step
        
        # Clamp
        lot = max(self.min_lot, min(self.max_lot, lot))
        
        return round(lot, 2)
    
    # ============================================================
    # CIRCUIT BREAKER
    # ============================================================
    
    def can_open_trade(self, now: datetime) -> bool:
        """Returns False if we're currently in cooldown."""
        return not self.is_in_cooldown(now)
    
    def is_in_cooldown(self, now: datetime) -> bool:
        """
        Checks if cooldown is active. Auto-clears the cooldown if it expired.
        """
        if self.cooldown_until is None:
            return False
        if now >= self.cooldown_until:
            self.log.info(f'[RISK] Cooldown ended at {now}. Resuming trading.')
            self.cooldown_until = None
            self.save_state()
            return False
        return True
    
    def _check_and_trigger_cooldown(self, now: datetime) -> None:
        """
        If RETURN-SPACE drawdown <= threshold, start cooldown.
        Uses breaker_drawdown (summed-reward space) to match the notebook exactly,
        NOT money drawdown — so position sizing doesn't change when we pause.
        """
        dd = self.breaker_drawdown
        if dd <= self.dd_threshold and self.cooldown_until is None:
            self.cooldown_until = now + timedelta(hours=self.cooldown_hours)
            self.log.warning(
                f'[RISK] CIRCUIT BREAKER TRIGGERED: return-DD={dd*100:+.2f}% <= '
                f'{self.dd_threshold*100:.2f}%. Cooldown until {self.cooldown_until}. '
                f'(money DD={self.current_drawdown_pct*100:+.2f}%)'
            )
    
    # ============================================================
    # PNL CALLBACK (called by trade_manager on close)
    # ============================================================
    
    def record_close(self, ticket: int, pnl_usd: float, reason: str,
                     close_dt: datetime, deal_id: Optional[int] = None,
                     reward_pct: Optional[float] = None) -> None:
        """
        Called by trade_manager when a position closes.
        
        Updates money equity (for sizing/reporting) AND the return-space series
        (for the circuit breaker). Triggers cooldown off return-space drawdown.
        
        reward_pct: the trade's realized return as a FRACTION (e.g. -0.006 = -0.6%),
                    computed from actual fill prices: (close-open)/open for LONG,
                    (open-close)/open for SHORT. This already includes real spread
                    and slippage. If None, we fall back to deriving it from pnl_usd
                    (less precise), but trade_manager should always supply it.
        deal_id: optional MT5 deal ID for deduplication across restarts.
        """
        # Deduplication
        if deal_id is not None and deal_id in self.processed_deal_ids:
            self.log.debug(f'[RISK] Deal {deal_id} already processed, skipping')
            return
        
        prev_equity = self.current_equity
        self.cumulative_pnl += pnl_usd
        new_equity = self.current_equity
        
        # Update money peak (reporting / sizing only)
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
        
        # === Update RETURN-SPACE series (drives the circuit breaker) ===
        if reward_pct is None:
            # Fallback: approximate the return from money PnL relative to equity
            # BEFORE this trade. Less accurate (uses money, not price-return), but
            # better than nothing if trade_manager didn't supply reward_pct.
            reward_pct = pnl_usd / prev_equity if prev_equity > 0 else 0.0
            self.log.debug('[RISK] reward_pct not supplied; approximated from PnL.')
        self.cum_return += reward_pct
        if self.cum_return > self.ret_peak:
            self.ret_peak = self.cum_return
        
        # Check cooldown trigger (uses return-space drawdown)
        self._check_and_trigger_cooldown(close_dt)
        
        # Track deal as processed
        if deal_id is not None:
            self.processed_deal_ids.add(deal_id)
        
        self.n_trades_closed += 1
        
        self.log.info(
            f'[RISK] Trade #{ticket} closed by {reason}: '
            f'PnL=${pnl_usd:+.2f} reward={reward_pct*100:+.3f}% | '
            f'equity ${prev_equity:.2f}→${new_equity:.2f} | '
            f'peak ${self.peak_equity:.2f} | money_DD={self.current_drawdown_pct*100:+.2f}% | '
            f'breaker_DD={self.breaker_drawdown*100:+.2f}%'
        )
        
        self.save_state()
    
    # ============================================================
    # STATUS LINE FOR HEARTBEAT
    # ============================================================
    
    def status_line(self) -> str:
        eq = self.current_equity
        dd = self.current_drawdown_pct
        bdd = self.breaker_drawdown
        mult = self._compute_mult()
        mpo = self.base_mpo * mult
        
        cool = ''
        if self.cooldown_until is not None:
            cool = f' COOLDOWN_until={self.cooldown_until.strftime("%Y-%m-%d %H:%M")}'
        
        return (
            f'[RISK] eq=${eq:.2f} peak=${self.peak_equity:.2f} '
            f'money_dd={dd*100:+.2f}% breaker_dd={bdd*100:+.2f}% '
            f'mult={mult:.3f} mpo=${mpo:.2f} '
            f'closed={self.n_trades_closed}{cool}'
        )
    
    # ============================================================
    # PERSISTENCE
    # ============================================================
    
    def load_state(self) -> None:
        if not self.state_path.exists():
            self.log.info(f'[RISK] No state file at {self.state_path}, starting fresh.')
            return
        
        try:
            with open(self.state_path, 'r') as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            self.log.warning(f'[RISK] Could not load state: {e}. Starting fresh.')
            return
        
        # Validate that yaml params match what was saved (defensive check)
        saved_initial = d.get('initial_equity')
        if saved_initial is not None and abs(saved_initial - self.initial_equity) > 0.01:
            self.log.warning(
                f'[RISK] WARNING: initial_equity in yaml (${self.initial_equity}) '
                f'differs from saved state (${saved_initial}). Using yaml value but '
                f'keeping cumulative_pnl from state. Verify this is intentional.'
            )
        
        self.cumulative_pnl = float(d.get('cumulative_pnl', 0.0))
        self.peak_equity = float(d.get('peak_equity', self.initial_equity))
        self.n_trades_closed = int(d.get('n_trades_closed', 0))
        
        # Return-space series for the circuit breaker
        self.cum_return = float(d.get('cum_return', 0.0))
        self.ret_peak = float(d.get('ret_peak', 0.0))
        
        cu = d.get('cooldown_until')
        self.cooldown_until = datetime.fromisoformat(cu) if cu else None
        
        deal_ids = d.get('processed_deal_ids', [])
        self.processed_deal_ids = set(deal_ids)
        
        self.log.info(
            f'[RISK] Loaded state: eq=${self.current_equity:.2f}, '
            f'peak=${self.peak_equity:.2f}, closed={self.n_trades_closed}, '
            f'cum_return={self.cum_return*100:+.2f}%, breaker_DD={self.breaker_drawdown*100:+.2f}%, '
            f'cooldown={"YES" if self.cooldown_until else "NO"}'
        )
    
    def save_state(self) -> None:
        d = {
            'initial_equity': self.initial_equity,
            'cumulative_pnl': round(self.cumulative_pnl, 4),
            'peak_equity': round(self.peak_equity, 4),
            'cum_return': round(self.cum_return, 8),
            'ret_peak': round(self.ret_peak, 8),
            'n_trades_closed': self.n_trades_closed,
            'cooldown_until': self.cooldown_until.isoformat() if self.cooldown_until else None,
            'processed_deal_ids': sorted(list(self.processed_deal_ids))[-1000:],  # cap to last 1000
            'updated_at': datetime.now().isoformat(),
        }
        try:
            with open(self.state_path, 'w') as f:
                json.dump(d, f, indent=2)
        except OSError as e:
            self.log.error(f'[RISK] Failed to save state: {e}')
