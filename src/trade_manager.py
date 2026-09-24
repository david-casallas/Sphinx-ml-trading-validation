"""
trade_manager.py

Handles trade lifecycle:
- Opening positions with TP/SL based on ATR and DYNAMIC lot size from risk_manager
- BAR-COUNT timeout (matches backtest exactly, pauses during weekend gaps)
- Querying realized PnL from MT5 history → callback to risk_manager
- Persisting state to survive restarts

==========================================================================
TIMEOUT CHANGE (vs previous version)
==========================================================================
Previous: wall-clock deadline (entry_time + H * 15min). Problem: weekend gaps
caused trades to "expire" during market closure and close immediately at reopen.

New: bar-count. We track entry_bar_time per trade. On each tick we ask MT5
for the latest closed 15M bar. If (latest_bar - entry_bar) >= (H-1) bars,
we close the trade. During weekend gaps, no new bars appear → no false
expirations.
==========================================================================
"""

from __future__ import annotations
from typing import Optional, List, Dict, Callable
from datetime import datetime, timedelta
from pathlib import Path
import time
import json

import MetaTrader5 as mt5
from mt5_wrapper import MT5Wrapper, MT5Error

# Broker-time clock function. Imported lazily inside methods to avoid circular import.
def _now():
    try:
        from bot import broker_now
        return broker_now()
    except Exception:
        return datetime.now()


def _floor_to_15m(dt: datetime) -> datetime:
    """Floor a datetime to its containing 15M bar (M15 timestamp = bar OPEN)."""
    minute = (dt.minute // 15) * 15
    return dt.replace(minute=minute, second=0, microsecond=0)


class TradeManager:
    """
    Opens trades with dynamic lot sizing, monitors deadlines by BAR COUNT
    (matches backtest, weekend-safe), closes timed-out positions, and reports
    realized PnL back to risk_manager via on_close callback.
    """
    
    def __init__(
        self,
        mt5_wrapper: MT5Wrapper,
        symbol: str,
        magic: int,
        lot_size_provider: Callable[[float], float],
        k_tp: float,
        k_sl: float,
        horizon_candles: int,
        candle_minutes: int = 15,
        state_path: str = '../data/trade_state.json',
        on_close: Optional[Callable[[int, float, str, datetime, Optional[int]], None]] = None,
    ):
        self.mt5 = mt5_wrapper
        self.symbol = symbol
        self.magic = int(magic)
        self.lot_size_provider = lot_size_provider
        self.k_tp = float(k_tp)
        self.k_sl = float(k_sl)
        self.horizon_candles = int(horizon_candles)
        self.candle_minutes = int(candle_minutes)
        self.state_path = Path(state_path)
        self.on_close = on_close
        
        # tracked[ticket] = {
        #   'entry_bar_time': datetime  (M15 floor of entry moment — the bar where we entered)
        #   'open_dt':       datetime  (actual moment of order placement, wall-clock)
        #   'direction':     str        ('LONG' or 'SHORT')
        #   'lot':           float
        #   'open_price':    float
        # }
        self.tracked: Dict[int, Dict] = {}
        
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
    
    # ============================================================
    # MARKET BAR HELPERS
    # ============================================================
    
    @staticmethod
    def _compute_reward_pct(direction: str, open_price: float, close_price: float) -> float:
        """
        Realized return as a FRACTION, in the same price-percent space the model uses.
          LONG:  (close - open) / open
          SHORT: (open - close) / open
        Computed from actual fill prices, so real spread/slippage is already baked in.
        Returns 0.0 if prices are unusable (defensive).
        """
        if not open_price or open_price <= 0:
            return 0.0
        if direction == 'LONG':
            return (close_price - open_price) / open_price
        elif direction == 'SHORT':
            return (open_price - close_price) / open_price
        return 0.0
    
    def _bars_closed_since(self, entry_bar_time: datetime) -> Optional[int]:
        """
        Counts the number of ACTUAL closed 15M bars in MT5 since (and including)
        entry_bar_time. Uses copy_rates_range so weekend gaps are naturally excluded —
        no bars exist during market closure, so the count doesn't grow.
        
        Returns None if MT5 query fails.
        """
        try:
            now = datetime.now()
            bars = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_M15, entry_bar_time, now)
        except Exception as e:
            print(f'[TM] copy_rates_range failed: {e}')
            return None
        
        if bars is None:
            return None
        
        return int(len(bars))
    
    # ============================================================
    # STATE PERSISTENCE
    # ============================================================
    
    def load_state(self) -> None:
        """Load tracked positions from disk. Reconciles with open positions."""
        if not self.state_path.exists():
            self.tracked = {}
            return
        
        try:
            with open(self.state_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f'[TM] Failed to load state ({e}), starting fresh')
            self.tracked = {}
            return
        
        # Parse — supports both new format and legacy (deadline-only) for migration
        loaded: Dict[int, Dict] = {}
        try:
            if 'tracked' in data:
                # New format
                for t_str, meta in data['tracked'].items():
                    t = int(t_str)
                    loaded[t] = {
                        'entry_bar_time': datetime.fromisoformat(meta['entry_bar_time']),
                        'open_dt': datetime.fromisoformat(meta['open_dt']),
                        'direction': meta['direction'],
                        'lot': float(meta['lot']),
                        'open_price': float(meta['open_price']),
                    }
            elif 'deadlines' in data:
                # Legacy v2 format (deadline + ticket_meta)
                deadlines_old = data['deadlines']
                meta_old = data.get('meta', {})
                for t_str, deadline_iso in deadlines_old.items():
                    t = int(t_str)
                    m = meta_old.get(t_str, {})
                    open_dt = datetime.fromisoformat(m['open_dt']) if 'open_dt' in m else datetime.fromisoformat(deadline_iso) - timedelta(minutes=self.horizon_candles * self.candle_minutes)
                    loaded[t] = {
                        'entry_bar_time': _floor_to_15m(open_dt),
                        'open_dt': open_dt,
                        'direction': m.get('direction', '?'),
                        'lot': float(m.get('lot', 0.0)),
                        'open_price': float(m.get('open_price', 0.0)),
                    }
                print(f'[TM] Migrated {len(loaded)} positions from legacy state format')
            else:
                # Oldest format: {ticket: deadline_iso}
                for t_str, deadline_iso in data.items():
                    t = int(t_str)
                    deadline = datetime.fromisoformat(deadline_iso)
                    open_dt = deadline - timedelta(minutes=self.horizon_candles * self.candle_minutes)
                    loaded[t] = {
                        'entry_bar_time': _floor_to_15m(open_dt),
                        'open_dt': open_dt,
                        'direction': '?',
                        'lot': 0.0,
                        'open_price': 0.0,
                    }
                print(f'[TM] Migrated {len(loaded)} positions from legacy state format')
        except Exception as e:
            print(f'[TM] State parse failed: {e}. Starting fresh.')
            self.tracked = {}
            return
        
        # Reconcile with currently open positions
        open_pos = self.mt5.get_open_positions(magic=self.magic)
        open_tickets = {p['ticket'] for p in open_pos}
        self.tracked = {t: m for t, m in loaded.items() if t in open_tickets}
        
        orphans = open_tickets - set(loaded.keys())
        if orphans:
            print(f'[TM] WARNING: {len(orphans)} open positions without tracking: {orphans}')
        
        self._save_state()
        print(f'[TM] Loaded {len(self.tracked)} tracked positions')
    
    def _save_state(self) -> None:
        data = {
            'tracked': {
                str(t): {
                    'entry_bar_time': m['entry_bar_time'].isoformat(),
                    'open_dt': m['open_dt'].isoformat(),
                    'direction': m['direction'],
                    'lot': m['lot'],
                    'open_price': m['open_price'],
                }
                for t, m in self.tracked.items()
            },
        }
        try:
            with open(self.state_path, 'w') as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            print(f'[TM] Failed to save state: {e}')
    
    # ============================================================
    # PNL QUERY FROM MT5 HISTORY
    # ============================================================
    
    def _get_realized_pnl(self, ticket: int) -> Optional[Dict]:
        """
        Queries MT5 deal history for the given position ticket.
        Returns dict with {pnl_usd, deal_id_exit, close_price, close_dt} or None.
        Defensive filters protect against MT5 API quirks (like returning balance deals).
        """
        try:
            deals = mt5.history_deals_get(position=ticket)
        except Exception as e:
            print(f'[TM] history_deals_get failed for #{ticket}: {e}')
            return None
        
        if deals is None or len(deals) == 0:
            try:
                from_dt = datetime.now() - timedelta(days=7)
                to_dt = datetime.now() + timedelta(days=1)
                deals = mt5.history_deals_get(from_dt, to_dt)
            except Exception as e:
                print(f'[TM] history_deals_get fallback failed for #{ticket}: {e}')
                return None
        
        if deals is None or len(deals) == 0:
            print(f'[TM] No deals found in history for #{ticket}')
            return None
        
        total_pnl = 0.0
        exit_deal = None
        
        for d in deals:
            deal_position_id = getattr(d, 'position_id', None)
            if deal_position_id is None or deal_position_id != ticket:
                continue
            
            deal_type = getattr(d, 'type', None)
            if deal_type is not None and deal_type not in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL):
                continue
            
            deal_symbol = getattr(d, 'symbol', '')
            if deal_symbol and deal_symbol != self.symbol:
                continue
            
            total_pnl += float(d.profit)
            if hasattr(d, 'commission'):
                total_pnl += float(d.commission)
            if hasattr(d, 'swap'):
                total_pnl += float(d.swap)
            
            if d.entry == mt5.DEAL_ENTRY_OUT:
                exit_deal = d
        
        if exit_deal is None:
            print(f'[TM] No matching exit deal found for #{ticket}. PnL retrieval failed.')
            return None
        
        return {
            'pnl_usd': total_pnl,
            'deal_id_exit': int(exit_deal.ticket),
            'close_price': float(exit_deal.price),
            'close_dt': datetime.utcfromtimestamp(exit_deal.time),
        }
    
    # ============================================================
    # TRADE LIFECYCLE
    # ============================================================
    
    def open_trade(self, direction: str, current_price: float, atr: float,
                   comment_extra: str = '') -> Optional[Dict]:
        """Open a new market position. Lot size from risk_manager, timeout by bar count."""
        direction = direction.upper()
        if direction not in ('LONG', 'SHORT'):
            raise ValueError(f'direction must be LONG or SHORT, got {direction}')
        
        try:
            volume = self.lot_size_provider(current_price)
        except Exception as e:
            print(f'[TM] lot_size_provider failed: {e}')
            return None
        
        if volume <= 0:
            print(f'[TM] Skipping trade: lot_size={volume:.4f} (insufficient equity)')
            return None
        
        if direction == 'LONG':
            order_type = 'BUY'
            tp_price = current_price + self.k_tp * atr
            sl_price = current_price - self.k_sl * atr
        else:
            order_type = 'SELL'
            tp_price = current_price - self.k_tp * atr
            sl_price = current_price + self.k_sl * atr
        
        # Determine entry bar (M15 floor of broker_now). This is the bar in which
        # we're entering. Backtest equivalent: "entry at Open of bar X".
        now = _now()
        entry_bar_time = _floor_to_15m(now)
        
        # Comment: encode direction and entry bar epoch for traceability
        entry_bar_epoch = int(entry_bar_time.timestamp())
        comment = f'{direction}_{entry_bar_epoch}'
        if comment_extra:
            comment = f'{comment}_{comment_extra}'
        comment = comment[:31]
        
        try:
            result = self.mt5.place_order(
                symbol=self.symbol,
                order_type=order_type,
                volume=volume,
                sl_price=sl_price,
                tp_price=tp_price,
                magic=self.magic,
                comment=comment,
            )
        except MT5Error as e:
            print(f'[TM] Open trade failed: {e}')
            return None
        
        ticket = result['ticket']
        self.tracked[ticket] = {
            'entry_bar_time': entry_bar_time,
            'open_dt': now,
            'direction': direction,
            'lot': volume,
            'open_price': result['price'],
        }
        self._save_state()
        
        # Compute the "expected" close bar: when H bars have closed since entry,
        # we close. The H-th bar to close after entry is bar (entry + (H-1)*15min).
        expected_close_bar = entry_bar_time + timedelta(minutes=(self.horizon_candles - 1) * self.candle_minutes)
        
        result['direction'] = direction
        result['entry_bar_time'] = entry_bar_time
        result['expected_close_bar'] = expected_close_bar
        result['atr_used'] = atr
        result['lot'] = volume
        
        print(f'[TM] OPENED {direction} #{ticket} '
              f'@ {result["price"]:.5f} lot={volume:.2f} | '
              f'TP={tp_price:.5f} SL={sl_price:.5f} | '
              f'entry_bar={entry_bar_time.strftime("%Y-%m-%d %H:%M")} '
              f'close_after_bar={expected_close_bar.strftime("%H:%M")}')
        return result
    
    def check_and_close_expired(self) -> List[Dict]:
        """
        Scan open positions. Detect natural TP/SL closes. Close positions that
        have lived long enough (by BAR COUNT, not wall-clock).
        
        Bar-count timeout matches backtest exactly:
          - Entry at Open of bar X.
          - Watch bars X through X+H-1 for TP/SL.
          - If no hit, close at the close of bar X+H-1.
          - In real-time: when latest_closed_bar.timestamp >= entry_bar.timestamp + (H-1)*15min, close.
        
        During market closure (weekends), no new bars appear, so no false timeouts.
        """
        open_pos = self.mt5.get_open_positions(magic=self.magic)
        open_tickets = {p['ticket'] for p in open_pos}
        
        events = []
        
        # === 1. Detect natural closes (broker hit TP or SL) ===
        closed_externally = [t for t in list(self.tracked.keys()) if t not in open_tickets]
        for ticket in closed_externally:
            pnl_info = self._get_realized_pnl(ticket)
            if pnl_info is not None:
                pnl = pnl_info['pnl_usd']
                close_price = pnl_info['close_price']
                close_dt = pnl_info['close_dt']
                deal_id = pnl_info['deal_id_exit']
                
                meta = self.tracked.get(ticket, {})
                direction = meta.get('direction', '?')
                open_price = meta.get('open_price', 0)
                if direction == 'LONG':
                    reason = 'TP' if close_price > open_price else 'SL'
                elif direction == 'SHORT':
                    reason = 'TP' if close_price < open_price else 'SL'
                else:
                    reason = 'TP_or_SL'
                
                # Realized return in PRICE-PERCENT space (matches the model's reward).
                # Uses actual fill prices, so real spread/slippage is already included.
                reward_pct = self._compute_reward_pct(direction, open_price, close_price)
                
                print(f'[TM] Position #{ticket} closed naturally by {reason}: '
                      f'PnL=${pnl:+.2f} reward={reward_pct*100:+.3f}% @ {close_price:.5f}')
                
                if self.on_close is not None:
                    try:
                        self.on_close(ticket, pnl, reason, close_dt, deal_id, reward_pct)
                    except Exception as e:
                        print(f'[TM] on_close callback failed: {e}')
                
                events.append({
                    'ticket': ticket, 'reason': reason,
                    'close_price': close_price, 'close_time': close_dt,
                    'pnl_usd': pnl,
                })
            else:
                print(f'[TM] Position #{ticket} closed but could not retrieve PnL')
            
            del self.tracked[ticket]
        
        # === 2. Bar-count timeout check ===
        # Counts ACTUAL closed M15 bars from MT5 — weekend-safe by design.
        if self.tracked:
            for ticket in list(self.tracked.keys()):
                meta = self.tracked[ticket]
                entry_bar_time = meta['entry_bar_time']
                bars_closed = self._bars_closed_since(entry_bar_time)
                
                if bars_closed is None:
                    # Can't determine. Likely MT5 unreachable. Skip this tick.
                    continue
                
                # Backtest convention: enter at Open of bar X, watch bars X..X+H-1,
                # timeout closes at end of bar X+H-1. In actual-bar terms: when H bars
                # have closed since entry (inclusive), trigger close.
                if bars_closed >= self.horizon_candles:
                    try:
                        close_info = self.mt5.close_position(ticket)
                        print(f'[TM] CLOSED #{ticket} by TIMEOUT after {bars_closed} bars '
                              f'@ {close_info["close_price"]:.5f}')
                    except MT5Error as e:
                        # Market may be momentarily closed (broker maintenance window).
                        # Don't spam; just try again next tick.
                        print(f'[TM] Failed to close #{ticket} (will retry next tick): {e}')
                        continue
                    
                    time.sleep(0.5)
                    pnl_info = self._get_realized_pnl(ticket)
                    if pnl_info is not None:
                        pnl = pnl_info['pnl_usd']
                        deal_id = pnl_info['deal_id_exit']
                        close_dt = pnl_info['close_dt']
                        close_price = pnl_info['close_price']
                        
                        direction = meta.get('direction', '?')
                        open_price = meta.get('open_price', 0)
                        reward_pct = self._compute_reward_pct(direction, open_price, close_price)
                        
                        print(f'[TM]   Timeout PnL=${pnl:+.2f} reward={reward_pct*100:+.3f}%')
                        
                        if self.on_close is not None:
                            try:
                                self.on_close(ticket, pnl, 'TIMEOUT', close_dt, deal_id, reward_pct)
                            except Exception as e:
                                print(f'[TM] on_close callback failed: {e}')
                        
                        events.append({
                            'ticket': ticket, 'reason': 'TIMEOUT',
                            'close_price': close_info['close_price'],
                            'close_time': close_dt, 'pnl_usd': pnl,
                        })
                    
                    del self.tracked[ticket]
        
        if closed_externally or events:
            self._save_state()
        
        return events
    
    def get_active_count(self) -> int:
        positions = self.mt5.get_open_positions(magic=self.magic)
        return len(positions)
    
    def status_line(self) -> str:
        n = self.get_active_count()
        n_tracked = len(self.tracked)
        return f'[TM] active_positions={n} | tracked={n_tracked}'
