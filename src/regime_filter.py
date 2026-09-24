"""
regime_filter.py

Determines whether the current market regime allows trading.

Logic (from a development notebook that is not part of this repository):
    - Query VXX.US daily bars from MT5
    - Calculate SMA20 and SMA200 of VXX Close
    - If SMA20 - SMA200 > THRESHOLD_DIFF, the regime is "high uncertainty"
    - During high uncertainty, the bot does NOT open new trades

Important:
    - Uses VXX from yesterday's close (shift of 1 day) to avoid lookahead.
      We can't know today's VXX close at the start of the trading day.
    - Caches the daily result. Refreshes only when the date changes.
"""

from __future__ import annotations
from typing import Optional, Dict
from datetime import datetime, date
import pandas as pd
from mt5_wrapper import MT5Wrapper, MT5Error


class RegimeFilter:
    """
    Provides a binary trade/no-trade decision based on VXX regime.
    
    Usage:
        regime = RegimeFilter(mt5_wrapper, symbol='VXX.US',
                              sma_short=20, sma_long=200, threshold_diff=10)
        if regime.can_trade():
            # proceed with model decision
        else:
            # skip this candle
    """
    
    def __init__(
        self,
        mt5_wrapper: MT5Wrapper,
        symbol: str = 'VXX.US',
        sma_short: int = 20,
        sma_long: int = 200,
        threshold_diff: float = 10.0,
        timeframe: str = 'D1',
    ):
        self.mt5 = mt5_wrapper
        self.symbol = symbol
        self.sma_short = int(sma_short)
        self.sma_long = int(sma_long)
        self.threshold_diff = float(threshold_diff)
        self.timeframe = timeframe
        
        # Cache
        self._cached_date: Optional[date] = None
        self._cached_can_trade: Optional[bool] = None
        self._cached_details: Optional[Dict] = None
    
    def _compute_regime(self) -> Dict:
        """
        Fetch VXX daily, compute SMAs, return regime info.
        
        Returns dict with:
            can_trade: bool
            vxx_close: float       (yesterday's close)
            sma_short: float       (SMA20 calculated up to yesterday)
            sma_long: float        (SMA200 calculated up to yesterday)
            diff: float            (sma_short - sma_long)
            high_uncertainty: bool (diff > threshold)
        """
        # Need at least sma_long + 1 bars (the +1 is to allow shift)
        n_bars = self.sma_long + 50
        
        df = self.mt5.get_bars(self.symbol, self.timeframe, n_bars)
        if len(df) < self.sma_long + 1:
            raise MT5Error(
                f'Insufficient VXX history: got {len(df)}, need {self.sma_long + 1}. '
                f'Make sure {self.symbol} is available in your MT5 with enough history.'
            )
        
        # Calculate SMAs
        df['sma_short'] = df['Close'].rolling(self.sma_short, min_periods=self.sma_short).mean()
        df['sma_long'] = df['Close'].rolling(self.sma_long, min_periods=self.sma_long).mean()
        
        # IMPORTANT: shift(1) so we use yesterday's values for today's decision
        df['sma_short_lag'] = df['sma_short'].shift(1)
        df['sma_long_lag'] = df['sma_long'].shift(1)
        df['close_lag'] = df['Close'].shift(1)
        
        # Last row has the most recent values (yesterday's SMAs)
        last = df.iloc[-1]
        
        sma_s = float(last['sma_short_lag'])
        sma_l = float(last['sma_long_lag'])
        close_y = float(last['close_lag'])
        diff = sma_s - sma_l
        high_unc = diff > self.threshold_diff
        
        return {
            'can_trade': not high_unc,
            'vxx_close': close_y,
            'sma_short': sma_s,
            'sma_long': sma_l,
            'diff': diff,
            'threshold': self.threshold_diff,
            'high_uncertainty': high_unc,
            'computed_at': datetime.now(),
            'data_date': last['Datetime'],
        }
    
    def can_trade(self, force_refresh: bool = False) -> bool:
        """
        Returns True if trading is allowed in the current regime.
        
        Caches the result per date. Use force_refresh=True to skip cache.
        """
        today = datetime.now().date()
        
        if not force_refresh and self._cached_date == today and self._cached_can_trade is not None:
            return self._cached_can_trade
        
        info = self._compute_regime()
        self._cached_date = today
        self._cached_can_trade = info['can_trade']
        self._cached_details = info
        
        return info['can_trade']
    
    def get_details(self) -> Optional[Dict]:
        """Return the cached regime details (or compute fresh if no cache)."""
        if self._cached_details is None:
            self.can_trade()
        return self._cached_details
    
    def status_line(self) -> str:
        """One-line status message for logs/Telegram."""
        d = self.get_details()
        if d is None:
            return '[REGIME] not computed yet'
        return (
            f'[REGIME] VXX={d["vxx_close"]:.2f} '
            f'SMA{self.sma_short}={d["sma_short"]:.2f} '
            f'SMA{self.sma_long}={d["sma_long"]:.2f} '
            f'diff={d["diff"]:+.2f} '
            f'(threshold={d["threshold"]:.0f}) → '
            f'{"NO TRADE (high uncertainty)" if d["high_uncertainty"] else "TRADE OK"}'
        )
