"""
feature_builder.py

Builds features in real time, IDENTICALLY to the backtest (notebook 02).

CRITICAL TIME-ALIGNMENT RULE:
=============================
In the backtest, the features of row i use information UP TO THE CLOSE of bar i-1.
Entry is simulated at the Open of bar i.

In production, the equivalent logic is:
  - The bot waits until bar t-1 has fully closed
  - It computes the features from the bars up to t-1 INCLUSIVE
  - It enters at the Open of bar t (the current price at decision time)

Parity with the research pipeline is checked by tests/test_feature_parity.py.
"""

from __future__ import annotations
from typing import List, Dict
import numpy as np
import pandas as pd


ATR_PERIOD_DEFAULT = 50
SMA_WINDOWS = [20, 50, 200]
TENSION_WINDOWS = [50, 200]
RET_WINDOWS = [5, 20, 80]


class FeatureBuilder:
    """
    Builds features aligned with the backtest.
    
    INPUT: DataFrame with columns Datetime, Open, High, Low, Close,
           sorted chronologically, at least 250 bars.
    
    OUTPUT: dict {feature_name: float} for the next bar.
    """
    
    def __init__(self, features: List[str], atr_period: int = ATR_PERIOD_DEFAULT, min_history: int = 250):
        self.features = list(features)
        self.atr_period = atr_period
        self.min_history = min_history
        
        self._needs_atr = any(c.startswith(('dist_sma_', 'tension_', 'vol_', 'slope_', 'curvature', 'regime_')) for c in features)
        self._needs_returns = any(c.startswith('ret_') for c in features)
        self._needs_volret = any('volret' in c for c in features)
        self._needs_microstructure = any(c in features for c in ['body_pct', 'upper_wick', 'lower_wick', 'close_position'])
        self._needs_time = any(c in features for c in ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'])
    
    def _validate_input(self, df: pd.DataFrame) -> None:
        required_cols = ['Datetime', 'Open', 'High', 'Low', 'Close']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f'Missing columns: {missing}')
        if len(df) < self.min_history:
            raise ValueError(f'Insufficient history: {len(df)} < {self.min_history} bars')
        if not df['Datetime'].is_monotonic_increasing:
            raise ValueError('DataFrame is not sorted by ascending Datetime')
    
    def build_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Computes the features for the NEXT bar using information up to df.iloc[-1] inclusive.
        df.iloc[-1] is the last CLOSED bar.
        The returned features are the ones that apply to bar t = (last closed + 1).
        """
        self._validate_input(df)
        d = df.copy().reset_index(drop=True)
        
        # ATR
        if self._needs_atr or 'atr_50' in self.features:
            tr = pd.concat([
                d['High'] - d['Low'],
                (d['High'] - d['Close'].shift(1)).abs(),
                (d['Low'] - d['Close'].shift(1)).abs()
            ], axis=1).max(axis=1)
            d['atr_50'] = tr.rolling(self.atr_period, min_periods=self.atr_period).mean()
            d['atr_200'] = tr.rolling(200, min_periods=200).mean()
        
        # Returns
        if self._needs_returns:
            for n in RET_WINDOWS:
                d[f'ret_{n}'] = d['Close'].pct_change(n)
            rolling_std = d['Close'].pct_change().rolling(200, min_periods=100).std()
            for n in [5, 20, 80]:
                d[f'ret_{n}_z'] = d[f'ret_{n}'] / (rolling_std * np.sqrt(n))
        
        # Mean reversion
        if any(f.startswith('dist_sma_') for f in self.features):
            for n in SMA_WINDOWS:
                sma = d['Close'].rolling(n, min_periods=n).mean()
                d[f'dist_sma_{n}'] = (d['Close'] - sma) / d['atr_50']
        
        # Tension
        if any(f.startswith('tension_') for f in self.features):
            for n in TENSION_WINDOWS:
                rmax = d['High'].rolling(n, min_periods=n).max()
                rmin = d['Low'].rolling(n, min_periods=n).min()
                rmid = (rmax + rmin) / 2
                d[f'tension_{n}'] = 2 * (d['Close'] - rmid) / (rmax - rmin)
        
        # Trend regime
        if any(f.startswith('regime_') for f in self.features):
            sma_20 = d['Close'].rolling(20, min_periods=20).mean()
            sma_50 = d['Close'].rolling(50, min_periods=50).mean()
            sma_200 = d['Close'].rolling(200, min_periods=200).mean()
            d['regime_20_50'] = (sma_20 - sma_50) / d['atr_50']
            d['regime_50_200'] = (sma_50 - sma_200) / d['atr_50']
        
        # Volatility
        if 'vol_ratio' in self.features:
            d['vol_ratio'] = d['atr_50'] / d['atr_200']
        if self._needs_volret:
            ret_1 = d['Close'].pct_change()
            d['volret_50'] = ret_1.rolling(50, min_periods=50).std()
            d['volret_200'] = ret_1.rolling(200, min_periods=200).std()
            d['volret_ratio'] = d['volret_50'] / d['volret_200']
        
        # Curvature
        if any(f in self.features for f in ['slope_short', 'slope_long', 'curvature']):
            ema_5 = d['Close'].ewm(span=5, adjust=False).mean()
            ema_20 = d['Close'].ewm(span=20, adjust=False).mean()
            ema_50 = d['Close'].ewm(span=50, adjust=False).mean()
            d['slope_short'] = (ema_5 - ema_20) / d['atr_50']
            d['slope_long'] = (ema_20 - ema_50) / d['atr_50']
            d['curvature'] = d['slope_short'] - d['slope_short'].shift(5)
        
        # Candle microstructure
        if self._needs_microstructure:
            range_t = (d['High'] - d['Low']).replace(0, np.nan)
            d['body_pct'] = (d['Close'] - d['Open']) / range_t
            d['upper_wick'] = (d['High'] - np.maximum(d['Open'], d['Close'])) / range_t
            d['lower_wick'] = (np.minimum(d['Open'], d['Close']) - d['Low']) / range_t
            d['close_position'] = (d['Close'] - d['Low']) / range_t
        
        # Time features: they apply to the NEXT bar
        if self._needs_time:
            next_dt = d['Datetime'].iloc[-1] + pd.Timedelta(minutes=15)
            hour = next_dt.hour
            dow = next_dt.dayofweek
            time_features = {
                'hour_sin': float(np.sin(2*np.pi*hour/24)),
                'hour_cos': float(np.cos(2*np.pi*hour/24)),
                'dow_sin': float(np.sin(2*np.pi*dow/5)),
                'dow_cos': float(np.cos(2*np.pi*dow/5)),
            }
        else:
            time_features = {}
        
        # Output: equivalent to the backtest's global shift(1) → take d.iloc[-1]
        # This replicates that, in the backtest, the features of row i are those computed at the close of i-1
        result = {}
        last_row = d.iloc[-1]
        for feat in self.features:
            if feat in time_features:
                result[feat] = time_features[feat]
            elif feat in d.columns:
                value = last_row[feat]
                if pd.isna(value):
                    raise ValueError(f'Feature {feat} is NaN. Historical bars are probably missing.')
                result[feat] = float(value)
            else:
                raise ValueError(f'Feature {feat} was not computed. Bug in feature_builder.')
        return result
    
    def build_features_array(self, df: pd.DataFrame) -> np.ndarray:
        d = self.build_features(df)
        return np.array([d[f] for f in self.features], dtype=np.float64)
