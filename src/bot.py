"""
bot.py

Main orchestrator for the USDJPY 15M trading bot.

NEW IN THIS VERSION:
- Integrates risk_manager.py for:
    - Strategy-level equity tracking
    - Dynamic lot sizing (MPO * Mult / contract_size)
    - Drawdown circuit breaker (-5% DD → cooldown; 4h in the validated configuration)
- trade_manager uses lot_size_provider callback (dynamic per trade)
- on_close callback feeds realized PnL back to risk_manager
"""

from __future__ import annotations
import sys
import csv
import time
import signal
import pickle
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from mt5_wrapper import MT5Wrapper, MT5Error
from feature_builder import FeatureBuilder
from regime_filter import RegimeFilter
from trade_manager import TradeManager
from risk_manager import RiskManager

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# ============================================================
# CONFIGURATION HELPERS
# ============================================================

def load_yaml_config(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('bot')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    
    return logger


# ============================================================
# BROKER TIME HELPERS (timezone-independent)
# ============================================================

_BROKER_OFFSET = timedelta(0)


def broker_now() -> datetime:
    return datetime.utcnow() + _BROKER_OFFSET


def calibrate_broker_offset(wrapper, symbol: str) -> timedelta:
    global _BROKER_OFFSET
    df = wrapper.get_bars(symbol, 'M1', 5)
    if df is None or len(df) == 0:
        return _BROKER_OFFSET
    
    bar_broker_time = df.iloc[-1]['Datetime']
    real_utc = datetime.utcnow()
    
    bar_minute = bar_broker_time.replace(second=0, microsecond=0)
    utc_minute = real_utc.replace(second=0, microsecond=0)
    
    offset = bar_minute - utc_minute
    rounded_hours = round(offset.total_seconds() / 3600)
    _BROKER_OFFSET = timedelta(hours=rounded_hours)
    
    return _BROKER_OFFSET


# ============================================================
# THE BOT
# ============================================================

class TradingBot:
    LOOP_SLEEP_SEC = 5
    MARKET_FRESHNESS_LIMIT_MIN = 20
    HISTORY_BARS = 500
    RECONNECT_BACKOFF_BASE_SEC = 30
    RECONNECT_BACKOFF_MAX_SEC = 600
    HEARTBEAT_MIN = 15
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = load_yaml_config(config_path)
        
        # Logging
        log_path = Path(self.config['logs']['file'])
        self.log = setup_logging(log_path)
        self.log.info('=' * 70)
        self.log.info('BOT STARTING')
        self.log.info(f'Config: {config_path}')
        
        # Load model payload
        model_pkl = Path(self.config['model']['pkl'])
        with open(model_pkl, 'rb') as f:
            self.payload = pickle.load(f)
        
        self.model = self.payload['model']
        self.features = self.payload['features']
        self.thresholds = self.payload['thresholds']
        self.env = self.payload['env_params']
        self.regime_cfg = self.payload['regime_filter']
        
        self.symbol = self.env['symbol']
        self.k_tp = self.env['k_tp']
        self.k_sl = self.env['k_sl']
        self.horizon = self.env['horizon']
        self.atr_period = self.env['atr_period']
        self.exclude_hours = set(self.env.get('exclude_hours_broker', self.env.get('exclude_hours_gmt', [])))
        
        self.log.info(f'Model loaded: {self.symbol} {self.env["timeframe"]}')
        self.log.info(f'Features: {len(self.features)}')
        self.log.info(f'Thresholds: LONG>{self.thresholds["long"]} | SHORT<{self.thresholds["short"]}')
        self.log.info(f'K_TP={self.k_tp}, K_SL={self.k_sl}, H={self.horizon}')
        self.log.info(f'Excluded broker hours: {sorted(self.exclude_hours)}')
        
        # Risk management config (from yaml)
        rm_cfg = self.config['risk_management']
        self.log.info(f'Risk: initial_eq=${rm_cfg["initial_equity"]} | '
                      f'base_mpo=${rm_cfg["base_mpo"]} | leverage=1:{rm_cfg["leverage"]:.0f} | '
                      f'dd_threshold={rm_cfg["dd_threshold"]*100:.1f}% | '
                      f'cooldown={rm_cfg["cooldown_hours"]}h')
        
        # Components (initialized after MT5 connect)
        self.wrapper: Optional[MT5Wrapper] = None
        self.feature_builder: Optional[FeatureBuilder] = None
        self.regime: Optional[RegimeFilter] = None
        self.risk: Optional[RiskManager] = None
        self.tm: Optional[TradeManager] = None
        
        # State
        self.last_bar_time: Optional[datetime] = None
        self.last_heartbeat: datetime = broker_now()
        self.shutdown_requested = False
        self.consecutive_disconnects = 0
        
        # CSV trade event log (separate from text log)
        log_dir = Path(self.config['logs']['file']).parent
        self.csv_log_path = log_dir / 'trade_events.csv'
        self._init_csv_log()
        
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        self.log.info(f'Signal {signum} received, shutting down gracefully')
        self.shutdown_requested = True
    
    # === CSV trade event log ===
    
    def _init_csv_log(self) -> None:
        """Create CSV log with headers if it doesn't exist."""
        if not self.csv_log_path.exists():
            self.csv_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_log_path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow([
                    'timestamp_utc', 'broker_time', 'event',
                    'ticket', 'direction', 'prob', 'price', 'lot',
                    'pnl_usd', 'outcome',
                    'equity', 'peak', 'dd_pct', 'mult', 'mpo',
                    'notes'
                ])
    
    def _log_event(self, event: str, **kwargs) -> None:
        """Append one row to the CSV trade event log."""
        try:
            row = [
                datetime.utcnow().isoformat(timespec='seconds'),
                broker_now().isoformat(timespec='seconds'),
                event,
                kwargs.get('ticket', ''),
                kwargs.get('direction', ''),
                kwargs.get('prob', ''),
                kwargs.get('price', ''),
                kwargs.get('lot', ''),
                kwargs.get('pnl_usd', ''),
                kwargs.get('outcome', ''),
                round(self.risk.current_equity, 2) if self.risk else '',
                round(self.risk.peak_equity, 2) if self.risk else '',
                round(self.risk.current_drawdown_pct * 100, 4) if self.risk else '',
                round(self.risk._compute_mult(), 3) if self.risk else '',
                round(self.risk.base_mpo * self.risk._compute_mult(), 2) if self.risk else '',
                kwargs.get('notes', ''),
            ]
            with open(self.csv_log_path, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            self.log.error(f'CSV log write failed: {e}')
    
    def _on_trade_close(self, ticket: int, pnl_usd: float, reason: str,
                       close_dt, deal_id=None, reward_pct=None) -> None:
        """Callback fired when any position closes. Updates risk + CSV log."""
        self.risk.record_close(ticket, pnl_usd, reason, close_dt, deal_id, reward_pct)
        self._log_event('CLOSE', ticket=ticket, pnl_usd=round(pnl_usd, 2), outcome=reason,
                        notes=f'reward={reward_pct*100:+.3f}%' if reward_pct is not None else '')
    
    # === Connection management ===
    
    def connect(self) -> bool:
        try:
            acct = self.config['account']
            self.wrapper = MT5Wrapper(
                login=int(acct['login']),
                password=acct['password'],
                server=acct['server'],
                path=acct.get('path'),
            )
            self.wrapper.connect()
            
            offset = calibrate_broker_offset(self.wrapper, self.symbol)
            self.log.info(f'Broker time offset: UTC{offset.total_seconds()/3600:+.0f}h | broker_now={broker_now()}')
            
            self.feature_builder = FeatureBuilder(
                features=self.features,
                atr_period=self.atr_period,
                min_history=250,
            )
            
            if self.config.get('regime_filter', {}).get('enabled', True):
                     self.regime = RegimeFilter(
                         mt5_wrapper=self.wrapper,
                         symbol=self.regime_cfg['symbol'],
                         sma_short=self.regime_cfg['sma_short'],
                         sma_long=self.regime_cfg['sma_long'],
                         threshold_diff=self.regime_cfg['threshold_diff'],
                         timeframe=self.regime_cfg.get('timeframe', 'D1'),
                     )
            else:
                    self.regime = None
                    self.log.info('Regime filter DISABLED in config')
            # Risk manager
            rm_cfg = self.config['risk_management']
            self.risk = RiskManager(
                initial_equity=rm_cfg['initial_equity'],
                base_mpo=rm_cfg['base_mpo'],
                leverage=rm_cfg['leverage'],
                contract_size=rm_cfg['contract_size'],
                base_for_mult=rm_cfg.get('base_for_mult', 1.25),
                max_n=rm_cfg.get('max_n', 6.0),
                min_mult=rm_cfg.get('min_mult', 1.0),
                min_mult_downside=rm_cfg.get('min_mult_downside', 0.3),
                downside_beta=rm_cfg.get('downside_beta', 0.5),
                dd_threshold=rm_cfg.get('dd_threshold', -0.05),
                cooldown_hours=rm_cfg.get('cooldown_hours', 1.8),
                min_lot=rm_cfg.get('min_lot', 0.01),
                max_lot=rm_cfg.get('max_lot', 100.0),
                lot_step=rm_cfg.get('lot_step', 0.01),
                state_path=self.config['logs'].get('risk_state_file', '../data/risk_state.json'),
                logger=self.log,
            )
            
            # Trade manager: uses risk_manager for lot sizing and on_close
            trade_cfg = self.config['trade']
            self.tm = TradeManager(
                mt5_wrapper=self.wrapper,
                symbol=self.symbol,
                magic=int(trade_cfg['magic']),
                lot_size_provider=self.risk.compute_lot_size,
                k_tp=self.k_tp,
                k_sl=self.k_sl,
                horizon_candles=self.horizon,
                candle_minutes=15,
                state_path=self.config['logs'].get('state_file', '../data/trade_state.json'),
                on_close=self._on_trade_close,
            )
            self.tm.load_state()
            
            self.consecutive_disconnects = 0
            self.log.info('All components initialized')
            self.log.info(self.risk.status_line())
            return True
        except (MT5Error, Exception) as e:
            self.log.error(f'Connection failed: {e}')
            self.consecutive_disconnects += 1
            return False
    
    def reconnect_with_backoff(self) -> bool:
        backoff = min(
            self.RECONNECT_BACKOFF_BASE_SEC * (2 ** self.consecutive_disconnects),
            self.RECONNECT_BACKOFF_MAX_SEC,
        )
        self.log.warning(f'Disconnected. Retrying in {backoff}s (attempt #{self.consecutive_disconnects + 1})')
        
        for _ in range(backoff):
            if self.shutdown_requested:
                return False
            time.sleep(1)
        
        if self.wrapper:
            try:
                self.wrapper.disconnect()
            except Exception:
                pass
        
        return self.connect()
    
    # === State detection ===
    
    def is_market_fresh(self, latest_bar_time: datetime) -> bool:
        age = broker_now() - latest_bar_time
        return age < timedelta(minutes=self.MARKET_FRESHNESS_LIMIT_MIN)
    
    def is_excluded_hour(self, broker_dt: datetime) -> bool:
        return broker_dt.hour in self.exclude_hours
    
    # === Signal evaluation ===
    
    def evaluate_and_trade(self, history: pd.DataFrame) -> None:
        # Build features
        try:
            features_dict = self.feature_builder.build_features(history)
            features_array = np.array([features_dict[f] for f in self.features], dtype=np.float64).reshape(1, -1)
        except Exception as e:
            self.log.error(f'Feature build failed: {e}')
            return
        
        # Predict
        try:
            prob = float(self.model.predict(features_array, num_iteration=self.model.best_iteration)[0])
        except Exception as e:
            self.log.error(f'Prediction failed: {e}')
            return
        
        # Decision
        thr_long = self.thresholds['long']
        thr_short = self.thresholds['short']
        
        if prob > thr_long:
            decision = 'LONG'
        elif prob < thr_short:
            decision = 'SHORT'
        else:
            decision = 'HOLD'
        
        last_close_dt = history.iloc[-1]['Datetime']
        self.log.info(
            f'SIGNAL last_close={last_close_dt.strftime("%H:%M")} | '
            f'prob={prob:.4f} | thr=[{thr_short},{thr_long}] → {decision}'
        )
        
        # CSV log every signal (including HOLD)
        self._log_event('SIGNAL', direction=decision, prob=round(prob, 4),
                        notes=f'last_close={last_close_dt.strftime("%H:%M")}')
        
        if decision == 'HOLD':
            return
        
        # === CIRCUIT BREAKER CHECK ===
        if not self.risk.can_open_trade(broker_now()):
            self.log.warning(
                f'  Trade BLOCKED by circuit breaker. {self.risk.status_line()}'
            )
            self._log_event('BLOCKED_COOLDOWN', direction=decision, prob=round(prob, 4))
            return
        
        # Get current price + ATR
        try:
            tick = self.wrapper.get_current_tick(self.symbol)
        except MT5Error as e:
            self.log.error(f'Tick fetch failed: {e}')
            return
        
        current_price = tick['ask'] if decision == 'LONG' else tick['bid']
        
        atr = self._compute_atr(history)
        if atr is None or atr <= 0:
            self.log.error(f'Invalid ATR: {atr}')
            return
        
        # Open trade (trade_manager will get dynamic lot from risk_manager)
        result = self.tm.open_trade(
            direction=decision,
            current_price=current_price,
            atr=atr,
            comment_extra=f'p{int(prob*100)}',
        )
        
        if result:
            slippage = result['price'] - current_price if decision == 'LONG' else current_price - result['price']
            self.log.info(f'  Slippage: {slippage:+.5f} pts | Reference: {current_price:.5f} | Lot: {result["lot"]:.2f}')
            self._log_event('OPEN',
                            ticket=result['ticket'], direction=decision,
                            prob=round(prob, 4), price=result['price'],
                            lot=result['lot'],
                            notes=f'slippage={slippage:+.5f}')
    
    def _compute_atr(self, history: pd.DataFrame) -> Optional[float]:
        if len(history) < self.atr_period + 1:
            return None
        tr = pd.concat([
            history['High'] - history['Low'],
            (history['High'] - history['Close'].shift(1)).abs(),
            (history['Low'] - history['Close'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_series = tr.rolling(self.atr_period, min_periods=self.atr_period).mean()
        last_atr = atr_series.iloc[-1]
        return float(last_atr) if not pd.isna(last_atr) else None
    
    # === Heartbeat ===
    
    def maybe_log_heartbeat(self, state: str, extra: str = '') -> None:
        now = broker_now()
        if (now - self.last_heartbeat).total_seconds() >= self.HEARTBEAT_MIN * 60:
            tm_status = self.tm.status_line() if self.tm else '[TM] not init'
            risk_status = self.risk.status_line() if self.risk else '[RISK] not init'
            regime_status = self.regime.status_line() if self.regime else '[REGIME] disabled'
            self.log.info(f'HEARTBEAT state={state} | {tm_status} | {risk_status} | {regime_status} | {extra}')
            self.last_heartbeat = now
    
    # === Main loop ===
    
    def run(self) -> None:
        while not self.connect():
            if self.shutdown_requested:
                return
            self.reconnect_with_backoff()
        
        self.log.info('Entering main loop')
        
        while not self.shutdown_requested:
            try:
                self._tick()
            except MT5Error as e:
                self.log.error(f'MT5 error in tick: {e}')
                if not self.reconnect_with_backoff():
                    if self.shutdown_requested:
                        break
            except Exception as e:
                self.log.exception(f'Unexpected error in tick: {e}')
            
            for _ in range(self.LOOP_SLEEP_SEC):
                if self.shutdown_requested:
                    break
                time.sleep(1)
        
        self._cleanup()
    
    def _tick(self) -> None:
        # 1. Always check timeouts (regardless of state)
        try:
            self.tm.check_and_close_expired()
        except MT5Error as e:
            self.log.error(f'Timeout check failed: {e}')
            raise
        
        # 2. Fetch latest bar
        try:
            df = self.wrapper.get_bars(self.symbol, self.env['timeframe'], self.HISTORY_BARS)
        except MT5Error as e:
            self.log.error(f'Bar fetch failed: {e}')
            raise
        
        if len(df) < 250:
            self.log.warning(f'Only {len(df)} bars available, need 250')
            self.maybe_log_heartbeat('INSUFFICIENT_HISTORY')
            return
        
        latest_bar_dt = df.iloc[-1]['Datetime']
        
        # 3. State detection
        if not self.is_market_fresh(latest_bar_dt):
            self.maybe_log_heartbeat('MARKET_CLOSED', f'last_bar={latest_bar_dt}')
            return
        
        # The backtest excludes a trade based on the ENTRY bar's hour, and entry
        # happens at the bar that opens AFTER the last closed bar. So we must gate
        # on (last_closed + 15min), not on the last closed bar itself. Gating on the
        # closed bar shifts the exclusion window 15 min early and breaks bot/theory parity.
        entry_bar_dt = latest_bar_dt + timedelta(minutes=15)
        if self.is_excluded_hour(entry_bar_dt):
            self.maybe_log_heartbeat('EXCLUDED_HOUR', f'entry_hour={entry_bar_dt.hour}')
            return
        
        # 4. Detect new candle
        if self.last_bar_time is None:
            self.last_bar_time = latest_bar_dt
            self.log.info(f'Initialized last_bar_time={latest_bar_dt}. Will trade on next new candle.')
            self.maybe_log_heartbeat('TRADING')
            return
        
        if latest_bar_dt <= self.last_bar_time:
            self.maybe_log_heartbeat('TRADING', f'last_bar={latest_bar_dt}')
            return
        
        # New candle closed!
        self.log.info(f'NEW CANDLE closed @ {latest_bar_dt}')
        self.last_bar_time = latest_bar_dt
        
        # 5. Regime check (skip if disabled in config)
        if self.regime is not None:
            try:
                if not self.regime.can_trade():
                    self.log.info(f'  Regime blocks trading. {self.regime.status_line()}')
                    return
            except (MT5Error, Exception) as e:
                self.log.error(f'Regime check failed: {e}. Defaulting to NO TRADE for safety.')
                return
        
        # 6. Evaluate and trade (this checks risk_manager cooldown internally)
        self.evaluate_and_trade(df)
    
    def _cleanup(self) -> None:
        self.log.info('Cleaning up...')
        if self.risk:
            try:
                self.risk.save_state()
            except Exception:
                pass
        if self.wrapper:
            try:
                self.wrapper.disconnect()
            except Exception:
                pass
        self.log.info('BOT STOPPED')


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='USDJPY 15M trading bot')
    parser.add_argument('--config', '-c', default='../config/usdjpy.yaml',
                        help='Path to YAML config file')
    args = parser.parse_args()
    
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f'Config not found: {config_path}')
        sys.exit(1)
    
    bot = TradingBot(config_path)
    bot.run()


if __name__ == '__main__':
    main()
