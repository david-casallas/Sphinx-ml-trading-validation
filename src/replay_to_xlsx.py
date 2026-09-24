"""
replay_to_xlsx.py — USDJPY version

Replays the model on a USDJPY 15M CSV, bar by bar. For each bar:
  - Builds features (using feature_builder.py — same logic as production)
  - Predicts probability with the model from the .pkl
  - Decides LONG / SHORT / HOLD based on thresholds
  - If a trade is triggered, computes TP/SL/timeout outcome from future bars
  - Writes everything to a single Excel file

Excludes bars in exclude_hours_broker (matching backtest behavior).
NO regime filter (USDJPY doesn't use VXX).

USAGE:
    cd src
    python replay_to_xlsx.py                          # uses defaults below
    python replay_to_xlsx.py --start-date 2026-05-25  # only recent bars
    python replay_to_xlsx.py --csv ../data/X.csv --pkl ../outputs/models/Y.pkl --out ../outputs/Z.xlsx
"""

from __future__ import annotations
import sys
import argparse
import pickle
import pathlib
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from feature_builder import FeatureBuilder


# Default paths matching the USDJPY project structure
DEFAULT_CSV = '../data/USDJPY_15M.csv'
DEFAULT_PKL = '../outputs/models/usdjpy_15m.pkl'
DEFAULT_OUT = '../outputs/replay.xlsx'


def load_csv(path: pathlib.Path) -> pd.DataFrame:
    """Loads MT5-format CSV with renamed columns."""
    df = pd.read_csv(path, sep='\t')
    rename_map = {
        '<DATE>': 'Date', '<TIME>': 'Time',
        '<OPEN>': 'Open', '<HIGH>': 'High', '<LOW>': 'Low', '<CLOSE>': 'Close',
        '<TICKVOL>': 'TickVol', '<VOL>': 'Vol', '<SPREAD>': 'Spread'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    for c in ['Open', 'High', 'Low', 'Close', 'TickVol', 'Spread']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def compute_atr(history: pd.DataFrame, period: int) -> float:
    if len(history) < period + 1:
        return np.nan
    tr = pd.concat([
        history['High'] - history['Low'],
        (history['High'] - history['Close'].shift(1)).abs(),
        (history['Low'] - history['Close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period, min_periods=period).mean().iloc[-1])


def evaluate_trade_outcome(future: pd.DataFrame, direction: str,
                           tp_price: float, sl_price: float,
                           horizon: int) -> dict:
    if len(future) < horizon:
        return {'outcome': 'INSUFFICIENT_FUTURE', 'exit_price': np.nan, 'exit_offset': np.nan}
    
    H = future['High'].values[:horizon]
    L = future['Low'].values[:horizon]
    
    for k in range(horizon):
        if direction == 'LONG':
            tp_hit = H[k] >= tp_price
            sl_hit = L[k] <= sl_price
        else:
            tp_hit = L[k] <= tp_price
            sl_hit = H[k] >= sl_price
        
        if tp_hit and sl_hit:
            return {'outcome': 'TIE_AS_SL', 'exit_price': sl_price, 'exit_offset': k + 1}
        elif tp_hit:
            return {'outcome': 'TP', 'exit_price': tp_price, 'exit_offset': k + 1}
        elif sl_hit:
            return {'outcome': 'SL', 'exit_price': sl_price, 'exit_offset': k + 1}
    
    timeout_close = future['Close'].iloc[horizon - 1]
    return {'outcome': 'TIMEOUT', 'exit_price': float(timeout_close), 'exit_offset': horizon}


def main():
    parser = argparse.ArgumentParser(description='Replay model on USDJPY CSV history')
    parser.add_argument('--csv', default=DEFAULT_CSV, help=f'Path to USDJPY 15M MT5 CSV (default: {DEFAULT_CSV})')
    parser.add_argument('--pkl', default=DEFAULT_PKL, help=f'Path to model .pkl (default: {DEFAULT_PKL})')
    parser.add_argument('--out', default=DEFAULT_OUT, help=f'Output XLSX path (default: {DEFAULT_OUT})')
    parser.add_argument('--start-date', default=None, help='Optional: only process bars from this date (YYYY-MM-DD)')
    args = parser.parse_args()
    
    print(f'[1/5] Loading model from {args.pkl}')
    with open(args.pkl, 'rb') as f:
        payload = pickle.load(f)
    model = payload['model']
    features = payload['features']
    thresholds = payload['thresholds']
    env = payload['env_params']
    
    K_TP = env['k_tp']
    K_SL = env['k_sl']
    HORIZON = env['horizon']
    ATR_PERIOD = env['atr_period']
    EXCLUDE_HOURS = set(env.get('exclude_hours_broker', env.get('exclude_hours_gmt', [])))
    THR_LONG = thresholds['long']
    THR_SHORT = thresholds['short']
    
    print(f'      Symbol: {env.get("symbol", "?")}')
    print(f'      Model: {len(features)} features, K_TP={K_TP}, K_SL={K_SL}, H={HORIZON}')
    print(f'      Thresholds: LONG>{THR_LONG} SHORT<{THR_SHORT}')
    print(f'      Excluded hours: {sorted(EXCLUDE_HOURS)}')
    
    print(f'\n[2/5] Loading CSV from {args.csv}')
    df = load_csv(pathlib.Path(args.csv))
    if args.start_date:
        df = df[df['Datetime'] >= pd.to_datetime(args.start_date)].reset_index(drop=True)
    print(f'      Bars: {len(df):,} | {df["Datetime"].min()} → {df["Datetime"].max()}')
    
    print(f'\n[3/5] Building feature_builder')
    fb = FeatureBuilder(features=features, atr_period=ATR_PERIOD, min_history=250)
    
    print(f'\n[4/5] Running replay...')
    rows = []
    last_progress = 0
    n_total = len(df) - HORIZON
    
    for i in range(250, n_total):
        if i - last_progress >= 5000:
            pct = (i - 250) / (n_total - 250) * 100
            print(f'      {pct:5.1f}%  ({i:,} / {n_total:,})')
            last_progress = i
        
        history = df.iloc[:i + 1]
        next_bar = df.iloc[i + 1]
        
        bar_dt = next_bar['Datetime']
        bar_hour = bar_dt.hour
        
        skip_reason = None
        if bar_hour in EXCLUDE_HOURS:
            skip_reason = 'EXCLUDED_HOUR'
        
        try:
            feat_dict = fb.build_features(history)
            feat_array = np.array([feat_dict[f] for f in features], dtype=np.float64).reshape(1, -1)
        except Exception as e:
            rows.append({
                'Datetime': bar_dt, 'Open': next_bar['Open'], 'High': next_bar['High'],
                'Low': next_bar['Low'], 'Close': next_bar['Close'],
                'skip_reason': f'FEATURE_ERROR: {e}',
                'prob': np.nan, 'decision': 'ERROR',
            })
            continue
        
        prob = float(model.predict(feat_array, num_iteration=model.best_iteration)[0])
        
        if skip_reason:
            decision = 'SKIP'
        elif prob > THR_LONG:
            decision = 'LONG'
        elif prob < THR_SHORT:
            decision = 'SHORT'
        else:
            decision = 'HOLD'
        
        outcome = {'outcome': '', 'exit_price': np.nan, 'exit_offset': np.nan}
        reward_pct = np.nan
        
        if decision in ('LONG', 'SHORT'):
            atr = compute_atr(history, ATR_PERIOD)
            entry_price = next_bar['Open']
            
            if decision == 'LONG':
                tp_price = entry_price + K_TP * atr
                sl_price = entry_price - K_SL * atr
            else:
                tp_price = entry_price - K_TP * atr
                sl_price = entry_price + K_SL * atr
            
            future = df.iloc[i + 1:i + 1 + HORIZON]
            outcome = evaluate_trade_outcome(future, decision, tp_price, sl_price, HORIZON)
            
            if not pd.isna(outcome['exit_price']):
                if decision == 'LONG':
                    reward_pct = (outcome['exit_price'] - entry_price) / entry_price * 100
                else:
                    reward_pct = (entry_price - outcome['exit_price']) / entry_price * 100
        
        row = {
            'Datetime': bar_dt,
            'Open': next_bar['Open'],
            'High': next_bar['High'],
            'Low': next_bar['Low'],
            'Close': next_bar['Close'],
            'prob': prob,
            'decision': decision,
            'skip_reason': skip_reason or '',
            'outcome': outcome['outcome'],
            'exit_price': outcome['exit_price'],
            'exit_offset_bars': outcome['exit_offset'],
            'reward_pct': reward_pct,
        }
        for f in features:
            row[f'feat_{f}'] = feat_dict[f]
        rows.append(row)
    
    print(f'      Total rows: {len(rows):,}')
    
    print(f'\n[5/5] Writing XLSX to {args.out}')
    out_df = pd.DataFrame(rows)
    
    main_cols = ['Datetime', 'Open', 'High', 'Low', 'Close', 'prob', 'decision',
                 'skip_reason', 'outcome', 'exit_price', 'exit_offset_bars', 'reward_pct']
    feat_cols = [c for c in out_df.columns if c.startswith('feat_')]
    out_df = out_df[main_cols + feat_cols]
    
    trades = out_df[out_df['decision'].isin(['LONG', 'SHORT'])].copy()
    summary = pd.DataFrame({
        'metric': [
            'Total bars analyzed',
            'Trades opened (LONG+SHORT)',
            'LONG trades',
            'SHORT trades',
            'HOLD signals',
            'SKIP signals',
            'TP outcomes',
            'SL outcomes',
            'TIMEOUT outcomes',
            'TIE_AS_SL outcomes',
            'Mean reward % per trade',
            'Sum reward %',
            'Hit rate (reward>0)',
            'LONG mean reward %',
            'SHORT mean reward %',
        ],
        'value': [
            len(out_df),
            len(trades),
            (out_df['decision'] == 'LONG').sum(),
            (out_df['decision'] == 'SHORT').sum(),
            (out_df['decision'] == 'HOLD').sum(),
            (out_df['decision'] == 'SKIP').sum(),
            (trades['outcome'] == 'TP').sum(),
            (trades['outcome'] == 'SL').sum(),
            (trades['outcome'] == 'TIMEOUT').sum(),
            (trades['outcome'] == 'TIE_AS_SL').sum(),
            f"{trades['reward_pct'].mean():+.4f}" if len(trades) > 0 else 'N/A',
            f"{trades['reward_pct'].sum():+.2f}" if len(trades) > 0 else 'N/A',
            f"{(trades['reward_pct'] > 0).mean():.4f}" if len(trades) > 0 else 'N/A',
            f"{trades[trades['decision']=='LONG']['reward_pct'].mean():+.4f}" if (trades['decision']=='LONG').any() else 'N/A',
            f"{trades[trades['decision']=='SHORT']['reward_pct'].mean():+.4f}" if (trades['decision']=='SHORT').any() else 'N/A',
        ]
    })
    
    with pd.ExcelWriter(args.out, engine='openpyxl') as writer:
        out_df.to_excel(writer, sheet_name='replay', index=False)
        summary.to_excel(writer, sheet_name='summary', index=False)
        if len(trades) > 0:
            trades.to_excel(writer, sheet_name='trades_only', index=False)
    
    print(f'      Done. {len(out_df):,} rows | {len(trades):,} trades')
    print(f'\n=== SUMMARY ===')
    for _, r in summary.iterrows():
        print(f'  {r["metric"]:<35} {r["value"]}')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
