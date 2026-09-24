"""
export_model.py — Reproduce the model artifact (.pkl) the production bot consumes.

Trains the SAME frozen model that was validated in notebooks/03_model_and_validation.ipynb:
  - trained on the TRAIN split only (first 60% of filtered data, embargo applied)
  - early stopping on the last 20% of TRAIN
  - frozen decision thresholds and risk parameters baked into the payload

This deliberately exports the *validated* model, not one retrained on all data,
so the artifact matches the model whose locked-test results are reported.

Prerequisites: run notebooks 01 and 02 first (they create the parquet + feature list).

Usage:
    python scripts/export_model.py
"""

from __future__ import annotations
import pathlib
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'USDJPY_15M_features.parquet'
FEATS = ROOT / 'data' / 'feature_list.txt'
OUT = ROOT / 'outputs' / 'models' / 'usdjpy_15m.pkl'

# === Frozen configuration (must match notebook 03) ===
TRAIN_FRAC = 0.60
EMBARGO = 18
THRESHOLDS = {'long': 0.55, 'short': 0.45}
ENV_PARAMS = {
    'symbol': 'USDJPY', 'timeframe': '15M',
    'k_tp': 2.5, 'k_sl': 2.5, 'horizon': 18, 'atr_period': 50,
    'tick': 0.001, 'exclude_hours_broker': [20, 21, 22, 23],
}
RISK_PARAMS = {'dd_threshold': -0.05, 'cooldown_hours': 4.0,
               'note': 'circuit breaker operates in RETURN space (summed per-trade returns)'}
LGB_PARAMS = {
    'objective': 'binary', 'metric': 'binary_logloss',
    'learning_rate': 0.02, 'num_leaves': 31, 'min_data_in_leaf': 200,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'lambda_l2': 1.0, 'verbose': -1, 'num_threads': -1, 'seed': 42,
}


def main() -> int:
    assert DATA.exists(), f'Missing {DATA} — run notebooks 01 and 02 first.'
    with open(FEATS) as f:
        features = [ln.strip() for ln in f if ln.strip()]

    df = pd.read_parquet(DATA).sort_values('Datetime').reset_index(drop=True)
    n = len(df)
    train = df.iloc[:int(n * TRAIN_FRAC) - EMBARGO]
    vs = int(len(train) * 0.8)

    dtr = lgb.Dataset(train.iloc[:vs][features].values,
                      label=train.iloc[:vs]['target'].values, feature_name=features)
    des = lgb.Dataset(train.iloc[vs:][features].values,
                      label=train.iloc[vs:]['target'].values, feature_name=features,
                      reference=dtr)
    model = lgb.train(LGB_PARAMS, dtr, num_boost_round=2000,
                      valid_sets=[des],
                      callbacks=[lgb.early_stopping(100, verbose=False)])

    p_es = model.predict(train.iloc[vs:][features].values,
                         num_iteration=model.best_iteration)
    auc = roc_auc_score(train.iloc[vs:]['target'].values, p_es)

    payload = {
        'model': model,
        'features': features,
        'thresholds': THRESHOLDS,
        'env_params': ENV_PARAMS,
        'risk_params': RISK_PARAMS,
        # bot.py reads this key. The USDJPY model does not use the VXX regime filter
        # (see regime_filter in config/usdjpy.example.yaml, which must be disabled).
        'regime_filter': None,
        'lgb_params': LGB_PARAMS,
        'meta': {
            'created_at': datetime.now().isoformat(),
            'train_rows': int(len(train)),
            'best_iteration': int(model.best_iteration),
            'auc_early_stop_split': float(auc),
            'honest_note': ('This model did NOT demonstrate a deployable edge on the '
                            'locked test set; see reports/technical_report.pdf.'),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'wb') as f:
        pickle.dump(payload, f)

    print(f'Exported: {OUT}')
    print(f'  features={len(features)} | best_iteration={model.best_iteration} | '
          f'AUC(early-stop split)={auc:.4f}')
    print(f'  thresholds={THRESHOLDS} | horizon={ENV_PARAMS["horizon"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
