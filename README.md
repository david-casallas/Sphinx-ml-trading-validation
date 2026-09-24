# Sphinx — An ML Trading System and the Validation Framework That Killed It

email: juan.casallas@correounivalle.edu.co
LinkedIn: www.linkedin.com/in/juandacasallas97
GitHub: david-casallas

**A two-year research project in machine-learning trading, reported honestly: the strategy did not beat the market, and this repository documents exactly why.**

Sphinx is a complete, production-grade algorithmic trading system: a LightGBM classifier predicting short-horizon price moves on USDJPY (15-minute bars), a live execution bot for MetaTrader 5 with dynamic position sizing and a drawdown circuit breaker, and — most importantly — a validation framework rigorous enough to demonstrate that the strategy's apparent edge was not real.

The headline result, stated plainly: **on a locked, never-touched test set (10 months of unseen data), the strategy earned +8.2% with a −7.7% maximum drawdown, an edge of +0.009% per trade whose 95% confidence interval touches zero, with ~88% of the profit concentrated in a single week.** An expanding walk-forward showed the edge's sign flips across market regimes. On other instruments (US500, EURGBP), apparent profits were shown to be directional drift — being long in a rising market — rather than timing skill. The project was wound down on this evidence.

The value of this repository is not a profitable strategy. It is (1) a clean, reproducible ML research pipeline with leak-aware methodology, (2) a working production trading system whose live behavior was validated against theory to 100% decision parity, and (3) a worked example of building experiments designed to *disconfirm* one's own hypothesis — and accepting the answer.

---

## Key results (USDJPY 15M, locked test set: Jul 2025 – May 2026)

| Metric | Value |
|---|---|
| Trades (after risk-management cooldown) | 894 |
| Mean return per trade | +0.0092% |
| 95% CI of mean return (bootstrap) | [+0.0002%, +0.0181%] |
| Cumulative return | +8.23% |
| Hit rate | 55.0% |
| Maximum drawdown | −7.72% |
| Return without the single best week | +1.00% |
| Expanding walk-forward consistency | 1 of 3 evaluable folds positive |
| Per-bar predictive power (AUC) | 0.49 – 0.54 across folds |

**Interpretation:** the test technically passed its four pre-registered checks, but marginally. An edge of ~1.4 pips per trade against a ~1.5-pip round-trip cost floor, with regime-dependent sign flips, is not a deployable edge. The full diagnosis is in [`reports/technical_report.pdf`](reports/technical_report.pdf).

---

## Repository structure

```
sphinx/
├── data/                  Data documentation: how to obtain the MT5 price export + checksum
├── notebooks/             The research pipeline, run in order:
│   ├── 01_labeling.ipynb            TP-before-SL labeling with ex-ante ATR
│   ├── 02_features.ipynb            21 features, temporal-leakage discipline
│   ├── 03_model_and_validation.ipynb  Train/dev/test split, model, expanding
│   │                                  walk-forward, locked-test validation
│   │                                  (01–03 committed WITH executed outputs)
│   └── 04_live_vs_replay.ipynb      Live-account vs. theoretical-model comparison
├── src/                   The production bot (MetaTrader 5, Windows)
│   ├── bot.py                       Main loop: candle detection → features → signal → order
│   ├── feature_builder.py           Real-time features, parity-tested to 1e-10 vs research code
│   ├── trade_manager.py             Order lifecycle; bar-count timeout (weekend-safe)
│   ├── risk_manager.py              Dynamic sizing + return-space drawdown circuit breaker
│   ├── mt5_wrapper.py               Broker API isolation layer
│   └── replay_to_xlsx.py            Offline replay of the model on historical CSV
├── tests/                 Unit and parity tests (risk manager runs anywhere; see below)
├── scripts/export_model.py  Reproduces the .pkl model artifact the bot consumes
├── config/usdjpy.example.yaml  Bot configuration template (no credentials)
├── reports/               Technical report (PDF + Markdown) and figures
└── results/summary.md     One-page numerical summary
```

## Reproducing the results

Requirements: Python 3.11+, ~10 minutes of compute.

```bash
pip install -r requirements.txt
jupyter lab
```

The price data is **not included** in the repository (see below). Export it first, following `data/README.md`, and save it as `data/USDJPY_15M.csv`.

Then run the notebooks **in order**: `01_labeling` → `02_features` → `03_model_and_validation`. Each writes its outputs to `data/` and `outputs/`; notebook 03 ends with the locked-test validation and verdict. Notebooks 01–03 are committed *with* their executed outputs, so the results can be inspected without running anything.

The data is a standard MetaTrader 5 history export of USDJPY 15-minute bars, obtainable at no cost from an MT5 demo account. It is not redistributed here because the right to republish a broker's price data has not been verified. `data/README.md` gives the export procedure, the schema, and a SHA-256 checksum to confirm an identical copy.

To reproduce the model artifact the bot consumes: `python scripts/export_model.py`.

## Testing the engineering

The research pipeline runs anywhere. The bot itself requires MetaTrader 5 (Windows) and a broker account, but its core logic is testable offline:

```bash
pytest tests/ -q        # fresh clone: 8 passed, 2 skipped | after notebooks 01–02: 9 passed, 1 skipped
```

`test_risk_manager.py` covers sizing math, the circuit breaker, and state persistence with no broker needed (also runnable directly: `python tests/test_risk_manager.py`). `test_feature_parity.py` verifies that the production feature builder reproduces the research notebook's features to within 1e-10 on a random sample of bars; it needs the parquet and feature list written by notebooks 01–02, and skips until they exist. One known exception is documented in the test: on bars that immediately follow a gap in the price data (2 of the 57,251 modelling rows), the builder's time-of-day features differ, because it infers the next bar's time as the last close + 15 minutes. `test_mt5_regime.py` is a live-connection smoke test that skips automatically on non-Windows platforms.

Live behavior was validated against theory in `notebooks/04_live_vs_replay.ipynb`: across matched demo-account trades, the bot showed **100% direction parity** with the theoretical model and a per-trade reward correlation of **0.999**, with measured entry slippage of +0.78 pips (≈ the broker half-spread).

## Why it failed — the short version

Four findings, each documented in the report and reproducible from the notebooks:

1. **The edge sits at the cost floor.** Theoretical mean return ≈ 1–2 pips/trade; round-trip spread cost ≈ 1.5 pips; measured live slippage consumed ~40% of the theoretical edge.
2. **The edge is regime-dependent.** The expanding walk-forward returned −22%, +10%, −15% across consecutive periods; the locked test (+8%) is one more sample from a sign-flipping process, not a confirmation.
3. **Apparent cross-asset successes were drift, not skill.** On US500 the model was long 10:1 and an exposure-matched random null *itself* earned +25.6% — the "edge" was mostly being long a rising index. The framework's null test was built precisely to catch this.
4. **No economic mechanism.** There is no structural reason a retail participant should be compensated for 15-minute directional bets against institutional counterparties — and the data agree.

## Honesty statement

Nothing in this repository is selected to flatter the strategy. The committed notebook outputs are from the actual final run; the negative walk-forward folds, the marginal confidence interval, and the one-week profit concentration are reported as found. Earlier project iterations contained real bugs (a temporal-leakage error that inflated AUC from 0.50 to 0.72, a 100× transaction-cost unit error on a second instrument, a PnL-attribution bug in the live bot) — these are documented in the report's lessons section because finding and fixing them was most of the work.

## License

MIT (code and documentation). The price data is not included; see `data/README.md`.
