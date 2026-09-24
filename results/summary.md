# Sphinx — Results Summary (one page)

All numbers come from the committed, executed `notebooks/03_model_and_validation.ipynb`
(USDJPY 15M) and `notebooks/04_live_vs_replay.ipynb` (live demo-account validation).

## Setup

Data: USDJPY 15-minute bars, 2022-05 → 2026-05 (~100k bars; ~46k after filtering to
TP/SL-resolved, liquidity-screened observations). Split: 60% train / 20% dev / 20%
locked test, 18-bar embargo between splits. Model: LightGBM binary classifier on
"take-profit hit before stop-loss within 18 bars" (TP/SL = 2.5×ATR(50)). Frozen
decision rule chosen on dev full-data: LONG if p>0.55, SHORT if p<0.45. Transaction
cost: 2× spread per trade (round trip). Risk overlay: trading pauses 4h when the
cumulative summed return draws down 5% from its peak.

## Development-set result (basis for the threshold choice)

1,314 trades, mean +0.0208%/trade, sum +27.28%, bootstrap 95% CI [+0.0101%, +0.0313%].

## Expanding walk-forward (consistency check, train+dev span)

| Fold | Test window | AUC | Trades (full data) | Sum return |
|---|---|---|---|---|
| 0 | 2024-02 → 2024-07 | 0.493 | 1,510 | −22.23% |
| 1 | 2024-07 → 2024-11 | 0.528 | 0 | — |
| 2 | 2024-11 → 2025-03 | 0.538 | 227 | +9.81% |
| 3 | 2025-03 → 2025-07 | 0.544 | 1,687 | −15.03% |

Verdict: inconsistent — 1 of 3 evaluable folds positive; the edge's sign depends on
the market regime.

## Locked test set (2025-07-29 → 2026-05-27, never touched during development)

| Metric | Value |
|---|---|
| Signals / trades taken (post-cooldown) | 2,137 / 894 |
| Mean return per trade | +0.0092% |
| Bootstrap 95% CI of the mean | [+0.0002%, +0.0181%] |
| Sum return / max drawdown | +8.23% / −7.72% |
| Hit rate | 55.0% |
| Exposure-matched null (timing vs drift) | model at percentile 100 (null mean −4.99%) |
| Return without the single best week | +1.00% (≈88% of profit in one week) |
| Within-test halves positive | 2 of 2 |

The test passed its four pre-registered checks, but marginally: the CI lower bound is
+0.0002%, return/maxDD ≈ 1.1, and the profit is heavily concentrated in one week.

## Live execution validation (demo account vs theoretical replay)

21 matched trades: 100% direction parity, per-trade reward correlation 0.999, mean
reward real +0.0125% vs theoretical +0.0130%. Measured entry slippage +0.78 pips
against the trader (95% CI [0.68, 0.89]) — consistent with the broker half-spread,
consuming roughly 40% of the theoretical edge.

## Cross-asset checks (summarized; details in the report)

US500: locked test +54%, but the model was long 10:1 and an exposure-matched random
null itself earned +25.6% — the result is directional drift, not timing skill.
EURGBP: edge appeared only one-sided at extreme thresholds — same diagnosis.

## Conclusion

The strategy's edge (~1.4 pips/trade) sits at the round-trip cost floor (~1.5 pips),
flips sign across regimes, and concentrates in rare episodes. The system was wound
down on this evidence. The negative result is the finding.
