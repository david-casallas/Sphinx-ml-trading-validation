# Sphinx: A Machine-Learning Trading System and an Honest Account of Why It Did Not Beat the Market

**Juan David Casallas** · June 2026 · Code and data: see repository README

## Abstract

Sphinx is a complete machine-learning trading system — research pipeline, trained model, and live execution bot — built over two years to test one question: can a retail participant extract a systematic timing edge from 15-minute currency bars? The answer, established on a locked test set the model never saw and confirmed by three independent diagnostics, is no. The strategy's measured edge (~+0.009% per trade, 95% CI [+0.0002%, +0.0181%]) sits at the transaction-cost floor, flips sign across market regimes, and concentrates almost entirely in rare episodes; apparent successes on other instruments were shown to be directional market drift rather than skill. This report documents the data, the methodology, the results, and — its main contribution — the validation framework designed to disconfirm the strategy, which worked. The negative result is reported as the finding, not hidden as a failure.

## 1. Motivation and question

Machine learning applied to price prediction is a natural attraction for an economist: markets generate enormous, freely available data, and the hypothesis — that short-horizon price movements contain exploitable structure — is precisely testable. The project asked a narrow, falsifiable version of that hypothesis: *given only past prices and a standard retail broker, does a supervised classifier on 15-minute USD/JPY bars produce trading decisions that earn more than their transaction costs, consistently across time?*

The honest prior should have been skeptical. At 15-minute horizons the counterparties are institutional firms with better data, faster execution, and lower costs; financial economics gives no structural reason why a retail participant should be compensated for directional bets at that timescale. Part of what this project documents is the empirical price of testing that prior properly rather than assuming an edge exists because a backtest says so.

## 2. Data and pipeline

**Data.** USD/JPY 15-minute OHLCV bars with broker spread, May 2022 to May 2026 (~100,000 bars), exported from MetaTrader 5. The data is free and the export procedure is documented in the repository, so every result is reproducible from a clean machine.

**Labeling (notebook 01).** Each bar is labeled by a first-passage rule: from the next bar's open, does price reach a take-profit at +2.5×ATR before a stop-loss at −2.5×ATR within 18 bars (4.5 hours)? ATR — the Average True Range, a standard volatility measure — is computed *ex-ante* (using only information available before entry). Bars where neither barrier is hit within the horizon ("timeouts," about a third of all bars) are excluded from training but, crucially, *included* in evaluation, because a live system cannot exclude them. Low-liquidity hours are screened out. This leaves ~46,000 labeled training observations with a near-balanced target.

**Features (notebook 02).** Twenty-one features organized by economic hypothesis: multi-horizon returns normalized by rolling volatility (momentum/reversion), distances between price and moving averages scaled by ATR (trend regime), volatility ratios (regime shifts), candle microstructure (wicks, bodies, ranges), and cyclical time-of-day encodings. Every price-derived feature is shifted one bar so that the model only ever sees information available before the decision moment. This discipline matters: an early version of the pipeline contained a subtle look-ahead error that inflated the model's apparent accuracy (AUC) from 0.50 to 0.72 — a result that would have looked like a discovery and was in fact a bug. Feature selection used mutual information benchmarked against shuffled-noise baselines and intra-block correlation pruning, not predictive performance, to avoid selection leakage.

## 3. Modeling approach

The model is a gradient-boosted tree classifier (LightGBM) predicting the probability that take-profit is reached before stop-loss. Trees were chosen over deep architectures deliberately: tabular features, modest sample size, strong regularization needs, and interpretability via feature importance. Hyperparameters were fixed at conservative values (slow learning rate, 31 leaves, minimum 200 observations per leaf, feature and row subsampling, L2 penalty) and — by design — **never tuned by mass search**. With dozens of free choices (instruments, barriers, horizons, thresholds, hyperparameters), an exhaustive optimizer is statistically guaranteed to produce an impressive backtest by chance alone; avoiding that trap was treated as a methodological requirement, not a missed opportunity.

The classifier's probability becomes a three-way decision through fixed thresholds: long above 0.55, short below 0.45, otherwise abstain. Thresholds were chosen once, on the development set, using the realistic evaluation that includes timeout bars and round-trip costs.

## 4. Validation methodology

This section is the core of the project. Each component exists to give the strategy a specific way to fail.

**Three-way temporal split with embargo.** The data was divided chronologically: 60% training, 20% development (threshold choice, all analysis), 20% test — with an 18-bar gap between segments so that labels reaching into the future cannot leak across the boundary. The test segment was locked: no training, no tuning, no peeking. It was evaluated exactly once, with every parameter frozen, as the final act of the project.

**Realistic accounting.** All reported returns include timeout trades (the optimistic "barriers-only" view is shown alongside precisely to display how much it flatters) and a round-trip transaction cost of twice the recorded spread, charged on every trade — entry and exit each cross the spread.

**Expanding walk-forward.** Beyond a single split, the model was retrained on a growing window (always starting from the first observation, mimicking how a live system accumulates history) and evaluated on consecutive unseen blocks. This asks not "is the average good?" but "is the edge present period after period?" — the question a single backtest number hides.

**An exposure-matched null for timing versus drift.** A strategy that is mostly long while the market rises earns money without skill. To separate timing from drift, the framework simulates thousands of counterfactual strategies that trade the *same bars* with the *same long/short proportions* but with directions assigned at random. This null retains all the directional drift and destroys only the per-bar timing decision. A model with real skill must beat the 95th percentile of this distribution; a drift-capture strategy cannot.

**Distribution stress tests.** Total return is recomputed after removing the best trades and the best week, and the longest "underwater" stretch is measured — because an edge that lives in one lucky episode is not an edge one could have planned to hold through.

**Live-versus-theory validation.** The production bot ran on a demo account, and its actual trades were matched, bar by bar, against an offline replay of the model on the same price history. Across matched trades the bot showed 100% direction parity with the theoretical model and a per-trade reward correlation of 0.999, and the measured entry slippage (+0.78 pips against the trader, 95% CI [0.68, 0.89]) was identified as the broker half-spread rather than adverse execution. This established that any failure of the strategy is a failure of the *idea*, not of the implementation.

## 5. Results

**Development set.** At the chosen thresholds, 1,314 trades, mean +0.0208% per trade, cumulative +27.3%, bootstrap 95% CI [+0.0101%, +0.0313%]. Taken alone, this looks like a discovery. The remaining diagnostics show why it is not.

**Expanding walk-forward.** Across consecutive unseen blocks the strategy returned **−22.2%, (no trades), +9.8%, and −15.0%**, with per-bar AUC between 0.49 and 0.54 — barely above a coin flip. One of three evaluable periods was positive. The edge's *sign* depends on the market regime.

**Locked test (July 2025 – May 2026).** With everything frozen: 894 trades after the risk overlay, mean **+0.0092%** per trade, 95% CI **[+0.0002%, +0.0181%]**, cumulative **+8.23%**, maximum drawdown **−7.72%**, hit rate 55.0%. The exposure-matched null placed the model at its 100th percentile (the null averaged −5.0%), and both chronological halves were positive. The test therefore *passed* its four pre-registered checks — and an honest reading still rejects the strategy: the confidence interval's lower bound is two ten-thousandths of a percent from zero; removing the single best week reduces ten months of profit from +8.23% to **+1.00%** (one week carried ~88% of the result); and the return-to-drawdown ratio of ~1.1 falls below 1 once live slippage (~40% of the theoretical edge, as measured) is applied. A marginal pass on one ten-month sample, drawn from a process the walk-forward showed to be sign-flipping, is not evidence of a deployable edge. It is one more draw from a high-variance, near-zero-mean process.

**Cross-asset replications.** The same pipeline applied to US500 produced a +54% locked-test return — and the framework's null test exposed it: the model was long 10-to-1, and the exposure-matched random null *itself* earned +25.6%, because the index rose strongly over the window. The result was drift wearing the costume of skill. On EURGBP, an apparent edge existed only one-sided and only at the most extreme probability thresholds — the same diagnosis. The consistent pattern across instruments is the project's clearest finding: *the strategy looked successful exactly when it could disguise directional market drift as prediction, and looked marginal when it could not.*

## 6. Why the system did not outperform

Four causes, in order of importance, each supported by the evidence above.

**The edge sits at the cost floor.** The measured edge of roughly 1.4 pips per trade faces a round-trip spread cost of roughly 1.5 pips, and live measurement showed slippage consuming about 40% of the theoretical edge before any adverse selection. Whatever weak signal the features carry, transaction costs absorb it. This is the textbook microstructure outcome for short horizons, observed here directly.

**Regime dependence rather than persistence.** A genuine edge produces many small wins distributed across conditions. This strategy produced long flat or losing stretches punctuated by concentrated profitable episodes — visible in the walk-forward sign flips and in the one-week concentration of the test profit. Such a return profile cannot be distinguished, on available data, from occasionally catching a trend by chance.

**Drift masquerading as timing.** The exposure-matched null demonstrated that the largest apparent successes (US500, EURGBP) were predominantly directional exposure in trending markets. Without that null, the project would have concluded — wrongly — that the method "works on indices."

**No economic mechanism.** The question every strategy must answer — *who is on the other side, and why are they paying you?* — has no good answer at this horizon for a retail participant. The empirical results are exactly what that theoretical null predicts. In hindsight, asking this question on day one would have predicted the outcome of two years of work; it is now the first question I ask of any empirical project.

## 7. What I learned

The methodological lessons generalize well beyond trading. **Sequence disconfirmation first:** the holdout, the null tests, and the walk-forward were built in the project's final phase; built first, they would have returned the same verdict two years earlier at a fraction of the cost. **Pre-register the test:** writing down, before touching data, what result would count as failure is the cheapest protection against fooling oneself. **Distrust monotone improvements toward extreme corners of a parameter grid** — that shape is the signature of selection on noise. **Resist the optimizer:** a mass hyperparameter search over thousands of configurations guarantees a beautiful backtest and proves nothing; declining to run one was the right call even though it left the feeling of an unexplored escape hatch. **Account in the right units:** a 100× transaction-cost error from a wrong tick size, a profit-and-loss attribution bug in the broker API, and a wall-clock-versus-market-time discrepancy in trade exits were each caught only because independent numbers were cross-checked against each other; every one of them would have silently corrupted conclusions. And finally: **an instrument built to say "no" is the most valuable artifact a researcher owns.** This project's framework caught a +54% backtest and correctly labeled it drift. That capability — not the bot — is what the two years actually produced.

## 8. Reproducibility

The repository contains the three research notebooks (committed with their executed outputs, so every reported number can be inspected without running anything), the production bot source with its offline-runnable test suite, and a script that reproduces the exact model artifact. The price data, a standard MT5 history export, is not redistributed; the repository documents how to obtain it and a checksum to verify it. Notebooks 01→02→03 then reproduce every figure and table in a few minutes.

---

*This report describes a system that was wound down after the analysis above. No part of it constitutes investment advice; its purpose is methodological.*
