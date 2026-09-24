# Data

The pipeline expects `data/USDJPY_15M.csv`: USD/JPY 15-minute OHLCV bars with broker
spread, May 2022 → May 2026 (99,976 bars). It is a standard MetaTrader 5 history export,
obtainable at no cost from an MT5 demo account.

**The file is not included in this repository.** Downloading the data is free, but the
right to *republish* a broker's price data has not been verified, so it is not redistributed.

## The exact file used

| | |
|---|---|
| Source | ActivTrades MT5 server (demo account), symbol `USDJPY`, timeframe M15 |
| First bar | `2022.05.20 05:30:00` |
| Last bar | `2026.05.27 21:30:00` |
| Rows | 99,976 bars + 1 header line |
| SHA-256 | `c15b2770866862cd41c887ec2342c82902fcd15b9bbeb9d896fd056ae9f9bbac` |

To check that your export is byte-identical to the one used here:

```bash
sha256sum data/USDJPY_15M.csv        # Linux / Git Bash
certutil -hashfile data\USDJPY_15M.csv SHA256   # Windows cmd
```

A file from another broker, or a different date range, will have a different checksum.
The pipeline still runs on it, but the numbers will differ from those reported; the
committed notebook outputs remain the reference.

## Schema (tab-separated)

| Column | Meaning |
|---|---|
| `<DATE>`, `<TIME>` | Bar open timestamp, broker server time (UTC+0 for the broker used) |
| `<OPEN> <HIGH> <LOW> <CLOSE>` | Prices |
| `<TICKVOL>` | Tick volume |
| `<SPREAD>` | Spread in points (1 point = 0.001 for USDJPY) — used for transaction-cost modeling |

## How to export it (any instrument)

1. Open MetaTrader 5 (free demo account from any broker)
2. Tools → Options → Charts → set *Max bars in chart* to Unlimited
3. Open the instrument chart at M15, press Home to load history
4. File → Save As → CSV

## Generated artifacts (not committed)

Running the notebooks creates, in this folder: `USDJPY_15M_target.parquet` (notebook 01),
`USDJPY_15M_features.parquet`, `USDJPY_15M_features_FULL.parquet` and `feature_list.txt`
(notebook 02). They are gitignored; regenerate them by running the notebooks in order.

One detail that matters when adapting to other instruments: the tick size used for cost
modeling is asset-specific (0.001 for JPY pairs, 0.00001 for most other FX pairs, 0.01
for index CFDs). A wrong tick silently scales transaction costs by 100×; this exact
mistake — and how the validation caught it — is documented in the technical report.
