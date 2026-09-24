# Diagnosis of the V3.1 reports used to design V3.2

## What the reports said

The V3.1 adaptive-risk model produced **+2.2494%** in the 2026 evaluation window, with **65.22% win rate**, **1.7783% max drawdown**, **1.7599 profit factor**, **1.0272 Sharpe**, and 23 trades.

A fixed-notional version made more return (+2.8036%) but with materially higher drawdown (2.9369%). A fixed 1.00% risk-per-trade version reached +3.5701%, but average leverage rose to 1.114x, max leverage to 1.463x and drawdown to 3.0624%. This means much of that extra return came from exposure, not a proven improvement in directional edge.

Exit research showed that indiscriminate trailing/breakeven rules were not helpful. The fade+trail challenger preserved similar return (+2.1826%) while improving risk-adjusted statistics (PF 2.322, Sharpe 1.1765, drawdown 1.136%). This motivates delayed/state-dependent profit protection, not a trailing stop from trade inception.

The largest weakness was regime dependence. Frozen annual V3.1 results were positive in 2020, 2021, 2023 and 2026, but negative in 2022, 2024 and 2025. The parameter-selection walk-forward was even less convincing. V3.2 therefore targets regime/trend quality rather than adding another correlated oscillator.

The existing ML meta-label was not promoted. In 2026 it rejected all 156 candidates at the production threshold, and the last reported calibration AUC was below 0.50. V3.2 keeps ML research-only and adds a model-health AUC gate.

## Design response in V3.2

1. EMA50/EMA200 + EMA50 slope + ADX + directional DMI trend score.
2. Explicit rejection of range-like momentum setups.
3. Hybrid ATR + recent-structure stop.
4. Reject trades whose structurally valid stop exceeds a configurable ATR ceiling.
5. Volatility-regime-dependent ATR stop distance.
6. Target expressed in R relative to the actual stop distance.
7. Dynamic EUR risk sizing modulated by trend quality and volatility.
8. MAE/MFE/R-multiple/capture-ratio recorded for every trade.
9. False-stop diagnostics at 3/5/10 bars after SL.
10. Delayed breakeven/trailing logic implemented but disabled by default until it wins walk-forward.
11. Fixed-architecture annual walk-forward; no yearly parameter cherry-picking in the main validation.
12. ML remains research-only until its validation AUC and OOS trade results justify promotion.
