# Project status — V3.2

Implemented:
- current-only dashboard and trade history
- EUR account capital and dynamic position sizing
- trend score / market regime
- EMA50/EMA200 / slope / ADX / DMI confirmation
- range rejection
- hybrid structure + volatility stop
- maximum valid stop distance
- volatility-regime ATR scaling
- target based on R
- quality-aware risk sizing
- optional delayed breakeven/trailing (research switch)
- MAE/MFE/R/capture diagnostics
- post-stop false-stop analysis
- V3.2 ablation and robustness grid
- frozen annual walk-forward
- ML validation health gate; ML remains research-only
- 21 automated tests

Next decision gate: run the GitHub Full V3.2 research suite on real EUR/USD OHLC and compare V3.2 with the frozen V3.1 benchmark across annual out-of-sample slices. Do not promote parameter changes from 2026 alone.
