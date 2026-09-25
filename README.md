# Swing-trading---ML — V3.3 research branch

EUR/USD daily swing-research project. V3.3 keeps the interpretable V3.1 technical direction core and rebuilds risk, exits and ML around the actual trade objective.

## Current V3.3 assumptions

- Account balance is denominated in **EUR**; default research balance: **€1,000**.
- Position size is displayed as **stake EUR**, not FX units.
- Gross exposure = stake EUR × leverage.
- **x1** means exposure equals the cash stake (spot-equivalent convention in this simulator).
- Leverage search is integer **x1 … x30**.
- The selected leverage is the highest integer whose estimated loss at the stop remains inside the configured EUR risk budget. It is not a return forecast.
- Directional core: RSI + SMA50 + MACD.
- Trend/volatility context scales size softly by default; it does not hard-veto the technical signal.
- Default trade lifecycle has **no fixed-day exit**. A trade remains open until TP/SL or a market-state exit explicitly enabled by research.
- A position still open at the end of a backtest window is marked to market and reported as OPEN instead of being force-liquidated.

## ML

The old ML target used a five-bar TIME exit and was misaligned with the intended strategy.

V3.3 labels a candidate as:
- `1`: TP is reached before SL;
- `0`: SL is reached before TP;
- unresolved events: censored / not labeled.

Research compares:
- regularized Logistic Regression;
- Random Forest;
- HistGradientBoosting.

Validation is chronological and purged: a training event is admitted only if its TP/SL outcome was already known before the test block starts.

ML never invents BUY/SELL direction and never hard-rejects a V3.3 signal. A healthy ML model may only adjust the stake/risk modestly.

## Important files

- `src/v33_engine.py` — V3.3 execution, EUR stake, leverage and TP/SL engine.
- `src/ml_v33.py` — TP-before-SL ML research.
- `config.json` — account, stake, leverage, risk and execution assumptions.
- `scripts/backtest_v33.py` — V3.1 vs V3.2 vs V3.3 comparison.
- `scripts/v33_policy_research.py` — stop/RR search using 2020–2025, then frozen 2026 test.
- `scripts/v33_exit_research.py` — non-time exit research.
- `scripts/ml_v33_research.py` — ML family comparison and soft-sizing test.

## GitHub Actions

### Tests
Runs chronology, anti-look-ahead, EUR stake, leverage, exit and ML tests.

### Update EURUSD V3.3 dashboard
Refreshes market data and the V3.3 dashboard.

### Full V3.3 research suite
Freezes one 10-year OHLC sample and runs all V3.3 research on the same data.

## Development status

V3.3 is developed in branch `v3.3-risk-ml-rework` and PR #2 remains a draft until its out-of-sample evidence is reviewed. Do not treat a higher in-sample return as sufficient reason to promote a configuration.

Research software only; not financial advice or broker execution.
