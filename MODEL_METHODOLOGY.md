# EUR/USD V3.2 — Trend + Structural Risk Engine

## Current model

The production/research dashboard shows **only V3.2**. Historical V3.1 outputs are kept only under `reports/input-v3.1/` for audit and are not part of the live UI.

### 1. Candidate direction

Long candidate:
- RSI > configured long threshold
- close > SMA50
- MACD histogram > 0

Short candidate is symmetric.

### 2. Trend quality / regime

Each candidate receives a 0–5 trend score from:
- price vs EMA200
- EMA50 vs EMA200
- EMA50 slope over N bars
- directional DMI (+DI vs -DI)
- ADX above threshold

A low-ADX + flat-EMA condition is classified as range-like and rejected for the momentum strategy.

### 3. Stop construction

The initial stop is the farther of:
- volatility stop = dynamic ATR multiple
- recent market structure + ATR buffer

The stop must remain within `maxStopATR`. If market structure requires a wider stop, the trade is skipped rather than forcing a poor stop or increasing risk.

ATR multiple changes with a past-only volatility regime:
- low volatility
- normal volatility
- high volatility

### 4. Target

Take-profit is defined as a multiple of the **actual stop distance** (`targetRR`), preserving a consistent R framework when structural stops widen or narrow.

### 5. Position sizing

EUR account equity is the base. Units are calculated from:
- current equity
- target risk %
- exact EUR loss at the stop
- volatility scaling
- trend-quality scaling
- maximum leverage cap

The dashboard reports units, lots, EUR notional, EUR risk, effective leverage and whether leverage is used.

### 6. Execution chronology

- Signal uses completed close t.
- Trend, ATR, structure, volatility and risk inputs are frozen at close t.
- Entry occurs at open t+1 plus modeled spread/slippage.
- SL/TP checks use daily High/Low.
- If SL and TP touch in the same daily candle, SL is assumed first.

### 7. Profit protection

Delayed breakeven/trailing is implemented but disabled in the default configuration because earlier trailing tests harmed results. Research can enable it after a position has already earned a configurable R multiple. Stop changes take effect on the next bar only.

### 8. Diagnostics

Every trade records:
- R multiple
- MAE in R
- MFE in R
- winner capture ratio
- trend score/regime
- volatility regime
- structural vs ATR stop source
- stop distance in ATR

`stop_diagnostics.py` measures how often an SL would subsequently have reached its original TP within 3, 5 and 10 sessions.

### 9. Machine learning

ML is **not part of the current production signal**. It remains a research-only meta-label challenger. A validation AUC health gate makes the model abstain when its most recent chronological validation has insufficient discrimination.

### 10. Promotion rule

No 2026 winner is automatically promoted. A modification should improve a combination of return, expectancy, Sharpe/Sortino, PF and drawdown and should remain credible in fixed-architecture annual walk-forward and parameter robustness tests.
