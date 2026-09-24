# Model methodology V3.1

## Objective

Separate four questions that were previously mixed:

1. Does the technical signal have directional edge?
2. Does ML improve candidate selection?
3. Does sizing improve capital efficiency without unacceptable drawdown?
4. Does the result survive time, costs and parameter perturbations?

## Account accounting

Account currency: EUR.

For EUR/USD, base units are EUR. A trade of `U` units therefore has approximately `U EUR` notional. Quote-currency P&L is:

`PnL_USD = (exit - entry) * units * side`

and account P&L is converted at exit:

`PnL_EUR = PnL_USD / exit`.

## Risk sizing

For each candidate:

`risk_budget_eur = current_equity_eur * applied_risk_pct`

The program estimates EUR loss per unit at the configured stop and chooses units so that stop loss is near that budget, then applies:

- lot-step rounding;
- minimum units;
- absolute-unit cap if configured;
- `maxLeverage * current_equity` notional cap.

Adaptive risk scales the base risk with realized volatility. ML confidence may scale risk only after the candidate passes the ML threshold.

## Technical baseline

Strict directional signal:

- LONG = RSI > rsiLong AND Close > SMA50 AND MACD histogram > 0.
- SHORT = symmetric conditions.

Signal is observed at close and cannot fill until next open.

## ML challenger

The broad candidate generator lowers entry strictness and asks ML whether the setup is worth trading. The model uses only features observable at the signal close.

Meta-label outcome is generated using the same next-open execution and barrier logic used by the strategy. `outcomeEnd` records when that label becomes knowable. A training sample may only contain events with `outcomeEnd < predictionDate`.

This purge rule prevents a trade whose result finishes in the future from leaking into a prediction made today.

## Why logistic regression first

A small regularized logistic model was chosen deliberately before neural networks/XGBoost because:

- event count is limited;
- price-derived indicators are correlated;
- probability calibration matters more than raw classification accuracy;
- coefficients and failure modes are easier to audit;
- it creates a clean baseline against which more complex ML can later be justified.

Nested feature sets test whether extra variables actually improve out-of-sample results.

## Validation hierarchy

1. Unit tests / no-look-ahead tests.
2. 2026 same-sample comparison.
3. Cost and parameter sensitivity.
4. Annual walk-forward strict model.
5. Annual ML out-of-sample comparison.
6. Only then consider promotion of ML or additional complexity.

No script automatically rewrites production parameters based on whichever 2026 variant has the highest return.
