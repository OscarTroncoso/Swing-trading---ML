# EUR/USD Swing Trading Research v3.1

Research project for an interpretable EUR/USD daily swing strategy with realistic next-bar execution, EUR-denominated account risk, dynamic position sizing, leverage reporting, an ML meta-label challenger, robustness tests and walk-forward validation.

## What changed from the original project

The original prototype treated `10000` as FX units. V3.1 treats **€10,000 as account equity**. EUR/USD units are calculated separately for every trade. For EUR/USD, one unit is one euro of base-currency notional; 100,000 units = 1 standard lot.

The strict live/research baseline remains intentionally small: RSI + SMA50 + MACD direction, ATR stop/target, signal at close `t`, execution at open `t+1`. ML is a **challenger meta-filter**: a broader technical rule proposes direction and logistic meta-labeling estimates whether to accept it. ML never silently replaces the baseline.

## Dynamic position sizing

For risk sizing, the engine starts from a risk budget:

`risk_budget_EUR = current_equity_EUR × applied_risk_pct`

It then computes the exact EUR loss per EUR/USD unit if the stop is hit, estimates desired units, rounds to the configured lot step and applies the leverage cap.

Every setup exposes:
- model equity in EUR;
- applied risk % and estimated EUR loss at stop;
- EUR units and lots;
- notional EUR and effective leverage;
- whether leverage is used;
- target gain and reward/risk;
- a **no-leverage alternative** for comparison.

`adaptive_risk` scales the base risk using realized volatility and, only for the ML challenger, calibrated ML confidence. Direction and exposure are kept conceptually separate.

## 2025 warm-up / 2026 evaluation

`config.json` defines `evaluationStart: 2026-01-01`. The data download contains a much longer history. Indicators are calculated on the full history before the evaluation window is sliced, so 2025 and earlier observations form SMA50, RSI, MACD, ATR and ADX before 1 January 2026. **P&L starts on 1 January 2026; warm-up history does not create pre-2026 P&L.**

The ML challenger needs more than 2025 because it trains on resolved historical candidate events. The standard GitHub workflow therefore downloads 10 years, but 2026 remains the YTD evaluation period.

## ML design

Default ML (`core`) uses seven compact features: RSI centered around 50, distance from SMA50, MACD/ATR, realized volatility, ADX, 5-day return and candidate side. Candidate direction comes from a broad RSI 52/48 + SMA50 rule.

Training is time-safe:
- labels use next-open entry and the same stop/target logic as the trading engine;
- a training event is usable only after its outcome date is strictly before the prediction date;
- retraining occurs every 60 bars;
- chronological Platt calibration is used when there is enough data;
- 2026 is not used to auto-select a threshold or feature set.

`Full research suite` also compares `core`, `compact` and `full` feature sets. This is an ablation test, not automatic feature selection.

## Configure capital and risk

Edit only `config.json` for normal use. Important fields:

```json
"account": { "initialCapitalEUR": 10000.0 },
"risk": {
  "sizingMode": "adaptive_risk",
  "baseRiskPerTrade": 0.005,
  "minRiskPerTrade": 0.0025,
  "maxRiskPerTrade": 0.01,
  "maxLeverage": 3.0
}
```

Examples: `initialCapitalEUR: 25000` starts the simulation with €25,000. `maxLeverage: 1.0` forbids leverage. A 3.0 cap means the model may use at most €3 notional per €1 of model equity; it does **not** assert what a broker will allow.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python scripts/update_data.py --period 10y --backtest-start 2026-01-01
python scripts/backtest_2026.py --csv data/eurusd_daily.csv --start 2026-01-01
```

Open the project through a local web server, for example `python -m http.server 8000`, then visit `http://localhost:8000`.

## GitHub Actions

`Tests` runs on every push. `Update EURUSD dashboard` runs manually or after the daily FX session and refreshes `data.json`, the exact OHLC sample, strict/ML equity curves and the fair 2026 comparison. `Full research suite` is manual and runs sizing/leverage/exit attribution, the 324-config baseline stress grid, baseline annual walk-forward, ML annual walk-forward and ML feature/threshold ablation.

## Main folders

```text
src/                 core model, sizing, ML and experimental exits
scripts/             data update, backtests and research jobs
tests/               leakage/sizing/chronology tests
reports/             generated research results
archive/v1/          original project preserved for audit
archive/v2-exploratory/ earlier exploratory work
.github/workflows/   CI, daily update and full research suite
```

## Important interpretation

A larger return caused only by larger position size is **not a new edge**. The research reports separate signal changes, exit changes, risk sizing and leverage. ML is kept as a challenger until multi-year out-of-sample evidence supports promotion. Daily OHLC backtests cannot reproduce tick-level fills, news gaps or broker-specific swap/financing perfectly; those belong in the later broker/execution validation stage.
