# Swing-trading---ML — V4 research branch

EUR/USD daily swing-research project. V4 is designed around a **EUR-denominated account**, **EUR trade allocation**, explicit **x1–x30 trade leverage**, soft trend context, adaptive ATR/structure stops, and leakage-controlled ML used only as an advisory risk scaler.

## Why V4 exists

V3.2 over-filtered the 2026 sample: the trend gate and wide-stop rejection removed many otherwise valid technical candidates. The previous ML meta-label also behaved like an almost total hard gate. V4 separates four questions:

1. **Direction** — RSI + SMA50 + MACD technical signal.
2. **Context** — trend/ADX changes trade quality and risk; only an extreme opposing regime can block a signal.
3. **Risk geometry** — ATR + recent price structure sets SL/TP. If structure is too far away, V4 falls back to ATR rather than rejecting the trade.
4. **Capital allocation** — position is shown in EUR. Leverage is chosen from x1 to x30 to use, but not exceed, the trade risk budget.

## Position / leverage semantics

Default research account:

- Account equity: **€1,000**
- Base allocated position: **€100**
- x1: €100 position -> €100 market exposure (fully funded / spot-equivalent)
- x10: €100 position -> €1,000 market exposure
- x30: €100 position -> €3,000 market exposure

Leverage is **not** selected by highest historical return. For each entry V4 computes the stop first, establishes a risk budget from account equity, and then chooses the highest allowed leverage whose loss at the stop remains inside that budget. The dashboard shows the alternatives and the selected leverage.

A simplified financing stress is charged to the borrowed part of exposure (`notional - positionEUR`) while a leveraged trade is open. Research compares 0%, 4% and 8% annual financing assumptions.

## Exit rule

There is **no forced 3-day or 5-day exit** in V4.

A trade stays open until:

- Stop Loss, or
- Take Profit.

If the evaluation period ends while a position remains active, it stays `OPEN`; final equity is marked to market instead of inventing an end-of-period trade close.

## ML in V4

ML does not create direction and does not reject trades.

The research layer compares:

- regularized logistic regression
- histogram gradient boosting

using chronological validation and only labels whose outcomes were already resolved before each prediction date. A healthy model can scale the risk budget modestly. If validation AUC is below the health threshold, ML abstains and the technical strategy is unchanged.

The V4 label is aligned with the strategy objective: **TP before SL** within a research labeling horizon. Unresolved events are excluded from training.

## Run locally

```bash
pip install -r requirements.txt
pytest -q

python scripts/update_data.py --period 10y --backtest-start 2026-01-01 --out data.json --market-csv-out data/eurusd_daily.csv
python scripts/backtest_2026.py --csv data/eurusd_daily.csv --start 2026-01-01
python scripts/v4_research.py --csv data/eurusd_daily.csv --first-year 2020 --last-year 2026
```

## Research outputs

- `reports/backtest_2026_comparison.csv` — V3.1 vs V4 vs V4+ML advisory
- `reports/backtest_2026.json` — detailed trades/predictions
- `reports/v4_research_full.csv` — full 2020–2026 variant comparison
- `reports/v4_research_summary.csv` — annual robustness summary
- `reports/v4_research_*_annual.csv` — yearly result for each variant
- `reports/v4_research_ml_predictions.csv` — past-only ML model/AUC/probability log

Variants include R/R, ATR stop width, leverage caps x1/x5/x10/x20/x30, financing assumptions, risk budgets, and ML advisory sizing.

## Important files

- `src/strategy_v4.py` — V4 strategy, EUR sizing, leverage, financing, SL/TP-only backtest
- `src/ml_v4.py` — chronological logistic vs histogram-gradient-boosting advisory layer
- `src/strategy_engine.py` — V3.2 retained for research / indicator compatibility
- `src/baseline_v31.py` — frozen V3.1 benchmark
- `config.json` — account, EUR position, leverage, risk, trend, stop, ML settings
- `scripts/update_data.py` — current V4 dashboard generator
- `scripts/backtest_2026.py` — same-data V3.1/V4 comparison
- `scripts/v4_research.py` — multi-year architecture/leverage/ML research

## Validation principles

- Signal at close t, execution at next open.
- Signal-bar ATR/trend/structure inputs are frozen before next-open execution.
- Spread and slippage are included.
- Stop gaps are handled at the next available open rather than the ideal stop price.
- Leveraged financing is stressed explicitly.
- ML training is chronological and purged by outcome completion date.
- 2026 is not used to auto-promote a parameter solely because it has the highest return.

Research software only; not financial advice or a broker execution system.
