# Project status

## Ready for GitHub

- Account equity: EUR.
- Default initial equity: €10,000; editable in `config.json`.
- Dynamic risk sizing: enabled.
- Explicit leverage calculation and configurable cap: enabled.
- No-leverage alternative: enabled.
- Strict technical baseline: enabled.
- ML meta-label challenger: enabled.
- Chronological label purge/calibration: enabled.
- 2025/prior-history indicator warm-up with 2026 P&L start: enabled.
- V1 fair comparison: enabled.
- Robustness grid: enabled.
- Exit research: enabled.
- Strict walk-forward: enabled.
- ML annual walk-forward: enabled.
- GitHub Pages dashboard: enabled.
- Automated weekday refresh: enabled.

## Local verification

- Python compile check: passed.
- Pytest: 17/17 passed.
- End-to-end scripts tested with synthetic OHLC for software validation.

Synthetic results are intentionally not stored as strategy evidence. The first GitHub Action run downloads real EUR/USD OHLC and generates the actual 2026 reports.

## Important next files after GitHub Action

- `data/eurusd_daily.csv`
- `reports/backtest_2026_comparison.csv`
- `reports/robustness_2026.csv`
- `reports/ml_research.csv`
- `reports/walk_forward.csv`
- `reports/ml_walk_forward.csv`
