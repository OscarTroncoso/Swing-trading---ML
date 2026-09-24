# GitHub setup — upload and run

1. Create a new GitHub repository, e.g. `forex-swingtrading`.
2. Upload **the contents of this folder** to the repository root. `.github/` must remain at root level.
3. In **Settings → Actions → General → Workflow permissions**, enable **Read and write permissions** so the daily job can refresh generated data.
4. Open **Actions → Tests**. It should pass before running research.
5. Open **Actions → Update EURUSD dashboard → Run workflow**. This downloads one 10-year EUR/USD daily OHLC sample, uses pre-2026 data as warm-up/training, evaluates from 1 January 2026, writes `data.json` and the fair comparison, then commits the refreshed outputs.
6. For deeper testing, run **Actions → Full research suite**. Download the `eurusd-full-research` artifact when complete.
7. To publish the dashboard, enable **Settings → Pages → Deploy from branch → main → /(root)**.

## Change the account size

Edit `config.json`:

```json
"account": { "initialCapitalEUR": 10000.0 }
```

That value is account equity, not EUR/USD units. The engine recalculates units for every setup.

## Change leverage/risk

In the same file:

```json
"risk": {
  "baseRiskPerTrade": 0.005,
  "minRiskPerTrade": 0.0025,
  "maxRiskPerTrade": 0.01,
  "maxLeverage": 3.0
}
```

Use `maxLeverage: 1.0` to prohibit leverage completely. The dashboard always shows both the model risk-target size and a no-leverage alternative.

## What to inspect after the first run

The most useful files are `reports/backtest_2026_comparison.csv`, `reports/improvement_research.csv`, `reports/robustness_2026.csv`, `reports/walk_forward.csv`, `reports/ml_walk_forward.csv` and `reports/ml_research.csv`. Do not promote the row with the highest 2026 return automatically; prioritize stability across cost stress, parameter neighborhoods and unseen annual folds.
