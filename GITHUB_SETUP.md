# Update your existing GitHub repository to V3.2

1. Back up the current repository or create a branch/tag such as `v3.1-backup`.
2. Copy the contents of this ZIP into the repository root, replacing matching files.
3. Delete the old `archive/` folder if it still exists. It is no longer needed by the UI.
4. Commit and push, for example: `V3.2 trend and structural risk engine`.
5. In GitHub open **Actions → Tests**. It should pass all tests.
6. Run **Actions → Update EURUSD V3.2 dashboard**.
7. Confirm that `data.json`, `data/eurusd_daily.csv`, `reports/backtest_2026.*` and `reports/stop_diagnostics.*` are updated.
8. Run **Actions → Full V3.2 research suite**.
9. Review `reports/walk_forward.json`, `reports/robustness_2026.json`, `reports/v32_ablation.csv` and `reports/stop_diagnostics.json` before changing any default parameter.
10. GitHub Pages remains **Settings → Pages → Deploy from a branch → main → /(root)**.

The Full Research workflow has `contents: write` and commits its reports automatically.
