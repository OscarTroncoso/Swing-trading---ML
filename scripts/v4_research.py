"""V4 research matrix (final branch validation).

The script avoids choosing a winner from one 2026 number. It evaluates simple structural
choices on the same frozen OHLC sample and reports:
- full-period results
- year-by-year results 2020..2026
- positive-year share and median annual return
- risk/leverage utilization

ML is compared as advisory sizing only, never as a hard gate.
"""
from __future__ import annotations
import argparse, json, sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_v4_params
from src.strategy_engine import load_csv
from src.strategy_v4 import V4Params, backtest_v4
from src.ml_v4 import MLV4Params, build_probability_maps


def metric_row(name: str, result: dict, scope: str) -> dict:
    return {"variant": name, "scope": scope, **result["metrics"]}


def yearly_eval(df, p: V4Params, first=2020, last=2026):
    rows = []
    for y in range(first, last + 1):
        start = f"{y}-01-01"
        end = f"{y}-12-31"
        r = backtest_v4(df, p, start=start, end=end)
        rows.append({"year": y, **r["metrics"]})
    return pd.DataFrame(rows)


def summarize_years(name: str, y: pd.DataFrame) -> dict:
    rets = y["totalReturn"].astype(float)
    return {
        "variant": name,
        "years": int(len(y)),
        "positiveYearsPct": round(float((rets > 0).mean() * 100), 2),
        "medianAnnualReturnPct": round(float(rets.median()), 4),
        "meanAnnualReturnPct": round(float(rets.mean()), 4),
        "worstAnnualReturnPct": round(float(rets.min()), 4),
        "bestAnnualReturnPct": round(float(rets.max()), 4),
        "medianSharpe": round(float(y["sharpe"].median()), 4),
        "medianProfitFactor": round(float(y["profitFactor"].median()), 4),
        "maxObservedDrawdownPct": round(float(y["maxDrawdown"].max()), 4),
        "medianClosedTrades": round(float(y["totalTrades"].median()), 2),
        "meanExposurePct": round(float(y["exposurePct"].mean()), 2),
        "meanTradeLeverage": round(float(y["avgTradeLeverage"].mean()), 2),
        "maxTradeLeverageUsed": round(float(y["maxTradeLeverageUsed"].max()), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--first-year", type=int, default=2020)
    ap.add_argument("--last-year", type=int, default=2026)
    ap.add_argument("--out-prefix", default="reports/v4_research")
    args = ap.parse_args()

    df = load_csv(args.csv)
    p = load_v4_params(ROOT / args.config)

    variants = {
        "v4_thesis_rr1.8": p,
        "v4_barrier_only_rr1.8": replace(p, thesis_exit_enabled=False),
        "v4_no_hard_trend_reject": replace(p, hard_trend_reject=False),
        "v4_rr1.0": replace(p, target_rr=1.0),
        "v4_rr1.2": replace(p, target_rr=1.2),
        "v4_rr1.5": replace(p, target_rr=1.5),
        "v4_rr2.0": replace(p, target_rr=2.0),
        "v4_atr_tighter": replace(p, stop_atr_low_vol=1.20, stop_atr_normal=1.35, stop_atr_high_vol=1.55),
        "v4_atr_wider": replace(p, stop_atr_low_vol=1.50, stop_atr_normal=1.70, stop_atr_high_vol=1.90),
        "v4_lev_cap_1x": replace(p, max_trade_leverage=1.0),
        "v4_lev_cap_5x": replace(p, max_trade_leverage=5.0),
        "v4_lev_cap_10x": replace(p, max_trade_leverage=10.0),
        "v4_lev_cap_20x": replace(p, max_trade_leverage=20.0),
        "v4_lev_cap_30x": replace(p, max_trade_leverage=30.0),
        "v4_financing_0pct": replace(p, financing_annual_pct_borrowed=0.0),
        "v4_financing_4pct": replace(p, financing_annual_pct_borrowed=0.04),
        "v4_financing_8pct": replace(p, financing_annual_pct_borrowed=0.08),
        "v4_risk_0.50pct": replace(p, risk_per_trade=.005, min_risk_per_trade=.003, max_risk_per_trade=.0075),
        "v4_risk_0.75pct": p,
        "v4_risk_1.00pct": replace(p, risk_per_trade=.010, min_risk_per_trade=.005, max_risk_per_trade=.0125),
    }

    detail_rows = []
    summary_rows = []

    full_start = f"{args.first_year}-01-01"
    full_end = f"{args.last_year}-12-31"

    for name, vp in variants.items():
        full = backtest_v4(df, vp, start=full_start, end=full_end)
        detail_rows.append(metric_row(name, full, f"{args.first_year}-{args.last_year}"))
        yearly = yearly_eval(df, vp, args.first_year, args.last_year)
        yearly.insert(0, "variant", name)
        yearly.to_csv(ROOT / f"{args.out_prefix}_{name}_annual.csv", index=False)
        summary_rows.append(summarize_years(name, yearly))

    # ML advisory comparison for the base architecture.
    raw = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    m = raw.get("ml", {})
    mlp = MLV4Params(
        retrain_every_bars=m.get("retrainEveryBars",60),
        min_train_events=m.get("minTrainEvents",120),
        validation_fraction=m.get("calibrationFraction",.20),
        max_train_years=m.get("maxTrainYears",8),
        min_validation_auc=m.get("minValidationAUC",.52),
        label_horizon_bars=m.get("labelHorizonBars",60),
        logistic_c=m.get("C",.7),
    )
    probs, healthy, predlog = build_probability_maps(df, p, mlp, start=full_start, end=full_end)
    ml_full = backtest_v4(df, p, start=full_start, end=full_end, ml_probability_by_date=probs, ml_healthy_by_date=healthy)
    detail_rows.append(metric_row("v4_ml_advisory", ml_full, f"{args.first_year}-{args.last_year}"))

    # ML annual result uses the same past-only probability map.
    ml_years = []
    for y in range(args.first_year, args.last_year + 1):
        yy = backtest_v4(
            df, p, start=f"{y}-01-01", end=f"{y}-12-31",
            ml_probability_by_date=probs, ml_healthy_by_date=healthy
        )
        ml_years.append({"year": y, **yy["metrics"]})
    ml_years = pd.DataFrame(ml_years)
    ml_years.insert(0, "variant", "v4_ml_advisory")
    ml_years.to_csv(ROOT / f"{args.out_prefix}_v4_ml_advisory_annual.csv", index=False)
    summary_rows.append(summarize_years("v4_ml_advisory", ml_years))

    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)
    detail_path = ROOT / f"{args.out_prefix}_full.csv"
    summary_path = ROOT / f"{args.out_prefix}_summary.csv"
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    pred = pd.DataFrame(predlog)
    pred.to_csv(ROOT / f"{args.out_prefix}_ml_predictions.csv", index=False)

    print("\n=== FULL PERIOD ===")
    cols = ["variant","totalReturn","maxDrawdown","profitFactor","sharpe","totalTrades","openTrades","avgTradeLeverage","maxTradeLeverageUsed","avgAccountExposureX"]
    print(detail[[c for c in cols if c in detail.columns]].to_string(index=False))
    print("\n=== ANNUAL ROBUSTNESS SUMMARY ===")
    print(summary.sort_values(["positiveYearsPct","medianAnnualReturnPct"], ascending=False).to_string(index=False))
    if len(pred):
        healthy_rate = float(pred["healthy"].mean()*100)
        print(f"\nML healthy predictions: {healthy_rate:.2f}%")
        if "selectedModel" in pred:
            print(pred["selectedModel"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
