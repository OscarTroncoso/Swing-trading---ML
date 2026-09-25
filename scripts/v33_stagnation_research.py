"""Research dynamic stale-trade exits without imposing a fixed holding period.

A position may stay open beyond the review horizon if it has made sufficient MFE
or the technical thesis still has at least two of RSI/SMA50/MACD confirmations.
Selection uses 2020-2025 only; 2026 is a frozen test.
"""
from __future__ import annotations
import argparse, json, sys
from dataclasses import replace, asdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.config import load_params, load_backtest_settings
from src.strategy_engine import fetch_yahoo, load_csv, backtest_v31_benchmark
from src.v33_engine import load_v33_policy, backtest_v33


def score(rows):
    ret=np.array([r["totalReturn"] for r in rows],float)
    dd=np.array([r["maxDrawdown"] for r in rows],float)
    sh=np.array([r["sharpe"] for r in rows],float)
    pf=np.array([r["profitFactor"] for r in rows],float)
    return {
      "medianReturnTrain":float(np.median(ret)),"meanReturnTrain":float(np.mean(ret)),
      "positiveYearsTrain":int((ret>0).sum()),"worstYearTrain":float(ret.min()),
      "medianDDTrain":float(np.median(dd)),"medianSharpeTrain":float(np.median(sh)),
      "medianPFTrain":float(np.median(pf)),
      "robustScore":float(np.median(ret)+.35*np.median(sh)-.20*np.median(dd)+.25*(ret>0).mean()),
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--csv"); ap.add_argument("--config",default="config.json")
    ap.add_argument("--out",default="reports/v33_stagnation_research.json")
    ap.add_argument("--csv-out",default="reports/v33_stagnation_research.csv")
    args=ap.parse_args()
    cfg=ROOT/args.config; p=load_params(cfg); base=load_v33_policy(cfg); bc=load_backtest_settings(cfg)
    df=load_csv(args.csv) if args.csv else fetch_yahoo("EURUSD=X",period=bc.get("downloadHistory","10y"))

    policies=[]
    for bars in [3,5,8,12,20]:
      for min_mfe in [.25,.50,.75]:
        for require_weak in [True,False]:
          policies.append(replace(base,max_holding_bars=0,stagnation_bars=bars,
                                  stagnation_min_mfe_r=min_mfe,stagnation_min_close_r=0.0,
                                  stagnation_require_weak_thesis=require_weak))

    rows=[]; detail={}
    for idx,pol in enumerate(policies):
        yrs=[]
        for year in range(2020,2026):
            b=backtest_v33(df,p,pol,start=f"{year}-01-01",end=f"{year}-12-31")
            yrs.append({"year":year,**b["metrics"]})
        rows.append({"configId":idx,"stagnationBars":pol.stagnation_bars,
                     "stagnationMinMFER":pol.stagnation_min_mfe_r,
                     "requireWeakThesis":pol.stagnation_require_weak_thesis,**score(yrs)})
        detail[str(idx)]=yrs

    tab=pd.DataFrame(rows).sort_values(["robustScore","medianReturnTrain"],ascending=False).reset_index(drop=True)
    best_id=int(tab.iloc[0].configId); best=policies[best_id]
    test=backtest_v33(df,p,best,start="2026-01-01")
    static=backtest_v33(df,p,replace(base,stagnation_bars=0,max_holding_bars=0),start="2026-01-01")
    legacy=backtest_v31_benchmark(df,p,start="2026-01-01")
    tab["selectedFor2026Test"]=tab.configId.eq(best_id)
    outcsv=ROOT/args.csv_out; outcsv.parent.mkdir(parents=True,exist_ok=True); tab.to_csv(outcsv,index=False)
    report={
      "selectionRule":"dynamic stagnation parameters selected on 2020-2025 only; 2026 tested once",
      "definition":"No fixed max holding. Exit next open only if review age reached, MFE remains below threshold, current R < 0, and optionally RSI/SMA50/MACD thesis has fewer than 2 confirmations.",
      "selectedPolicy":{"stagnationBars":best.stagnation_bars,"stagnationMinMFER":best.stagnation_min_mfe_r,
                        "requireWeakThesis":best.stagnation_require_weak_thesis,"maxHoldingBars":0},
      "selectedTrainYears":detail[str(best_id)],
      "legacyFiveBar2026":legacy["metrics"],
      "staticTPSL2026":static["metrics"],
      "selectedDynamic2026":test["metrics"],
      "allPolicies":tab.to_dict(orient="records"),
      "promotionAutomatic":False
    }
    (ROOT/args.out).write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    print(tab.head(12).to_string(index=False))
    print("\n2026 comparison")
    print(pd.DataFrame([{"variant":"legacy_5bar",**legacy["metrics"]},{"variant":"static_tp_sl",**static["metrics"]},{"variant":"dynamic_stagnation",**test["metrics"]}]).to_string(index=False))


if __name__=="__main__": main()
