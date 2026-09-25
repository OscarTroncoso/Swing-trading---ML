"""Robust V3.3 parameter research.

Configuration ranking uses only 2020-2025. The selected candidate is then frozen
and reported on 2026. Nothing is auto-promoted to production.
"""
from __future__ import annotations
import argparse, json, sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.config import load_params, load_backtest_settings
from src.strategy_engine import fetch_yahoo, load_csv
from src.v33_engine import backtest_v33, load_v33_policy


def score_years(rows: list[dict]) -> dict:
    ret=np.array([r["totalReturn"] for r in rows],float)
    dd=np.array([r["maxDrawdown"] for r in rows],float)
    sh=np.array([r["sharpe"] for r in rows],float)
    return {
        "medianReturnTrain":float(np.median(ret)),
        "meanReturnTrain":float(np.mean(ret)),
        "positiveYearsTrain":int((ret>0).sum()),
        "medianDDTrain":float(np.median(dd)),
        "medianSharpeTrain":float(np.median(sh)),
        "worstYearReturnTrain":float(np.min(ret)),
        "robustScore":float(np.median(ret)+0.30*np.median(sh)-0.20*np.median(dd)+0.20*(ret>0).mean()),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--config",default="config.json")
    ap.add_argument("--out",default="reports/v33_policy_research.json")
    ap.add_argument("--csv-out",default="reports/v33_policy_research.csv")
    args=ap.parse_args()

    cfg=ROOT/args.config; p=load_params(cfg); base=load_v33_policy(cfg); bc=load_backtest_settings(cfg)
    df=load_csv(args.csv) if args.csv else fetch_yahoo("EURUSD=X",period=bc.get("downloadHistory","10y"))

    configs=[]
    for stop_atr in [1.20,1.50,1.80,2.10]:
      for rr in [1.50,1.80,2.20,2.60,3.00]:
        for soft in [False,True]:
          configs.append(replace(base,stop_atr=stop_atr,target_rr=rr,soft_trend_sizing=soft,max_holding_bars=0))

    rows=[]
    details={}
    for idx,pol in enumerate(configs):
        yearly=[]
        for year in range(2020,2026):
            b=backtest_v33(df,p,pol,start=f"{year}-01-01",end=f"{year}-12-31")
            yearly.append({"year":year,**b["metrics"]})
        s=score_years(yearly)
        row={"configId":idx,"stopATR":pol.stop_atr,"targetRR":pol.target_rr,"softTrendSizing":pol.soft_trend_sizing,**s}
        rows.append(row); details[str(idx)]=yearly

    tab=pd.DataFrame(rows).sort_values(["robustScore","medianReturnTrain"],ascending=False).reset_index(drop=True)
    best_id=int(tab.iloc[0].configId)
    best=configs[best_id]
    test=backtest_v33(df,p,best,start="2026-01-01")
    tab["selectedFor2026Test"]=tab.configId.eq(best_id)
    out_csv=ROOT/args.csv_out; out_csv.parent.mkdir(parents=True,exist_ok=True); tab.to_csv(out_csv,index=False)

    report={
      "selectionRule":"rank on 2020-2025 only; freeze top config; evaluate 2026 once",
      "selected":{
        "configId":best_id,"stopATR":best.stop_atr,"targetRR":best.target_rr,
        "softTrendSizing":best.soft_trend_sizing,"maxHoldingBars":best.max_holding_bars,
      },
      "selectedTrainYearMetrics":details[str(best_id)],
      "selected2026Test":test["metrics"],
      "allConfigs":tab.to_dict(orient="records"),
      "promotionAutomatic":False,
    }
    out=ROOT/args.out; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    print(tab.head(12).to_string(index=False))
    print("\nFrozen 2026 test:")
    print(pd.DataFrame([test["metrics"]]).to_string(index=False))


if __name__=="__main__":
    main()
