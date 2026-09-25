"""V3.3 exit research with no fixed-day liquidation.

All candidate policies stay open until TP/SL or a market-state event (break-even,
trailing, opposite signal). Selection uses 2020-2025 only; 2026 is frozen test.
"""
from __future__ import annotations
import argparse,json,sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.config import load_params,load_backtest_settings
from src.strategy_engine import fetch_yahoo,load_csv
from src.v33_engine import backtest_v33,load_v33_policy


def score(rows):
    ret=np.array([r["totalReturn"] for r in rows],float)
    dd=np.array([r["maxDrawdown"] for r in rows],float)
    sh=np.array([r["sharpe"] for r in rows],float)
    pf=np.array([r["profitFactor"] for r in rows],float)
    return {
      "medianReturnTrain":float(np.median(ret)),
      "meanReturnTrain":float(np.mean(ret)),
      "positiveYearsTrain":int((ret>0).sum()),
      "medianDDTrain":float(np.median(dd)),
      "medianSharpeTrain":float(np.median(sh)),
      "medianPFTrain":float(np.median(pf)),
      "worstYearTrain":float(np.min(ret)),
      "robustScore":float(np.median(ret)+.35*np.median(sh)+.15*np.median(np.minimum(pf,3))-0.20*np.median(dd)+.20*(ret>0).mean())
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--csv"); ap.add_argument("--config",default="config.json")
    ap.add_argument("--out",default="reports/v33_exit_research.json"); ap.add_argument("--csv-out",default="reports/v33_exit_research.csv")
    args=ap.parse_args()
    cfg=ROOT/args.config; p=load_params(cfg); base=load_v33_policy(cfg); bc=load_backtest_settings(cfg)
    df=load_csv(args.csv) if args.csv else fetch_yahoo("EURUSD=X",period=bc.get("downloadHistory","10y"))

    variants=[
      ("static_tp_sl",replace(base,max_holding_bars=0,breakeven_trigger_r=0,trailing_trigger_r=0,exit_on_opposite_signal=False)),
      ("be_0.75R",replace(base,max_holding_bars=0,breakeven_trigger_r=.75,trailing_trigger_r=0,exit_on_opposite_signal=False)),
      ("be_1.00R",replace(base,max_holding_bars=0,breakeven_trigger_r=1.0,trailing_trigger_r=0,exit_on_opposite_signal=False)),
      ("be_1.25R",replace(base,max_holding_bars=0,breakeven_trigger_r=1.25,trailing_trigger_r=0,exit_on_opposite_signal=False)),
      ("be1_trail1.5R_ATR1.5",replace(base,max_holding_bars=0,breakeven_trigger_r=1.0,trailing_trigger_r=1.5,trailing_atr=1.5,exit_on_opposite_signal=False)),
      ("be1_trail2R_ATR1.5",replace(base,max_holding_bars=0,breakeven_trigger_r=1.0,trailing_trigger_r=2.0,trailing_atr=1.5,exit_on_opposite_signal=False)),
      ("opposite_signal",replace(base,max_holding_bars=0,breakeven_trigger_r=0,trailing_trigger_r=0,exit_on_opposite_signal=True)),
      ("be1_opposite",replace(base,max_holding_bars=0,breakeven_trigger_r=1.0,trailing_trigger_r=0,exit_on_opposite_signal=True)),
      ("be1_trail1.5_opposite",replace(base,max_holding_bars=0,breakeven_trigger_r=1.0,trailing_trigger_r=1.5,trailing_atr=1.5,exit_on_opposite_signal=True)),
    ]
    rows=[]; yearly={}
    for name,pol in variants:
        ys=[]
        for y in range(2020,2026):
            b=backtest_v33(df,p,pol,start=f"{y}-01-01",end=f"{y}-12-31")
            ys.append({"year":y,**b["metrics"]})
        s=score(ys); rows.append({"variant":name,**s}); yearly[name]=ys
    tab=pd.DataFrame(rows).sort_values(["robustScore","medianReturnTrain"],ascending=False).reset_index(drop=True)
    selected=str(tab.iloc[0].variant); pol=dict(variants)[selected]
    test=backtest_v33(df,p,pol,start="2026-01-01")
    tab["selectedFor2026Test"]=tab.variant.eq(selected)
    outcsv=ROOT/args.csv_out; outcsv.parent.mkdir(parents=True,exist_ok=True); tab.to_csv(outcsv,index=False)
    report={
      "selectionRule":"exit policy selected only from 2020-2025; 2026 evaluated once after freeze",
      "constraint":"No candidate uses a fixed-day TIME exit.",
      "selected":selected,
      "selectedPolicy":{
        "breakevenTriggerR":pol.breakeven_trigger_r,"trailingTriggerR":pol.trailing_trigger_r,
        "trailingATR":pol.trailing_atr,"exitOnOppositeSignal":pol.exit_on_opposite_signal,
        "maxHoldingBars":pol.max_holding_bars
      },
      "selectedTrainYears":yearly[selected],
      "selected2026Test":test["metrics"],
      "allVariants":tab.to_dict(orient="records"),
      "promotionAutomatic":False
    }
    out=ROOT/args.out; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    print(tab.to_string(index=False)); print("\nFrozen 2026 test"); print(pd.DataFrame([test["metrics"]]).to_string(index=False))

if __name__=="__main__": main()
