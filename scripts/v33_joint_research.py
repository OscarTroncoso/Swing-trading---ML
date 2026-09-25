"""Joint V3.3 search: stop/TP geometry + conditional extension.

Selection sees 2020-2025 only. 2026 is evaluated exactly once after freeze.
The search is intentionally small/interpretable to avoid parameter fishing.
"""
from __future__ import annotations
import argparse, json, sys
from dataclasses import replace
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
    pos=(ret>0).mean(); worst=ret.min()
    # Strong penalty for unstable/worst-year behavior.
    return {
      "medianReturnTrain":float(np.median(ret)),"meanReturnTrain":float(np.mean(ret)),
      "positiveYearsTrain":int((ret>0).sum()),"worstYearTrain":float(worst),
      "medianDDTrain":float(np.median(dd)),"medianSharpeTrain":float(np.median(sh)),
      "robustScore":float(np.median(ret)+.35*np.median(sh)-.25*np.median(dd)+.5*pos+.55*worst),
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--csv"); ap.add_argument("--config",default="config.json")
    ap.add_argument("--out",default="reports/v33_joint_research.json")
    ap.add_argument("--csv-out",default="reports/v33_joint_research.csv")
    args=ap.parse_args()
    cfg=ROOT/args.config; p=load_params(cfg); base=load_v33_policy(cfg); bc=load_backtest_settings(cfg)
    df=load_csv(args.csv) if args.csv else fetch_yahoo("EURUSD=X",period=bc.get("downloadHistory","10y"))

    configs=[]
    for stop in [1.5,1.8,2.1]:
      for rr in [1.40,1.50,1.60]:
        for review in [3,5]:
          for mfe in [.50,.75]:
            for confirms in [2,3]:
              for soft in [False,True]:
                configs.append(replace(base,stop_atr=stop,target_rr=rr,soft_trend_sizing=soft,
                    max_holding_bars=0,stagnation_bars=0,extension_review_bars=review,
                    extension_min_mfe_r=mfe,extension_min_close_r=0.0,
                    extension_min_confirmations=confirms,hard_countertrend_veto=False))

    rows=[]; details={}
    for idx,pol in enumerate(configs):
        yrs=[]
        for year in range(2020,2026):
            b=backtest_v33(df,p,pol,start=f"{year}-01-01",end=f"{year}-12-31")
            yrs.append({"year":year,**b["metrics"]})
        rows.append({"configId":idx,"stopATR":pol.stop_atr,"targetRR":pol.target_rr,
          "reviewBars":pol.extension_review_bars,"minMFER":pol.extension_min_mfe_r,
          "minConfirmations":pol.extension_min_confirmations,"softTrendSizing":pol.soft_trend_sizing,
          **score(yrs)})
        details[str(idx)]=yrs

    tab=pd.DataFrame(rows).sort_values(["robustScore","medianReturnTrain"],ascending=False).reset_index(drop=True)
    best_id=int(tab.iloc[0].configId); best=configs[best_id]
    test=backtest_v33(df,p,best,start="2026-01-01")
    legacy=backtest_v31_benchmark(df,p,start="2026-01-01")
    tab["selectedFor2026Test"]=tab.configId.eq(best_id)
    outcsv=ROOT/args.csv_out; outcsv.parent.mkdir(parents=True,exist_ok=True); tab.to_csv(outcsv,index=False)
    report={
      "selectionRule":"small joint grid selected on 2020-2025 only; 2026 frozen one-shot",
      "selected":{"stopATR":best.stop_atr,"targetRR":best.target_rr,"reviewBars":best.extension_review_bars,
        "minMFER":best.extension_min_mfe_r,"minConfirmations":best.extension_min_confirmations,
        "softTrendSizing":best.soft_trend_sizing,"maxHoldingBars":0},
      "selectedTrainYears":details[str(best_id)],
      "legacyFiveBar2026":legacy["metrics"],"selectedJoint2026":test["metrics"],
      "allConfigs":tab.to_dict(orient="records"),"promotionAutomatic":False
    }
    (ROOT/args.out).write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    print(tab.head(20).to_string(index=False))
    print("\n2026")
    print(pd.DataFrame([{"variant":"legacy_5bar",**legacy["metrics"]},{"variant":"joint_dynamic",**test["metrics"]}]).to_string(index=False))


if __name__=="__main__": main()
