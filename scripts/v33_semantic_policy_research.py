"""Research V3.3 stop/target semantics close to the original V3.1 geometry.

V3.1 used stop=1.5 ATR and take=2.2 ATR, equivalent to about 1.47R.
V3.3 had interpreted 2.2 as 2.2R, moving the target much farther away.
This study selects only on 2020-2025 and heavily penalizes bad worst years.
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


def robust_score(rows):
    ret=np.array([r["totalReturn"] for r in rows],float); dd=np.array([r["maxDrawdown"] for r in rows],float)
    sh=np.array([r["sharpe"] for r in rows],float); pf=np.array([r["profitFactor"] for r in rows],float)
    pos=(ret>0).mean(); worst=ret.min()
    return {
      "medianReturnTrain":float(np.median(ret)),"meanReturnTrain":float(np.mean(ret)),
      "positiveYearsTrain":int((ret>0).sum()),"worstYearTrain":float(worst),
      "medianDDTrain":float(np.median(dd)),"medianSharpeTrain":float(np.median(sh)),
      "medianPFTrain":float(np.median(pf)),
      "robustScore":float(np.median(ret)+.35*np.median(sh)-.25*np.median(dd)+.45*pos+.45*worst),
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--csv"); ap.add_argument("--config",default="config.json")
    ap.add_argument("--out",default="reports/v33_semantic_policy_research.json")
    ap.add_argument("--csv-out",default="reports/v33_semantic_policy_research.csv")
    args=ap.parse_args()
    cfg=ROOT/args.config; p=load_params(cfg); base=load_v33_policy(cfg); bc=load_backtest_settings(cfg)
    df=load_csv(args.csv) if args.csv else fetch_yahoo("EURUSD=X",period=bc.get("downloadHistory","10y"))

    configs=[]
    for stop in [1.5,1.8,2.1]:
      for rr in [1.40,1.47,1.50,1.60,1.80,2.00]:
        for soft in [False,True]:
          configs.append(replace(base,stop_atr=stop,target_rr=rr,soft_trend_sizing=soft,
                                 max_holding_bars=0,stagnation_bars=0,extension_review_bars=0,
                                 hard_countertrend_veto=False))

    rows=[]; details={}
    for idx,pol in enumerate(configs):
        yrs=[]
        for year in range(2020,2026):
            b=backtest_v33(df,p,pol,start=f"{year}-01-01",end=f"{year}-12-31")
            yrs.append({"year":year,**b["metrics"]})
        rows.append({"configId":idx,"stopATR":pol.stop_atr,"targetRR":pol.target_rr,
                     "softTrendSizing":pol.soft_trend_sizing,**robust_score(yrs)})
        details[str(idx)]=yrs

    tab=pd.DataFrame(rows).sort_values(["robustScore","medianReturnTrain"],ascending=False).reset_index(drop=True)
    best_id=int(tab.iloc[0].configId); best=configs[best_id]
    test=backtest_v33(df,p,best,start="2026-01-01")
    legacy=backtest_v31_benchmark(df,p,start="2026-01-01")
    tab["selectedFor2026Test"]=tab.configId.eq(best_id)
    outcsv=ROOT/args.csv_out; outcsv.parent.mkdir(parents=True,exist_ok=True); tab.to_csv(outcsv,index=False)
    report={"selectionRule":"2020-2025 only; robust score penalizes worst year and DD; 2026 frozen",
            "v31Geometry":"1.5 ATR stop / 2.2 ATR take ~= 1.47R",
            "selected":{"stopATR":best.stop_atr,"targetRR":best.target_rr,"softTrendSizing":best.soft_trend_sizing,"maxHoldingBars":0},
            "selectedTrainYears":details[str(best_id)],"legacyFiveBar2026":legacy["metrics"],
            "selectedTPSLOnly2026":test["metrics"],"allConfigs":tab.to_dict(orient="records"),"promotionAutomatic":False}
    (ROOT/args.out).write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    print(tab.head(15).to_string(index=False)); print("\n2026")
    print(pd.DataFrame([{"variant":"legacy_5bar",**legacy["metrics"]},{"variant":"selected_tp_sl_only",**test["metrics"]}]).to_string(index=False))


if __name__=="__main__": main()
