"""Research a narrow counter-trend veto on top of conditional extensions.

This deliberately avoids the broad V3.2 trend filter. A signal is vetoed only
when its trend score is very poor AND ADX says the opposing trend is strong.
All selection is pre-2026.
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
    ret=np.array([r["totalReturn"] for r in rows],float); dd=np.array([r["maxDrawdown"] for r in rows],float)
    sh=np.array([r["sharpe"] for r in rows],float); pf=np.array([r["profitFactor"] for r in rows],float)
    return {"medianReturnTrain":float(np.median(ret)),"meanReturnTrain":float(np.mean(ret)),
            "positiveYearsTrain":int((ret>0).sum()),"worstYearTrain":float(ret.min()),
            "medianDDTrain":float(np.median(dd)),"medianSharpeTrain":float(np.median(sh)),
            "medianPFTrain":float(np.median(pf)),
            "robustScore":float(np.median(ret)+.40*np.median(sh)-.20*np.median(dd)+.30*(ret>0).mean())}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--csv"); ap.add_argument("--config",default="config.json")
    ap.add_argument("--out",default="reports/v33_entry_filter_research.json")
    ap.add_argument("--csv-out",default="reports/v33_entry_filter_research.csv")
    args=ap.parse_args()
    cfg=ROOT/args.config; p=load_params(cfg); base=load_v33_policy(cfg); bc=load_backtest_settings(cfg)
    df=load_csv(args.csv) if args.csv else fetch_yahoo("EURUSD=X",period=bc.get("downloadHistory","10y"))

    # Extension parameters were selected using 2020-2025 in the preceding study.
    extension=replace(base,max_holding_bars=0,stagnation_bars=0,extension_review_bars=3,
                      extension_min_mfe_r=.75,extension_min_close_r=0.0,extension_min_confirmations=3)

    policies=[("off",replace(extension,hard_countertrend_veto=False))]
    for max_score in [0,1,2]:
      for adx in [20.0,24.0,28.0,32.0,36.0]:
        policies.append((f"score<={max_score}_adx>={adx:g}",replace(extension,hard_countertrend_veto=True,
                        countertrend_max_score=max_score,countertrend_min_adx=adx)))

    rows=[]; detail={}
    for idx,(name,pol) in enumerate(policies):
        yrs=[]
        for year in range(2020,2026):
            b=backtest_v33(df,p,pol,start=f"{year}-01-01",end=f"{year}-12-31")
            yrs.append({"year":year,**b["metrics"]})
        rows.append({"configId":idx,"variant":name,"veto":pol.hard_countertrend_veto,
                     "maxTrendScore":pol.countertrend_max_score,"minADX":pol.countertrend_min_adx,**score(yrs)})
        detail[str(idx)]=yrs

    tab=pd.DataFrame(rows).sort_values(["robustScore","medianReturnTrain"],ascending=False).reset_index(drop=True)
    best_id=int(tab.iloc[0].configId); name,best=policies[best_id]
    test=backtest_v33(df,p,best,start="2026-01-01")
    no_veto=backtest_v33(df,p,policies[0][1],start="2026-01-01")
    legacy=backtest_v31_benchmark(df,p,start="2026-01-01")
    tab["selectedFor2026Test"]=tab.configId.eq(best_id)
    outcsv=ROOT/args.csv_out; outcsv.parent.mkdir(parents=True,exist_ok=True); tab.to_csv(outcsv,index=False)
    report={"selectionRule":"counter-trend veto selected on 2020-2025 only; 2026 frozen",
            "baseExit":"conditional extension review=3, minMFE=.75R, 3/3 confirmations",
            "selected":{"variant":name,"hardCountertrendVeto":best.hard_countertrend_veto,
                        "countertrendMaxScore":best.countertrend_max_score,"countertrendMinADX":best.countertrend_min_adx},
            "selectedTrainYears":detail[str(best_id)],"legacyFiveBar2026":legacy["metrics"],
            "extensionNoVeto2026":no_veto["metrics"],"selectedFilter2026":test["metrics"],
            "allVariants":tab.to_dict(orient="records"),"promotionAutomatic":False}
    (ROOT/args.out).write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    print(tab.to_string(index=False)); print("\n2026")
    print(pd.DataFrame([{"variant":"legacy_5bar",**legacy["metrics"]},{"variant":"extension_no_veto",**no_veto["metrics"]},{"variant":name,**test["metrics"]}]).to_string(index=False))


if __name__=="__main__": main()
