"""Research whether multiple independent TP/SL tranches solve the opportunity-cost problem.

All portfolio-policy selection uses 2020-2025 only. The selected policy is frozen
before the one-shot 2026 evaluation.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.config import load_params, load_backtest_settings
from src.strategy_engine import fetch_yahoo, load_csv
from src.v33_engine import load_v33_policy, backtest_v33
from src.portfolio_engine import PortfolioPolicy, backtest_portfolio


def score_years(rows):
    ret=np.array([r["totalReturn"] for r in rows],float)
    dd=np.array([r["maxDrawdown"] for r in rows],float)
    sh=np.array([r["sharpe"] for r in rows],float)
    pf=np.array([r["profitFactor"] for r in rows],float)
    return {
        "medianReturnTrain":float(np.median(ret)),"meanReturnTrain":float(np.mean(ret)),
        "positiveYearsTrain":int((ret>0).sum()),"worstYearTrain":float(ret.min()),
        "medianDDTrain":float(np.median(dd)),"medianSharpeTrain":float(np.median(sh)),
        "medianPFTrain":float(np.median(pf)),
        "robustScore":float(np.median(ret)+0.35*np.median(sh)-0.20*np.median(dd)+0.25*(ret>0).mean()),
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--csv"); ap.add_argument("--config",default="config.json")
    ap.add_argument("--out",default="reports/v33_portfolio_research.json")
    ap.add_argument("--csv-out",default="reports/v33_portfolio_research.csv")
    args=ap.parse_args()
    cfg=ROOT/args.config; p=load_params(cfg); pol=load_v33_policy(cfg); bc=load_backtest_settings(cfg)
    df=load_csv(args.csv) if args.csv else fetch_yahoo("EURUSD=X",period=bc.get("downloadHistory","10y"))

    configs=[]
    for n in [1,2,3]:
      for cooldown in [1,3,5]:
        for risk in [0.02,0.03,0.04]:
          for gross in [2.5,4.0]:
            configs.append(PortfolioPolicy(max_concurrent_positions=n,min_bars_between_entries=cooldown,
                                           max_portfolio_risk_pct=risk,max_gross_exposure_multiple=gross,
                                           same_direction_only=True))

    rows=[]; yearly_detail={}
    for idx,pp in enumerate(configs):
        yearly=[]
        for year in range(2020,2026):
            b=backtest_portfolio(df,p,pol,pp,start=f"{year}-01-01",end=f"{year}-12-31")
            yearly.append({"year":year,**b["metrics"]})
        s=score_years(yearly)
        rows.append({"configId":idx,**asdict(pp),**s})
        yearly_detail[str(idx)]=yearly

    tab=pd.DataFrame(rows).sort_values(["robustScore","medianReturnTrain"],ascending=False).reset_index(drop=True)
    best_id=int(tab.iloc[0].configId); best=configs[best_id]
    test=backtest_portfolio(df,p,pol,best,start="2026-01-01")
    single=backtest_v33(df,p,pol,start="2026-01-01")
    tab["selectedFor2026Test"]=tab.configId.eq(best_id)
    outcsv=ROOT/args.csv_out; outcsv.parent.mkdir(parents=True,exist_ok=True); tab.to_csv(outcsv,index=False)
    report={
      "hypothesis":"Keep trades until TP/SL but allow independent later signals so one long-lived trade does not block the strategy.",
      "selectionRule":"portfolio policy selected only on 2020-2025; 2026 is frozen one-shot test",
      "selectedPolicy":asdict(best),
      "selectedTrainYears":yearly_detail[str(best_id)],
      "singlePosition2026":single["metrics"],
      "selectedPortfolio2026":test["metrics"],
      "selectedPortfolioOpenPositions":test.get("openPositions",[]),
      "allPolicies":tab.to_dict(orient="records"),
      "promotionAutomatic":False,
    }
    out=ROOT/args.out; out.write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    print(tab.head(15).to_string(index=False))
    print("\n2026 single vs selected portfolio")
    print(pd.DataFrame([{"variant":"single",**single["metrics"]},{"variant":"portfolio",**test["metrics"]}]).to_string(index=False))


if __name__=="__main__": main()
