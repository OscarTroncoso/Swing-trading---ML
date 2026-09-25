"""V3.3 ML model comparison and soft-sizing backtest."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.config import load_params, load_backtest_settings
from src.strategy_engine import fetch_yahoo, load_csv
from src.v33_engine import backtest_v33, load_v33_policy
from src.ml_v33 import ML33Params, build_resolved_events, chronological_model_comparison, expanding_probabilities


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--config",default="config.json")
    ap.add_argument("--start",default="2026-01-01")
    ap.add_argument("--end")
    ap.add_argument("--out",default="reports/ml_v33_research.json")
    ap.add_argument("--csv-out",default="reports/ml_v33_research.csv")
    args=ap.parse_args()

    cfg=ROOT/args.config
    p=load_params(cfg)
    policy=load_v33_policy(cfg)
    bc=load_backtest_settings(cfg)
    df=load_csv(args.csv) if args.csv else fetch_yahoo("EURUSD=X",period=bc.get("downloadHistory","10y"))

    events=build_resolved_events(df,p,policy)
    start_ts=pd.Timestamp(args.start,tz="UTC")
    pre=events[events.outcomeEnd < start_ts].copy()
    mlp=ML33Params()
    comparison=chronological_model_comparison(pre,mlp,folds=5)

    probs,meta=expanding_probabilities(df,p,policy,mlp,start=args.start,end=args.end)
    base=backtest_v33(df,p,policy,start=args.start,end=args.end)
    soft=backtest_v33(df,p,policy,start=args.start,end=args.end,ml_probabilities=probs)

    rows=[]
    for name,r in [("V3.3 no ML",base),("V3.3 ML soft sizing",soft)]:
        rows.append({"variant":name,**r["metrics"]})
    tab=pd.DataFrame(rows)
    out_csv=ROOT/args.csv_out; out_csv.parent.mkdir(parents=True,exist_ok=True); tab.to_csv(out_csv,index=False)

    report={
        "labelDefinition":"technical signal; label=1 only if TP is reached before SL; unresolved signals are censored; no time-exit labels",
        "preEvaluationEvents":int(len(pre)),
        "positiveRatePct":round(float(pre.label.mean()*100),2) if len(pre) else None,
        "modelComparison":comparison.to_dict(orient="records"),
        "lastModelMeta":meta.get("lastModelMeta",{}),
        "backtests":tab.to_dict(orient="records"),
        "predictions":meta.get("predictions",[]),
        "promotionAutomatic":False
    }
    out=ROOT/args.out; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    print("\nMODEL COMPARISON")
    print(comparison.to_string(index=False) if not comparison.empty else "No valid comparison")
    print("\nBACKTEST")
    print(tab.to_string(index=False))


if __name__=="__main__":
    main()
