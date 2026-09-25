"""Compare V3.1/V3.2 benchmarks with V3.3 TP/SL-only logic."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.config import load_params, load_backtest_settings
from src.strategy_engine import backtest as backtest_v32, backtest_v31_benchmark, fetch_yahoo, load_csv
from src.v33_engine import backtest_v33, load_v33_policy


def flat(name,r):
    return {"model":name,**r["metrics"]}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--config",default="config.json")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--out",default="reports/backtest_v33.json")
    ap.add_argument("--csv-out",default="reports/backtest_v33_comparison.csv")
    args=ap.parse_args()

    cfg=ROOT/args.config
    p=load_params(cfg)
    policy=load_v33_policy(cfg)
    bc=load_backtest_settings(cfg)
    start=args.start or bc.get("evaluationStart","2026-01-01")
    df=load_csv(args.csv) if args.csv else fetch_yahoo("EURUSD=X",period=bc.get("downloadHistory","10y"))

    v31=backtest_v31_benchmark(df,p,start=start,end=args.end)
    v32=backtest_v32(df,p,start=start,end=args.end)
    v33=backtest_v33(df,p,policy,start=start,end=args.end)

    table=pd.DataFrame([
        flat("V3.1 benchmark (legacy 5-bar exit)",v31),
        flat("V3.2 trend/structural filter",v32),
        flat("V3.3 EUR stake + TP/SL only",v33),
    ])
    out_csv=ROOT/args.csv_out; out_csv.parent.mkdir(parents=True,exist_ok=True); table.to_csv(out_csv,index=False)
    report={
        "window":{"start":start,"end":args.end or str(df.index.max().date())},
        "methodology":[
            "All strategies receive the same OHLC sample and next-open execution convention.",
            "V3.3 uses the V3.1 technical direction core without V3.2 hard trend vetoes.",
            "V3.3 position size is shown as EUR stake; gross exposure = stake x leverage.",
            "V3.3 leverage is an integer x1..x30, selected as the highest level whose SL loss stays inside the EUR risk budget.",
            "V3.3 has no mandatory time exit; positions remain open until TP/SL or the dataset ends.",
            "An unresolved V3.3 position at sample end is left open and marked to market, not force-closed as a realized trade."
        ],
        "comparison":table.to_dict(orient="records"),
        "v33Trades":v33["trades"],
        "v33OpenTrade":v33.get("openTrade"),
    }
    out=ROOT/args.out; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    print(table.to_string(index=False))


if __name__=="__main__":
    main()
