from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.config import load_params, load_backtest_settings
from src.strategy_engine import features, fetch_yahoo, load_csv
from src.v33_engine import backtest_v33, current_snapshot_v33, load_v33_policy
from src.ml_v33 import ML33Params, expanding_probabilities


def _n(v,d=6):
    try: return None if v!=v else round(float(v),d)
    except Exception: return None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--ticker",default="EURUSD=X")
    ap.add_argument("--period"); ap.add_argument("--backtest-start"); ap.add_argument("--backtest-end")
    ap.add_argument("--out",default="data.json"); ap.add_argument("--config",default="config.json")
    ap.add_argument("--market-csv-out")
    args=ap.parse_args()

    cfg=ROOT/args.config
    p=load_params(cfg)
    policy=load_v33_policy(cfg)
    bc=load_backtest_settings(cfg)
    period=args.period or bc.get("downloadHistory","10y")
    start=args.backtest_start or bc.get("evaluationStart",f"{datetime.now(timezone.utc).year}-01-01")
    df=load_csv(args.csv) if args.csv else fetch_yahoo(args.ticker,period=period)

    if args.market_csv_out:
        outcsv=ROOT/args.market_csv_out; outcsv.parent.mkdir(parents=True,exist_ok=True)
        df.reset_index().rename(columns={"index":"date"}).to_csv(outcsv,index=False)

    probs,ml_meta=expanding_probabilities(df,p,policy,ML33Params(),start=start,end=args.backtest_end)
    bt=backtest_v33(df,p,policy,start=start,end=args.backtest_end,ml_probabilities=probs)

    x=features(df,p)
    latest_prob=probs.get(pd.Timestamp(x.index[-1])) if "pd" in globals() else None
    if latest_prob is None:
        try:
            import pandas as pd
            latest_prob=probs.get(pd.Timestamp(x.index[-1]))
        except Exception:
            latest_prob=None

    snap=current_snapshot_v33(x,p,policy,equity_eur=bt["metrics"]["finalCapitalEUR"],ml_probability=latest_prob)
    tail=x.tail(220)
    payload={
      "modelVersion":"3.3.0",
      "modelName":"EUR Stake + Risk-Constrained Leverage Engine",
      "lastUpdate":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
      "pair":"EUR/USD","accountCurrency":"EUR","liveMode":"V3_3_CURRENT_ONLY",
      "backtestWindow":{"start":start,"end":args.backtest_end or str(df.index.max().date())},
      "warmup":{"historyAvailableFrom":str(df.index.min().date()),"preEvaluationHistoryUsedOnlyForIndicators":True},
      **snap,
      "ml":{
        "mode":"soft_sizing_only",
        "latestProbability":latest_prob,
        "lastModelMeta":ml_meta.get("lastModelMeta",{}),
        "note":"ML never vetoes a V3.3 technical signal; when healthy it only adjusts stake/risk modestly."
      },
      "backtest":bt["metrics"],
      "openTrade":bt.get("openTrade"),
      "trades":bt["trades"][-100:],
      "chart":{
        "dates":[str(i.date()) for i in tail.index],
        "prices":[_n(v) for v in tail.close],
        "ema50":[_n(v) for v in tail.ema50],
        "ema200":[_n(v) for v in tail.ema200],
        "sma50":[_n(v) for v in tail.sma50],
      }
    }
    out=ROOT/args.out
    out.write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8")
    bt["equity"].to_csv(out.with_suffix(".equity.csv"))
    print(json.dumps({"snapshot":snap,"backtest":bt["metrics"],"openTrade":bt.get("openTrade")},indent=2,default=str))


if __name__=="__main__":
    main()
