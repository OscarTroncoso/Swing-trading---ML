"""Fair 2026 comparison: legacy, strict V3.1, and leakage-controlled ML challenger."""
from __future__ import annotations
import argparse,json,sys
from dataclasses import replace
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.config import load_params,load_ml_params,load_backtest_settings
from src.strategy_engine import backtest,backtest_legacy_fair,fetch_yahoo,load_csv
from src.ml_meta import backtest_ml


def flat(name,r): return {'model':name,**r['metrics']}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv'); ap.add_argument('--config',default='config.json')
    ap.add_argument('--out',default='reports/backtest_2026.json'); ap.add_argument('--csv-out',default='reports/backtest_2026_comparison.csv')
    ap.add_argument('--start',default=None); ap.add_argument('--end'); ap.add_argument('--period',default=None); args=ap.parse_args()
    cfg=ROOT/args.config; p=load_params(cfg); mlp=load_ml_params(cfg); bc=load_backtest_settings(cfg)
    start=args.start or bc.get('evaluationStart','2026-01-01'); period=args.period or bc.get('downloadHistory','10y')
    df=load_csv(args.csv) if args.csv else fetch_yahoo('EURUSD=X',period=period)
    legacy=backtest_legacy_fair(df,p,start=start,end=args.end)
    fixed=backtest(df,replace(p,sizing_mode='fixed_notional'),start=start,end=args.end)
    adaptive=backtest(df,p,start=start,end=args.end)
    ml=backtest_ml(df,p,mlp,start=start,end=args.end)
    table=pd.DataFrame([flat('Legacy signal / fair execution / € fixed notional',legacy),flat('V3.1 strict / € fixed notional',fixed),flat('V3.1 strict / adaptive risk',adaptive),flat('V3.1 ML challenger / adaptive risk',ml)])
    co=ROOT/args.csv_out; co.parent.mkdir(parents=True,exist_ok=True); table.to_csv(co,index=False)
    report={
      'window':{'start':start,'end':args.end or str(df.index.max().date())},'historyStart':str(df.index.min().date()),'rows':len(df),
      'methodology':['Pre-2026 history warms indicators; P&L starts exactly at the requested start.','Each model generates its own orders independently.','Signal at close t; fill at open t+1.','ATR/sizing inputs are frozen on signal bar.','Account equity, risk and P&L are EUR; EUR/USD units are EUR notional.','ML is a challenger meta-label trained only on events whose outcomes ended before each prediction date.','ML parameters are not auto-promoted from 2026 results.'],
      'comparison':table.to_dict(orient='records'),'legacyTrades':legacy['trades'],'fixedTrades':fixed['trades'],'strictTrades':adaptive['trades'],'mlTrades':ml['trades'],'mlPredictions':ml.get('predictions',[]),'mlLastModel':ml.get('lastModelMeta',{})}
    o=ROOT/args.out; o.parent.mkdir(parents=True,exist_ok=True); o.write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    print(table.to_string(index=False))
if __name__=='__main__': main()
