"""2026 comparison: frozen V3.1 benchmark versus current V3.2 engine."""
from __future__ import annotations
import argparse,json,sys
from dataclasses import replace
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.config import load_params,load_backtest_settings
from src.strategy_engine import backtest,backtest_v31_benchmark,fetch_yahoo,load_csv

def flat(name,r): return {'model':name,**r['metrics']}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv'); ap.add_argument('--config',default='config.json'); ap.add_argument('--start'); ap.add_argument('--end'); ap.add_argument('--period')
    ap.add_argument('--out',default='reports/backtest_2026.json'); ap.add_argument('--csv-out',default='reports/backtest_2026_comparison.csv'); args=ap.parse_args()
    cfg=ROOT/args.config; p=load_params(cfg); bc=load_backtest_settings(cfg); start=args.start or bc.get('evaluationStart','2026-01-01'); period=args.period or bc.get('downloadHistory','10y')
    df=load_csv(args.csv) if args.csv else fetch_yahoo('EURUSD=X',period=period)
    v31=backtest_v31_benchmark(df,p,start=start,end=args.end)
    v32=backtest(df,p,start=start,end=args.end)
    v32_fixed=backtest(df,replace(p,sizing_mode='fixed_notional',fixed_notional_eur=p.initial_capital_eur,max_leverage=1.0),start=start,end=args.end)
    table=pd.DataFrame([flat('V3.1 benchmark / adaptive risk',v31),flat('V3.2 current / adaptive quality risk',v32),flat('V3.2 current / 1.0x fixed notional',v32_fixed)])
    co=ROOT/args.csv_out; co.parent.mkdir(parents=True,exist_ok=True); table.to_csv(co,index=False)
    report={'window':{'start':start,'end':args.end or str(df.index.max().date())},'historyStart':str(df.index.min().date()),'rows':len(df),
            'methodology':['2025/prior history warms indicators; P&L starts at requested evaluation date.','Each model generates its own signals independently.','Signal at close t; fill at open t+1.','Signal-bar ATR, structure and trend inputs are frozen before next-open execution.','V3.2 uses trend regime confirmation plus hybrid ATR/market-structure stops.','All account P&L and risk are EUR.'],
            'comparison':table.to_dict(orient='records'),'currentTrades':v32['trades'],'benchmarkTrades':v31['trades'],'rejections':v32.get('rejections',{})}
    o=ROOT/args.out; o.parent.mkdir(parents=True,exist_ok=True); o.write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    print(table.to_string(index=False))
if __name__=='__main__': main()
