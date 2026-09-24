"""Declared 2026 sensitivity grid. This is a stress test, not an optimizer."""
from __future__ import annotations
import argparse,itertools,sys
from dataclasses import replace
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.config import load_params,load_backtest_settings
from src.strategy_engine import backtest,fetch_yahoo,load_csv

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv'); ap.add_argument('--start'); ap.add_argument('--end'); ap.add_argument('--config',default='config.json'); ap.add_argument('--out',default='reports/robustness_2026.csv'); args=ap.parse_args()
    cfg=ROOT/args.config; base=load_params(cfg); bc=load_backtest_settings(cfg); start=args.start or bc.get('evaluationStart','2026-01-01')
    df=load_csv(args.csv) if args.csv else fetch_yahoo('EURUSD=X',period=bc.get('downloadHistory','10y'))
    rows=[]
    for rhi,(stop,take),hold,spread,lev in itertools.product([52.,55.,58.],[(1.2,1.8),(1.5,2.2),(1.8,2.8)],[3,5,8],[0.4,0.8,1.5,2.0],[1.0,2.0,3.0]):
        p=replace(base,rsi_long=rhi,rsi_short=100-rhi,stop_atr=stop,take_atr=take,max_holding_bars=hold,spread_pips=spread,max_leverage=lev)
        m=backtest(df,p,start=start,end=args.end)['metrics']
        rows.append({'rsi_long':rhi,'rsi_short':100-rhi,'stop_atr':stop,'take_atr':take,'holding':hold,'spread_pips':spread,'max_leverage_cap':lev,**m})
    result=pd.DataFrame(rows); out=ROOT/args.out; out.parent.mkdir(parents=True,exist_ok=True); result.to_csv(out,index=False)
    print({'configs':len(result),'profitable_share_pct':round((result.totalReturn>0).mean()*100,2),'median_return_pct':round(result.totalReturn.median(),4),'p10_return_pct':round(result.totalReturn.quantile(.1),4),'p90_return_pct':round(result.totalReturn.quantile(.9),4),'median_trades':round(result.totalTrades.median(),2),'median_max_drawdown_pct':round(result.maxDrawdown.median(),4)})
if __name__=='__main__': main()
