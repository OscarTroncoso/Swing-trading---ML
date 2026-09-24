"""V3.2 structural/trend sensitivity grid. Stress test, not optimizer."""
from __future__ import annotations
import argparse,itertools,sys,json
from dataclasses import replace
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.config import load_params,load_backtest_settings
from src.strategy_engine import backtest,fetch_yahoo,load_csv

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv'); ap.add_argument('--start'); ap.add_argument('--end'); ap.add_argument('--config',default='config.json')
    ap.add_argument('--out',default='reports/robustness_2026.csv'); ap.add_argument('--json-out',default='reports/robustness_2026.json'); args=ap.parse_args()
    base=load_params(ROOT/args.config); bc=load_backtest_settings(ROOT/args.config); start=args.start or bc.get('evaluationStart','2026-01-01')
    df=load_csv(args.csv) if args.csv else fetch_yahoo('EURUSD=X',period=bc.get('downloadHistory','10y')); rows=[]
    for score,lookback,maxstop,rr,adx in itertools.product([2,3,4],[8,12,20],[2.0,2.5,3.0],[1.4,1.6,1.8],[15.,18.,22.]):
        p=replace(base,min_trend_score=score,structure_lookback=lookback,max_stop_atr=maxstop,target_rr=rr,min_adx=adx)
        m=backtest(df,p,start=start,end=args.end)['metrics']; rows.append({'minTrendScore':score,'structureLookback':lookback,'maxStopATR':maxstop,'targetRR':rr,'minADX':adx,**m})
    tab=pd.DataFrame(rows); o=ROOT/args.out; o.parent.mkdir(parents=True,exist_ok=True); tab.to_csv(o,index=False)
    summary={'configs':len(tab),'positiveReturnPct':round(float((tab.totalReturn>0).mean()*100),2),'positiveSharpePct':round(float((tab.sharpe>0).mean()*100),2),
             'medianReturn':round(float(tab.totalReturn.median()),4),'p10Return':round(float(tab.totalReturn.quantile(.1)),4),'p90Return':round(float(tab.totalReturn.quantile(.9)),4),
             'medianMaxDrawdown':round(float(tab.maxDrawdown.median()),4),'medianTrades':round(float(tab.totalTrades.median()),2)}
    (ROOT/args.json_out).write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(summary)
if __name__=='__main__': main()
