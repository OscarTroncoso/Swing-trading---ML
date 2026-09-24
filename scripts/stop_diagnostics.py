"""Stop/exit diagnostics for the current V3.2 model.
Quantifies MAE/MFE, winner capture, and whether stopped trades subsequently reached the original TP.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.config import load_params,load_backtest_settings
from src.strategy_engine import backtest,fetch_yahoo,load_csv


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv'); ap.add_argument('--config',default='config.json'); ap.add_argument('--start'); ap.add_argument('--end')
    ap.add_argument('--out',default='reports/stop_diagnostics.csv'); ap.add_argument('--json-out',default='reports/stop_diagnostics.json'); args=ap.parse_args()
    p=load_params(ROOT/args.config); bc=load_backtest_settings(ROOT/args.config); start=args.start or bc.get('evaluationStart','2026-01-01')
    df=load_csv(args.csv) if args.csv else fetch_yahoo('EURUSD=X',period=bc.get('downloadHistory','10y')); bt=backtest(df,p,start=start,end=args.end)
    rows=[]; horizons=[3,5,10]
    for t in bt['trades']:
        row=dict(t); exit_ts=pd.Timestamp(t['exit']); i=int(df.index.searchsorted(exit_ts,side='left')); side=1 if t['side']=='LONG' else -1
        for h in horizons:
            fut=df.iloc[i+1:min(len(df),i+1+h)]
            if len(fut)==0:
                row[f'tpAfterStop_{h}d']=False; row[f'returnToEntry_{h}d']=False; continue
            if side==1:
                row[f'tpAfterStop_{h}d']=bool((fut.high>=float(t['takeProfit'])).any()) if t['reason']=='SL' else False
                row[f'returnToEntry_{h}d']=bool((fut.high>=float(t['entryPrice'])).any()) if t['reason']=='SL' else False
            else:
                row[f'tpAfterStop_{h}d']=bool((fut.low<=float(t['takeProfit'])).any()) if t['reason']=='SL' else False
                row[f'returnToEntry_{h}d']=bool((fut.low<=float(t['entryPrice'])).any()) if t['reason']=='SL' else False
        rows.append(row)
    tab=pd.DataFrame(rows); out=ROOT/args.out; out.parent.mkdir(parents=True,exist_ok=True); tab.to_csv(out,index=False)
    stopped=tab[tab.reason=='SL'] if len(tab) else tab
    summary={'trades':int(len(tab)),'stoppedTrades':int(len(stopped)),'avgMFER':float(tab.mfeR.mean()) if len(tab) else 0.0,
             'avgMAER':float(tab.maeR.mean()) if len(tab) else 0.0,'medianWinnerCaptureRatio':float(tab.loc[tab.profitEUR>0,'captureRatio'].median()) if (tab.profitEUR>0).any() else 0.0}
    for h in horizons:
        summary[f'falseStopRate_{h}dPct']=round(float(stopped[f'tpAfterStop_{h}d'].mean()*100),2) if len(stopped) else 0.0
        summary[f'returnToEntryAfterStop_{h}dPct']=round(float(stopped[f'returnToEntry_{h}d'].mean()*100),2) if len(stopped) else 0.0
    (ROOT/args.json_out).write_text(json.dumps({'summary':summary,'definition':'False stop = SL trade whose original TP is touched within N subsequent daily bars. Daily OHLC cannot reveal intraday path ordering beyond the conservative backtest rule.'},indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
