"""Out-of-sample-oriented ML ablation and threshold research.
No variant is auto-promoted into the live baseline.
"""
from __future__ import annotations
import argparse,json,sys
from dataclasses import replace
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.config import load_params,load_ml_params,load_backtest_settings
from src.strategy_engine import backtest,fetch_yahoo,load_csv
from src.ml_meta import MLParams,backtest_ml,build_event_labels

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv'); ap.add_argument('--config',default='config.json'); ap.add_argument('--period',default=None); ap.add_argument('--start',default=None); ap.add_argument('--end'); ap.add_argument('--out',default='reports/ml_research.csv'); ap.add_argument('--json-out',default='reports/ml_research.json'); args=ap.parse_args()
    cfg=ROOT/args.config; p=load_params(cfg); base=load_ml_params(cfg); bc=load_backtest_settings(cfg); start=args.start or bc.get('evaluationStart','2026-01-01'); period=args.period or bc.get('downloadHistory','10y')
    df=load_csv(args.csv) if args.csv else fetch_yahoo('EURUSD=X',period=period)
    strict=backtest(df,p,start=start,end=args.end); rows=[{'variant':'strict_baseline','featureSet':None,'threshold':None,**strict['metrics']}]
    for fs in ['core','compact','full']:
      for th in [0.55,0.60,0.65]:
        mlp=replace(base,feature_set=fs,threshold=th)
        r=backtest_ml(df,p,mlp,start=start,end=args.end)
        rows.append({'variant':f'ml_{fs}_{th:.2f}','featureSet':fs,'threshold':th,**r['metrics']})
    tab=pd.DataFrame(rows); out=ROOT/args.out; out.parent.mkdir(parents=True,exist_ok=True); tab.to_csv(out,index=False)
    events=build_event_labels(df,p,base)
    # Research gate is descriptive only: it never edits config.json.
    candidates=tab[tab.variant!='strict_baseline'].copy(); base_ret=float(strict['metrics']['totalReturn']); base_dd=float(strict['metrics']['maxDrawdown'])
    pass_mask=(candidates.totalReturn>base_ret)&(candidates.profitFactor>1.0)&(candidates.maxDrawdown<=max(base_dd*1.25,0.5))&(candidates.totalTrades>=max(5,strict['metrics']['totalTrades']//2))
    payload={'status':'RESEARCH_PASS' if pass_mask.any() else 'KEEP_BASELINE','promotionIsAutomatic':False,'methodology':['ML only accepts/rejects broad directional candidates; it does not invent direction.','Feature sets are nested to expose whether added complexity contributes out of sample.','Training uses only labels resolved before each prediction date.','Thresholds are reported side by side and are not selected automatically from 2026.'],'events':{'count':len(events),'positiveRatePct':round(float(events.label.mean()*100),2) if len(events) else None},'variants':tab.to_dict(orient='records')}
    jo=ROOT/args.json_out; jo.parent.mkdir(parents=True,exist_ok=True); jo.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(tab.to_string(index=False)); print('\nGate:',payload['status'])
if __name__=='__main__': main()
