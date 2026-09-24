"""Fixed-architecture annual walk-forward comparison.
No yearly parameter selection: compares frozen V3.1 benchmark with frozen V3.2 current rules.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.config import load_params,load_backtest_settings
from src.strategy_engine import backtest,backtest_v31_benchmark,fetch_yahoo,load_csv

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv'); ap.add_argument('--config',default='config.json'); ap.add_argument('--first-test-year',type=int,default=2020); ap.add_argument('--last-test-year',type=int,default=2026)
    ap.add_argument('--out',default='reports/walk_forward.csv'); ap.add_argument('--json-out',default='reports/walk_forward.json'); args=ap.parse_args()
    p=load_params(ROOT/args.config); bc=load_backtest_settings(ROOT/args.config); df=load_csv(args.csv) if args.csv else fetch_yahoo('EURUSD=X',period=bc.get('downloadHistory','10y')); rows=[]
    for year in range(args.first_test_year,args.last_test_year+1):
        if pd.Timestamp(f'{year}-01-01',tz='UTC')>df.index.max(): break
        for model,func in [('V3.1 benchmark',backtest_v31_benchmark),('V3.2 current',backtest)]:
            try: m=func(df,p,start=f'{year}-01-01',end=f'{year}-12-31')['metrics']
            except ValueError: continue
            rows.append({'testYear':year,'model':model,**m})
    tab=pd.DataFrame(rows); o=ROOT/args.out; o.parent.mkdir(parents=True,exist_ok=True); tab.to_csv(o,index=False)
    summary={}
    for model,g in tab.groupby('model'):
        summary[model]={'years':int(len(g)),'positiveYears':int((g.totalReturn>0).sum()),'meanAnnualReturn':round(float(g.totalReturn.mean()),4),
                        'compoundedReturn':round(float((g.totalReturn.div(100).add(1).prod()-1)*100),4),'meanSharpe':round(float(g.sharpe.mean()),4),
                        'worstYearReturn':round(float(g.totalReturn.min()),4),'worstYearDrawdown':round(float(g.maxDrawdown.max()),4)}
    (ROOT/args.json_out).write_text(json.dumps({'methodology':'Frozen architecture by calendar-year out-of-sample slices; no yearly tuning.','summary':summary,'folds':rows},indent=2,default=str),encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
