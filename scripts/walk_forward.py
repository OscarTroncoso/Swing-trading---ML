"""Expanding annual walk-forward for the interpretable baseline.
Parameters are selected only on past years, then frozen for the unseen next year.
"""
from __future__ import annotations
import argparse,itertools,json,sys
from dataclasses import replace,asdict
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.config import load_params,load_backtest_settings
from src.strategy_engine import backtest,fetch_yahoo,load_csv

GRID=[dict(rsi_long=rhi,rsi_short=100-rhi,stop_atr=stop,take_atr=take,max_holding_bars=hold,risk_per_trade=risk,min_risk_per_trade=min(.0025,risk),max_risk_per_trade=max(.01,risk)) for rhi,(stop,take),hold,risk in itertools.product([52.,55.],[(1.5,2.2),(1.8,2.8)],[5,8],[0.005,0.0075])]
def score(m): return -999.0 if m['totalTrades']<12 else float(m['sharpe'])-0.05*float(m['maxDrawdown'])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv'); ap.add_argument('--period'); ap.add_argument('--config',default='config.json'); ap.add_argument('--first-test-year',type=int,default=2020); ap.add_argument('--last-test-year',type=int,default=2026); ap.add_argument('--min-train-years',type=int,default=3); ap.add_argument('--out',default='reports/walk_forward.csv'); ap.add_argument('--json-out',default='reports/walk_forward.json'); args=ap.parse_args()
    cfg=ROOT/args.config; base=load_params(cfg); bc=load_backtest_settings(cfg); period=args.period or bc.get('downloadHistory','10y'); df=load_csv(args.csv) if args.csv else fetch_yahoo('EURUSD=X',period=period); first_year=int(df.index.min().year); rows=[]
    for test_year in range(args.first_test_year,args.last_test_year+1):
        if test_year-first_year<args.min_train_years: continue
        test_start=f'{test_year}-01-01'; test_end=f'{test_year}-12-31'; train_end=f'{test_year-1}-12-31'
        if pd.Timestamp(test_start,tz='UTC')>df.index.max(): break
        cand=[]
        for g in GRID:
            try: m=backtest(df,replace(base,**g),start=f'{first_year}-01-01',end=train_end)['metrics']
            except ValueError: continue
            cand.append((score(m),g,m))
        if not cand: continue
        cand.sort(key=lambda z:z[0],reverse=True); train_score,chosen,train_m=cand[0]
        try: test_m=backtest(df,replace(base,**chosen),start=test_start,end=test_end)['metrics']
        except ValueError: continue
        rows.append({'test_year':test_year,'train_start':str(df.index.min().date()),'train_end':train_end,'train_score':train_score,'selected_params':json.dumps(chosen,sort_keys=True),**{f'train_{k}':v for k,v in train_m.items()},**{f'test_{k}':v for k,v in test_m.items()}})
    tab=pd.DataFrame(rows); out=ROOT/args.out; out.parent.mkdir(parents=True,exist_ok=True); tab.to_csv(out,index=False); (ROOT/args.json_out).write_text(json.dumps({'method':'expanding walk-forward; past-only selection, next-year frozen test','grid_size':len(GRID),'base_params':asdict(base),'folds':rows},indent=2,default=str),encoding='utf-8')
    print(tab[[c for c in ['test_year','test_totalReturn','test_winRate','test_maxDrawdown','test_profitFactor','test_totalTrades','test_sharpe'] if c in tab.columns]].to_string(index=False) if len(tab) else 'No valid folds.')
if __name__=='__main__': main()
