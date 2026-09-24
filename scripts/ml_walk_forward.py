"""Annual out-of-sample comparison of strict baseline vs fixed ML challenger."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.config import load_params,load_ml_params,load_backtest_settings
from src.strategy_engine import backtest,fetch_yahoo,load_csv
from src.ml_meta import backtest_ml

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv'); ap.add_argument('--period'); ap.add_argument('--config',default='config.json'); ap.add_argument('--first-test-year',type=int,default=2020); ap.add_argument('--last-test-year',type=int,default=2026); ap.add_argument('--out',default='reports/ml_walk_forward.csv'); ap.add_argument('--json-out',default='reports/ml_walk_forward.json'); args=ap.parse_args()
    cfg=ROOT/args.config; p=load_params(cfg); mlp=load_ml_params(cfg); bc=load_backtest_settings(cfg); period=args.period or bc.get('downloadHistory','10y'); df=load_csv(args.csv) if args.csv else fetch_yahoo('EURUSD=X',period=period); rows=[]
    for year in range(args.first_test_year,args.last_test_year+1):
        if pd.Timestamp(f'{year}-01-01',tz='UTC')>df.index.max(): break
        try: strict=backtest(df,p,start=f'{year}-01-01',end=f'{year}-12-31')['metrics']; ml=backtest_ml(df,p,mlp,start=f'{year}-01-01',end=f'{year}-12-31')['metrics']
        except ValueError: continue
        rows += [{'testYear':year,'model':'strict',**strict},{'testYear':year,'model':'ml_meta_label',**ml}]
    tab=pd.DataFrame(rows); out=ROOT/args.out; out.parent.mkdir(parents=True,exist_ok=True); tab.to_csv(out,index=False); (ROOT/args.json_out).write_text(json.dumps({'methodology':'Each test year uses the fixed ML architecture/threshold and retrains only from previously resolved events; no yearly threshold tuning.','folds':tab.to_dict(orient='records')},indent=2,default=str),encoding='utf-8'); print(tab[[c for c in ['testYear','model','totalReturn','maxDrawdown','totalTrades','winRate','profitFactor','sharpe'] if c in tab.columns]].to_string(index=False) if len(tab) else 'No folds')
if __name__=='__main__': main()
