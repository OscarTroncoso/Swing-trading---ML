"""Attribution research: signal edge, sizing, leverage, exits and ML.
No variant is auto-promoted. The purpose is to identify why return changes.
"""
from __future__ import annotations
import argparse,json,sys
from dataclasses import replace
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.config import load_params,load_ml_params,load_backtest_settings
from src.strategy_engine import backtest,fetch_yahoo,load_csv
from src.experimental import backtest_exit_policy,standard_exit_policies
from src.ml_meta import backtest_ml

def row(label,result,category,extra=None):
    out={'variant':label,'category':category,**result['metrics']}
    if extra: out.update(extra)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv'); ap.add_argument('--period'); ap.add_argument('--start'); ap.add_argument('--end'); ap.add_argument('--config',default='config.json'); ap.add_argument('--out',default='reports/improvement_research.csv'); ap.add_argument('--json-out',default='reports/improvement_research.json'); args=ap.parse_args()
    cfg=ROOT/args.config; base=load_params(cfg); mlbase=load_ml_params(cfg); bc=load_backtest_settings(cfg); start=args.start or bc.get('evaluationStart','2026-01-01'); period=args.period or bc.get('downloadHistory','10y')
    df=load_csv(args.csv) if args.csv else fetch_yahoo('EURUSD=X',period=period); rows=[]
    fixed=replace(base,sizing_mode='fixed_notional',fixed_notional_eur=base.initial_capital_eur,max_leverage=1.0)
    rows.append(row('fixed_100pct_equity_notional',backtest(df,fixed,start=start,end=args.end),'baseline'))
    for risk in [0.0025,0.005,0.0075,0.01]:
        p=replace(base,sizing_mode='risk',risk_per_trade=risk,min_risk_per_trade=risk,max_risk_per_trade=risk)
        rows.append(row(f'fixed_risk_{risk*100:.2f}pct',backtest(df,p,start=start,end=args.end),'sizing',{'riskPerTradePct':risk*100}))
    rows.append(row('adaptive_risk_vol_target',backtest(df,base,start=start,end=args.end),'sizing'))
    for lev in [1.0,1.5,2.0,3.0]:
        p=replace(base,max_leverage=lev); rows.append(row(f'adaptive_max_leverage_{lev:.1f}x',backtest(df,p,start=start,end=args.end),'leverage',{'maxLeverage':lev}))
    for policy in standard_exit_policies():
        rows.append(row(f'exit_{policy.name}',backtest_exit_policy(df,base,policy,start=start,end=args.end),'exit',{'exitPolicy':policy.name}))
    for th in [0.55,0.60,0.65]:
        mlp=replace(mlbase,threshold=th); rows.append(row(f'ml_core_{th:.2f}',backtest_ml(df,base,mlp,start=start,end=args.end),'ml',{'threshold':th,'featureSet':mlp.feature_set}))
    table=pd.DataFrame(rows); out=ROOT/args.out; out.parent.mkdir(parents=True,exist_ok=True); table.to_csv(out,index=False)
    payload={'window':{'start':start,'end':args.end or str(df.index.max().date())},'principles':['€10,000 is account equity, never a hard-coded number of FX units.','Sizing variants keep the same strict signal so return from exposure is separated from return from edge.','Leverage is capped and reported trade by trade.','ML is a past-only meta-filter challenger, not an automatic replacement.','Exit variants and ML variants are research challengers and are not selected by highest 2026 return.'],'variants':table.to_dict(orient='records')}
    jo=ROOT/args.json_out; jo.parent.mkdir(parents=True,exist_ok=True); jo.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8')
    cols=[c for c in ['variant','category','totalReturn','maxDrawdown','totalTrades','winRate','profitFactor','sharpe','avgLeverage','maxLeverageUsed'] if c in table.columns]; print(table[cols].sort_values(['totalReturn','sharpe'],ascending=False).to_string(index=False))
if __name__=='__main__': main()
