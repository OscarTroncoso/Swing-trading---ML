"""V3.2 return/risk attribution research. No variant is auto-promoted."""
from __future__ import annotations
import argparse,json,sys
from dataclasses import replace
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.config import load_params,load_backtest_settings
from src.strategy_engine import backtest,fetch_yahoo,load_csv

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv'); ap.add_argument('--config',default='config.json'); ap.add_argument('--start'); ap.add_argument('--end')
    ap.add_argument('--out',default='reports/improvement_research.csv'); ap.add_argument('--json-out',default='reports/improvement_research.json'); args=ap.parse_args()
    base=load_params(ROOT/args.config); bc=load_backtest_settings(ROOT/args.config); start=args.start or bc.get('evaluationStart','2026-01-01')
    df=load_csv(args.csv) if args.csv else fetch_yahoo('EURUSD=X',period=bc.get('downloadHistory','10y')); variants=[]
    def add(name,cat,p): variants.append({'variant':name,'category':cat,**backtest(df,p,start=start,end=args.end)['metrics']})
    add('v32_current','current',base)
    for risk in [.0025,.005,.0075]: add(f'fixed_risk_{risk*100:.2f}pct','sizing',replace(base,sizing_mode='risk',risk_per_trade=risk,min_risk_per_trade=risk,max_risk_per_trade=risk))
    for score in [2,3,4]: add(f'trend_score_{score}','trend',replace(base,min_trend_score=score))
    add('no_trend_filter','trend',replace(base,require_trend_filter=False))
    add('atr_only_stop','stop',replace(base,stop_mode='atr'))
    for ms in [2.0,2.5,3.0]: add(f'max_stop_{ms:.1f}atr','stop',replace(base,max_stop_atr=ms))
    for rr in [1.4,1.6,1.8,2.0]: add(f'target_rr_{rr:.1f}','target',replace(base,target_rr=rr))
    add('delayed_profit_protection','exit',replace(base,profit_protection_enabled=True))
    tab=pd.DataFrame(variants); o=ROOT/args.out; o.parent.mkdir(parents=True,exist_ok=True); tab.to_csv(o,index=False)
    payload={'promotionAutomatic':False,'principles':['Separate signal quality, stop geometry, risk sizing and exit management.','Do not select the highest-return 2026 row as production without walk-forward support.','ML remains research-only because prior report showed no useful discrimination.'],'variants':tab.to_dict(orient='records')}
    (ROOT/args.json_out).write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8')
    print(tab[['variant','category','totalReturn','maxDrawdown','profitFactor','sharpe','totalTrades','expectancyR','avgMFER','avgMAER']].sort_values(['sharpe','totalReturn'],ascending=False).to_string(index=False))
if __name__=='__main__': main()
