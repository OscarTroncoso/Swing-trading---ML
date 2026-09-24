"""V3.2 component ablation. No variant is auto-promoted."""
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
    ap.add_argument('--out',default='reports/v32_ablation.csv'); ap.add_argument('--json-out',default='reports/v32_ablation.json'); args=ap.parse_args()
    p=load_params(ROOT/args.config); bc=load_backtest_settings(ROOT/args.config); start=args.start or bc.get('evaluationStart','2026-01-01')
    df=load_csv(args.csv) if args.csv else fetch_yahoo('EURUSD=X',period=bc.get('downloadHistory','10y'))
    variants=[
      ('V3.2 full',p),
      ('no trend filter',replace(p,require_trend_filter=False)),
      ('ATR-only stop',replace(p,stop_mode='atr')),
      ('no volatility-adaptive stop',replace(p,stop_atr_low_vol=1.5,stop_atr_normal=1.5,stop_atr_high_vol=1.5)),
      ('no quality sizing',replace(p,quality_scale_min=1.0,quality_scale_max=1.0)),
      ('trend score >=4',replace(p,min_trend_score=4)),
      ('profit protection delayed',replace(p,profit_protection_enabled=True)),
      ('fixed risk 0.50%',replace(p,sizing_mode='risk',risk_per_trade=.005,min_risk_per_trade=.005,max_risk_per_trade=.005)),
    ]
    rows=[]
    for name,v in variants:
        b=backtest(df,v,start=start,end=args.end); rows.append({'variant':name,**b['metrics']})
    tab=pd.DataFrame(rows); o=ROOT/args.out; o.parent.mkdir(parents=True,exist_ok=True); tab.to_csv(o,index=False)
    (ROOT/args.json_out).write_text(json.dumps({'promotionAutomatic':False,'variants':tab.to_dict(orient='records')},indent=2,default=str),encoding='utf-8')
    print(tab[['variant','totalReturn','maxDrawdown','profitFactor','sharpe','totalTrades','expectancyR','avgMFER','avgMAER']].to_string(index=False))
if __name__=='__main__': main()
