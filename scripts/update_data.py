from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.config import load_params, load_backtest_settings
from src.strategy_engine import backtest, current_snapshot, features, fetch_yahoo, load_csv, params_dict


def _n(v, digits=6):
    try:
        return None if v != v else round(float(v), digits)
    except Exception:
        return None


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv'); ap.add_argument('--ticker',default='EURUSD=X'); ap.add_argument('--period')
    ap.add_argument('--backtest-start'); ap.add_argument('--backtest-end'); ap.add_argument('--out',default='data.json'); ap.add_argument('--config',default='config.json')
    ap.add_argument('--market-csv-out'); args=ap.parse_args()
    cfg=ROOT/args.config; p=load_params(cfg); bc=load_backtest_settings(cfg); period=args.period or bc.get('downloadHistory','10y')
    start=args.backtest_start or bc.get('evaluationStart',f'{datetime.now(timezone.utc).year}-01-01')
    df=load_csv(args.csv) if args.csv else fetch_yahoo(args.ticker,period=period)
    if args.market_csv_out:
        outcsv=ROOT/args.market_csv_out; outcsv.parent.mkdir(parents=True,exist_ok=True)
        df.reset_index().rename(columns={'index':'date'}).to_csv(outcsv,index=False)
    bt=backtest(df,p,start=start,end=args.backtest_end); x=features(df,p)
    snap=current_snapshot(x,p,equity_eur=bt['metrics']['finalCapitalEUR'])
    tail=x.tail(220)
    payload={
      'modelVersion':'3.2.0','modelName':'Trend + Structural Risk Engine','lastUpdate':datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
      'pair':'EUR/USD','accountCurrency':'EUR','liveMode':'V3_2_CURRENT_ONLY',
      'backtestWindow':{'start':start,'end':args.backtest_end or str(df.index.max().date())},
      'warmup':{'historyAvailableFrom':str(df.index.min().date()),'preEvaluationHistoryUsedOnlyForIndicators':True},
      **snap,'parameters':params_dict(p),'backtest':bt['metrics'],'trades':bt['trades'][-80:],
      'chart':{
        'dates':[str(i.date()) for i in tail.index], 'prices':[_n(v) for v in tail.close],
        'ema50':[_n(v) for v in tail.ema50], 'ema200':[_n(v) for v in tail.ema200],
        'sma50':[_n(v) for v in tail.sma50], 'upperBand':[_n(v) for v in tail.bb_upper], 'lowerBand':[_n(v) for v in tail.bb_lower]
      }
    }
    out=ROOT/args.out; out.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); bt['equity'].to_csv(out.with_suffix('.equity.csv'))
    print(json.dumps({'snapshot':snap,'backtest':bt['metrics']},indent=2,default=str))

if __name__=='__main__': main()
