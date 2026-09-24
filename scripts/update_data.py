from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_params, load_ml_params, load_backtest_settings
from src.strategy_engine import backtest, current_snapshot, features, fetch_yahoo, load_csv, params_dict, position_plan, broad_candidate_row
from src.ml_meta import backtest_ml, current_ml_decision


def pd_isna(v):
    return v != v


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv')
    ap.add_argument('--ticker',default='EURUSD=X')
    ap.add_argument('--period',default=None)
    ap.add_argument('--backtest-start',default=None)
    ap.add_argument('--backtest-end',default=None)
    ap.add_argument('--out',default='data.json')
    ap.add_argument('--config',default='config.json')
    ap.add_argument('--no-ml',action='store_true')
    ap.add_argument('--market-csv-out',default=None)
    args=ap.parse_args()

    cfg_path=ROOT/args.config
    p=load_params(cfg_path); mlp=load_ml_params(cfg_path); bt_cfg=load_backtest_settings(cfg_path)
    period=args.period or bt_cfg.get('downloadHistory','10y')
    start=args.backtest_start or bt_cfg.get('evaluationStart',f'{datetime.now(timezone.utc).year}-01-01')
    df=load_csv(args.csv) if args.csv else fetch_yahoo(args.ticker,period=period)
    if args.market_csv_out:
        m=ROOT/args.market_csv_out; m.parent.mkdir(parents=True,exist_ok=True)
        df.reset_index().rename(columns={'index':'date'}).to_csv(m,index=False)

    strict=backtest(df,p,start=start,end=args.backtest_end)
    ml_bt=None if args.no_ml else backtest_ml(df,p,mlp,start=start,end=args.backtest_end)
    x=features(df,p)
    strict_snap=current_snapshot(x,p,equity_eur=strict['metrics']['finalCapitalEUR'])
    ml_dec={} if args.no_ml else current_ml_decision(df,p,mlp)
    if ml_dec:
        side=int(ml_dec.get('side',0) or 0)
        ml_dec={**ml_dec, 'candidateSignal': 'BUY (LONG)' if side==1 else ('SELL (SHORT)' if side==-1 else 'NONE'), 'mlProbability': ml_dec.get('probability'), 'mlAccepted': bool(ml_dec.get('accepted',False))}

    ml_plan=None
    if ml_dec.get('accepted') and ml_dec.get('side') in (-1,1):
        r=x.iloc[-1]
        ml_plan=position_plan(strict['metrics']['finalCapitalEUR'],float(r.close),int(ml_dec['side']),float(r.atr),r,p,ml_dec.get('probability'))

    tail=x.tail(180)
    payload={
        'modelVersion':'3.1.0',
        'lastUpdate':datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
        'pair':'EUR/USD','accountCurrency':'EUR',
        'liveMode':'STRICT_BASELINE_WITH_ML_CHALLENGER',
        'backtestWindow':{'start':start,'end':args.backtest_end or str(df.index.max().date())},
        'warmup':{'historyAvailableFrom':str(df.index.min().date()),'indicatorsUsePre2026History':True},
        **strict_snap,
        'ml':{**ml_dec,'positionPlan':ml_plan,'backtest':None if ml_bt is None else ml_bt['metrics']},
        'parameters':params_dict(p),
        'mlParameters':None if args.no_ml else mlp.__dict__,
        'backtest':strict['metrics'],
        'strictBacktest':strict['metrics'],
        'mlBacktest':None if ml_bt is None else ml_bt['metrics'],
        'trades':strict['trades'][-50:],
        'mlTrades':[] if ml_bt is None else ml_bt['trades'][-50:],
        'chart':{
            'dates':[str(i.date()) for i in tail.index],
            'prices':[round(float(v),6) for v in tail.close],
            'upperBand':[None if pd_isna(v) else round(float(v),6) for v in tail.bb_upper],
            'lowerBand':[None if pd_isna(v) else round(float(v),6) for v in tail.bb_lower],
            'sma50':[None if pd_isna(v) else round(float(v),6) for v in tail.sma50],
        }
    }
    out=ROOT/args.out; out.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    strict['equity'].to_csv(out.with_suffix('.equity.csv'))
    if ml_bt is not None:
        ml_bt['equity'].to_csv(out.with_name(out.stem+'.ml_equity.csv'))
    print(json.dumps({'strictSnapshot':strict_snap,'ml':ml_dec,'strictBacktest':strict['metrics'],'mlBacktest':None if ml_bt is None else ml_bt['metrics']},indent=2))

if __name__=='__main__': main()
