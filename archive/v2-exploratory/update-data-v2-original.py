"""EUR/USD strategy engine v2.

Key upgrades vs the uploaded prototype:
- OHLC-aware indicators and intraday stop/TP checks.
- Signal at close t, execution at open t+1 (no same-close look-ahead).
- Spread + slippage + optional commission.
- Wilder RSI, true ATR, ADX, MACD, Bollinger history, SMA/EMA trend filters.
- Fixed-unit or risk-based sizing.
- Conservative same-bar stop/TP ordering (stop assumed first if both touched).
- Full equity curve and risk metrics.
- Current default strategy is the momentum challenger that was most stable in
  the available 90-close diagnostic sample. Range/mean-reversion is optional,
  not silently mixed into the production signal.

Use:
  python update-data-v2.py --csv EURUSD.csv
or, if yfinance is installed and internet is available:
  python update-data-v2.py --ticker EURUSD=X --period 10y
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd

PIP = 0.0001

@dataclass
class Params:
    initial_capital: float = 10_000.0
    fixed_units: float = 10_000.0
    sizing_mode: str = "fixed"          # fixed | risk
    risk_per_trade: float = 0.005        # 0.5% equity
    max_units: float = 50_000.0
    spread_pips: float = 0.8             # round-trip modelled as half-spread each side
    slippage_pips_per_side: float = 0.1
    commission_per_million_per_side: float = 0.0
    rsi_period: int = 14
    rsi_long: float = 55.0
    rsi_short: float = 45.0
    sma_trend: int = 50
    bb_period: int = 20
    atr_period: int = 14
    adx_period: int = 14
    stop_atr: float = 1.5
    take_atr: float = 2.2
    max_holding_bars: int = 5
    require_macd: bool = True
    require_adx: bool = False
    min_adx: float = 18.0
    enable_range_module: bool = False
    range_adx_max: float = 17.0
    range_z: float = 1.5
    range_rsi_low: float = 35.0
    range_rsi_high: float = 65.0


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    x=df.copy()
    x.columns=[str(c).strip().lower() for c in x.columns]
    aliases={'datetime':'date','timestamp':'date','price':'ignore'}
    x=x.rename(columns={k:v for k,v in aliases.items() if k in x.columns})
    if 'date' in x.columns:
        x['date']=pd.to_datetime(x['date'],utc=True,errors='coerce')
        x=x.dropna(subset=['date']).set_index('date')
    if not isinstance(x.index,pd.DatetimeIndex):
        x.index=pd.to_datetime(x.index,utc=True,errors='coerce')
    needed=['open','high','low','close']
    missing=[c for c in needed if c not in x.columns]
    if missing:
        raise ValueError(f'Missing OHLC columns: {missing}')
    for c in needed:
        x[c]=pd.to_numeric(x[c],errors='coerce')
    return x[needed].dropna().sort_index().loc[lambda d: ~d.index.duplicated(keep='last')]


def load_csv(path: str) -> pd.DataFrame:
    # Handles ordinary OHLC CSV and Yahoo multi-header exports like the public sample.
    try:
        raw=pd.read_csv(path)
        if {'Open','High','Low','Close'}.issubset(raw.columns):
            return _norm_cols(raw)
    except Exception:
        pass
    raw=pd.read_csv(path,skiprows=[1])
    return _norm_cols(raw)


def fetch_yahoo(ticker='EURUSD=X',period='10y') -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError('yfinance is not installed. Use --csv or install yfinance.') from e
    d=yf.download(ticker,period=period,interval='1d',auto_adjust=True,progress=False)
    if d.empty: raise RuntimeError('Yahoo returned no data.')
    if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
    d=d.reset_index().rename(columns={'Date':'date','Open':'open','High':'high','Low':'low','Close':'close'})
    return _norm_cols(d)


def rsi_wilder(close, n=14):
    delta=close.diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    ag=gain.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    al=loss.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    rs=ag/al.replace(0,np.nan)
    out=100-100/(1+rs)
    out[(al==0)&(ag>0)]=100; out[(al==0)&(ag==0)]=50
    return out


def true_range(df):
    pc=df.close.shift(1)
    return pd.concat([(df.high-df.low).abs(),(df.high-pc).abs(),(df.low-pc).abs()],axis=1).max(axis=1)


def atr_wilder(df,n=14):
    return true_range(df).ewm(alpha=1/n,adjust=False,min_periods=n).mean()


def adx_wilder(df,n=14):
    up=df.high.diff(); dn=-df.low.diff()
    plus_dm=up.where((up>dn)&(up>0),0.0)
    minus_dm=dn.where((dn>up)&(dn>0),0.0)
    atr=atr_wilder(df,n)
    plus_di=100*plus_dm.ewm(alpha=1/n,adjust=False,min_periods=n).mean()/atr
    minus_di=100*minus_dm.ewm(alpha=1/n,adjust=False,min_periods=n).mean()/atr
    dx=100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False,min_periods=n).mean()


def features(df: pd.DataFrame,p: Params) -> pd.DataFrame:
    x=df.copy(); c=x.close
    x['rsi']=rsi_wilder(c,p.rsi_period)
    x['sma50']=c.rolling(p.sma_trend).mean()
    x['ema20']=c.ewm(span=20,adjust=False).mean(); x['ema50']=c.ewm(span=50,adjust=False).mean()
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean()
    x['macd']=e12-e26; x['macd_signal']=x.macd.ewm(span=9,adjust=False).mean(); x['macd_hist']=x.macd-x.macd_signal
    x['bb_mid']=c.rolling(p.bb_period).mean(); x['bb_std']=c.rolling(p.bb_period).std(ddof=0)
    x['bb_upper']=x.bb_mid+2*x.bb_std; x['bb_lower']=x.bb_mid-2*x.bb_std
    x['bb_z']=(c-x.bb_mid)/x.bb_std.replace(0,np.nan)
    x['atr']=atr_wilder(x,p.atr_period); x['adx']=adx_wilder(x,p.adx_period)
    x['rv20']=c.pct_change().rolling(20).std(ddof=0)*math.sqrt(252)
    return x


def signal_row(r,p:Params) -> int:
    if pd.isna(r.rsi) or pd.isna(r.sma50) or pd.isna(r.atr): return 0
    long_mom=(r.rsi>p.rsi_long and r.close>r.sma50)
    short_mom=(r.rsi<p.rsi_short and r.close<r.sma50)
    if p.require_macd:
        long_mom=long_mom and r.macd_hist>0
        short_mom=short_mom and r.macd_hist<0
    if p.require_adx and pd.notna(r.adx):
        long_mom=long_mom and r.adx>=p.min_adx
        short_mom=short_mom and r.adx>=p.min_adx
    if long_mom: return 1
    if short_mom: return -1
    if p.enable_range_module and pd.notna(r.adx) and r.adx<=p.range_adx_max:
        if r.bb_z<=-p.range_z and r.rsi<=p.range_rsi_low: return 1
        if r.bb_z>=p.range_z and r.rsi>=p.range_rsi_high: return -1
    return 0


def commission(units,p:Params):
    return (units/1_000_000.0)*p.commission_per_million_per_side


def backtest(df:pd.DataFrame,p:Params, start=None,end=None):
    x=features(df,p)
    if start is not None: x=x.loc[pd.Timestamp(start,tz='UTC'):] if pd.Timestamp(start).tzinfo is None else x.loc[pd.Timestamp(start):]
    if end is not None: x=x.loc[:pd.Timestamp(end,tz='UTC')] if pd.Timestamp(end).tzinfo is None else x.loc[:pd.Timestamp(end)]
    warm=max(p.sma_trend,p.bb_period,p.atr_period,p.adx_period,30)
    if len(x)<=warm+2: raise ValueError('Not enough data for backtest/warm-up.')

    cap=p.initial_capital; pos=0; pending=0; units=0.; entry=sl=tp=np.nan; entry_i=None; entry_date=None
    trades=[]; eq=[]; eq_dates=[]; exposure=0
    half_spread=p.spread_pips*PIP/2; slip=p.slippage_pips_per_side*PIP

    for i in range(warm,len(x)):
        r=x.iloc[i]
        # Fill yesterday's signal at today's OPEN, never today's close.
        if pos==0 and pending:
            pos=pending
            adverse=half_spread+slip
            entry=float(r.open + adverse if pos==1 else r.open-adverse)
            atr=float(r.atr)
            stop_dist=max(p.stop_atr*atr, PIP)
            if p.sizing_mode=='risk':
                units=min(p.max_units,max(1000.,(cap*p.risk_per_trade)/stop_dist))
            else: units=p.fixed_units
            cap-=commission(units,p)
            sl=entry-stop_dist if pos==1 else entry+stop_dist
            tp=entry+p.take_atr*atr if pos==1 else entry-p.take_atr*atr
            entry_i=i; entry_date=x.index[i]; pending=0

        exit_reason=None; exit_raw=None
        if pos:
            exposure+=1
            # Conservative if both barriers touched: assume stop first.
            if pos==1:
                stop_hit=r.low<=sl; take_hit=r.high>=tp
                if stop_hit: exit_reason='SL'; exit_raw=sl
                elif take_hit: exit_reason='TP'; exit_raw=tp
            else:
                stop_hit=r.high>=sl; take_hit=r.low<=tp
                if stop_hit: exit_reason='SL'; exit_raw=sl
                elif take_hit: exit_reason='TP'; exit_raw=tp
            if exit_reason is None and i-entry_i>=p.max_holding_bars:
                exit_reason='TIME'; exit_raw=float(r.close)
            if exit_reason:
                adverse=half_spread+slip
                exit_px=exit_raw-adverse if pos==1 else exit_raw+adverse
                pnl=(exit_px-entry)*units*pos-commission(units,p)
                cap+=pnl
                trades.append({'entry':str(entry_date),'exit':str(x.index[i]),'side':'LONG' if pos==1 else 'SHORT',
                               'entryPrice':entry,'exitPrice':exit_px,'units':units,'profit':pnl,'reason':exit_reason})
                pos=0; units=0.; entry_i=None

        mtm=cap
        if pos:
            mark=float(r.close-half_spread if pos==1 else r.close+half_spread)
            mtm+=(mark-entry)*units*pos
        eq.append(mtm); eq_dates.append(x.index[i])

        # Signal produced at today's CLOSE for execution at tomorrow's OPEN.
        if pos==0 and pending==0: pending=signal_row(r,p)

    # Liquidate any remaining position at last close.
    if pos:
        r=x.iloc[-1]; adverse=half_spread+slip
        exit_px=float(r.close-adverse if pos==1 else r.close+adverse)
        pnl=(exit_px-entry)*units*pos-commission(units,p); cap+=pnl
        trades.append({'entry':str(entry_date),'exit':str(x.index[-1]),'side':'LONG' if pos==1 else 'SHORT',
                       'entryPrice':entry,'exitPrice':exit_px,'units':units,'profit':pnl,'reason':'EOD'})
        eq[-1]=cap

    equity=pd.Series(eq,index=eq_dates,name='equity',dtype=float)
    rets=equity.pct_change().fillna(0); peak=equity.cummax(); dd=equity/peak-1
    pn=np.array([t['profit'] for t in trades],float); wins=pn[pn>0]; losses=-pn[pn<0]
    vol=rets.std(ddof=0); downside=rets[rets<0].std(ddof=0)
    metrics={
        'initialCapital':p.initial_capital,'finalCapital':round(cap,2),'totalReturn':round((cap/p.initial_capital-1)*100,4),
        'winRate':round((pn>0).mean()*100,2) if len(pn) else 0.0,'maxDrawdown':round(-dd.min()*100,4),
        'profitFactor':round(wins.sum()/losses.sum(),4) if losses.sum()>0 else (999.0 if wins.sum()>0 else 0.0),
        'totalTrades':int(len(trades)),'sharpe':round(rets.mean()/vol*math.sqrt(252),4) if vol>0 else 0.0,
        'sortino':round(rets.mean()/downside*math.sqrt(252),4) if downside and downside>0 else 0.0,
        'avgTrade':round(pn.mean(),2) if len(pn) else 0.0,'exposurePct':round(exposure/max(1,len(equity))*100,2)
    }
    return {'metrics':metrics,'trades':trades,'equity':equity,'features':x}


def current_snapshot(x:pd.DataFrame,p:Params):
    r=x.iloc[-1]; s=signal_row(r,p)
    return {
        'date':str(x.index[-1]),'currentPrice':round(float(r.close),5),'rsi':round(float(r.rsi),2) if pd.notna(r.rsi) else None,
        'sma50':round(float(r.sma50),5) if pd.notna(r.sma50) else None,
        'macdHist':round(float(r.macd_hist),6) if pd.notna(r.macd_hist) else None,
        'atr':round(float(r.atr),6) if pd.notna(r.atr) else None,'adx':round(float(r.adx),2) if pd.notna(r.adx) else None,
        'signal':'BUY (LONG)' if s==1 else ('SELL (SHORT)' if s==-1 else 'NEUTRAL (WAIT)')
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv'); ap.add_argument('--ticker',default='EURUSD=X'); ap.add_argument('--period',default='10y')
    ap.add_argument('--out',default='data-v2.json'); ap.add_argument('--risk-sizing',action='store_true')
    args=ap.parse_args(); p=Params(sizing_mode='risk' if args.risk_sizing else 'fixed')
    df=load_csv(args.csv) if args.csv else fetch_yahoo(args.ticker,args.period)
    bt=backtest(df,p); x=features(df,p); snap=current_snapshot(x,p)
    tail=x.tail(120)
    out={**snap,'parameters':asdict(p),'backtest':bt['metrics'],'trades':bt['trades'][-20:],
         'chart':{
             'dates':[str(i.date()) for i in tail.index],
             'prices':[round(float(v),6) for v in tail['close']],
             'upperBand':[None if pd.isna(v) else round(float(v),6) for v in tail['bb_upper']],
             'lowerBand':[None if pd.isna(v) else round(float(v),6) for v in tail['bb_lower']],
             'sma50':[None if pd.isna(v) else round(float(v),6) for v in tail['sma50']]
         }}
    Path(args.out).write_text(json.dumps(out,indent=2))
    bt['equity'].to_csv(Path(args.out).with_suffix('.equity.csv'))
    print(json.dumps({'snapshot':snap,'backtest':bt['metrics']},indent=2))

if __name__=='__main__': main()
