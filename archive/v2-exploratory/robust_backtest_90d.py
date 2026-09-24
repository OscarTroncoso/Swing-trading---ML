import json, math
from pathlib import Path
import numpy as np
import pandas as pd

DATA_PATH = Path('/mnt/data/data.json')
INITIAL_CAPITAL = 10_000.0
PIP = 0.0001


def load_close_data(path=DATA_PATH):
    raw = json.loads(path.read_text())
    df = pd.DataFrame({'date': pd.to_datetime(raw['dates']), 'close': pd.Series(raw['prices'], dtype=float)})
    df = df.drop_duplicates('date').sort_values('date').set_index('date')
    return df, raw


def add_indicators(df):
    x = df.copy()
    c = x['close']
    x['sma20'] = c.rolling(20).mean()
    x['std20'] = c.rolling(20).std(ddof=0)
    x['bb_upper'] = x['sma20'] + 2*x['std20']
    x['bb_lower'] = x['sma20'] - 2*x['std20']
    x['z'] = (c-x['sma20'])/x['std20'].replace(0, np.nan)
    x['sma50'] = c.rolling(50).mean()
    x['ema20'] = c.ewm(span=20, adjust=False).mean()
    x['ema50'] = c.ewm(span=50, adjust=False).mean()
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    x['macd'] = ema12-ema26
    x['macd_signal'] = x['macd'].ewm(span=9, adjust=False).mean()
    x['macd_hist'] = x['macd']-x['macd_signal']

    # Wilder RSI
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    x['rsi'] = 100 - 100/(1+rs)
    x.loc[(avg_loss==0)&(avg_gain>0),'rsi'] = 100
    x.loc[(avg_loss==0)&(avg_gain==0),'rsi'] = 50

    # Close-only ATR proxy (real ATR requires OHLC)
    x['atr_proxy'] = c.diff().abs().ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    x['ret'] = c.pct_change()
    x['rv20'] = x['ret'].rolling(20).std(ddof=0)*math.sqrt(252)
    x['ema_gap'] = (x['ema20']/x['ema50'] - 1)
    x['ema20_slope5'] = x['ema20'].pct_change(5)
    return x


def original_rsi(prices, periods=14):
    if len(prices) < periods+1:
        return 50.0
    gains=[]; losses=[]
    for i in range(len(prices)-periods, len(prices)):
        change=prices[i]-prices[i-1]
        if change>0: gains.append(change)
        else: losses.append(abs(change))
    avg_gain=sum(gains)/periods
    avg_loss=sum(losses)/periods
    if avg_loss==0: return 100.0
    rs=avg_gain/avg_loss
    return 100-(100/(1+rs))


def original_as_is(df, initial=INITIAL_CAPITAL):
    # Exact logic in uploaded update-data.py, applied to the 90 closes retained in data.json.
    closes=df['close'].to_list(); dates=df.index
    capital=initial; pos=0; entry=0.; sl=tp=0.; entry_date=None
    trades=[]; peak=initial; maxdd=0
    for i in range(50,len(closes)):
        s=closes[i-20:i]
        sma=sum(s)/20
        var=sum((v-sma)**2 for v in s)/20
        std=var**0.5
        upper=sma+2*std; lower=sma-2*std
        p=closes[i]; rsi=original_rsi(closes[:i+1],14)
        if pos==0:
            if p<=lower or rsi<45:
                pos=1; entry=p; entry_date=dates[i]; sl=p-1.2*std; tp=p+1.8*std
            elif p>=upper or rsi>55:
                pos=-1; entry=p; entry_date=dates[i]; sl=p+1.2*std; tp=p-1.8*std
        elif pos==1 and (p<=sl or p>=tp):
            pnl=(p-entry)*10000; capital+=pnl
            trades.append((entry_date, dates[i],1,entry,p,pnl)); pos=0
        elif pos==-1 and (p>=sl or p<=tp):
            pnl=(entry-p)*10000; capital+=pnl
            trades.append((entry_date, dates[i],-1,entry,p,pnl)); pos=0
        peak=max(peak,capital); maxdd=max(maxdd,(peak-capital)/peak)
    return summarize_realized(trades, initial, capital, maxdd, 'Original exact (90d)')


def summarize_realized(trades, initial, capital, maxdd, name):
    pnls=np.array([t[-1] for t in trades],float) if trades else np.array([])
    wins=pnls[pnls>0]; losses=-pnls[pnls<0]
    pf=(wins.sum()/losses.sum()) if losses.sum()>0 else (np.inf if wins.sum()>0 else 0)
    return {
        'model':name,'final_capital':capital,'return_pct':(capital/initial-1)*100,
        'trades':len(trades),'win_rate_pct':(pnls>0).mean()*100 if len(pnls) else 0,
        'profit_factor':pf,'max_drawdown_pct':maxdd*100,
        'avg_pnl':pnls.mean() if len(pnls) else 0,'trades_detail':trades
    }


def calc_signal(row, mode='improved', z_thr=1.1, range_gap=0.0015):
    if pd.isna(row['rsi']) or pd.isna(row['z']) or pd.isna(row['atr_proxy']): return 0
    if mode=='original':
        if row['close'] <= row['bb_lower'] or row['rsi'] < 45: return 1
        if row['close'] >= row['bb_upper'] or row['rsi'] > 55: return -1
        return 0
    # Regime-aware hybrid
    trend_up = row['ema_gap'] > range_gap and row['macd_hist'] > 0 and row['ema20_slope5'] > 0
    trend_dn = row['ema_gap'] < -range_gap and row['macd_hist'] < 0 and row['ema20_slope5'] < 0
    range_regime = abs(row['ema_gap']) <= range_gap
    # buy pullbacks in uptrends / sell rallies in downtrends
    if trend_up and row['close'] <= row['ema20']*1.001 and 38 <= row['rsi'] <= 58:
        return 1
    if trend_dn and row['close'] >= row['ema20']*0.999 and 42 <= row['rsi'] <= 62:
        return -1
    # mean reversion only while regime is range-bound
    if range_regime and row['z'] <= -z_thr and row['rsi'] < 42:
        return 1
    if range_regime and row['z'] >= z_thr and row['rsi'] > 58:
        return -1
    return 0


def backtest_close_only(df, mode='original', initial=INITIAL_CAPITAL, spread_pips=0.8,
                        stop_mult=1.5, tp_mult=2.5, risk_pct=None, fixed_units=10000,
                        max_units=50000, max_holding=12, z_thr=1.1, range_gap=0.0015,
                        start_idx=50, name=None):
    """Close-only conservative engine: signal at close t; fill at close t+1 as a proxy.
    Costs charged half-spread each entry/exit. Intraday SL/TP cannot be modeled without OHLC.
    """
    x=add_indicators(df)
    n=len(x); capital=initial; position=0; units=0.; entry=sl=tp=np.nan; entry_i=None; entry_date=None
    pending=0; trades=[]; equities=[]; dates=[]; exposure=0
    half_spread = spread_pips*PIP/2

    for i in range(max(1,start_idx), n):
        row=x.iloc[i]
        price=float(row['close'])
        # Execute prior close's signal at this bar close proxy, with adverse half-spread.
        if position==0 and pending!=0:
            position=pending
            fill = price + half_spread if position==1 else price-half_spread
            atr=float(row['atr_proxy']) if pd.notna(row['atr_proxy']) and row['atr_proxy']>0 else price*0.005
            stop_dist=stop_mult*atr
            if risk_pct is None:
                units=float(fixed_units)
            else:
                units=min(max_units, max(1000.0, (capital*risk_pct)/max(stop_dist,1e-8)))
            entry=fill; entry_i=i; entry_date=x.index[i]
            if position==1:
                sl=entry-stop_dist; tp=entry+tp_mult*atr
            else:
                sl=entry+stop_dist; tp=entry-tp_mult*atr
            pending=0

        exit_reason=None; exit_price=None
        if position!=0:
            exposure += 1
            holding=i-entry_i
            # Close-only exits. On OHLC version this is replaced with high/low barrier logic.
            if position==1:
                if price<=sl: exit_reason='SL'; exit_price=price-half_spread
                elif price>=tp: exit_reason='TP'; exit_price=price-half_spread
            else:
                if price>=sl: exit_reason='SL'; exit_price=price+half_spread
                elif price<=tp: exit_reason='TP'; exit_price=price+half_spread
            if exit_reason is None and holding>=max_holding:
                exit_reason='TIME'; exit_price=price-half_spread if position==1 else price+half_spread

            # Optional exit on opposite model signal (evaluated at today's close, exited next iteration not same bar)
            # kept disabled to avoid same-bar decision/fill advantage.

            if exit_reason:
                pnl=(exit_price-entry)*units*position
                capital += pnl
                trades.append({'entry':entry_date,'exit':x.index[i],'side':position,'entry_price':entry,
                               'exit_price':exit_price,'units':units,'pnl':pnl,'reason':exit_reason})
                position=0; units=0.; entry_i=None; entry_date=None

        mtm=capital
        if position!=0:
            mark = price-half_spread if position==1 else price+half_spread
            mtm += (mark-entry)*units*position
        equities.append(mtm); dates.append(x.index[i])

        # Generate signal after marking/exits; execute next bar.
        if position==0 and pending==0:
            pending=calc_signal(row, mode=mode, z_thr=z_thr, range_gap=range_gap)

    # Liquidate at final close with spread to avoid hiding open risk
    if position!=0:
        price=float(x.iloc[-1]['close'])
        exit_price=price-half_spread if position==1 else price+half_spread
        pnl=(exit_price-entry)*units*position
        capital += pnl
        trades.append({'entry':entry_date,'exit':x.index[-1],'side':position,'entry_price':entry,
                       'exit_price':exit_price,'units':units,'pnl':pnl,'reason':'EOD'})
        if equities: equities[-1]=capital

    eq=pd.Series(equities,index=dates,dtype=float)
    r=eq.pct_change().fillna(0)
    peak=eq.cummax(); dd=(eq/peak-1)
    sharpe=(r.mean()/r.std(ddof=0)*math.sqrt(252)) if r.std(ddof=0)>0 else 0
    downside=r[r<0].std(ddof=0)
    sortino=(r.mean()/downside*math.sqrt(252)) if downside and downside>0 else 0
    pnls=np.array([t['pnl'] for t in trades],float) if trades else np.array([])
    wins=pnls[pnls>0]; losses=-pnls[pnls<0]
    pf=wins.sum()/losses.sum() if losses.sum()>0 else (np.inf if wins.sum()>0 else 0)
    out={
        'model': name or mode, 'final_capital':capital,'return_pct':(capital/initial-1)*100,
        'trades':len(trades),'win_rate_pct':(pnls>0).mean()*100 if len(pnls) else 0,
        'profit_factor':pf,'max_drawdown_pct':(-dd.min()*100) if len(dd) else 0,
        'sharpe':sharpe,'sortino':sortino,'avg_pnl':pnls.mean() if len(pnls) else 0,
        'avg_win':wins.mean() if len(wins) else 0,'avg_loss':-losses.mean() if len(losses) else 0,
        'exposure_pct': exposure/max(1,len(eq))*100,'trades_detail':trades,'equity':eq
    }
    return out


def robustness_grid(df):
    rows=[]
    for spread in [0.4,0.8,1.5,2.0]:
        for z in [0.9,1.1,1.3]:
            for gap in [0.0010,0.0015,0.0020]:
                for slm,tpm in [(1.2,2.0),(1.5,2.5),(1.8,3.0)]:
                    r=backtest_close_only(df,mode='improved',spread_pips=spread,stop_mult=slm,tp_mult=tpm,
                                          risk_pct=0.005,z_thr=z,range_gap=gap,
                                          name='grid')
                    rows.append({'spread_pips':spread,'z_thr':z,'range_gap':gap,'sl_mult':slm,'tp_mult':tpm,
                                 'return_pct':r['return_pct'],'trades':r['trades'],'win_rate_pct':r['win_rate_pct'],
                                 'profit_factor':r['profit_factor'],'max_drawdown_pct':r['max_drawdown_pct'],'sharpe':r['sharpe']})
    return pd.DataFrame(rows)


def main():
    df,raw=load_close_data()
    results=[]
    results.append(original_as_is(df))
    results.append(backtest_close_only(df,mode='original',spread_pips=0.8,stop_mult=1.2,tp_mult=1.8,
                                       risk_pct=None,fixed_units=10000,max_holding=999,
                                       name='Original corrected: next-bar + 0.8 pip'))
    results.append(backtest_close_only(df,mode='improved',spread_pips=0.8,stop_mult=1.5,tp_mult=2.5,
                                       risk_pct=None,fixed_units=10000,
                                       name='Hybrid regime, fixed 10k'))
    results.append(backtest_close_only(df,mode='improved',spread_pips=0.8,stop_mult=1.5,tp_mult=2.5,
                                       risk_pct=0.005,max_units=50000,
                                       name='Hybrid regime + 0.5% risk sizing'))

    summary=pd.DataFrame([{k:v for k,v in r.items() if k not in ('trades_detail','equity')} for r in results])
    summary.to_csv('/mnt/data/backtest_comparison_90d.csv',index=False)
    grid=robustness_grid(df)
    grid.to_csv('/mnt/data/robustness_grid_90d.csv',index=False)

    # Year-reported metrics from user's saved backtest for reference, not directly recomputed here.
    reported = raw.get('backtest',{})
    report={
      'data_window': {'start':str(df.index.min().date()),'end':str(df.index.max().date()),'observations':len(df)},
      'reported_original_1y': {k:reported.get(k) for k in ['initialCapital','finalCapital','totalReturn','winRate','maxDrawdown','profitFactor','totalTrades']},
      'local_90d_models': summary.replace([np.inf,-np.inf],None).to_dict(orient='records'),
      'robustness': {
        'configs':len(grid),
        'profitable_share_pct':float((grid['return_pct']>0).mean()*100),
        'positive_sharpe_share_pct':float((grid['sharpe']>0).mean()*100),
        'median_return_pct':float(grid['return_pct'].median()),
        'return_p10_pct':float(grid['return_pct'].quantile(.10)),
        'return_p90_pct':float(grid['return_pct'].quantile(.90)),
        'median_max_drawdown_pct':float(grid['max_drawdown_pct'].median()),
        'median_trades':float(grid['trades'].median())
      },
      'limitations':['Only 90 daily closes are retained in data.json.','No OHLC, so true ATR and intraday stop/take-profit ordering cannot be tested.','Long-horizon walk-forward requires the full historical dataset.']
    }
    Path('/mnt/data/backtest_report_90d.json').write_text(json.dumps(report,indent=2,default=str))
    print(summary.to_string(index=False))
    print('\nRobustness:\n',json.dumps(report['robustness'],indent=2))

if __name__=='__main__': main()

# --- Challenger signal suite for diagnostic search (fixed, interpretable rules) ---
def challenger_signal(row, kind):
    if pd.isna(row['rsi']) or pd.isna(row['sma50']) or pd.isna(row['z']): return 0
    c=row['close']
    if kind=='mr_trend_filter':
        # Mean-reversion only in direction of 50d trend
        if (c<=row['bb_lower'] or row['rsi']<40) and c>row['sma50']: return 1
        if (c>=row['bb_upper'] or row['rsi']>60) and c<row['sma50']: return -1
    elif kind=='mr_soft_trend_filter':
        if (c<=row['bb_lower'] or row['rsi']<42) and row['ema20']>=row['ema50']*0.998: return 1
        if (c>=row['bb_upper'] or row['rsi']>58) and row['ema20']<=row['ema50']*1.002: return -1
    elif kind=='momentum':
        if row['rsi']>55 and c>row['sma50'] and row['macd_hist']>0: return 1
        if row['rsi']<45 and c<row['sma50'] and row['macd_hist']<0: return -1
    elif kind=='momentum_simple':
        if row['rsi']>55 and c>row['sma50']: return 1
        if row['rsi']<45 and c<row['sma50']: return -1
    elif kind=='breakout':
        if c>row['bb_upper'] and row['ema20']>row['ema50']: return 1
        if c<row['bb_lower'] and row['ema20']<row['ema50']: return -1
    elif kind=='score':
        score=0
        score += 1 if row['ema20']>row['ema50'] else -1
        score += 1 if row['macd_hist']>0 else -1
        score += 1 if row['rsi']>52 else (-1 if row['rsi']<48 else 0)
        score += 1 if row['z']>0 else -1
        if score>=3: return 1
        if score<=-3: return -1
    return 0


def backtest_challenger(df, kind, spread_pips=.8, stop_mult=1.5,tp_mult=2.2,max_holding=8,risk_pct=None):
    # clone of close-only engine with alternate signal
    x=add_indicators(df); n=len(x); capital=INITIAL_CAPITAL; position=0; units=0.; pending=0
    entry=sl=tp=np.nan; entry_i=None; entry_date=None; trades=[]; equities=[]; dates=[]; exposure=0
    hs=spread_pips*PIP/2
    for i in range(50,n):
        row=x.iloc[i]; price=float(row.close)
        if position==0 and pending:
            position=pending; fill=price+hs if position==1 else price-hs
            atr=float(row.atr_proxy) if pd.notna(row.atr_proxy) and row.atr_proxy>0 else price*.005
            sd=stop_mult*atr
            units=10000. if risk_pct is None else min(50000,max(1000,(capital*risk_pct)/sd))
            entry=fill; entry_i=i; entry_date=x.index[i]
            sl=entry-sd if position==1 else entry+sd
            tp=entry+tp_mult*atr if position==1 else entry-tp_mult*atr
            pending=0
        reason=None
        if position:
            exposure+=1; hold=i-entry_i
            if position==1:
                if price<=sl: reason='SL'; out=price-hs
                elif price>=tp: reason='TP'; out=price-hs
            else:
                if price>=sl: reason='SL'; out=price+hs
                elif price<=tp: reason='TP'; out=price+hs
            if reason is None and hold>=max_holding:
                reason='TIME'; out=price-hs if position==1 else price+hs
            if reason:
                pnl=(out-entry)*units*position; capital+=pnl
                trades.append({'entry':entry_date,'exit':x.index[i],'side':position,'pnl':pnl,'reason':reason})
                position=0; units=0
        mtm=capital
        if position:
            mark=price-hs if position==1 else price+hs
            mtm+=(mark-entry)*units*position
        equities.append(mtm); dates.append(x.index[i])
        if position==0 and pending==0: pending=challenger_signal(row,kind)
    if position:
        price=float(x.iloc[-1].close); out=price-hs if position==1 else price+hs
        pnl=(out-entry)*units*position; capital+=pnl
        trades.append({'entry':entry_date,'exit':x.index[-1],'side':position,'pnl':pnl,'reason':'EOD'})
        equities[-1]=capital
    eq=pd.Series(equities,index=dates); rr=eq.pct_change().fillna(0); dd=eq/eq.cummax()-1
    pn=np.array([t['pnl'] for t in trades]); wins=pn[pn>0]; losses=-pn[pn<0]
    return {'model':kind,'return_pct':(capital/INITIAL_CAPITAL-1)*100,'final_capital':capital,'trades':len(pn),
            'win_rate_pct':(pn>0).mean()*100 if len(pn) else 0,'profit_factor':wins.sum()/losses.sum() if losses.sum()>0 else (np.inf if wins.sum()>0 else 0),
            'max_drawdown_pct':-dd.min()*100 if len(dd) else 0,'sharpe':rr.mean()/rr.std(ddof=0)*np.sqrt(252) if rr.std(ddof=0)>0 else 0,
            'avg_pnl':pn.mean() if len(pn) else 0,'exposure_pct':exposure/max(1,len(eq))*100,'trades_detail':trades}

if __name__=='__main__':
    df,_=load_close_data()
    kinds=['mr_trend_filter','mr_soft_trend_filter','momentum','momentum_simple','breakout','score']
    outs=[]
    for k in kinds:
        for hold in [5,8,12,20]:
            for sl,tp in [(1.2,1.8),(1.5,2.2),(1.8,2.8)]:
                r=backtest_challenger(df,k,stop_mult=sl,tp_mult=tp,max_holding=hold)
                outs.append({**{kk:vv for kk,vv in r.items() if kk!='trades_detail'},'hold':hold,'sl_mult':sl,'tp_mult':tp})
    o=pd.DataFrame(outs).sort_values(['return_pct','sharpe'],ascending=False)
    o.to_csv('/mnt/data/challenger_search_90d.csv',index=False)
    print('\nTOP CHALLENGERS')
    print(o.head(20).to_string(index=False))
    print('\nBY MODEL MEDIANS')
    print(o.groupby('model')[['return_pct','sharpe','max_drawdown_pct','trades','win_rate_pct']].median().sort_values('return_pct',ascending=False).to_string())
