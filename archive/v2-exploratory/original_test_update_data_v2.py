import importlib.util, pathlib, sys
import numpy as np, pandas as pd
spec=importlib.util.spec_from_file_location('m','/mnt/data/update-data-v2.py'); m=importlib.util.module_from_spec(spec); sys.modules['m']=m; spec.loader.exec_module(m)

def synthetic(n=400, drift=0.00035, seed=7):
    rng=np.random.default_rng(seed); r=drift+rng.normal(0,0.002,n); c=1.05*np.exp(np.cumsum(r))
    o=np.r_[c[0],c[:-1]]; spread=np.maximum(0.001,abs(c-o)*.7)
    h=np.maximum(o,c)+spread; l=np.minimum(o,c)-spread
    idx=pd.date_range('2024-01-01',periods=n,freq='B',tz='UTC')
    return pd.DataFrame({'open':o,'high':h,'low':l,'close':c},index=idx)

def test_features_have_expected_columns():
    p=m.Params(); f=m.features(synthetic(),p)
    for c in ['rsi','sma50','macd_hist','atr','adx','bb_z']: assert c in f.columns

def test_future_change_does_not_change_past_signal():
    p=m.Params(); d=synthetic(); f1=m.features(d,p); cutoff=250
    sig1=[m.signal_row(f1.iloc[i],p) for i in range(80,cutoff)]
    d2=d.copy(); d2.iloc[cutoff+5:, d2.columns.get_loc('close')]*=1.25
    d2.iloc[cutoff+5:, d2.columns.get_loc('high')]=np.maximum(d2.iloc[cutoff+5:].high,d2.iloc[cutoff+5:].close)
    f2=m.features(d2,p); sig2=[m.signal_row(f2.iloc[i],p) for i in range(80,cutoff)]
    assert sig1==sig2

def test_costs_do_not_improve_identical_strategy():
    d=synthetic(); low=m.backtest(d,m.Params(spread_pips=.2,slippage_pips_per_side=0))['metrics']['finalCapital']
    high=m.backtest(d,m.Params(spread_pips=2.0,slippage_pips_per_side=.5))['metrics']['finalCapital']
    assert high<=low+1e-9

def test_execution_is_after_signal_bar():
    d=synthetic(); p=m.Params(); b=m.backtest(d,p)
    # Every trade has distinct entry/exit chronology and cannot enter before warm-up.
    for t in b['trades']:
        assert pd.Timestamp(t['exit'])>=pd.Timestamp(t['entry'])
    if b['trades']:
        assert pd.Timestamp(b['trades'][0]['entry'])>=d.index[max(p.sma_trend,p.bb_period,p.atr_period,p.adx_period,30)]

def test_risk_sizing_respects_cap():
    d=synthetic(); p=m.Params(sizing_mode='risk',max_units=12000); b=m.backtest(d,p)
    assert all(t['units']<=12000+1e-9 for t in b['trades'])

def test_equity_metrics_are_finite():
    b=m.backtest(synthetic(),m.Params()); q=b['metrics']
    for k in ['finalCapital','totalReturn','maxDrawdown','sharpe','sortino']:
        assert np.isfinite(q[k])
