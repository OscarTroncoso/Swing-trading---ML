from __future__ import annotations
from dataclasses import replace
import numpy as np
import pandas as pd

from src.strategy_engine import Params, backtest, backtest_legacy_fair, features, position_plan, signal_row, pnl_eur, adaptive_risk_pct
from src.ml_meta import MLParams, build_event_labels, fit_model_for_date, backtest_ml, FEATURE_SETS
from src.config import load_params, load_ml_params


def synthetic(n=1800, drift=0.00008, seed=7):
    rng=np.random.default_rng(seed)
    reg=np.where((np.arange(n)//180)%2==0, drift, -drift/2)
    cycle=.0007*np.sin(np.arange(n)/18)
    ret=reg+cycle+rng.normal(0,0.003,n)
    close=1.05*np.exp(np.cumsum(ret))
    open_=np.r_[close[0],close[:-1]*(1+rng.normal(0,0.00035,n-1))]
    wiggle=np.maximum(0.0007,np.abs(close-open_)*0.7+rng.uniform(0.0001,0.0012,n))
    high=np.maximum(open_,close)+wiggle; low=np.minimum(open_,close)-wiggle
    idx=pd.date_range('2019-01-01',periods=n,freq='B',tz='UTC')
    return pd.DataFrame({'open':open_,'high':high,'low':low,'close':close},index=idx)


def test_features_expected_columns():
    f=features(synthetic(500),Params())
    for c in ['rsi','sma50','macd_hist','atr','adx','bb_z','rv20','sma_gap','macd_atr']:
        assert c in f.columns


def test_future_changes_do_not_change_past_signals():
    p=Params(); d=synthetic(600); cutoff=380
    f1=features(d,p); s1=[signal_row(f1.iloc[i],p) for i in range(80,cutoff)]
    d2=d.copy(); d2.iloc[cutoff+5:]*=1.20
    f2=features(d2,p); s2=[signal_row(f2.iloc[i],p) for i in range(80,cutoff)]
    assert s1==s2


def test_signal_precedes_entry():
    b=backtest(synthetic(600),Params())
    for t in b['trades']:
        assert pd.Timestamp(t['signalDate']) < pd.Timestamp(t['entry']) <= pd.Timestamp(t['exit'])


def test_start_uses_prior_history_as_warmup():
    d=synthetic(600); start=d.index[240]
    b=backtest(d,Params(),start=start)
    assert b['equity'].index[0] == start


def test_account_is_eur_and_notional_is_not_hardcoded_units():
    d=synthetic(650); p=Params(initial_capital_eur=17_500,sizing_mode='fixed_notional',fixed_notional_eur=17_500,max_leverage=1.0)
    b=backtest(d,p)
    assert b['metrics']['initialCapitalEUR']==17500
    assert b['metrics']['accountCurrency']=='EUR'
    assert all(t['notionalEUR']<=17_500+1e-9 for t in b['trades'])


def test_eurusd_pnl_is_converted_back_to_eur():
    # 10k EUR units, +100 pips = +100 USD, translated at the exit FX rate.
    assert abs(pnl_eur(1.10,1.11,10_000,1)-(100/1.11))<1e-9


def test_position_plan_reports_leverage_and_no_leverage_alternative():
    d=synthetic(500); p=Params(max_leverage=3.0,sizing_mode='risk',risk_per_trade=.01)
    r=features(d,p).dropna().iloc[-1]
    plan=position_plan(10_000,float(r.close),1,float(r.atr),r,p)
    assert plan['recommended']['leverage']<=3.0+1e-9
    assert plan['noLeverageAlternative']['leverage']<=1.0+1e-9
    assert isinstance(plan['recommended']['usesLeverage'],bool)
    assert plan['recommended']['riskAtStopEUR']>=0


def test_absolute_unit_cap_is_respected():
    p=Params(sizing_mode='risk',risk_per_trade=.02,max_leverage=10.0,absolute_max_units=12000)
    b=backtest(synthetic(700),p)
    assert all(t['units']<=12000+1e-9 for t in b['trades'])


def test_adaptive_risk_is_bounded():
    r=features(synthetic(500),Params()).dropna().iloc[-1]; p=Params(min_risk_per_trade=.0025,max_risk_per_trade=.01)
    for prob in [None,.60,.70,.90]:
        x=adaptive_risk_pct(r,p,prob); assert .0025-1e-12<=x<=.01+1e-12


def test_metrics_finite():
    q=backtest(synthetic(650),Params())['metrics']
    for k in ['finalCapitalEUR','totalReturn','maxDrawdown','sharpe','sortino','avgTradeEUR','avgLeverage','maxLeverageUsed']:
        assert np.isfinite(q[k])


def test_disabled_adx_does_not_change_signal_requirement():
    d=synthetic(600); p=Params(require_adx=False); row=features(d,p).dropna().iloc[-1]; s=signal_row(row,p)
    assert signal_row(row,replace(p,min_adx=99.0))==s


def test_legacy_fair_generates_own_order_timeline():
    b=backtest_legacy_fair(synthetic(650),Params())
    for t in b['trades']:
        assert pd.Timestamp(t['signalDate']) < pd.Timestamp(t['entry']) <= pd.Timestamp(t['exit'])


def test_sizing_changes_exposure_not_strict_signal_timeline():
    d=synthetic(700)
    fixed=backtest(d,Params(sizing_mode='fixed_notional',fixed_notional_eur=10000,max_leverage=1))
    risk=backtest(d,Params(sizing_mode='risk',risk_per_trade=.005,max_leverage=3))
    assert [t['signalDate'] for t in fixed['trades']]==[t['signalDate'] for t in risk['trades']]


def test_ml_events_are_chronological_and_purged():
    d=synthetic(n=1800,seed=22); p=Params(); mlp=MLParams(min_train_events=30,feature_set='core')
    events=build_event_labels(d,p,mlp)
    assert len(events)>30
    assert (events.entryDate>events.signalDate).all(); assert (events.outcomeEnd>=events.entryDate).all()
    asof=events.signalDate.iloc[-20]
    _,meta=fit_model_for_date(events,asof,mlp)
    expected=((events.outcomeEnd<asof)&(events.signalDate>=asof-pd.DateOffset(years=mlp.max_train_years))).sum()
    assert meta['trainingEvents']<=int(expected)


def test_ml_backtest_is_chronological_and_respects_leverage():
    d=synthetic(n=1900,seed=15); p=Params(max_leverage=2.0); mlp=MLParams(min_train_events=30,retrain_every_bars=60,threshold=.55)
    b=backtest_ml(d,p,mlp,start='2024-01-01')
    for t in b['trades']:
        assert pd.Timestamp(t['signalDate'])<pd.Timestamp(t['entry'])<=pd.Timestamp(t['exit'])
        assert t['leverage']<=2.0+1e-9


def test_ml_feature_sets_are_nested():
    assert set(FEATURE_SETS['core']).issubset(FEATURE_SETS['compact'])
    assert set(FEATURE_SETS['compact']).issubset(FEATURE_SETS['full'])


def test_nested_config_loads_account_and_ml_settings():
    p=load_params('config.json'); m=load_ml_params('config.json')
    assert p.initial_capital_eur==10_000
    assert p.max_leverage==3.0
    assert m.feature_set=='core' and m.threshold==.60
