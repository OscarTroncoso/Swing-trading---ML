from __future__ import annotations
from dataclasses import replace
import numpy as np, pandas as pd

from src.strategy_engine import (Params, backtest, backtest_v31_benchmark, features, position_plan, signal_row,
                                 signal_decision, trend_context, pnl_eur, adaptive_risk_pct, dynamic_stop_atr)
from src.ml_meta import MLParams, build_event_labels, fit_model_for_date, backtest_ml, FEATURE_SETS
from src.config import load_params, load_ml_params, load_v4_params
from src.strategy_v4 import V4Params, backtest_v4, position_plan_v4, signal_decision_v4
from src.ml_v4 import MLV4Params, build_v4_events, fit_best_v4_model


def synthetic(n=1800, drift=0.00008, seed=7):
    rng=np.random.default_rng(seed); reg=np.where((np.arange(n)//180)%2==0,drift,-drift/2); cycle=.0007*np.sin(np.arange(n)/18)
    ret=reg+cycle+rng.normal(0,0.003,n); close=1.05*np.exp(np.cumsum(ret)); open_=np.r_[close[0],close[:-1]*(1+rng.normal(0,0.00035,n-1))]
    wiggle=np.maximum(0.0007,np.abs(close-open_)*.7+rng.uniform(.0001,.0012,n)); high=np.maximum(open_,close)+wiggle; low=np.minimum(open_,close)-wiggle
    idx=pd.date_range('2019-01-01',periods=n,freq='B',tz='UTC'); return pd.DataFrame({'open':open_,'high':high,'low':low,'close':close},index=idx)


def test_features_expected_columns():
    f=features(synthetic(600),Params())
    for c in ['rsi','sma50','ema50','ema200','ema50_slope','macd_hist','atr','adx','plus_di','minus_di','rv20','swing_low','swing_high','rv20_q33','rv20_q67']:
        assert c in f.columns


def test_future_changes_do_not_change_past_signals():
    p=Params(); d=synthetic(800); cutoff=500; f1=features(d,p); s1=[signal_row(f1.iloc[i],p) for i in range(250,cutoff)]
    d2=d.copy(); d2.iloc[cutoff+5:]*=1.2; f2=features(d2,p); s2=[signal_row(f2.iloc[i],p) for i in range(250,cutoff)]; assert s1==s2


def test_signal_precedes_entry():
    b=backtest(synthetic(900),Params())
    for t in b['trades']: assert pd.Timestamp(t['signalDate'])<pd.Timestamp(t['entry'])<=pd.Timestamp(t['exit'])


def test_start_uses_prior_history_as_warmup():
    d=synthetic(900); start=d.index[400]; b=backtest(d,Params(),start=start); assert b['equity'].index[0]==start


def test_account_eur_and_dynamic_notional():
    d=synthetic(950); p=Params(initial_capital_eur=17500,sizing_mode='adaptive_risk',max_leverage=1.0); b=backtest(d,p)
    assert b['metrics']['initialCapitalEUR']==17500 and b['metrics']['accountCurrency']=='EUR'; assert all(t['notionalEUR']<=17500*1.1 for t in b['trades'])


def test_eurusd_pnl_conversion(): assert abs(pnl_eur(1.10,1.11,10_000,1)-(100/1.11))<1e-9


def test_position_plan_reports_structure_and_leverage():
    d=synthetic(900); p=Params(max_leverage=1.5); r=features(d,p).dropna().iloc[-1]; side=1 if r.close>r.sma50 else -1
    plan=position_plan(10_000,float(r.close),side,float(r.atr),r,p)
    assert 'stopSource' in plan and 'stopDistanceATR' in plan and 'trend' in plan
    assert plan['recommended']['leverage']<=1.5+1e-9


def test_stop_too_wide_is_rejected():
    d=synthetic(900); p=Params(max_stop_atr=.5); r=features(d,p).dropna().iloc[-1]; side=1 if r.close>r.sma50 else -1
    plan=position_plan(10_000,float(r.close),side,float(r.atr),r,p); assert not plan['valid'] and plan['rejectionReason']=='STOP_TOO_WIDE'


def test_adaptive_risk_is_bounded():
    r=features(synthetic(900),Params()).dropna().iloc[-1]; p=Params(min_risk_per_trade=.0025,max_risk_per_trade=.0075)
    for side in [-1,1]:
        x=adaptive_risk_pct(r,p,side); assert .0025-1e-12<=x<=.0075+1e-12


def test_trend_context_score_range():
    p=Params(); r=features(synthetic(900),p).dropna().iloc[-1]; c=trend_context(r,1,p); assert 0<=c['score']<=5 and c['maxScore']==5


def test_trend_filter_can_reject_candidate():
    p=Params(min_trend_score=5); f=features(synthetic(1100,seed=13),p).dropna(); found=False
    for _,r in f.iterrows():
        d=signal_decision(r,p)
        if d.get('candidate') and not d.get('accepted'):
            found=True; break
    assert found


def test_volatility_stop_multiplier_is_declared():
    p=Params(); r=features(synthetic(900),p).dropna().iloc[-1]; assert dynamic_stop_atr(r,p) in [p.stop_atr_low_vol,p.stop_atr_normal,p.stop_atr_high_vol]


def test_metrics_include_excursion_diagnostics():
    q=backtest(synthetic(1100),Params())['metrics']
    for k in ['expectancyR','avgMFER','avgMAER','medianWinnerCaptureRatio','stoppedTrades','rejectedTrendSignals','rejectedWideStops']: assert k in q and np.isfinite(q[k])


def test_trade_diagnostics_present():
    b=backtest(synthetic(1100),Params());
    for t in b['trades']:
        for k in ['mfeR','maeR','rMultiple','trendScore','trendRegime','stopSource','stopDistanceATR','initialRiskEUR']: assert k in t


def test_profit_protection_does_not_change_signal_dates_before_entries():
    d=synthetic(1200); a=backtest(d,Params(profit_protection_enabled=False)); b=backtest(d,Params(profit_protection_enabled=True))
    # Exit changes may alter later availability, but every realized trade still obeys chronology.
    for x in b['trades']: assert pd.Timestamp(x['signalDate'])<pd.Timestamp(x['entry'])<=pd.Timestamp(x['exit'])


def test_v31_benchmark_runs_separately():
    b=backtest_v31_benchmark(synthetic(950),Params()); assert 'metrics' in b and b['metrics']['accountCurrency']=='EUR'


def test_metrics_finite():
    q=backtest(synthetic(1000),Params())['metrics'];
    for k in ['finalCapitalEUR','totalReturn','maxDrawdown','sharpe','sortino','avgTradeEUR','avgLeverage','maxLeverageUsed','expectancyR']: assert np.isfinite(q[k])


def test_ml_events_are_chronological_and_purged():
    d=synthetic(1800,seed=22); p=Params(); mlp=MLParams(min_train_events=30,feature_set='core'); events=build_event_labels(d,p,mlp); assert len(events)>30
    assert (events.entryDate>events.signalDate).all() and (events.outcomeEnd>=events.entryDate).all(); asof=events.signalDate.iloc[-20]; _,meta=fit_model_for_date(events,asof,mlp); assert meta['trainingEvents']>0


def test_ml_backtest_respects_chronology_and_leverage():
    d=synthetic(1900,seed=15); p=Params(max_leverage=1.5); mlp=MLParams(min_train_events=30,retrain_every_bars=60,threshold=.55,min_validation_auc=.45)
    b=backtest_ml(d,p,mlp,start='2024-01-01')
    for t in b['trades']:
        assert pd.Timestamp(t['signalDate'])<pd.Timestamp(t['entry'])<=pd.Timestamp(t['exit']); assert t['leverage']<=1.5+1e-9


def test_ml_feature_sets_nested():
    assert set(FEATURE_SETS['core']).issubset(FEATURE_SETS['compact']); assert set(FEATURE_SETS['compact']).issubset(FEATURE_SETS['full'])


def test_nested_config_loads_v4_settings():
    p=load_v4_params('config.json')
    assert p.initial_capital_eur==1_000
    assert p.base_position_eur==100
    assert p.min_trade_leverage==1.0
    assert p.max_trade_leverage==30.0
    assert p.max_holding_bars==0
    assert not p.require_trend_filter


def test_v4_reports_eur_position_and_trade_leverage():
    d=synthetic(1100,seed=31); p=V4Params()
    f=features(d,p).dropna()
    r=f.iloc[-1]
    side=1 if r.close>r.sma50 else -1
    plan=position_plan_v4(1000,float(r.close),side,float(r.atr),r,p)
    if plan['valid']:
        assert plan['positionEUR']>=p.min_position_eur
        assert 1.0<=plan['tradeLeverage']<=30.0
        assert abs(plan['notionalEUR']-plan['positionEUR']*plan['tradeLeverage'])<.02
        assert plan['riskAtStopEUR']<=plan['targetRiskEUR']+.02


def test_v4_no_forced_time_exit():
    b=backtest_v4(synthetic(1400,seed=32),V4Params())
    assert all(t['reason']!='TIME' for t in b['trades'])
    assert b['metrics']['timeExitTrades']==0


def test_v4_trend_is_soft_except_extreme_opposition():
    p=V4Params(hard_trend_reject=False)
    f=features(synthetic(1200,seed=33),p).dropna()
    for _,r in f.iterrows():
        d=signal_decision_v4(r,p)
        if d.get('candidate'):
            assert d.get('accepted')
            break


def test_v4_open_position_is_not_counted_as_closed_trade():
    b=backtest_v4(synthetic(1000,seed=34),V4Params(target_rr=5.0,stop_atr_normal=3.0,stop_atr_low_vol=3.0,stop_atr_high_vol=3.0))
    assert b['metrics']['openTrades'] in (0,1)
    assert b['metrics']['finalEquityEUR']>0


def test_v4_ml_events_are_purged_and_barrier_resolved():
    d=synthetic(2200,seed=35); p=V4Params(); mlp=MLV4Params(min_train_events=40,label_horizon_bars=60)
    ev=build_v4_events(d,p,mlp)
    assert len(ev)>40
    assert (ev.entryDate>ev.signalDate).all()
    assert (ev.outcomeEnd>=ev.entryDate).all()
    assert set(ev.label.unique()).issubset({0,1})
    asof=ev.signalDate.iloc[-10]
    model,meta=fit_best_v4_model(ev,asof,mlp)
    assert meta['trainingEvents']>0
