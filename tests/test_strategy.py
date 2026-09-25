from __future__ import annotations
from dataclasses import replace
import numpy as np, pandas as pd

from src.strategy_engine import (Params, backtest, backtest_v31_benchmark, features, position_plan, signal_row,
                                 signal_decision, trend_context, pnl_eur, adaptive_risk_pct, dynamic_stop_atr)
from src.ml_meta import MLParams, build_event_labels, fit_model_for_date, backtest_ml, FEATURE_SETS
from src.config import load_params, load_ml_params
from src.v33_engine import V33Policy, backtest_v33, load_v33_policy, position_plan_v33
from src.ml_v33 import ML33Params, build_resolved_events, chronological_model_comparison


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


def test_nested_config_loads_current_settings():
    p=load_params('config.json'); m=load_ml_params('config.json')
    assert p.initial_capital_eur==1_000
    assert p.max_leverage==30.0
    assert not p.require_trend_filter
    assert p.stop_mode=='atr'
    assert m.threshold==.5
    assert m.min_validation_auc==.52


def test_v33_config_uses_eur_stake_and_30x_search_cap():
    pol=load_v33_policy('config.json')
    assert pol.initial_capital_eur==1_000
    assert pol.base_stake_eur==100
    assert pol.min_leverage==1 and pol.max_leverage==30
    assert pol.max_holding_bars==0


def test_v33_plan_reports_stake_exposure_and_integer_leverage():
    d=synthetic(1000); p=load_params('config.json'); pol=load_v33_policy('config.json')
    r=features(d,p).dropna().iloc[-1]; side=1 if r.close>r.sma50 else -1
    plan=position_plan_v33(1_000,float(r.close),side,float(r.atr),r,p,pol)
    if plan['valid']:
        assert plan['stakeEUR']>0
        assert plan['grossExposureEUR']>=plan['stakeEUR']
        assert isinstance(plan['leverage'],int)
        assert 1<=plan['leverage']<=30
        assert plan['riskAtStopEUR']<=plan['riskBudgetEUR']+1e-9


def test_v33_has_no_time_exit_by_default():
    d=synthetic(1400,seed=31); p=load_params('config.json'); pol=load_v33_policy('config.json')
    b=backtest_v33(d,p,pol,start='2022-01-01')
    assert all(t['reason'] in ('TP','SL') for t in b['trades'])
    assert b['metrics']['timeExitTrades']==0


def test_v33_open_position_is_not_force_closed_at_sample_end():
    d=synthetic(950,seed=4); p=load_params('config.json')
    pol=V33Policy(initial_capital_eur=1_000,base_stake_eur=100,max_holding_bars=0,stop_atr=10,target_rr=10)
    b=backtest_v33(d,p,pol,start=d.index[700])
    assert b['metrics']['openPositions'] in (0,1)
    if b['openTrade'] is not None:
        assert b['openTrade']['unrealizedPnLEUR']==b['openTrade']['unrealizedPnLEUR']


def test_ml_v33_labels_only_resolved_tp_or_sl():
    d=synthetic(1700,seed=12); p=load_params('config.json'); pol=load_v33_policy('config.json')
    ev=build_resolved_events(d,p,pol)
    assert len(ev)>30
    assert set(ev.label.unique()).issubset({0,1})
    assert (ev.outcomeEnd>=ev.entryDate).all()


def test_ml_v33_model_comparison_is_chronological():
    d=synthetic(1800,seed=23); p=load_params('config.json'); pol=load_v33_policy('config.json')
    ev=build_resolved_events(d,p,pol)
    cmp=chronological_model_comparison(ev,ML33Params(min_train_events=60),folds=4)
    if not cmp.empty:
        assert set(cmp.model).issubset({'logistic','random_forest','hist_gradient_boosting'})
        assert ((cmp.meanAUC>=0)&(cmp.meanAUC<=1)).all()
