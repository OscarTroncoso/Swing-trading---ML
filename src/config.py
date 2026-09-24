from __future__ import annotations
import json
from dataclasses import fields
from pathlib import Path

from .strategy_engine import Params
from .ml_meta import MLParams


def _read(path: str | Path | None) -> dict:
    if path is None: return {}
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_params(path: str | Path | None = None, **overrides) -> Params:
    raw=_read(path)
    if any(k in raw for k in ('account','risk','execution','strategy','trend','stop','profitProtection','ml','backtest')):
        a=raw.get('account',{}); r=raw.get('risk',{}); e=raw.get('execution',{}); s=raw.get('strategy',{})
        t=raw.get('trend',{}); st=raw.get('stop',{}); pp=raw.get('profitProtection',{}); m=raw.get('ml',{})
        data={
            'initial_capital_eur':a.get('initialCapitalEUR',10_000.0),
            'sizing_mode':r.get('sizingMode','adaptive_risk'),'fixed_notional_eur':r.get('fixedNotionalEUR',10_000.0),
            'risk_per_trade':r.get('baseRiskPerTrade',0.005),'min_risk_per_trade':r.get('minRiskPerTrade',0.0025),
            'max_risk_per_trade':r.get('maxRiskPerTrade',0.0075),'max_leverage':r.get('maxLeverage',1.5),
            'lot_step_units':r.get('lotStepUnits',1000.0),'min_units':r.get('minUnits',1000.0),'absolute_max_units':r.get('absoluteMaxUnits',0.0),
            'volatility_target':r.get('volatilityTarget',0.08),'vol_scale_min':r.get('volScaleMin',0.65),'vol_scale_max':r.get('volScaleMax',1.25),
            'quality_scale_min':r.get('qualityScaleMin',0.75),'quality_scale_max':r.get('qualityScaleMax',1.15),
            'spread_pips':e.get('spreadPips',0.8),'slippage_pips_per_side':e.get('slippagePipsPerSide',0.1),
            'commission_per_million_per_side_eur':e.get('commissionPerMillionPerSideEUR',0.0),
            'rsi_period':s.get('rsiPeriod',14),'rsi_long':s.get('rsiLong',55.0),'rsi_short':s.get('rsiShort',45.0),
            'sma_trend':s.get('smaTrend',50),'require_macd':s.get('requireMACD',True),
            'bb_period':s.get('bbPeriod',20),'atr_period':s.get('atrPeriod',14),'adx_period':s.get('adxPeriod',14),
            'trend_ema_fast':t.get('emaFast',50),'trend_ema_slow':t.get('emaSlow',200),'trend_slope_bars':t.get('slopeBars',10),
            'min_trend_score':t.get('minScore',3),'min_adx':t.get('minADX',18.0),'range_reject_adx':t.get('rangeRejectADX',15.0),
            'flat_ema_slope_abs':t.get('flatEMASlopeAbs',0.0008),'require_trend_filter':t.get('enabled',True),
            'vol_regime_lookback':t.get('volRegimeLookback',252),'vol_regime_min_obs':t.get('volRegimeMinObs',60),
            'stop_mode':st.get('mode','hybrid_structure'),'stop_atr_low_vol':st.get('atrLowVol',1.35),
            'stop_atr_normal':st.get('atrNormal',1.50),'stop_atr_high_vol':st.get('atrHighVol',1.75),
            'structure_lookback':st.get('structureLookback',8),'structure_buffer_atr':st.get('structureBufferATR',0.15),
            'max_stop_atr':st.get('maxStopATR',2.50),'target_rr':st.get('targetRR',1.60),'max_holding_bars':st.get('maxHoldingBars',5),
            'profit_protection_enabled':pp.get('enabled',False),'breakeven_trigger_r':pp.get('breakevenTriggerR',1.25),
            'trailing_trigger_r':pp.get('trailingTriggerR',1.75),'trailing_atr':pp.get('trailingATR',1.50),
            'trailing_structure_lookback':pp.get('structureLookback',5),
            'enable_range_module':s.get('enableRangeModule',False),'range_adx_max':s.get('rangeADXMax',17.0),
            'range_z':s.get('rangeZ',1.5),'range_rsi_low':s.get('rangeRSILow',35.0),'range_rsi_high':s.get('rangeRSIHigh',65.0),
            'ml_probability_threshold':m.get('threshold',0.55),'ml_confidence_reference':m.get('confidenceReference',0.70),
        }
    else:
        valid={f.name for f in fields(Params)}; unknown=sorted(set(raw)-valid)
        if unknown: raise ValueError(f'Unknown strategy config keys: {unknown}')
        data=dict(raw)
    data.update({k:v for k,v in overrides.items() if v is not None})
    return Params(**data)


def load_ml_params(path: str | Path | None = None, **overrides) -> MLParams:
    raw=_read(path); m=raw.get('ml',{}) if isinstance(raw,dict) else {}
    data={'threshold':m.get('threshold',0.55),'retrain_every_bars':m.get('retrainEveryBars',60),'min_train_events':m.get('minTrainEvents',100),
          'calibration_fraction':m.get('calibrationFraction',0.20),'feature_set':m.get('featureSet','core'),'max_train_years':m.get('maxTrainYears',8),'C':m.get('C',0.7),'min_validation_auc':m.get('minValidationAUC',0.52)}
    data.update({k:v for k,v in overrides.items() if v is not None}); return MLParams(**data)


def load_backtest_settings(path: str | Path | None = None) -> dict:
    raw=_read(path); return dict(raw.get('backtest',{})) if isinstance(raw,dict) else {}


def load_v4_params(path: str | Path | None = None, **overrides):
    from .strategy_v4 import V4Params
    raw = _read(path)
    a = raw.get('account', {}) if isinstance(raw, dict) else {}
    r = raw.get('risk', {}) if isinstance(raw, dict) else {}
    e = raw.get('execution', {}) if isinstance(raw, dict) else {}
    s = raw.get('strategy', {}) if isinstance(raw, dict) else {}
    t = raw.get('trend', {}) if isinstance(raw, dict) else {}
    st = raw.get('stop', {}) if isinstance(raw, dict) else {}
    pp = raw.get('profitProtection', {}) if isinstance(raw, dict) else {}
    data = {
        'initial_capital_eur': a.get('initialCapitalEUR', 1000.0),
        'risk_per_trade': r.get('baseRiskPerTrade', 0.0075),
        'min_risk_per_trade': r.get('minRiskPerTrade', 0.0035),
        'max_risk_per_trade': r.get('maxRiskPerTrade', 0.0100),
        'base_position_eur': r.get('basePositionEUR', 100.0),
        'min_position_eur': r.get('minPositionEUR', 25.0),
        'max_position_pct_equity': r.get('maxPositionPctEquity', 0.25),
        'min_trade_leverage': r.get('minTradeLeverage', 1.0),
        'max_trade_leverage': r.get('maxTradeLeverage', 30.0),
        'leverage_step': r.get('leverageStep', 1.0),
        'volatility_target': r.get('volatilityTarget', 0.08),
        'vol_scale_min': r.get('volScaleMin', 0.70),
        'vol_scale_max': r.get('volScaleMax', 1.15),
        'quality_scale_min': r.get('qualityScaleMin', 0.70),
        'quality_scale_max': r.get('qualityScaleMax', 1.20),
        'spread_pips': e.get('spreadPips', 0.8),
        'slippage_pips_per_side': e.get('slippagePipsPerSide', 0.1),
        'commission_per_million_per_side_eur': e.get('commissionPerMillionPerSideEUR', 0.0),
        'rsi_period': s.get('rsiPeriod', 14),
        'rsi_long': s.get('rsiLong', 55.0),
        'rsi_short': s.get('rsiShort', 45.0),
        'sma_trend': s.get('smaTrend', 50),
        'require_macd': s.get('requireMACD', True),
        'bb_period': s.get('bbPeriod', 20),
        'atr_period': s.get('atrPeriod', 14),
        'adx_period': s.get('adxPeriod', 14),
        'trend_ema_fast': t.get('emaFast', 50),
        'trend_ema_slow': t.get('emaSlow', 200),
        'trend_slope_bars': t.get('slopeBars', 10),
        'min_trend_score': t.get('minScore', 3),
        'min_adx': t.get('minADX', 18.0),
        'range_reject_adx': t.get('rangeRejectADX', 15.0),
        'flat_ema_slope_abs': t.get('flatEMASlopeAbs', 0.0008),
        'require_trend_filter': False,
        'hard_trend_reject': t.get('hardRejectEnabled', True),
        'hard_trend_reject_adx': t.get('hardRejectADX', 28.0),
        'hard_trend_max_alignment_score': t.get('hardRejectMaxAlignmentScore', 1),
        'vol_regime_lookback': t.get('volRegimeLookback', 252),
        'vol_regime_min_obs': t.get('volRegimeMinObs', 60),
        'stop_mode': st.get('mode', 'adaptive_structure'),
        'stop_atr_low_vol': st.get('atrLowVol', 1.35),
        'stop_atr_normal': st.get('atrNormal', 1.50),
        'stop_atr_high_vol': st.get('atrHighVol', 1.70),
        'structure_lookback': st.get('structureLookback', 8),
        'structure_buffer_atr': st.get('structureBufferATR', 0.10),
        'max_stop_atr': st.get('maxStructureATR', 2.75),
        'target_rr': st.get('targetRR', 1.80),
        'max_holding_bars': 0,
        'profit_protection_enabled': pp.get('enabled', False),
        'breakeven_trigger_r': pp.get('breakevenTriggerR', 1.25),
        'trailing_trigger_r': pp.get('trailingTriggerR', 1.75),
        'trailing_atr': pp.get('trailingATR', 1.5),
        'trailing_structure_lookback': pp.get('structureLookback', 5),
        'enable_range_module': s.get('enableRangeModule', False),
        'range_adx_max': s.get('rangeADXMax', 17.0),
        'range_z': s.get('rangeZ', 1.5),
        'range_rsi_low': s.get('rangeRSILow', 35.0),
        'range_rsi_high': s.get('rangeRSIHigh', 65.0),
    }
    data.update({k:v for k,v in overrides.items() if v is not None})
    return V4Params(**data)
