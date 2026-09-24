from __future__ import annotations
import json
from dataclasses import fields
from pathlib import Path

from .strategy_engine import Params
from .ml_meta import MLParams


def _read(path: str | Path | None) -> dict:
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_params(path: str | Path | None = None, **overrides) -> Params:
    """Load strategy/account settings.

    Supports the user-friendly nested ``config.json`` and the older flat Params JSON.
    Unknown keys in a flat config are rejected so typos do not silently change tests.
    """
    raw = _read(path)
    if any(k in raw for k in ('account','risk','execution','strategy','ml','backtest')):
        a=raw.get('account',{}); r=raw.get('risk',{}); e=raw.get('execution',{}); s=raw.get('strategy',{}); m=raw.get('ml',{})
        data = {
            'initial_capital_eur': a.get('initialCapitalEUR', 10_000.0),
            'sizing_mode': r.get('sizingMode','adaptive_risk'),
            'fixed_notional_eur': r.get('fixedNotionalEUR',10_000.0),
            'risk_per_trade': r.get('baseRiskPerTrade',0.005),
            'min_risk_per_trade': r.get('minRiskPerTrade',0.0025),
            'max_risk_per_trade': r.get('maxRiskPerTrade',0.01),
            'max_leverage': r.get('maxLeverage',3.0),
            'lot_step_units': r.get('lotStepUnits',1000.0),
            'min_units': r.get('minUnits',1000.0),
            'absolute_max_units': r.get('absoluteMaxUnits',0.0),
            'volatility_target': r.get('volatilityTarget',0.08),
            'vol_scale_min': r.get('volScaleMin',0.65),
            'vol_scale_max': r.get('volScaleMax',1.25),
            'spread_pips': e.get('spreadPips',0.8),
            'slippage_pips_per_side': e.get('slippagePipsPerSide',0.1),
            'commission_per_million_per_side_eur': e.get('commissionPerMillionPerSideEUR',0.0),
            'rsi_period': s.get('rsiPeriod',14),
            'rsi_long': s.get('rsiLong',55.0),
            'rsi_short': s.get('rsiShort',45.0),
            'sma_trend': s.get('smaTrend',50),
            'bb_period': s.get('bbPeriod',20),
            'atr_period': s.get('atrPeriod',14),
            'adx_period': s.get('adxPeriod',14),
            'stop_atr': s.get('stopATR',1.5),
            'take_atr': s.get('takeATR',2.2),
            'max_holding_bars': s.get('maxHoldingBars',5),
            'require_macd': s.get('requireMACD',True),
            'require_adx': s.get('requireADX',False),
            'min_adx': s.get('minADX',18.0),
            'enable_range_module': s.get('enableRangeModule',False),
            'range_adx_max': s.get('rangeADXMax',17.0),
            'range_z': s.get('rangeZ',1.5),
            'range_rsi_low': s.get('rangeRSILow',35.0),
            'range_rsi_high': s.get('rangeRSIHigh',65.0),
            'ml_probability_threshold': m.get('threshold',0.60),
            'ml_confidence_reference': m.get('confidenceReference',0.78),
        }
    else:
        valid={f.name for f in fields(Params)}
        unknown=sorted(set(raw)-valid)
        if unknown:
            raise ValueError(f'Unknown strategy config keys: {unknown}')
        data=dict(raw)
    data.update({k:v for k,v in overrides.items() if v is not None})
    return Params(**data)


def load_ml_params(path: str | Path | None = None, **overrides) -> MLParams:
    raw=_read(path); m=raw.get('ml',{}) if isinstance(raw,dict) else {}
    data={
        'threshold': m.get('threshold',0.60),
        'retrain_every_bars': m.get('retrainEveryBars',60),
        'min_train_events': m.get('minTrainEvents',100),
        'calibration_fraction': m.get('calibrationFraction',0.20),
        'feature_set': m.get('featureSet','core'),
        'max_train_years': m.get('maxTrainYears',8),
        'C': m.get('C',0.7),
    }
    data.update({k:v for k,v in overrides.items() if v is not None})
    return MLParams(**data)


def load_backtest_settings(path: str | Path | None = None) -> dict:
    raw=_read(path)
    return dict(raw.get('backtest',{})) if isinstance(raw,dict) else {}
