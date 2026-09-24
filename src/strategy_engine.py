"""EUR/USD swing strategy engine v3.2 — trend + structural risk.

Design goals
------------
* EUR account equity (10,000 means EUR 10,000, never 10,000 hard-coded FX units).
* Signal at close t; execution at open t+1.
* All sizing/ATR/trend inputs are frozen on the signal bar.
* Trend-regime filter separates directional momentum from range/noise.
* Hybrid stop uses both volatility (ATR) and recent market structure.
* Trades whose structurally valid stop is excessively wide are skipped.
* Position size is derived from EUR risk budget and reports leverage explicitly.
* Optional delayed profit protection never uses current-bar information retroactively.
* Backtest records MAE/MFE/R-multiples/capture ratio for stop diagnostics.

Research software only. Backtests do not guarantee executable fills or future returns.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PIP = 0.0001


@dataclass(frozen=True)
class Params:
    # Account / sizing
    initial_capital_eur: float = 10_000.0
    sizing_mode: str = "adaptive_risk"  # fixed_notional | risk | adaptive_risk
    fixed_notional_eur: float = 10_000.0
    risk_per_trade: float = 0.005
    min_risk_per_trade: float = 0.0025
    max_risk_per_trade: float = 0.0075
    max_leverage: float = 1.5
    lot_step_units: float = 1_000.0
    min_units: float = 1_000.0
    absolute_max_units: float = 0.0
    volatility_target: float = 0.08
    vol_scale_min: float = 0.65
    vol_scale_max: float = 1.25
    quality_scale_min: float = 0.75
    quality_scale_max: float = 1.15

    # Execution
    spread_pips: float = 0.8
    slippage_pips_per_side: float = 0.1
    commission_per_million_per_side_eur: float = 0.0

    # Core direction signal
    rsi_period: int = 14
    rsi_long: float = 55.0
    rsi_short: float = 45.0
    sma_trend: int = 50
    require_macd: bool = True

    # Trend / regime confirmation
    trend_ema_fast: int = 50
    trend_ema_slow: int = 200
    trend_slope_bars: int = 10
    min_trend_score: int = 3            # out of 5
    min_adx: float = 18.0
    range_reject_adx: float = 15.0
    flat_ema_slope_abs: float = 0.0008   # ~0.08% over slope window
    require_trend_filter: bool = True

    # Indicators / volatility
    bb_period: int = 20
    atr_period: int = 14
    adx_period: int = 14
    vol_regime_lookback: int = 252
    vol_regime_min_obs: int = 60

    # Hybrid stop + target
    stop_mode: str = "hybrid_structure"  # atr | hybrid_structure
    stop_atr_low_vol: float = 1.35
    stop_atr_normal: float = 1.50
    stop_atr_high_vol: float = 1.75
    structure_lookback: int = 8
    structure_buffer_atr: float = 0.15
    max_stop_atr: float = 2.50
    target_rr: float = 1.60
    max_holding_bars: int = 5

    # Optional delayed profit protection; deliberately off until it wins OOS.
    profit_protection_enabled: bool = False
    breakeven_trigger_r: float = 1.25
    trailing_trigger_r: float = 1.75
    trailing_atr: float = 1.50
    trailing_structure_lookback: int = 5

    # Optional old range module retained only for experiments, off in production.
    enable_range_module: bool = False
    range_adx_max: float = 17.0
    range_z: float = 1.5
    range_rsi_low: float = 35.0
    range_rsi_high: float = 65.0

    # ML remains a challenger; these fields only support research integration.
    ml_probability_threshold: float = 0.55
    ml_confidence_reference: float = 0.70


# ------------------------------- data --------------------------------------
def _utc_ts(value) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    aliases = {"datetime": "date", "timestamp": "date", "time": "date"}
    x = x.rename(columns={k: v for k, v in aliases.items() if k in x.columns})
    if "date" in x.columns:
        x["date"] = pd.to_datetime(x["date"], utc=True, errors="coerce")
        x = x.dropna(subset=["date"]).set_index("date")
    if not isinstance(x.index, pd.DatetimeIndex):
        x.index = pd.to_datetime(x.index, utc=True, errors="coerce")
    elif x.index.tz is None:
        x.index = x.index.tz_localize("UTC")
    else:
        x.index = x.index.tz_convert("UTC")
    needed = ["open", "high", "low", "close"]
    missing = [c for c in needed if c not in x.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")
    for c in needed:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x[needed].dropna().sort_index()
    return x.loc[~x.index.duplicated(keep="last")]


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    raw = pd.read_csv(path)
    if not {"Open", "High", "Low", "Close"}.issubset(raw.columns) and not {"open", "high", "low", "close"}.issubset(raw.columns):
        raw = pd.read_csv(path, skiprows=[1])
    return _norm_cols(raw)


def fetch_yahoo(ticker: str = "EURUSD=X", period: str = "10y", start: str | None = None, end: str | None = None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed. Run pip install -r requirements.txt") from exc
    kwargs = dict(interval="1d", auto_adjust=True, progress=False)
    if start is not None:
        kwargs["start"] = start
        if end is not None:
            kwargs["end"] = end
    else:
        kwargs["period"] = period
    d = yf.download(ticker, **kwargs)
    if d.empty:
        raise RuntimeError("Yahoo Finance returned no data.")
    now_utc = pd.Timestamp.now(tz="UTC")
    if len(d) and pd.Timestamp(d.index[-1]).date() == now_utc.date() and now_utc.hour < 22:
        d = d.iloc[:-1]
    if d.empty:
        raise RuntimeError("Yahoo Finance returned only an incomplete current-day bar.")
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.reset_index().rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close"})
    return _norm_cols(d)


# ---------------------------- indicators ----------------------------------
def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out[(avg_loss == 0) & (avg_gain > 0)] = 100
    out[(avg_loss == 0) & (avg_gain == 0)] = 50
    return out


def rsi_original(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(n).sum() / n
    avg_loss = losses.rolling(n).sum() / n
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out[(avg_loss == 0) & (avg_gain > 0)] = 100
    out[(avg_loss == 0) & (avg_gain == 0)] = 50
    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df.close.shift(1)
    return pd.concat([(df.high - df.low).abs(), (df.high - prev_close).abs(), (df.low - prev_close).abs()], axis=1).max(axis=1)


def atr_wilder(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def dmi_adx_wilder(df: pd.DataFrame, n: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    up = df.high.diff()
    down = -df.low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr = atr_wilder(df, n)
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return plus_di, minus_di, adx


def adx_wilder(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return dmi_adx_wilder(df, n)[2]


def features(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    x = _norm_cols(df)
    c = x.close
    x["rsi"] = rsi_wilder(c, p.rsi_period)
    x["rsi_original"] = rsi_original(c, p.rsi_period)
    x["sma50"] = c.rolling(p.sma_trend).mean()
    x["ema20"] = c.ewm(span=20, adjust=False).mean()
    x["ema50"] = c.ewm(span=p.trend_ema_fast, adjust=False).mean()
    x["ema200"] = c.ewm(span=p.trend_ema_slow, adjust=False, min_periods=p.trend_ema_slow).mean()
    x["ema50_slope"] = x.ema50.pct_change(p.trend_slope_bars)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    x["macd"] = ema12 - ema26
    x["macd_signal"] = x.macd.ewm(span=9, adjust=False).mean()
    x["macd_hist"] = x.macd - x.macd_signal
    x["bb_mid"] = c.rolling(p.bb_period).mean()
    x["bb_std"] = c.rolling(p.bb_period).std(ddof=0)
    x["bb_upper"] = x.bb_mid + 2 * x.bb_std
    x["bb_lower"] = x.bb_mid - 2 * x.bb_std
    x["bb_z"] = (c - x.bb_mid) / x.bb_std.replace(0, np.nan)
    x["atr"] = atr_wilder(x, p.atr_period)
    plus_di, minus_di, adx = dmi_adx_wilder(x, p.adx_period)
    x["plus_di"] = plus_di
    x["minus_di"] = minus_di
    x["adx"] = adx
    x["rv20"] = c.pct_change().rolling(20).std(ddof=0) * math.sqrt(252)
    x["ret1"] = c.pct_change(1)
    x["ret5"] = c.pct_change(5)
    x["ret20"] = c.pct_change(20)
    x["sma_gap"] = c / x.sma50 - 1
    x["ema200_gap"] = c / x.ema200 - 1
    x["atr_pct"] = x.atr / c
    x["macd_atr"] = x.macd_hist / x.atr.replace(0, np.nan)
    x["range_pct"] = (x.high - x.low) / c
    x["body_pct"] = (x.close - x.open) / c

    # Past-only regime references: today's rv20 is compared only to PRIOR history.
    prior_rv = x.rv20.shift(1)
    x["rv20_q33"] = prior_rv.rolling(p.vol_regime_lookback, min_periods=p.vol_regime_min_obs).quantile(0.33)
    x["rv20_q67"] = prior_rv.rolling(p.vol_regime_lookback, min_periods=p.vol_regime_min_obs).quantile(0.67)

    # Structure is known at signal close. Including the completed signal bar is not look-ahead.
    x["swing_low"] = x.low.rolling(p.structure_lookback, min_periods=max(3, p.structure_lookback // 2)).min()
    x["swing_high"] = x.high.rolling(p.structure_lookback, min_periods=max(3, p.structure_lookback // 2)).max()
    x["trail_swing_low"] = x.low.rolling(p.trailing_structure_lookback, min_periods=2).min()
    x["trail_swing_high"] = x.high.rolling(p.trailing_structure_lookback, min_periods=2).max()
    return x


# ----------------------------- trend / signals -----------------------------
def volatility_regime(r: pd.Series) -> str:
    rv = r.get("rv20", np.nan); q33 = r.get("rv20_q33", np.nan); q67 = r.get("rv20_q67", np.nan)
    if pd.isna(rv) or pd.isna(q33) or pd.isna(q67):
        return "NORMAL"
    if rv <= q33:
        return "LOW"
    if rv >= q67:
        return "HIGH"
    return "NORMAL"


def trend_context(r: pd.Series, side: int, p: Params) -> dict:
    if side not in (-1, 1):
        return {"score": 0, "maxScore": 5, "accepted": False, "regime": "NO_SIGNAL", "components": {}}
    needed = ["ema50", "ema200", "ema50_slope", "adx", "plus_di", "minus_di"]
    if any(pd.isna(r.get(c, np.nan)) for c in needed):
        return {"score": 0, "maxScore": 5, "accepted": False, "regime": "WARMUP", "components": {}}
    slope = float(r.ema50_slope)
    components = {
        "priceVsEMA200": bool(r.close > r.ema200) if side == 1 else bool(r.close < r.ema200),
        "EMA50VsEMA200": bool(r.ema50 > r.ema200) if side == 1 else bool(r.ema50 < r.ema200),
        "EMA50Slope": bool(slope > p.flat_ema_slope_abs) if side == 1 else bool(slope < -p.flat_ema_slope_abs),
        "DMI": bool(r.plus_di > r.minus_di) if side == 1 else bool(r.minus_di > r.plus_di),
        "ADX": bool(r.adx >= p.min_adx),
    }
    score = sum(int(v) for v in components.values())
    flat = abs(slope) < p.flat_ema_slope_abs
    range_like = bool(r.adx < p.range_reject_adx and flat)
    if range_like:
        regime = "RANGE"
    elif score >= 4 and r.adx >= max(p.min_adx, 22):
        regime = "STRONG_TREND"
    elif score >= p.min_trend_score:
        regime = "TREND"
    else:
        regime = "WEAK_OR_MISALIGNED"
    accepted = (score >= p.min_trend_score) and not range_like
    return {"score": int(score), "maxScore": 5, "accepted": bool(accepted), "regime": regime,
            "volatilityRegime": volatility_regime(r), "components": components}




def market_trend_context(r: pd.Series, p: Params) -> dict:
    if pd.isna(r.get("ema50", np.nan)) or pd.isna(r.get("ema200", np.nan)):
        return {"score":0,"maxScore":5,"accepted":False,"regime":"WARMUP","volatilityRegime":volatility_regime(r),"components":{},"bias":"NEUTRAL"}
    if r.ema50 > r.ema200:
        side=1; bias="BULLISH"
    elif r.ema50 < r.ema200:
        side=-1; bias="BEARISH"
    else:
        side=1; bias="NEUTRAL"
    ctx=trend_context(r,side,p)
    return {**ctx,"bias":bias,"side":side if bias!="NEUTRAL" else 0}

def technical_candidate_row(r: pd.Series, p: Params) -> int:
    if pd.isna(r.get("rsi", np.nan)) or pd.isna(r.get("sma50", np.nan)) or pd.isna(r.get("atr", np.nan)):
        return 0
    long_mom = r.rsi > p.rsi_long and r.close > r.sma50
    short_mom = r.rsi < p.rsi_short and r.close < r.sma50
    if p.require_macd:
        long_mom = long_mom and r.macd_hist > 0
        short_mom = short_mom and r.macd_hist < 0
    if long_mom:
        return 1
    if short_mom:
        return -1
    if p.enable_range_module and pd.notna(r.adx) and r.adx <= p.range_adx_max:
        if r.bb_z <= -p.range_z and r.rsi <= p.range_rsi_low: return 1
        if r.bb_z >= p.range_z and r.rsi >= p.range_rsi_high: return -1
    return 0


def signal_decision(r: pd.Series, p: Params) -> dict:
    side = technical_candidate_row(r, p)
    if side == 0:
        return {"side": 0, "candidate": False, "accepted": False, "trend": market_trend_context(r, p), "reason": "NO_TECHNICAL_CANDIDATE"}
    ctx = trend_context(r, side, p)
    accepted = True if not p.require_trend_filter else ctx["accepted"]
    return {"side": side if accepted else 0, "candidateSide": side, "candidate": True, "accepted": bool(accepted),
            "trend": ctx, "reason": "ACCEPT" if accepted else f"REJECT_{ctx['regime']}"}


def signal_row(r: pd.Series, p: Params) -> int:
    return int(signal_decision(r, p)["side"])


def broad_candidate_row(r: pd.Series) -> int:
    if pd.isna(r.get("rsi", np.nan)) or pd.isna(r.get("sma50", np.nan)):
        return 0
    if r.rsi > 52 and r.close > r.sma50: return 1
    if r.rsi < 48 and r.close < r.sma50: return -1
    return 0


def legacy_signal_row(r: pd.Series) -> int:
    if pd.isna(r.rsi_original) or pd.isna(r.bb_lower) or pd.isna(r.bb_upper): return 0
    if r.close <= r.bb_lower or r.rsi_original < 45: return 1
    if r.close >= r.bb_upper or r.rsi_original > 55: return -1
    return 0


# ----------------------- account / stop / leverage -------------------------
def _commission(units: float, p: Params) -> float:
    return (units / 1_000_000.0) * p.commission_per_million_per_side_eur


def pnl_eur(entry: float, exit_px: float, units: float, side: int) -> float:
    if exit_px <= 0: raise ValueError("exit_px must be positive")
    return ((exit_px - entry) * units * side) / exit_px


def dynamic_stop_atr(r: pd.Series, p: Params) -> float:
    regime = volatility_regime(r)
    if regime == "LOW": return p.stop_atr_low_vol
    if regime == "HIGH": return p.stop_atr_high_vol
    return p.stop_atr_normal


def quality_multiplier(r: pd.Series, side: int, p: Params) -> float:
    ctx = trend_context(r, side, p)
    score = ctx["score"]
    if score <= p.min_trend_score:
        q = p.quality_scale_min
    elif score >= ctx["maxScore"]:
        q = p.quality_scale_max
    else:
        span = max(1, ctx["maxScore"] - p.min_trend_score)
        frac = (score - p.min_trend_score) / span
        q = p.quality_scale_min + frac * (p.quality_scale_max - p.quality_scale_min)
    if ctx.get("regime") == "STRONG_TREND": q = min(p.quality_scale_max, q * 1.05)
    if ctx.get("volatilityRegime") == "HIGH": q *= 0.90
    return float(np.clip(q, p.quality_scale_min, p.quality_scale_max))


def adaptive_risk_pct(r: pd.Series, p: Params, side: int = 1, ml_probability: float | None = None) -> float:
    risk = p.risk_per_trade
    if pd.notna(r.get("rv20", np.nan)) and float(r.rv20) > 0:
        vol_scale = float(np.clip(p.volatility_target / float(r.rv20), p.vol_scale_min, p.vol_scale_max))
        risk *= vol_scale
    risk *= quality_multiplier(r, side, p)
    if ml_probability is not None:
        if ml_probability < p.ml_probability_threshold: return 0.0
        den = max(p.ml_confidence_reference - p.ml_probability_threshold, 1e-6)
        q = float(np.clip((ml_probability - p.ml_probability_threshold) / den, 0.0, 1.0))
        risk *= 0.85 + 0.30 * q
    return float(np.clip(risk, p.min_risk_per_trade, p.max_risk_per_trade))


def _round_units_down(units: float, step: float) -> float:
    if step <= 0: return units
    return math.floor(max(0.0, units) / step) * step


def _stop_and_target(entry: float, side: int, atr_signal: float, r: pd.Series, p: Params) -> dict:
    mult = dynamic_stop_atr(r, p)
    atr_dist = max(mult * atr_signal, PIP)
    if side == 1:
        atr_stop = entry - atr_dist
        structure = float(r.swing_low) - p.structure_buffer_atr * atr_signal if pd.notna(r.get("swing_low", np.nan)) else atr_stop
        raw_stop = min(atr_stop, structure) if p.stop_mode == "hybrid_structure" else atr_stop
        stop_dist = entry - raw_stop
    else:
        atr_stop = entry + atr_dist
        structure = float(r.swing_high) + p.structure_buffer_atr * atr_signal if pd.notna(r.get("swing_high", np.nan)) else atr_stop
        raw_stop = max(atr_stop, structure) if p.stop_mode == "hybrid_structure" else atr_stop
        stop_dist = raw_stop - entry
    stop_atr_dist = stop_dist / max(atr_signal, PIP)
    valid = bool(stop_atr_dist <= p.max_stop_atr and stop_dist > 0)
    source = "ATR"
    if p.stop_mode == "hybrid_structure" and abs(raw_stop - atr_stop) > PIP / 10:
        source = "STRUCTURE"
    raw_take = entry + side * p.target_rr * stop_dist
    return {"valid": valid, "rejectionReason": None if valid else "STOP_TOO_WIDE",
            "stopLoss": float(max(raw_stop, PIP)), "takeProfit": float(max(raw_take, PIP)),
            "stopDistance": float(stop_dist), "stopDistanceATR": float(stop_atr_dist),
            "stopATRMultiplier": float(mult), "stopSource": source, "targetRR": float(p.target_rr)}


def position_plan(equity_eur: float, entry: float, side: int, atr_signal: float, signal_row_data: pd.Series,
                  p: Params, ml_probability: float | None = None) -> dict:
    if equity_eur <= 0 or entry <= 0 or side not in (-1, 1): raise ValueError("Invalid equity/entry/side")
    adverse = p.spread_pips * PIP / 2 + p.slippage_pips_per_side * PIP
    levels = _stop_and_target(entry, side, atr_signal, signal_row_data, p)
    trend = trend_context(signal_row_data, side, p)
    if not levels["valid"]:
        return {"valid": False, "rejectionReason": levels["rejectionReason"], "equityEUR": round(equity_eur,2),
                "trend": trend, **levels, "recommended": {"units":0.0,"lots":0.0,"notionalEUR":0.0,"leverage":0.0,
                "usesLeverage":False,"riskAtStopEUR":0.0,"riskAtStopPct":0.0,"targetGainEUR":0.0,"rewardRisk":0.0},
                "noLeverageAlternative": {"units":0.0,"lots":0.0,"notionalEUR":0.0,"leverage":0.0,"usesLeverage":False,
                "riskAtStopEUR":0.0,"riskAtStopPct":0.0,"targetGainEUR":0.0,"rewardRisk":0.0}}

    raw_stop = levels["stopLoss"]; raw_take = levels["takeProfit"]
    stop_fill = max(PIP, raw_stop - adverse if side == 1 else raw_stop + adverse)
    take_fill = max(PIP, raw_take - adverse if side == 1 else raw_take + adverse)

    if p.sizing_mode == "fixed_notional":
        target_risk_pct = None; desired_units = p.fixed_notional_eur
    else:
        target_risk_pct = p.risk_per_trade if p.sizing_mode == "risk" else adaptive_risk_pct(signal_row_data, p, side, ml_probability)
        risk_budget = equity_eur * target_risk_pct
        loss_per_unit = abs(pnl_eur(entry, stop_fill, 1.0, side)) + 2*(p.commission_per_million_per_side_eur/1_000_000.0)
        desired_units = risk_budget / max(loss_per_unit, 1e-12)

    leverage_cap = equity_eur * p.max_leverage
    if p.absolute_max_units > 0: leverage_cap = min(leverage_cap, p.absolute_max_units)
    chosen_units = _round_units_down(min(desired_units, leverage_cap), p.lot_step_units)
    if chosen_units < p.min_units: chosen_units = 0.0
    no_lev_units = _round_units_down(min(desired_units, equity_eur), p.lot_step_units)
    if no_lev_units < p.min_units: no_lev_units = 0.0

    def option(units: float, name: str) -> dict:
        if units <= 0:
            return {"name":name,"units":0.0,"lots":0.0,"notionalEUR":0.0,"leverage":0.0,"usesLeverage":False,
                    "riskAtStopEUR":0.0,"riskAtStopPct":0.0,"targetGainEUR":0.0,"rewardRisk":0.0}
        risk_eur = abs(pnl_eur(entry, stop_fill, units, side)) + 2*_commission(units,p)
        gain_eur = pnl_eur(entry, take_fill, units, side) - 2*_commission(units,p)
        lev = units/equity_eur
        return {"name":name,"units":round(units,0),"lots":round(units/100_000,4),"notionalEUR":round(units,2),
                "leverage":round(lev,3),"usesLeverage":bool(lev>1.0001),"riskAtStopEUR":round(risk_eur,2),
                "riskAtStopPct":round(risk_eur/equity_eur*100,3),"targetGainEUR":round(gain_eur,2),
                "rewardRisk":round(gain_eur/risk_eur,3) if risk_eur>0 else 0.0}

    rec = option(chosen_units,"risk_target"); nolev=option(no_lev_units,"no_leverage")
    if chosen_units <= 0: label="SKIP_SIZE_TOO_SMALL"
    elif desired_units > leverage_cap + 1e-9: label="LEVERAGE_CAPPED"
    elif rec["usesLeverage"]: label="RISK_TARGET_WITH_LEVERAGE"
    else: label="RISK_TARGET_NO_LEVERAGE"
    return {"valid":chosen_units>0,"rejectionReason":None if chosen_units>0 else "SIZE_TOO_SMALL",
            "equityEUR":round(equity_eur,2),"sizingMode":p.sizing_mode,"baseRiskPct":round(p.risk_per_trade*100,3),
            "appliedRiskPct":None if target_risk_pct is None else round(target_risk_pct*100,3),
            "qualityMultiplier":round(quality_multiplier(signal_row_data,side,p),3),"maxLeverage":p.max_leverage,
            "desiredUnits":round(desired_units,2),"desiredLeverage":round(desired_units/equity_eur,3),
            "leverageCapHit":bool(desired_units>leverage_cap+1e-9),"recommended":rec,"noLeverageAlternative":nolev,
            "selection":label,"trend":trend,**levels,"levelsIndicativeUntilNextOpen":True,
            "mlProbability":None if ml_probability is None else round(float(ml_probability),4)}


# ----------------------------- metrics ------------------------------------
def _metrics(equity: pd.Series, trades: list[dict], initial: float, final_capital: float, exposure: int,
             rejected: dict | None = None) -> dict:
    rets=equity.pct_change().fillna(0.0); peak=equity.cummax(); dd=equity/peak-1
    pnl=np.array([t["profitEUR"] for t in trades],float) if trades else np.array([],float)
    wins=pnl[pnl>0]; losses=-pnl[pnl<0]; vol=rets.std(ddof=0); downside=rets[rets<0].std(ddof=0)
    pf=wins.sum()/losses.sum() if losses.sum()>0 else (999.0 if wins.sum()>0 else 0.0)
    rmult=np.array([t.get("rMultiple",np.nan) for t in trades],float) if trades else np.array([])
    mfe=np.array([t.get("mfeR",np.nan) for t in trades],float) if trades else np.array([])
    mae=np.array([t.get("maeR",np.nan) for t in trades],float) if trades else np.array([])
    caps=np.array([t.get("captureRatio",np.nan) for t in trades if t.get("profitEUR",0)>0],float) if trades else np.array([])
    out={"accountCurrency":"EUR","initialCapitalEUR":round(initial,2),"finalCapitalEUR":round(final_capital,2),
         "totalReturn":round((final_capital/initial-1)*100,4),"winRate":round((pnl>0).mean()*100,2) if len(pnl) else 0.0,
         "maxDrawdown":round(float(-dd.min()*100),4) if len(dd) else 0.0,"profitFactor":round(float(pf),4),
         "totalTrades":int(len(trades)),"sharpe":round(float(rets.mean()/vol*math.sqrt(252)),4) if vol>0 else 0.0,
         "sortino":round(float(rets.mean()/downside*math.sqrt(252)),4) if pd.notna(downside) and downside>0 else 0.0,
         "avgTradeEUR":round(float(pnl.mean()),2) if len(pnl) else 0.0,"exposurePct":round(exposure/max(1,len(equity))*100,2),
         "avgLeverage":round(float(np.mean([t.get("leverage",0) for t in trades])),3) if trades else 0.0,
         "maxLeverageUsed":round(float(max([t.get("leverage",0) for t in trades],default=0)),3),
         "leveragedTrades":int(sum(bool(t.get("usesLeverage")) for t in trades)),
         "expectancyR":round(float(np.nanmean(rmult)),3) if len(rmult) and np.isfinite(rmult).any() else 0.0,
         "avgMFER":round(float(np.nanmean(mfe)),3) if len(mfe) and np.isfinite(mfe).any() else 0.0,
         "avgMAER":round(float(np.nanmean(mae)),3) if len(mae) and np.isfinite(mae).any() else 0.0,
         "medianWinnerCaptureRatio":round(float(np.nanmedian(caps)),3) if len(caps) and np.isfinite(caps).any() else 0.0,
         "stoppedTrades":int(sum(t.get("reason")=="SL" for t in trades)),
         "takeProfitTrades":int(sum(t.get("reason")=="TP" for t in trades)),
         "timeExitTrades":int(sum(t.get("reason")=="TIME" for t in trades))}
    if rejected:
        out.update({"rejectedTrendSignals":int(rejected.get("trend",0)),"rejectedWideStops":int(rejected.get("wide_stop",0)),
                    "rejectedSize":int(rejected.get("size",0))})
    return out


# ----------------------------- backtest -----------------------------------
def _update_profit_protection(stop: float, entry: float, side: int, initial_risk_price: float, r: pd.Series, p: Params) -> tuple[float,str|None]:
    if not p.profit_protection_enabled or initial_risk_price<=0: return stop,None
    close_r=((float(r.close)-entry)*side)/initial_risk_price
    new_stop=stop; label=None
    cost_buffer=(p.spread_pips/2+p.slippage_pips_per_side)*PIP
    if close_r>=p.breakeven_trigger_r:
        be=entry+side*cost_buffer
        new_stop=max(new_stop,be) if side==1 else min(new_stop,be); label="BREAKEVEN"
    if close_r>=p.trailing_trigger_r and pd.notna(r.atr):
        atrtrail=float(r.close)-side*p.trailing_atr*float(r.atr)
        if side==1:
            struct=float(r.trail_swing_low)-p.structure_buffer_atr*float(r.atr) if pd.notna(r.trail_swing_low) else atrtrail
            candidate=max(atrtrail,struct)
            new_stop=max(new_stop,candidate)
        else:
            struct=float(r.trail_swing_high)+p.structure_buffer_atr*float(r.atr) if pd.notna(r.trail_swing_high) else atrtrail
            candidate=min(atrtrail,struct)
            new_stop=min(new_stop,candidate)
        label="TRAIL"
    return float(new_stop),label


def backtest(df: pd.DataFrame, p: Params, start=None, end=None) -> dict:
    x=features(df,p)
    warm=max(p.trend_ema_slow,p.sma_trend,p.bb_period,p.atr_period,p.adx_period,30)
    start_ts,end_ts=_utc_ts(start),_utc_ts(end)
    first_i=warm if start_ts is None else max(warm,int(x.index.searchsorted(start_ts,side="left")))
    last_i=len(x)-1 if end_ts is None else min(len(x)-1,int(x.index.searchsorted(end_ts,side="right"))-1)
    if last_i<=first_i: raise ValueError("Not enough data in requested backtest window after warm-up.")

    capital=p.initial_capital_eur; position=0; units=0.0; entry=stop=take=np.nan; entry_i=None
    entry_date=signal_date=None; pending=None; active_plan=None; initial_risk_price=np.nan
    fav_price=adv_price=np.nan; protection_events=[]; trades=[]; eq=[]; dates=[]; exposure=0
    rejected={"trend":0,"wide_stop":0,"size":0}; half_spread=p.spread_pips*PIP/2; slip=p.slippage_pips_per_side*PIP

    for i in range(first_i,last_i+1):
        r=x.iloc[i]
        if position==0 and pending is not None:
            position=int(pending["side"]); adverse=half_spread+slip
            entry=float(r.open+adverse if position==1 else r.open-adverse)
            plan=position_plan(capital,entry,position,float(pending["atr"]),pending["row"],p,pending.get("ml_probability"))
            if not plan.get("valid",False):
                if plan.get("rejectionReason")=="STOP_TOO_WIDE": rejected["wide_stop"]+=1
                else: rejected["size"]+=1
                position=0; pending=None
            else:
                units=float(plan["recommended"]["units"]); capital-=_commission(units,p)
                stop=float(plan["stopLoss"]); take=float(plan["takeProfit"]); initial_risk_price=abs(entry-stop)
                entry_i=i; entry_date=x.index[i]; signal_date=pending["signal_date"]; active_plan=plan; pending=None
                fav_price=entry; adv_price=entry; protection_events=[]

        reason=raw_exit=None
        if position:
            exposure+=1
            if position==1:
                fav_price=max(float(fav_price),float(r.high)); adv_price=min(float(adv_price),float(r.low))
                if r.low<=stop: reason,raw_exit="SL",stop
                elif r.high>=take: reason,raw_exit="TP",take
            else:
                fav_price=min(float(fav_price),float(r.low)); adv_price=max(float(adv_price),float(r.high))
                if r.high>=stop: reason,raw_exit="SL",stop
                elif r.low<=take: reason,raw_exit="TP",take
            if reason is None and i-entry_i>=p.max_holding_bars: reason,raw_exit="TIME",float(r.close)
            if reason:
                adverse=half_spread+slip; exit_px=float(raw_exit-adverse if position==1 else raw_exit+adverse)
                trade_pnl=pnl_eur(entry,exit_px,units,position)-_commission(units,p); capital+=trade_pnl
                mfe_price=max(0.0,(float(fav_price)-entry)*position); mae_price=max(0.0,-(float(adv_price)-entry)*position)
                risk_eur=max(float(active_plan["recommended"].get("riskAtStopEUR",0)),1e-9)
                r_mult=trade_pnl/risk_eur; realized_price=(exit_px-entry)*position
                capture=realized_price/mfe_price if trade_pnl>0 and mfe_price>0 else np.nan
                trades.append({"signalDate":str(signal_date),"entry":str(entry_date),"exit":str(x.index[i]),
                    "side":"LONG" if position==1 else "SHORT","entryPrice":round(entry,6),"exitPrice":round(exit_px,6),
                    "stopLoss":round(float(active_plan["stopLoss"]),6),"finalStop":round(stop,6),"takeProfit":round(take,6),
                    "units":round(units,0),"lots":round(units/100_000,4),"notionalEUR":round(units,2),
                    "leverage":active_plan["recommended"]["leverage"],"usesLeverage":active_plan["recommended"]["usesLeverage"],
                    "appliedRiskPct":active_plan["appliedRiskPct"],"initialRiskEUR":round(risk_eur,2),"profitEUR":round(float(trade_pnl),2),
                    "rMultiple":round(float(r_mult),3),"mfeR":round(mfe_price/initial_risk_price,3),"maeR":round(mae_price/initial_risk_price,3),
                    "captureRatio":None if pd.isna(capture) else round(float(capture),3),"reason":reason,
                    "trendScore":active_plan["trend"]["score"],"trendRegime":active_plan["trend"]["regime"],
                    "volatilityRegime":active_plan["trend"].get("volatilityRegime"),"stopSource":active_plan["stopSource"],
                    "stopDistanceATR":round(float(active_plan["stopDistanceATR"]),3),"qualityMultiplier":active_plan.get("qualityMultiplier"),
                    "protectionEvents":protection_events})
                position=0; units=0.0; entry_i=entry_date=signal_date=None; active_plan=None
            else:
                new_stop,event=_update_profit_protection(stop,entry,position,initial_risk_price,r,p)
                if event and abs(new_stop-stop)>1e-12: protection_events.append({"date":str(x.index[i]),"event":event,"newStop":round(new_stop,6)})
                stop=new_stop

        mtm=capital
        if position:
            mark=float(r.close-half_spread if position==1 else r.close+half_spread); mtm+=pnl_eur(entry,mark,units,position)
        eq.append(mtm); dates.append(x.index[i])

        if position==0 and pending is None and i<last_i:
            dec=signal_decision(r,p)
            if dec.get("candidate") and not dec.get("accepted"): rejected["trend"]+=1
            if dec.get("side"):
                pending={"side":int(dec["side"]),"signal_date":x.index[i],"atr":float(r.atr),"row":r.copy(),"decision":dec}

    if position:
        r=x.iloc[last_i]; adverse=half_spread+slip; exit_px=float(r.close-adverse if position==1 else r.close+adverse)
        trade_pnl=pnl_eur(entry,exit_px,units,position)-_commission(units,p); capital+=trade_pnl
        mfe_price=max(0.0,(float(fav_price)-entry)*position); mae_price=max(0.0,-(float(adv_price)-entry)*position)
        risk_eur=max(float(active_plan["recommended"].get("riskAtStopEUR",0)),1e-9); capture=((exit_px-entry)*position)/mfe_price if trade_pnl>0 and mfe_price>0 else np.nan
        trades.append({"signalDate":str(signal_date),"entry":str(entry_date),"exit":str(x.index[last_i]),"side":"LONG" if position==1 else "SHORT",
            "entryPrice":round(entry,6),"exitPrice":round(exit_px,6),"stopLoss":round(float(active_plan["stopLoss"]),6),"finalStop":round(stop,6),
            "takeProfit":round(take,6),"units":round(units,0),"lots":round(units/100_000,4),"notionalEUR":round(units,2),
            "leverage":active_plan["recommended"]["leverage"],"usesLeverage":active_plan["recommended"]["usesLeverage"],"appliedRiskPct":active_plan["appliedRiskPct"],
            "initialRiskEUR":round(risk_eur,2),"profitEUR":round(float(trade_pnl),2),"rMultiple":round(float(trade_pnl/risk_eur),3),
            "mfeR":round(mfe_price/initial_risk_price,3),"maeR":round(mae_price/initial_risk_price,3),"captureRatio":None if pd.isna(capture) else round(float(capture),3),
            "reason":"EOD","trendScore":active_plan["trend"]["score"],"trendRegime":active_plan["trend"]["regime"],
            "volatilityRegime":active_plan["trend"].get("volatilityRegime"),"stopSource":active_plan["stopSource"],
            "stopDistanceATR":round(float(active_plan["stopDistanceATR"]),3),"qualityMultiplier":active_plan.get("qualityMultiplier"),"protectionEvents":protection_events})
        eq[-1]=capital

    equity=pd.Series(eq,index=dates,name="equity_eur",dtype=float)
    return {"metrics":_metrics(equity,trades,p.initial_capital_eur,capital,exposure,rejected),"trades":trades,"equity":equity,"features":x,"rejections":rejected}


# ------------------------ fair legacy comparator ---------------------------
def _v31_params_from_current(p: Params, sizing_mode: str = "adaptive_risk"):
    from .baseline_v31 import Params as V31Params
    return V31Params(
        initial_capital_eur=p.initial_capital_eur, sizing_mode=sizing_mode, fixed_notional_eur=p.fixed_notional_eur,
        risk_per_trade=0.005, min_risk_per_trade=0.0025, max_risk_per_trade=0.0100, max_leverage=3.0,
        # Research benchmark is normalized to a small EUR 1k account; allow fine notional granularity
        # so the legacy risk budget does not collapse to zero trades merely because of a 1k-unit lot step.
        lot_step_units=1.0, min_units=1.0, absolute_max_units=p.absolute_max_units,
        volatility_target=0.08, vol_scale_min=0.65, vol_scale_max=1.25, spread_pips=p.spread_pips,
        slippage_pips_per_side=p.slippage_pips_per_side, commission_per_million_per_side_eur=p.commission_per_million_per_side_eur,
        rsi_period=14, rsi_long=55.0, rsi_short=45.0, sma_trend=50, bb_period=20, atr_period=14, adx_period=14,
        stop_atr=1.5, take_atr=2.2, max_holding_bars=5, require_macd=True, require_adx=False, min_adx=18.0,
        enable_range_module=False, range_adx_max=17.0, range_z=1.5, range_rsi_low=35.0, range_rsi_high=65.0,
        ml_probability_threshold=0.60, ml_confidence_reference=0.78)


def backtest_v31_benchmark(df: pd.DataFrame, p: Params, start=None, end=None) -> dict:
    from .baseline_v31 import backtest as _v31
    return _v31(df, _v31_params_from_current(p), start=start, end=end)


def backtest_legacy_fair(df: pd.DataFrame, p: Params, start=None, end=None) -> dict:
    from .baseline_v31 import backtest_legacy_fair as _legacy
    return _legacy(df, _v31_params_from_current(p, sizing_mode="fixed_notional"), start=start, end=end)


# ----------------------------- snapshot -----------------------------------
def current_snapshot(x: pd.DataFrame, p: Params, equity_eur: float | None = None, ml_probability: float | None = None,
                     override_signal: int | None = None) -> dict:
    r=x.iloc[-1]; dec=signal_decision(r,p)
    s=int(dec["side"]) if override_signal is None else int(override_signal)
    equity=p.initial_capital_eur if equity_eur is None else float(equity_eur)
    plan=None; sl=tp=None
    if s and pd.notna(r.atr):
        plan=position_plan(equity,float(r.close),s,float(r.atr),r,p,ml_probability)
        if plan.get("valid"):
            sl,tp=plan["stopLoss"],plan["takeProfit"]
        else:
            s=0
            if override_signal is None:
                dec = {**dec, "reason": plan.get("rejectionReason", "RISK_PLAN_REJECTED")}
    trend=dec.get("trend") if override_signal is None else trend_context(r,s,p)
    return {"date":str(x.index[-1]),"currentPrice":round(float(r.close),6),"rsi":round(float(r.rsi),2) if pd.notna(r.rsi) else None,
            "sma50":round(float(r.sma50),6) if pd.notna(r.sma50) else None,"ema50":round(float(r.ema50),6) if pd.notna(r.ema50) else None,
            "ema200":round(float(r.ema200),6) if pd.notna(r.ema200) else None,"ema50Slope":round(float(r.ema50_slope),6) if pd.notna(r.ema50_slope) else None,
            "macdHist":round(float(r.macd_hist),7) if pd.notna(r.macd_hist) else None,"atr":round(float(r.atr),7) if pd.notna(r.atr) else None,
            "adx":round(float(r.adx),2) if pd.notna(r.adx) else None,"plusDI":round(float(r.plus_di),2) if pd.notna(r.plus_di) else None,
            "minusDI":round(float(r.minus_di),2) if pd.notna(r.minus_di) else None,"rv20":round(float(r.rv20),5) if pd.notna(r.rv20) else None,
            "volatilityRegime":volatility_regime(r),"trendScore":trend.get("score",0),"trendScoreMax":trend.get("maxScore",5),
            "trendRegime":trend.get("regime","—"),"marketBias":trend.get("bias", "BULLISH" if trend.get("components",{}).get("EMA50VsEMA200") else "BEARISH"),
            "trendComponents":trend.get("components",{}),
            "candidateSignal":"BUY (LONG)" if dec.get("candidateSide")==1 else ("SELL (SHORT)" if dec.get("candidateSide")==-1 else "NONE"),
            "signal":"BUY (LONG)" if s==1 else ("SELL (SHORT)" if s==-1 else "NEUTRAL (WAIT)"),
            "signalReason":dec.get("reason"),"stopLoss":sl,"takeProfit":tp,"levelsAreIndicative":True,
            "execution":"Signal at close; final size/SL/TP recalculated at following daily open with signal-bar risk inputs frozen.",
            "positionPlan":plan}


def params_dict(p: Params) -> dict:
    return asdict(p)
