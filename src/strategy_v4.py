"""EUR/USD swing strategy V4 — soft trend, adaptive stop, EUR margin and risk-based leverage.

Key changes versus V3.2:
- Trend is a quality/risk modifier. It only blocks a signal under a strong opposing regime.
- Structural stops never reject a trade merely because the swing is far away; the engine falls back to ATR.
- No forced 3/5-day exit. A trade remains open until SL/TP or the available dataset ends.
- Account equity and trade allocation are EUR. The user-facing position is margin/stake EUR, not FX units.
- Trade leverage is searched from x1 to x30 to use the risk budget without exceeding it.
- The account exposure multiple is reported separately from trade leverage.
- Open positions at the end are marked to market instead of being silently counted as closed trades.

Research software only. Broker margin rules, financing, gaps and execution can differ.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Optional

import numpy as np
import pandas as pd

from .strategy_engine import (
    PIP,
    Params as BaseParams,
    _commission,
    _utc_ts,
    features,
    fetch_yahoo,
    load_csv,
    pnl_eur,
    technical_candidate_row,
    trend_context,
    volatility_regime,
)


@dataclass(frozen=True)
class V4Params(BaseParams):
    # Account
    initial_capital_eur: float = 1_000.0

    # Risk budget. This controls account loss at the initial SL, independently of leverage.
    sizing_mode: str = "eur_margin_leverage"
    risk_per_trade: float = 0.0075
    min_risk_per_trade: float = 0.0035
    max_risk_per_trade: float = 0.0100
    volatility_target: float = 0.08
    vol_scale_min: float = 0.70
    vol_scale_max: float = 1.15
    quality_scale_min: float = 0.70
    quality_scale_max: float = 1.20

    # EUR allocated as margin/stake before leverage.
    base_position_eur: float = 100.0
    min_position_eur: float = 25.0
    max_position_pct_equity: float = 0.25

    # x1 means unlevered/spot-equivalent on the allocated trade amount.
    min_trade_leverage: float = 1.0
    max_trade_leverage: float = 30.0
    leverage_step: float = 1.0

    # Simplified financing stress: annual cost applied to borrowed EUR exposure
    # (notional - allocated position). x1 therefore has no financing charge.
    financing_annual_pct_borrowed: float = 0.04

    # Trend is soft by default; only a very strong opposite trend is blocked.
    require_trend_filter: bool = False
    hard_trend_reject: bool = True
    hard_trend_reject_adx: float = 28.0
    hard_trend_max_alignment_score: int = 1

    # Stop/target
    stop_mode: str = "adaptive_structure"
    stop_atr_low_vol: float = 1.35
    stop_atr_normal: float = 1.50
    stop_atr_high_vol: float = 1.70
    structure_lookback: int = 8
    structure_buffer_atr: float = 0.10
    max_stop_atr: float = 2.75  # structure farther than this is ignored, not rejected
    target_rr: float = 1.80
    max_holding_bars: int = 0  # 0 = no time exit

    # Profit protection remains opt-in because prior tests cut winners too early.
    profit_protection_enabled: bool = False


def _quality_multiplier(r: pd.Series, side: int, p: V4Params) -> float:
    ctx = trend_context(r, side, p)
    score = float(ctx.get("score", 0))
    # Map 0..5 approximately into configured quality range.
    q = p.quality_scale_min + (score / 5.0) * (p.quality_scale_max - p.quality_scale_min)
    if ctx.get("regime") == "STRONG_TREND":
        q *= 1.04
    if ctx.get("volatilityRegime") == "HIGH":
        q *= 0.92
    return float(np.clip(q, p.quality_scale_min, p.quality_scale_max))


def adaptive_risk_pct_v4(
    r: pd.Series,
    side: int,
    p: V4Params,
    ml_probability: float | None = None,
    ml_healthy: bool = False,
) -> float:
    risk = float(p.risk_per_trade)
    rv = r.get("rv20", np.nan)
    if pd.notna(rv) and float(rv) > 0:
        risk *= float(np.clip(p.volatility_target / float(rv), p.vol_scale_min, p.vol_scale_max))
    risk *= _quality_multiplier(r, side, p)

    # ML is advisory/risk-scaling only. An unhealthy model is ignored.
    if ml_healthy and ml_probability is not None and np.isfinite(ml_probability):
        # 0.50 => neutral, 0.65 => +20%, 0.35 => -20%.
        ml_scale = float(np.clip(1.0 + (float(ml_probability) - 0.50) * 1.3333, 0.80, 1.20))
        risk *= ml_scale

    return float(np.clip(risk, p.min_risk_per_trade, p.max_risk_per_trade))


def signal_decision_v4(r: pd.Series, p: V4Params) -> dict:
    side = technical_candidate_row(r, p)
    if side == 0:
        return {
            "side": 0,
            "candidate": False,
            "accepted": False,
            "reason": "NO_TECHNICAL_CANDIDATE",
            "trend": {"score": 0, "maxScore": 5, "regime": "NO_SIGNAL", "components": {}},
        }

    ctx = trend_context(r, side, p)
    hard_opposition = bool(
        p.hard_trend_reject
        and pd.notna(r.get("adx", np.nan))
        and float(r.adx) >= p.hard_trend_reject_adx
        and int(ctx.get("score", 0)) <= p.hard_trend_max_alignment_score
    )
    accepted = not hard_opposition
    return {
        "side": side if accepted else 0,
        "candidateSide": side,
        "candidate": True,
        "accepted": accepted,
        "reason": "ACCEPT" if accepted else "REJECT_STRONG_OPPOSING_TREND",
        "trend": ctx,
        "quality": _quality_multiplier(r, side, p),
    }


def dynamic_stop_multiplier_v4(r: pd.Series, p: V4Params) -> float:
    regime = volatility_regime(r)
    if regime == "LOW":
        return p.stop_atr_low_vol
    if regime == "HIGH":
        return p.stop_atr_high_vol
    return p.stop_atr_normal


def stop_target_v4(entry: float, side: int, atr_signal: float, r: pd.Series, p: V4Params) -> dict:
    mult = dynamic_stop_multiplier_v4(r, p)
    atr_dist = max(mult * atr_signal, PIP)

    if side == 1:
        atr_stop = entry - atr_dist
        struct = (
            float(r.swing_low) - p.structure_buffer_atr * atr_signal
            if pd.notna(r.get("swing_low", np.nan))
            else atr_stop
        )
        struct_dist = entry - struct
        use_structure = 0 < struct_dist <= p.max_stop_atr * atr_signal
        stop = min(atr_stop, struct) if use_structure else atr_stop
        stop_dist = entry - stop
    else:
        atr_stop = entry + atr_dist
        struct = (
            float(r.swing_high) + p.structure_buffer_atr * atr_signal
            if pd.notna(r.get("swing_high", np.nan))
            else atr_stop
        )
        struct_dist = struct - entry
        use_structure = 0 < struct_dist <= p.max_stop_atr * atr_signal
        stop = max(atr_stop, struct) if use_structure else atr_stop
        stop_dist = stop - entry

    if stop_dist <= 0:
        return {"valid": False, "rejectionReason": "INVALID_STOP"}

    source = "STRUCTURE" if use_structure and abs(stop - atr_stop) > PIP / 10 else (
        "ATR_FALLBACK" if not use_structure and abs(struct - atr_stop) > PIP / 10 else "ATR"
    )
    take = entry + side * p.target_rr * stop_dist
    return {
        "valid": True,
        "rejectionReason": None,
        "stopLoss": float(max(stop, PIP)),
        "takeProfit": float(max(take, PIP)),
        "stopDistance": float(stop_dist),
        "stopDistanceATR": float(stop_dist / max(atr_signal, PIP)),
        "stopATRMultiplier": float(mult),
        "stopSource": source,
        "structureUsed": bool(use_structure),
        "targetRR": float(p.target_rr),
    }


def _commission_notional(notional_eur: float, p: V4Params) -> float:
    return (notional_eur / 1_000_000.0) * p.commission_per_million_per_side_eur


def _round_down(value: float, step: float) -> float:
    if step <= 0:
        return max(0.0, value)
    return math.floor(max(0.0, value) / step) * step


def position_plan_v4(
    equity_eur: float,
    entry: float,
    side: int,
    atr_signal: float,
    signal_row: pd.Series,
    p: V4Params,
    ml_probability: float | None = None,
    ml_healthy: bool = False,
) -> dict:
    if equity_eur <= 0 or entry <= 0 or side not in (-1, 1):
        raise ValueError("Invalid equity/entry/side")

    levels = stop_target_v4(entry, side, atr_signal, signal_row, p)
    trend = trend_context(signal_row, side, p)
    if not levels["valid"]:
        return {"valid": False, "rejectionReason": levels["rejectionReason"], "trend": trend, **levels}

    adverse = p.spread_pips * PIP / 2 + p.slippage_pips_per_side * PIP
    stop_fill = max(PIP, levels["stopLoss"] - adverse if side == 1 else levels["stopLoss"] + adverse)
    take_fill = max(PIP, levels["takeProfit"] - adverse if side == 1 else levels["takeProfit"] + adverse)

    target_risk_pct = adaptive_risk_pct_v4(signal_row, side, p, ml_probability, ml_healthy)
    risk_budget = equity_eur * target_risk_pct
    loss_per_eur_notional = abs(pnl_eur(entry, stop_fill, 1.0, side)) + 2 * (
        p.commission_per_million_per_side_eur / 1_000_000.0
    )

    quality = _quality_multiplier(signal_row, side, p)
    scaled_base = p.base_position_eur * (equity_eur / max(p.initial_capital_eur, 1e-9))
    margin_eur = scaled_base * float(np.clip(quality, 0.75, 1.20))
    max_margin = max(p.min_position_eur, equity_eur * p.max_position_pct_equity)
    margin_eur = float(np.clip(margin_eur, p.min_position_eur, max_margin))

    # If x1 already risks too much, reduce the EUR position rather than using leverage < x1.
    risk_x1 = loss_per_eur_notional * margin_eur
    if risk_x1 > risk_budget and loss_per_eur_notional > 0:
        margin_eur = risk_budget / loss_per_eur_notional
        margin_eur = min(margin_eur, max_margin)
        if margin_eur < p.min_position_eur:
            return {
                "valid": False,
                "rejectionReason": "POSITION_EUR_TOO_SMALL_FOR_RISK_CAP",
                "equityEUR": round(equity_eur, 2),
                "targetRiskEUR": round(risk_budget, 2),
                "trend": trend,
                **levels,
            }

    def risk_for(lev: float, margin: float) -> float:
        notional = margin * lev
        return loss_per_eur_notional * notional + 2 * _commission_notional(notional, p)

    # Search x1..x30 for the highest risk-budget utilization without exceeding it.
    lev_values = np.arange(p.min_trade_leverage, p.max_trade_leverage + 1e-9, p.leverage_step)
    feasible = [float(l) for l in lev_values if risk_for(float(l), margin_eur) <= risk_budget + 1e-9]
    leverage = max(feasible) if feasible else p.min_trade_leverage

    # If x30 still underuses the budget, increase the EUR margin up to the configured cap.
    if leverage >= p.max_trade_leverage - 1e-9 and loss_per_eur_notional > 0:
        required_margin = risk_budget / max(loss_per_eur_notional * leverage, 1e-12)
        margin_eur = float(np.clip(max(margin_eur, required_margin), p.min_position_eur, max_margin))
        if risk_for(leverage, margin_eur) > risk_budget + 1e-9:
            margin_eur = risk_budget / max(loss_per_eur_notional * leverage, 1e-12)

    notional_eur = margin_eur * leverage
    actual_risk = risk_for(leverage, margin_eur)
    gain_eur = pnl_eur(entry, take_fill, notional_eur, side) - 2 * _commission_notional(notional_eur, p)
    account_exposure_x = notional_eur / equity_eur

    return {
        "valid": True,
        "rejectionReason": None,
        "equityEUR": round(equity_eur, 2),
        "positionEUR": round(margin_eur, 2),
        "marginEUR": round(margin_eur, 2),
        "tradeLeverage": round(leverage, 2),
        "spotEquivalent": bool(abs(leverage - 1.0) < 1e-9),
        "notionalEUR": round(notional_eur, 2),
        "accountExposureX": round(account_exposure_x, 3),
        "targetRiskPct": round(target_risk_pct * 100, 3),
        "targetRiskEUR": round(risk_budget, 2),
        "riskAtStopEUR": round(actual_risk, 2),
        "riskAtStopPct": round(actual_risk / equity_eur * 100, 3),
        "targetGainEUR": round(gain_eur, 2),
        "rewardRisk": round(gain_eur / actual_risk, 3) if actual_risk > 0 else 0.0,
        "riskBudgetUtilizationPct": round(actual_risk / risk_budget * 100, 2) if risk_budget > 0 else 0.0,
        "leverageSelection": "RISK_BUDGET_OPTIMAL",
        "qualityMultiplier": round(quality, 3),
        "trend": trend,
        "mlProbability": None if ml_probability is None else round(float(ml_probability), 4),
        "mlHealthy": bool(ml_healthy),
        **levels,
    }


def _metrics_v4(
    equity: pd.Series,
    trades: list[dict],
    initial: float,
    realized_capital: float,
    final_equity: float,
    exposure_bars: int,
    open_position: Optional[dict],
    rejected: dict,
) -> dict:
    rets = equity.pct_change().fillna(0.0)
    peak = equity.cummax()
    dd = equity / peak - 1
    pnl = np.array([t["profitEUR"] for t in trades], dtype=float) if trades else np.array([], dtype=float)
    wins = pnl[pnl > 0]
    losses = -pnl[pnl < 0]
    vol = rets.std(ddof=0)
    downside = rets[rets < 0].std(ddof=0)
    pf = wins.sum() / losses.sum() if losses.sum() > 0 else (999.0 if wins.sum() > 0 else 0.0)
    rmult = np.array([t.get("rMultiple", np.nan) for t in trades], dtype=float) if trades else np.array([])
    return {
        "accountCurrency": "EUR",
        "initialCapitalEUR": round(initial, 2),
        "finalCapitalEUR": round(realized_capital, 2),
        "finalEquityEUR": round(final_equity, 2),
        "realizedReturnPct": round((realized_capital / initial - 1) * 100, 4),
        "totalReturn": round((final_equity / initial - 1) * 100, 4),
        "winRate": round((pnl > 0).mean() * 100, 2) if len(pnl) else 0.0,
        "maxDrawdown": round(float(-dd.min() * 100), 4) if len(dd) else 0.0,
        "profitFactor": round(float(pf), 4),
        "totalTrades": int(len(trades)),
        "openTrades": int(open_position is not None),
        "sharpe": round(float(rets.mean() / vol * math.sqrt(252)), 4) if vol > 0 else 0.0,
        "sortino": round(float(rets.mean() / downside * math.sqrt(252)), 4)
        if pd.notna(downside) and downside > 0
        else 0.0,
        "avgTradeEUR": round(float(pnl.mean()), 2) if len(pnl) else 0.0,
        "expectancyR": round(float(np.nanmean(rmult)), 3) if len(rmult) and np.isfinite(rmult).any() else 0.0,
        "exposurePct": round(exposure_bars / max(1, len(equity)) * 100, 2),
        "avgTradeLeverage": round(float(np.mean([t.get("tradeLeverage", 0) for t in trades])), 2) if trades else 0.0,
        "maxTradeLeverageUsed": round(float(max([t.get("tradeLeverage", 0) for t in trades], default=0)), 2),
        "avgAccountExposureX": round(float(np.mean([t.get("accountExposureX", 0) for t in trades])), 3) if trades else 0.0,
        "stoppedTrades": int(sum(t.get("reason") == "SL" for t in trades)),
        "takeProfitTrades": int(sum(t.get("reason") == "TP" for t in trades)),
        "timeExitTrades": 0,
        "rejectedStrongOpposingTrend": int(rejected.get("trend", 0)),
        "rejectedSize": int(rejected.get("size", 0)),
    }


def _gap_aware_exit(r: pd.Series, side: int, stop: float, take: float) -> tuple[str | None, float | None]:
    if side == 1:
        if float(r.open) <= stop:
            return "SL_GAP", float(r.open)
        if float(r.low) <= stop:
            return "SL", stop
        if float(r.high) >= take:
            return "TP", take
    else:
        if float(r.open) >= stop:
            return "SL_GAP", float(r.open)
        if float(r.high) >= stop:
            return "SL", stop
        if float(r.low) <= take:
            return "TP", take
    return None, None


def backtest_v4(
    df: pd.DataFrame,
    p: V4Params,
    start=None,
    end=None,
    ml_probability_by_date: dict | None = None,
    ml_healthy_by_date: dict | None = None,
) -> dict:
    x = features(df, p)
    warm = max(p.trend_ema_slow, p.sma_trend, p.bb_period, p.atr_period, p.adx_period, 30)
    start_ts, end_ts = _utc_ts(start), _utc_ts(end)
    first_i = warm if start_ts is None else max(warm, int(x.index.searchsorted(start_ts, side="left")))
    last_i = len(x) - 1 if end_ts is None else min(len(x) - 1, int(x.index.searchsorted(end_ts, side="right")) - 1)
    if last_i <= first_i:
        raise ValueError("Not enough data in requested backtest window after warm-up")

    capital = float(p.initial_capital_eur)
    position = 0
    notional_eur = 0.0
    entry = stop = take = np.nan
    entry_date = signal_date = None
    active_plan = None
    initial_risk_price = np.nan
    fav_price = adv_price = np.nan
    pending = None
    trades: list[dict] = []
    equity_vals: list[float] = []
    equity_dates: list[pd.Timestamp] = []
    exposure = 0
    rejected = {"trend": 0, "size": 0}
    half_spread = p.spread_pips * PIP / 2
    slip = p.slippage_pips_per_side * PIP
    financing_accum = 0.0
    entry_commission = 0.0

    for i in range(first_i, last_i + 1):
        r = x.iloc[i]

        # Financing accrues only for a position already open from the prior daily bar.
        if position and i > first_i:
            calendar_days = max(0, (x.index[i] - x.index[i-1]).days)
            borrowed = max(0.0, notional_eur - float(active_plan.get("positionEUR", 0.0)))
            financing_accum += borrowed * p.financing_annual_pct_borrowed * calendar_days / 365.0

        if position == 0 and pending is not None:
            position = int(pending["side"])
            adverse = half_spread + slip
            entry = float(r.open + adverse if position == 1 else r.open - adverse)
            signal_dt = pending["signal_date"]
            ml_prob = None if ml_probability_by_date is None else ml_probability_by_date.get(signal_dt)
            ml_ok = False if ml_healthy_by_date is None else bool(ml_healthy_by_date.get(signal_dt, False))
            plan = position_plan_v4(
                capital,
                entry,
                position,
                float(pending["atr"]),
                pending["row"],
                p,
                ml_probability=ml_prob,
                ml_healthy=ml_ok,
            )
            if not plan.get("valid", False):
                rejected["size"] += 1
                position = 0
                pending = None
            else:
                active_plan = plan
                notional_eur = float(plan["notionalEUR"])
                entry_commission = _commission_notional(notional_eur, p)
                capital -= entry_commission
                financing_accum = 0.0
                stop = float(plan["stopLoss"])
                take = float(plan["takeProfit"])
                initial_risk_price = abs(entry - stop)
                entry_date = x.index[i]
                signal_date = signal_dt
                fav_price = entry
                adv_price = entry
                pending = None

        if position:
            exposure += 1
            if position == 1:
                fav_price = max(float(fav_price), float(r.high))
                adv_price = min(float(adv_price), float(r.low))
            else:
                fav_price = min(float(fav_price), float(r.low))
                adv_price = max(float(adv_price), float(r.high))

            reason, raw_exit = _gap_aware_exit(r, position, stop, take)
            if reason is not None:
                adverse = half_spread + slip
                # For a stop gap, the open already represents the gap; only execution friction is added.
                if reason == "SL_GAP":
                    exit_px = float(raw_exit - adverse if position == 1 else raw_exit + adverse)
                else:
                    exit_px = float(raw_exit - adverse if position == 1 else raw_exit + adverse)
                price_pnl = pnl_eur(entry, exit_px, notional_eur, position)
                exit_commission = _commission_notional(notional_eur, p)
                capital_delta = price_pnl - exit_commission - financing_accum
                capital += capital_delta
                trade_pnl = price_pnl - entry_commission - exit_commission - financing_accum

                mfe_price = max(0.0, (float(fav_price) - entry) * position)
                mae_price = max(0.0, -(float(adv_price) - entry) * position)
                initial_risk_eur = max(float(active_plan["riskAtStopEUR"]), 1e-9)
                r_mult = trade_pnl / initial_risk_eur
                realized_price = (exit_px - entry) * position
                capture = realized_price / mfe_price if trade_pnl > 0 and mfe_price > 0 else np.nan
                trades.append({
                    "signalDate": str(signal_date),
                    "entry": str(entry_date),
                    "exit": str(x.index[i]),
                    "status": "CLOSED",
                    "side": "LONG" if position == 1 else "SHORT",
                    "entryPrice": round(entry, 6),
                    "exitPrice": round(exit_px, 6),
                    "stopLoss": round(float(active_plan["stopLoss"]), 6),
                    "takeProfit": round(take, 6),
                    "positionEUR": active_plan["positionEUR"],
                    "tradeLeverage": active_plan["tradeLeverage"],
                    "notionalEUR": active_plan["notionalEUR"],
                    "accountExposureX": active_plan["accountExposureX"],
                    "targetRiskEUR": active_plan["targetRiskEUR"],
                    "initialRiskEUR": round(initial_risk_eur, 2),
                    "profitEUR": round(float(trade_pnl), 2),
                    "pricePnLEUR": round(float(price_pnl), 2),
                    "financingCostEUR": round(float(financing_accum), 2),
                    "entryCommissionEUR": round(float(entry_commission), 2),
                    "exitCommissionEUR": round(float(exit_commission), 2),
                    "rMultiple": round(float(r_mult), 3),
                    "mfeR": round(float(mfe_price / initial_risk_price), 3),
                    "maeR": round(float(mae_price / initial_risk_price), 3),
                    "captureRatio": None if pd.isna(capture) else round(float(capture), 3),
                    "reason": reason,
                    "trendScore": active_plan["trend"].get("score", 0),
                    "trendRegime": active_plan["trend"].get("regime"),
                    "volatilityRegime": active_plan["trend"].get("volatilityRegime"),
                    "stopSource": active_plan["stopSource"],
                    "stopDistanceATR": round(float(active_plan["stopDistanceATR"]), 3),
                    "riskBudgetUtilizationPct": active_plan["riskBudgetUtilizationPct"],
                })
                position = 0
                notional_eur = 0.0
                entry_date = signal_date = None
                active_plan = None
                financing_accum = 0.0
                entry_commission = 0.0

        mtm = capital
        if position:
            mark = float(r.close - half_spread if position == 1 else r.close + half_spread)
            mtm += pnl_eur(entry, mark, notional_eur, position) - financing_accum
        equity_vals.append(float(mtm))
        equity_dates.append(x.index[i])

        if position == 0 and pending is None and i < last_i:
            dec = signal_decision_v4(r, p)
            if dec.get("candidate") and not dec.get("accepted"):
                rejected["trend"] += 1
            if dec.get("side"):
                pending = {
                    "side": int(dec["side"]),
                    "signal_date": x.index[i],
                    "atr": float(r.atr),
                    "row": r.copy(),
                    "decision": dec,
                }

    open_position = None
    final_equity = capital
    if position:
        r = x.iloc[last_i]
        mark = float(r.close - half_spread if position == 1 else r.close + half_spread)
        unrealized_price = pnl_eur(entry, mark, notional_eur, position)
        unrealized = unrealized_price - financing_accum - entry_commission
        final_equity = capital + unrealized_price - financing_accum
        open_position = {
            "signalDate": str(signal_date),
            "entry": str(entry_date),
            "status": "OPEN",
            "side": "LONG" if position == 1 else "SHORT",
            "entryPrice": round(entry, 6),
            "currentPrice": round(mark, 6),
            "stopLoss": round(stop, 6),
            "takeProfit": round(take, 6),
            "positionEUR": active_plan["positionEUR"],
            "tradeLeverage": active_plan["tradeLeverage"],
            "notionalEUR": active_plan["notionalEUR"],
            "accountExposureX": active_plan["accountExposureX"],
            "initialRiskEUR": active_plan["riskAtStopEUR"],
            "unrealizedPnLEUR": round(unrealized, 2),
            "financingCostEUR": round(financing_accum, 2),
            "entryCommissionEUR": round(entry_commission, 2),
            "trendScore": active_plan["trend"].get("score", 0),
            "trendRegime": active_plan["trend"].get("regime"),
            "stopSource": active_plan["stopSource"],
        }

    equity = pd.Series(equity_vals, index=equity_dates, name="equity_eur", dtype=float)
    metrics = _metrics_v4(
        equity,
        trades,
        p.initial_capital_eur,
        capital,
        final_equity,
        exposure,
        open_position,
        rejected,
    )
    return {
        "metrics": metrics,
        "trades": trades,
        "openPosition": open_position,
        "equity": equity,
        "features": x,
        "rejections": rejected,
    }


def current_snapshot_v4(
    x: pd.DataFrame,
    p: V4Params,
    equity_eur: float | None = None,
    ml_advisory: dict | None = None,
) -> dict:
    r = x.iloc[-1]
    dec = signal_decision_v4(r, p)
    side = int(dec.get("side", 0))
    equity = p.initial_capital_eur if equity_eur is None else float(equity_eur)

    ml_prob = None
    ml_healthy = False
    if ml_advisory:
        ml_prob = ml_advisory.get("probability")
        ml_healthy = bool(ml_advisory.get("healthy", False))

    plan = None
    if side and pd.notna(r.atr):
        plan = position_plan_v4(
            equity,
            float(r.close),
            side,
            float(r.atr),
            r,
            p,
            ml_probability=ml_prob,
            ml_healthy=ml_healthy,
        )
        if not plan.get("valid"):
            side = 0

    trend = dec.get("trend", {})
    return {
        "date": str(x.index[-1]),
        "currentPrice": round(float(r.close), 6),
        "rsi": round(float(r.rsi), 2) if pd.notna(r.rsi) else None,
        "sma50": round(float(r.sma50), 6) if pd.notna(r.sma50) else None,
        "ema50": round(float(r.ema50), 6) if pd.notna(r.ema50) else None,
        "ema200": round(float(r.ema200), 6) if pd.notna(r.ema200) else None,
        "macdHist": round(float(r.macd_hist), 7) if pd.notna(r.macd_hist) else None,
        "adx": round(float(r.adx), 2) if pd.notna(r.adx) else None,
        "plusDI": round(float(r.plus_di), 2) if pd.notna(r.plus_di) else None,
        "minusDI": round(float(r.minus_di), 2) if pd.notna(r.minus_di) else None,
        "volatilityRegime": volatility_regime(r),
        "trendScore": trend.get("score", 0),
        "trendScoreMax": trend.get("maxScore", 5),
        "trendRegime": trend.get("regime", "—"),
        "candidateSignal": "BUY (LONG)" if dec.get("candidateSide") == 1 else (
            "SELL (SHORT)" if dec.get("candidateSide") == -1 else "NONE"
        ),
        "signal": "BUY (LONG)" if side == 1 else ("SELL (SHORT)" if side == -1 else "NEUTRAL (WAIT)"),
        "signalReason": dec.get("reason"),
        "positionPlan": plan,
        "mlAdvisory": ml_advisory,
        "execution": "Signal at close; execution next open. No forced time exit; position remains until SL/TP.",
    }


def params_dict_v4(p: V4Params) -> dict:
    return asdict(p)
