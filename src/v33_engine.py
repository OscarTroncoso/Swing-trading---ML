"""V3.3 EUR/USD engine: soft trend context, EUR stake sizing and TP/SL-only exits.

The strategy keeps the proven V3.1 directional core (RSI + SMA50 + MACD), removes
V3.2's hard trend/structural-stop vetoes, expresses position size as EUR stake,
and selects integer leverage x1..x30 subject to a per-trade EUR risk budget.

Research software only. Historical simulation is not a guarantee of future results.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .strategy_engine import (
    PIP,
    Params,
    _commission,
    _metrics,
    _utc_ts,
    features,
    pnl_eur,
    technical_candidate_row,
    trend_context,
    volatility_regime,
)


@dataclass(frozen=True)
class V33Policy:
    initial_capital_eur: float = 1_000.0
    base_stake_eur: float = 100.0
    min_stake_eur: float = 25.0
    max_stake_pct_equity: float = 0.25
    base_risk_pct_equity: float = 0.01
    min_risk_pct_equity: float = 0.005
    max_risk_pct_equity: float = 0.015
    min_leverage: int = 1
    max_leverage: int = 30
    stop_atr: float = 1.50
    target_rr: float = 2.20
    structure_lookback: int = 8
    structure_buffer_atr: float = 0.10
    max_structure_extension_atr: float = 0.25
    soft_trend_sizing: bool = True
    hard_countertrend_veto: bool = False
    countertrend_max_score: int = 1
    countertrend_min_adx: float = 28.0
    max_holding_bars: int = 0  # 0 = no time exit.
    breakeven_trigger_r: float = 0.0  # 0 = disabled; effective next bar.
    trailing_trigger_r: float = 0.0   # 0 = disabled; effective next bar.
    trailing_atr: float = 1.5
    exit_on_opposite_signal: bool = False  # execute next open, never same close.
    invalidation_mode: str = "none"  # none | macd | sma50 | rsi50 | core2
    partial_take_r: float = 0.0       # 0 = disabled
    partial_fraction: float = 0.0     # fraction of exposure closed at partial_take_r
    partial_move_stop_to_be: bool = True
    ml_soft_min_multiplier: float = 0.75
    ml_soft_max_multiplier: float = 1.20


def load_v33_policy(path: str | Path = "config.json") -> V33Policy:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    a = raw.get("account", {})
    pos = raw.get("position", {})
    st = raw.get("stop", {})
    tr = raw.get("trend", {})
    ml = raw.get("ml", {})
    return V33Policy(
        initial_capital_eur=float(a.get("initialCapitalEUR", 1_000.0)),
        base_stake_eur=float(pos.get("baseStakeEUR", 100.0)),
        min_stake_eur=float(pos.get("minStakeEUR", 25.0)),
        max_stake_pct_equity=float(pos.get("maxStakePctEquity", 0.25)),
        base_risk_pct_equity=float(pos.get("baseRiskPctEquity", 0.01)),
        min_risk_pct_equity=float(pos.get("minRiskPctEquity", 0.005)),
        max_risk_pct_equity=float(pos.get("maxRiskPctEquity", 0.015)),
        min_leverage=int(pos.get("minLeverage", 1)),
        max_leverage=int(pos.get("maxLeverage", 30)),
        stop_atr=float(st.get("atrNormal", st.get("stopATR", 1.50))),
        target_rr=float(st.get("targetRR", 2.20)),
        structure_lookback=int(st.get("structureLookback", 8)),
        structure_buffer_atr=float(st.get("structureBufferATR", 0.10)),
        max_structure_extension_atr=float(st.get("maxStructureExtensionATR", 0.25)),
        soft_trend_sizing=bool(tr.get("softSizing", True)),
        hard_countertrend_veto=bool(tr.get("hardCountertrendVeto", False)),
        countertrend_max_score=int(tr.get("countertrendMaxScore", 1)),
        countertrend_min_adx=float(tr.get("countertrendMinADX", 28.0)),
        max_holding_bars=int(st.get("maxHoldingBars", 0)),
        breakeven_trigger_r=float(st.get("breakevenTriggerR", 0.0)),
        trailing_trigger_r=float(st.get("trailingTriggerR", 0.0)),
        trailing_atr=float(st.get("trailingATR", 1.5)),
        exit_on_opposite_signal=bool(st.get("exitOnOppositeSignal", False)),
        invalidation_mode=str(st.get("invalidationMode", "none")),
        partial_take_r=float(st.get("partialTakeR", 0.0)),
        partial_fraction=float(st.get("partialFraction", 0.0)),
        partial_move_stop_to_be=bool(st.get("partialMoveStopToBE", True)),
        ml_soft_min_multiplier=float(ml.get("softMinMultiplier", 0.75)),
        ml_soft_max_multiplier=float(ml.get("softMaxMultiplier", 1.20)),
    )


def soft_trend_multiplier(row: pd.Series, side: int, p: Params, policy: V33Policy) -> tuple[float, dict]:
    ctx = trend_context(row, side, p)
    score = int(ctx.get("score", 0))
    # Trend is context, not a hard requirement. The score scales capital at risk.
    mapping = {0: 0.70, 1: 0.80, 2: 0.90, 3: 1.00, 4: 1.10, 5: 1.20}
    mult = mapping.get(score, 1.0)
    if ctx.get("regime") == "RANGE":
        mult *= 0.90
    if ctx.get("volatilityRegime") == "HIGH":
        mult *= 0.90
    if not policy.soft_trend_sizing:
        mult = 1.0
    return float(np.clip(mult, 0.60, 1.25)), ctx


def should_veto_countertrend(row: pd.Series, side: int, p: Params, policy: V33Policy) -> bool:
    if not policy.hard_countertrend_veto:
        return False
    ctx = trend_context(row, side, p)
    adx = float(row.get("adx", np.nan))
    return bool(
        ctx.get("score", 0) <= policy.countertrend_max_score
        and np.isfinite(adx)
        and adx >= policy.countertrend_min_adx
    )


def _stop_target(entry: float, side: int, atr: float, row: pd.Series, policy: V33Policy) -> dict:
    """ATR stop with a *capped* structural extension.

    V3.2 widened stops to full recent structure and rejected many otherwise valid
    signals. V3.3 starts from ATR and only extends to structure when the extra
    distance is modest. This preserves signal coverage.
    """
    atr = max(float(atr), PIP)
    base_dist = max(policy.stop_atr * atr, PIP)
    if side == 1:
        stop = entry - base_dist
        swing = row.get("swing_low", np.nan)
        if pd.notna(swing):
            structural = float(swing) - policy.structure_buffer_atr * atr
            extra = (stop - structural) / atr
            if 0 < extra <= policy.max_structure_extension_atr:
                stop = structural
        dist = entry - stop
    else:
        stop = entry + base_dist
        swing = row.get("swing_high", np.nan)
        if pd.notna(swing):
            structural = float(swing) + policy.structure_buffer_atr * atr
            extra = (structural - stop) / atr
            if 0 < extra <= policy.max_structure_extension_atr:
                stop = structural
        dist = stop - entry
    take = entry + side * policy.target_rr * dist
    return {
        "stopLoss": float(max(stop, PIP)),
        "takeProfit": float(max(take, PIP)),
        "stopDistance": float(dist),
        "stopDistanceATR": float(dist / atr),
        "targetRR": float(policy.target_rr),
    }


def _ml_multiplier(probability: float | None, policy: V33Policy) -> float:
    if probability is None or not np.isfinite(probability):
        return 1.0
    # For a TP at R times the stop, the pre-cost break-even hit probability is
    # 1/(1+R), not 50%. A calibrated ML probability is assessed around that
    # economic threshold. ML remains a soft modifier only.
    breakeven_p = 1.0 / (1.0 + max(policy.target_rr, 1e-6))
    q = float(np.clip((probability - breakeven_p) / 0.15, -1.0, 1.0))
    if q >= 0:
        return 1.0 + q * (policy.ml_soft_max_multiplier - 1.0)
    return 1.0 + (-q) * (policy.ml_soft_min_multiplier - 1.0)


def position_plan_v33(
    equity_eur: float,
    entry: float,
    side: int,
    atr_signal: float,
    row: pd.Series,
    p: Params,
    policy: V33Policy,
    ml_probability: float | None = None,
) -> dict:
    if equity_eur <= 0 or entry <= 0 or side not in (-1, 1):
        raise ValueError("Invalid equity, entry, or side")

    levels = _stop_target(entry, side, atr_signal, row, policy)
    trend_mult, ctx = soft_trend_multiplier(row, side, p, policy)
    ml_mult = _ml_multiplier(ml_probability, policy)

    # The stake is the cash/margin amount the user sees (e.g. EUR100).
    max_stake = max(policy.min_stake_eur, equity_eur * policy.max_stake_pct_equity)
    desired_stake = policy.base_stake_eur * trend_mult * ml_mult
    stake = float(np.clip(desired_stake, policy.min_stake_eur, max_stake))

    # Risk budget changes modestly with setup quality but remains bounded.
    risk_pct = policy.base_risk_pct_equity * trend_mult * ml_mult
    risk_pct = float(np.clip(risk_pct, policy.min_risk_pct_equity, policy.max_risk_pct_equity))
    risk_budget = equity_eur * risk_pct

    adverse = p.spread_pips * PIP / 2 + p.slippage_pips_per_side * PIP
    stop_fill = max(PIP, levels["stopLoss"] - adverse if side == 1 else levels["stopLoss"] + adverse)
    take_fill = max(PIP, levels["takeProfit"] - adverse if side == 1 else levels["takeProfit"] + adverse)

    def projected(stake_eur: float, lev: int) -> tuple[float, float, float]:
        exposure = stake_eur * lev
        risk = abs(pnl_eur(entry, stop_fill, exposure, side)) + 2 * _commission(exposure, p)
        gain = pnl_eur(entry, take_fill, exposure, side) - 2 * _commission(exposure, p)
        return exposure, risk, gain

    # If x1 already violates the risk budget, reduce the EUR stake first.
    _, risk_x1, _ = projected(stake, 1)
    if risk_x1 > risk_budget and risk_x1 > 0:
        stake *= risk_budget / risk_x1
    stake = min(stake, max_stake, equity_eur)
    if stake < policy.min_stake_eur - 1e-9:
        return {
            "valid": False,
            "rejectionReason": "STAKE_TOO_SMALL_FOR_RISK_BUDGET",
            "equityEUR": round(equity_eur, 2),
            "stakeEUR": round(max(stake, 0.0), 2),
            "trend": ctx,
            **levels,
        }

    feasible = []
    for lev in range(max(1, policy.min_leverage), max(1, policy.max_leverage) + 1):
        exposure, risk, gain = projected(stake, lev)
        if risk <= risk_budget + 1e-9:
            feasible.append((lev, exposure, risk, gain))

    if not feasible:
        return {
            "valid": False,
            "rejectionReason": "NO_LEVERAGE_LEVEL_WITHIN_RISK_BUDGET",
            "equityEUR": round(equity_eur, 2),
            "stakeEUR": round(stake, 2),
            "trend": ctx,
            **levels,
        }

    # "Optimal" leverage here means highest integer leverage that still respects
    # the explicit EUR loss budget at the stop. It is not a return forecast.
    lev, exposure, risk, gain = feasible[-1]
    return {
        "valid": True,
        "equityEUR": round(equity_eur, 2),
        "stakeEUR": round(stake, 2),
        "grossExposureEUR": round(exposure, 2),
        "leverage": int(lev),
        "accountExposureMultiple": round(exposure / equity_eur, 3),
        "leverageRange": f"x{policy.min_leverage}-x{policy.max_leverage}",
        "selectionMethod": "highest_integer_leverage_within_stop_risk_budget",
        "riskBudgetEUR": round(risk_budget, 2),
        "riskAtStopEUR": round(risk, 2),
        "riskAtStopPct": round(risk / equity_eur * 100, 3),
        "targetGainEUR": round(gain, 2),
        "rewardRisk": round(gain / risk, 3) if risk > 0 else 0.0,
        "trendMultiplier": round(trend_mult, 3),
        "mlMultiplier": round(ml_mult, 3),
        "mlProbability": None if ml_probability is None else round(float(ml_probability), 4),
        "trend": ctx,
        **levels,
    }


def _thesis_invalidated(row: pd.Series, side: int, p: Params, policy: V33Policy) -> bool:
    """Close-based thesis invalidation; execution occurs at next open."""
    mode = str(policy.invalidation_mode or "none").lower()
    if mode == "none":
        return False
    if mode == "macd":
        return bool((row.macd_hist <= 0) if side == 1 else (row.macd_hist >= 0))
    if mode == "sma50":
        return bool((row.close <= row.sma50) if side == 1 else (row.close >= row.sma50))
    if mode == "rsi50":
        return bool((row.rsi <= 50) if side == 1 else (row.rsi >= 50))
    if mode == "core2":
        confirmations = 0
        confirmations += int(row.rsi > 50) if side == 1 else int(row.rsi < 50)
        confirmations += int(row.close > row.sma50) if side == 1 else int(row.close < row.sma50)
        confirmations += int(row.macd_hist > 0) if side == 1 else int(row.macd_hist < 0)
        return confirmations < 2
    raise ValueError(f"Unknown invalidation_mode={policy.invalidation_mode}")


def _protective_stop_after_close(
    current_stop: float,
    entry: float,
    side: int,
    initial_risk_price: float,
    row: pd.Series,
    policy: V33Policy,
) -> tuple[float, str | None]:
    """Update a protective stop using only the completed bar close.

    The returned stop becomes active on the *next* bar, so a bar that creates
    the trigger cannot retroactively stop the trade on the same bar.
    """
    if initial_risk_price <= 0:
        return current_stop, None
    close_r = ((float(row.close) - entry) * side) / initial_risk_price
    new_stop = current_stop
    event = None
    if policy.breakeven_trigger_r > 0 and close_r >= policy.breakeven_trigger_r:
        be = entry
        new_stop = max(new_stop, be) if side == 1 else min(new_stop, be)
        event = "BREAKEVEN"
    if policy.trailing_trigger_r > 0 and close_r >= policy.trailing_trigger_r and pd.notna(row.get("atr", np.nan)):
        atr = float(row.atr)
        candidate = float(row.close) - side * policy.trailing_atr * atr
        if side == 1:
            new_stop = max(new_stop, candidate)
        else:
            new_stop = min(new_stop, candidate)
        event = "TRAIL"
    return float(new_stop), event


def _trade_metrics(
    entry: float,
    exit_px: float,
    stake: float,
    leverage: int,
    side: int,
    risk_eur: float,
    fav_price: float,
    adv_price: float,
    initial_risk_price: float,
) -> dict:
    mfe_price = max(0.0, (fav_price - entry) * side)
    mae_price = max(0.0, -(adv_price - entry) * side)
    realized = (exit_px - entry) * side
    return {
        "rMultiple": realized * stake * leverage / max(risk_eur * exit_px, 1e-12),
        "mfeR": mfe_price / max(initial_risk_price, 1e-12),
        "maeR": mae_price / max(initial_risk_price, 1e-12),
        "captureRatio": realized / mfe_price if realized > 0 and mfe_price > 0 else np.nan,
    }


def backtest_v33(
    df: pd.DataFrame,
    p: Params,
    policy: V33Policy,
    start=None,
    end=None,
    ml_probabilities: dict | None = None,
) -> dict:
    x = features(df, p)
    warm = max(p.trend_ema_slow, p.sma_trend, p.bb_period, p.atr_period, p.adx_period, 30)
    start_ts, end_ts = _utc_ts(start), _utc_ts(end)
    first_i = warm if start_ts is None else max(warm, int(x.index.searchsorted(start_ts, side="left")))
    last_i = len(x) - 1 if end_ts is None else min(len(x) - 1, int(x.index.searchsorted(end_ts, side="right")) - 1)
    if last_i <= first_i:
        raise ValueError("Not enough data in requested window")

    capital = float(policy.initial_capital_eur)
    position = 0
    pending = None
    pending_exit = None
    active = None
    entry_i = None
    fav = adv = np.nan
    trades: list[dict] = []
    equity_values = []
    equity_dates = []
    exposure_bars = 0
    vetoed = 0
    half_spread = p.spread_pips * PIP / 2
    slip = p.slippage_pips_per_side * PIP

    for i in range(first_i, last_i + 1):
        row = x.iloc[i]

        # A close-based invalidation/opposite signal exits at the NEXT open.
        if position and active is not None and pending_exit is not None:
            exit_px = float(row.open - half_spread - slip if position == 1 else row.open + half_spread + slip)
            final_leg_pnl = pnl_eur(active["entry"], exit_px, active["exposure"], position) - _commission(active["exposure"], p)
            capital += final_leg_pnl
            pnl = active.get("partialPnL", 0.0) + final_leg_pnl
            diag = _trade_metrics(
                active["entry"], exit_px, active["stake"], active["leverage"], position,
                active["risk"], float(fav), float(adv), abs(active["entry"] - active["initialStop"]),
            )
            ctx = active["plan"]["trend"]
            trades.append({
                "signalDate":str(active["signalDate"]),"entry":str(active["entryDate"]),"exit":str(x.index[i]),
                "side":"LONG" if position==1 else "SHORT","entryPrice":round(active["entry"],6),"exitPrice":round(exit_px,6),
                "stopLoss":round(active["initialStop"],6),"finalStop":round(active["stop"],6),"takeProfit":round(active["take"],6),
                "stakeEUR":round(active["stake"],2),"grossExposureEUR":round(active["exposure"],2),"leverage":active["leverage"],
                "usesLeverage":bool(active["leverage"]>1),"initialRiskEUR":round(active["risk"],2),"partialTaken":bool(active.get("partialTaken")),"partialPnLEUR":round(float(active.get("partialPnL",0.0)),2),"partialDate":None if active.get("partialDate") is None else str(active.get("partialDate")),"profitEUR":round(float(pnl),2),
                "reason":pending_exit,"trendScore":int(ctx.get("score",0)),"trendRegime":ctx.get("regime"),
                "volatilityRegime":ctx.get("volatilityRegime"),"mlProbability":None if active["mlProbability"] is None else round(float(active["mlProbability"]),4),
                "rMultiple":round(float(pnl / max(active["risk"], 1e-12)),3),"mfeR":round(float(diag["mfeR"]),3),"maeR":round(float(diag["maeR"]),3),
                "captureRatio":None if pd.isna(diag["captureRatio"]) else round(float(diag["captureRatio"]),3),
            })
            position=0; active=None; entry_i=None; pending_exit=None

        if position == 0 and pending is not None:
            side = int(pending["side"])
            entry = float(row.open + half_spread + slip if side == 1 else row.open - half_spread - slip)
            prob = None
            if ml_probabilities is not None:
                prob = ml_probabilities.get(pd.Timestamp(pending["signalDate"]))
            plan = position_plan_v33(capital, entry, side, float(pending["atr"]), pending["row"], p, policy, prob)
            if plan.get("valid"):
                position = side
                active = {
                    "plan": plan,
                    "signalDate": pending["signalDate"],
                    "entryDate": x.index[i],
                    "entry": entry,
                    "stop": float(plan["stopLoss"]),
                    "initialStop": float(plan["stopLoss"]),
                    "take": float(plan["takeProfit"]),
                    "stake": float(plan["stakeEUR"]),
                    "leverage": int(plan["leverage"]),
                    "exposure": float(plan["grossExposureEUR"]),
                    "initialExposure": float(plan["grossExposureEUR"]),
                    "risk": float(plan["riskAtStopEUR"]),
                    "partialTaken": False,
                    "partialPnL": 0.0,
                    "partialDate": None,
                    "beNextBar": False,
                    "mlProbability": prob,
                }
                capital -= _commission(active["exposure"], p)
                entry_i = i
                fav = adv = entry
            pending = None

        reason = raw_exit = None
        if position and active is not None:
            exposure_bars += 1
            # A break-even stop created by a partial take becomes active only
            # from the next bar, never retroactively on the trigger bar.
            if active.get("beNextBar"):
                active["stop"] = max(active["stop"], active["entry"]) if position == 1 else min(active["stop"], active["entry"])
                active["beNextBar"] = False
            initial_risk_price = abs(active["entry"] - active["initialStop"])
            partial_level = active["entry"] + position * policy.partial_take_r * initial_risk_price if policy.partial_take_r > 0 else np.nan
            if position == 1:
                fav = max(float(fav), float(row.high))
                adv = min(float(adv), float(row.low))
                if row.open <= active["stop"]:
                    reason, raw_exit = "SL_GAP", float(row.open)
                elif row.low <= active["stop"]:
                    # Conservative same-bar ordering: stop wins over partial/TP.
                    reason, raw_exit = "SL", active["stop"]
                else:
                    if (
                        not active["partialTaken"] and policy.partial_take_r > 0
                        and 0 < policy.partial_fraction < 1 and row.high >= partial_level
                    ):
                        frac = float(policy.partial_fraction)
                        close_exposure = active["initialExposure"] * frac
                        partial_fill = float(partial_level - half_spread - slip)
                        ppnl = pnl_eur(active["entry"], partial_fill, close_exposure, position) - _commission(close_exposure, p)
                        capital += ppnl
                        active["partialPnL"] += ppnl
                        active["partialTaken"] = True
                        active["partialDate"] = x.index[i]
                        active["exposure"] = max(0.0, active["exposure"] - close_exposure)
                        if policy.partial_move_stop_to_be:
                            active["beNextBar"] = True
                    if row.open >= active["take"]:
                        reason, raw_exit = "TP", active["take"]
                    elif row.high >= active["take"]:
                        reason, raw_exit = "TP", active["take"]
            else:
                fav = min(float(fav), float(row.low))
                adv = max(float(adv), float(row.high))
                if row.open >= active["stop"]:
                    reason, raw_exit = "SL_GAP", float(row.open)
                elif row.high >= active["stop"]:
                    reason, raw_exit = "SL", active["stop"]
                else:
                    if (
                        not active["partialTaken"] and policy.partial_take_r > 0
                        and 0 < policy.partial_fraction < 1 and row.low <= partial_level
                    ):
                        frac = float(policy.partial_fraction)
                        close_exposure = active["initialExposure"] * frac
                        partial_fill = float(partial_level + half_spread + slip)
                        ppnl = pnl_eur(active["entry"], partial_fill, close_exposure, position) - _commission(close_exposure, p)
                        capital += ppnl
                        active["partialPnL"] += ppnl
                        active["partialTaken"] = True
                        active["partialDate"] = x.index[i]
                        active["exposure"] = max(0.0, active["exposure"] - close_exposure)
                        if policy.partial_move_stop_to_be:
                            active["beNextBar"] = True
                    if row.open <= active["take"]:
                        reason, raw_exit = "TP", active["take"]
                    elif row.low <= active["take"]:
                        reason, raw_exit = "TP", active["take"]

            if policy.max_holding_bars > 0 and reason is None and i - entry_i >= policy.max_holding_bars:
                reason, raw_exit = "TIME", float(row.close)

            if reason:
                exit_px = float(raw_exit - half_spread - slip if position == 1 else raw_exit + half_spread + slip)
                final_leg_pnl = pnl_eur(active["entry"], exit_px, active["exposure"], position) - _commission(active["exposure"], p)
                capital += final_leg_pnl
                pnl = active.get("partialPnL", 0.0) + final_leg_pnl
                diag = _trade_metrics(
                    active["entry"], exit_px, active["stake"], active["leverage"], position,
                    active["risk"], float(fav), float(adv), abs(active["entry"] - active["initialStop"]),
                )
                ctx = active["plan"]["trend"]
                trades.append({
                    "signalDate": str(active["signalDate"]),
                    "entry": str(active["entryDate"]),
                    "exit": str(x.index[i]),
                    "side": "LONG" if position == 1 else "SHORT",
                    "entryPrice": round(active["entry"], 6),
                    "exitPrice": round(exit_px, 6),
                    "stopLoss": round(active["initialStop"], 6),
                    "finalStop": round(active["stop"], 6),
                    "takeProfit": round(active["take"], 6),
                    "stakeEUR": round(active["stake"], 2),
                    "grossExposureEUR": round(active["exposure"], 2),
                    "leverage": active["leverage"],
                    "usesLeverage": bool(active["leverage"] > 1),
                    "accountExposureMultiple": round(active["initialExposure"] / max(active["plan"]["equityEUR"], 1e-12), 3),
                    "initialRiskEUR": round(active["risk"], 2),
                    "partialTaken": bool(active.get("partialTaken")),
                    "partialPnLEUR": round(float(active.get("partialPnL", 0.0)), 2),
                    "partialDate": None if active.get("partialDate") is None else str(active.get("partialDate")),
                    "profitEUR": round(float(pnl), 2),
                    "reason": reason,
                    "trendScore": int(ctx.get("score", 0)),
                    "trendRegime": ctx.get("regime"),
                    "volatilityRegime": ctx.get("volatilityRegime"),
                    "mlProbability": None if active["mlProbability"] is None else round(float(active["mlProbability"]), 4),
                    "rMultiple": round(float(pnl / max(active["risk"], 1e-12)), 3),
                    "mfeR": round(float(diag["mfeR"]), 3),
                    "maeR": round(float(diag["maeR"]), 3),
                    "captureRatio": None if pd.isna(diag["captureRatio"]) else round(float(diag["captureRatio"]), 3),
                })
                position = 0
                active = None
                entry_i = None
                pending_exit = None
            else:
                # Protection is derived from this completed close and becomes
                # effective next bar only.
                new_stop, _ = _protective_stop_after_close(
                    active["stop"], active["entry"], position,
                    abs(active["entry"] - active["initialStop"]), row, policy
                )
                active["stop"] = new_stop
                if policy.exit_on_opposite_signal:
                    opp = technical_candidate_row(row, p)
                    if opp == -position:
                        pending_exit = "OPPOSITE_SIGNAL"
                if pending_exit is None and _thesis_invalidated(row, position, p, policy):
                    pending_exit = "THESIS_INVALIDATION"

        mtm = capital
        if position and active is not None:
            mark = float(row.close - half_spread if position == 1 else row.close + half_spread)
            mtm += pnl_eur(active["entry"], mark, active["exposure"], position)
        equity_values.append(mtm)
        equity_dates.append(x.index[i])

        if position == 0 and pending is None and i < last_i:
            side = technical_candidate_row(row, p)
            if side:
                if should_veto_countertrend(row, side, p, policy):
                    vetoed += 1
                else:
                    pending = {
                        "side": int(side),
                        "signalDate": x.index[i],
                        "atr": float(row.atr),
                        "row": row.copy(),
                    }

    open_trade = None
    if position and active is not None:
        row = x.iloc[last_i]
        mark = float(row.close - half_spread if position == 1 else row.close + half_spread)
        unrealized = pnl_eur(active["entry"], mark, active["exposure"], position)
        ctx = active["plan"]["trend"]
        open_trade = {
            "signalDate": str(active["signalDate"]),
            "entry": str(active["entryDate"]),
            "side": "LONG" if position == 1 else "SHORT",
            "entryPrice": round(active["entry"], 6),
            "currentPrice": round(mark, 6),
            "stopLoss": round(active["initialStop"], 6),
            "currentStop": round(active["stop"], 6),
            "takeProfit": round(active["take"], 6),
            "stakeEUR": round(active["stake"], 2),
            "grossExposureEUR": round(active["exposure"], 2),
            "initialGrossExposureEUR": round(active["initialExposure"], 2),
            "partialTaken": bool(active.get("partialTaken")),
            "partialPnLEUR": round(float(active.get("partialPnL", 0.0)), 2),
            "leverage": active["leverage"],
            "usesLeverage": bool(active["leverage"] > 1),
            "accountExposureMultiple": round(active["initialExposure"] / max(active["plan"]["equityEUR"], 1e-12), 3),
            "riskAtStopEUR": round(active["risk"], 2),
            "unrealizedPnLEUR": round(float(unrealized), 2),
            "trendScore": int(ctx.get("score", 0)),
            "trendRegime": ctx.get("regime"),
        }

    equity = pd.Series(equity_values, index=equity_dates, name="equity_eur", dtype=float)
    final_equity = float(equity.iloc[-1])
    metrics = _metrics(equity, trades, policy.initial_capital_eur, final_equity, exposure_bars)
    metrics.update({
        "realizedCapitalEUR": round(capital, 2),
        "openPositions": 1 if open_trade else 0,
        "countertrendVetoes": int(vetoed),
        "avgStakeEUR": round(float(np.mean([t["stakeEUR"] for t in trades])), 2) if trades else 0.0,
        "avgGrossExposureEUR": round(float(np.mean([t["grossExposureEUR"] for t in trades])), 2) if trades else 0.0,
        "avgLeverage": round(float(np.mean([t["leverage"] for t in trades])), 3) if trades else 0.0,
        "maxLeverageUsed": int(max([t["leverage"] for t in trades], default=0)),
    })
    return {
        "metrics": metrics,
        "trades": trades,
        "openTrade": open_trade,
        "equity": equity,
        "features": x,
    }


def current_snapshot_v33(
    x: pd.DataFrame,
    p: Params,
    policy: V33Policy,
    equity_eur: float | None = None,
    ml_probability: float | None = None,
) -> dict:
    row = x.iloc[-1]
    side = technical_candidate_row(row, p)
    equity = policy.initial_capital_eur if equity_eur is None else float(equity_eur)
    vetoed = bool(side and should_veto_countertrend(row, side, p, policy))
    plan = None
    if side and not vetoed and pd.notna(row.atr):
        plan = position_plan_v33(equity, float(row.close), side, float(row.atr), row, p, policy, ml_probability)
        if not plan.get("valid"):
            side = 0
    ctx = trend_context(row, side if side else (1 if row.close >= row.sma50 else -1), p)
    return {
        "date": str(x.index[-1]),
        "currentPrice": round(float(row.close), 6),
        "signal": "BUY (LONG)" if side == 1 else ("SELL (SHORT)" if side == -1 else "NEUTRAL (WAIT)"),
        "rsi": round(float(row.rsi), 2) if pd.notna(row.rsi) else None,
        "sma50": round(float(row.sma50), 6) if pd.notna(row.sma50) else None,
        "macdHist": round(float(row.macd_hist), 7) if pd.notna(row.macd_hist) else None,
        "adx": round(float(row.adx), 2) if pd.notna(row.adx) else None,
        "trendScore": int(ctx.get("score", 0)),
        "trendRegime": ctx.get("regime"),
        "volatilityRegime": volatility_regime(row),
        "countertrendVeto": vetoed,
        "positionPlan": plan,
        "exitPolicy": "TP_SL_ONLY" if policy.max_holding_bars <= 0 else f"TP_SL_OR_{policy.max_holding_bars}_BARS",
    }
