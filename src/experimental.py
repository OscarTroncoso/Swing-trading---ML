"""Research-only strategy variants.

Nothing in this module is used by the production dashboard unless a user explicitly
runs a research script. The purpose is to test whether return can be improved by
risk sizing or exit management without contaminating the small production signal.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import numpy as np
import pandas as pd

from .strategy_engine import PIP, Params, _commission, _metrics, _utc_ts, features, signal_row, position_plan, pnl_eur


@dataclass(frozen=True)
class ExitPolicy:
    name: str = "time"
    max_holding_bars: int = 5
    trailing_atr: float | None = None
    breakeven_trigger_atr: float | None = None
    momentum_fade: bool = False
    hard_max_holding_bars: int = 20


def _fade_condition(row: pd.Series, side: int) -> bool:
    """A simple, interpretable loss-of-momentum condition evaluated at bar close."""
    if side == 1:
        return bool(row.rsi < 50 or row.macd_hist < 0 or row.close < row.ema20)
    return bool(row.rsi > 50 or row.macd_hist > 0 or row.close > row.ema20)


def backtest_exit_policy(df: pd.DataFrame, p: Params, policy: ExitPolicy, start=None, end=None) -> dict:
    """Backtest production entries with an alternative exit policy.

    Important anti-look-ahead rules:
    * entry signals are produced at close t and filled at open t+1;
    * ATR used for initial stop/TP and sizing is frozen on signal bar t;
    * trailing/breakeven stops are updated only after bar close and become active
      on the next bar; they never retroactively trigger inside the bar that created
      them;
    * momentum-fade exits are decided at close and filled at next open.
    """
    x = features(df, p)
    warm = max(p.sma_trend, p.bb_period, p.atr_period, p.adx_period, 30)
    start_ts, end_ts = _utc_ts(start), _utc_ts(end)
    first_i = max(warm, int(x.index.searchsorted(start_ts, side="left"))) if start_ts is not None else warm
    last_i = len(x) - 1 if end_ts is None else min(len(x) - 1, int(x.index.searchsorted(end_ts, side="right")) - 1)
    if last_i <= first_i:
        raise ValueError("Not enough data in requested experimental backtest window.")

    capital = p.initial_capital_eur
    position = 0
    units = 0.0
    entry = stop = take = np.nan
    entry_i = None
    entry_date = active_signal_date = None
    entry_atr = np.nan
    best_price = np.nan
    pending_entry = None
    pending_exit = None
    active_plan = None
    trades: list[dict] = []
    equity_values, equity_dates = [], []
    exposure = 0
    half_spread = p.spread_pips * PIP / 2
    slip = p.slippage_pips_per_side * PIP

    def close_position(raw_price: float, reason: str, date: pd.Timestamp):
        nonlocal capital, position, units, entry, stop, take, entry_i, entry_date, active_signal_date, pending_exit, active_plan
        adverse = half_spread + slip
        exit_px = float(raw_price - adverse if position == 1 else raw_price + adverse)
        pnl = pnl_eur(entry, exit_px, units, position) - _commission(units, p)
        capital += pnl
        trades.append({
            "signalDate": str(active_signal_date),
            "entry": str(entry_date),
            "exit": str(date),
            "side": "LONG" if position == 1 else "SHORT",
            "entryPrice": round(float(entry), 6),
            "exitPrice": round(exit_px, 6),
            "stopLoss": round(float(stop), 6),
            "takeProfit": round(float(take), 6),
            "units": round(float(units), 2),
            "lots": round(float(units)/100_000.0,4),
            "notionalEUR": round(float(units),2),
            "leverage": active_plan["recommended"]["leverage"] if active_plan else 0.0,
            "usesLeverage": active_plan["recommended"]["usesLeverage"] if active_plan else False,
            "appliedRiskPct": active_plan.get("appliedRiskPct") if active_plan else None,
            "profitEUR": round(float(pnl), 2),
            "reason": reason,
            "exitPolicy": policy.name,
        })
        position = 0
        units = 0.0
        entry = stop = take = np.nan
        entry_i = None
        entry_date = active_signal_date = None
        pending_exit = None
        active_plan = None

    for i in range(first_i, last_i + 1):
        r = x.iloc[i]

        # Discretionary exit decided at previous close -> today's open.
        if position and pending_exit is not None:
            close_position(float(r.open), pending_exit, x.index[i])

        # Entry decided at previous close -> today's open.
        if position == 0 and pending_entry is not None:
            position = int(pending_entry["side"])
            adverse = half_spread + slip
            entry = float(r.open + adverse if position == 1 else r.open - adverse)
            entry_atr = float(pending_entry["atr"])
            plan = position_plan(capital, entry, position, entry_atr, pending_entry["row"], p)
            active_plan = plan
            units = float(plan["recommended"]["units"])
            if units <= 0:
                position = 0
                pending_entry = None
                continue
            capital -= _commission(units, p)
            stop = float(plan["stopLoss"])
            take = float(plan["takeProfit"])
            entry_i = i
            entry_date = x.index[i]
            active_signal_date = pending_entry["signal_date"]
            best_price = entry
            pending_entry = None

        # Intraday barriers use only levels known before this bar.
        if position:
            exposure += 1
            exit_reason = None
            exit_raw = None
            if position == 1:
                if r.low <= stop:
                    exit_reason, exit_raw = "SL", stop
                elif r.high >= take:
                    exit_reason, exit_raw = "TP", take
            else:
                if r.high >= stop:
                    exit_reason, exit_raw = "SL", stop
                elif r.low <= take:
                    exit_reason, exit_raw = "TP", take
            if exit_reason:
                close_position(float(exit_raw), exit_reason, x.index[i])

        # Mark equity after any intraday exit.
        mtm = capital
        if position:
            mark = float(r.close - half_spread if position == 1 else r.close + half_spread)
            mtm += pnl_eur(entry, mark, units, position)
        equity_values.append(mtm)
        equity_dates.append(x.index[i])

        # End-of-bar risk management. Changes take effect on the NEXT bar.
        if position:
            best_price = max(float(best_price), float(r.high)) if position == 1 else min(float(best_price), float(r.low))
            favourable = (float(r.close) - entry) * position
            if policy.breakeven_trigger_atr is not None and favourable >= policy.breakeven_trigger_atr * entry_atr:
                stop = max(stop, entry) if position == 1 else min(stop, entry)
            if policy.trailing_atr is not None:
                candidate = best_price - policy.trailing_atr * entry_atr if position == 1 else best_price + policy.trailing_atr * entry_atr
                stop = max(stop, candidate) if position == 1 else min(stop, candidate)

            held = i - entry_i
            if policy.momentum_fade and _fade_condition(r, position) and i < last_i:
                pending_exit = "FADE_NEXT_OPEN"
            elif policy.name == "time" and held >= policy.max_holding_bars:
                # Production-style deterministic time exit at close.
                close_position(float(r.close), "TIME", x.index[i])
                equity_values[-1] = capital
            elif policy.name != "time" and held >= policy.hard_max_holding_bars and i < last_i:
                pending_exit = "MAX_HOLD_NEXT_OPEN"

        # Generate a new entry signal only when flat and no exit is pending.
        if position == 0 and pending_entry is None and i < last_i:
            s = signal_row(r, p)
            if s:
                pending_entry = {"side": s, "signal_date": x.index[i], "atr": float(r.atr), "row": r.copy()}

    if position:
        r = x.iloc[last_i]
        close_position(float(r.close), "EOD", x.index[last_i])
        equity_values[-1] = capital

    equity = pd.Series(equity_values, index=equity_dates, name="equity", dtype=float)
    return {
        "metrics": _metrics(equity, trades, p.initial_capital_eur, capital, exposure),
        "trades": trades,
        "equity": equity,
        "features": x,
    }


def standard_exit_policies() -> list[ExitPolicy]:
    """Small predeclared research set; deliberately not an optimizer."""
    return [
        ExitPolicy(name="time", max_holding_bars=5),
        ExitPolicy(name="fade", momentum_fade=True, hard_max_holding_bars=20),
        ExitPolicy(name="trail_1.5", trailing_atr=1.5, hard_max_holding_bars=20),
        ExitPolicy(name="trail_2.0", trailing_atr=2.0, hard_max_holding_bars=20),
        ExitPolicy(name="breakeven_trail", trailing_atr=2.0, breakeven_trigger_atr=1.0, hard_max_holding_bars=20),
        ExitPolicy(name="fade_trail", trailing_atr=2.0, breakeven_trigger_atr=1.0, momentum_fade=True, hard_max_holding_bars=20),
    ]
