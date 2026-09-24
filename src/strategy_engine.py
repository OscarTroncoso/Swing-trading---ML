"""EUR/USD swing strategy engine v3.0.

Key design choices
------------------
* Account equity is denominated in EUR. `initial_capital_eur=10_000` means
  10,000 EUR of account equity, not 10,000 FX units.
* EUR/USD position units are EUR base-currency units (100,000 units = 1 lot).
* Signal at close t; execution at open t+1.
* ATR/position risk inputs are frozen on the signal bar: no entry-bar look-ahead.
* Stop/TP checks use OHLC high/low with conservative stop-first ordering.
* P&L generated in USD is converted back to EUR at the exit/mark FX rate.
* Position size can be fixed-notional, fixed-risk, or adaptive-risk.
* Adaptive risk uses volatility targeting and (optionally) an ML probability,
  but the directional baseline remains interpretable RSI + SMA50 + MACD.

Research software only. It does not guarantee executable broker fills or returns.
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
    risk_per_trade: float = 0.005       # base risk = 0.50% of current equity
    min_risk_per_trade: float = 0.0025
    max_risk_per_trade: float = 0.0100
    max_leverage: float = 3.0           # internal strategy cap, not a broker promise
    lot_step_units: float = 1_000.0     # 0.01 standard lot
    min_units: float = 1_000.0
    absolute_max_units: float = 0.0     # 0 => only equity*max_leverage cap
    volatility_target: float = 0.08     # annualized EUR/USD realized-vol target
    vol_scale_min: float = 0.65
    vol_scale_max: float = 1.25

    # Execution costs, denominated/converted to account EUR where applicable
    spread_pips: float = 0.8
    slippage_pips_per_side: float = 0.1
    commission_per_million_per_side_eur: float = 0.0

    # Signal / risk model
    rsi_period: int = 14
    rsi_long: float = 55.0
    rsi_short: float = 45.0
    sma_trend: int = 50
    bb_period: int = 20
    atr_period: int = 14
    adx_period: int = 14
    stop_atr: float = 1.5
    take_atr: float = 2.2
    max_holding_bars: int = 5
    require_macd: bool = True

    # Optional research switches; off by default.
    require_adx: bool = False
    min_adx: float = 18.0
    enable_range_module: bool = False
    range_adx_max: float = 17.0
    range_z: float = 1.5
    range_rsi_low: float = 35.0
    range_rsi_high: float = 65.0

    # ML risk integration. Direction can still be produced without ML.
    ml_probability_threshold: float = 0.60
    ml_confidence_reference: float = 0.78


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
        # Handles Yahoo two-header exports.
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


def adx_wilder(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df.high.diff()
    down = -df.low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr = atr_wilder(df, n)
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def features(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    x = _norm_cols(df)
    c = x.close
    x["rsi"] = rsi_wilder(c, p.rsi_period)
    x["rsi_original"] = rsi_original(c, p.rsi_period)
    x["sma50"] = c.rolling(p.sma_trend).mean()
    x["ema20"] = c.ewm(span=20, adjust=False).mean()
    x["ema50"] = c.ewm(span=50, adjust=False).mean()
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
    x["adx"] = adx_wilder(x, p.adx_period)
    x["rv20"] = c.pct_change().rolling(20).std(ddof=0) * math.sqrt(252)
    x["ret1"] = c.pct_change(1)
    x["ret5"] = c.pct_change(5)
    x["ret20"] = c.pct_change(20)
    x["sma_gap"] = c / x.sma50 - 1
    x["atr_pct"] = x.atr / c
    x["macd_atr"] = x.macd_hist / x.atr.replace(0, np.nan)
    x["range_pct"] = (x.high - x.low) / c
    x["body_pct"] = (x.close - x.open) / c
    # Past-only volatility percentile, useful for risk scaling/research.
    x["rv20_median_252"] = x.rv20.shift(1).rolling(252, min_periods=60).median()
    return x


# ----------------------------- signals ------------------------------------
def signal_row(r: pd.Series, p: Params) -> int:
    """Small production directional signal: RSI + SMA50 + optional MACD."""
    if pd.isna(r.rsi) or pd.isna(r.sma50) or pd.isna(r.atr):
        return 0
    long_mom = r.rsi > p.rsi_long and r.close > r.sma50
    short_mom = r.rsi < p.rsi_short and r.close < r.sma50
    if p.require_macd:
        long_mom = long_mom and r.macd_hist > 0
        short_mom = short_mom and r.macd_hist < 0
    if p.require_adx:
        if pd.isna(r.adx):
            return 0
        long_mom = long_mom and r.adx >= p.min_adx
        short_mom = short_mom and r.adx >= p.min_adx
    if long_mom:
        return 1
    if short_mom:
        return -1
    if p.enable_range_module and pd.notna(r.adx) and r.adx <= p.range_adx_max:
        if r.bb_z <= -p.range_z and r.rsi <= p.range_rsi_low:
            return 1
        if r.bb_z >= p.range_z and r.rsi >= p.range_rsi_high:
            return -1
    return 0


def broad_candidate_row(r: pd.Series) -> int:
    """Broader ML candidate generator. ML may accept/reject; it never sees future data."""
    if pd.isna(r.rsi) or pd.isna(r.sma50):
        return 0
    if r.rsi > 52 and r.close > r.sma50:
        return 1
    if r.rsi < 48 and r.close < r.sma50:
        return -1
    return 0


def legacy_signal_row(r: pd.Series) -> int:
    if pd.isna(r.rsi_original) or pd.isna(r.bb_lower) or pd.isna(r.bb_upper):
        return 0
    if r.close <= r.bb_lower or r.rsi_original < 45:
        return 1
    if r.close >= r.bb_upper or r.rsi_original > 55:
        return -1
    return 0


# ----------------------- account / risk / leverage -------------------------
def _commission(units: float, p: Params) -> float:
    return (units / 1_000_000.0) * p.commission_per_million_per_side_eur


def pnl_eur(entry: float, exit_px: float, units: float, side: int) -> float:
    """EUR/USD P&L converted from quote USD back to account EUR at exit rate."""
    if exit_px <= 0:
        raise ValueError("exit_px must be positive")
    pnl_usd = (exit_px - entry) * units * side
    return pnl_usd / exit_px


def adaptive_risk_pct(r: pd.Series, p: Params, ml_probability: float | None = None) -> float:
    """Risk budget based on current equity, vol target, and optional ML confidence.

    This changes exposure, not trade direction. All inputs are from the signal bar.
    """
    risk = p.risk_per_trade
    if pd.notna(r.get("rv20", np.nan)) and float(r.rv20) > 0:
        vol_scale = p.volatility_target / float(r.rv20)
        vol_scale = float(np.clip(vol_scale, p.vol_scale_min, p.vol_scale_max))
        risk *= vol_scale
    if ml_probability is not None:
        if ml_probability < p.ml_probability_threshold:
            return 0.0
        # Smoothly scale confidence from 0.80x at threshold to 1.25x at reference.
        den = max(p.ml_confidence_reference - p.ml_probability_threshold, 1e-6)
        q = float(np.clip((ml_probability - p.ml_probability_threshold) / den, 0.0, 1.0))
        risk *= 0.80 + 0.45 * q
    return float(np.clip(risk, p.min_risk_per_trade, p.max_risk_per_trade))


def _round_units_down(units: float, step: float) -> float:
    if step <= 0:
        return units
    return math.floor(max(0.0, units) / step) * step


def position_plan(
    equity_eur: float,
    entry: float,
    side: int,
    atr_signal: float,
    signal_row_data: pd.Series,
    p: Params,
    ml_probability: float | None = None,
) -> dict:
    """Compute position alternatives and the risk-aware recommended size.

    For EUR/USD the notional in account/base EUR equals the number of EUR units.
    Therefore leverage = EUR units / EUR equity.
    """
    if equity_eur <= 0 or entry <= 0 or side not in (-1, 1):
        raise ValueError("Invalid equity/entry/side")
    adverse = (p.spread_pips * PIP / 2) + (p.slippage_pips_per_side * PIP)
    stop_dist = max(p.stop_atr * atr_signal, PIP)
    take_dist = max(p.take_atr * atr_signal, PIP)
    raw_stop = entry - stop_dist if side == 1 else entry + stop_dist
    raw_take = entry + take_dist if side == 1 else entry - take_dist
    stop_fill = raw_stop - adverse if side == 1 else raw_stop + adverse
    take_fill = raw_take - adverse if side == 1 else raw_take + adverse
    stop_fill = max(stop_fill, PIP)
    take_fill = max(take_fill, PIP)

    if p.sizing_mode == "fixed_notional":
        target_risk_pct = None
        desired_units = p.fixed_notional_eur
    else:
        target_risk_pct = p.risk_per_trade if p.sizing_mode == "risk" else adaptive_risk_pct(signal_row_data, p, ml_probability)
        risk_budget = equity_eur * target_risk_pct
        # Exact-ish account-EUR loss per one EUR unit at stop, including both commissions.
        stop_loss_per_unit = abs(pnl_eur(entry, stop_fill, 1.0, side)) + 2 * (p.commission_per_million_per_side_eur / 1_000_000.0)
        desired_units = risk_budget / max(stop_loss_per_unit, 1e-12)

    no_leverage_cap = equity_eur
    leverage_cap = equity_eur * p.max_leverage
    if p.absolute_max_units > 0:
        leverage_cap = min(leverage_cap, p.absolute_max_units)
    chosen_units = min(desired_units, leverage_cap)
    chosen_units = _round_units_down(chosen_units, p.lot_step_units)
    if chosen_units < p.min_units:
        chosen_units = 0.0

    no_lev_units = _round_units_down(min(desired_units, no_leverage_cap), p.lot_step_units)
    if no_lev_units < p.min_units:
        no_lev_units = 0.0

    def option(units: float, name: str) -> dict:
        if units <= 0:
            return {"name": name, "units": 0.0, "lots": 0.0, "notionalEUR": 0.0, "leverage": 0.0, "usesLeverage": False,
                    "riskAtStopEUR": 0.0, "targetGainEUR": 0.0, "rewardRisk": 0.0}
        risk_eur = abs(pnl_eur(entry, stop_fill, units, side)) + 2 * _commission(units, p)
        gain_eur = pnl_eur(entry, take_fill, units, side) - 2 * _commission(units, p)
        lev = units / equity_eur
        return {
            "name": name,
            "units": round(units, 0),
            "lots": round(units / 100_000.0, 4),
            "notionalEUR": round(units, 2),
            "leverage": round(lev, 3),
            "usesLeverage": bool(lev > 1.0001),
            "riskAtStopEUR": round(risk_eur, 2),
            "riskAtStopPct": round(risk_eur / equity_eur * 100, 3),
            "targetGainEUR": round(gain_eur, 2),
            "rewardRisk": round(gain_eur / risk_eur, 3) if risk_eur > 0 else 0.0,
        }

    recommended = option(chosen_units, "risk_target")
    no_leverage = option(no_lev_units, "no_leverage")
    desired_leverage = desired_units / equity_eur if equity_eur > 0 else 0.0
    if chosen_units <= 0:
        label = "SKIP_SIZE_TOO_SMALL"
    elif recommended["usesLeverage"]:
        label = "RISK_TARGET_WITH_LEVERAGE"
    else:
        label = "RISK_TARGET_NO_LEVERAGE"
    if desired_units > leverage_cap + 1e-9:
        label = "LEVERAGE_CAPPED"

    return {
        "equityEUR": round(equity_eur, 2),
        "sizingMode": p.sizing_mode,
        "baseRiskPct": round(p.risk_per_trade * 100, 3),
        "appliedRiskPct": None if target_risk_pct is None else round(target_risk_pct * 100, 3),
        "maxLeverage": p.max_leverage,
        "desiredUnits": round(desired_units, 2),
        "desiredLeverage": round(desired_leverage, 3),
        "leverageCapHit": bool(desired_units > leverage_cap + 1e-9),
        "recommended": recommended,
        "noLeverageAlternative": no_leverage,
        "modelSelectedOption": "recommended",
        "selection": label,
        "stopLoss": round(raw_stop, 6),
        "takeProfit": round(raw_take, 6),
        "levelsIndicativeUntilNextOpen": True,
        "mlProbability": None if ml_probability is None else round(float(ml_probability), 4),
    }


# ----------------------------- backtest -----------------------------------
def _metrics(equity: pd.Series, trades: list[dict], initial: float, final_capital: float, exposure: int) -> dict:
    rets = equity.pct_change().fillna(0.0)
    peak = equity.cummax()
    drawdown = equity / peak - 1
    pnl = np.array([t["profitEUR"] for t in trades], dtype=float) if trades else np.array([], dtype=float)
    wins = pnl[pnl > 0]
    losses = -pnl[pnl < 0]
    vol = rets.std(ddof=0)
    downside = rets[rets < 0].std(ddof=0)
    pf = wins.sum() / losses.sum() if losses.sum() > 0 else (999.0 if wins.sum() > 0 else 0.0)
    return {
        "accountCurrency": "EUR",
        "initialCapitalEUR": round(initial, 2),
        "finalCapitalEUR": round(final_capital, 2),
        "totalReturn": round((final_capital / initial - 1) * 100, 4),
        "winRate": round((pnl > 0).mean() * 100, 2) if len(pnl) else 0.0,
        "maxDrawdown": round(float(-drawdown.min() * 100), 4) if len(drawdown) else 0.0,
        "profitFactor": round(float(pf), 4),
        "totalTrades": int(len(trades)),
        "sharpe": round(float(rets.mean() / vol * math.sqrt(252)), 4) if vol > 0 else 0.0,
        "sortino": round(float(rets.mean() / downside * math.sqrt(252)), 4) if pd.notna(downside) and downside > 0 else 0.0,
        "avgTradeEUR": round(float(pnl.mean()), 2) if len(pnl) else 0.0,
        "exposurePct": round(exposure / max(1, len(equity)) * 100, 2),
        "avgLeverage": round(float(np.mean([t.get("leverage", 0) for t in trades])), 3) if trades else 0.0,
        "maxLeverageUsed": round(float(max([t.get("leverage", 0) for t in trades], default=0)), 3),
        "leveragedTrades": int(sum(bool(t.get("usesLeverage")) for t in trades)),
    }


def backtest(df: pd.DataFrame, p: Params, start=None, end=None) -> dict:
    """V3 baseline: independent signals + EUR-equity risk sizing."""
    x = features(df, p)
    warm = max(p.sma_trend, p.bb_period, p.atr_period, p.adx_period, 30)
    start_ts, end_ts = _utc_ts(start), _utc_ts(end)
    first_i = warm if start_ts is None else max(warm, int(x.index.searchsorted(start_ts, side="left")))
    last_i = len(x) - 1 if end_ts is None else min(len(x) - 1, int(x.index.searchsorted(end_ts, side="right")) - 1)
    if last_i <= first_i:
        raise ValueError("Not enough data in requested backtest window after warm-up.")

    capital = p.initial_capital_eur
    position = 0
    units = 0.0
    entry = stop = take = np.nan
    entry_i = None
    entry_date = active_signal_date = None
    pending = None
    trades: list[dict] = []
    equity_values, equity_dates = [], []
    exposure = 0
    half_spread = p.spread_pips * PIP / 2
    slip = p.slippage_pips_per_side * PIP

    for i in range(first_i, last_i + 1):
        r = x.iloc[i]
        if position == 0 and pending is not None:
            position = int(pending["side"])
            adverse = half_spread + slip
            entry = float(r.open + adverse if position == 1 else r.open - adverse)
            plan = position_plan(capital, entry, position, float(pending["atr"]), pending["row"], p, pending.get("ml_probability"))
            units = float(plan["recommended"]["units"])
            if units <= 0:
                position = 0
                pending = None
            else:
                capital -= _commission(units, p)
                stop = float(plan["stopLoss"])
                take = float(plan["takeProfit"])
                entry_i = i
                entry_date = x.index[i]
                active_signal_date = pending["signal_date"]
                active_plan = plan
                pending = None

        exit_reason = exit_raw = None
        if position:
            exposure += 1
            if position == 1:
                stop_hit, take_hit = r.low <= stop, r.high >= take
                if stop_hit:
                    exit_reason, exit_raw = "SL", stop
                elif take_hit:
                    exit_reason, exit_raw = "TP", take
            else:
                stop_hit, take_hit = r.high >= stop, r.low <= take
                if stop_hit:
                    exit_reason, exit_raw = "SL", stop
                elif take_hit:
                    exit_reason, exit_raw = "TP", take
            if exit_reason is None and i - entry_i >= p.max_holding_bars:
                exit_reason, exit_raw = "TIME", float(r.close)
            if exit_reason:
                adverse = half_spread + slip
                exit_px = float(exit_raw - adverse if position == 1 else exit_raw + adverse)
                trade_pnl = pnl_eur(entry, exit_px, units, position) - _commission(units, p)
                capital += trade_pnl
                trades.append({
                    "signalDate": str(active_signal_date), "entry": str(entry_date), "exit": str(x.index[i]),
                    "side": "LONG" if position == 1 else "SHORT", "entryPrice": round(entry, 6),
                    "exitPrice": round(exit_px, 6), "stopLoss": round(stop, 6), "takeProfit": round(take, 6),
                    "units": round(units, 0), "lots": round(units / 100_000.0, 4),
                    "notionalEUR": round(units, 2), "leverage": active_plan["recommended"]["leverage"],
                    "usesLeverage": active_plan["recommended"]["usesLeverage"],
                    "appliedRiskPct": active_plan["appliedRiskPct"], "profitEUR": round(float(trade_pnl), 2),
                    "reason": exit_reason,
                })
                position = 0
                units = 0.0
                entry_i = entry_date = active_signal_date = None

        mtm = capital
        if position:
            mark = float(r.close - half_spread if position == 1 else r.close + half_spread)
            mtm += pnl_eur(entry, mark, units, position)
        equity_values.append(mtm)
        equity_dates.append(x.index[i])

        if position == 0 and pending is None and i < last_i:
            s = signal_row(r, p)
            if s:
                pending = {"side": s, "signal_date": x.index[i], "atr": float(r.atr), "row": r.copy()}

    if position:
        r = x.iloc[last_i]
        adverse = half_spread + slip
        exit_px = float(r.close - adverse if position == 1 else r.close + adverse)
        trade_pnl = pnl_eur(entry, exit_px, units, position) - _commission(units, p)
        capital += trade_pnl
        trades.append({
            "signalDate": str(active_signal_date), "entry": str(entry_date), "exit": str(x.index[last_i]),
            "side": "LONG" if position == 1 else "SHORT", "entryPrice": round(entry, 6), "exitPrice": round(exit_px, 6),
            "stopLoss": round(stop, 6), "takeProfit": round(take, 6), "units": round(units, 0),
            "lots": round(units / 100_000.0, 4), "notionalEUR": round(units, 2),
            "leverage": active_plan["recommended"]["leverage"], "usesLeverage": active_plan["recommended"]["usesLeverage"],
            "appliedRiskPct": active_plan["appliedRiskPct"], "profitEUR": round(float(trade_pnl), 2), "reason": "EOD",
        })
        equity_values[-1] = capital

    equity = pd.Series(equity_values, index=equity_dates, name="equity_eur", dtype=float)
    return {"metrics": _metrics(equity, trades, p.initial_capital_eur, capital, exposure), "trades": trades, "equity": equity, "features": x}


def backtest_legacy_fair(df: pd.DataFrame, p: Params, start=None, end=None) -> dict:
    """V1 signals, but next-open execution and EUR account accounting for fair comparison."""
    x = features(df, p)
    warm = max(50, p.bb_period, p.rsi_period)
    start_ts, end_ts = _utc_ts(start), _utc_ts(end)
    first_i = warm if start_ts is None else max(warm, int(x.index.searchsorted(start_ts, side="left")))
    last_i = len(x) - 1 if end_ts is None else min(len(x) - 1, int(x.index.searchsorted(end_ts, side="right")) - 1)
    if last_i <= first_i:
        raise ValueError("Not enough data in requested legacy backtest window.")

    # Legacy nominal is interpreted as EUR notional equal to configured fixed amount.
    lp = Params(**{**asdict(p), "sizing_mode": "fixed_notional"})
    capital = p.initial_capital_eur
    position = 0
    units = 0.0
    entry = stop = take = np.nan
    entry_date = signal_date = None
    pending = None
    trades, eq, dates = [], [], []
    exposure = 0
    hs = p.spread_pips * PIP / 2
    slip = p.slippage_pips_per_side * PIP

    for i in range(first_i, last_i + 1):
        r = x.iloc[i]
        if position == 0 and pending is not None:
            position = int(pending["side"])
            adverse = hs + slip
            entry = float(r.open + adverse if position == 1 else r.open - adverse)
            units = min(lp.fixed_notional_eur, capital * p.max_leverage)
            units = _round_units_down(units, p.lot_step_units)
            sd = max(float(pending["std20"]) * 1.2, PIP)
            stop = entry - sd if position == 1 else entry + sd
            take_dist = max(float(pending["std20"]) * 1.8, PIP)
            take = entry + take_dist if position == 1 else entry - take_dist
            entry_date, signal_date = x.index[i], pending["signal_date"]
            pending = None

        reason = raw_exit = None
        if position:
            exposure += 1
            if position == 1:
                if r.low <= stop: reason, raw_exit = "SL", stop
                elif r.high >= take: reason, raw_exit = "TP", take
            else:
                if r.high >= stop: reason, raw_exit = "SL", stop
                elif r.low <= take: reason, raw_exit = "TP", take
            if reason:
                adverse = hs + slip
                exit_px = float(raw_exit - adverse if position == 1 else raw_exit + adverse)
                trade_pnl = pnl_eur(entry, exit_px, units, position)
                capital += trade_pnl
                lev = units / max(capital - trade_pnl, 1e-12)
                trades.append({"signalDate": str(signal_date), "entry": str(entry_date), "exit": str(x.index[i]),
                               "side": "LONG" if position == 1 else "SHORT", "entryPrice": round(entry, 6),
                               "exitPrice": round(exit_px, 6), "units": units, "lots": round(units/100000,4),
                               "notionalEUR": units, "leverage": round(lev,3), "usesLeverage": bool(lev>1.0001),
                               "appliedRiskPct": None, "profitEUR": round(float(trade_pnl),2), "reason": reason})
                position = 0

        mtm = capital
        if position:
            mark = float(r.close - hs if position == 1 else r.close + hs)
            mtm += pnl_eur(entry, mark, units, position)
        eq.append(mtm); dates.append(x.index[i])

        if position == 0 and pending is None and i < last_i:
            s = legacy_signal_row(r)
            if s and pd.notna(r.bb_std):
                pending = {"side": s, "signal_date": x.index[i], "std20": float(r.bb_std)}

    if position:
        r = x.iloc[last_i]
        adverse = hs + slip
        exit_px = float(r.close - adverse if position == 1 else r.close + adverse)
        trade_pnl = pnl_eur(entry, exit_px, units, position)
        capital += trade_pnl
        lev = units / max(capital - trade_pnl, 1e-12)
        trades.append({"signalDate": str(signal_date), "entry": str(entry_date), "exit": str(x.index[last_i]),
                       "side": "LONG" if position == 1 else "SHORT", "entryPrice": round(entry, 6),
                       "exitPrice": round(exit_px, 6), "units": units, "lots": round(units/100000,4),
                       "notionalEUR": units, "leverage": round(lev,3), "usesLeverage": bool(lev>1.0001),
                       "appliedRiskPct": None, "profitEUR": round(float(trade_pnl),2), "reason": "EOD"})
        eq[-1] = capital

    equity = pd.Series(eq, index=dates, name="equity_eur", dtype=float)
    return {"metrics": _metrics(equity, trades, p.initial_capital_eur, capital, exposure), "trades": trades, "equity": equity, "features": x}


# --------------------------- dashboard snapshot ----------------------------
def current_snapshot(x: pd.DataFrame, p: Params, equity_eur: float | None = None, ml_probability: float | None = None, override_signal: int | None = None) -> dict:
    r = x.iloc[-1]
    s = signal_row(r, p) if override_signal is None else int(override_signal)
    equity = p.initial_capital_eur if equity_eur is None else float(equity_eur)
    if s and pd.notna(r.atr):
        # Current-close entry is only an indicative pre-open sizing estimate.
        indicative_entry = float(r.close)
        plan = position_plan(equity, indicative_entry, s, float(r.atr), r, p, ml_probability)
        indicative_sl, indicative_tp = plan["stopLoss"], plan["takeProfit"]
    else:
        plan = None
        indicative_sl = indicative_tp = None
    confirmations = {
        "rsiLong": bool(pd.notna(r.rsi) and r.rsi > p.rsi_long),
        "rsiShort": bool(pd.notna(r.rsi) and r.rsi < p.rsi_short),
        "aboveSMA50": bool(pd.notna(r.sma50) and r.close > r.sma50),
        "belowSMA50": bool(pd.notna(r.sma50) and r.close < r.sma50),
        "macdPositive": bool(pd.notna(r.macd_hist) and r.macd_hist > 0),
        "macdNegative": bool(pd.notna(r.macd_hist) and r.macd_hist < 0),
    }
    return {
        "date": str(x.index[-1]), "currentPrice": round(float(r.close), 6),
        "rsi": round(float(r.rsi), 2) if pd.notna(r.rsi) else None,
        "sma50": round(float(r.sma50), 6) if pd.notna(r.sma50) else None,
        "macdHist": round(float(r.macd_hist), 7) if pd.notna(r.macd_hist) else None,
        "atr": round(float(r.atr), 7) if pd.notna(r.atr) else None,
        "adx": round(float(r.adx), 2) if pd.notna(r.adx) else None,
        "rv20": round(float(r.rv20), 5) if pd.notna(r.rv20) else None,
        "signal": "BUY (LONG)" if s == 1 else ("SELL (SHORT)" if s == -1 else "NEUTRAL (WAIT)"),
        "stopLoss": indicative_sl, "takeProfit": indicative_tp,
        "levelsAreIndicative": True,
        "execution": "Signal at close; final size/SL/TP recalculated at following daily open.",
        "confirmations": confirmations,
        "positionPlan": plan,
    }


def params_dict(p: Params) -> dict:
    return asdict(p)
