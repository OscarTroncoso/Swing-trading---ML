"""Portfolio extension for V3.3.

Allows several independent EUR-stake trades to coexist so a long-lived TP/SL
position does not block later valid signals. Every tranche keeps its own entry,
SL, TP, EUR stake and leverage. Portfolio-level risk/exposure caps prevent the
absence of a time exit from turning into uncontrolled stacking.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
import pandas as pd

from .strategy_engine import PIP, Params, _commission, _utc_ts, features, pnl_eur, technical_candidate_row
from .v33_engine import V33Policy, position_plan_v33, should_veto_countertrend


@dataclass(frozen=True)
class PortfolioPolicy:
    max_concurrent_positions: int = 3
    min_bars_between_entries: int = 3
    max_portfolio_risk_pct: float = 0.03
    max_gross_exposure_multiple: float = 4.0
    same_direction_only: bool = True


def _portfolio_metrics(eq: pd.Series, trades: list[dict], initial: float, realized: float,
                       open_positions: list[dict], exposure_bars: int, skipped: dict) -> dict:
    rets=eq.pct_change().fillna(0.0)
    dd=eq/eq.cummax()-1
    pnl=np.array([t["profitEUR"] for t in trades],float) if trades else np.array([],float)
    wins=pnl[pnl>0]; losses=-pnl[pnl<0]
    vol=rets.std(ddof=0); downside=rets[rets<0].std(ddof=0)
    pf=wins.sum()/losses.sum() if losses.sum()>0 else (999.0 if wins.sum()>0 else 0.0)
    final=float(eq.iloc[-1]) if len(eq) else realized
    return {
        "accountCurrency":"EUR","initialCapitalEUR":round(initial,2),"realizedCapitalEUR":round(realized,2),
        "finalCapitalEUR":round(final,2),"totalReturn":round((final/initial-1)*100,4),
        "winRate":round((pnl>0).mean()*100,2) if len(pnl) else 0.0,
        "maxDrawdown":round(float(-dd.min()*100),4) if len(dd) else 0.0,
        "profitFactor":round(float(pf),4),"totalTrades":int(len(trades)),"openPositions":int(len(open_positions)),
        "sharpe":round(float(rets.mean()/vol*math.sqrt(252)),4) if vol>0 else 0.0,
        "sortino":round(float(rets.mean()/downside*math.sqrt(252)),4) if pd.notna(downside) and downside>0 else 0.0,
        "avgTradeEUR":round(float(pnl.mean()),2) if len(pnl) else 0.0,
        "exposurePct":round(exposure_bars/max(1,len(eq))*100,2),
        "avgStakeEUR":round(float(np.mean([t.get("stakeEUR",0) for t in trades])),2) if trades else 0.0,
        "avgLeverage":round(float(np.mean([t.get("leverage",0) for t in trades])),3) if trades else 0.0,
        "maxLeverageUsed":int(max([t.get("leverage",0) for t in trades],default=0)),
        "takeProfitTrades":int(sum(t.get("reason")=="TP" for t in trades)),
        "stoppedTrades":int(sum(t.get("reason") in ("SL","SL_GAP") for t in trades)),
        "timeExitTrades":0,
        "signalsSkippedPositionCap":int(skipped.get("position_cap",0)),
        "signalsSkippedCooldown":int(skipped.get("cooldown",0)),
        "signalsSkippedPortfolioRisk":int(skipped.get("portfolio_risk",0)),
        "signalsSkippedExposure":int(skipped.get("exposure",0)),
        "signalsSkippedOppositeBook":int(skipped.get("opposite_book",0)),
    }


def backtest_portfolio(df: pd.DataFrame, p: Params, policy: V33Policy, pp: PortfolioPolicy,
                       start=None, end=None, ml_probabilities: Optional[dict]=None) -> dict:
    x=features(df,p)
    warm=max(p.trend_ema_slow,p.sma_trend,p.bb_period,p.atr_period,p.adx_period,30)
    start_ts,end_ts=_utc_ts(start),_utc_ts(end)
    first_i=warm if start_ts is None else max(warm,int(x.index.searchsorted(start_ts,side="left")))
    last_i=len(x)-1 if end_ts is None else min(len(x)-1,int(x.index.searchsorted(end_ts,side="right"))-1)
    if last_i<=first_i: raise ValueError("Not enough data in requested window")

    capital=float(policy.initial_capital_eur)
    positions=[]; pending=[]; trades=[]; eq=[]; dates=[]
    half=p.spread_pips*PIP/2; slip=p.slippage_pips_per_side*PIP
    last_entry_i=-10**9; exposure_bars=0
    skipped={"position_cap":0,"cooldown":0,"portfolio_risk":0,"exposure":0,"opposite_book":0}

    for i in range(first_i,last_i+1):
        row=x.iloc[i]

        # Execute yesterday's accepted signals at today's open, if portfolio limits still allow them.
        queued=pending; pending=[]
        for q in queued:
            if len(positions)>=pp.max_concurrent_positions:
                skipped["position_cap"]+=1; continue
            if i-last_entry_i<pp.min_bars_between_entries:
                skipped["cooldown"]+=1; continue
            side=int(q["side"])
            if pp.same_direction_only and positions and any(pos["side"]!=side for pos in positions):
                skipped["opposite_book"]+=1; continue
            entry=float(row.open+half+slip if side==1 else row.open-half-slip)
            prob=None if ml_probabilities is None else ml_probabilities.get(pd.Timestamp(q["signalDate"]))
            plan=position_plan_v33(capital,entry,side,float(q["atr"]),q["row"],p,policy,prob)
            if not plan.get("valid"): continue

            existing_risk=sum(float(pos["risk"]) for pos in positions)
            existing_exposure=sum(float(pos["exposure"]) for pos in positions)
            new_risk=float(plan["riskAtStopEUR"]); new_exp=float(plan["grossExposureEUR"])
            if existing_risk+new_risk > capital*pp.max_portfolio_risk_pct+1e-9:
                skipped["portfolio_risk"]+=1; continue
            if existing_exposure+new_exp > capital*pp.max_gross_exposure_multiple+1e-9:
                skipped["exposure"]+=1; continue

            capital-=_commission(new_exp,p)
            positions.append({
                "side":side,"signalDate":q["signalDate"],"entryDate":x.index[i],"entry":entry,
                "stop":float(plan["stopLoss"]),"take":float(plan["takeProfit"]),"stake":float(plan["stakeEUR"]),
                "leverage":int(plan["leverage"]),"exposure":new_exp,"risk":new_risk,"plan":plan,
                "fav":entry,"adv":entry,"mlProbability":prob,
            })
            last_entry_i=i

        # Manage every tranche independently. No fixed-day exit exists.
        survivors=[]
        for pos in positions:
            side=pos["side"]; reason=None; raw_exit=None
            if side==1:
                pos["fav"]=max(float(pos["fav"]),float(row.high)); pos["adv"]=min(float(pos["adv"]),float(row.low))
                if row.open<=pos["stop"]: reason,raw_exit="SL_GAP",float(row.open)
                elif row.low<=pos["stop"]: reason,raw_exit="SL",pos["stop"]
                elif row.high>=pos["take"]: reason,raw_exit="TP",pos["take"]
            else:
                pos["fav"]=min(float(pos["fav"]),float(row.low)); pos["adv"]=max(float(pos["adv"]),float(row.high))
                if row.open>=pos["stop"]: reason,raw_exit="SL_GAP",float(row.open)
                elif row.high>=pos["stop"]: reason,raw_exit="SL",pos["stop"]
                elif row.low<=pos["take"]: reason,raw_exit="TP",pos["take"]

            if reason:
                exit_px=float(raw_exit-half-slip if side==1 else raw_exit+half+slip)
                pnl=pnl_eur(pos["entry"],exit_px,pos["exposure"],side)-_commission(pos["exposure"],p)
                capital+=pnl
                initial_risk_price=abs(pos["entry"]-pos["stop"])
                mfe=max(0.0,(pos["fav"]-pos["entry"])*side)/max(initial_risk_price,1e-12)
                mae=max(0.0,-(pos["adv"]-pos["entry"])*side)/max(initial_risk_price,1e-12)
                ctx=pos["plan"]["trend"]
                trades.append({
                    "signalDate":str(pos["signalDate"]),"entry":str(pos["entryDate"]),"exit":str(x.index[i]),
                    "side":"LONG" if side==1 else "SHORT","entryPrice":round(pos["entry"],6),"exitPrice":round(exit_px,6),
                    "stopLoss":round(pos["stop"],6),"takeProfit":round(pos["take"],6),"stakeEUR":round(pos["stake"],2),
                    "grossExposureEUR":round(pos["exposure"],2),"leverage":pos["leverage"],"initialRiskEUR":round(pos["risk"],2),
                    "profitEUR":round(float(pnl),2),"rMultiple":round(float(pnl/max(pos["risk"],1e-12)),3),
                    "mfeR":round(float(mfe),3),"maeR":round(float(mae),3),"reason":reason,
                    "trendScore":int(ctx.get("score",0)),"trendRegime":ctx.get("regime"),
                    "mlProbability":None if pos["mlProbability"] is None else round(float(pos["mlProbability"]),4),
                })
            else:
                survivors.append(pos)
        positions=survivors
        if positions: exposure_bars+=1

        mtm=capital
        for pos in positions:
            mark=float(row.close-half if pos["side"]==1 else row.close+half)
            mtm+=pnl_eur(pos["entry"],mark,pos["exposure"],pos["side"])
        eq.append(mtm); dates.append(x.index[i])

        # Candidate is generated after today's completed close and may fill next open.
        if i<last_i:
            side=technical_candidate_row(row,p)
            if side and not should_veto_countertrend(row,side,p,policy):
                pending.append({"side":int(side),"signalDate":x.index[i],"atr":float(row.atr),"row":row.copy()})

    open_positions=[]
    row=x.iloc[last_i]
    for pos in positions:
        mark=float(row.close-half if pos["side"]==1 else row.close+half)
        open_positions.append({
            "signalDate":str(pos["signalDate"]),"entry":str(pos["entryDate"]),
            "side":"LONG" if pos["side"]==1 else "SHORT","entryPrice":round(pos["entry"],6),"currentPrice":round(mark,6),
            "stopLoss":round(pos["stop"],6),"takeProfit":round(pos["take"],6),"stakeEUR":round(pos["stake"],2),
            "grossExposureEUR":round(pos["exposure"],2),"leverage":pos["leverage"],"riskAtStopEUR":round(pos["risk"],2),
            "unrealizedPnLEUR":round(float(pnl_eur(pos["entry"],mark,pos["exposure"],pos["side"])),2),
        })

    equity=pd.Series(eq,index=dates,name="equity_eur",dtype=float)
    metrics=_portfolio_metrics(equity,trades,policy.initial_capital_eur,capital,open_positions,exposure_bars,skipped)
    return {"metrics":metrics,"trades":trades,"openPositions":open_positions,"equity":equity,"features":x,"skippedSignals":skipped}
