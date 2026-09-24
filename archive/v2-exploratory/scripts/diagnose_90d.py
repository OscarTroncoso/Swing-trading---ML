"""Reproduce the known 90-close diagnostic and explain why +1.99% is not directly a 90-day strategy return.

The first 50 observations are required for SMA50 warm-up, so only 40 bars are
tradable in this archived close-only diagnostic. This script is intentionally
kept separate from the OHLC production backtester.
"""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
archive_script = ROOT/'archive/v2-exploratory/robust_backtest_90d.py'
spec=importlib.util.spec_from_file_location('archived_robust',archive_script)
mod=importlib.util.module_from_spec(spec); sys.modules['archived_robust']=mod; spec.loader.exec_module(mod)

df,_=mod.load_close_data(ROOT/'archive/v1/data-v1.json')
base=mod.backtest_challenger(df,'momentum',spread_pips=.8,stop_mult=1.5,tp_mult=2.2,max_holding=5)
rows=[]
for risk in [None,.0025,.005,.0075,.01,.015,.02]:
    r=mod.backtest_challenger(df,'momentum',spread_pips=.8,stop_mult=1.5,tp_mult=2.2,max_holding=5,risk_pct=risk)
    rows.append({
        'sizing':'fixed_10k' if risk is None else f'risk_{risk*100:.2f}pct',
        'risk_pct':None if risk is None else risk*100,
        'return_pct':r['return_pct'],'max_drawdown_pct':r['max_drawdown_pct'],
        'trades':r['trades'],'win_rate_pct':r['win_rate_pct'],'profit_factor':r['profit_factor'],
        'sharpe':r['sharpe'],'exposure_pct':r['exposure_pct'],
    })

reasons=pd.DataFrame([{'reason':t['reason'],'pnl':t['pnl']} for t in base['trades_detail']])
reason_summary=reasons.groupby('reason').agg(trades=('pnl','size'),pnl=('pnl','sum'),avg_pnl=('pnl','mean')).reset_index()
reason_summary['pnl_share_pct']=reason_summary.pnl/reasons.pnl.sum()*100

payload={
    'archived_data': {
        'observations':len(df),'start':str(df.index.min().date()),'end':str(df.index.max().date()),
        'warmup_bars':50,'tradable_bars':len(df)-50,'tradable_start':str(df.index[50].date()),
    },
    'baseline_momentum': {k:v for k,v in base.items() if k!='trades_detail'},
    'trades': [{**t,'entry':str(t['entry'].date()),'exit':str(t['exit'].date())} for t in base['trades_detail']],
    'exit_contribution':reason_summary.to_dict(orient='records'),
    'sizing_sensitivity':rows,
    'interpretation': [
        'The quoted +1.99% comes from only 40 tradable daily bars after the SMA50 warm-up, not from 90 fully tradable sessions.',
        'Two TP trades generated about 80% of total P&L in this archived sample.',
        'Four time exits generated the remaining ~20%; alternative close-only trailing/breakeven/fade screens did not beat the 5-bar baseline in the archived sample.',
        'Risk sizing increases return by using more notional exposure on the exact same signals; it does not prove a stronger forecasting edge.',
    ]
}
(ROOT/'reports/diagnosis_90d.json').write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8')
pd.DataFrame(rows).to_csv(ROOT/'reports/sizing_sensitivity_90d.csv',index=False)
reason_summary.to_csv(ROOT/'reports/exit_contribution_90d.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
print('\nExit contribution:')
print(reason_summary.to_string(index=False))
