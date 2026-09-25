import numpy as np
import pandas as pd

from src.config import load_params
from src.v33_engine import load_v33_policy
from src.portfolio_engine import PortfolioPolicy, backtest_portfolio


def synthetic(n=1500,seed=91):
    rng=np.random.default_rng(seed)
    r=0.00005+rng.normal(0,0.003,n)
    c=1.08*np.exp(np.cumsum(r)); o=np.r_[c[0],c[:-1]]
    span=np.maximum(0.0012,np.abs(c-o)*0.7)
    h=np.maximum(o,c)+span; l=np.minimum(o,c)-span
    idx=pd.date_range("2020-01-01",periods=n,freq="B",tz="UTC")
    return pd.DataFrame({"open":o,"high":h,"low":l,"close":c},index=idx)


def test_portfolio_has_no_time_exits():
    p=load_params("config.json"); pol=load_v33_policy("config.json")
    pp=PortfolioPolicy(max_concurrent_positions=3,min_bars_between_entries=2)
    b=backtest_portfolio(synthetic(),p,pol,pp,start="2022-01-01")
    assert all(t["reason"] in ("TP","SL","SL_GAP") for t in b["trades"])
    assert b["metrics"]["timeExitTrades"]==0


def test_portfolio_risk_and_exposure_caps_are_respected_at_end():
    p=load_params("config.json"); pol=load_v33_policy("config.json")
    pp=PortfolioPolicy(max_concurrent_positions=3,max_portfolio_risk_pct=.02,max_gross_exposure_multiple=3.0)
    b=backtest_portfolio(synthetic(seed=92),p,pol,pp,start="2022-01-01")
    assert len(b["openPositions"])<=pp.max_concurrent_positions
    total_risk=sum(x["riskAtStopEUR"] for x in b["openPositions"])
    total_exp=sum(x["grossExposureEUR"] for x in b["openPositions"])
    eq=b["metrics"]["finalCapitalEUR"]
    assert total_risk<=eq*pp.max_portfolio_risk_pct+2.0
    assert total_exp<=eq*pp.max_gross_exposure_multiple+5.0


def test_portfolio_reports_eur_stake_not_units():
    p=load_params("config.json"); pol=load_v33_policy("config.json")
    b=backtest_portfolio(synthetic(seed=93),p,pol,PortfolioPolicy(),start="2022-01-01")
    for t in b["trades"]:
        assert "stakeEUR" in t and "grossExposureEUR" in t and "leverage" in t
        assert "units" not in t
