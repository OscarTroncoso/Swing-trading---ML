import numpy as np
import pandas as pd
from dataclasses import replace

from src.config import load_params
from src.v33_engine import load_v33_policy, backtest_v33


def synthetic(n=1200,seed=111):
    rng=np.random.default_rng(seed); r=rng.normal(0,0.003,n)
    c=1.08*np.exp(np.cumsum(r)); o=np.r_[c[0],c[:-1]]
    span=np.maximum(0.0012,np.abs(c-o)*0.7)
    h=np.maximum(o,c)+span; l=np.minimum(o,c)-span
    idx=pd.date_range("2020-01-01",periods=n,freq="B",tz="UTC")
    return pd.DataFrame({"open":o,"high":h,"low":l,"close":c},index=idx)


def test_dynamic_stagnation_never_creates_time_exit():
    p=load_params("config.json"); base=load_v33_policy("config.json")
    pol=replace(base,max_holding_bars=0,stagnation_bars=1,stagnation_min_mfe_r=100.0,
                stagnation_min_close_r=100.0,stagnation_require_weak_thesis=False)
    b=backtest_v33(synthetic(),p,pol,start="2022-01-01")
    assert all(t["reason"]!="TIME" for t in b["trades"])
    if b["trades"]:
        assert any(t["reason"]=="STAGNATION" for t in b["trades"])
