"""Compare raw versus direction-aligned ML features for V3.3.

A LONG and SHORT setup should be economically symmetric. Multiplying directional
features by side lets a linear model learn momentum in trade direction rather
than separately rediscovering LONG/SHORT interactions.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.config import load_params, load_backtest_settings
from src.strategy_engine import fetch_yahoo, load_csv
from src.v33_engine import load_v33_policy
from src.ml_v33 import build_resolved_events, FEATURES


def add_aligned(e):
    z=e.copy()
    for src,dst in [
        ("rsi_centered","rsi_aligned"),("sma_gap","sma_aligned"),("macd_atr","macd_aligned"),
        ("di_spread","di_aligned"),("ema200_gap","ema200_aligned"),("ema50_slope","slope_aligned"),
        ("ret1","ret1_aligned"),("ret5","ret5_aligned"),("ret20","ret20_aligned"),("bb_z","bb_aligned")
    ]:
        z[dst]=z[src]*z["side"]
    return z


ALIGNED_CORE=["rsi_aligned","sma_aligned","macd_aligned","di_aligned","ema200_aligned","slope_aligned","adx","rv20","atr_pct"]
ALIGNED_FULL=ALIGNED_CORE+["ret1_aligned","ret5_aligned","ret20_aligned","bb_aligned"]


def models(seed=17):
    return {
      "logistic":Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=.7,max_iter=3000,random_state=seed))]),
      "random_forest":RandomForestClassifier(n_estimators=350,max_depth=5,min_samples_leaf=12,random_state=seed,n_jobs=-1),
      "hist_gradient_boosting":HistGradientBoostingClassifier(max_depth=3,learning_rate=.05,max_iter=180,min_samples_leaf=20,l2_regularization=1.0,random_state=seed),
    }


def cv(events, features, model_name, folds=5, min_train=120):
    e=events.dropna(subset=features+["label"]).sort_values("signalDate").reset_index(drop=True)
    n=len(e); start=max(min_train,int(n*.45))
    cuts=np.linspace(start,n-20,folds,dtype=int)
    aucs=[]; briers=[]; ns=[]
    for k,cut in enumerate(cuts):
        nxt=cuts[k+1] if k+1<len(cuts) else n
        if nxt<=cut: continue
        test=e.iloc[cut:nxt]; train=e.iloc[:cut]
        if len(test)<15: continue
        train=train[train.outcomeEnd<test.signalDate.iloc[0]]
        if train.label.nunique()<2 or test.label.nunique()<2: continue
        m=models()[model_name]; m.fit(train[features],train.label.astype(int))
        prob=m.predict_proba(test[features])[:,1]
        aucs.append(roc_auc_score(test.label,prob)); briers.append(brier_score_loss(test.label,prob)); ns.append(len(test))
    if not aucs: return None
    base=float(e.iloc[:cuts[-1]].label.mean())
    eval_y=e.iloc[cuts[0]:].label.to_numpy()
    baseline=float(np.mean((eval_y-base)**2))
    brier=float(np.mean(briers)); skill=1-brier/baseline if baseline>0 else -999
    return {"folds":len(aucs),"testEvents":int(sum(ns)),"meanAUC":float(np.mean(aucs)),"medianAUC":float(np.median(aucs)),
            "meanBrier":brier,"baselineBrier":baseline,"brierSkill":float(skill),"score":float(np.mean(aucs)+.25*skill)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--csv"); ap.add_argument("--config",default="config.json")
    ap.add_argument("--out",default="reports/ml_v33_features.json"); ap.add_argument("--csv-out",default="reports/ml_v33_features.csv")
    args=ap.parse_args()
    cfg=ROOT/args.config; p=load_params(cfg); pol=load_v33_policy(cfg); bc=load_backtest_settings(cfg)
    df=load_csv(args.csv) if args.csv else fetch_yahoo("EURUSD=X",period=bc.get("downloadHistory","10y"))
    ev=add_aligned(build_resolved_events(df,p,pol))
    cutoff=pd.Timestamp("2026-01-01",tz="UTC"); pre=ev[ev.outcomeEnd<cutoff].copy()
    sets={"raw_full":FEATURES,"aligned_core":ALIGNED_CORE,"aligned_full":ALIGNED_FULL}
    rows=[]
    for setname,featureset in sets.items():
      for m in models():
        q=cv(pre,featureset,m)
        if q: rows.append({"featureSet":setname,"model":m,**q})
    tab=pd.DataFrame(rows).sort_values(["score","meanAUC"],ascending=False).reset_index(drop=True)
    outcsv=ROOT/args.csv_out; outcsv.parent.mkdir(parents=True,exist_ok=True); tab.to_csv(outcsv,index=False)
    report={"label":"TP-before-SL, unresolved censored, pre-2026 only","events":int(len(pre)),"positiveRatePct":float(pre.label.mean()*100),
            "comparison":tab.to_dict(orient="records"),"promotionAutomatic":False}
    (ROOT/args.out).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(tab.to_string(index=False))


if __name__=="__main__": main()
