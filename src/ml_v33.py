"""V3.3 ML research: predict whether TP is hit before SL, with no time-exit label.

Model families are compared chronologically. ML is a soft sizing input only when
out-of-sample validation is demonstrably better than random.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .strategy_engine import Params, features, technical_candidate_row
from .v33_engine import V33Policy, _stop_target


FEATURES = [
    "rsi_centered", "sma_gap", "macd_atr", "adx", "di_spread",
    "ema200_gap", "ema50_slope", "rv20", "atr_pct", "ret1", "ret5",
    "ret20", "bb_z", "side",
]


@dataclass(frozen=True)
class ML33Params:
    min_train_events: int = 120
    max_train_years: int = 8
    min_validation_auc: float = 0.52
    retrain_every_bars: int = 60
    random_state: int = 17


def feature_table(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    x = features(df, p).copy()
    x["rsi_centered"] = (x.rsi - 50.0) / 50.0
    x["di_spread"] = (x.plus_di - x.minus_di) / 100.0
    return x


def _row_features(r: pd.Series, side: int) -> dict:
    return {
        "rsi_centered": float((r.rsi - 50.0) / 50.0),
        "sma_gap": float(r.sma_gap),
        "macd_atr": float(r.macd_atr),
        "adx": float(r.adx),
        "di_spread": float((r.plus_di - r.minus_di) / 100.0),
        "ema200_gap": float(r.ema200_gap),
        "ema50_slope": float(r.ema50_slope),
        "rv20": float(r.rv20),
        "atr_pct": float(r.atr_pct),
        "ret1": float(r.ret1),
        "ret5": float(r.ret5),
        "ret20": float(r.ret20),
        "bb_z": float(r.bb_z),
        "side": float(side),
    }


def build_resolved_events(df: pd.DataFrame, p: Params, policy: V33Policy) -> pd.DataFrame:
    """Label only signals that eventually resolve at TP or SL.

    There is deliberately no five-day/time-exit label. If both barriers are
    touched in one daily bar, SL is assigned first (conservative ordering).
    """
    x = feature_table(df, p)
    required = FEATURES[:-1] + ["atr"]
    warm = max(p.trend_ema_slow, p.sma_trend, p.bb_period, p.atr_period, p.adx_period, 30)
    rows = []
    for i in range(warm, len(x) - 1):
        r = x.iloc[i]
        if any(pd.isna(r.get(c, np.nan)) for c in required):
            continue
        side = technical_candidate_row(r, p)
        if not side:
            continue
        entry_i = i + 1
        entry = float(x.iloc[entry_i].open)
        levels = _stop_target(entry, side, float(r.atr), r, policy)
        stop = float(levels["stopLoss"])
        take = float(levels["takeProfit"])
        outcome = None
        outcome_i = None
        for j in range(entry_i, len(x)):
            rr = x.iloc[j]
            if side == 1:
                stop_hit = rr.low <= stop
                take_hit = rr.high >= take
            else:
                stop_hit = rr.high >= stop
                take_hit = rr.low <= take
            if stop_hit:
                outcome, outcome_i = 0, j
                break
            if take_hit:
                outcome, outcome_i = 1, j
                break
        if outcome is None:
            # Unresolved events are censored, not forced into a label.
            continue
        rec = {
            "signalDate": x.index[i],
            "entryDate": x.index[entry_i],
            "outcomeEnd": x.index[outcome_i],
            "label": int(outcome),
            "side": int(side),
        }
        rec.update(_row_features(r, side))
        rows.append(rec)
    return pd.DataFrame(rows)


def _models(seed: int):
    return {
        "logistic": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.7, max_iter=3000, random_state=seed)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=350, max_depth=5, min_samples_leaf=12,
            random_state=seed, n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=180,
            min_samples_leaf=20, l2_regularization=1.0, random_state=seed,
        ),
    }


def chronological_model_comparison(events: pd.DataFrame, params: ML33Params, folds: int = 5) -> pd.DataFrame:
    e = events.dropna(subset=FEATURES + ["label"]).sort_values("signalDate").reset_index(drop=True)
    if len(e) < params.min_train_events + 40:
        return pd.DataFrame()
    n = len(e)
    start = max(params.min_train_events, int(n * 0.45))
    cut_points = np.linspace(start, n - 20, folds, dtype=int)
    rows = []
    for name, proto in _models(params.random_state).items():
        aucs, briers, counts = [], [], []
        for k, cut in enumerate(cut_points):
            nxt = cut_points[k + 1] if k + 1 < len(cut_points) else n
            if nxt <= cut:
                continue
            test = e.iloc[cut:nxt]
            test_start = test.signalDate.iloc[0]
            # Purged chronological fold: a training event is admissible only if
            # its TP/SL outcome was already known before the test block starts.
            train = e.iloc[:cut]
            train = train[train.outcomeEnd < test_start]
            if train.label.nunique() < 2 or test.label.nunique() < 2 or len(test) < 15:
                continue
            model = _models(params.random_state)[name]
            model.fit(train[FEATURES], train.label.astype(int))
            prob = model.predict_proba(test[FEATURES])[:, 1]
            aucs.append(roc_auc_score(test.label, prob))
            briers.append(brier_score_loss(test.label, prob))
            counts.append(len(test))
        if aucs:
            # Baseline Brier from the historical class prevalence. A probability
            # model used for sizing should beat this naive forecast, not only rank.
            base_rate = float(e.iloc[:cut_points[-1]].label.mean())
            baseline_brier = float(np.mean([(base_rate - int(y))**2 for y in e.iloc[cut_points[0]:].label]))
            mean_brier = float(np.mean(briers))
            brier_skill = 1.0 - mean_brier / baseline_brier if baseline_brier > 0 else -np.inf
            rows.append({
                "model": name,
                "folds": len(aucs),
                "testEvents": int(sum(counts)),
                "meanAUC": float(np.mean(aucs)),
                "medianAUC": float(np.median(aucs)),
                "meanBrier": mean_brier,
                "baselineBrier": baseline_brier,
                "brierSkill": float(brier_skill),
                "score": float(np.mean(aucs) + 0.25*brier_skill),
            })
    return pd.DataFrame(rows).sort_values(["score", "meanAUC"], ascending=[False, False]).reset_index(drop=True)


def select_model(events: pd.DataFrame, asof, params: ML33Params):
    asof = pd.Timestamp(asof)
    train = events[events.outcomeEnd < asof].copy()
    if params.max_train_years > 0:
        train = train[train.signalDate >= asof - pd.DateOffset(years=params.max_train_years)]
    train = train.dropna(subset=FEATURES + ["label"]).sort_values("signalDate")
    if len(train) < params.min_train_events or train.label.nunique() < 2:
        return None, {"status": "INSUFFICIENT_HISTORY", "trainingEvents": int(len(train))}
    cmp = chronological_model_comparison(train, params, folds=4)
    if cmp.empty:
        return None, {"status": "NO_VALID_MODEL_COMPARISON", "trainingEvents": int(len(train))}
    best = cmp.iloc[0]
    healthy = bool(best.meanAUC >= params.min_validation_auc and best.brierSkill > 0)
    if not healthy:
        return None, {
            "status": "MODEL_HEALTH_GATE",
            "trainingEvents": int(len(train)),
            "selectedModel": str(best.model),
            "validationAUC": float(best.meanAUC),
            "validationBrier": float(best.meanBrier),
            "brierSkill": float(best.brierSkill),
        }
    model = _models(params.random_state)[str(best.model)]
    model.fit(train[FEATURES], train.label.astype(int))
    return model, {
        "status": "OK",
        "trainingEvents": int(len(train)),
        "selectedModel": str(best.model),
        "validationAUC": float(best.meanAUC),
        "validationBrier": float(best.meanBrier),
        "brierSkill": float(best.brierSkill),
        "comparison": cmp.to_dict(orient="records"),
    }


def expanding_probabilities(
    df: pd.DataFrame,
    p: Params,
    policy: V33Policy,
    params: ML33Params,
    start,
    end=None,
) -> tuple[dict, dict]:
    x = feature_table(df, p)
    events = build_resolved_events(df, p, policy)
    start_ts = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start).tz_convert("UTC")
    end_ts = x.index[-1] if end is None else (pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).tz_convert("UTC"))
    probs = {}
    logs = []
    model = None
    meta = {}
    last_fit_i = -10**9
    for i, (dt, r) in enumerate(x.loc[(x.index >= start_ts) & (x.index <= end_ts)].iterrows()):
        side = technical_candidate_row(r, p)
        if not side or any(pd.isna(r.get(c, np.nan)) for c in FEATURES if c != "side"):
            continue
        if model is None or i - last_fit_i >= params.retrain_every_bars:
            model, meta = select_model(events, dt, params)
            last_fit_i = i
        prob = None
        if model is not None:
            row_df = pd.DataFrame([_row_features(r, side)], index=[dt])
            prob = float(model.predict_proba(row_df[FEATURES])[:, 1][0])
            probs[pd.Timestamp(dt)] = prob
        logs.append({
            "signalDate": str(dt), "side": int(side), "probability": prob,
            "modelStatus": meta.get("status"), "model": meta.get("selectedModel"),
            "validationAUC": meta.get("validationAUC"), "trainingEvents": meta.get("trainingEvents", 0),
        })
    return probs, {"events": events, "predictions": logs, "lastModelMeta": meta}
