"""V4 leakage-controlled ML advisory layer.

This module does not decide direction and does not hard-filter the trading strategy.
It compares two deliberately small models on chronological validation:
- regularized logistic regression
- histogram gradient boosting

The best healthy model only scales the risk budget slightly. If validation AUC is
below the configured health threshold, ML abstains and V4 behaves exactly as the
non-ML strategy.

Labels are barrier outcomes: a technical candidate is positive only if its V4 TP
is reached before its V4 SL within the research label horizon. Unresolved events
are excluded rather than forced into a class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .strategy_engine import PIP, features, technical_candidate_row
from .strategy_v4 import V4Params, stop_target_v4, thesis_invalidated_v4


V4_FEATURES = [
    "rsi_centered",
    "signed_sma_gap",
    "signed_macd_atr",
    "rv20",
    "adx",
    "signed_dmi",
    "signed_ret5",
    "signed_ret20",
    "atr_pct",
    "signed_bb_z",
    "signed_ema200_gap",
    "signed_ema50_slope",
]


@dataclass(frozen=True)
class MLV4Params:
    retrain_every_bars: int = 60
    min_train_events: int = 120
    validation_fraction: float = 0.20
    max_train_years: int = 8
    min_validation_auc: float = 0.52
    label_horizon_bars: int = 60
    logistic_c: float = 0.7


def _row_features(r: pd.Series, side: int) -> dict:
    dmi = (float(r.plus_di) - float(r.minus_di)) if pd.notna(r.plus_di) and pd.notna(r.minus_di) else np.nan
    return {
        "rsi_centered": (float(r.rsi) - 50.0) / 50.0,
        "signed_sma_gap": side * float(r.sma_gap),
        "signed_macd_atr": side * float(r.macd_atr),
        "rv20": float(r.rv20),
        "adx": float(r.adx),
        "signed_dmi": side * dmi,
        "signed_ret5": side * float(r.ret5),
        "signed_ret20": side * float(r.ret20),
        "atr_pct": float(r.atr_pct),
        "signed_bb_z": side * float(r.bb_z),
        "signed_ema200_gap": side * float(r.ema200_gap),
        "signed_ema50_slope": side * float(r.ema50_slope),
    }


def build_v4_events(df: pd.DataFrame, p: V4Params, mlp: MLV4Params) -> pd.DataFrame:
    """Create past-only candidate labels using the same V4 exit thesis.

    A label becomes known only when the simulated event actually exits by SL, TP,
    gap, or next-open thesis invalidation. Events unresolved inside the research
    horizon are omitted rather than forced into a class.
    """
    from .strategy_engine import pnl_eur

    x = features(df, p)
    warm = max(p.trend_ema_slow, p.sma_trend, p.bb_period, p.atr_period, p.adx_period, 30)
    adverse = p.spread_pips * PIP / 2 + p.slippage_pips_per_side * PIP
    rows = []

    for i in range(warm, len(x) - 1):
        r = x.iloc[i]
        side = technical_candidate_row(r, p)
        if side == 0:
            continue
        feat = _row_features(r, side)
        if any(pd.isna(v) or not np.isfinite(v) for v in feat.values()):
            continue

        entry_i = i + 1
        entry_row = x.iloc[entry_i]
        entry = float(entry_row.open + adverse if side == 1 else entry_row.open - adverse)
        levels = stop_target_v4(entry, side, float(r.atr), r, p)
        if not levels.get("valid"):
            continue
        stop = float(levels["stopLoss"])
        take = float(levels["takeProfit"])
        last = min(len(x) - 1, entry_i + mlp.label_horizon_bars)

        exit_px = None
        outcome_end = None
        outcome_reason = None
        pending_thesis = False

        for j in range(entry_i, last + 1):
            rr = x.iloc[j]
            o, h, l = float(rr.open), float(rr.high), float(rr.low)

            if pending_thesis:
                # Overnight barriers have priority over discretionary next-open exit.
                if side == 1 and o <= stop:
                    exit_px = o - adverse; outcome_reason = "SL_GAP"
                elif side == -1 and o >= stop:
                    exit_px = o + adverse; outcome_reason = "SL_GAP"
                elif side == 1 and o >= take:
                    exit_px = take - adverse; outcome_reason = "TP_GAP"
                elif side == -1 and o <= take:
                    exit_px = take + adverse; outcome_reason = "TP_GAP"
                else:
                    exit_px = o - adverse if side == 1 else o + adverse
                    outcome_reason = "THESIS_INVALIDATED"
                outcome_end = x.index[j]
                break

            # Conservative daily OHLC ordering: SL first when both barriers are inside a bar.
            if side == 1:
                if o <= stop:
                    exit_px = o - adverse; outcome_reason = "SL_GAP"; outcome_end = x.index[j]; break
                if l <= stop:
                    exit_px = stop - adverse; outcome_reason = "SL"; outcome_end = x.index[j]; break
                if h >= take:
                    exit_px = take - adverse; outcome_reason = "TP"; outcome_end = x.index[j]; break
            else:
                if o >= stop:
                    exit_px = o + adverse; outcome_reason = "SL_GAP"; outcome_end = x.index[j]; break
                if h >= stop:
                    exit_px = stop + adverse; outcome_reason = "SL"; outcome_end = x.index[j]; break
                if l <= take:
                    exit_px = take + adverse; outcome_reason = "TP"; outcome_end = x.index[j]; break

            if j < last:
                pending_thesis = thesis_invalidated_v4(rr, side, p, holding_bars=max(0, j-entry_i))

        if exit_px is None or outcome_end is None:
            continue

        unit_pnl = float(pnl_eur(entry, float(exit_px), 1.0, side))
        rec = {
            "signalDate": x.index[i],
            "entryDate": x.index[entry_i],
            "outcomeEnd": outcome_end,
            "side": int(side),
            "label": int(unit_pnl > 0),
            "unitPnlEUR": unit_pnl,
            "reason": outcome_reason,
        }
        rec.update(feat)
        rows.append(rec)

    return pd.DataFrame(rows)


def _balanced_weights(y: pd.Series) -> np.ndarray:
    n = len(y)
    pos = max(1, int((y == 1).sum()))
    neg = max(1, int((y == 0).sum()))
    return np.where(y.to_numpy() == 1, n / (2 * pos), n / (2 * neg))


def _fit_candidates(X_train: pd.DataFrame, y_train: pd.Series):
    logistic = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.7, class_weight="balanced", max_iter=2000, random_state=17)),
    ])
    logistic.fit(X_train, y_train)

    hist = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=180,
        max_depth=3,
        min_samples_leaf=25,
        l2_regularization=1.0,
        random_state=17,
    )
    hist.fit(X_train, y_train, sample_weight=_balanced_weights(y_train))
    return {"logistic": logistic, "hist_gradient_boosting": hist}


def fit_best_v4_model(events: pd.DataFrame, asof, mlp: MLV4Params):
    asof = pd.Timestamp(asof)
    train = events[events.outcomeEnd < asof].copy().sort_values("signalDate")
    if mlp.max_train_years > 0:
        train = train[train.signalDate >= asof - pd.DateOffset(years=mlp.max_train_years)]
    train = train.dropna(subset=V4_FEATURES + ["label"])

    if len(train) < mlp.min_train_events or train.label.nunique() < 2:
        return None, {
            "healthy": False,
            "status": "INSUFFICIENT_HISTORY",
            "trainingEvents": int(len(train)),
        }

    split = max(60, int(len(train) * (1 - mlp.validation_fraction)))
    split = min(split, len(train) - 30)
    fit = train.iloc[:split]
    val = train.iloc[split:]
    if len(val) < 30 or fit.label.nunique() < 2 or val.label.nunique() < 2:
        return None, {
            "healthy": False,
            "status": "INSUFFICIENT_TEMPORAL_VALIDATION",
            "trainingEvents": int(len(train)),
        }

    models = _fit_candidates(fit[V4_FEATURES], fit.label.astype(int))
    leaderboard = []
    for name, model in models.items():
        prob = model.predict_proba(val[V4_FEATURES])[:, 1]
        auc = float(roc_auc_score(val.label, prob))
        brier = float(brier_score_loss(val.label, prob))
        leaderboard.append({"model": name, "auc": auc, "brier": brier})

    leaderboard.sort(key=lambda z: (-z["auc"], z["brier"]))
    best = leaderboard[0]
    healthy = bool(best["auc"] >= mlp.min_validation_auc)
    meta = {
        "healthy": healthy,
        "status": "OK" if healthy else "MODEL_HEALTH_GATE",
        "selectedModel": best["model"],
        "validationAUC": round(best["auc"], 4),
        "validationBrier": round(best["brier"], 4),
        "validationEvents": int(len(val)),
        "trainingEvents": int(len(train)),
        "positiveRatePct": round(float(train.label.mean() * 100), 2),
        "leaderboard": leaderboard,
    }
    if not healthy:
        return None, meta

    # Refit selected architecture on all events already resolved before asof.
    if best["model"] == "logistic":
        final_model = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=mlp.logistic_c, class_weight="balanced", max_iter=2000, random_state=17)),
        ])
        final_model.fit(train[V4_FEATURES], train.label.astype(int))
    else:
        final_model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=180,
            max_depth=3,
            min_samples_leaf=25,
            l2_regularization=1.0,
            random_state=17,
        )
        final_model.fit(train[V4_FEATURES], train.label.astype(int), sample_weight=_balanced_weights(train.label))

    return final_model, meta


def build_probability_maps(
    df: pd.DataFrame,
    p: V4Params,
    mlp: MLV4Params,
    start=None,
    end=None,
) -> tuple[dict, dict, list[dict]]:
    x = features(df, p)
    events = build_v4_events(df, p, mlp)
    warm = max(p.trend_ema_slow, p.sma_trend, p.bb_period, p.atr_period, p.adx_period, 30)
    start_ts = pd.Timestamp(start, tz="UTC") if start and pd.Timestamp(start).tzinfo is None else (pd.Timestamp(start).tz_convert("UTC") if start else None)
    end_ts = pd.Timestamp(end, tz="UTC") if end and pd.Timestamp(end).tzinfo is None else (pd.Timestamp(end).tz_convert("UTC") if end else None)
    first_i = warm if start_ts is None else max(warm, int(x.index.searchsorted(start_ts, side="left")))
    last_i = len(x) - 1 if end_ts is None else min(len(x) - 1, int(x.index.searchsorted(end_ts, side="right")) - 1)

    probs: dict = {}
    healthy_map: dict = {}
    log: list[dict] = []
    model = None
    meta = {}
    last_fit_i = -10**9

    for i in range(first_i, last_i + 1):
        r = x.iloc[i]
        side = technical_candidate_row(r, p)
        if side == 0:
            continue
        feat = _row_features(r, side)
        if any(pd.isna(v) or not np.isfinite(v) for v in feat.values()):
            continue

        if model is None or i - last_fit_i >= mlp.retrain_every_bars:
            model, meta = fit_best_v4_model(events, x.index[i], mlp)
            last_fit_i = i

        dt = x.index[i]
        healthy = bool(model is not None and meta.get("healthy"))
        healthy_map[dt] = healthy
        if healthy:
            row = pd.DataFrame([feat], index=[dt])[V4_FEATURES]
            prob = float(model.predict_proba(row)[0, 1])
            probs[dt] = prob
        else:
            prob = np.nan

        log.append({
            "signalDate": str(dt),
            "side": int(side),
            "probability": None if not np.isfinite(prob) else round(prob, 4),
            "healthy": healthy,
            "selectedModel": meta.get("selectedModel"),
            "validationAUC": meta.get("validationAUC"),
            "trainingEvents": meta.get("trainingEvents", 0),
            "status": meta.get("status"),
        })

    return probs, healthy_map, log


def current_v4_ml_advisory(df: pd.DataFrame, p: V4Params, mlp: MLV4Params) -> dict:
    x = features(df, p)
    r = x.iloc[-1]
    side = technical_candidate_row(r, p)
    if side == 0:
        return {"candidate": False, "healthy": False, "probability": None, "status": "NO_TECHNICAL_CANDIDATE"}

    events = build_v4_events(df.iloc[:-1], p, mlp)
    model, meta = fit_best_v4_model(events, x.index[-1], mlp)
    if model is None:
        return {"candidate": True, "healthy": False, "probability": None, **meta}

    feat = _row_features(r, side)
    row = pd.DataFrame([feat], index=[x.index[-1]])[V4_FEATURES]
    prob = float(model.predict_proba(row)[0, 1])
    return {
        "candidate": True,
        "healthy": True,
        "probability": round(prob, 4),
        "side": int(side),
        **meta,
    }
