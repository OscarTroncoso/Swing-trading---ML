"""2026 comparison: V3.1 benchmark, V4, and V4 with ML advisory sizing."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_v4_params, load_backtest_settings
from src.strategy_engine import backtest_v31_benchmark, fetch_yahoo, load_csv
from src.strategy_v4 import backtest_v4
from src.ml_v4 import MLV4Params, build_probability_maps


def flat(name, r):
    return {"model": name, **r["metrics"]}


def ml_cfg(path: Path) -> MLV4Params:
    raw = json.loads(path.read_text(encoding="utf-8"))
    m = raw.get("ml", {})
    return MLV4Params(
        retrain_every_bars=m.get("retrainEveryBars", 60),
        min_train_events=m.get("minTrainEvents", 120),
        validation_fraction=m.get("calibrationFraction", 0.20),
        max_train_years=m.get("maxTrainYears", 8),
        min_validation_auc=m.get("minValidationAUC", 0.52),
        label_horizon_bars=m.get("labelHorizonBars", 60),
        logistic_c=m.get("C", 0.7),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--period")
    ap.add_argument("--out", default="reports/backtest_2026.json")
    ap.add_argument("--csv-out", default="reports/backtest_2026_comparison.csv")
    args = ap.parse_args()

    cfg = ROOT / args.config
    p = load_v4_params(cfg)
    bc = load_backtest_settings(cfg)
    start = args.start or bc.get("evaluationStart", "2026-01-01")
    period = args.period or bc.get("downloadHistory", "10y")
    df = load_csv(args.csv) if args.csv else fetch_yahoo("EURUSD=X", period=period)

    v31 = backtest_v31_benchmark(df, p, start=start, end=args.end)
    v4 = backtest_v4(df, p, start=start, end=args.end)

    mlp = ml_cfg(cfg)
    probs, healthy, predlog = build_probability_maps(df, p, mlp, start=start, end=args.end)
    v4ml = backtest_v4(
        df,
        p,
        start=start,
        end=args.end,
        ml_probability_by_date=probs,
        ml_healthy_by_date=healthy,
    )

    table = pd.DataFrame([
        flat("V3.1 benchmark / 5-day time exit", v31),
        flat("V4 / SL-TP only / EUR sizing", v4),
        flat("V4 + ML advisory risk scaling", v4ml),
    ])

    co = ROOT / args.csv_out
    co.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(co, index=False)

    report = {
        "window": {"start": start, "end": args.end or str(df.index.max().date())},
        "historyStart": str(df.index.min().date()),
        "rows": len(df),
        "methodology": [
            "All models use signal-at-close and next-open execution.",
            "V4 keeps a trade open until SL/TP; no forced 3/5-day exit.",
            "V4 position is expressed as EUR margin/stake and leverage is x1..x30.",
            "Leverage is selected to use the risk budget without exceeding it.",
            "ML cannot invent direction or reject trades in V4; a healthy model only scales risk modestly.",
            "Open positions at the evaluation end remain OPEN and are marked to market.",
        ],
        "comparison": table.to_dict(orient="records"),
        "v4Trades": v4["trades"],
        "v4OpenPosition": v4.get("openPosition"),
        "v4MLTrades": v4ml["trades"],
        "v4MLOpenPosition": v4ml.get("openPosition"),
        "mlPredictions": predlog,
    }

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
