"""Download/freeze market OHLC without running strategy or ML."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.strategy_engine import fetch_yahoo

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ticker",default="EURUSD=X")
    ap.add_argument("--period",default="10y")
    ap.add_argument("--out",default="data/research_10y.csv")
    args=ap.parse_args()
    df=fetch_yahoo(args.ticker,period=args.period)
    out=ROOT/args.out; out.parent.mkdir(parents=True,exist_ok=True)
    df.reset_index().rename(columns={"index":"date"}).to_csv(out,index=False)
    print(f"saved {len(df)} OHLC rows to {out}")

if __name__=="__main__": main()
