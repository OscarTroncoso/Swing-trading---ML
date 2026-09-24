# Audit trail from V1 to V3.1

The original V1 used Bollinger/RSI mean reversion, fixed 10,000 EUR/USD units, same-close decision/fill, close-only stop/TP and no realistic execution cost. Its saved one-year result was +6.16%, 45% win rate, 2.79% max drawdown and PF 1.60, but those figures were produced under the original permissive execution assumptions.

The exploratory V2 corrected next-bar execution and costs and identified a simple momentum challenger: RSI + SMA50 + MACD. On the retained 90 closes it produced an exploratory +1.99%, but only six trades were available and only ~40 bars were actually evaluable after warm-up. This was never sufficient evidence to declare a production edge.

V3.1 adds EUR-account accounting, dynamic risk sizing, explicit leverage, a no-leverage alternative, 2025/prehistory warm-up, independent V1/V3 order generation, ML meta-label research, feature ablation, annual walk-forward and a single user-facing configuration file. The strict technical signal remains the baseline until research supports a change.
