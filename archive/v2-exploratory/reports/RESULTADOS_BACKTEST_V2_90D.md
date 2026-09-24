# EUR/USD Model V2 — Auditoría y pruebas

## Base original reportada
- Capital inicial: 10,000
- Capital final: 10,615.97
- Retorno: +6.16%
- Win rate: 45.0%
- Max drawdown: 2.79%
- Profit factor: 1.60
- Trades: 20

## Reproducción sobre los 90 cierres conservados en data.json
- Original exacto: -1.82%, win rate 20.0%, PF 0.43, MDD 1.86%, 5 trades.
- Original corregido (señal en t, ejecución siguiente barra, 0.8 pip): -1.42%, win rate 28.57%, PF 0.53, MDD 1.77%, 7 trades.

## Challenger seleccionado para investigación
Momentum confirmado:
- LONG: RSI > 55, close > SMA50 y MACD histogram > 0.
- SHORT: RSI < 45, close < SMA50 y MACD histogram < 0.
- Salida temporal central: 5 barras.
- Stops/TP: en el diagnóstico close-only se usó proxy de volatilidad; en V2 se sustituye por ATR real con OHLC.

Resultado central en la ventana evaluable:
- Retorno: +1.99%
- Win rate: 83.33%
- Profit factor: 7.16
- Max drawdown: 0.37%
- Sharpe: 3.39
- Trades: 6

## Robustez paramétrica
576 combinaciones de:
- spread: 0.4 / 0.8 / 1.5 / 2.0 pips
- RSI: 52/48, 55/45, 58/42
- SMA50/EMA50
- MACD requerido / no requerido
- holding: 3 / 5 / 8 / 12
- SL/TP: 1.2/1.8, 1.5/2.2, 1.8/2.8

Resultados:
- 90.97% de configuraciones con retorno positivo.
- Retorno mediano: +0.87%.
- P10: +0.07%.
- P90: +1.99%.
- Drawdown mediano: 1.03%.
- Trades medianos: 6.

## Sensibilidad temporal
20 variantes cambiando el inicio del backtest y truncando el final:
- Momentum positivo: 100% de variantes.
- Momentum superó al baseline corregido: 100% de variantes.
- Mejora mediana de retorno vs baseline: +1.48 puntos porcentuales.

## Qué corrige V2
- Entrada al open siguiente después de la señal del close.
- OHLC para stops/take-profit intradía.
- ATR real.
- RSI Wilder.
- MACD real.
- ADX.
- Bollinger históricas.
- Spread y slippage.
- Comisión configurable.
- Sizing fijo o por riesgo.
- Equity curve con Sharpe/Sortino/MDD/exposure.
- Tratamiento conservador si stop y TP se tocan en la misma vela.
- Dashboard sin probabilidades inventadas.

## Tests automatizados
6/6 tests pasan:
1. columnas de indicadores,
2. ausencia de look-ahead en señales pasadas,
3. mayores costes no mejoran el mismo sistema,
4. cronología de ejecución,
5. cap de position sizing,
6. métricas finitas.

## Limitación crítica
El `data.json` recibido conserva solo 90 cierres, aunque el script original descargó un año. Por tanto, los resultados del challenger son exploratorios y no sustituyen un backtest OHLC de 5–10 años con walk-forward. El motor V2 queda preparado para ejecutar ese análisis con un CSV OHLC o con yfinance en un entorno con acceso a internet.
