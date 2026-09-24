# Auditoría metodológica — EUR/USD V3.1

## Cambios estructurales

V3.1 corrige varios problemas del prototipo original:

- €10.000 representa equity de cuenta, no 10.000 unidades fijas.
- P&L de EUR/USD se convierte de USD a EUR.
- señal en cierre t y ejecución en apertura t+1;
- ATR/sizing congelados en la barra de señal;
- SL/TP con High/Low y supuesto conservador stop-first;
- spread y slippage;
- sizing dinámico según riesgo y volatilidad;
- apalancamiento explícito y limitado;
- alternativa sin apalancamiento;
- ML temporal como meta-label, no como generador opaco de dirección;
- walk-forward separado del periodo usado para descubrir ideas.

## Capital y posición

Para EUR/USD, las unidades son EUR de notional base. El motor calcula el tamaño necesario para aproximarse a un presupuesto de pérdida en stop:

`risk_budget = equity_eur * applied_risk_pct`

Luego limita el tamaño por:

1. paso mínimo de lote;
2. tamaño mínimo;
3. cap absoluto opcional;
4. `equity * maxLeverage`.

La salida incluye `units`, `lots`, `notionalEUR`, `leverage`, `usesLeverage`, `riskAtStopEUR`, `targetGainEUR` y la alternativa sin leverage.

## Sizing adaptativo

El riesgo base se escala con volatilidad realizada respecto a un target. Si un candidato ML es aceptado, su probabilidad puede modular adicionalmente el riesgo dentro de límites predefinidos. Esto cambia exposición, no dirección.

## ML

El challenger ML parte de candidatos técnicos más amplios y estima si conviene ejecutarlos.

Salvaguardas:

- features conocidas al cierre de señal;
- labels generados con next-open + barreras;
- `outcomeEnd` registra cuándo el label se vuelve observable;
- solo se entrena con `outcomeEnd < predictionDate`;
- calibración cronológica;
- reentrenamiento periódico;
- feature sets anidados para medir coste de complejidad;
- threshold fijo en config;
- ningún script modifica automáticamente la producción según el mejor retorno de 2026.

## Riesgo de sobreajuste

Sigue existiendo. Los indicadores técnicos provienen del mismo precio y pueden ser redundantes. Por eso el ML inicial utiliza regularización y un conjunto pequeño de variables, y se mantiene como challenger hasta superar validación temporal.

Una mejora solo debería considerarse seria si mantiene resultados razonables en:

- múltiples años;
- distintos spreads/slippage;
- perturbaciones de parámetros;
- suficiente número de trades;
- drawdowns controlados;
- walk-forward out-of-sample.

## 2025 → 2026

El backtest 2026 no desperdicia 50 sesiones al comenzar enero. La historia anterior ya está disponible para formar SMA50/ATR/RSI/MACD. El P&L comienza exactamente el 01-01-2026.

Para ML se conserva más historia que 2025 porque un solo año puede producir pocos eventos etiquetados. Esto no contamina 2026: solo outcomes ya resueltos antes de cada predicción entran al training.

## Estado de validación local

El código se valida con tests sintéticos diseñados para verificar cronología y lógica. Los resultados financieros de esos datos sintéticos no tienen significado económico y no deben compararse con EUR/USD real.

El resultado real 2026 se genera en GitHub Actions sobre el OHLC descargado y congelado en `data/eurusd_daily.csv`.
