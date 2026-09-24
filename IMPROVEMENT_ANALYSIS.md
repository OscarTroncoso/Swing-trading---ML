# Análisis de mejora de rentabilidad — evolución hacia V3.1

## Por qué +1.99% no debe convertirse en un target

El +1.99% fue un resultado exploratorio sobre una muestra corta. El análisis mostró una señal preliminar prometedora, pero no una rentabilidad anual validada. Forzar parámetros hasta superar ese número introduciría selection bias.

La mejora se separa ahora en tres fuentes:

1. **edge de señal**: seleccionar mejor dirección/entrada;
2. **capital efficiency**: dimensionar mejor la posición;
3. **trade management**: mejorar salidas sin cortar tendencias útiles.

## Hallazgo previo: sizing

En el diagnóstico corto, modificar salidas no superó de forma consistente el TIME=5. En cambio, risk sizing elevó el retorno porque utilizó mejor el capital. Eso no crea edge nuevo: amplifica tanto ganancias como pérdidas.

V3.1 reemplaza la idea de “10k units” por equity EUR real. La posición depende de stop distance, riesgo objetivo, volatilidad y límite de leverage.

## Qué prueba ahora el proyecto

### Baseline interpretable

RSI + SMA50 + MACD con ATR para riesgo.

### Sizing

- fixed notional igual al equity inicial;
- 0.25%, 0.50%, 0.75%, 1.00% de riesgo fijo;
- adaptive risk por volatilidad;
- leverage caps 1x / 1.5x / 2x / 3x.

### Salidas

- TIME;
- momentum fade;
- ATR trailing;
- break-even + trailing;
- fade + trailing.

### ML

Meta-label con Logistic Regression regularizada y calibración cronológica. Se prueban feature sets `core`, `compact`, `full` y thresholds predefinidos. El resultado de 2026 no auto-selecciona el ganador.

## Qué puede aumentar retorno de forma saludable

Una subida de rentabilidad es más convincente si proviene de:

- mejor expectancy por trade;
- más candidatos válidos sin deteriorar PF;
- sizing que mantiene riesgo estable cuando cambia ATR;
- menor capital ocioso sin incremento desproporcionado de drawdown;
- leverage utilizado solo cuando el risk budget lo requiere;
- consistencia en años distintos.

Es menos convincente si proviene únicamente de:

- elevar leverage;
- bajar artificialmente threshold hasta aceptar casi todo;
- escoger retrospectivamente el mejor holding/stop;
- añadir muchos indicadores altamente correlacionados;
- entrenar ML con eventos cuyo resultado todavía no era conocido.

## Próxima decisión

Después del primer workflow real, comparar:

- strict baseline;
- strict adaptive sizing;
- ML challenger;
- robustness 2026;
- annual walk-forward.

Solo entonces tiene sentido decidir si ML pasa de `challenger` a una capa productiva o si el baseline interpretable sigue siendo superior.
