# X1 — Reanálisis de las curvas de inactivación a 50 °C

Fecha: 2026-11-19. Datos: **digitalizados de la figura publicada**, no crudos.

## Procedencia y su límite

No hay acceso a los números del laboratorio. Los puntos se recuperaron de la
imagen con `scripts/digitize_figure.py`: calibración derivada de las marcas de
los ejes detectadas en la propia imagen, no estimadas a ojo. El control de que
la calibración es correcta es que los tiempos recuperados caen a menos de
0.3 min de 15, 30, 45, 60, 120, 180, 240, 300 y 400.

Incertidumbre por digitalización: ~1–2 puntos porcentuales en actividad.

No se pudieron resolver DOUBLE en t = 15, 30, 45, 60 y 180 ni D314A en t = 240,
porque las dos series se superponen ahí. DOUBLE queda con 5 puntos, todos salvo
t=0 sobre la meseta.

**Todo lo que sigue es hipótesis a confirmar contra los datos crudos.**

## Modelo

Sadana & Henley (1987): desactivación unimolecular de un paso cuyo estado final
conserva actividad, lo que produce decaimiento no de primer orden hacia una
meseta no nula.

    A(t) = A_inf + (100 - A_inf) · e^(−k·t)

## Resultados

| Variante | n | k (1/min) | IC95 k | A_inf (%) | IC95 A_inf | t½ | k identificable |
|---|---|---|---|---|---|---|---|
| WT | 10 | 0.108 | [0.077, 0.159] | 60.0 | [58.0, 61.9] | **no definido** | sí |
| DOUBLE | 5 | 0.022 | [0.014, 3.74] | 48.4 | [45.1, 51.1] | 161.7 min | **no** |
| D314A | 9 | 0.109 | [0.079, 0.149] | 47.0 | [44.6, 49.3] | 26.4 min | sí |
| S126R | 10 | 0.353 | [0.234, 8.29] | 22.1 | [21.3, 22.9] | 2.9 min | **no** |

## Qué es medible y qué no

**A_inf es el observable robusto.** Intervalo de ±2 puntos para las cuatro,
sobre un rango de 22 % a 60 %. El ordenamiento es seguro.

**k solo es identificable para WT y D314A.** Para S126R y DOUBLE el decaimiento
es más rápido que el primer punto de muestreo (15 min), así que el intervalo
bootstrap abarca órdenes de magnitud. Esos dos valores de k no son mediciones:
son artefactos de la grilla temporal.

**t½ no está definido para WT** (meseta en 60 %, la curva nunca cruza el 50 %) y
**es indecidible para DOUBLE**, cuyo IC de A_inf cruza el 50 %. Ordenar las
cuatro variantes por vida media no es posible con estos datos, contra cualquier
modelo que se compare.

Esto responde la pregunta de Yovine del 12/8 con más precisión que la pregunta
original: no es solo que no exista un modelo que prediga el half-time; es que el
half-time no es una cantidad bien definida para este conjunto de variantes.

## Dos observaciones

**WT y D314A se inactivan a la misma velocidad.** k = 0.108 y 0.109, con
intervalos casi superpuestos. Lo que difiere es solo la meseta: 60 % contra
47 %. D314A no cambia qué tan rápido se inactiva la enzima, cambia qué fracción
sobrevive. Son afirmaciones físicas distintas y conviene no confundirlas.

**La doble se parece a D314A sola, no a la suma.** S126R sola cae a 22 %, pero
S126R junto con D314A queda en 48 %, dentro del error de D314A sola (47 %). Los
efectos no son aditivos: D314A parece suprimir lo que hace S126R. Si se sostiene
en los datos crudos, es epistasis, y ningún modelo que prediga Tm tiene ruta
para capturarla.

## Pendientes

- Confirmar ambas observaciones contra los números crudos.
- Puntos tempranos (1, 2, 5, 10 min) si alguna vez hay otro ensayo: sin ellos k
  es inaccesible para las dos variantes que más rápido se caen.
- El residuo faltante del C-terminal (423 + 3 = 426 contra 427 canónicos).
