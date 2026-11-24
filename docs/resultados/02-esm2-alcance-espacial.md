# X2 — ESM-2 frente a la medición: el efecto cae fuera del alcance del modelo

Fecha: 2026-11-24. Datos de laboratorio digitalizados de figura (ver `01-reanalisis-curvas-inactivacion.md`).
Modelo: `facebook/esm2_t33_650M_UR50D`.

## Resultado en una línea

ESM-2 codifica acoplamiento espacial genuino entre residuos, con un alcance de
unos 12 Å. La epistasis que midió el laboratorio ocurre entre posiciones
separadas por 18.4 Å, fuera de ese alcance. El modelo no puede verla, y ahora
sabemos exactamente por qué.

## Lo que se midió

La doble mutante se comporta como D314A sola (48.4 % contra 47.0 %), no como la
suma de los dos efectos simples. En escala logit, la epistasis medida es
**−1.72 RT**: el modelo aditivo predice que la doble debería quedar en 14 % y
quedó en 48 %.

(La transformación logit supone que la meseta es una partición de equilibrio
entre dos estados. Es un supuesto declarado, no una medición.)

## Lo que dio ESM-2

Dos esquemas de scoring zero-shot:

| Esquema | Epistasis | Lectura |
|---|---|---|
| `wt_marginals` | 0.0000 | aditivo por construcción; es el nulo, no un competidor |
| `mutant_marginals` | +0.1003 | puede expresar acoplamiento; dio algo, en la dirección correcta |

El signo de +0.1003 coincide en dirección con la medición: ambos dicen que la
doble es mejor que la suma. Las dos escalas corren en sentidos opuestos
(energía: más alto = peor; log-verosimilitud: más alto = mejor), así que los
signos numéricos opuestos describen el mismo hecho.

## Por qué ese +0.1003 no significa nada

Un número solo no dice si es señal. Se generó un nulo: 300 pares de posiciones
al azar con sustituciones al azar, misma cuenta.

- Mediana del nulo: 0.0519. Percentil 90: 0.2194.
- **El par real cae en el percentil 71.** Dentro del grueso de la distribución.
- Contra pares a distancia comparable (14–22 Å, n=69): **percentil 67**.

Esa última es la comparación justa, porque mantiene constante el alcance
espacial del modelo. El par real es indistinguible de dos posiciones elegidas al
azar.

## El hallazgo que sí vale: el modelo tiene alcance espacial

| Distancia Cα (Å) | n | mediana \|epistasis\| |
|---|---|---|
| <8 | 5 | 0.691 |
| 8–12 | 10 | 0.189 |
| 12–16 | 25 | 0.061 |
| 16–20 | 32 | 0.055 |
| 20–25 | 53 | 0.074 |
| 25–30 | 45 | 0.032 |
| >30 | 130 | 0.042 |

Spearman(distancia, |epistasis|) = **−0.361**, p = 1.2e-10 sobre los 300 pares.
Mann-Whitney cerca (<10 Å) contra lejos (>20 Å): p = 1.7e-05.

La caída es fuerte por debajo de 12 Å y plana después. La no monotonía del bin
20–25 confirma que más allá de 12 Å queda solo fluctuación.

### Descarte del confundidor

Los residuos cercanos en el espacio suelen serlo también en la secuencia
(rho = +0.508 entre las dos distancias). Un transformer con atención local
produciría la misma caída sin saber nada de estructura. Dos chequeos
independientes lo descartan:

- Correlación parcial controlando por separación en secuencia: **rho = −0.252**.
- Restringiendo a los 233 pares separados por más de 50 residuos en secuencia,
  donde la atención local no ayuda: **rho = −0.262, p = 5.2e-05**.

La caída sobrevive por regresión y por estratificación. El acoplamiento es
espacial.

## Límites

- El bin <8 Å tiene n=5, así que la magnitud del acoplamiento de corto alcance
  está mal estimada. La tendencia no depende de ese bin: el Spearman usa los 300
  pares y el test estratificado usa 233.
- "Epistasis" acá es desviación de la aditividad en log-verosimilitud, no una
  energía libre. Se comparan signos y ordenamientos, nunca magnitudes ni
  cocientes entre las dos escalas.
- Una proteína, un modelo, un par real. No hay base para generalizar el valor de
  12 Å a otros sistemas.
- El lado experimental viene de datos digitalizados.

## Qué queda

Predictores no corridos, por orden de interés: **ThermoMPNN-D**, único con modo
epistático explícito — y cuyo umbral de acoplamiento por default es 5 Å Cα–Cα,
contra los 18.4 Å de este par, así que caería en aditivo por su propio criterio.
Después SaProt y TmProt (los de la tesis de Agustina) y FoldX.
