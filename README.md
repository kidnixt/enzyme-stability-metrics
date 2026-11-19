# enzyme-stability-metrics

¿Miden los predictores de estabilidad lo que el laboratorio realmente mide?

Caso de estudio: cuatro variantes de citosina desaminasa de *E. coli* (`codA`,
UniProt P25524) — WT, D314A, S126R y la doble D314A+S126R.

Repo separado de `master_thesis`. La línea del protocolo de selección con PLMs
queda en standby; esto es otra cosa.

## El problema

El laboratorio midió **estabilidad cinética**: actividad residual en el tiempo a
50 °C. Los modelos que se probaron (SaProt, TmProt) predicen **Tm**, que es
estabilidad termodinámica. No son la misma magnitud, y en la literatura de
ingeniería de proteínas no correlacionan: en un conjunto independiente de 51
mutantes no se encontró correlación significativa entre Rosetta ΔΔG / FoldX y el
T50 observado.

Ese desajuste es el objeto de estudio, no un bug.

Además, las curvas decaen a una **meseta no nula**, así que el half-life no está
definido para las variantes cuya meseta queda por encima del 50 %.

## Plan

| | Experimento | Estado | Cómputo |
|---|---|---|---|
| X1 | Reajuste de las curvas: (k, A_inf) por variante | **hecho** sobre datos digitalizados | CPU |
| X2 | Desajuste predictor (Tm/ΔΔG) vs medición cinética | siguiente | CPU |
| X3 | Clúster de sitio activo con UMA: separar efecto catalítico de estructural | listo para empezar | GPU |

## Datos

No hay acceso a los números crudos del laboratorio. `data/raw/inactivation_50C_digitized.csv`
se recuperó de la figura publicada con `scripts/digitize_figure.py`. La
calibración se derivó de las marcas de los ejes detectadas en la imagen, y se
valida sola: los tiempos recuperados caen a menos de 0.3 min de 15, 30, 45, 60,
120, 180, 240, 300 y 400. Incertidumbre ~1–2 puntos porcentuales en actividad.

Todo resultado que salga de ahí es una hipótesis a confirmar contra los datos
crudos, no un resultado.

## Numeración de secuencias

Las secuencias provistas no incluyen los primeros 3 residuos de la entrada
canónica:

    posicion_canonica = indice_en_string + 3

No es un supuesto: `sequences.verify_mutations()` lo confirma porque las dos
mutaciones caen exactamente en 126 y 314 con ese offset y con ningún otro.

**Resuelto (2026-11-19)**: los cuatro PDB abarcan los residuos 4..426, así que
los residuos 1-3 y el 427 simplemente no están modelados en 1K6W. Las secuencias
coinciden exactamente con las estructuras y el mapeo índice+3 no necesita más
corrección.

## Estructura

```
src/stability/
  sequences.py     # las 4 secuencias, offset, verificación de mutaciones
  active_site.py   # residuos del sitio activo, aguas, metal, histidinas del clúster
  kinetics.py      # Sadana-Henley: ajuste, bootstrap, t_1/2 con el caso indefinido
scripts/
  digitize_figure.py
notebooks/
  01_inactivation_curves.ipynb
docs/resultados/
```

## Uso

En tu PC (alcanza para X1 y X2):

```bash
pip install -r requirements-analysis.txt
PYTHONPATH=src python -m stability.sequences      # verifica las 4 secuencias
jupyter lab notebooks/01_inactivation_curves.ipynb
```

En el cluster de ORT, incluido el entorno con GPU para X3: ver
`docs/setup-cluster.md`.
