# X4 — uma-s-1p1 calcula mal los cationes; 1p2 lo corrige

Fecha: 2026-11-24. Scripts: `scripts/mol_validate_ie.py`,
`scripts/mol_compare_checkpoints.py`.
Datos: `data/processed/ie_validation.csv`, `data/processed/checkpoint_comparison.csv`.

## Por qué se hizo este chequeo

Para la mitad molecular del trabajo hacen falta descriptores energéticos
comparables entre moléculas. La energía total no lo es (escala con la
composición), así que se usan diferencias sobre la misma geometría,
aprovechando que OMol25 está condicionado por carga:

```
IE = E(carga +1) − E(carga 0)      energía de ionización vertical
EA = E(carga 0) − E(carga −1)      afinidad electrónica vertical
```

Al revisar los 761 valores producidos con `uma-s-1p1`, la distribución global
era razonable (mediana 8.56 eV) pero una minoría daba valores imposibles:
ésteres de sulfonato cerca de 2 eV y carbonilos chicos cerca de 5 eV, donde lo
real ronda 10.

Dos hipótesis se descartaron antes de seguir:

1. **Convención de espín.** La documentación de FAIRChem indica que `spin` es
   la multiplicidad total y su valor por defecto es 1, que es lo que se pasó.
2. **Geometría MMFF mala.** Las moléculas afectadas tienen fuerza residual
   *menor* (mediana 1.40 eV/Å) que las no afectadas (1.74), así que no es que
   estuvieran peor construidas.

Un tercer indicio localizó el problema: la EA salía bien mientras la IE salía
mal, y como las dos comparten `E(carga 0)`, el error tenía que estar en las
energías de los **cationes**.

## Referencia

DFT real con PySCF sobre las mismas geometrías y con la misma definición:
B3LYP/def2-SVP, Kohn-Sham restringido para el neutro y no restringido para el
catión. B3LYP no es el funcional de OMol25 (que es ωB97M-V), así que un sesgo
constante sería esperable y poco interesante; lo que importa es el orden y la
magnitud del error.

25 moléculas de ≤12 átomos, mezclando deliberadamente las sospechosas con las
normales.

## Resultado

| modelo | n | sesgo | MAE | Spearman |
|---|---|---|---|---|
| uma-s-1p1 | 12 | −2.90 eV | 3.16 eV | 0.755 |
| **uma-s-1p2** | 12 | **+0.17 eV** | **0.29 eV** | **0.902** |

Mediana de la IE por grupo de moléculas sonda:

| modelo | grupo "rota" | grupo "sana" |
|---|---|---|
| uma-s-1p1 | 5.36 | 11.91 |
| uma-s-1p2 | 10.71 | 11.94 |

El patrón es el que discrimina: **1p2 corrige el grupo roto sin alterar el
sano.** El peor caso de 1p1, `COS(C)(=O)=O`, pasa de 2.88 eV (DFT: 10.51) a
11.02 eV.

Un MAE de 0.29 eV contra B3LYP está dentro del error del propio método de
referencia frente a experimento, así que la IE de `uma-s-1p2` por Δ-SCF es
utilizable como descriptor cuantitativo, no solo como ranking.

## Checkpoints que no sirvieron

- `uma-s-1p2p1`: no existe en la versión de `fairchem-core` que el proyecto
  tiene clavada en `<2.22` (ver `requirements-mlip.txt`: desde 2.22 fairchem
  pide torch con CUDA 13 y el cluster tiene driver de la serie 575).
- `uma-m-1p1`: checkpoint de 11 GB, no entra en la RTX 2080 Ti de 11 GB
  compartida con otro proceso. OOM en las doce moléculas.

## Consecuencias

1. **Todo lo molecular se recalcula con `uma-s-1p2`.** El default de
   `scripts/mol_quantum.py` quedó cambiado.
2. **Las fuerzas de las proteínas (X3) también se recalculan.** Ese resultado
   se produjo con 1p1. La conclusión de X3 es geométrica — las dos mutaciones
   están a 14.9 Å y la esfera más grande que entra es de 9, así que el cálculo
   es aditivo por construcción — y no depende de los valores numéricos. Pero
   usar un checkpoint que este mismo trabajo demuestra defectuoso es una
   objeción evitable, y rehacerlo cuesta 88 esferas chicas.
3. **El hallazgo vale por sí mismo.** Un modelo potencial universal puede tener
   su condicionamiento por carga roto en una familia química entera sin que
   nada en la salida lo señale: las energías no vienen con barras de error y
   la distribución global parecía sana. La única forma de detectarlo fue
   contrastar contra química cuántica de verdad en un puñado de moléculas
   baratas. Es el mismo patrón que el resto del trabajo: medir el alcance de
   la herramienta antes de confiarle una conclusión.
