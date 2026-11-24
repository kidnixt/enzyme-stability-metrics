# X2b — ThermoMPNN-D: el acoplamiento que predice casi no depende de la distancia

Fecha: 2026-11-24. Modelo: ThermoMPNN-D (Dieckhaus & Kuhlman, Protein Science 2025).
Estructura: `WT (4).pdb`. Medición digitalizada de figura.

## Tres resultados, en orden de fuerza

### 1. Corrido como lo diseñaron, no mira el par

Con el corte por default de 5 Å Cα–Cα, la salida tiene 192.774 predicciones y
la distancia máxima es 5.00 Å. **Cero filas mencionan 126 y 314 juntas.** Las
posiciones están a 18.4 Å. El único predictor que intenta modelar acoplamiento,
en su configuración de fábrica, no considera que estas dos posiciones puedan
interactuar.

### 2. Forzado a mirarlo, acierta el signo de los simples pero no el de la doble

| | ddG predicho | percentil entre 8.460 mutaciones |
|---|---|---|
| D314A | −0.438 kcal/mol | **0.6** (entre las más estabilizantes) |
| S126R | +1.380 kcal/mol | 69.7 |

Ordenamientos, del más robusto al menos:

- predicho: D314A > WT > DOUBLE > S126R
- medido: WT > DOUBLE > D314A > S126R

S126R último en los dos (Spearman 0.4, no significativo con n=4). Pero predice
D314A como **estabilizante**, entre las veinte mejores mutaciones posibles de la
proteína, cuando el laboratorio midió que baja la meseta de 60 % a 47 %.

Epistasis: aditivo +0.942, epistático +1.147, **predicha +0.205 kcal/mol**. El
signo es el contrario al medido: predice la doble algo peor que la suma, se
midió dramáticamente mejor.

### 3. Pero ese +0.205 está por debajo de su propio ruido

Nulo construido con las 101.441 predicciones del run dirigido (281 pares):

- mediana |epistasis| = **0.728 kcal/mol**, percentil 90 = 1.681
- el par real cae en el **percentil 15.2**
- contra pares a distancia comparable (14–22 Å, n=27.075): percentil **14.7**

El modelo le asigna a este par *menos* acoplamiento que al 85 % de los pares al
azar.

## El hallazgo principal

| Distancia Cα (Å) | n | mediana \|epistasis\| |
|---|---|---|
| <8 | 14.440 | 0.833 |
| 8–12 | 14.440 | 0.839 |
| 12–16 | 14.440 | 0.650 |
| 16–20 | 14.801 | 0.792 |
| 20–25 | 14.440 | 0.840 |
| 25–30 | 14.440 | 0.727 |
| >30 | 14.440 | 0.535 |

Spearman(distancia, |epistasis|) = −0.118. Significativo por el tamaño de
muestra, pero minúsculo: la tabla es casi plana y no es monótona.

**El modelo predice ~0.5 kcal/mol de acoplamiento entre residuos separados por
más de 30 Å**, donde no puede existir interacción física directa. Y el contraste
entre su rango de diseño (<8 Å: 0.833) y ese piso (>30 Å: 0.535) es de apenas
un factor de 1.5.

### Comparación entre modelos

La razón cerca/lejos no tiene unidades, así que sí es comparable entre modelos
aunque las escalas no lo sean:

| Modelo | cerca | lejos | razón |
|---|---|---|---|
| ESM-2 (log-verosimilitud) | 0.372 (<10 Å) | 0.045 (>20 Å) | **8.2** |
| ThermoMPNN-D (kcal/mol) | 0.833 (<8 Å) | 0.535 (>30 Å) | **1.56** |

**El modelo de secuencia discrimina distancia unas ocho veces mejor que el
modelo basado en estructura.** ThermoMPNN-D recibe las coordenadas como entrada
y aun así su término epistático es casi insensible a ellas.

## Honestidad sobre el alcance

- Usar el modelo a 18.4 Å es **fuera de su rango de diseño**. El corte de 5 Å es
  decisión de sus autores, probablemente porque conocen este comportamiento. Que
  emita ruido afuera no es un defecto oculto. Lo que sí es informativo es que el
  contraste dentro y fuera del rango sea de solo 1.5×: el acoplamiento "real"
  dentro de 5 Å no se separa bien de su propio piso.
- ThermoMPNN predice **estabilidad termodinámica de plegado**; lo medido es
  **cinética de inactivación**. Son observables distintos y la comparación de
  magnitudes en kcal/mol hay que leerla con esa salvedad al lado.
- El lado experimental viene de datos digitalizados y la transformación logit
  supone una partición de dos estados.
- Una proteína, un par real. No se generaliza.

## Qué queda

SaProt y TmProt (los de la tesis de Agustina) y FoldX. Con dos modelos de
familias distintas fallando por la misma razón geométrica, agregar un tercero
de la familia de secuencia aportaría menos que uno físico clásico.
