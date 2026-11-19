# Instalación en el cluster de ORT

Cluster: JupyterHub, usuario `jovyan`, 1× RTX 2080 Ti (11 GB), Xeon W-2195
36 vCPU, driver 575.51 / CUDA 12.9. Sin Slurm, sin root: las corridas se lanzan
a mano y los entornos van en `$HOME`.

## Dos entornos, no uno

```bash
bash scripts/setup_cluster.sh analysis   # X1 y X2 — solo CPU
bash scripts/setup_cluster.sh mlip       # X3 — fairchem + UMA, GPU
```

Separados a propósito: `fairchem` arrastra su propia versión de torch y
convive mal con todo lo demás. El script registra cada uno como kernel de
Jupyter (`enzyme-stability (analysis)` y `enzyme-stability (mlip)`), así que
desde JupyterHub se eligen desde el selector de kernel.

**X1 y X2 no necesitan el cluster.** Son numpy, scipy y pandas sobre cuatro
curvas. Corren en tu PC en segundos. El cluster hace falta para X3 y para los
predictores basados en PLMs.

Conda no se usa: fairchem dejó de soportarlo y pytorch también lo discontinuó.

## Las tres cosas que cuestan tiempo si no se saben antes

**1. Los pesos de UMA están gated, y el formulario no se puede editar.**
Hay que crear cuenta en HuggingFace y aceptar la FAIR Chemistry License v1 en
`https://huggingface.co/facebook/UMA`. El formulario pide nombre legal
completo, fecha de nacimiento y nombre completo de la organización, y **no se
puede editar después de enviarlo**. Después hace falta un token con permiso de
lectura sobre repos gated, y `huggingface-cli login`. La aprobación suele ser
inmediata, pero hasta que no esté no se baja nada: conviene hacerlo primero.

**2. No usar el checkpoint `uma-s-1`.** Tiene un bug conocido de
extensividad, y la extensividad es exactamente la propiedad de la que dependen
las energías de interacción E(complejo) − E(partes), que es lo que calcula X3.
Usar `uma-s-1p1` o `uma-s-1p2p1`. El modelo mediano es `uma-m-1p1`.

**3. La 2080 Ti es Turing (compute capability 7.5): no tiene bfloat16 nativo.**
Si algo del stack intenta bf16, va a ser lento o fallar. `check_mlip_env.py` lo
detecta y avisa. Quedarse en fp32.

Aparte: el cache de HuggingFace pesa varios GB y por defecto va a
`~/.cache/huggingface`. Si el home tiene cuota, exportar `HF_HOME` a un disco
con espacio antes de bajar los pesos.

## Verificación

```bash
source ~/venvs/mlip/bin/activate
python scripts/check_mlip_env.py
```

Imprime versión de torch, GPU, capacidad de cómputo, avisa si falta bf16, carga
UMA y calcula energía y fuerzas de una molécula de agua. Si eso sale, el
entorno está listo para X3.

## Tamaño esperado del cálculo

Los clústeres de sitio activo son de 200–250 átomos: 13 residuos truncados en
Cβ/Cγ, 4 aguas catalíticas, el metal y el sustrato. Eso entra holgado en 11 GB
y queda por debajo del techo de 350 átomos con el que se entrenó OMol25. No hace
falta el modelo mediano para empezar.
