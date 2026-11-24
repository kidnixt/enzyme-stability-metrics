"""El modelo clasico falla en el test publicado. Es el corrimiento, o soy yo?

Los 814 descriptores del paper dan AUC 0.745 en validacion cruzada sobre las
234 de entrenamiento y 0.460 sobre las 527 de test. Un AUC por debajo de 0.5
en transferencia tiene dos lecturas incompatibles y hay que separarlas antes
de afirmar nada:

  (a) los dos conjuntos estan genuinamente corridos entre si
  (b) el 0.745 era sobreajuste disfrazado, y no habia senal que transferir

Las cuatro celdas lo deciden:

  CV solo en train    alto + CV solo en test    alto + transferencia baja  -> (a)
  CV solo en train    alto + CV solo en test    bajo                       -> ojo
  CV sobre el conjunto unido alto                                          -> (a)
  CV sobre el conjunto unido bajo                                          -> (b)

Si al unir los dos conjuntos y hacer validacion cruzada la senal reaparece,
entonces los descriptores si distinguen toxicos de no toxicos y lo que falla
es especificamente la particion publicada. Si al unirlos tampoco hay senal,
el 0.745 era un artefacto de 234 muestras con 814 variables.

Se agrega un nulo por permutacion de etiquetas para saber donde cae un 0.46
cuando no hay nada que aprender.

Uso:
    python scripts/mol_transfer.py
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from mol_analysis import QUANTUM, load, pipe      # misma tuberia, sin duplicar


def cv_auc(p, X, y, repeats, seed=0):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    return cross_val_score(p, X, y, cv=cv, scoring="roc_auc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desc", default="data/processed/mol_descriptors.csv")
    ap.add_argument("--quant", default="data/processed/mol_quantum_1p2.csv")
    ap.add_argument("--model", default="lr", choices=["lr", "gb"])
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--perms", type=int, default=200)
    args = ap.parse_args()

    m, classical = load(args.desc, args.quant)
    tr = m[m.split == "train"].reset_index(drop=True)
    te = m[m.split == "test"].reset_index(drop=True)
    cols = classical

    print(f"train n={len(tr)} ({tr.y.mean()*100:.0f}% tox), "
          f"test n={len(te)} ({te.y.mean()*100:.0f}% tox), "
          f"{len(cols)} descriptores clasicos, modelo {args.model}\n")

    print("=" * 66)
    print("Validacion cruzada DENTRO de cada conjunto")
    print("=" * 66)
    for nm, g in (("solo train", tr), ("solo test", te),
                  ("los dos juntos", m.reset_index(drop=True))):
        s = cv_auc(pipe(args.model, cols), g, g.y, args.repeats)
        print(f"   {nm:16s} n={len(g):4d}   AUC = {s.mean():.3f} +- {s.std(ddof=1):.3f}")

    print("\n" + "=" * 66)
    print("Transferencia entre conjuntos")
    print("=" * 66)
    for nm, a, b in (("train -> test", tr, te), ("test -> train", te, tr)):
        p = pipe(args.model, cols)
        p.fit(a, a.y)
        auc = roc_auc_score(b.y, p.predict_proba(b)[:, 1])
        print(f"   {nm:16s} AUC = {auc:.3f}")

    print("\n" + "=" * 66)
    print(f"Nulo: {args.perms} permutaciones de las etiquetas de entrenamiento")
    print("=" * 66)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(args.perms):
        yp = rng.permutation(tr.y.values)
        p = pipe(args.model, cols)
        p.fit(tr, yp)
        null.append(roc_auc_score(te.y, p.predict_proba(te)[:, 1]))
    null = np.array(null)
    p_real = pipe(args.model, cols)
    p_real.fit(tr, tr.y)
    real = roc_auc_score(te.y, p_real.predict_proba(te)[:, 1])
    print(f"   nulo: media {null.mean():.3f}, sd {null.std(ddof=1):.3f}, "
          f"rango [{null.min():.3f}, {null.max():.3f}]")
    print(f"   real: {real:.3f}")
    print(f"   percentil del real dentro del nulo: "
          f"{(null < real).mean()*100:.1f}%")
    print("\n   Si el real cae en el medio del nulo, el modelo no aprendio nada")
    print("   transferible. Si cae por debajo, aprendio algo que en el test")
    print("   apunta al reves, que es mas raro y mas informativo.")


if __name__ == "__main__":
    main()
