"""La huella aprendida de UMA contra los descriptores clasicos.

Por que esto puede salir positivo, a diferencia de todo lo anterior
-------------------------------------------------------------------
Los 814 descriptores del paper son casi todos bidimensionales: cuentan atomos,
anillos, enlaces, fragmentos. La huella de UMA sale de una representacion que
el modelo construyo mirando la geometria tridimensional real, y que usa para
predecir energias con precision de quimica cuantica.

Son dos descripciones de la misma molecula con informacion distinta. La
pregunta tiene respuesta de las dos formas, y ninguna esta determinada de
antemano por el diseno.

La prueba que mas importa
-------------------------
Los clasicos NO transfieren: entrenados en las 234 dan AUC 0.460 sobre las 527,
que es el percentil 20 del nulo por permutacion. Si la huella SI transfiere,
eso es un resultado positivo y concreto: una representacion aprendida de la
geometria generaliza donde los descriptores de conteo no.

Si tampoco transfiere, el resultado es igual de informativo pero negativo, y
apunta al conjunto de datos y no a la representacion.

Uso:
    python scripts/mol_embeddings_analysis.py          # en el entorno analysis
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from mol_analysis import QUANTUM, TRAP, load, pipe, nadeau_bengio_p

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desc", default="data/processed/mol_descriptors.csv")
    ap.add_argument("--quant", default="data/processed/mol_quantum_1p2.csv")
    ap.add_argument("--emb", default="data/processed/mol_embeddings.csv")
    ap.add_argument("--model", default="lr", choices=["lr", "gb"])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    m, classical = load(args.desc, args.quant)
    e = pd.read_csv(args.emb).drop(columns=["y", "split"], errors="ignore")
    emb_cols = [c for c in e.columns if c.startswith("emb")]
    m = m.merge(e, on="mol_id", how="inner")

    tr = m[m.split == "train"].reset_index(drop=True)
    te = m[m.split == "test"].reset_index(drop=True)
    print(f"train n={len(tr)} ({tr.y.mean()*100:.0f}% tox), "
          f"test n={len(te)} ({te.y.mean()*100:.0f}% tox)")
    print(f"{len(classical)} clasicos, {len(emb_cols)} de la huella, "
          f"{len(QUANTUM)} cuanticos   |   modelo {args.model}\n")

    SETS = {
        "CLASICOS": classical,
        "HUELLA UMA": emb_cols,
        "CUANTICOS": QUANTUM,
        "TRAMPA": TRAP,
        "CLASICOS+HUELLA": classical + emb_cols,
    }

    # ---------------- 1. validacion cruzada en entrenamiento
    print("=" * 70)
    print("1. Validacion cruzada sobre las 234 de entrenamiento")
    print("=" * 70)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats,
                                 random_state=0)
    folds = list(cv.split(tr, tr.y))
    scores = {}
    for name, cols in SETS.items():
        s = []
        for itr, ite in folds:
            p = pipe(args.model, cols)
            p.fit(tr.iloc[itr], tr.y.iloc[itr])
            s.append(roc_auc_score(tr.y.iloc[ite],
                                   p.predict_proba(tr.iloc[ite])[:, 1]))
        scores[name] = np.array(s)
        mark = "   <- control negativo" if name == "TRAMPA" else ""
        print(f"   {name:18s} AUC = {scores[name].mean():.3f} "
              f"+- {scores[name].std(ddof=1):.3f}{mark}")

    n_tr, n_te = len(tr) * 0.8, len(tr) * 0.2
    for a, b in [("HUELLA UMA", "CLASICOS"), ("CLASICOS+HUELLA", "CLASICOS")]:
        d = scores[a] - scores[b]
        t, p_ = nadeau_bengio_p(d, n_tr, n_te)
        print(f"\n   {a} vs {b}: {d.mean():+.4f} AUC   "
              f"t = {t:+.2f}   p = {p_:.3f}")

    # ---------------- 2. la prueba que importa
    print("\n" + "=" * 70)
    print("2. TRANSFERENCIA: entrenar en las 234, predecir las 527")
    print("=" * 70)
    rng = np.random.default_rng(0)
    preds = {}
    print(f"   {'conjunto':18s} {'AUC':>7s} {'MCC':>7s}")
    for name, cols in SETS.items():
        p = pipe(args.model, cols)
        p.fit(tr, tr.y)
        s = p.predict_proba(te)[:, 1]
        preds[name] = s
        print(f"   {name:18s} {roc_auc_score(te.y, s):7.3f} "
              f"{matthews_corrcoef(te.y, (s >= 0.5).astype(int)):7.3f}")

    print(f"\n   Nulo por permutacion ({args.perms} corridas, huella):")
    null = []
    for _ in range(args.perms):
        p = pipe(args.model, emb_cols)
        p.fit(tr, rng.permutation(tr.y.values))
        null.append(roc_auc_score(te.y, p.predict_proba(te)[:, 1]))
    null = np.array(null)
    real = roc_auc_score(te.y, preds["HUELLA UMA"])
    pct = (null < real).mean() * 100
    print(f"      nulo: media {null.mean():.3f}, sd {null.std(ddof=1):.3f}")
    print(f"      real: {real:.3f}   -> percentil {pct:.1f} del nulo")
    print(f"      p (una cola) = {(null >= real).mean():.3f}")

    print("\n" + "=" * 70)
    print("VEREDICTO")
    print("=" * 70)
    cls = roc_auc_score(te.y, preds["CLASICOS"])
    if pct > 95:
        print(f"   La huella SI transfiere (percentil {pct:.0f} del nulo) donde")
        print(f"   los clasicos dan {cls:.3f}. Es un resultado positivo.")
    elif pct > 80:
        print(f"   La huella queda en el percentil {pct:.0f}: senal debil, no")
        print("   concluyente. Hace falta mas potencia o mejor agrupamiento.")
    else:
        print(f"   La huella tampoco transfiere (percentil {pct:.0f}). Con los")
        print(f"   clasicos en {cls:.3f}, el problema esta en la particion de")
        print("   los datos y no en la representacion.")

    idx = np.arange(len(te))
    dif = [roc_auc_score(te.y.values[b], preds["CLASICOS+HUELLA"][b])
           - roc_auc_score(te.y.values[b], preds["CLASICOS"][b])
           for b in (rng.choice(idx, len(idx), replace=True)
                     for _ in range(args.boot))
           if te.y.values[b].std() > 0] if True else []
    if dif:
        lo, hi = np.percentile(dif, [2.5, 97.5])
        print(f"\n   Sumar la huella a los clasicos, en el test: "
              f"{np.mean(dif):+.4f} AUC")
        print(f"   IC 95% bootstrap: [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
