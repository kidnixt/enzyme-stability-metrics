"""El embedding hereda el defecto del modelo, o sobrevive?

La pregunta
-----------
X4 mostro que uma-s-1p1 calcula mal la energia de ionizacion de una familia
quimica entera: error de hasta 7 eV contra DFT, mientras que 1p2 acierta a
0.29 eV. El defecto esta en la SALIDA del modelo.

Pero la salida es el final de una cadena: el modelo construye primero una
representacion interna de cada atomo y despues una cabeza la convierte en
energia. El defecto puede estar en cualquiera de los dos lugares, y la
diferencia importa mucho:

  - Si la REPRESENTACION esta sana y solo la cabeza falla, entonces extraer
    embeddings es mas robusto que leer predicciones. Eso es una recomendacion
    practica para cualquiera que use este metodo.

  - Si la representacion tambien esta rota, entonces extraer features hereda
    fallas silenciosas que ninguna metrica de la salida revela. Eso es una
    advertencia.

Las dos respuestas sirven. La literatura que extrae embeddings de potenciales
interatomicos no se hizo esta pregunta.

El diseno
---------
Se usa la IE de 1p2 como verdad, porque esta validada contra DFT a 0.29 eV de
error, sobre las 761 moleculas. Tres mediciones contra esa verdad:

  A. La cabeza de 1p1              - se sabe que falla en el grupo roto
  B. Una sonda lineal sobre los embeddings de 1p1   <- LA PREGUNTA
  C. La misma sonda, pero prediciendo la IE EQUIVOCADA de la propia 1p1

A y B contra la verdad dicen si la representacion sobrevive. C dice si la
representacion contiene ademas el error: si C es precisa, el embedding
codifica aquello que la cabeza leyo mal, y el problema es de lectura.

El grupo "roto" se define por donde falla la cabeza de 1p1, que es una
cantidad distinta de la que se mide despues (la precision de la sonda), asi
que no hay circularidad.

Con --strict los pliegues no son al azar sino por familia quimica: se agrupan
las moleculas por similitud estructural y se deja cada familia ENTERA afuera.
Sin eso, al retirar un sulfonato quedan sus hermanos en entrenamiento y la
sonda puede acertar por vecindad. La version estricta pregunta algo mas duro y
mas util: la representacion acierta en quimica que nunca vio?

Uso:
    python scripts/uma_embeddings.py --layer module.backbone --model uma-s-1p1 \\
        --out data/processed/mol_embeddings_1p1.csv
    python scripts/embedding_vs_head.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def probe():
    return Pipeline([("sc", StandardScaler()),
                     ("r", RidgeCV(alphas=np.logspace(-2, 5, 25)))])


def report(name, pred, true, mask_broken):
    err = np.abs(pred - true)
    print(f"   {name:46s} {err.mean():7.3f} {err[mask_broken].mean():9.3f} "
          f"{err[~mask_broken].mean():9.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb1p1", default="data/processed/mol_embeddings_1p1.csv")
    ap.add_argument("--emb1p2", default="data/processed/mol_embeddings.csv")
    ap.add_argument("--q1p1", default="data/processed/mol_quantum.csv")
    ap.add_argument("--q1p2", default="data/processed/mol_quantum_1p2.csv")
    ap.add_argument("--broken-eV", type=float, default=1.0)
    ap.add_argument("--strict", action="store_true",
                    help="dejar familias quimicas enteras afuera, no moleculas sueltas")
    ap.add_argument("--tanimoto", type=float, default=0.6,
                    help="corte de similitud para armar las familias")
    args = ap.parse_args()

    q1 = pd.read_csv(args.q1p1)[["mol_id", "smiles", "IE_eV"]].rename(
        columns={"IE_eV": "ie_1p1"})
    q2 = pd.read_csv(args.q1p2)[["mol_id", "IE_eV"]].rename(
        columns={"IE_eV": "ie_1p2"})
    e1 = pd.read_csv(args.emb1p1)
    e2 = pd.read_csv(args.emb1p2)
    cols = [c for c in e1.columns if c.startswith("emb")]

    d = q1.merge(q2, on="mol_id").merge(
        e1[["mol_id"] + cols], on="mol_id").reset_index(drop=True)
    X1 = d[cols].values
    true = d.ie_1p2.values
    head = d.ie_1p1.values
    broken = np.abs(head - true) > args.broken_eV

    print(f"{len(d)} moleculas, {len(cols)} dimensiones de embedding")
    print(f"grupo roto (|cabeza 1p1 - verdad| > {args.broken_eV} eV): "
          f"{broken.sum()} ({broken.mean()*100:.1f}%)")
    print(f"grupo sano: {(~broken).sum()}\n")

    if args.strict:
        # Agrupar por similitud estructural y dejar grupos ENTEROS afuera. Con
        # pliegues al azar, al retirar un sulfonato quedan sus diez hermanos en
        # entrenamiento, asi que la sonda puede acertar por vecindad en vez de
        # por entender la quimica. Esto lo impide.
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.ML.Cluster import Butina
        from rdkit import DataStructs
        from sklearn.model_selection import LeaveOneGroupOut
        RDLogger.DisableLog("rdApp.*")

        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fps = [gen.GetFingerprint(Chem.MolFromSmiles(s_)) for s_ in d.smiles]
        dists = []
        for i in range(1, len(fps)):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend(1 - x for x in sims)
        clusters = Butina.ClusterData(dists, len(fps), 1 - args.tanimoto,
                                      isDistData=True)
        groups = np.empty(len(fps), dtype=int)
        for gi, cl in enumerate(clusters):
            for i in cl:
                groups[i] = gi
        sizes = pd.Series(groups).value_counts()
        print(f"MODO ESTRICTO: {len(clusters)} familias quimicas "
              f"(Tanimoto {args.tanimoto}); la mayor tiene {sizes.iloc[0]}")
        print(f"   familias que tocan el grupo roto: "
              f"{len(set(groups[broken]))}\n")
        cv = LeaveOneGroupOut()
        p_true = cross_val_predict(probe(), X1, true, cv=cv, groups=groups)
        p_head = cross_val_predict(probe(), X1, head, cv=cv, groups=groups)
    else:
        cv = KFold(5, shuffle=True, random_state=0)
        p_true = cross_val_predict(probe(), X1, true, cv=cv)
        p_head = cross_val_predict(probe(), X1, head, cv=cv)

    print("=" * 72)
    print("Error absoluto medio contra la verdad (IE de 1p2, validada vs DFT)")
    print("=" * 72)
    print(f"   {'':46s} {'TODAS':>7s} {'ROTAS':>9s} {'SANAS':>9s}")
    report("A. cabeza de 1p1 (la salida del modelo)", head, true, broken)
    report("B. sonda sobre embeddings de 1p1", p_true, true, broken)
    print()
    print(f"   {'':46s} {'TODAS':>7s} {'ROTAS':>9s} {'SANAS':>9s}")
    report("C. sonda prediciendo la IE ERRONEA de 1p1", p_head, head, broken)

    ga = np.abs(head - true)[broken].mean()
    gb = np.abs(p_true - true)[broken].mean()
    print("\n" + "=" * 72)
    print("VEREDICTO sobre el grupo donde la cabeza falla")
    print("=" * 72)
    print(f"   cabeza: {ga:.3f} eV      sonda sobre el embedding: {gb:.3f} eV")
    if gb < ga * 0.5:
        print("\n   LA REPRESENTACION SOBREVIVE. El embedding conserva la quimica")
        print("   que la cabeza lee mal. Extraer features es mas robusto que")
        print("   leer predicciones, y eso es una recomendacion concreta.")
    elif gb < ga:
        print("\n   Mejora parcial: el embedding conserva algo pero no todo.")
    else:
        print("\n   EL EMBEDDING HEREDA EL DEFECTO. Extraer features no protege:")
        print("   el error esta en la representacion y es invisible desde la")
        print("   salida. Es una advertencia para quien use este metodo.")

    # ---- los dos espacios codifican la misma quimica?
    common = set(e1.mol_id) & set(e2.mol_id)
    a = e1[e1.mol_id.isin(common)].sort_values("mol_id")
    b = e2[e2.mol_id.isin(common)].sort_values("mol_id")
    br = pd.Series(broken.astype(bool), index=d.mol_id).reindex(a.mol_id).values

    def rsa(ia):
        """Similitud entre moleculas dentro de cada espacio, comparada entre
        espacios. Los dos checkpoints no comparten sistema de coordenadas, asi
        que se compara la GEOMETRIA del espacio y no los vectores."""
        A = StandardScaler().fit_transform(a.loc[ia, cols].values)
        B = StandardScaler().fit_transform(b.loc[ia, cols].values)
        ca = np.corrcoef(A)[np.triu_indices(len(A), 1)]
        cb = np.corrcoef(B)[np.triu_indices(len(B), 1)]
        return spearmanr(ca, cb)[0]

    print("\n" + "=" * 72)
    print("Coinciden los dos espacios de representacion?")
    print("=" * 72)
    idx = a.index.values
    print(f"   todas  : rho = {rsa(idx):+.3f}")
    if br.sum() > 5:
        print(f"   rotas  : rho = {rsa(idx[br]):+.3f}")
        print(f"   sanas  : rho = {rsa(idx[~br]):+.3f}")
        print("\n   Si el acuerdo cae en las rotas, 1p1 tambien representa mal")
        print("   esa quimica, no solo la puntua mal.")


if __name__ == "__main__":
    main()
