"""Does UMA's energy tell us anything about toxicity that chemistry does not?

The question has to be asked in that form. In a test set that is 92% toxicant,
almost any descriptor will look predictive, and the paper already has 818
classical descriptors. So "omega correlates with toxicity" is not a result.
The result is whether the quantum block adds anything ON TOP of the classical
block, and whether what it adds survives the obvious confounds.

Four feature blocks, and the third one is there on purpose:

    CLASSICAL  814 descriptors from the paper
    QUANTUM    IE, EA, chi, eta, omega        - composition cancels out
    TRAP       total energy, energy per atom  - composition does NOT cancel
    STRAIN     forces at the MMFF geometry    - quality of our conformer

TRAP is a negative control. Total energy is a near-perfect proxy for molecular
size, so if it "predicts" toxicity that is a demonstration of the confound,
not a finding. Keeping it in the comparison makes the confound visible instead
of letting it hide inside a combined feature set.

Statistics
----------
Repeated stratified CV on the 234 training molecules, with every feature set
scored on the SAME folds so the comparison is paired. The p-value uses the
Nadeau-Bengio correction, because repeated-CV folds share training data and an
uncorrected paired t-test on them is badly anticonservative.

Then the published 527-molecule test set, once, with a bootstrap interval.

Usage:
    python scripts/mol_analysis.py        # on the analysis env
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

QUANTUM = ["IE_eV", "EA_eV", "chi_eV", "eta_eV", "omega_eV"]
TRAP = ["E_neutral_eV", "E_per_atom_eV"]
STRAIN = ["max_force_eV_per_A", "rms_force_eV_per_A"]


def load(desc_csv, quant_csv):
    d = pd.read_csv(desc_csv)
    q = pd.read_csv(quant_csv)
    q = q.drop(columns=[c for c in ["smiles", "y", "split"] if c in q.columns])
    m = d.merge(q, on="mol_id", how="inner")
    classical = [c for c in m.columns if c.startswith(("rdkit.", "mordred."))]
    m[classical] = m[classical].replace([np.inf, -np.inf], np.nan)
    m[QUANTUM + TRAP + STRAIN] = m[QUANTUM + TRAP + STRAIN].replace(
        [np.inf, -np.inf], np.nan)
    return m, classical


def pipe(kind, cols):
    pre = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("var", VarianceThreshold(1e-12)),
                    ("sc", StandardScaler())])
    if kind == "lr":
        clf = LogisticRegression(penalty="l2", C=0.1, max_iter=5000,
                                 class_weight="balanced")
    else:
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                             max_leaf_nodes=15,
                                             l2_regularization=1.0,
                                             random_state=0)
    return Pipeline([("pre", ColumnTransformer([("x", pre, cols)])), ("clf", clf)])


def nadeau_bengio_p(diffs, n_train, n_test):
    """Paired t on CV folds, with the variance correction they share training
    data demands. Without it the p-value is meaningless."""
    d = np.asarray(diffs, float)
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return np.nan, np.nan
    corrected_var = d.var(ddof=1) * (1.0 / n + n_test / n_train)
    t = d.mean() / np.sqrt(corrected_var)
    return float(t), float(2 * stats.t.sf(abs(t), df=n - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desc", default="data/processed/mol_descriptors.csv")
    ap.add_argument("--quant", default="data/processed/mol_quantum.csv")
    ap.add_argument("--model", default="lr", choices=["lr", "gb"])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="data/processed/mol_analysis.csv")
    args = ap.parse_args()

    m, classical = load(args.desc, args.quant)
    tr = m[m.split == "train"].reset_index(drop=True)
    te = m[m.split == "test"].reset_index(drop=True)
    print(f"entrenamiento n={len(tr)} ({tr.y.mean()*100:.0f}% toxico), "
          f"test n={len(te)} ({te.y.mean()*100:.0f}% toxico)")
    print(f"{len(classical)} descriptores clasicos, {len(QUANTUM)} cuanticos\n")

    # ---------------- 1. univariate
    print("=" * 72)
    print("1. Cada descriptor cuantico por separado contra la toxicidad")
    print("=" * 72)
    print(f"   {'descriptor':18s} {'AUC train':>10s} {'p train':>10s} "
          f"{'AUC test':>10s} {'p test':>10s}")
    uni = []
    for c in QUANTUM + TRAP + STRAIN:
        row = {"descriptor": c}
        cells = []
        for nm, g in (("train", tr), ("test", te)):
            a, b = g.loc[g.y == 1, c].dropna(), g.loc[g.y == 0, c].dropna()
            if len(a) < 3 or len(b) < 3:
                cells += ["      n/a", "      n/a"]; continue
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            auc = u / (len(a) * len(b))        # AUC of the descriptor as a score
            row[f"auc_{nm}"], row[f"p_{nm}"] = auc, p
            cells += [f"{auc:10.3f}", f"{p:10.1e}"]
        uni.append(row)
        mark = ""
        if c in TRAP:
            mark = "   <- control negativo"
        print(f"   {c:18s} " + " ".join(cells) + mark)
    print("\n   AUC 0.5 = sin informacion. Lejos de 0.5 en AMBOS conjuntos es")
    print("   lo unico que cuenta; en uno solo puede ser el desbalance.")

    # ---------------- 2. confound
    print("\n" + "=" * 72)
    print("2. Es omega solo el tamano y la lipofilia disfrazados?")
    print("=" * 72)
    conf = [c for c in ["rdkit.MolWt", "rdkit.MolLogP"] if c in m.columns]
    if conf:
        sub = m[[*conf, "omega_eV", "y"]].dropna()
        Z = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in conf])
        def resid(v):
            beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
            return v - Z @ beta
        ro, po = stats.spearmanr(sub.omega_eV, sub.y)
        rr, pr = stats.spearmanr(resid(sub.omega_eV.values),
                                 resid(sub.y.values.astype(float)))
        print(f"   omega vs toxicidad, crudo    : rho = {ro:+.3f}  p = {po:.2e}")
        print(f"   controlando {', '.join(conf)}: rho = {rr:+.3f}  p = {pr:.2e}")
        print("\n   Si el segundo se derrumba, lo que veia era peso y lipofilia.")
    else:
        print("   faltan MolWt/MolLogP en la matriz")

    # ---------------- 3. nested models
    print("\n" + "=" * 72)
    print(f"3. Aporta algo sobre los clasicos?  (modelo: {args.model})")
    print("=" * 72)
    sets = {
        "CLASICOS": classical,
        "CUANTICOS": QUANTUM,
        "TRAMPA": TRAP,
        "CLASICOS+CUANTICOS": classical + QUANTUM,
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats,
                                 random_state=0)
    folds = list(cv.split(tr[classical + QUANTUM + TRAP], tr.y))
    scores = {k: [] for k in sets}
    for name, cols in sets.items():
        for itr, ite in folds:
            p = pipe(args.model, cols)
            p.fit(tr.iloc[itr], tr.y.iloc[itr])
            s = p.predict_proba(tr.iloc[ite])[:, 1]
            scores[name].append(roc_auc_score(tr.y.iloc[ite], s))
    print(f"   {'conjunto':22s} {'AUC medio':>10s} {'sd':>7s}")
    for k, v in scores.items():
        v = np.array(v)
        mark = "   <- control negativo" if k == "TRAMPA" else ""
        print(f"   {k:22s} {v.mean():10.3f} {v.std(ddof=1):7.3f}{mark}")

    d = np.array(scores["CLASICOS+CUANTICOS"]) - np.array(scores["CLASICOS"])
    t, p = nadeau_bengio_p(d, n_train=len(tr) * 0.8, n_test=len(tr) * 0.2)
    print(f"\n   ganancia de sumar los cuanticos: {d.mean():+.4f} AUC")
    print(f"   t corregido Nadeau-Bengio = {t:+.2f}   p = {p:.3f}")
    print("   (la correccion es obligatoria: los pliegues comparten datos de")
    print("    entrenamiento y un t pareado crudo daria p absurdamente chico)")

    # ---------------- 4. external test set
    print("\n" + "=" * 72)
    print("4. Conjunto de test publicado, una sola vez")
    print("=" * 72)
    rng = np.random.default_rng(0)
    preds, rows = {}, []
    for name, cols in sets.items():
        p = pipe(args.model, cols)
        p.fit(tr, tr.y)
        s = p.predict_proba(te)[:, 1]
        preds[name] = s
        yhat = (s >= 0.5).astype(int)
        rows.append({"conjunto": name,
                     "AUC": roc_auc_score(te.y, s),
                     "MCC": matthews_corrcoef(te.y, yhat),
                     "bal_acc": balanced_accuracy_score(te.y, yhat)})
    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))
    print(f"\n   referencia: decir 'toxico' siempre da exactitud "
          f"{te.y.mean()*100:.0f}%, AUC 0.5 y MCC 0.0")

    idx = np.arange(len(te))
    dif = []
    for _ in range(args.boot):
        b = rng.choice(idx, len(idx), replace=True)
        if te.y.values[b].std() == 0:
            continue
        dif.append(roc_auc_score(te.y.values[b], preds["CLASICOS+CUANTICOS"][b])
                   - roc_auc_score(te.y.values[b], preds["CLASICOS"][b]))
    lo, hi = np.percentile(dif, [2.5, 97.5])
    print(f"\n   diferencia AUC (clasicos+cuanticos) - clasicos: "
          f"{np.mean(dif):+.4f}")
    print(f"   IC 95% bootstrap: [{lo:+.4f}, {hi:+.4f}]")
    print("   Si el intervalo cruza el cero, los cuanticos no aportan.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(uni).to_csv(args.out, index=False)
    res.to_csv(args.out.replace(".csv", "_modelos.csv"), index=False)
    print(f"\nescritos {args.out} y {args.out.replace('.csv','_modelos.csv')}")


if __name__ == "__main__":
    main()
