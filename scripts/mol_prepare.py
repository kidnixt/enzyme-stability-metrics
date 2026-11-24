"""SMILES -> 3D geometry with hydrogens, ready for ASE and for descriptors.

The TIRESIA files are two sets with different roles and very different class
balance, so the split is recorded per molecule and never shuffled together:

    ci2c01126_si_002.txt   234 chemicals, CAESAR training set, 68% toxicant
    ci2c01126_si_003.txt   527 chemicals, test set,            92% toxicant

Output is one SDF holding every molecule that survived, each carrying its
SMILES, label, split, formal charge and electron count as properties. Anything
that failed is written to a separate CSV rather than dropped silently: a
pipeline that quietly loses molecules will quietly bias the result.

Usage:
    python scripts/mol_prepare.py
"""

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

SOURCES = [
    ("data/raw/ci2c01126_si_002.txt", "train"),   # CAESAR, 234
    ("data/raw/ci2c01126_si_003.txt", "test"),    # 527
]


def load(path, split):
    df = pd.read_csv(path)
    df.columns = ["smiles", "label"]
    df["label"] = df.label.str.strip()
    df["y"] = (df.label == "Toxicant").astype(int)
    df["split"] = split
    return df


def embed(mol, seed):
    """ETKDG then MMFF. Returns None if no conformer can be produced."""
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.maxIterations = 1000
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
    except Exception:
        pass        # an unoptimised conformer still beats no conformer
    return mol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/molecules.sdf")
    ap.add_argument("--failures", default="data/processed/molecules_failed.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.concat([load(p, s) for p, s in SOURCES], ignore_index=True)
    print(f"leidos {len(df)} SMILES")
    for s, g in df.groupby("split"):
        print(f"   {s}: n={len(g)}, toxicos={g.y.sum()} ({g.y.mean()*100:.0f}%)")

    # duplicates across the two sets would leak the test set into training
    dup = df[df.duplicated("smiles", keep=False)].sort_values("smiles")
    if len(dup):
        cross = dup.groupby("smiles").split.nunique()
        n_cross = int((cross > 1).sum())
        print(f"\nSMILES duplicados: {len(dup)} filas, "
              f"{n_cross} presentes en AMBOS conjuntos")
        if n_cross:
            print("   OJO: eso es fuga del test al entrenamiento en el diseño publicado.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(args.out)
    failures, kept = [], 0
    uncharger = rdMolStandardize.Uncharger()   # only used to report, not applied

    for i, r in df.iterrows():
        mol = Chem.MolFromSmiles(r.smiles)
        if mol is None:
            failures.append({**r.to_dict(), "reason": "parse"})
            continue
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:                      # none in this dataset, kept as a guard
            mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
            note = "multifragment"
        else:
            note = ""
        mol = Chem.AddHs(mol)
        mol = embed(mol, args.seed + i)
        if mol is None:
            failures.append({**r.to_dict(), "reason": "embed"})
            continue

        charge = Chem.GetFormalCharge(mol)
        n_elec = sum(a.GetAtomicNum() for a in mol.GetAtoms()) - charge
        n_rad = Descriptors.NumRadicalElectrons(mol)

        mol.SetProp("_Name", f"mol{i:04d}")
        mol.SetProp("mol_id", f"mol{i:04d}")
        mol.SetProp("smiles", r.smiles)
        mol.SetProp("label", r.label)
        mol.SetProp("y", str(int(r.y)))
        mol.SetProp("split", r.split)
        mol.SetProp("charge", str(charge))
        mol.SetProp("n_elec", str(n_elec))
        mol.SetProp("n_atoms", str(mol.GetNumAtoms()))
        mol.SetProp("n_radical", str(n_rad))
        mol.SetProp("note", note)
        writer.write(mol)
        kept += 1
    writer.close()

    print(f"\nescritos {kept} en {args.out}")
    if failures:
        f = pd.DataFrame(failures)
        f.to_csv(args.failures, index=False)
        print(f"fallaron {len(f)} -> {args.failures}")
        print(f.reason.value_counts().to_string())
    else:
        print("sin fallos")


if __name__ == "__main__":
    main()
