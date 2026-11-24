"""The classical baseline: the same 818 descriptors the TIRESIA paper used.

ci2c01126_si_004.txt is not a feature matrix, it is the LIST of descriptor
names they computed: 606 from Mordred and 212 from RDKit. So the matrix is
rebuilt here by name, which keeps the baseline theirs rather than one invented
for the occasion. Names that cannot be matched in the installed versions are
reported, not silently dropped.

This matrix is what the quantum descriptors have to beat. Without it, "omega
correlates with toxicity" says nothing: almost everything correlates with
toxicity in a set that is 92% toxic, and the real question is whether omega
carries anything the 818 do not already have.

Usage:
    python scripts/mol_descriptors.py        # on the analysis env
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")


def wanted_names(path):
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["Descriptors"])
    df["Descriptors"] = df.Descriptors.str.strip()
    df = df[df.Descriptors != ""]
    rd = df[df.Library.str.strip() == "RDKit"].Descriptors.tolist()
    mo = df[df.Library.str.strip() == "Mordred"].Descriptors.tolist()
    return rd, mo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdf", default="data/processed/molecules.sdf")
    ap.add_argument("--names", default="data/raw/ci2c01126_si_004.txt")
    ap.add_argument("--out", default="data/processed/mol_descriptors.csv")
    args = ap.parse_args()

    rd_want, mo_want = wanted_names(args.names)
    print(f"el paper lista {len(rd_want)} RDKit y {len(mo_want)} Mordred")

    mols = [m for m in Chem.SDMolSupplier(args.sdf, removeHs=False) if m is not None]
    meta = pd.DataFrame([{"mol_id": m.GetProp("mol_id"),
                          "smiles": m.GetProp("smiles"),
                          "y": int(m.GetProp("y")),
                          "split": m.GetProp("split")} for m in mols])
    print(f"{len(mols)} moleculas")

    # ---- RDKit
    have_rd = {n: f for n, f in Descriptors._descList}
    miss_rd = [n for n in rd_want if n not in have_rd]
    use_rd = [n for n in rd_want if n in have_rd]
    rows = []
    for m in mols:
        r = {}
        for n in use_rd:
            try:
                v = have_rd[n](m)
            except Exception:
                v = np.nan
            r[f"rdkit.{n}"] = v
        rows.append(r)
    rd_df = pd.DataFrame(rows)
    print(f"RDKit: {len(use_rd)} calculados, {len(miss_rd)} sin correspondencia")
    if miss_rd:
        print("   faltan:", ", ".join(miss_rd[:12]), "..." if len(miss_rd) > 12 else "")

    # ---- Mordred (2D only: the 3D ones would depend on the conformer we built,
    # which the paper did not have)
    try:
        from mordred import Calculator, descriptors as mdesc
        calc = Calculator(mdesc, ignore_3D=True)
        all_names = [str(d) for d in calc.descriptors]
        keep = [d for d in calc.descriptors if str(d) in set(mo_want)]
        miss_mo = sorted(set(mo_want) - set(all_names))
        calc2 = Calculator(keep, ignore_3D=True)
        print(f"Mordred: {len(keep)} calculados, {len(miss_mo)} sin correspondencia"
              f" (calculando, tarda unos minutos)")
        mo_df = calc2.pandas([Chem.RemoveHs(m) for m in mols], quiet=True)
        mo_df = mo_df.apply(pd.to_numeric, errors="coerce")
        mo_df.columns = [f"mordred.{c}" for c in mo_df.columns]
        if miss_mo:
            print("   faltan:", ", ".join(miss_mo[:12]),
                  "..." if len(miss_mo) > 12 else "")
    except ModuleNotFoundError:
        print("Mordred no instalado: solo se calculan los RDKit")
        mo_df = pd.DataFrame(index=range(len(mols)))

    X = pd.concat([meta, rd_df, mo_df], axis=1)

    n_all_nan = int(X.filter(regex=r"^(rdkit|mordred)\.").isna().all().sum())
    print(f"\n{X.shape[1] - 4} descriptores, de los cuales {n_all_nan} "
          f"son NaN en todas las moleculas")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    X.to_csv(args.out, index=False)
    print(f"escrito {args.out}")


if __name__ == "__main__":
    main()
