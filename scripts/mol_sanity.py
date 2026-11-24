"""Are the quantum descriptors physically sensible at all?

This runs before any modelling. UMA was trained on OMol25 to reproduce
wB97M-V/def2-TZVPD energies, but nothing guarantees that the charge
conditioning gives usable ionization energies on THIS chemistry. If the
numbers are nonsense, every correlation downstream is nonsense too, and it is
far cheaper to find out here.

Four checks:
  1. Ranges. Organic molecules have IE roughly 7-12 eV and eta roughly 3-7 eV.
     Numbers far outside that mean the Delta-SCF is not working.
  2. Sign of eta. eta <= 0 is unphysical; a nonzero rate means some molecules
     are being described badly.
  3. Named small molecules. Their IE can be checked against the literature.
  4. The composition trap. Total energy must correlate almost perfectly with
     atom count, and the derived descriptors must not. This is the check that
     the whole design rests on.

Usage:
    python scripts/mol_sanity.py        # on the analysis env
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

Q = ["IE_eV", "EA_eV", "chi_eV", "eta_eV", "omega_eV"]
TRAP = ["E_neutral_eV", "E_per_atom_eV"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/mol_quantum.csv")
    args = ap.parse_args()
    d = pd.read_csv(args.csv)
    print(f"{len(d)} moleculas\n")

    print("=" * 68)
    print("1. Rangos  (organicas tipicas: IE 7-12 eV, eta 3-7 eV, omega 0.5-4)")
    print("=" * 68)
    print(d[Q + TRAP + ["n_atoms"]].describe().T[
        ["mean", "std", "min", "25%", "50%", "75%", "max"]
    ].round(3).to_string())

    print("\n" + "=" * 68)
    print("2. Valores imposibles")
    print("=" * 68)
    bad_eta = int((d.eta_eV <= 0).sum())
    bad_ie = int((d.IE_eV <= 0).sum())
    nan_om = int(d.omega_eV.isna().sum())
    print(f"   eta <= 0        : {bad_eta} ({bad_eta/len(d)*100:.1f}%)   "
          f"<- dureza negativa, no tiene sentido fisico")
    print(f"   IE <= 0         : {bad_ie}   <- ionizar liberaria energia")
    print(f"   omega sin valor : {nan_om}")
    print(f"   radicales RDKit : {int((d.n_radical > 0).sum())}")
    print(f"   con carga formal: {int((d.charge != 0).sum())}")

    print("\n" + "=" * 68)
    print("3. Las moleculas mas chicas, para contrastar con literatura")
    print("=" * 68)
    small = d.nsmallest(10, "n_atoms")[
        ["smiles", "n_atoms", "charge", "IE_eV", "EA_eV", "eta_eV", "omega_eV"]]
    print(small.round(3).to_string(index=False))

    print("\n" + "=" * 68)
    print("4. La trampa de la composicion")
    print("=" * 68)
    print("   Spearman contra el numero de atomos:")
    for c in TRAP + Q:
        rho, p = spearmanr(d[c], d.n_atoms, nan_policy="omit")
        flag = ""
        if c in TRAP and abs(rho) > 0.9:
            flag = "  <- como debe ser: es un proxy del tamano"
        elif c in Q and abs(rho) > 0.5:
            flag = "  <- PROBLEMA: deberia haberse cancelado"
        print(f"      {c:16s} rho = {rho:+.3f}   p = {p:.1e}{flag}")

    print("\n" + "=" * 68)
    print("5. Correlacion entre los propios descriptores cuanticos")
    print("=" * 68)
    print(d[Q].corr(method="spearman").round(2).to_string())
    print("\n   (si omega es casi identico a chi, aporta poco por encima de el)")


if __name__ == "__main__":
    main()
