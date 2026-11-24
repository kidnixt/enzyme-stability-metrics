"""Is UMA's ionization energy trustworthy on THIS chemistry?

Why this exists
---------------
The sanity check found that the bulk of UMA's Delta-SCF ionization energies
look like organic molecules should (median 8.6 eV), but a minority are clearly
wrong: sulfonate esters come out near 2 eV, small carbonyls near 5 eV, where
the real values are around 10. Bad MMFF geometries were ruled out - the
offending molecules carry LOWER residual force than the rest.

So the question is no longer "are the numbers pretty". It is whether UMA's
charged-state energies track real quantum chemistry. As a descriptor we do not
need them accurate, we need them MONOTONE: a feature that ranks molecules
correctly is useful even if every value is offset. A feature that ranks them
wrongly is noise, and no amount of modelling downstream will fix it.

The reference
-------------
Real DFT, from PySCF, on the SAME geometry and with the SAME definition of a
vertical ionization energy:

    neutral : restricted Kohn-Sham
    cation  : unrestricted Kohn-Sham, one electron removed, doublet
    IE      = E(cation) - E(neutral)

B3LYP is not the functional OMol25 was trained on (that is wB97M-V), so a
constant offset between the two is expected and uninteresting. What matters is
the rank correlation. B3LYP ionization energies are good to a few tenths of an
eV, which is far better than the ~5 eV discrepancies being investigated.

Only small molecules are used, because this reference has to be affordable.

Usage:
    pip install pyscf
    python scripts/mol_validate_ie.py --max-atoms 12 --n 25
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdf", default="data/processed/molecules.sdf")
    ap.add_argument("--uma", default="data/processed/mol_quantum.csv")
    ap.add_argument("--out", default="data/processed/ie_validation.csv")
    ap.add_argument("--max-atoms", type=int, default=12)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--basis", default="def2-SVP")
    ap.add_argument("--xc", default="B3LYP")
    args = ap.parse_args()

    from pyscf import gto, dft
    from pyscf.lib import logger
    from rdkit import Chem

    uma = pd.read_csv(args.uma).set_index("mol_id")
    mols = [m for m in Chem.SDMolSupplier(args.sdf, removeHs=False) if m is not None]

    # the smallest neutral, non-radical molecules: the reference must be cheap
    cand = [m for m in mols
            if m.GetNumAtoms() <= args.max_atoms
            and int(m.GetProp("charge")) == 0
            and int(m.GetProp("n_radical")) == 0]
    cand.sort(key=lambda m: m.GetNumAtoms())

    # deliberately mix the suspicious ones with ordinary ones, so the test can
    # distinguish "UMA is wrong everywhere" from "UMA is wrong on a subset"
    ids = [m.GetProp("mol_id") for m in cand]
    sus = [m for m in cand if uma.loc[m.GetProp("mol_id"), "IE_eV"] < 6.5]
    rest = [m for m in cand if uma.loc[m.GetProp("mol_id"), "IE_eV"] >= 6.5]
    pick = sus[: args.n // 2] + rest[: args.n - len(sus[: args.n // 2])]
    print(f"{len(cand)} candidatas de <= {args.max_atoms} atomos; "
          f"se usan {len(pick)} ({len(sus[:args.n//2])} sospechosas)")
    print(f"referencia: {args.xc}/{args.basis}\n")

    rows = []
    for i, m in enumerate(pick):
        mid = m.GetProp("mol_id")
        conf = m.GetConformer()
        atom = [(a.GetSymbol(), tuple(conf.GetAtomPosition(a.GetIdx())))
                for a in m.GetAtoms()]
        t0 = time.time()
        try:
            mol0 = gto.M(atom=atom, basis=args.basis, charge=0, spin=0,
                         unit="Angstrom", verbose=0)
            e0 = dft.RKS(mol0, xc=args.xc).density_fit().kernel()
            mol1 = gto.M(atom=atom, basis=args.basis, charge=1, spin=1,
                         unit="Angstrom", verbose=0)
            e1 = dft.UKS(mol1, xc=args.xc).density_fit().kernel()
        except Exception as e:
            print(f"   {mid} {m.GetProp('smiles')[:28]:28s} fallo: {type(e).__name__}")
            continue
        ie_dft = (e1 - e0) * 27.211386245988      # Hartree -> eV
        ie_uma = float(uma.loc[mid, "IE_eV"])
        rows.append({"mol_id": mid, "smiles": m.GetProp("smiles"),
                     "n_atoms": m.GetNumAtoms(),
                     "IE_uma_eV": ie_uma, "IE_dft_eV": ie_dft,
                     "diff_eV": ie_uma - ie_dft})
        print(f"   [{i+1:2d}/{len(pick)}] {m.GetProp('smiles')[:30]:30s} "
              f"UMA {ie_uma:6.2f}   DFT {ie_dft:6.2f}   "
              f"dif {ie_uma-ie_dft:+6.2f}   ({time.time()-t0:4.0f}s)")

    r = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(args.out, index=False)

    from scipy.stats import spearmanr, pearsonr
    rho, prho = spearmanr(r.IE_uma_eV, r.IE_dft_eV)
    pr, ppr = pearsonr(r.IE_uma_eV, r.IE_dft_eV)
    print("\n" + "=" * 66)
    print(f"n = {len(r)} moleculas")
    print(f"Spearman UMA vs DFT : rho = {rho:+.3f}   p = {prho:.2e}   <- lo que importa")
    print(f"Pearson             : r   = {pr:+.3f}   p = {ppr:.2e}")
    print(f"sesgo medio         : {r.diff_eV.mean():+.2f} eV")
    print(f"error absoluto medio: {r.diff_eV.abs().mean():.2f} eV")
    print(f"peor caso           : {r.diff_eV.abs().max():.2f} eV  "
          f"({r.loc[r.diff_eV.abs().idxmax(),'smiles']})")
    print("=" * 66)
    print("\nrho alto  -> el descriptor ordena bien y sirve pese al sesgo")
    print("rho bajo  -> es ruido, y la mitad molecular se detiene aca")
    print(f"\nescrito {args.out}")


if __name__ == "__main__":
    main()
