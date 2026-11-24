"""Is the bad cation energy a property of UMA, or only of uma-s-1p1?

The validation found uma-s-1p1 underestimating ionization energies by about
5 eV on a subset of molecules, with the error localised in the cation branch:
EA comes out right, IE does not. Before concluding anything about UMA as a
family, the obvious check is whether a newer checkpoint behaves differently.
Charge conditioning is exactly the kind of thing a point release fixes.

The molecule set is chosen to separate two hypotheses:
  "broken"  - molecules uma-s-1p1 gets badly wrong
  "fine"    - molecules it already matches literature on
A checkpoint that fixes the first group without disturbing the second is the
one to switch to. A checkpoint that changes both is suspicious for a different
reason.

Usage:
    python scripts/mol_compare_checkpoints.py            # on the mlip env
    python scripts/mol_compare_checkpoints.py --models uma-s-1p1,uma-s-1p2p1
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# smiles -> (group, reference IE in eV). References are B3LYP/def2-SVP from
# scripts/mol_validate_ie.py where available, otherwise left as NaN and filled
# from data/processed/ie_validation.csv when that file exists.
PROBE = [
    ("C=O",              "rota"),
    ("CC=O",             "rota"),
    ("C=CCl",            "rota"),
    ("CCO",              "rota"),
    ("C1CO1",            "rota"),
    ("COS(C)(=O)=O",     "rota"),
    ("CC#N",             "sana"),
    ("ClC(Cl)(Cl)Cl",    "sana"),
    ("S=C=S",            "sana"),
    ("N#CC(Cl)Cl",       "sana"),
    ("N#CCCl",           "sana"),
    ("CO",               "rota"),
]

DEFAULT_MODELS = "uma-s-1p1,uma-s-1p2,uma-s-1p2p1,uma-m-1p1"


def multiplicity(n_elec):
    return 1 if n_elec % 2 == 0 else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdf", default="data/processed/molecules.sdf")
    ap.add_argument("--ref", default="data/processed/ie_validation.csv")
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="data/processed/checkpoint_comparison.csv")
    args = ap.parse_args()

    from ase import Atoms
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    from rdkit import Chem

    want = {s: g for s, g in PROBE}
    mols = {}
    for m in Chem.SDMolSupplier(args.sdf, removeHs=False):
        if m is None:
            continue
        s = m.GetProp("smiles")
        if s in want and s not in mols:
            mols[s] = m
    missing = [s for s in want if s not in mols]
    if missing:
        print(f"no estan en el SDF: {missing}")
    print(f"{len(mols)} moleculas sonda\n")

    ref = {}
    if Path(args.ref).exists():
        rv = pd.read_csv(args.ref)
        ref = dict(zip(rv.smiles, rv.IE_dft_eV))
        print(f"referencia DFT disponible para {len(ref)} de ellas\n")

    rows = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"--- {name}")
        try:
            predictor = pretrained_mlip.get_predict_unit(name, device=args.device)
            calc = FAIRChemCalculator(predictor, task_name="omol")
        except Exception as e:
            print(f"    no se pudo cargar: {type(e).__name__}: "
                  f"{str(e)[:140]}\n")
            continue
        for s, m in mols.items():
            q0 = int(m.GetProp("charge"))
            n_elec = int(m.GetProp("n_elec"))
            conf = m.GetConformer()
            pos = np.array([list(conf.GetAtomPosition(i))
                            for i in range(m.GetNumAtoms())])
            sym = [a.GetSymbol() for a in m.GetAtoms()]
            e = {}
            try:
                for dq in (0, +1, -1):
                    a = Atoms(symbols=sym, positions=pos)
                    a.info.update({"charge": q0 + dq,
                                   "spin": multiplicity(n_elec - dq)})
                    a.calc = calc
                    e[dq] = float(a.get_potential_energy())
            except Exception as ex:
                print(f"    {s:18s} fallo: {type(ex).__name__}")
                continue
            ie = e[+1] - e[0]
            rows.append({"model": name, "smiles": s, "group": want[s],
                         "IE_eV": ie, "EA_eV": e[0] - e[-1],
                         "IE_dft_eV": ref.get(s, np.nan)})
            d = ref.get(s)
            extra = f"   DFT {d:6.2f}   dif {ie-d:+6.2f}" if d == d and d else ""
            print(f"    {s:18s} [{want[s]}] IE {ie:7.2f}{extra}")
        print()

    r = pd.DataFrame(rows)
    if r.empty:
        raise SystemExit("ningun modelo pudo correr")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(args.out, index=False)

    print("=" * 70)
    print("IE por modelo y grupo (mediana)")
    print("=" * 70)
    print(r.pivot_table(index="model", columns="group",
                        values="IE_eV", aggfunc="median").round(2).to_string())

    sub = r.dropna(subset=["IE_dft_eV"])
    if len(sub):
        print("\n" + "=" * 70)
        print("Contra la referencia DFT")
        print("=" * 70)
        from scipy.stats import spearmanr
        print(f"{'model':16s} {'n':>3s} {'sesgo':>8s} {'MAE':>8s} {'rho':>8s}")
        for name, g in sub.groupby("model"):
            rho = spearmanr(g.IE_eV, g.IE_dft_eV)[0] if len(g) > 2 else np.nan
            d = g.IE_eV - g.IE_dft_eV
            print(f"{name:16s} {len(g):3d} {d.mean():+8.2f} "
                  f"{d.abs().mean():8.2f} {rho:+8.3f}")
        print("\nEl que sirve es el que tenga MAE chico Y rho alto.")
    print(f"\nescrito {args.out}")


if __name__ == "__main__":
    main()
