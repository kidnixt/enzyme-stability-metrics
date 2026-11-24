"""Energetic descriptors from UMA, built so that they are comparable across
molecules.

The trap
--------
A total energy is not comparable between two different molecules: it scales
with how many atoms and which elements they contain, so any correlation with
it reads the molecular weight. Same mistake as comparing E(S126R) with E(WT).

The fix
-------
OMol25 conditions on charge, so three single points on the SAME geometry give
energy DIFFERENCES, in which the composition cancels:

    IE = E(q+1) - E(q)      vertical ionization energy
    EA = E(q)   - E(q-1)    vertical electron affinity

and from those, the conceptual-DFT quantities:

    chi   = (IE + EA) / 2           electronegativity
    eta   = (IE - EA) / 2           chemical hardness
    omega = chi^2 / (2 eta)         Parr electrophilicity index

omega is the one with a mechanistic story for toxicity: an electrophile is a
molecule that wants electrons, and the damaging reactions of reactive
toxicants are attacks on the electron-rich parts of DNA and proteins. It is
an established QSAR descriptor, not something invented here.

Spin is set to the lowest multiplicity consistent with the electron count:
singlet when even, doublet when odd. Molecules RDKit flags as radicals are
marked in the output so they can be excluded later.

Writes incrementally and resumes, because this is ~3x761 forward passes.

Usage:
    python scripts/mol_quantum.py              # on the mlip env
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


def multiplicity(n_elec):
    return 1 if n_elec % 2 == 0 else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdf", default="data/processed/molecules.sdf")
    ap.add_argument("--out", default="data/processed/mol_quantum.csv")
    ap.add_argument("--model", default="uma-s-1p1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="0 = todas")
    args = ap.parse_args()

    from ase import Atoms
    from fairchem.core import pretrained_mlip, FAIRChemCalculator

    done = set()
    out = Path(args.out)
    if out.exists():
        done = set(pd.read_csv(out).mol_id)
        print(f"reanudando: ya hay {len(done)} moleculas en {out}")

    predictor = pretrained_mlip.get_predict_unit(args.model, device=args.device)
    calc = FAIRChemCalculator(predictor, task_name="omol")

    supplier = Chem.SDMolSupplier(args.sdf, removeHs=False)
    mols = [m for m in supplier if m is not None]
    if args.limit:
        mols = mols[: args.limit]
    print(f"{len(mols)} moleculas en el SDF, {len(mols)-len(done)} por calcular\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not out.exists()
    n_err = 0

    for k, m in enumerate(mols):
        mid = m.GetProp("mol_id")
        if mid in done:
            continue
        q0 = int(m.GetProp("charge"))
        n_elec = int(m.GetProp("n_elec"))
        conf = m.GetConformer()
        pos = np.array([list(conf.GetAtomPosition(i))
                        for i in range(m.GetNumAtoms())])
        sym = [a.GetSymbol() for a in m.GetAtoms()]

        energies, forces0 = {}, None
        try:
            for dq in (0, +1, -1):
                a = Atoms(symbols=sym, positions=pos)
                a.info.update({"charge": q0 + dq,
                               "spin": multiplicity(n_elec - dq)})
                a.calc = calc
                energies[dq] = float(a.get_potential_energy())
                if dq == 0:
                    forces0 = a.get_forces()
        except Exception as e:
            n_err += 1
            print(f"   {mid}: fallo ({type(e).__name__}) con {len(sym)} atomos")
            continue

        ie = energies[+1] - energies[0]
        ea = energies[0] - energies[-1]
        chi = (ie + ea) / 2.0
        eta = (ie - ea) / 2.0
        # eta <= 0 is unphysical and means the three states are not describing
        # one molecule's frontier orbitals; flagged rather than silently used
        omega = (chi ** 2) / (2.0 * eta) if eta > 1e-6 else np.nan

        fmag = np.linalg.norm(forces0, axis=1)
        row = {
            "mol_id": mid,
            "smiles": m.GetProp("smiles"),
            "y": int(m.GetProp("y")),
            "split": m.GetProp("split"),
            "charge": q0,
            "n_atoms": int(m.GetProp("n_atoms")),
            "n_radical": int(m.GetProp("n_radical")),
            "E_neutral_eV": energies[0],
            "E_per_atom_eV": energies[0] / len(sym),
            "IE_eV": ie,
            "EA_eV": ea,
            "chi_eV": chi,
            "eta_eV": eta,
            "omega_eV": omega,
            "eta_nonpositive": bool(eta <= 1e-6),
            "max_force_eV_per_A": float(fmag.max()),
            "rms_force_eV_per_A": float(np.sqrt((fmag ** 2).mean())),
        }
        pd.DataFrame([row]).to_csv(out, mode="a", header=header_needed, index=False)
        header_needed = False

        if (k + 1) % 25 == 0:
            print(f"   {k+1}/{len(mols)}")

    print(f"\nescrito {out}   (fallos: {n_err})")


if __name__ == "__main__":
    main()
