"""Forces from UMA on the four variants: the original question, done directly.

The question
------------
Are the mutations more or less favourable according to what UMA computes?

Why forces and not energies
---------------------------
Total energy depends on atomic composition. Arg has more atoms than Ser, so
E_total(S126R) and E_total(WT) are not comparable and any correlation with
them reads the molecular weight. A force is local, in eV/A, and does not
depend on how many atoms the system has. The force on the atoms around
position 126 in one variant IS comparable to the same region in another.

What the number means
---------------------
MOE placed the mutated sidechains on a rigid backbone, without letting the
surroundings respond. A residue that cannot sit there without strain carries
large residual forces. That is a crude proxy for the steric cost of the
substitution -- the same logic as clash detection in mutation scanning. It is
not a free energy, and it partly measures how well MOE placed the sidechain.

The control that makes it interpretable
---------------------------------------
Asking "is the force large?" has no answer. So each position is scored against
the force distribution of every other residue IN THE SAME STRUCTURE: the
z-score of the mutated position within its own protein. Each variant is its
own control and global differences normalise away.

Usage
-----
    python scripts/uma_forces.py --pdb-dir data/raw --out data/processed/uma_forces.csv
"""

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np

VARIANTS = {"WT (4)": "WT", "D314A (1)": "D314A",
            "S126R (2)": "S126R", "S126-D314A (1)": "DOUBLE"}
MUTATED = {"WT": [], "D314A": [314], "S126R": [126], "DOUBLE": [126, 314]}
WATCH = [126, 314]


def protonate(pdb_in, pdb_out, ph=7.4):
    """pdb2pqr adds the hydrogens an all-atom model needs and reports the charge."""
    pqr = str(pdb_out).replace(".pdb", ".pqr")
    cmd = (f'pdb2pqr30 --ff=AMBER --titration-state-method=propka --with-ph={ph} '
           f'--keep-chain --pdb-output="{pdb_out}" "{pdb_in}" "{pqr}" 2>&1')
    os.system(cmd)
    charge = 0.0
    if Path(pqr).exists():
        for ln in open(pqr):
            if ln.startswith(("ATOM", "HETATM")):
                parts = ln.split()
                try:
                    charge += float(parts[-2])
                except (ValueError, IndexError):
                    pass
    return pdb_out, int(round(charge))


def read_atoms(path, drop_het=True):
    syms, xyz, resnum = [], [], []
    for ln in open(path):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        if drop_het and ln.startswith("HETATM"):
            continue
        el = ln[76:78].strip()
        if not el:
            el = re.sub(r"[^A-Za-z]", "", ln[12:16].strip())[:1]
        syms.append(el.capitalize())
        xyz.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        resnum.append(int(ln[22:26]))
    return syms, np.array(xyz), np.array(resnum)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-dir", default="data/raw")
    ap.add_argument("--work", default="data/processed/protonated")
    ap.add_argument("--out", default="data/processed/uma_forces.csv")
    ap.add_argument("--model", default="uma-s-1p1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--keep-metal", action="store_true",
                    help="keep Fe and waters. Off by default: Fe in a high-spin "
                         "state is open-shell, which is exactly the regime "
                         "OMol25-class models are documented as untested on. "
                         "Position 126 sits 12.9 A away, so dropping it barely "
                         "touches the forces there; position 314 is 5.4 A away, "
                         "so for that one rerun with this flag and compare.")
    args = ap.parse_args()

    from ase import Atoms
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    import pandas as pd

    Path(args.work).mkdir(parents=True, exist_ok=True)
    predictor = pretrained_mlip.get_predict_unit(args.model, device=args.device)
    calc = FAIRChemCalculator(predictor, task_name="omol")
    print(f"loaded {args.model} on {args.device}\n")

    rows = []
    for stem, name in VARIANTS.items():
        src = Path(args.pdb_dir) / f"{stem}.pdb"
        if not src.exists():
            print(f"missing {src}, skipping"); continue
        prot = Path(args.work) / f"{name}_H.pdb"
        if not prot.exists():
            print(f"protonating {name} ...")
        _, charge = protonate(src, prot)

        syms, xyz, resnum = read_atoms(prot, drop_het=not args.keep_metal)
        print(f"{name}: {len(syms)} atoms, net charge {charge:+d}")

        atoms = Atoms(symbols=syms, positions=xyz)
        atoms.info.update({"charge": charge, "spin": 1})
        atoms.calc = calc
        f = atoms.get_forces()
        fmag = np.linalg.norm(f, axis=1)

        # force per residue, then the z-score of each watched position within
        # this structure's own distribution
        per_res = {}
        for r in np.unique(resnum):
            per_res[int(r)] = float(fmag[resnum == r].max())
        vals = np.array(list(per_res.values()))
        mu, sd = vals.mean(), vals.std()

        for pos in WATCH:
            if pos not in per_res:
                continue
            rows.append({
                "variant": name, "position": pos,
                "mutated_here": pos in MUTATED[name],
                "max_force_eV_per_A": round(per_res[pos], 4),
                "z_within_structure": round((per_res[pos] - mu) / sd, 3),
                "structure_median": round(float(np.median(vals)), 4),
                "n_atoms": len(syms), "net_charge": charge,
            })
        print(f"   median per-residue max force {np.median(vals):.4f} eV/A")
        for pos in WATCH:
            if pos in per_res:
                print(f"   position {pos}: {per_res[pos]:.4f} eV/A"
                      f"   z = {(per_res[pos]-mu)/sd:+.2f}"
                      f"   {'<- MUTATED' if pos in MUTATED[name] else ''}")
        print()

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {args.out}")
    print("\nThe comparison to make: does the z-score at a mutated position "
          "track the measured plateau (WT 60, DOUBLE 48, D314A 47, S126R 22)?")


if __name__ == "__main__":
    main()
