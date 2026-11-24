"""Forces from UMA around the mutated positions: the original question, done directly.

The question
------------
Are the mutations more or less favourable according to what UMA computes?

Why forces, not energies
------------------------
Total energy depends on composition. Arg has more atoms than Ser, so
E(S126R) and E(WT) are not comparable and any correlation with them reads the
molecular weight. A force is local, in eV/A, and does not care how many atoms
the system has.

Why spheres, not the whole protein
----------------------------------
The protonated protein is ~7,000 atoms and does not fit in 11 GB. It is also
far outside the size range OMol25 was trained on. A sphere around the position
of interest is both affordable and closer to the training regime.

What the number means
---------------------
MOE placed the mutated sidechains on a rigid backbone, without letting the
surroundings respond. A residue that cannot sit there without strain carries
large residual forces. That is a proxy for the steric cost of the substitution,
not a free energy, and it partly measures how well MOE placed the sidechain.

The control
-----------
"Is the force large?" has no answer on its own, so the same sphere calculation
runs at N random OTHER positions in the SAME structure. The target is reported
as a z-score within that null. Each variant is its own control.

Usage
-----
    python scripts/uma_forces.py --radius 10 --n-controls 20
"""

import argparse
import os
import random
import re
from pathlib import Path

import numpy as np

VARIANTS = {"WT (4)": "WT", "D314A (1)": "D314A",
            "S126R (2)": "S126R", "S126-D314A (1)": "DOUBLE"}
MUTATED = {"WT": [], "D314A": [314], "S126R": [126], "DOUBLE": [126, 314]}
WATCH = [126, 314]
BACKBONE = {"N", "CA", "C", "O", "OXT"}


def protonate(pdb_in, out_stem):
    pdb_out, pqr = f"{out_stem}.pdb", f"{out_stem}.pqr"
    if not Path(pqr).exists():
        os.system(f'pdb2pqr30 --ff=AMBER --titration-state-method=propka '
                  f'--with-ph=7.4 --pdb-output="{pdb_out}" "{pdb_in}" "{pqr}" '
                  f'>/dev/null 2>&1')
    return pqr


def read_pqr(path):
    """PQR carries per-atom charges, so the cluster charge is summed, not guessed."""
    atoms = []
    for ln in open(path):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        p = ln.split()
        try:
            x, y, z = float(p[-5]), float(p[-4]), float(p[-3])
            q = float(p[-2])
        except (ValueError, IndexError):
            continue
        name = p[2]
        el = re.sub(r"[^A-Za-z]", "", name)[:1].upper()
        resnum = int(re.sub(r"[^0-9-]", "", p[-6])) if not p[-6].lstrip("-").isdigit() \
            else int(p[-6])
        atoms.append({"name": name, "el": el, "resnum": resnum,
                      "xyz": np.array([x, y, z]), "q": q})
    return atoms


def cut_sphere(atoms, center_res, radius):
    """Whole residues with any atom within `radius` of the centre residue's sidechain."""
    core = [a for a in atoms if a["resnum"] == center_res]
    side = [a for a in core if a["name"] not in BACKBONE] or core
    if not core:
        return None
    centre = np.array([a["xyz"] for a in side])

    keep = set()
    for a in atoms:
        if np.min(np.linalg.norm(centre - a["xyz"], axis=1)) <= radius:
            keep.add(a["resnum"])

    sel = [a for a in atoms if a["resnum"] in keep]

    # cap broken peptide bonds with hydrogen along the original bond vector
    caps = []
    by_res = {}
    for a in sel:
        by_res.setdefault(a["resnum"], {})[a["name"]] = a
    allres = {a["resnum"]: None for a in atoms}
    for rn, d in by_res.items():
        if rn - 1 in allres and rn - 1 not in keep and "N" in d:
            prev_c = next((a for a in atoms
                           if a["resnum"] == rn - 1 and a["name"] == "C"), None)
            if prev_c is not None:
                v = prev_c["xyz"] - d["N"]["xyz"]
                v = v / np.linalg.norm(v)
                caps.append({"el": "H", "xyz": d["N"]["xyz"] + 1.01 * v, "q": 0.0,
                             "resnum": rn, "name": "HCAP"})
        if rn + 1 in allres and rn + 1 not in keep and "C" in d:
            nxt_n = next((a for a in atoms
                          if a["resnum"] == rn + 1 and a["name"] == "N"), None)
            if nxt_n is not None:
                v = nxt_n["xyz"] - d["C"]["xyz"]
                v = v / np.linalg.norm(v)
                caps.append({"el": "H", "xyz": d["C"]["xyz"] + 1.09 * v, "q": 0.0,
                             "resnum": rn, "name": "HCAP"})

    full = sel + caps
    charge = int(round(sum(a["q"] for a in full)))
    return full, charge, len(keep)


def max_force_on(atoms_list, forces, resnum):
    idx = [i for i, a in enumerate(atoms_list)
           if a["resnum"] == resnum and a["name"] not in BACKBONE
           and a["name"] != "HCAP"]
    if not idx:
        idx = [i for i, a in enumerate(atoms_list) if a["resnum"] == resnum]
    return float(np.linalg.norm(forces[idx], axis=1).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-dir", default="data/raw")
    ap.add_argument("--work", default="data/processed/protonated")
    ap.add_argument("--out", default="data/processed/uma_forces.csv")
# uma-s-1p2 y no 1p1: medimos que 1p1 calcula mal las energias de
# cationes en quimica organica comun (MAE 3.16 eV contra B3LYP, sesgo
# -2.90), y que 1p2 lo corrige (MAE 0.29 eV, rho 0.902) sin alterar las
# moleculas que 1p1 ya acertaba. Ver docs/resultados/05-uma-checkpoints.md
    ap.add_argument("--model", default="uma-s-1p2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--n-controls", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        from ase import Atoms
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        import pandas as pd
    except ModuleNotFoundError as e:
        raise SystemExit(f"falta {e.name}: corré esto en el entorno mlip")

    Path(args.work).mkdir(parents=True, exist_ok=True)
    predictor = pretrained_mlip.get_predict_unit(args.model, device=args.device)
    calc = FAIRChemCalculator(predictor, task_name="omol")
    print(f"{args.model} en {args.device}, esferas de {args.radius} A\n")

    rng = random.Random(args.seed)
    rows, controls = [], None

    for stem, name in VARIANTS.items():
        src = Path(args.pdb_dir) / f"{stem}.pdb"
        if not src.exists():
            print(f"falta {src}"); continue
        pqr = protonate(src, str(Path(args.work) / name))
        atoms_all = read_pqr(pqr)
        positions = sorted({a["resnum"] for a in atoms_all})

        if controls is None:   # same control positions for every variant
            pool = [p for p in positions if p not in WATCH]
            controls = rng.sample(pool, min(args.n_controls, len(pool)))
            print(f"posiciones control: {sorted(controls)}\n")

        print(f"--- {name}")
        for pos in WATCH + controls:
            cut = cut_sphere(atoms_all, pos, args.radius)
            if cut is None:
                continue
            sel, charge, nres = cut
            a = Atoms(symbols=[x["el"] for x in sel],
                      positions=np.array([x["xyz"] for x in sel]))
            a.info.update({"charge": charge, "spin": 1})
            a.calc = calc
            try:
                f = a.get_forces()
            except Exception as e:
                print(f"   pos {pos}: fallo ({type(e).__name__}); "
                      f"{len(sel)} atomos. Probá --radius menor.")
                continue
            fm = max_force_on(sel, f, pos)
            rows.append({"variant": name, "position": pos,
                         "is_target": pos in WATCH,
                         "mutated_here": pos in MUTATED[name],
                         "max_force_eV_per_A": round(fm, 4),
                         "n_atoms": len(sel), "n_residues": nres,
                         "charge": charge})
            if pos in WATCH:
                print(f"   pos {pos}: {fm:.4f} eV/A   ({len(sel)} atomos, "
                      f"carga {charge:+d}){'   <- MUTADA' if pos in MUTATED[name] else ''}")

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print("\n" + "=" * 62)
    print("z-score de cada posicion dentro de su propia estructura")
    print("=" * 62)
    for name in VARIANTS.values():
        g = df[df.variant == name]
        if g.empty:
            continue
        ctrl = g[~g.is_target].max_force_eV_per_A
        if len(ctrl) < 3:
            continue
        mu, sd = ctrl.mean(), ctrl.std()
        print(f"{name:7s} (control: mediana {ctrl.median():.3f}, n={len(ctrl)})")
        for pos in WATCH:
            r = g[(g.position == pos) & g.is_target]
            if r.empty:
                continue
            v = float(r.max_force_eV_per_A.iloc[0])
            tag = "MUTADA" if bool(r.mutated_here.iloc[0]) else "wt"
            print(f"    pos {pos}: {v:.3f} eV/A   z = {(v-mu)/sd:+.2f}   [{tag}]")
    print(f"\nescrito {args.out}")
    print("\nLa comparacion: el z-score de la posicion mutada, sigue el orden de "
          "las mesetas medidas? (WT 60, DOUBLE 48, D314A 47, S126R 22)")


if __name__ == "__main__":
    main()
