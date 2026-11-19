"""Place a substrate into the apo variant structures by superposition.

All four variant structures are apo: iron and water only. An interaction energy
needs a substrate in the site, so one is transplanted from a holo structure of
the same enzyme.

Donor structure: 1RA0 (bacterial cytosine deaminase D314G) carries FPY, the
4S-5-fluoro-4-hydroxy-3,4-dihydropyrimidin-2(1H)-one intermediate, together
with the iron, at 1.12 A. That it comes from a mutant does not matter: only the
ligand coordinates are taken, after superposition.

Superposition is done on the active-site CA atoms rather than the whole chain.
Local superposition places the ligand better, and the RMSD of both is reported
so the choice is visible rather than implicit.

Usage
-----
    python scripts/transplant_ligand.py --donor data/raw/1RA0.pdb --ligand FPY
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stability.active_site import ACTIVE_SITE_RESIDUES


def read_pdb(path):
    rows = []
    for ln in open(path):
        if ln.startswith(("ATOM", "HETATM")):
            rows.append({
                "line": ln.rstrip("\n"),
                "record": ln[:6].strip(),
                "name": ln[12:16].strip(),
                "resname": ln[17:20].strip(),
                "chain": ln[21],
                "resnum": int(ln[22:26]),
                "xyz": np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]),
            })
    return rows


def kabsch(P, Q):
    """Rotation+translation mapping P onto Q. Returns (R, t, rmsd)."""
    cp, cq = P.mean(0), Q.mean(0)
    H = (P - cp).T @ (Q - cq)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cq - R @ cp
    rmsd = float(np.sqrt((((P @ R.T + t) - Q) ** 2).sum(1).mean()))
    return R, t, rmsd


def ca_map(rows):
    return {r["resnum"]: r["xyz"] for r in rows if r["record"] == "ATOM" and r["name"] == "CA"}


def superpose(donor, target, subset=None):
    dm, tm = ca_map(donor), ca_map(target)
    common = sorted(set(dm) & set(tm))
    if subset is not None:
        common = [n for n in common if n in subset]
    if len(common) < 3:
        raise SystemExit(f"only {len(common)} matched CA atoms; cannot superpose")
    P = np.array([dm[n] for n in common])
    Q = np.array([tm[n] for n in common])
    R, t, rmsd = kabsch(P, Q)
    return R, t, rmsd, len(common)


def pdb_line_with_xyz(line, xyz):
    return f"{line[:30]}{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{line[54:]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", required=True, help="holo PDB carrying the ligand")
    ap.add_argument("--ligand", default="FPY", help="residue name of the ligand")
    ap.add_argument("--targets", nargs="*", default=None,
                    help="variant PDBs; default is every .pdb in data/raw except the donor")
    ap.add_argument("--outdir", default="data/processed/with_ligand")
    args = ap.parse_args()

    donor = read_pdb(args.donor)
    lig = [r for r in donor if r["resname"] == args.ligand]
    if not lig:
        names = sorted({r["resname"] for r in donor if r["record"] == "HETATM"})
        raise SystemExit(f"ligand {args.ligand} not found. HETATM present: {names}")
    print(f"donor {args.donor}: ligand {args.ligand} with {len(lig)} atoms")

    targets = args.targets or [str(p) for p in sorted(Path("data/raw").glob("*.pdb"))
                               if Path(p).resolve() != Path(args.donor).resolve()]
    site = {n for _, n in ACTIVE_SITE_RESIDUES}
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for tpath in targets:
        target = read_pdb(tpath)
        R_all, t_all, rmsd_all, n_all = superpose(donor, target)
        R, t, rmsd, n = superpose(donor, target, subset=site)
        print(f"\n{Path(tpath).name}")
        print(f"   global superposition: {n_all} CA, RMSD {rmsd_all:.2f} A")
        print(f"   active-site only:     {n} CA, RMSD {rmsd:.2f} A   <- used")

        out = outdir / (Path(tpath).stem.replace(" ", "_") + f"_with_{args.ligand}.pdb")
        with open(out, "w") as fh:
            for r in target:
                fh.write(r["line"] + "\n")
            for r in lig:
                fh.write(pdb_line_with_xyz(r["line"], R @ r["xyz"] + t) + "\n")
            fh.write("END\n")
        print(f"   wrote {out}")


if __name__ == "__main__":
    main()
