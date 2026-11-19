"""Build active-site cluster models from the variant PDB files.

Scope and honest limits
-----------------------
This module does the geometric part: select residues, cut them, collect the
metal and the catalytic waters, count the formal charge and report the size.

It does NOT protonate and it does NOT place a substrate. Both are required
before anything is handed to an MLIP, and both are decisions, not defaults:

* The input structures carry essentially no hydrogens (the WT file has none).
  An MLIP is an all-atom model, so every hydrogen has to be added explicitly.
* The four histidines in the cluster each have three possible protonation
  states. His61 and His214 coordinate the iron directly, which constrains
  theirs; the other two do not. The choice fixes the cluster's total charge.
* The structures are apo: only FE and water. There is no substrate to compute
  an interaction energy against.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

from .active_site import ACTIVE_SITE_RESIDUES, CATALYTIC_WATERS, CLUSTER_HISTIDINES

BACKBONE = {"N", "CA", "C", "O", "OXT"}

# Formal charge at pH 7 of each residue's sidechain, used to report the cluster
# charge. HIS is listed as 0 (neutral tautomer); flip to +1 for any His you
# decide to model as protonated (HIP).
SIDECHAIN_CHARGE = {"ASP": -1, "GLU": -1, "LYS": +1, "ARG": +1, "HIS": 0}


@dataclass
class Atom:
    record: str
    name: str
    resname: str
    resnum: int
    element: str
    xyz: Tuple[float, float, float]


def read_pdb(path: str) -> List[Atom]:
    out = []
    for ln in open(path):
        if ln.startswith(("ATOM", "HETATM")):
            el = ln[76:78].strip() or ln[12:16].strip()[0]
            out.append(Atom(ln[:6].strip(), ln[12:16].strip(), ln[17:20].strip(),
                            int(ln[22:26]), el,
                            (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))))
    return out


def distance(a, b) -> float:
    return math.dist(a, b)


def build_cluster(atoms: List[Atom],
                  extra_residues: Optional[List[int]] = None,
                  sidechain_only: bool = True) -> Dict:
    """Select the cluster atoms. Returns the atoms plus a report.

    `sidechain_only` cuts each residue at the Ca-Cb bond; the dangling valence
    must later be capped with hydrogen. Residues contiguous in sequence are
    reported so you can decide to keep their backbone instead of cutting twice.
    """
    wanted = {n for _, n in ACTIVE_SITE_RESIDUES} | set(extra_residues or [])
    picked, charge, composition = [], 0, {}

    for a in atoms:
        if a.record == "ATOM" and a.resnum in wanted:
            if sidechain_only and a.name in BACKBONE:
                continue
            picked.append(a)
        elif a.record == "HETATM" and (
                a.resname == "FE" or
                (a.resname == "HOH" and a.resnum in CATALYTIC_WATERS)):
            picked.append(a)

    seen = {}
    for a in atoms:
        if a.record == "ATOM" and a.resnum in wanted:
            seen[a.resnum] = a.resname
    for num, rn in seen.items():
        charge += SIDECHAIN_CHARGE.get(rn, 0)
        composition[f"{rn}{num}"] = rn
    charge += 2  # Fe(II); set to +3 for Fe(III)

    contiguous = sorted(n for n in seen if (n + 1) in seen)
    his_in_cluster = sorted(set(CLUSTER_HISTIDINES) & set(seen))
    n_heavy = sum(1 for a in picked if a.element != "H")

    return {
        "atoms": picked,
        "residues": composition,
        "n_atoms_heavy": n_heavy,
        "n_waters": len([a for a in picked if a.resname == "HOH"]),
        "formal_charge_no_H": charge,
        "histidines_to_decide": his_in_cluster,
        "contiguous_pairs": contiguous,
        "needs_protonation": True,
        "needs_substrate": True,
    }


def write_xyz(atoms: List[Atom], path: str, comment: str = "") -> None:
    with open(path, "w") as fh:
        fh.write(f"{len(atoms)}\n{comment}\n")
        for a in atoms:
            fh.write(f"{a.element:2s} {a.xyz[0]:12.6f} {a.xyz[1]:12.6f} {a.xyz[2]:12.6f}\n")
