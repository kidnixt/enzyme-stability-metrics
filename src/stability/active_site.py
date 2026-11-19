"""Active-site definition for E. coli cytosine deaminase, as provided by the lab.

Residue numbers are canonical (PDB 1K6W numbering).
"""

ACTIVE_SITE_RESIDUES = [
    ("PRO", 60), ("HIS", 61), ("ILE", 62), ("LEU", 64),
    ("VAL", 213), ("HIS", 214), ("CYS", 215),
    ("SER", 245), ("HIS", 246), ("THR", 247),
    ("HIS", 312), ("ASP", 313), ("ASP", 314),
]

CATALYTIC_WATERS = [740, 742, 888, 1000]

METAL = "FE"  # as modelled in 1K6W; Zn-substituted structures: 3R0D, 3O7U

# Each of these can be HID / HIE / HIP. The choice fixes the cluster's total
# charge and its H-bond network. Decide from the structure, never by default.
CLUSTER_HISTIDINES = [61, 214, 246, 312]

MUTATED_POSITIONS = {"D314A": 314, "S126R": 126}


def is_active_site(position):
    return position in {p for _, p in ACTIVE_SITE_RESIDUES}


def mutation_classes():
    """D314 is in the active site, S126 is not. That asymmetry is the hypothesis."""
    return {name: ("active_site" if is_active_site(pos) else "distal")
            for name, pos in MUTATED_POSITIONS.items()}
