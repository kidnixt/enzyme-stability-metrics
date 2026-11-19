"""Sequences of the four E. coli cytosine deaminase (codA) variants under study.

Numbering
---------
The sequences as provided by the lab are missing the first 3 residues of the
canonical UniProt P25524 entry:

    canonical_position = index_in_string + SEQ_OFFSET

The offset is not an assumption. It is confirmed by the data itself: both
mutations land exactly on their named positions (126 and 314) with this offset
and with no other. `verify_mutations()` enforces it.

Open issue
----------
The provided sequences are 423 residues; 423 + 3 = 426, but canonical P25524 is
427. One residue is missing, and it cannot be at the N-terminus (that would
shift the mutations to 127 and 315). It must be at the C-terminus. Resolve
before aligning sequence to structure (1K6W).
"""

SEQ_OFFSET = 3
CANONICAL_LENGTH = 427  # UniProt P25524 / PDB 1K6W

_WT = (
    "ALQTIINARLPGEEGLWQIHLQDGKISAIDAQSGVMPITENSLDAEQGLVIPPFVEPHIHLDTTQTAGQPNWNQSGTLFEGIERWAERKALLTHDDVKQRAWQTLKWQIANGIQHVRTHV"
    "DVSDATLTALKAMLEVKQEVAPWIDLQIVAFPQEGILSYPNGEALLEEALRLGADVVGAIPHFEFTREYGVESLHKTFALAQKYDRLIDVHCDEIDDEQSRFVETVAALAHHEGMGARVT"
    "ASHTTAMHSYNGAYTSRLFRLLKMSGINFVANPLVNIHLQGRFDTYPKRRGITRVKEMLESGINVCFGHDDVFDPWYPLGTANMLQVLHMGLHVCQLMGYGQINDGLNLITHHSARTLNL"
    "QDYGIAAGNSANLIILPAENGFDALRRQVPVRYSVRGGKVIASTQPAQTTVYLEQPEAIDYKR"
)


def _apply(seq, canonical_pos, wt_aa, mut_aa):
    i = canonical_pos - SEQ_OFFSET - 1
    assert seq[i] == wt_aa, f"expected {wt_aa} at canonical {canonical_pos}, found {seq[i]}"
    return seq[:i] + mut_aa + seq[i + 1:]


SEQUENCES = {
    "WT": _WT,
    "D314A": _apply(_WT, 314, "D", "A"),
    "S126R": _apply(_WT, 126, "S", "R"),
    "DOUBLE": _apply(_apply(_WT, 314, "D", "A"), 126, "S", "R"),
}

EXPECTED_MUTATIONS = {
    "WT": [],
    "D314A": [(314, "D", "A")],
    "S126R": [(126, "S", "R")],
    "DOUBLE": [(126, "S", "R"), (314, "D", "A")],
}


def diff_vs_wt(seq):
    """[(canonical_pos, wt_aa, mut_aa), ...] for every position differing from WT."""
    assert len(seq) == len(_WT), f"length mismatch: {len(seq)} vs WT {len(_WT)}"
    return [(i + 1 + SEQ_OFFSET, _WT[i], seq[i])
            for i in range(len(_WT)) if _WT[i] != seq[i]]


def verify_mutations():
    """Fail loudly if a variant does not carry exactly the mutations it is named after."""
    for name, seq in SEQUENCES.items():
        found, expected = diff_vs_wt(seq), EXPECTED_MUTATIONS[name]
        assert found == expected, f"{name}: expected {expected}, found {found}"


def length_report():
    n = len(_WT)
    return {"provided": n, "plus_offset": n + SEQ_OFFSET, "canonical": CANONICAL_LENGTH,
            "unaccounted_residues": CANONICAL_LENGTH - (n + SEQ_OFFSET)}


def write_fasta(path):
    with open(path, "w") as fh:
        for name, seq in SEQUENCES.items():
            fh.write(f">{name} codA_Ecoli_P25524 offset={SEQ_OFFSET} len={len(seq)}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")


if __name__ == "__main__":
    verify_mutations()
    print("verify_mutations(): OK")
    for name, seq in SEQUENCES.items():
        print(f"  {name:8s} len={len(seq)}  {diff_vs_wt(seq)}")
    print("  length_report():", length_report())
