"""ThermoMPNN-D on selected residue pairs instead of the whole protein.

Why this exists
---------------
`v2_ssm.py --mode epistatic` enumerates every residue pair within a Ca-Ca
cutoff. Our pair sits at 18.4 A, and a 19 A cutoff on this 423-residue protein
means 20,675 pairs x 361 substitutions = 7.5 million rows. Not runnable.

But the pair list is built in one place, `get_ssm_mutations_double(pdb, dthresh)`,
which derives it from a distance mask. Replacing that function with one that
returns an explicit pair list leaves every downstream step untouched and brings
the job down to ~120k rows.

What it produces
----------------
The epistatic ddG for the real pair (S126R + D314A) and for a set of random
pairs spanning a range of distances, so the real pair can be read against a
null exactly as was done for ESM-2. Additive ddGs come from the single-mutant
run, so epistasis = epistatic - additive.

Usage
-----
    export PYTHONPATH=$PYTHONPATH:/path/to/ThermoMPNN-D
    cd /path/to/ThermoMPNN-D
    python /path/to/this/thermompnn_targeted.py \
        --pdb /path/to/enzyme-stability-metrics/data/raw/"WT (4).pdb" \
        --out /path/to/enzyme-stability-metrics/data/processed/thermompnn_pairs.csv

NOTE: this driver has not been executed against the real model. The hook and
the data shapes were read from the source, but the first run may need a tweak.
Report the traceback rather than working around it.
"""

import argparse
import itertools
import math
import random
import sys

import numpy as np
import pandas as pd

# Canonical numbering -> index into pdb["seq"].
# The structures start at residue 4, so seq_pos = canonical - 4.
FIRST_RESIDUE = 4
REAL_PAIR = ((126, "S", "R"), (314, "D", "A"))
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def seq_pos(canonical):
    return canonical - FIRST_RESIDUE


def ca_coords(pdb_path):
    out = {}
    for ln in open(pdb_path):
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            out[int(ln[22:26])] = np.array(
                [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    return out


def build_pair_list(ca, n_random, seed, stratify):
    """Explicit list of (seq_pos_a, seq_pos_b), real pair first."""
    rng = random.Random(seed)
    pairs = [(seq_pos(126), seq_pos(314))]

    positions = sorted(ca)
    if stratify:
        BINS = [(0, 8), (8, 12), (12, 16), (16, 20), (20, 25), (25, 30), (30, 1e9)]
        buckets = {b: [] for b in BINS}
        for p, q in itertools.combinations(positions, 2):
            d = float(np.linalg.norm(ca[p] - ca[q]))
            for lo, hi in BINS:
                if lo <= d < hi:
                    buckets[(lo, hi)].append((p, q))
                    break
        per_bin = max(1, n_random // len(BINS))
        chosen = []
        for b, cand in buckets.items():
            chosen += rng.sample(cand, min(per_bin, len(cand)))
    else:
        chosen = [tuple(rng.sample(positions, 2)) for _ in range(n_random)]

    for p, q in chosen:
        a, b = seq_pos(min(p, q)), seq_pos(max(p, q))
        if (a, b) not in pairs:
            pairs.append((a, b))
    return pairs


def patched_enumerator(pair_list):
    """Replacement for get_ssm_mutations_double: same outputs, our pairs."""
    def _inner(pdb, dthresh):
        seq = pdb["seq"]
        combos = np.array(pair_list)                      # [P, 2]
        wt = np.zeros_like(combos)
        for i, (a, b) in enumerate(combos):
            wt[i, 0] = ALPHABET.index(seq[a])
            wt[i, 1] = ALPHABET.index(seq[b])

        one = np.arange(20).repeat(20)
        two = np.tile(np.arange(20), 20)
        mutAA = np.tile(np.stack([one, two]).T, (combos.shape[0], 1))   # [P*400, 2]
        wtAA = np.repeat(wt, 400, axis=0)
        pos = np.repeat(combos, 400, axis=0)

        keep = ~np.sum(mutAA == wtAA, -1).astype(bool)    # drop self/single mutants
        return pos[keep], wtAA[keep], mutAA[keep]
    return _inner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-random", type=int, default=280)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stratify", action="store_true", default=True)
    args = ap.parse_args()

    import v2_ssm

    ca = ca_coords(args.pdb)
    pairs = build_pair_list(ca, args.n_random, args.seed, args.stratify)
    print(f"{len(pairs)} pairs selected (real pair first), "
          f"~{len(pairs)*380:,} mutation rows")

    v2_ssm.get_ssm_mutations_double = patched_enumerator(pairs)

    cfg = v2_ssm.get_config("epistatic")
    model = v2_ssm.get_model("epistatic", cfg)
    pdb_data = v2_ssm.load_pdb(args.pdb, None)

    ddg, mutations = v2_ssm.run_epistatic_ssm(
        pdb_data, cfg, model,
        distance=1e9,          # ignored: the enumerator is patched
        threshold=1e6,         # keep everything
        batch_size=args.batch_size,
    )

    df = pd.DataFrame({"ddG_epistatic": ddg, "Mutation": mutations})
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(df)} rows)")

    want = f"{REAL_PAIR[0][1]}{seq_pos(126)}{REAL_PAIR[0][2]}"
    hits = df[df.Mutation.astype(str).str.contains(want, na=False)]
    print(f"\nrows mentioning {want}: {len(hits)}")
    print(hits.head(10).to_string())


if __name__ == "__main__":
    main()
