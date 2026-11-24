"""Null distribution for ESM-2's apparent epistasis between two positions.

The question this answers
-------------------------
For the real pair (S126R, D314A), the `mutant_marginals` scheme gave an
epistasis of +0.1003 log-likelihood units. That is 0.7 % of the total effect.
Is it a small real coupling, or the model's numerical floor?

A single number cannot say. This script samples random position pairs, applies
random substitutions, and computes the same quantity, producing a null
distribution. The real pair is then placed in it as a percentile.

It also records the sequence separation and the 3D distance of every pair, so
you can see whether ESM-2's apparent coupling decays with distance at all. If
it does not, the model has no structural coupling to offer and the real pair's
value is noise whatever its size.

Usage
-----
    python scripts/esm_epistasis_null.py --n-pairs 300 --device cuda
"""

import argparse
import itertools
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stability import plm
from stability.sequences import SEQUENCES, SEQ_OFFSET

AA = "ACDEFGHIKLMNPQRSTVWY"
REAL_PAIR = [(126, "S", "R"), (314, "D", "A")]


def ca_coords(pdb_path):
    """Canonical residue number -> CA coordinates."""
    out = {}
    for ln in open(pdb_path):
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            out[int(ln[22:26])] = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
    return out


def pair_epistasis(tok, model, device, wt_seq, mut_a, mut_b):
    """epistasis = score(both, in context) - [score(a) + score(b)], all vs WT."""
    s_a = plm.score_variant(tok, model, device, wt_seq, [mut_a], "wt_marginals")
    s_b = plm.score_variant(tok, model, device, wt_seq, [mut_b], "wt_marginals")
    s_ab = plm.score_variant(tok, model, device, wt_seq, [mut_a, mut_b],
                             "mutant_marginals")
    return s_ab - (s_a + s_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=300)
    ap.add_argument("--model", default=plm.DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pdb", default="data/raw/WT (4).pdb")
    ap.add_argument("--out", default="data/processed/esm2_epistasis_null.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stratified", action="store_true",
                    help="balance the sample across CA-distance bins instead of "
                         "sampling uniformly; random pairs in a globular protein "
                         "are nearly all far apart, so uniform sampling starves "
                         "the short-distance bins")
    ap.add_argument("--per-bin", type=int, default=40,
                    help="pairs per distance bin when --stratified is set")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    wt = SEQUENCES["WT"]
    ca = ca_coords(args.pdb)
    positions = sorted(ca)

    tok, model, device = plm.load_model(args.model, device=args.device)
    print(f"loaded {args.model} on {device}", flush=True)

    def residues_ok(p, q):
        try:
            return wt[plm.string_index(p)] in AA and wt[plm.string_index(q)] in AA
        except IndexError:
            return False

    if args.stratified:
        BINS = [(0, 8), (8, 12), (12, 16), (16, 20), (20, 25), (25, 30), (30, 1e9)]
        buckets = {b: [] for b in BINS}
        for p, q in itertools.combinations(positions, 2):
            if not residues_ok(p, q):
                continue
            d = math.dist(ca[p], ca[q])
            for lo, hi in BINS:
                if lo <= d < hi:
                    buckets[(lo, hi)].append((p, q))
                    break
        pairs = []
        for b, candidates in buckets.items():
            take = min(args.per_bin, len(candidates))
            pairs += rng.sample(candidates, take)
            print(f"  bin {b[0]}-{b[1]} A: {len(candidates)} available, taking {take}")
        rng.shuffle(pairs)
        print(f"stratified sample: {len(pairs)} pairs total", flush=True)
    else:
        pairs = []
        while len(pairs) < args.n_pairs:
            p, q = rng.sample(positions, 2)
            if residues_ok(p, q):
                pairs.append((p, q))

    rows = []
    for i, (p, q) in enumerate(pairs):
        wt_p = wt[plm.string_index(p)]
        wt_q = wt[plm.string_index(q)]
        mut_p = rng.choice([a for a in AA if a != wt_p])
        mut_q = rng.choice([a for a in AA if a != wt_q])

        eps = pair_epistasis(tok, model, device, wt,
                             (p, wt_p, mut_p), (q, wt_q, mut_q))
        rows.append({
            "pos_a": p, "pos_b": q,
            "mut_a": f"{wt_p}{p}{mut_p}", "mut_b": f"{wt_q}{q}{mut_q}",
            "seq_separation": abs(p - q),
            "ca_distance": round(math.dist(ca[p], ca[q]), 2),
            "epistasis": eps,
        })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(pairs)}", flush=True)

    real = pair_epistasis(tok, model, device, wt, REAL_PAIR[0], REAL_PAIR[1])
    d_real = math.dist(ca[126], ca[314])

    import pandas as pd
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    vals = df.epistasis.abs()
    pct = (vals < abs(real)).mean() * 100

    print()
    print("=" * 64)
    print(f"real pair S126R + D314A:  epistasis = {real:+.4f}")
    print(f"   sequence separation {abs(126-314)} residues, CA distance {d_real:.1f} A")
    print()
    print(f"null over {len(df)} random pairs,  |epistasis|:")
    print(f"   median {vals.median():.4f}   90th pct {vals.quantile(0.9):.4f}"
          f"   max {vals.max():.4f}")
    print()
    print(f"the real pair sits at the {pct:.1f}th percentile of the null")
    if pct < 90:
        print("   -> indistinguishable from random pairs: this is the noise floor")
    else:
        print("   -> above the bulk of random pairs: worth taking seriously")
    print()
    far = df[df.ca_distance > 20]
    near = df[df.ca_distance < 10]
    if len(near) and len(far):
        print(f"|epistasis| for CA distance < 10 A : median {near.epistasis.abs().median():.4f}  (n={len(near)})")
        print(f"|epistasis| for CA distance > 20 A : median {far.epistasis.abs().median():.4f}  (n={len(far)})")
        print("   if these are the same, ESM-2 has no distance-dependent coupling at all")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
