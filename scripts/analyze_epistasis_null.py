"""Is ESM-2's apparent epistasis distance-dependent, or is it all noise?

The summary printed by esm_epistasis_null.py splits the pairs at 10 A and 20 A
and compares medians. With few pairs in the near bin that comparison is
suggestive at best. This script does it properly: binned medians, a rank
correlation over all pairs, and a Mann-Whitney test between near and far.

    python scripts/analyze_epistasis_null.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

REAL_EPISTASIS = 0.1003
REAL_CA_DISTANCE = 18.4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/esm2_epistasis_null.csv")
    ap.add_argument("--near", type=float, default=10.0)
    ap.add_argument("--far", type=float, default=20.0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["abs_epistasis"] = df.epistasis.abs()
    print(f"{len(df)} random pairs from {args.csv}\n")

    # --- where does the real pair sit overall
    pct = (df.abs_epistasis < abs(REAL_EPISTASIS)).mean() * 100
    print(f"real pair |epistasis| = {abs(REAL_EPISTASIS):.4f} at {REAL_CA_DISTANCE} A"
          f"  ->  {pct:.1f}th percentile of the full null\n")

    # --- binned by distance
    bins = [0, 8, 12, 16, 20, 25, 30, np.inf]
    labels = ["<8", "8-12", "12-16", "16-20", "20-25", "25-30", ">30"]
    df["bin"] = pd.cut(df.ca_distance, bins=bins, labels=labels)
    tab = df.groupby("bin", observed=True).agg(
        n=("abs_epistasis", "size"),
        median=("abs_epistasis", "median"),
        p90=("abs_epistasis", lambda s: s.quantile(0.9)),
    ).round(4)
    print("median |epistasis| by CA distance:")
    print(tab.to_string())
    print()

    # --- rank correlation over everything, no binning
    rho, p = spearmanr(df.ca_distance, df.abs_epistasis)
    print(f"Spearman(CA distance, |epistasis|) over all pairs:"
          f"  rho = {rho:+.3f}   p = {p:.2g}")
    if p < 0.05 and rho < 0:
        print("   -> apparent coupling does decay with distance: the model has"
              " a spatial range")
    elif p >= 0.05:
        print("   -> no distance dependence detectable: apparent coupling is"
              " noise at every distance")
    print()

    # --- near vs far, without relying on a median of ten points
    near = df[df.ca_distance < args.near].abs_epistasis
    far = df[df.ca_distance > args.far].abs_epistasis
    print(f"near (< {args.near:g} A): n={len(near):3d}  median {near.median():.4f}")
    print(f"far  (> {args.far:g} A): n={len(far):3d}  median {far.median():.4f}")
    if len(near) >= 3 and len(far) >= 3:
        u, pu = mannwhitneyu(near, far, alternative="greater")
        print(f"Mann-Whitney (near > far): p = {pu:.2g}")
        if pu < 0.05:
            print("   -> near pairs really do show more apparent coupling")
        else:
            print(f"   -> not significant; with n={len(near)} near pairs this test"
                  " is underpowered, rerun with more pairs")
    print()

    # --- IS IT REALLY 3D DISTANCE, OR JUST SEQUENCE PROXIMITY?
    # Residues close in space are often close in sequence too (a helix turn is
    # i, i+3-4). A transformer attending locally in sequence would produce the
    # same apparent decay with no spatial knowledge at all. These two checks
    # separate the explanations.
    print("-" * 64)
    print("CONFOUND CHECK: 3D distance vs sequence separation")
    r_de, _ = spearmanr(df.ca_distance, df.abs_epistasis)
    r_se, _ = spearmanr(df.seq_separation, df.abs_epistasis)
    r_ds, _ = spearmanr(df.ca_distance, df.seq_separation)
    print(f"  rho(3D distance, |epistasis|)      = {r_de:+.3f}")
    print(f"  rho(seq separation, |epistasis|)   = {r_se:+.3f}")
    print(f"  rho(3D distance, seq separation)   = {r_ds:+.3f}   <- how entangled they are")

    denom = np.sqrt((1 - r_ds ** 2) * (1 - r_se ** 2))
    partial = (r_de - r_ds * r_se) / denom if denom > 0 else float("nan")
    print(f"  PARTIAL rho(3D distance, |epistasis| | seq separation) = {partial:+.3f}")

    # Same question by stratification instead of regression: keep only pairs
    # that are far apart in sequence, where local attention cannot help.
    distant = df[df.seq_separation > 50]
    if len(distant) > 20:
        rho_d, p_d = spearmanr(distant.ca_distance, distant.abs_epistasis)
        print(f"  among pairs >50 residues apart in sequence (n={len(distant)}):"
              f" rho = {rho_d:+.3f}, p = {p_d:.2g}")
        if p_d < 0.05 and rho_d < 0:
            print("     -> the decay survives with sequence proximity removed:"
                  " genuinely spatial")
        else:
            print("     -> the decay does NOT survive: what looked spatial was"
                  " sequence proximity")
    print()

    # --- the comparison that matters for us: the real pair against its own bin
    band = df[(df.ca_distance > REAL_CA_DISTANCE - 4) &
              (df.ca_distance < REAL_CA_DISTANCE + 4)]
    if len(band) >= 5:
        pct_band = (band.abs_epistasis < abs(REAL_EPISTASIS)).mean() * 100
        print(f"against random pairs at a COMPARABLE distance"
              f" ({REAL_CA_DISTANCE-4:.0f}-{REAL_CA_DISTANCE+4:.0f} A, n={len(band)}):")
        print(f"   median {band.abs_epistasis.median():.4f},"
              f" real pair at the {pct_band:.1f}th percentile")
        print("   this is the fair comparison: same distance regime, so the"
              " model's spatial range is held constant")


if __name__ == "__main__":
    main()
