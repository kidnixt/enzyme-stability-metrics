"""Null distribution for ThermoMPNN-D's predicted epistasis.

The targeted run already produced 101,441 predictions over 281 residue pairs,
so the null costs nothing extra: every (pair, substitution) row is a sample of
what the epistatic model says when it is asked about two positions.

For each row:
    epistasis = ddG_epistatic - [ ddG_single(mut_a) + ddG_single(mut_b) ]

The real pair is then placed in that distribution, and the same distance
checks run as for ESM-2: does the model's apparent coupling decay with
distance, and does it survive controlling for sequence separation.

    python scripts/thermompnn_null.py --single ~/thermompnn_single.csv \
        --pairs data/processed/thermompnn_pairs.csv
"""

import argparse
import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PAIRS_OFFSET = 3     # pairs file numbers positions as canonical - 3
REAL = ("S", 123, "R", "D", 311, "A")   # in pairs-file numbering


def load_singles(path):
    """{(pos_in_pairs_numbering, wt, mut): ddG}"""
    df = pd.read_csv(path)
    mut_col = next(c for c in df.columns if "mutation" in c.lower())
    ddg_col = next(c for c in df.columns if "ddg" in c.lower())
    parts = df[mut_col].astype(str).str.extract(r"^(.)(.)(\d+)(.)$")
    parts.columns = ["c1", "c2", "pos", "mut"]
    df = df.join(parts)
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce")

    table = {}
    for _, r in df.iterrows():
        if pd.isna(r["pos"]):
            continue
        # singles use canonical numbering; pairs use canonical - 3
        pos = int(r["pos"]) - PAIRS_OFFSET
        for wt in (r["c1"], r["c2"]):
            table[(pos, wt, r["mut"])] = float(r[ddg_col])
    return table


def ca_coords(pdb_path):
    out = {}
    for ln in open(pdb_path):
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            out[int(ln[22:26])] = np.array(
                [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--pdb", default="data/raw/WT (4).pdb")
    ap.add_argument("--out", default="data/processed/thermompnn_epistasis.csv")
    args = ap.parse_args()

    singles = load_singles(args.single)
    ca = ca_coords(args.pdb)
    df = pd.read_csv(args.pairs)
    mut_col = next(c for c in df.columns if "mutation" in c.lower())
    ddg_col = next(c for c in df.columns if "ddg" in c.lower())

    ex = df[mut_col].astype(str).str.extract(
        r"^(.)(\d+)(.):(.)(\d+)(.)$")
    ex.columns = ["wt1", "p1", "m1", "wt2", "p2", "m2"]
    df = df.join(ex).dropna(subset=["p1", "p2"])
    df["p1"] = df["p1"].astype(int)
    df["p2"] = df["p2"].astype(int)

    add, ok = [], []
    for _, r in df.iterrows():
        a = singles.get((r["p1"], r["wt1"], r["m1"]))
        b = singles.get((r["p2"], r["wt2"], r["m2"]))
        if a is None or b is None:
            add.append(np.nan); ok.append(False)
        else:
            add.append(a + b); ok.append(True)
    df["ddG_additive"] = add
    df = df[pd.Series(ok, index=df.index)].copy()
    df["epistasis"] = df[ddg_col] - df["ddG_additive"]

    # canonical numbering for distances
    df["ca_distance"] = [
        float(np.linalg.norm(ca[p1 + PAIRS_OFFSET] - ca[p2 + PAIRS_OFFSET]))
        for p1, p2 in zip(df.p1, df.p2)]
    df["seq_separation"] = (df.p2 - df.p1).abs()
    df["abs_epistasis"] = df.epistasis.abs()

    df.to_csv(args.out, index=False)
    print(f"{len(df)} predictions with a matched additive baseline"
          f"  ->  {args.out}\n")

    real = df[(df.wt1 == REAL[0]) & (df.p1 == REAL[1]) & (df.m1 == REAL[2]) &
              (df.wt2 == REAL[3]) & (df.p2 == REAL[4]) & (df.m2 == REAL[5])]
    if len(real) != 1:
        print(f"real pair rows found: {len(real)}  (expected 1)"); return
    r_eps = float(real.epistasis.iloc[0])
    r_d = float(real.ca_distance.iloc[0])

    print(f"REAL PAIR S126R + D314A")
    print(f"   epistatic {float(real[ddg_col].iloc[0]):+.3f}"
          f"   additive {float(real.ddG_additive.iloc[0]):+.3f}"
          f"   epistasis {r_eps:+.3f} kcal/mol   at {r_d:.1f} A\n")

    v = df.abs_epistasis
    pct = (v < abs(r_eps)).mean() * 100
    print(f"null over {len(df)} predictions, |epistasis|:")
    print(f"   median {v.median():.4f}   90th {v.quantile(0.9):.4f}"
          f"   max {v.max():.4f}")
    print(f"   real pair at the {pct:.1f}th percentile")

    band = df[(df.ca_distance > r_d - 4) & (df.ca_distance < r_d + 4)]
    if len(band) > 20:
        pb = (band.abs_epistasis < abs(r_eps)).mean() * 100
        print(f"   against pairs at comparable distance "
              f"({r_d-4:.0f}-{r_d+4:.0f} A, n={len(band)}): {pb:.1f}th percentile")

    print()
    rho, p = spearmanr(df.ca_distance, df.abs_epistasis)
    print(f"Spearman(CA distance, |epistasis|) = {rho:+.3f}  p = {p:.2g}")
    distant = df[df.seq_separation > 50]
    if len(distant) > 20:
        rd, pd_ = spearmanr(distant.ca_distance, distant.abs_epistasis)
        print(f"   among pairs >50 apart in sequence (n={len(distant)}):"
              f" rho = {rd:+.3f}, p = {pd_:.2g}")

    print("\nmedian |epistasis| by CA distance:")
    bins = [0, 8, 12, 16, 20, 25, 30, np.inf]
    labels = ["<8", "8-12", "12-16", "16-20", "20-25", "25-30", ">30"]
    df["bin"] = pd.cut(df.ca_distance, bins=bins, labels=labels)
    print(df.groupby("bin", observed=True).agg(
        n=("abs_epistasis", "size"),
        median=("abs_epistasis", "median")).round(4).to_string())


if __name__ == "__main__":
    main()
