"""Extract the ThermoMPNN predictions for our two mutations and their double.

Mutation string formats, learned from the actual output:
  single file : {chain}{wt}{pos}{mut}      e.g. "AD314A"  (PDB numbering)
  pairs file  : {wt}{pos}{mut}:{wt}{pos}{mut}  with 0-based sequence indices,
                because the targeted driver skips the renumbering step

Canonical position p maps to sequence index p - 4 (structures start at 4).

    python scripts/thermompnn_compare.py --single ~/thermompnn_single.csv \
        [--pairs data/processed/thermompnn_pairs.csv]
"""

import argparse

import pandas as pd

MEASURED_A_INF = {"WT": 60.0, "D314A": 47.0, "S126R": 22.1, "DOUBLE": 48.4}
FIRST_RESIDUE = 4
TARGETS = {"D314A": ("D", 314, "A"), "S126R": ("S", 126, "R")}


def seq_index(canonical):
    return canonical - FIRST_RESIDUE


def cols_of(df):
    mut = next(c for c in df.columns if "mutation" in c.lower())
    ddg = next(c for c in df.columns if "ddg" in c.lower())
    return mut, ddg


def do_single(path):
    df = pd.read_csv(path)
    mut_col, ddg_col = cols_of(df)
    print(f"--- singles: {len(df)} rows, format e.g. {df[mut_col].iloc[1]!r}\n")

    found = {}
    for name, (wt, pos, mut) in TARGETS.items():
        # chain prefix may or may not be there, so anchor at the end
        hit = df[df[mut_col].astype(str).str.endswith(f"{wt}{pos}{mut}")]
        if len(hit) != 1:
            print(f"{name}: {len(hit)} matches; rows at position {pos}:")
            print(df[df[mut_col].astype(str).str.contains(f"{wt}{pos}",
                                                          na=False)].to_string())
            continue
        ddg = float(hit.iloc[0][ddg_col])
        pctile = (df[ddg_col] < ddg).mean() * 100
        found[name] = ddg
        print(f"{name:7s}  ddG = {ddg:+7.3f} kcal/mol"
              f"   (percentile {pctile:5.1f} of {len(df)} mutations)")

    if len(found) == 2:
        additive = found["D314A"] + found["S126R"]
        print(f"\nDOUBLE additive = {additive:+7.3f} kcal/mol"
              "   (what any additive model must say)")
        pred = {"WT": 0.0, **found, "DOUBLE": additive}
        print("\npredicted ranking (least destabilising first):")
        print("   " + " > ".join(sorted(pred, key=lambda v: -pred[v])))
        print("measured ranking  (most robust first):")
        print("   " + " > ".join(sorted(MEASURED_A_INF,
                                        key=lambda v: -MEASURED_A_INF[v])))
        return found, additive
    return found, None


def do_pairs(path, additive):
    df = pd.read_csv(path)
    mut_col, ddg_col = cols_of(df)
    print(f"\n--- pairs: {len(df)} rows, format e.g. {df[mut_col].iloc[0]!r}")

    a, b = seq_index(126), seq_index(314)
    s = df[mut_col].astype(str)
    both = df[s.str.contains(f"S{a}R", na=False) & s.str.contains(f"D{b}A", na=False)]
    print(f"rows with S{a}R and D{b}A together: {len(both)}")

    if len(both) == 0:
        at_a = df[s.str.contains(f"{a}", na=False)]
        print(f"\nrows mentioning index {a} at all: {len(at_a)}")
        print(at_a.head(10).to_string())
        print("\nfirst rows of the file, to read the real format:")
        print(df.head(10).to_string())
        return

    ddg_epi = float(both.iloc[0][ddg_col])
    print(f"\nepistatic ddG for the real double = {ddg_epi:+7.3f} kcal/mol")
    if additive is not None:
        print(f"additive  ddG                     = {additive:+7.3f} kcal/mol")
        print(f"PREDICTED EPISTASIS               = {ddg_epi - additive:+7.3f}")
        print("\nmeasured: the double behaves like D314A alone, far better than"
              " the sum of its parts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", required=True)
    ap.add_argument("--pairs")
    args = ap.parse_args()

    _, additive = do_single(args.single)
    if args.pairs:
        do_pairs(args.pairs, additive)


if __name__ == "__main__":
    main()
