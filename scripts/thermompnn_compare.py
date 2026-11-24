"""Read ThermoMPNN single-mutant output and set up the comparison.

Produces, for D314A and S126R:
  - the predicted ddG
  - where each sits among all 8,460 predicted mutations (percentile)
  - the additive prediction for the double, which is what any additive model
    must say
and contrasts the resulting ranking with the measured A_inf ordering.

Robust to the exact Mutation string format: it reports the columns and a
sample, then locates the mutations by regex on position and amino acids.

    python scripts/thermompnn_compare.py --single ~/thermompnn_single.csv
"""

import argparse
import re

import pandas as pd

MEASURED_A_INF = {"WT": 60.0, "D314A": 47.0, "S126R": 22.1, "DOUBLE": 48.4}
TARGETS = {"D314A": ("D", 314, "A"), "S126R": ("S", 126, "R")}


def find_mutation(df, col, wt, pos, mut):
    """Match e.g. D314A however the column spells it."""
    pat = re.compile(rf"\b{wt}\s*{pos}\s*{mut}\b", re.I)
    hit = df[df[col].astype(str).str.contains(pat, na=False)]
    if len(hit) == 0:   # fall back: just the position, let the caller look
        pat2 = re.compile(rf"{wt}{pos}", re.I)
        return df[df[col].astype(str).str.contains(pat2, na=False)], False
    return hit, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.single)
    print(f"{len(df)} rows")
    print(f"columns: {list(df.columns)}\n")
    print("sample rows:")
    print(df.head(5).to_string(), "\n")

    mut_col = next((c for c in df.columns if "mutation" in c.lower()), df.columns[-1])
    ddg_col = next((c for c in df.columns if "ddg" in c.lower()), None)
    if ddg_col is None:
        raise SystemExit(f"no ddG column found among {list(df.columns)}")
    print(f"using mutation column '{mut_col}', ddG column '{ddg_col}'\n")

    found = {}
    for name, (wt, pos, mut) in TARGETS.items():
        hit, exact = find_mutation(df, mut_col, wt, pos, mut)
        if not exact:
            print(f"{name}: no exact match. Rows at position {pos}:")
            print(hit.head(25).to_string(), "\n")
            continue
        row = hit.iloc[0]
        ddg = float(row[ddg_col])
        pctile = (df[ddg_col] < ddg).mean() * 100
        found[name] = ddg
        print(f"{name:7s}  ddG = {ddg:+7.3f} kcal/mol"
              f"   (percentile {pctile:5.1f} of all {len(df)} mutations)")

    if len(found) == 2:
        additive = found["D314A"] + found["S126R"]
        print(f"\nDOUBLE, additive prediction = {additive:+7.3f} kcal/mol")
        print("   any additive model must say exactly this; the epistatic model"
              " is free to differ")

        pred = {"WT": 0.0, "D314A": found["D314A"], "S126R": found["S126R"],
                "DOUBLE": additive}
        print("\nranking, predicted (least destabilising first):")
        print("   " + " > ".join(sorted(pred, key=lambda v: -pred[v])))
        print("ranking, measured (most robust first):")
        print("   " + " > ".join(sorted(MEASURED_A_INF,
                                        key=lambda v: -MEASURED_A_INF[v])))
        print("\nmeasured A_inf:", {k: MEASURED_A_INF[k] for k in MEASURED_A_INF})


if __name__ == "__main__":
    main()
