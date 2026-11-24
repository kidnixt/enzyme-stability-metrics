"""Verdict on the forces: do they track the measured plateaus, and can they?

Three things, in the order that matters.

1. The wildtype baseline. Both watched positions carry force well above a
   random position even in WT, where nothing is mutated. So the raw z-score
   is not the cost of the substitution: part of it is that these positions
   are intrinsically strained. The quantity with meaning is the difference
   against WT at the SAME position.

2. The additivity check. If the computation can express an interaction
   between the two mutations, the double mutant's value at a position must
   differ from the single mutant that carries that same mutation. It does
   not. The interaction term is reported here next to the measured one.

3. The reach figure. The distance between the two mutated positions against
   the spatial range of every method tried. This is why (2) comes out the
   way it does, and it is the result.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# residual activity at the plateau, from the digitized published figure
PLATEAU = {"WT": 60.0, "D314A": 47.0, "S126R": 22.0, "DOUBLE": 48.0}
CARRIES = {"D314A": {314}, "S126R": {126}, "DOUBLE": {126, 314}, "WT": set()}

# measured in the PDBs: minimum atom-atom distance between positions 126 and 314
SEPARATION_A = 14.9

REACH = [
    ("ThermoMPNN-D", 5.0, "cutoff hardcoded in the model"),
    ("ESM-2", 9.0, "measured: coupling is noise beyond this"),
    ("UMA (this run)", 9.0, "largest sphere that fits in 11 GB"),
]


def logit(a):
    """Puts plateaus on an additive scale, assuming a two-state partition."""
    return -np.log(a / (100.0 - a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/uma_forces_r9.csv")
    ap.add_argument("--fig", default="docs/figures/reach_vs_separation.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # z-score of each watched position within its own structure's null
    z = {}
    raw = {}
    for variant, g in df.groupby("variant"):
        ctrl = g[~g.is_target].max_force_eV_per_A
        mu, sd = ctrl.mean(), ctrl.std()
        for _, r in g[g.is_target].iterrows():
            z[(variant, int(r.position))] = (r.max_force_eV_per_A - mu) / sd
            raw[(variant, int(r.position))] = r.max_force_eV_per_A

    positions = sorted({p for _, p in z})

    print("=" * 72)
    print("1. El WT ya esta tensionado en las dos posiciones")
    print("=" * 72)
    for p in positions:
        print(f"   pos {p}: WT da z = {z[('WT', p)]:+.2f} sin mutacion alguna")
    print("\n   Por eso lo que sigue es la diferencia contra el WT en la misma")
    print("   posicion, no el z crudo.")

    print("\n" + "=" * 72)
    print("2. Delta contra WT, por posicion")
    print("=" * 72)
    print(f"   {'variante':9s}" + "".join(f"{'pos '+str(p):>14s}" for p in positions)
          + f"{'suma':>10s}{'meseta':>9s}")
    dz_total = {}
    for v in ["WT", "D314A", "S126R", "DOUBLE"]:
        cells, tot = [], 0.0
        for p in positions:
            d = z[(v, p)] - z[("WT", p)]
            tot += d
            mark = "*" if p in CARRIES[v] else " "
            cells.append(f"{d:+13.2f}{mark}")
        dz_total[v] = tot
        print(f"   {v:9s}" + "".join(cells) + f"{tot:+10.2f}{PLATEAU[v]:9.0f}")
    print("\n   (* = la mutacion esta en esa posicion)")

    print("\n" + "=" * 72)
    print("3. Hay termino de interaccion?")
    print("=" * 72)
    print("   Si el calculo pudiera ver las dos mutaciones juntas, el DOUBLE")
    print("   tendria que diferir del simple que lleva esa misma mutacion.\n")
    pairs = [(126, "S126R"), (314, "D314A")]
    for p, single in pairs:
        a, b = raw[(single, p)], raw[("DOUBLE", p)]
        print(f"   pos {p}:  {single:7s} {a:.4f}   DOUBLE {b:.4f}   "
              f"diferencia {abs(a-b):.4f} eV/A ({abs(a-b)/a*100:.2f}%)")

    eps_force = dz_total["DOUBLE"] - (dz_total["D314A"] + dz_total["S126R"])
    print(f"\n   epistasia en fuerzas  = {eps_force:+.3f} sigma")

    g = {v: logit(PLATEAU[v]) for v in PLATEAU}
    dg = {v: g[v] - g["WT"] for v in PLATEAU}
    eps_meas = dg["DOUBLE"] - (dg["D314A"] + dg["S126R"])
    print(f"   epistasia medida      = {eps_meas:+.3f} unidades logit")
    print(f"      prediccion aditiva del DOUBLE: {dg['D314A'] + dg['S126R']:+.3f}")
    print(f"      observado:                     {dg['DOUBLE']:+.3f}")
    print(f"      la interaccion borra el {abs(eps_meas)/(dg['D314A']+dg['S126R'])*100:.0f}% "
          f"de lo que predice la suma")

    print("\n" + "=" * 72)
    print("4. Orden predicho contra orden medido")
    print("=" * 72)
    loss = {v: PLATEAU["WT"] - PLATEAU[v] for v in PLATEAU if v != "WT"}
    by_pred = sorted(loss, key=lambda v: -dz_total[v])
    by_meas = sorted(loss, key=lambda v: -loss[v])
    print(f"   por tension predicha (peor primero): {by_pred}")
    print(f"   por perdida medida   (peor primero): {by_meas}")
    print(f"\n   n = 3 mutantes. La p minima de una permutacion exacta es 1/6 =")
    print("   0.167, asi que ningun orden alcanza significancia. Solo se puede")
    print("   decir si coincide, y nada mas.")

    # ---- the reach figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.6, 3.2))
        y = np.arange(len(REACH))
        ax.barh(y, [r[1] for r in REACH], color="#6b7f8e", height=0.55)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in REACH])
        ax.invert_yaxis()
        ax.axvline(SEPARATION_A, color="#c0392b", lw=2)
        ax.text(SEPARATION_A + 0.35, 0.1,
                f"the two mutations\nsit {SEPARATION_A:.1f} $\\AA$ apart",
                color="#c0392b", va="center", fontsize=9)
        for i, (_, _, why) in enumerate(REACH):
            ax.text(0.3, i, why, va="center", fontsize=8, color="white")
        ax.set_xlabel(r"spatial reach ($\AA$)")
        ax.set_xlim(0, SEPARATION_A + 7)
        ax.set_title("Each method is blind at the distance the effect lives at",
                     fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig, dpi=200)
        print(f"\nfigura escrita en {args.fig}")
    except ModuleNotFoundError:
        print("\n(sin matplotlib en este entorno: la figura no se genero)")


if __name__ == "__main__":
    main()
