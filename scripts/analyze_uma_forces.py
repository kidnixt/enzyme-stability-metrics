"""Verdict on the forces: do they track the measured plateaus, and can they?

Two outputs:

1. The table. z-score of each mutated position within its own structure,
   next to the measured residual activity. With three mutants there is no
   statistic to run, so this reports the ordering and says so plainly
   instead of dressing n=3 up as a correlation.

2. The reach figure. The distance between the two mutated positions against
   the spatial range of every method tried. This is the actual result: the
   effect sits outside all of them, for three unrelated reasons.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# residual activity at the plateau, from the digitized published figure
PLATEAU = {"WT": 60.0, "D314A": 47.0, "S126R": 22.0, "DOUBLE": 48.0}

# measured in the PDBs: minimum atom-atom distance between positions 126 and 314
SEPARATION_A = 14.9

REACH = [
    ("ThermoMPNN-D", 5.0, "cutoff hardcoded in the model"),
    ("ESM-2", 9.0, "measured: coupling is noise beyond this"),
    ("UMA (this run)", 10.0, "memory ceiling at ~1100 atoms"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/uma_forces_r9.csv")
    ap.add_argument("--fig", default="docs/figures/reach_vs_separation.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    rows = []
    for variant, g in df.groupby("variant"):
        ctrl = g[~g.is_target].max_force_eV_per_A
        if len(ctrl) < 3:
            continue
        mu, sd = ctrl.mean(), ctrl.std()
        for _, r in g[g.is_target].iterrows():
            rows.append({
                "variant": variant,
                "position": int(r.position),
                "mutated": bool(r.mutated_here),
                "force": r.max_force_eV_per_A,
                "z": (r.max_force_eV_per_A - mu) / sd,
                "n_ctrl": len(ctrl),
                "plateau": PLATEAU.get(variant, np.nan),
            })
    t = pd.DataFrame(rows)

    print("=" * 70)
    print("Posiciones vigiladas, z dentro de su propia estructura")
    print("=" * 70)
    print(f"{'variante':8s} {'pos':>4s} {'mut':>4s} {'eV/A':>7s} {'z':>7s} "
          f"{'n_ctrl':>7s} {'meseta':>7s}")
    for _, r in t.sort_values(["position", "variant"]).iterrows():
        print(f"{r.variant:8s} {r.position:4d} {'si' if r.mutated else '-':>4s} "
              f"{r.force:7.3f} {r.z:+7.2f} {r.n_ctrl:7d} {r.plateau:7.0f}")

    mut = t[t.mutated].copy()
    print("\n" + "=" * 70)
    print("La pregunta: el z de la posicion mutada sigue a la meseta medida?")
    print("=" * 70)
    if len(mut):
        # one number per variant: the largest strain among its mutated positions
        per_var = mut.groupby("variant").agg(z=("z", "max"),
                                            plateau=("plateau", "first"))
        per_var = per_var.sort_values("plateau")
        print("\nordenado por meseta creciente (peor variante arriba):")
        for v, r in per_var.iterrows():
            print(f"   {v:8s} meseta {r.plateau:3.0f}   z = {r.z:+.2f}")
        print(f"\nn = {len(per_var)} mutantes. Con tres puntos no hay correlacion que")
        print("reportar: la p minima de una permutacion exacta es 1/6 = 0.167, asi")
        print("que ningun orden puede alcanzar significancia. Lo que se puede decir")
        print("es si el orden coincide, y nada mas que eso.")

        print(f"\norden por tension (mayor z = peor): "
              f"{per_var.sort_values('z', ascending=False).index.tolist()}")
        print(f"orden por meseta (peor variante primero): {per_var.index.tolist()}")

    # ---- the reach figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        y = np.arange(len(REACH))
        ax.barh(y, [r[1] for r in REACH], color="#7a8b99", height=0.55)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in REACH])
        ax.invert_yaxis()
        ax.axvline(SEPARATION_A, color="#c0392b", lw=2)
        ax.text(SEPARATION_A + 0.3, len(REACH) - 0.5,
                f"separation of the\ntwo mutations\n{SEPARATION_A:.1f} $\\AA$",
                color="#c0392b", va="center", fontsize=9)
        for i, (_, d, why) in enumerate(REACH):
            ax.text(0.3, i, why, va="center", fontsize=8, color="white")
        ax.set_xlabel(r"spatial reach ($\AA$)")
        ax.set_xlim(0, SEPARATION_A + 6)
        ax.set_title("Every method is blind at the distance the effect lives at",
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
