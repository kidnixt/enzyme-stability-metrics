"""The main figure: what was measured, and why nothing available can see it.

Left panel  - the measurement. Residual activity of each variant, with where
              the double mutant would sit if the two mutations acted
              independently. The gap between that marker and the observed bar
              is the interaction, and it is the dominant feature of the data.
Right panel - the spatial reach of every method tried, against the distance
              between the two mutated positions. Three unrelated reasons,
              one shared blind spot.

All numbers are final: plateaus from the digitized figure, forces from
data/processed/uma_forces_r9.csv, separation measured in the PDBs.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLATEAU = {"WT": 60.0, "D314A": 47.0, "S126R": 22.0, "DOUBLE": 48.0}
SEPARATION_A = 14.9
REACH = [
    ("ThermoMPNN-D", 5.0, "hardcoded cutoff"),
    ("ESM-2", 9.0, "measured: noise past this"),
    ("UMA forces", 9.0, "largest sphere that fits"),
]

logit = lambda a: -np.log(a / (100.0 - a))
inv = lambda g: 100.0 / (1.0 + np.exp(g))

dg = {v: logit(PLATEAU[v]) - logit(PLATEAU["WT"]) for v in PLATEAU}
additive_dg = dg["D314A"] + dg["S126R"]
additive_plateau = inv(logit(PLATEAU["WT"]) + additive_dg)
epistasis = dg["DOUBLE"] - additive_dg

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 3.9),
                               gridspec_kw={"width_ratios": [1.05, 1]})

# ---------- left: the measurement
order = ["WT", "D314A", "S126R", "DOUBLE"]
x = np.arange(len(order))
colors = ["#4a5c6a", "#6b7f8e", "#6b7f8e", "#6b7f8e"]
ax1.bar(x, [PLATEAU[v] for v in order], color=colors, width=0.56, zorder=3)
for i, v in enumerate(order):
    ax1.text(i, PLATEAU[v] + 1.6, f"{PLATEAU[v]:.0f}", ha="center",
             fontsize=9, color="#333", zorder=4)

ax1.bar([3], [additive_plateau], width=0.56, facecolor="none",
        edgecolor="#c0392b", lw=1.6, linestyle="--", zorder=5)
ax1.annotate(f"if the two mutations acted\nindependently: {additive_plateau:.0f}%",
             xy=(2.73, additive_plateau), xytext=(2.0, 34),
             fontsize=8.5, color="#c0392b", ha="center",
             arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.1,
                             shrinkA=2, shrinkB=2))
ax1.annotate("", xy=(3.32, additive_plateau), xytext=(3.32, PLATEAU["DOUBLE"]),
             arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.3))
ax1.text(3.42, (additive_plateau + PLATEAU["DOUBLE"]) / 2,
         f"interaction\n{epistasis:+.2f} logit", fontsize=8.5,
         color="#c0392b", va="center")

ax1.set_xticks(x)
ax1.set_xticklabels(order, fontsize=9)
ax1.set_ylabel("residual activity at the plateau (%)")
ax1.set_ylim(0, 72)
ax1.set_xlim(-0.6, 4.5)
ax1.set_title("The double mutant is rescued, not additive", fontsize=10.5)
ax1.grid(axis="y", color="#e8e8e8", zorder=0)
ax1.spines[["top", "right"]].set_visible(False)

# ---------- right: reach
y = np.arange(len(REACH))
ax2.barh(y, [r[1] for r in REACH], color="#6b7f8e", height=0.5, zorder=3)
ax2.set_yticks(y)
ax2.set_yticklabels([f"{r[0]}\n{r[2]}" for r in REACH], fontsize=8.5)
ax2.invert_yaxis()
ax2.axvline(SEPARATION_A, color="#c0392b", lw=2, zorder=4)
ax2.text(SEPARATION_A + 0.4, 0.05,
         f"the two mutations\nsit {SEPARATION_A:.1f} $\\AA$ apart",
         color="#c0392b", va="center", fontsize=8.5)
for i, (_, d, _why) in enumerate(REACH):
    ax2.text(d + 0.35, i, f"{d:.0f} $\\AA$", va="center", fontsize=9,
             color="#555", zorder=5)
ax2.set_xlabel(r"spatial reach ($\AA$)")
ax2.set_xlim(0, SEPARATION_A + 5.5)
ax2.set_title("No method reaches that far", fontsize=10.5)
ax2.grid(axis="x", color="#e8e8e8", zorder=0)
ax2.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
out = Path("docs/figures/main_result.png")
fig.savefig(out, dpi=200)
print(f"escrito {out}")
print(f"  prediccion aditiva del DOUBLE: {additive_plateau:.1f}% "
      f"(observado {PLATEAU['DOUBLE']:.0f}%)")
print(f"  epistasia: {epistasis:+.3f} logit, borra el "
      f"{abs(epistasis)/additive_dg*100:.0f}% de la suma")
