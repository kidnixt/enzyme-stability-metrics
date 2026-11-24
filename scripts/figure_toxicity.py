"""La figura de la mitad molecular: los cuanticos no aportan, y el benchmark
publicado no transfiere.

Left  - cross-validated AUC on the 234 training molecules, by feature block and
        model family. The point is the negative control: total energy is pure
        composition, and gradient boosting still gets 0.700 out of it. Whatever
        the training set is teaching, a good part of it is not chemistry.
Right - the transfer diagnostic. Signal exists inside each set and does not
        cross between them. The grey band is the permutation null: what
        train->test looks like when the labels carry no information.

Numbers from scripts/mol_analysis.py and scripts/mol_transfer.py.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SETS = ["CLASICOS", "CUANTICOS", "TRAMPA", "CLASICOS\n+CUANTICOS"]
LR = [0.745, 0.595, 0.423, 0.744]
LR_SD = [0.067, 0.077, 0.078, 0.067]
GB = [0.809, 0.568, 0.700, 0.813]
GB_SD = [0.060, 0.075, 0.067, 0.060]

TRANSFER = [("CV dentro\nde train", 0.743, 0.071),
            ("CV dentro\nde test", 0.612, 0.073),
            ("CV sobre\nlos dos", 0.637, 0.059),
            ("train $\\rightarrow$ test", 0.460, None)]
NULL_MEAN, NULL_SD = 0.506, 0.052

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.0),
                               gridspec_kw={"width_ratios": [1.15, 1]})

# ---------- left
x = np.arange(len(SETS))
w = 0.36
ax1.bar(x - w/2, LR, w, yerr=LR_SD, capsize=3, label="regresion logistica",
        color="#8fa3b0", zorder=3)
ax1.bar(x + w/2, GB, w, yerr=GB_SD, capsize=3, label="gradient boosting",
        color="#44596b", zorder=3)
ax1.axhline(0.5, color="#888", lw=1, ls="--", zorder=2)
ax1.text(3.46, 0.515, "azar", fontsize=8, color="#888")
ax1.annotate("el control negativo\nllega a 0.70",
             xy=(2 + w/2, 0.700), xytext=(1.35, 0.90),
             fontsize=8.5, color="#c0392b", ha="center",
             arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.1))
ax1.set_xticks(x)
ax1.set_xticklabels(SETS, fontsize=8.5)
ax1.set_ylabel("AUC en validacion cruzada (train, n=234)")
ax1.set_ylim(0.3, 1.0)
ax1.legend(fontsize=8, frameon=False, loc="lower left")
ax1.set_title("Sumar los descriptores cuanticos no cambia nada",
              fontsize=10.5)
ax1.grid(axis="y", color="#ececec", zorder=0)
ax1.spines[["top", "right"]].set_visible(False)

# ---------- right
y = np.arange(len(TRANSFER))
vals = [v for _, v, _ in TRANSFER]
errs = [e if e else 0 for _, _, e in TRANSFER]
cols = ["#8fa3b0"] * 3 + ["#c0392b"]
ax2.axhspan(NULL_MEAN - 1.96*NULL_SD, NULL_MEAN + 1.96*NULL_SD,
            color="#d9d9d9", zorder=1)
ax2.axhline(NULL_MEAN, color="#999", lw=1, zorder=2)
ax2.bar(y, vals, 0.5, yerr=errs, capsize=3, color=cols, zorder=3)
ax2.text(3.42, NULL_MEAN + 1.96*NULL_SD - 0.012,
         "nulo por permutacion\n(95%)", fontsize=8, color="#666",
         ha="right", va="top")
ax2.set_xticks(y)
ax2.set_xticklabels([n for n, _, _ in TRANSFER], fontsize=8.5)
ax2.set_ylabel("AUC")
ax2.set_ylim(0.3, 0.9)
ax2.set_xlim(-0.6, 3.6)
ax2.set_title("Hay senal dentro de cada conjunto; no cruza entre ellos",
              fontsize=10.5)
ax2.grid(axis="y", color="#ececec", zorder=0)
ax2.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
out = Path("docs/figures/toxicity_result.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200)
print(f"escrito {out}")
