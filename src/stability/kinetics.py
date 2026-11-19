"""Thermal inactivation kinetics for enzymes retaining residual activity.

Model (Sadana & Henley, Biotechnol. Bioeng. 1987, 30, 717): single-step
unimolecular deactivation whose final state is NOT fully inactivated, giving
non-first-order decay to a non-zero plateau:

    A(t) = A_inf + (A_0 - A_inf) * exp(-k * t)

Reported per variant:
    k      [1/min]  rate of the fast decay phase
    A_inf  [%]      plateau residual activity
    tau    [min]    1/k, always defined
    t_half [min]    time to reach A_0/2, ONLY defined when A_inf < A_0/2

The half-life does not exist when the plateau sits at or above half the initial
activity: the curve never crosses the 50% line. Ranking variants by half-life
compares a quantity that is undefined for some of them.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit


def residual_activity(t, k, a_inf, a0=100.0):
    return a_inf + (a0 - a_inf) * np.exp(-k * np.asarray(t, dtype=float))


def half_life(k, a_inf, a0=100.0) -> Optional[float]:
    target = a0 / 2.0
    if a_inf >= target:
        return None
    return float(-np.log((target - a_inf) / (a0 - a_inf)) / k)


@dataclass
class InactivationFit:
    variant: str
    k: float
    a_inf: float
    tau: float
    t_half: Optional[float]
    t_half_defined: bool
    k_ci: Optional[Tuple[float, float]] = None
    a_inf_ci: Optional[Tuple[float, float]] = None
    n_points: int = 0
    k_identifiable: bool = True
    notes: list = field(default_factory=list)

    def as_row(self):
        d = asdict(self)
        d["t_half"] = "undefined" if self.t_half is None else round(self.t_half, 2)
        d["notes"] = "; ".join(self.notes)
        return d


def fit_variant(t, activity, variant="", a0=100.0, n_boot=0, seed=0) -> InactivationFit:
    """Fit the two-parameter model to one inactivation curve.

    n_boot > 0 adds percentile bootstrap CIs by resampling residuals.
    Flags k as non-identifiable when the decay is faster than the sampling grid.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(activity, dtype=float)

    def model(tt, k, a_inf):
        return residual_activity(tt, k, a_inf, a0=a0)

    popt, _ = curve_fit(model, t, y, p0=(0.05, float(np.min(y))),
                        bounds=([1e-6, 0.0], [10.0, a0]), maxfev=20000)
    k, a_inf = float(popt[0]), float(popt[1])

    k_ci = a_inf_ci = None
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        resid = y - model(t, k, a_inf)
        ks, ais = [], []
        for _ in range(n_boot):
            yb = model(t, k, a_inf) + rng.choice(resid, size=resid.size, replace=True)
            try:
                pb, _ = curve_fit(model, t, yb, p0=popt,
                                  bounds=([1e-6, 0.0], [10.0, a0]), maxfev=20000)
                ks.append(pb[0]); ais.append(pb[1])
            except RuntimeError:
                continue
        if ks:
            k_ci = tuple(float(v) for v in np.percentile(ks, [2.5, 97.5]))
            a_inf_ci = tuple(float(v) for v in np.percentile(ais, [2.5, 97.5]))

    # k is only identifiable if the sampling grid resolves the fast phase:
    # at least one measured point must fall before ~2 time constants.
    t_pos = t[t > 0]
    identifiable = bool(t_pos.size and t_pos.min() < 2.0 / k)
    notes = []
    if not identifiable:
        notes.append(
            f"k not identifiable: first point at {t_pos.min():.0f} min but tau={1/k:.1f} min; "
            "need earlier samples. A_inf remains well determined.")

    th = half_life(k, a_inf, a0=a0)
    return InactivationFit(variant=variant, k=k, a_inf=a_inf, tau=1.0 / k,
                           t_half=th, t_half_defined=th is not None,
                           k_ci=k_ci, a_inf_ci=a_inf_ci, n_points=int(t.size),
                           k_identifiable=identifiable, notes=notes)


def fit_table(df, variant_col="variant", time_col="time_min", value_col="activity_pct",
              n_boot=500, seed=1):
    """Fit every variant in a tidy dataframe. Returns a list of InactivationFit."""
    out = []
    for variant, g in df.groupby(variant_col):
        g = g.sort_values(time_col)
        out.append(fit_variant(g[time_col].values, g[value_col].values,
                               variant=str(variant), n_boot=n_boot, seed=seed))
    return out
