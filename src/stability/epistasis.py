"""Additivity and ranking tests for the four variants.

Why a transform is needed
-------------------------
A_inf is a surviving fraction, not an energy, and fractions are not additive.
Asking whether a double mutant is "the sum of its singles" only makes sense on a
scale where effects add. If the plateau reflects a partition between an active
and an inactive population, the natural scale is the logit:

    G(A) = -ln( A / (100 - A) )

in units of RT. On that scale an additive double mutant satisfies

    G(AB) - G(wt) = [G(A) - G(wt)] + [G(B) - G(wt)]

and the deviation from it is the epistasis.

THIS IS AN ASSUMPTION, not a measurement: it treats the plateau as a two-state
equilibrium partition. It is consistent with the Sadana-Henley picture (a final
state that retains activity) and with the oligomer-dissociation hypothesis, but
it is not established. If the plateau is something else, the transform is not
meaningful and the epistasis number below is not either. State it when using it.
"""

import math
from itertools import permutations
from typing import Dict, Sequence, Tuple


def logit_energy(a_inf_pct: float) -> float:
    """Pseudo free energy in RT units from a plateau percentage."""
    if not 0.0 < a_inf_pct < 100.0:
        raise ValueError("a_inf must be strictly between 0 and 100 %")
    return -math.log(a_inf_pct / (100.0 - a_inf_pct))


def inverse_logit_energy(g: float) -> float:
    return 100.0 / (1.0 + math.exp(g))


def epistasis(wt: float, single_a: float, single_b: float, double: float) -> Dict:
    """Deviation of the double mutant from additivity, on the logit scale.

    All four arguments are A_inf percentages.
    """
    g_wt, g_a, g_b, g_ab = (logit_energy(x) for x in (wt, single_a, single_b, double))
    dd_a, dd_b, dd_ab = g_a - g_wt, g_b - g_wt, g_ab - g_wt
    additive = dd_a + dd_b
    return {
        "ddG_single_a": dd_a,
        "ddG_single_b": dd_b,
        "ddG_double_observed": dd_ab,
        "ddG_double_additive": additive,
        "epistasis": dd_ab - additive,
        "a_inf_double_observed": double,
        "a_inf_double_if_additive": inverse_logit_energy(g_wt + additive),
    }


def exact_rank_pvalue(predicted: Sequence[float], measured: Sequence[float],
                      higher_is_better: bool = True) -> Tuple[float, float]:
    """Exact permutation p-value for rank agreement. Feasible because n is tiny.

    Returns (spearman_rho, p_value). With n = 4 the smallest achievable p-value
    for a perfect match is 1/24 = 0.0417, so perfect agreement is the only
    outcome that reaches significance at 0.05. Anything less is descriptive.
    """
    n = len(measured)
    assert n == len(predicted)

    def rho(x, y):
        rx = [sorted(x).index(v) for v in x]
        ry = [sorted(y).index(v) for v in y]
        mx, my = sum(rx) / n, sum(ry) / n
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
        return num / den if den else 0.0

    obs = rho(predicted, measured)
    if not higher_is_better:
        obs = -obs
    perms = list(permutations(predicted))
    count = sum(1 for p in perms if rho(p, measured) >= obs)
    return obs, count / len(perms)
