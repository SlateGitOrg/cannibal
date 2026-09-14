"""Distribution tails and multiple-testing procedures, written from scratch.

The house rule for this catalogue is standard library only, so there is no
scipy.stats.chi2 and no statsmodels.multipletests. Both are reimplemented here.
The incomplete-gamma routines follow the standard series / continued-fraction
split (Numerical Recipes 6.2): the power series converges fast for x < a+1 and
the Lentz continued fraction for the upper tail converges fast for x >= a+1.
Using the wrong branch does not merely lose accuracy, it fails to converge.
"""

import math

# Relative convergence target for the iterative gamma routines. 1e-14 is about
# four bits above double epsilon (2.2e-16) -- tight enough that the p-values are
# exact to every digit we ever print, loose enough that the loops always stop.
_EPS = 1e-14
_MAX_ITER = 500

# Smallest positive normal double. Used to bootstrap the continued fraction so
# the first division is never 0/0.
_TINY = 1e-300


def _lower_gamma_series(a, x):
    """Regularised lower incomplete gamma P(a, x) by power series."""
    if x <= 0.0:
        return 0.0
    ap = a
    total = 1.0 / a
    term = total
    for _ in range(_MAX_ITER):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _upper_gamma_cf(a, x):
    """Regularised upper incomplete gamma Q(a, x) by continued fraction."""
    b = x + 1.0 - a
    c = 1.0 / _TINY
    d = 1.0 / b
    h = d
    for i in range(1, _MAX_ITER + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _TINY:
            d = _TINY
        c = b + an / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def gamma_sf(a, x):
    """Q(a, x) = 1 - P(a, x), the regularised upper incomplete gamma."""
    if x < 0.0 or a <= 0.0:
        raise ValueError("gamma_sf requires a > 0 and x >= 0")
    if x == 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _lower_gamma_series(a, x)
    return _upper_gamma_cf(a, x)


def chi2_sf(statistic, df):
    """Upper-tail probability P(X > statistic) for chi-square with df degrees."""
    if df <= 0:
        raise ValueError("chi2_sf requires df >= 1")
    if statistic <= 0.0:
        return 1.0
    return gamma_sf(df / 2.0, statistic / 2.0)


def normal_sf(z):
    """Upper-tail probability of the standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def benjamini_hochberg(pvalues, alpha):
    """Benjamini-Hochberg step-up FDR control.

    Returns (rejected, qvalues). `rejected` is a list of bools aligned with the
    input order; `qvalues` are the BH-adjusted p-values.

    The step-up part is the part a hand-rolled version usually gets wrong. You
    do NOT reject exactly the hypotheses whose own p_(i) <= i/m*alpha. You find
    the LARGEST i satisfying that inequality and reject everything ranked at or
    below it, including hypotheses that individually fail their own threshold.
    Testing each p-value against its own line independently is a different,
    less powerful procedure and is not the BH procedure.
    """
    m = len(pvalues)
    rejected = [False] * m
    qvalues = [1.0] * m
    if m == 0:
        return rejected, qvalues

    order = sorted(range(m), key=lambda i: pvalues[i])

    k = 0  # number of hypotheses to reject, i.e. the largest passing rank
    for rank, idx in enumerate(order, start=1):
        if pvalues[idx] <= (rank / m) * alpha:
            k = rank
    for rank, idx in enumerate(order, start=1):
        if rank <= k:
            rejected[idx] = True

    # Adjusted p-values: q_(i) = min over j >= i of (m/j) * p_(j). The running
    # minimum from the largest rank downwards enforces monotonicity, which is
    # what makes "q <= alpha" agree exactly with the step-up decision above.
    running = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        running = min(running, (m / rank) * pvalues[idx])
        qvalues[idx] = min(1.0, running)
    return rejected, qvalues


def bonferroni(pvalues, alpha):
    """Bonferroni FWER control: reject where p <= alpha/m."""
    m = len(pvalues)
    if m == 0:
        return []
    return [p <= alpha / m for p in pvalues]


def uncorrected(pvalues, alpha):
    """The naive procedure: reject wherever p <= alpha, ignoring multiplicity."""
    return [p <= alpha for p in pvalues]
