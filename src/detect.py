"""Cannibalisation test statistics and the detection pipeline.

Two statistics live here on purpose:

  * `share_homogeneity_test` -- the one we ship.
  * `raw_correlation_test`   -- the generic version, kept so the README's claim
                                about it is measured rather than asserted.
"""

import math
from dataclasses import dataclass

from .generator import all_pairs
from .stats import benjamini_hochberg, bonferroni, chi2_sf, normal_sf, uncorrected


@dataclass
class PairResult:
    query: str
    page_a: str
    page_b: str
    statistic: float
    df: int
    pvalue: float
    share: float          # pooled share of clicks going to page_a
    total_clicks: int


def share_homogeneity_test(clicks_a, clicks_b):
    """Test whether page_a's share of the PAIR's clicks is constant over days.

    Why this statistic and not a correlation of the two click series:

    Both pages ride the same query's daily demand, which has seasonality and a
    trend. Any statistic computed on raw volumes is dominated by that shared
    factor -- both series rise at Christmas, so every pair on the site looks
    related. Conditioning on the pair's daily TOTAL removes the shared factor
    exactly, not approximately: if clicks_a[t] ~ Poisson(D_t * w_a) and
    clicks_b[t] ~ Poisson(D_t * w_b), then conditional on the total n_t the
    split is Binomial(n_t, w_a/(w_a+w_b)) and D_t has cancelled out of the
    distribution entirely. Whatever the demand curve does, the null split is
    the same constant proportion every day.

    So cannibalisation -- one page systematically displacing the other for the
    same intent, day to day -- shows up as heterogeneity of that proportion.
    The classic test for equality of proportions across groups is Pearson's
    chi-square; with days as the groups it has df = (days used) - 1. For two
    days it reduces exactly to the squared two-proportion z statistic, which is
    the test the spec names; using all days instead of two halves simply uses
    all the information.

    Detrending the volumes and correlating the residuals would be the other
    defensible route. It was rejected because it needs a model of the trend
    (and gets a wrong null if the model is wrong), whereas conditioning on the
    pair total needs no model at all.
    """
    used_a = []
    used_n = []
    for a, b in zip(clicks_a, clicks_b):
        n = a + b
        if n > 0:
            used_a.append(a)
            used_n.append(n)

    df = len(used_n) - 1
    if df < 1:
        return 0.0, 0, 1.0, 0.0, 0

    total_n = sum(used_n)
    total_a = sum(used_a)
    p = total_a / total_n
    if p <= 0.0 or p >= 1.0:
        # One page took every click on every day. That is a constant split, so
        # there is no heterogeneity evidence -- and the chi-square variance term
        # is zero, so the statistic is undefined rather than infinite.
        return 0.0, df, 1.0, p, total_n

    stat = 0.0
    for a, n in zip(used_a, used_n):
        expected = n * p
        stat += (a - expected) ** 2 / (n * p * (1.0 - p))
    return stat, df, chi2_sf(stat, df), p, total_n


def raw_correlation_test(clicks_a, clicks_b):
    """The generic alternative: is the Pearson correlation of raw clicks != 0?

    Kept to be measured, not to be used. p-value via the Fisher z transform,
    which is adequate here because the point is that this test fires on
    essentially every pair, not where its tail is accurate to three digits.
    """
    n = len(clicks_a)
    if n < 5:
        return 0.0, 1.0
    mean_a = sum(clicks_a) / n
    mean_b = sum(clicks_b) / n
    sab = sum((a - mean_a) * (b - mean_b) for a, b in zip(clicks_a, clicks_b))
    saa = sum((a - mean_a) ** 2 for a in clicks_a)
    sbb = sum((b - mean_b) ** 2 for b in clicks_b)
    if saa <= 0.0 or sbb <= 0.0:
        return 0.0, 1.0
    r = sab / math.sqrt(saa * sbb)
    r = max(-0.999999, min(0.999999, r))
    z = 0.5 * math.log((1.0 + r) / (1.0 - r)) * math.sqrt(n - 3)
    return r, 2.0 * normal_sf(abs(z))


def test_all_pairs(corpus, statistic="share"):
    """Run one hypothesis test per within-query page pair."""
    results = []
    for query, a, b in all_pairs(corpus):
        ca, cb = query.clicks[a], query.clicks[b]
        if statistic == "share":
            stat, df, p, share, total = share_homogeneity_test(ca, cb)
        elif statistic == "raw_correlation":
            stat, p = raw_correlation_test(ca, cb)
            df = len(ca) - 2
            total = sum(ca) + sum(cb)
            share = (sum(ca) / total) if total else 0.0
        else:
            raise ValueError("unknown statistic: %r" % (statistic,))
        results.append(PairResult(query.name, a, b, stat, df, p, share, total))
    return results


@dataclass
class Report:
    method: str
    alpha: float
    n_tests: int
    flagged: list          # list of PairResult
    n_true_positive: int
    n_false_positive: int
    n_true_pairs: int

    @property
    def fdp(self):
        """Realised false discovery proportion for this single run."""
        n = len(self.flagged)
        return (self.n_false_positive / n) if n else 0.0

    @property
    def power(self):
        return (self.n_true_positive / self.n_true_pairs) if self.n_true_pairs else 0.0


def decide(results, truth, alpha=0.05, method="bh"):
    """Apply a multiplicity procedure and score it against planted truth."""
    pvalues = [r.pvalue for r in results]
    if method == "bh":
        flags, _ = benjamini_hochberg(pvalues, alpha)
    elif method == "uncorrected":
        flags = uncorrected(pvalues, alpha)
    elif method == "bonferroni":
        flags = bonferroni(pvalues, alpha)
    else:
        raise ValueError("unknown method: %r" % (method,))

    flagged = [r for r, f in zip(results, flags) if f]
    tp = sum(1 for r in flagged if frozenset((r.page_a, r.page_b)) in truth)
    return Report(
        method=method,
        alpha=alpha,
        n_tests=len(results),
        flagged=sorted(flagged, key=lambda r: r.pvalue),
        n_true_positive=tp,
        n_false_positive=len(flagged) - tp,
        n_true_pairs=len(truth),
    )
