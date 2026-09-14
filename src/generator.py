"""Synthetic Search-Console-shaped corpus with planted cannibalisation.

Shape: for each query we have a daily click count per landing page over a
window of days. Every query shares a single site-wide demand curve with strong
seasonality and an upward trend, because that confound is the whole reason the
naive statistic fails.

Ground truth: a small, known set of page pairs genuinely cannibalise. Every
other pair is independent given the day's demand.
"""

import math
import random
from dataclasses import dataclass, field

# Seasonal amplitude of site-wide demand. 0.6 means peak days carry 4x the
# clicks of trough days -- a course catalogue's enrolment peak is at least that
# sharp, and it is what makes raw click correlation useless.
SEASONAL_AMPLITUDE = 0.6
SEASON_PERIOD_DAYS = 30.0

# Multiplicative growth per day. 0.008 compounds to about +60% over a 60-day
# window: a site that is growing, so every page's series trends up together.
DAILY_TREND = 0.008

# Query popularity is heavily skewed on a real catalogue: a handful of head
# queries carry an order of magnitude more traffic than the long tail. This is
# not decoration -- without it the pair with the most clicks is systematically
# the pair from a two-page query, and "rank by overlap volume" would recover the
# planted pairs for a reason that has nothing to do with cannibalisation. The
# floor keeps every query's daily pair counts large enough for the chi-square
# approximation to the binomial split to hold.
POPULARITY_LOG_SIGMA = 0.7
MIN_QUERY_CLICKS = 40


@dataclass
class Query:
    """One search query and the daily clicks of each page that ranks for it."""

    name: str
    pages: list = field(default_factory=list)        # page identifiers
    clicks: dict = field(default_factory=dict)       # page -> list of daily clicks


@dataclass
class Corpus:
    days: int
    queries: list = field(default_factory=list)
    # Ground truth: set of frozenset({page_a, page_b}) that genuinely compete.
    cannibal_pairs: set = field(default_factory=set)


def _query_popularity(rng, mean_clicks):
    """Lognormal head/tail popularity with the mean preserved."""
    draw = math.exp(rng.gauss(0.0, POPULARITY_LOG_SIGMA) - 0.5 * POPULARITY_LOG_SIGMA ** 2)
    return max(MIN_QUERY_CLICKS, mean_clicks * draw)


def _demand_series(rng, days, mean_clicks):
    """Daily demand for one query: seasonality x trend x multiplicative noise."""
    series = []
    for t in range(days):
        seasonal = 1.0 + SEASONAL_AMPLITUDE * math.sin(2.0 * math.pi * t / SEASON_PERIOD_DAYS)
        trend = (1.0 + DAILY_TREND) ** t
        # Lognormal day noise, so the shared shock is multiplicative like real
        # traffic rather than additive.
        noise = math.exp(rng.gauss(0.0, 0.15))
        series.append(max(1, int(round(mean_clicks * seasonal * trend * noise))))
    return series


def _binomial(rng, n, p):
    """Exact Binomial(n, p) draw.

    random.Random.binomialvariate (Python 3.12+, BTPE) is exact and O(1)
    expected; the per-trial Bernoulli loop is the exact fallback for 3.11. The
    loop made the 500-site simulations take minutes, which is why it is not the
    default.
    """
    if p <= 0.0:
        return 0
    if p >= 1.0:
        return n
    if hasattr(rng, "binomialvariate"):
        return rng.binomialvariate(n, p)
    return sum(1 for _ in range(n) if rng.random() < p)


def _multinomial(rng, n, weights):
    """Split n items over len(weights) buckets by sequential binomial draws."""
    total = sum(weights)
    out = []
    remaining = n
    remaining_weight = total
    for w in weights[:-1]:
        if remaining <= 0 or remaining_weight <= 0:
            out.append(0)
            remaining_weight -= w
            continue
        p = min(1.0, max(0.0, w / remaining_weight))
        k = _binomial(rng, remaining, p)
        out.append(k)
        remaining -= k
        remaining_weight -= w
    out.append(max(0, remaining))
    return out


def _dirichlet_like(rng, k, concentration=4.0):
    draws = [rng.gammavariate(concentration, 1.0) for _ in range(k)]
    s = sum(draws)
    return [d / s for d in draws]


def generate_corpus(
    seed,
    days=60,
    n_null_queries=60,
    n_cannibal_queries=12,
    mean_query_clicks=120,
    displacement=0.22,
):
    """Build one site.

    `displacement` is the planted effect size: on a cannibalising pair, a latent
    daily coin flip moves this fraction of the pair's combined clicks from one
    page to the other. Crucially the pair TOTAL is untouched, so the planted
    signal is invisible to any statistic that looks at volume or overlap -- it
    lives only in the day-to-day split. That is what makes this a fair test of
    the within-pair-share statistic rather than a giveaway.
    """
    rng = random.Random(seed)
    corpus = Corpus(days=days)

    for qi in range(n_null_queries):
        # 2-4 pages rank for a typical query; all independent given the day.
        k = rng.randint(2, 4)
        weights = _dirichlet_like(rng, k)
        pages = ["null-q%03d-p%d" % (qi, j) for j in range(k)]
        demand = _demand_series(rng, days, _query_popularity(rng, mean_query_clicks))
        clicks = {p: [] for p in pages}
        for t in range(days):
            parts = _multinomial(rng, demand[t], weights)
            for p, c in zip(pages, parts):
                clicks[p].append(c)
        corpus.queries.append(Query(name="null-q%03d" % qi, pages=pages, clicks=clicks))

    for qi in range(n_cannibal_queries):
        # Cannibalising queries carry exactly two pages on purpose. If a third
        # page shared the query, its pair with a switching page would also show
        # a heterogeneous split -- genuinely so -- and the "ground truth" label
        # would be wrong rather than the detector. Keeping k=2 keeps the planted
        # truth unambiguous.
        pages = ["cann-q%03d-p0" % qi, "cann-q%03d-p1" % qi]
        base = rng.uniform(0.35, 0.65)
        demand = _demand_series(rng, days, _query_popularity(rng, mean_query_clicks))
        clicks = {p: [] for p in pages}
        for t in range(days):
            swing = displacement if rng.random() < 0.5 else -displacement
            p_a = min(0.95, max(0.05, base + swing))
            parts = _multinomial(rng, demand[t], [p_a, 1.0 - p_a])
            for p, c in zip(pages, parts):
                clicks[p].append(c)
        corpus.queries.append(Query(name="cann-q%03d" % qi, pages=pages, clicks=clicks))
        corpus.cannibal_pairs.add(frozenset(pages))

    return corpus


def all_pairs(corpus):
    """Every within-query page pair -- one hypothesis test each."""
    out = []
    for q in corpus.queries:
        for i in range(len(q.pages)):
            for j in range(i + 1, len(q.pages)):
                out.append((q, q.pages[i], q.pages[j]))
    return out
