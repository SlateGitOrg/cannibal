"""Repeated-simulation evaluation of the multiplicity procedures.

A single simulated site tells you almost nothing: the realised false discovery
proportion of one run is a ratio of two small integers and swings wildly. FDR
is an EXPECTATION over repeated experiments, so it is only meaningful averaged
over many corpora, with a Monte Carlo standard error attached.
"""

import math
from dataclasses import dataclass

from .detect import decide, test_all_pairs
from .generator import generate_corpus


@dataclass
class Summary:
    method: str
    runs: int
    mean_fdr: float
    se_fdr: float
    mean_power: float
    se_power: float
    mean_flagged: float
    mean_false_positives: float
    mean_tests: float
    # Per-run values, aligned by corpus seed across methods, so paired
    # comparisons between procedures can use the correct paired standard error.
    fdps: list = None
    powers: list = None

    def fdr_upper_bound(self, k=3.0):
        """Mean FDR plus k Monte Carlo standard errors.

        The assertion bar is derived, not chosen: k=3 is a ~99.7% one-sided
        normal band on the simulation mean, so a procedure that truly controls
        FDR at alpha clears it and one that does not, does not.
        """
        return self.mean_fdr + k * self.se_fdr


def _mean_se(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def run_simulation(
    runs=200,
    alpha=0.05,
    methods=("uncorrected", "bh", "bonferroni"),
    statistic="share",
    base_seed=1000,
    **corpus_kwargs
):
    """Simulate `runs` independent sites and score every method on each."""
    fdps = {m: [] for m in methods}
    powers = {m: [] for m in methods}
    flagged = {m: [] for m in methods}
    falses = {m: [] for m in methods}
    tests_per_run = []

    for i in range(runs):
        corpus = generate_corpus(seed=base_seed + i, **corpus_kwargs)
        results = test_all_pairs(corpus, statistic=statistic)
        tests_per_run.append(len(results))
        for m in methods:
            rep = decide(results, corpus.cannibal_pairs, alpha=alpha, method=m)
            fdps[m].append(rep.fdp)
            powers[m].append(rep.power)
            flagged[m].append(len(rep.flagged))
            falses[m].append(rep.n_false_positive)

    summaries = {}
    for m in methods:
        mf, sf = _mean_se(fdps[m])
        mp, sp = _mean_se(powers[m])
        summaries[m] = Summary(
            method=m,
            runs=runs,
            mean_fdr=mf,
            se_fdr=sf,
            mean_power=mp,
            se_power=sp,
            mean_flagged=sum(flagged[m]) / runs,
            mean_false_positives=sum(falses[m]) / runs,
            mean_tests=sum(tests_per_run) / runs,
            fdps=fdps[m],
            powers=powers[m],
        )
    return summaries
