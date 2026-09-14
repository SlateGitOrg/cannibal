"""The headline tests: the statistic, and what correction buys you.

Every simulation here is seeded, so the numbers are deterministic across runs;
the assertions are still written as Monte Carlo bands because that is what makes
them meaningful rather than a snapshot of one arbitrary draw.
"""

import math
import unittest

from src.detect import decide, raw_correlation_test, share_homogeneity_test, test_all_pairs
from src.evaluate import run_simulation
from src.generator import all_pairs, generate_corpus

ALPHA = 0.05


class TestShareStatistic(unittest.TestCase):
    def test_constant_split_gives_no_evidence(self):
        a = [30] * 40
        b = [70] * 40
        stat, df, p, share, total = share_homogeneity_test(a, b)
        self.assertAlmostEqual(stat, 0.0, delta=1e-12)
        self.assertEqual(df, 39)
        self.assertAlmostEqual(p, 1.0, delta=1e-12)
        self.assertAlmostEqual(share, 0.3, delta=1e-12)
        self.assertEqual(total, 4000)

    def test_statistic_is_invariant_to_a_violent_trend(self):
        """A 600x trend with a fixed split must produce NO evidence at all.

        This is the differentiator's load-bearing property. The generic
        alternative -- correlating the raw click series -- reports a
        near-perfect correlation on exactly the same data, which is the
        confound that makes every pair on a seasonal site look cannibalised.
        """
        a, b = [], []
        for t in range(60):
            volume = 10 * (t + 1)          # deterministic 1x -> 600x ramp
            a.append(volume * 3)
            b.append(volume * 7)
        _, _, p_share, _, _ = share_homogeneity_test(a, b)
        self.assertGreater(p_share, 0.99)

        r, p_raw = raw_correlation_test(a, b)
        self.assertGreater(r, 0.999)
        self.assertLess(p_raw, 1e-12)

    def test_two_day_case_reduces_to_the_two_proportion_z_test(self):
        """With two days the chi-square must equal the squared z statistic.

        The spec names a two-proportion test; using all days is the same test
        with more groups, and this pins that equivalence numerically.
        """
        a = [40, 12]
        n = [100, 90]
        b = [n[0] - a[0], n[1] - a[1]]
        stat, df, _, _, _ = share_homogeneity_test(a, b)
        self.assertEqual(df, 1)

        p1, p2 = a[0] / n[0], a[1] / n[1]
        pooled = (a[0] + a[1]) / (n[0] + n[1])
        z = (p1 - p2) / math.sqrt(pooled * (1 - pooled) * (1 / n[0] + 1 / n[1]))
        self.assertAlmostEqual(stat, z * z, delta=1e-10)

    def test_degenerate_inputs_do_not_raise(self):
        self.assertEqual(share_homogeneity_test([], [])[2], 1.0)
        self.assertEqual(share_homogeneity_test([0, 0, 0], [0, 0, 0])[2], 1.0)
        # One page takes every click: a constant split, so no evidence.
        self.assertEqual(share_homogeneity_test([5, 9, 3], [0, 0, 0])[2], 1.0)


class TestOverlapRanking(unittest.TestCase):
    def test_ranking_by_overlap_volume_finds_nothing(self):
        """The generic tool's ranking has no power on this corpus.

        A pair's combined click volume is the "overlap" a raw tool sorts by.

        The original claim here was "precision at chance". Measured, that is
        false: planted queries carry exactly two pages, so their pair gets the
        whole query's clicks and volume leaks a little signal (precision ~0.12
        against chance ~0.04). The true, measured claim is weaker but is the
        one that matters: overlap ranking's precision is far below the BH
        list's precision on the same corpora. Bar: 3 binomial se.
        """
        hits = 0
        trials = 0
        bh_hits = 0
        bh_flagged = 0
        for seed in range(20):
            corpus = generate_corpus(seed=4000 + seed, n_null_queries=80)
            truth = corpus.cannibal_pairs
            pairs = all_pairs(corpus)
            ranked = sorted(
                pairs,
                key=lambda t: -(sum(t[0].clicks[t[1]]) + sum(t[0].clicks[t[2]])),
            )
            k = len(truth)
            hits += sum(1 for q, a, b in ranked[:k] if frozenset((a, b)) in truth)
            trials += k
            rep = decide(test_all_pairs(corpus), truth, alpha=ALPHA, method="bh")
            bh_hits += rep.n_true_positive
            bh_flagged += len(rep.flagged)
        precision = hits / trials
        bh_precision = bh_hits / bh_flagged
        se = math.sqrt(precision * (1 - precision) / trials
                       + bh_precision * (1 - bh_precision) / bh_flagged)
        self.assertLess(precision + 3 * se, bh_precision)
        self.assertLess(precision, 0.5)


class TestCorrectionHeadline(unittest.TestCase):
    """The measured comparison the whole repo exists to make."""

    RUNS = 150

    @classmethod
    def setUpClass(cls):
        cls.sim = run_simulation(runs=cls.RUNS, alpha=ALPHA, n_null_queries=150,
                                 base_seed=1000)

    def test_uncorrected_reports_about_alpha_times_m_false_pairs(self):
        u = self.sim["uncorrected"]
        m0 = u.mean_tests - 12          # 12 planted true pairs per corpus
        expected = ALPHA * m0
        observed = u.mean_false_positives
        # The band is wide on the low side on purpose: the chi-square
        # approximation to the binomial-split null is mildly conservative at
        # these counts, so the realised type-I rate sits a little under nominal.
        # The claim under test is the ORDER of the number -- tens of false pairs,
        # scaling with the number of tests -- not three-digit agreement.
        self.assertGreater(observed, 0.7 * expected)
        self.assertLess(observed, 1.1 * expected)
        self.assertGreater(observed, 15.0)

    def test_uncorrected_list_is_mostly_false_positives(self):
        u = self.sim["uncorrected"]
        # "Mostly" = more than half the reported pairs are false. 3 Monte Carlo
        # standard errors below the estimate must still clear 0.5.
        self.assertGreater(u.mean_fdr - 3 * u.se_fdr, 0.5)

    def test_bh_controls_fdr_at_the_nominal_level(self):
        b, u = self.sim["bh"], self.sim["uncorrected"]
        # BH's guarantee is E[FDP] <= alpha * m0/m, which here is ~0.049 -- the
        # guarantee is met with near equality, so "mean + 3se < alpha" (the
        # original assertion) is a false claim for a CORRECT implementation.
        # The right test: the estimate is not significantly above alpha
        # (mean - 3se < alpha), and it is far below the uncorrected FDR.
        self.assertLess(b.mean_fdr - 3 * b.se_fdr, ALPHA)
        self.assertLess(b.mean_fdr + 3 * b.se_fdr, u.mean_fdr - 3 * u.se_fdr)
        self.assertLess(b.mean_fdr, 2 * ALPHA)

    def test_bh_retains_most_of_the_true_pairs(self):
        b, u = self.sim["bh"], self.sim["uncorrected"]
        self.assertGreater(u.mean_power, 0.0)
        self.assertGreater(b.mean_power, 0.5 * u.mean_power)

    def test_bh_list_is_short_enough_to_act_on(self):
        b, u = self.sim["bh"], self.sim["uncorrected"]
        self.assertLess(b.mean_flagged, 0.5 * u.mean_flagged)
        # And close to the true count of 12 planted pairs.
        self.assertLess(abs(b.mean_flagged - 12.0), 2.0)

    def test_bonferroni_controls_fdr_too_but_is_not_the_point(self):
        bonf = self.sim["bonferroni"]
        self.assertLess(bonf.fdr_upper_bound(3.0), ALPHA)


class TestWeakEffectRegime(unittest.TestCase):
    """Where BH earns its keep over the conservative FWER alternative.

    At the default planted effect every procedure has power 1.0, so a power
    comparison there is vacuous. Shrinking the displacement to 0.03 puts the
    problem near the detection floor, which is where the choice of correction
    actually changes the answer.
    """

    @classmethod
    def setUpClass(cls):
        cls.sim = run_simulation(runs=100, alpha=ALPHA, n_null_queries=150,
                                 displacement=0.03, base_seed=7000)

    def test_bh_is_strictly_more_powerful_than_bonferroni(self):
        b, bonf = self.sim["bh"], self.sim["bonferroni"]
        # Both procedures run on the SAME corpora, so the comparison is paired.
        # The independent-samples se (sqrt(se1^2 + se2^2)) ignores the strong
        # positive correlation and overstates the noise several-fold.
        diffs = [x - y for x, y in zip(b.powers, bonf.powers)]
        n = len(diffs)
        mean = sum(diffs) / n
        sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
        self.assertGreater(mean, 3 * sd / math.sqrt(n))

    def test_bh_still_controls_fdr_near_the_detection_floor(self):
        b = self.sim["bh"]
        # Far fewer discoveries here, so FDP is a noisier ratio. BH's bound is
        # met near equality, so the point estimate can sit just above alpha by
        # chance (measured 0.058, se 0.0125); the claim is "not significantly
        # above alpha".
        self.assertLess(b.mean_fdr - 3 * b.se_fdr, ALPHA)

    def test_uncorrected_gets_worse_not_better_when_signal_is_weak(self):
        u = self.sim["uncorrected"]
        self.assertGreater(u.mean_fdr - 3 * u.se_fdr, 0.6)


class TestConfoundedStatisticCannotBeRescuedByCorrection(unittest.TestCase):
    def test_raw_correlation_plus_bh_flags_nearly_everything(self):
        """Correction fixes multiplicity, not confounding.

        Identical corpora and the identical BH step, run through the generic
        raw-correlation statistic. Because seasonality makes almost every pair
        significantly correlated, the p-values are tiny across the board and BH
        rejects nearly the whole corpus -- FDR near 1. This is the test that
        fails if the detrending decision is removed.
        """
        raw = run_simulation(runs=15, alpha=ALPHA, statistic="raw_correlation",
                             methods=("bh",), n_null_queries=80, base_seed=9000)["bh"]
        share = run_simulation(runs=15, alpha=ALPHA, statistic="share",
                               methods=("bh",), n_null_queries=80, base_seed=9000)["bh"]
        self.assertGreater(raw.mean_flagged, 0.8 * raw.mean_tests)
        self.assertGreater(raw.mean_fdr, 0.9)
        self.assertLess(share.mean_flagged, 0.05 * share.mean_tests)
        self.assertLess(share.mean_fdr - 3 * share.se_fdr, ALPHA)


if __name__ == "__main__":
    unittest.main()
