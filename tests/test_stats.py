"""Tests for the hand-rolled distribution tails and the BH procedure."""

import math
import random
import unittest

from src.stats import (
    benjamini_hochberg,
    bonferroni,
    chi2_sf,
    gamma_sf,
    normal_sf,
    uncorrected,
)


class TestDistributionTails(unittest.TestCase):
    def test_gamma_sf_matches_closed_form_exponential(self):
        # Q(1, x) = exp(-x) exactly. This crosses the series/continued-fraction
        # branch boundary at x = a + 1 = 2, so it catches a wrong branch split.
        for x in (0.01, 0.5, 1.0, 1.9, 2.0, 2.1, 5.0, 20.0, 60.0):
            self.assertAlmostEqual(gamma_sf(1.0, x), math.exp(-x), delta=1e-13)

    def test_chi2_sf_two_df_closed_form(self):
        # Chi-square with 2 df is Exponential(1/2): sf(x) = exp(-x/2).
        for x in (0.1, 1.0, 4.0, 9.21034, 30.0, 100.0):
            self.assertAlmostEqual(chi2_sf(x, 2), math.exp(-x / 2.0), delta=1e-13)

    def test_chi2_sf_matches_published_critical_values(self):
        # Standard 95th-percentile chi-square critical values (5-6 dp, from
        # published tables). sf at the critical value must be 0.05.
        table = {1: 3.841459, 2: 5.991465, 5: 11.070498,
                 10: 18.307038, 30: 43.772972, 59: 77.930524}
        for df, crit in table.items():
            self.assertAlmostEqual(chi2_sf(crit, df), 0.05, delta=1e-6)

    def test_chi2_sf_upper_tail_percentiles(self):
        # 99.9th percentiles, to exercise the far tail where the continued
        # fraction is doing the work.
        for df, crit in {1: 10.827566, 5: 20.515001, 10: 29.588298}.items():
            self.assertAlmostEqual(chi2_sf(crit, df), 0.001, delta=2e-6)

    def test_chi2_sf_is_monotone_decreasing(self):
        prev = 1.0
        for x in [i * 0.5 for i in range(1, 120)]:
            cur = chi2_sf(x, 7)
            self.assertLess(cur, prev)
            prev = cur

    def test_normal_sf(self):
        self.assertAlmostEqual(normal_sf(0.0), 0.5, delta=1e-15)
        self.assertAlmostEqual(normal_sf(1.959963985), 0.025, delta=1e-9)
        self.assertAlmostEqual(normal_sf(-1.959963985), 0.975, delta=1e-9)


class TestBenjaminiHochberg(unittest.TestCase):
    def test_worked_example(self):
        # The canonical BH(1995) style worked example. Thresholds i/m * alpha
        # are 0.005, 0.010, ..., 0.050; only ranks 1 and 2 clear their line and
        # no higher rank does, so exactly two hypotheses are rejected.
        p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205, 0.212, 0.216]
        rejected, _ = benjamini_hochberg(p, 0.05)
        self.assertEqual(rejected, [True, True] + [False] * 8)

    def test_step_up_rejects_below_the_largest_passing_rank(self):
        # THE step-up property, and the thing a naive implementation gets wrong.
        # m=4, alpha=0.05 -> lines 0.0125, 0.025, 0.0375, 0.05.
        # p=0.030 FAILS its own rank-2 line, but rank 3 (p=0.031 <= 0.0375)
        # passes, so BH rejects ranks 1..3 including the 0.030.
        p = [0.001, 0.030, 0.031, 0.900]
        rejected, _ = benjamini_hochberg(p, 0.05)
        self.assertEqual(rejected, [True, True, True, False])
        # A per-hypothesis ("each p against its own line") reading rejects only
        # two here. Confirm the two readings really do differ, so this test
        # would fail against that wrong implementation.
        own_line = [p[i] <= ((rank + 1) / len(p)) * 0.05
                    for rank, i in enumerate(sorted(range(len(p)), key=lambda j: p[j]))]
        self.assertEqual(sum(own_line), 2)
        self.assertEqual(sum(rejected), 3)

    def test_qvalues_agree_with_rejections(self):
        rng = random.Random(11)
        for _ in range(200):
            m = rng.randint(1, 60)
            p = [rng.random() ** 3 for _ in range(m)]
            alpha = rng.choice([0.01, 0.05, 0.1])
            rejected, q = benjamini_hochberg(p, alpha)
            self.assertEqual(rejected, [qi <= alpha + 1e-15 for qi in q])

    def test_qvalues_are_monotone_in_p(self):
        rng = random.Random(12)
        p = sorted(rng.random() for _ in range(50))
        _, q = benjamini_hochberg(p, 0.05)
        for a, b in zip(q, q[1:]):
            self.assertLessEqual(a, b + 1e-15)

    def test_bh_is_between_bonferroni_and_uncorrected(self):
        # BH must never be stricter than Bonferroni nor looser than no
        # correction at all; violating either bracket is a real bug.
        rng = random.Random(13)
        for _ in range(200):
            m = rng.randint(5, 100)
            p = [rng.random() ** 2 for _ in range(m)]
            bh, _ = benjamini_hochberg(p, 0.05)
            bonf = bonferroni(p, 0.05)
            none = uncorrected(p, 0.05)
            self.assertLessEqual(sum(bonf), sum(bh))
            self.assertLessEqual(sum(bh), sum(none))
            for i in range(m):
                if bonf[i]:
                    self.assertTrue(bh[i])
                if bh[i]:
                    self.assertTrue(none[i])

    def test_controls_fdr_under_the_global_null(self):
        """Independent uniform p-values: BH must hold E[FDP] at or below alpha.

        Under the complete null BH's FDR equals alpha exactly (m0 = m), so this
        is a two-sided check against a closed-form target, not a loose bound.
        """
        rng = random.Random(2024)
        alpha, m, runs = 0.05, 200, 2000
        fdps = []
        for _ in range(runs):
            p = [rng.random() for _ in range(m)]
            rejected, _ = benjamini_hochberg(p, alpha)
            n = sum(rejected)
            fdps.append(1.0 if n else 0.0)
        mean = sum(fdps) / runs
        # Under the global null every rejection is false, so FDP is 1 whenever
        # anything is rejected and 0 otherwise; E[FDP] = P(any rejection),
        # which BH pins at exactly alpha. Binomial se of that estimate:
        se = math.sqrt(alpha * (1 - alpha) / runs)
        self.assertLess(abs(mean - alpha), 3 * se)


if __name__ == "__main__":
    unittest.main()
