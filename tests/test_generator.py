"""Tests that the planted ground truth is actually what the generator plants."""

import math
import statistics
import unittest

from src.detect import raw_correlation_test, share_homogeneity_test
from src.generator import Corpus, all_pairs, generate_corpus


def split_pairs(corpus):
    truth = corpus.cannibal_pairs
    null_pairs, true_pairs = [], []
    for q, a, b in all_pairs(corpus):
        (true_pairs if frozenset((a, b)) in truth else null_pairs).append((q, a, b))
    return null_pairs, true_pairs


class TestPlantedTruth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = generate_corpus(seed=99, n_null_queries=80, n_cannibal_queries=12)
        cls.null_pairs, cls.true_pairs = split_pairs(cls.corpus)

    def test_ground_truth_counts(self):
        self.assertEqual(len(self.corpus.cannibal_pairs), 12)
        self.assertEqual(len(self.true_pairs), 12)
        self.assertGreater(len(self.null_pairs), 100)

    def test_every_page_appears_in_exactly_one_query(self):
        seen = set()
        for q in self.corpus.queries:
            for p in q.pages:
                self.assertNotIn(p, seen)
                seen.add(p)

    def test_null_pairs_have_a_calibrated_chi_square_null(self):
        """Null pairs must produce statistics whose mean is the df.

        If the generator's "independent" pages were not actually conditionally
        multinomial on the day's demand, this mean would drift away from df and
        every downstream FDR number would be meaningless. The tolerance is the
        Monte Carlo se of the mean: Var(chi2_df) = 2*df, so the mean of N such
        statistics has se sqrt(2*df/N); dividing by df gives sqrt(2/(df*N)).
        """
        ratios = []
        for q, a, b in self.null_pairs:
            stat, df, _, _, _ = share_homogeneity_test(q.clicks[a], q.clicks[b])
            self.assertGreater(df, 0)
            ratios.append(stat / df)
        n = len(ratios)
        df = self.corpus.days - 1
        se = math.sqrt(2.0 / (df * n))
        self.assertLess(abs(statistics.fmean(ratios) - 1.0), 3 * se)

    def test_planted_pairs_are_strongly_overdispersed(self):
        # Every planted pair should sit far above the null's mean ratio of 1.
        for q, a, b in self.true_pairs:
            stat, df, p, _, _ = share_homogeneity_test(q.clicks[a], q.clicks[b])
            self.assertGreater(stat / df, 2.0)
            self.assertLess(p, 1e-6)

    def test_planted_effect_is_invisible_in_pair_volume(self):
        """The displacement moves share, never the pair's total clicks.

        This is what makes the corpus a fair test: a detector that ranks by
        volume or by overlap size has nothing to latch onto. If the generator
        leaked the signal into totals, the share statistic would be untestable.
        """
        def cv(pair):
            q, a, b = pair
            totals = [x + y for x, y in zip(q.clicks[a], q.clicks[b])]
            return statistics.pstdev(totals) / statistics.fmean(totals)

        null_cv = statistics.fmean(cv(p) for p in self.null_pairs)
        true_cv = statistics.fmean(cv(p) for p in self.true_pairs)
        # Shared seasonality dominates the coefficient of variation of the pair
        # total, so the two families must look alike. 3 se of the null family's
        # own spread is the bar.
        spread = statistics.stdev([cv(p) for p in self.null_pairs])
        se = spread / math.sqrt(len(self.true_pairs))
        self.assertLess(abs(true_cv - null_cv), 3 * se)

    def test_the_seasonal_confound_is_present(self):
        """Raw click series co-move even for pairs that do not compete.

        Without this the detrending argument would be hypothetical. Independent
        pages in DIFFERENT queries share only the shape of demand, and that
        alone already gives strongly positive correlations.
        """
        qs = [q for q in self.corpus.queries if q.name.startswith("null-")][:40]
        rs = []
        for i in range(0, len(qs) - 1, 2):
            a = qs[i].pages[0]
            b = qs[i + 1].pages[0]
            r, p = raw_correlation_test(qs[i].clicks[a], qs[i + 1].clicks[b])
            rs.append(r)
        self.assertGreater(statistics.fmean(rs), 0.5)

    def test_reproducible_for_a_seed(self):
        a = generate_corpus(seed=5, n_null_queries=4, n_cannibal_queries=2)
        b = generate_corpus(seed=5, n_null_queries=4, n_cannibal_queries=2)
        self.assertEqual([q.clicks for q in a.queries], [q.clicks for q in b.queries])
        c = generate_corpus(seed=6, n_null_queries=4, n_cannibal_queries=2)
        self.assertNotEqual([q.clicks for q in a.queries], [q.clicks for q in c.queries])

    def test_empty_corpus_is_handled(self):
        self.assertEqual(all_pairs(Corpus(days=0)), [])


if __name__ == "__main__":
    unittest.main()
