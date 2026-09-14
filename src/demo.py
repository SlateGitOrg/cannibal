"""60-second artefact: the noisy list, the short list, and why.

ASCII only -- the target console is Windows cp1252.
"""

from .detect import decide, test_all_pairs
from .evaluate import run_simulation
from .generator import generate_corpus

ALPHA = 0.05
DEMO_SEED = 7

# The showcase site is catalogue-scale: ~1100 queries gives ~3600 simultaneous
# pair tests, which is the regime the README's headline numbers describe.
DEMO_NULL_QUERIES = 1100

# The repeated-simulation sites are smaller (~510 tests) purely for runtime.
# FDR is a ratio, so the qualitative result does not depend on the scale.
SIM_RUNS = 80
SIM_NULL_QUERIES = 150

# The weak-effect regime exists to separate BH from Bonferroni. At the default
# planted effect size every procedure has power 1.0, so a power comparison there
# would be vacuous.
WEAK_DISPLACEMENT = 0.03
WEAK_RUNS = 60


def rule(char="-", width=78):
    print(char * width)


def header(text):
    print()
    rule("=")
    print(text)
    rule("=")


def single_site():
    corpus = generate_corpus(seed=DEMO_SEED, n_null_queries=DEMO_NULL_QUERIES)
    results = test_all_pairs(corpus, statistic="share")
    truth = corpus.cannibal_pairs

    header("ONE SITE: %d queries, %d days, %d page pairs tested"
           % (len(corpus.queries), corpus.days, len(results)))
    print("Planted cannibalising pairs (ground truth): %d" % len(truth))
    print()

    rows = []
    for method in ("uncorrected", "bh", "bonferroni"):
        rep = decide(results, truth, alpha=ALPHA, method=method)
        rows.append(rep)

    print("%-14s %8s %8s %8s %10s %8s" % ("procedure", "flagged", "real", "false", "false %", "found"))
    rule()
    for rep in rows:
        print("%-14s %8d %8d %8d %9.1f%% %7d/%d" % (
            rep.method, len(rep.flagged), rep.n_true_positive, rep.n_false_positive,
            100.0 * rep.fdp, rep.n_true_positive, rep.n_true_pairs))

    from .generator import all_pairs
    ranked = sorted(all_pairs(corpus),
                    key=lambda t: -(sum(t[0].clicks[t[1]]) + sum(t[0].clicks[t[2]])))
    top = ranked[:len(truth)]
    hit = sum(1 for q, a, b in top if frozenset((a, b)) in truth)
    print()
    print("Raw overlap ranking (sort pairs by shared click volume), top %d: %d real, %d false."
          % (len(truth), hit, len(truth) - hit))

    naive = [r for r in rows if r.method == "uncorrected"][0]
    bh = [r for r in rows if r.method == "bh"][0]

    print()
    print("Top of the UNCORRECTED list (what a generic tool hands the content team):")
    rule()
    print("%-16s %-26s %-26s %10s %6s" % ("query", "page A", "page B", "p", "real?"))
    for r in naive.flagged[:12]:
        real = "yes" if frozenset((r.page_a, r.page_b)) in truth else "NO"
        print("%-16s %-26s %-26s %10.2e %6s" % (r.query, r.page_a, r.page_b, r.pvalue, real))
    print("... %d flagged in total, %d of them false." % (len(naive.flagged), naive.n_false_positive))

    print()
    print("The BH list at alpha=%.2f (q-value controlled):" % ALPHA)
    rule()
    print("%-16s %-26s %-26s %10s %6s" % ("query", "page A", "page B", "p", "real?"))
    for r in bh.flagged:
        real = "yes" if frozenset((r.page_a, r.page_b)) in truth else "NO"
        print("%-16s %-26s %-26s %10.2e %6s" % (r.query, r.page_a, r.page_b, r.pvalue, real))
    print("... %d flagged in total, %d of them false." % (len(bh.flagged), bh.n_false_positive))


def confound_demo():
    header("WHY NOT JUST CORRELATE THE TWO CLICK SERIES?")
    print("Every page on the site rides the same seasonal demand curve, so raw")
    print("click series co-move whether or not the pages compete. Same corpora,")
    print("same BH correction, only the statistic changes:")
    print()
    summaries = run_simulation(runs=10, alpha=ALPHA, statistic="raw_correlation",
                               methods=("bh",), base_seed=5000)
    raw = summaries["bh"]
    summaries = run_simulation(runs=10, alpha=ALPHA, statistic="share",
                               methods=("bh",), base_seed=5000)
    share = summaries["bh"]
    print("%-34s %10s %10s %10s" % ("statistic (both with BH)", "flagged", "FDR", "power"))
    rule()
    print("%-34s %10.1f %10.3f %10.3f" % ("raw click correlation", raw.mean_flagged, raw.mean_fdr, raw.mean_power))
    print("%-34s %10.1f %10.3f %10.3f" % ("within-pair share homogeneity", share.mean_flagged, share.mean_fdr, share.mean_power))
    print()
    print("Correction cannot rescue a confounded statistic: the raw-correlation")
    print("p-values are tiny for almost every pair, so BH rejects almost every pair.")


def simulation_table(title, runs, **kwargs):
    header(title)
    summaries = run_simulation(runs=runs, alpha=ALPHA, **kwargs)
    print("%d simulated sites, alpha=%.2f, mean %.0f tests per site"
          % (runs, ALPHA, summaries["bh"].mean_tests))
    print()
    print("%-14s %12s %12s %12s %12s" % ("procedure", "mean FDR", "se", "mean power", "se"))
    rule()
    for m in ("uncorrected", "bh", "bonferroni"):
        s = summaries[m]
        print("%-14s %12.4f %12.4f %12.4f %12.4f"
              % (m, s.mean_fdr, s.se_fdr, s.mean_power, s.se_power))
    print()
    print("Mean false pairs reported per site: uncorrected %.2f, BH %.2f, Bonferroni %.2f"
          % (summaries["uncorrected"].mean_false_positives,
             summaries["bh"].mean_false_positives,
             summaries["bonferroni"].mean_false_positives))
    return summaries


def main():
    print("cannibal -- search cannibalisation detection with FDR control")
    single_site()
    confound_demo()
    main_sim = simulation_table(
        "FDR AND POWER OVER %d SIMULATED SITES (strong planted effect)" % SIM_RUNS,
        runs=SIM_RUNS, n_null_queries=SIM_NULL_QUERIES)
    weak_sim = simulation_table(
        "SAME, WEAK PLANTED EFFECT (displacement %.2f)" % WEAK_DISPLACEMENT,
        runs=WEAK_RUNS, displacement=WEAK_DISPLACEMENT,
        n_null_queries=SIM_NULL_QUERIES)

    header("HEADLINE")
    u, b = main_sim["uncorrected"], main_sim["bh"]
    print("Uncorrected: %.1f%% of the reported pairs are false (%.1f false pairs per site)."
          % (100.0 * u.mean_fdr, u.mean_false_positives))
    print("BH at alpha=%.2f: FDR %.4f (se %.4f), power %.4f -- a short list you can act on."
          % (ALPHA, b.mean_fdr, b.se_fdr, b.mean_power))
    print("Where the effect is weak, BH power %.3f beats Bonferroni power %.3f"
          % (weak_sim["bh"].mean_power, weak_sim["bonferroni"].mean_power))
    print("while both still control error -- BH is the right correction, not just a correction.")


if __name__ == "__main__":
    main()
