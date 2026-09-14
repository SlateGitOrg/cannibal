# cannibal

> Search cannibalisation detection with multiple-testing correction, so the list is twelve real pairs instead of 180 noisy ones.

> **Implementation note.** The runnable reference is pure standard-library
> Python: a hand-written chi-square tail (incomplete gamma), Benjamini-Hochberg
> and Bonferroni, a seeded synthetic Search-Console-shaped generator with
> planted cannibalising pairs, and `unittest`. It uses daily click splits in
> place of GSC impressions, and no statsmodels, pandas, DuckDB, Streamlit or
> live GSC ingestion. Consolidation recommendations and before/after tracking
> are not built. Those remain the target stack and scope.


`COMPACT` · **Marketing Analyst** · Intermediate · ~5 days · Education - course catalogue with 4,000 pages

**Primary language:** Python
**Tags:** `seo`, `hypothesis-testing`, `multiple-comparisons`, `duckdb`, `search-console`

---

## The problem

Four pages on the same site compete for the same search query. Each ranks worse than one consolidated page would, and the content team keeps writing more pages on the topic because the topic 'underperforms' - actively making the problem worse with every article.

## ⭐ The differentiator

Tests whether two pages genuinely compete using a **two-proportion test on query-level impression share with Benjamini-Hochberg correction across the thousands of query-page pairs examined**. Ranking page pairs by raw overlap percentage, across that many comparisons, surfaces noise at the top of the list every single time. A generic cannibalisation tool sorts by overlap and hands the content team a list of false positives to act on.

This is the sentence to lead with when someone asks you to walk through the
project. Everything else in this repo exists to make it true and to prove it.

## Data

**Google Search Console API** (free for any site owner), plus a synthetic generator with planted cannibalisation pairs *and* planted benign overlaps, so precision is measurable rather than asserted.

> No paid API key is required to run or demo this project. Where a paid
> service would add value it is wired as an optional enhancement behind an
> interface with an offline mock as the default implementation.

## Stack

- Python: statsmodels, pandas
- DuckDB
- Streamlit
- pytest

## Core capabilities

- GSC query-page performance ingestion with date-range windowing
- Competing-pair detection with multiple-testing correction and an effect-size floor
- Rank-instability analysis detecting pages swapping for the same query over time
- Consolidation recommendation with modelled traffic impact and a redirect plan
- Before/after tracking for consolidations actually actioned

## Repository layout

```
src/ingest/
src/detect/
src/recommend/
generator/
app/
test/
```

## Build plan

1. Generator with both planted cannibalisation and planted benign overlap. The benign cases are what prove the correction works.
2. Raw overlap ranking first, deliberately - see the 180 pairs, then fix it.
3. Add the test and BH correction. Report both lists side by side.
4. Recommendations and before/after tracking last.

## Testing strategy

Assert planted cannibalisation pairs are detected **and planted benign-overlap pairs are not**. Assert the false-discovery rate stays below the stated level across 500 simulated sites - the test that demonstrates the correction is doing real work rather than being cited.

Tests assert **correctness**, not merely that the code runs. A green suite on
this repo is a claim about behaviour under adversarial conditions; treat any
test that would pass against a deliberately broken implementation as a bug in
the test.

## Quality & safety layer

Uses the official Search Console API under the site owner's own credentials. No scraping of search results, which would breach terms of service.

## Measurable outcome

> A ranked list of twelve genuinely competing page sets - against the 180 pairs that raw-overlap ranking would have reported.

State it in these terms — business units, not technical ones — in your CV
bullet and in the first thirty seconds of describing the project.

## Measured results

From `python -m src.demo` (seeded, reproducible; about 15 s). The test for each
pair is a chi-square homogeneity test of page A's share of the pair's daily
clicks. With two days it is exactly the two-proportion z-test, and a unit test
checks that equivalence. Conditioning on the pair's daily total cancels shared
seasonality and trend.

**One catalogue-scale site** (1,112 queries, 60 days, 3,575 page pairs, 12 planted cannibalising pairs, alpha = 0.05):

| Procedure | Flagged | Real | False | Planted found |
|---|---|---|---|---|
| No correction | 173 | 12 | 161 | 12/12 |
| Benjamini-Hochberg | 14 | 12 | 2 | 12/12 |
| Bonferroni | 12 | 12 | 0 | 12/12 |
| Raw overlap ranking (top 12 by shared volume) | 12 | 1 | 11 | 1/12 |

**Repeated simulation** (80 sites, ~514 tests each, strong planted effect):

| Procedure | Mean FDR (se) | Mean power | False pairs per site |
|---|---|---|---|
| No correction | 0.650 (0.007) | 1.000 | 23.15 |
| BH | 0.042 (0.007) | 1.000 | 0.57 |
| Bonferroni | 0.001 (0.001) | 1.000 | 0.01 |

**Weak planted effect** (60 sites, displacement 0.03): BH power 0.285 against Bonferroni 0.218. BH FDR is 0.059 (se 0.015) and uncorrected FDR is 0.755.

**Confounded statistic, same BH step** (10 sites): raw click correlation flags 198 of ~200 pairs, with FDR 1.000. The share statistic flags 12.3, with FDR 0.023.

The measurable outcome holds in the measured form: twelve real pairs (plus two false ones under BH) against 173 uncorrected. The unit tests use their own seeds and 150 or 100 sites per regime.

### Limitations

- All numbers come from synthetic data. The planted effect is a day-to-day swing in share within a pair, and real GSC cannibalisation may not look like that.
- BH controls FDR at alpha·m0/m with near equality, so measured FDR sits close to 0.05 and can land slightly above it on a finite run. The tests assert that FDR is "not significantly above alpha" (mean - 3 se < alpha), not "below alpha".
- Ranking by overlap volume is not at chance here. Planted queries have two pages, so volume leaks a little signal: precision is about 0.12 in the tests against about 0.04 at chance. It is still far below BH's precision.
- The FDR tests use 150 simulated sites, not the 500 in the testing strategy, to keep the suite under a minute.
- Sampling uses `random.binomialvariate` (Python 3.12+). On 3.11 the exact fallback works but is much slower, and seeded numbers differ between versions.
- Not built: GSC ingestion, the effect-size floor, consolidation recommendations, before/after tracking, and the Streamlit/DuckDB layer.

## Interview questions this project answers

- **Why do you need multiple-testing correction here?**
- **What is the difference between two pages overlapping and two pages competing?**
- **How would you validate a consolidation worked?**

## What this deliberately is *not*

- Not a rank tracker, and not a scraper. Owner-authorised API data only.


## Run it now

```bash
python -m unittest discover -s tests -v   # the suite
python -m src.demo                        # the 60-second artefact
```

Run from the repo root. Requires Python 3.12+ for the fast exact binomial
sampler; 3.11 works but is slow. The runnable core uses **only the standard
library**, so there is nothing to install.

## Getting started

```bash
git clone https://github.com/SlateGitOrg/cannibal.git
cd cannibal
python -m unittest discover -s tests -v   # the suite
python -m src.demo                        # the 60-second artefact
```

Nothing to install - the runnable reference has no dependencies.

<details>
<summary>Target workflow for the full build (not yet implemented)</summary>

These commands describe the production stack this project grows into.
None of them work in this repository today.

```bash
# git clone <your-fork-url> cannibal
# cd cannibal
# pip install -e .
# python -m generator --pages 4000
# python -m src.detect run --correction bh
# streamlit run app/main.py
# pytest
```

</details>

Docker is supported but optional — every path above works on a plain
Windows/macOS/Linux laptop without a cloud account.

## Definition of done

- [ ] The differentiator above is implemented, and a test proves it
- [ ] The measurable outcome is produced by a command anyone can run
- [ ] `README` explains the one decision a generic version gets wrong
- [ ] CI runs the full suite on every push and is green on `main`
- [ ] A recruiter can see the headline artefact in under 60 seconds

## Licence

MIT — see [LICENSE](LICENSE).
