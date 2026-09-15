# BENCH v4: Python vs NetLogo vs Niamir et al. (2024)

Audit date: 2026-09-14. Sources compared:

* `netlogo/BENCH_ v04_ B-NLD.ESP.nlogox` (header says "Version: B-NLD.ESP 03", 30 Jan 2023, re-saved by NetLogo 7.0.4)
* `bench_v4/*.py`
* Niamir, Mastrucci, van Ruijven (2024), *Energy Research & Social Science* 110:103445

---

## 0. Headline

1. **The Python is a faithful port.** `params.py` is an exact transcription of the NetLogo
   `setup` block (verified by script: all 14 groups, every range and break point). The tick
   logic matches. The divergences found are small (1 to 4 % on cumulative renovations). The
   one that matters is qualitative rather than large: `Slow dynamics` reaches about one agent
   per run with peer learning, making it effectively `No learning`.

2. **The jaggedness is real model behaviour, not a port bug.** It comes from the renovation
   cooldown (15 / 7 / 2 years by dwelling vintage) synchronising households into cohort waves.
   Freezing `update.dwelling` makes it *worse*, not better (period-7 spikes at 2030, 2037, 2044).

3. **Most of the apparent mismatch with the paper is a reporting mismatch.** The paper's
   Figs. 5 to 7 plot a renovation rate "observed over a 5-year period". `plotting.py` plots
   the single-year value at 2020, 2025, .... Recomputed the paper's way, the Python reproduces
   the published numbers closely:

   | Case / scenario | cohort | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 |
   |---|---|---|---|---|---|---|---|---|
   | NL SD paper  | 11-35 | 6.8 | 6.8 | 2.0 | 1.5 | 2.0 | 2.5 | 1.2 |
   | NL SD python | 11-35 | 6.9 | 7.3 | 3.1 | 2.2 | 1.7 | 1.0 | 0.2 |
   | NL SD paper  | <10   | 6.0 | 1.0 | 2.0 | 1.5 | 1.5 | 1.5 | 0.5 |
   | NL SD python | <10   | 6.4 | 0.0 | 3.1 | 2.2 | 1.7 | 0.7 | 0.2 |
   | ES SD paper  | all   | ~1  | ~0  | ~0  | ~0  | ~0  | ~0  | ~0  |
   | ES SD python | >35   | 2.7 | 0.6 | 0.7 | 0.1 | 0.2 | 0.1 | 0.1 |
   | ES ID paper  | <10   | 2.5 | 1.0 | 2.0 | 0.5 | 1.5 | 0.5 | 0.5 |
   | ES ID python | <10   | 2.2 | 1.5 | 2.8 | 2.0 | 1.3 | 0.9 | 0.3 |
   | NL ID paper  | >35   | 15  | 27  | 4.5 | 4.0 | 4.5 | 3.0 | 1.0 |
   | NL ID python | >35   | 21.6| 21.5| 9.2 | 9.0 | 6.4 | 3.9 | 2.6 |

   (Python = mean of 30 seeds; renovations inside the 5-year window / mean cohort size.
   Paper values read off Fig. 5.) The systematic residual is that Python decays more slowly
   than the paper in the late years.

4. **Two structural problems that neither the code nor the paper resolves**: the probit
   utility never binds, and there is no affordability constraint anywhere. See section 2.3.

---

## 1. Python vs NetLogo

### 1.1 Verified equivalent

* `params.py` against the NetLogo `setup` block: all income-group cumulative shares, income /
  gas / know / cee.aw / ed.aw / pn / sn / pbcI / pbcC / pbcS / ene.pat ranges, all `erI` error
  terms, all age / ecom / edu / dw.age / dw.size / dw.elab break ladders, and the owner,
  dwelling-type, gender and provider thresholds. **No differences.**
* Tick order, guilt / motivation / consideration stickiness, the U1 formula, cooldown table,
  re-entry conditions (`dw_elab > 1`), `dw_elab` decrement, recall probabilities,
  `update.dwelling` vintage tables, grid half-width 44.
* The NetLogo `action` procedure's two sequential `if`s are equivalent to the Python
  vectorised expression **except** for the `insulated` case (see 1.2 #2).

### 1.2 Divergences, ranked by impact

#### #1. `Slow dynamics` is effectively `No learning` in Python. (Qualitative.)

NetLogo builds the social network with `create-links-to other turtles-on neighbors` and
**never clears the links**, so `link-neighbors` is a cumulative directed network that grows
every year. The gate `if count link-neighbors > 4` therefore opens progressively.

Python re-derives neighbours from the static patch grid each tick, and that grid is nearly
empty: 759 agents on an 89x89 world is 0.096 agents per patch, so an agent's 8-patch
neighbourhood holds **0.77 agents on average**. Measured over 200 seeds, the number of agents
that clear `len(nbrs) > SLOW_NEIGHBOR_MIN` is:

```
NL  0.89 agents per run (of 759)   at least one in  96/200 runs   max seen 8
ES  1.04 agents per run (of 793)   at least one in 101/200 runs   max seen 7
```

So peer learning reaches roughly **one agent in 776 per run**. Net effect on cumulative
renovations, 200 seeds:

```
NL  No learning 197.7   Slow dynamics 198.1   +0.24 %   13/200 seeds differ
ES  No learning  25.7   Slow dynamics  25.8   +0.04 %    1/200 seeds differ
```

The paper's SD scenario, its behavioural baseline, is therefore *effectively* `No learning` in
the Python: not bit-identical, but within a quarter of a percent.

(Correction, 2026-09-15: an earlier draft of this section said the gate "is never true" and
that the two modes were "identical to the digit". That was inferred from a single seed whose
maximum neighbour count happened to be exactly 4. The gate does open, for about one agent per
run. The conclusion is unchanged; the absolute claim was wrong.)

Fix: keep a persistent cumulative link set (`self._links: list[set[int]]`), add
`turtles-on neighbors` to it each tick for acting agents, and gate on its size. Keep the
current behaviour behind a flag so both can be reproduced.

#### #2. `insulated` is a permanent lock in Python, inert in NetLogo. (Small: +1.7 to +3.8 %.)

NetLogo `action`:

```
if U1 <= 0 or invest1 = True or h.sta = "insulated" [set act1 False]
if (U1 > 0 and invest1 = False and dw.elab > 1)     [set act1 True ...]
```

The second `if` runs after the first and does **not** test `h.sta`, so it overrides it. The
`"insulated"` test is dead code. Python's `_action` has `& ~self._insulated`, which locks the
recall cohort (1.2 to 3.0 % of agents) out for the whole run.

Fix: drop `~self._insulated` from `_action`; keep the array for reporting.

#### #3. Learning growth caps. (Negligible: <0.3 %.)

NetLogo gates on `x < 6.6` then does `x + x*0.05` with no cap, so values reach 6.93.
Python clamps: `min(x*1.05, 6.6)` in the neighbour loop, and `min(x*1.05, 6.65)` in the
Informative broadcast (`lc + lr`, which is not a NetLogo quantity at all).

Fix: drop both clamps; keep only the `< LEARNING_CAP` entry gate.

#### #4. World wrapping.

The NetLogo view has `wrappingAllowedX/Y="true"` (a torus). `_build_neighbor_index` discards
out-of-bounds neighbours instead of wrapping, so perimeter agents (~4.5 % at N=793) have
artificially few neighbours. Fix: `nx % W`, `ny % W`.

#### #5. Run length. (Python is right; the saved experiment config is wrong.)

The NetLogo **model code** runs 35 ticks, 2016 to 2050. `setup` sets `year 2016`; `go`
processes the year then increments it; `if year >= 2051 [stop]` fires at the end of tick 35,
having processed 2050. Python matches this exactly.

Three things agree on 35 ticks:

* the `if year >= 2051 [stop]` guard;
* `update.income` reads `item n`, with `n = 0..34` over 35 ticks, consuming exactly the 35 rows
  of `cge-es-ssp2-h.csv`. The author's own comment says "35 years of data (0 to 34)";
* the paper's Figs. 5 to 7 show exactly 7 five-year windows, which needs 35 years.

Only the BehaviorSpace experiments disagree: `timeLimit="34"` truncates every run to 2016-2049,
which is 34 years and does not divide into 5. So the saved experiment config contradicts the
model's own stop condition, its own data file length, and the paper's figures. This is further
evidence that the supplied `.nlogox` is not the configuration behind the published runs
(see section 3.1).

#### #5b. The NL income file is mis-indexed. (No effect today; a trap for later.)

`cge-nl-ssp2-h.csv` holds 175 rows: 35 distinct annual growth factors, each repeated 5 times.
`cge-es-ssp2-h.csv` holds 35 rows with no repetition. Both are read the same way,
`item 0 item n` with `n = 0..34`, so the NL run consumes rows 0-34 and sees only the **first 7**
of its 35 distinct values, stretched across the whole run. Whatever the repetition encodes
(5 income quintiles per year, or an annual series at 5x resolution), the indexing does not
account for it.

Harmless right now, because `income` is never read by any decision rule (see #8 and
section 2.3). It becomes a real bug the moment an affordability constraint is added, which is
exactly what section 2.3 item 5 recommends. Fix the indexing at the same time.

#### #6. Dead-code differences.

No effect while only insulation is active, but they will bite if conservation or switching are
ever enabled:

* `_consideration` ANDs `ene_pat != 3` into `cC`; NetLogo's second `if` overrides that test.
* Python applies the owner test to all three `cI` slots; NetLogo applies it only to `cI1`/`cI2`.
* Python has no `ene_prov` array, so the NetLogo `cS1`/`cS2`/`cS3` provider gating is absent.
* `h.sta = "Efficient"` is never set in Python (`n.hh.eff` has no Python counterpart).

#### #7. ES recall, group 3.

NetLogo has overlapping conditions (`if (aa <= 2.9)[...]` followed by
`if (aa > 1.7) and (aa <= 100)[set invest1 False]`), so the effective `invest1` recall
probability is 1.7 %, not the 2.9 % in `RECALL_PROB`. Combined with #2 this is the largest
single parameter discrepancy in the recall step.

#### #8. `_update_income` position.

Moved ahead of `_save_energy`. Harmless, because **`self._income` is never read by anything**.
See section 2.3.

### 1.3 Deliberate Python-only choices (document, do not "fix")

* `_place_on_grid` scales the world by `sqrt(N / N_default)` to hold density at ~0.1
  agents/patch. NetLogo has a fixed 89x89 world. So a 10 000-agent Python run is **not** what
  NetLogo would produce at 10 000 agents (NetLogo would be about 12x denser, with far more
  learning).
* `_create_arrays_from_df` (the `copula` and `survey755` population paths) is a new
  specification, not a port: it sets `erI = -0.02` constant, `ene_pat = 1`, and broadcasts a
  single survey `pn` / `sn` / `pbc` score to all three action slots. This removes the
  idiosyncratic probit error term, the only source of within-cell heterogeneity in U1.
  All configs in `configs/` use `population: copula`, so **none of the current runs use the
  NetLogo initialisation.** Run at least one `population: groupparams` scenario for comparison.

---

## 2. Models vs paper

### 2.1 What the paper and the code agree on

* v04 is insulation-only ("BENCH (version 04) focuses on a household decision on building
  renovation"). The `act2`..`act9` machinery in the .nlogox is declared but never computed, and
  `utility` produces only `U1`. That is consistent with the paper, not a sign of damage.
* SD / ID map to `Learning = "Slow dynamics"` / `"Informative"`. `"Fast dynamics"` is an extra
  scenario not in the paper. `"Informative-Soft"` and `"Promoting"` are empty stubs.
* The BehaviorSpace metrics (`a1`, `number.groupN`, `groupN.a1`, `number.dwageN`, `dwageN.a1`)
  are exactly the ingredients for Figs. 5 and 7, and they are **annual**. So the 5-year
  aggregation in the paper was done in post-processing.
* The falling-over-time shape and the SD below ID ordering match Figs. 5 and 8.

### 2.2 Reporting mismatches (fix these first, they are cheap)

| Paper | Code |
|---|---|
| Rate "observed over a 5-year period" | `_plot_ms_income_histogram` and `_SNAPSHOT_YEARS` take single-year values |
| "means across 100 random runs" | BehaviorSpace experiments are `repetitions="1"`; configs use `runs: 64` |
| "over 33 years (2017-2050)" | Model code and Python both run 2016-2050 (35 ticks), which is what the 7 five-year windows need. Only the saved BehaviorSpace `timeLimit="34"` disagrees (see 1.2 #5) |
| Figs. 5-7 start at 2020 | **Resolved.** 2016-2050 = 35 years = exactly 7 windows of 5, and the paper shows exactly 7 points. The windows tile the run and are labelled by their end year, so 2016-2019 sit inside the first bin rather than being omitted. `REPORT_MIN_YEAR = None` |

### 2.3 Under-specified or missing in the paper

1. **SD definition.** Table 1 says "individuals interact with four available neighbours".
   The code says `count link-neighbors > 4` (strictly more than four), and on a *cumulative*
   link network rather than the patch neighbourhood. Neither the threshold semantics nor the
   network definition is stated. This one sentence decides whether SD is a social scenario or
   a no-learning scenario.

2. **`update.dwelling` is not described, and contradicts the text.** The paper says
   "households' dwelling arrangements remain static, with the model only tracking the age of
   the buildings". The code **redraws every agent's `dw.age` independently every year** from a
   hardcoded 5-year MESSAGEix vintage-share table. A household's building can go from >35y to
   <10y and back in three consecutive years. This is the biggest specification gap, and it
   interacts directly with the jaggedness: the cooldown length is looked up from the *current*
   (randomly redrawn) `dw.age`, so a household part-way through a 15-year cooldown can be
   dumped into the 2-year bucket and become eligible again immediately.

3. **The renovation cooldown (15 / 7 / 2 years by vintage) appears nowhere in the paper.**
   No source, no calibration, no sensitivity analysis. It is the dominant driver of the time
   profile.

4. **Energy saving per renovation.** Paper: 20 % of renovations deep (60 % saving), the rest
   averaging 30 %. Code: a flat 20 % of *gas only*, electricity untouched. It is unclear
   whether the 60/30 split is applied only downstream in MESSAGEix / STURM.

5. **Affordability.** The Fig. 7 discussion rests on "renovation affordability: the majority of
   household ... cannot afford the costs". **There is no budget constraint in either model.**
   `I1.cost = 3000 EUR` is accumulated for accounting only; income is updated from the EXIOMOD
   CGE every year and then **never read by any decision rule**. The paper's affordability
   conclusion is not produced by the model as supplied.

6. **The probit utility never binds.** Evaluated over the whole NL population, U1 ranges from
   0.368 to 1.875: **`U1 > 0` for 100 % of agents**, and `(U1 > 0)` equals the consideration
   mask exactly at every tick. Therefore education, age, dwelling type, dwelling size and gas
   consumption have *no* influence on outcomes. Everything is decided by the guilt / motivation
   / PBC thresholds and the owner gate. Two plausible explanations, both worth chasing: the
   published probit had a (negative) constant term that was dropped, or the action step was
   probabilistic (adopt with probability `Phi(U1)`) rather than `U1 > 0`. Either would restore
   the coefficients' role, and would also smooth the output considerably.

7. **Threshold provenance.** Guilt 4.6 / 5.2, the six PN/SN thresholds, PBC 1.0 (NL) / 2.2 (ES)
   / 3.5 (ES switching). Cited to ref. [33] but not tabulated. PBC = 1.0 is the *bottom* of the
   1-7 scale, so the NL investment PBC gate is vacuous.

8. **Learning parameters.** The 5 %/yr multiplicative growth, the 6.6 cap, and what "intense
   information policy" means quantitatively. Not given.

9. **Scale.** BENCH is provincial (Navarre 793, Overijssel 759). MESSAGEix-Buildings is
   national. The upscaling from province to country is not described.

10. **Income groups.** The model has 7 income *brackets* collapsed to 5 for reporting
    (ES g1 = 11.4 %, g2 = 46.8 %). The paper labels them 5 *quintiles*. The mapping is not stated.

11. **Missing data in this repo.** Only the CGE income-growth column ships
    (`cge-{es,nl}-ssp2-h.csv`). Absent: PRIMES price series, `cge-*-con`, the MESSAGEix/STURM
    vintage time series behind the hardcoded `update.dwelling` table, and the NL survey file.

---

## 3. Next steps

### 3.1 Evidence that this .nlogox is not the paper's run version

* Code header: **"Version: B-NLD.ESP 03", 30 Jan 2023.** The filename says v04. The paper says
  v04 and was received 3 July 2023, revised 23 January 2024.
* Every BehaviorSpace experiment is `repetitions="1"` with `Generate-seed? = true`.
  The paper reports **means across 100 random runs**, and the seeds are not logged, so these
  experiments as configured are neither reproducible nor what produced Figs. 5 to 7.
* Experiments come in identical duplicate pairs (`...-SD` and `...-SD-S`) with byte-identical
  parameters and metrics. Something was being varied outside BehaviorSpace.
* The file was last written by NetLogo 7.0.4 (a 2025 release), so it has been opened and
  re-saved well after publication, and `.nlogox` is itself a format conversion from `.nlogo`.
* Every experiment sets `timeLimit="34"` (2016-2049). The model's own stop condition, its
  35-row ES income file, and the paper's 7 five-year windows all require 35 ticks (2016-2050).
  The saved experiment config is inconsistent with all three.
* Large amounts of half-wired machinery (a2..a9, conservation, switching, electricity, prices,
  providers, `invs.a2..a9`) are declared, initialised and plotted but never computed.

### 3.2 What to ask your boss, in priority order

1. **The post-processing script** (R or Python) that turned BehaviorSpace CSV into Figs. 5 to 8.
   This is the highest-value artefact: it settles the 5-year aggregation, the denominator,
   whether 2016 is dropped, and how 100 runs were averaged.
2. The exact `.nlogo` / `.nlogox` plus data directory used for the published runs, and the seed list.
3. Source and justification for the 15 / 7 / 2-year renovation cooldown.
4. Is the action step `U1 > 0`, or probabilistic? Did the probit have a constant term?
   (As supplied, `U1 > 0` always, so the probit does nothing.)
5. Was an affordability or budget constraint present in the paper's version? If not, how is the
   Fig. 7 affordability narrative supported?
6. Is `update.dwelling` meant to redraw `dw.age` per agent per year, or to preserve building
   identity and reassign only the demolished / newly built share?
7. The MESSAGEix / STURM vintage-share time series behind the hardcoded table.

### 3.3 Code work, in order

1. ~~**Reporting (cheap, may resolve the complaint on its own).** Add a 5-year cumulative
   renovation rate to `plotting.py` alongside the annual series, and a drop-first-tick option.~~
   **Done (2026-09-14).** `bench_v4/aggregate.py` holds the definition;
   `BENCHv4.renovation_rate_5yr_by_vintage()` / `_by_income()` expose it programmatically;
   `renovation_by_vintage_5yr.png` and `renovation_by_income_5yr.png` are produced per scenario;
   `multi_renovation_by_income_histogram.png` now uses windows instead of single-year snapshots.
   Raw cohort counts are written to `annual_results.csv` so the denominator is correct.
   Windows tile 2016-2050 exactly (`aggregate.REPORT_MIN_YEAR = None`); see section 2.2.
   Still outstanding: report means over at least 100 seeds (configs currently use `runs: 64`).

2. **Decouple the cooldown from the current-year `dw_age`.** Record the vintage at the moment
   of renovation and use *that* cooldown. This removes the largest artefact: households being
   re-bucketed mid-cooldown.

3. **The 2016 tick exposes a probable initialisation bug.** The reporting question is settled
   (2016 stays in, see section 2.2), but working it out surfaced a real model problem.

   `_recall_memory` gives every household `act1_year = 0`, so on tick 1 the entire eligible
   pool renovates at once and then serves its cooldown together. NL Slow Dynamics, annual
   counts, mean of 30 seeds:

   | cohort | cooldown | 2016 | 2017 | 2018 | 2019 | 2020 | ... | 2023 |
   |---|---|---|---|---|---|---|---|---|
   | <10 yr | 15 yr | 5.9 | 0 | 0 | 0 | 0 | | 0 |
   | 11-35 yr | 7 yr | 22.7 | 0 | 0 | 0 | 0 | | **22.7** |
   | >35 yr | 2 yr | 20.5 | 0 | 20.5 | 0 | 16.3 | | 0 |

   The 11-35 yr cohort fires 22.7 in 2016 and exactly 22.7 again seven years later: a cohort
   marching in perfect lockstep. That is the jaggedness in its purest form, and it means the
   two long-cooldown cohorts have *no* renovations at all between 2017 and 2020.

   The model therefore has two regimes: rigidly synchronised to 2024, then desynchronised once
   `_update_dwelling` starts reshuffling `dw_age` annually in 2025.

   The paper's Fig. 5 has no such hole in any cohort, which suggests its version seeded
   **partially elapsed cooldowns** at setup, so adoption is spread from the start rather than
   released in one burst. The NetLogo `recallmemory` has the same gap (it sets `invest1` but
   never `act1.year`), which is further evidence the supplied file is not the paper's version.
   Seeding `act1.year` uniformly over each household's cooldown length is roughly a one-line
   change and is the most likely single fix for the remaining shape mismatch in Fig. 5.
4. **Make `update_dwelling` preserve building identity.** Age buildings deterministically and
   reassign only the demolished / new share implied by the MESSAGEix table, applied on 5-year
   steps rather than annually. Keep the current behaviour behind a flag for NetLogo parity.
5. **Fix `Slow dynamics`** (cumulative link network, section 1.2 #1). Until this is done, do not
   report SD as a social-dynamics scenario.
6. Small parity fixes: the `insulated` lock, the learning caps, torus wrapping, run length.
7. **Build a regression harness.** Run the `.nlogox` headless (`netlogo-headless` with a fixed
   `Seed-for-random` and `Generate-seed? = false`), dump per-agent state per tick via
   `debugfiles`, and diff against the Python. Nothing currently verifies the port beyond reading.
8. Once items 1 to 4 are in, re-run the SEM analysis. With `U1 > 0` always true, any SEM path through
   the probit covariates (education, dwelling type, size, gas) is structurally zero in the
   current model, which will make those paths look spurious.
