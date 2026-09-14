# BENCH v4 — Python ABM

Python translation of the BENCH NetLogo model (v4), replicating the renovation/insulation module from:

> Niamir, L., et al. (2024). *Energizing building renovation: Unraveling the dynamic interplay of building stock evolution, individual behaviour, and social norms.* Energy Research & Social Science, 110. https://doi.org/10.1016/j.erss.2024.103445

---

## What the model does

BENCH_v.4 (Behavioral change in ENergy Consumption of Households) is an agent-based model of residential building renovation. Each agent represents a household that may decide to insulate their home. The model is calibrated from survey data for two regional case studies:

- **NL** — Netherlands, Overijssel province (759 households)
- **ES** — Spain, Navarre region (793 households)

The simulation runs from **2016 to 2050**. Each annual tick, households go through a sequential decision pipeline and decide whether to renovate. Social learning spreads awareness and motivation between spatial neighbours. Income grows over time following SSP2 scenario trajectories from CGE (Computable General Equilibrium) data. From 2025 onwards, dwelling age distributions are updated from MESSAGEix-Buildings projections.

---

## Decision pipeline

Each household follows a four-stage behavioural chain grounded in the **Theory of Planned Behaviour** and **Norm Activation Theory**:

```
Knowledge  -->  Motivation  -->  Consideration  -->  Utility  -->  Action
```

**1. Knowledge**
Awareness is the mean of three knowledge attributes (`know`, `cee_aw`, `ed_aw`), all on a 1–7 Likert scale. If awareness exceeds a case-study threshold (NL: 4.6, ES: 5.2) the household becomes "guilty" and activates the motivation stage.

**2. Motivation**
Guilty households check whether personal norm (`pn`) and social norm (`sn`) exceed thresholds for each behaviour type (investment, conservation, switching). Each behaviour gets a motivation status: `H` (high) or `L` (low).

**3. Consideration**
Motivated households evaluate perceived behavioural control (`pbcI`) and dwelling ownership. Only owner-occupiers with sufficient PBC pass to the utility calculation.

**4. Utility (discrete choice)**
A linear utility function determines renovation preference:

```
U1 = edu*0.0563 + age*0.0008 + dw_elab*(-0.0770) + dw_type*0.4265
   + dw_age*0.0883 + dw_size*0.0857 + gas*0.0000488 + pn1*0.0528 + erI1
```

**5. Action**
If `U1 > 0` and the household has not renovated recently (cooldown period by dwelling age: new=15 yr, middle=7 yr, old=2 yr), renovation proceeds: the energy label improves by one step, gas consumption falls by 20%, and the household records an investment cost of €3,000.

### Social learning modes

After each tick (from 2017 onwards), social learning updates neighbours' attributes:

| Mode | Mechanism |
|---|---|
| **No learning** | No attribute updating |
| **Slow dynamics** | Active households boost neighbours only if they have >4 spatial neighbours |
| **Fast dynamics** | Active households boost neighbours unconditionally |
| **Informative** | All households get a 5% annual knowledge boost; active households also propagate to neighbours |

---

## Project structure

```
bench_v4/
  params.py        All empirical parameters and initialization distributions
  household.py     Household agent class (mirrors NetLogo turtle-own variables)
  model.py         BENCHv4 simulation engine (go-procedure logic)
  aggregate.py     5-year window aggregation, matching the paper's reported rates
  output.py        Per-run CSV / JSON / TXT saving
  plotting.py      Single-scenario and multi-scenario plots (mean +/- 95% CI)
  __init__.py

configs/
  bench_v4_scenarios.yaml   Multi-scenario YAML config (8 scenarios, NL + ES)

data/
  cge-nl-ssp2-h.csv         CGE income growth multipliers for NL (175 rows)
  cge-es-ssp2-h.csv         CGE income growth multipliers for ES (35 rows)

main.py            CLI entry point
```

---

## Module reference

### `bench_v4/params.py`

Stores every empirical constant and initialization distribution from the NetLogo source, with nothing embedded in the model logic. This includes utility-function coefficients, guilt and motivation thresholds (different for NL and ES), PBC thresholds for each behaviour type, social-learning rate and cap, memory-recall probabilities, cooldown periods by dwelling age, and the MESSAGEix-Buildings dwelling-age update tables used from 2025 onwards. All per-income-group distributions for behavioral attributes (income, gas use, knowledge, personal norm, social norm, PBC, etc.) are defined here as `NL_GROUPS` and `ES_GROUPS` lists.

### `bench_v4/household.py`

The `Household` class represents a single agent, mapping one-to-one to a NetLogo turtle. Attribute names follow the NetLogo originals with dots replaced by underscores (e.g. `dw.elab` → `dw_elab`). On construction it stochastically initialises all attributes by drawing from the per-income-group distributions in `params.py`. The module also provides `_rand_num()`, an exact Python translation of NetLogo's `randomNumber` reporter (stepped uniform distribution over a discrete grid). `__slots__` is used throughout to keep per-agent memory low when running large ensembles.

### `bench_v4/model.py`

The simulation engine. `BENCHv4.run()` calls `setup()` then iterates `go()` from 2016 to 2050. Each tick of `go()` follows the NetLogo procedure order exactly:

1. `_update_info` — reset `act1` to False
2. `_recall_memory` — assign pre-2016 renovation history in the first year
3. `_update_dwelling` — probabilistically shift dwelling age from 2025 using MESSAGEix projections
4. `_knowledge` — compute household awareness and guilt status
5. `_motivation` — set motivation status for investment, conservation, and switching
6. `_consideration` — filter by PBC and ownership constraints
7. `_utility` — calculate renovation utility score `U1`
8. `_action` — set `act1=True` for households with `U1 > 0` who have not recently renovated
9. `_save_energy` / `_invest` — record gas savings and renovation cost
10. `_learn` — social learning between spatial neighbours (mode-dependent)
11. `_update_income` — apply CGE growth multiplier to household income
12. `_update_energy` — improve energy label for renovating households
13. `_update_memory` — increment renovation cooldown counter and reset when expired

At the end of each tick, `_collect_stats()` assembles an `AnnualStats` dataclass record that is appended to `model.history`.

### `bench_v4/aggregate.py`

The single definition of the paper's reporting window. `multi_year_rate()` turns an annual
series of renovation counts plus cohort sizes into the "% of cohort renovating over a 5-year
period" figure that Niamir et al. (2024) plot in Figs. 5 to 7. `multi_year_rate_from_df()` and
`stack_runs()` apply it to `annual_results.csv` DataFrames, with a fallback (and a warning) for
CSVs written before the raw count columns existed. See "Reporting: annual vs 5-year rates".

### `bench_v4/output.py`

`save_run(model, run_dir)` writes three files for one completed run:

- `annual_results.csv` — one row per simulation year, all 31 metric columns
- `run_config.json` — the parameters used (`case_study`, `learning`, `seed`, `n_households`, `memory`, `start_year`, `end_year`)
- `summary.txt` — a human-readable table of year, renovation count, percentage, gas saved, and investment

### `bench_v4/plotting.py`

Two public functions for visualising ensemble results:

**`plot_all(config_dir)`** reads all `runs/*/annual_results.csv` files, aggregates them into mean ± 95% CI (1.96 × std / sqrt(n)), and saves 10 PNG plots to `config_dir/plots/`. Plots cover overall renovation rate, vintage breakdown (annual and cumulative), income-group breakdown, behaviour counts, investment, energy savings, motivation states, gas savings, and awareness.

**`plot_multi_scenario(parent_dir)`** auto-discovers all scenario subfolders by reading the `run_config.json` inside each, organises them into a grid (rows = learning mode, columns = case study), and saves two comparison plots to `parent_dir/multi_scenario_plots/`:
- a subplot grid showing renovation rate by vintage with CI bands
- a subplot grid showing grouped bar charts of renovation by income group at five-year snapshots (2020–2050)

---

## Installation

Requires Python 3.12+. Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Or with pip:

```bash
pip install matplotlib numpy pandas pyyaml
```

---

## Running the model

### Single run

```bash
uv run python main.py --case NL --learning Informative --seed 42 --verbose
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--case` | `NL` | Case study: `NL` or `ES` |
| `--learning` | `Informative` | `No learning`, `Slow dynamics`, `Fast dynamics`, `Informative` |
| `--seed` | random | Fixed random seed (single runs only) |
| `--runs` | `1` | Number of Monte Carlo seed runs |
| `--no-memory` | off | Disable pre-2016 renovation recall |
| `--output-dir` | `output/` | Root folder for all saved results |
| `--no-plot` | off | Skip plot generation |
| `--verbose` | off | Print year-by-year progress |

### Multi-run ensemble

```bash
uv run python main.py --case NL --learning Informative --runs 100
```

Results and plots are saved automatically. The mean annual renovation table is printed to stdout.

### Run from a YAML config file

```bash
uv run python main.py --config configs/bench_v4_scenarios.yaml
```

The YAML file defines a list of scenarios run in sequence. See [configs/bench_v4_scenarios.yaml](configs/bench_v4_scenarios.yaml) for the full format.

---

## Output structure

### Direct CLI run

A single timestamped folder is created for the scenario:

```
output/
  NL_Informative_20260101_120000/
    runs/
      run_001_seed_386129255/
        annual_results.csv      Per-year metrics (one row per year)
        run_config.json         Seed, case, learning, n_households
        summary.txt             Human-readable summary table
      run_002_seed_.../
        ...
    plots/
      renovation_rate_overall.png
      renovation_by_vintage.png
      renovation_by_vintage_5yr.png        5-year windows, as reported in the paper
      renovation_by_vintage_cumulative.png
      renovation_by_income.png
      renovation_by_income_5yr.png         5-year windows, as reported in the paper
      behaviour_count_by_type.png
      investment_by_type.png
      energy_saved_by_type.png
      motivation_over_time.png
      gas_savings_annual.png
      avg_awareness.png
```

### Config-file run

One parent folder is created for the entire config run. Each scenario gets a subfolder inside it. After all scenarios complete, multi-scenario comparison plots are saved alongside the scenario folders:

```
output/
  bench_v4_scenarios_20260101_120000/   <- one folder per config run
    NL_Informative/
      runs/
        run_001_seed_.../
          annual_results.csv
          run_config.json
          summary.txt
        ...
      plots/
        (12 plots, same as above)
    NL_Slow_dynamics/
      ...
    ES_Informative/
      ...
    multi_scenario_plots/
      multi_renovation_by_vintage.png          Vintage renovation grid (rows=learning, cols=case)
      multi_renovation_by_income_histogram.png Income group bars, 5-year windows
```

All single-scenario plots show the **mean line** across seed runs with a **shaded 95% confidence interval** band.

### annual_results.csv columns

| Column | Description |
|---|---|
| `year` | Calendar year (2016–2050) |
| `n_renovated` | Households renovating this year |
| `n_conservation` | Households conserving (0 in v4, forward-compatible) |
| `n_switching` | Households switching energy source (0 in v4, forward-compatible) |
| `pct_renovated` | % of total households renovating |
| `pct_conservation` | % of total households conserving |
| `pct_switching` | % of total households switching |
| `renov_pct_dwage1` | % renovating — new dwellings (<10 yr) |
| `renov_pct_dwage2` | % renovating — middle dwellings (11–35 yr) |
| `renov_pct_dwage3` | % renovating — old dwellings (>35 yr) |
| `renov_cum_pct_dwage1` | Cumulative renovation rate — new dwellings |
| `renov_cum_pct_dwage2` | Cumulative renovation rate — middle dwellings |
| `renov_cum_pct_dwage3` | Cumulative renovation rate — old dwellings |
| `renov_pct_grp1` | % renovating — income group 1 (lowest) |
| `renov_pct_grp2` | % renovating — income group 2 |
| `renov_pct_grp3` | % renovating — income group 3 |
| `renov_pct_grp4` | % renovating — income group 4 |
| `renov_pct_grp5` | % renovating — income group 5 (highest) |
| `n_renov_dwage1..3` | Renovation **count** per dwelling-vintage cohort |
| `n_total_dwage1..3` | Cohort **size** per dwelling-vintage cohort |
| `n_renov_grp1..5` | Renovation **count** per income group |
| `n_total_grp1..5` | Group **size** per income group |
| `total_gas_saved_kwh` | Annual gas savings from renovation (kWh) |
| `total_energy_conservation_kwh` | Annual energy savings from conservation (kWh) |
| `total_energy_switching_kwh` | Annual energy savings from fuel switching (kWh) |
| `total_investment_eur` | Annual renovation expenditure (EUR) |
| `total_invest_conservation_eur` | Annual conservation expenditure (EUR) |
| `total_invest_switching_eur` | Annual switching expenditure (EUR) |
| `avg_aware` | Mean household awareness (1–7 Likert) |
| `avg_pn1` | Mean personal norm for renovation (1–7 Likert) |
| `avg_sn1` | Mean social norm for renovation (1–7 Likert) |
| `high_guilt_pct` | % households with awareness above guilt threshold |
| `high_m1_pct` | % households with high motivation for renovation |
| `high_m2_pct` | % households with high motivation for conservation |
| `high_m3_pct` | % households with high motivation for switching |

---

## Reporting: annual vs 5-year rates

The paper (Niamir et al. 2024) reports renovation rates in Figs. 5, 6 and 7 as

> "the percentage of households renovation within specific age cohorts (<10, 11-35, >35),
> relative to the total number of households in each cohort, **observed over a 5-year period**"

so the value plotted at year Y is

```
100 * (renovations during [Y-4, Y]) / (households in that cohort)
```

not the single-year rate at Y. The distinction matters here: the renovation cooldown
(15 / 7 / 2 years by dwelling vintage) synchronises households into cohort waves, so the
**annual** series oscillates with a period of 2 to 7 years while the **5-year** aggregate is
smooth and comparable with the published figures.

`bench_v4/aggregate.py` holds the single definition. Both plotting and the programmatic API
use it:

```python
from bench_v4 import BENCHv4

m = BENCHv4(case_study="NL", learning="Informative", seed=1)
m.run()

end_years, rates = m.renovation_rate_5yr_by_vintage()
# end_years -> [2020, 2025, 2030, 2035, 2040, 2045, 2050]
# rates     -> {1: array([...]), 2: ..., 3: ...}   % of cohort renovating per 5 yr

end_years, rates = m.renovation_rate_5yr_by_income()
```

Both accept the `aggregate.multi_year_rate` keyword arguments:

| Argument | Default | Meaning |
|---|---|---|
| `end_years` | `[2020, 2025, ..., 2050]` | Window end years to report |
| `window` | `5` | Window length in years |
| `denominator` | `"mean"` | `"mean"` = mean cohort size over the window, `"last"` = cohort size in the end year. Cohort size is not constant, because `_update_dwelling` redraws `dw_age` every year from 2025 |
| `min_year` | `None` | Drop years before this from every window. `None` keeps every model year, which is what reproduces the paper |

### Why the paper's BENCH figures start at 2020

The run is 2016-2050 = 35 years = **exactly 7 windows of 5**, and Figs. 5 to 7 show exactly 7
points. The windows tile the run with no gap or overlap, each labelled by its **end** year:

| label | covers | label | covers |
|---|---|---|---|
| 2020 | 2016-2020 | 2040 | 2036-2040 |
| 2025 | 2021-2025 | 2045 | 2041-2045 |
| 2030 | 2026-2030 | 2050 | 2046-2050 |
| 2035 | 2031-2035 | | |

So 2016-2019 are not missing from the paper's charts, they are inside the first window. 2016
therefore belongs in the aggregation, and `min_year` defaults to `None`.

Setting `min_year=2017` to trim the 2016 initialisation tick is available but changes results
far more than "one year in five" suggests. On tick 1 every household already past the
behavioural gates renovates at once, then serves its cooldown (15 / 7 / 2 years by vintage).
For the two long-cooldown cohorts there are *no* further renovations before 2021, so 2016 is
100 % of their first window and trimming it sends them to zero. See `MODEL_AUDIT.md`
section 3.3 item 3.

Annual plots are still produced alongside the 5-year ones; use the annual series to diagnose
model dynamics and the 5-year series to compare against the paper.

CSVs written before the `n_renov_*` / `n_total_*` count columns existed still work: the
aggregation falls back to summing the annual percentage columns and emits a `RuntimeWarning`.
That fallback is exact only while cohort size is constant, so it drifts by up to ~1 percentage
point from 2025 onwards. Re-run the model to regenerate the CSVs for exact values.

---

## YAML scenario config

The config file is a YAML list. Each entry runs one scenario:

```yaml
- case_study: NL
  learning: Informative
  runs: 100
  memory: true
  run_label: NL_Informative
  description: Netherlands - informative social learning
```

| Field | Required | Description |
|---|---|---|
| `case_study` | yes | `NL` or `ES` |
| `learning` | yes | Learning mode (see table above) |
| `runs` | yes | Number of seed runs |
| `memory` | no | Pre-2016 renovation recall (default: `true`) |
| `run_label` | no | Output folder prefix (auto-generated if omitted) |
| `description` | no | Free-text note, not used by the model |

---

## Programmatic use

```python
from bench_v4 import BENCHv4

model = BENCHv4(case_study="NL", learning="Informative", seed=42)
history = model.run()

# Renovation rates by dwelling vintage (dict of lists)
rates = model.renovation_rate_by_vintage()
# {1: [pct_2016, pct_2017, ...], 2: [...], 3: [...]}

# Save and plot
from bench_v4.output import save_run
from bench_v4.plotting import plot_all

save_run(model, "my_output/run_001")
plot_all("my_output")
```

