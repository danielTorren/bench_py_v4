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
  output.py        Per-run CSV / JSON / TXT saving
  plotting.py      Batch plots with mean +/- 95% CI across seed runs
  __init__.py

configs/
  bench_v4_scenarios.yaml   Example multi-scenario YAML config

netlogo/
  BENCH_ v04_ B-NLD.ESP.nlogox   Original NetLogo source (read-only reference)
  cge-nl-ssp2-h.csv              CGE income growth data for NL (175 rows)
  cge-es-ssp2-h.csv              CGE income growth data for ES (35 rows)

main.py            CLI entry point
```

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

Each run creates a timestamped folder:

```
output/
  NL_Informative_20260101_120000/
    runs/
      run_001_seed_386129255/
        annual_results.csv      Per-year metrics
        run_config.json         Seed, case, learning, n_households
        summary.txt             Human-readable table
      run_002_seed_.../
        ...
    plots/
      renovation_rate_overall.png
      renovation_by_vintage.png
      renovation_by_vintage_cumulative.png
      renovation_by_income.png
      behaviour_count_by_type.png
      investment_by_type.png
      energy_saved_by_type.png
      motivation_over_time.png
      gas_savings_annual.png
      avg_awareness.png
```

All plots show the **mean line** across runs with a **shaded 95% confidence interval** band.

### annual_results.csv columns

| Column | Description |
|---|---|
| `year` | Calendar year (2016–2050) |
| `n_renovated` | Households renovating this year |
| `n_conservation` | Households conserving (0 in v4, forward-compatible) |
| `n_switching` | Households switching energy source (0 in v4) |
| `pct_renovated` | % of total households renovating |
| `renov_pct_dwage1/2/3` | % renovating by dwelling vintage (new/middle/old) |
| `renov_cum_pct_dwage1/2/3` | Cumulative renovation rate by vintage |
| `renov_pct_grp1..5` | % renovating by income group |
| `total_gas_saved_kwh` | Annual gas savings from renovation (kWh) |
| `total_investment_eur` | Annual renovation expenditure (EUR) |
| `avg_aware` | Mean household awareness (0–7) |
| `high_guilt_pct` | % households with awareness above threshold |
| `high_m1_pct` | % households motivated for renovation |
| `high_m2_pct` | % households motivated for conservation |
| `high_m3_pct` | % households motivated for switching |

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

---

## Key parameters

| Parameter | NL | ES | Source |
|---|---|---|---|
| Awareness threshold (guilt) | 4.6 | 5.2 | Survey calibration |
| Motivation threshold pn1 | 4.7 | 5.67 | Survey calibration |
| Motivation threshold sn1 | 3.5 | 4.77 | Survey calibration |
| PBC investment threshold | 1.0 | 2.2 | Survey calibration |
| Renovation cost | €3,000 | €3,000 | Literature |
| Gas saving fraction | 20% | 20% | Engineering estimate |
| Cooldown: new dwelling | 15 yr | 15 yr | NetLogo v4 |
| Cooldown: middle dwelling | 7 yr | 7 yr | NetLogo v4 |
| Cooldown: old dwelling | 2 yr | 2 yr | NetLogo v4 |
| Learning rate | 5%/yr | 5%/yr | NetLogo v4 |
