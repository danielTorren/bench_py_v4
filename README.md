# BENCH v4: Python port

A Python port of the BENCH v4 agent-based model (NetLogo), as used in:

> Niamir, L., Mastrucci, A., & van Ruijven, B. (2024). *Energizing building renovation: Unraveling the dynamic interplay of building stock evolution, individual behaviour, and social norms.* Energy Research & Social Science, 110, 103445. https://doi.org/10.1016/j.erss.2024.103445

The port is literal. It reproduces the logic of the original NetLogo code (`BENCH_ v04_ B-NLD.ESP.nlogox`), including its known defects (see [Known issues](#known-issues-in-the-original-model)). It does not correct or extend the model.

## The model

BENCH (Behavioural change in ENergy Consumption of Households) is a spatially explicit agent-based model. Each agent is a household that decides each year whether to insulate and renovate its dwelling. Version 4 covers this one decision only. In the paper, BENCH is soft-linked to the building stock model MESSAGEix-Buildings: the stock composition by vintage comes from MESSAGEix-Buildings (STURM), and the renovation decisions from BENCH go back to it.

**Case studies.** Two EU provinces, calibrated from a household survey:

| Case | Region | Households |
|---|---|---|
| `ES` | Navarre, Spain | 793 |
| `NL` | Overijssel, The Netherlands | 759 |

**Agents.** Households differ in socio-economic and dwelling attributes (income group, education, age, tenure, energy label, dwelling type, age and size, gas consumption) and in behavioural factors (knowledge, awareness of consequences, information on energy investments, personal norms, subjective norms, perceived behavioural control). Initial values are drawn from survey-based distributions per income group. Income grows each year by the SSP2 growth factors of the EXIOMOD CGE model (`data/cge-*-ssp2-h.csv`).

**Space.** Households are placed at random on an 89 x 89 torus. A household's neighbours are the households on the 8 surrounding patches.

**Time.** One tick is one year, from 2016 to 2050 (35 ticks).

**Decision.** Following the Theory of Planned Behaviour and Norm Activation Theory, each household goes through these steps every year:

1. **Knowledge activation.** Awareness is the mean of three knowledge scores. Above a threshold (NL 4.6, ES 5.2) the household feels responsible ("guilt").
2. **Motivation.** A responsible household is motivated if its personal and subjective norms pass thresholds.
3. **Consideration.** A motivated household considers renovation if its perceived behavioural control passes a threshold (NL 1.0, ES 2.2) and it owns its dwelling.
4. **Utility.** A probit utility from the survey estimation:
   `U1 = 0.0563 edu + 0.0008 age - 0.0770 label + 0.4265 type + 0.0883 dw_age + 0.0857 size + 0.0000488 gas + 0.0528 pn + error`
5. **Action.** The household renovates if `U1 > 0`, its energy label can improve, and it has not renovated within its cooldown period (15, 7 or 2 years for dwellings < 10, 11-35 and > 35 years old). A renovation improves the energy label by one step and reduces gas use by 20 %.

At the first tick, a share of households (1.2 % to 3.6 % by income group) is set as already renovated before 2016. From 2025, dwelling ages are redrawn every year from the MESSAGEix-Buildings vintage shares.

**Social learning.** From 2017, households that renovate or are in their cooldown period influence their neighbours. A neighbour's knowledge, norms and perceived control increase by 5 % if they are below the neighbourhood level (the larger of mean and median), up to a ceiling of 6.6 on the 1-7 scale.

| Scenario | Paper name | Rule |
|---|---|---|
| `No learning` | | No social learning |
| `Slow dynamics` | SD (baseline) | Neighbours learn only if the active household has more than 4 neighbours |
| `Informative` | ID | Neighbours always learn, and every household gains 5 % knowledge per year (information policy) |
| `Fast dynamics` | not in the paper | Neighbours always learn, no information policy |

**Outputs.** The model records the number of renovations per year, by income group and by dwelling age cohort. The paper reports the renovation rate per cohort over 5-year periods: renovations in the period divided by the households in the cohort (`bench_v4/aggregate.py`).

## Usage

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python main.py --config configs/paper_reproduction.yaml
```

`configs/paper_reproduction.yaml` runs the six paper scenarios (ES and NL, each with No learning, SD and ID) with 100 runs each. Run `i` uses seed `seed + i`. Results go to `output/<config>_<timestamp>/<scenario>/runs/run_NNN_seed_S/` (`annual_results.csv`, `run_config.json`, `summary.txt`), with plots in each scenario folder.

From Python:

```python
from bench_v4 import BENCHv4

m = BENCHv4(case_study="ES", learning="Slow dynamics", seed=1)
m.run()
end_years, rates = m.renovation_rate_5yr_by_vintage()
```

## Validation against NetLogo

`validation/` runs the original NetLogo model headless (NetLogo 7.0.4) and compares it with the Python port.

```bash
uv run python validation/run_netlogo.py --runs 100     # 600 NetLogo runs
uv run python validation/compare_netlogo.py            # writes validation/comparison.md
```

The NetLogo model file is expected at `netlogo/BENCH_ v04_ B-NLD.ESP.nlogox`. The two models use different random number generators, so single runs do not match. The comparison is between ensembles with the same seeds: for each scenario, metric and year, the difference of the means is divided by its standard error. A faithful port gives about 5 % of values above 1.96 by chance.

One change is made to the NetLogo copy that is run: the `debug` procedure, which writes every agent to `debug.csv` on every tick, is replaced by `ask turtles [ ]`. This keeps the random number stream unchanged.

**Year labels in BehaviorSpace output.** NetLogo's `go` increments `year` at the end of the tick, so in a BehaviorSpace table the row labelled `year = Y` holds the results of model year `Y - 1`. The first row (2016) is zero, and model year 2050 is never recorded. The Python output labels each year by the model year.

`tests/` checks specific NetLogo behaviours (`uv run pytest`).

## Project structure

```
bench_v4/
  params.py       Parameters and initial distributions from the NetLogo setup
  model.py        BENCHv4 model (NetLogo go procedure)
  aggregate.py    5-year renovation rates as in the paper
  output.py       Per-run CSV / JSON / TXT output
  plotting.py     Scenario plots (mean and 95 % CI over runs)
configs/          Scenario files
data/             EXIOMOD income growth factors (ES, NL)
validation/       Headless NetLogo runner and ensemble comparison
tests/            NetLogo parity tests
main.py           Command-line entry point
```

## Known issues in the original model

The port keeps these behaviours of the NetLogo code on purpose:

- **Vintage shares in reverse order.** From 2025, the largest MESSAGEix-Buildings share is assigned to dwellings < 10 years old, not > 35 years old.
- **Dwelling age is redrawn every year.** From 2025, each household's dwelling age is drawn again each year, independent of the previous value. The cooldown uses the current dwelling age, so a household can move to a shorter cooldown during its cooldown period.
- **Slow dynamics is close to No learning.** The rule needs more than 4 neighbours, but households have 0.8 neighbours on average, so very few households can pass on learning.
- **Synchronised start.** Every eligible household renovates in 2016 and then waits for the same cooldown, which gives waves in the annual results.
- **The utility never blocks a decision.** `U1 > 0` for all households that reach this step, so the utility coefficients have no effect on the results.
- **Pre-2016 recall, ES income group 3.** The code gives 2.9 %, but a second condition overrides it to 1.7 %.
- **NL income file.** The file has 175 rows (35 values, each repeated 5 times), but only rows 1-35 are read. Income does not affect any decision.
