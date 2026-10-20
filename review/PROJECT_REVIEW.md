# SustainaBuild review — 18 October 2026

This is an AI-assisted review/development artifact, not a student-authored report or evidence of coordinator approval. The supplied guidelines prohibit generative AI in submitted work. The original executed notebooks and the Google Drive copy have been preserved.

## Verdict

The project implements a meaningful sequential control problem and substantially delivers the proposed comparison of DQN, REINFORCE, A2C and conventional control. It is not a verified real-building energy-saving system, and it is not yet defensible to describe it as perfect, fully compliant or demonstrably preferable to predictive control.

The most important academic risk is the lecturer's earlier objection about RL suitability: the two-step lookahead has the lowest mean objective in every saved experimental scenario. The problem has three continuous physical storage states (air temperature, thermal mass and battery energy), known schedules and relatively simple model equations. Fourteen observation features do not by themselves establish that a conventional optimiser cannot solve it effectively. Online inference speed is not a strong practical argument here: both search and RL run in under a millisecond in the saved tests, while decisions are hourly.

This does not make the comparison worthless or require inventing a harder problem. It requires an honest research question, stronger conventional comparisons, careful treatment of uncertainty, and coordinator acceptance of the resulting scope. No approval of this particular proposal was found in the supplied files.

## What matches the proposal and guidelines

| Requirement | Evidence | Assessment |
|---|---|---|
| Three distinct RL families | DQN cell 45, REINFORCE cell 47, A2C cell 49 | Present. Double DQN is only a DQN variant, not a fourth family. |
| Own algorithm implementations | Replay buffer/target network; reward-to-go policy gradient; actor and value critic | Present in the notebook, with pseudocode tags and equations. Authorship itself cannot be verified by inspecting code. |
| Sequential, coupled decisions | Cooling affects future temperatures; battery use changes future storage; joint 15-action policy | Present and meaningful. |
| Physically motivated transitions | Two-node RC model, substeps, charge/discharge efficiency and clipping, grid balance | Present; parameters are assumed, not calibrated to a measured building. |
| Chronological split | 423 training days, 92 validation days, 121 test days; warm months only | Present. Validation data are not independent final test data. |
| Tuning and repeated training | Learning-rate sweeps, second-factor sweeps, three final seeds per algorithm | Present; full run uses nominal 300,000 transitions per final run. Batched on-policy rollouts slightly overshoot. |
| E1 / E2 / E3 | Standard, +2h office schedule, +4°C temperature, 50% solar | Present, with all metrics exposed. |
| Final conclusions | Cell 80 is still a writing prompt | Incomplete. Results need the group's own interpretation. |
| Report and presentation | Not present in the inspected project files | Cannot assess. Report Part 2 requires three improvements, three distinct future directions and six academic sources for that part. Hyperparameter tuning is not an eligible improvement. |
| Single documented notebook | Main notebook includes training, simulator, tests and results | Main work fits; the separate dashboard is supplementary. Include every presented numerical result in the submitted notebook. |
| Reproducibility | Executed notebooks, Drive checkpoints/caches, frozen CSV provided | Strong starting point; packaging and recorded environment versions need attention. |

The guidelines provided are dated 29 July 2026 and warn that Blackboard updates can supersede them. This review does not check Blackboard. Code/presentation deadline: 23 October; report: 30 October. Report limit: eight A4 pages in IEEE two-column format, with the stated exclusions. All members must understand the whole project and provide the required contribution statement.

## What the saved results actually support

E1 means on 121 test days; RL figures average the three training seeds:

| Controller | Weighted objective | Electricity bill ($/day) | Discomfort (°C·h/day) | Mean daily maximum import (kW) |
|---|---:|---:|---:|---:|
| Rule | 26.05 | 18.11 | 0.31 | 18.61 |
| Two-step lookahead | 17.30 | 13.26 | 0.92 | 6.80 |
| DQN | 17.46 | 12.23 | 0.86 | 10.76 |
| REINFORCE | 19.51 | 15.18 | 0.10 | 14.36 |
| A2C | 17.57 | 12.79 | 0.51 | 8.43 |

DQN's mean simulated bill is about 32.5% below the fixed rule, but its mean discomfort is higher. A2C has a different cost/comfort/battery-use trade-off. REINFORCE has the lowest mean discomfort, while lookahead has the best combined objective. These are different claims; none is an unconditional winner.

Lookahead mean objectives across E1 / E2 / heat / low solar are 17.30 / 21.28 / 20.63 / 35.69. The best RL means are 17.46 / 22.67 / 20.86 / 37.67. Small differences are not proof of statistical significance. The displayed ± values are population standard deviations across three seed means, not confidence intervals and not variability across weather days.

On the showcased 19 February 2024 heatwave, every RL agent has a higher electricity bill than the fixed rule. Their weighted objectives are lower because the objective also prices discomfort, exceedance, battery use and terminal energy. A demo must not call the objective difference an electricity saving.

Independent re-evaluation of all nine final checkpoints and both conventional controllers reproduced all four scenario tables to their displayed precision (excluding latency). No retraining was performed. Energy conservation, battery bounds, finite zero-PV observations and the dashboard export also passed.

Use `saved_colab_results.json` for the exact extracted tables and their source notebook hash. `verification.json` and `recomputed_scenario_means.csv`, when produced by the verification script, distinguish newly checked results from saved notebook output. No full training rerun is implied.

## Findings requiring attention

### 1. The current-hour irradiance observation is idealised

The weather loader shifts shortwave radiation one position earlier to use an average for [t,t+1). That is an appropriate interval alignment for simulation physics, but the agent also sees PV derived from that interval average before choosing its action at t. At that point in real operation, the realised whole-hour average is not yet known. Consequently, “what a sensor would read now” and “no actual future realised weather” are too strong as currently written.

Separate the plant's realised irradiance from the controller's information. Use a causal last-measured value or a forecast available at the decision time, applying it to all comparators. Alternatively explicitly call this perfect within-hour information. A causal observation change requires retraining and reevaluation; this review has not silently changed the trained observation semantics.

Open-Meteo documents solar radiation as an average over the preceding hour: https://open-meteo.com/en/docs/historical-weather-api

### 2. Claimed identical information is not strictly true

Lookahead queries yesterday's weather at the next hour through `persistence_forecast`; that forecast is not among the 14 RL observation features. Lookahead also has the exact simulator model. No future test-day weather beyond the current interval is supplied to it, but “same 14 observations” does not describe its total information access. Declare the model/information advantage explicitly, or add the same causal forecast features to all RL methods and retrain. A model-aware comparator is legitimate if honestly specified.

### 3. The free-floating check used the wrong action — fixed in the editable notebook

In original cell 27, `e.step(7)` is labelled cooling off / battery hold. The encoding is `5*cool + batt`, so 7 means half cooling / hold. The corrected action is 2. The earlier numerical convergence calculation is for a half-cooled building, and the prose claiming a no-cooling temperature response must be revisited. On 19 February 2024 the corrected no-cooling peak is 33.55°C; the original half-cooling test peaks at 26.08°C. The correction changes this validation test, not the training dynamics. The original executed outputs are retained as historical evidence and are not presented as corrected output.

### 4. Comfort uses the hourly mean, not integrated temperature violations

`reward_terms` applies the comfort-band penalty after averaging the twelve five-minute temperatures. Positive and negative deviations can be hidden by this averaging. A reported zero degree-hours therefore does not guarantee that every substep stayed within 22–25°C. Evaluate the violation at each substep and integrate it if a true degree-hour metric is intended. That changes the reward and would require a full experiment rerun. The dashboard now labels hourly means explicitly; the review script quantifies the difference without changing saved training results. On the demonstration heatwave, REINFORCE reports 0.000 degree-hours under the original metric but 0.092 degree-hours when five-minute violations are integrated. DQN changes from 5.560 to 6.068 degree-hours.

### 5. The baseline needs a stronger justification

The fixed rule has no shown validation tuning. Its PV surplus test subtracts only the non-cooling load, although cooling may also consume PV. With the quantised charge levels, it can import power while attempting to charge from apparent surplus. This is an allowed heuristic but a potentially weak reference, not established best conventional practice. Tune a small thermostat/battery-rule grid using training/validation only. Retain the lookahead and, if feasible, compare an appropriately bounded longer-horizon MPC/search baseline. Do not tune either method to final test performance.

### 6. Daily resets limit deployment and savings claims

Air, mass and battery states reset every midnight. The terminal battery valuation reduces storage depletion incentives but does not price thermal state and does not demonstrate multi-day operation. It is a value per stored kWh, not an exact refill-cost calculation. Run a continuous 7-day carry-over evaluation before claiming ongoing control or multiplying daily savings into annual savings. The proposal promises a check of thermal boundary exploitation; a completed carry-over check was not found.

### 7. Observation is not demonstrated to be a Markov state

The simulator's hidden day selects an entire weather trajectory. The actor sees neither that day nor the complete weather-generating state. Time remaining repairs finite-horizon accounting but does not prove the weather observations are Markov. Describe the policies as using a limited observation of exogenous weather, or justify an approximate Markov representation. A recurrent model is a possible future extension, not a mandatory last-minute addition.

### 8. Demo configuration bugs — fixed

- PV size 0 previously divided by zero in `_obs`. It now produces a finite zero PV feature.
- The rule decoded PV using the default 40 kWp rating when the dashboard changed the PV rating. It now uses `env.cfg`, including the active tariff scale.
- Data/results lookup accepts existing uppercase or lowercase directory names on case-sensitive Colab/Linux systems.
- The dashboard distinguishes validation from test dates and removes the unconnected live-forecast choice.
- Hourly action tables include actual charge/discharge powers rather than implying a clipped requested action was fully executed.

These corrections are made in the shared core and relevant editable notebook definitions. The default-setting learned dynamics and the stored Colab experiments are unchanged. The Drive copy and executed notebooks have not been rewritten.

### 9. Reproducibility and presentation

The initial local results folder was empty, but the supplied `Google Drive/RL_Assignment/Results` contains all final caches and weights. The exported weights have now been copied into the root results folder for the revised dashboard. The two weather files match byte-for-byte, and Drive notebook source matches the executed Colab source.

Saved setup output reports Torch 2.11.0+cu130, Gymnasium 1.3.0 and NumPy 2.1.3; requirements.txt specifies different versions. Record the full original Colab package list if exact training reproduction is needed. Add dashboard dependencies through requirements-dashboard.txt. Do not promise an exact runtime: the saved learning-rate sweep alone took 19.7 minutes and final runs 30.9 minutes, before the second sweep and checks.

Mean daily maxima must not be labelled the maximum peak over the whole test period. Weekday-only comfort results and paired per-day differences would make the trade-offs clearer. Three seeds offer limited statistical evidence. Selecting a demo seed on validation is appropriate, but that single checkpoint is not the three-seed mean.

## Real-world data and a stronger practical direction

The existing weather is already real historical reanalysis input, although it is not a local building sensor record. The useful application is an operator's control-policy comparison: when to pre-cool, retain battery energy or discharge, subject to price and grid constraints. This is a legitimate decision-support prototype. It does not demonstrate installed performance or measured carbon reductions.

For the remaining time, keep the building problem and prioritise causal observations, correct comfort accounting, continuous-state evaluation and conventional comparisons. Additional decorative complexity or a live-weather API will not solve the scientific weaknesses.

Three substantive routes are available; choose one for future development rather than promising all of them now:

1. **A measured load case study.** Building Data Genome 2 supplies hourly measured meters with building metadata. Choose one office, document cleaning and a chronological split, and use the data to ground demand variability. Whole-building electricity includes existing HVAC: do not use it unchanged as non-cooling demand and add simulated cooling again. Meter histories alone cannot identify how new cooling actions change indoor temperature. https://github.com/buds-lab/building-data-genome-project-2
2. **A better-established building benchmark.** BOPTEST provides building-control test cases and performance evaluation tools. This would improve external comparability but requires an integration/runtime pilot and remains simulation, not physical deployment. It is a larger change than a dashboard update. https://github.com/ibpsa/project1-boptest
3. **Control under realistic model mismatch.** Ground a small range of thermal/load uncertainty in evidence, train policies across that range, and test unseen parameter combinations against the same well-tuned conventional controls. This tests a plausible reason to learn feedback under imperfect knowledge. Do not deliberately weaken the MPC or randomly add dimensions to force an RL win. Retraining and new held-out experiments are required.

For a more creative demonstration, the revised interface already makes the operator's decisions inspectable: heatwave, reduced solar and delayed-office scenarios; bill-versus-comfort consequences; actual battery flows; and downloadable action plans. This adds practical clarity without inventing performance.

## What changed during this review

- Added `sustainabuild_dashboard.py`: redesigned Gradio/Plotly interface with working model replay, scenario controls, objective decomposition, recorded evidence, hourly inspection and export.
- Replaced the root demo notebook's interface with a launcher for that module; retained original executed notebook.
- Fixed zero-PV handling and active-configuration rule decoding in the core and editable main notebook.
- Fixed the action in the editable free-float check and portable directory lookup.
- Added `review/saved_colab_results.json`, verification script and this review.
- Added dashboard requirements. No full training run or change to learned reward/observation semantics was performed.

The remaining scientific and assessment issues above are intentionally visible. Frontend polish cannot certify lecturer approval, submission compliance or full marks.
