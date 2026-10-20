# SustainaBuild - Peak-Aware Building Energy Management with RL

COMP6008 Reinforcement Learning, Group 10
Dasith Gamage, Priyandan Francis, Ushan Gunarathne

A simulated solar-powered office building in Perth. Every hour a controller chooses a cooling level (off / half / full) and a battery action (full or half discharge, hold, half or full charge), giving 15 joint actions. We implemented three RL agents ourselves in PyTorch (DQN, REINFORCE and A2C) and compared them with a rule-based controller and a two-step lookahead controller on held-out weather days.

**18 October review:** see [the detailed audit](review/PROJECT_REVIEW.md) and [verification results](review/verification.json). The revised dashboard and source fixes were AI-assisted development/review work. The supplied assignment prohibits generative AI in submitted work; these changes do not certify submission compliance. Original executed notebooks remain in `ColabNoteBooks/`; the supplied Drive archive remains in `Google Drive/RL_Assignment/`.

## Folder layout

```
SustainaBuild/
├── SustainaBuild_RL_Group10.ipynb   editable main notebook (summary still needs the group's analysis)
├── SustainaBuild_Demo_App.ipynb     interactive dashboard (Gradio) that replays days with the trained agents
├── sustainabuild_core.py            simulator + controller code with documented review fixes
├── sustainabuild_dashboard.py       revised Gradio/Plotly dashboard
├── data/
│   └── perth_hourly_weather_2021_2024.csv   frozen Open-Meteo extract (hourly temp + irradiance)
├── results/                          exported weights copied from the supplied Drive archive
├── review/                           audit, extracted Colab evidence, verification script and results
├── ColabNoteBooks/                   original executed notebooks
├── Google Drive/RL_Assignment/       original complete Drive copy including all training caches
├── requirements-dashboard.txt
├── requirements.txt
└── README.md
```

## Running on Google Colab (recommended)

Everything is kept in the Google Drive folder **`MyDrive/RL_Assignment`**.

1. In `MyDrive/RL_Assignment/` you need the notebook, `sustainabuild_core.py`, `sustainabuild_dashboard.py`, the weather CSV in `data/` or `Data/`, and saved weights in `results/` or `Results/`. Include `review/saved_colab_results.json` for the dashboard's evidence tab. The revised code recognises either existing directory spelling. Keep existing Results files when rerunning.
2. Open `SustainaBuild_RL_Group10.ipynb` in Colab and choose `Runtime > Run all`. Allow Google Drive access when asked. A CPU runtime is enough.
3. Results are written to `MyDrive/RL_Assignment/Results/`, so they stay after the Colab session ends:
   - `tune_*.pkl`, `final_*.pkl`: every training run (weights + learning history)
   - `trained_policies.pt`: best seed of each agent, used by the dashboard
   - `tuning_table.csv`, `E1_test_days.csv`, `E2_shifted_schedule.csv`, `E3_high_heat.csv`, `E3_low_solar.csv`, `scenario_summary.csv`
   - `learning_curves.png`, `E1_test_days.png`, `scenario_comparison.png`, `demo_day.png`
4. Save the executed notebook (`File > Save`) with its outputs. The marking requires that every result shown in the presentation and report appears in the notebook.
5. Optional: open the revised `SustainaBuild_Demo_App.ipynb`. Local mode defaults to no public share link. For Colab, deliberately enable `SHARE_PUBLIC = True` if a public Gradio tunnel is needed.

The demo installs missing Gymnasium, Gradio and Plotly packages. It expects Gradio 6; use `requirements-dashboard.txt` for a local installation. Colab already supplies the main scientific libraries.

## Dashboard (SustainaBuild_Demo_App.ipynb)

Pick a clearly labelled validation or test day, select controllers and inspect one controller's bill, discomfort, import peak and weighted objective. Compare interactive traces, inspect actual battery power, explore scenario presets and export the replay. The evidence tab displays saved multi-day results separately. The rule and inspected controller are always included. Changed controls require another replay; the current result remains labelled with its original scenario.

The building is simulated and the weather is historical. There is no live forecast or building connection. The loaded checkpoints are selected using validation results; validation dates are not independent test dates. If weights are unavailable, only conventional controllers are enabled.

## Energy Studio: custom web frontend

The new local interface combines an architectural dashboard, animated energy flows, a Perth map, and automatic/manual control. It serves `frontend/` through `sustainabuild_web.py` and reuses `Dashboard`, the existing environment and exported Torch policies. No training or building equations are changed by this interface.

From this directory, with the existing review runtime installed:

```powershell
.\Start-Studio.ps1
```

Open **http://127.0.0.1:8000**. Keep the terminal open; Ctrl+C stops the server. On another machine, install `requirements-web.txt` in a short-path Python environment, then run `python sustainabuild_web.py`. No Node build or API key is needed. An alternative port can be supplied with `--port 8001`.

- **Auto:** compare all five available controllers, select a focus controller, play/pause the 24-hour replay, adjust playback speed, or scrub through the day.
- **Manual:** select cooling and battery actions and advance the real simulator one hour. Undo restores the previous physical state. Let a controller take the next hour or complete the remaining hours from your current state. The decision log records who chose each action and the actual power after physical limits. Completed sessions can be compared fairly with full controller days.
- **Scenario settings:** explore heat, solar availability, shifted occupancy, battery size, PV size and import threshold. Every controller is rerun under the same conditions. Existing trained weights are reused, so changed equipment is an exploratory test, not a newly trained policy.
- **Site & weather:** interactive OpenStreetMap and descriptive weather charts for October 2023–April 2024. The pin is the weather reference point, not a surveyed building location. These averages are not long-term climate normals.
- **Experiment evidence:** saved multi-day Colab results remain separate from interactive one-day results. No claim that RL beats the best conventional benchmark is made.
- **Export:** download a JSON snapshot of the current results or a CSV of hourly decisions.

The app binds only to this computer. Manual sessions are temporary: refreshing the page starts a new workspace, restarting the server clears sessions, and inactive server sessions expire after one hour. Export a result before leaving if you need to retain it. The map and optional Google Fonts need internet; the simulator and local interface work without them.

Verification: `python review/test_web.py` checks automatic/backend equivalence, all five policy handovers, manual actions, undo (including the terminal hour), stale-action protection, session isolation, energy balance, input validation and a zero-PV stress scenario. Browser checks cover playback, manual decisions, handover, settings, navigation and responsive layouts. This interface is AI-assisted review/demo material; apply the assignment's authorship rules before including it in assessed work.

## Running locally

The review verified inference and the dashboard with Python 3.12.14, PyTorch 2.14.1 CPU, Gymnasium 1.4.0, Gradio 6.30.0 and Plotly 7.1.0. The original Colab runtime had different versions (see the audit). On Windows, use a short virtual-environment path to avoid PyTorch's long-path installation error:

```powershell
python -m venv "$env:TEMP\sb-env"
& "$env:TEMP\sb-env\Scripts\Activate.ps1"
pip install -r requirements-dashboard.txt
python sustainabuild_dashboard.py
```

The dashboard opens at `http://127.0.0.1:7860`. No public tunnel is created unless `--share` is explicitly supplied. For the main notebook:

```text
pip install notebook
jupyter notebook SustainaBuild_RL_Group10.ipynb
```

For a CPU-only PyTorch install, use `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Switches at the top of the notebook (Section 1)

| Variable | Default | Effect |
|---|---|---|
| `QUICK_RUN` | `False` | `True` cuts every training budget so the whole notebook runs in a few minutes. Use it only to check the code; the results will be poor. |
| `SAVE_TO_DRIVE` | `True` | On Colab, mounts Google Drive and uses `MyDrive/RL_Assignment/` for `Data/` and `Results/`. Ignored when running locally. |
| `RETRAIN` | `True` | `False` loads the saved runs from `Results/` (if present) instead of training again. Useful for re-making the plots. |

Runtime depends on the machine. In the saved Colab run, the first tuning sweep took 19.7 minutes and final training took 30.9 minutes, excluding the second sweep, checks and evaluation. Reserve more time than 30–45 minutes for a full run.

## Reproducibility

- Data split is chronological. Train: Oct 2021 to Apr 2022 and Oct 2022 to Apr 2023. Validation: Oct to Dec 2023. Test: Jan to Apr 2024.
- Every run is seeded (`set_seed`). Tuning uses seeds 0 and 1; final runs use seeds 100, 101 and 102.
- Hyperparameter selection uses validation days. The original numerical-integration check also inspected a test heatwave before final evaluation; use training/validation days for future model-design checks.
- PyTorch CPU results can differ slightly between machines and versions, so numbers may not match the saved outputs to the last decimal.

## Static website build (runs in the browser)

`frontend/engine.js` is a JavaScript port of `sustainabuild_core.py` (building model, environment, rule-based and two-step lookahead controllers) plus forward passes of the saved DQN, REINFORCE and A2C networks. It answers the same API requests as `sustainabuild_web.py`, so the full interface (all 213 validation/test days, custom scenarios and manual control) works on a static host without Python.

```bash
python export_web_engine.py <output-folder>               # index.html + assets/, ready to host
python export_web_engine.py <output-folder> --reference   # also writes Python reference results
node review/check_engine.mjs <output-folder>               # compares the JavaScript engine with Python
```

The check covers 6 scenarios × 213 days × 5 controllers: every hourly action matched and the largest numerical difference was below 1e-13.
