'use strict';

// Presentation layer only. All numbers come from the Python API or explicitly
// labelled snapshots exported by that same API for the portfolio demonstration.
const agentNotes = {
  DQN: {family: 'Value-based', title: 'Learn the value of each action.', text: 'DQN learns a future-return score for each of the 15 actions, then selects the highest-scoring action during this replay.'},
  REINFORCE: {family: 'Policy gradient', title: 'Learn from complete days.', text: 'REINFORCE learns action probabilities using complete-day returns. This replay selects the most probable action from its trained policy.'},
  A2C: {family: 'Actor–critic', title: 'A policy with a learning partner.', text: 'The actor learns action probabilities. A critic estimates future return to help train the actor. This replay uses the trained actor.'},
  'Rule-based': {family: 'Conventional benchmark', title: 'Follow a fixed operating rule.', text: 'A programmed temperature and battery rule selects the action. It does not learn from training episodes.'},
  'Lookahead (2-step)': {family: 'Planning benchmark', title: 'Simulate the next two moves.', text: 'The planner evaluates 225 two-action combinations with the building model and a persistence forecast, then takes the first action.'}
};
const tourSteps = [
  ['overview', 'Start with the building.', 'Choose a controller and press Play. Watch its cooling and battery decisions change energy flows. The readings show the selected simulated hour.'],
  ['compare', 'Compare the trade-offs.', 'Choose two controllers. They face the same weather, equipment and starting state. A lower electricity bill does not necessarily mean better comfort.'],
  ['lab', 'Change the conditions.', 'Try extra heat, reduced sunshine or a later office schedule. The environment changes; the saved agents are not retrained.'],
  ['evidence', 'Look beyond one day.', 'Read the held-out, multi-seed results and original training curves. The strongest conventional benchmark still wins the mean combined objective.'],
  ['about', 'Connect the demo to the research.', 'Review the observation, action, reward and experiment design, then the assignment evidence and remaining submission requirements.']
];
function tourBanner() {
  if (S.tour < 0) return '';
  const [page, title, text] = tourSteps[S.tour];
  return `<section class="walkthrough" aria-label="Guided demonstration"><span class="tour-count">DEMO ${S.tour + 1} / ${tourSteps.length}</span><div><h3>${title}</h3><p>${text}</p></div><div class="tour-controls"><button id="tour-back" ${S.tour === 0 ? 'disabled' : ''}>Back</button><button id="tour-next" class="next-tour">${S.tour === tourSteps.length - 1 ? 'Finish tour' : 'Next →'}</button><button id="tour-close" aria-label="Close guided demonstration">×</button></div></section>`;
}
function snapshotBanner() {
  if (window.SB_ENGINE_DATA) return `<div class="snapshot-banner"><b>Runs in your browser.</b> The building simulator, rule-based and lookahead controllers and our three trained networks (DQN, REINFORCE, A2C) run here in JavaScript, ported from our Python code and checked against its results. All 213 validation and test days, custom scenarios and manual control are available.</div>`;
  return window.SB_DEMO ? `<div class="snapshot-banner"><b>Recorded demonstration.</b> These are actual outputs exported from the Python simulator and saved policies. Explore two historical days and four preset scenarios. Manual control and custom settings run in the <a href="${esc(window.SB_DEMO.repo)}" target="_blank" rel="noopener">local Python app</a>.</div>` : '';
}
function liveReadings() {
  const r = currentFlow(), manual = S.mode === 'manual';
  const indoor = manual ? S.manual.current.Ta : r.Ta;
  const occupied = r.occ > 0;
  return `<div class="live-readings">
    <div class="live-reading ${occupied && (indoor < 22 || indoor > 25) ? 'warn' : ''}"><small>${manual ? 'INDOOR NOW' : 'INDOOR · HOUR MEAN'}</small><strong>${num(indoor)}°C</strong><span>${occupied ? 'Occupied target: 22–25°C' : 'Office unoccupied this hour'}</span></div>
    <div class="live-reading"><small>BATTERY · ${r.notApplied ? 'INITIAL' : 'END OF HOUR'}</small><strong>${num(r.soc * 100, 0)}%</strong><span>${num(r.soc * scenario().battery)} kWh stored</span></div>
    <div class="live-reading"><small>OUTDOOR TEMPERATURE</small><strong>${num(r.t_out)}°C</strong><span>Historical weather + scenario change</span></div>
    <div class="live-reading"><small>ELECTRICITY PRICE</small><strong>$${num(r.price, 2)}</strong><span>per kWh · assumed tariff</span></div>
  </div>`;
}
function decisionPanel() {
  const r = currentFlow(), name = S.mode === 'manual' ? (r.actor || 'You') : S.focus;
  const note = agentNotes[name] || {family: 'Manual control', title: 'You choose the next action.', text: 'Your actions change the same simulated building used by every trained agent.'};
  const chosen = !r.notApplied;
  const cool = chosen ? ['Off · 0 kW', 'Half · 6 kW', 'Full · 12 kW'][Math.floor(r.action / 5)] : 'Awaiting a decision';
  const battery = chosen ? ['Discharge up to 20 kW', 'Discharge up to 10 kW', 'Hold', 'Charge up to 10 kW', 'Charge up to 20 kW'][r.action % 5] : 'Awaiting a decision';
  return `<aside class="panel decision-panel"><div><span class="eyebrow">THE DECISION · ${chosen ? hour(r.hour) : 'READY'}</span><h3>${esc(name)}</h3><span class="agent-family">${note.family}</span><p>${note.text}</p></div><div><div class="decision-fact"><small>COOLING REQUEST</small><strong>${cool}</strong></div><div class="decision-fact"><small>BATTERY REQUEST</small><strong>${battery}</strong></div><div class="decision-fact"><small>THIS HOUR'S ELECTRICITY COST</small><strong>${chosen ? '$' + num(r.cost, 2) : '—'}</strong></div></div><p class="decision-foot">${chosen ? `Action ${r.action} of 0–14. Actual battery power may be smaller because of physical limits.` : 'Choose an action in the manual controls.'} This explains the method and observed choice, not the network’s hidden reasoning.</p></aside>`;
}
function decisionLoop() {
  const r = currentFlow();
  return `<div class="decision-loop ${S.playing ? 'is-playing' : ''}" aria-label="Observe, decide, act, measure">
    <div class="loop-step"><span class="step-no">01 / OBSERVE</span><b>Read the building</b>14 inputs: temperature, battery, weather, prices and schedule.</div>
    <div class="loop-step"><span class="step-no">02 / DECIDE</span><b>Choose one action</b><span id="loop-action">${r.notApplied ? '3 cooling levels × 5 battery choices.' : `Action ${r.action}: cooling + battery together.`}</span></div>
    <div class="loop-step"><span class="step-no">03 / ACT</span><b>Advance one hour</b>The thermal model and battery respond to that decision.</div>
    <div class="loop-step"><span class="step-no">04 / MEASURE</span><b>Score the outcome</b>Electricity cost, comfort, grid demand and battery use.</div>
  </div>`;
}
function simulatorPage() {
  const manual = S.mode === 'manual';
  const hero = `<section class="sim-hero"><div><span class="eyebrow">SUSTAINABUILD / PERTH ENERGY LAB</span><h1>${manual ? 'Your building.<br><em>Your next move.</em>' : 'Less energy.<br><em>Smarter decisions.</em>'}</h1><p>One office. Solar, a battery and air conditioning. Explore how three reinforcement learning agents balance energy cost and indoor comfort.</p><div class="hero-actions"><button class="button primary" id="start-tour">${icon('play')}Take the guided demo</button><button class="button secondary" id="go-compare">${icon('chart')}Compare agents</button><button class="button secondary" id="export-button">${icon('download')}Export</button></div></div><div class="hero-stamp"><span>Real research</span><b>3 RL</b><span>agent families</span></div></section>`;
  return tourBanner() + hero + toolbar() +
    `<div id="live-readings">${liveReadings()}</div>` +
    (manual
      ? `<div class="manual-layout">${manualPanel()}<div>${energyPanel(true)}<div id="decision-loop">${decisionLoop()}</div></div></div><div id="decision-panel" style="margin-top:20px">${decisionPanel()}</div>`
      : `<div class="sim-layout">${energyPanel()}<div id="decision-panel">${decisionPanel()}</div></div><div id="decision-loop">${decisionLoop()}</div>`) +
    `<div class="source-note">Playback visualises an already simulated day. Training happened earlier; the saved policies do not learn during this demonstration.</div>` +
    `<details class="details-panel"><summary>${manual ? 'Your session totals and comparison' : 'Daily totals and all five controllers'}</summary><div class="details-body">${stats()}${insight()}${comparison()}</div></details>` +
    `<details class="details-panel"><summary>Temperature and grid charts</summary><div class="details-body">${manual ? chartPanel('Indoor temperature', 'Completed hours only · hourly means', 'Ta', {yMin: Math.min(20, ...S.manual.trace.map(r => r.Ta)) - 1, yMax: Math.max(28, ...S.manual.trace.map(r => r.Ta)) + 1, unit: '°C', band: [22, 25]}) : charts()}</div></details>` +
    `<details class="details-panel"><summary>Every hourly decision · inspect or download</summary><div class="details-body">${actionTable()}</div></details>`;
}
function agentCards() {
  return `<div class="agent-grid">${['DQN', 'REINFORCE', 'A2C'].map((name, i) => `<article class="agent-card"><div class="agent-letter">${['Q', 'π', 'A'][i]}</div><h3>${name}</h3><small>${agentNotes[name].family.toUpperCase()}</small><p>${agentNotes[name].title} ${i === 0 ? 'Replay memory and a target network stabilise learning.' : i === 1 ? 'A complete day supplies the returns used to improve the policy.' : 'The critic supplies a baseline and, for shorter rollouts, a bootstrapped return.'}</p></article>`).join('')}</div>`;
}
function comparePage() {
  const names = S.meta.controllers;
  const a = names.includes(S.compareA) ? S.compareA : names[0];
  const b = names.includes(S.compareB) ? S.compareB : names.at(-1);
  const ar = S.replay.results[a], br = S.replay.results[b];
  const option = value => names.map(n => `<option ${n === value ? 'selected' : ''}>${esc(n)}</option>`).join('');
  const duel = [['energy_cost', 'ELECTRICITY BILL', '$', 2], ['discomfort_degh', 'DISCOMFORT · °C·h', '', 2], ['peak_import_kw', 'PEAK GRID · kW', '', 1], ['objective', 'COMBINED OBJECTIVE', '', 2]];
  const chart = (field, title, unit, opts = {}) => {
    const series = [a, b].map(name => ({name, y: S.replay.results[name].trace.map(r => field === 'soc' ? 100 * r.soc : r[field]), x: S.replay.results[name].trace.map(r => field === 'soc' ? r.hour + 1 : r.hour)}));
    const vals = series.flatMap(s => s.y);
    const bounds = field === 'Ta' ? {yMin: Math.min(20, ...vals) - 1, yMax: Math.max(28, ...vals) + 1, band: [22, 25]} : {};
    return `<section class="panel"><div class="panel-head"><div><h3>${title}</h3><p>${field === 'soc' ? 'Charge at the end of each hour' : field === 'Ta' ? 'Hourly mean · comfort band 22–25°C' : 'Actual simulated hourly power'}</p></div><span class="panel-tag">${unit}</span></div><div class="chart">${chartSVG(series, {...bounds, ...opts, unit})}</div>${legend(series)}</section>`;
  };
  return tourBanner() + heading('COMPARE AGENTS / SAME CONDITIONS', 'Different brains.<br>Different trade-offs.', 'Compare two controllers on the same day, then check the full experiment results.') +
    `<div class="scenario-note">${titleDate(scenario().date)} · ${scenario().heat > 0 ? '+' : ''}${scenario().heat}°C change · ${num(scenario().solar * 100, 0)}% sunshine · ${scenario().battery} kWh battery</div>` +
    `<div class="compare-selects"><label>CONTROLLER A<select id="compare-a">${option(a)}</select></label><span class="vs">VS</span><label>CONTROLLER B<select id="compare-b">${option(b)}</select></label><button class="button secondary" id="go-evidence">See held-out results ↗</button></div>` +
    `<div class="metric-duel">${duel.map(([key, title, prefix, decimals]) => `<div class="duel-card"><small>${title}</small>${[[a, ar], [b, br]].map(([n, r]) => `<div class="duel-line"><span><i class="dot" style="background:${colors[n]}"></i>${esc(n)}</span><b>${prefix}${num(r.metrics[key], decimals)}</b></div>`).join('')}</div>`).join('')}</div>` +
    `<p class="duel-note">Lower is better for these four measures. The combined objective includes penalties and an end-of-day battery adjustment; it is not the electricity bill. This is one day, not the mean of the research experiment.</p>` +
    `<div class="chart-grid">${chart('Ta', 'Comfort over the day', '°C')}${chart('soc', 'Saving energy for later', '%', {yMin: 0, yMax: 100})}${chart('p_import', 'Electricity from the grid', 'kW')}${chart('p_cool', 'Cooling decisions', 'kW')}</div>` +
    `<div class="section-intro"><h2>Three genuinely different methods.</h2><p>Each agent controls the entire building independently. They are not three agents collaborating in one episode.</p></div>${agentCards()}` +
    `<div class="source-note">Rule-based and two-step lookahead are conventional benchmarks. They do not count as the three learning agents.</div>`;
}
const labPresets = [
  {key: 'standard', icon: 'sun', name: 'A regular day', tag: 'ORIGINAL CONDITIONS', text: 'Replay the recorded weather with the original equipment, tariff and office schedule.'},
  {key: 'heat', icon: 'thermo', name: 'Turn up the heat', tag: '+4°C OUTDOORS', text: 'Hotter air increases the cooling challenge. Does the same saved policy protect occupied comfort?'},
  {key: 'solar', icon: 'bolt', name: 'Less sunshine', tag: '50% SOLAR IRRADIATION', text: 'Reduce available solar energy. Watch when controllers rely on the grid or stored energy.'},
  {key: 'shift', icon: 'sliders', name: 'A later workday', tag: 'OFFICE SHIFTED +2 HOURS', text: 'Move occupancy and the linked building load two hours later while the tariff stays fixed.'}
];
function labScenario(key) {
  const c = defaults(scenario().date);
  if (key === 'heat') c.heat = 4;
  if (key === 'solar') c.solar = .5;
  if (key === 'shift') c.shift = 2;
  return c;
}
function labPage() {
  return tourBanner() + heading('SCENARIO LAB / TEST THE SAME POLICIES', 'Change the day.<br>Watch the response.', 'Run a controlled experiment. Every controller gets the same changed conditions.') +
    `<div class="toolbar"><div class="field-inline"><label for="day-select">HISTORICAL DAY</label><select id="day-select" aria-label="Historical replay day" data-mutate>${S.meta.days.map(d => `<option value="${d.date}" ${d.date === scenario().date ? 'selected' : ''}>${d.date} · ${d.split} · ${num(d.max_temp)}°C</option>`).join('')}</select></div><span class="duel-note">Your current replay: ${titleDate(scenario().date)}</span></div>` +
    `<div class="scenario-grid">${labPresets.map(p => `<article class="scenario-card"><div class="scenario-icon">${icon(p.icon)}</div><div><h3>${p.name}</h3><span class="scenario-chip">${p.tag}</span><p>${p.text}</p><button class="button primary" data-scenario="${p.key}" data-mutate>Run this scenario ${icon('arrow')}</button></div></article>`).join('')}</div>` +
    `<section class="panel note-card"><span class="eyebrow">ONE CHANGE AT A TIME</span><h2>A fair test starts with the same conditions.</h2><p>These presets reset equipment to the baseline: 40 kWp solar, 60 kWh battery and a 15 kW import threshold. They change one factor only. Use Scenario settings for a custom combination${window.SB_DEMO ? ' in the local Python app' : ''}.</p><p>The policies are reused without retraining. A large equipment change can take an agent outside its training conditions.</p></section>`;
}
function researchExtras() {
  return `<figure class="training-figure"><h3>What happened during training?</h3><p class="source-note">Original saved Colab figure · final training runs, three seeds per agent.</p><img src="assets/research/learning_curves.png" alt="Original learning curves showing training returns and validation objective for DQN, REINFORCE and A2C" loading="lazy"><figcaption>Training returns include exploration. Validation evaluates checkpoint quality on separate days. These curves describe saved training runs; playing the simulator does not train an agent.</figcaption><div class="training-links"><a href="assets/research/learning_curves.png" target="_blank" rel="noopener">Open full-size curves ↗</a><a href="assets/research/tuning_table.csv" download>Download recorded tuning table</a></div></figure><details class="details-panel"><summary>Compare all four recorded experiment scenarios</summary><div class="details-body"><figure class="training-figure"><img src="assets/research/scenario_comparison.png" alt="Saved mean objective across standard, shifted occupancy, high heat and low solar experiments" loading="lazy"><figcaption>Original multi-day experiment figure. Error bars describe variation across training seeds; they are not a guarantee of performance on a real building.</figcaption></figure></div></details>`;
}
function guidePage() {
  const rows = [
    ['Define a real decision problem', 'Choose cooling and battery use jointly. A decision changes later temperatures and available energy.', 'Explain why sequential control matters and why lookahead is a serious alternative. Lecturer approval is still required.'],
    ['Specify the RL environment', 'Notebook §§3–5: thermal transitions, battery limits, 14 observation values, 15 actions and hourly reward.', 'State the idealised PV information, hidden weather and daily reset assumptions.'],
    ['Implement three agent families', 'Notebook §§8–10: DQN, REINFORCE and A2C. Two conventional controllers provide reference points.', 'Explain the implementation and link the required pseudocode to the code.'],
    ['Tune and repeat training', 'Notebook §§11–12: two-stage tuning on validation data; final runs use three seeds per agent.', 'Show tuning effects, learning curves and sample-efficiency/convergence evidence.'],
    ['Test generalisation', 'Notebook §§13–15: unseen days, shifted occupancy, extra heat and reduced solar.', 'Report variation across seeds and distinguish test evidence from an illustrative single day.'],
    ['Critically analyse results', 'Saved results: lookahead has the best mean objective in all four scenarios. RL agents make different cost–comfort trade-offs.', 'Discuss why, what was expected, limitations and possible confounding information differences.'],
    ['Propose improvements and extensions', 'Candidate directions: better weather information, more realistic comfort/state models and longer episodes; extensions could add flexible loads or multiple buildings.', 'These are proposals, not implemented features. The report requires at least 3 improvements, 3 extensions and 6 academic sources with scientific justification.'],
    ['Deliver and explain the work', 'The notebook is the implementation record; this interface is a companion demonstration.', 'Submission also requires the report, presentation, contribution statements and individual understanding. A polished frontend cannot certify marks or compliance.']
  ];
  return tourBanner() + heading('PROJECT GUIDE / FROM DECISION TO EVIDENCE', 'What we built.<br>How it works. What it proves.', 'A plain-language guide to the research and its assignment objectives.', false) +
    `<p class="guide-lead">SustainaBuild asks a practical question: <b>how should an office use cooling, solar and a battery over a day?</b> Buying less electricity is useful, but an uncomfortable office is not a good outcome. The agents learn to balance both, together with grid demand and battery use.</p>` +
    `<div class="guide-flow"><article><span class="guide-number">01</span><h3>Define the world</h3><p>Historical Perth temperature and sunlight drive a simulated office. A thermal model tracks air and building mass. A battery model tracks stored energy.</p></article><article><span class="guide-number">02</span><h3>Learn by experience</h3><p>During training, an agent tries actions, receives rewards and updates its network. Each episode is a 24-hour day.</p></article><article><span class="guide-number">03</span><h3>Select fairly</h3><p>Tune settings and select checkpoints on validation days. Retrain each selected configuration with three final seeds.</p></article><article><span class="guide-number">04</span><h3>Test and question</h3><p>Evaluate unseen days and changed conditions. Compare against conventional rules and planning, then analyse where each method succeeds or fails.</p></article></div>` +
    `<div class="section-intro"><h2>What are the agents actually learning?</h2><p>A rule for choosing the next cooling-and-battery action from the current observation.</p></div>${agentCards()}` +
    `<div class="notes-grid"><section class="panel note-card"><span class="eyebrow">OBSERVATION / 14 NUMBERS</span><h2>What can the agent see?</h2><p>Indoor air and building-mass temperatures; battery charge; outdoor temperature and its change; solar output; building load; current and next-hour prices; current and next-hour occupancy; two time-of-day values; and time remaining.</p><p>It does not receive the future weather sequence or the identity of the day. These observations do not guarantee that hidden weather is fully Markov.</p></section><section class="panel note-card"><span class="eyebrow">ACTION / 15 CHOICES</span><h2>What can it control?</h2><p><b>Cooling:</b> off, 6 kW or 12 kW electrical power.<br><b>Battery:</b> request discharge at 20 or 10 kW, hold, or charge at 10 or 20 kW.</p><p>3 cooling levels × 5 battery levels = 15 joint actions. The simulator clips the battery request to its physical limits and available energy.</p></section></div>` +
    `<section class="panel note-card" style="margin-top:20px"><span class="eyebrow">REWARD / FEEDBACK FOR LEARNING</span><h2>A low bill is only part of success.</h2><div class="reward-formula">Hourly penalty = <span>electricity cost</span> + <span>2 × discomfort</span> + <span>grid exceedance</span> + <span>0.02 × battery throughput</span></div><p>The reward is the negative of this penalty. Discomfort counts occupied degree-hours outside 22–25°C. Grid exceedance counts imported energy above the 15 kW baseline threshold. At the final hour, an energy valuation accounts for how much battery charge remains. Lower total objective means a better result under these chosen weights.</p></section>` +
    `<div class="section-intro"><h2>The experiment in one view.</h2></div><div class="agent-grid"><article class="agent-card"><span class="eyebrow">TRAIN / 423 DAYS</span><h3>Learn the policy</h3><p>Warm-season days in October 2021–April 2023. The agents interact with the simulator, rather than learning actions from a labelled real-building dataset.</p></article><article class="agent-card"><span class="eyebrow">VALIDATION / 92 DAYS</span><h3>Select settings</h3><p>October–December 2023. Used for hyperparameter choices and checkpoint selection, so this is not independent test evidence.</p></article><article class="agent-card"><span class="eyebrow">TEST / 121 DAYS</span><h3>Evaluate generalisation</h3><p>January–April 2024. Standard conditions and three changed-condition experiments, evaluated across three final seeds per RL agent.</p></article></div>` +
    `<div class="insight-bar"><span class="insight-icon">${icon('info')}</span><p><b>The finding is nuanced.</b> RL improves the combined objective over the fixed rule in the recorded scenarios, but two-step lookahead has the lowest mean objective. DQN's low electricity cost and REINFORCE's strong comfort are distinct outcomes. This is evidence about this simulator, not measured savings in an occupied building.</p></div>` +
    `<div class="section-intro"><h2>Where the assignment objectives appear.</h2><p>Evidence mapping, not a promise of full marks or approval.</p></div><section class="panel"><div class="table-wrap"><table class="objective-table"><thead><tr><th>Objective</th><th>Project evidence</th><th>What still needs explaining / delivering</th></tr></thead><tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div></section>` +
    `<details class="details-panel"><summary>Model assumptions, data provenance and loaded policies</summary><div class="details-body">${aboutPage()}</div></details>` +
    `<p class="source-note">Mapping based on the supplied COMP6008 guidelines dated 29 July 2026 and the lecturer’s requirement for three distinct algorithm families. This AI-assisted demo is supplementary; the supplied assignment prohibits generative AI in submitted work. It must not be presented as proof of authorship or assignment compliance.</p><button class="button secondary" id="go-site" style="margin-top:20px">${icon('map')}Explore the Perth weather reference</button>`;
}
function refreshExperience() {
  if ($('#live-readings')) $('#live-readings').innerHTML = liveReadings();
  if ($('#decision-panel')) $('#decision-panel').innerHTML = decisionPanel();
  const loop = $('#decision-loop .decision-loop'); if (loop) loop.classList.toggle('is-playing', S.playing);
  if ($('#loop-action')) { const r=currentFlow(); $('#loop-action').textContent=r.notApplied?'3 cooling levels × 5 battery choices.':`Action ${r.action}: cooling + battery together.`; }
}
function bindExperience() {
  const on = (id, fn, event = 'click') => { const e = $('#' + id); if (e) e.addEventListener(event, fn); };
  on('start-tour', () => { pause(); S.mode = 'auto'; S.tour = 0; setPage('overview'); });
  on('tour-close', () => { S.tour = -1; render(); });
  on('tour-back', () => { S.tour = Math.max(0, S.tour - 1); setPage(tourSteps[S.tour][0]); });
  on('tour-next', () => { if (S.tour === tourSteps.length - 1) { S.tour = -1; setPage('overview'); } else { S.tour++; setPage(tourSteps[S.tour][0]); } });
  on('go-compare', () => setPage('compare'));
  on('go-evidence', () => setPage('evidence'));
  on('go-site', () => setPage('site'));
  on('compare-a', e => { S.compareA = e.target.value; if (S.compareA === S.compareB) S.compareB = S.meta.controllers.find(n => n !== S.compareA); render(); }, 'change');
  on('compare-b', e => { S.compareB = e.target.value; if (S.compareB === S.compareA) S.compareA = S.meta.controllers.find(n => n !== S.compareB); render(); }, 'change');
  $$('[data-scenario]').forEach(b => b.onclick = async () => { await applyScenario(labScenario(b.dataset.scenario)); setPage('overview'); });
  if (window.SB_DEMO && $('#manual-mode')) { $('#manual-mode').disabled = true; $('#manual-mode').title = 'Manual actions require the local Python simulator.'; }
}

// No invented offline simulation: a website without the Python backend can only
// replay the exact scenario snapshots included in the exported demonstration.
let snapshotPromise;
async function recordedApi(path, body) {
  if (!snapshotPromise) snapshotPromise = fetch(window.SB_DEMO.data).then(r => { if (!r.ok) throw Error('Recorded demonstration data could not load.'); return r.json(); });
  const data = await snapshotPromise;
  if (path === '/meta') return data.meta;
  if (path === '/evidence') return data.evidence;
  if (path === '/climate') return data.climate;
  if (path === '/replay') {
    const match = data.replays.find(r => Object.keys(r.scenario).every(k => r.scenario[k] === body[k]));
    if (match) return match;
    throw Error('This recorded demo supports the four preset scenarios. Use the local Python app for custom conditions.');
  }
  throw Error('Manual control requires the local Python simulator.');
}
init();
