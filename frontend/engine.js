'use strict';
// SustainaBuild engine for the static website.
// A line-by-line JavaScript port of sustainabuild_core.py (building model, environment, rule-based and
// two-step lookahead controllers) plus forward passes of the saved DQN / REINFORCE / A2C networks.
// It answers the same /api requests as sustainabuild_web.py, so the interface works without Python.
// Where the Python code does arithmetic in float32 (the observation, and the controllers that read it)
// Math.fround reproduces numpy's float32 rounding. check_engine.mjs compares this file with Python.
(function (root) {
  const f32 = Math.fround;
  const COOL = [0, 0.5, 1];
  const BATT = [-1, -0.5, 0, 0.5, 1];
  const N_ACTIONS = 15;
  const LOOK = 'Lookahead (2-step)';
  const mod = (a, n) => ((a % n) + n) % n;

  // ---------- schedules and PV (sustainabuild_core: occupancy, non_cooling_load, tariff, pv_power) ----------
  function occupancy(c, hour, weekday) {
    const h = mod(hour - c.schedule_shift, 24);
    return h >= c.open_hour && h < c.close_hour && weekday ? 1 : 0;
  }
  const nonCoolingLoad = (c, hour, wd) => c.base_load + (c.occupied_load - c.base_load) * occupancy(c, hour, wd);
  function tariff(c, hour) {
    const h = mod(hour, 24);
    return h >= 15 && h < 21 ? c.price_peak : h >= 7 && h < 15 ? c.price_shoulder : c.price_offpeak;
  }
  function pvPower(c, ghi, tOut) {
    const tCell = tOut + 0.03 * ghi;
    return Math.max(c.pv_kwp * (ghi / 1000) * c.pv_derate * (1 + c.pv_temp_coef * (tCell - 25)), 0);
  }

  // ---------- one-hour transition (building_step) ----------
  // f32State = true reproduces the lookahead's first step, where Ta, Tm and E arrive as numpy float32
  // scalars read back from the observation; everywhere else the state is float64.
  function buildingStep(c, Ta, Tm, E, ci, bi, tOut, ghi, pLoad, occ, f32State) {
    const dt = c.dt;
    const pCool = COOL[ci] * c.p_cool_max;
    const pReq = BATT[bi] * c.batt_power;
    const pPv = pvPower(c, ghi, tOut);
    const eMin = c.soc_min * c.batt_capacity, eMax = c.soc_max * c.batt_capacity;
    let chRoom, disRoom;
    if (f32State) {
      chRoom = Math.max(0, f32(f32(f32(eMax) - E) / f32(c.eta_c * dt)));
      disRoom = Math.max(0, f32(f32(f32(f32(E - f32(eMin)) * f32(c.eta_d))) / f32(dt)));
    } else {
      chRoom = Math.max(0, (eMax - E) / (c.eta_c * dt));
      disRoom = Math.max(0, (E - eMin) * c.eta_d / dt);
    }
    const net = Math.max(0, pLoad + pCool - pPv);
    const pCh = Math.min(Math.max(pReq, 0), chRoom);
    const pDis = Math.min(Math.min(Math.max(-pReq, 0), disRoom), net);
    const ENext = E + c.eta_c * pCh * dt - pDis / c.eta_d * dt;
    const qInt = c.occupant_gain * occ + c.load_heat_frac * pLoad + c.solar_aperture * ghi / 1000;
    const h = dt / c.n_substeps;
    let TaSum = 0;
    for (let k = 0; k < c.n_substeps; k++) {
      const qOa = (tOut - Ta) / c.R_oa;
      let qMa, TmNew;
      if (f32State && k === 0) {
        qMa = f32(f32(Tm - Ta) / f32(c.R_am));
        TmNew = f32(Tm - f32(f32(h / c.C_m) * qMa));
      } else {
        qMa = (Tm - Ta) / c.R_am;
        TmNew = Tm - h / c.C_m * qMa;
      }
      const TaNew = Ta + h / c.C_a * (qOa + qMa + qInt - c.cop * pCool);
      Tm = TmNew;
      Ta = TaNew;
      TaSum = TaSum + Ta;
    }
    const pGrid = pLoad + pCool + pCh - pDis - pPv;
    return {Ta, Tm, E: ENext, Ta_mean: TaSum / c.n_substeps, p_cool: pCool, p_ch: pCh, p_dis: pDis,
            p_pv: pPv, p_load: pLoad, p_import: Math.max(pGrid, 0), p_curtail: Math.max(-pGrid, 0)};
  }

  function rewardTerms(c, r, price, occ) {
    const dt = c.dt;
    const cost = price * r.p_import * dt;
    const discomfort = occ * (Math.max(0, r.Ta_mean - c.t_high) + Math.max(0, c.t_low - r.Ta_mean)) * dt;
    const exceed = Math.max(0, r.p_import - c.peak_limit) * dt;
    const throughput = (r.p_ch + r.p_dis) * dt;
    const reward = -(cost + c.lambda_d * discomfort + c.lambda_p * exceed + c.lambda_b * throughput);
    return {reward, terms: {cost, discomfort, exceed, throughput}};
  }

  // ---------- environment (SustainaBuildEnv) ----------
  class Env {
    constructor(day, cfg) { this.day = day; this.cfg = cfg; this.t = 0; }
    weather(hour, previous) {
      const d = this.day;
      return [(previous ? d.prev_temp : d.temp)[hour] + this.cfg.temp_offset,
              (previous ? d.prev_ghi : d.ghi)[hour] * this.cfg.solar_scale];
    }
    obs() {
      const c = this.cfg, h = this.t, wd = this.day.weekday;
      const [tOut, ghi] = this.weather(h);
      const tPrev = h > 0 ? this.weather(h - 1)[0] : this.weather(23, true)[0];
      return Float32Array.from([
        (this.Ta - 24) / 5, (this.Tm - 24) / 5, this.E / c.batt_capacity, (tOut - 25) / 10, (tOut - tPrev) / 5,
        c.pv_kwp > 0 ? pvPower(c, ghi, tOut) / c.pv_kwp : 0, nonCoolingLoad(c, h, wd) / 20,
        tariff(c, h) / c.price_peak, tariff(c, h + 1) / c.price_peak, occupancy(c, h, wd), occupancy(c, h + 1, wd),
        Math.sin(2 * Math.PI * h / 24), Math.cos(2 * Math.PI * h / 24), (24 - h) / 24]);
    }
    reset() {
      const c = this.cfg;
      this.t = 0; this.Ta = c.t_air_init; this.Tm = c.t_mass_init;
      this.E = c.soc_init * c.batt_capacity; this.E_initial = this.E;
      return this.obs();
    }
    step(action) {
      const c = this.cfg, h = this.t, wd = this.day.weekday;
      const ci = Math.floor(action / 5), bi = action % 5;
      const [tOut, ghi] = this.weather(h);
      const occ = occupancy(c, h, wd), pLoad = nonCoolingLoad(c, h, wd), price = tariff(c, h);
      const res = buildingStep(c, this.Ta, this.Tm, this.E, ci, bi, tOut, ghi, pLoad, occ, false);
      let {reward, terms} = rewardTerms(c, res, price, occ);
      this.Ta = res.Ta; this.Tm = res.Tm; this.E = res.E;
      this.t += 1;
      const done = this.t >= 24;
      let terminalAdj = 0;
      if (done) { terminalAdj = c.p_ref * (this.E_initial - this.E); reward -= terminalAdj; }
      const info = {hour: h, t_out: tOut, occ, price, terminal_adj: terminalAdj, Ta: res.Ta_mean, Tm: this.Tm,
                    soc: this.E / c.batt_capacity, p_cool: res.p_cool, p_ch: res.p_ch, p_dis: res.p_dis, p_pv: res.p_pv,
                    p_load: res.p_load, p_import: res.p_import, p_curtail: res.p_curtail, ...terms};
      return {obs: done ? new Float32Array(14) : this.obs(), reward, done, info};
    }
    knownSchedule(hour) { const c = this.cfg, wd = this.day.weekday; return [occupancy(c, hour, wd), nonCoolingLoad(c, hour, wd), tariff(c, hour)]; }
    currentWeather() { return this.weather(this.t); }
    persistenceForecast(hour) { return this.weather(mod(hour, 24), true); }
  }

  // obs_to_dict: numpy float32 arithmetic on the observation
  function obsToDict(o, c) {
    return {Ta: f32(f32(o[0] * 5) + 24), Tm: f32(f32(o[1] * 5) + 24), soc: o[2], E: f32(o[2] * f32(c.batt_capacity)),
            pv: f32(o[5] * f32(c.pv_kwp)), load: f32(o[6] * 20), price: f32(o[7] * f32(c.price_peak)),
            occ: o[9], occ_next: o[10], hour: Math.round(f32(24 - f32(o[13] * 24)))};
  }

  // ---------- controllers ----------
  const rule = {
    name: 'Rule-based',
    act(o, env) {
      const c = env.cfg, s = obsToDict(o, c);
      let cool = 0;
      if (s.occ > 0.5 || s.occ_next > 0.5) cool = s.Ta > 24 ? 2 : s.Ta > 23 ? 1 : 0;
      const surplus = f32(s.pv - s.load);
      let batt;
      if (surplus > 5) batt = 4;
      else if (surplus > 1) batt = 3;
      else if (s.price >= f32(c.price_peak - 1e-6)) batt = 0;
      else if (s.hour < 7 && s.soc < 0.5) batt = 3;
      else batt = 2;
      return cool * 5 + batt;
    }
  };

  function lookahead(cfg) {
    return {
      name: LOOK,
      act(o, env) {
        const c = env.cfg, h = env.t, s = obsToDict(o, c), E0 = s.E;
        const [tOut, ghi] = env.currentWeather();
        const [occ, load, price] = env.knownSchedule(h);
        let best = -Infinity, bestA = 0;
        const first = [];
        for (let a1 = 0; a1 < N_ACTIONS; a1++) {
          const r1 = buildingStep(c, s.Ta, s.Tm, E0, Math.floor(a1 / 5), a1 % 5, tOut, ghi, load, occ, true);
          first.push([r1, rewardTerms(c, r1, price, occ).reward]);
        }
        let next = null;
        if (h + 1 < 24) next = [...env.persistenceForecast(h + 1), ...env.knownSchedule(h + 1)];
        for (let a1 = 0; a1 < N_ACTIONS; a1++) {
          const [r1, rew1] = first[a1];
          for (let a2 = 0; a2 < N_ACTIONS; a2++) {
            let total = 0 + rew1, EEnd = r1.E;
            if (next) {
              const [t2, g2, occ2, load2, price2] = next;
              const r2 = buildingStep(c, r1.Ta, r1.Tm, r1.E, Math.floor(a2 / 5), a2 % 5, t2, g2, load2, occ2, false);
              total += rewardTerms(c, r2, price2, occ2).reward;
              EEnd = r2.E;
            }
            total -= c.p_ref * (E0 - EEnd);
            if (total > best) { best = total; bestA = a1; }       // first maximum, like np.argmax
          }
        }
        return bestA;
      }
    };
  }

  // ---------- saved policy networks (float32 forward pass) ----------
  function decodeF32(b64) {
    const bin = atob(b64), bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }
  function policy(name, spec) {
    const layers = spec.layers.map(l => ({rows: l.rows, cols: l.cols, W: decodeF32(l.W), b: decodeF32(l.b)}));
    const act = spec.activation === 'relu' ? x => (x > 0 ? x : 0) : x => f32(Math.tanh(x));
    return {
      name,
      act(o) {
        let x = o;
        layers.forEach((l, li) => {
          const y = new Float32Array(l.rows);
          for (let i = 0; i < l.rows; i++) {
            let sum = 0;
            const off = i * l.cols;
            for (let j = 0; j < l.cols; j++) sum += l.W[off + j] * x[j];
            const v = f32(sum + l.b[i]);
            y[i] = li < layers.length - 1 ? act(v) : v;
          }
          x = y;
        });
        let bestA = 0;
        for (let i = 1; i < x.length; i++) if (x[i] > x[bestA]) bestA = i;
        return bestA;
      }
    };
  }

  // ---------- the same JSON API as sustainabuild_web.py ----------
  let data, policies = {};
  const sessions = new Map();
  const LIMITS = {heat: [-4, 6], solar: [0.2, 1.2], shift: [0, 4], battery: [10, 120], pv: [0, 80], limit: [8, 30]};

  function scenarioOf(body) {
    const s = {date: String(body.date), heat: 0, solar: 1, shift: 0, battery: 60, pv: 40, limit: 15};
    for (const k of Object.keys(LIMITS)) {
      if (body[k] === undefined) continue;
      const v = Number(body[k]);
      if (!Number.isFinite(v) || v < LIMITS[k][0] || v > LIMITS[k][1] || (k === 'shift' && !Number.isInteger(v)))
        throw Error(`Scenario setting "${k}" is outside the supported range.`);
      s[k] = v;
    }
    return s;
  }
  function configOf(s) {
    return {...data.base_config, temp_offset: s.heat, solar_scale: s.solar, schedule_shift: s.shift,
            batt_capacity: s.battery, pv_kwp: s.pv, peak_limit: s.limit};
  }
  function dayOf(s) {
    const d = data.days.find(x => x.date === s.date);
    if (!d) throw Error('Choose a date from the available historical days.');
    return d;
  }
  function controller(name, cfg) {
    if (name === 'Rule-based') return rule;
    if (name === LOOK) return lookahead(cfg);
    if (policies[name]) return policies[name];
    throw Error('That controller is not available. Check the trained model file.');
  }
  function metrics(trace, c, completed) {
    if (!trace.length) return {objective: 0, energy_cost: 0, discomfort_degh: 0, peak_import_kw: 0, exceed_kwh: 0,
      throughput_kwh: 0, terminal_adj: 0, final_soc: c.soc_init, violation_hours: 0, import_kwh: 0,
      curtailed_kwh: 0, completed: false, hours: 0};
    const total = k => trace.reduce((a, r) => a + r[k], 0);
    const cost = total('cost'), discomfort = total('discomfort'), exceed = total('exceed'), throughput = total('throughput');
    const terminal = total('terminal_adj');
    return {objective: cost + c.lambda_d * discomfort + c.lambda_p * exceed + c.lambda_b * throughput + terminal,
            energy_cost: cost, discomfort_degh: discomfort, peak_import_kw: Math.max(...trace.map(r => r.p_import)),
            exceed_kwh: exceed, throughput_kwh: throughput, terminal_adj: terminal, final_soc: trace.at(-1).soc,
            violation_hours: trace.filter(r => r.discomfort > 1e-6).length, import_kwh: total('p_import') * c.dt,
            curtailed_kwh: total('p_curtail') * c.dt, completed, hours: trace.length};
  }

  function runDay(name, day, cfg) {
    const ctrl = controller(name, cfg), env = new Env(day, cfg);
    let o = env.reset(), done = false, time = 0;
    const trace = [];
    while (!done) {
      const t0 = performance.now();
      const a = ctrl.act(o, env);
      time += performance.now() - t0;
      const r = env.step(a);
      o = r.obs; done = r.done;
      trace.push({...r.info, action: a, date: day.date});
    }
    return {trace, metrics: metrics(trace, cfg, true), latency_ms: time / 24};
  }

  function replay(body) {
    const s = scenarioOf(body), cfg = configOf(s), day = dayOf(s), results = {};
    for (const name of data.meta.controllers) results[name] = runDay(name, day, cfg);
    return {scenario: s, config: cfg, results, weather_sha256: data.weather_sha256, policy_info: data.policy_info,
            kind: 'historical_simulation', interval_information: 'Realised PV average for the current hour is known to the controller.'};
  }

  class Session {
    constructor(s) { this.scenario = s; this.cfg = configOf(s); this.day = dayOf(s); this.restart(); }
    restart() { this.env = new Env(this.day, this.cfg); this.obs = this.env.reset(); this.history = []; this.trace = []; }
    step(action, actor) {
      const r = this.env.step(action);
      this.obs = r.obs;
      this.history.push({action, actor});
      this.trace.push({...r.info, hour: r.info.hour, action, actor});
    }
    response(sid) {
      const e = this.env, done = e.t === 24;
      const current = {hour: e.t, Ta: e.Ta, Tm: e.Tm, soc: e.E / this.cfg.batt_capacity};
      if (!done) {
        const [out, ghi] = e.currentWeather(), [occ, load, price] = e.knownSchedule(e.t);
        Object.assign(current, {t_out: out, p_pv: pvPower(this.cfg, ghi, out), occ, p_load: load, price});
      }
      return structuredClone({session_id: sid, scenario: this.scenario, current, trace: this.trace,
        metrics: metrics(this.trace, this.cfg, done), done, weather_sha256: data.weather_sha256, policy_info: data.policy_info});
    }
  }
  function session(sid) {
    const s = sessions.get(sid);
    if (!s) throw Error('This manual session has expired. Start a new day.');
    return s;
  }
  function checkHour(s, expected) {
    if (s.env.t !== expected) throw Error('The hour has already changed. Refresh the manual session before continuing.');
    if (s.env.t >= 24) throw Error('The day is complete. Start a new day or undo the last action.');
  }

  async function load(url) {
    if (data) return;
    const r = await fetch(url);
    if (!r.ok) throw Error('The simulator data could not load.');
    data = await r.json();
    for (const [name, spec] of Object.entries(data.policies)) policies[name] = policy(name, spec);
  }

  async function api(path, body) {
    await load(root.SB_ENGINE_DATA);
    const clone = x => structuredClone(x);
    if (path === '/meta') return clone(data.meta);
    if (path === '/evidence') return clone(data.evidence);
    if (path === '/climate') return clone(data.climate);
    if (path === '/replay') return replay(body);
    if (path === '/manual') {
      const s = new Session(scenarioOf(body)), sid = crypto.randomUUID().replace(/-/g, '');
      sessions.set(sid, s);
      return s.response(sid);
    }
    const m = path.match(/^\/manual\/([0-9a-f]+)(?:\/(step|finish|undo))?$/);
    if (!m) throw Error('Unknown request.');
    const [, sid, kind] = m, s = session(sid);
    if (!kind) return s.response(sid);
    if (kind === 'step') {
      checkHour(s, body.expected_hour);
      const manualAction = body.action !== undefined && body.action !== null;
      if (manualAction === (body.controller !== undefined && body.controller !== null)) throw Error('Choose either a manual action or a controller.');
      if (manualAction && !(Number.isInteger(body.action) && body.action >= 0 && body.action < N_ACTIONS)) throw Error('Choose an action from 0 to 14.');
      const a = manualAction ? body.action : controller(body.controller, s.cfg).act(s.obs, s.env);
      s.step(a, manualAction ? 'You' : body.controller);
      return s.response(sid);
    }
    if (kind === 'finish') {
      checkHour(s, body.expected_hour);
      const ctrl = controller(body.controller, s.cfg);
      while (s.env.t < 24) s.step(ctrl.act(s.obs, s.env), body.controller);
      return s.response(sid);
    }
    if (s.env.t !== body.expected_hour) throw Error('The session has changed. Refresh it before undoing.');
    const history = s.history.slice(0, -1);
    s.restart();
    for (const r of history) s.step(r.action, r.actor);
    return s.response(sid);
  }

  root.SB_ENGINE = {api, _internal: {Env, buildingStep, rule, lookahead, policy, runDay, configOf, scenarioOf,
    setData: d => { data = d; policies = {}; for (const [n, sp] of Object.entries(d.policies)) policies[n] = policy(n, sp); }}};
})(typeof window !== 'undefined' ? window : globalThis);
