"""SustainaBuild core: building simulator, Gymnasium environment and controllers.

Extracted from SustainaBuild_RL_Group10.ipynb for use without retraining.
Review fixes: zero-PV observations remain finite; rule observations use the active
building configuration. Default-configuration training dynamics are unchanged.
"""
import time
from dataclasses import dataclass, field, replace, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

@dataclass
class DaySet:
    dates: pd.DatetimeIndex
    temp: np.ndarray        # (n_days, 24) outdoor temperature, degC
    ghi: np.ndarray         # (n_days, 24) irradiance, W/m2
    prev_temp: np.ndarray   # previous day's temperature (persistence forecast only)
    prev_ghi: np.ndarray
    weekday: np.ndarray     # True for Mon-Fri

    def __len__(self):
        return len(self.dates)

    def subset(self, idx):
        return DaySet(self.dates[idx], self.temp[idx], self.ghi[idx],
                      self.prev_temp[idx], self.prev_ghi[idx], self.weekday[idx])


def build_dayset(df):
    groups = [(d, g) for d, g in df.groupby("date") if len(g) == 24]
    dates = pd.DatetimeIndex([d for d, _ in groups])
    temp = np.stack([g["temp_c"].to_numpy() for _, g in groups])
    ghi = np.stack([g["ghi_wm2"].to_numpy() for _, g in groups])
    # previous calendar day; the first day of the file has none so it is dropped below
    prev_temp = np.vstack([np.full((1, 24), np.nan), temp[:-1]])
    prev_ghi = np.vstack([np.full((1, 24), np.nan), ghi[:-1]])
    consecutive = np.r_[False, np.diff(dates.values).astype("timedelta64[D]").astype(int) == 1]
    keep = consecutive & dates.month.isin([10, 11, 12, 1, 2, 3, 4])
    ds = DaySet(dates, temp, ghi, prev_temp, prev_ghi, dates.dayofweek.values < 5)
    return ds.subset(np.where(keep)[0])


@dataclass
class BuildingConfig:
    # thermal model
    C_a: float = 4.0            # kWh/degC
    C_m: float = 40.0           # kWh/degC
    R_oa: float = 2.0           # degC/kW
    R_am: float = 0.1           # degC/kW
    solar_aperture: float = 12.0    # effective glazing area (m2) -> kW of gain per kW/m2
    occupant_gain: float = 5.0      # kW from people when fully occupied
    load_heat_frac: float = 0.8     # share of electrical load released as heat inside
    n_substeps: int = 12            # 5-minute integration inside each hour
    dt: float = 1.0                 # hours per decision step
    # HVAC
    p_cool_max: float = 12.0    # kW electric
    cop: float = 3.0
    # PV
    pv_kwp: float = 40.0
    pv_derate: float = 0.85
    pv_temp_coef: float = -0.004
    # battery
    batt_capacity: float = 60.0     # kWh
    batt_power: float = 20.0        # kW rated, both directions
    eta_c: float = 0.95
    eta_d: float = 0.95
    soc_min: float = 0.10
    soc_max: float = 0.95
    soc_init: float = 0.50
    # load and occupancy
    base_load: float = 4.0
    occupied_load: float = 14.0
    open_hour: int = 8
    close_hour: int = 18
    schedule_shift: int = 0         # E2: shift occupancy and load by this many hours
    # tariff ($/kWh)
    price_offpeak: float = 0.15
    price_shoulder: float = 0.25
    price_peak: float = 0.45
    # comfort and initial temperatures
    t_low: float = 22.0
    t_high: float = 25.0
    t_air_init: float = 24.0
    t_mass_init: float = 25.0
    # reward weights (fixed on training/validation data before testing)
    peak_limit: float = 15.0        # kW
    lambda_d: float = 2.0           # $ per degree-hour of occupied discomfort
    lambda_p: float = 1.0           # $ per kWh imported above the peak limit
    lambda_b: float = 0.02          # $ per kWh battery throughput (wear proxy)
    p_ref: float = 0.15             # $ per kWh for the terminal battery adjustment (off-peak refill cost)
    # scenario modifiers for stress tests (E3)
    temp_offset: float = 0.0
    solar_scale: float = 1.0


BASE_CFG = BuildingConfig()


COOL_LEVELS = np.array([0.0, 0.5, 1.0])


BATT_LEVELS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])


N_ACTIONS = len(COOL_LEVELS) * len(BATT_LEVELS)


def decode_action(a):
    """joint action index -> (cooling index, battery index). a = 5*cool + batt"""
    a = np.asarray(a)
    return a // len(BATT_LEVELS), a % len(BATT_LEVELS)


ACTION_NAMES = [f"cool={['off','half','full'][c]}, batt={['-full','-half','hold','+half','+full'][b]}"
                for c in range(3) for b in range(5)]


def occupancy(cfg, hour, weekday):
    h = (np.asarray(hour) - cfg.schedule_shift) % 24
    occ = (h >= cfg.open_hour) & (h < cfg.close_hour) & np.asarray(weekday)
    return occ.astype(float)


def non_cooling_load(cfg, hour, weekday):
    return cfg.base_load + (cfg.occupied_load - cfg.base_load) * occupancy(cfg, hour, weekday)


def tariff(cfg, hour):
    h = np.asarray(hour) % 24
    return np.where((h >= 15) & (h < 21), cfg.price_peak,
                    np.where((h >= 7) & (h < 15), cfg.price_shoulder, cfg.price_offpeak))


def pv_power(cfg, ghi, t_out):
    t_cell = t_out + 0.03 * ghi
    p = cfg.pv_kwp * (ghi / 1000.0) * cfg.pv_derate * (1 + cfg.pv_temp_coef * (t_cell - 25.0))
    return np.maximum(p, 0.0)


def building_step(cfg, Ta, Tm, E, cool_idx, batt_idx, t_out, ghi, p_load, occ):
    """Advance the building by one hour. Works on scalars or numpy arrays (broadcast)."""
    dt = cfg.dt
    p_cool = COOL_LEVELS[cool_idx] * cfg.p_cool_max
    p_req = BATT_LEVELS[batt_idx] * cfg.batt_power           # +ve = charge, -ve = discharge
    p_pv = pv_power(cfg, ghi, t_out)

    # --- (3) enforce battery limits ---
    e_min, e_max = cfg.soc_min * cfg.batt_capacity, cfg.soc_max * cfg.batt_capacity
    p_ch_room = np.maximum(0.0, (e_max - E) / (cfg.eta_c * dt))     # max charge that still fits
    p_dis_room = np.maximum(0.0, (E - e_min) * cfg.eta_d / dt)      # max discharge available
    net_demand = np.maximum(0.0, p_load + p_cool - p_pv)            # no battery export
    p_ch = np.minimum(np.maximum(p_req, 0.0), p_ch_room)
    p_dis = np.minimum(np.minimum(np.maximum(-p_req, 0.0), p_dis_room), net_demand)

    # --- (4) battery energy and RC thermal update ---
    E_next = E + cfg.eta_c * p_ch * dt - p_dis / cfg.eta_d * dt
    q_int = cfg.occupant_gain * occ + cfg.load_heat_frac * p_load + cfg.solar_aperture * ghi / 1000.0
    h = dt / cfg.n_substeps
    Ta_sum = 0.0
    for _ in range(cfg.n_substeps):
        q_oa = (t_out - Ta) / cfg.R_oa
        q_ma = (Tm - Ta) / cfg.R_am
        Ta_new = Ta + h / cfg.C_a * (q_oa + q_ma + q_int - cfg.cop * p_cool)
        Tm = Tm - h / cfg.C_m * q_ma
        Ta = Ta_new
        Ta_sum = Ta_sum + Ta
    Ta_mean = Ta_sum / cfg.n_substeps      # used for comfort during the hour

    # --- (5) grid balance ---
    p_grid = p_load + p_cool + p_ch - p_dis - p_pv
    p_import = np.maximum(p_grid, 0.0)
    p_curtail = np.maximum(-p_grid, 0.0)
    return dict(Ta=Ta, Tm=Tm, E=E_next, Ta_mean=Ta_mean, p_cool=p_cool, p_ch=p_ch, p_dis=p_dis,
                p_pv=p_pv, p_load=p_load, p_import=p_import, p_curtail=p_curtail)


def reward_terms(cfg, res, price, occ):
    dt = cfg.dt
    cost = price * res["p_import"] * dt
    discomfort = occ * (np.maximum(0.0, res["Ta_mean"] - cfg.t_high) +
                        np.maximum(0.0, cfg.t_low - res["Ta_mean"])) * dt
    exceed = np.maximum(0.0, res["p_import"] - cfg.peak_limit) * dt
    throughput = (res["p_ch"] + res["p_dis"]) * dt
    reward = -(cost + cfg.lambda_d * discomfort + cfg.lambda_p * exceed + cfg.lambda_b * throughput)
    return reward, dict(cost=cost, discomfort=discomfort, exceed=exceed, throughput=throughput)


class SustainaBuildEnv(gym.Env):
    metadata = {"render_modes": []}
    OBS_DIM = 14

    def __init__(self, days, cfg=BASE_CFG, seed=None):
        super().__init__()
        self.days = days
        self.cfg = cfg
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(self.OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.np_random_days = np.random.default_rng(seed)
        self.day = 0
        self.t = 0

    # exogenous inputs for an hour of the current day, with any stress-test modifiers applied
    def _weather(self, hour, previous_day=False):
        temp = self.days.prev_temp if previous_day else self.days.temp
        ghi = self.days.prev_ghi if previous_day else self.days.ghi
        return (temp[self.day, hour] + self.cfg.temp_offset,
                ghi[self.day, hour] * self.cfg.solar_scale)

    def _obs(self):
        c, h = self.cfg, self.t
        wd = self.days.weekday[self.day]
        t_out, ghi = self._weather(h)
        t_prev = self._weather(h - 1)[0] if h > 0 else self._weather(23, previous_day=True)[0]
        obs = np.array([
            (self.Ta - 24.0) / 5.0,
            (self.Tm - 24.0) / 5.0,
            self.E / c.batt_capacity,
            (t_out - 25.0) / 10.0,
            (t_out - t_prev) / 5.0,
            pv_power(c, ghi, t_out) / c.pv_kwp if c.pv_kwp > 0 else 0.0,
            non_cooling_load(c, h, wd) / 20.0,
            tariff(c, h) / c.price_peak,
            tariff(c, h + 1) / c.price_peak,
            occupancy(c, h, wd),
            occupancy(c, h + 1, wd),
            np.sin(2 * np.pi * h / 24),
            np.cos(2 * np.pi * h / 24),
            (24 - h) / 24.0,
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random_days = np.random.default_rng(seed)
        if options is not None and "day" in options:
            self.day = int(options["day"])
        else:
            self.day = int(self.np_random_days.integers(len(self.days)))
        c = self.cfg
        self.t = 0
        self.Ta, self.Tm = c.t_air_init, c.t_mass_init
        self.E = c.soc_init * c.batt_capacity
        self.E_initial = self.E
        return self._obs(), {"date": str(self.days.dates[self.day].date())}

    def step(self, action):
        c, h = self.cfg, self.t
        cool_idx, batt_idx = decode_action(int(action))                       # (2) joint action
        wd = self.days.weekday[self.day]
        t_out, ghi = self._weather(h)
        occ, p_load, price = occupancy(c, h, wd), non_cooling_load(c, h, wd), tariff(c, h)

        res = building_step(c, self.Ta, self.Tm, self.E, cool_idx, batt_idx,   # (3)-(5)
                            t_out, ghi, p_load, occ)
        reward, terms = reward_terms(c, res, price, occ)

        self.Ta, self.Tm, self.E = float(res["Ta"]), float(res["Tm"]), float(res["E"])
        self.t += 1
        terminated = self.t >= 24
        terminal_adj = 0.0
        if terminated:
            terminal_adj = c.p_ref * (self.E_initial - self.E)        # Eq. (5)
            reward -= terminal_adj

        info = dict(hour=h, t_out=t_out, occ=occ, price=price, terminal_adj=terminal_adj,
                    Ta=float(res["Ta_mean"]), Tm=self.Tm, soc=self.E / c.batt_capacity,
                    **{k: float(v) for k, v in res.items() if k.startswith("p_")},
                    **{k: float(v) for k, v in terms.items()})
        obs = self._obs() if not terminated else np.zeros(self.OBS_DIM, dtype=np.float32)
        return obs, float(reward), terminated, False, info

    # ---- information the conventional controllers are allowed to use ----
    def known_schedule(self, hour):
        """occupancy, load and price are scheduled in advance, so they can be looked up"""
        wd = self.days.weekday[self.day]
        return occupancy(self.cfg, hour, wd), non_cooling_load(self.cfg, hour, wd), tariff(self.cfg, hour)

    def current_weather(self):
        """what a sensor would read now (the same values that are in the observation)"""
        return self._weather(self.t)

    def persistence_forecast(self, hour):
        """forecast for a future hour of today = same hour yesterday (past information only)"""
        return self._weather(hour % 24, previous_day=True)


def obs_to_dict(obs, cfg=BASE_CFG):
    """undo the observation scaling (used by the hand-written controllers)"""
    return dict(Ta=obs[0] * 5 + 24, Tm=obs[1] * 5 + 24, soc=obs[2],
                E=obs[2] * cfg.batt_capacity, t_out=obs[3] * 10 + 25, pv=obs[5] * cfg.pv_kwp,
                load=obs[6] * 20, price=obs[7] * cfg.price_peak, occ=obs[9], occ_next=obs[10],
                hour=int(round(24 - obs[13] * 24)))


class RuleBasedController:
    name = "Rule-based"

    def act(self, obs, env=None):
        cfg = env.cfg if env is not None else BASE_CFG
        s = obs_to_dict(obs, cfg)
        # thermostat with one hour of pre-cooling
        if s["occ"] > 0.5 or s["occ_next"] > 0.5:
            cool = 2 if s["Ta"] > 24.0 else (1 if s["Ta"] > 23.0 else 0)
        else:
            cool = 0
        # battery rule
        surplus = s["pv"] - s["load"]
        if surplus > 5.0:
            batt = 4
        elif surplus > 1.0:
            batt = 3
        elif s["price"] >= cfg.price_peak - 1e-6:
            batt = 0
        elif s["hour"] < 7 and s["soc"] < 0.5:
            batt = 3
        else:
            batt = 2
        return int(cool * 5 + batt)


class LookaheadController:
    name = "Lookahead (2-step)"

    def __init__(self, cfg=BASE_CFG):
        self.cfg = cfg
        a1, a2 = np.meshgrid(np.arange(N_ACTIONS), np.arange(N_ACTIONS), indexing="ij")
        self.a1, self.a2 = a1.ravel(), a2.ravel()               # 225 sequences
        self.c1, self.b1 = decode_action(self.a1)
        self.c2, self.b2 = decode_action(self.a2)

    def act(self, obs, env):
        cfg, h = env.cfg, env.t
        s = obs_to_dict(obs, cfg)
        E0 = s["E"]
        total = np.zeros(len(self.a1))
        # hour t: measured weather
        t_out, ghi = env.current_weather()
        occ, load, price = env.known_schedule(h)
        r1 = building_step(cfg, s["Ta"], s["Tm"], E0, self.c1, self.b1, t_out, ghi, load, occ)
        total += reward_terms(cfg, r1, price, occ)[0]
        E_end = r1["E"]
        if h + 1 < 24:
            # hour t+1: persistence forecast for weather, schedule is known
            t_out2, ghi2 = env.persistence_forecast(h + 1)
            occ2, load2, price2 = env.known_schedule(h + 1)
            r2 = building_step(cfg, r1["Ta"], r1["Tm"], r1["E"], self.c2, self.b2,
                               t_out2, ghi2, load2, occ2)
            total += reward_terms(cfg, r2, price2, occ2)[0]
            E_end = r2["E"]
        total -= cfg.p_ref * (E0 - E_end)        # value of energy left in the battery
        return int(self.a1[np.argmax(total)])


def run_controller(controller, day_set, cfg=BASE_CFG, days=None, keep_trace=False):
    env = SustainaBuildEnv(day_set, cfg)
    days = range(len(day_set)) if days is None else days
    rows, traces, latencies = [], [], []
    for d in days:
        obs, _ = env.reset(options={"day": d})
        done, ret, hours = False, 0.0, []
        while not done:
            t0 = time.perf_counter()
            a = controller.act(obs, env)
            latencies.append(time.perf_counter() - t0)
            obs, r, done, _, inf = env.step(a)
            ret += r
            inf["action"] = a
            hours.append(inf)
        h = pd.DataFrame(hours)
        rows.append(dict(
            date=day_set.dates[d].date(), weekday=bool(day_set.weekday[d]),
            objective=-ret,                                   # daily objective incl. terminal adj ($)
            energy_cost=h["cost"].sum(),
            discomfort_degh=h["discomfort"].sum(),
            violation_hours=int(((h["discomfort"] > 1e-6)).sum()),
            peak_import_kw=h["p_import"].max(),
            exceed_kwh=h["exceed"].sum(),
            exceed_hours=int((h["exceed"] > 1e-6).sum()),
            throughput_kwh=h["throughput"].sum(),
            terminal_adj=h["terminal_adj"].sum(),
            final_soc=h["soc"].iloc[-1],
            final_Ta=env.Ta,
            curtailed_kwh=h["p_curtail"].sum(),
        ))
        if keep_trace:
            h["date"] = day_set.dates[d].date()
            traces.append(h)
    out = pd.DataFrame(rows)
    out.attrs["latency_ms"] = 1000 * float(np.mean(latencies))
    if keep_trace:
        out.attrs["trace"] = pd.concat(traces, ignore_index=True)
    return out


def summarise(df):
    keys = ["objective", "energy_cost", "discomfort_degh", "violation_hours", "peak_import_kw",
            "exceed_kwh", "exceed_hours", "throughput_kwh", "final_soc"]
    return df[keys].mean()


def mlp(in_dim, out_dim, hidden=(128, 128), act=nn.ReLU):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class TorchPolicy:
    """wraps a network so it has the same act(obs, env) interface as the rule-based controllers"""
    def __init__(self, net, name):
        self.net, self.name = net, name

    @torch.no_grad()
    def act(self, obs, env=None):
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return int(self.net(x).argmax(1).item())
