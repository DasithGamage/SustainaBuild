"""Local web interface for the existing SustainaBuild simulator and trained policies.

python sustainabuild_web.py --port 8000
No model training or changes to the original dynamics/reward are performed here.
"""
from dataclasses import asdict, replace
from pathlib import Path
from threading import RLock
import time
import uuid

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

import sustainabuild_core as sb
from sustainabuild_dashboard import Dashboard

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(2)
service = Dashboard(ROOT)
app = FastAPI(title="SustainaBuild Studio", docs_url="/api/docs")
app.mount("/assets", StaticFiles(directory=ROOT / "frontend"), name="assets")


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    date: str
    heat: float = Field(default=0, ge=-4, le=6)
    solar: float = Field(default=1, ge=.2, le=1.2)
    shift: int = Field(default=0, ge=0, le=4)
    battery: float = Field(default=60, ge=10, le=120)
    pv: float = Field(default=40, ge=0, le=80)
    limit: float = Field(default=15, ge=8, le=30)

    def config(self):
        return replace(sb.BASE_CFG, temp_offset=self.heat, solar_scale=self.solar,
                       schedule_shift=self.shift, batt_capacity=self.battery,
                       pv_kwp=self.pv, peak_limit=self.limit)


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_hour: int = Field(ge=0, le=23)
    action: int | None = Field(default=None, ge=0, le=14)
    controller: str | None = None

    @model_validator(mode="after")
    def one_source(self):
        if (self.action is None) == (self.controller is None):
            raise ValueError("Choose either a manual action or a controller.")
        return self


class Finish(BaseModel):
    expected_hour: int = Field(ge=0, le=23)
    controller: str


class Undo(BaseModel):
    expected_hour: int = Field(ge=1, le=24)


def day_for(scenario):
    matches = np.where(service.days.dates.strftime("%Y-%m-%d") == scenario.date)[0]
    if not len(matches):
        raise HTTPException(422, "Choose a date from the available historical days.")
    return service.days.subset([int(matches[0])])


def get_controller(name, cfg):
    if name not in service.names:
        raise HTTPException(422, "That controller is not available. Check the trained model file.")
    return service.controller(name, cfg)


def clean_trace(frame):
    return [{k: (str(v) if k == "date" else int(v) if k in ("action", "hour") else float(v))
             for k, v in r.items()} for r in frame.to_dict("records")]


def metrics(trace, cfg, completed):
    if not trace:
        return dict(objective=0, energy_cost=0, discomfort_degh=0, peak_import_kw=0,
                    exceed_kwh=0, throughput_kwh=0, terminal_adj=0, final_soc=cfg.soc_init,
                    violation_hours=0, import_kwh=0, curtailed_kwh=0, completed=False, hours=0)
    def total(key): return float(sum(r[key] for r in trace))
    cost, discomfort, exceed, throughput = [total(k) for k in ("cost", "discomfort", "exceed", "throughput")]
    terminal = total("terminal_adj")
    return dict(objective=cost+cfg.lambda_d*discomfort+cfg.lambda_p*exceed+cfg.lambda_b*throughput+terminal,
                energy_cost=cost, discomfort_degh=discomfort, peak_import_kw=max(r["p_import"] for r in trace),
                exceed_kwh=exceed, throughput_kwh=throughput, terminal_adj=terminal,
                final_soc=trace[-1]["soc"], violation_hours=sum(r["discomfort"] > 1e-6 for r in trace),
                import_kwh=total("p_import")*cfg.dt, curtailed_kwh=total("p_curtail")*cfg.dt,
                completed=completed, hours=len(trace))


@app.get("/")
def home():
    return FileResponse(ROOT / "frontend/index.html")


@app.get("/api/meta")
def meta():
    days = [dict(date=str(date.date()), weekday=bool(wd), max_temp=float(temp.max()),
                 split="Test" if date >= pd.Timestamp("2024-01-01") else "Validation")
            for date, wd, temp in zip(service.days.dates, service.days.weekday, service.days.temp)]
    return dict(controllers=service.names, policy_info=service.policy_info, model_messages=service.load_messages,
                days=days, default_date=service.default[:10], config=asdict(sb.BASE_CFG),
                weather_sha256=service.weather_hash, location=dict(lat=-32.005, lon=115.894),
                timezone="Australia/Perth", interface_version="2.0")


@app.get("/api/evidence")
def evidence():
    return service.evidence or {"tables": {}, "provenance": "No saved experiment file available."}


@app.get("/api/climate")
def climate():
    # A descriptive view of the available evaluation weather, not long-term climate normals.
    rows=[]
    for month in [10,11,12,1,2,3,4]:
        ids=np.where(service.days.dates.month == month)[0]
        rows.append(dict(month=month, days=len(ids), mean_temp=float(service.days.temp[ids].mean()),
                         mean_daily_max=float(service.days.temp[ids].max(1).mean()),
                         solar_kwh_m2_day=float(service.days.ghi[ids].sum(1).mean()/1000)))
    return dict(period="October 2023 – April 2024", months=rows)


@app.post("/api/replay")
def replay(scenario: Scenario):
    day, cfg = day_for(scenario), scenario.config()
    results={}
    for name in service.names:
        df=sb.run_controller(get_controller(name,cfg),day,cfg,days=[0],keep_trace=True)
        trace=clean_trace(df.attrs["trace"])
        results[name]=dict(trace=trace, metrics=metrics(trace,cfg,True), latency_ms=df.attrs["latency_ms"])
    return dict(scenario=scenario.model_dump(), config=asdict(cfg), results=results,
                weather_sha256=service.weather_hash, policy_info=service.policy_info,
                kind="historical_simulation", interval_information="Realised PV average for the current hour is known to the controller.")


class Session:
    def __init__(self, scenario):
        self.scenario=scenario
        self.cfg=scenario.config()
        self.day=day_for(scenario)
        self.env=sb.SustainaBuildEnv(self.day,self.cfg)
        self.obs,_=self.env.reset(options={"day":0})
        self.history=[]
        self.trace=[]
        self.updated=time.monotonic()
        self.lock=RLock()

    def step(self, action, actor):
        self.obs,_,_,_,info=self.env.step(action)
        self.history.append(dict(action=action,actor=actor))
        row={k:float(v) for k,v in info.items()}
        row.update(hour=int(info["hour"]),action=action,actor=actor)
        self.trace.append(row)

    def response(self, sid):
        e=self.env; done=e.t==24
        current=dict(hour=e.t, Ta=e.Ta,Tm=e.Tm,soc=e.E/self.cfg.batt_capacity)
        if not done:
            out,ghi=e.current_weather();occ,load,price=e.known_schedule(e.t)
            current.update(t_out=float(out),p_pv=float(sb.pv_power(self.cfg,ghi,out)),
                           occ=float(occ),p_load=float(load),price=float(price))
        return dict(session_id=sid,scenario=self.scenario.model_dump(),current=current,
                    trace=self.trace,metrics=metrics(self.trace,self.cfg,done),done=done,
                    weather_sha256=service.weather_hash, policy_info=service.policy_info)


sessions={}
sessions_lock=RLock()


def find_session(sid):
    with sessions_lock:
        s=sessions.get(sid)
        if s is None or time.monotonic()-s.updated>3600:
            sessions.pop(sid,None)
            raise HTTPException(404,"This manual session has expired. Start a new day.")
        s.updated=time.monotonic()
        return s


def check_hour(s,expected):
    if s.env.t != expected:
        raise HTTPException(409,"The hour has already changed. Refresh the manual session before continuing.")
    if s.env.t>=24:
        raise HTTPException(409,"The day is complete. Start a new day or undo the last action.")


@app.post("/api/manual")
def start_manual(scenario: Scenario):
    s=Session(scenario);sid=uuid.uuid4().hex
    with sessions_lock:
        for key in list(sessions):
            if time.monotonic()-sessions[key].updated>3600:
                del sessions[key]
        if len(sessions)>=100:
            raise HTTPException(429,"Too many active sessions. Try again after older sessions expire.")
        sessions[sid]=s
    return s.response(sid)


@app.get("/api/manual/{sid}")
def manual_state(sid:str):
    s=find_session(sid)
    with s.lock:return s.response(sid)


@app.post("/api/manual/{sid}/step")
def manual_step(sid:str,action:Action):
    s=find_session(sid)
    with s.lock:
        check_hour(s,action.expected_hour)
        a=action.action if action.action is not None else get_controller(action.controller,s.cfg).act(s.obs,s.env)
        s.step(int(a),"You" if action.action is not None else action.controller)
        return s.response(sid)


@app.post("/api/manual/{sid}/finish")
def finish_manual(sid:str,request:Finish):
    s=find_session(sid)
    with s.lock:
        check_hour(s,request.expected_hour)
        controller=get_controller(request.controller,s.cfg)
        while s.env.t<24:
            s.step(controller.act(s.obs,s.env),request.controller)
        return s.response(sid)


@app.post("/api/manual/{sid}/undo")
def undo_manual(sid:str,request:Undo):
    s=find_session(sid)
    with s.lock:
        if s.env.t!=request.expected_hour:
            raise HTTPException(409,"The session has changed. Refresh it before undoing.")
        history=s.history[:-1]
        s.obs,_=s.env.reset(options={"day":0});s.history=[];s.trace=[]
        for record in history:s.step(record["action"],record["actor"])
        return s.response(sid)


if __name__=="__main__":
    import argparse
    import uvicorn
    parser=argparse.ArgumentParser(description="Run the SustainaBuild Studio locally")
    parser.add_argument("--port",type=int,default=8000)
    args=parser.parse_args()
    uvicorn.run(app,host="127.0.0.1",port=args.port)
