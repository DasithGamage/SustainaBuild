"""Interactive review/demo interface. Runs the project's real simulator.

Run from this directory: python sustainabuild_dashboard.py
The executed Colab notebooks remain the evidence for the original training runs.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn

import sustainabuild_core as sb

ROOT = Path(__file__).resolve().parent
LOOK = "Lookahead (2-step)"
COLORS = {"Rule-based": "#98a3a0", LOOK: "#243e37", "DQN": "#2587b3",
          "REINFORCE": "#cf7734", "A2C": "#9365ad"}
PRESETS = {"Standard day": (0, 1, 0), "Heatwave +4°C": (4, 1, 0),
           "Half the sunshine": (0, .5, 0), "Office opens 2h later": (0, 1, 2)}
CSS = """
.gradio-container {max-width:1440px!important; background:#f7f8f4!important; color:#203b33!important;}
footer {display:none!important}
.sb-head {display:flex;align-items:center;justify-content:space-between;padding:12px 2px 20px;border-bottom:1px solid #dce3da;margin-bottom:20px;gap:16px;flex-wrap:wrap}
.sb-brand {font-size:22px;font-weight:750;letter-spacing:-.8px;color:#203b33!important}.sb-mark {display:inline-block;background:#204e3f;color:#d9f2bd;padding:7px 12px;border-radius:12px;margin-right:10px}
.sb-eyebrow {font-size:11px;letter-spacing:2px;text-transform:uppercase;font-weight:700;color:#5d7366}
.sb-hero {background:#203f34;color:#f4f7ee;padding:30px 34px;border-radius:18px;margin-bottom:12px;position:relative;overflow:hidden}
.sb-hero h1 {font-size:clamp(28px,3.1vw,42px);line-height:1.12;letter-spacing:-1.5px;margin:12px 0;color:#f4f7ee!important;font-weight:650}
.sb-hero p {color:#cfdfd4;max-width:730px;line-height:1.6;margin:0}.sb-hero .sb-eyebrow {color:#bde39d}
.sb-badge {display:inline-block;font-size:11px;font-weight:650;border:1px solid #cddacc;background:#eef4e8;color:#35513e;border-radius:50px;padding:6px 11px;margin:4px 6px 4px 0}
.sb-notice {padding:12px 16px;border:1px solid #dfd5bb;background:#fcf7e9;border-radius:12px;color:#6a5632;font-size:13px;line-height:1.5}
.sb-cards {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:10px 0 18px}
.sb-card {border:1px solid #dae2d7;background:#fff;border-radius:14px;padding:17px 18px;min-width:0}
.sb-card strong {display:block;font-size:29px;font-weight:600;letter-spacing:-1px;margin:8px 0;color:#203f34}.sb-card small {font-size:12px;color:#65766b;display:block;line-height:1.5}
.sb-label {font-size:11px;letter-spacing:1px;text-transform:uppercase;color:#627263;font-weight:650}
.sb-run {font-size:12px;color:#67786c;padding:4px 0 8px}.sb-run b {color:#274c3c}
.sb-hour {display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.sb-hour>div {background:#eff4eb;border-radius:10px;padding:14px;font-size:12px;color:#536659}.sb-hour b {display:block;font-size:23px;color:#244a38;margin-top:5px}
.sb-insight {background:white;border-left:4px solid #a6c77e;border-radius:4px 12px 12px 4px;padding:16px 20px;margin:8px 0;line-height:1.6;font-size:14px;color:#3b5446}
#run-day {background:#254e3c!important;color:#fff!important;border-radius:10px!important;min-height:46px!important;font-weight:650}
#control-panel {background:#fff;border:1px solid #dce3da;border-radius:16px;padding:20px!important}
@media(max-width:760px){.sb-cards,.sb-hour{grid-template-columns:repeat(2,minmax(0,1fr))}.sb-hero{padding:23px}.sb-card{padding:12px}.sb-card strong{font-size:24px}}
"""


def find_file(root, folders, filename):
    for folder in folders:
        p = root / folder / filename
        if p.exists():
            return p
    return root / folders[0] / filename


class Dashboard:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        csv = find_file(self.root, ("data", "Data"), "perth_hourly_weather_2021_2024.csv")
        if not csv.exists():
            raise FileNotFoundError(f"Weather CSV missing: {csv}. Run the training notebook's weather section first.")
        w = pd.read_csv(csv, parse_dates=["time"]).sort_values("time")
        if w["time"].duplicated().any() or w[["temp_c", "ghi_wm2"]].isna().any().any():
            raise ValueError("Weather must have unique timestamps and no missing measurements.")
        if not (w["time"].diff().dropna() == pd.Timedelta(hours=1)).all():
            raise ValueError("Weather must be hourly and consecutive before interval alignment.")
        # Preserve the training convention; disclose its perfect within-hour PV assumption.
        w["ghi_wm2"] = w["ghi_wm2"].shift(-1).fillna(0.0)
        w["date"] = w["time"].dt.normalize()
        all_days = sb.build_dayset(w)
        self.days = all_days.subset(np.where(all_days.dates >= "2023-10-01")[0])
        self.labels = [f"{d:%Y-%m-%d} · {d:%a} · {t:.1f}°C max · " +
                       ("TEST" if d >= pd.Timestamp("2024-01-01") else "VALIDATION")
                       for d, t in zip(self.days.dates, self.days.temp.max(1))]
        hot = np.where(self.days.weekday & (self.days.dates >= "2024-01-01"))[0]
        self.default = self.labels[int(hot[np.argmax(self.days.temp.max(1)[hot])])]
        self.policies, self.policy_info, self.load_messages = {}, {}, []
        self.policy_path = find_file(self.root, ("Results", "results"), "trained_policies.pt")
        if self.policy_path.exists():
            try:
                ckpt = torch.load(self.policy_path, map_location="cpu", weights_only=True)
                if ckpt.get("building_config") != asdict(sb.BASE_CFG):
                    raise ValueError("checkpoint building configuration differs from the simulator defaults")
                for name, d in ckpt["agents"].items():
                    if name not in ("DQN", "REINFORCE", "A2C"):
                        continue
                    net = sb.mlp(sb.SustainaBuildEnv.OBS_DIM, sb.N_ACTIONS,
                                 tuple(d["config"]["hidden"]), nn.ReLU if name == "DQN" else nn.Tanh)
                    net.load_state_dict(d["state_dict"])
                    net.eval()
                    self.policies[name] = sb.TorchPolicy(net, name)
                    self.policy_info[name] = f"seed {d['seed']}; validation objective {d['val_objective']:.2f}"
            except Exception as exc:
                self.policies.clear()
                self.policy_info.clear()
                self.load_messages.append(f"Checkpoint could not be loaded: {exc}")
        self.names = ["Rule-based", LOOK] + list(self.policies)
        self.evidence = None
        ep = self.root / "review" / "saved_colab_results.json"
        if ep.exists():
            self.evidence = json.loads(ep.read_text(encoding="utf-8"))
        cache = self.root / ".dashboard_cache"
        cache.mkdir(exist_ok=True)
        self.exports = TemporaryDirectory(prefix="replay_", dir=cache)
        self.weather_hash = hashlib.sha256(csv.read_bytes()).hexdigest()

    def controller(self, name, cfg):
        if name == "Rule-based":
            return sb.RuleBasedController()
        if name == LOOK:
            return sb.LookaheadController(cfg)
        return self.policies[name]

    def run(self, day_label, selected, focus, heat, sun, shift, battery, pv, limit):
        if day_label not in self.labels or focus not in self.names:
            raise gr.Error("Choose an available day and controller.")
        cfg = replace(sb.BASE_CFG, temp_offset=float(heat), solar_scale=float(sun),
                      schedule_shift=int(shift), batt_capacity=float(battery),
                      pv_kwp=float(pv), peak_limit=float(limit))
        if not (10 <= cfg.batt_capacity <= 120 and 0 <= cfg.pv_kwp <= 80 and
                8 <= cfg.peak_limit <= 30 and -4 <= cfg.temp_offset <= 6 and
                .2 <= cfg.solar_scale <= 1.2 and 0 <= cfg.schedule_shift <= 4):
            raise gr.Error("Scenario settings are outside the supported demonstration range.")
        selected = selected or []
        if any(n not in self.names for n in selected):
            raise gr.Error("A selected controller is not available in this session.")
        # Always include the reference and focused controller; no misleading unmatched comparison.
        names = [n for n in self.names if n in selected or n in (focus, "Rule-based")]
        day = self.days.subset([self.labels.index(day_label)])
        results = {n: sb.run_controller(self.controller(n, cfg), day, cfg, days=[0], keep_trace=True)
                   for n in names}
        payload = {"results": results, "cfg": cfg, "day": day, "focus": focus,
                   "day_label": day_label, "weather_sha256": self.weather_hash}
        return (self.cards(payload), self.insight(payload), self.plot(payload), self.cost_plot(payload),
                self.metrics(payload), self.plan(payload), self.hour(payload, 15), payload, 15, None,
                "Results match the settings shown. Changing a control requires a new replay.")

    def cards(self, p):
        r = p["results"][p["focus"]].iloc[0]
        ref = p["results"]["Rule-based"].iloc[0]
        values = [("Electricity bill", f"${r.energy_cost:.2f}", f"${r.energy_cost-ref.energy_cost:+.2f} vs rule"),
                  ("Occupied discomfort", f"{r.discomfort_degh:.2f}", f"°C·h · {int(r.violation_hours)} hourly-mean violations"),
                  ("Peak grid import", f"{r.peak_import_kw:.1f} kW", f"{r.exceed_kwh:.2f} kWh above limit"),
                  ("Weighted objective", f"{r.objective:.2f}", "Bill + model penalties + terminal adjustment")]
        cards = ''.join(f'<div class="sb-card"><span class="sb-label">{a}</span><strong>{b}</strong><small>{c}</small></div>' for a,b,c in values)
        c=p["cfg"]
        scenario=(f"Replay maximum {p['day'].temp[0].max()+c.temp_offset:.1f}°C · solar {c.solar_scale:.0%} · "
                  f"office shift {c.schedule_shift:+d}h · battery {c.batt_capacity:g} kWh · "
                  f"PV {c.pv_kwp:g} kWp · import threshold {c.peak_limit:g} kW")
        return f'<div class="sb-run">REPLAY / <b>{html.escape(p["focus"])}</b> / {html.escape(p["day_label"])}</div><div class="sb-run">{scenario}</div><div class="sb-cards">{cards}</div>'

    def insight(self, p):
        r, base = p["results"][p["focus"]].iloc[0], p["results"]["Rule-based"].iloc[0]
        names = p["results"]
        best = min(names, key=lambda n: names[n].iloc[0].objective)
        words = f"<b>{html.escape(best)}</b> has the lowest weighted objective among the controllers in this replay. "
        words += f"For {html.escape(p['focus'])}, the bill changes by ${r.energy_cost-base.energy_cost:+.2f} and occupied discomfort by {r.discomfort_degh-base.discomfort_degh:+.2f} °C·h versus the rule. "
        words += "This is one simulated day; use the experiment evidence for multi-day conclusions."
        if p["cfg"] != sb.BASE_CFG:
            words += " <b>Changed conditions:</b> policies are not retrained for these settings; equipment changes are outside the original experiments."
        return '<div class="sb-insight">'+words+'</div>'

    def plot(self, p):
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Indoor temperature · hourly mean", "Grid import & available solar", "Battery charge · end of hour", "Cooling electricity"), vertical_spacing=.19, horizontal_spacing=.09)
        c = p["cfg"]
        trace = next(iter(p["results"].values())).attrs["trace"]
        fig.add_hrect(y0=c.t_low, y1=c.t_high, fillcolor="#d3eac2", opacity=.4, line_width=0, row=1, col=1)
        fig.add_hline(y=c.peak_limit, line_dash="dot", line_color="#bf704d", row=1, col=2)
        for name, df in p["results"].items():
            tr = df.attrs["trace"]
            for field,row,col,mult in [("Ta",1,1,1),("p_import",1,2,1),("soc",2,1,100),("p_cool",2,2,1)]:
                x,y=tr.hour.tolist(),(tr[field]*mult).tolist()
                if field=="soc":
                    x,y=[0]+(tr.hour+1).tolist(),[100*c.soc_init]+y
                elif field in ("p_import","p_cool"):
                    x,y=x+[24],y+[y[-1]]
                fig.add_trace(go.Scatter(x=x, y=y, name=name,
                    legendgroup=name, showlegend=(row==1 and col==1),
                    line=dict(color=COLORS[name], width=3 if name==p["focus"] else 1.5,
                              shape="hv" if field in ("p_import","p_cool") else "linear"),
                    hovertemplate="%{x}:00 · %{y:.2f}<extra>%{fullData.name}</extra>"),row=row,col=col)
        fig.add_trace(go.Scatter(x=trace.hour.tolist(), y=trace.p_pv.tolist(),name="Available PV",line=dict(color="#d2a449",dash="dot")),row=1,col=2)
        for h in trace.loc[trace.occ>0,"hour"]:
            for row,col in [(1,1),(1,2),(2,1),(2,2)]:
                fig.add_vrect(x0=h,x1=h+1,fillcolor="#91a69a",opacity=.045,line_width=0,row=row,col=col)
        for row,col,title in [(1,1,"°C"),(1,2,"kW"),(2,1,"%"),(2,2,"kW")]:
            fig.update_yaxes(title_text=title, gridcolor="#edf0e9",row=row,col=col)
        fig.update_yaxes(range=[0,100],row=2,col=1)
        fig.update_xaxes(range=[0,24],dtick=6,tickvals=[0,6,12,18,24],ticktext=["00:00","06:00","12:00","18:00","24:00"],gridcolor="#edf0e9")
        fig.update_layout(height=590,margin=dict(l=48,r=20,t=50,b=65),paper_bgcolor="white",plot_bgcolor="white",font=dict(family="Arial",color="#3b5145",size=11),legend=dict(orientation="h",y=-.14,x=0),hovermode="x unified")
        return fig

    def cost_plot(self, p):
        fig=go.Figure(); names=list(p["results"]); c=p["cfg"]
        components=[("Electricity bill","energy_cost",1,"#376c56"),("Comfort penalty","discomfort_degh",c.lambda_d,"#d4ad65"),("Peak exceedance penalty","exceed_kwh",c.lambda_p,"#bd765b"),("Battery-use proxy","throughput_kwh",c.lambda_b,"#9b97b1"),("Terminal adjustment","terminal_adj",1,"#adbcaa")]
        for name,key,weight,color in components:
            fig.add_trace(go.Bar(name=name,x=names,y=[float(p['results'][n].iloc[0][key])*weight for n in names],marker_color=color))
        fig.update_layout(barmode="relative",height=340,paper_bgcolor="white",plot_bgcolor="white",margin=dict(l=45,r=10,t=20,b=90),font=dict(family="Arial",color="#3b5145",size=11),legend=dict(orientation="h",y=-.3),yaxis_title="Objective components ($-equivalent)")
        return fig

    def metrics(self, p):
        cols={"objective":"Objective", "energy_cost":"Bill ($)","discomfort_degh":"Discomfort (°C·h)","violation_hours":"Violation hours", "peak_import_kw":"Peak (kW)","exceed_kwh":"Above limit (kWh)","throughput_kwh":"Battery use (kWh)","final_soc":"Final SoC", "terminal_adj":"Terminal adjustment"}
        rows=[]
        for n,df in p["results"].items():
            rows.append({"Controller":n,**{label:round(float(df.iloc[0][key]),3) for key,label in cols.items()}})
        return pd.DataFrame(rows).sort_values("Objective")

    def plan(self, p):
        tr=p["results"][p["focus"]].attrs["trace"]
        cool,batt=sb.decode_action(tr.action.values)
        return pd.DataFrame({"Hour":[f"{h:02d}:00" for h in tr.hour],"Occupied":np.where(tr.occ>0,"Yes","No"),
          "Cooling requested":[("Off","Half","Full")[i] for i in cool],
          "Battery requested":[("Full discharge","Half discharge","Hold","Half charge","Full charge")[i] for i in batt],
          "Cooling actual (kW)":tr.p_cool.round(2),"Charge actual (kW)":tr.p_ch.round(2),"Discharge actual (kW)":tr.p_dis.round(2),
          "Indoor mean (°C)":tr.Ta.round(2),"End SoC (%)":(100*tr.soc).round(1),"Grid (kW)":tr.p_import.round(2),"PV (kW)":tr.p_pv.round(2),"Curtailed (kW)":tr.p_curtail.round(2)})

    def hour(self, p, h):
        if not p:
            return "Run a replay to inspect an hour."
        r=p["results"][p["focus"]].attrs["trace"].iloc[int(h)]
        vals=[("Solar available",f"{r.p_pv:.1f} kW"),("Building + cooling",f"{r.p_load+r.p_cool:.1f} kW"),("Battery to building",f"{r.p_dis-r.p_ch:+.1f} kW"),("Grid import",f"{r.p_import:.1f} kW")]
        return '<div class="sb-hour">'+''.join(f'<div>{k}<b>{v}</b></div>' for k,v in vals)+'</div>'+f'<div class="sb-run">{int(h):02d}:00 · {html.escape(p["focus"])} · Battery positive = discharge, negative = charge. Curtailed PV: {r.p_curtail:.1f} kW. Physical limits can reduce the requested battery action.</div>'

    def export(self, p):
        if not p:
            raise gr.Error("Run a replay before exporting.")
        from uuid import uuid4
        path=Path(self.exports.name)/f"sustainabuild_{p['day'].dates[0]:%Y%m%d}_{uuid4().hex[:8]}.zip"
        with ZipFile(path,"w") as z:
            z.writestr("metrics.csv",self.metrics(p).to_csv(index=False))
            for n,df in p["results"].items():
                z.writestr(n.replace(" ","_")+"_hourly.csv",df.attrs['trace'].to_csv(index=False))
            meta={"day":p["day_label"],"focus":p["focus"],"config":asdict(p["cfg"]),"weather_sha256":p["weather_sha256"],"controllers":list(p["results"]),"checkpoint_details":self.policy_info,"note":"Simulated building, historical reanalysis weather; model objective is not the electricity bill. Hourly irradiance is known over the current action interval."}
            z.writestr("scenario.json",json.dumps(meta,indent=2))
        return str(path)


def build_app(root=ROOT):
    d=Dashboard(root)
    with gr.Blocks(title="SustainaBuild · Building control lab", analytics_enabled=False) as app:
        gr.HTML('<div class="sb-head"><div class="sb-brand"><span class="sb-mark">S</span>SustainaBuild</div><div class="sb-eyebrow">Building control lab / Perth, Australia</div></div><div class="sb-hero"><div class="sb-eyebrow">Solar · storage · comfort</div><h1>A smarter day for<br>building energy.</h1><p>Replay historical weather. Compare control decisions. See exactly where electricity cost, comfort and peak demand trade off.</p></div><span class="sb-badge">Historical weather</span><span class="sb-badge">Simulated office</span><span class="sb-badge">24 hourly decisions</span>')
        if not d.policies:
            gr.HTML('<div class="sb-notice"><b>Conventional controllers available.</b> Trained RL weights are missing or could not be loaded. Add Results/trained_policies.pt and restart to enable DQN, REINFORCE and A2C. The evidence tab contains their saved Colab results.</div>')
        if d.load_messages:
            gr.Markdown("\n".join(d.load_messages))
        with gr.Tab("Day replay"):
            state=gr.State()
            with gr.Row(equal_height=False):
                with gr.Column(scale=1,min_width=285,elem_id="control-panel"):
                    gr.Markdown("### Set the scene\nA matched day for every controller.")
                    day=gr.Dropdown(d.labels,value=d.default,label="Historical day")
                    preset=gr.Radio(list(PRESETS),value="Standard day",label="Scenario preset")
                    focus=gr.Dropdown(d.names,value=LOOK,label="Inspect controller")
                    selected=gr.CheckboxGroup(d.names,value=d.names,label="Compare controllers")
                    with gr.Accordion("Weather & office hours",open=False):
                        heat=gr.Slider(-4,6,value=0,step=.5,label="Temperature change (°C)")
                        sun=gr.Slider(.2,1.2,value=1,step=.1,label="Solar multiplier")
                        shift=gr.Slider(0,4,value=0,step=1,label="Office hours shifted later (h)")
                    with gr.Accordion("Building equipment",open=False):
                        gr.Markdown("Equipment changes are exploratory: trained agents are not adapted to the new building.")
                        battery=gr.Slider(10,120,value=60,step=10,label="Battery capacity (kWh)")
                        pv=gr.Slider(0,80,value=40,step=5,label="Solar array (kWp)")
                        limit=gr.Slider(8,30,value=15,step=1,label="Peak import threshold (kW)")
                    run=gr.Button("Replay this day →",variant="primary",elem_id="run-day")
                    status=gr.Markdown("Run a replay to view results.")
                    gr.Markdown("The rule-based reference and inspected controller are always included. A preset resets the weather and schedule controls only.")
                with gr.Column(scale=3,min_width=350):
                    cards=gr.HTML();insight=gr.HTML()
                    with gr.Tab("Energy & comfort"):
                        chart=gr.Plot(show_label=False)
                        gr.Markdown("Green band: 22–25°C. Shading: occupied hours. Temperature is an hourly mean; battery charge is measured at the end of each hour.")
                    with gr.Tab("What makes up the objective?"):
                        costs=gr.Plot(show_label=False)
                        gr.Markdown("Only the electricity component is a simulated purchase bill. Comfort, peak exceedance and battery wear are modelling penalties. The terminal adjustment values remaining battery energy.")
                    with gr.Tab("Controller comparison"):
                        table=gr.Dataframe(interactive=False,label="This day · lower objective is better")
            with gr.Accordion("Inspect the hourly decisions",open=True):
                hour=gr.Slider(0,23,value=15,step=1,label="Hour of the day")
                flows=gr.HTML()
                plan=gr.Dataframe(interactive=False,label="Requested actions and actual power · inspected controller",max_height=350)
            with gr.Row():
                download=gr.Button("Export this replay")
                export_file=gr.File(label="Metrics, hourly traces & scenario settings",interactive=False)
        with gr.Tab("Experiment evidence"):
            gr.Markdown("### The full experiment, beyond one day\nRecorded outputs from the executed Colab notebook. These tables are not recalculated by the replay controls. RL entries use three training seeds; ± is the population standard deviation across seed means, not a confidence interval.\n\n**What the saved runs show:** the two-step lookahead has the lowest mean objective in E1, E2, high heat and low solar. RL improves on the fixed rule in mean objective, but does not establish superiority over predictive control. Cost and comfort have different rankings.")
            if d.evidence:
                for name,item in d.evidence["tables"].items():
                    label={"E3 heat and solar 1":"E3 · +4°C heat", "E3 heat and solar 3":"E3 · 50% solar"}.get(name,name)
                    with gr.Accordion(label,open=name in ("E1 standard","Scenario summary")):
                        gr.Dataframe(pd.DataFrame(item["rows"],columns=item["columns"]),interactive=False,show_label=False)
                gr.Markdown("Source: `"+d.evidence['source']+"`. The exported checkpoint selects a seed by validation performance; a single replay is not the three-seed experiment average.")
            else:
                gr.Markdown("Saved-output evidence has not been included in this copy. See the executed Colab notebook.")
        with gr.Tab("Model & data"):
            gr.Markdown("""### What is real, and what is modelled?

**Weather:** a frozen historical Open-Meteo extract for Perth. The archive uses reanalysis/model products; this is not a sensor stream from the building. Training: warm-season days in October 2021–April 2023. Validation: October–December 2023. Test: January–April 2024. Validation days influenced model selection.

**Building:** an uncalibrated two-node thermal model, synthetic office load and occupancy, assumed tariffs, simplified PV and battery operation. Each day starts from the same indoor temperatures and battery charge. There is no connected building or live control.

**Decisions:** three cooling levels × five battery levels = 15 joint actions. DQN, REINFORCE and A2C are separate implementations; the conventional comparators use fixed rules or two-hour search.

**Objective:** purchase bill + 2 × occupied degree-hours + 1 × kWh above the import limit + 0.02 × battery throughput + 0.15 × change in stored battery energy. Penalties and terminal valuation are assumptions, not verified utility charges. Comfort is calculated from the hourly mean temperature, which may hide brief excursions.

**Information limits:** the replay preserves the training convention in which current-hour PV is known for the entire action interval. This is an idealisation, not an instantaneous sensor observation. Lookahead also uses yesterday's next-hour weather; that forecast is not included in the RL agents' observation vector.

**Interpretation:** use this as a control-policy experiment and decision-support prototype. A lower simulated bill does not establish measured savings, carbon reduction, or safe deployment in a real building.

[Historical weather documentation](https://open-meteo.com/en/docs/historical-weather-api)
""")
            gr.Markdown("**Loaded policies:** " + ("; ".join(f"{k}: {v}" for k,v in d.policy_info.items()) or "None"))
        inputs=[day,selected,focus,heat,sun,shift,battery,pv,limit]
        outputs=[cards,insight,chart,costs,table,plan,flows,state,hour,export_file,status]
        preset.change(lambda name:PRESETS[name],preset,[heat,sun,shift]).then(lambda:"Settings changed. Replay the day to update the results.",outputs=status)
        for control in inputs:
            control.change(lambda:"Settings changed. Displayed results still describe the previous replay.",outputs=status)
        run.click(d.run,inputs,outputs)
        app.load(d.run,inputs,outputs)
        hour.change(d.hour,[state,hour],flows)
        download.click(d.export,state,export_file)
    app.sustainabuild_dashboard=d
    return app


def launch(root=ROOT, share=False, **kwargs):
    app=build_app(root)
    theme=gr.themes.Soft(primary_hue="green",neutral_hue="stone",font=["Arial","sans-serif"])
    # Keep the editorial light palette readable even when the host browser prefers dark mode.
    palette={"body_background_fill":"#f7f8f4", "body_text_color":"#203b33",
             "body_text_color_subdued":"#647569", "background_fill_primary":"#ffffff",
             "background_fill_secondary":"#eef3e9", "border_color_primary":"#dce3da",
             "border_color_accent":"#63876c", "block_background_fill":"#ffffff",
             "block_border_color":"#dce3da", "block_label_background_fill":"#ffffff",
             "block_label_text_color":"#425c4d", "block_title_text_color":"#203b33",
             "block_title_background_fill":"#ffffff", "block_info_text_color":"#647569",
             "panel_background_fill":"#f7f8f4", "panel_border_color":"#dce3da",
             "input_background_fill":"#f5f7f2", "input_background_fill_focus":"#ffffff",
             "input_border_color":"#dce3da", "accordion_text_color":"#203b33",
             "checkbox_label_background_fill":"#f7f8f4", "checkbox_label_text_color":"#425c4d",
             "checkbox_label_background_fill_selected":"#e1edd8", "checkbox_label_text_color_selected":"#244e3d",
             "checkbox_label_border_color":"#dce3da", "checkbox_label_border_color_selected":"#78946a",
             "table_text_color":"#203b33", "table_even_background_fill":"#f7f8f4",
             "table_odd_background_fill":"#ffffff", "table_border_color":"#dce3da",
             "button_secondary_background_fill":"#edf3e9", "button_secondary_text_color":"#244e3d"}
    theme.set(**{key+suffix:value for key,value in palette.items() for suffix in ("","_dark")})
    # Gradio 6 accepts CSS and theme at launch rather than construction.
    return app.launch(share=share,css=CSS,theme=theme,**kwargs)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--project-dir",type=Path,default=ROOT)
    parser.add_argument("--port",type=int,default=7860)
    parser.add_argument("--share",action="store_true",help="Explicitly create a public Gradio share link")
    args=parser.parse_args()
    launch(args.project_dir,share=args.share,server_name="127.0.0.1",server_port=args.port)
