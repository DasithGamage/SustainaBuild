"""Export everything the in-browser engine (frontend/engine.js) needs.

  python export_web_engine.py <destination folder>              -> index.html + assets/ (static site)
  python export_web_engine.py <destination folder> --reference  -> also writes reference.json for check_engine.mjs

The browser engine is a line-by-line JavaScript port of sustainabuild_core.py. This script supplies
its inputs (the same weather days, configuration and trained weights the Python app uses) and,
with --reference, Python outputs for every day so the port can be checked against the original.
"""
import argparse
import base64
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import sustainabuild_web as web

sb = web.sb
FILES = ("app.js", "experience.js", "engine.js", "style.css", "energy.css", "logo.svg")


def network(name, policy):
    """Linear layers of the saved policy as base64 float32 (exact weights) plus the activation."""
    layers = []
    for module in policy.net:
        if isinstance(module, torch.nn.Linear):
            w = module.weight.detach().numpy().astype("<f4")
            b = module.bias.detach().numpy().astype("<f4")
            layers.append(dict(rows=int(w.shape[0]), cols=int(w.shape[1]),
                               W=base64.b64encode(w.tobytes()).decode(),
                               b=base64.b64encode(b.tobytes()).decode()))
    return dict(activation="relu" if name == "DQN" else "tanh", layers=layers)


def engine_data():
    days = web.service.days
    return dict(
        meta=web.meta(),
        evidence=web.evidence(),
        climate=web.climate(),
        base_config=asdict(sb.BASE_CFG),
        days=[dict(date=str(d.date()), weekday=bool(wd), temp=t.tolist(), ghi=g.tolist(),
                   prev_temp=pt.tolist(), prev_ghi=pg.tolist())
              for d, wd, t, g, pt, pg in zip(days.dates, days.weekday, days.temp, days.ghi,
                                             days.prev_temp, days.prev_ghi)],
        policies={name: network(name, web.service.policies[name]) for name in web.service.policies},
        policy_info=web.service.policy_info,
        weather_sha256=web.service.weather_hash,
    )


SCENARIOS = [dict(), dict(heat=4), dict(solar=.5), dict(shift=2),
             dict(heat=-2.5, solar=1.2, battery=110, pv=0, limit=9),
             dict(heat=6, solar=.2, shift=4, battery=10, pv=80, limit=30)]


def reference():
    """Python results for every available day under several scenarios (actions and key metrics)."""
    out = []
    dates = [str(d.date()) for d in web.service.days.dates]
    for change in SCENARIOS:
        for date in dates:
            r = web.replay(web.Scenario(date=date, **change))
            out.append(dict(scenario=r["scenario"], results={
                n: dict(actions=[row["action"] for row in v["trace"]],
                        objective=v["metrics"]["objective"], energy_cost=v["metrics"]["energy_cost"],
                        Ta=[row["Ta"] for row in v["trace"]], soc=[row["soc"] for row in v["trace"]])
                for n, v in r["results"].items()}))
    return out


def export(destination: Path, with_reference=False):
    assets = destination / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    frontend = web.ROOT / "frontend"
    for name in FILES:
        shutil.copyfile(frontend / name, assets / name)
    shutil.copyfile(frontend / "index.html", destination / "index.html")
    shutil.copytree(frontend / "research", assets / "research", dirs_exist_ok=True)
    (assets / "engine-data.json").write_text(json.dumps(engine_data(), allow_nan=False), encoding="utf-8")
    (assets / "runtime.js").write_text(
        '// Static build: the simulator and saved policies run in the browser (assets/engine.js).\n'
        'window.SB_DEMO = null;\n'
        'window.SB_ENGINE_DATA = "assets/engine-data.json";\n', encoding="utf-8")
    old = assets / "recorded.json"
    if old.exists():
        old.unlink()
    if with_reference:
        (destination / "reference.json").write_text(json.dumps(reference(), allow_nan=False), encoding="utf-8")
    print("exported static engine build to", destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    export(args.destination, args.reference)
