"""Export real simulator replays for the static portfolio demonstration.

The exported UI cannot perform new manual actions without the Python backend.
It labels this boundary and only offers scenarios present in recorded.json.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import sustainabuild_web as web


def export(destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    assets = destination / "assets"
    assets.mkdir(exist_ok=True)
    frontend = web.ROOT / "frontend"
    for name in ("app.js", "experience.js", "style.css", "energy.css", "logo.svg"):
        shutil.copyfile(frontend / name, assets / name)
    shutil.copyfile(frontend / "index.html", destination / "index.html")
    shutil.copytree(frontend / "research", assets / "research", dirs_exist_ok=True)
    dates = ("2024-01-18", "2024-02-19")
    meta = web.meta()
    meta["days"] = [d for d in meta["days"] if d["date"] in dates]
    scenarios = ({}, {"heat": 4}, {"solar": .5}, {"shift": 2})
    replays = []
    for date in dates:
        for change in scenarios:
            result = web.replay(web.Scenario(date=date, **change))
            assert all(len(r["trace"]) == 24 for r in result["results"].values())
            replays.append(result)
    data = {
        "mode": "recorded_python_outputs",
        "meta": meta,
        "evidence": web.evidence(),
        "climate": web.climate(),
        "replays": replays,
        "provenance": {
            "generator": "export_portfolio.py calls sustainabuild_web.replay directly",
            "core_sha256": hashlib.sha256((web.ROOT / "sustainabuild_core.py").read_bytes()).hexdigest(),
            "weights_sha256": hashlib.sha256((web.ROOT / "results/trained_policies.pt").read_bytes()).hexdigest(),
            "limitation": "Snapshots, not a browser reimplementation or live building connection.",
        },
    }
    (assets / "recorded.json").write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    (assets / "runtime.js").write_text(
        'window.SB_DEMO = {data:"assets/recorded.json",'
        'repo:"https://github.com/DasithGamage/SustainaBuild"};\n', encoding="utf-8"
    )
    print(f"Exported {len(replays)} real scenario replays to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    export(parser.parse_args().destination)
