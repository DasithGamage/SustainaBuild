"""Review-only verification. Re-evaluates supplied weights; does not train agents."""
import copy
import json
import pickle
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sustainabuild_core as sb
from sustainabuild_dashboard import Dashboard

sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(2)


def verify():
    app = Dashboard(ROOT)
    assert set(app.policies) == {"DQN", "REINFORCE", "A2C"}, app.load_messages
    print("All three exported policies loaded:", app.policy_info, flush=True)
    summary = {"trained_from_scratch": False, "policies": app.policy_info, "checks": []}
    for pv, capacity in [(40,60),(0,60),(80,10),(5,120)]:
        cfg = replace(sb.BASE_CFG, pv_kwp=pv, batt_capacity=capacity)
        day = app.days.subset([app.labels.index(app.default)])
        for name in app.names:
            env = sb.SustainaBuildEnv(day, cfg)
            obs,_ = env.reset(options={"day":0})
            for h in range(24):
                assert np.isfinite(obs).all(), (pv, capacity, name, h)
                action = app.controller(name,cfg).act(obs,env)
                before = env.E
                obs,r,done,_,info = env.step(action)
                assert np.isfinite(r)
                assert np.isclose(env.E, before+cfg.eta_c*info['p_ch']-info['p_dis']/cfg.eta_d)
                assert cfg.soc_min-1e-8 <= info['soc'] <= cfg.soc_max+1e-8
                assert np.isclose(info['p_import']+info['p_pv']+info['p_dis'], info['p_load']+info['p_cool']+info['p_ch']+info['p_curtail'])
                assert done == (h==23)
    summary['checks'].append("Finite observations, battery bounds, energy conservation and 24-step termination: all 5 controllers at 4 equipment settings")

    # Confirm the test formerly labelled 'free float' used half cooling.
    true, old = [], []
    for action,target in [(2,true),(7,old)]:
        env = sb.SustainaBuildEnv(day)
        env.reset(options={"day":0})
        for h in range(24):
            env.step(action);target.append(env.Ta)
    summary['free_float_hottest_weekday'] = dict(date=str(day.dates[0].date()),
        corrected_peak_c=max(true), previous_half_cooling_peak_c=max(old))

    # Reconstruct five-minute temperatures, without altering the stored training reward.
    comfort=[]
    for name in app.names:
        cfg=sb.BASE_CFG;env=sb.SustainaBuildEnv(day);obs,_=env.reset(options={"day":0})
        hourly=integrated=0.
        for h in range(24):
            a=app.controller(name,cfg).act(obs,env);cool,batt=sb.decode_action(a)
            Ta,Tm=env.Ta,env.Tm;outside,ghi=env.current_weather();occ,load,_=env.known_schedule(h)
            dt=cfg.dt/cfg.n_substeps
            for _ in range(cfg.n_substeps):
                qoa=(outside-Ta)/cfg.R_oa;qma=(Tm-Ta)/cfg.R_am
                gain=cfg.occupant_gain*occ+cfg.load_heat_frac*load+cfg.solar_aperture*ghi/1000
                Ta,Tm=Ta+dt/cfg.C_a*(qoa+qma+gain-cfg.cop*sb.COOL_LEVELS[cool]*cfg.p_cool_max),Tm-dt/cfg.C_m*qma
                integrated += occ*(max(0,Ta-cfg.t_high)+max(0,cfg.t_low-Ta))*dt
            obs,_,_,_,info=env.step(a);hourly+=info['discomfort']
            assert np.isclose(Ta,env.Ta) and np.isclose(Tm,env.Tm)
        comfort.append(dict(controller=name,hourly_mean_degree_hours=hourly,integrated_degree_hours=integrated))
    summary['comfort_diagnostic']=comfort

    # Restore the supplied cache classes through their own source definitions.
    # These are user-provided training files; never use this on untrusted pickle files.
    notebook=json.loads((ROOT/'ColabNoteBooks/SustainaBuild_RL_Group10.ipynb').read_text(encoding='utf-8'))
    globals().update({k:v for k,v in vars(sb).items() if not k.startswith('_')})
    for i in [43,45,47,49]:
        exec(compile(''.join(notebook['cells'][i]['source']),f'notebook_cell_{i}','exec'),globals())
    saved=ROOT/'Google Drive/RL_Assignment/Results'
    policies={}
    for name in ('DQN','REINFORCE','A2C'):
        policies[name]=[]
        for seed in range(3):
            with (saved/f'final_{name}_s{seed}.pkl').open('rb') as f:
                record=pickle.load(f)
            net=sb.mlp(14,15,record['cfg'].hidden, nn.ReLU if name=='DQN' else nn.Tanh)
            net.load_state_dict(record['log'].best_state);net.eval()
            policies[name].append(sb.TorchPolicy(net,name))
    test=app.days.subset(np.where(app.days.dates>='2024-01-01')[0])
    scenarios={'E1_test_days':sb.BASE_CFG,'E2_shifted_schedule':replace(sb.BASE_CFG,schedule_shift=2),
        'E3_high_heat':replace(sb.BASE_CFG,temp_offset=4), 'E3_low_solar':replace(sb.BASE_CFG,solar_scale=.5)}
    all_rows=[]
    for scenario,cfg in scenarios.items():
        expected=pd.read_csv(saved/(scenario+'.csv'),index_col=0)
        for name in app.names:
            controllers=policies[name] if name in policies else [app.controller(name,cfg)]
            frames=[sb.run_controller(c,test,cfg) for c in controllers]
            actual=pd.DataFrame([sb.summarise(f) for f in frames]).mean()
            # Published tables round to two decimals. Allow half a last digit plus floating error.
            for metric in sb.summarise(frames[0]).index:
                value=float(str(expected.loc[name,metric]).split('±')[0].strip())
                assert abs(actual[metric]-value)<=.0051,(scenario,name,metric,actual[metric],value)
            all_rows.append({'scenario':scenario,'controller':name,**actual.to_dict()})
        print('Verified saved table:',scenario,flush=True)
    summary['checks'].append('All four saved scenario tables reproduced to displayed precision across all nine final RL checkpoints and both conventional controllers; latency excluded')
    pd.DataFrame(all_rows).to_csv(ROOT/'review/recomputed_scenario_means.csv',index=False)

    # Exercise the dashboard callback and downloadable record using actual model weights.
    output=app.run(app.default,app.names,'DQN',0,1,0,60,40,15)
    payload=output[7]
    assert len(output[4])==5 and len(output[5])==24
    assert Path(app.export(payload)).exists()
    app.run(app.default,[], 'A2C',4,.5,2,10,0,8)
    summary['checks'].append('Dashboard full replay, empty comparison selection with mandatory reference/focus, combined stress/zero-PV replay, and ZIP export')
    (ROOT/'review/verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    verify()
