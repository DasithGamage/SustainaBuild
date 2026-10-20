"""Integration checks for real replay, manual state and policy handover."""
import sys
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import sustainabuild_web as web

client=TestClient(web.app)
scenario=dict(date="2024-02-19",heat=0,solar=1,shift=0,battery=60,pv=40,limit=15)


def post(path,body):
    response=client.post(path,json=body)
    assert response.status_code==200,response.text
    return response.json()


def verify():
    assert client.get('/').status_code==200
    assert client.get('/assets/app.js').status_code==200
    assert len(client.get('/api/meta').json()['controllers'])==5
    replay=post('/api/replay',scenario)
    for name,result in replay['results'].items():
        assert len(result['trace'])==24
        assert result['metrics']['completed']
        day=web.day_for(web.Scenario(**scenario))
        expected=web.sb.run_controller(web.get_controller(name,web.sb.BASE_CFG),day,days=[0])
        assert np.isclose(result['metrics']['objective'],expected.iloc[0].objective)
    print('All five web replay results match the existing backend.',flush=True)

    for name in replay['results']:
        session=post('/api/manual',scenario)
        finished=post(f"/api/manual/{session['session_id']}/finish",dict(expected_hour=0,controller=name))
        assert finished['done']
        assert np.isclose(finished['metrics']['objective'],replay['results'][name]['metrics']['objective'])
        assert [x['action'] for x in finished['trace']]==[x['action'] for x in replay['results'][name]['trace']]
    print('Manual sessions handed entirely to each policy reproduce its automatic day.',flush=True)

    session=post('/api/manual',scenario);sid=session['session_id']
    step=post(f'/api/manual/{sid}/step',dict(expected_hour=0,action=14))
    assert step['current']['hour']==1 and step['trace'][0]['actor']=='You'
    assert step['trace'][0]['p_cool']==12
    assert client.post(f'/api/manual/{sid}/step',json=dict(expected_hour=0,action=2)).status_code==409
    undo=post(f'/api/manual/{sid}/undo',dict(expected_hour=1))
    assert undo['current']==session['current'] and undo['trace']==[]
    # Different sessions cannot accidentally overwrite each other's building state.
    other=post('/api/manual',scenario)
    post(f'/api/manual/{sid}/step',dict(expected_hour=0,action=14))
    assert client.get(f"/api/manual/{other['session_id']}").json()['current']['hour']==0
    mixed=post(f'/api/manual/{sid}/finish',dict(expected_hour=1,controller='DQN'))
    assert mixed['trace'][0]['actor']=='You'
    assert all(x['actor']=='DQN' for x in mixed['trace'][1:])
    assert len(mixed['trace'])==24
    for tr in mixed['trace']:
        assert np.isclose(tr['p_import']+tr['p_pv']+tr['p_dis'],tr['p_load']+tr['p_cool']+tr['p_ch']+tr['p_curtail'])
    undone=post(f'/api/manual/{sid}/undo',dict(expected_hour=24))
    assert not undone['done'] and undone['metrics']['terminal_adj']==0
    assert client.post('/api/replay',json={**scenario,'date':'2000-01-01'}).status_code==422
    assert client.post('/api/replay',json={**scenario,'battery':0}).status_code==422
    assert client.post(f'/api/manual/{sid}/step',json={'expected_hour':23,'action':16}).status_code==422
    assert client.post(f'/api/manual/{sid}/step',json={'expected_hour':23,'action':2,'controller':'DQN'}).status_code==422
    assert client.post(f'/api/manual/{sid}/finish',json={'expected_hour':23,'controller':'missing'}).status_code==422
    zero=post('/api/replay',{**scenario,'pv':0,'battery':10,'heat':4,'solar':.5,'shift':2})
    assert all(np.isfinite(x['metrics']['objective']) for x in zero['results'].values())
    assert client.get('/api/evidence').json()['tables']['E1 standard']['rows'][0][0]=='Rule-based'
    assert len(client.get('/api/climate').json()['months'])==7
    print('Manual actions, undo, stale-action protection, session isolation, handover, energy balance, validation and zero-PV stress passed.',flush=True)


if __name__=='__main__':verify()
