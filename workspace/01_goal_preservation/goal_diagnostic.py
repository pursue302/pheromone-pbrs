"""Exact small goal-preservation diagnostic, not a training benchmark."""
from pathlib import Path
import json
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'environments'))
from gridworld import GridWorld
HERE=Path(__file__).resolve().parent

def solve(env,beta,kind,cap=40,gamma=.99):
    n=env.n_states;h=np.zeros(n);h[env._rc_to_s(4,2)]=1.
    v=np.zeros((cap+1,n));policy=np.zeros((cap,n),dtype=int)
    for t in range(cap-1,-1,-1):
        for s in range(n):
            q=np.zeros(4)
            for a in range(4):
                for p,sp,r,term in env.P[s][a]:
                    end=term or t==cap-1
                    next_h=0. if end else h[sp]
                    f=0. if kind=='plain' else beta*h[sp] if kind=='naive' else beta*(gamma*next_h-h[s])
                    q[a]+=p*(r+f+(0. if end else gamma*v[t+1,sp]))
            policy[t,s]=int(q.argmax());v[t,s]=q.max()
    s=env.start_state;trajectory=[env.s_to_rc(s)];original=shaped=shaping=0.;success=False
    for t in range(cap):
        a=int(policy[t,s]);ts=env.P[s][a];assert len(ts)==1
        p,sp,r,term=ts[0];assert p==1.
        end=term or t==cap-1;next_h=0. if end else h[sp]
        f=0. if kind=='plain' else beta*h[sp] if kind=='naive' else beta*(gamma*next_h-h[s])
        original+=gamma**t*r;shaped+=gamma**t*(r+f);shaping+=gamma**t*f
        trajectory.append(env.s_to_rc(sp));s=sp
        if end:success=bool(term);break
    return dict(beta=beta,kind=kind,original_discounted_return=original,shaped_return=shaped,
        success=success,length=t+1,shaping_return=shaping,trajectory=trajectory),v

def main():
    layout=['......G','.......','.......','.......','S......']
    env=GridWorld(layout,slip=0.,step_reward=0.,goal_reward=1.,continuing=False)
    result=[];plain,reference=solve(env,0.,'plain');max_error=0.
    h=np.zeros(env.n_states);h[env._rc_to_s(4,2)]=1.
    for beta in [0.,.01,.03,.05,.1,.3,1.,3.,10.]:
        for kind in ('naive','pbrs'):
            r,v=solve(env,beta,kind);result.append(r)
            if kind=='pbrs':
                error=float(np.max(abs(v[:-1]-reference[:-1]+beta*h)))
                max_error=max(max_error,error)
                assert abs(r['original_discounted_return']-plain['original_discounted_return'])<1e-10
                assert r['success'] and abs(r['shaping_return'])<1e-10
    assert max_error<1e-10
    assert any(not r['success'] for r in result if r['kind']=='naive')
    out=dict(layout=layout,cap=40,gamma=.99,tempting_cell=[4,2],baseline=plain,
        results=result,max_value_shift_error=max_error,
        scope='Exact finite-horizon model planning; no training, no empirical sample-efficiency or novelty claim.')
    (HERE/'goal_diagnostic.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    print(json.dumps(dict(baseline=plain['original_discounted_return'],max_error=max_error,
        outcomes=[(r['kind'],r['beta'],round(r['original_discounted_return'],3),r['success']) for r in result])))

if __name__=='__main__':main()
