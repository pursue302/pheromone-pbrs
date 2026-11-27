from pathlib import Path
import time
import numpy as np
import run_replication as paper
from historical import Historic
from sarsa import evaluate, update

ROOT = Path(__file__).resolve().parent.parent
old = paper.numeric

def run_repair(job, save):
    out,method,seed=job;started=time.perf_counter()
    role='plain';prior=None
    if method.startswith('static'):
        role='static';prior,_=paper.load_prior(str(ROOT/'02_source_audit/results/priors/historical_A__mature.pkl'))
    env,agent,raw,features,rng=paper.setup('historical',seed,role,'frozen',prior)
    w,z=agent.tile_code_weights,agent.tile_code_trace_weights
    canonical=method in ('canonical','static_canonical')
    reset=method!='literal';eps=.025;times=[];episodes=[]
    from gym_legacy.compat import spaces
    def decision(obs):
        ids=features(obs);q=old.values(w,ids)
        chance=rng.random_sample()
        a=int(np.argmax(q)) if chance>eps else int(spaces.np_random.randint(3))
        phi=float(old.values(agent.pheromone_trace,ids)[a]/10) if role=='static' else 0.
        return a,ids,q,phi
    for episode in range(200):
        before=time.perf_counter();obs=env.reset()
        if reset:z.fill(0.)
        a,ids,q,p=decision(obs);qold=0.
        for t in range(1000):
            eps*=.99
            sp,r,done,_=env.step(a);ap,ids2,q2,p2=decision(sp)
            # Repair arms always use actual initial Phi and zero endpoint Phi.
            if done:p2=0.
            future=0. if done else q2[ap]
            update(w,z,ids[a],a,agent.alpha,1.,.9,r+p2-p+future-q[a],q[a]-qold,canonical)
            qold=future;p=p2;a=ap;ids=ids2;obs=sp;q=old.values(w,ids)
            if done:break
        times.append(time.perf_counter()-before);episodes.append(dict(env.episodes[-1]))
    final=evaluate(lambda:Historic(seed+918273),features,w,10)
    length=np.array([e['length']-1 for e in episodes]);ix=int(length.argmin())
    result=dict(method=method,seed=seed,episodes=episodes,steps=env.total_steps,episode_seconds=times,
        author_metrics=dict(values=float(length[ix]),indexes=ix,times=sum(times[:ix]),execution_times=sum(times)),
        final=final,wall_seconds=time.perf_counter()-started,finite=bool(np.isfinite(w).all()))
    save(out,method,seed,result,dict(weights=w,traces=z))
    return dict(method=method,seed=seed,steps=env.total_steps,final=final)

