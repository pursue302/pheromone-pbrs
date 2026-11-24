"""Causal control: call the ACTUAL released learn(), reset traces only.
No rewrite of its transition, action selection, reward, or numerical update.
"""
from study import *

def one(args):
    out,seed=args;env,agent,raw,features,rng=paper.setup('historical',seed,'plain')
    times=[]
    from unittest.mock import patch
    with patch.object(np.random,'random',rng.random_sample):
        for _ in range(200):
            agent.tile_code_trace_weights.fill(0.)
            t=time.perf_counter();agent.learn(env);times.append(time.perf_counter()-t)
    metrics=paper.find_best_results([times],[agent.return_last_steps()])
    result=dict(method='actual_source_reset_only',seed=seed,steps=env.total_steps,episodes=env.episodes,
        final=paper.evaluate(agent,'historical',seed),author_metrics={k:float(v[0]) for k,v in metrics.items()},
        episode_seconds=times,finite=bool(np.isfinite(agent.tile_code_weights).all()),backend='actual_author_learn_with_trace_reset')
    result['final']['success_rate']=result['final']['success']
    save(out,result['method'],seed,result,dict(weights=agent.tile_code_weights,traces=agent.tile_code_trace_weights))
    return dict(seed=seed,steps=env.total_steps)

if __name__=='__main__':
    out=HERE/'direct_reset';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('immutable results')
    m=dict(status='running',seeds=100,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(m,indent=2));start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=6) as pool:
        rows=[]
        for i,r in enumerate(pool.map(one,[(str(out),s) for s in range(20000,20100)]),1):
            rows.append(r)
            if i%10==0:print(json.dumps(dict(completed=i,total=100,seconds=time.perf_counter()-start)),flush=True)
    m.update(status='complete',steps=sum(x['steps'] for x in rows),seconds=time.perf_counter()-start)
    (out/'manifest.json').write_text(json.dumps(m,indent=2))
