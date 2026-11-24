"""Paper-near replication through actual full author learn() methods."""
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
from unittest.mock import patch
import argparse
import contextlib
import copy
import hashlib
import importlib
import io
import json
import os
import pickle
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'pheromone_feedback_2019'))
import experiment as numeric
from gym_legacy.mountain_car import MountainCarEnv
from gym_legacy.compat import spaces
from paper_author.ApproximatedSarsaLambdaAgent import ApproximatedSarsaLambdaAgent as Plain
from paper_author.HAApproximatedSarsaLambdaAgent import HAApproximatedSarsaLambdaAgent as HA
from author_metrics import find_best_results

class Environment:
    def __init__(self,kind,seed,record_steps=False):
        self.kind=kind;self.key='mountain_car';self.cap=self._max_episode_steps=1000
        self.gamma=1.;self.learning_rate=.5
        self.base=MountainCarEnv();self.base.seed(int(seed));self.base.reset()
        if kind=='paper_text':self.base.goal_position=.6
        self.low=self.base.observation_space.low;self.high=self.base.observation_space.high
        self.actions=self.base.action_space;self.action_space=self.actions
        self.record_steps=record_steps;self.data=[];self.episodes=[];self.total_steps=0
        self.last_success=False;self._phase=0
    def reset(self,position=None):
        obs=self.base.reset()
        if position is not None:self.base.state=np.array([position,0.],dtype=float);obs=self.base.state.copy()
        self.t=0;self.last_success=False;self.reward_total=0.;self.current_obs=obs.copy();self.start_obs=obs.copy()
        return obs
    def step(self,action):
        obs,reward,terminated,info=self.base.step(action)
        if self.kind=='paper_text' and terminated:reward=1.
        self.t+=1;self.total_steps+=1;self.last_success=terminated;self.reward_total+=reward
        done=bool(terminated or self.t>=self.cap)
        if self.record_steps:self.data.append((self._phase,self.current_obs.copy(),int(action),float(reward),obs.copy(),done))
        self.current_obs=obs.copy()
        if done:self.episodes.append(dict(length=self.t,success=bool(terminated),original_return=float(self.reward_total),
            cumulative_steps=self.total_steps,start_observation=self.start_obs.tolist()))
        return obs,reward,done,info


def config(feature_seed,role):
    return dict(Strategy='TrueOnline',decrease_exploration=True,decrease_exploration_rate=.99,
        learning_rate=.5,eps=.025,**{'lambda':.9},discount=1.,n_iter=1000,
        tile_seed=int(feature_seed),rho=.99,psi=.3,nu=10,Pheromone_strategy='hard',
        heuristic_dynamic=role not in ('static','static_flag'),model_based=False,model=None)


def setup(kind,seed,role,version='source',prior=None,same_tiles=False,record=False,fast=True):
    env=Environment(kind,seed+451,record)
    conf=config(seed,role)
    cls=HA if role=='donor' else (Plain if role=='plain' else importlib.import_module('paper_author.pa_'+version).PAApproximatedSarsaLambdaAgent)
    with contextlib.redirect_stdout(io.StringIO()):agent=cls(env.low,env.high,env.actions,[10,10],[10],my_config=conf)
    if prior is not None:
        coders=copy.deepcopy(prior['coders'])
        if hasattr(agent,'set_tilecoder'):agent.set_tilecoder(coders)
        else:agent.tile_coder=coders
        if not same_tiles:agent.set_pheromone_trace(prior['tau'].copy())
    raw=copy.deepcopy(agent.tile_coder)
    feature=numeric.Features(agent,env)
    if fast:agent.tile_coder={a:numeric.FastCoder(feature,a) for a in range(3)}
    spaces.np_random.seed((int(seed)+800081)%2**32)
    rng=np.random.RandomState((int(seed)+2027)%2**32)
    return env,agent,raw,feature,rng


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def evaluate(agent,kind,seed):
    env=Environment(kind,seed+918273);returns=[];lengths=[];success=[]
    with patch.object(np.random,'random',lambda:1.):
        for pos in np.linspace(-.6,-.4,10):
            obs=env.reset(float(pos));total=0.
            for _ in range(env.cap):
                obs,reward,done,_=env.step(agent.act(obs,eps=0.));total+=reward
                if done:break
            returns.append(total);lengths.append(env.t);success.append(env.last_success)
    return dict(original_return=float(np.mean(returns)),success=float(np.mean(success)),
        steps=float(np.mean(lengths)),per_start_returns=returns)


def train_episodes(env,agent,rng,count):
    times=[]
    with patch.object(np.random,'random',rng.random_sample):
        for _ in range(count):
            before=len(env.episodes);t=time.perf_counter();agent.learn(env)
            times.append(time.perf_counter()-t)
            assert len(env.episodes)==before+1
    return times


def arrays(agent):
    out=dict(weights=agent.tile_code_weights.copy(),traces=agent.tile_code_trace_weights.copy())
    if hasattr(agent,'pheromone_trace'):out['tau']=agent.pheromone_trace.copy()
    return out


def export_prior(folder,bank,stage,agent,raw,episodes,env,times,stable):
    payload=dict(tau=agent.pheromone_trace.copy(),coders=copy.deepcopy(raw))
    stem=f'{bank}__{stage}'
    with (folder/'priors'/f'{stem}.pkl').open('wb') as f:pickle.dump(payload,f,protocol=pickle.HIGHEST_PROTOCOL)
    meta=dict(bank=bank,stage=stage,episodes=episodes,steps=env.total_steps,training_seconds=sum(times),stable=stable,
        tau_sha256=digest(payload['tau']),coder_sha256=hashlib.sha256(pickle.dumps(raw)).hexdigest(),
        best_actual_length=min(e['length'] for e in env.episodes),last100_success=float(np.mean([e['success'] for e in env.episodes[-100:]])),
        last100_mean_length=float(np.mean([e['length'] for e in env.episodes[-100:]])))
    (folder/'priors'/f'{stem}.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    return meta


def teacher_job(args):
    path,bank,kind,seed,smoke=args;folder=Path(path)
    env,agent,raw,features,rng=setup(kind,seed,'donor',record=False)
    times=[];checks=[];stable=False;limit=6 if smoke else 2000;minimum=6 if smoke else 1000
    checkpoint=3 if smoke else 200
    for end in range(checkpoint,limit+1,checkpoint):
        times.extend(train_episodes(env,agent,rng,checkpoint))
        if end==checkpoint:export_prior(folder,bank,'e200',agent,raw,end,env,times,False)
        if end>=minimum:
            e=env.episodes;last=e[-200:];prior=e[-200:-100];recent=e[-100:]
            previous_best=min(x['length'] for x in e[:-200]) if len(e)>200 else min(x['length'] for x in e)
            new_best=min(x['length'] for x in last)
            old_mean=np.mean([x['length'] for x in prior]) if prior else np.mean([x['length'] for x in recent])
            new_mean=np.mean([x['length'] for x in recent])
            stable=bool(np.mean([x['success'] for x in last])>=.99 and new_best>=previous_best and abs(new_mean-old_mean)<=.05*old_mean)
            checks.append(dict(episodes=end,stable=stable,previous_best=previous_best,recent_best=new_best,previous100_mean=float(old_mean),recent100_mean=float(new_mean)))
        print(json.dumps(dict(teacher=bank,episodes=end,stable=stable)),flush=True)
        if end>=minimum and (stable or smoke):break
    meta=export_prior(folder,bank,'mature',agent,raw,len(env.episodes),env,times,stable)
    meta.update(episodes_data=env.episodes,author_last_steps=list(map(float,agent.return_last_steps())),episode_seconds=times,stability_checks=checks,
        final_greedy=evaluate(agent,kind,seed),kind=kind,seed=seed)
    (folder/'teachers'/f'{bank}.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    np.savez_compressed(folder/'teachers'/f'{bank}.npz',**arrays(agent))
    return dict(bank=bank,episodes=len(env.episodes),steps=env.total_steps,stable=stable)


@lru_cache(maxsize=16)
def load_prior(path):
    path=Path(path)
    with path.open('rb') as f:p=pickle.load(f)
    m=json.loads(path.with_suffix('.json').read_text())
    assert digest(p['tau'])==m['tau_sha256']
    return p,m


def student_job(args):
    path,job,smoke=args;folder=Path(path);seed=job['seed'];bank=job.get('bank');stage=job.get('stage','mature')
    prior=meta=None
    if bank:prior,meta=load_prior(str(folder/'priors'/f'{bank}__{stage}.pkl'))
    env,agent,raw,features,rng=setup(job['kind'],seed,job['role'],job.get('version','source'),prior,job.get('same_tiles',False),record=job.get('record',False))
    assert np.count_nonzero(agent.tile_code_weights)==0
    if hasattr(agent,'best_trajectory'):assert not agent.best_trajectory['trajectory'] and not agent.heuristic_activate
    initial_tau=agent.pheromone_trace.copy() if hasattr(agent,'pheromone_trace') else None
    n=3 if smoke else 200;times=train_episodes(env,agent,rng,n)
    assert len(agent.return_last_steps())==n and len(env.episodes)==n
    finite=bool(np.isfinite(agent.tile_code_weights).all())
    if hasattr(agent,'pheromone_trace'):finite=finite and bool(np.isfinite(agent.pheromone_trace).all())
    live_change=float(np.max(np.abs(agent.pheromone_trace-initial_tau))) if initial_tau is not None else 0.
    if job['role']=='static':assert live_change==0.
    metric=find_best_results([times],[agent.return_last_steps()])
    metrics={key:float(value[0]) for key,value in metric.items()}
    stem=job['id'];result=dict(**job,backend='actual_author_learn',episodes=env.episodes,
        author_last_steps=list(map(float,agent.return_last_steps())),episode_seconds=times,
        author_metrics=metrics,training_seconds=sum(times),steps=env.total_steps,finite=finite,
        live_change=live_change,final_epsilon=float(agent.config['eps']),config=agent.config|{'model':None},
        donor_steps=meta['steps'] if meta and not job.get('same_tiles') else 0,
        donor_reference=meta,static_tau_sha256=digest(initial_tau) if initial_tau is not None else None,
        coder_sha256=hashlib.sha256(pickle.dumps(raw)).hexdigest(),final_greedy=evaluate(agent,job['kind'],seed) if finite else None)
    (folder/'runs'/f'{stem}.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    np.savez_compressed(folder/'models'/f'{stem}.npz',**arrays(agent))
    if env.data:
        np.savez_compressed(folder/'trajectories'/f'{stem}.npz',obs=np.array([x[1] for x in env.data]),
            action=np.array([x[2] for x in env.data],np.uint8),reward=np.array([x[3] for x in env.data]),
            next_obs=np.array([x[4] for x in env.data]),done=np.array([x[5] for x in env.data]))
    return dict(id=stem,steps=env.total_steps,finite=finite,cohort=job['cohort'])


def jobs(smoke=False):
    out=[]
    main_seeds=range(20000,20002 if smoke else 20100)
    specs={
      'sarsa':dict(role='plain'), 'dynamic':dict(role='dynamic'),
      'static_flag':dict(role='static_flag',bank='historical_A'),
      'static_frozen':dict(role='static',version='frozen',bank='historical_A'),
      'static_200':dict(role='static',version='frozen',bank='historical_A',stage='e200'),
      'sarsa_same_tiles':dict(role='plain',bank='historical_A',same_tiles=True),
      'dynamic_same_tiles':dict(role='dynamic',bank='historical_A',same_tiles=True)}
    for seed in main_seeds:
        for name,spec in specs.items():out.append(dict(id=f'main__{name}__{seed}',cohort='main',method=name,kind='historical',seed=seed,record=seed==20000,**spec))
    for bank in ('historical_B','historical_C'):
        for seed in range(22000,22001 if smoke else 22030):
            out.append(dict(id=f'prior__{bank}__{seed}',cohort='prior',method=bank,kind='historical',seed=seed,role='static',version='frozen',bank=bank))
    for seed in range(21000,21001 if smoke else 21020):
        for kind,formulas,bank in [('historical',['eq38'],'historical_A'),('paper_text',['source','eq38'],'paper_text_A')]:
            out.append(dict(id=f'sensitivity__{kind}__sarsa__{seed}',cohort='sensitivity',method=kind+'__sarsa',kind=kind,seed=seed,role='plain'))
            for formula in formulas:
                for role in ('dynamic','static'):
                    version=('frozen' if role=='static' else 'source')+('_eq38' if formula=='eq38' else '')
                    out.append(dict(id=f'sensitivity__{kind}__{role}_{formula}__{seed}',cohort='sensitivity',method=kind+'__'+role+'_'+formula,
                        kind=kind,seed=seed,role=role,version=version,**({'bank':bank} if role=='static' else {})))
    return out


def warmup():
    numeric.feature_indices(np.array([-.5,0.]),np.zeros((3,2,10)),np.array([-1.2,-.07]),np.array([.6,.07]),10)


def hashes():
    paths=[HERE/'run_replication.py',HERE/'PROTOCOL.txt',HERE/'author_metrics.py']
    for folder in ('paper_author','gym_legacy'):paths+=list((HERE/folder).glob('*.py'))
    paths+=[ROOT/'pheromone_feedback_2019/experiment.py']
    return {os.path.relpath(p,HERE):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);p.add_argument('--smoke',action='store_true');p.add_argument('--out',default=None);a=p.parse_args()
    folder=HERE/(a.out or ('smoke' if a.smoke else 'results'))
    if (folder/'manifest.json').exists():raise RuntimeError('Immutable results exist; never overwrite.')
    for name in ('priors','teachers','runs','models','trajectories'):(folder/name).mkdir(parents=True,exist_ok=True)
    joblist=jobs(a.smoke);assert len(joblist)==(24 if a.smoke else 920)
    m=dict(status='running',smoke=a.smoke,source_hashes=hashes(),workers=a.workers,
        author_commit='95786b39b9ee8c5e004e71ccfa790304cbb3f684',episodes=3 if a.smoke else 200,
        student_runs=len(joblist),seed_counts=dict(main=2 if a.smoke else 100,prior=1 if a.smoke else 30,sensitivity=1 if a.smoke else 20))
    mp=folder/'manifest.json';mp.write_text(json.dumps(m,indent=2),encoding='utf8');began=time.perf_counter()
    teacher_specs=[('historical_A','historical',120001),('historical_B','historical',120002),('historical_C','historical',120003),('paper_text_A','paper_text',120004)]
    with ProcessPoolExecutor(max_workers=a.workers,initializer=warmup) as pool:
        teacher_results=list(pool.map(teacher_job,[(str(folder),*s,a.smoke) for s in teacher_specs]))
        print(json.dumps(dict(phase='teachers_complete',results=teacher_results)),flush=True)
        order=np.random.RandomState(887766).permutation(len(joblist));counts=[]
        fs=[pool.submit(student_job,(str(folder),joblist[i],a.smoke)) for i in order]
        for i,f in enumerate(as_completed(fs),1):
            counts.append(f.result())
            if i%20==0 or i==len(fs):print(json.dumps(dict(phase='students',complete=i,total=len(fs),seconds=round(time.perf_counter()-began,1))),flush=True)
    m.update(status='complete',teacher_results=teacher_results,physical_teacher_steps=sum(r['steps'] for r in teacher_results),
        physical_student_steps=sum(r['steps'] for r in counts),all_finite=all(r['finite'] for r in counts),elapsed_seconds=time.perf_counter()-began)
    mp.write_text(json.dumps(m,indent=2),encoding='utf8');print(json.dumps(m|{'source_hashes':'saved'}),flush=True)


if __name__=='__main__':main()
