"""New experiments; previous result directories remain immutable.

repair: exact old Gym dynamics/author features, 200 complete episodes.
discovery/confirmation: equal total interaction budgets, common warm-up.
All reported rewards are environmental rewards, never shaped rewards.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'1')
import sys, json, time, hashlib, argparse, copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import numpy as np
from numba import njit

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'paper_static_replication_20261120'))
import run_replication as paper
old=paper.numeric

@njit(cache=True)
def update(w,z,idx,a,alpha,gamma,lam,delta,dq,canonical):
    dot=0.
    for j in idx:dot+=z[a,j]
    if canonical:
        z*=gamma*lam
        add=1.-alpha*gamma*lam*dot
    else:add=1.-alpha*gamma*lam*dot
    for j in idx:z[a,j]+=add
    # Preserve upstream arithmetic association in the source-compatible arm.
    for b in range(w.shape[0]):
        for j in range(w.shape[1]):w[b,j]+=alpha*delta*z[b,j]+alpha*dq*z[b,j]
    for j in idx:w[a,j]-=alpha*dq
    if not canonical:z*=gamma*lam

@dataclass
class Config:
    mc_steps:int=60000
    grid_steps:int=30000
    mc_cap:int=1000
    grid_cap:int=200
    evaluation_episodes:int=20
    epsilon:float=.025
    rho:float=.99
    warmup:float=.25
    publish:int=10
    scale:float=.1
    beta:float=.5
    limit:float=1.
    kappa:float=5.

class Historic(paper.Environment):
    def __init__(self,seed):
        super().__init__('historical',seed)
        self.n_count=900
        self._max_episode_steps=self.cap
    def count_bin(self,obs):
        x=np.minimum(29,np.maximum(0,((obs-self.low)/(self.high-self.low)*30).astype(int)))
        return int(x[0]*30+x[1])

def environment(key,seed,cfg):
    return Historic(seed) if key=='mountain_car' else old.GridEnv(key,seed,cfg.grid_cap)

def evaluate(env_factory,features,w,n=20):
    env=env_factory();rows=[]
    n=n if env.key in ('mountain_car','slippery_rooms') else 1
    for pos in np.linspace(-.6,-.4,n):
        obs=env.reset(float(pos) if env.key=='mountain_car' else None)
        total=disc=0.;factor=1.;seen=set()
        for t in range(env.cap):
            if env.key!='slippery_rooms':
                s=tuple(map(float,env.base.state)) if env.key=='mountain_car' else env.s
                if s in seen:
                    remain=env.cap-t;r=-1.
                    total+=remain*r
                    disc+=factor*r*(remain if env.gamma==1 else (1-env.gamma**remain)/(1-env.gamma))
                    break
                seen.add(s)
            a=int(np.argmax(old.values(w,features(obs))))
            obs,r,done,_=env.step(a);total+=r;disc+=factor*r;factor*=env.gamma
            if done:break
        rows.append((total,disc,float(env.last_success)))
    return dict(zip(('original_return','original_discounted_return','success_rate'),map(float,np.mean(rows,axis=0))))

def repair_job(job):
    out,method,seed=job;started=time.perf_counter()
    role='plain';prior=None
    if method.startswith('static'):
        role='static';prior,_=paper.load_prior(str(ROOT/'paper_static_replication_20261120/results/priors/historical_A__mature.pkl'))
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

class Memory:
    """Independent completed trajectories, not visits or elite replays."""
    def __init__(self,n):
        self.weight=np.zeros(n);self.target=np.zeros(n);self.count=np.zeros(n)
    def add(self,states,rewards,gamma,success,quality=True):
        # Failed trajectories remain valid observations, but do not reinforce
        # the success pheromone memory. This is predeclared, not eval-dependent.
        if not success:return
        future=0.;g=np.empty(len(rewards))
        for i in range(len(rewards)-1,-1,-1):future=rewards[i]+gamma*future;g[i]=future
        # A tile can appear many times during one trajectory: keep the final
        # visit, so a single looping ant cannot multiply the evidence count.
        states=np.asarray(states);_,rev=np.unique(states[::-1],return_index=True)
        pos=len(states)-1-rev;s=states[pos];v=g[pos]
        q=1./len(rewards) if quality else 1.
        self.weight*=.99;self.target*=.99
        self.weight[s]+=q;self.target[s]+=q*v;self.count[s]+=1
    def mean(self,default):
        return np.divide(self.target,self.weight,out=np.full(len(self.weight),default),where=self.weight>1e-200)

METHODS=('plain','raw_static','raw_dynamic','raw_mix',
         'return_static','return_dynamic','return_mix','return_gate','return_gate_unbounded',
         'mc_dynamic','velocity','velocity_shift','manhattan')

def run_group(args):
    out,key,seed,methods,cfg=args;began=time.perf_counter()
    results=[]
    for method in methods:
        env=environment(key,seed+451,cfg)
        agent=old.create_agent(env,seed,cfg);features=old.Features(agent,env)
        agent.tile_coder={a:old.FastCoder(features,a) for a in range(env.actions.n)}
        w,z=agent.tile_code_weights,agent.tile_code_trace_weights
        rng=np.random.default_rng(seed+2027)
        budget=cfg.mc_steps if key=='mountain_car' else cfg.grid_steps
        warmup=int(budget*cfg.warmup);step=0;ep=0;anchor=None;anchor_tau=None
        memory=Memory(env.n_count);online=Memory(env.n_count)
        phi=np.zeros(env.n_count);anchor_phi=np.zeros_like(phi)
        trajectory={};episodes=[];records=[];lastpub=0;error=0.;phi_drift=0.
        default=-min(200.,float(env.cap));raw_active=False
        prefix=hashlib.sha256();shaping2=0.;publications=0
        def decision(obs):
            ids=features(obs);q=old.values(w,ids)
            a=int(rng.integers(env.actions.n)) if rng.random()<cfg.epsilon else int(np.argmax(q))
            return a,ids,q
        def pot(obs,ids,a):
            if anchor is None or method=='plain':return 0.
            if method.startswith('raw_'):
                x=old.values(anchor_tau,ids)[a]/10
                if method=='raw_static':return float(x)
                y=old.values(agent.pheromone_trace,ids)[a]/10
                return float(y if method=='raw_dynamic' else (1-cfg.beta)*x+cfg.beta*y)
            if method.startswith('velocity'):
                return float(abs(obs[1])/.07+(1. if method=='velocity_shift' else 0.))
            if method=='manhattan':
                gr,gc=env.base.s_to_rc(next(iter(env.base.terminal_states)))
                return -float(abs(obs[0]-gr)+abs(obs[1]-gc))
            return float(phi[env.count_bin(obs)])
        def check():
            m=evaluate(lambda:environment(key,seed+918273,cfg),features,w,cfg.evaluation_episodes)
            records.append(dict(step=step,**m))
        check()
        while step<budget:
            obs=env.reset();z.fill(0.);qold=0.;a,ids,q=decision(obs)
            p=pot(obs,ids,a);initial=p;shaped=0.;factor=1.;total=disc=0.;start=step
            state_list=[];reward_list=[];trajectory={};past=ids[a].copy();aold=a
            for t in range(env.cap):
                state_list.append(env.count_bin(obs));trajectory[(float(obs[0]),float(obs[1]),a)]=dict(state=obs.copy(),action=a,time=t)
                sp,r,done,_=env.step(a);ap,ids2,q2=decision(sp);p2=0. if done else pot(sp,ids2,ap)
                future=0. if done else q2[ap];f=env.gamma*p2-p
                update(w,z,ids[a],a,agent.alpha,env.gamma,.9,r+f+env.gamma*future-q[a],q[a]-qold,True)
                # Released author's local/global pheromone rules are retained
                # for raw controls, separate from the new trajectory memory.
                if agent.heuristic_activate and (anchor is None or method in ('raw_dynamic','raw_mix')):
                    old.local_pheromone_update(agent.pheromone_trace,past,ids[a],ids2[ap],aold,a,ap,.99,10)
                if anchor is None:
                    prefix.update(np.asarray(obs,dtype=np.float64).tobytes());prefix.update(bytes([a]))
                reward_list.append(r);step+=1;total+=r;disc+=factor*r;shaped+=factor*f;factor*=env.gamma;shaping2+=f*f
                qold=future;p=p2;past=ids[a].copy();aold=a;obs=sp;a=ap;ids=ids2;q=old.values(w,ids)
                if done:
                    success=bool(env.last_success);ep+=1
                    if success:agent.heuristic_activate=True
                    if agent.heuristic_activate and (anchor is None or method in ('raw_dynamic','raw_mix')):
                        agent.handle_trajectory(result=success,t=max(t,1),trajectory=trajectory)
                        agent.reinforce_heuristic_model()
                    memory.add(state_list,reward_list,env.gamma,success,method!='mc_dynamic')
                    if anchor is not None:online.add(state_list,reward_list,env.gamma,success,method!='mc_dynamic')
                    if anchor is None and step>=warmup:
                        anchor=copy.deepcopy(memory);anchor_tau=agent.pheromone_trace.copy()
                        anchor_phi=cfg.scale*anchor.mean(default);phi=anchor_phi.copy()
                        # Pure dynamic starts without an offline memory.
                        if method in ('return_dynamic','mc_dynamic'):phi=cfg.scale*online.mean(default)
                        lastpub=ep
                    elif anchor is not None and ep-lastpub>=cfg.publish and method in ('return_dynamic','return_mix','return_gate','return_gate_unbounded','mc_dynamic'):
                        live=cfg.scale*online.mean(default)
                        if method in ('return_dynamic','mc_dynamic'):candidate=live
                        elif method=='return_mix':candidate=(1-cfg.beta)*anchor_phi+cfg.beta*live
                        else:
                            beta=cfg.beta*online.count/(online.count+cfg.kappa)
                            # Unobserved online states keep the offline anchor.
                            candidate=(1-beta)*anchor_phi+beta*live
                        limit=np.inf if method=='return_gate_unbounded' else cfg.limit
                        new=phi+np.clip(candidate-phi,-limit,limit)
                        phi_drift+=float(np.mean(abs(new-phi)));phi=new;lastpub=ep;publications+=1
                    e=abs(shaped+initial);error=max(error,e)
                    episodes.append(dict(start=start,end=step,length=t+1,success=success,original_return=total,
                        original_discounted_return=disc,initial_potential=initial,telescoping_error=e))
                if step%5000==0 or step==budget:check()
                if done or step==budget:break
        assert error<1e-7 and np.isfinite(w).all()
        result=dict(method=method,scenario=key,seed=seed,config=asdict(cfg),steps=step,warmup_steps=episodes[next(i for i,x in enumerate(episodes) if x['end']>=warmup)]['end'],
            prefix_hash=prefix.hexdigest(),episodes=episodes,records=records,final=records[-1],finite=True,
            max_telescoping_error=error,potential_drift=phi_drift,publications=publications,rms_shaping=float(np.sqrt(shaping2/budget)))
        save(out,key+'__'+method,seed,result,dict(weights=w,phi=phi,anchor_phi=anchor_phi,online_count=online.count))
        results.append(dict(method=method,steps=step,final=result['final']))
    return dict(scenario=key,seed=seed,results=results,seconds=time.perf_counter()-began)

def save(out,method,seed,result,arrays):
    folder=Path(out);(folder/'runs').mkdir(parents=True,exist_ok=True);(folder/'models').mkdir(exist_ok=True)
    name=f'{method}__{seed}'
    path=folder/'runs'/f'{name}.json'
    if path.exists():raise RuntimeError('immutable run already exists: '+str(path))
    path.write_text(json.dumps(result,indent=2),encoding='utf8')
    np.savez_compressed(folder/'models'/f'{name}.npz',**arrays)

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['repair','discover','confirm']);p.add_argument('--seeds',type=int,default=8);p.add_argument('--start',type=int,default=30000)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--out',default=None);p.add_argument('--methods',default=None);p.add_argument('--scenarios',default='mountain_car,u_detour,four_rooms,slippery_rooms')
    p.add_argument('--scale',type=float,default=.1);p.add_argument('--beta',type=float,default=.5);p.add_argument('--limit',type=float,default=1.);a=p.parse_args()
    out=HERE/(a.out or a.phase);out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('immutable experiment directory')
    cfg=Config(scale=a.scale,beta=a.beta,limit=a.limit)
    methods=a.methods.split(',') if a.methods else (['reset','canonical','static_reset','static_canonical'] if a.phase=='repair' else list(METHODS[:-1]))
    if a.phase=='repair':jobs=[(str(out),m,s) for s in range(a.start,a.start+a.seeds) for m in methods];fn=repair_job
    else:
        jobs=[(str(out),k,s,[m for m in methods if not (m.startswith('velocity') and k!='mountain_car') and not (m=='manhattan' and k=='mountain_car')],cfg)
            for s in range(a.start,a.start+a.seeds) for k in a.scenarios.split(',')];fn=run_group
    manifest=dict(status='running',args=vars(a),config=asdict(cfg),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),jobs=len(jobs))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8');start=time.perf_counter();steps=0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(fn,j) for j in jobs]
        for i,f in enumerate(as_completed(futures),1):
            r=f.result();steps+=r.get('steps',sum(x['steps'] for x in r.get('results',[])))
            print(json.dumps(dict(completed=i,total=len(jobs),seconds=round(time.perf_counter()-start,1),result=r)),flush=True)
    manifest.update(status='complete',physical_steps=steps,elapsed_seconds=time.perf_counter()-start)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')

if __name__=='__main__':main()
