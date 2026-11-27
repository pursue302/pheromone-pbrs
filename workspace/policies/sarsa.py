import numpy as np
from numba import njit


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

def epsilon_greedy(values, action_count, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(action_count))
    return int(np.argmax(values))


@njit(cache=True)
def values(weights, indices):
    out = np.zeros(indices.shape[0])
    for a in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            out[a] += weights[a, indices[a, j]]
    return out


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
            a=int(np.argmax(values(w,features(obs))))
            obs,r,done,_=env.step(a);total+=r;disc+=factor*r;factor*=env.gamma
            if done:break
        rows.append((total,disc,float(env.last_success)))
    return dict(zip(('original_return','original_discounted_return','success_rate'),map(float,np.mean(rows,axis=0))))

