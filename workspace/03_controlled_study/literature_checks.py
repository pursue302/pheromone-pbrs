"""Small exact checks of published theoretical examples, not paper replications."""
from pathlib import Path
import json
import numpy as np

def main():
    here=Path(__file__).resolve().parent
    gamma=.99
    # Grzes 2017 Fig 1: unequal terminal potentials invert the preferred goal.
    original=np.array([0.,100.]);bad=original+gamma*np.array([1000.,10.]);fixed=original.copy()
    assert original.argmax()==fixed.argmax()==1 and bad.argmax()==0
    # Wiewiora 2003: unshaped Q_init=Phi versus shaped Q_init=0 in a
    # finite tabular episodic problem; terminal potential exactly zero.
    rng=np.random.default_rng(75121);n=12;phi=-np.arange(n,0,-1,dtype=float);phi[-1]=0.
    qs=np.zeros((n,2));qo=np.broadcast_to(phi[:,None],qs.shape).copy();alpha=.1
    maxerror=0.;same=True;s=0
    for t in range(5000):
        exploratory=rng.random()<.1;random_a=int(rng.integers(2))
        a=random_a if exploratory else int(qs[s].argmax())
        ao=random_a if exploratory else int(qo[s].argmax());same=same and a==ao
        sp=min(n-1,s+1) if a==1 else max(0,s-1);r=-1.;done=sp==n-1
        future_s=0. if done else qs[sp].max();future_o=0. if done else qo[sp].max()
        qs[s,a]+=alpha*(r+gamma*phi[sp]-phi[s]+gamma*future_s-qs[s,a])
        qo[s,a]+=alpha*(r+gamma*future_o-qo[s,a])
        maxerror=max(maxerror,float(np.max(abs(qo-qs-phi[:,None]))))
        s=0 if done else sp
    assert same and maxerror<1e-10
    result=dict(grzes2017_terminal_counterexample=dict(gamma=gamma,original_action_values=original.tolist(),
        nonzero_terminal_shaped_values=bad.tolist(),zero_terminal_shaped_values=fixed.tolist(),
        correct_preference_preserved=bool(fixed.argmax()==original.argmax())),
        wiewiora2003_tabular_identity=dict(transitions=5000,identical_decisions=same,max_coordinate_error=maxerror),
        scope='Exact small theoretical checks; not the full experiments or original author implementations.')
    (here/'literature_checks.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
