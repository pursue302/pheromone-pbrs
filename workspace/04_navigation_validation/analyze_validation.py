from pathlib import Path
import json,sys,collections,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'03_controlled_study'))
from analyze import stats,paired,holm

def main():
    folder=HERE/'experiments/fresh_nav';manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['status']=='complete'
    current_hash = hashlib.sha256((HERE/'experiments/engine.py').read_bytes()).hexdigest()
    archive_hash = json.loads((HERE/'experiments/provenance.json').read_text())['engine_sha256']
    assert manifest['source_sha256'] in (current_hash, archive_hash)
    rows=[json.loads(p.read_text()) for p in sorted((folder/'runs').glob('*.json'))];assert len(rows)==600
    groups=collections.defaultdict(list);prefix=collections.defaultdict(set);max_error=0.
    for r in rows:
        assert r['steps']==30000 and r['finite']
        r['auc']=float(np.trapezoid([z['original_return'] for z in r['records']],[z['step'] for z in r['records']])/r['steps'])
        groups[r['scenario'],r['method']].append(r);prefix[r['scenario'],r['seed']].add(r['prefix_hash'])
        max_error=max(max_error,r['max_telescoping_error'])
    assert all(len(x)==1 for x in prefix.values()) and max_error<1e-7
    summary={}
    for (k,m),rr in groups.items():
        assert len(rr)==20
        summary.setdefault(k,{})[m]={z:stats([r['final'][z] for r in rr]) for z in ['original_return','original_discounted_return','success_rate']}
        summary[k][m]['auc']=stats([r['auc'] for r in rr])
    families={}
    for name,controls in [('primary',['raw_static','raw_dynamic']),
        ('representation',['state_raw_static','state_raw_matched','return_static','mc_dynamic_all_unbounded','plain']),
        ('strength',['gate_scale003','gate_scale030'])]:
        tests=[]
        for k in sorted(summary):
            x={r['seed']:r['final']['original_return'] for r in groups[k,'return_gate_unbounded']}
            for b in controls:
                tests.append(dict(scenario=k,a='return_gate_unbounded',b=b,**paired(x,{r['seed']:r['final']['original_return'] for r in groups[k,b]})))
        families[name]=holm(tests)
    out=dict(summary=summary,tests=families,runs=600,steps=18000000,warmup_identical=True,max_telescoping_error=max_error,
        interpretation='New validation family fixed before these outcomes; old primary failures remain unchanged.')
    (HERE/'validation_summary.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    print(json.dumps(dict(means={k:{m:round(v['original_return']['mean'],3) for m,v in z.items()} for k,z in summary.items()},
        tests={f:[(t['scenario'],t['b'],round(t['difference'],3),round(t['p_holm'],4)) for t in z] for f,z in families.items()}),indent=2))

if __name__=='__main__':main()
