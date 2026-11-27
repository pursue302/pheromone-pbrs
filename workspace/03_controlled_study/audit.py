"""Verify actual completion, frozen source, budgets and model finiteness."""
from pathlib import Path
import json, hashlib, collections
import numpy as np
HERE=Path(__file__).resolve().parent
EXPECTED={'repair':400,'direct_reset':100,'discovery':360,'confirm_mc':960,'confirm_grids':990,
          'strong_mc':160,'strong_grids':180,'unbounded_mc':240,'unbounded_grids':270}

def main():
    result={};steps=0;prefix=collections.defaultdict(set);max_telescope=0.
    for name,count in EXPECTED.items():
        folder=(HERE.parent/'02_source_audit' if name in ('repair','direct_reset') else HERE)/name;manifest=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
        assert manifest['status']=='complete',(name,manifest['status'])
        paths=sorted((folder/'runs').glob('*.json'));assert len(paths)==count,(name,len(paths),count)
        source=HERE/('study_v1.py' if name in ('repair','discovery') else 'direct_reset.py' if name=='direct_reset' else
            'study_strong_controls.py' if name.startswith('strong') else 'study_unbounded_controls.py' if name.startswith('unbounded') else 'study.py')
        if name=='direct_reset':source=HERE.parent/'02_source_audit/direct_reset.py'
        assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['source_sha256'],name
        phase_steps=0
        for path in paths:
            r=json.loads(path.read_text(encoding='utf8'));assert r['finite'];phase_steps+=r['steps']
            assert r['final'] is not None
            assert np.isfinite(r['final']['original_return'])
            if 'scenario' in r:
                budget=60000 if r['scenario']=='mountain_car' else 30000;assert r['steps']==budget
                prefix[r['scenario'],r['seed']].add(r['prefix_hash'])
                max_telescope=max(max_telescope,r['max_telescoping_error']);assert r['max_telescoping_error']<1e-7
            else:assert len(r['episodes'])==200
            model=folder/'models'/path.with_suffix('.npz').name
            with np.load(model) as arrays:
                assert all(np.isfinite(arrays[k]).all() for k in arrays.files),str(model)
        assert phase_steps==manifest.get('physical_steps',manifest.get('steps'))
        steps+=phase_steps;result[name]=dict(runs=count,steps=phase_steps,status='verified',source_sha256=manifest['source_sha256'])
    assert all(len(v)==1 for v in prefix.values()),'warmup mismatch'
    assert not(set(range(30000,30008))&set(range(31000,31080)))
    eq=json.loads((HERE.parent/'02_source_audit/direct_reset_equivalence.json').read_text());assert eq['max_weight_error']<1e-10
    result.update(total_runs=sum(EXPECTED.values()),total_steps=steps,max_telescoping_error=max_telescope,
        identical_warmup_prefixes=True,direct_source_equivalence_max_error=eq['max_weight_error'],all_models_finite=True)
    (HERE/'AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
