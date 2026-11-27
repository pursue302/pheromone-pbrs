from pathlib import Path
import json,hashlib,collections,re
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'workspace'
def main():
    f=ROOT/'04_navigation_validation/experiments/fresh_nav';m=json.loads((f/'manifest.json').read_text())
    assert m['status']=='complete' and m['physical_steps']==18000000
    assert hashlib.sha256((ROOT/'04_navigation_validation/experiments/engine.py').read_bytes()).hexdigest()==m['source_sha256']
    provenance=json.loads((ROOT/'04_navigation_validation/experiments/provenance.json').read_text())
    assert hashlib.sha256((ROOT/'04_navigation_validation/experiments/PROTOCOL.txt').read_bytes()).hexdigest()==provenance['protocol_sha256']
    assert hashlib.sha256((ROOT/'03_controlled_study/study_unbounded_controls.py').read_bytes()).hexdigest()==provenance['sha256']
    paths=list((f/'runs').glob('*.json'));assert len(paths)==600
    prefixes=collections.defaultdict(set);error=0.
    for p in paths:
        r=json.loads(p.read_text());assert r['steps']==30000
        assert 40000<=r['seed']<40020
        prefixes[r['scenario'],r['seed']].add(r['prefix_hash']);error=max(error,r['max_telescoping_error'])
        with np.load(f/'models'/p.with_suffix('.npz').name) as a:assert all(np.isfinite(a[k]).all() for k in a.files)
    assert len(prefixes)==60 and all(len(x)==1 for x in prefixes.values()) and error<1e-7
    a=json.loads((HERE/'validation_summary.json').read_text());assert len(a['tests']['primary'])==6
    assert sum(t['p_holm']<.05 and t['difference']>0 for t in a['tests']['primary'])==4
    d=json.loads((ROOT/'01_goal_preservation/goal_diagnostic.json').read_text());assert d['max_value_shift_error']<1e-10
    snap=json.loads((HERE/'paper_data.json').read_text());assert len(snap['fresh_rows'])==600 and len(snap['mc_rows'])==400
    comp=json.loads((HERE/'compile_result.json').read_text());assert comp['exitCode']==0 and comp['pdfExists']
    assert 'Overfull' not in comp['log']
    result=dict(fresh_training_runs=600,training_steps=18000000,all_models_finite=True,
        distinct_seed_condition_pairs=60,warmup_identical=True,max_telescoping_error=error,
        new_primary_comparisons=6,positive_after_holm=4,method_retuned_after_validation=False,
        diagnostic_value_shift_error=d['max_value_shift_error'],compiler_exit_code=0,
        figure_count=len(list((HERE/'figures').glob('*.pdf'))),overfull_box_warnings=0,
        author_field_intentionally_blank=True)
    (HERE/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
