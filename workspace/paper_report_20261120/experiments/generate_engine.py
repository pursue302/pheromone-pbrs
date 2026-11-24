"""Generate a transparent extension of the frozen study, without editing it."""
from pathlib import Path
import difflib,hashlib,json
p=Path(__file__).resolve().parent;root=p.parents[1]
source=root/'research_story_20261120/study_unbounded_controls.py'
s=original=source.read_text(encoding='utf8')
old='HERE=Path(__file__).resolve().parent;ROOT=HERE.parent'
assert old in s;s=s.replace(old,'HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]')
s=s.replace('from dataclasses import dataclass, asdict','from dataclasses import dataclass, asdict, replace')
old='    results=[]\n    for method in methods:\n'
new='''    results=[]
    initial_cfg=cfg
    for tag in methods:
        method='return_gate_unbounded' if tag.startswith('gate_scale') else tag
        cfg=replace(initial_cfg,scale={'gate_scale003':.03,'gate_scale030':.3}.get(tag,initial_cfg.scale))
'''
assert old in s;s=s.replace(old,new)
old='                        anchor_phi=cfg.scale*anchor.mean(default);phi=anchor_phi.copy()'
new=old+'''
                        if method in ('state_raw_static','state_raw_matched'):
                            profile=np.array([old.values(anchor_tau,features(x)).mean()/10 for x in env.observations])
                            if method=='state_raw_matched':
                                seen=anchor.count>0
                                if seen.sum()>=2 and profile[seen].std()>1e-12:
                                    profile=anchor_phi[seen].mean()+(profile-profile[seen].mean())*anchor_phi[seen].std()/profile[seen].std()
                                else:profile=np.full_like(profile,anchor_phi[seen].mean() if seen.any() else 0.)
                            phi=profile.copy()
'''
assert old in s;s=s.replace(old,new)
s=s.replace('result=dict(method=method,scenario=key,seed=seed,config=asdict(cfg)', 'result=dict(method=tag,scenario=key,seed=seed,config=asdict(cfg)')
s=s.replace("save(out,key+'__'+method,seed,result", "save(out,key+'__'+tag,seed,result")
s=s.replace('results.append(dict(method=method,steps=step', 'results.append(dict(method=tag,steps=step')
assert s!=original
(p/'engine.py').write_text(s,encoding='utf8')
(p/'engine.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile=str(source.relative_to(root)),tofile='paper_report_20261120/experiments/engine.py')),encoding='utf8')
(p/'provenance.json').write_text(json.dumps({'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'engine_sha256':hashlib.sha256((p/'engine.py').read_bytes()).hexdigest(),
    'protocol_sha256':hashlib.sha256((p/'PROTOCOL.txt').read_bytes()).hexdigest()},indent=2),encoding='utf8')
print('Generated frozen extension and diff.')
