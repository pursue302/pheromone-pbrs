"""Compact, inspectable data for regenerating the paper without legacy paths."""
from pathlib import Path
import json,hashlib,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(HERE/'experiments'))
import engine

def main():
    inputs={};records={}
    def read(path):
        inputs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text(encoding='utf8'))
    def compact(folder,methods=None):
        rows=[]
        for p in sorted(folder.glob('*.json')):
            r=read(p)
            if methods is not None and r['method'] not in methods:continue
            rows.append({k:r[k] for k in ['method','scenario','seed','steps','final','records']})
        return rows
    old=read(ROOT/'research_story_20261120/analysis.json');validation=read(HERE/'validation_summary.json')
    fresh=compact(HERE/'experiments/fresh_nav/runs')
    methods=['plain','raw_static','return_gate_unbounded','return_dynamic_all_unbounded','mc_dynamic_all_unbounded']
    mc=compact(ROOT/'research_story_20261120/confirm_mc/runs',methods)+compact(ROOT/'research_story_20261120/unbounded_mc/runs',methods)
    assert len(fresh)==600 and len(mc)==400
    out=dict(previous_analysis=old,validation=validation,fresh_rows=fresh,mc_rows=mc,
        diagnostic=read(HERE/'experiments/goal_diagnostic.json'),
        layouts={k:engine.old.scenario_layout(k) for k in ['u_detour','four_rooms','slippery_rooms']},
        provenance='Lossless selection of saved metrics used in the paper. Episodes and models remain in the raw experiment archives.')
    (HERE/'paper_data.json').write_text(json.dumps(out,separators=(',',':')),encoding='utf8')
    (HERE/'data_provenance.json').write_text(json.dumps(inputs,indent=2),encoding='utf8')
    print('Wrote standalone figure data:',len(fresh),'fresh records,',len(mc),'MC records.')

if __name__=='__main__':main()
