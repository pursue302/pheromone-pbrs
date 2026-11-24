"""Package complete new evidence and the minimum original-source dependencies."""
from pathlib import Path
import zipfile, hashlib, json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def main():
    audit=json.loads((HERE/'AUDIT.json').read_text());assert audit['total_runs']==3660
    paths=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    base=ROOT/'paper_static_replication_20261120'
    paths += [base/x for x in ('run_replication.py','author_metrics.py','PROTOCOL.txt')]
    paths += list((base/'paper_author').glob('*.py'))+list((base/'gym_legacy').glob('*.py'))
    paths += [base/'results/priors'/('historical_A__mature'+x) for x in ('.pkl','.json')]
    for m in ('sarsa','static_frozen','dynamic'):paths+=list((base/'results/runs').glob('main__'+m+'__*.json'))
    paths += [ROOT/'pheromone_feedback_2019/experiment.py']+list((ROOT/'pheromone_feedback_2019/author_port').glob('*.py'))
    paths += [ROOT/'cs4620j_nb1/cs4620j_nb1/cs4620j/envs.py']
    orig=ROOT/'project_2_8_research/author_code/RLACOSarsaLambda_95786b39'
    paths += [p for p in orig.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts]
    paths=sorted(set(paths));target=ROOT/'research_story_20261120.zip'
    if target.exists():raise RuntimeError('archive exists; never overwrite silently')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in paths:z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(paths)
    sha=hashlib.sha256(target.read_bytes()).hexdigest()
    info=dict(path=str(target),bytes=target.stat().st_size,files=len(paths),sha256=sha)
    target.with_suffix('.zip.sha256.txt').write_text(sha+'  '+target.name+'\n',encoding='ascii')
    (HERE/'PACKAGE.json').write_text(json.dumps(info,indent=2),encoding='utf8');print(json.dumps(info,indent=2))
if __name__=='__main__':main()
