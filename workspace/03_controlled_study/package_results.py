"""Package complete new evidence and the minimum original-source dependencies."""
from pathlib import Path
import zipfile, hashlib, json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def main():
    audit=json.loads((HERE/'AUDIT.json').read_text());assert audit['total_runs']==3660
    paths=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    base=ROOT/'02_source_audit'
    paths += [base/x for x in ('run_replication.py','author_metrics.py','PROTOCOL.txt')]
    paths += list((ROOT/'policies/paper_author').glob('*.py'))+list((ROOT/'environments/gym_legacy').glob('*.py'))
    paths += [base/'results/priors'/('historical_A__mature'+x) for x in ('.pkl','.json')]
    for m in ('sarsa','static_frozen','dynamic'):paths+=list((base/'results/runs').glob('main__'+m+'__*.json'))
    paths += [ROOT/'02_source_audit/feedback/experiment.py']+list((ROOT/'policies/author_port').glob('*.py'))
    paths += [ROOT/'environments/gridworld.py']
    paths += list((ROOT/'policies').glob('*.py')) + list((ROOT/'environments').glob('*.py'))
    paths += [ROOT/'02_source_audit/trace_repair.py']
    orig=ROOT/'02_source_audit/upstream_author'
    paths += [p for p in orig.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts]
    paths=sorted(set(paths));target=ROOT/'03_controlled_study.zip'
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
