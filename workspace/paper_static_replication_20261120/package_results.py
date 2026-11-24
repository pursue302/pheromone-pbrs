"""Portable source, frozen protocols, all results/models and original priors."""
from pathlib import Path
import hashlib
import json
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def main():
    assert json.loads((HERE/'results/manifest.json').read_text())['status']=='complete'
    assert (HERE/'summary.json').is_file() and (HERE/'复现结果.html').is_file()
    selected={}
    for p in HERE.rglob('*'):
        if not p.is_file():continue
        pp=p.relative_to(HERE).parts
        if '__pycache__' in pp or 'smoke' in pp or p.suffix in ('.pyc','.nbc','.nbi') or p.name=='package_inventory.json':continue
        selected[p.relative_to(ROOT).as_posix()]=p
    legacy=ROOT/'pheromone_feedback_2019'
    p=legacy/'experiment.py';selected[p.relative_to(ROOT).as_posix()]=p
    for p in (legacy/'author_port').glob('*'):
        if p.is_file() and (p.suffix=='.py' or p.name=='LICENSE'):selected[p.relative_to(ROOT).as_posix()]=p
    reference=ROOT/'author_direct_comparison_20261120/author_agents'
    for name in ('__init__.py','ApproximatedSarsaLambdaAgent.py','HAApproximatedSarsaLambdaAgent.py','pa_original.py','representation.py','LICENSE'):
        p=reference/name;selected[p.relative_to(ROOT).as_posix()]=p
    inventory=dict(files=[dict(path=k,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in sorted(selected.items())],
        includes='all formal training records, models, teacher files, tau+tilecoding priors, sampled interactions, author code, historical Gym, source patches, reports and required projection dependencies',
        omitted='short smoke cohort and runtime caches only')
    p=HERE/'package_inventory.json';p.write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf8');selected[p.relative_to(ROOT).as_posix()]=p
    target=ROOT/'paper_static_replication_20261120.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,p in sorted(selected.items()):z.write(p,'paper_replication_workspace/'+name)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))
    print(json.dumps(dict(path=str(target),bytes=target.stat().st_size,files=len(selected),sha256=hashlib.sha256(target.read_bytes()).hexdigest()),ensure_ascii=False))


if __name__=='__main__':main()
