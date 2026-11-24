from pathlib import Path
import json,hashlib,zipfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def main():
    a=json.loads((HERE/'audit.json').read_text());q=json.loads((HERE/'pdf_qa.json').read_text())
    assert a['fresh_training_runs']==600 and q['main_text_pages']==7 and q['pages']==8
    ignore={'qa','__pycache__','build'}
    files=[p for p in HERE.rglob('*') if p.is_file() and not any(x in ignore for x in p.relative_to(HERE).parts)
        and p.name not in {'FILE_MANIFEST.json','DELIVERY.json','latex_runtime.json','extracted_text.txt','extracted_layout.txt'}
        and not p.name.endswith('.log')]
    for name in ['study_unbounded_controls.py','analyze.py','PROTOCOL.txt','SELECTION.json','UNBOUNDED_CONTROL_PROTOCOL.txt','literature.json']:
        files.append(ROOT/'research_story_20261120'/name)
    p=ROOT/'paper_static_replication_20261120'
    files += [p/x for x in ['run_replication.py','author_metrics.py','PROTOCOL.txt']]
    files += list((p/'paper_author').glob('*.py'))+list((p/'gym_legacy').glob('*.py'))
    files += [p/'results/priors'/('historical_A__mature'+x) for x in ['.pkl','.json']]
    files += [ROOT/'pheromone_feedback_2019/experiment.py']+list((ROOT/'pheromone_feedback_2019/author_port').glob('*.py'))
    files += [ROOT/'cs4620j_nb1/cs4620j_nb1/cs4620j/envs.py',ROOT/'output/pdf/pheromone_shaping_report.pdf']
    orig=ROOT/'project_2_8_research/author_code/RLACOSarsaLambda_95786b39'
    files += [p for p in orig.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts]
    files=sorted(set(files));hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest=HERE/'FILE_MANIFEST.json';manifest.write_text(json.dumps(hashes,indent=2),encoding='utf8');files.append(manifest)
    target=ROOT/'pheromone_shaping_latex_and_experiments.zip'
    if target.exists():raise RuntimeError('Delivery archive exists; choose a new version explicitly.')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    digest=hashlib.sha256(target.read_bytes()).hexdigest();target.with_suffix('.zip.sha256.txt').write_text(digest+'  '+target.name+'\n')
    out=dict(archive=str(target),bytes=target.stat().st_size,files=len(files),sha256=digest,final_pdf=q['final_path'])
    (HERE/'DELIVERY.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
