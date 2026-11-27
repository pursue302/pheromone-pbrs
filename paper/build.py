import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare_paper(out):
    out.mkdir(parents=True)
    for name in ['main.tex','paper_data.json']:
        shutil.copy2(ROOT / 'paper' / name, out / name)


def figures(out, run_logged):
    prepare_paper(out)
    run_logged([sys.executable,ROOT / 'paper/make_figures.py', '--out', out], out)


def report(out, dry_run, run_logged):
    candidate = os.environ.get('CODEX_TECTONIC_PATH')
    tectonic = candidate if candidate and Path(candidate).is_file() else shutil.which('tectonic')
    latexmk = shutil.which('latexmk')
    pdflatex = shutil.which('pdflatex')
    if tectonic:
        command = [tectonic,'-X','compile','--untrusted','--outdir',str(out / 'build'),'main.tex']
    elif latexmk:
        command = [latexmk,'-norc','-pdf','-interaction=nonstopmode','-halt-on-error','-outdir=build','main.tex']
    elif pdflatex:
        command = [pdflatex,'-interaction=nonstopmode','-halt-on-error','-output-directory=build','main.tex']
    else:
        raise RuntimeError('No LaTeX compiler found. The ready-made report is paper/report.pdf. Rebuilding needs Tectonic, latexmk, or pdflatex.')
    if dry_run:
        print(json.dumps(dict(command=command,output=str(out)),indent=2));return
    prepare_paper(out);(out / 'build').mkdir()
    run_logged([sys.executable,ROOT / 'paper/make_figures.py', '--out', out],out)
    run_logged(command,out,cwd=out)
    if not tectonic and not latexmk:
        run_logged(command,out,cwd=out)
    if not (out / 'build/main.pdf').is_file():raise RuntimeError('Compiler produced no PDF')
    print('PDF:',out / 'build/main.pdf')


