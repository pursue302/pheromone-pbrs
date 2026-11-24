"""Portable entry points for the frozen pheromone-PBRS research project."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / 'workspace'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def fresh_output(name, category='runs', label='run'):
    """Keep new outputs apart from the frozen source and experiment records."""
    base = (ROOT / category).resolve()
    if name:
        requested = Path(name)
        if requested.is_absolute():
            target = requested.resolve()
        elif requested.parts[0] == category:
            target = (ROOT / requested).resolve()
        else:
            target = (base / requested).resolve()
    else:
        target = base / (label + '_' + uuid.uuid4().hex[:8])
    if target == base or base not in target.parents:
        raise ValueError('Output must be a new directory inside ' + str(base))
    if target.exists():
        raise ValueError('Output already exists; choose a different --out: ' + str(target))
    return target


def run_logged(command, out, cwd=ROOT):
    out.mkdir(parents=True, exist_ok=True)
    log = out / 'execution.log'
    print('Running:', subprocess.list2cmdline(list(map(str, command))), flush=True)
    print('Log:', log, flush=True)
    with log.open('w', encoding='utf-8') as handle:
        result = subprocess.run(list(map(str, command)), cwd=cwd, stdout=handle,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        print(log.read_text(encoding='utf-8', errors='replace')[-5000:], file=sys.stderr)
        raise RuntimeError('Command failed with exit code ' + str(result.returncode))
    print('Completed:', out, flush=True)
    return out


def presets():
    return {p.stem: read_json(p) for p in sorted((ROOT / 'configs').glob('*.json'))}


def list_project(_args):
    print(json.dumps(read_json(ROOT / 'docs/INVENTORY.json'), ensure_ascii=False, indent=2))
    print('\nTraining presets:')
    for name, config in presets().items():
        print('  ' + name + ': ' + config['description'])


def verify(args):
    manifest = read_json(ROOT / 'MANIFEST.json')
    errors = []
    checked = 0
    for relative, record in manifest['files'].items():
        path = ROOT / relative
        if not path.is_file():
            errors.append('Missing: ' + relative)
            continue
        if path.stat().st_size != record['bytes']:
            errors.append('Size changed: ' + relative)
            continue
        # Fast checks still verify source/configuration integrity.
        if not args.fast or path.suffix in {'.py', '.tex', '.txt'} or relative.startswith('configs/'):
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
                errors.append('Hash changed: ' + relative)
        checked += 1
    if manifest.get('artifact_type') == 'source_repository':
        if errors:
            raise RuntimeError('\n'.join(errors[:30]))
        print(json.dumps(dict(status='verified',files=checked,
            mode='fast' if args.fast else 'full_sha256',artifact_type='source_repository',
            note='Frozen raw training records and models are distributed separately.'),indent=2))
        return
    inventory = read_json(ROOT / 'docs/INVENTORY.json')
    counts = {
        'author_replication_students': len(list((WORKSPACE / 'paper_static_replication_20261120/results/runs').glob('*.json'))),
        'author_replication_teachers': len(list((WORKSPACE / 'paper_static_replication_20261120/results/teachers').glob('*.json'))),
        'controlled_training_runs': sum(len(list((WORKSPACE / 'research_story_20261120' / name / 'runs').glob('*.json')))
            for name in ['repair','direct_reset','discovery','confirm_mc','confirm_grids','strong_mc','strong_grids','unbounded_mc','unbounded_grids']),
        'fresh_validation_runs': len(list((WORKSPACE / 'paper_report_20261120/experiments/fresh_nav/runs').glob('*.json'))),
    }
    for name, count in counts.items():
        if count != inventory[name]:
            errors.append('Unexpected record count: ' + name + ' = ' + str(count))
    for folder in [WORKSPACE / 'paper_static_replication_20261120/results',
                   WORKSPACE / 'research_story_20261120',
                   WORKSPACE / 'paper_report_20261120/experiments/fresh_nav']:
        for subfolder in folder.rglob('*'):
            if subfolder.is_dir() and subfolder.name in {'runs', 'teachers', 'models', 'priors', 'trajectories'}:
                for path in subfolder.iterdir():
                    if path.is_file() and path.relative_to(ROOT).as_posix() not in manifest['files']:
                        errors.append('Untracked frozen evidence: ' + path.relative_to(ROOT).as_posix())
    if errors:
        raise RuntimeError('\n'.join(errors[:30]))
    print(json.dumps(dict(status='verified',files=checked,mode='fast' if args.fast else 'full_sha256',
                          training_counts=counts), indent=2))


def train(args):
    available = presets()
    if args.preset not in available:
        raise ValueError('Unknown preset; use python run.py list')
    config = available[args.preset]
    out = fresh_output(args.out, label=args.preset)
    driver = config['driver']
    if args.quick and driver != 'source':
        raise ValueError('--quick is only for author-replication; use smoke or --seeds 1 for other presets')
    if args.scenarios:
        requested = args.scenarios.split(',')
        if not config.get('scenarios') or not set(requested) <= set(config['scenarios']):
            raise ValueError('--scenarios must be a subset of this preset\'s declared scenarios')
    command = [sys.executable]
    if driver == 'source':
        if args.seeds is not None or args.start is not None or args.scenarios:
            raise ValueError('The author-replication preset has fixed seed cohorts; use --quick for its smoke cohort.')
        command += [WORKSPACE / 'paper_static_replication_20261120/run_replication.py', '--out', out,
                    '--workers', str(args.workers)]
        if args.quick:
            command.append('--smoke')
    elif driver == 'direct_reset':
        command += [ROOT / 'tools/run_direct_reset.py', '--out', out,
                    '--seeds', str(args.seeds if args.seeds is not None else config['seeds']),
                    '--start', str(args.start if args.start is not None else config['start']),
                    '--workers', str(args.workers)]
    else:
        scripts = {
            'study': WORKSPACE / 'research_story_20261120/study.py',
            'study_v1': WORKSPACE / 'research_story_20261120/study_v1.py',
            'strong': WORKSPACE / 'research_story_20261120/study_strong_controls.py',
            'unbounded': WORKSPACE / 'research_story_20261120/study_unbounded_controls.py',
            'validation': WORKSPACE / 'paper_report_20261120/experiments/engine.py',
        }
        command += [scripts[driver], config['phase'], '--out', out,
                    '--seeds', str(args.seeds if args.seeds is not None else config['seeds']),
                    '--start', str(args.start if args.start is not None else config['start']),
                    '--workers', str(args.workers), '--methods', ','.join(config['methods'])]
        if config.get('scenarios'):
            command += ['--scenarios', args.scenarios or ','.join(config['scenarios'])]
        for name, value in config.get('parameters', {}).items():
            command += ['--' + name, str(value)]
    if args.dry_run:
        print(json.dumps(dict(config=config,command=list(map(str,command)),output=str(out)), ensure_ascii=False,indent=2))
        return
    run_logged(command, out)
    (out / 'bundle_config.json').write_text(json.dumps(dict(preset=args.preset,config=config,
        command=list(map(str,command))), indent=2), encoding='utf-8')


def smoke(args):
    import numpy as np
    out = fresh_output(args.out, label='smoke')
    engine_dir = WORKSPACE / 'paper_report_20261120/experiments'
    sys.path.insert(0, str(engine_dir))
    engine = importlib.import_module('engine')
    config = engine.Config(grid_steps=1000, evaluation_episodes=2)
    methods = ['plain','raw_static','state_raw_static','state_raw_matched','return_gate_unbounded','mc_dynamic_all_unbounded']
    engine.run_group((str(out / 'data'), 'u_detour', 50000, methods, config))
    records = [read_json(p) for p in sorted((out / 'data/runs').glob('*.json'))]
    assert len(records) == len(methods)
    assert len({r['prefix_hash'] for r in records}) == 1
    assert all(r['steps'] == 1000 and r['finite'] and r['max_telescoping_error'] < 1e-7 for r in records)
    for path in (out / 'data/models').glob('*.npz'):
        with np.load(path) as data:
            assert all(np.isfinite(data[k]).all() for k in data.files)
    result = dict(status='passed',scenario='u_detour',methods=methods,training_steps=6000,
                  warmup_identical=True,purpose='Execution and numerical checks, not a performance experiment.')
    (out / 'smoke_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result, indent=2));print('Output:', out)


def diagnostic(args):
    out = fresh_output(args.out, label='goal_diagnostic');out.mkdir(parents=True)
    sys.path.insert(0, str(WORKSPACE / 'paper_report_20261120/experiments'))
    module = importlib.import_module('goal_diagnostic');module.HERE = out;module.main()
    expected = read_json(WORKSPACE / 'paper_report_20261120/experiments/goal_diagnostic.json')
    assert read_json(out / 'goal_diagnostic.json') == expected
    print('Exact diagnostic reproduced:', out)


def prepare_paper(out):
    out.mkdir(parents=True)
    for name in ['main.tex','make_figures.py','paper_data.json']:
        shutil.copy2(ROOT / 'paper' / name, out / name)


def figures(args):
    out = fresh_output(args.out, label='figures');prepare_paper(out)
    run_logged([sys.executable,out / 'make_figures.py'], out)


def report(args):
    out = fresh_output(args.out, label='report')
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
    if args.dry_run:
        print(json.dumps(dict(command=command,output=str(out)),indent=2));return
    prepare_paper(out);(out / 'build').mkdir()
    run_logged([sys.executable,out / 'make_figures.py'],out)
    run_logged(command,out,cwd=out)
    if not tectonic and not latexmk:
        run_logged(command,out,cwd=out)
    if not (out / 'build/main.pdf').is_file():raise RuntimeError('Compiler produced no PDF')
    print('PDF:',out / 'build/main.pdf')


def analyze(args):
    out = fresh_output(args.out, category='analysis_results',label=args.dataset)
    input_folder = None
    if args.input:
        if args.dataset not in {'fresh','source'}:
            raise ValueError('--input is supported for fresh or source analysis')
        input_folder = Path(args.input)
        if not input_folder.is_absolute():input_folder = ROOT / input_folder
        input_folder = input_folder.resolve()
        if not (input_folder / 'manifest.json').is_file():
            raise ValueError('Input must contain a completed experiment manifest.json')
    if input_folder is None and read_json(ROOT / 'MANIFEST.json').get('artifact_type') == 'source_repository':
        raise RuntimeError('Frozen raw records are distributed in the offline evidence archive. '
            'For new runs use analyze fresh/source --input runs/<name>; '
            'or restore the offline evidence before recomputing its statistics.')
    if args.dataset == 'all':
        for dataset in ['fresh','controlled','source']:
            run_logged([sys.executable,ROOT / 'run.py','analyze',dataset,'--out',out / dataset],out / ('log_' + dataset))
        return
    if args.dataset == 'source':
        script = WORKSPACE / 'paper_static_replication_20261120/analyze_replication.py'
        command = [sys.executable,script,'--out',input_folder or 'results','--report',out]
    else:
        command = [sys.executable,ROOT / 'tools' / ('recompute_' + args.dataset + '.py'),out]
        if input_folder:command.append(input_folder)
    run_logged(command,out)


def test(args):
    out = fresh_output(args.out,label='tests')
    for name,folder,pattern in [
        ('numerics','research_story_20261120','test_study.py'),
        ('source','paper_static_replication_20261120','test_replication.py')]:
        run_logged([sys.executable,'-m','unittest','discover','-s',WORKSPACE / folder,'-p',pattern,'-v'],out / name)
    print('Both verification suites passed:',out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action',required=True)
    sub.add_parser('list',help='Show contents and training presets').set_defaults(function=list_project)
    p=sub.add_parser('verify',help='Verify package file hashes and frozen trial counts');p.add_argument('--fast',action='store_true');p.set_defaults(function=verify)
    for command,fn in [('smoke',smoke),('diagnostic',diagnostic),('figures',figures),('test',test),('report',report)]:
        p=sub.add_parser(command);p.add_argument('--out');p.set_defaults(function=fn)
        if command=='report':p.add_argument('--dry-run',action='store_true')
    p=sub.add_parser('train');p.add_argument('preset');p.add_argument('--out');p.add_argument('--seeds',type=int);p.add_argument('--start',type=int)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--scenarios');p.add_argument('--quick',action='store_true');p.add_argument('--dry-run',action='store_true');p.set_defaults(function=train)
    p=sub.add_parser('analyze');p.add_argument('dataset',choices=['fresh','controlled','source','all']);p.add_argument('--out');p.add_argument('--input');p.set_defaults(function=analyze)
    args=parser.parse_args()
    if getattr(args,'start',None) is not None and args.start<0:parser.error('--start must be nonnegative')
    for name in ['seeds','workers']:
        if getattr(args,name,None) is not None and getattr(args,name)<=0:parser.error('--'+name+' must be positive')
    try:args.function(args)
    except (ValueError,RuntimeError) as error:parser.exit(1,'Error: '+str(error)+'\n')


if __name__=='__main__':main()
