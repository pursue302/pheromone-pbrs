from pathlib import Path
import ast
import difflib
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def main():
    target=HERE/'paper_author';target.mkdir(exist_ok=True)
    (target/'__init__.py').write_text('')
    original=ROOT/'author_direct_comparison_20261120/author_agents'
    for name in ('ApproximatedSarsaLambdaAgent.py','HAApproximatedSarsaLambdaAgent.py','representation.py','LICENSE'):
        shutil.copyfile(original/name,target/name)
    raw=(original/'pa_original.py').read_text()
    frozen=raw.replace('if self.heuristic_activate:','if self.heuristic_activate and self.heuristic_dynamic:')
    assert raw.count('if self.heuristic_activate:')==1
    versions={'source':raw,'frozen':frozen}
    for name,s in list(versions.items()):
        term='shaping = (self.gamma*potential- old_potential)/self.num_tiling'
        assert s.count(term)==1
        versions[name+'_eq38']=s.replace(term,'shaping = self.rho*potential - old_potential')
    for name,s in versions.items():
        (target/('pa_'+name+'.py')).write_text(s,encoding='utf8')
        (HERE/('PA_'+name+'.patch')).write_text(''.join(difflib.unified_diff(raw.splitlines(True),s.splitlines(True),fromfile='author_PA.py',tofile='pa_'+name+'.py')),encoding='utf8')
    legacy=HERE/'gym_legacy';legacy.mkdir(exist_ok=True);(legacy/'__init__.py').write_text('')
    s=(HERE/'upstream/gym_v0105_mountain_car.py').read_text()
    s=s.replace('import gym\n','').replace('from gym import spaces','from .compat import gym, spaces').replace('from gym.utils import seeding','from . import seeding')
    (legacy/'mountain_car.py').write_text(s,encoding='utf8')
    s=(HERE/'upstream/gym_v0105_seeding.py').read_text().replace('from gym import error','from . import error')
    (legacy/'seeding.py').write_text(s,encoding='utf8')
    (legacy/'error.py').write_text('class Error(Exception):\n    pass\n')
    # Use the author's actual table statistics, including zero-based argmin and
    # its time-to-best convention (sum excludes the best episode itself).
    s=(HERE/'upstream/PlotResultsM3D.py').read_text()
    body='def find_best_results'+s.split('def find_best_results',1)[1].split('def print_results',1)[0]
    ast.parse(body)
    (HERE/'author_metrics.py').write_text('import numpy as np\n\n'+body,encoding='utf8')
    hashes={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob('*') if p.is_file()}
    (HERE/'source_versions.json').write_text(json.dumps(dict(author_commit='95786b39b9ee8c5e004e71ccfa790304cbb3f684',hashes=hashes,
        changes='Only Python3/import/tile-seed compatibility in original classes. Frozen adds one local dynamic guard. eq38 changes only the TrueOnline shaping expression. No trace, initial/terminal potential or success-invariant fixes.'),indent=2),encoding='utf8')
    print('Prepared full author learners, exact historical Gym methods and original metric function.')


if __name__=='__main__':main()
