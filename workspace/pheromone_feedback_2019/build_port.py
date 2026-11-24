"""Rebuild a Python 3 compatibility port without modifying the author snapshot."""
from pathlib import Path
import difflib
import json
import re

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'project_2_8_research/author_code/RLACOSarsaLambda_95786b39'

def build():
    out = HERE / 'author_port'
    out.mkdir(exist_ok=True)
    (out / '__init__.py').write_text('', encoding='utf-8')
    diffs = []
    paths = ['agents/ApproximatedSarsaLambdaAgent.py',
             'agents/PAApproximatedSarsaLambdaAgent.py']
    for path in paths:
        original = (SOURCE / path).read_text(encoding='utf-8')
        port = original.replace('from gym.spaces import discrete\n', '').replace('import gym\n', '')
        port = port.replace('from tilecoding.representation import TileCoding',
                            'from .representation import TileCoding')
        port = port.replace('from ApproximatedSarsaLambdaAgent import ApproximatedSarsaLambdaAgent',
                            'from .ApproximatedSarsaLambdaAgent import ApproximatedSarsaLambdaAgent')
        port = port.replace('print self.strategy', 'print(self.strategy)')
        port = port.replace('rnd_stream = np.random.RandomState()',
                            'rnd_stream = np.random.RandomState(self.config.get("tile_seed", 0) + 17*i)')
        (out / Path(path).name).write_text(port, encoding='utf-8')
        diffs.extend(difflib.unified_diff(original.splitlines(True), port.splitlines(True),
                     fromfile='upstream/' + path, tofile='author_port/' + Path(path).name))
    original = (HERE / 'upstream_tilecoding/representation.py').read_text(encoding='utf-8')
    port = original.replace('from itertools import chain, izip', 'from itertools import chain\nizip = zip')
    port = re.sub(r'\bxrange\b', 'range', port)
    port = re.sub(r'np\.int\b', 'int', port)
    port = port.replace('np.cumsum(map(lambda x: x.size, self.tilings[:-1]))',
                        'np.cumsum(list(map(lambda x: x.size, self.tilings[:-1])))')
    port = port.replace('np.hstack(chain((t(state) for t in self.tilings),\n'
                        '                                      [np.zeros((state.shape[0], 1), dtype=\'int\')]))',
                        'np.hstack(tuple(chain((t(state) for t in self.tilings),\n'
                        '                                      [np.zeros((state.shape[0], 1), dtype=\'int\')])))')
    port = port.replace('np.hstack((t(state) for t in self.tilings))',
                        'np.hstack(tuple(t(state) for t in self.tilings))')
    (out / 'representation.py').write_text(port, encoding='utf-8')
    diffs.extend(difflib.unified_diff(original.splitlines(True), port.splitlines(True),
                 fromfile='upstream_tilecoding/representation.py', tofile='author_port/representation.py'))
    (HERE / 'PYTHON3_PORT.patch').write_text(''.join(diffs), encoding='utf-8')
    (HERE / 'COPYING_AUTHOR.txt').write_text((SOURCE / 'LICENSE').read_text(encoding='utf-8'), encoding='utf-8')
    print(json.dumps({'ported_author_files': paths, 'tilecoding_commit':
          '3fd90d31539e92d229996f48c4227ef5ea4663f6'}, indent=2))

if __name__ == '__main__':
    build()
