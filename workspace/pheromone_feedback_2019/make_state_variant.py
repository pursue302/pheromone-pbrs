"""Generate an explicitly separate state-only potential robustness variant.

The primary confirmation source is left unchanged. Only potential aggregation
changes from the selected action's pheromone to the mean across actions.
"""
from pathlib import Path
import difflib

HERE = Path(__file__).resolve().parent
original = (HERE/'experiment.py').read_text(encoding='utf-8')
changed = original
replacements = {
    'old_potential = pher[action] / agent.num_tiling':
        'old_potential = pher.mean() / agent.num_tiling',
    'potential = 0.0 if done else next_pher[ap] / agent.num_tiling':
        'potential = 0.0 if done else next_pher.mean() / agent.num_tiling',
    "sources = [HERE / 'experiment.py', HERE / 'build_port.py']":
        "sources = [HERE / 'experiment_state.py', HERE / 'make_state_variant.py', HERE / 'build_port.py']",
    "author_commit='95786b39b9ee8c5e004e71ccfa790304cbb3f684',":
        "potential_variant='state_mean', author_commit='95786b39b9ee8c5e004e71ccfa790304cbb3f684',",
}
for before, after in replacements.items():
    assert changed.count(before) == 1, before
    changed = changed.replace(before, after)
(HERE/'experiment_state.py').write_text(changed, encoding='utf-8')
(HERE/'STATE_POTENTIAL.patch').write_text(''.join(difflib.unified_diff(
    original.splitlines(True),changed.splitlines(True),fromfile='experiment.py',
    tofile='experiment_state.py')),encoding='utf-8')
print('Generated state-only robustness variant; primary experiment unchanged.')
