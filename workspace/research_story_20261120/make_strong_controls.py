"""Freeze confirmed source; add stronger full-history controls explicitly."""
from pathlib import Path
import difflib, hashlib, json
p=Path(__file__).resolve().parent
original=(p/'study.py').read_text(encoding='utf8')
patched=original.replace("method!='mc_dynamic'","method not in ('mc_dynamic','mc_dynamic_all')")
old="('return_dynamic','return_mix','return_gate','return_gate_unbounded','mc_dynamic')"
new="('return_dynamic','return_mix','return_gate','return_gate_unbounded','mc_dynamic','return_dynamic_all','mc_dynamic_all')"
assert old in patched;patched=patched.replace(old,new)
old="live=cfg.scale*online.mean(default)"
new="live=cfg.scale*(memory if method in ('return_dynamic_all','mc_dynamic_all') else online).mean(default)"
assert old in patched;patched=patched.replace(old,new)
old="if method in ('return_dynamic','mc_dynamic'):candidate=live"
new="if method in ('return_dynamic','mc_dynamic','return_dynamic_all','mc_dynamic_all'):candidate=live"
assert old in patched;patched=patched.replace(old,new)
assert patched!=original
(p/'study_strong_controls.py').write_text(patched,encoding='utf8')
(p/'strong_controls.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),patched.splitlines(True),fromfile='study.py',tofile='study_strong_controls.py')),encoding='utf8')
(p/'STRONG_CONTROL_PROTOCOL.txt').write_text('Added after initial confirmation results, without changing the selected candidate.\nThe original pure-online return/MC controls use only post-warmup trajectories. Add stronger dynamic estimates that retain ALL warmup samples and keep updating the same memory, with no data discard. Both weighted ant memory and unweighted MC get this control. Same 80 MC seeds and 30 per grid, same total budgets, features, Q and evaluation. Comparisons are supplementary/exploratory, not replacements for the original 20-test primary family. Initial published Phi is based on all warmup samples; publication every 10 episodes and potential change limit 1 remain the same as the original pure dynamic control. The candidate and its results remain frozen.\n',encoding='utf8')
print(json.dumps(dict(original=hashlib.sha256(original.encode()).hexdigest(),extended=hashlib.sha256(patched.encode()).hexdigest())))
