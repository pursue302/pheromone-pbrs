"""Match the selected candidate's unlimited publication amplitude in controls."""
from pathlib import Path
import difflib
p=Path(__file__).resolve().parent;src=(p/'study_strong_controls.py').read_text(encoding='utf8');s=src
s=s.replace("('mc_dynamic','mc_dynamic_all')","('mc_dynamic','mc_dynamic_all','mc_dynamic_all_unbounded')")
s=s.replace("('return_dynamic','return_mix','return_gate','return_gate_unbounded','mc_dynamic','return_dynamic_all','mc_dynamic_all')",
    "('return_dynamic','return_mix','return_gate','return_gate_unbounded','mc_dynamic','return_dynamic_all','mc_dynamic_all','return_dynamic_all_unbounded','mc_dynamic_all_unbounded','return_mix_unbounded')")
s=s.replace("('return_dynamic_all','mc_dynamic_all') else online)","('return_dynamic_all','mc_dynamic_all','return_dynamic_all_unbounded','mc_dynamic_all_unbounded') else online)")
s=s.replace("('return_dynamic','mc_dynamic','return_dynamic_all','mc_dynamic_all'):candidate=live",
    "('return_dynamic','mc_dynamic','return_dynamic_all','mc_dynamic_all','return_dynamic_all_unbounded','mc_dynamic_all_unbounded'):candidate=live")
s=s.replace("elif method=='return_mix':candidate=", "elif method in ('return_mix','return_mix_unbounded'):candidate=")
s=s.replace("if method=='return_gate_unbounded' else cfg.limit", "if method in ('return_gate_unbounded','return_dynamic_all_unbounded','mc_dynamic_all_unbounded','return_mix_unbounded') else cfg.limit")
assert src!=s
(p/'study_unbounded_controls.py').write_text(s,encoding='utf8')
(p/'unbounded_controls.diff').write_text(''.join(difflib.unified_diff(src.splitlines(True),s.splitlines(True),fromfile='study_strong_controls.py',tofile='study_unbounded_controls.py')),encoding='utf8')
(p/'UNBOUNDED_CONTROL_PROTOCOL.txt').write_text('Additional supplementary controls, after outcomes, without altering the candidate. Remove the publication amplitude limit from full-history weighted dynamic, full-history unweighted MC dynamic, and the 50/50 static+new-data mixture, to match the selected unbounded gate. Otherwise differences could reflect clipping rather than mixing. Same 80 MC and 30 per grid paired seeds and all fixed budgets. Report exploratory paired comparisons separately, never substitute for the failed original primary family.\n',encoding='utf8')
