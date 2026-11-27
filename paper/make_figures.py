"""Build paper figures and tables from the published data snapshot."""
import argparse
import csv
import json
import sys
from importlib import import_module
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from workspace.plot_style import KEYS, LABEL, TITLE


def tables(data, output):
    PAPER=data['previous_analysis']
    VALID=data['validation']
    TAB=output/'tables'
    TAB.mkdir(exist_ok=True)
    methods=['plain','raw_static','raw_dynamic','return_static','return_gate_unbounded','return_dynamic_all_unbounded','mc_dynamic_all_unbounded']
    z=PAPER['confirmation_joint']['summary'];keys=['u_detour','four_rooms','slippery_rooms','mountain_car']
    lines=[]
    for m in methods:
        cells=[LABEL[m]]
        for k in keys:
            v=z[k][m]['final_return'];cells.append(f"{v['mean']:.2f} $\\pm$ {v['sd']:.2f}")
        lines.append(' & '.join(cells)+r' \\')
    (TAB/'previous_results.tex').write_text('\n'.join(lines)+'\n',encoding='utf8')
    methods=['plain','raw_static','raw_dynamic','state_raw_static','state_raw_matched','return_static','return_gate_unbounded','mc_dynamic_all_unbounded','gate_scale003','gate_scale030']
    lines=[]
    for m in methods:
        label=LABEL.get(m,{'gate_scale003':r'Return-Gate ($c=.03$)','gate_scale030':r'Return-Gate ($c=.30$)'}.get(m,m))
        cells=[label]
        for k in KEYS:
            v=VALID['summary'][k][m]['original_return'];cells.append(f"{v['mean']:.2f} $\\pm$ {v['sd']:.2f}")
        lines.append(' & '.join(cells)+r' \\')
    (TAB/'validation_results.tex').write_text('\n'.join(lines)+'\n',encoding='utf8')
    lines=[]
    for t in VALID['tests']['primary']:
        cells=[TITLE[t['scenario']],LABEL[t['b']],f"{t['difference']:+.2f}",f"[{t['ci95'][0]:+.2f}, {t['ci95'][1]:+.2f}]",f"{t['p_holm']:.3f}"]
        lines.append(' & '.join(cells)+r' \\')
    (TAB/'primary_tests.tex').write_text('\n'.join(lines)+'\n',encoding='utf8')
    compact='\n'.join(lines)
    for full,short in [('U-detour','U'),('Four rooms','F'),('Slippery rooms','S')]:compact=compact.replace(full,short)
    (TAB/'primary_tests_compact.tex').write_text(compact+'\n',encoding='utf8')
    lines=[]
    for k in KEYS:
        cells=[TITLE[k]]
        for m in ['raw_static','raw_dynamic','return_gate_unbounded','state_raw_static','state_raw_matched']:
            cells.append(f"{100*VALID['summary'][k][m]['success_rate']['mean']:.1f}\\%")
        lines.append(' & '.join(cells)+r' \\')
    (TAB/'success.tex').write_text('\n'.join(lines)+'\n',encoding='utf8')
    # Full exported values remain machine-readable in the source package.
    with (output/'results_export.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['cohort','scenario','method','n','mean_original_return','sd','success','auc'])
        for k,zz in VALID['summary'].items():
            for m,v in zz.items():w.writerow(['fresh',k,m,v['original_return']['n'],v['original_return']['mean'],v['original_return']['sd'],v['success_rate']['mean'],v['auc']['mean']])

def main(output):
    output.mkdir(parents=True, exist_ok=True)
    data = json.loads((HERE / 'paper_data.json').read_text())
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    goal = import_module('workspace.01_goal_preservation.plot_goal')
    source = import_module('workspace.02_source_audit.plot_source')
    controlled = import_module('workspace.03_controlled_study.plot_controls')
    navigation = import_module('workspace.04_navigation_validation.plot_navigation')
    navigation.maps(data, figures)
    goal.goal(data, figures)
    navigation.validation(data, figures)
    controlled.controls(data, figures, source.source_panel)
    tables(data, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE)
    main(parser.parse_args().out)
