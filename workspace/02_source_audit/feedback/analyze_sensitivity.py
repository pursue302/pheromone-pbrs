from pathlib import Path
import json
import numpy as np
from analyze import statistics

HERE=Path(__file__).resolve().parent
base_folder=HERE/'results'
base_manifest=json.loads((base_folder/'manifest.json').read_text())
scenarios=base_manifest['scenarios']
rows=[]
for gain,folder in [(.25,'sensitivity_025'),(1.,'results'),(4.,'sensitivity_4')]:
    path=HERE/folder
    manifest=json.loads((path/'manifest.json').read_text())
    assert manifest['status']=='complete'
    for scenario in scenarios:
        for method in ('dynamic_count','dynamic_feedback'):
            differences=[]
            returns=[]
            auc_differences=[]
            for seed in range(1000,1010):
                target=json.loads((path/'runs'/f'{scenario}__{method}__{seed}.json').read_text())
                base=json.loads((base_folder/'runs'/f'{scenario}__dynamic__{seed}.json').read_text())
                returns.append(target['final']['original_return'])
                differences.append(target['final']['original_discounted_return']-base['final']['original_discounted_return'])
                def auc(run):
                    x=np.array([r['step'] for r in run['records']])
                    y=np.array([r['original_return'] for r in run['records']])
                    return np.trapezoid(y,x)/(x[-1]-x[0])
                auc_differences.append(auc(target)-auc(base))
            result=dict(scenario=scenario,method=method,gain=gain,
                        mean_original_return=float(np.mean(returns)),
                        discounted_difference=statistics(differences,scenario+method+str(gain)),
                        auc_difference=statistics(auc_differences,scenario+method+str(gain)+'auc'))
            rows.append(result)
output=dict(status='exploratory strength check; same ten paired seeds; no independent confirmation',
            rows=rows,extra_training_steps=sum(json.loads((HERE/f/'manifest.json').read_text())['unique_training_steps']
                                             for f in ('sensitivity_025','sensitivity_4')))
(HERE/'sensitivity_summary.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
lines=['反馈强度敏感性检查（探索性，不并入预设主检验）','任务 | 方法 | 强度 | 原始回报均值 | 相对动态基线的原始折扣回报差值及95%区间']
for r in rows:
    d=r['discounted_difference']
    lines.append(f'{r["scenario"]} | {r["method"]} | {r["gain"]} | {r["mean_original_return"]:.3f} | '
                 f'{d["mean"]:+.4f} [{d["ci95"][0]:+.4f}, {d["ci95"][1]:+.4f}]')
(HERE/'sensitivity_report_zh.txt').write_text('\n'.join(lines),encoding='utf-8')
print('\n'.join(lines))
