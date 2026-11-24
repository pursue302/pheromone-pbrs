from pathlib import Path
import json
import numpy as np
from analyze import statistics, holm

HERE=Path(__file__).resolve().parent
methods=['sarsa','sarsa_feedback','dynamic','dynamic_feedback','dynamic_count']
seeds=list(range(4000,4020))
data={}
for method in methods:
    folder='final_mc_count025' if method=='dynamic_count' else 'final_mc_feedback4'
    manifest=json.loads((HERE/folder/'manifest.json').read_text())
    assert manifest['status']=='complete'
    data[method]=[json.loads((HERE/folder/'runs'/f'mountain_car__{method}__{s}.json').read_text()) for s in seeds]

def auc(run):
    x=np.array([r['step'] for r in run['records']])
    y=np.array([r['original_return'] for r in run['records']])
    return float(np.trapezoid(y,x)/(x[-1]-x[0]))

summary=dict(scenario='mountain_car',seeds=seeds,methods={},contrasts={},
             status='final fixed-size selected-task independent check; no further tuning')
for method,runs in data.items():
    m={k:float(np.mean([r['final'][k] for r in runs])) for k in
       ('original_return','original_discounted_return','success_rate')}
    m['auc_original_return']=float(np.mean([auc(r) for r in runs]))
    summary['methods'][method]=m
comparisons=[('dynamic_feedback','dynamic'),('dynamic_feedback','dynamic_count'),
             ('dynamic_feedback','sarsa')]
for a,b in comparisons:
    name=a+'__minus__'+b
    differences=[x['final']['original_return']-y['final']['original_return'] for x,y in zip(data[a],data[b])]
    summary['contrasts'][name]=statistics(differences,'final_mc'+name)
    summary['contrasts'][name]['auc_difference']=statistics(
        [auc(x)-auc(y) for x,y in zip(data[a],data[b])],'final_mc_auc'+name)
d=[(a['final']['original_return']-b['final']['original_return'])-
   (c['final']['original_return']-e['final']['original_return']) for a,b,c,e in
   zip(data['dynamic_feedback'],data['dynamic'],data['sarsa_feedback'],data['sarsa'])]
summary['contrasts']['extra_over_sampling_only']=statistics(d,'final_mc_interaction')
keys=list(summary['contrasts'])
adjusted=holm([summary['contrasts'][k]['p_sign_flip'] for k in keys])
for key,p in zip(keys,adjusted):
    summary['contrasts'][key]['p_holm_four_comparisons']=float(p)
summary['training_steps']=sum(json.loads((HERE/f/'manifest.json').read_text())['unique_training_steps']
                               for f in ('final_mc_feedback4','final_mc_count025'))
(HERE/'final_mc_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
lines=['Mountain Car 最终独立验证：固定二十个全新种子',
       '种子 4000--4019；每种方法 60,000 训练步。反馈强度4，普通计数强度.25。',
       '没有把前面的试运行、强度选择或十种子验证混进统计。该单任务验证不能代替跨任务结论。','',
       '方法 | 原始回报均值 | 成功率 | 原始回报AUC']
for key,m in summary['methods'].items():
    lines.append(f'{key} | {m["original_return"]:.3f} | {m["success_rate"]:.3f} | {m["auc_original_return"]:.3f}')
lines += ['', '配对原始回报差值（gamma=1）；区间是点区间，检验做四项 Holm 校正']
for key,d in summary['contrasts'].items():
    lines.append(f'{key}: {d["mean"]:+.3f}, 95% CI {d["ci95"]}, Holm p={d["p_holm_four_comparisons"]:.5f}')
(HERE/'final_mc_report_zh.txt').write_text('\n'.join(lines),encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
labels=['SARSA','SARSA + feedback','Dynamic','Dynamic + feedback','Dynamic + count']
colors=['#555555','#4575b4','#d73027','#1b9e77','#984ea3']
for key,label,color in zip(methods,labels,colors):
    matrix=np.array([[r['original_return'] for r in run['records']] for run in data[key]])
    x=np.array([r['step'] for r in data[key][0]['records']])/1000
    mean=matrix.mean(axis=0)
    axes[0].plot(x,mean,label=label,color=color,lw=2)
axes[0].set(xlabel='Training interactions (thousands)',ylabel='Original return',title='Learning curves')
axes[0].grid(alpha=.15)
axes[0].legend(frameon=False,fontsize=8)
matrix=np.array([[r['final']['original_return'] for r in data[m]] for m in methods])
for j in range(len(seeds)):
    axes[1].plot(np.arange(5),matrix[:,j],'-o',color='gray',alpha=.25,lw=.6,ms=2)
axes[1].scatter(np.arange(5),matrix.mean(axis=1),c=colors,s=75,zorder=3)
axes[1].set(xticks=np.arange(5),xticklabels=['SARSA','SARSA+FB','Dynamic','Dynamic+FB','Dynamic+count'],
            ylabel='Final original return',title='Paired final outcomes; large dots = means')
axes[1].tick_params(axis='x',labelrotation=20)
axes[1].grid(axis='y',alpha=.15)
fig.suptitle('Final independent Mountain Car check: 20 previously unused seeds')
fig.savefig(HERE/'final_mc.png',dpi=170)
plt.close(fig)
print('\n'.join(lines))
