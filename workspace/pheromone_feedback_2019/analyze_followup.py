from pathlib import Path
import json
import numpy as np
from analyze import statistics, holm

HERE=Path(__file__).resolve().parent
folders={'dynamic':'followup_feedback4','dynamic_feedback':'followup_feedback4',
         'dynamic_count':'followup_count025'}
methods=list(folders)
data={}
for method,folder in folders.items():
    manifest=json.loads((HERE/folder/'manifest.json').read_text())
    assert manifest['status']=='complete'
    data[method]=[json.loads((HERE/folder/'runs'/f'mountain_car__{method}__{s}.json').read_text())
                  for s in range(3000,3010)]
summary=dict(seeds=list(range(3000,3010)),scenario='mountain_car',methods={},contrasts={})
for method,runs in data.items():
    summary['methods'][method]={k:float(np.mean([r['final'][k] for r in runs]))
                               for k in ('original_return','original_discounted_return','success_rate')}
comparisons=[('dynamic_feedback','dynamic'),('dynamic_count','dynamic'),
             ('dynamic_feedback','dynamic_count')]
for target,base in comparisons:
    d=[a['final']['original_return']-b['final']['original_return'] for a,b in zip(data[target],data[base])]
    summary['contrasts'][target+'__minus__'+base]=statistics(d,'fresh'+target+base)
adjusted=holm([summary['contrasts'][a+'__minus__'+b]['p_sign_flip'] for a,b in comparisons])
for (a,b),p in zip(comparisons,adjusted):
    summary['contrasts'][a+'__minus__'+b]['p_holm_three_comparisons']=float(p)
summary['training_steps']=sum(json.loads((HERE/f/'manifest.json').read_text())['unique_training_steps']
                              for f in set(folders.values()))
(HERE/'fresh_followup.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
lines=['Mountain Car 新种子验证：强反馈与普通计数',
       '种子 3000--3009；每种方法 60,000 训练步；探索加分不计入原始回报。',
       '强度由之前十个敏感性种子选择，这里的种子没有参与选择。',
       '该单任务验证不能替代跨任务检验。','', '方法 | 原始回报均值 | 成功率']
for method,m in summary['methods'].items():
    lines.append(f'{method} | {m["original_return"]:.3f} | {m["success_rate"]:.3f}')
lines += ['', '配对原始回报差值（Mountain Car 的 gamma=1，因此也是原始折扣回报差值）']
for key,d in summary['contrasts'].items():
    lines.append(f'{key}: {d["mean"]:+.3f}, 95% CI {d["ci95"]}, Holm p={d["p_holm_three_comparisons"]:.5f}')
(HERE/'fresh_followup_zh.txt').write_text('\n'.join(lines),encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
matrix=np.array([[r['final']['original_return'] for r in data[m]] for m in methods])
for j in range(matrix.shape[1]):
    ax.plot(np.arange(3),matrix[:,j],'-o',color='#999999',alpha=.45,lw=.9,ms=3)
ax.plot(np.arange(3),matrix.mean(axis=1),'-o',color='#1b9e77',lw=3,ms=8,label='Mean over 10 fresh seeds')
ax.set(xticks=np.arange(3),xticklabels=['Dynamic baseline','Gated feedback (4)','Ordinary count (.25)'],
       ylabel='Original return',title='Fresh-seed Mountain Car check: 60,000 training steps')
ax.grid(axis='y',alpha=.2)
ax.legend(frameon=False)
fig.savefig(HERE/'fresh_followup.png',dpi=170)
plt.close(fig)
print('\n'.join(lines))
