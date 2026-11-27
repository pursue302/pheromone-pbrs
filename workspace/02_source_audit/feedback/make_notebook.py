from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
cells = []
def markdown(text):
    cells.append({'cell_type':'markdown','id':f'cell-{len(cells)}','metadata':{},'source':text.splitlines(True)})
def code(text):
    cells.append({'cell_type':'code','id':f'cell-{len(cells)}','execution_count':None,'metadata':{},'outputs':[],
                  'source':text.splitlines(True)})

markdown('''# 基于 2019 年作者代码的探索反馈实验

问题：信息素引导采样可能过早集中。根据局部信息素集中程度和动作访问次数提供探索反馈，是否改善原始奖励表现？

来源：[2019 年论文](https://doi.org/10.1007/s10732-019-09408-x)、[作者公开代码](https://github.com/SB-BISS/RLACOSarsaLambda)。这是兼容移植、共同修正和受控扩展；不是论文表 7 的完整复现。

先看结果，不预设反馈有效。主实验与状态势稳健性检查使用不同种子，分别统计。''')
code('''from pathlib import Path
import json
from IPython.display import display, Image, HTML
HERE = Path.cwd()
if not (HERE / 'README.txt').exists():
    HERE = Path(r'E:\\homework\\462\\02_source_audit/feedback')
primary = json.loads((HERE/'results/summary.json').read_text(encoding='utf-8'))
state = json.loads((HERE/'state_results/summary.json').read_text(encoding='utf-8'))
print('主实验训练步数:', primary['manifest']['unique_training_steps'])
print('状态势检查训练步数:', state['manifest']['unique_training_steps'])
print('数值核对:', primary['audits'])
''')
markdown('''## 方法与预算

保留作者 SARSA、tile coding、局部信息素调整、全局蒸发和最佳轨迹沉积。

候选反馈：$b(s,a)=C(s)/\\sqrt{1+N(s,a)}$，其中 $C(s)$ 是局部信息素的相对集中程度。普通计数探索则不乘 $C(s)$。反馈只改变采样动作评分；TD 奖励和信息素质量评价均不包含探索加分。

固定组在前 25% 的总预算内学习信息素，之后同时冻结局部与全局更新，保留 Q。动态组继续更新。全部预热交互计入预算，所有组保持相同基础随机探索。另设更强随机探索和不塑形的 SARSA 对照。

重要限制：均匀信息素下 $C(s)=0$，因此该候选机制不帮助首次成功前的探索。动作相关势版本不直接适用状态 PBRS 的目标不变定理；状态势检查将各动作信息素取均值，其他规则不变。''')
code('''print(json.dumps(primary['manifest']['config'], ensure_ascii=False, indent=2))
print((HERE/'EXPERIMENT_PROTOCOL.txt').read_text(encoding='utf-8'))
''')
markdown('''## 全部场景的原始奖励结果

负回报越接近零越好。训练和评估均保留失败及超时，不从均值中删除。''')
code('''def result_table(summary):
    methods = ['sarsa','fixed','dynamic','dynamic_count','dynamic_feedback','dynamic_random']
    rows = ['<table><tr><th>任务</th><th>方法</th><th>原始回报</th><th>原始折扣回报</th><th>成功率</th><th>回报 AUC</th></tr>']
    for scenario, values in summary['methods'].items():
        for method in methods:
            m = values[method]
            data = [scenario, method] + [f"{m[k]['mean']:.3f}" for k in
                ['original_return','original_discounted_return','success_rate','auc_original_return']]
            rows.append('<tr>' + ''.join('<td>'+v+'</td>' for v in data) + '</tr>')
    display(HTML(''.join(rows)+'</table>'))
result_table(primary)
''')
code('''display(Image(filename=str(HERE/'results/learning_curves.png')))
display(Image(filename=str(HERE/'results/feedback_effects.png')))
''')
markdown('''## 配对检验与动态特异性

主指标是最终原始折扣回报。95% bootstrap 区间为点区间；主检验另做五任务 Holm 校正。AUC、覆盖率及其他对照帮助解释结果，不替代主指标。

动态特异性：$(动态+反馈-动态)-(固定+反馈-固定)$。

超出采样收益：$(动态+反馈-动态)-(SARSA+反馈-SARSA)$。这两个交互差值均为探索性分析。''')
code('''for scenario, comparisons in primary['contrasts'].items():
    print('\\n', scenario)
    for name in ['dynamic_feedback__minus__dynamic','dynamic_feedback__minus__dynamic_count',
                 'dynamic_specificity','extra_over_sampling_only']:
        d = comparisons[name]['original_discounted_return']
        print(name, '差值', round(d['mean'],4), '95% CI', d['ci95'],
              '主检验校正p', d.get('p_holm_five_scenarios','探索性'))
''')
markdown('''## 状态势稳健性检查

这里使用独立的十个种子；只把动作相关势评分改为动作信息素均值构成的状态势。与主实验分开报告。''')
code('''result_table(state)
display(Image(filename=str(HERE/'state_results/learning_curves.png')))
display(Image(filename=str(HERE/'state_results/feedback_effects.png')))
''')
code('''print((HERE/'results/report_zh.txt').read_text(encoding='utf-8'))
print((HERE/'state_results/report_zh.txt').read_text(encoding='utf-8'))
''')
markdown('''## 探索性强度检查与最终判断

补测强度 0.25 和 4，与强度 1 使用相同的十个种子及动态基线。全部结果都报告，不把事后选出的最好强度当作独立验证。''')
code('''print((HERE/'sensitivity_report_zh.txt').read_text(encoding='utf-8'))
print((HERE/'fresh_followup_zh.txt').read_text(encoding='utf-8'))
display(Image(filename=str(HERE/'fresh_followup.png')))
print((HERE/'final_mc_report_zh.txt').read_text(encoding='utf-8'))
display(Image(filename=str(HERE/'final_mc.png')))
print((HERE/'findings_zh.txt').read_text(encoding='utf-8'))
''')
markdown('''## 源码与复现

阅读 `README.txt`、`PYTHON3_PORT.patch`、`STATE_POTENTIAL.patch` 和两个实验协议。

`test_experiment.py` 将一条完整固定轨迹在作者移植版和加速实现中运行，比较权重、资格迹与局部信息素；另外检查上游 tile 投影一致性、真正冻结、回报抵消和零反馈对照。

运行命令见 README。不要从这里的曲线直接声称方法首次提出或普遍有效；创新论证还需要具体设计差异和相关工作核对。''')
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
     'language_info':{'name':'python','version':'3.13.5'}},'nbformat':4,'nbformat_minor':5}
(HERE/'Pheromone_feedback_2019.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
print('Wrote Pheromone_feedback_2019.ipynb')
