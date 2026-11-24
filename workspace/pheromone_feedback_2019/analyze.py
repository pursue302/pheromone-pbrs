"""Summarize every registered scenario, paired contrasts, and original rewards."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
LABELS = {'mountain_car':'Mountain Car', 'open':'Open grid', 'u_detour':'U detour',
          'four_rooms':'Four rooms', 'slippery_rooms':'Slippery rooms'}
SHORT = {'sarsa':'SARSA', 'sarsa_count':'SARSA + count', 'sarsa_feedback':'SARSA + feedback',
         'fixed':'Frozen', 'fixed_count':'Frozen + count', 'fixed_feedback':'Frozen + feedback',
         'dynamic':'Dynamic', 'dynamic_count':'Dynamic + count',
         'dynamic_feedback':'Dynamic + feedback', 'dynamic_random':'Dynamic + random (.1)'}


def statistics(differences, tag):
    d = np.asarray(differences, dtype=float)
    seed = int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    bootstrap = d[rng.integers(len(d), size=(20000, len(d)))].mean(axis=1)
    signs = rng.choice(np.array([-1., 1.]), size=(50000, len(d)))
    shuffled = (signs * d).mean(axis=1)
    p = (1 + np.count_nonzero(np.abs(shuffled) >= abs(d.mean()) - 1e-12)) / (len(shuffled) + 1)
    return dict(mean=float(d.mean()), ci95=[float(x) for x in np.quantile(bootstrap, [.025,.975])],
                p_sign_flip=float(p), n=len(d), wins=int(np.sum(d>1e-9)),
                losses=int(np.sum(d < -1e-9)), ties=int(np.sum(np.abs(d)<=1e-9)))


def holm(values):
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1., running)
    return adjusted


def metrics(run):
    rows = run['records']
    x = np.array([r['step'] for r in rows])
    result = dict(run['final'])
    for name in ('original_return','original_discounted_return','success_rate'):
        y = np.array([r[name] for r in rows])
        result['auc_'+name] = float(np.trapezoid(y,x)/(x[-1]-x[0]))
        result['tail_'+name] = float(y[-3:].mean())
    result['first_goal_step'] = run['first_goal_step'] if run['first_goal_step'] is not None else x[-1]
    result['training_successes'] = run['training_successes']
    return result


def analyze(out, plots=True):
    manifest = json.loads((out/'manifest.json').read_text())
    runs = [json.loads(p.read_text()) for p in sorted((out/'runs').glob('*.json'))]
    expected = len(manifest['methods'])*len(manifest['scenarios'])*manifest['config']['seeds']
    if len(runs) != expected or manifest['status'] != 'complete':
        raise RuntimeError(f'Incomplete experiment: {len(runs)}/{expected}')
    index = {(r['scenario'],r['method'],r['seed']):r for r in runs}
    seeds = list(range(manifest['config']['seed_start'],
                       manifest['config']['seed_start']+manifest['config']['seeds']))
    summary = dict(manifest=manifest, methods={}, contrasts={}, audits={})
    numerical_metrics = ('original_return','original_discounted_return','success_rate',
                         'auc_original_return','auc_original_discounted_return','tail_original_return',
                         'visited_bins','visited_state_actions','mean_concentration',
                         'cumulative_local_potential_drift','first_goal_step','training_successes')
    comparisons = [('dynamic_feedback','dynamic'), ('dynamic_count','dynamic'),
                   ('dynamic_feedback','dynamic_count'), ('dynamic_feedback','dynamic_random'),
                   ('dynamic_feedback','sarsa'), ('dynamic_feedback','sarsa_feedback'),
                   ('fixed_feedback','fixed'), ('sarsa_feedback','sarsa')]
    flat = []
    for scenario in manifest['scenarios']:
        summary['methods'][scenario] = {}
        summary['contrasts'][scenario] = {}
        for method in manifest['methods']:
            samples = [metrics(index[scenario,method,s]) for s in seeds]
            aggregate = {}
            for name in numerical_metrics:
                data = np.array([s[name] for s in samples],float)
                aggregate[name] = dict(mean=float(data.mean()), sd=float(data.std(ddof=1)) if len(data)>1 else 0.)
            summary['methods'][scenario][method] = aggregate
        for target, base in comparisons:
            if target not in manifest['methods'] or base not in manifest['methods']:
                continue
            tag = target+'__minus__'+base
            comparison = {}
            for name in numerical_metrics:
                d = [metrics(index[scenario,target,s])[name]-metrics(index[scenario,base,s])[name] for s in seeds]
                stat = statistics(d, scenario+tag+name)
                comparison[name] = stat
                flat.append(dict(scenario=scenario, target=target, baseline=base, metric=name,
                                 mean_difference=stat['mean'], ci_low=stat['ci95'][0],ci_high=stat['ci95'][1],
                                 p_sign_flip=stat['p_sign_flip'],n=stat['n']))
            summary['contrasts'][scenario][tag] = comparison
        interactions = [('dynamic_specificity', 'fixed_feedback','fixed'),
                        ('extra_over_sampling_only', 'sarsa_feedback','sarsa')]
        if 'dynamic_feedback' in manifest['methods'] and 'dynamic' in manifest['methods']:
            for tag, other_target, other_base in interactions:
                if other_target not in manifest['methods'] or other_base not in manifest['methods']:
                    continue
                comparison = {}
                for name in ('original_return','original_discounted_return','auc_original_return',
                             'auc_original_discounted_return'):
                    d = []
                    for s in seeds:
                        a,b,c,e = [metrics(index[scenario,m,s])[name] for m in
                                   ('dynamic_feedback','dynamic',other_target,other_base)]
                        d.append((a-b)-(c-e))
                    comparison[name] = statistics(d,scenario+tag+name)
                summary['contrasts'][scenario][tag] = comparison
    primary = []
    for scenario in manifest['scenarios']:
        contrast = summary['contrasts'][scenario].get('dynamic_feedback__minus__dynamic')
        if contrast:
            primary.append((scenario,contrast['original_discounted_return']))
    if primary:
        adjusted = holm([x[1]['p_sign_flip'] for x in primary])
        for (scenario, stat), p in zip(primary,adjusted):
            stat['p_holm_five_scenarios'] = float(p)
            stat['positive_after_holm'] = bool(stat['mean']>0 and p<.05)
    prefix_mismatches = 0
    for scenario in manifest['scenarios']:
        for suffix in ('','_count','_feedback'):
            if 'fixed'+suffix not in manifest['methods'] or 'dynamic'+suffix not in manifest['methods']:
                continue
            budget = manifest['config']['mc_steps'] if scenario=='mountain_car' else manifest['config']['grid_steps']
            freeze = int(budget*manifest['config']['freeze_fraction'])
            for seed in seeds:
                fixed = index[scenario,'fixed'+suffix,seed]
                dynamic = index[scenario,'dynamic'+suffix,seed]
                for a,b in zip(fixed['records'], dynamic['records']):
                    if a['step'] <= freeze and a != b:
                        prefix_mismatches += 1
    summary['audits'] = dict(
        all_runs_use_equal_scenario_budgets=all(r['final']['step']==(
            manifest['config']['mc_steps'] if r['scenario']=='mountain_car' else manifest['config']['grid_steps']) for r in runs),
        max_frozen_pheromone_change=max([r['frozen_pheromone_change'] for r in runs if r['frozen_pheromone_change'] is not None] or [0]),
        max_telescoping_error=max(r['max_telescoping_error'] for r in runs),
        fixed_dynamic_prefix_mismatches=prefix_mismatches)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if flat:
        with (out/'paired_comparisons.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    state_only = manifest.get('potential_variant') == 'state_mean'
    lines = ['2019 年作者代码探索反馈实验：正式结果' + ('（状态势稳健性检查）' if state_only else '（作者动作势版本）'),
             '这是作者算法的受控移植和扩展，不是论文表 7 的数值复现。',
             f'种子数：{len(seeds)}；任务数：{len(manifest["scenarios"])}；方法数：{len(manifest["methods"])}。',
             f'全部训练交互：{manifest["unique_training_steps"]:,}；含冻结前的采集阶段。',
             '主指标：最终部署策略的原始折扣回报。探索加分不进入奖励评价。',
             '95% 区间是种子配对差值的 bootstrap 区间，未经多重比较校正；主检验另给五任务 Holm 校正 p 值。',
             'AUC 为检查点原始回报曲线按环境步数积分后除以预算；尾部均值是最后三个检查点的补充指标。','']
    for scenario in manifest['scenarios']:
        lines += [LABELS[scenario], '方法 | 最终原始回报 | 最终原始折扣回报 | 成功率 | 回报 AUC']
        for method in manifest['methods']:
            m=summary['methods'][scenario][method]
            lines.append(f'{method} | {m["original_return"]["mean"]:.3f} | '
                         f'{m["original_discounted_return"]["mean"]:.4f} | '
                         f'{m["success_rate"]["mean"]:.3f} | {m["auc_original_return"]["mean"]:.3f}')
        c=summary['contrasts'][scenario].get('dynamic_feedback__minus__dynamic')
        if c:
            for metric in ('original_discounted_return','auc_original_return','visited_state_actions'):
                d=c[metric]
                lines.append(f'反馈减动态基线，{metric}: {d["mean"]:+.4f}, 95% CI '
                             f'[{d["ci95"][0]:+.4f}, {d["ci95"][1]:+.4f}]')
            lines.append(f'主检验 Holm p={c["original_discounted_return"]["p_holm_five_scenarios"]:.5f}')
        for name in ('dynamic_specificity','extra_over_sampling_only'):
            if name in summary['contrasts'][scenario]:
                d=summary['contrasts'][scenario][name]['original_discounted_return']
                lines.append(f'{name} 原始折扣回报交互差值: {d["mean"]:+.4f}, 95% CI {d["ci95"]}')
        lines.append('')
    lines += ['数值核对：'+json.dumps(summary['audits'],ensure_ascii=False), '',
              '解释边界：覆盖率上升不能直接等同于原始回报改善；未做中介干预，不能把相关变化当成因果机制证明。',
              ('本组使用动作信息素均值形成状态势；与作者动作势结果单独统计。' if state_only else
               '本组保留作者的动作相关势评分；不能据此宣称适用状态 PBRS 的最优策略不变定理。'),
              '冻结对照保留原 Q，并把全部预热交互计入预算；它不是另行预训练后重置 Q 的静态迁移实验。',
              '参数、共同修正、环境差异和对照定义见 EXPERIMENT_PROTOCOL.txt。']
    (out/'report_zh.txt').write_text('\n'.join(lines),encoding='utf-8')
    if plots:
        render(out,summary,index,seeds)
    for scenario,stat in primary:
        print(scenario, 'delta discounted return',round(stat['mean'],4), 'CI',stat['ci95'],
              'Holm p',round(stat['p_holm_five_scenarios'],5))
    print('Audits',summary['audits'])
    return summary


def render(out, summary, index, seeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=['sarsa','fixed','dynamic','dynamic_count','dynamic_feedback','dynamic_random']
    colors={'sarsa':'#555555','fixed':'#4575b4','dynamic':'#d73027',
            'dynamic_count':'#984ea3','dynamic_feedback':'#1b9e77','dynamic_random':'#e6ab02'}
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for ax,scenario in zip(axes.flat,summary['manifest']['scenarios']):
        for method in selected:
            if method not in summary['manifest']['methods']:
                continue
            samples=[index[scenario,method,s]['records'] for s in seeds]
            x=np.array([r['step'] for r in samples[0]])
            y=np.array([[r['original_return'] for r in sample] for sample in samples])
            mean=y.mean(axis=0)
            se=y.std(axis=0,ddof=1)/np.sqrt(len(y)) if len(y)>1 else np.zeros_like(mean)
            ax.plot(x/1000,mean,label=SHORT[method],color=colors[method],lw=1.8)
            ax.fill_between(x/1000,mean-1.96*se,mean+1.96*se,color=colors[method],alpha=.10)
        ax.axvline(x[-1]/1000*summary['manifest']['config']['freeze_fraction'],ls=':',color='gray')
        ax.set(title=LABELS[scenario],xlabel='Training interactions (thousands)',ylabel='Original return')
        ax.grid(alpha=.15)
    axes.flat[-1].axis('off')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles,labels,loc='center',frameon=False,fontsize=11)
    fig.suptitle('Author-derived pheromone shaping: original-reward learning curves\n'
                 f'{len(seeds)} independent paired seeds; shading = approximate 95% pointwise intervals',fontsize=13)
    fig.savefig(out/'learning_curves.png',dpi=170)
    plt.close(fig)
    scenarios=summary['manifest']['scenarios']
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    for ax,metric,title in [(axes[0],'original_discounted_return','Original discounted return'),
                            (axes[1],'visited_state_actions','Visited state-action bins')]:
        means=[];low=[];high=[]
        for scenario in scenarios:
            d=summary['contrasts'][scenario]['dynamic_feedback__minus__dynamic'][metric]
            means.append(d['mean']);low.append(max(0,d['mean']-d['ci95'][0]));high.append(max(0,d['ci95'][1]-d['mean']))
        ax.errorbar(means,np.arange(len(scenarios)),xerr=[low,high],fmt='o',color='#1b9e77',capsize=4)
        ax.axvline(0,color='gray',lw=1)
        ax.set(yticks=np.arange(len(scenarios)),yticklabels=[LABELS[s] for s in scenarios],
               xlabel='Gated feedback minus dynamic baseline',title=title)
        ax.grid(axis='x',alpha=.15)
    fig.suptitle('Paired mean differences and 95% bootstrap intervals')
    fig.savefig(out/'feedback_effects.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='results')
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    analyze(HERE/args.out,not args.no_plots)
