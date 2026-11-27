"""Produce a concise, evidence-linked finding note after every run finishes."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
primary=json.loads((HERE/'results/summary.json').read_text())
state=json.loads((HERE/'state_results/summary.json').read_text())
sensitivity=json.loads((HERE/'sensitivity_summary.json').read_text())
fresh=json.loads((HERE/'fresh_followup.json').read_text())
final_mc=json.loads((HERE/'final_mc_summary.json').read_text())
manifests=[json.loads((HERE/f/'manifest.json').read_text())
           for f in ('results','state_results','sensitivity_025','sensitivity_4',
                     'followup_feedback4','followup_count025',
                     'final_mc_feedback4','final_mc_count025')]
for manifest in manifests:
    assert manifest['status']=='complete'
    for name,expected in manifest['source_hashes'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected, name
total_steps=sum(m['unique_training_steps'] for m in manifests)
total_runs=sum(len(m['methods'])*len(m['scenarios'])*m['config']['seeds'] for m in manifests)
primary_significant=sum(c['dynamic_feedback__minus__dynamic']['original_discounted_return']
                        ['positive_after_holm'] for c in primary['contrasts'].values())
state_significant=sum(c['dynamic_feedback__minus__dynamic']['original_discounted_return']
                      ['positive_after_holm'] for c in state['contrasts'].values())
local=final_mc['contrasts']['dynamic_feedback__minus__dynamic']
local_confirmed=local['mean']>0 and local['p_holm_four_comparisons']<.05
local_sentence=('最终独立单任务验证支持强反馈改善动态基线，但不代表跨任务有效或已经证明独立创新。'
                if local_confirmed else
                '最终独立单任务验证也未充分确立强反馈优于动态基线；不能把探索性改善直接当成已确认创新。')
lines=['基于 2019 年作者代码的探索反馈实验：结论与证据','',
       '结论：固定强度1的跨任务实验没有稳定的最终回报优势。'+local_sentence,
       '这不是证明所有探索反馈均无效；结论限定于本次机制、任务、预算和参数范围。','',
       f'正式主实验：5 个任务、10 种方法、20 个配对种子，共 1,000 次训练。',
       '主实验原始折扣回报：五项反馈对照的 95% 配对区间均包含零；'
       f'Holm 校正后正向比较数 {primary_significant}/5。',
       f'状态势稳健性检查：独立 10 个种子，正向主比较数 {state_significant}/5。',
       f'含两档探索性强度检查及新种子验证，共 {total_runs:,} 次训练、{total_steps:,} 个环境交互步。',
       '强度检查复用相同十个种子的动态基线；没有重复花费基线采样预算。',
       '另有不混入正式统计的小样本试运行及数值检查。','',
       '主实验最终原始回报均值（越高越好）',
       '任务 | SARSA | 动态基线 | 动态+普通计数 | 动态+候选反馈']
for scenario,methods in primary['methods'].items():
    values=[methods[m]['original_return']['mean'] for m in
            ('sarsa','dynamic','dynamic_count','dynamic_feedback')]
    lines.append(scenario+' | '+' | '.join(f'{v:.3f}' for v in values))
lines += ['', '具体发现与解释',
    '1. 覆盖增加与策略收益脱节。Mountain Car 中候选反馈平均多访问约 26.9 个离散状态动作格，'
    '但最终原始回报从 -277.84 变为 -310.84。覆盖指标不能替代原始奖励。',
    '2. 候选机制没有稳定的动态特异性证据：动态/固定交互差值的区间不能排除零，'
    '也没有稳定证明收益超出同样采样反馈对普通 SARSA 的帮助。',
    '3. 主实验中普通 SARSA 的最终原始回报均值在五个任务里都高于动态+候选反馈。'
    '修补动态方法后略有局部改善，也不能忽略这个更简单的基线。',
    '4. 均匀信息素时门控反馈为零，所以它无法帮助发现首次成功轨迹；它只能在已有信息素差异后介入。',
    '5. 信息素集中可能意味着正确利用，也可能意味着错误锁定；当前集中度门控没有可靠区分二者。'
    '这是设计限制，不是由本实验单独证明的失败中介机制。',
    '', '探索性强度检查（各强度十个相同种子，未经独立确认）']
for gain in (.25,1.,4.):
    subset=[r for r in sensitivity['rows'] if r['gain']==gain and r['method']=='dynamic_feedback']
    lines.append('gain='+str(gain)+': '+ '; '.join(
        f'{r["scenario"]} 差值 {r["discounted_difference"]["mean"]:+.3f}, CI {r["discounted_difference"]["ci95"]}'
        for r in subset))
lines += ['', '状态势版本的学习速度（次要指标，区间未做多重校正）']
for scenario,contrasts in state['contrasts'].items():
    d=contrasts['dynamic_feedback__minus__dynamic']['auc_original_return']
    lines.append(f'{scenario}: AUC 差值 {d["mean"]:+.3f}, 95% CI {d["ci95"]}')
lines += ['', '选定强度的全新种子验证（Mountain Car，10 个全新种子）',
          (HERE/'fresh_followup_zh.txt').read_text(encoding='utf-8'),
          '', '最终独立验证（Mountain Car，另外20个全新种子）',
          (HERE/'final_mc_report_zh.txt').read_text(encoding='utf-8'),
          '', '来源与限制',
    '算法基于作者公开的 PAApproximatedSarsaLambdaAgent；实际复用上游全局更新，'
    '数值核对上游一条完整轨迹的权重、资格迹和局部更新。代码兼容修改见 PYTHON3_PORT.patch。',
    '主实验保留动作相关势；状态势检查只改变聚合方式，差异见 STATE_POTENTIAL.patch。',
    '共同修正、持续探索率、任务奖励、冻结定义均已记录。不能称为论文表 7 的原样复现，'
    '也没有证明论文结果由覆盖不足或某个源码细节导致。',
    '没有做独立的覆盖中介干预或相同数据的势质量回放，故不把覆盖变化解释成已验证的因果链。',
    '不同任务的回报尺度不同；不把它们直接平均来制造总体优势。',
    '所有训练源码哈希已与各组 manifest 核对一致；图表已检查，固定组信息素变动为零。',
    '', '对课程项目的判断',
    '可以把本次工作写成一项有对照和边界分析的研究；贡献表述必须限定到实际得到支持的任务、指标与强度，'
    '不能宣称探索反馈普遍修复了动态塑形。',
    '后续若继续设计，应让反馈判断原始回报是否停滞、势评分是否可靠；'
    '不能仅因为某个分支少访问、信息素集中，就假定增加探索会改善策略。']
(HERE/'findings_zh.txt').write_text('\n'.join(lines),encoding='utf-8')
(HERE/'completion_audit.json').write_text(json.dumps(dict(total_confirmatory_and_sensitivity_runs=total_runs,
    total_training_steps=total_steps,primary_positive_after_holm=primary_significant,
    state_positive_after_holm=state_significant,training_source_hashes_match=True,
    fresh_followup=fresh,
    final_mc=final_mc,
    primary_audits=primary['audits'],state_audits=state['audits']),indent=2),encoding='utf-8')
print('\n'.join(lines[:16]))
