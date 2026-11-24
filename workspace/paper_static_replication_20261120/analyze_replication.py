"""All registered cells, author-defined metrics and paired static contrasts."""
from pathlib import Path
import argparse
import hashlib
import html
import json
import copy
import pickle
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from author_metrics import find_best_results

HERE=Path(__file__).resolve().parent
DATA=HERE/'results'
REPORT=HERE
sys.path.insert(0,str(HERE.parent/'pheromone_feedback_2019'))
from analyze import statistics,holm

NAMES={'sarsa':'作者SARSA','dynamic':'作者动态PA','static_flag':'作者原静态开关',
 'static_frozen':'真正冻结／成熟先验','static_200':'真正冻结／200回合先验',
 'sarsa_same_tiles':'SARSA／同一tile编码','dynamic_same_tiles':'动态PA／同一tile编码'}
PAPER={'sarsa':dict(ABS=103.2,AIBS=96.8,ATBS=10.,ATST=17.6),
 'dynamic':dict(ABS=102.2,AIBS=89.,ATBS=15.,ATST=27.),
 'static_frozen':dict(ABS=101.4,AIBS=75.,ATBS=14.,ATST=27.)}
METRICS={'ABS':'values','AIBS':'indexes','ATBS':'times','ATST':'execution_times'}


def aggregate(rr):
    out=dict(n=len(rr))
    for label,key in METRICS.items():
        a=np.array([r['author_metrics'][key] for r in rr])
        out[label]=dict(mean=float(a.mean()),sd=float(a.std(ddof=1)),author_print_SEM=float(a.std(ddof=0)/np.sqrt(len(a))))
    out.update(actual_best_steps=float(np.mean([min(e['length'] for e in r['episodes']) for r in rr])),
       final_original_return=float(np.mean([r['final_greedy']['original_return'] for r in rr])),
       final_success=float(np.mean([r['final_greedy']['success'] for r in rr])),
       student_steps=float(np.mean([r['steps'] for r in rr])),
       donor_steps=float(np.mean([r['donor_steps'] for r in rr])),
       last20_training_return=float(np.mean([np.mean([e['original_return'] for e in r['episodes'][-20:]]) for r in rr])))
    return out


def contrast(a,b,tag):
    aa={r['seed']:r for r in a};bb={r['seed']:r for r in b};assert aa.keys()==bb.keys()
    return {label:statistics([aa[s]['author_metrics'][key]-bb[s]['author_metrics'][key] for s in sorted(aa)],tag+label)
        for label,key in METRICS.items()}


def audit():
    folder=DATA;m=json.loads((folder/'manifest.json').read_text());assert m['status']=='complete' and not m['smoke']
    for relative,expected in m['source_hashes'].items():assert hashlib.sha256((HERE/relative).read_bytes()).hexdigest()==expected,relative
    rr=[json.loads(p.read_text()) for p in sorted((folder/'runs').glob('*.json'))]
    assert len(rr)==len(list((folder/'models').glob('*.npz')))==920
    assert sum(r['steps'] for r in rr)==m['physical_student_steps']
    assert all(r['finite'] and r['backend']=='actual_author_learn' and len(r['episodes'])==200 for r in rr)
    # Pickle load can canonicalize ndarray layout and change byte serialization.
    # Compare the actual loaded coder payload, then verify projection equality.
    import run_replication as runner
    loaded_coder_hashes={};reencoding=[]
    for p in (folder/'priors').glob('*.pkl'):
        with p.open('rb') as f:payload=pickle.load(f)
        meta=json.loads(p.with_suffix('.json').read_text())
        h=hashlib.sha256(pickle.dumps(payload['coders'])).hexdigest();loaded_coder_hashes[p.stem]=h
        if h!=meta['coder_sha256']:reencoding.append(p.stem)
        teacher=json.loads((folder/'teachers'/f'{meta["bank"]}.json').read_text())
        _,_,reconstructed,_,_=runner.setup(teacher['kind'],teacher['seed'],'donor')
        for a in range(3):
            left=payload['coders'][a];right=reconstructed[a]
            assert left.size==right.size and len(left.tilings)==len(right.tilings)
            for x,y in zip(left.tilings,right.tilings):
                for key in ('offset','ntiles','index_offset'):np.testing.assert_array_equal(getattr(x,key),getattr(y,key))
                for xbound,ybound in zip(x.state_range,y.state_range):np.testing.assert_array_equal(xbound,ybound)
        clone=copy.deepcopy(copy.deepcopy(payload['coders']))
        probes=np.random.RandomState(78).uniform([-1.2,-.07],[.6,.07],(300,2))
        for s in probes:
            for a in range(3):np.testing.assert_array_equal(payload['coders'][a](s),clone[a](s))
    for r in rr:
        assert r['steps']==sum(e['length'] for e in r['episodes'])
        real=find_best_results([r['episode_seconds']],[r['author_last_steps']])
        for key,value in real.items():assert abs(float(value[0])-r['author_metrics'][key])<1e-8
        if r['role']=='static':assert r['live_change']==0.
        if r['donor_reference']:
            assert r['coder_sha256']==loaded_coder_hashes[r['bank']+'__'+r.get('stage','mature')],r['id']
            if not r.get('same_tiles'):assert r['static_tau_sha256']==r['donor_reference']['tau_sha256']
        # Actual rewards differ only in the explicitly named paper-text cell.
        for e in r['episodes']:
            expected=-float(e['length'])+(2 if r['kind']=='paper_text' and e['success'] else 0)
            assert e['original_return']==expected
    for p in (folder/'trajectories').glob('*.npz'):
        r=next(r for r in rr if r['id']==p.stem)
        with np.load(p) as d:
            assert len(d['reward'])==r['steps']
            assert np.count_nonzero(d['done'])==200
            assert d['reward'].sum()==sum(e['original_return'] for e in r['episodes'])
    return m,rr,dict(source_hashes_match=True,all_200_complete=True,all_finite=True,
        author_metrics_exactly_recalculated=True,loaded_prior_and_coders_verified=True,
        deserialized_coder_hashes=loaded_coder_hashes,pickle_reencoding_byte_changes=reencoding,
        coder_geometry_reconstructed_from_teacher_seed=True,
        frozen_versions_unchanged=True,all_rewards_verified=True,
        physical_student_steps=m['physical_student_steps'],physical_teacher_steps=m['physical_teacher_steps'])


def tables(headers,rows):return '\n'.join([' | '.join(headers)]+[' | '.join(map(str,row)) for row in rows])


def figures(groups):
    fig,axes=plt.subplots(1,2,figsize=(13,5.2),layout='constrained')
    colors={'sarsa':'#222222','dynamic':'#777fc1','static_flag':'#d29c45','static_frozen':'#167f82','static_200':'#bb6b77'}
    for name,color in colors.items():
        y=np.array([r['author_last_steps'] for r in groups['main'][name]])
        for ax,z in zip(axes,[y,np.minimum.accumulate(y,axis=1)]):
            mu=z.mean(axis=0);se=z.std(axis=0,ddof=1)/np.sqrt(len(z));x=np.arange(200)
            ax.plot(x,mu,label=name,color=color,lw=1.35);ax.fill_between(x,mu-1.96*se,mu+1.96*se,color=color,alpha=.07)
    axes[0].set(title='Training trajectory (raw author metric)',yscale='log',ylabel='Source last_steps')
    axes[1].set(title='Best trajectory found so far',ylabel='Minimum source last_steps')
    for ax in axes:ax.set_xlabel('Episode index (author zero-based)');ax.grid(alpha=.15)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=5,frameon=False,fontsize=9)
    fig.suptitle('Paper-near source implementation:100 repetitions,200 complete episodes\nOne fixed mature pheromone file; every method uses source exploration decay',fontsize=11)
    fig.savefig(REPORT/'paper_near_learning_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for name,color in [('sarsa_same_tiles','#222222'),('dynamic_same_tiles','#777fc1'),('static_frozen','#167f82')]:
        y=np.array([r['author_last_steps'] for r in groups['main'][name]])
        axes[0].plot(np.arange(200),y.mean(axis=0),label=name,color=color)
        axes[1].plot(np.arange(200),np.minimum.accumulate(y,axis=1).mean(axis=0),label=name,color=color)
    axes[0].set(yscale='log',title='Same tile coding, training trajectories',ylabel='Source last_steps')
    axes[1].set(title='Same tile coding, best so far',ylabel='Minimum source last_steps')
    for ax in axes:ax.set_xlabel('Episode index');ax.grid(alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
    fig.savefig(REPORT/'same_tile_controls.png',dpi=160);plt.close(fig)


def report(out,groups):
    ss=out['summary']['main'];parts=[]
    def add(title,*p):parts.append((title,list(p)))
    qualitative=out['effect_assessment']['lower_mean_AIBS_than_both'] and out['effect_assessment']['no_worse_mean_ABS_than_both']
    judgement='出现静态提前学习的均值趋势，统计支持另见下表' if qualitative else '未复现论文的静态提前学习优势'
    lead=(f'本轮主对照{judgement}：真正冻结静态AIBS={ss["static_frozen"]["AIBS"]["mean"]:.2f}，'
          f'SARSA={ss["sarsa"]["AIBS"]["mean"]:.2f}，动态PA={ss["dynamic"]["AIBS"]["mean"]:.2f}（均为100次重复，越小越好）。'
          '这是对可获得作者代码和重新生成先验的复现尝试，不能据此断定原论文错误。')
    add('一、这次对齐了什么',
        '本轮仅检验论文表7的静态信息素效果。直接执行作者普通SARSA、HA、PA完整learn()；没有混合、探索反馈、自适应控制或另写TD循环。主队列100个新种子，每种200完整回合。',
        '主环境直接使用Gymv0.10.5的MountainCar step/reset/seed函数体，目标x=.5，全部奖励-1，float64观察和float32观察边界，历史RandomState/hash随机数。具体原Gym版本未在论文说明，故它是有来源的历史近似。alpha=.5/10、gamma1、lambda.9、10x10tile与10层、rho.99。所有方法eps=.025，依作者脚本每步乘.99。',
        '静态共用一份独立预训练HA导出的tau和配套原tile编码；调用作者两个setter。学生Q=0、资格迹=0、历史轨迹空、不复制老师Q或激活标志。保留作者原初始势、终止势、跨回合资格迹和成功判断。真正冻结版本只补一个局部dynamic判断，原静态开关另列。')
    add('二、先验是否比以前充分',
        tables(['文件','实际预训练回合','实际采样步数','稳定标准通过','最后100训练回合均值步数','最后100成功率'],
            [[b,r['episodes'],r['steps'],r['stable'],f'{r["last100_mean_length"]:.2f}',f'{r["last100_success"]:.1%}'] for b,r in out['teachers'].items()]),
        '至少1000回合，每200回合检查训练表现平台，最多2000；具体标准在PROTOCOL.txt。通过只表示满足预设经验稳定条件，不等于理论收敛。教师种子事先固定，没有按学生成绩挑选。另保留同一老师200回合快照，主对照未将先验采集称为免费。')
    add('三、主队列与论文表7',
        '下列ABS/AIBS/ATBS/ATST完全调用作者PlotResultsM3D.py的find_best_results：ABS=min(last_steps)，AIBS=argmin，从0计；ATBS按原脚本只累计最佳回合之前的耗时。它与“包含最佳回合的真正到达时间”不同，两者不能混称。普通SARSA/PA的last_steps是实际步数-1。',
        tables(['方法','论文ABS','本机ABS','论文AIBS','本机AIBS','论文ATBS秒','本机ATBS秒','本机ATST秒'],
            [[NAMES[n],p['ABS'],f'{ss[n]["ABS"]["mean"]:.2f}',p['AIBS'],f'{ss[n]["AIBS"]["mean"]:.2f}',p['ATBS'],f'{ss[n]["ATBS"]["mean"]:.2f}',f'{ss[n]["ATST"]["mean"]:.2f}'] for n,p in PAPER.items()]),
        tables(['方法','样本数','ABS均值±原作者SEM','AIBS均值±原作者SEM','学生交互均值','最终固定策略原始回报','最终固定策略成功率'],
            [[NAMES[n],r['n'],f'{r["ABS"]["mean"]:.2f} ± {r["ABS"]["author_print_SEM"]:.2f}',f'{r["AIBS"]["mean"]:.2f} ± {r["AIBS"]["author_print_SEM"]:.2f}',f'{r["student_steps"]:.0f}',f'{r["final_original_return"]:.2f}',f'{r["final_success"]:.1%}'] for n,r in ss.items()]),
        '原作者打印的Interval在公开脚本中是std/sqrt(n)，因此这里明确标作SEM；配对区间另用bootstrap95%。曲线未平滑。绝对运行时间受硬件、8进程和等价tile投影优化影响，不用来声称复现论文的秒数。最终策略栏是另做10个固定起点的原作者act(eps=0)评估，不替代训练中的最好轨迹指标。')
    add('四、静态加速是否出现',
        lead,
        f'同一老师的200回合先验和成熟先验，静态AIBS均值分别为{ss["static_200"]["AIBS"]["mean"]:.2f}和{ss["static_frozen"]["AIBS"]["mean"]:.2f}。均值差及统计支持以配对表为准；区间跨零不构成加速证据，也不能反过来证明静态必然更差。',
        json.dumps(out['effect_assessment'],ensure_ascii=False,indent=2),
        '差值定义为真正冻结静态减相应基线，负数较好。两项主比较×ABS/AIBS共4项Holm；置信区间是100个学生种子的配对bootstrap，只条件于固定先验A，不代表不同先验之间的全部不确定性。',
        tables(['比较','指标','均值差','配对95%区间','4项Holm p'],
            [[tag,k,f'{r[k]["mean"]:+.3f}',str([round(x,3) for x in r[k]['ci95']]),f'{r[k]["p_holm4"]:.5f}'] for tag,r in out['primary'].items() for k in ('ABS','AIBS')]))
    add('五、固定编码与先验文件敏感性',
        'sarsa_same_tiles、dynamic_same_tiles只使用老师的随机tile布局，不加载训练信息素。这能判断单份静态文件的优势是否同时来自特征布局。下列匹配编码比较属于预设诊断，其区间不与主检验混合。',
        '由于AIBS追踪的是各方法自己的最佳轨迹，目标质量不相同，同编码对照仍需结合ABS、区间和完整曲线判断，不能单凭AIBS均值宣布加速。',
        tables(['同编码比较','ABS差','AIBS差','AIBS配对95%区间'],[[tag,f'{r["ABS"]["mean"]:+.3f}',f'{r["AIBS"]["mean"]:+.3f}',str([round(x,3) for x in r['AIBS']['ci95']])] for tag,r in out['same_tile'].items()]),
        tables(['独立先验','学生n','ABS均值','AIBS均值','最终原始回报'],[[n,r['n'],f'{r["ABS"]["mean"]:.2f}',f'{r["AIBS"]["mean"]:.2f}',f'{r["final_original_return"]:.2f}'] for n,r in out['prior_summary'].items()]),
        'B/C是另外两个事先指定的老师，分别30次学生训练。B/C共用22000起始学生种子，A主实验使用20000起始种子，因此A与B/C间差异还包含学生样本流差异，不能全归因于文件。它们不能合并成“300个独立先验”；只有3份历史环境先验。')
    add('六、论文文字／代码歧义对照',
        '论文文字写目标x=.6、到达奖励+1；公开MountainCar脚本调用Gym默认环境，目标x=.5、全程-1。这里另有paper_text环境，只更改目标和终点奖励，动力学仍按作者mc_model及旧Gym的先更新速度再更新位置。',
        '论文式38是rho*Pnext-Pold；公开PA TrueOnline是(gamma*Pnext-Pold)/10。原式对照只替换这一行，不选择额外scale。其余源码行为不变，所有20种子结果都保留；不是筛出赢家后替换主基线。',
        tables(['环境/公式','方法','学生n','ABS','AIBS','训练最后20回合回报','最终固定策略回报'],
            [[tag,n,r['n'],f'{r["ABS"]["mean"]:.2f}',f'{r["AIBS"]["mean"]:.2f}',f'{r["last20_training_return"]:.2f}',f'{r["final_original_return"]:.2f}'] for tag,cell in out['sensitivity_cells'].items() for n,r in cell.items()]))
    add('七、成本、来源和完整性',
        f'正式学生训练920次，教师4次，训练交互总数{out["audit"]["physical_student_steps"]+out["audit"]["physical_teacher_steps"]:,}。先验A一次采集{out["teachers"]["historical_A"]["steps"]:,}步。单个冷启动任务应加上全部先验成本；复用100次时可另外报摊销，不能将学生提前找到轨迹直接说成总成本下降。',
        json.dumps(out['audit'],ensure_ascii=False,indent=2),
        '原静态pickle未在所查固定提交中找到，文件提交历史查询及releases也未找到对应分发。我们重新生成，不能声称取得作者原文件。原示例脚本当前选用另一APHARS模型，本轮明确执行其公开PA backward类，不声称整份示例脚本原封运行。',
        '论文：https://link.springer.com/content/pdf/10.1007/s10732-019-09408-x.pdf；作者代码：https://github.com/SB-BISS/RLACOSarsaLambda/tree/95786b39b9ee8c5e004e71ccfa790304cbb3f684；旧环境：https://github.com/openai/gym/blob/v0.10.5/gym/envs/classic_control/mountain_car.py。',
        '结果：results/runs、models、priors、teachers；源码：paper_author、gym_legacy；明确差异：PA_*.patch；计划：PROTOCOL.txt；原作者统计函数：author_metrics.py；全部统计：summary.json。')
    text='论文静态信息素效果复现尝试\n\n'+lead+'\n\n'+'\n\n'.join(t+'\n\n'+'\n\n'.join(pp) for t,pp in parts)
    (REPORT/'复现结果.txt').write_text(text,encoding='utf8')
    body=['<h1>论文静态信息素效果复现尝试</h1><p>100次主重复 · 200完整回合 · 固定成熟先验 · 作者完整学习函数</p><p><a href="复现结果.txt">文字结果</a> · <a href="summary.json">全部数值</a> · <a href="PROTOCOL.txt">事前实验设置</a></p>']
    body.append('<p style="padding:16px;background:#fff2dc;border-left:4px solid #bd842e">'+html.escape(lead)+'</p>')
    for title,pp in parts:
        body.append('<h2>'+html.escape(title)+'</h2>')
        for p in pp:
            ls=p.splitlines()
            if len(ls)>1 and all(' | ' in l for l in ls):
                body.append('<div class="scroll"><table>')
                for i,l in enumerate(ls):
                    tag='th' if i==0 else 'td';body.append('<tr>'+''.join('<'+tag+'>'+html.escape(v)+'</'+tag+'>' for v in l.split(' | '))+'</tr>')
                body.append('</table></div>')
            else:body.append('<p>'+html.escape(p).replace('\n','<br>')+'</p>')
        if title.startswith('三、'):body.append('<img src="paper_near_learning_curves.png" alt="原作者指标学习曲线">')
        if title.startswith('五、'):body.append('<img src="same_tile_controls.png" alt="同编码对照">')
    css='body{font-family:Segoe UI,Microsoft YaHei,sans-serif;background:#f1f5f7;color:#203c48;line-height:1.85;margin:0}article{max-width:1250px;margin:22px auto;padding:34px;background:white}h1,h2{color:#145567}h2{margin-top:34px;border-bottom:1px solid #dbe5e8;padding-bottom:8px}p{overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #d6e2e6;padding:9px;text-align:left}th{background:#eaf3f5}tr:nth-child(even){background:#f7fafb}.scroll{overflow:auto;margin:20px 0}img{max-width:100%;height:auto}a{color:#147e90}@media(max-width:700px){article{margin:0;padding:16px}}'
    (REPORT/'复现结果.html').write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>论文静态信息素复现</title><style>'+css+'</style></head><body><article>'+''.join(body)+'</article></body></html>',encoding='utf8')


def main():
    m,rr,checks=audit();groups={c:{} for c in ('main','prior','sensitivity')}
    for r in rr:groups[r['cohort']].setdefault(r['method'],[]).append(r)
    summary={c:{n:aggregate(a) for n,a in group.items()} for c,group in groups.items()}
    primary={};stats=[]
    for base in ('sarsa','dynamic'):
        tag='static_frozen__minus__'+base
        primary[tag]=contrast(groups['main']['static_frozen'],groups['main'][base],tag)
        stats.extend(primary[tag][k] for k in ('ABS','AIBS'))
    for s,p in zip(stats,holm([s['p_sign_flip'] for s in stats])):s['p_holm4']=float(p)
    same={base:contrast(groups['main']['static_frozen'],groups['main'][base],base) for base in ('sarsa_same_tiles','dynamic_same_tiles')}
    prior={'historical_A':summary['main']['static_frozen'],**summary['prior']}
    cells={}
    for kind,formula in [('historical','eq38'),('paper_text','source'),('paper_text','eq38')]:
        cells[kind+'__'+formula]={role:summary['sensitivity'][kind+'__'+(role if role=='sarsa' else role+'_'+formula)] for role in ('sarsa','dynamic','static')}
    frozen=summary['main']['static_frozen']
    assessment=dict(conditional_prior='historical_A',
       lower_mean_AIBS_than_both=all(frozen['AIBS']['mean']<summary['main'][b]['AIBS']['mean'] for b in ('sarsa','dynamic')),
       no_worse_mean_ABS_than_both=all(frozen['ABS']['mean']<=summary['main'][b]['ABS']['mean'] for b in ('sarsa','dynamic')),
       AIBS_both_Holm_p_below05=all(primary['static_frozen__minus__'+b]['AIBS']['p_holm4']<.05 and primary['static_frozen__minus__'+b]['AIBS']['mean']<0 for b in ('sarsa','dynamic')),
       numeric_reproduction_of_original_static_file=False)
    teachers={p.stem:json.loads(p.read_text()) for p in (DATA/'teachers').glob('*.json')}
    for r in teachers.values():
        for key in ('episodes_data','author_last_steps','episode_seconds','final_greedy'):r.pop(key,None)
    out=dict(summary=summary,primary=primary,same_tile=same,prior_summary=prior,sensitivity_cells=cells,
        effect_assessment=assessment,teachers=teachers,audit=checks,paper_table7=PAPER,labels=NAMES)
    (REPORT/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
    figures(groups);report(out,groups)
    print(json.dumps(dict(effect=assessment,main=summary['main'],primary=primary,sensitivity=cells),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results');p.add_argument('--report',default=None);a=p.parse_args()
    DATA=HERE/a.out
    REPORT=HERE/a.report if a.report else (HERE if a.out=='results' else DATA/'report')
    REPORT.mkdir(parents=True,exist_ok=True)
    main()
