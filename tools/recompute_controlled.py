from pathlib import Path
import json, collections, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
BASE=Path(__file__).resolve().parents[1]
HERE=BASE/'workspace/research_story_20261120';ROOT=BASE/'workspace'
REPORT_OUT=Path(sys.argv[1]).resolve();REPORT_OUT.mkdir(parents=True,exist_ok=True)


def read(folder):
    return [json.loads(p.read_text(encoding='utf8')) for p in sorted((folder/'runs').glob('*.json'))]

def stats(a):
    a=np.asarray(a,dtype=float)
    return dict(mean=float(a.mean()),sd=float(a.std(ddof=1)) if len(a)>1 else 0.,n=len(a))

def paired(x,y,seed=75831):
    ids=sorted(set(x)&set(y));d=np.array([x[s]-y[s] for s in ids])
    rng=np.random.default_rng(seed);n=len(d)
    boot=np.empty(20000);null=np.empty(20000)
    for begin in range(0,20000,1000):
        boot[begin:begin+1000]=d[rng.integers(n,size=(1000,n))].mean(axis=1)
        null[begin:begin+1000]=(d*rng.choice([-1.,1.],size=(1000,n))).mean(axis=1)
    p=float((1+np.count_nonzero(abs(null)>=abs(d.mean())-1e-12))/20001)
    return dict(n=n,difference=float(d.mean()),ci95=list(map(float,np.percentile(boot,[2.5,97.5]))),p=p,
        wins=int((d>1e-9).sum()),ties=int((abs(d)<=1e-9).sum()),losses=int((d< -1e-9).sum()))

def holm(rows):
    order=sorted(range(len(rows)),key=lambda i:rows[i]['p']);last=0.
    for rank,i in enumerate(order):
        last=max(last,min(1.,rows[i]['p']*(len(rows)-rank)));rows[i]['p_holm']=last
    return rows

def ownrepair():
    rows=read(HERE/'repair')+read(HERE/'direct_reset');source=read(ROOT/'paper_static_replication_20261120/results')
    for r in source:
        if r.get('cohort')=='main' and r['method'] in ('sarsa','static_frozen','dynamic'):
            r['method']='source_'+r['method'];r['final']=r['final_greedy'];r['final']['success_rate']=r['final']['success'];rows.append(r)
    groups=collections.defaultdict(list)
    for r in rows:groups[r['method']].append(r)
    summary={m:dict(ABS=stats([r['author_metrics']['values'] for r in rr]),
        AIBS=stats([r['author_metrics']['indexes'] for r in rr]),
        final_return=stats([r['final']['original_return'] for r in rr]),
        success=stats([r['final']['success_rate'] for r in rr]),steps=stats([r['steps'] for r in rr])) for m,rr in groups.items()}
    tests=[]
    for a,b in [('actual_source_reset_only','source_sarsa'),('reset','source_sarsa'),('canonical','reset'),
        ('static_reset','source_static_frozen'),('static_canonical','source_static_frozen'),('static_canonical','canonical')]:
        if a in groups and b in groups:
            tests.append(dict(a=a,b=b,**paired({r['seed']:r['final']['original_return'] for r in groups[a]},
                {r['seed']:r['final']['original_return'] for r in groups[b]})))
    holm(tests)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    show=['source_sarsa','actual_source_reset_only','canonical','source_static_frozen','static_canonical']
    colors=['#b73e3e','#238c70','#2876ad','#9f6a3a','#7451a8']
    for m,c in zip(show,colors):
        if m not in groups:continue
        rr=groups[m];v=np.array([[e['original_return'] for e in r['episodes']] for r in rr]);
        # Rolling training return is distinct from frozen-policy evaluation.
        smooth=np.array([np.convolve(x,np.ones(10)/10,'valid') for x in v])
        axes[0].plot(np.arange(10,201),smooth.mean(axis=0),label=m,color=c)
        ys=[r['final']['original_return'] for r in rr];x=show.index(m)
        axes[1].bar(x,np.mean(ys),color=c);axes[1].errorbar(x,np.mean(ys),yerr=1.96*np.std(ys,ddof=1)/np.sqrt(len(ys)),color='black',capsize=4)
    axes[0].set(xlabel='Completed training episodes',ylabel='Original training return (rolling 10)');axes[0].legend(fontsize=7)
    axes[1].set_xticks(range(len(show)),show,rotation=25,ha='right',fontsize=7);axes[1].set_ylabel('Final greedy policy: original return')
    fig.tight_layout();fig.savefig(REPORT_OUT/'repair_evidence.png',dpi=180);plt.close(fig)
    return dict(summary=summary,tests=tests)

def phase(folder):
    rows=read(folder);groups=collections.defaultdict(list)
    for r in rows:
        # Normalized AUC: larger original return is better. Evaluation steps
        # are common and excluded from the interaction budget.
        y=np.array([x['original_return'] for x in r['records']]);x=np.array([x['step'] for x in r['records']])
        r['auc']=float(np.trapezoid(y,x)/(x[-1]-x[0]));groups[r['scenario'],r['method']].append(r)
    summary={}
    for (k,m),rr in groups.items():
        summary.setdefault(k,{})[m]=dict(final_return=stats([r['final']['original_return'] for r in rr]),
            final_discounted_return=stats([r['final']['original_discounted_return'] for r in rr]),
            success=stats([r['final']['success_rate'] for r in rr]),auc=stats([r['auc'] for r in rr]),
            warmup_steps=stats([r['warmup_steps'] for r in rr]),
            max_telescope=max(r['max_telescoping_error'] for r in rr))
    # No method can see a different warm-up prefix.
    prefixes=collections.defaultdict(set)
    for r in rows:prefixes[r['scenario'],r['seed']].add(r['prefix_hash'])
    prefix_ok=all(len(v)==1 for v in prefixes.values())
    selection=HERE/'SELECTION.json'
    proposal=json.loads(selection.read_text())['proposal'] if selection.exists() else 'return_gate'
    tests=[]
    controls=['return_static','return_dynamic','return_mix','plain','mc_dynamic'] if proposal.startswith('return') else ['raw_static','raw_dynamic','plain','return_mix','mc_dynamic']
    for k in sorted(summary):
        if (k,proposal) not in groups:continue
        for b in controls:
            if (k,b) in groups:
                tests.append(dict(scenario=k,a=proposal,b=b,**paired({r['seed']:r['final']['original_return'] for r in groups[k,proposal]},
                    {r['seed']:r['final']['original_return'] for r in groups[k,b]})))
    holm(tests)
    if len(summary)==1:fig,axes=plt.subplots(1,1,figsize=(11,5.8));panels=[axes]
    else:fig,axes=plt.subplots(2,2,figsize=(12,8));panels=list(axes.ravel())
    show=['plain','raw_static','raw_dynamic','raw_mix','return_static','return_dynamic','return_mix','return_gate','return_gate_unbounded','mc_dynamic','velocity_shift','manhattan','return_dynamic_all','mc_dynamic_all','return_dynamic_all_unbounded','mc_dynamic_all_unbounded','return_mix_unbounded']
    for ax,k in zip(panels,sorted(summary)):
        for m in show:
            if (k,m) not in groups:continue
            rr=groups[k,m];x=np.array([z['step'] for z in rr[0]['records']]);v=np.array([[z['original_return'] for z in r['records']] for r in rr]);ax.plot(x,v.mean(axis=0),label=m,linewidth=2.8 if m==proposal else 1.25)
        ax.set(title=k,xlabel='Total training interactions (warm-up included)',ylabel='Original evaluation return')
    for ax in panels[len(summary):]:ax.set_visible(False)
    handles,labels=panels[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8)
    fig.tight_layout(rect=(0,.14 if len(summary)==1 else .10,1,1));(REPORT_OUT/folder.name).mkdir(exist_ok=True);fig.savefig(REPORT_OUT/folder.name/'learning_curves.png',dpi=180);plt.close(fig)
    secondary=[]
    for k in sorted(summary):
        for b in ('raw_static','raw_dynamic'):
            if (k,'raw_mix') in groups and (k,b) in groups:
                secondary.append(dict(scenario=k,a='raw_mix',b=b,**paired({r['seed']:r['final']['original_return'] for r in groups[k,'raw_mix']},
                    {r['seed']:r['final']['original_return'] for r in groups[k,b]})))
    holm(secondary)
    return dict(summary=summary,tests=tests,secondary_tests=secondary,prefixes_identical=prefix_ok,runs=len(rows),steps=sum(r['steps'] for r in rows))

def main():
    output={'repair':ownrepair()}
    for name in ('discovery','confirmation','confirm_mc','confirm_grids','strong_mc','strong_grids','unbounded_mc','unbounded_grids'):
        if (HERE/name/'runs').exists():output[name]=phase(HERE/name)
    # Confirmation split by scenario is one testing family, not two chances.
    if 'confirm_mc' in output and 'confirm_grids' in output:
        tests=output['confirm_mc']['tests']+output['confirm_grids']['tests'];holm(tests)
        output['confirmation_joint']=dict(summary=output['confirm_mc']['summary']|output['confirm_grids']['summary'],tests=tests,
            runs=output['confirm_mc']['runs']+output['confirm_grids']['runs'],
            steps=output['confirm_mc']['steps']+output['confirm_grids']['steps'],
            prefixes_identical=output['confirm_mc']['prefixes_identical'] and output['confirm_grids']['prefixes_identical'])
        sec=output['confirm_mc']['secondary_tests']+output['confirm_grids']['secondary_tests'];holm(sec)
        output['confirmation_joint']['secondary_tests']=sec
        # Explanatory analyses added after the primary outcome was visible.
        # They cannot rescue or replace the prespecified 20-test family.
        rows=read(HERE/'confirm_mc')+read(HERE/'confirm_grids');gg=collections.defaultdict(list)
        for r in rows:gg[r['scenario'],r['method']].append(r)
        proposal=json.loads((HERE/'SELECTION.json').read_text())['proposal']
        rawtests=[];auctests=[]
        for k in sorted(output['confirmation_joint']['summary']):
            aa={r['seed']:r['final']['original_return'] for r in gg[k,proposal]}
            for b in ('raw_static','raw_dynamic'):
                rawtests.append(dict(scenario=k,a=proposal,b=b,**paired(aa,{r['seed']:r['final']['original_return'] for r in gg[k,b]})))
            def areas(rr):
                return {r['seed']:float(np.trapezoid([x['original_return'] for x in r['records']],[x['step'] for x in r['records']])/r['steps']) for r in rr}
            for b in ('return_static','return_dynamic','return_mix','plain','mc_dynamic'):
                auctests.append(dict(scenario=k,a=proposal,b=b,**paired(areas(gg[k,proposal]),areas(gg[k,b]))))
        holm(rawtests);holm(auctests)
        output['confirmation_joint']['posthoc_raw_controls']=rawtests
        output['confirmation_joint']['posthoc_auc_tests']=auctests
        extra=[]
        for name in ('strong_mc','strong_grids','unbounded_mc','unbounded_grids'):
            if name not in output:continue
            extra+=read(HERE/name)
            for k,methods in output[name]['summary'].items():output['confirmation_joint']['summary'][k].update(methods)
        if extra:
            ext=collections.defaultdict(list)
            for r in extra:ext[r['scenario'],r['method']].append(r)
            supp=[]
            for (k,b),rr in sorted(ext.items()):
                supp.append(dict(scenario=k,a=proposal,b=b,**paired({r['seed']:r['final']['original_return'] for r in gg[k,proposal]},
                    {r['seed']:r['final']['original_return'] for r in rr})))
            holm(supp);output['confirmation_joint']['strong_control_tests']=supp
            output['confirmation_joint']['supplementary_runs']=len(extra)
    (REPORT_OUT/'analysis.json').write_text(json.dumps(output,indent=2),encoding='utf8')
    print(json.dumps({k:{'runs':v.get('runs'),'prefixes_identical':v.get('prefixes_identical'),'summary':v['summary']} for k,v in output.items()},indent=2))

if __name__=='__main__':main()
