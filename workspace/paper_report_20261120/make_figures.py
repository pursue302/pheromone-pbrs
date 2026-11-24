"""Publication figures and LaTeX tables, generated only from saved data."""
from pathlib import Path
import json,collections,sys,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
DATA=json.loads((HERE/'paper_data.json').read_text())
PAPER=DATA['previous_analysis']
VALID=DATA['validation']
FIG=HERE/'figures';FIG.mkdir(exist_ok=True);TAB=HERE/'tables';TAB.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
    'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight'})
COL={'plain':'#343d49','raw_static':'#bc7040','raw_dynamic':'#b53d4a','state_raw_static':'#3a8e9d',
    'state_raw_matched':'#84a435','return_static':'#9c7cae','return_gate_unbounded':'#235cc3',
    'return_dynamic_all_unbounded':'#a06aae','mc_dynamic_all_unbounded':'#4c9a65'}
LABEL={'plain':'SARSA','raw_static':'Raw-S','raw_dynamic':'Raw-D','state_raw_static':'State-S',
    'state_raw_matched':'Matched-S','return_static':'Return-S','return_gate_unbounded':'Return-Gate',
    'return_dynamic_all_unbounded':'Return-D (all)','mc_dynamic_all_unbounded':'MC (all)'}
KEYS=['u_detour','four_rooms','slippery_rooms'];TITLE={'u_detour':'U-detour','four_rooms':'Four rooms','slippery_rooms':'Slippery rooms','mountain_car':'MountainCar'}
def export(fig,name):
    fig.savefig(FIG/(name+'.pdf'));fig.savefig(FIG/(name+'.png'),dpi=240);plt.close(fig)
def read(folder):return [json.loads(p.read_text()) for p in sorted(folder.glob('*.json'))]
def interval(a):
    a=np.asarray(a);rng=np.random.default_rng(78322)
    b=np.concatenate([a[rng.integers(len(a),size=(1000,len(a)))].mean(axis=1) for _ in range(10)])
    return np.quantile(b,[.025,.975])
def maps():
    fig,axs=plt.subplots(1,3,figsize=(7.05,2.05))
    for ax,k in zip(axs,KEYS):
        board=DATA['layouts'][k]
        for r,row in enumerate(board):
            for c,x in enumerate(row):
                ax.add_patch(Rectangle((c-.5,r-.5),1,1,facecolor='#354657' if x=='#' else '#f3f6fa',edgecolor='white',linewidth=.4))
                if x in 'SG':ax.text(c,r,x,ha='center',va='center',fontweight='bold',color='#206b54' if x=='G' else '#235cc3',fontsize=8)
        ax.set(xlim=(-.5,10.5),ylim=(10.5,-.5),title=TITLE[k]+(' (slip = 0.15)' if k=='slippery_rooms' else ''))
        ax.set_aspect('equal');ax.axis('off')
    fig.tight_layout(w_pad=2.);export(fig,'navigation_maps')
def goal():
    d=DATA['diagnostic'];fig,(a,b)=plt.subplots(1,2,figsize=(7.05,2.15),gridspec_kw={'width_ratios':[.85,1.4]})
    for r in range(5):
        for c in range(7):a.add_patch(Rectangle((c-.5,r-.5),1,1,facecolor='#f1f4f8',edgecolor='white'))
    for pos,text,color in [((4,0),'S','#235cc3'),((0,6),'G','#208467'),((4,2),'B','#b53d4a')]:
        a.text(pos[1],pos[0],text,ha='center',va='center',color=color,fontweight='bold',fontsize=11)
    r=next(x for x in d['results'] if x['kind']=='pbrs' and x['beta']==1.)
    xy=np.array(r['trajectory']);a.plot(xy[:,1],xy[:,0],color='#235cc3',lw=1.4,alpha=.7)
    a.annotate('repeat visits',xy=(2,4),xytext=(3,2.5),arrowprops={'arrowstyle':'->','color':'#b53d4a'},color='#b53d4a',fontsize=7)
    a.set(xlim=(-.5,6.5),ylim=(4.5,-.5),title='(a) A tempting bonus cell B');a.set_aspect('equal');a.axis('off')
    for kind,name,color in [('naive','Naive arrival bonus','#b53d4a'),('pbrs','Potential difference','#235cc3')]:
        rr=[x for x in d['results'] if x['kind']==kind];b.plot([x['beta'] for x in rr],[x['original_discounted_return'] for x in rr],'-o',ms=3,label=name,color=color)
    b.axhline(d['baseline']['original_discounted_return'],ls=':',color='#343d49',lw=1,label='Original optimum')
    b.set_xscale('symlog',linthresh=.01);b.set(xlabel=r'Shaping strength $\beta$',ylabel='Original discounted return',xlim=(0,10.5),ylim=(-.04,1.03),title='(b) Exact optimal policy for the shaped objective')
    b.legend(loc='lower left',fontsize=6.4);fig.tight_layout(w_pad=1.5);export(fig,'goal_preservation')
def validation():
    rows=DATA['fresh_rows'];fig,axs=plt.subplots(1,3,figsize=(7.05,2.6))
    methods=['plain','raw_static','raw_dynamic','state_raw_static','state_raw_matched','return_gate_unbounded']
    for ax,k in zip(axs,KEYS):
        for x,m in enumerate(methods):
            a=np.array([r['final']['original_return'] for r in rows if r['scenario']==k and r['method']==m]);lo,hi=interval(a);mu=a.mean()
            ax.bar(x,mu,color=COL[m],width=.72);ax.errorbar(x,mu,yerr=[[mu-lo],[hi-mu]],fmt='none',ecolor='#1f2d3a',capsize=2,lw=.8)
        ax.set_xticks(range(len(methods)),[LABEL[m] for m in methods],rotation=55,ha='right',fontsize=7.2)
        ax.set(title=TITLE[k],ylabel='Final original return' if k==KEYS[0] else '',ylim=(-140,3));ax.grid(axis='y',alpha=.18)
    fig.tight_layout(w_pad=1.1);export(fig,'fresh_validation')
def controls():
    fig,(ax,b)=plt.subplots(1,2,figsize=(7.05,2.45),gridspec_kw={'width_ratios':[1.,1.15]})
    methods=['source_sarsa','source_static_frozen','actual_source_reset_only','static_canonical']
    names=['Released SARSA','Released static','SARSA + reset','Repaired static']
    colors=['#b53d4a','#bc7040','#4c9a65','#235cc3'];rep=PAPER['repair']['summary']
    for i,(m,c) in enumerate(zip(methods,colors)):
        mean=rep[m]['final_return']['mean'];sem=rep[m]['final_return']['sd']/10
        ax.barh(i,mean,color=c,height=.65);ax.errorbar(mean,i,xerr=1.96*sem,fmt='none',color='#24313d',capsize=2,lw=.8)
    ax.set_yticks(range(4),names,fontsize=7);ax.invert_yaxis();ax.set(xlabel='Final original return',title='(a) Direct source control, 100 seeds',xlim=(-1030,0))
    rows=DATA['mc_rows']
    for m in ['plain','raw_static','return_gate_unbounded','return_dynamic_all_unbounded','mc_dynamic_all_unbounded']:
        rr=[r for r in rows if r['method']==m];x=np.array([z['step'] for z in rr[0]['records']]);y=np.array([[z['original_return'] for z in r['records']] for r in rr]);mu=y.mean(axis=0);sem=y.std(axis=0,ddof=1)/np.sqrt(len(y))
        b.plot(x/1000,mu,label=LABEL[m],color=COL[m],lw=1.8 if m=='return_gate_unbounded' else 1.)
        if m=='return_gate_unbounded':b.fill_between(x/1000,mu-1.96*sem,mu+1.96*sem,color=COL[m],alpha=.12)
    b.axvspan(0,15,color='#9ba8b6',alpha=.12);b.set(xlabel='Total interactions (thousands)',ylabel='Original evaluation return',title='(b) MountainCar, 80 seeds')
    b.legend(loc='lower right',fontsize=6.8);fig.tight_layout(w_pad=1.5);export(fig,'source_and_mountaincar')
def tables():
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
    with (HERE/'results_export.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['cohort','scenario','method','n','mean_original_return','sd','success','auc'])
        for k,zz in VALID['summary'].items():
            for m,v in zz.items():w.writerow(['fresh',k,m,v['original_return']['n'],v['original_return']['mean'],v['original_return']['sd'],v['success_rate']['mean'],v['auc']['mean']])
def main():
    maps();goal();validation();controls();tables();print('Generated four figures and data-driven LaTeX tables.')
if __name__=='__main__':main()
