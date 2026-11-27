import numpy as np
from workspace.plot_style import plt, export, COL, LABEL
def controls(data, output, source_panel):
    fig,(ax,b)=plt.subplots(1,2,figsize=(7.05,2.45),gridspec_kw={'width_ratios':[1.,1.15]})
    source_panel(ax, data['previous_analysis']['repair']['summary'])
    rows=data['mc_rows']
    for m in ['plain','raw_static','return_gate_unbounded','return_dynamic_all_unbounded','mc_dynamic_all_unbounded']:
        rr=[r for r in rows if r['method']==m];x=np.array([z['step'] for z in rr[0]['records']]);y=np.array([[z['original_return'] for z in r['records']] for r in rr]);mu=y.mean(axis=0);sem=y.std(axis=0,ddof=1)/np.sqrt(len(y))
        b.plot(x/1000,mu,label=LABEL[m],color=COL[m],lw=1.8 if m=='return_gate_unbounded' else 1.)
        if m=='return_gate_unbounded':b.fill_between(x/1000,mu-1.96*sem,mu+1.96*sem,color=COL[m],alpha=.12)
    b.axvspan(0,15,color='#9ba8b6',alpha=.12);b.set(xlabel='Total interactions (thousands)',ylabel='Original evaluation return',title='(b) MountainCar, 80 seeds')
    b.legend(loc='lower right',fontsize=6.8);fig.tight_layout(w_pad=1.5);export(fig,'source_and_mountaincar', output)
