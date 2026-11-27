import numpy as np
from workspace.plot_style import plt, export, COL, LABEL, TITLE, KEYS
from matplotlib.patches import Rectangle
def maps(data, output):
    fig,axs=plt.subplots(1,3,figsize=(7.05,2.05))
    for ax,k in zip(axs,KEYS):
        board=data['layouts'][k]
        for r,row in enumerate(board):
            for c,x in enumerate(row):
                ax.add_patch(Rectangle((c-.5,r-.5),1,1,facecolor='#354657' if x=='#' else '#f3f6fa',edgecolor='white',linewidth=.4))
                if x in 'SG':ax.text(c,r,x,ha='center',va='center',fontweight='bold',color='#206b54' if x=='G' else '#235cc3',fontsize=8)
        ax.set(xlim=(-.5,10.5),ylim=(10.5,-.5),title=TITLE[k]+(' (slip = 0.15)' if k=='slippery_rooms' else ''))
        ax.set_aspect('equal');ax.axis('off')
    fig.tight_layout(w_pad=2.);export(fig,'navigation_maps', output)

def interval(a):
    a=np.asarray(a);rng=np.random.default_rng(78322)
    b=np.concatenate([a[rng.integers(len(a),size=(1000,len(a)))].mean(axis=1) for _ in range(10)])
    return np.quantile(b,[.025,.975])
def validation(data, output):
    rows=data['fresh_rows'];fig,axs=plt.subplots(1,3,figsize=(7.05,2.6))
    methods=['plain','raw_static','raw_dynamic','state_raw_static','state_raw_matched','return_gate_unbounded']
    for ax,k in zip(axs,KEYS):
        for x,m in enumerate(methods):
            a=np.array([r['final']['original_return'] for r in rows if r['scenario']==k and r['method']==m]);lo,hi=interval(a);mu=a.mean()
            ax.bar(x,mu,color=COL[m],width=.72);ax.errorbar(x,mu,yerr=[[mu-lo],[hi-mu]],fmt='none',ecolor='#1f2d3a',capsize=2,lw=.8)
        ax.set_xticks(range(len(methods)),[LABEL[m] for m in methods],rotation=55,ha='right',fontsize=7.2)
        ax.set(title=TITLE[k],ylabel='Final original return' if k==KEYS[0] else '',ylim=(-140,3));ax.grid(axis='y',alpha=.18)
    fig.tight_layout(w_pad=1.1);export(fig,'fresh_validation', output)
