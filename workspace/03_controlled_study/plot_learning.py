import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_learning(groups, summary, proposal, show, output):
    if len(summary)==1:fig,axes=plt.subplots(1,1,figsize=(11,5.8));panels=[axes]
    else:fig,axes=plt.subplots(2,2,figsize=(12,8));panels=list(axes.ravel())
    for ax,k in zip(panels,sorted(summary)):
        for m in show:
            if (k,m) not in groups:continue
            rr=groups[k,m];x=np.array([z['step'] for z in rr[0]['records']]);v=np.array([[z['original_return'] for z in r['records']] for r in rr]);ax.plot(x,v.mean(axis=0),label=m,linewidth=2.8 if m==proposal else 1.25)
        ax.set(title=k,xlabel='Total training interactions (warm-up included)',ylabel='Original evaluation return')
    for ax in panels[len(summary):]:ax.set_visible(False)
    handles,labels=panels[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8)
    fig.tight_layout(rect=(0,.14 if len(summary)==1 else .10,1,1));fig.savefig(output,dpi=180);plt.close(fig)
