import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_repair(groups, output):
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
    fig.tight_layout();fig.savefig(output,dpi=180);plt.close(fig)
