import numpy as np
from workspace.plot_style import plt, export
from matplotlib.patches import Rectangle
def goal(data, output):
    d=data['diagnostic'];fig,(a,b)=plt.subplots(1,2,figsize=(7.05,2.15),gridspec_kw={'width_ratios':[.85,1.4]})
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
    b.legend(loc='lower left',fontsize=6.4);fig.tight_layout(w_pad=1.5);export(fig,'goal_preservation', output)
