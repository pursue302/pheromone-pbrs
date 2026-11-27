def source_panel(ax, summary):
    methods=['source_sarsa','source_static_frozen','actual_source_reset_only','static_canonical']
    names=['Released SARSA','Released static','SARSA + reset','Repaired static']
    colors=['#b53d4a','#bc7040','#4c9a65','#235cc3'];rep=summary
    for i,(m,c) in enumerate(zip(methods,colors)):
        mean=rep[m]['final_return']['mean'];sem=rep[m]['final_return']['sd']/10
        ax.barh(i,mean,color=c,height=.65);ax.errorbar(mean,i,xerr=1.96*sem,fmt='none',color='#24313d',capsize=2,lw=.8)
    ax.set_yticks(range(4),names,fontsize=7);ax.invert_yaxis();ax.set(xlabel='Final original return',title='(a) Direct source control, 100 seeds',xlim=(-1030,0))
