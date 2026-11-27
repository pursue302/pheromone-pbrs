import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
def export(fig, name, output):
    fig.savefig(output/(name+'.pdf'));fig.savefig(output/(name+'.png'),dpi=240);plt.close(fig)
