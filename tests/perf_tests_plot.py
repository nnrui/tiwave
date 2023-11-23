import json
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
fig_width_pt = 3*246.0                  
inches_per_pt = 1.0/72.27               
golden_mean = (np.sqrt(5)-1.0)/2.0      
fig_width = fig_width_pt*inches_per_pt  
fig_height = fig_width*golden_mean      
fig_size =  [fig_width,fig_height]
params = {'axes.labelsize': 20,
          'font.family': 'serif',
          'font.serif': 'Computer Modern Raman',
          'font.size': 28,
          'legend.fontsize': 20,
          'xtick.labelsize': 18,
          'ytick.labelsize': 18,
          'axes.grid' : True,
          'text.usetex': False,
          'savefig.dpi' : 300,
          'lines.markersize' : 18,
          'figure.figsize': fig_size}
matplotlib.rcParams.update(params)

with open('time_consuming_gpu_ti.json', 'r') as f:
    data_ti = json.load(f)
with open('time_consuming_gpu_ripple.json', 'r') as f:
    data_ripple = json.load(f)
with open('time_consuming_gpu_ti.json', 'r') as f:
    data_bbhx = json.load(f)

fig, ax = plt.subplots()
ax.loglog(data_ti['data_length'],     data_ti['ti'],         label='wf4ti')
ax.loglog(data_ripple['data_length'], data_ripple['ripple'], label='ripple')
ax.loglog(data_bbhx['data_length'],   data_bbhx['bbhx'],     label='bbhx')
ax.legend()
fig.savefig('perf_gpu.png')
