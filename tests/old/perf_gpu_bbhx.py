'''
After import ripple, there will be strange unknown error when using taichi.
So, we have to seperate these 3 tests into 3 files.
'''
import sys 
path_prefix = '/home/changfenggroup/nrui/works/'
import time
import json

import numpy as np
import pandas as pd
import bilby
import lal


# set max_mass to avoid too many 0 at the end of waveform
cadance = 5
minimum_frequency = 1e-5
maximum_frequency = 1e-1
PhenomD_f_cut = 0.2
max_mass = maximum_frequency/PhenomD_f_cut/lal.MTSUN_SI
# print(max_mass)

time_consuming_bbhx = []
data_length_list = []
powers_of_2 = range(10, 31)
# powers_of_2 = range(10, 12)
num_tests = 100

rng = np.random.default_rng()
parameters = {}
parameters['total_mass'] = rng.uniform(1e3, max_mass, num_tests)
parameters['mass_ratio'] = rng.uniform(0.2, 1.0, num_tests)
parameters['chi_1'] = rng.uniform(-1.0, 1.0, num_tests)
parameters['chi_2'] = rng.uniform(-1.0, 1.0, num_tests)
parameters['luminosity_distance'] = rng.uniform(1000.0, 10000.0, num_tests)
parameters['inclination'] = rng.uniform(0, np.pi, num_tests)
parameters['reference_phase'] = rng.uniform(0, 2*np.pi, num_tests)
parameters['coalescence_time'] = np.zeros(num_tests)
parameters = bilby.gw.conversion.generate_mass_parameters(parameters)
parameters = pd.DataFrame(parameters)
# print(parameters.keys())
# print(parameters)


# NOTE: since different codes has different output form, some output amplitude and phase, 
# some output polarizations, the comparison maybe not absolutely fair.
# The output of wf4ti includes polarizations and tf which could be thought having the most 
# compution steps.
##########################################################################################
# bbhx
from bbhx.waveforms.phenomhm import PhenomHMAmpPhase
for p in powers_of_2:
    duration = 2**p
    f_array = np.arange(0, 1.0/(2*cadance), 1.0/duration)
    bound = ((f_array >= minimum_frequency) * (f_array <= maximum_frequency))
    f_array = f_array[bound]
    data_length = len(f_array)
    data_length_list.append(data_length)

    bbhx_wf = PhenomHMAmpPhase(use_gpu=True, run_phenomd=True)
    st = time.perf_counter()
    for i in range(num_tests):
        bbhx_wf(parameters.iloc[i]['mass_1'],
                parameters.iloc[i]['mass_2'],
                parameters.iloc[i]['chi_1'],
                parameters.iloc[i]['chi_2'],
                parameters.iloc[i]['luminosity_distance']*1e6*lal.PC_SI,
                0.0,
                0.0,
                parameters.iloc[i]['coalescence_time'],
                len(f_array),
                freqs=f_array)
        amps = bbhx_wf.amp[0][0]  # shape (num_bin_all, num_modes, length)
        phase = bbhx_wf.phase[0][0]  # shape (num_bin_all, num_modes, length)
        tf = bbhx_wf.tf[0][0]
    ed = time.perf_counter()
    time_consuming = (ed - st)/num_tests
    print(f'bbhx {p}th, time:{time_consuming}')
    time_consuming_bbhx.append(time_consuming)


save_data = {'data_length': data_length_list,
             'bbhx': time_consuming_bbhx
            }
with open('time_consuming_gpu_bbhx.json', 'w') as f:
    json.dump(save_data, f)


# import matplotlib
# from matplotlib import pyplot as plt
# fig_width_pt = 3*246.0                  
# inches_per_pt = 1.0/72.27               
# golden_mean = (np.sqrt(5)-1.0)/2.0      
# fig_width = fig_width_pt*inches_per_pt  
# fig_height = fig_width*golden_mean      
# fig_size =  [fig_width,fig_height]
# params = {'axes.labelsize': 20,
#           'font.family': 'serif',
#           'font.serif': 'Computer Modern Raman',
#           'font.size': 28,
#           'legend.fontsize': 20,
#           'xtick.labelsize': 18,
#           'ytick.labelsize': 18,
#           'axes.grid' : True,
#           'text.usetex': False,
#           'savefig.dpi' : 300,
#           'lines.markersize' : 18,
#           'figure.figsize': fig_size}
# matplotlib.rcParams.update(params)
# fig, ax = plt.subplots()
# ax.loglog(save_data['data_length'], save_data['ti'],     label='wf4ti')
# ax.loglog(save_data['data_length'], save_data['ripple'], label='ripple')
# ax.loglog(save_data['data_length'], save_data['bbhx'],   label='bbhx')
# ax.legend()
# fig.savefig('perf_gpu.png')

