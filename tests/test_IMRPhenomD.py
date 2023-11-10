import sys 
sys.path.append('/home/hydrogen/workspace/Space_GW/wf4ti')
import time

import numpy as np
from matplotlib import pyplot as plt
import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f64)

from wf4ti.waveforms.IMRPhenomD import IMRPhenomD

parameters = {}
parameters['total_mass'] = 7e6
parameters['mass_ratio'] = 0.8
parameters['chi_1'] = 0.2
parameters['chi_2'] = 0.3
parameters['luminosity_distance'] = 3000.0
parameters['inclination'] = 0.15
parameters['reference_phase'] = 0.0
parameters['coalescence_time'] = 0.0
rng = np.random.default_rng()

cadance = 10
duration = 365*24*3600
f_array = np.arange(0, 1.0/(2*cadance), 1.0/duration)
minimum_frequency = 1e-4
maximum_frequency = 1e-1
bound = ((f_array >= minimum_frequency) * (f_array <= maximum_frequency))
f_array = f_array[bound]
data_length = len(f_array)

frequencies = ti.field(ti.f64, shape=(data_length,))
frequencies.from_numpy(f_array)

wf = IMRPhenomD(frequencies, returned_form='amplitude_phase')
wf.get_waveform(parameters)
wf_array = wf.np_array_of_waveform_container()

fig, ax = plt.subplots()
ax.loglog(f_array, wf_array['amplitude'])
fig.savefig('amplitude.png')
fig, ax = plt.subplots()
ax.semilogx(f_array, wf_array['phase'])
fig.savefig('phase.png')
fig, ax = plt.subplots()
ax.semilogx(f_array, wf_array['tf'])
fig.savefig('tf.png')
