import numpy as np
from matplotlib import pyplot as plt
import matplotlib
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
          'text.usetex': True,
          'savefig.dpi' : 300,
          'lines.markersize' : 18,
          'figure.figsize': fig_size}
matplotlib.rcParams.update(params)

import lal
import lalsimulation as lalsim
import bilby


parameters = {}
parameters['total_mass'] = 5e6
parameters['mass_ratio'] = 0.8
parameters['chi_1'] = 0.2
parameters['chi_2'] = 0.3
parameters['luminosity_distance'] = 3000.0
parameters['inclination'] = 0.15
parameters['reference_phase'] = 0.0
parameters['coalescence_time'] = 0.0
parameters = bilby.gw.conversion.generate_all_bbh_parameters(parameters)

cadance = 10
duration = 3600*24*7
f_array = np.arange(0, 1.0/(2*cadance), 1.0/duration)
minimum_frequency = 1e-5
maximum_frequency = 1e-1
bound = ((f_array >= minimum_frequency) * (f_array <= maximum_frequency))
f_array = f_array[bound]
data_length = len(f_array)

extraParams = lal.CreateDict()
# require the modified lalsimulaiton (https://github.com/niuiniuin/lalsuite)
# checkout to the branch `phenomD_amp_phase`
amplitude, phase = lalsim.SimIMRPhenomDFrequencySequenceAmpPhase(f_array,
                                                                 parameters['reference_phase'],
                                                                 f_array[0],
                                                                 parameters['mass_1']*lal.MSUN_SI,
                                                                 parameters['mass_2']*lal.MSUN_SI,
                                                                 parameters['chi_1'],
                                                                 parameters['chi_2'],
                                                                 parameters['luminosity_distance']*1e6*lal.PC_SI,
                                                                 extraParams,
                                                                 lalsim.NoNRT_V
                                                                 )
amplitude_array = amplitude.data.data
phase_array = phase.data.data
dphase = (phase_array[1:] -  phase_array[:-1]) * duration / 2 / lal.PI

f_peak = lalsim.IMRPhenomDGetPeakFreq(parameters['mass_1'],
                                      parameters['mass_2'],
                                      parameters['chi_1'],
                                      parameters['chi_2']
                                      )

# fig, ax = plt.subplots()
# ax.loglog(f_array, amplitude_array)
# fig.savefig('amp_lal.png')

fig, ax = plt.subplots()
ax.loglog(f_array[:-1], np.abs(dphase))
ax.axvline(f_peak, color='tab:red', label='peak frequency', linestyle='dashed')
ax.set_xlabel(r'$f$(Hz)')
ax.set_ylabel(r'$|\frac{{\rm d}\Phi}{2\pi{\rm d}f}|$')
ax.legend()
# ax.set_title('t0 include C2MRD')
# fig.savefig('dphase_C2MRD_lal_mHz.png')
ax.set_title('Original t0 without C2MRD')
fig.savefig('dphase_lal_mHz.png')

