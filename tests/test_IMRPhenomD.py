import sys 
sys.path.append('/home/hydrogen/workspace/Space_GW/wf4ti')
import time

import numpy as np
import taichi as ti
ti.init(arch=ti.cpu)

from wf4ti.waveforms import IMRPhenomD

cadance = 10
duration = 7*24*3600
f_array = np.arange(0, 1.0/(2*cadance), 1.0/duration)
minimum_frequency = 1e-4
maximum_frequency = 1e-1
bound = ((f_array >= minimum_frequency) * (f_array <= maximum_frequency))
f_array = f_array[bound]
data_length = len(f_array)

frequencies = ti.field(ti.f64, shape=(data_length,))
frequencies.from_numpy(f_array)

wf = IMRPhenomD.IMRPhenomD(frequencies)
wf.get_waveform()

