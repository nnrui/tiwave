import bilby
import numpy as np
from matplotlib import pyplot as plt

parameters = {
    "total_mass": 64.95150288,
    "mass_ratio": 0.6560166,
    "chi_1": -0.31793288,
    "chi_2": -0.23096549,
    "phase": 0.0,
    "theta_jn": np.pi / 2,
    "luminosity_distance": 800.0,
}

duration = 16.0
sampling_frequency = 4096
start_time = 0.0

appro_1 = "NRHybSur3dq8"
appro_2 = "IMRPhenomXAS"

wf_1_args = dict(
    reference_frequency=20.0,
    minimum_frequency=20.0,
    maximum_frequency=2048.0,
    waveform_approximant=appro_1,
    catch_waveform_errors=True,
    mode_array=[[2,2]]
)
wf_1 = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    start_time=start_time,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=wf_1_args,
)
wf_2_args = dict(
    reference_frequency=20.0,
    minimum_frequency=20.0,
    maximum_frequency=2048.0,
    waveform_approximant=appro_2,
    catch_waveform_errors=True,
    mode_array=[[2,2]]
)
wf_2 = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    start_time=start_time,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=wf_2_args,
)

hf_1 = wf_1.frequency_domain_strain(parameters)
hf_2 = wf_2.frequency_domain_strain(parameters)
freq = wf_1.frequency_array
freq_mask = (freq >= wf_1_args["minimum_frequency"]) * (
    freq <= wf_1_args["maximum_frequency"]
)

fig, ax = plt.subplots()
ax.loglog(freq[freq_mask], np.abs(hf_1["plus"])[freq_mask], label="nrhyb")
ax.loglog(freq[freq_mask], np.abs(hf_2["plus"])[freq_mask], label="xas")
ax.legend()
fig.savefig("abs")

fig, ax = plt.subplots()
ax.semilogx(freq[freq_mask], hf_1["plus"][freq_mask].real, label="nrhyb")
ax.semilogx(freq[freq_mask], hf_2["plus"][freq_mask].real, label="xas")
ax.legend()
fig.savefig("real")

fig, ax = plt.subplots()
ax.semilogx(freq[freq_mask], hf_1["plus"][freq_mask].imag, label="nrhyb")
ax.semilogx(freq[freq_mask], hf_2["plus"][freq_mask].imag, label="xas")
ax.legend()
fig.savefig("imag")
