from multiprocessing import Pool
from functools import partial
import bilby
import numpy as np
import scipy
from matplotlib import pyplot as plt


def normalising(hf, psd_array, delta_f):
    return hf / np.sqrt(
        bilby.gw.utils.noise_weighted_inner_product(hf, hf, psd_array, 1 / delta_f).real
    )


def mismatch(h1_norm, h2_norm, psd_array, delta_f):
    return (
        1
        - bilby.gw.utils.noise_weighted_inner_product(
            h1_norm, h2_norm, psd_array, 1 / delta_f
        ).real
    )


def mismatch_with_time_phase_shift(
    time_phase_shift, freq, delta_f, psd_array, h1_norm, h2_norm
):
    shift = 1j * (2 * np.pi * freq * time_phase_shift[0] + time_phase_shift[1])
    h1_norm_shifted = h1_norm * np.exp(shift)
    return mismatch(h1_norm_shifted, h2_norm, psd_array, delta_f)


def minimized_mismatch(
    total_mass,
    mass_ratio,
    chi_1,
    chi_2,
    wf_gen_1,
    wf_gen_2,
    freq,
    freq_mask,
    psd_array,
    delta_f,
):
    params = {
        "total_mass": total_mass,
        "mass_ratio": mass_ratio,
        "chi_1": chi_1,
        "chi_2": chi_2,
        "phase": 0.0,
        "theta_jn": np.pi / 2,
        "luminosity_distance": 800.0,
    }
    h1 = wf_gen_1.frequency_domain_strain(params)["plus"][freq_mask]
    h2 = wf_gen_2.frequency_domain_strain(params)["plus"][freq_mask]
    h1_norm = normalising(h1, psd_array, delta_f)
    h2_norm = normalising(h2, psd_array, delta_f)

    minimizing_res = scipy.optimize.dual_annealing(
        mismatch_with_time_phase_shift,
        bounds=((-0.1, 0.1), (-np.pi, np.pi)),
        args=(freq[freq_mask], delta_f, psd_array, h1_norm, h2_norm),
    )
    if minimizing_res.success:
        return minimizing_res.fun
    else:
        raise ValueError("minimizing func exited fail")


if __name__ == "__main__":

    nproc = 72

    appro_nrhyb = "NRHybSur3dq8"
    appro_xas = "IMRPhenomXAS"
    appro_d = "IMRPhenomD"

    duration = 16.0
    sampling_frequency = 4096.0
    start_time = 0.0
    reference_frequency = 20.0
    minimum_frequency = 20.0
    maximum_frequency = 2048.0

    wf_args_nrhyb = dict(
        reference_frequency=reference_frequency,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        waveform_approximant=appro_nrhyb,
        mode_array=[[2, 2]],
    )
    wf_args_xas = dict(
        reference_frequency=reference_frequency,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        waveform_approximant=appro_xas,
        mode_array=[[2, 2]],
    )
    wf_args_d = dict(
        reference_frequency=reference_frequency,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        waveform_approximant=appro_d,
        mode_array=[[2, 2]],
    )
    wf_nrhyb = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        start_time=start_time,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=wf_args_nrhyb,
    )
    wf_xas = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        start_time=start_time,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=wf_args_xas,
    )
    wf_d = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        start_time=start_time,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=wf_args_d,
    )

    freq = wf_nrhyb.frequency_array
    freq_mask = freq_mask = (freq >= wf_args_nrhyb["minimum_frequency"]) * (
        freq <= wf_args_nrhyb["maximum_frequency"]
    )
    delta_f = freq[1] - freq[0]

    PSD_Aplus = bilby.gw.detector.PowerSpectralDensity(asd_file="AplusDesign.txt")
    psd_array = PSD_Aplus.get_power_spectral_density_array(freq[freq_mask])

    # num = 1000
    # parameters = {}
    # rng = np.random.default_rng(seed=20241031)
    # parameters["total_mass"] = rng.uniform(20, 250, num)
    # parameters["mass_ratio"] = rng.uniform(0.125, 1.0, num)
    # parameters["chi_1"] = rng.uniform(-0.8, 0.8, num)
    # parameters["chi_2"] = rng.uniform(-0.8, 0.8, num)
    # params_array = np.vstack(
    #     [
    #         parameters["total_mass"],
    #         parameters["mass_ratio"],
    #         parameters["chi_1"],
    #         parameters["chi_2"],
    #     ]
    # ).T

    # mismtach_nrhyb_xas = partial(
    #     minimized_mismatch,
    #     wf_gen_1=wf_nrhyb,
    #     wf_gen_2=wf_xas,
    #     freq=freq,
    #     freq_mask=freq_mask,
    #     psd_array=psd_array,
    #     delta_f=delta_f,
    # )
    # with Pool(processes=nproc) as pool:
    #     mism = np.array(pool.starmap(mismtach_nrhyb_xas, params_array))
    # parameters["mismatch_nrhyb_xas"] = mism

    # mismtach_nrhyb_d = partial(
    #     minimized_mismatch,
    #     wf_gen_1=wf_nrhyb,
    #     wf_gen_2=wf_d,
    #     freq=freq,
    #     freq_mask=freq_mask,
    #     psd_array=psd_array,
    #     delta_f=delta_f,
    # )
    # with Pool(processes=nproc) as pool:
    #     mism = np.array(pool.starmap(mismtach_nrhyb_d, params_array))
    # parameters["mismatch_nrhyb_d"] = mism

    # param_keys = list(parameters.keys())
    # structed_params = np.array(
    #     list(zip(*[parameters[key] for key in param_keys])),
    #     dtype=[(key, "float") for key in param_keys],
    # )
    # np.save(f"mismatch_nrhyb_xaslal_d", structed_params)
########################################################################################
    parameters = np.load("mismatch_nrhyb_xaslal_d.npy")
    parameters = {key: parameters[key] for key in parameters.dtype.names}

    params_array = np.vstack(
        [
            parameters["total_mass"],
            parameters["mass_ratio"],
            parameters["chi_1"],
            parameters["chi_2"],
        ]
    ).T

    mismtach_nrhyb_xas = partial(
        minimized_mismatch,
        wf_gen_1=wf_nrhyb,
        wf_gen_2=wf_xas,
        freq=freq,
        freq_mask=freq_mask,
        psd_array=psd_array,
        delta_f=delta_f,
    )
    with Pool(processes=nproc) as pool:
        mism = np.array(pool.starmap(mismtach_nrhyb_xas, params_array))
    parameters["mismatch_nrhyb_xasmod"] = mism

    param_keys = list(parameters.keys())
    structed_params = np.array(
        list(zip(*[parameters[key] for key in param_keys])),
        dtype=[(key, "float") for key in param_keys],
    )
    np.save(f"mismatch_nrhyb_xasmod", structed_params)

    fig, ax = plt.subplots()
    ax.hist(
        np.log10(parameters[f"mismatch_nrhyb_xas"]),
        density=True,
        bins=20,
        histtype="step",
        label="nrhyb_xaslal",
    )
    ax.hist(
    np.log10(parameters[f"mismatch_nrhyb_xasmod"]),
    density=True,
    bins=20,
    histtype="step",
    label="nrhyb_xasmod",
    )
    ax.hist(
        np.log10(parameters[f"mismatch_nrhyb_d"]),
        density=True,
        bins=20,
        histtype="step",
        label="nrhyb_d",
    )
    ax.legend()
    fig.savefig("mismatch_lal")


