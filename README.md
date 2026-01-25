# `tiwave` - gravitational waveforms implemented using taichi-lang

![last commit](https://img.shields.io/github/last-commit/nnrui/tiwave)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![docs](https://img.shields.io/badge/docs-under%20development-yellow)]()
<!-- [![docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://nrui.github.io/tiwave/) -->

> [!WARNING]
> This is an experimental project under active development. The design and APIs are not stable and may change frequently.

`tiwave` is a python implementation of several gravitational waveform models powered by [`taichi-lang`](https://www.taichi-lang.org/), developed primarily for the preparatory science of space-borne gravitaional-wave missions.
This package enables high-performance waveform generation across CPUs and GPUs, while keeping pythonic usability and maintainability. 


## Installation
Install from PyPI:
```bash
pip install tiwave
```
Install the latest or specific commit version:
```bash
# install the latest development version
pip install git+https://github.com/nnrui/tiwave
# install a specific commit
pip install git+https://github.com/nnrui/tiwave@<commit-hash>
```

## Usage
```python
import taichi as ti
ti.init(arch=ti.cpu, default_fp=ti.f64)

import numpy as np
from tiwave.waveforms import IMRPhenomXAS

reference_frequency = 20.0
minimum_frequency = 20.0
maximum_frequency = 2048.0
sampling_rate = 4096
duration = 4.0

num_samples = int(duration * sampling_rate)
full_freqs = np.fft.rfftfreq(num_samples, 1 / sampling_rate)
freqs_mask = (full_freqs <= maximum_frequency) * (full_freqs >= minimum_frequency)
freqs = full_freqs[freqs_mask]
freqs_ti = ti.field(ti.f64, shape=freqs.shape)
freqs_ti.from_numpy(freqs)

params = dict(
    mass_1=36.0,
    mass_2=29.0,
    chi_1=-0.4,
    chi_2=0.02,
    luminosity_distance=800.0,
    inclination=0.4,
    reference_phase=1.2,
)

xas_tiw = IMRPhenomXAS(freqs_ti, reference_frequency)
xas_tiw.update_waveform(params)
xas_tiw.waveform_container_numpy
```
More examples or tutorials can be found in the [document]().


## Other Tools
If `tiwave` cannot meet your needs, you may find other packages for gravitaional waveform generation (welcome to open issues or pull requests if you know more):
- [`lalsimulation`](https://github.com/lscsoft/lalsuite)
- [`bbhx`](https://github.com/mikekatz04/BBHx)
- [`wf4py`](https://github.com/CosmoStatGW/WF4Py)
- [`ripple`](https://github.com/tedwards2412/ripple)


## Known Limitations
- For waveforms of X family, only the recommended configuration are implemented, except the multibanding settings which is currently not supported in `tiwave`. The details of recommended settings can be found in the document of `lalsimulation`, [LALSimIMRPhenomX.c](https://docs.ligo.org/lscsoft/lalsuite/lalsimulation/group___l_a_l_sim_i_m_r_phenom_x__c.html);
- The 32 mode in IMRPhenomXHM has relatively large numerical errors. However, we believe it is still safe to use for sources with moderate mass ratio and spins (more details can be found in the [notebook]() and Appendix A in the [paper]());
- Currently, automatic differentiation is only available for IMRPhenomXAS, and limited to the backward mode;


## Contact
The author strive to make this package easy-to-use and maintainable. But the author's experience and knowledge in software engineering is limited. Any feedback, comments, and suggestions are greatly appreciated. Feel free to open issues or contact.


## Citation
If you think this package is useful, please considering cite [arxiv: 2601.xxxx]().

The development of this package depending on many previous works, Please cite the original works for the corresponding modules you have used.
> IMRPhenomD: (validation for reimplemented IMRPhenomD in `tiwave` over the entire parameter space is still in progress)
> - Sascha Husa, et al. *Frequency-domain gravitational waves from nonprecessing black-hole binaries. I. New numerical waveforms and anatomy of the signal*, [Phys. Rev. D 93, 044006](https://doi.org/10.1103/PhysRevD.93.044006)
> - Sebastian Khan, et al. *Frequency-domain gravitational waves from nonprecessing black-hole binaries. II. A phenomenological model for the advanced detector era*, [Phys. Rev. D 93, 044007](https://doi.org/10.1103/PhysRevD.93.044007)
> 
> IMRPhenomXAS:
> - Geraint Pratten, et al. *Setting the cornerstone for a family of models for gravitational waves from compact binaries: The dominant harmonic for nonprecessing quasicircular black holes*, [Phys. Rev. D 102, 064001](https://doi.org/10.1103/PhysRevD.102.064001)
> 
> IMRPhenomXHM:
> - Cecilio García-Quirós, et al. *Multimode frequency-domain model for the gravitational wave signal from nonprecessing black-hole binaries*, [Phys. Rev. D 102, 064002](https://doi.org/10.1103/PhysRevD.102.064002)
> 
> IMRPhenomXPHM: (support for the precession effects in `tiwave` is still ongoing)
> - Geraint Pratten, et al. *Computationally efficient models for the dominant and subdominant harmonic modes of precessing binary black holes*, [Phys. Rev. D 103, 104056](https://doi.org/10.1103/PhysRevD.103.104056)
> - (the single-spin model) Mark Hannam, et al. *Simple Model of Complete Precessing Black-Hole-Binary Gravitational Waveforms*, [Phys. Rev. Lett. 113, 151101](https://doi.org/10.1103/PhysRevLett.113.151101)
> - (the double-spin model) Katerina Chatziioannou, et, al. *Constructing gravitational waves from generic spin-precessing compact binary inspirals*, [Phys. Rev. D 95, 104004](https://doi.org/10.1103/PhysRevD.95.104004)

