# tiwave - commonly used gravitational waveform models implemented with taichi-lang

![last commit](https://img.shields.io/github/last-commit/niuiniuin/tiwave)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![docs:]()]
[![test.yaml:]()]
[![test coverage:]()]
[![downloads:]()]

`tiwave` is a python implementation of several commonly used gravitational waveform models powered by [`taichi-lang`](https://www.taichi-lang.org/), developed mainly for preparatory science of space-borne gravitaional wave observation projects. It can be used to generate waveforms in high performance on both CPUs and GPUs, with useful features like automatic differentiation, while keeping python's virtues of usability and maintainability. 


## Supported models

- [x] TaylorF2 (arXiv: )
- [x] IMRPhenomD (arXiv: [1508.07250](https://arxiv.org/abs/1508.07250), [1508.07253](https://arxiv.org/abs/1508.07253))
- [x] IMRPhenomXAS (arXiv: [2001.11412](https://arxiv.org/abs/2001.11412))
- [x] IMRPhenomXHM (arXiv: [2001.10914](https://arxiv.org/abs/2001.10914))
- [ ] IMRPhenomXPHM (arXiv: [2004.06503](https://arxiv.org/abs/2004.06503))
- [ ] IMRPhenomXP 
- [ ] FASTGB (arXiv: [0704.1808](https://arxiv.org/abs/0704.1808))
- [ ] AAK (arXiv: )

*(For waveforms of X family, only the default configuration are implemented, see [docs](https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimIMRPhenomX.c#L136) in lalsimulation for more details.)*


## Installation
Using pip is the easiest way to install.
```sh
pip install git+https://github.com/niuiniuin/tiwave.git@master
```

While, if you are concerned that this may mess up your environment, you can use conda to create an isolated environment.
```sh
conda create -f https://raw.githubusercontent.com/niuiniuin/tiwave/refs/heads/master/environment.yml
conda activate tiwave
pip install git+https://github.com/niuiniuin/tiwave.git@master
```


## Quickstart
```python
import taichi as ti
ti.init(arch=ti.cpu, default_fp=ti.f64)

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

params_in = dict(
    mass_1=36.0,
    mass_2=29.0,
    chi_1=-0.4,
    chi_2=0.02,
    luminosity_distance=800.0,
    inclination=0.4,
    reference_phase=1.2,
)

xas_tiw = IMRPhenomXAS(freqs_ti, reference_frequency)
xas_tiw.update_waveform(parameters)
xas_tiw.waveform_container_numpy
```
More examples and detail APIs can be found in the [document]().


## Testing with lalsimulation

*Comparing the waveform*

*Mismatch in the whole parameter space*

*Performance*

(The performance tests are highly depending on the hardware and software environment. The results shown above are obtained on the [hanhai20](https://scc.ustc.edu.cn/zlsc/user_doc/html/introduction/hanhai20-introduction.html) system in USTC with Intel Xeon Scale 6248 and NVIDIA Tesla V100.)

More thorough tests can be found [here]().


## Other tools
If `tiwave` cannot meet your needs, you may find other packages for gravitaional waveform generation (Please open a new issue or PR if you know more):

- [`lalsimulation`](https://github.com/lscsoft/lalsuite)
- [`bbhx`](https://github.com/mikekatz04/BBHx)
- [`wf4py`](https://github.com/CosmoStatGW/WF4Py)
- [`ripple`](https://github.com/tedwards2412/ripple)


## Contact
Any suggestions and comments are extremely welcome. Feel free to open issues or email me (nrui@mail.ustc.edu.cn).


## Citation
If you think this package is useful, please considering cite
```
```
The development of this package depending on many previous works including:
```
```
Please cite the original works for the corresponding modules you have used.



