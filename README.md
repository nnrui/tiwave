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

params_in = dict(total_mass=2590590.325070,
                 mass_ratio=0.6560166,
                 chi_1=-0.448101,
                 chi_2=0.014488,
                 luminosity_distance=800.0,
                 inclination=2.123734,
                 reference_phase=3.211444,
                 reference_frequency=1e-5,
                 coalescence_time=0.0,)
minimum_frequency = 1e-5
maximum_frequency = 0.1

wf = IMRPhenomXAS(freqs_ti, reference_frequency=params_in['reference_frequency'])
wf.update_waveform(params_in)
```
More examples and detail APIs can be found in the document ().


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



