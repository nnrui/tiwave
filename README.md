# tiwave - Gravitational waveforms using taichi-lang

`tiwave` is a python implementation for some gravitational waveform models using [`taichi-lang`](https://www.taichi-lang.org/) which is designed for high-performance parallel programming and allows the codes to run fast while keeping the python's virtues of readability and maintainability.

## Supported waveforms
- IMRPhenomD
- IMRPhenomXAS
- IMRPhenomXHM
- IMRPhenomXPHM
- TaylorF2
- FASTGB

## Installation

## Basic usage

More examples and detail api can be found in the document().

## Testing with lalsimulation
(TODO: testing in whole parameter space)
Note, the performance is depend on the hardware and software environment. You may obtain differenc results when runing the same test script. The result shown above are obtained on hanhai20 system in USTC with H100

## Development
Code Organization
The codes are modularized as much as possible, which allow to easily modify and add some effects. A example of adding environment effects (cite[]) in inspiral can be found here.
(TODO: development installation)


## Similar tools
Other similar tools for gravitaional waveform generation which may meet your different needs (Due to the limitation of author's knowledge, this list may be not complete. Please open a new issue or PR if you know others.):

- [`lalsimulation`](https://github.com/lscsoft/lalsuite), the most authoritative and inclusive waveform library developed by LVK.
- [`bbhx`](https://github.com/mikekatz04/BBHx), using `cuda` to generate waveforms on GPU.
- [`wf4py`](https://github.com/CosmoStatGW/WF4Py), a python implementation of gravitaional waveforms.
- [`ripple`](https://github.com/tedwards2412/ripple), a python implementation of gravitaional wavefroms with `jax`.


## Citation
If you think this package is useful, please considering cite
```

```
The development of this package depending on many previous works including:

Please cite the original works for the correnpond modules you have used.


## Contact
Any suggestions and comments are extremely welcome. Feel free to open issues or email me (nrui@mail.ustc.edu.cn).


