import os
import warnings
from typing import Optional

import taichi as ti
import taichi.math as tm
import numpy as np

from ..constants import *
from ..utils import ComplexNumber
from .base_waveform import BaseWaveform





@ti.data_oriented
class IMRPhenomXAS(BaseWaveform):
    def __init__(self, 
                 frequencies:ti.ScalarField, 
                 waveform_container: Optional[ti.StructField] = None, 
                 reference_frequency: Optional[float] = None, 
                 returned_form:str = 'polarizations', 
                 include_tf:bool=True, 
                 parameter_sanity_check:bool=False)->None:
        '''
        Parameters
        ==========
        frequencies: ti.field of f64, note that frequencies may not uniform spaced
        waveform_container: ti.Struct.field or None
            {} or {}
        returned_form: str
            `polarizations` or `amplitude_phase`, if waveform_container is given, this attribute will be neglected.
        parameter_sanity_check: bool

        Returns:
        ========
        array, the A channel without the prefactor which is determined by the TDI generation.

        TODO:
        check whether passed in `waveform_containter` consistent with `returned_form`
        '''
        self.frequencies = frequencies
        if reference_frequency is None:
            self.reference_frequency = self.frequencies[0]
        elif reference_frequency <= 0.0:
            raise ValueError(f'you are set reference_frequency={reference_frequency}, which must be postive.')
        else:
            self.reference_frequency = reference_frequency
            
        # TODO: make the sanity checks do not depend on the taichi debug mode
        self.parameter_sanity_check = parameter_sanity_check
        if self.parameter_sanity_check:
            warnings.warn('`parameter_sanity_check` is turn-on, make sure taichi is initialized with debug mode')
        else:
            warnings.warn('`parameter_sanity_check` is disable, make sure all parameters passed in are valid.')

        if waveform_container is not None:
            if not (waveform_container.shape == frequencies.shape):
                raise ValueError('passed in `waveform_container` and `frequencies` have different shape')
            self.waveform_container=waveform_container
            ret_content = self.waveform_container.keys
            if 'tf' in ret_content:
                include_tf = True
                ret_content.remove('tf')
            else:
                include_tf = False
            if all([item in ret_content for item in ['hplus', 'hcross']]):
                returned_form = 'polarizations'
                [ret_content.remove(item) for item in ['hplus', 'hcross']]
            elif all([item in ret_content for item in ['amplitude', 'phase']]):
                returned_form = 'amplitude_phase'
                [ret_content.remove(item) for item in ['amplitude', 'phase']]
            if len(ret_content) > 0:
                raise ValueError(f'`waveform_container` contains additional unknown keys {ret_content}.')
            self.returned_form = returned_form
            self.include_tf = include_tf
            print(f'Using `waveform_container` passed in, updating returned_form={self.returned_form}, include_tf={self.include_tf}')
        else:
            self._initialize_waveform_container(returned_form, include_tf)
            self.returned_form = returned_form
            self.include_tf = include_tf
            print(f'`waveform_container` is not given, initializing one with returned_form={returned_form}, include_tf={include_tf}')
            
        # initializing data struct with 0, and instantiating fields for global accessing
        self.source_parameters = SourceParameters.field(shape=())
        self.phase_coefficients = PhaseCoefficients.field(shape=())
        self.amplitude_coefficients = AmplitudeCoefficients.field(shape=())
        self.pn_prefactors = PostNewtonianPrefactors.field(shape=())

    def _initialize_waveform_container(self, returned_form:str, include_tf:bool)->None:
        ret_content = {}
        if returned_form == 'polarizations':
            ret_content.update({'hplus': ComplexNumber, 'hcross': ComplexNumber})
        elif returned_form == 'amplitude_phase':
            ret_content.update({'amplitude': ti.f64, 'phase': ti.f64})
        else:
            raise Exception(f'{returned_form} is unknown. `returned_form` can only be one of `polarizations` and `amplitude_phase`')

        if include_tf:
            ret_content.update({'tf': ti.f64})

        self.waveform_container = ti.Struct.field(ret_content, shape=(self.frequencies.length,),)
        return None

    def update_waveform(self, parameters:dict[str, float]):
        '''
        necessary preparation which need to be finished in python scope for waveform computation 
        (this function may be awkward, since no interpolation function in taichi-lang)
        '''
        self.source_parameters[None].generate_all_source_parameters(parameters)
        self._update_waveform_kernel()
    
    @ti.kernel
    def _update_waveform_kernel(self):
        if ti.static(self.parameter_sanity_check):
            self._parameter_check()
    
        self.pn_prefactors[None].compute_PN_prefactors(self.source_parameters[None])
        self.amplitude_coefficients[None].compute_amplitude_coefficients(self.source_parameters[None], self.pn_prefactors[None])
        self.phase_coefficients[None].compute_phase_coefficients(self.source_parameters[None], self.pn_prefactors[None])
        
        powers_of_Mf = UsefulPowers()

        powers_of_Mf.updating(self.amplitude_coefficients[None].f_peak)
        t0 = _d_phase_merge_ringdown_ansatz(powers_of_Mf, self.phase_coefficients[None], self.source_parameters[None].f_ring, self.source_parameters[None].f_damp)/self.source_parameters[None].eta
        # t0 = (_d_phase_merge_ringdown_ansatz(powers_of_Mf, self.phase_coefficients[None], self.source_parameters[None].f_ring, self.source_parameters[None].f_damp) + self.phase_coefficients[None].C2_merge_ringdown)/self.source_parameters[None].eta
        time_shift = t0 - 2*PI*self.source_parameters[None].tc/self.source_parameters[None].M_sec
        Mf_ref = self.source_parameters[None].M_sec * self.reference_frequency
        powers_of_Mf.updating(Mf_ref)
        phase_ref_temp = _compute_phase(powers_of_Mf, self.phase_coefficients[None], self.pn_prefactors[None], self.source_parameters[None].f_ring, self.source_parameters[None].f_damp, self.source_parameters[None].eta)
        phase_shift = 2.0*self.source_parameters[None].phase_ref + phase_ref_temp

        for idx in self.frequencies:
            Mf = self.source_parameters[None].M_sec * self.frequencies[idx]
            if Mf < FREQUENCY_CUT:
                powers_of_Mf.updating(Mf)            
                amplitude = _compute_amplitude(powers_of_Mf, self.amplitude_coefficients[None], self.pn_prefactors[None], self.source_parameters[None].f_ring, self.source_parameters[None].f_damp)
                phase = _compute_phase(powers_of_Mf, self.phase_coefficients[None], self.pn_prefactors[None], self.source_parameters[None].f_ring, self.source_parameters[None].f_damp, self.source_parameters[None].eta)
                phase -= time_shift * (Mf-Mf_ref) + phase_shift
                # remember multiple amp0 and shift phase and 1/eta
                if ti.static(self.returned_form == 'amplitude_phase'):
                    self.waveform_container[idx].amplitude = amplitude
                    self.waveform_container[idx].phase = phase
                if ti.static(self.returned_form == 'polarizations'):
                    self.waveform_container[idx].hcross, self.waveform_container[idx].hplus = _get_polarization_from_amplitude_phase(amplitude, phase, self.source_parameters[None].iota)
                if ti.static(self.include_tf):
                    tf = _compute_tf(powers_of_Mf, self.phase_coefficients[None], self.pn_prefactors[None], self.source_parameters[None].f_ring, self.source_parameters[None].f_damp, self.source_parameters[None].eta)
                    tf -= time_shift
                    tf *= self.source_parameters[None].M_sec / PI / 2
                    self.waveform_container[idx].tf = tf
            else:
                if ti.static(self.returned_form == 'amplitude_phase'):
                    self.waveform_container[idx].amplitude = 0.0
                    self.waveform_container[idx].phase = 0.0
                if ti.static(self.returned_form == 'polarization'):
                    self.waveform_container[idx].hcross.fill(0.0)
                    self.waveform_container[idx].hplus.fill(0.0)
                if ti.static(self.include_tf):
                    self.waveform_container[idx].tf = 0.0

    @ti.func
    def _parameter_check(self):
        assert (self.source_parameters[None].mass_1 > self.source_parameters[None].mass_2), \
               f'require m1 > m2, you are passing m1: {self.source_parameters[None].mass_1}, m2:{self.source_parameters[None].mass_2}'
        assert (self.source_parameters[None].q > 0.0 and self.source_parameters[None].q < 1.0), \
               f'require 0 < q < 1, you are passing q: {self.source_parameters[None].q}'
        assert (self.source_parameters[None].chi_1 > -1.0 and self.source_parameters[None].chi_1 < 1.0), \
               f'require -1 < chi_1 < 1, you are passing chi_1: {self.source_parameters[None].chi_1}'
        assert (self.source_parameters[None].chi_2 > -1.0 and self.source_parameters[None].chi_2 < 1.0), \
               f'require -1 < chi_2 < 1, you are passing chi_2: {self.source_parameters[None].chi_2}'

        # TODO more parameter check 
        
    def np_array_of_waveform_container(self):
        ret = {}
        if self.returned_form=='polarizations':
            hcross_array = self.waveform_container.hcross.to_numpy().view(dtype=np.complex128).reshape((self.frequencies.shape))
            hplus_array = self.waveform_container.hplus.to_numpy().view(dtype=np.complex128).reshape((self.frequencies.shape))
            ret['hcross'] = hcross_array
            ret['hplus'] = hplus_array
        elif self.returned_form=='amplitude_phase':
            amp_array = self.waveform_container.amplitude.to_numpy()
            phase_array = self.waveform_container.phase.to_numpy()
            ret['amplitude'] = amp_array
            ret['phase'] = phase_array
        if self.include_tf:
            tf_array = self.waveform_container.tf.to_numpy()
            ret['tf'] = tf_array
        return ret
        


@ti.data_oriented
class IMRPhenomXHM(BaseWaveform):
    pass


@ti.data_oriented
class IMRPhenomXP(BaseWaveform):
    pass


@ti.data_oriented
class IMRPhenomXPHM(BaseWaveform):
    pass