# TODO:
# - spectial case of q=1
# - all idx begin from 0
# - computing power_of_Mf ahead of the main loop!!
import warnings
import os
from typing import Optional

import taichi as ti
import taichi.math as tm
import numpy as np

from ..constants import *
from ..utils import ComplexNumber, gauss_elimination, UsefulPowers, sub_struct_from
from .common import PostNewtonianCoefficients
from .base_waveform import BaseWaveform
from .IMRPhenomXAS import IMRPhenomXAS
from .IMRPhenomXAS import SourceParameters as SourceParametersMode22
from .IMRPhenomXAS import PhaseCoefficients as PhaseCoefficientsMode22
from .IMRPhenomXAS import AmplitudeCoefficients as AmplitudeCoefficientsMode22
from .IMRPhenomXAS import _time_shift_psi4_to_strain

# TODO: temp
from .IMRPhenomXAS import _compute_phase, _compute_tf


useful_powers_pi = UsefulPowers()
useful_powers_pi.update(PI)
useful_powers_2pi = UsefulPowers()
useful_powers_2pi.update(2.0 * PI)
useful_powers_2pi_over_3 = UsefulPowers()
useful_powers_2pi_over_3.update(2.0 * PI / 3.0)
useful_powers_pi_over_2 = UsefulPowers()
useful_powers_pi_over_2.update(0.5 * PI)
eta_EMR = 0.05
QNM_frequencies_struct = ti.types.struct(
    f_ring=ti.f64,
    f_damp=ti.f64,
    # cache for repeatedly using
    f_ring_pow2=ti.f64,
    f_damp_pow2=ti.f64,
)


@ti.func
def _amp_ins_f_end_EMR(m: ti.f64, source_params: ti.template()) -> ti.f64:
    """The end frequency of inspiral amplitude for extreme mass ratio."""
    return m * (
        1.25
        * (
            (
                0.011671068725758493
                - 0.0000858396080377194 * source_params.chi_1
                + 0.000316707064291237 * source_params.chi_1 * source_params.chi_1
            )
            * (0.8447212540381764 + 6.2873167352395125 * source_params.eta)
        )
        / (1.2857082764038923 - 0.9977728883419751 * source_params.chi_1)
    )


@sub_struct_from(PostNewtonianCoefficients)
class PostNewtonianCoefficientsHighModesBase:

    @ti.func
    def _set_rescaling_phase_coefficients(
        self, m: ti.f64, pn_coefficients_22: ti.template()
    ):
        """
        Note in the XAS model, we only implemente the default 104 inspiral configuration
        where the canonical TaylorF2 3.5PN phase along with spin corrections at 4PN
        are incorporated (TODO: check the TaylorF2 implementation!!). So we do not set coefficients of higher PN orders here.
        """
        m_over_2 = 0.5 * m

        self.phi_0 = m_over_2 ** (8.0 / 3.0) * pn_coefficients_22.phi_0
        self.phi_1 = m_over_2 ** (7.0 / 3.0) * pn_coefficients_22.phi_1
        self.phi_2 = m_over_2 ** (2.0) * pn_coefficients_22.phi_2
        self.phi_3 = m_over_2 ** (5.0 / 3.0) * pn_coefficients_22.phi_3
        self.phi_4 = m_over_2 ** (4.0 / 3.0) * pn_coefficients_22.phi_4
        self.phi_5l = m_over_2 * pn_coefficients_22.phi_5l
        self.phi_6 = m_over_2 ** (2.0 / 3.0) * (
            pn_coefficients_22.phi_6 - tm.log(m_over_2) * pn_coefficients_22.phi_6l
        )
        self.phi_6l = m_over_2 ** (2.0 / 3.0) * pn_coefficients_22.phi_6l
        self.phi_7 = m_over_2 ** (1.0 / 3.0) * pn_coefficients_22.phi_7
        self.phi_8 = (
            pn_coefficients_22.phi_8 - tm.log(m_over_2) * pn_coefficients_22.phi_8l
        )
        self.phi_8l = pn_coefficients_22.phi_8l


@ti.dataclass
class AmplitudeCoefficientsHighModesBase:
    # Inspiral
    rho_1: ti.f64
    rho_2: ti.f64
    rho_3: ti.f64
    ins_f_end: ti.f64
    ins_colloc_points: ti.types.vector(3, ti.f64)
    ins_colloc_values: ti.types.vector(3, ti.f64)
    # Intermediate
    # (note the implementaion of PhenomXHMReleaseVersion 122022 has many difference with that shown in the paper)
    int_f_end: ti.f64
    int_colloc_points: ti.types.vector(6, ti.f64)
    int_colloc_values: ti.types.vector(8, ti.f64)
    int_Ab: ti.types.matrix(8, 9, dtype=ti.f64)
    int_ansatz_coefficients: ti.types.vector(8, ti.f64)
    # Merge-ringdown
    MRD_f_falloff: ti.f64
    MRD_colloc_points: ti.types.vector(3, ti.f64)
    MRD_colloc_values: ti.types.vector(3, ti.f64)
    MRD_ansatz_coefficients: ti.types.vector(5, ti.f64)
    #
    useful_powers: ti.types.struct(
        ins_f1=UsefulPowers, ins_f2=UsefulPowers, ins_f3=UsefulPowers
    )

    @ti.func
    def amplitude_inspiral_ansatz(
        self, pn_coefficients_HM: ti.template(), powers_of_Mf: ti.template()
    ) -> ti.f64:
        """
        without the leading prefactor pi*sqrt(2*eta/3)*(pi Mf)^(-7/2)
        """
        return (
            pn_coefficients_HM.PN_amplitude(powers_of_Mf)
            + self.rho_1 * powers_of_Mf.seven_thirds
            + self.rho_2 * powers_of_Mf.eight_thirds
            + self.rho_3 * powers_of_Mf.three
        )

    @ti.func
    def d_amplitude_inspiral_ansatz(
        self,
        pn_coefficients_HM: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> ti.f64:
        return pn_coefficients_HM.d_PN_amplitude(powers_of_Mf) + (
            +7.0 / 3.0 * self.rho_1 * powers_of_Mf.four_thirds
            + 8.0 / 3.0 * self.rho_2 * powers_of_Mf.five_thirds
            + 9.0 / 3.0 * self.rho_3 * powers_of_Mf.two
        )

    @ti.func
    def amplitude_intermediate_ansatz(self, Mf: ti.f64) -> ti.f64:
        """
        Note the ansatz for amplitude intermediate used in ReleaseVersion 122022 is different with that shown in the paper.
        without PN0
        """
        fpower = 1.0
        ret = 0.0
        for i in ti.static(range(8)):
            ret += self.int_ansatz_coefficients[i] * fpower
            fpower *= Mf
        return ret

    @ti.func
    def amplitdue_merge_ringdown_ansatz_Lorentzian(
        self, f_minus_fring: ti.f64
    ) -> ti.f64:
        return (
            self.gamma1_gamma3_fdamp
            / (f_minus_fring * f_minus_fring + self.gamma3_fdamp_pow2)
            * tm.exp(-f_minus_fring * self.gamma2_over_gamma3_fdamp)
        )

    @ti.func
    def amplitude_merge_ringdown_ansatz_falloff(self, Mf: ti.f64) -> ti.f64:
        return self.MRD_ansatz_coefficients[3] * tm.exp(
            -self.MRD_ansatz_coefficients[4] * (Mf - self.MRD_f_falloff)
        )

    @ti.func
    def amplitude_merge_ringdown_ansatz(self, Mf: ti.f64) -> ti.f64:
        """used for no mixing cases 21, 33, 44, for 32 mode, this method need to be overrided"""
        if (self.MRD_f_falloff > 0.0) and (
            Mf >= self.MRD_f_falloff
        ):  # exponential falloff
            return self.amplitude_merge_ringdown_ansatz_falloff(self, Mf)
        else:  # Lorentzian, same with the MRD ansatz of PhenomXAS
            return self.amplitdue_merge_ringdown_ansatz_Lorentzian(self, Mf)

    @ti.func
    def d_amplitude_merge_ringdown_ansatz(self, source_params: ti.f64) -> ti.f64:
        pass

    @ti.func
    def update_amplitude_coefficients(
        self,
        pn_coefficients_HM: ti.template(),
        source_params: ti.template(),
    ):
        self._set_ins_colloc_points(
            source_params,
            source_params,
        )
        self._set_inspiral_coefficients(
            source_params,
            source_params,
            pn_coefficients_HM,
        )
        self._set_merge_ringdown_coefficients()


@ti.dataclass
class PhaseCoefficientsHighModesBase:
    # Inspiral
    sigma_1: ti.f64
    sigma_2: ti.f64
    sigma_3: ti.f64
    sigma_4: ti.f64
    # only consider 4 pseudo-PN coefficients in the default 104 inspiral configuration of XAS model
    Lambda_lm: ti.f64  # corrections for the complex PN amplitudes, eq 4.9
    ins_f_end: ti.f64
    ins_C0: ti.f64
    ins_C1: ti.f64

    # Intermediate
    c_0: ti.f64
    c_1: ti.f64
    c_2: ti.f64
    c_3: ti.f64  # only used for mode 32, set to 0.0 for mode 21, 33, 44
    c_4: ti.f64
    c_L: ti.f64
    int_f_end: ti.f64
    int_colloc_points: ti.types.vector(6, ti.f64)
    int_colloc_values: ti.types.vector(6, ti.f64)

    # Merge-ringdown
    alpha_2: ti.f64
    alpha_L: ti.f64
    MRD_C0: ti.f64
    MRD_C1: ti.f64

    # constant for aligning each mode under the choice of tetrad convention
    delta_phi_lm: ti.f64

    @ti.func
    def _set_ins_rescaling_coefficients(
        self,
        m: ti.f64,
        phase_coefficients_22: ti.template(),
    ):
        m_over_2 = 0.5 * m
        self.sigma_1 = phase_coefficients_22.sigma_1
        self.sigma_2 = phase_coefficients_22.sigma_2 / m_over_2 ** (1.0 / 3.0)
        self.sigma_3 = phase_coefficients_22.sigma_3 / m_over_2 ** (2.0 / 3.0)
        self.sigma_4 = phase_coefficients_22.sigma_4 / m_over_2

    @ti.func
    def _set_MRD_rescaling_coefficients(
        self,
        w_lm: ti.f64,
        QNM_freqs_lm: ti.template(),
        source_params: ti.template(),
    ):
        """
        for 33, 44 mode, w_lm = 2.0
        for 21 mode, w_lm = 1/3.0
        """
        # TODO:1/eta ??
        # used for modes without significant mixing
        self.alpha_L = self._fit_mode22_alpha_L(source_params)
        self.alpha_2 = (
            w_lm / QNM_freqs_lm.f_ring_pow2 * self._fit_mode22_alpha_2(source_params)
        )

    @ti.func
    def _fit_mode22_alpha_2(self, source_params: ti.template()) -> ti.f64:
        return (
            0.2088669311744758
            - 0.37138987533788487 * source_params.eta
            + 6.510807976353186 * source_params.eta_pow2
            - 31.330215053905395 * source_params.eta_pow3
            + 55.45508989446867 * source_params.eta_pow4
            + (
                (
                    0.2393965714370633
                    + 1.6966740823756759 * source_params.eta
                    - 16.874355161681766 * source_params.eta_pow2
                    + 38.61300158832203 * source_params.eta_pow3
                )
                * source_params.S_tot_hat
            )
            / (1.0 - 0.633218538432246 * source_params.S_tot_hat)
            + source_params.delta_chi
            * (
                0.9088578269496244 * source_params.eta**2.5
                + 15.619592332008951 * source_params.delta_chi * source_params.eta**3.5
            )
            * source_params.delta
        )

    @ti.func
    def _fit_mode22_alpha_L(self, source_params: ti.template()) -> ti.f64:
        return (
            (
                -1.1926122248825484
                + 2.5400257699690143 * source_params.eta
                - 16.504334734464244 * source_params.eta_pow2
                + 27.623649807617376 * source_params.eta_pow3
            )
            + source_params.eta_pow2
            * source_params.S_tot_hat
            * (
                35.803988443700824
                + 9.700178927988006 * source_params.S_tot_hat
                - 77.2346297158916 * source_params.S_tot_hat_pow2
            )
            + source_params.S_tot_hat
            * (
                0.1034526554654983
                - 0.21477847929548569 * source_params.S_tot_hat
                - 0.06417449517826644 * source_params.S_tot_hat_pow2
            )
            + source_params.eta
            * source_params.S_tot_hat
            * (
                -4.7282481007397825
                + 0.8743576195364632 * source_params.S_tot_hat
                + 8.170616575493503 * source_params.S_tot_hat_pow2
            )
            + source_params.eta_pow3
            * source_params.S_tot_hat
            * (
                -72.50310678862684
                - 39.83460092417137 * source_params.S_tot_hat
                + 180.8345521274853 * source_params.S_tot_hat_pow2
            )
            + (
                -0.7428134042821221
                * source_params.chi_1
                * source_params.eta_pow2
                * source_params.eta_sqrt
                + 0.7428134042821221
                * source_params.chi_2
                * source_params.eta_pow2
                * source_params.eta_sqrt
                + 17.588573345324154
                * source_params.chi_1_pow2
                * source_params.eta_pow3
                * source_params.eta_sqrt
                - 35.17714669064831
                * source_params.chi_1
                * source_params.chi_2
                * source_params.eta_pow3
                * source_params.eta_sqrt
                + 17.588573345324154
                * source_params.chi_2_pow2
                * source_params.eta_pow3
                * source_params.eta_sqrt
            )
            * source_params.delta
        )

    @ti.func
    def _set_delta_phi_lm(
        self, source_params: ti.template(), phase_coefficients_22: ti.template()
    ):
        pass

    @ti.func
    def _inspiral_phase(
        self,
        pn_coefficients_lm: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> ti.f64:
        return (
            pn_coefficients_lm.PN_phase(powers_of_Mf)
            + (
                +self.sigma_1 * powers_of_Mf.one
                + 0.75 * self.sigma_2 * powers_of_Mf.four_thirds
                + 0.6 * self.sigma_3 * powers_of_Mf.five_thirds
                + 0.5 * self.sigma_4 * powers_of_Mf.two
            )
            + self.Lambda_lm * powers_of_Mf.one
        )

    @ti.func
    def _inspiral_d_phase(
        self,
        pn_coefficients_lm: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> ti.f64:
        return (
            pn_coefficients_lm.d_PN_phase(powers_of_Mf)
            + (
                self.sigma_1
                + self.sigma_2 * powers_of_Mf.third
                + self.sigma_3 * powers_of_Mf.two_thirds
                + self.sigma_4 * powers_of_Mf.one
            )
            + self.Lambda_lm
        )

    @ti.func
    def _intermediate_phase(
        self,
        QNM_freqs_lm: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> ti.f64:
        return (
            self.c_0 * powers_of_Mf.one
            + self.c_1 * powers_of_Mf.log
            - self.c_2 / powers_of_Mf.one
            - self.c_3 / 2.0 / powers_of_Mf.two
            - self.c_4 / 3.0 / powers_of_Mf.three
            + self.c_L
            * tm.atan2((powers_of_Mf.one - QNM_freqs_lm.f_ring), QNM_freqs_lm.f_damp)
        )

    @ti.func
    def _intermediate_d_phase(
        self,
        QNM_freqs_lm: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> ti.f64:
        return (
            self.c_0
            + self.c_1 / powers_of_Mf.one
            + self.c_2 / powers_of_Mf.two
            + self.c_3 / powers_of_Mf.three
            + self.c_4 / powers_of_Mf.four
            + self.c_L
            * QNM_freqs_lm.f_damp
            / (QNM_freqs_lm.f_damp_pow2 + (powers_of_Mf.one - QNM_freqs_lm.f_ring) ** 2)
        )

    @ti.func
    def _merge_ringdown_phase(
        self, QNM_freqs_lm: ti.template(), powers_of_Mf: ti.template()
    ) -> ti.f64:
        """for no mixing modes"""
        return (
            -self.alpha_2 * QNM_freqs_lm.f_ring_pow2 / powers_of_Mf.one
            + self.alpha_L
            * tm.atan2((powers_of_Mf.one - QNM_freqs_lm.f_ring), QNM_freqs_lm.f_damp)
        )

    @ti.func
    def _merge_ringdown_d_phase(
        self, QNM_freqs_lm: ti.template(), powers_of_Mf: ti.template()
    ) -> ti.f64:
        """for no mixing modes"""
        return (
            self.alpha_2 * QNM_freqs_lm.f_ring_pow2 / powers_of_Mf.two
            + self.alpha_L
            * QNM_freqs_lm.f_damp
            / (QNM_freqs_lm.f_damp_pow2 + (powers_of_Mf.one - QNM_freqs_lm.f_ring) ** 2)
        )

    # @ti.func
    # def phase(
    #     self,
    #     pn_coefficients_lm: ti.template(),
    #     source_params: ti.template(),
    #     powers_of_Mf: ti.template(),
    # ):
    #     phase = 0.0
    #     if powers_of_Mf.one < self.ins_f_end:
    #         phase = (
    #             self.inspiral_phase(pn_coefficients_lm, source_params)
    #             + self.ins_C0
    #             + self.int_C1 * powers_of_Mf.one
    #         )
    #     elif (
    #         powers_of_Mf.one > self.int_f_end
    #     ):
    #         phase(
    #             self.merge_ringdown_phase()
    #             + +self.C0_MRD
    #             + self.C1_MRD * powers_of_Mf.one
    #         )
    #     else:
    #         phase = self.intermediate_phase()

    #     return phase / source_params.eta

    # @ti.func
    # def d_phase(self, powers_of_Mf: ti.template()):
    #     pass


@sub_struct_from(SourceParametersMode22)
class SourceParametersHighModes:
    eta_sqrt: ti.f64
    eta_pow7: ti.f64
    eta_pow8: ti.f64

    chi_1_pow2: ti.f64
    chi_2_pow2: ti.f64
    delta_chi_half: ti.f64
    delta_chi_half_pow2: ti.f64

    # the common factor of amplitude of the dominant PN order
    amp_common_factor: ti.f64
    # scale factor of mass and distance
    scale_factor: ti.f64
    # the fit of the time-difference between peak of strain and psi4
    dt_psi4_to_strain: ti.f64
    # QNM frequencies
    QNM_freqs_lm: ti.types.struct(
        **{
            "21": QNM_frequencies_struct,
            "33": QNM_frequencies_struct,
            "32": QNM_frequencies_struct,
            "44": QNM_frequencies_struct,
        }
    )
    # rescaling frequencies
    f_MECO_lm: ti.types.struct(
        **{"21": ti.f64, "33": ti.f64, "32": ti.f64, "44": ti.f64}
    )
    f_ISCO_lm: ti.types.struct(
        **{"21": ti.f64, "33": ti.f64, "32": ti.f64, "44": ti.f64}
    )

    @ti.func
    def update_source_parameters(
        self,
        mass_1: ti.f64,
        mass_2: ti.f64,
        chi_1: ti.f64,
        chi_2: ti.f64,
        luminosity_distance: ti.f64,
        inclination: ti.f64,
        reference_phase: ti.f64,
        coalescence_time: ti.f64,
        high_modes: ti.template(),
    ):
        self._parent_update_source_parameters(
            mass_1,
            mass_2,
            chi_1,
            chi_2,
            luminosity_distance,
            inclination,
            reference_phase,
            coalescence_time,
        )
        self.eta_sqrt = tm.sqrt(self.eta)
        self.eta_pow7 = self.eta * self.eta_pow6
        self.eta_pow8 = self.eta * self.eta_pow7

        self.chi_1_pow2 = self.chi_1 * self.chi_1
        self.chi_2_pow2 = self.chi_2 * self.chi_2
        self.delta_chi_half = self.delta_chi * 0.5
        self.delta_chi_half_pow2 = self.delta_chi_half * self.delta_chi_half

        self.amp_common_factor = (
            0.25 * tm.sqrt(10.0 / 3.0 * self.eta) / useful_powers_pi.two_thirds
        )
        self.scale_factor = self.M**2 / self.dL_SI * MRSUN_SI * MTSUN_SI
        self.dt_psi4_to_strain = -2.0 * PI * (500.0 + _time_shift_psi4_to_strain(self))

        if ti.static("21" in high_modes):
            self._set_QNM_frequencies_21()
            self.QNM_freqs_lm["21"].f_ring_pow2 = self.QNM_freqs_lm["21"].f_ring ** 2
            self.QNM_freqs_lm["21"].f_damp_pow2 = self.QNM_freqs_lm["21"].f_damp ** 2
            self.f_MECO_lm["21"] = 0.5 * self.f_MECO
            self.f_ISCO_lm["21"] = 0.5 * self.f_ISCO
        if ti.static("33" in high_modes):
            self._set_QNM_frequencies_33()
            self.QNM_freqs_lm["33"].f_ring_pow2 = self.QNM_freqs_lm["33"].f_ring ** 2
            self.QNM_freqs_lm["33"].f_damp_pow2 = self.QNM_freqs_lm["33"].f_damp ** 2
            self.f_MECO_lm["33"] = 1.5 * self.f_MECO
            self.f_ISCO_lm["33"] = 1.5 * self.f_ISCO
        if ti.static("32" in high_modes):
            self._set_QNM_frequencies_32()
            self.QNM_freqs_lm["32"].f_ring_pow2 = self.QNM_freqs_lm["32"].f_ring ** 2
            self.QNM_freqs_lm["32"].f_damp_pow2 = self.QNM_freqs_lm["32"].f_damp ** 2
            self.f_MECO_lm["32"] = self.f_MECO
            self.f_ISCO_lm["32"] = self.f_ISCO
        if ti.static("44" in high_modes):
            self._set_QNM_frequencies_44()
            self.QNM_freqs_lm["44"].f_ring_pow2 = self.QNM_freqs_lm["44"].f_ring ** 2
            self.QNM_freqs_lm["44"].f_damp_pow2 = self.QNM_freqs_lm["44"].f_damp ** 2
            self.f_MECO_lm["44"] = 2.0 * self.f_MECO
            self.f_ISCO_lm["44"] = 2.0 * self.f_ISCO

    @ti.func
    def _set_QNM_frequencies_21(self):
        self.QNM_freqs_lm["21"].f_ring = (
            (
                0.059471695665734674
                - 0.07585416297991414 * self.final_spin
                + 0.021967909664591865 * self.final_spin_pow2
                - 0.0018964744613388146 * self.final_spin_pow3
                + 0.001164879406179587 * self.final_spin_pow4
                - 0.0003387374454044957 * self.final_spin_pow5
            )
            / (
                1
                - 1.4437415542456158 * self.final_spin
                + 0.49246920313191234 * self.final_spin_pow2
            )
            / self.final_mass
        )
        self.QNM_freqs_lm["21"].f_damp = (
            (
                2.0696914454467294
                - 3.1358071947583093 * self.final_spin
                + 0.14456081596393977 * self.final_spin_pow2
                + 1.2194717985037946 * self.final_spin_pow3
                - 0.2947372598589144 * self.final_spin_pow4
                + 0.002943057145913646 * self.final_spin_pow5
            )
            / (
                146.1779212636481
                - 219.81790388304876 * self.final_spin
                + 17.7141194900164 * self.final_spin_pow2
                + 75.90115083917898 * self.final_spin_pow3
                - 18.975287709794745 * self.final_spin_pow4
            )
            / self.final_mass
        )

    @ti.func
    def _set_QNM_frequencies_33(self):
        self.QNM_freqs_lm["33"].f_ring = (
            (
                0.09540436245212061
                - 0.22799517865876945 * self.final_spin
                + 0.13402916709362475 * self.final_spin_pow2
                + 0.03343753057911253 * self.final_spin_pow3
                - 0.030848060170259615 * self.final_spin_pow4
                - 0.006756504382964637 * self.final_spin_pow5
                + 0.0027301732074159835 * self.final_spin_pow6
            )
            / (
                1
                - 2.7265947806178334 * self.final_spin
                + 2.144070539525238 * self.final_spin_pow2
                - 0.4706873667569393 * self.final_spin_pow4
                + 0.05321818246993958 * self.final_spin_pow6
            )
            / self.final_mass
        )
        self.QNM_freqs_lm["33"].f_damp = (
            (
                0.014754148319335946
                - 0.03124423610028678 * self.final_spin
                + 0.017192623913708124 * self.final_spin_pow2
                + 0.001034954865629645 * self.final_spin_pow3
                - 0.0015925124814622795 * self.final_spin_pow4
                - 0.0001414350555699256 * self.final_spin_pow5
            )
            / (
                1
                - 2.0963684630756894 * self.final_spin
                + 1.196809702382645 * self.final_spin_pow2
                - 0.09874113387889819 * self.final_spin_pow4
            )
            / self.final_mass
        )

    @ti.func
    def _set_QNM_frequencies_32(self):
        self.QNM_freqs_lm["32"].f_ring = (
            (
                0.09540436245212061
                - 0.13628306966373951 * self.final_spin
                + 0.030099881830507727 * self.final_spin_pow2
                - 0.000673589757007597 * self.final_spin_pow3
                + 0.0118277880067919 * self.final_spin_pow4
                + 0.0020533816327907334 * self.final_spin_pow5
                - 0.0015206141948469621 * self.final_spin_pow6
            )
            / (
                1
                - 1.6531854335715193 * self.final_spin
                + 0.5634705514193629 * self.final_spin_pow2
                + 0.12256204148002939 * self.final_spin_pow4
                - 0.027297817699401976 * self.final_spin_pow6
            )
            / self.final_mass
        )
        self.QNM_freqs_lm["32"].f_damp = (
            (
                0.014754148319335946
                - 0.03445752346074498 * self.final_spin
                + 0.02168855041940869 * self.final_spin_pow2
                + 0.0014945908223317514 * self.final_spin_pow3
                - 0.0034761714223258693 * self.final_spin_pow4
            )
            / (
                1
                - 2.320722660848874 * self.final_spin
                + 1.5096146036915865 * self.final_spin_pow2
                - 0.18791187563554512 * self.final_spin_pow4
            )
            / self.final_mass
        )

    @ti.func
    def _set_QNM_frequencies_44(self):
        self.QNM_freqs_lm["44"].f_ring = (
            (
                0.1287821193485683
                - 0.21224284094693793 * self.final_spin
                + 0.0710926778043916 * self.final_spin_pow2
                + 0.015487322972031054 * self.final_spin_pow3
                - 0.002795401084713644 * self.final_spin_pow4
                + 0.000045483523029172406 * self.final_spin_pow5
                + 0.00034775290179000503 * self.final_spin_pow6
            )
            / (
                1
                - 1.9931645124693607 * self.final_spin
                + 1.0593147376898773 * self.final_spin_pow2
                - 0.06378640753152783 * self.final_spin_pow4
            )
            / self.final_mass
        )
        self.QNM_freqs_lm["44"].f_damp = (
            (
                0.014986847152355699
                - 0.01722587715950451 * self.final_spin
                - 0.0016734788189065538 * self.final_spin_pow2
                + 0.0002837322846047305 * self.final_spin_pow3
                + 0.002510528746148588 * self.final_spin_pow4
                + 0.00031983835498725354 * self.final_spin_pow5
                + 0.000812185411753066 * self.final_spin_pow6
            )
            / (
                1
                - 1.1350205970682399 * self.final_spin
                - 0.0500827971270845 * self.final_spin_pow2
                + 0.13983808071522857 * self.final_spin_pow4
                + 0.051876225199833995 * self.final_spin_pow6
            )
            / self.final_mass
        )


@sub_struct_from(PostNewtonianCoefficientsHighModesBase)
class PostNewtonianCoefficientsMode21:

    @ti.func
    def update_pn_coefficients(
        self,
        pn_coefficients_22: ti.template(),
        source_params: ti.template(),
    ):
        self._set_rescaling_phase_coefficients(1.0, pn_coefficients_22)

        self.amp_global = tm.sqrt(2.0) / 3.0
        self.A_0 = 0.0
        self.A_1 = source_params.delta * useful_powers_2pi.third
        self.A_2 = (
            -3.0
            / 2.0
            * (source_params.chi_a + source_params.chi_s * source_params.delta)
            * useful_powers_2pi.two_thirds
        )
        self.A_3 = (
            3.35 / 6.72 * source_params.delta
            + 11.7 / 5.6 * source_params.delta * source_params.eta
        ) * useful_powers_2pi.one
        self.A_4 = (
            tm.length(
                ComplexNumber(
                    [
                        (0.5 + 2.0 * tm.log(2.0)) * source_params.delta,
                        (
                            -source_params.delta * PI
                            + 3.427 / 1.344 * source_params.chi_a
                            - 21.01 / 3.36 * source_params.chi_a * source_params.eta
                            + 3.427 / 1.344 * source_params.chi_s * source_params.delta
                            - 9.65
                            / 3.36
                            * source_params.chi_s
                            * source_params.delta
                            * source_params.eta
                        ),
                    ]
                )
            )
            * useful_powers_2pi.four_thirds
        )
        self.A_5 = (
            -0.964357 / 8.128512 * source_params.delta
            - 3.6529 / 1.2544 * source_params.delta * source_params.eta
            + 21.365 / 8.064 * source_params.delta * source_params.eta_pow2
            + 3.0 * source_params.chi_a * PI
            + 3.0 * source_params.chi_s * source_params.delta * PI
            - 30.7 / 3.2 * source_params.chi_a_pow2 * source_params.delta
            + 10.0 * source_params.chi_a_pow2 * source_params.delta * source_params.eta
            - 30.7 / 3.2 * source_params.chi_s_pow2 * source_params.delta
            + 39.0
            / 8.0
            * source_params.chi_s_pow2
            * source_params.delta
            * source_params.eta
            - 30.7 / 1.6 * source_params.chi_a * source_params.chi_s
            + 213.0
            / 4.0
            * source_params.chi_a
            * source_params.chi_s
            * source_params.eta
        ) * useful_powers_2pi.five_thirds
        self.A_6 = (
            tm.length(
                ComplexNumber(
                    [
                        (
                            (3.35 / 13.44 + 3.35 / 3.36 * tm.log(2.0))
                            * source_params.delta
                            + (14.89 / 1.12 + 8.9 / 2.8 * tm.log(2.0))
                            * source_params.delta
                            * source_params.eta
                        ),
                        (
                            -2.455 / 1.344 * source_params.delta * PI
                            + 4.17 / 1.12 * source_params.delta * source_params.eta * PI
                            + 143.063173 / 5.419008 * source_params.chi_a
                            - 227.58317
                            / 2.25792
                            * source_params.chi_a
                            * source_params.eta
                            + 42.617
                            / 1.792
                            * source_params.chi_a
                            * source_params.eta_pow2
                            + 143.063173
                            / 5.419008
                            * source_params.chi_s
                            * source_params.delta
                            - 70.49629
                            / 2.25792
                            * source_params.chi_s
                            * source_params.delta
                            * source_params.eta
                            - 5.47
                            / 7.68
                            * source_params.chi_s
                            * source_params.delta
                            * source_params.eta_pow2
                            + 24.3 / 6.4 * source_params.chi_a_pow3
                            - 15.0 * source_params.chi_a_pow3 * source_params.eta
                            + 24.3
                            / 6.4
                            * source_params.chi_a_pow3
                            * source_params.delta
                            - 3.0
                            / 16.0
                            * source_params.chi_s_pow3
                            * source_params.delta
                            * source_params.eta
                            + 72.9
                            / 6.4
                            * source_params.chi_a
                            * source_params.chi_s_pow2
                            + 72.9
                            / 6.4
                            * source_params.chi_a_pow2
                            * source_params.chi_s
                            * source_params.delta
                            - 15.0
                            * source_params.chi_a_pow2
                            * source_params.chi_s
                            * source_params.delta
                            * source_params.eta
                            - 48.9
                            / 1.6
                            * source_params.chi_a
                            * source_params.chi_s_pow2
                            * source_params.eta
                        ),
                    ]
                )
            )
            * useful_powers_2pi.two
        )


@sub_struct_from(PostNewtonianCoefficientsHighModesBase)
class PostNewtonianCoefficientsMode33:

    @ti.func
    def update_pn_coefficients(
        self,
        pn_coefficients_22: ti.template(),
        source_params: ti.template(),
    ):
        self._set_rescaling_phase_coefficients(3.0, pn_coefficients_22)

        self.amp_global = 0.75 * tm.sqrt(5.0 / 7.0)  # TODO: minus??
        self.A_0 = 0.0
        self.A_1 = source_params.delta * useful_powers_2pi_over_3.third
        self.A_2 = 0.0
        self.A_3 = (
            -19.45 / 6.72 * source_params.delta
            + 27.0 / 8.0 * source_params.delta * source_params.eta
        ) * useful_powers_2pi_over_3.one
        self.A_4 = (
            tm.length(
                ComplexNumber(
                    [
                        (21.0 / 5.0 - 6.0 * tm.log(1.5)) * source_params.delta,
                        (
                            source_params.delta * PI
                            + 6.5 / 2.4 * source_params.chi_a
                            - 28.0 / 3.0 * source_params.chi_a * source_params.eta
                            + 6.5 / 2.4 * source_params.chi_s * source_params.delta
                            - 2.0
                            / 3.0
                            * source_params.chi_s
                            * source_params.delta
                            * source_params.eta
                        ),
                    ]
                )
            )
            * useful_powers_2pi_over_3.four_thirds
        )
        self.A_5 = (
            -10.77664867 / 4.47068160 * source_params.delta
            - 117.58073 / 8.87040 * source_params.delta * source_params.eta
            + 42.0389 / 6.3360 * source_params.delta * source_params.eta_pow2
            - 8.1 / 3.2 * source_params.chi_a_pow2 * source_params.delta
            - 8.1 / 3.2 * source_params.chi_s_pow2 * source_params.delta
            + 10.0 * source_params.chi_a_pow2 * source_params.delta * source_params.eta
            + 1.0
            / 8.0
            * source_params.chi_s_pow2
            * source_params.delta
            * source_params.eta
            - 8.1 / 1.6 * source_params.chi_a * source_params.chi_s
            + 81.0 / 4.0 * source_params.chi_a * source_params.chi_s * source_params.eta
        ) * useful_powers_2pi_over_3.five_thirds
        self.A_6 = (
            tm.length(
                ComplexNumber(
                    [
                        (
                            (-38.9 / 3.2 + 19.45 / 1.12 * tm.log(1.5))
                            * source_params.delta
                            + (440.957 / 9.720 - 69.0 / 4.0 * tm.log(1.5))
                            * source_params.delta
                            * source_params.eta
                        ),
                        (
                            -5.675 / 1.344 * source_params.delta * PI
                            + 13.1 / 1.6 * source_params.delta * source_params.eta * PI
                            + 16.3021 / 1.6128 * source_params.chi_a
                            - 148.501 / 4.032 * source_params.chi_a * source_params.eta
                            - 13.7 / 2.4 * source_params.chi_a * source_params.eta_pow2
                            + 16.3021
                            / 1.6128
                            * source_params.chi_s
                            * source_params.delta
                            - 58.745
                            / 4.032
                            * source_params.chi_s
                            * source_params.delta
                            * source_params.eta
                            - 6.7
                            / 2.4
                            * source_params.chi_s
                            * source_params.delta
                            * source_params.eta_pow2
                        ),
                    ]
                )
            )
            * useful_powers_2pi_over_3.two
        )


@sub_struct_from(PostNewtonianCoefficientsHighModesBase)
class PostNewtonianCoefficientsMode32:

    @ti.func
    def update_pn_coefficients(
        self,
        pn_coefficients_22: ti.template(),
        source_params: ti.template(),
    ):
        self._set_rescaling_phase_coefficients(2.0, pn_coefficients_22)

        self.amp_global = tm.sqrt(5.0 / 7.0) / 3.0  # TODO: minus??
        self.A_0 = 0.0
        self.A_1 = 0.0
        self.A_2 = (-1.0 + 3.0 * source_params.eta) * useful_powers_pi.two_thirds
        self.A_3 = -4.0 * source_params.chi_s * source_params.eta * useful_powers_pi.one
        self.A_4 = (
            1.0471 / 1.0080
            - 12.325 / 2.016 * source_params.eta
            + 58.9 / 7.2 * source_params.eta_pow2
        ) * useful_powers_pi.four_thirds
        self.A_5 = (
            tm.length(
                ComplexNumber(
                    [
                        (
                            -11.3 / 2.4 * source_params.chi_a * source_params.delta
                            + 113.0
                            / 8.0
                            * source_params.chi_a
                            * source_params.delta
                            * source_params.eta
                            - 11.3 / 2.4 * source_params.chi_s
                            + 108.1 / 8.4 * source_params.chi_s * source_params.eta
                            - 15.0 * source_params.chi_s * source_params.eta_pow2
                        ),
                        (3.0 - 66.0 / 5.0 * source_params.eta),
                    ]
                )
            )
            * useful_powers_pi.five_thirds
        )
        self.A_6 = (
            8.24173699 / 4.47068160
            - 8.689883 / 149.022720 * source_params.eta
            - 78.584047 / 2.661120 * source_params.eta_pow2
            + 83.7223 / 6.3360 * source_params.eta_pow3
            + 8.0 * source_params.chi_s * source_params.eta * PI
            + 8.1 / 3.2 * source_params.chi_a_pow2
            + 8.1 / 3.2 * source_params.chi_s_pow2
            - 56.3 / 3.2 * source_params.chi_a_pow2 * source_params.eta
            + 30.0 * source_params.chi_a_pow2 * source_params.eta_pow2
            - 254.9 / 9.6 * source_params.chi_s_pow2 * source_params.eta
            + 31.3 / 2.4 * source_params.chi_s_pow2 * source_params.eta_pow2
            + 8.1
            / 1.6
            * source_params.chi_a
            * source_params.chi_s
            * source_params.delta
            - 163.3
            / 4.8
            * source_params.chi_a
            * source_params.chi_s
            * source_params.delta
            * source_params.eta
        ) * useful_powers_pi.two


@sub_struct_from(PostNewtonianCoefficientsHighModesBase)
class PostNewtonianCoefficientsMode44:

    @ti.func
    def update_pn_coefficients(
        self,
        pn_coefficients_22: ti.template(),
        source_params: ti.template(),
    ):
        self._set_rescaling_phase_coefficients(4.0, pn_coefficients_22)

        self.amp_global = 4.0 / 9.0 * tm.sqrt(10.0 / 7.0)  # TODO: minus??
        self.A_0 = 0.0
        self.A_1 = 0.0
        self.A_2 = (1.0 - 3.0 * source_params.eta) * useful_powers_pi_over_2.two_thirds
        self.A_3 = 0.0
        self.A_4 = (
            -15.8383 / 3.6960
            + 128.221 / 7.392 * source_params.eta
            - 106.3 / 8.8 * source_params.eta_pow2
        ) * useful_powers_pi_over_2.four_thirds
        self.A_5 = (
            tm.length(
                ComplexNumber(
                    [
                        (
                            2.0 * PI
                            - 6.0 * source_params.eta * PI
                            + 11.3 / 2.4 * source_params.chi_a * source_params.delta
                            - 113.0
                            / 8.0
                            * source_params.chi_a
                            * source_params.delta
                            * source_params.eta
                            + 11.3 / 2.4 * source_params.chi_s
                            - 41.5 / 2.4 * source_params.chi_s * source_params.eta
                            + 19.0 / 2.0 * source_params.chi_s * source_params.eta_pow2
                        ),
                        (
                            -42.0 / 5.0
                            + 8.0 * tm.log(2.0)
                            + (119.3 / 4.0 - 24.0 * tm.log(2.0)) * source_params.eta
                        ),
                    ]
                )
            )
            * useful_powers_pi_over_2.five_thirds
        )
        self.A_6 = (
            0.7888301437 / 2.9059430400
            - 225.80029007 / 8.80588800 * source_params.eta
            + 90.1461137 / 1.1531520 * source_params.eta_pow2
            - 76.06537 / 2.74560 * source_params.eta_pow3
            - 8.1 / 3.2 * source_params.chi_a_pow2
            - 8.1 / 3.2 * source_params.chi_s_pow2
            + 56.3 / 3.2 * source_params.chi_a_pow2 * source_params.eta
            - 30.0 * source_params.chi_a_pow2 * source_params.eta_pow2
            + 24.7 / 3.2 * source_params.chi_s_pow2 * source_params.eta
            - 3.0 / 8.0 * source_params.chi_s_pow2 * source_params.eta_pow2
            - 8.1
            / 1.6
            * source_params.chi_a
            * source_params.chi_s
            * source_params.delta
            + 24.3
            / 1.6
            * source_params.chi_a
            * source_params.chi_s
            * source_params.delta
            * source_params.eta
        ) * useful_powers_pi_over_2.two


# @ti.dataclass
# class AmplitudeCoefficientsMode21:
#     # Inspiral
#     rho_1: ti.f64
#     rho_2: ti.f64
#     rho_3: ti.f64
#     ins_f_end: ti.f64
#     ins_colloc_points: ti.types.vector(3, ti.f64)
#     ins_colloc_values: ti.types.vector(3, ti.f64)
#     useful_powers: ti.types.struct(
#         ins_f1=UsefulPowers, ins_f2=UsefulPowers, ins_f3=UsefulPowers
#     )
#     # Intermediate
#     # (note the implementaion of PhenomXHMReleaseVersion 122022 has many difference with that shown in the paper)
#     int_f_end: ti.f64
#     int_colloc_points: ti.types.vector(6, ti.f64)
#     int_colloc_values: ti.types.vector(8, ti.f64)
#     int_Ab: ti.types.matrix(8, 9, dtype=ti.f64)
#     int_ansatz_coefficients: ti.types.vector(8, ti.f64)
#     # Merge-ringdown
#     MRD_f_falloff: ti.f64
#     MRD_colloc_points: ti.types.vector(3, ti.f64)
#     MRD_colloc_values: ti.types.vector(3, ti.f64)
#     MRD_ansatz_coefficients: ti.types.vector(5, ti.f64)

#     @ti.func
#     def _set_ins_colloc_points(
#         self, source_params: ti.template()
#     ):
#         # Inspiral collocation points
#         if source_params.eta < 0.023795359904818562:  # for extreme mass ratios (q>40)
#             self.ins_f_end = source_params_HM.f_amp_ins_end_emr
#         else:
#             # note we use f_MECO_22 and f_ISCO_22 here, remember multiply with m/2
#             self.ins_f_end = (
#                 source_params.f_MECO
#                 + (
#                     0.75
#                     - 0.235 * source_params.chi_eff
#                     - 5.0 / 6.0 * source_params.chi_eff_pow2
#                 )
#                 * ti.abs(source_params.f_ISCO - source_params.f_MECO)
#                 * 0.5
#             )
#         self.ins_colloc_points[0] = 0.5 * self.ins_f_end
#         self.ins_colloc_points[1] = 0.75 * self.ins_f_end
#         self.ins_colloc_points[2] = self.ins_f_end

#         self.useful_powers.ins_f1.update(self.ins_colloc_points[0])
#         self.useful_powers.ins_f2.update(self.ins_colloc_points[1])
#         self.useful_powers.ins_f3.update(self.ins_colloc_points[2])

#         # Intermediate collocation points
#         self.int_f_end = 0.75 * source_params_HM.f_ring_21
#         f_space_int = (self.int_f_end - self.ins_f_end) / 5.0
#         self.int_colloc_points = [
#             self.ins_f_end,
#             self.ins_f_end + f_space_int,
#             self.ins_f_end + 2.0 * f_space_int,
#             self.ins_f_end + 3.0 * f_space_int,
#             self.ins_f_end + 4.0 * f_space_int,
#             self.int_f_end,
#         ]
#         # Merge-ringdown
#         self.MRD_f_falloff = source_params_HM.f_ring_21 + 2 * source_params_HM.f_damp_21
#         self.MRD_colloc_points[0] = (
#             source_params_HM.f_ring_21 - source_params_HM.f_damp_21
#         )
#         self.MRD_colloc_points[1] = source_params_HM.f_ring_21
#         self.MRD_colloc_points[2] = (
#             source_params_HM.f_ring_21 + source_params_HM.f_damp_21
#         )

#     @ti.func
#     def _set_inspiral_coefficients(
#         self,
#         pn_coefficients_21: ti.template(),
#         source_params: ti.template(),
#     ):
#         # Note the PhenomXHMReleaseVersion 122022 dose not use votes
#         self.ins_colloc_values[0] = (
#             source_params.delta
#             * (
#                 0.037868557189995156
#                 + 0.10740090317702103 * source_params.eta
#                 + 1.963812986867654 * source_params.eta_pow2
#                 - 16.706455229589558 * source_params.eta_pow3
#                 + 69.75910808095745 * source_params.eta_pow4
#                 - 98.3062466823662 * source_params.eta_pow5
#             )
#             + source_params.delta
#             * source_params.chi_PN_hat
#             * (
#                 -0.007963757232702219
#                 + 0.10627108779259965 * source_params.eta
#                 - 0.008044970210401218 * source_params.chi_PN_hat
#                 + source_params.eta_pow2
#                 * (
#                     -0.4735861262934258
#                     - 0.5985436493302649 * source_params.chi_PN_hat
#                     - 0.08217216660522082 * source_params.chi_PN_hat_pow2
#                 )
#             )
#             - 0.257787704938017
#             * source_params.delta_chi
#             * source_params.eta_pow2
#             * (1.0 + 8.75928187268504 * source_params.eta_pow2)
#             - 0.2597503605427412
#             * source_params.delta_chi
#             * source_params.eta_pow2
#             * source_params.chi_PN_hat
#         )
#         self.ins_colloc_values[1] = (
#             source_params.delta
#             * (
#                 0.05511628628738656
#                 - 0.12579599745414977 * source_params.eta
#                 + 2.831411618302815 * source_params.eta_pow2
#                 - 14.27268643447161 * source_params.eta_pow3
#                 + 28.3307320191161 * source_params.eta_pow4
#             )
#             + source_params.delta
#             * source_params.chi_PN_hat
#             * (
#                 -0.008692738851491525
#                 + source_params.eta
#                 * (0.09512553997347649 + 0.116470975986383 * source_params.chi_PN_hat)
#                 - 0.009520793625590234 * source_params.chi_PN_hat
#                 + source_params.eta_pow2
#                 * (
#                     -0.3409769288480959
#                     - 0.8321002363767336 * source_params.chi_PN_hat
#                     - 0.13099477081654226 * source_params.chi_PN_hat_pow2
#                 )
#                 - 0.006383232900211555 * source_params.chi_PN_hat_pow2
#             )
#             - 0.2962753588645467
#             * source_params.delta_chi
#             * source_params.eta_pow2
#             * (1.0 + 1.3993978458830476 * source_params.eta_pow2)
#             - 0.17100612756133535
#             * source_params.delta_chi
#             * source_params.eta_pow2
#             * source_params.chi_PN_hat
#             * (1.0 + 18.974303741922743 * source_params.eta_pow2 * source_params.delta)
#         )
#         self.ins_colloc_values[2] = (
#             source_params.delta
#             * (
#                 0.059110044024271766
#                 - 0.0024538774422098405 * source_params.eta
#                 + 0.2428578654261086 * source_params.eta_pow2
#             )
#             + source_params.delta
#             * source_params.chi_PN_hat
#             * (
#                 -0.007044339356171243
#                 - 0.006952154764487417 * source_params.chi_PN_hat
#                 + source_params.eta_pow2
#                 * (
#                     -0.016643018304732624
#                     - 0.12702579620537421 * source_params.chi_PN_hat
#                     + 0.004623467175906347 * source_params.chi_PN_hat_pow2
#                 )
#                 - 0.007685497720848461 * source_params.chi_PN_hat_pow2
#             )
#             - 0.3172310538516028
#             * source_params.delta_chi
#             * (1.0 - 2.9155919835488024 * source_params.eta_pow2)
#             * source_params.eta_pow2
#             - 0.11975485688200693
#             * source_params.delta_chi
#             * source_params.eta_pow2
#             * source_params.chi_PN_hat
#             * (1.0 + 17.27626751837825 * source_params.eta_pow2 * source_params.delta)
#         )
#         # Note here we get the pseudeo-PN coefficients with the denominator of f_lm^Ins powers
#         v1 = self.ins_colloc_values[
#             0
#         ] * self.useful_powers.ins_f1.seven_sixths / source_params.amp_common_factor - pn_coefficients_21.amplitude_inspiral_PN_ansatz(
#             self.useful_powers.ins_f1
#         )
#         v2 = self.ins_colloc_values[
#             1
#         ] * self.useful_powers.ins_f2.seven_sixths / source_params.amp_common_factor - pn_coefficients_21.amplitude_inspiral_PN_ansatz(
#             self.useful_powers.ins_f2
#         )
#         v3 = self.ins_colloc_values[
#             2
#         ] * self.useful_powers.ins_f3.seven_sixths / source_params.amp_common_factor - pn_coefficients_21.amplitude_inspiral_PN_ansatz(
#             self.useful_powers.ins_f3
#         )
#         Ab_ins = ti.Matrix(
#             [
#                 [
#                     self.useful_powers.ins_f1.seven_thirds,
#                     self.useful_powers.ins_f1.eight_thirds,
#                     self.useful_powers.ins_f1.three,
#                     v1,
#                 ],
#                 [
#                     self.useful_powers.ins_f2.seven_thirds,
#                     self.useful_powers.ins_f2.eight_thirds,
#                     self.useful_powers.ins_f2.three,
#                     v2,
#                 ],
#                 [
#                     self.useful_powers.ins_f3.seven_thirds,
#                     self.useful_powers.ins_f3.eight_thirds,
#                     self.useful_powers.ins_f3.three,
#                     v3,
#                 ],
#             ],
#             dt=ti.f64,
#         )
#         self.rho_1, self.rho_2, self.rho_3 = gauss_elimination(Ab_ins)

#     @ti.func
#     def _set_merge_ringdown_coefficients(
#         self, source_params: ti.template()
#     ):
#         MRD_v1 = ti.abs(
#             source_params.delta
#             * source_params.eta
#             * (
#                 12.880905080761432
#                 - 23.5291063016996 * source_params.eta
#                 + 92.6090002736012 * source_params.eta_pow2
#                 - 175.16681482428694 * source_params.eta_pow3
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 26.89427230731867 * source_params.eta
#                 - 710.8871223808559 * source_params.eta_pow2
#                 + 2255.040486907459 * source_params.eta_pow3
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 21.402708785047853 * source_params.eta
#                 - 232.07306353130417 * source_params.eta_pow2
#                 + 591.1097623278739 * source_params.eta_pow3
#             )
#             * source_params.chi_PN_hat
#             + source_params.delta
#             * source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 -10.090867481062709
#                 * (
#                     0.9580746052260011
#                     + 5.388149112485179 * source_params.eta
#                     - 107.22993216128548 * source_params.eta_pow2
#                     + 801.3948756800821 * source_params.eta_pow3
#                     - 2688.211889175019 * source_params.eta_pow4
#                     + 3950.7894052628735 * source_params.eta_pow5
#                     - 1992.9074348833092 * source_params.eta_pow6
#                 )
#                 - 0.42972412296628143
#                 * (
#                     1.9193131231064235
#                     + 139.73149069609775 * source_params.eta
#                     - 1616.9974609915555 * source_params.eta_pow2
#                     - 3176.4950303461164 * source_params.eta_pow3
#                     + 107980.65459735804 * source_params.eta_pow4
#                     - 479649.75188253267 * source_params.eta_pow5
#                     + 658866.0983367155 * source_params.eta_pow6
#                 )
#                 * source_params.chi_PN_hat
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.eta_pow5
#             * (
#                 -1512.439342647443
#                 + 175.59081294852444 * source_params.chi_PN_hat
#                 + 10.13490934572329 * source_params.chi_PN_hat_pow2
#             )
#         )
#         MRD_v2 = ti.abs(
#             source_params.delta
#             * (9.112452928978168 - 7.5304766811877455 * source_params.eta)
#             * source_params.eta
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 16.236533863306132 * source_params.eta
#                 - 500.11964987628926 * source_params.eta_pow2
#                 + 1618.0818430353293 * source_params.eta_pow3
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 2.7866868976718226 * source_params.eta
#                 - 0.4210629980868266 * source_params.eta_pow2
#                 - 20.274691328125606 * source_params.eta_pow3
#             )
#             * source_params.chi_PN_hat
#             + source_params_HM.delta_chi_half
#             * source_params.eta_pow5
#             * (
#                 -1116.4039232324135
#                 + 245.73200219767514 * source_params.chi_PN_hat
#                 + 21.159179960295855 * source_params.chi_PN_hat_pow2
#             )
#             + source_params.delta
#             * source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 -8.236485576091717
#                 * (
#                     0.8917610178208336
#                     + 5.1501231412520285 * source_params.eta
#                     - 87.05136337926156 * source_params.eta_pow2
#                     + 519.0146702141192 * source_params.eta_pow3
#                     - 997.6961311502365 * source_params.eta_pow4
#                 )
#                 + 0.2836840678615208
#                 * (
#                     -0.19281297100324718
#                     - 57.65586769647737 * source_params.eta
#                     + 586.7942442434971 * source_params.eta_pow2
#                     - 1882.2040277496196 * source_params.eta_pow3
#                     + 2330.3534917059906 * source_params.eta_pow4
#                 )
#                 * source_params.chi_PN_hat
#                 + 0.40226131643223145
#                 * (
#                     -3.834742668014861
#                     + 190.42214703482531 * source_params.eta
#                     - 2885.5110686004946 * source_params.eta_pow2
#                     + 16087.433824017446 * source_params.eta_pow3
#                     - 29331.524552164105 * source_params.eta_pow4
#                 )
#                 * source_params.chi_PN_hat_pow2
#             )
#         )
#         MRD_v3 = ti.abs(
#             source_params.delta
#             * (2.920930733198033 - 3.038523690239521 * source_params.eta)
#             * source_params.eta
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 6.3472251472354975 * source_params.eta
#                 - 171.23657247338042 * source_params.eta_pow2
#                 + 544.1978232314333 * source_params.eta_pow3
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 1.9701247529688362 * source_params.eta
#                 - 2.8616711550845575 * source_params.eta_pow2
#                 - 0.7347258030219584 * source_params.eta_pow3
#             )
#             * source_params.chi_PN_hat
#             + source_params_HM.delta_chi_half
#             * source_params.eta_pow5
#             * (
#                 -334.0969956136684
#                 + 92.91301644484749 * source_params.chi_PN_hat
#                 - 5.353399481074393 * source_params.chi_PN_hat_pow2
#             )
#             + source_params.delta
#             * source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 -2.7294297839371824
#                 * (
#                     1.148166706456899
#                     - 4.384077347340523 * source_params.eta
#                     + 36.120093043420326 * source_params.eta_pow2
#                     - 87.26454353763077 * source_params.eta_pow3
#                 )
#                 + 0.23949142867803436
#                 * (
#                     -0.6931516433988293
#                     + 33.33372867559165 * source_params.eta
#                     - 307.3404155231787 * source_params.eta_pow2
#                     + 862.3123076782916 * source_params.eta_pow3
#                 )
#                 * source_params.chi_PN_hat
#                 + 0.1930861073906724
#                 * (
#                     3.7735099269174106
#                     - 19.11543562444476 * source_params.eta
#                     - 78.07256429516346 * source_params.eta_pow2
#                     + 485.67801863289293 * source_params.eta_pow3
#                 )
#                 * source_params.chi_PN_hat_pow2
#             )
#         )

#         if MRD_v3 >= MRD_v2**2 / MRD_v1:
#             MRD_v3 = 0.5 * MRD_v2**2 / MRD_v1
#         if MRD_v3 > MRD_v2:
#             MRD_v3 = 0.5 * MRD_v2
#         if (MRD_v1 < MRD_v2) and (MRD_v3 > MRD_v1):
#             MRD_v3 = MRD_v1
#         self.MRD_colloc_values[0] = MRD_v1
#         self.MRD_colloc_values[1] = MRD_v2
#         self.MRD_colloc_values[2] = MRD_v3

#         deno = tm.sqrt(MRD_v1 / MRD_v3) - MRD_v1 / MRD_v2
#         if deno <= 0.0:
#             deno = 1e-16
#         self.MRD_ansatz_coefficients[0] = (
#             self.MRD_colloc_values[0] * source_params_HM.f_damp_21 / deno
#         )
#         self.MRD_ansatz_coefficients[2] = tm.sqrt(
#             self.MRD_ansatz_coefficients[0]
#             / (self.MRD_colloc_values[1] * source_params_HM.f_damp_21)
#         )
#         self.MRD_ansatz_coefficients[1] = (
#             0.5
#             * self.MRD_ansatz_coefficients[2]
#             * tm.log(self.MRD_colloc_values[0] / self.MRD_colloc_values[2])
#         )
#         if self.MRD_f_falloff > 0.0:
#             self.MRD_f_falloff = 0.0
#             self.MRD_ansatz_coefficients[3] = self.merge_ringdown_ansatz()
#             self.MRD_ansatz_coefficients[4] = self.d_merge_ringdown_ansatz()
#             self.MRD_f_falloff = temp

#         # 33
#         MRD_v1 = ti.abs(
#             source_params.delta
#             * source_params.eta
#             * (
#                 12.439702602599235
#                 - 4.436329538596615 * source_params.eta
#                 + 22.780673360839497 * source_params.eta_pow2
#             )
#             + source_params.delta
#             * source_params.eta
#             * (
#                 source_params_HM.delta_chi_half
#                 * (
#                     -41.04442169938298 * source_params.eta
#                     + 502.9246970179746 * source_params.eta_pow2
#                     - 1524.2981907688634 * source_params.eta_pow3
#                 )
#                 + source_params_HM.delta_chi_half_pow2
#                 * (
#                     32.23960072974939 * source_params.eta
#                     - 365.1526474476759 * source_params.eta_pow2
#                     + 1020.6734178547847 * source_params.eta_pow3
#                 )
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 -52.85961155799673 * source_params.eta
#                 + 577.6347407795782 * source_params.eta_pow2
#                 - 1653.496174539196 * source_params.eta_pow3
#             )
#             * source_params.chi_PN_hat
#             + source_params_HM.delta_chi_half
#             * source_params.eta_pow5
#             * (
#                 257.33227387984863
#                 - 34.5074027042393 * source_params_HM.delta_chi_half_pow2
#                 - 21.836905132600755 * source_params.chi_PN_hat
#                 - 15.81624534976308 * source_params.chi_PN_hat_pow2
#             )
#             + 13.499999999999998
#             * source_params.delta
#             * source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 -0.13654149379906394
#                 * (
#                     2.719687834084113
#                     + 29.023992126142304 * source_params.eta
#                     - 742.1357702210267 * source_params.eta_pow2
#                     + 4142.974510926698 * source_params.eta_pow3
#                     - 6167.08766058184 * source_params.eta_pow4
#                     - 3591.1757995710486 * source_params.eta_pow5
#                 )
#                 - 0.06248535354306988
#                 * (
#                     6.697567446351289
#                     - 78.23231700361792 * source_params.eta
#                     + 444.79350113344543 * source_params.eta_pow2
#                     - 1907.008984765889 * source_params.eta_pow3
#                     + 6601.918552659412 * source_params.eta_pow4
#                     - 10056.98422430965 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat
#             )
#             * pow(-3.9329308614837704 + source_params.chi_PN_hat, -1)
#         )

#         MRD_v2 = ti.abs(
#             source_params.delta
#             * source_params.eta
#             * (8.425057692276933 + 4.543696144846763 * source_params.eta)
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 -32.18860840414171 * source_params.eta
#                 + 412.07321398189293 * source_params.eta_pow2
#                 - 1293.422289802462 * source_params.eta_pow3
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 -17.18006888428382 * source_params.eta
#                 + 190.73514518113845 * source_params.eta_pow2
#                 - 636.4802385540647 * source_params.eta_pow3
#             )
#             * source_params.chi_PN_hat
#             + source_params.delta
#             * source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 0.1206817303851239
#                 * (
#                     8.667503604073314
#                     - 144.08062755162752 * source_params.eta
#                     + 3188.189172446398 * source_params.eta_pow2
#                     - 35378.156133055556 * source_params.eta_pow3
#                     + 163644.2192178668 * source_params.eta_pow4
#                     - 265581.70142471837 * source_params.eta_pow5
#                 )
#                 + 0.08028332044013944
#                 * (
#                     12.632478544060636
#                     - 322.95832000179297 * source_params.eta
#                     + 4777.45310151897 * source_params.eta_pow2
#                     - 35625.58409457366 * source_params.eta_pow3
#                     + 121293.97832549023 * source_params.eta_pow4
#                     - 148782.33687815256 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.eta_pow5
#             * (
#                 159.72371180117415
#                 - 29.10412708633528 * source_params_HM.delta_chi_half_pow2
#                 - 1.873799747678187 * source_params.chi_PN_hat
#                 + 41.321480132899524 * source_params.chi_PN_hat_pow2
#             )
#         )

#         MRD_v3 = ti.abs(
#             source_params.delta
#             * source_params.eta
#             * (2.485784720088995 + 2.321696430921996 * source_params.eta)
#             + source_params.delta
#             * source_params.eta
#             * (
#                 source_params_HM.delta_chi_half
#                 * (
#                     -10.454376404653859 * source_params.eta
#                     + 147.10344302665484 * source_params.eta_pow2
#                     - 496.1564538739011 * source_params.eta_pow3
#                 )
#                 + source_params_HM.delta_chi_half_pow2
#                 * (
#                     -5.9236399792925996 * source_params.eta
#                     + 65.86115501723127 * source_params.eta_pow2
#                     - 197.51205149250532 * source_params.eta_pow3
#                 )
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 -10.27418232676514 * source_params.eta
#                 + 136.5150165348149 * source_params.eta_pow2
#                 - 473.30988537734174 * source_params.eta_pow3
#             )
#             * source_params.chi_PN_hat
#             + source_params_HM.delta_chi_half
#             * source_params.eta_pow5
#             * (
#                 32.07819766300362
#                 - 3.071422453072518 * source_params_HM.delta_chi_half_pow2
#                 + 35.09131921815571 * source_params.chi_PN_hat
#                 + 67.23189816732847 * source_params.chi_PN_hat_pow2
#             )
#             + 13.499999999999998
#             * source_params.delta
#             * source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 0.0011484326782460882
#                 * (
#                     4.1815722950796035
#                     - 172.58816646768219 * source_params.eta
#                     + 5709.239330076732 * source_params.eta_pow2
#                     - 67368.27397765424 * source_params.eta_pow3
#                     + 316864.0589150127 * source_params.eta_pow4
#                     - 517034.11171277676 * source_params.eta_pow5
#                 )
#                 - 0.009496797093329243
#                 * (
#                     0.9233282181397624
#                     - 118.35865186626413 * source_params.eta
#                     + 2628.6024206791726 * source_params.eta_pow2
#                     - 23464.64953722729 * source_params.eta_pow3
#                     + 94309.57566199072 * source_params.eta_pow4
#                     - 140089.40725211444 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat
#             )
#             * pow(
#                 0.09549360183532198
#                 - 0.41099904730526465 * source_params.chi_PN_hat
#                 + source_params.chi_PN_hat_pow2,
#                 -1,
#             )
#         )

#         # 44
#         MRD_v1 = ti.abs(
#             source_params.eta
#             * (
#                 source_params_HM.delta_chi_half
#                 * source_params.delta
#                 * (
#                     -8.51952446214978 * source_params.eta
#                     + 117.76530248141987 * source_params.eta_pow2
#                     - 297.2592736781142 * source_params.eta_pow3
#                 )
#                 + source_params_HM.delta_chi_half_pow2
#                 * (
#                     -0.2750098647982238 * source_params.eta
#                     + 4.456900599347149 * source_params.eta_pow2
#                     - 8.017569928870929 * source_params.eta_pow3
#                 )
#             )
#             + source_params.eta
#             * (
#                 5.635069974807398
#                 - 33.67252878543393 * source_params.eta
#                 + 287.9418482197136 * source_params.eta_pow2
#                 - 3514.3385364216438 * source_params.eta_pow3
#                 + 25108.811524802128 * source_params.eta_pow4
#                 - 98374.18361532023 * source_params.eta_pow5
#                 + 158292.58792484726 * source_params.eta_pow6
#             )
#             + source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 -0.4360849737360132
#                 * (
#                     -0.9543114627170375
#                     - 58.70494649755802 * source_params.eta
#                     + 1729.1839588870455 * source_params.eta_pow2
#                     - 16718.425586396803 * source_params.eta_pow3
#                     + 71236.86532610047 * source_params.eta_pow4
#                     - 111910.71267453219 * source_params.eta_pow5
#                 )
#                 - 0.024861802943501172
#                 * (
#                     -52.25045490410733
#                     + 1585.462602954658 * source_params.eta
#                     - 15866.093368857853 * source_params.eta_pow2
#                     + 35332.328181283 * source_params.eta_pow3
#                     + 168937.32229060197 * source_params.eta_pow4
#                     - 581776.5303770923 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat
#                 + 0.005856387555754387
#                 * (
#                     186.39698091707513
#                     - 9560.410655118145 * source_params.eta
#                     + 156431.3764198244 * source_params.eta_pow2
#                     - 1.0461268207440731e6 * source_params.eta_pow3
#                     + 3.054333578686424e6 * source_params.eta_pow4
#                     - 3.2369858387064277e6 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat_pow2
#             )
#         )
#         MRD_v2 = ti.abs(
#             source_params.eta
#             * (
#                 source_params_HM.delta_chi_half
#                 * source_params.delta
#                 * (
#                     -2.861653255976984 * source_params.eta
#                     + 50.50227103211222 * source_params.eta_pow2
#                     - 123.94152825700999 * source_params.eta_pow3
#                 )
#                 + source_params_HM.delta_chi_half_pow2
#                 * (
#                     2.9415751419018865 * source_params.eta
#                     - 28.79779545444817 * source_params.eta_pow2
#                     + 72.40230240887851 * source_params.eta_pow3
#                 )
#             )
#             + source_params.eta
#             * (
#                 3.2461722686239307
#                 + 25.15310593958783 * source_params.eta
#                 - 792.0167314124681 * source_params.eta_pow2
#                 + 7168.843978909433 * source_params.eta_pow3
#                 - 30595.4993786313 * source_params.eta_pow4
#                 + 49148.57065911245 * source_params.eta_pow5
#             )
#             + source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 -0.23311779185707152
#                 * (
#                     -1.0795711755430002
#                     - 20.12558747513885 * source_params.eta
#                     + 1163.9107546486134 * source_params.eta_pow2
#                     - 14672.23221502075 * source_params.eta_pow3
#                     + 73397.72190288734 * source_params.eta_pow4
#                     - 127148.27131388368 * source_params.eta_pow5
#                 )
#                 + 0.025805905356653
#                 * (
#                     11.929946153728276
#                     + 350.93274421955806 * source_params.eta
#                     - 14580.02701600596 * source_params.eta_pow2
#                     + 174164.91607515427 * source_params.eta_pow3
#                     - 819148.9390278616 * source_params.eta_pow4
#                     + 1.3238624538095295e6 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat
#                 + 0.019740635678180102
#                 * (
#                     -7.046295936301379
#                     + 1535.781942095697 * source_params.eta
#                     - 27212.67022616794 * source_params.eta_pow2
#                     + 201981.0743810629 * source_params.eta_pow3
#                     - 696891.1349708183 * source_params.eta_pow4
#                     + 910729.0219043035 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat_pow2
#             )
#         )
#         MRD_v3 = ti.abs(
#             source_params.eta
#             * (
#                 source_params_HM.delta_chi_half
#                 * source_params.delta
#                 * (
#                     2.4286414692113816 * source_params.eta
#                     - 23.213332913737403 * source_params.eta_pow2
#                     + 66.58241012629095 * source_params.eta_pow3
#                 )
#                 + source_params_HM.delta_chi_half_pow2
#                 * (
#                     3.085167288859442 * source_params.eta
#                     - 31.60440418701438 * source_params.eta_pow2
#                     + 78.49621016381445 * source_params.eta_pow3
#                 )
#             )
#             + source_params.eta
#             * (
#                 0.861883217178703
#                 + 13.695204704208976 * source_params.eta
#                 - 337.70598252897696 * source_params.eta_pow2
#                 + 2932.3415281149432 * source_params.eta_pow3
#                 - 12028.786386004691 * source_params.eta_pow4
#                 + 18536.937955014455 * source_params.eta_pow5
#             )
#             + source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 -0.048465588779596405
#                 * (
#                     -0.34041762314288154
#                     - 81.33156665674845 * source_params.eta
#                     + 1744.329802302927 * source_params.eta_pow2
#                     - 16522.343895064576 * source_params.eta_pow3
#                     + 76620.18243090731 * source_params.eta_pow4
#                     - 133340.93723954144 * source_params.eta_pow5
#                 )
#                 + 0.024804027856323612
#                 * (
#                     -8.666095805675418
#                     + 711.8727878341302 * source_params.eta
#                     - 13644.988225595187 * source_params.eta_pow2
#                     + 112832.04975245205 * source_params.eta_pow3
#                     - 422282.0368440555 * source_params.eta_pow4
#                     + 584744.0406581408 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat
#             )
#         )

#     @ti.func
#     def _set_intermediate_coefficients(
#         self,
#         pn_coefficients_21: ti.template(),
#         source_params: ti.template(),
#     ):
#         """
#         Require inspiral and merge-ringdown coefficients, can only be called after updating
#         inspiral and merge-ringdown coefficients.
#         """
#         self.int_colloc_values[0] = self.amplitude_inspiral_ansatz(
#             pn_coefficients_21, self.useful_powers.ins_f3
#         )
#         self.int_colloc_values[1] = ti.abs(
#             source_params.delta
#             * source_params.eta
#             * (
#                 source_params_HM.delta_chi_half_pow2
#                 * (
#                     5.159755997682368 * source_params.eta
#                     - 30.293198248154948 * source_params.eta_pow2
#                     + 63.70715919820867 * source_params.eta_pow3
#                 )
#                 + source_params_HM.delta_chi_half
#                 * (
#                     8.262642080222694 * source_params.eta
#                     - 415.88826990259116 * source_params.eta_pow2
#                     + 1427.5951158851076 * source_params.eta_pow3
#                 )
#             )
#             + source_params.delta
#             * source_params.eta
#             * (
#                 18.55363583212328
#                 - 66.46950491124205 * source_params.eta
#                 + 447.2214642597892 * source_params.eta_pow2
#                 - 1614.178472020212 * source_params.eta_pow3
#                 + 2199.614895727586 * source_params.eta_pow4
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.eta_pow5
#             * (
#                 -1698.841763891122
#                 - 195.27885562092342 * source_params.chi_PN_hat
#                 - 1.3098861736238572 * source_params.chi_PN_hat_pow2
#             )
#             + source_params.delta
#             * source_params.eta
#             * (
#                 source_params_HM.delta_chi_half
#                 * (
#                     34.17829404207186 * source_params.eta
#                     - 386.34587928670015 * source_params.eta_pow2
#                     + 1022.8553774274128 * source_params.eta_pow3
#                 )
#                 * source_params.chi_PN_hat
#                 + source_params_HM.delta_chi_half
#                 * (
#                     56.76554600963724 * source_params.eta
#                     - 491.4593694689354 * source_params.eta_pow2
#                     + 1016.6019654342113 * source_params.eta_pow3
#                 )
#                 * source_params.chi_PN_hat_pow2
#             )
#             + source_params.delta
#             * source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 -8.276366844994188
#                 * (
#                     1.0677538075697492
#                     - 24.12941323757896 * source_params.eta
#                     + 516.7886322104276 * source_params.eta_pow2
#                     - 4389.799658723288 * source_params.eta_pow3
#                     + 16770.447637953577 * source_params.eta_pow4
#                     - 23896.392706809565 * source_params.eta_pow5
#                 )
#                 - 1.6908277400304084
#                 * (
#                     3.4799140066657928
#                     - 29.00026389706585 * source_params.eta
#                     + 114.8330693231833 * source_params.eta_pow2
#                     - 184.13091281984674 * source_params.eta_pow3
#                     + 592.300353344717 * source_params.eta_pow4
#                     - 2085.0821513466053 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat
#                 - 0.46006975902558517
#                 * (
#                     -2.1663474937625975
#                     + 826.026625945615 * source_params.eta
#                     - 17333.549622759732 * source_params.eta_pow2
#                     + 142904.08962903373 * source_params.eta_pow3
#                     - 528521.6231015554 * source_params.eta_pow4
#                     + 731179.456702448 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat_pow2
#             )
#         )
#         self.int_colloc_values[2] = ti.abs(
#             source_params.delta
#             * source_params.eta
#             * (
#                 13.757856231617446
#                 - 12.783698329428516 * source_params.eta
#                 + 12.048194546899204 * source_params.eta_pow2
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 15.107530092096438 * source_params.eta
#                 - 416.811753638553 * source_params.eta_pow2
#                 + 1333.6181181686939 * source_params.eta_pow3
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.eta_pow5
#             * (
#                 -1549.6199518612063
#                 - 102.34716990474509 * source_params.chi_PN_hat
#                 - 3.3637011939285015 * source_params.chi_PN_hat_pow2
#             )
#             + source_params.delta
#             * source_params.eta
#             * (
#                 source_params_HM.delta_chi_half
#                 * (
#                     36.358142200869295 * source_params.eta
#                     - 384.2123173145321 * source_params.eta_pow2
#                     + 984.6826660818275 * source_params.eta_pow3
#                 )
#                 * source_params.chi_PN_hat
#                 + source_params_HM.delta_chi_half
#                 * (
#                     4.159271594881928 * source_params.eta
#                     + 105.10911749116399 * source_params.eta_pow2
#                     - 639.190132707115 * source_params.eta_pow3
#                 )
#                 * source_params.chi_PN_hat_pow2
#             )
#             + source_params.delta
#             * source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 -8.097876227116853
#                 * (
#                     0.6569459700232806
#                     + 9.861355377849485 * source_params.eta
#                     - 116.88834714736281 * source_params.eta_pow2
#                     + 593.8035334117192 * source_params.eta_pow3
#                     - 1063.0692862578455 * source_params.eta_pow4
#                 )
#                 - 1.0546375154878165
#                 * (
#                     0.745557030602097
#                     + 65.25215540635162 * source_params.eta
#                     - 902.5751736558435 * source_params.eta_pow2
#                     + 4350.442990924205 * source_params.eta_pow3
#                     - 7141.611333893155 * source_params.eta_pow4
#                 )
#                 * source_params.chi_PN_hat
#                 - 0.5006664599166409
#                 * (
#                     10.289020582277626
#                     - 212.00728173197498 * source_params.eta
#                     + 2334.0029399672358 * source_params.eta_pow2
#                     - 11939.621138801092 * source_params.eta_pow3
#                     + 21974.8201355744 * source_params.eta_pow4
#                 )
#                 * source_params.chi_PN_hat_pow2
#             )
#         )
#         self.int_colloc_values[3] = ti.abs(
#             source_params.delta
#             * source_params.eta
#             * (
#                 13.318990196097973
#                 - 21.755549987331054 * source_params.eta
#                 + 76.14884211156267 * source_params.eta_pow2
#                 - 127.62161159798488 * source_params.eta_pow3
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 17.704321326939414 * source_params.eta
#                 - 434.4390350012534 * source_params.eta_pow2
#                 + 1366.2408490833282 * source_params.eta_pow3
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 11.877985158418596 * source_params.eta
#                 - 131.04937626836355 * source_params.eta_pow2
#                 + 343.79587860999874 * source_params.eta_pow3
#             )
#             * source_params.chi_PN_hat
#             + source_params_HM.delta_chi_half
#             * source_params.eta_pow5
#             * (
#                 -1522.8543551416456
#                 - 16.639896279650678 * source_params.chi_PN_hat
#                 + 3.0053086651515843 * source_params.chi_PN_hat_pow2
#             )
#             + source_params.delta
#             * source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 -8.665646058245033
#                 * (
#                     0.7862132291286934
#                     + 8.293609541933655 * source_params.eta
#                     - 111.70764910503321 * source_params.eta_pow2
#                     + 576.7172598056907 * source_params.eta_pow3
#                     - 1001.2370065269745 * source_params.eta_pow4
#                 )
#                 - 0.9459820574514348
#                 * (
#                     1.309016452198605
#                     + 48.94077040282239 * source_params.eta
#                     - 817.7854010574645 * source_params.eta_pow2
#                     + 4331.56002883546 * source_params.eta_pow3
#                     - 7518.309520232795 * source_params.eta_pow4
#                 )
#                 * source_params.chi_PN_hat
#                 - 0.4308267743835775
#                 * (
#                     9.970654092010587
#                     - 302.9708323417439 * source_params.eta
#                     + 3662.099161055873 * source_params.eta_pow2
#                     - 17712.883990278668 * source_params.eta_pow3
#                     + 29480.158198408903 * source_params.eta_pow4
#                 )
#                 * source_params.chi_PN_hat_pow2
#             )
#         )
#         self.int_colloc_values[4] = ti.abs(
#             source_params.delta
#             * source_params.eta
#             * (
#                 13.094382343446163
#                 - 22.831152256559523 * source_params.eta
#                 + 83.20619262213437 * source_params.eta_pow2
#                 - 139.25546924151664 * source_params.eta_pow3
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 20.120192352555357 * source_params.eta
#                 - 458.2592421214168 * source_params.eta_pow2
#                 + 1430.3698681181 * source_params.eta_pow3
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 12.925363020014743 * source_params.eta
#                 - 126.87194512915104 * source_params.eta_pow2
#                 + 280.6003655502327 * source_params.eta_pow3
#             )
#             * source_params.chi_PN_hat
#             + source_params_HM.delta_chi_half
#             * source_params.eta_pow5
#             * (
#                 -1528.956015503355
#                 + 74.44462583487345 * source_params.chi_PN_hat
#                 - 2.2456928156392197 * source_params.chi_PN_hat_pow2
#             )
#             + source_params.delta
#             * source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 -9.499741513411829
#                 * (
#                     0.912120958549489
#                     + 2.400945118514037 * source_params.eta
#                     - 33.651192908287236 * source_params.eta_pow2
#                     + 166.04254881175257 * source_params.eta_pow3
#                     - 248.5050377498615 * source_params.eta_pow4
#                 )
#                 - 0.7850652143322492
#                 * (
#                     1.534131218043425
#                     + 60.81773903539479 * source_params.eta
#                     - 1032.1319480683567 * source_params.eta_pow2
#                     + 5381.481380750608 * source_params.eta_pow3
#                     - 9077.037917192794 * source_params.eta_pow4
#                 )
#                 * source_params.chi_PN_hat
#                 - 0.21540359093306097
#                 * (
#                     9.42805409480658
#                     - 109.06544597367301 * source_params.eta
#                     + 385.8345793110262 * source_params.eta_pow2
#                     + 1889.9613367802453 * source_params.eta_pow3
#                     - 9835.416414460055 * source_params.eta_pow4
#                 )
#                 * source_params.chi_PN_hat_pow2
#             )
#         )
#         self.int_colloc_values[5] = self.amplitude_merge_ringdown_ansatz()
#         self.int_colloc_values[6] = self.d_amplitude_inspiral_ansatz(
#             pn_coefficients_21, self.useful_powers.ins_f3
#         )
#         self.int_colloc_values[7] = self.d_amplitude_merge_ringdown_ansatz()

#         # set the matrix consisting of powers of collocation frequencies
#         # note the intermediate ansatz of release version 122022 is different with that show in the paper
#         row_idx = 0
#         for i in ti.static(range(6)):
#             # set the fit value on the collocation point
#             self.int_Ab[row_idx][-1] = self.int_colloc_values[row_idx]
#             # set the coefficient matrix of frequency powers
#             # (1, fi, fi^2, fi^3, fi^4, fi^5, fi^6, fi^7) * fi^(-7/6)
#             fi = self.int_colloc_points[i]
#             fpower = fi ** (-7.0 / 6.0)
#             for j in ti.static(range(8)):
#                 self.int_Ab[row_idx][j] = fpower
#                 fpower *= fi
#             # next row
#             row_idx += 1
#         # for two derivatives at the boundaries
#         for i in ti.static([0, -1]):
#             # set the derivatives value on the boundaries
#             self.int_Ab[row_idx][-1] = self.int_colloc_values[row_idx]
#             # set the coefficient matrix of powers of frequency powers
#             # ( (-7/6)fi_-1, (-7/6+1), (-7/6+2)fi, (-7/6+3)fi^2, (-7/6+4)fi^3, (-7/6+5)fi^4, (-7/6+6)fi^5, (-7/6+7)fi^6) * fi^(-7/6)
#             fi = self.int_colloc_points[i]
#             fpower = fi ** (-13.0 / 6.0)
#             for j in ti.static(range(8)):
#                 self.int_Ab[row_idx][j] = (-7.0 / 6.0 + j) * fpower
#                 fpower *= fi
#             # next row
#             row_idx += 1

#         self.int_ansatz_coefficients = gauss_elimination(self.int_Ab)

#     @ti.func
#     def amplitude_inspiral_ansatz(
#         self,
#         pn_coefficients_21: ti.template(),
#         powers_of_Mf: ti.template(),
#     ) -> ti.f64:
#         return _amplitude_inspiral_ansatz(self, pn_coefficients_21, powers_of_Mf)

#     @ti.func
#     def d_amplitude_inspiral_ansatz(
#         self,
#         pn_coefficients_21: ti.template(),
#         powers_of_Mf: ti.template(),
#     ) -> ti.f64:
#         return _d_amplitude_inspiral_ansatz(self, pn_coefficients_21, powers_of_Mf)

#     @ti.func
#     def amplitude_intermediate_ansatz(self, powers_of_Mf: ti.template()):
#         return _amplitude_intermediate_ansatz(self, powers_of_Mf)

#     # @ti.func
#     # def amplitude_merge_ringdown_ansatz(
#     #     self,
#     #     pn_coefficients_21: ti.template(),
#     #     powers_of_Mf: ti.template(),
#     # ) -> ti.f64:
#     #     return _amplitude_merge_ringdown_ansatz(self, pn_coefficients_21, powers_of_Mf)

#     # @ti.func
#     # def d_amplitude_merge_ringdown_ansatz(
#     #     self,
#     #     pn_coefficients_21: ti.template(),
#     #     powers_of_Mf: ti.template(),
#     # ) -> ti.f64:
#     #     return _d_amplitude_merge_ringdown_ansatz(self, pn_coefficients_21, powers_of_Mf)

#     @ti.func
#     def update_amplitude_coefficients(
#         self,
#         pn_coefficients_21: ti.template(),
#         source_params: ti.template(),
#     ):
#         self._set_ins_colloc_points(
#             source_params,
#             source_params_HM,
#         )
#         self._set_inspiral_coefficients(
#             source_params,
#             source_params_HM,
#             pn_coefficients_21,
#         )
#         self._set_merge_ringdown_coefficients()


# @ti.dataclass
# class AmplitudeCoefficientsMode33:
#     # Inspiral
#     rho_1: ti.f64
#     rho_2: ti.f64
#     rho_3: ti.f64
#     ins_f_end: ti.f64
#     ins_colloc_points: ti.types.vector(3, ti.f64)
#     ins_colloc_values: ti.types.vector(3, ti.f64)
#     useful_powers: ti.types.struct(
#         ins_f1=UsefulPowers, ins_f2=UsefulPowers, ins_f3=UsefulPowers
#     )
#     # Intermediate
#     # (note the implementaion of PhenomXHMReleaseVersion 122022 has many difference with that shown in the paper)
#     int_f_end: ti.f64
#     int_colloc_points: ti.types.vector(6, ti.f64)
#     int_colloc_values: ti.types.vector(8, ti.f64)
#     int_Ab: ti.types.matrix(8, 9, dtype=ti.f64)
#     int_ansatz_coefficients: ti.types.vector(8, ti.f64)
#     # Merge-ringdown
#     MRD_f_falloff: ti.f64
#     MRD_colloc_points: ti.types.vector(3, ti.f64)
#     MRD_colloc_values: ti.types.vector(3, ti.f64)
#     MRD_ansatz_coefficients: ti.types.vector(5, ti.f64)

#     @ti.func
#     def _set_ins_colloc_points(
#         self, source_params: ti.template()
#     ):
#         if source_params.eta < 0.04535147392290249:  # for extreme mass ratios (q>20)
#             self.ins_f_end = 3.0 * source_params_HM.f_amp_ins_end_emr
#         else:
#             # note we use f_MECO_22 and f_ISCO_22 here, remember multiply with m/2
#             sself.ins_f_end = (
#                 source_params.f_MECO
#                 + (
#                     0.75
#                     - 0.235 * source_params.chi_eff
#                     - 5.0 / 6.0 * source_params.chi_eff
#                 )
#                 * ti.abs(source_params.f_ISCO - source_params.f_MECO)
#                 * 1.5
#             )
#         self.ins_colloc_points[0] = 0.5 * self.ins_f_end
#         self.ins_colloc_points[1] = 0.75 * self.ins_f_end
#         self.ins_colloc_points[2] = self.ins_f_end

#         self.useful_powers.ins_f1.update(self.ins_colloc_points[0])
#         self.useful_powers.ins_f2.update(self.ins_colloc_points[1])
#         self.useful_powers.ins_f3.update(self.ins_colloc_points[2])

#         # Intermediate collocation points
#         self.int_f_end = 0.95 * source_params_HM.f_ring_33
#         f_space_int = (self.int_f_end - self.ins_f_end) / 5.0
#         self.int_colloc_points = [
#             self.ins_f_end,
#             self.ins_f_end + f_space_int,
#             self.ins_f_end + 2.0 * f_space_int,
#             self.ins_f_end + 3.0 * f_space_int,
#             self.ins_f_end + 4.0 * f_space_int,
#             self.int_f_end,
#         ]
#         # Merge-ringdown
#         self.MRD_f_falloff = source_params_HM.f_ring_33 + 2 * source_params_HM.f_damp_33
#         self.MRD_colloc_points[0] = (
#             source_params_HM.f_ring_33 - source_params_HM.f_damp_33
#         )
#         self.MRD_colloc_points[1] = source_params_HM.f_ring_33
#         self.MRD_colloc_points[2] = (
#             source_params_HM.f_ring_33 + source_params_HM.f_damp_33
#         )

#     @ti.func
#     def _set_inspiral_coefficients(
#         self,
#         pn_coefficients_33: ti.template(),
#         source_params: ti.template(),
#     ):
#         self.ins_colloc_values[0] = (
#             (
#                 source_params.delta
#                 * (
#                     -0.056586690934283326
#                     - 0.14374841547279146 * source_params.eta
#                     + 0.5584776628959615 * source_params.eta_pow2
#                 )
#             )
#             / (-0.3996185676368123 + source_params.eta)
#             + source_params.delta
#             * source_params.chi_PN_hat
#             * (
#                 (0.056042044149691175 + 0.12482426029674777 * source_params.chi_PN_hat)
#                 * source_params.chi_PN_hat
#                 + source_params.eta
#                 * (
#                     2.1108074577110343
#                     - 1.7827773156978863 * source_params.chi_PN_hat_pow2
#                 )
#                 + source_params.eta_pow2
#                 * (
#                     -7.657635515668849
#                     - 0.07646730296478217 * source_params.chi_PN_hat
#                     + 5.343277927456605 * source_params.chi_PN_hat_pow2
#                 )
#             )
#             + 0.45866449225302536
#             * source_params.delta_chi
#             * (1.0 - 9.603750707244906 * source_params.eta_pow2)
#             * source_params.eta_pow2
#         )
#         self.ins_colloc_values[1] = (
#             source_params.delta
#             * (
#                 0.2137734510411439
#                 - 0.7692194209223682 * source_params.eta
#                 + 26.10570221351058 * source_params.eta_pow2
#                 - 316.0643979123107 * source_params.eta_pow3
#                 + 2090.9063511488234 * source_params.eta_pow4
#                 - 6897.3285171507105 * source_params.eta_pow5
#                 + 8968.893362362503 * source_params.eta_pow6
#             )
#             + source_params.delta
#             * source_params.chi_PN_hat
#             * (
#                 0.018546836505210842
#                 + 0.05924304311104228 * source_params.chi_PN_hat
#                 + source_params.eta
#                 * (
#                     1.6484440612224325
#                     - 0.4683932646001618 * source_params.chi_PN_hat
#                     - 2.110311135456494 * source_params.chi_PN_hat_pow2
#                 )
#                 + 0.10701786057882816 * source_params.chi_PN_hat_pow2
#                 + source_params.eta_pow2
#                 * (
#                     -6.51575737684721
#                     + 1.6692205620001157 * source_params.chi_PN_hat
#                     + 8.351789152096782 * source_params.chi_PN_hat_pow2
#                 )
#             )
#             + 0.3929315188124088
#             * source_params.delta_chi
#             * (1.0 - 11.289452844364227 * source_params.eta_pow2)
#             * source_params.eta_pow2
#         )
#         self.ins_colloc_values[2] = (
#             source_params.delta
#             * (
#                 0.2363760327127446
#                 + 0.2855410252403732 * source_params.eta
#                 - 10.159877125359897 * source_params.eta_pow2
#                 + 162.65372389693505 * source_params.eta_pow3
#                 - 1154.7315106095564 * source_params.eta_pow4
#                 + 3952.61320206691 * source_params.eta_pow5
#                 - 5207.67472857814 * source_params.eta_pow6
#             )
#             + source_params.delta
#             * source_params.chi_PN_hat
#             * (
#                 0.04573095188775319
#                 + 0.048249943132325494 * source_params.chi_PN_hat
#                 + source_params.eta
#                 * (
#                     0.15922377052827502
#                     - 0.1837289613228469 * source_params.chi_PN_hat
#                     - 0.2834348500565196 * source_params.chi_PN_hat_pow2
#                 )
#                 + 0.052963737236081304 * source_params.chi_PN_hat_pow2
#             )
#             + 0.25187274502769835
#             * source_params.delta_chi
#             * (1.0 - 12.172961866410864 * source_params.eta_pow2)
#             * source_params.eta_pow2
#         )
#         # Note here we get the pseudeo-PN coefficients with the denominator of f_lm^Ins powers
#         v1 = self.ins_colloc_values[
#             0
#         ] * self.useful_powers.ins_f1.seven_sixths / source_params.amp_common_factor - pn_coefficients_33.amplitude_inspiral_PN_ansatz(
#             self.useful_powers.ins_f1
#         )
#         v2 = self.ins_colloc_values[
#             1
#         ] * self.useful_powers.ins_f2.seven_sixths / source_params.amp_common_factor - pn_coefficients_33.amplitude_inspiral_PN_ansatz(
#             self.useful_powers.ins_f2
#         )
#         v3 = self.ins_colloc_values[
#             2
#         ] * self.useful_powers.ins_f3.seven_sixths / source_params.amp_common_factor - pn_coefficients_33.amplitude_inspiral_PN_ansatz(
#             self.useful_powers.ins_f3
#         )
#         Ab_ins = ti.Matrix(
#             [
#                 [
#                     self.useful_powers.ins_f1.seven_thirds,
#                     self.useful_powers.ins_f1.eight_thirds,
#                     self.useful_powers.ins_f1.three,
#                     v1,
#                 ],
#                 [
#                     self.useful_powers.ins_f2.seven_thirds,
#                     self.useful_powers.ins_f2.eight_thirds,
#                     self.useful_powers.ins_f2.three,
#                     v2,
#                 ],
#                 [
#                     self.useful_powers.ins_f3.seven_thirds,
#                     self.useful_powers.ins_f3.eight_thirds,
#                     self.useful_powers.ins_f3.three,
#                     v3,
#                 ],
#             ],
#             dt=ti.f64,
#         )
#         self.rho_1, self.rho_2, self.rho_3 = gauss_elimination(Ab_ins)

#     @ti.func
#     def _set_merge_ringdown_coefficients(
#         self, source_params: ti.template()
#     ):
#         MRD_v1 = ti.abs(
#             source_params.delta
#             * source_params.eta
#             * (
#                 12.439702602599235
#                 - 4.436329538596615 * source_params.eta
#                 + 22.780673360839497 * source_params.eta_pow2
#             )
#             + source_params.delta
#             * source_params.eta
#             * (
#                 source_params_HM.delta_chi_half
#                 * (
#                     -41.04442169938298 * source_params.eta
#                     + 502.9246970179746 * source_params.eta_pow2
#                     - 1524.2981907688634 * source_params.eta_pow3
#                 )
#                 + source_params_HM.delta_chi_half_pow2
#                 * (
#                     32.23960072974939 * source_params.eta
#                     - 365.1526474476759 * source_params.eta_pow2
#                     + 1020.6734178547847 * source_params.eta_pow3
#                 )
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 -52.85961155799673 * source_params.eta
#                 + 577.6347407795782 * source_params.eta_pow2
#                 - 1653.496174539196 * source_params.eta_pow3
#             )
#             * source_params.chi_PN_hat
#             + source_params_HM.delta_chi_half
#             * source_params.eta_pow5
#             * (
#                 257.33227387984863
#                 - 34.5074027042393 * source_params_HM.delta_chi_half_pow2
#                 - 21.836905132600755 * source_params.chi_PN_hat
#                 - 15.81624534976308 * source_params.chi_PN_hat_pow2
#             )
#             + 13.499999999999998
#             * source_params.delta
#             * source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 -0.13654149379906394
#                 * (
#                     2.719687834084113
#                     + 29.023992126142304 * source_params.eta
#                     - 742.1357702210267 * source_params.eta_pow2
#                     + 4142.974510926698 * source_params.eta_pow3
#                     - 6167.08766058184 * source_params.eta_pow4
#                     - 3591.1757995710486 * source_params.eta_pow5
#                 )
#                 - 0.06248535354306988
#                 * (
#                     6.697567446351289
#                     - 78.23231700361792 * source_params.eta
#                     + 444.79350113344543 * source_params.eta_pow2
#                     - 1907.008984765889 * source_params.eta_pow3
#                     + 6601.918552659412 * source_params.eta_pow4
#                     - 10056.98422430965 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat
#             )
#             * pow(-3.9329308614837704 + source_params.chi_PN_hat, -1)
#         )
#         MRD_v2 = ti.abs(
#             source_params.delta
#             * source_params.eta
#             * (8.425057692276933 + 4.543696144846763 * source_params.eta)
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 -32.18860840414171 * source_params.eta
#                 + 412.07321398189293 * source_params.eta_pow2
#                 - 1293.422289802462 * source_params.eta_pow3
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 -17.18006888428382 * source_params.eta
#                 + 190.73514518113845 * source_params.eta_pow2
#                 - 636.4802385540647 * source_params.eta_pow3
#             )
#             * source_params.chi_PN_hat
#             + source_params.delta
#             * source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 0.1206817303851239
#                 * (
#                     8.667503604073314
#                     - 144.08062755162752 * source_params.eta
#                     + 3188.189172446398 * source_params.eta_pow2
#                     - 35378.156133055556 * source_params.eta_pow3
#                     + 163644.2192178668 * source_params.eta_pow4
#                     - 265581.70142471837 * source_params.eta_pow5
#                 )
#                 + 0.08028332044013944
#                 * (
#                     12.632478544060636
#                     - 322.95832000179297 * source_params.eta
#                     + 4777.45310151897 * source_params.eta_pow2
#                     - 35625.58409457366 * source_params.eta_pow3
#                     + 121293.97832549023 * source_params.eta_pow4
#                     - 148782.33687815256 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.eta_pow5
#             * (
#                 159.72371180117415
#                 - 29.10412708633528 * source_params_HM.delta_chi_half_pow2
#                 - 1.873799747678187 * source_params.chi_PN_hat
#                 + 41.321480132899524 * source_params.chi_PN_hat_pow2
#             )
#         )
#         MRD_v3 = ti.abs(
#             source_params.delta
#             * source_params.eta
#             * (2.485784720088995 + 2.321696430921996 * source_params.eta)
#             + source_params.delta
#             * source_params.eta
#             * (
#                 source_params_HM.delta_chi_half
#                 * (
#                     -10.454376404653859 * source_params.eta
#                     + 147.10344302665484 * source_params.eta_pow2
#                     - 496.1564538739011 * source_params.eta_pow3
#                 )
#                 + source_params_HM.delta_chi_half_pow2
#                 * (
#                     -5.9236399792925996 * source_params.eta
#                     + 65.86115501723127 * source_params.eta_pow2
#                     - 197.51205149250532 * source_params.eta_pow3
#                 )
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 -10.27418232676514 * source_params.eta
#                 + 136.5150165348149 * source_params.eta_pow2
#                 - 473.30988537734174 * source_params.eta_pow3
#             )
#             * source_params.chi_PN_hat
#             + source_params_HM.delta_chi_half
#             * source_params.eta_pow5
#             * (
#                 32.07819766300362
#                 - 3.071422453072518 * source_params_HM.delta_chi_half_pow2
#                 + 35.09131921815571 * source_params.chi_PN_hat
#                 + 67.23189816732847 * source_params.chi_PN_hat_pow2
#             )
#             + 13.499999999999998
#             * source_params.delta
#             * source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 0.0011484326782460882
#                 * (
#                     4.1815722950796035
#                     - 172.58816646768219 * source_params.eta
#                     + 5709.239330076732 * source_params.eta_pow2
#                     - 67368.27397765424 * source_params.eta_pow3
#                     + 316864.0589150127 * source_params.eta_pow4
#                     - 517034.11171277676 * source_params.eta_pow5
#                 )
#                 - 0.009496797093329243
#                 * (
#                     0.9233282181397624
#                     - 118.35865186626413 * source_params.eta
#                     + 2628.6024206791726 * source_params.eta_pow2
#                     - 23464.64953722729 * source_params.eta_pow3
#                     + 94309.57566199072 * source_params.eta_pow4
#                     - 140089.40725211444 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat
#             )
#             * pow(
#                 0.09549360183532198
#                 - 0.41099904730526465 * source_params.chi_PN_hat
#                 + source_params.chi_PN_hat_pow2,
#                 -1,
#             )
#         )

#         if MRD_v3 >= MRD_v2**2 / MRD_v1:
#             MRD_v3 = 0.5 * MRD_v2**2 / MRD_v1
#         if MRD_v3 > MRD_v2:
#             MRD_v3 = 0.5 * MRD_v2
#         if (MRD_v1 < MRD_v2) and (MRD_v3 > MRD_v1):
#             MRD_v3 = MRD_v1
#         self.MRD_colloc_values[0] = MRD_v1
#         self.MRD_colloc_values[1] = MRD_v2
#         self.MRD_colloc_values[2] = MRD_v3

#         deno = tm.sqrt(MRD_v1 / MRD_v3) - MRD_v1 / MRD_v2
#         if deno <= 0.0:
#             deno = 1e-16
#         self.MRD_ansatz_coefficients[0] = (
#             self.MRD_colloc_values[0] * source_params_HM.f_damp_21 / deno
#         )
#         self.MRD_ansatz_coefficients[2] = tm.sqrt(
#             self.MRD_ansatz_coefficients[0]
#             / (self.MRD_colloc_values[1] * source_params_HM.f_damp_21)
#         )
#         self.MRD_ansatz_coefficients[1] = (
#             0.5
#             * self.MRD_ansatz_coefficients[2]
#             * tm.log(self.MRD_colloc_values[0] / self.MRD_colloc_values[2])
#         )
#         if self.MRD_f_falloff > 0.0:
#             self.MRD_f_falloff = 0.0
#             self.MRD_ansatz_coefficients[3] = self.merge_ringdown_ansatz()
#             self.MRD_ansatz_coefficients[4] = self.d_merge_ringdown_ansatz()
#             self.MRD_f_falloff = temp

#     @ti.func
#     def _set_intermediate_coefficients(
#         self,
#         pn_coefficients_33: ti.template(),
#         source_params: ti.template(),
#     ):
#         self.int_colloc_values[0] = _amplitude_inspiral_ansatz(
#             pn_coefficients_33, self.useful_powers.ins_f3
#         )
#         self.int_colloc_values[1] = (
#             source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 -0.3516244197696068 * source_params.eta
#                 + 40.425151307421416 * source_params.eta_pow2
#                 - 148.3162618111991 * source_params.eta_pow3
#             )
#             + source_params.delta
#             * source_params.eta
#             * (
#                 26.998512565991778
#                 - 146.29035440932105 * source_params.eta
#                 + 914.5350366065115 * source_params.eta_pow2
#                 - 3047.513201789169 * source_params.eta_pow3
#                 + 3996.417635728702 * source_params.eta_pow4
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 5.575274516197629 * source_params.eta
#                 - 44.592719238427094 * source_params.eta_pow2
#                 + 99.91399033058927 * source_params.eta_pow3
#             )
#             * source_params.chi_PN_hat
#             + source_params.delta
#             * source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 -0.5383304368673182
#                 * (
#                     -7.456619067234563
#                     + 129.36947401891433 * source_params.eta
#                     - 843.7897535238325 * source_params.eta_pow2
#                     + 3507.3655567272644 * source_params.eta_pow3
#                     - 9675.194644814854 * source_params.eta_pow4
#                     + 11959.83533107835 * source_params.eta_pow5
#                 )
#                 - 0.28042799223829407
#                 * (
#                     -6.212827413930676
#                     + 266.69059813274475 * source_params.eta
#                     - 4241.537539226717 * source_params.eta_pow2
#                     + 32634.43965039936 * source_params.eta_pow3
#                     - 119209.70783201039 * source_params.eta_pow4
#                     + 166056.27237509796 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.eta_pow5
#             * (
#                 199.6863414922219
#                 + 53.36849263931051 * source_params.chi_PN_hat
#                 + 7.650565415855383 * source_params.chi_PN_hat_pow2
#             )
#         )
#         self.int_colloc_values[2] = (
#             source_params.delta
#             * source_params.eta
#             * (
#                 17.42562079069636
#                 - 28.970875603981295 * source_params.eta
#                 + 50.726220750178435 * source_params.eta_pow2
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 -7.861956897615623 * source_params.eta
#                 + 93.45476935080045 * source_params.eta_pow2
#                 - 273.1170921735085 * source_params.eta_pow3
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 -0.3265505633310564 * source_params.eta
#                 - 9.861644053348053 * source_params.eta_pow2
#                 + 60.38649425562178 * source_params.eta_pow3
#             )
#             * source_params.chi_PN_hat
#             + source_params_HM.delta_chi_half
#             * source_params.eta_pow5
#             * (
#                 234.13476431269862
#                 + 51.2153901931183 * source_params.chi_PN_hat
#                 - 10.05114600643587 * source_params.chi_PN_hat_pow2
#             )
#             + source_params.delta
#             * source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 0.3104472390387834
#                 * (
#                     6.073591341439855
#                     + 169.85423386969634 * source_params.eta
#                     - 4964.199967099143 * source_params.eta_pow2
#                     + 42566.59565666228 * source_params.eta_pow3
#                     - 154255.3408672655 * source_params.eta_pow4
#                     + 205525.13910847943 * source_params.eta_pow5
#                 )
#                 + 0.2295327944679772
#                 * (
#                     19.236275867648594
#                     - 354.7914372697625 * source_params.eta
#                     + 1876.408148917458 * source_params.eta_pow2
#                     + 2404.4151687877525 * source_params.eta_pow3
#                     - 41567.07396803811 * source_params.eta_pow4
#                     + 79210.33893514868 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat
#                 + 0.30983324991828787
#                 * (
#                     11.302200127272357
#                     - 719.9854052004307 * source_params.eta
#                     + 13278.047199998868 * source_params.eta_pow2
#                     - 104863.50453518033 * source_params.eta_pow3
#                     + 376409.2335857397 * source_params.eta_pow4
#                     - 504089.07690692553 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat_pow2
#             )
#         )
#         self.int_colloc_values[3] = (
#             source_params.delta
#             * source_params.eta
#             * (
#                 14.555522136327964
#                 - 12.799844096694798 * source_params.eta
#                 + 16.79500349318081 * source_params.eta_pow2
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 -16.292654447108134 * source_params.eta
#                 + 190.3516012682791 * source_params.eta_pow2
#                 - 562.0936797781519 * source_params.eta_pow3
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 -7.048898856045782 * source_params.eta
#                 + 49.941617405768135 * source_params.eta_pow2
#                 - 73.62033985436068 * source_params.eta_pow3
#             )
#             * source_params.chi_PN_hat
#             + source_params_HM.delta_chi_half
#             * source_params.eta_pow5
#             * (
#                 263.5151703818307
#                 + 44.408527093031566 * source_params.chi_PN_hat
#                 + 10.457035444964653 * source_params.chi_PN_hat_pow2
#             )
#             + source_params.delta
#             * source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 0.4590550434774332
#                 * (
#                     3.0594364612798635
#                     + 207.74562213604057 * source_params.eta
#                     - 5545.0086137386525 * source_params.eta_pow2
#                     + 50003.94075934942 * source_params.eta_pow3
#                     - 195187.55422847517 * source_params.eta_pow4
#                     + 282064.174913521 * source_params.eta_pow5
#                 )
#                 + 0.657748992123043
#                 * (
#                     5.57939137343977
#                     - 124.06189543062042 * source_params.eta
#                     + 1276.6209573025596 * source_params.eta_pow2
#                     - 6999.7659193505915 * source_params.eta_pow3
#                     + 19714.675715229736 * source_params.eta_pow4
#                     - 20879.999628681435 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat
#                 + 0.3695850566805098
#                 * (
#                     6.077183107132255
#                     - 498.95526910874986 * source_params.eta
#                     + 10426.348944657859 * source_params.eta_pow2
#                     - 91096.64982858274 * source_params.eta_pow3
#                     + 360950.6686625352 * source_params.eta_pow4
#                     - 534437.8832860565 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat_pow2
#             )
#         )
#         self.int_colloc_values[4] = (
#             source_params.delta
#             * source_params.eta
#             * (
#                 13.312095699772305
#                 - 7.449975618083432 * source_params.eta
#                 + 17.098576301150125 * source_params.eta_pow2
#             )
#             + source_params.delta
#             * source_params.eta
#             * (
#                 source_params_HM.delta_chi_half
#                 * (
#                     -31.171150896110156 * source_params.eta
#                     + 371.1389274783572 * source_params.eta_pow2
#                     - 1103.1917047361735 * source_params.eta_pow3
#                 )
#                 + source_params_HM.delta_chi_half_pow2
#                 * (
#                     32.78644599730888 * source_params.eta
#                     - 395.15713118955387 * source_params.eta_pow2
#                     + 1164.9282236341376 * source_params.eta_pow3
#                 )
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 -46.85669289852532 * source_params.eta
#                 + 522.3965959942979 * source_params.eta_pow2
#                 - 1485.5134187612182 * source_params.eta_pow3
#             )
#             * source_params.chi_PN_hat
#             + source_params_HM.delta_chi_half
#             * source_params.eta_pow5
#             * (
#                 287.90444670305715
#                 - 21.102665129433042 * source_params_HM.delta_chi_half_pow2
#                 + 7.635582066682054 * source_params.chi_PN_hat
#                 - 29.471275170013012 * source_params.chi_PN_hat_pow2
#             )
#             + source_params.delta
#             * source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 0.6893003654021495
#                 * (
#                     3.1014226377197027
#                     - 44.83989278653052 * source_params.eta
#                     + 565.3767256471909 * source_params.eta_pow2
#                     - 4797.429130246123 * source_params.eta_pow3
#                     + 19514.812242035154 * source_params.eta_pow4
#                     - 27679.226582207506 * source_params.eta_pow5
#                 )
#                 + 0.7068016563068026
#                 * (
#                     4.071212304920691
#                     - 118.51094098279343 * source_params.eta
#                     + 1788.1730303291356 * source_params.eta_pow2
#                     - 13485.270489656365 * source_params.eta_pow3
#                     + 48603.96661003743 * source_params.eta_pow4
#                     - 65658.74746265226 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat
#                 + 0.2181399561677432
#                 * (
#                     -1.6754158383043574
#                     + 303.9394443302189 * source_params.eta
#                     - 6857.936471898544 * source_params.eta_pow2
#                     + 59288.71069769708 * source_params.eta_pow3
#                     - 216137.90827404748 * source_params.eta_pow4
#                     + 277256.38289831823 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat_pow2
#             )
#         )
#         self.int_colloc_values[5] = self.amplitude_merge_ringdown_ansatz()
#         self.int_colloc_values[6] = _d_amplitude_inspiral_ansatz(
#             pn_coefficients_33, self.useful_powers.ins_f3
#         )
#         self.int_colloc_values[7] = self.d_amplitude_merge_ringdown_ansatz()

#         # set the matrix consisting of powers of collocation frequencies
#         # note the intermediate ansatz of release version 122022 is different with that show in the paper
#         row_idx = 0
#         for i in ti.static(range(6)):
#             # set the fit value on the collocation point
#             self.int_Ab[row_idx][-1] = self.int_colloc_values[row_idx]
#             # set the coefficient matrix of frequency powers
#             # (1, fi, fi^2, fi^3, fi^4, fi^5, fi^6, fi^7) * fi^(-7/6)
#             fi = self.int_colloc_points[i]
#             fpower = fi ** (-7.0 / 6.0)
#             for j in ti.static(range(8)):
#                 self.int_Ab[row_idx][j] = fpower
#                 fpower *= fi
#             # next row
#             row_idx += 1
#         # for two derivatives at the boundaries
#         for i in ti.static([0, -1]):
#             # set the derivatives value on the boundaries
#             self.int_Ab[row_idx][-1] = self.int_colloc_values[row_idx]
#             # set the coefficient matrix of powers of frequency powers
#             # ( (-7/6)fi_-1, (-7/6+1), (-7/6+2)fi, (-7/6+3)fi^2, (-7/6+4)fi^3, (-7/6+5)fi^4, (-7/6+6)fi^5, (-7/6+7)fi^6) * fi^(-7/6)
#             fi = self.int_colloc_points[i]
#             fpower = fi ** (-13.0 / 6.0)
#             for j in ti.static(range(8)):
#                 self.int_Ab[row_idx][j] = (-7.0 / 6.0 + j) * fpower
#                 fpower *= fi
#             # next row
#             row_idx += 1

#         self.int_ansatz_coefficients = gauss_elimination(self.int_Ab)

#     @ti.func
#     def amplitude_inspiral_ansatz(
#         self,
#         pn_coefficients_21: ti.template(),
#         powers_of_Mf: ti.template(),
#     ) -> ti.f64:
#         return _amplitude_inspiral_ansatz(self, pn_coefficients_21, powers_of_Mf)

#     @ti.func
#     def d_amplitude_inspiral_ansatz(
#         self,
#         pn_coefficients_21: ti.template(),
#         powers_of_Mf: ti.template(),
#     ) -> ti.f64:
#         return _d_amplitude_inspiral_ansatz(self, pn_coefficients_21, powers_of_Mf)

#     @ti.func
#     def amplitude_intermediate_ansatz(self, powers_of_Mf: ti.template()):
#         return _amplitude_intermediate_ansatz(self, powers_of_Mf)

#     # @ti.func
#     # def amplitude_merge_ringdown_ansatz(
#     #     self,
#     #     pn_coefficients_21: ti.template(),
#     #     powers_of_Mf: ti.template(),
#     # ) -> ti.f64:
#     #     return _amplitude_merge_ringdown_ansatz(self, pn_coefficients_21, powers_of_Mf)

#     # @ti.func
#     # def d_amplitude_merge_ringdown_ansatz(
#     #     self,
#     #     pn_coefficients_21: ti.template(),
#     #     powers_of_Mf: ti.template(),
#     # ) -> ti.f64:
#     #     return _d_amplitude_merge_ringdown_ansatz(self, pn_coefficients_21, powers_of_Mf)
#     @ti.func
#     def update_amplitude_coefficients(
#         self,
#         pn_coefficients_21: ti.template(),
#         source_params: ti.template(),
#     ):
#         self._set_ins_colloc_points(
#             source_params,
#             source_params_HM,
#         )
#         self._set_inspiral_coefficients(
#             source_params,
#             source_params_HM,
#             pn_coefficients_21,
#         )
#         self._set_merge_ringdown_coefficients()


# @ti.dataclass
# class AmplitudeCoefficientsMode32:
#     # Inspiral
#     rho_1: ti.f64
#     rho_2: ti.f64
#     rho_3: ti.f64
#     ins_f_end: ti.f64
#     ins_colloc_points: ti.types.vector(3, ti.f64)
#     ins_colloc_values: ti.types.vector(3, ti.f64)
#     useful_powers: ti.types.struct(
#         ins_f1=UsefulPowers, ins_f2=UsefulPowers, ins_f3=UsefulPowers
#     )
#     # Intermediate
#     # (note the implementaion of PhenomXHMReleaseVersion 122022 has many difference with that shown in the paper)
#     int_f_end: ti.f64
#     int_colloc_points: ti.types.vector(6, ti.f64)
#     int_colloc_values: ti.types.vector(8, ti.f64)
#     int_Ab: ti.types.matrix(8, 9, dtype=ti.f64)
#     int_ansatz_coefficients: ti.types.vector(8, ti.f64)
#     # Merge-ringdown

#     @ti.func
#     def _set_ins_colloc_points(
#         self, source_params: ti.template()
#     ):
#         if source_params.eta < 0.04535147392290249:  # for extreme mass ratios (q>20)
#             self.ins_f_end = 2.0 * source_params_HM.f_amp_ins_end_emr
#         else:
#             # note we use f_MECO_22 and f_ISCO_22 here, remember multiply with m/2
#             self.ins_f_end = (
#                 (
#                     source_params.f_MECO
#                     + (0.75 - 0.235 * ti.abs(source_params.chi_eff))
#                     * ti.abs(source_params.f_ISCO - source_params.f_MECO)
#                 )
#                 * source_params_HM.f_ring_32
#                 / source_params.f_ring
#             )
#         self.ins_colloc_points[0] = 0.5 * self.ins_f_end
#         self.ins_colloc_points[1] = 0.75 * self.ins_f_end
#         self.ins_colloc_points[2] = self.ins_f_end

#         self.useful_powers.ins_f1.update(self.ins_colloc_points[0])
#         self.useful_powers.ins_f2.update(self.ins_colloc_points[1])
#         self.useful_powers.ins_f3.update(self.ins_colloc_points[2])

#         # Intermediate collocation points
#         if source_params.eta < 0.0453515:  # for extreme mass ratios (q>20)
#             self.int_f_end = (
#                 source_params_HM.f_ring_32 * tm.exp(2.5)
#                 + source_params_22.f_ring * tm.exp(5.0 * source_params.chi_1)
#             ) / (
#                 tm.exp(2.5) + tm.exp(5.0 * source_params.chi_1)
#             ) - source_params_HM.f_damp_32
#         else:
#             self.int_f_end = source_params_22.f_ring  # for comparable mass ratios
#         if (
#             (source_params.eta > 0.02126654064272212)
#             and (source_params.eta < 0.12244897959183673)
#             and (source_params.chi_1 > 0.95)
#         ):  # for 6 < q < 45
#             self.int_f_end = (
#                 source_params_HM.f_ring_32 - 2.0 * source_params_HM.f_damp_32
#             )
#         f_space_int = (self.int_f_end - self.ins_f_end) / 5.0
#         self.int_colloc_points = [
#             self.ins_f_end,
#             self.ins_f_end + f_space_int,
#             self.ins_f_end + 2.0 * f_space_int,
#             self.ins_f_end + 3.0 * f_space_int,
#             self.ins_f_end + 4.0 * f_space_int,
#             self.int_f_end,
#         ]

#     @ti.func
#     def _set_inspiral_coefficients(
#         self,
#         pn_coefficients_32: ti.template(),
#         source_params: ti.template(),
#     ):
#         self.ins_colloc_values[0] = (
#             tm.sqrt(1.0 - 3.0 * source_params.eta)
#             * (
#                 0.019069933430190773
#                 - 0.19396651989685837 * source_params.eta
#                 + 11.95224600241255 * source_params.eta_pow2
#                 - 158.90113442757382 * source_params.eta_pow3
#                 + 1046.65239329071 * source_params.eta_pow4
#                 - 3476.940285294999 * source_params.eta_pow5
#                 + 4707.249209858949 * source_params.eta_pow6
#             )
#             + tm.sqrt(1.0 - 3.0 * source_params.eta)
#             * source_params.chi_PN_hat
#             * (
#                 0.0046910348789512895
#                 + 0.40231360805609434 * source_params.eta
#                 - 0.0038263656140933152 * source_params.chi_PN_hat
#                 + 0.018963579407636953 * source_params.chi_PN_hat_pow2
#                 + source_params.eta_pow2
#                 * (
#                     -1.955352354930108
#                     + 2.3753413452420133 * source_params.chi_PN_hat
#                     - 0.9085620866763245 * source_params.chi_PN_hat_pow3
#                 )
#                 + 0.02738043801805805 * source_params.chi_PN_hat_pow3
#                 + source_params.eta_pow3
#                 * (
#                     7.977057990568723
#                     - 7.9259853291789515 * source_params.chi_PN_hat
#                     + 0.49784942656123987 * source_params.chi_PN_hat_pow2
#                     + 5.2255665027119145 * source_params.chi_PN_hat_pow3
#                 )
#             )
#             + 0.058560321425018165
#             * source_params.delta_chi_pow2
#             * (1.0 - 19.936477485971217 * source_params.eta_pow2)
#             * source_params.eta_pow2
#             + 1635.4240644598524
#             * source_params.delta_chi
#             * source_params.eta_pow8
#             * source_params.delta
#             + 0.2735219358839411
#             * source_params.delta_chi
#             * source_params.eta_pow2
#             * source_params.chi_PN_hat
#             * source_params.delta
#         )
#         self.ins_colloc_values[1] = (
#             ti.sqrt(1.0 - 3.0 * source_params.eta)
#             * (
#                 0.024621376891809633
#                 - 0.09692699636236377 * source_params.eta
#                 + 2.7200998230836158 * source_params.eta_pow2
#                 - 16.160563094841066 * source_params.eta_pow3
#                 + 32.930430889650836 * source_params.eta_pow4
#             )
#             + tm.sqrt(1.0 - 3.0 * source_params.eta)
#             * source_params.chi_PN_hat
#             * (
#                 0.008522695567479373
#                 - 1.1104639098529456 * source_params.eta_pow2
#                 - 0.00362963820787208 * source_params.chi_PN_hat
#                 + 0.016978054142418417 * source_params.chi_PN_hat_pow2
#                 + source_params.eta
#                 * (
#                     0.24280554040831698
#                     + 0.15878436411950506 * source_params.chi_PN_hat
#                     - 0.1470288177047577 * source_params.chi_PN_hat_pow3
#                 )
#                 + 0.029465887557447824 * source_params.chi_PN_hat_pow3
#                 + source_params.eta_pow3
#                 * (
#                     4.649438233164449
#                     - 0.7550771176087877 * source_params.chi_PN_hat
#                     + 0.3381436950547799 * source_params.chi_PN_hat_pow2
#                     + 2.5663386135613093 * source_params.chi_PN_hat_pow3
#                 )
#             )
#             - 0.007061187955941243
#             * source_params.delta_chi_pow2
#             * (1.0 - 2.024701925508361 * source_params.eta_pow2)
#             * source_params.eta_pow2
#             + 215.06940561269835
#             * source_params.delta_chi
#             * source_params.eta_pow8
#             * source_params.delta
#             + 0.1465612311350642
#             * source_params.delta_chi
#             * source_params.eta_pow2
#             * source_params.chi_PN_hat
#             * source_params.delta
#         )
#         self.ins_colloc_values[2] = (
#             tm.sqrt(1.0 - 3.0 * source_params.eta)
#             * (
#                 -0.006150151041614737
#                 + 0.017454430190035 * source_params.eta
#                 + 0.02620962593739105 * source_params.eta_pow2
#                 - 0.019043090896351363 * source_params.eta_pow3
#             )
#             / (-0.2655505633361449 + source_params.eta)
#             + tm.sqrt(1.0 - 3.0 * source_params.eta)
#             * source_params.chi_PN_hat
#             * (
#                 0.011073381681404716
#                 + 0.00347699923233349 * source_params.chi_PN_hat
#                 + source_params.eta
#                 * source_params.chi_PN_hat
#                 * (
#                     0.05592992411391443
#                     - 0.15666140197050316 * source_params.chi_PN_hat_pow2
#                 )
#                 + 0.012079324401547036 * source_params.chi_PN_hat_pow2
#                 + source_params.eta_pow2
#                 * (
#                     0.5440307361144313
#                     - 0.008730335213434078 * source_params.chi_PN_hat
#                     + 0.04615964369925028 * source_params.chi_PN_hat_pow2
#                     + 0.6703688097531089 * source_params.chi_PN_hat_pow3
#                 )
#                 + 0.016323101357296865 * source_params.chi_PN_hat_pow3
#             )
#             - 0.020140175824954427
#             * source_params.delta_chi_pow2
#             * (1.0 - 12.675522774051249 * source_params.eta_pow2)
#             * source_params.eta_pow2
#             - 417.3604094454253
#             * source_params.delta_chi
#             * source_params.eta8
#             * source_params.delta
#             + 0.10464021067936538
#             * source_params.delta_chi
#             * source_params.eta_pow2
#             * source_params.chi_PN_hat
#             * source_params.delta
#         )
#         # Note here we get the pseudeo-PN coefficients with the denominator of f_lm^Ins powers
#         v1 = self.ins_colloc_values[
#             0
#         ] * self.useful_powers.ins_f1.seven_sixths / source_params.amp_common_factor - pn_coefficients_32.amplitude_inspiral_PN_ansatz(
#             self.useful_powers.ins_f1
#         )
#         v2 = self.ins_colloc_values[
#             1
#         ] * self.useful_powers.ins_f2.seven_sixths / source_params.amp_common_factor - pn_coefficients_32.amplitude_inspiral_PN_ansatz(
#             self.useful_powers.ins_f2
#         )
#         v3 = self.ins_colloc_values[
#             2
#         ] * self.useful_powers.ins_f3.seven_sixths / source_params.amp_common_factor - pn_coefficients_32.amplitude_inspiral_PN_ansatz(
#             self.useful_powers.ins_f3
#         )
#         Ab_ins = ti.Matrix(
#             [
#                 [
#                     self.useful_powers.ins_f1.seven_thirds,
#                     self.useful_powers.ins_f1.eight_thirds,
#                     self.useful_powers.ins_f1.three,
#                     v1,
#                 ],
#                 [
#                     self.useful_powers.ins_f2.seven_thirds,
#                     self.useful_powers.ins_f2.eight_thirds,
#                     self.useful_powers.ins_f2.three,
#                     v2,
#                 ],
#                 [
#                     self.useful_powers.ins_f3.seven_thirds,
#                     self.useful_powers.ins_f3.eight_thirds,
#                     self.useful_powers.ins_f3.three,
#                     v3,
#                 ],
#             ],
#             dt=ti.f64,
#         )
#         self.rho_1, self.rho_2, self.rho_3 = gauss_elimination(Ab_ins)

#     @ti.func
#     def _set_merge_ringdown_coefficients(
#         self,
#     ):
#         pass

#     @ti.func
#     def _set_intermediate_coefficients(
#         self,
#         pn_coefficients_32: ti.template(),
#         source_params: ti.template(),
#     ):
#         self.int_colloc_values[0] = self.amplitude_inspiral_ansatz(
#             pn_coefficients_32, self.useful_powers.ins_f3
#         )
#         self.int_colloc_values[1] = (
#             (
#                 source_params_HM.delta_chi_half_pow2
#                 * (
#                     -0.2341404256829785 * source_params.eta
#                     + 2.606326837996192 * source_params.eta_pow2
#                     - 8.68296921440857 * source_params.eta_pow3
#                 )
#                 + source_params_HM.delta_chi_half
#                 * source_params.delta
#                 * (
#                     0.5454562486736877 * source_params.eta
#                     - 25.19759222940851 * source_params.eta_pow2
#                     + 73.40268975811729 * source_params.eta_pow3
#                 )
#             )
#             * sqroot
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * (
#                 0.4422257616009941 * source_params.eta
#                 - 8.490112284851655 * source_params.eta_pow2
#                 + 32.22238925527844 * source_params.eta_pow3
#             )
#             * source_params.chi_PN_hat
#             * sqroot
#             + source_params.chi_PN_hat
#             * (
#                 0.7067243321652764
#                 * (
#                     0.12885110296881636
#                     + 9.608999847549535 * source_params.eta
#                     - 85.46581740280585 * source_params.eta_pow2
#                     + 325.71940024255775 * source_params.eta_pow3
#                     + 175.4194342269804 * source_params.eta_pow4
#                     - 1929.9084724384807 * source_params.eta_pow5
#                 )
#                 + 0.1540566313813899
#                 * (
#                     -0.3261041495083288
#                     + 45.55785402900492 * source_params.eta
#                     - 827.591235943271 * source_params.eta_pow2
#                     + 7184.647314370326 * source_params.eta_pow3
#                     - 28804.241518798244 * source_params.eta_pow4
#                     + 43309.69769878964 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat
#             )
#             * sqroot
#             + (
#                 480.0434256230109 * source_params.eta
#                 + 25346.341240810478 * source_params.eta_pow2
#                 - 99873.4707358776 * source_params.eta_pow3
#                 + 106683.98302194536 * source_params.eta_pow4
#             )
#             * sqroot
#             * pow(
#                 1
#                 + 1082.6574834474493 * source_params.eta
#                 + 10083.297670051445 * source_params.eta_pow2,
#                 -1,
#             )
#         )
#         self.int_colloc_values[2] = (
#             source_params.eta
#             * (
#                 source_params_HM.delta_chi_half_pow2
#                 * (
#                     -4.175680729484314 * source_params.eta
#                     + 47.54281549129226 * source_params.eta_pow2
#                     - 128.88334273588077 * source_params.eta_pow3
#                 )
#                 + source_params_HM.delta_chi_half
#                 * source_params.delta
#                 * (
#                     -0.18274358639599947 * source_params.eta
#                     - 71.01128541687838 * source_params.eta_pow2
#                     + 208.07105580635888 * source_params.eta_pow3
#                 )
#             )
#             + source_params.eta
#             * (
#                 4.760999387359598
#                 - 38.57900689641654 * source_params.eta
#                 + 456.2188780552874 * source_params.eta_pow2
#                 - 4544.076411013166 * source_params.eta_pow3
#                 + 24956.9592553473 * source_params.eta_pow4
#                 - 69430.10468748478 * source_params.eta_pow5
#                 + 77839.74180254337 * source_params.eta_pow6
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 1.2198776533959694 * source_params.eta
#                 - 26.816651899746475 * source_params.eta_pow2
#                 + 68.72798751937934 * source_params.eta_pow3
#             )
#             * source_params.chi_PN_hat
#             + source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 1.5098291294292217
#                 * (
#                     0.4844667556328104
#                     + 9.848766999273414 * source_params.eta
#                     - 143.66427232396376 * source_params.eta_pow2
#                     + 856.9917885742416 * source_params.eta_pow3
#                     - 1633.3295758142904 * source_params.eta_pow4
#                 )
#                 + 0.32413108737204144
#                 * (
#                     2.835358206961064
#                     - 62.37317183581803 * source_params.eta
#                     + 761.6103793011912 * source_params.eta_pow2
#                     - 3811.5047139343505 * source_params.eta_pow3
#                     + 6660.304740652403 * source_params.eta_pow4
#                 )
#                 * source_params.chi_PN_hat
#             )
#         )
#         self.int_colloc_values[3] = (
#             3.881450518842405 * source_params.eta
#             - 12.580316392558837 * source_params.eta_pow2
#             + 1.7262466525848588 * source_params.eta_pow3
#             + source_params_HM.delta_chi_half_pow2
#             * (
#                 -7.065118823041031 * source_params.eta_pow2
#                 + 77.97950589523865 * source_params.eta_pow3
#                 - 203.65975422378446 * source_params.eta_pow4
#             )
#             - 58.408542930248046 * source_params.eta_pow4
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * (
#                 1.924723094787216 * source_params.eta_pow2
#                 - 90.92716917757797 * source_params.eta_pow3
#                 + 387.00162600306226 * source_params.eta_pow4
#             )
#             + 403.5748987560612 * source_params.eta_pow5
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * (
#                 -0.2566958540737833 * source_params.eta_pow2
#                 + 14.488550203412675 * source_params.eta_pow3
#                 - 26.46699529970884 * source_params.eta_pow4
#             )
#             * source_params.chi_PN_hat
#             + source_params.chi_PN_hat
#             * (
#                 0.3650871458400108
#                 * (
#                     71.57390929624825 * source_params.eta_pow2
#                     - 994.5272351916166 * source_params.eta_pow3
#                     + 6734.058809060536 * source_params.eta_pow4
#                     - 18580.859291282686 * source_params.eta_pow5
#                     + 16001.318492586077 * source_params.eta_pow6
#                 )
#                 + 0.0960146077440495
#                 * (
#                     451.74917589707513 * source_params.eta_pow2
#                     - 9719.470997418284 * source_params.eta_pow3
#                     + 83403.5743434538 * source_params.eta_pow4
#                     - 318877.43061174755 * source_params.eta_pow5
#                     + 451546.88775684836 * source_params.eta_pow6
#                 )
#                 * source_params.chi_PN_hat
#                 - 0.03985156529181297
#                 * (
#                     -304.92981902871617 * source_params.eta_pow2
#                     + 3614.518459296278 * source_params.eta_pow3
#                     - 7859.4784979916085 * source_params.eta_pow4
#                     - 46454.57664737511 * source_params.eta_pow5
#                     + 162398.81483375572 * source_params.eta_pow6
#                 )
#                 * source_params.chi_PN_hat_pow2
#             )
#         )
#         self.int_colloc_values[4] = (
#             source_params.eta
#             * (
#                 source_params_HM.delta_chi_half_pow2
#                 * (
#                     -8.572797326909152 * source_params.eta
#                     + 92.95723645687826 * source_params.eta_pow2
#                     - 236.2438921965621 * source_params.eta_pow3
#                 )
#                 + source_params_HM.delta_chi_half
#                 * source_params.delta
#                 * (
#                     6.674358856924571 * source_params.eta
#                     - 171.4826985994883 * source_params.eta_pow2
#                     + 645.2760206304703 * source_params.eta_pow3
#                 )
#             )
#             + source_params.eta
#             * (
#                 3.921660532875504
#                 - 16.57299637423352 * source_params.eta
#                 + 25.254017911686333 * source_params.eta_pow2
#                 - 143.41033155133266 * source_params.eta_pow3
#                 + 692.926425981414 * source_params.eta_pow4
#             )
#             + source_params_HM.delta_chi_half
#             * source_params.delta
#             * source_params.eta
#             * (
#                 -3.582040878719185 * source_params.eta
#                 + 57.75888914133383 * source_params.eta_pow2
#                 - 144.21651114700492 * source_params.eta_pow3
#             )
#             * source_params.chi_PN_hat
#             + source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 1.242750265695504
#                 * (
#                     -0.522172424518215
#                     + 25.168480118950065 * source_params.eta
#                     - 303.5223688400309 * source_params.eta_pow2
#                     + 1858.1518762309654 * source_params.eta_pow3
#                     - 3797.3561904195085 * source_params.eta_pow4
#                 )
#                 + 0.2927045241764365
#                 * (
#                     0.5056957789079993
#                     - 15.488754837330958 * source_params.eta
#                     + 471.64047356915603 * source_params.eta_pow2
#                     - 3131.5783196211587 * source_params.eta_pow3
#                     + 6097.887891566872 * source_params.eta_pow4
#                 )
#                 * source_params.chi_PN_hat
#             )
#         )
#         self.int_colloc_values[5] = self.amplitude_merge_ringdown_ansatz()
#         self.int_colloc_values[6] = self.d_amplitude_inspiral_ansatz(
#             pn_coefficients_32, self.useful_powers.ins_f3
#         )
#         self.int_colloc_values[7] = self.d_amplitude_merge_ringdown_ansatz()

#         # set the matrix consisting of powers of collocation frequencies
#         # note the intermediate ansatz of release version 122022 is different with that show in the paper
#         row_idx = 0
#         for i in ti.static(range(6)):
#             # set the fit value on the collocation point
#             self.int_Ab[row_idx][-1] = self.int_colloc_values[row_idx]
#             # set the coefficient matrix of frequency powers
#             # (1, fi, fi^2, fi^3, fi^4, fi^5, fi^6, fi^7) * fi^(-7/6)
#             fi = self.int_colloc_points[i]
#             fpower = fi ** (-7.0 / 6.0)
#             for j in ti.static(range(8)):
#                 self.int_Ab[row_idx][j] = fpower
#                 fpower *= fi
#             # next row
#             row_idx += 1
#         # for two derivatives at the boundaries
#         for i in ti.static([0, -1]):
#             # set the derivatives value on the boundaries
#             self.int_Ab[row_idx][-1] = self.int_colloc_values[row_idx]
#             # set the coefficient matrix of powers of frequency powers
#             # ( (-7/6)fi_-1, (-7/6+1), (-7/6+2)fi, (-7/6+3)fi^2, (-7/6+4)fi^3, (-7/6+5)fi^4, (-7/6+6)fi^5, (-7/6+7)fi^6) * fi^(-7/6)
#             fi = self.int_colloc_points[i]
#             fpower = fi ** (-13.0 / 6.0)
#             for j in ti.static(range(8)):
#                 self.int_Ab[row_idx][j] = (-7.0 / 6.0 + j) * fpower
#                 fpower *= fi
#             # next row
#             row_idx += 1

#         self.int_ansatz_coefficients = gauss_elimination(self.int_Ab)

#     @ti.func
#     def amplitude_inspiral_ansatz(
#         self,
#         pn_coefficients_21: ti.template(),
#         powers_of_Mf: ti.template(),
#     ) -> ti.f64:
#         return _amplitude_inspiral_ansatz(self, pn_coefficients_21, powers_of_Mf)

#     @ti.func
#     def d_amplitude_inspiral_ansatz(
#         self,
#         pn_coefficients_21: ti.template(),
#         powers_of_Mf: ti.template(),
#     ) -> ti.f64:
#         return _d_amplitude_inspiral_ansatz(self, pn_coefficients_21, powers_of_Mf)

#     @ti.func
#     def amplitude_intermediate_ansatz(self, powers_of_Mf: ti.template()):
#         return _amplitude_intermediate_ansatz(self, powers_of_Mf)

#     # @ti.func
#     # def amplitude_merge_ringdown_ansatz(
#     #     self,
#     #     pn_coefficients_21: ti.template(),
#     #     powers_of_Mf: ti.template(),
#     # ) -> ti.f64:
#     #     return _amplitude_merge_ringdown_ansatz(self, pn_coefficients_21, powers_of_Mf)

#     # @ti.func
#     # def d_amplitude_merge_ringdown_ansatz(
#     #     self,
#     #     pn_coefficients_21: ti.template(),
#     #     powers_of_Mf: ti.template(),
#     # ) -> ti.f64:
#     #     return _d_amplitude_merge_ringdown_ansatz(self, pn_coefficients_21, powers_of_Mf)

#     @ti.func
#     def update_amplitude_coefficients(
#         self,
#         pn_coefficients_21: ti.template(),
#         source_params: ti.template(),
#     ):
#         self._set_ins_colloc_points(
#             source_params,
#             source_params_HM,
#         )
#         self._set_inspiral_coefficients(
#             source_params,
#             source_params_HM,
#             pn_coefficients_21,
#         )
#         self._set_merge_ringdown_coefficients()


# @ti.dataclass
# class AmplitudeCoefficientsMode44:
#     # Inspiral
#     rho_1: ti.f64
#     rho_2: ti.f64
#     rho_3: ti.f64
#     ins_f_end: ti.f64
#     ins_colloc_points: ti.types.vector(3, ti.f64)
#     ins_colloc_values: ti.types.vector(3, ti.f64)
#     useful_powers: ti.types.struct(
#         ins_f1=UsefulPowers, ins_f2=UsefulPowers, ins_f3=UsefulPowers
#     )
#     # Intermediate
#     # (note the implementaion of PhenomXHMReleaseVersion 122022 has many difference with that shown in the paper)
#     int_f_end: ti.f64
#     int_colloc_points: ti.types.vector(6, ti.f64)
#     int_colloc_values: ti.types.vector(8, ti.f64)
#     int_Ab: ti.types.matrix(8, 9, dtype=ti.f64)
#     int_ansatz_coefficients: ti.types.vector(8, ti.f64)
#     # Merge-ringdown
#     MRD_f_falloff: ti.f64
#     MRD_colloc_points: ti.types.vector(3, ti.f64)
#     MRD_colloc_values: ti.types.vector(3, ti.f64)
#     MRD_ansatz_coefficients: ti.types.vector(5, ti.f64)

#     @ti.func
#     def _set_ins_colloc_points(
#         self, source_params: ti.template()
#     ):
#         if source_params.eta < 0.04535147392290249:  # for extreme mass ratios (q>20)
#             self.ins_f_end = 4.0 * source_params_HM.f_amp_ins_end_emr
#         else:
#             # note we use f_MECO_22 and f_ISCO_22 here, remember multiply with m/2
#             self.ins_f_end = (
#                 source_params.f_MECO
#                 + (0.75 - 0.235 * source_params.chi_eff)
#                 * ti.abs(source_params.f_ISCO - source_params.f_MECO)
#                 * 2.0
#             )
#         self.ins_colloc_points[0] = 0.5 * self.ins_f_end
#         self.ins_colloc_points[1] = 0.75 * self.ins_f_end
#         self.ins_colloc_points[2] = self.ins_f_end

#         self.useful_powers.ins_f1.update(self.ins_colloc_points[0])
#         self.useful_powers.ins_f2.update(self.ins_colloc_points[1])
#         self.useful_powers.ins_f3.update(self.ins_colloc_points[2])
#         # Intermediate collocation points
#         self.int_f_end = 0.9 * source_params_HM.f_ring_44
#         f_space_int = (self.int_f_end - self.ins_f_end) / 5.0
#         self.int_colloc_points = [
#             self.ins_f_end,
#             self.ins_f_end + f_space_int,
#             self.ins_f_end + 2.0 * f_space_int,
#             self.ins_f_end + 3.0 * f_space_int,
#             self.ins_f_end + 4.0 * f_space_int,
#             self.int_f_end,
#         ]
#         # Merge-ringdown
#         self.MRD_f_falloff = source_params_HM.f_ring_21 + 2 * source_params_HM.f_damp_21
#         self.MRD_colloc_points[0] = (
#             source_params_HM.f_ring_21 - source_params_HM.f_damp_21
#         )
#         self.MRD_colloc_points[1] = source_params_HM.f_ring_21
#         self.MRD_colloc_points[2] = (
#             source_params_HM.f_ring_21 + source_params_HM.f_damp_21
#         )

#     @ti.func
#     def _set_inspiral_coefficients(
#         self,
#         pn_coefficients_44: ti.template(),
#         source_params: ti.template(),
#     ):
#         self.ins_colloc_values[0] = (
#             tm.sqrt(1.0 - 3.0 * source_params.eta)
#             * (
#                 0.06190013067931406
#                 + 0.1928897813606222 * source_params.eta
#                 + 1.9024723168424225 * source_params.eta_pow2
#                 - 15.988716302668415 * source_params.eta_pow3
#                 + 35.21461767354364 * source_params.eta_pow4
#             )
#             + tm.sqrt(1.0 - 3.0 * source_params.eta)
#             * source_params.chi_PN_hat
#             * (
#                 0.011454874900772544
#                 + 0.044702230915643903 * source_params.chi_PN_hat
#                 + source_params.eta
#                 * (
#                     0.6600413908621988
#                     + 0.12149520289658673 * source_params.chi_PN_hat
#                     - 0.4482406547006759 * source_params.chi_PN_hat_pow2
#                 )
#                 + 0.07327810908370004 * source_params.chi_PN_hat_pow2
#                 + source_params.eta_pow2
#                 * (
#                     -2.1705970511116486
#                     - 0.6512813450832168 * source_params.chi_PN_hat
#                     + 1.1237234702682313 * source_params.chi_PN_hat_pow2
#                 )
#             )
#             + 0.4766851579723911
#             * source_params.delta_chi
#             * (1.0 - 15.950025762198988 * source_params.eta_pow2)
#             * source_params.eta_pow2
#             + 0.127900699645338
#             * source_params.delta_chi_pow2
#             * (1.0 - 15.79329306044842 * source_params.eta_pow2)
#             * source_params.eta_pow2
#         )
#         self.ins_colloc_values[1] = (
#             0.08406011695496626
#             - 0.1469952725049322 * source_params.eta
#             + 0.2997223283799925 * source_params.eta_pow2
#             - 1.2910560244510723 * source_params.eta_pow3
#             + (
#                 0.023924074703897662
#                 + 0.26110236039648027 * source_params.eta
#                 - 1.1536009170220438 * source_params.eta_pow2
#             )
#             * source_params.chi_PN_hat
#             + (
#                 0.04479727299752669
#                 - 0.1439868858871802 * source_params.eta
#                 + 0.05736387085230215 * source_params.eta_pow2
#             )
#             * source_params.chi_PN_hat_pow2
#             + (
#                 0.06028104440131858
#                 - 0.4759412992529712 * source_params.eta
#                 + 1.1090751649419717 * source_params.eta_pow2
#             )
#             * source_params.chi_PN_hat_pow3
#             + 0.10346324686812074
#             * source_params.delta_chi_pow2
#             * (1.0 - 16.135903382018213 * source_params.eta_pow2)
#             * source_params.eta_pow2
#             + 0.2648241309154185
#             * source_params.delta_chi
#             * source_params.eta_pow2
#             * source_params.delta
#         )
#         self.ins_colloc_values[2] = (
#             0.08212436946985402
#             - 0.025332770704783136 * source_params.eta
#             - 3.2466088293309885 * source_params.eta_pow2
#             + 28.404235115663706 * source_params.eta_pow3
#             - 111.36325359782991 * source_params.eta_pow4
#             + 157.05954559045156 * source_params.eta_pow5
#             + source_params.chi_PN_hat
#             * (
#                 0.03488890057062679
#                 + 0.039491331923244756 * source_params.chi_PN_hat
#                 + source_params.eta
#                 * (
#                     -0.08968833480313292
#                     - 0.12754920943544915 * source_params.chi_PN_hat
#                     - 0.11199012099701576 * source_params.chi_PN_hat_pow2
#                 )
#                 + 0.034468577523793176 * source_params.chi_PN_hat_pow2
#             )
#             + 0.2062291124580944
#             * source_params.delta_chi
#             * source_params.eta_pow2
#             * source_params.delta
#         )

#         # Note here we get the pseudeo-PN coefficients with the denominator of f_lm^Ins powers
#         v1 = self.ins_colloc_values[
#             0
#         ] * self.useful_powers.ins_f1.seven_sixths / source_params.amp_common_factor - pn_coefficients_44.amplitude_inspiral_PN_ansatz(
#             self.useful_powers.ins_f1
#         )
#         v2 = self.ins_colloc_values[
#             1
#         ] * self.useful_powers.ins_f2.seven_sixths / source_params.amp_common_factor - pn_coefficients_44.amplitude_inspiral_PN_ansatz(
#             self.useful_powers.ins_f2
#         )
#         v3 = self.ins_colloc_values[
#             2
#         ] * self.useful_powers.ins_f3.seven_sixths / source_params.amp_common_factor - pn_coefficients_44.amplitude_inspiral_PN_ansatz(
#             self.useful_powers.ins_f3
#         )
#         Ab_ins = ti.Matrix(
#             [
#                 [
#                     self.useful_powers.ins_f1.seven_thirds,
#                     self.useful_powers.ins_f1.eight_thirds,
#                     self.useful_powers.ins_f1.three,
#                     v1,
#                 ],
#                 [
#                     self.useful_powers.ins_f2.seven_thirds,
#                     self.useful_powers.ins_f2.eight_thirds,
#                     self.useful_powers.ins_f2.three,
#                     v2,
#                 ],
#                 [
#                     self.useful_powers.ins_f3.seven_thirds,
#                     self.useful_powers.ins_f3.eight_thirds,
#                     self.useful_powers.ins_f3.three,
#                     v3,
#                 ],
#             ],
#             dt=ti.f64,
#         )
#         self.rho_1, self.rho_2, self.rho_3 = gauss_elimination(Ab_ins)

#     @ti.func
#     def _set_merge_ringdown_coefficients(
#         self, source_params: ti.template()
#     ):
#         MRD_v1 = ti.abs(
#             source_params.eta
#             * (
#                 source_params_HM.delta_chi_half
#                 * source_params.delta
#                 * (
#                     -8.51952446214978 * source_params.eta
#                     + 117.76530248141987 * source_params.eta_pow2
#                     - 297.2592736781142 * source_params.eta_pow3
#                 )
#                 + source_params_HM.delta_chi_half_pow2
#                 * (
#                     -0.2750098647982238 * source_params.eta
#                     + 4.456900599347149 * source_params.eta_pow2
#                     - 8.017569928870929 * source_params.eta_pow3
#                 )
#             )
#             + source_params.eta
#             * (
#                 5.635069974807398
#                 - 33.67252878543393 * source_params.eta
#                 + 287.9418482197136 * source_params.eta_pow2
#                 - 3514.3385364216438 * source_params.eta_pow3
#                 + 25108.811524802128 * source_params.eta_pow4
#                 - 98374.18361532023 * source_params.eta_pow5
#                 + 158292.58792484726 * source_params.eta_pow6
#             )
#             + source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 -0.4360849737360132
#                 * (
#                     -0.9543114627170375
#                     - 58.70494649755802 * source_params.eta
#                     + 1729.1839588870455 * source_params.eta_pow2
#                     - 16718.425586396803 * source_params.eta_pow3
#                     + 71236.86532610047 * source_params.eta_pow4
#                     - 111910.71267453219 * source_params.eta_pow5
#                 )
#                 - 0.024861802943501172
#                 * (
#                     -52.25045490410733
#                     + 1585.462602954658 * source_params.eta
#                     - 15866.093368857853 * source_params.eta_pow2
#                     + 35332.328181283 * source_params.eta_pow3
#                     + 168937.32229060197 * source_params.eta_pow4
#                     - 581776.5303770923 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat
#                 + 0.005856387555754387
#                 * (
#                     186.39698091707513
#                     - 9560.410655118145 * source_params.eta
#                     + 156431.3764198244 * source_params.eta_pow2
#                     - 1.0461268207440731e6 * source_params.eta_pow3
#                     + 3.054333578686424e6 * source_params.eta_pow4
#                     - 3.2369858387064277e6 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat_pow2
#             )
#         )
#         MRD_v2 = ti.abs(
#             source_params.eta
#             * (
#                 source_params_HM.delta_chi_half
#                 * source_params.delta
#                 * (
#                     -2.861653255976984 * source_params.eta
#                     + 50.50227103211222 * source_params.eta_pow2
#                     - 123.94152825700999 * source_params.eta_pow3
#                 )
#                 + source_params_HM.delta_chi_half_pow2
#                 * (
#                     2.9415751419018865 * source_params.eta
#                     - 28.79779545444817 * source_params.eta_pow2
#                     + 72.40230240887851 * source_params.eta_pow3
#                 )
#             )
#             + source_params.eta
#             * (
#                 3.2461722686239307
#                 + 25.15310593958783 * source_params.eta
#                 - 792.0167314124681 * source_params.eta_pow2
#                 + 7168.843978909433 * source_params.eta_pow3
#                 - 30595.4993786313 * source_params.eta_pow4
#                 + 49148.57065911245 * source_params.eta_pow5
#             )
#             + source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 -0.23311779185707152
#                 * (
#                     -1.0795711755430002
#                     - 20.12558747513885 * source_params.eta
#                     + 1163.9107546486134 * source_params.eta_pow2
#                     - 14672.23221502075 * source_params.eta_pow3
#                     + 73397.72190288734 * source_params.eta_pow4
#                     - 127148.27131388368 * source_params.eta_pow5
#                 )
#                 + 0.025805905356653
#                 * (
#                     11.929946153728276
#                     + 350.93274421955806 * source_params.eta
#                     - 14580.02701600596 * source_params.eta_pow2
#                     + 174164.91607515427 * source_params.eta_pow3
#                     - 819148.9390278616 * source_params.eta_pow4
#                     + 1.3238624538095295e6 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat
#                 + 0.019740635678180102
#                 * (
#                     -7.046295936301379
#                     + 1535.781942095697 * source_params.eta
#                     - 27212.67022616794 * source_params.eta_pow2
#                     + 201981.0743810629 * source_params.eta_pow3
#                     - 696891.1349708183 * source_params.eta_pow4
#                     + 910729.0219043035 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat_pow2
#             )
#         )
#         MRD_v3 = ti.abs(
#             source_params.eta
#             * (
#                 source_params_HM.delta_chi_half
#                 * source_params.delta
#                 * (
#                     2.4286414692113816 * source_params.eta
#                     - 23.213332913737403 * source_params.eta_pow2
#                     + 66.58241012629095 * source_params.eta_pow3
#                 )
#                 + source_params_HM.delta_chi_half_pow2
#                 * (
#                     3.085167288859442 * source_params.eta
#                     - 31.60440418701438 * source_params.eta_pow2
#                     + 78.49621016381445 * source_params.eta_pow3
#                 )
#             )
#             + source_params.eta
#             * (
#                 0.861883217178703
#                 + 13.695204704208976 * source_params.eta
#                 - 337.70598252897696 * source_params.eta_pow2
#                 + 2932.3415281149432 * source_params.eta_pow3
#                 - 12028.786386004691 * source_params.eta_pow4
#                 + 18536.937955014455 * source_params.eta_pow5
#             )
#             + source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 -0.048465588779596405
#                 * (
#                     -0.34041762314288154
#                     - 81.33156665674845 * source_params.eta
#                     + 1744.329802302927 * source_params.eta_pow2
#                     - 16522.343895064576 * source_params.eta_pow3
#                     + 76620.18243090731 * source_params.eta_pow4
#                     - 133340.93723954144 * source_params.eta_pow5
#                 )
#                 + 0.024804027856323612
#                 * (
#                     -8.666095805675418
#                     + 711.8727878341302 * source_params.eta
#                     - 13644.988225595187 * source_params.eta_pow2
#                     + 112832.04975245205 * source_params.eta_pow3
#                     - 422282.0368440555 * source_params.eta_pow4
#                     + 584744.0406581408 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat
#             )
#         )

#         if MRD_v3 >= MRD_v2**2 / MRD_v1:
#             MRD_v3 = 0.5 * MRD_v2**2 / MRD_v1
#         if MRD_v3 > MRD_v2:
#             MRD_v3 = 0.5 * MRD_v2
#         if (MRD_v1 < MRD_v2) and (MRD_v3 > MRD_v1):
#             MRD_v3 = MRD_v1
#         self.MRD_colloc_values[0] = MRD_v1
#         self.MRD_colloc_values[1] = MRD_v2
#         self.MRD_colloc_values[2] = MRD_v3

#         deno = tm.sqrt(MRD_v1 / MRD_v3) - MRD_v1 / MRD_v2
#         if deno <= 0.0:
#             deno = 1e-16
#         self.MRD_ansatz_coefficients[0] = (
#             self.MRD_colloc_values[0] * source_params_HM.f_damp_21 / deno
#         )
#         self.MRD_ansatz_coefficients[2] = tm.sqrt(
#             self.MRD_ansatz_coefficients[0]
#             / (self.MRD_colloc_values[1] * source_params_HM.f_damp_21)
#         )
#         self.MRD_ansatz_coefficients[1] = (
#             0.5
#             * self.MRD_ansatz_coefficients[2]
#             * tm.log(self.MRD_colloc_values[0] / self.MRD_colloc_values[2])
#         )
#         if self.MRD_f_falloff > 0.0:
#             self.MRD_f_falloff = 0.0
#             self.MRD_ansatz_coefficients[3] = self.merge_ringdown_ansatz()
#             self.MRD_ansatz_coefficients[4] = self.d_merge_ringdown_ansatz()
#             self.MRD_f_falloff = temp

#     @ti.func
#     def _set_intermediate_coefficients(
#         self,
#         pn_coefficients_44: ti.template(),
#         source_params: ti.template(),
#     ):
#         self.int_colloc_values[0] = self.amplitude_inspiral_ansatz(
#             pn_coefficients_44, self.useful_powers.ins_f3
#         )
#         self.int_colloc_values[1] = (
#             source_params.eta
#             * (
#                 source_params_HM.delta_chi_half
#                 * source_params.delta
#                 * (
#                     1.5378890240544967 * source_params.eta
#                     - 3.4499418893734903 * source_params.eta_pow2
#                     + 16.879953490422782 * source_params.eta_pow3
#                 )
#                 + source_params_HM.delta_chi_half_pow2
#                 * (
#                     1.720226708214248 * source_params.eta
#                     - 11.87925165364241 * source_params.eta_pow2
#                     + 23.259283336239545 * source_params.eta_pow3
#                 )
#             )
#             + source_params.eta
#             * (
#                 8.790173464969538
#                 - 64.95499142822892 * source_params.eta
#                 + 324.1998823562892 * source_params.eta_pow2
#                 - 1111.9864921907126 * source_params.eta_pow3
#                 + 1575.602443847111 * source_params.eta_pow4
#             )
#             + source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 -0.062333275821238224
#                 * (
#                     -21.630297087123807
#                     + 137.4395894877131 * source_params.eta
#                     + 64.92115530780129 * source_params.eta_pow2
#                     - 1013.1110639471394 * source_params.eta_pow3
#                 )
#                 - 0.11014697070998722
#                 * (
#                     4.149721483857751
#                     - 108.6912882442823 * source_params.eta
#                     + 831.6073263887092 * source_params.eta_pow2
#                     - 1828.2527520190122 * source_params.eta_pow3
#                 )
#                 * source_params.chi_PN_hat
#                 - 0.07704777584463054
#                 * (
#                     4.581767671445529
#                     - 50.35070009227704 * source_params.eta
#                     + 344.9177692251726 * source_params.eta_pow2
#                     - 858.9168637051405 * source_params.eta_pow3
#                 )
#                 * source_params.chi_PN_hat_pow2
#             )
#         )
#         self.int_colloc_values[2] = (
#             source_params.eta
#             * (
#                 source_params_HM.delta_chi_half
#                 * source_params.delta
#                 * (
#                     2.3123974306694057 * source_params.eta
#                     - 12.237594841284904 * source_params.eta_pow2
#                     + 44.78225529547671 * source_params.eta_pow3
#                 )
#                 + source_params_HM.delta_chi_half_pow2
#                 * (
#                     2.9282931698944292 * source_params.eta
#                     - 25.624210264341933 * source_params.eta_pow2
#                     + 61.05270871360041 * source_params.eta_pow3
#                 )
#             )
#             + source_params.eta
#             * (
#                 6.98072197826729
#                 - 46.81443520117986 * source_params.eta
#                 + 236.76146303619544 * source_params.eta_pow2
#                 - 920.358408667518 * source_params.eta_pow3
#                 + 1478.050456337336 * source_params.eta_pow4
#             )
#             + source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 -0.07801583359561987
#                 * (
#                     -28.29972282146242
#                     + 752.1603553640072 * source_params.eta
#                     - 10671.072606753183 * source_params.eta_pow2
#                     + 83447.0461509547 * source_params.eta_pow3
#                     - 350025.2112501252 * source_params.eta_pow4
#                     + 760889.6919776166 * source_params.eta_pow5
#                     - 702172.2934567826 * source_params.eta_pow6
#                 )
#                 + 0.013159545629626014
#                 * (
#                     91.1469833190294
#                     - 3557.5003799977294 * source_params.eta
#                     + 52391.684517955284 * source_params.eta_pow2
#                     - 344254.9973814295 * source_params.eta_pow3
#                     + 1.0141877915334814e6 * source_params.eta_pow4
#                     - 1.1505186449682908e6 * source_params.eta_pow5
#                     + 268756.85659532435 * source_params.eta_pow6
#                 )
#                 * source_params.chi_PN_hat
#             )
#         )
#         self.int_colloc_values[3] = (
#             source_params.eta
#             * (
#                 source_params_HM.delta_chi_half
#                 * source_params.delta
#                 * (
#                     -0.8765502142143329 * source_params.eta
#                     + 22.806632458441996 * source_params.eta_pow2
#                     - 43.675503209991184 * source_params.eta_pow3
#                 )
#                 + source_params_HM.delta_chi_half_pow2
#                 * (
#                     0.48698617426180074 * source_params.eta
#                     - 4.302527065360426 * source_params.eta_pow2
#                     + 16.18571810759235 * source_params.eta_pow3
#                 )
#             )
#             + source_params.eta
#             * (
#                 6.379772583015967
#                 - 44.10631039734796 * source_params.eta
#                 + 269.44092930942793 * source_params.eta_pow2
#                 - 1285.7635006711453 * source_params.eta_pow3
#                 + 2379.538739132234 * source_params.eta_pow4
#             )
#             + source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 -0.23316184683282615
#                 * (
#                     -1.7279023138971559
#                     - 23.606399143993716 * source_params.eta
#                     + 409.3387618483284 * source_params.eta_pow2
#                     - 1115.4147472977265 * source_params.eta_pow3
#                 )
#                 - 0.09653777612560172
#                 * (
#                     -5.310643306559746
#                     - 2.1852511802701264 * source_params.eta
#                     + 541.1248219096527 * source_params.eta_pow2
#                     - 1815.7529908827103 * source_params.eta_pow3
#                 )
#                 * source_params.chi_PN_hat
#                 - 0.060477799540741804
#                 * (
#                     -14.578189130145661
#                     + 175.6116682068523 * source_params.eta
#                     - 569.4799973930861 * source_params.eta_pow2
#                     + 426.0861915646515 * source_params.eta_pow3
#                 )
#                 * source_params.chi_PN_hat_pow2
#             )
#         )
#         self.int_colloc_values[4] = (
#             source_params.eta
#             * (
#                 source_params_HM.delta_chi_half
#                 * source_params.delta
#                 * (
#                     -2.461738962276138 * source_params.eta
#                     + 45.3240543970684 * source_params.eta_pow2
#                     - 112.2714974622516 * source_params.eta_pow3
#                 )
#                 + source_params_HM.delta_chi_half_pow2
#                 * (
#                     0.9158352037567031 * source_params.eta
#                     - 8.724582331021695 * source_params.eta_pow2
#                     + 28.44633544874233 * source_params.eta_pow3
#                 )
#             )
#             + source_params.eta
#             * (
#                 6.098676337298138
#                 - 45.42463610529546 * source_params.eta
#                 + 350.97192927929433 * source_params.eta_pow2
#                 - 2002.2013283876834 * source_params.eta_pow3
#                 + 4067.1685640401033 * source_params.eta_pow4
#             )
#             + source_params.eta
#             * source_params.chi_PN_hat
#             * (
#                 -0.36068516166901304
#                 * (
#                     -2.120354236840677
#                     - 47.56175350408845 * source_params.eta
#                     + 1618.4222330016048 * source_params.eta_pow2
#                     - 14925.514654896673 * source_params.eta_pow3
#                     + 60287.45399959349 * source_params.eta_pow4
#                     - 91269.3745059139 * source_params.eta_pow5
#                 )
#                 - 0.09635801207669747
#                 * (
#                     -11.824692837267394
#                     + 371.7551657959369 * source_params.eta
#                     - 4176.398139238679 * source_params.eta_pow2
#                     + 16655.87939259747 * source_params.eta_pow3
#                     - 4102.218189945819 * source_params.eta_pow4
#                     - 67024.98285179552 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat
#                 - 0.06565232123453196
#                 * (
#                     -26.15227471380236
#                     + 1869.0168486099005 * source_params.eta
#                     - 33951.35186039629 * source_params.eta_pow2
#                     + 253694.6032002248 * source_params.eta_pow3
#                     - 845341.6001856657 * source_params.eta_pow4
#                     + 1.0442282862506858e6 * source_params.eta_pow5
#                 )
#                 * source_params.chi_PN_hat_pow2
#             )
#         )
#         self.int_colloc_values[5] = self.amplitude_merge_ringdown_ansatz()
#         self.int_colloc_values[6] = self.d_amplitude_inspiral_ansatz(
#             pn_coefficients_44, self.useful_powers.ins_f3
#         )
#         self.int_colloc_values[7] = self.d_amplitude_merge_ringdown_ansatz()

#         # set the matrix consisting of powers of collocation frequencies
#         # note the intermediate ansatz of release version 122022 is different with that show in the paper
#         row_idx = 0
#         for i in ti.static(range(6)):
#             # set the fit value on the collocation point
#             self.int_Ab[row_idx][-1] = self.int_colloc_values[row_idx]
#             # set the coefficient matrix of frequency powers
#             # (1, fi, fi^2, fi^3, fi^4, fi^5, fi^6, fi^7) * fi^(-7/6)
#             fi = self.int_colloc_points[i]
#             fpower = fi ** (-7.0 / 6.0)
#             for j in ti.static(range(8)):
#                 self.int_Ab[row_idx][j] = fpower
#                 fpower *= fi
#             # next row
#             row_idx += 1
#         # for two derivatives at the boundaries
#         for i in ti.static([0, -1]):
#             # set the derivatives value on the boundaries
#             self.int_Ab[row_idx][-1] = self.int_colloc_values[row_idx]
#             # set the coefficient matrix of powers of frequency powers
#             # ( (-7/6)fi_-1, (-7/6+1), (-7/6+2)fi, (-7/6+3)fi^2, (-7/6+4)fi^3, (-7/6+5)fi^4, (-7/6+6)fi^5, (-7/6+7)fi^6) * fi^(-7/6)
#             fi = self.int_colloc_points[i]
#             fpower = fi ** (-13.0 / 6.0)
#             for j in ti.static(range(8)):
#                 self.int_Ab[row_idx][j] = (-7.0 / 6.0 + j) * fpower
#                 fpower *= fi
#             # next row
#             row_idx += 1

#         self.int_ansatz_coefficients = gauss_elimination(self.int_Ab)

#     @ti.func
#     def amplitude_inspiral_ansatz(
#         self,
#         pn_coefficients_21: ti.template(),
#         powers_of_Mf: ti.template(),
#     ) -> ti.f64:
#         return _amplitude_inspiral_ansatz(self, pn_coefficients_21, powers_of_Mf)

#     @ti.func
#     def d_amplitude_inspiral_ansatz(
#         self,
#         pn_coefficients_21: ti.template(),
#         powers_of_Mf: ti.template(),
#     ) -> ti.f64:
#         return _d_amplitude_inspiral_ansatz(self, pn_coefficients_21, powers_of_Mf)

#     @ti.func
#     def amplitude_intermediate_ansatz(self, powers_of_Mf: ti.template()):
#         return _amplitude_intermediate_ansatz(self, powers_of_Mf)

#     # @ti.func
#     # def amplitude_merge_ringdown_ansatz(
#     #     self,
#     #     pn_coefficients_21: ti.template(),
#     #     powers_of_Mf: ti.template(),
#     # ) -> ti.f64:
#     #     return _amplitude_merge_ringdown_ansatz(self, pn_coefficients_21, powers_of_Mf)

#     # @ti.func
#     # def d_amplitude_merge_ringdown_ansatz(
#     #     self,
#     #     pn_coefficients_21: ti.template(),
#     #     powers_of_Mf: ti.template(),
#     # ) -> ti.f64:
#     #     return _d_amplitude_merge_ringdown_ansatz(self, pn_coefficients_21, powers_of_Mf)

#     @ti.func
#     def update_amplitude_coefficients(
#         self,
#         pn_coefficients_21: ti.template(),
#         source_params: ti.template(),
#     ):
#         self._set_ins_colloc_points(
#             source_params,
#             source_params_HM,
#         )
#         self._set_inspiral_coefficients(
#             source_params,
#             source_params_HM,
#             pn_coefficients_21,
#         )
#         self._set_merge_ringdown_coefficients()


@sub_struct_from(PhaseCoefficientsHighModesBase)
class PhaseCoefficientsMode21:

    _useful_powers: ti.types.struct(
        int_f0=UsefulPowers,
        int_f1=UsefulPowers,
        int_f2=UsefulPowers,
        ins_f_end=UsefulPowers,
        int_f_end=UsefulPowers,
    )

    @ti.func
    def _set_colloc_points(self, source_params: ti.template()):
        self.ins_f_end = source_params.f_MECO_lm["21"]
        self.int_f_end = (
            source_params.QNM_freqs_lm["21"].f_ring
            - source_params.QNM_freqs_lm["21"].f_damp
        )
        # shifting forward the frequency of the first collocation points for small eta
        beta = 1.0 + 0.001 * (0.25 / source_params.eta - 1.0)

        self.int_colloc_points[0] = beta * self.ins_f_end
        self.int_colloc_points[1] = (
            tm.sqrt(3.0)
            * (self.int_colloc_points[0] - source_params.QNM_freqs_lm["21"].f_ring)
            + 2.0
            * (self.int_colloc_points[0] + source_params.QNM_freqs_lm["21"].f_ring)
        ) / 4.0
        self.int_colloc_points[2] = (
            3.0 * self.int_colloc_points[0] + source_params.QNM_freqs_lm["21"].f_ring
        ) / 4.0
        self.int_colloc_points[3] = (
            self.int_colloc_points[0] + source_params.QNM_freqs_lm["21"].f_ring
        ) / 2.0
        self.int_colloc_points[4] = (
            self.int_colloc_points[0] + 3.0 * source_params.QNM_freqs_lm["21"].f_ring
        ) / 4.0
        self.int_colloc_points[5] = (
            self.int_colloc_points[0] + 7.0 * source_params.QNM_freqs_lm["21"].f_ring
        ) / 8.0

        self._useful_powers.int_f0.update(self.int_colloc_points[0])
        self._useful_powers.int_f1.update(self.int_colloc_points[1])
        self._useful_powers.int_f2.update(self.int_colloc_points[2])
        self._useful_powers.ins_f_end.update(self.ins_f_end)
        self._useful_powers.int_f_end.update(self.int_f_end)

    @ti.func
    def _Lambda_21_PN(self) -> ti.f64:
        return 2.0 * PI * (0.5 + 2.0 * tm.log(2.0))

    @ti.func
    def _Lambda_21_fit(self, source_params: ti.template()) -> ti.f64:
        return (
            13.664473636545068
            - 170.08866400251395 * source_params.eta
            + 3535.657736681598 * source_params.eta_pow2
            - 26847.690494515424 * source_params.eta_pow3
            + 96463.68163125668 * source_params.eta_pow4
            - 133820.89317471132 * source_params.eta_pow5
            + (
                source_params.S_tot_hat
                * (
                    18.52571430563905
                    - 41.55066592130464 * source_params.S_tot_hat
                    + source_params.eta_pow3
                    * (
                        83493.24265292779
                        + 16501.749243703132 * source_params.S_tot_hat
                        - 149700.4915210766 * source_params.S_tot_hat_pow2
                    )
                    + source_params.eta
                    * (
                        3642.5891077598003
                        + 1198.4163078715173 * source_params.S_tot_hat
                        - 6961.484805326852 * source_params.S_tot_hat_pow2
                    )
                    + 33.8697137964237 * source_params.S_tot_hat_pow2
                    + source_params.eta_pow2
                    * (
                        -35031.361998480075
                        - 7233.191207000735 * source_params.S_tot_hat
                        + 62149.00902591944 * source_params.S_tot_hat_pow2
                    )
                )
            )
            / (6.880288191574696 + 1.0 * source_params.S_tot_hat)
            - 134.27742343186577
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
        )

    @ti.func
    def _set_intermediate_coefficients(
        self,
        pn_coefficients_22: ti.template(),
        phase_coefficients_22: ti.template(),
        source_params: ti.template(),
    ):
        """
        Setting intermediate coefficients for mode 21. For modes without significant
        mode-mixing, using 5 out of 6 collocation nodes determined according to spin and mass ratio, and setting c_3 = 0.
        for situation with (eta < etaEMR) or (emm == ell and STotR >= 0.8) or (modeTag == 33 and STotR < 0), using collocation nodes: 0, 1, 3, 4, 5.
        for situation with (STotR >= 0.8) and (modeTag == 21), using collocation nodes: 0, 1, 2, 4, 5.
        for remaining parameter space, using collocation nodes: 0, 1, 2, 3, 5
        """
        self.c_3 = 0.0

        self._update_int_colloc_value_0(source_params)
        self._update_int_colloc_value_1(source_params)
        self._update_int_colloc_value_2(source_params)
        self._update_int_colloc_value_5(source_params)

        # special operation for 21 mode to avoide sharp transitions in high-spin cases
        # TODO: 1/eta ??
        if source_params.S_tot_hat >= 0.8:
            ins_val_0 = phase_coefficients_22.compute_d_phase(
                pn_coefficients_22, source_params, self._useful_powers.int_f0
            )
            ins_val_1 = phase_coefficients_22.compute_d_phase(
                pn_coefficients_22, source_params, self._useful_powers.int_f1
            )
            ins_val_2 = phase_coefficients_22.compute_d_phase(
                pn_coefficients_22, source_params, self._useful_powers.int_f2
            )
            diff_01 = ins_val_0 - ins_val_1
            diff_12 = ins_val_1 - ins_val_2
            self.int_colloc_values[1] = self.int_colloc_values[2] + diff_12
            self.int_colloc_values[0] = self.int_colloc_values[1] + diff_01

        # simplified the conditional structure in LALSimIMRPhenomXHM_internals.c l.2108 for modeTag=21
        if source_params.eta < eta_EMR:  # using collocation nodes: 0, 1, 3, 4, 5
            self._set_intermediate_coefficients_01345(source_params)
        elif source_params.S_tot_hat >= 0.8:
            self._set_intermediate_coefficients_01245(source_params)
        else:
            self._set_intermediate_coefficients_01235(source_params)

    @ti.func
    def _set_intermediate_coefficients_01345(self, source_params: ti.template()):
        self.int_colloc_values[2] = 0.0
        self._update_int_colloc_value_3(source_params)
        self._update_int_colloc_value_4(source_params)

        Ab = self._int_no_mixing_augmented_matrix(
            source_params.QNM_freqs_lm["21"], [0, 1, 3, 4, 5]
        )
        self.c_0, self.c_1, self.c_2, self.c_4, self.c_L = gauss_elimination(Ab)

    @ti.func
    def _set_intermediate_coefficients_01245(self, source_params: ti.template()):
        self.int_colloc_values[3] = 0.0
        self._update_int_colloc_value_4(source_params)

        Ab = self._int_no_mixing_augmented_matrix(
            source_params.QNM_freqs_lm["21"], [0, 1, 2, 4, 5]
        )
        self.c_0, self.c_1, self.c_2, self.c_4, self.c_L = gauss_elimination(Ab)

    @ti.func
    def _set_intermediate_coefficients_01235(self, source_params: ti.template()):
        self._update_int_colloc_value_3(source_params)
        self.int_colloc_values[4] = 0.0

        Ab = self._int_no_mixing_augmented_matrix(
            source_params.QNM_freqs_lm["21"], [0, 1, 2, 3, 5]
        )
        self.c_0, self.c_1, self.c_2, self.c_4, self.c_L = gauss_elimination(Ab)

    @ti.func
    def _int_no_mixing_augmented_matrix(
        self, QNM_freqs_lm: ti.template(), idx: ti.template()
    ) -> ti.types.matrix(5, 6, ti.f64):
        Ab = ti.Matrix([[0.0] * 6 for _ in range(5)])
        for i in ti.static(range(5)):
            row = [
                1.0,
                self.int_colloc_points[idx[i]] ** (-1),
                self.int_colloc_points[idx[i]] ** (-2),
                self.int_colloc_points[idx[i]] ** (-4),
                QNM_freqs_lm.f_damp
                / (
                    QNM_freqs_lm.f_damp_pow2
                    + (self.int_colloc_points[idx[i]] - QNM_freqs_lm.f_ring) ** 2
                ),
                self.int_colloc_values[idx[i]],
            ]
            for j in ti.static(range(6)):
                Ab[i, j] = row[j]
        return Ab

    @ti.func
    def _update_int_colloc_value_0(self, source_params: ti.template()):
        self.int_colloc_values[0] = source_params.dt_psi4_to_strain + (
            4045.84
            + 7.63226 / source_params.eta
            - 1956.93 * source_params.eta
            - 23428.1 * source_params.eta_pow2
            + 369153.0 * source_params.eta_pow3
            - 2.28832e6 * source_params.eta_pow4
            + 6.82533e6 * source_params.eta_pow5
            - 7.86254e6 * source_params.eta_pow6
            - 347.273 * source_params.S_tot_hat
            + 83.5428 * source_params.S_tot_hat_pow2
            - 355.67 * source_params.S_tot_hat_pow3
            + (
                4.44457 * source_params.S_tot_hat
                + 16.5548 * source_params.S_tot_hat_pow2
                + 13.6971 * source_params.S_tot_hat_pow3
            )
            / source_params.eta
            + source_params.eta
            * (
                -79.761 * source_params.S_tot_hat
                - 355.299 * source_params.S_tot_hat_pow2
                + 1114.51 * source_params.S_tot_hat_pow3
                - 1077.75 * source_params.S_tot_hat_pow4
            )
            + 92.6654 * source_params.S_tot_hat_pow4
            + source_params.eta_pow2
            * (
                -619.837 * source_params.S_tot_hat
                - 722.787 * source_params.S_tot_hat_pow2
                + 2392.73 * source_params.S_tot_hat_pow3
                + 2689.18 * source_params.S_tot_hat_pow4
            )
            + (
                918.976 * source_params.chi_1 * source_params.delta
                - 918.976 * source_params.chi_2 * source_params.delta
            )
            * source_params.eta
            + (
                91.7679 * source_params.chi_1 * source_params.delta
                - 91.7679 * source_params.chi_2 * source_params.delta
            )
            * source_params.eta_pow2
        )

    @ti.func
    def _update_int_colloc_value_1(self, source_params: ti.template()):
        self.int_colloc_values[1] = source_params.dt_psi4_to_strain + (
            3509.09
            + 0.91868 / source_params.eta
            + 194.72 * source_params.eta
            - 27556.2 * source_params.eta_pow2
            + 369153.0 * source_params.eta_pow3
            - 2.28832e6 * source_params.eta_pow4
            + 6.82533e6 * source_params.eta_pow5
            - 7.86254e6 * source_params.eta_pow6
            + (
                (
                    0.7083999999999999
                    - 60.1611 * source_params.eta
                    + 131.815 * source_params.eta_pow2
                    - 619.837 * source_params.eta_pow3
                )
                * source_params.S_tot_hat
                + (
                    6.104720000000001
                    - 59.2068 * source_params.eta
                    + 278.588 * source_params.eta_pow2
                    - 722.787 * source_params.eta_pow3
                )
                * source_params.S_tot_hat_pow2
                + (
                    5.7791
                    + 117.913 * source_params.eta
                    - 1180.4 * source_params.eta_pow2
                    + 2392.73 * source_params.eta_pow3
                )
                * source_params.S_tot_hat_pow3
                + source_params.eta
                * (
                    92.6654
                    - 1077.75 * source_params.eta
                    + 2689.18 * source_params.eta_pow2
                )
                * source_params.S_tot_hat_pow4
            )
            / source_params.eta
            - 91.7679
            * source_params.delta
            * source_params.eta
            * (
                source_params.chi_1 * (-1.6012352903357276 - 1.0 * source_params.eta)
                + source_params.chi_2 * (1.6012352903357276 + 1.0 * source_params.eta)
            )
        )

    @ti.func
    def _update_int_colloc_value_2(self, source_params: ti.template()):
        self.int_colloc_values[2] = source_params.dt_psi4_to_strain + (
            3241.68
            + 890.016 * source_params.eta
            - 28651.9 * source_params.eta_pow2
            + 369153.0 * source_params.eta_pow3
            - 2.28832e6 * source_params.eta_pow4
            + 6.82533e6 * source_params.eta_pow5
            - 7.86254e6 * source_params.eta_pow6
            + (-2.2484 + 187.641 * source_params.eta - 619.837 * source_params.eta_pow2)
            * source_params.S_tot_hat
            + (3.22603 + 166.323 * source_params.eta - 722.787 * source_params.eta_pow2)
            * source_params.S_tot_hat_pow2
            + (117.913 - 1094.59 * source_params.eta + 2392.73 * source_params.eta_pow2)
            * source_params.S_tot_hat_pow3
            + (92.6654 - 1077.75 * source_params.eta + 2689.18 * source_params.eta_pow2)
            * source_params.S_tot_hat_pow4
            + 91.7679
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
        )

    @ti.func
    def _update_int_colloc_value_3(self, source_params: ti.template()):
        self.int_colloc_values[3] = source_params.dt_psi4_to_strain + (
            3160.88
            + 974.355 * source_params.eta
            - 28932.5 * source_params.eta_pow2
            + 369780.0 * source_params.eta_pow3
            - 2.28832e6 * source_params.eta_pow4
            + 6.82533e6 * source_params.eta_pow5
            - 7.86254e6 * source_params.eta_pow6
            + (26.3355 - 196.851 * source_params.eta + 438.401 * source_params.eta_pow2)
            * source_params.S_tot_hat
            + (45.9957 - 256.248 * source_params.eta + 117.563 * source_params.eta_pow2)
            * source_params.S_tot_hat_pow2
            + (-20.0261 + 467.057 * source_params.eta - 1613.0 * source_params.eta_pow2)
            * source_params.S_tot_hat_pow3
            + (
                -61.7446
                + 577.057 * source_params.eta
                - 1096.81 * source_params.eta_pow2
            )
            * source_params.S_tot_hat_pow4
            + 65.3326
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
        )

    @ti.func
    def _update_int_colloc_value_4(self, source_params: ti.template()):
        self.int_colloc_values[4] = source_params.dt_psi4_to_strain + (
            3102.36
            + 315.911 * source_params.eta
            - 1688.26 * source_params.eta_pow2
            + 3635.76 * source_params.eta_pow3
            + (-23.0959 + 320.93 * source_params.eta - 1029.76 * source_params.eta_pow2)
            * source_params.S_tot_hat
            + (
                -49.5435
                + 826.816 * source_params.eta
                - 3079.39 * source_params.eta_pow2
            )
            * source_params.S_tot_hat_pow2
            + (40.7054 - 365.842 * source_params.eta + 1094.11 * source_params.eta_pow2)
            * source_params.S_tot_hat_pow3
            + (81.8379 - 1243.26 * source_params.eta + 4689.22 * source_params.eta_pow2)
            * source_params.S_tot_hat_pow4
            + 119.014
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
        )

    @ti.func
    def _update_int_colloc_value_5(self, source_params: ti.template()):
        self.int_colloc_values[5] = source_params.dt_psi4_to_strain + (
            3089.18
            + 4.89194 * source_params.eta
            + 190.008 * source_params.eta_pow2
            - 255.245 * source_params.eta_pow3
            + (2.96997 + 57.1612 * source_params.eta - 432.223 * source_params.eta_pow2)
            * source_params.S_tot_hat
            + (
                -18.8929
                + 630.516 * source_params.eta
                - 2804.66 * source_params.eta_pow2
            )
            * source_params.S_tot_hat_pow2
            + (-24.6193 + 549.085 * source_params.eta_pow2)
            * source_params.S_tot_hat_pow3
            + (
                -12.8798
                - 722.674 * source_params.eta
                + 3967.43 * source_params.eta_pow2
            )
            * source_params.S_tot_hat_pow4
            + 74.0984
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
        )

    @ti.func
    def update_phase_coefficients(
        self,
        pn_coefficients_22: ti.template(),
        phase_coefficients_22: ti.template(),
        pn_coefficients_21: ti.template(),
        source_params: ti.template(),
    ):
        # intermediate
        self._set_colloc_points(source_params)
        self._set_intermediate_coefficients(
            pn_coefficients_22, phase_coefficients_22, source_params
        )
        # inspiral
        self._set_ins_rescaling_coefficients(1.0, phase_coefficients_22)
        if source_params.eta > 0.01:
            self.Lambda_lm = self._Lambda_21_PN()
        else:
            self.Lambda_lm = self._Lambda_21_fit(source_params)
        # merge-ringdown
        self._set_MRD_rescaling_coefficients(
            1.0 / 3.0, source_params.QNM_freqs_lm["21"], source_params
        )
        # continuous conditions
        self.ins_C1 = self._intermediate_d_phase(
            source_params.QNM_freqs_lm["21"], self._useful_powers.ins_f_end
        ) - self._inspiral_d_phase(pn_coefficients_21, self._useful_powers.ins_f_end)
        # Note we have dropped the constant of phi_5, ins_C0 is different with CINSP in 
        # lalsimulation. ins_C0 (tiwave) = CINSP (lalsim) - phi_5
        self.ins_C0 = (
            self._intermediate_phase(
                source_params.QNM_freqs_lm["21"], self._useful_powers.ins_f_end
            )
            - self._inspiral_phase(pn_coefficients_21, self._useful_powers.ins_f_end)
            - self.ins_C1 * self.ins_f_end
        )
        self.MRD_C1 = self._intermediate_d_phase(
            source_params.QNM_freqs_lm["21"], self._useful_powers.int_f_end
        ) - self._merge_ringdown_d_phase(
            source_params.QNM_freqs_lm["21"], self._useful_powers.int_f_end
        )
        self.MRD_C0 = (
            self._intermediate_phase(
                source_params.QNM_freqs_lm["21"], self._useful_powers.int_f_end
            )
            - self._merge_ringdown_phase(
                source_params.QNM_freqs_lm["21"], self._useful_powers.int_f_end
            )
            - self.MRD_C1 * self.int_f_end
        )


@sub_struct_from(PhaseCoefficientsHighModesBase)
class PhaseCoefficientsMode33:
    # # Inspiral
    # Lambda_PN: ti.f64  # corrections for the complex PN amplitudes, eq 4.9
    # C1_ins: ti.f64
    # C2_ins: ti.f64

    # # Intermediate
    # c_0: ti.f64
    # c_1: ti.f64
    # c_2: ti.f64
    # c_3: ti.f64
    # c_4: ti.f64
    # c_L: ti.f64
    # int_colloc_points: ti.types.vector(6, ti.f64)
    # int_colloc_values: ti.types.vector(6, ti.f64)

    # # Merge-ringdown
    # alpha_2: ti.f64
    # alpha_L: ti.f64

    # @ti.func
    # def _set_colloc_points(self, source_params: ti.template()):
    #     # Intermediate
    #     ins_f_end = (
    #         1.0
    #         + 0.001
    #         * (0.25 / source_params.eta - 1.0)
    #         * source_params.f_MECO
    #         * emm
    #         * 0.5
    #     )
    #     int_f_end = source_params.f_ring_21
    #     self.int_colloc_points[0] = ins_f_end
    #     self.int_colloc_points[1] = (
    #         tm.sqrt(3.0) * (ins_f_end - int_f_end) + 2.0 * (ins_f_end + int_f_end)
    #     ) / 4.0
    #     self.int_colloc_points[2] = (3.0 * ins_f_end + int_f_end) / 4.0
    #     self.int_colloc_points[3] = (ins_f_end + int_f_end) / 2.0
    #     self.int_colloc_points[4] = (ins_f_end + 3.0 * int_f_end) / 4.0
    #     self.int_colloc_points[5] = (ins_f_end + 7.0 * int_f_end) / 8.0
    #     self.int_colloc_points[6] = int_f_end

    # @ti.func
    # def _set_inspiral_coefficients(self, source_params: ti.template()):
    #     if source_params.eta > 0.01:
    #         self.Lambda_PN = 2.0 / 3.0 * PI * (21.0 / 5.0 - 6.0 * tm.log(1.5))
    #     else:
    #         self.Lambda_PN = (
    #             4.1138398568400705
    #             + 9.772510519809892 * source_params.eta
    #             - 103.92956504520747 * source_params.eta_pow2
    #             + 242.3428625556764 * source_params.eta_pow3
    #             + (
    #                 (
    #                     -0.13253553909611435
    #                     + 26.644159828590055 * source_params.eta
    #                     - 105.09339163109497 * source_params.eta_pow2
    #                 )
    #                 * source_params.S_tot_hat
    #             )
    #             / (1.0 + 0.11322426762297967 * source_params.S_tot_hat)
    #             - 19.705359163581168
    #             * source_params.delta_chi
    #             * source_params.eta_pow2
    #             * source_params.delta
    #         )

    # @ti.func
    # def _set_merge_ringdown_coefficients(self, source_params: ti.template()):
    #     pass

    # @ti.func
    # def _set_intermediate_coefficients(self, source_params: ti.template()):
    #     self.int_colloc_values[0] = source_params.dt_psi4_to_strain + (
    #         4360.19
    #         + 4.27128 / source_params.eta
    #         - 8727.4 * source_params.eta
    #         + 18485.9 * source_params.eta_pow2
    #         + 371303.00000000006 * source_params.eta_pow3
    #         - 3.22792e6 * source_params.eta_pow4
    #         + 1.01799e7 * source_params.eta_pow5
    #         - 1.15659e7 * source_params.eta_pow6
    #         + (
    #             (
    #                 11.6635
    #                 - 251.579 * source_params.eta
    #                 - 3255.6400000000003 * source_params.eta_pow2
    #                 + 19614.6 * source_params.eta_pow3
    #                 - 34860.2 * source_params.eta_pow4
    #             )
    #             * source_params.S_tot_hat
    #             + (
    #                 14.8017
    #                 + 204.025 * source_params.eta
    #                 - 5421.92 * source_params.eta_pow2
    #                 + 36587.3 * source_params.eta_pow3
    #                 - 74299.5 * source_params.eta_pow4
    #             )
    #             * source_params.S_tot_hat_pow2
    #         )
    #         / source_params.eta
    #         + source_params.eta
    #         * (
    #             223.65100000000004
    #             * source_params.chi_1
    #             * source_params.delta
    #             * (3.9201300240106223 + 1.0 * source_params.eta)
    #             - 223.65100000000004
    #             * source_params.chi_2
    #             * source_params.delta
    #             * (3.9201300240106223 + 1.0 * source_params.eta)
    #         )
    #     )
    #     self.int_colloc_values[1] = source_params.dt_psi4_to_strain + (
    #         3797.06
    #         + 0.786684 / source_params.eta
    #         - 2397.09 * source_params.eta
    #         - 25514.0 * source_params.eta_pow2
    #         + 518314.99999999994 * source_params.eta_pow3
    #         - 3.41708e6 * source_params.eta_pow4
    #         + 1.01799e7 * source_params.eta_pow5
    #         - 1.15659e7 * source_params.eta_pow6
    #     )
    #     +(
    #         (
    #             6.7812399999999995
    #             + 39.4668 * source_params.eta
    #             - 3520.37 * source_params.eta_pow2
    #             + 19614.6 * source_params.eta_pow3
    #             - 34860.2 * source_params.eta_pow4
    #         )
    #         * source_params.S_tot_hat
    #         + (
    #             4.80384
    #             + 293.215 * source_params.eta
    #             - 5914.61 * source_params.eta_pow2
    #             + 36587.3 * source_params.eta_pow3
    #             - 74299.5 * source_params.eta_pow4
    #         )
    #         * source_params.S_tot_hat_pow2
    #     ) / source_params.eta
    #     -223.65100000000004 * source_params.delta * source_params.eta * (
    #         source_params.chi_1 * (-1.3095134830606614 - 1.0 * source_params.eta)
    #         + source_params.chi_2 * (1.3095134830606614 + 1.0 * source_params.eta)
    #     )
    #     self.int_colloc_values[2] = source_params.dt_psi4_to_strain + (
    #         3321.83
    #         + 1796.03 * source_params.eta
    #         - 52406.1 * source_params.eta_pow2
    #         + 605028.0 * source_params.eta_pow3
    #         - 3.52532e6 * source_params.eta_pow4
    #         + 1.01799e7 * source_params.eta_pow5
    #         - 1.15659e7 * source_params.eta_pow6
    #         + (
    #             223.601
    #             - 3714.77 * source_params.eta
    #             + 19614.6 * source_params.eta_pow2
    #             - 34860.2 * source_params.eta_pow3
    #         )
    #         * source_params.S_tot_hat
    #         + (
    #             314.317
    #             - 5906.46 * source_params.eta
    #             + 36587.3 * source_params.eta_pow2
    #             - 74299.5 * source_params.eta_pow3
    #         )
    #         * source_params.S_tot_hat_pow2
    #         + 223.651
    #         * source_params.delta_chi
    #         * source_params.delta
    #         * source_params.eta_pow2
    #     )
    #     self.int_colloc_values[3] = source_params.dt_psi4_to_strain + (
    #         3239.44
    #         - 661.15 * source_params.eta
    #         + 5139.79 * source_params.eta_pow2
    #         + 3456.2 * source_params.eta_pow3
    #         - 248477.0 * source_params.eta_pow4
    #         + 1.17255e6 * source_params.eta_pow5
    #         - 1.70363e6 * source_params.eta_pow6
    #         + (
    #             225.859
    #             - 4150.09 * source_params.eta
    #             + 24364.0 * source_params.eta_pow2
    #             - 46537.3 * source_params.eta_pow3
    #         )
    #         * source_params.S_tot_hat
    #         + (
    #             35.2439
    #             - 994.971 * source_params.eta
    #             + 8953.98 * source_params.eta_pow2
    #             - 23603.5 * source_params.eta_pow3
    #         )
    #         * source_params.S_tot_hat_pow2
    #         + (
    #             -310.489
    #             + 5946.15 * source_params.eta
    #             - 35337.1 * source_params.eta_pow2
    #             + 67102.4 * source_params.eta_pow3
    #         )
    #         * source_params.S_tot_hat_pow3
    #         + 30.484
    #         * source_params.delta_chi
    #         * source_params.delta
    #         * source_params.eta_pow2
    #     )
    #     self.int_colloc_values[4] = source_params.dt_psi4_to_strain + (
    #         3114.3
    #         + 2143.06 * source_params.eta
    #         - 49428.3 * source_params.eta_pow2
    #         + 563997.0 * source_params.eta_pow3
    #         - 3.35991e6 * source_params.eta_pow4
    #         + 9.99745e6 * source_params.eta_pow5
    #         - 1.17123e7 * source_params.eta_pow6
    #         + (
    #             190.051
    #             - 3705.08 * source_params.eta
    #             + 23046.2 * source_params.eta_pow2
    #             - 46537.3 * source_params.eta_pow3
    #         )
    #         * source_params.S_tot_hat
    #         + (
    #             63.6615
    #             - 1414.2 * source_params.eta
    #             + 10166.1 * source_params.eta_pow2
    #             - 23603.5 * source_params.eta_pow3
    #         )
    #         * source_params.S_tot_hat_pow2
    #         + (
    #             -257.524
    #             + 5179.97 * source_params.eta
    #             - 33001.4 * source_params.eta_pow2
    #             + 67102.4 * source_params.eta_pow3
    #         )
    #         * source_params.S_tot_hat_pow3
    #         + 54.9833
    #         * source_params.delta_chi
    #         * source_params.delta
    #         * source_params.eta_pow2
    #     )
    #     self.int_colloc_values[5] = source_params.dt_psi4_to_strain + (
    #         3111.46
    #         + 384.121 * source_params.eta
    #         - 13003.6 * source_params.eta_pow2
    #         + 179537.0 * source_params.eta_pow3
    #         - 1.19313e6 * source_params.eta_pow4
    #         + 3.79886e6 * source_params.eta_pow5
    #         - 4.64858e6 * source_params.eta_pow6
    #         + (
    #             182.864
    #             - 3834.22 * source_params.eta
    #             + 24532.9 * source_params.eta_pow2
    #             - 50165.9 * source_params.eta_pow3
    #         )
    #         * source_params.S_tot_hat
    #         + (
    #             21.0158
    #             - 746.957 * source_params.eta
    #             + 6701.33 * source_params.eta_pow2
    #             - 17842.3 * source_params.eta_pow3
    #         )
    #         * source_params.S_tot_hat_pow2
    #         + (
    #             -292.855
    #             + 5886.62 * source_params.eta
    #             - 37382.4 * source_params.eta_pow2
    #             + 75501.8 * source_params.eta_pow3
    #         )
    #         * source_params.S_tot_hat_pow3
    #         + 75.5162
    #         * source_params.delta_chi
    #         * source_params.delta
    #         * source_params.eta_pow2
    #     )
    #     # special operation for 21 mode to avoide sharp transitions in high-spin cases
    #     if source_params.S_tot_hat >= 0.8:
    #         ins_val_0 = phase_coefficients_22.compute_dphase(
    #             source_params, pn_coefficients, self.useful_powers.int_f0
    #         )
    #         ins_val_1 = phase_coefficients_22.compute_dphase(
    #             source_params, pn_coefficients, self.useful_powers.int_f1
    #         )
    #         ins_val_2 = phase_coefficients_22.compute_dphase(
    #             source_params, pn_coefficients, self.useful_powers.int_f2
    #         )
    #         diff_01 = ins_val_0 - ins_val_1
    #         diff_12 = ins_val_1 - ins_val_2
    #         self.int_colloc_values[1] = self.int_colloc_values[2] + diff_12
    #         self.int_colloc_values[0] = self.int_colloc_values[1] + diff_01
    #     # collocation points need to be chosen according to spin and mass ratio

    # def _set_intermediate_coefficients_case_1(self, source_params: ti.template()):
    #     """
    #     using collocation points: 0, 1, 3, 4, 5
    #     for situation with (eta < etaEMR) or (emm == ell and STotR >= 0.8) or (modeTag == 33 and STotR < 0)
    #     """
    #     pass

    # def _set_intermediate_coefficients_case_2(self, source_params: ti.template()):
    #     """
    #     using collocation points: 0, 1, 2, 4, 5
    #     for situation with (STotR >= 0.8) and (modeTag == 21)"""
    #     pass

    # def _set_intermediate_coefficients_case_3(self, source_params: ti.template()):
    #     """
    #     using collocation points: 0, 1, 2, 3, 5
    #     remaining parameter space
    #     """
    #     pass

    # def _set_intermediate_coefficients(self, source_params: ti.template()):
    #     # simplify the conditional structure in LALSimIMRPhenomXHM_internals.c l.2108 for modeTag=33
    #     if (
    #         (source_params.eta < eta_EMR)
    #         or (source_params.S_tot_hat >= 0.8)
    #         or (source_params.S_tot_hat < 0.0)
    #     ):
    #         self._set_intermediate_coefficients_case_1(source_params)
    #     else:
    #         self._set_intermediate_coefficients_case_3(source_params)

    # # // choose collocation points according to spin/mass ratio
    # #     // current catalogue of simulations include some cases that create unphysical effects in the fits -> we need to use different subset of collocation points according to the parameters (we have to pick 5 out of 6 available fits)
    # #     /* cpoints_indices is an array of integers labelling the collocation points chosen in each case, e.g.
    # #      cpoints_indices={0,1,3,4,5} would mean that we are discarding the 3rd collocation points in the reconstructio */

    # #     int cpoints_indices[nCollocationPts_inter];
    # #     cpoints_indices[0]=0;
    # #     cpoints_indices[1]=1;
    # #     cpoints_indices[4]=5;

    # # if((pWF22->eta<pWFHM->etaEMR)||(emm==ell&&pWF22->STotR>=0.8)||(pWFHM->modeTag==33&&pWF22->STotR<0))
    # # {
    # #     cpoints_indices[2]=3;
    # #     cpoints_indices[3]=4;
    # # }
    # # else if(pWF22->STotR>=0.8&&pWFHM->modeTag==21){

    # #     cpoints_indices[2]=2;
    # #     cpoints_indices[3]=4;
    # # }

    # # else{
    # #     cpoints_indices[2]=2;
    # #     cpoints_indices[3]=3;
    # # }

    @ti.func
    def update_phase_coefficients(
        self,
        phase_coefficients_22: ti.template(),
        source_params: ti.template(),
    ):
        self._set_ins_rescaling_coefficients(3.0, phase_coefficients_22)

    # @ti.func
    # def phase_inspiral_ansatz(self):
    #     pass

    # @ti.func
    # def phase_intermediate_ansatz(self):
    #     pass

    # @ti.func
    # def phase_merge_ringdown_ansatz(self):
    #     pass


@sub_struct_from(PhaseCoefficientsHighModesBase)
class PhaseCoefficientsMode32:
    # # Inspiral
    # Lambda_PN: ti.f64  # corrections for the complex PN amplitudes, eq 4.9
    # C1_ins: ti.f64
    # C2_ins: ti.f64

    # # Intermediate
    # c_0: ti.f64
    # c_1: ti.f64
    # c_2: ti.f64
    # c_3: ti.f64
    # c_4: ti.f64
    # c_L: ti.f64
    # int_colloc_points: ti.types.vector(6, ti.f64)
    # int_colloc_values: ti.types.vector(6, ti.f64)

    # @ti.func
    # def _set_colloc_points(self, source_params: ti.template()):
    #     # Intermediate
    #     ins_f_end = (
    #         1.0
    #         + 0.001
    #         * (0.25 / source_params.eta - 1.0)
    #         * source_params.f_MECO
    #         * emm
    #         * 0.5
    #     )
    #     int_f_end = source_params.f_ring_21
    #     self.int_colloc_points[0] = ins_f_end
    #     self.int_colloc_points[1] = (
    #         tm.sqrt(3.0) * (ins_f_end - int_f_end) + 2.0 * (ins_f_end + int_f_end)
    #     ) / 4.0
    #     self.int_colloc_points[2] = (3.0 * ins_f_end + int_f_end) / 4.0
    #     self.int_colloc_points[3] = (ins_f_end + int_f_end) / 2.0
    #     self.int_colloc_points[4] = (ins_f_end + 3.0 * int_f_end) / 4.0
    #     self.int_colloc_points[5] = (ins_f_end + 7.0 * int_f_end) / 8.0
    #     self.int_colloc_points[6] = int_f_end

    # @ti.func
    # def _set_inspiral_coefficients(self, source_params: ti.template()):
    #     if source_params.eta > 0.01:
    #         self.Lambda_PN = (2376.0 * PI * (-5.0 + 22.0 * source_params.eta)) / (
    #             -3960.0 + 11880 * source_params.eta
    #         )
    #     else:
    #         self.Lambda_PN = (
    #             (
    #                 9.913819875501506
    #                 + 18.424900617803107 * source_params.eta
    #                 - 574.8672384388947 * source_params.eta_pow2
    #                 + 2671.7813055097877 * source_params.eta_pow3
    #                 - 6244.001932443913 * source_params.eta_pow4
    #             )
    #             / (1.0 - 0.9103118343073325 * source_params.eta)
    #             + (
    #                 -4.367632806613781
    #                 + 245.06757304950986 * source_params.eta
    #                 - 2233.9319708029775 * source_params.eta_pow2
    #                 + 5894.355429022858 * source_params.eta_pow3
    #             )
    #             * source_params.S_tot_hat
    #             + (
    #                 -1.375112297530783
    #                 - 1876.760129419146 * source_params.eta
    #                 + 17608.172965575013 * source_params.eta_pow2
    #                 - 40928.07304790013 * source_params.eta_pow3
    #             )
    #             * source_params.S_tot_hat_pow2
    #             + (
    #                 -1.28324755577382
    #                 - 138.36970336658558 * source_params.eta
    #                 + 708.1455154504333 * source_params.eta_pow2
    #                 - 273.23750933544176 * source_params.eta_pow3
    #             )
    #             * source_params.S_tot_hat_pow3
    #             + (
    #                 1.8403161863444328
    #                 + 2009.7361967331492 * source_params.eta
    #                 - 18636.271414571278 * source_params.eta_pow2
    #                 + 42379.205045791656 * source_params.eta_pow3
    #             )
    #             * source_params.S_tot_hat_pow4
    #             + source_params.delta_chi
    #             * source_params.delta
    #             * source_params.eta_pow2
    #             * (
    #                 -105.34550407768225
    #                 - 1566.1242344157668 * source_params.chi_1 * source_params.eta
    #                 + 1566.1242344157668 * source_params.chi_2 * source_params.eta
    #                 + 2155.472229664981 * source_params.eta * source_params.S_tot_hat
    #             )
    #         )

    # @ti.func
    # def _set_intermediate_coefficients(self, source_params: ti.template()):
    #     self.int_colloc_values[0] = source_params.dt_psi4_to_strain + (
    #         4414.11
    #         + 4.21564 / source_params.eta
    #         - 10687.8 * source_params.eta
    #         + 58234.6 * source_params.eta_pow2
    #         - 64068.40000000001 * source_params.eta_pow3
    #         - 704442.0 * source_params.eta_pow4
    #         + 2.86393e6 * source_params.eta_pow5
    #         - 3.26362e6 * source_params.eta_pow6
    #         + (
    #             (
    #                 6.39833
    #                 - 610.267 * source_params.eta
    #                 + 2095.72 * source_params.eta_pow2
    #                 - 3970.89 * source_params.eta_pow3
    #             )
    #             * source_params.S_tot_hat
    #             + (
    #                 22.956700000000005
    #                 - 99.1551 * source_params.eta
    #                 + 331.593 * source_params.eta_pow2
    #                 - 794.79 * source_params.eta_pow3
    #             )
    #             * source_params.S_tot_hat_pow2
    #             + (
    #                 10.4333
    #                 + 43.8812 * source_params.eta
    #                 - 541.261 * source_params.eta_pow2
    #                 + 294.289 * source_params.eta_pow3
    #             )
    #             * source_params.S_tot_hat_pow3
    #             + source_params.eta
    #             * (
    #                 106.047
    #                 - 1569.0299999999997 * source_params.eta
    #                 + 4810.61 * source_params.eta_pow2
    #             )
    #             * source_params.S_tot_hat_pow4
    #         )
    #         / source_params.eta
    #         + 132.244
    #         * source_params.delta
    #         * source_params.eta
    #         * (
    #             source_params.chi_1 * (6.227738120444028 - 1.0 * source_params.eta)
    #             + source_params.chi_2 * (-6.227738120444028 + 1.0 * source_params.eta)
    #         )
    #     )
    #     self.int_colloc_values[1] = source_params.dt_psi4_to_strain + (
    #         3980.7
    #         + 0.956703 / source_params.eta
    #         - 6202.38 * source_params.eta
    #         + 29218.1 * source_params.eta_pow2
    #         + 24484.2 * source_params.eta_pow3
    #         - 807629.0 * source_params.eta_pow4
    #         + 2.86393e6 * source_params.eta_pow5
    #         - 3.26362e6 * source_params.eta_pow6
    #         + (
    #             (
    #                 1.92692
    #                 - 226.825 * source_params.eta
    #                 + 75.246 * source_params.eta_pow2
    #                 + 1291.56 * source_params.eta_pow3
    #             )
    #             * source_params.S_tot_hat
    #             + (
    #                 15.328700000000001
    #                 - 99.1551 * source_params.eta
    #                 + 608.328 * source_params.eta_pow2
    #                 - 2402.94 * source_params.eta_pow3
    #             )
    #             * source_params.S_tot_hat_pow2
    #             + (
    #                 10.4333
    #                 + 43.8812 * source_params.eta
    #                 - 541.261 * source_params.eta_pow2
    #                 + 294.289 * source_params.eta_pow3
    #             )
    #             * source_params.S_tot_hat_pow3
    #             + source_params.eta
    #             * (
    #                 106.047
    #                 - 1569.0299999999997 * source_params.eta
    #                 + 4810.61 * source_params.eta_pow2
    #             )
    #             * source_params.S_tot_hat_pow4
    #         )
    #         / source_params.eta
    #         + 132.244
    #         * source_params.delta
    #         * source_params.eta
    #         * (
    #             source_params.chi_1 * (2.5769789177580837 - 1.0 * source_params.eta)
    #             + source_params.chi_2 * (-2.5769789177580837 + 1.0 * source_params.eta)
    #         )
    #     )
    #     self.int_colloc_values[2] = source_params.dt_psi4_to_strain + (
    #         3416.57
    #         + 2308.63 * source_params.eta
    #         - 84042.9 * source_params.eta_pow2
    #         + 1.01936e6 * source_params.eta_pow3
    #         - 6.0644e6 * source_params.eta_pow4
    #         + 1.76399e7 * source_params.eta_pow5
    #         - 2.0065e7 * source_params.eta_pow6
    #         + (
    #             24.6295
    #             - 282.354 * source_params.eta
    #             - 2582.55 * source_params.eta_pow2
    #             + 12750.0 * source_params.eta_pow3
    #         )
    #         * source_params.S_tot_hat
    #         + (
    #             433.675
    #             - 8775.86 * source_params.eta
    #             + 56407.8 * source_params.eta_pow2
    #             - 114798.0 * source_params.eta_pow3
    #         )
    #         * source_params.S_tot_hat_pow2
    #         + (
    #             559.705
    #             - 10627.4 * source_params.eta
    #             + 61581.0 * source_params.eta_pow2
    #             - 114029.0 * source_params.eta_pow3
    #         )
    #         * source_params.S_tot_hat_pow3
    #         + (106.047 - 1569.03 * source_params.eta + 4810.61 * source_params.eta_pow2)
    #         * source_params.S_tot_hat_pow4
    #         + 63.9466
    #         * source_params.delta_chi
    #         * source_params.delta
    #         * source_params.eta_pow2
    #     )
    #     self.int_colloc_values[3] = source_params.dt_psi4_to_strain + (
    #         3307.49
    #         - 476.909 * source_params.eta
    #         - 5980.37 * source_params.eta_pow2
    #         + 127610.0 * source_params.eta_pow3
    #         - 919108.0 * source_params.eta_pow4
    #         + 2.86393e6 * source_params.eta_pow5
    #         - 3.26362e6 * source_params.eta_pow6
    #         + (
    #             -5.02553
    #             - 282.354 * source_params.eta
    #             + 1291.56 * source_params.eta_pow2
    #         )
    #         * source_params.S_tot_hat
    #         + (
    #             -43.8823
    #             + 740.123 * source_params.eta
    #             - 2402.94 * source_params.eta_pow2
    #         )
    #         * source_params.S_tot_hat_pow2
    #         + (43.8812 - 370.362 * source_params.eta + 294.289 * source_params.eta_pow2)
    #         * source_params.S_tot_hat_pow3
    #         + (106.047 - 1569.03 * source_params.eta + 4810.61 * source_params.eta_pow2)
    #         * source_params.S_tot_hat_pow4
    #         - 132.244
    #         * source_params.delta_chi
    #         * source_params.delta
    #         * source_params.eta_pow2
    #     )
    #     self.int_colloc_values[4] = source_params.dt_psi4_to_strain + (
    #         3259.03
    #         - 3967.58 * source_params.eta
    #         + 111203.0 * source_params.eta_pow2
    #         - 1.81883e6 * source_params.eta_pow3
    #         + 1.73811e7 * source_params.eta_pow4
    #         - 9.56988e7 * source_params.eta_pow5
    #         + 2.75056e8 * source_params.eta_pow6
    #         - 3.15866e8 * source_params.eta_pow7
    #         + (19.7509 - 1104.53 * source_params.eta + 3810.18 * source_params.eta_pow2)
    #         * source_params.S_tot_hat
    #         + (-230.07 + 2314.51 * source_params.eta - 5944.49 * source_params.eta_pow2)
    #         * source_params.S_tot_hat_pow2
    #         + (
    #             -201.633
    #             + 2183.43 * source_params.eta
    #             - 6233.99 * source_params.eta_pow2
    #         )
    #         * source_params.S_tot_hat_pow3
    #         + (106.047 - 1569.03 * source_params.eta + 4810.61 * source_params.eta_pow2)
    #         * source_params.S_tot_hat_pow4
    #         + 112.714
    #         * source_params.delta_chi
    #         * source_params.delta
    #         * source_params.eta_pow2
    #     )
    #     self.int_colloc_values[5] = source_params.dt_psi4_to_strain + (
    #         3259.03
    #         - 3967.58 * source_params.eta
    #         + 111203.0 * source_params.eta_pow2
    #         - 1.81883e6 * source_params.eta_pow3
    #         + 1.73811e7 * source_params.eta_pow4
    #         - 9.56988e7 * source_params.eta_pow5
    #         + 2.75056e8 * source_params.eta_pow6
    #         - 3.15866e8 * source_params.eta_pow7
    #         + (19.7509 - 1104.53 * source_params.eta + 3810.18 * source_params.eta_pow2)
    #         * source_params.S_tot_hat
    #         + (-230.07 + 2314.51 * source_params.eta - 5944.49 * source_params.eta_pow2)
    #         * source_params.S_tot_hat_pow2
    #         + (
    #             -201.633
    #             + 2183.43 * source_params.eta
    #             - 6233.99 * source_params.eta_pow2
    #         )
    #         * source_params.S_tot_hat_pow3
    #         + (106.047 - 1569.03 * source_params.eta + 4810.61 * source_params.eta_pow2)
    #         * source_params.S_tot_hat_pow4
    #         + 112.714
    #         * source_params.delta_chi
    #         * source_params.delta
    #         * source_params.eta_pow2
    #     )
    @ti.func
    def update_phase_coefficients(
        self,
        phase_coefficients_22: ti.template(),
        source_params: ti.template(),
    ):
        self._set_ins_rescaling_coefficients(2.0, phase_coefficients_22)


@sub_struct_from(PhaseCoefficientsHighModesBase)
class PhaseCoefficientsMode44:
    # # Inspiral
    # Lambda_PN: ti.f64  # corrections for the complex PN amplitudes, eq 4.9
    # C1_ins: ti.f64
    # C2_ins: ti.f64
    # # Intermediate
    # c_0: ti.f64
    # c_1: ti.f64
    # c_2: ti.f64
    # c_3: ti.f64
    # c_4: ti.f64
    # c_L: ti.f64
    # int_colloc_points: ti.types.vector(6, ti.f64)
    # int_colloc_values: ti.types.vector(6, ti.f64)

    # # Merge-ringdown
    # alpha_2: ti.f64
    # alpha_L: ti.f64

    # @ti.func
    # def _set_colloc_points(self, source_params: ti.template()):
    #     # Intermediate
    #     ins_f_end = (
    #         1.0
    #         + 0.001
    #         * (0.25 / source_params.eta - 1.0)
    #         * source_params.f_MECO
    #         * emm
    #         * 0.5
    #     )
    #     int_f_end = source_params.f_ring_21
    #     self.int_colloc_points[0] = ins_f_end
    #     self.int_colloc_points[1] = (
    #         tm.sqrt(3.0) * (ins_f_end - int_f_end) + 2.0 * (ins_f_end + int_f_end)
    #     ) / 4.0
    #     self.int_colloc_points[2] = (3.0 * ins_f_end + int_f_end) / 4.0
    #     self.int_colloc_points[3] = (ins_f_end + int_f_end) / 2.0
    #     self.int_colloc_points[4] = (ins_f_end + 3.0 * int_f_end) / 4.0
    #     self.int_colloc_points[5] = (ins_f_end + 7.0 * int_f_end) / 8.0
    #     self.int_colloc_points[6] = int_f_end

    # @ti.func
    # def _set_inspiral_coefficients(self, source_params: ti.template()):
    #     if source_params.eta > 0.01:
    #         self.Lambda_PN = (
    #             45045.0
    #             * PI
    #             * (
    #                 336.0
    #                 - 1193.0 * source_params.eta
    #                 + 320.0 * (-1.0 + 3.0 * source_params.eta) * tm.log(2.0)
    #             )
    #             / (2.0 * (1801800.0 - 5405400.0 * source_params.eta))
    #         )

    #     else:
    #         self.Lambda_PN = (
    #             5.254484747463392
    #             - 21.277760168559862 * source_params.eta
    #             + 160.43721442910618 * source_params.eta_pow2
    #             - 1162.954360723399 * source_params.eta_pow3
    #             + 1685.5912722190276 * source_params.eta_pow4
    #             - 1538.6661348106031 * source_params.eta_pow5
    #             + (
    #                 0.007067861615983771
    #                 - 10.945895160727437 * source_params.eta
    #                 + 246.8787141453734 * source_params.eta_pow2
    #                 - 810.7773268493444 * source_params.eta_pow3
    #             )
    #             * source_params.S_tot_hat
    #             + (
    #                 0.17447830920234977
    #                 + 4.530539154777984 * source_params.eta
    #                 - 176.4987316167203 * source_params.eta_pow2
    #                 + 621.6920322846844 * source_params.eta_pow3
    #             )
    #             * source_params.S_tot_hat_pow2
    #             - 8.384066369867833
    #             * source_params.delta_chi
    #             * source_params.delta
    #             * source_params.eta_pow2
    #         )

    # @ti.func
    # def _set_merge_ringdown_coefficients(self, source_params: ti.template()):
    #     pass

    # @ti.func
    # def _set_intermediate_coefficients(self, source_params: ti.template()):
    #     self.int_colloc_values[0] = source_params.dt_psi4_to_strain + (
    #         4349.66
    #         + 4.34125 / source_params.eta
    #         - 8202.33 * source_params.eta
    #         + 5534.1 * source_params.eta_pow2
    #         + 536500.0 * source_params.eta_pow3
    #         - 4.33197e6 * source_params.eta_pow4
    #         + 1.37792e7 * source_params.eta_pow5
    #         - 1.60802e7 * source_params.eta_pow6
    #         + (
    #             (
    #                 12.0704
    #                 - 528.098 * source_params.eta
    #                 + 1822.9100000000003 * source_params.eta_pow2
    #                 - 9349.73 * source_params.eta_pow3
    #                 + 17900.9 * source_params.eta_pow4
    #             )
    #             * source_params.S_tot_hat
    #             + (
    #                 10.4092
    #                 + 253.334 * source_params.eta
    #                 - 5452.04 * source_params.eta_pow2
    #                 + 35416.6 * source_params.eta_pow3
    #                 - 71523.0 * source_params.eta_pow4
    #             )
    #             * source_params.S_tot_hat_pow2
    #             + source_params.eta
    #             * (
    #                 492.60300000000007
    #                 - 9508.5 * source_params.eta
    #                 + 57303.4 * source_params.eta_pow2
    #                 - 109418.0 * source_params.eta_pow3
    #             )
    #             * source_params.S_tot_hat_pow3
    #         )
    #         / source_params.eta
    #         - 262.143
    #         * source_params.delta
    #         * source_params.eta
    #         * (
    #             source_params.chi_1 * (-3.0782778864970646 - 1.0 * source_params.eta)
    #             + source_params.chi_2 * (3.0782778864970646 + 1.0 * source_params.eta)
    #         )
    #     )
    #     self.int_colloc_values[1] = source_params.dt_psi4_to_strain + (
    #         3804.19
    #         + 0.66144 / source_params.eta
    #         - 2421.77 * source_params.eta
    #         - 33475.8 * source_params.eta_pow2
    #         + 665951.0 * source_params.eta_pow3
    #         - 4.50145e6 * source_params.eta_pow4
    #         + 1.37792e7 * source_params.eta_pow5
    #         - 1.60802e7 * source_params.eta_pow6
    #         + (
    #             (
    #                 5.83038
    #                 - 172.047 * source_params.eta
    #                 + 926.576 * source_params.eta_pow2
    #                 - 7676.87 * source_params.eta_pow3
    #                 + 17900.9 * source_params.eta_pow4
    #             )
    #             * source_params.S_tot_hat
    #             + (
    #                 6.17601
    #                 + 253.334 * source_params.eta
    #                 - 5672.02 * source_params.eta_pow2
    #                 + 35722.1 * source_params.eta_pow3
    #                 - 71523.0 * source_params.eta_pow4
    #             )
    #             * source_params.S_tot_hat_pow2
    #             + source_params.eta
    #             * (
    #                 492.60300000000007
    #                 - 9508.5 * source_params.eta
    #                 + 57303.4 * source_params.eta_pow2
    #                 - 109418.0 * source_params.eta_pow3
    #             )
    #             * source_params.S_tot_hat_pow3
    #         )
    #         / source_params.eta
    #         - 262.143
    #         * source_params.delta
    #         * source_params.eta
    #         * (
    #             source_params.chi_1 * (-1.0543062374352932 - 1.0 * source_params.eta)
    #             + source_params.chi_2 * (1.0543062374352932 + 1.0 * source_params.eta)
    #         )
    #     )
    #     self.int_colloc_values[2] = source_params.dt_psi4_to_strain + (
    #         3308.97
    #         + 2353.58 * source_params.eta
    #         - 66340.1 * source_params.eta_pow2
    #         + 777272.0 * source_params.eta_pow3
    #         - 4.64438e6 * source_params.eta_pow4
    #         + 1.37792e7 * source_params.eta_pow5
    #         - 1.60802e7 * source_params.eta_pow6
    #         + (
    #             -21.5697
    #             + 926.576 * source_params.eta
    #             - 7989.26 * source_params.eta_pow2
    #             + 17900.9 * source_params.eta_pow3
    #         )
    #         * source_params.S_tot_hat
    #         + (
    #             353.539
    #             - 6403.24 * source_params.eta
    #             + 37599.5 * source_params.eta_pow2
    #             - 71523.0 * source_params.eta_pow3
    #         )
    #         * source_params.S_tot_hat_pow2
    #         + (
    #             492.603
    #             - 9508.5 * source_params.eta
    #             + 57303.4 * source_params.eta_pow2
    #             - 109418.0 * source_params.eta_pow3
    #         )
    #         * source_params.S_tot_hat_pow3
    #         + 262.143
    #         * source_params.delta_chi
    #         * source_params.delta
    #         * source_params.eta_pow2
    #     )
    #     self.int_colloc_values[3] = source_params.dt_psi4_to_strain + (
    #         3245.63
    #         - 928.56 * source_params.eta
    #         + 8463.89 * source_params.eta_pow2
    #         - 17422.6 * source_params.eta_pow3
    #         - 165169.0 * source_params.eta_pow4
    #         + 908279.0 * source_params.eta_pow5
    #         - 1.31138e6 * source_params.eta_pow6
    #         + (
    #             32.506
    #             - 590.293 * source_params.eta
    #             + 3536.61 * source_params.eta_pow2
    #             - 6758.52 * source_params.eta_pow3
    #         )
    #         * source_params.S_tot_hat
    #         + (
    #             -25.7716
    #             + 738.141 * source_params.eta
    #             - 4867.87 * source_params.eta_pow2
    #             + 9129.45 * source_params.eta_pow3
    #         )
    #         * source_params.S_tot_hat_pow2
    #         + (
    #             -15.7439
    #             + 620.695 * source_params.eta
    #             - 4679.24 * source_params.eta_pow2
    #             + 9582.58 * source_params.eta_pow3
    #         )
    #         * source_params.S_tot_hat_pow3
    #         + 87.0832
    #         * source_params.delta_chi
    #         * source_params.delta
    #         * source_params.eta_pow2
    #     )
    #     self.int_colloc_values[4] = source_params.dt_psi4_to_strain + (
    #         3108.38
    #         + 3722.46 * source_params.eta
    #         - 119588.0 * source_params.eta_pow2
    #         + 1.92148e6 * source_params.eta_pow3
    #         - 1.69796e7 * source_params.eta_pow4
    #         + 8.39194e7 * source_params.eta_pow5
    #         - 2.17143e8 * source_params.eta_pow6
    #         + 2.2829700000000003e8 * source_params.eta_pow7
    #         + (118.319 - 529.854 * source_params.eta)
    #         * source_params.eta
    #         * source_params.S_tot_hat
    #         + (21.0314 - 240.648 * source_params.eta + 516.333 * source_params.eta_pow2)
    #         * source_params.S_tot_hat_pow2
    #         + (20.3384 - 356.241 * source_params.eta + 999.417 * source_params.eta_pow2)
    #         * source_params.S_tot_hat_pow3
    #         + 97.1364
    #         * source_params.delta_chi
    #         * source_params.delta
    #         * source_params.eta_pow2
    #     )
    #     self.int_colloc_values[5] = source_params.dt_psi4_to_strain + (
    #         3096.03
    #         + 986.752 * source_params.eta
    #         - 20371.1 * source_params.eta_pow2
    #         + 220332.0 * source_params.eta_pow3
    #         - 1.31523e6 * source_params.eta_pow4
    #         + 4.29193e6 * source_params.eta_pow5
    #         - 6.01179e6 * source_params.eta_pow6
    #         + (
    #             -9.96292
    #             - 118.526 * source_params.eta
    #             + 2255.76 * source_params.eta_pow2
    #             - 6758.52 * source_params.eta_pow3
    #         )
    #         * source_params.S_tot_hat
    #         + (
    #             -14.4869
    #             + 370.039 * source_params.eta
    #             - 3605.8 * source_params.eta_pow2
    #             + 9129.45 * source_params.eta_pow3
    #         )
    #         * source_params.S_tot_hat_pow2
    #         + (
    #             17.0209
    #             + 70.1931 * source_params.eta
    #             - 3070.08 * source_params.eta_pow2
    #             + 9582.58 * source_params.eta_pow3
    #         )
    #         * source_params.S_tot_hat_pow3
    #         + 23.0759
    #         * source_params.delta_chi
    #         * source_params.delta
    #         * source_params.eta_pow2
    #     )

    # def _set_intermediate_coefficients_case_1(self, source_params: ti.template()):
    #     """
    #     using collocation points: 0, 1, 3, 4, 5
    #     for situation with (eta < etaEMR) or (emm == ell and STotR >= 0.8) or (modeTag == 33 and STotR < 0)
    #     """
    #     pass

    # def _set_intermediate_coefficients_case_2(self, source_params: ti.template()):
    #     """
    #     using collocation points: 0, 1, 2, 4, 5
    #     for situation with (STotR >= 0.8) and (modeTag == 21)"""
    #     pass

    # def _set_intermediate_coefficients_case_3(self, source_params: ti.template()):
    #     """
    #     using collocation points: 0, 1, 2, 3, 5
    #     remaining parameter space
    #     """
    #     pass

    # def _set_intermediate_coefficients(self, source_params: ti.template()):
    #     # simplify the conditional structure in LALSimIMRPhenomXHM_internals.c l.2108 for modeTag=44
    #     if (source_params.eta < eta_EMR) or (source_params.S_tot_hat >= 0.8):
    #         self._set_intermediate_coefficients_case_1(source_params)
    #     else:
    #         self._set_intermediate_coefficients_case_3(source_params)

    # # // choose collocation points according to spin/mass ratio
    # #     // current catalogue of simulations include some cases that create unphysical effects in the fits -> we need to use different subset of collocation points according to the parameters (we have to pick 5 out of 6 available fits)
    # #     /* cpoints_indices is an array of integers labelling the collocation points chosen in each case, e.g.
    # #      cpoints_indices={0,1,3,4,5} would mean that we are discarding the 3rd collocation points in the reconstructio */

    # #     int cpoints_indices[nCollocationPts_inter];
    # #     cpoints_indices[0]=0;
    # #     cpoints_indices[1]=1;
    # #     cpoints_indices[4]=5;

    # # if((pWF22->eta<pWFHM->etaEMR)||(emm==ell&&pWF22->STotR>=0.8)||(pWFHM->modeTag==33&&pWF22->STotR<0))
    # # {
    # #     cpoints_indices[2]=3;
    # #     cpoints_indices[3]=4;
    # # }
    # # else if(pWF22->STotR>=0.8&&pWFHM->modeTag==21){

    # #     cpoints_indices[2]=2;
    # #     cpoints_indices[3]=4;
    # # }

    # # else{
    # #     cpoints_indices[2]=2;
    # #     cpoints_indices[3]=3;
    # # }

    @ti.func
    def update_phase_coefficients(
        self,
        phase_coefficients_22: ti.template(),
        source_params: ti.template(),
    ):
        self._set_ins_rescaling_coefficients(4.0, phase_coefficients_22)

    # @ti.func
    # def phase_inspiral_ansatz(self):
    #     pass

    # @ti.func
    # def phase_intermediate_ansatz(self):
    #     pass

    # @ti.func
    # def phase_merge_ringdown_ansatz(self):
    #     pass


@ti.data_oriented
class IMRPhenomXHM(BaseWaveform):
    """
    only default configuration is implemented, except the multibanding threshold which is not implemented now.

    The referenced lalsutie version is the commit 9a106f0966b3683a25fbd7d5b22a6d7bea98b4b3

    The waveform computed here corresponds to the waveform given by lalsimulation with the configuration of
    /* IMRPhenomXHM Parameters */
    DEFINE_ISDEFAULT_FUNC(PhenomXHMReleaseVersion, INT4, "PhenomXHMReleaseVersion", 122022)
    DEFINE_ISDEFAULT_FUNC(PhenomXHMInspiralPhaseVersion, INT4, "InsPhaseHMVersion", 122019)
    DEFINE_ISDEFAULT_FUNC(PhenomXHMIntermediatePhaseVersion, INT4, "IntPhaseHMVersion", 122019)
    DEFINE_ISDEFAULT_FUNC(PhenomXHMRingdownPhaseVersion, INT4, "RDPhaseHMVersion", 122019)
    DEFINE_ISDEFAULT_FUNC(PhenomXHMInspiralAmpVersion, INT4, "InsAmpHMVersion", 3)
    DEFINE_ISDEFAULT_FUNC(PhenomXHMIntermediateAmpVersion, INT4, "IntAmpHMVersion", 2)
    DEFINE_ISDEFAULT_FUNC(PhenomXHMRingdownAmpVersion, INT4, "RDAmpHMVersion", 0)
    DEFINE_ISDEFAULT_FUNC(PhenomXHMInspiralAmpFitsVersion, INT4, "InsAmpFitsVersion", 122018)
    DEFINE_ISDEFAULT_FUNC(PhenomXHMIntermediateAmpFitsVersion, INT4, "IntAmpFitsVersion", 122018)
    DEFINE_ISDEFAULT_FUNC(PhenomXHMRingdownAmpFitsVersion, INT4, "RDAmpFitsVersion", 122018)
    DEFINE_ISDEFAULT_FUNC(PhenomXHMInspiralAmpFreqsVersion, INT4, "InsAmpFreqsVersion", 122018)
    DEFINE_ISDEFAULT_FUNC(PhenomXHMIntermediateAmpFreqsVersion, INT4, "IntAmpFreqsVersion", 122018)
    DEFINE_ISDEFAULT_FUNC(PhenomXHMRingdownAmpFreqsVersion, INT4, "RDAmpFreqsVersion", 122018)
    DEFINE_ISDEFAULT_FUNC(PhenomXHMPhaseRef21, REAL8, "PhaseRef21", 0.)
    DEFINE_ISDEFAULT_FUNC(PhenomXHMThresholdMband, REAL8, "ThresholdMband", 0.001)
    DEFINE_ISDEFAULT_FUNC(PhenomXHMAmpInterpolMB, INT4, "AmpInterpol", 1)
    DEFINE_ISDEFAULT_FUNC(DOmega220, REAL8, "domega220", 0)
    DEFINE_ISDEFAULT_FUNC(DTau220, REAL8, "dtau220", 0)
    DEFINE_ISDEFAULT_FUNC(DOmega210, REAL8, "domega210", 0)
    DEFINE_ISDEFAULT_FUNC(DTau210, REAL8, "dtau210", 0)
    DEFINE_ISDEFAULT_FUNC(DOmega330, REAL8, "domega330", 0)
    DEFINE_ISDEFAULT_FUNC(DTau330, REAL8, "dtau330", 0)
    DEFINE_ISDEFAULT_FUNC(DOmega440, REAL8, "domega440", 0)
    DEFINE_ISDEFAULT_FUNC(DTau440, REAL8, "dtau440", 0)
    DEFINE_ISDEFAULT_FUNC(DOmega550, REAL8, "domega550", 0)
    DEFINE_ISDEFAULT_FUNC(DTau550, REAL8, "dtau550", 0)
    """

    def __init__(
        self,
        frequencies: ti.ScalarField,
        waveform_container: Optional[ti.StructField] = None,
        reference_frequency: Optional[float] = None,
        returned_form: str = "polarizations",
        include_tf: bool = True,
        modes: list[str] = ["22", "21", "33", "32", "44"],
        combine_modes: bool = False,
        parameter_check: bool = False,
    ) -> None:
        """ """

    def update_waveform(self, parameters: dict[str, float]):
        """
        necessary preparation which need to be finished in python scope for waveform computation
        (this function may be awkward, since no interpolation function in taichi-lang)
        """
        # TODO: passed-in parameter conversion
        self._update_waveform_kernel(
            parameters["mass_1"],
            parameters["mass_2"],
            parameters["chi_1"],
            parameters["chi_2"],
            parameters["luminosity_distance"],
            parameters["inclination"],
            parameters["reference_phase"],
            parameters["coalescence_time"],
        )

    @ti.kernel
    def _update_waveform_kernel(
        self,
        mass_1: ti.f64,
        mass_2: ti.f64,
        chi_1: ti.f64,
        chi_2: ti.f64,
        luminosity_distance: ti.f64,
        inclination: ti.f64,
        reference_phase: ti.f64,
        coalescence_time: ti.f64,
    ):
        pass

    @ti.func
    def _parameter_check(self):
        # TODO
        pass
