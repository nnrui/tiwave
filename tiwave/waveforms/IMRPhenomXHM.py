# TODO:
# - special case of q=1
# - all idx begin from 0
# - only store coefficients used for wavefrom construction
# - check unwrap operation in PhaseCoefficientsMode32._set_intermediate_coefficients
# - fix high numerical error for intermediate phase of mode 32

import warnings
from typing import Optional

import taichi as ti
import taichi.math as tm

from ..constants import *
from ..utils import ComplexNumber, gauss_elimination, UsefulPowers, sub_struct_from
from .common import PostNewtonianCoefficients
from .base_waveform import BaseWaveform
from .IMRPhenomXAS import IMRPhenomXAS
from .IMRPhenomXAS import SourceParameters as SourceParametersMode22
from .IMRPhenomXAS import PhaseCoefficients as PhaseCoefficientsMode22
from .IMRPhenomXAS import AmplitudeCoefficients as AmplitudeCoefficientsMode22


eta_EMR = 0.05  # Limit for extreme mass ratio

useful_powers_pi = UsefulPowers()
useful_powers_pi.update(PI)
useful_powers_2pi = UsefulPowers()
useful_powers_2pi.update(2.0 * PI)
useful_powers_2pi_over_3 = UsefulPowers()
useful_powers_2pi_over_3.update(2.0 * PI / 3.0)
useful_powers_pi_over_2 = UsefulPowers()
useful_powers_pi_over_2.update(0.5 * PI)

QNM_frequencies_struct = ti.types.struct(
    f_ring=ti.f64,
    f_damp=ti.f64,
    # cache for repeatedly using
    f_damp_pow2=ti.f64,
)


@sub_struct_from(PostNewtonianCoefficients)
class PostNewtonianCoefficientsHighModesBase:

    # amplitude coeffients
    A_0: ComplexNumber
    A_1: ComplexNumber
    A_2: ComplexNumber
    A_3: ComplexNumber
    A_4: ComplexNumber
    A_5: ComplexNumber
    A_6: ComplexNumber

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

    @ti.func
    def PN_amplitude(self, powers_of_Mf: ti.template()) -> ti.f64:
        return tm.length(
            self.A_0
            + self.A_1 * powers_of_Mf.third
            + self.A_2 * powers_of_Mf.two_thirds
            + self.A_3 * powers_of_Mf.one
            + self.A_4 * powers_of_Mf.four_thirds
            + self.A_5 * powers_of_Mf.five_thirds
            + self.A_6 * powers_of_Mf.two
        )

    @ti.func
    def PN_d_amplitude(self, powers_of_Mf: ti.template()) -> ti.f64:
        """
        Note the derivative need to incorporate the absolute value function.
        """
        return tm.cmul(
            (
                1.0 / 3.0 * self.A_1 / powers_of_Mf.two_thirds
                + 2.0 / 3.0 * self.A_2 / powers_of_Mf.third
                + self.A_3
                + 4.0 / 3.0 * self.A_4 * powers_of_Mf.third
                + 5.0 / 3.0 * self.A_5 * powers_of_Mf.two_thirds
                + 2.0 * self.A_6 * powers_of_Mf.one
            ),
            tm.cconj(
                self.A_0
                + self.A_1 * powers_of_Mf.third
                + self.A_2 * powers_of_Mf.two_thirds
                + self.A_3 * powers_of_Mf.one
                + self.A_4 * powers_of_Mf.four_thirds
                + self.A_5 * powers_of_Mf.five_thirds
                + self.A_6 * powers_of_Mf.two
            ),
        )[0] / self.PN_amplitude(powers_of_Mf)


@ti.dataclass
class AmplitudeCoefficientsHighModesBase:
    # Inspiral
    rho_1: ti.f64
    rho_2: ti.f64
    rho_3: ti.f64
    # Intermediate (note the intermediate in PhenomXHMReleaseVersion 122022 is totally different from that presented in the paper)
    int_ansatz_coeffs: ti.types.vector(8, ti.f64)  # for modes with no-mixing
    # Merge-ringdown
    gamma_1: ti.f64  # scaling factor of Lorenrzian, a_R * f_damp * sigma, (in lalsim: RDCoefficient[0] * f_damp) # fmt: skip
    gamma_2: ti.f64  # decay rate of exponential, lambda / (f_damp * sigma), (in lalsim: RDCoefficient[1] / (RDCoefficient[2] * f_damp) ) # fmt: skip
    gamma_3: ti.f64  # width factor, (f_damp * sigma)^2, (in lalsim: (RDCoefficient[2] * f_damp)^2) # fmt: skip
    falloff_gamma_1: ti.f64  # scaling factor in fall-off range, (in lalsim: RDCoefficient[3]) # fmt: skip
    falloff_gamma_2: ti.f64  # decay rate in fall-off range, (in lalsim: RDCoefficient[4]) # fmt: skip
    # joint frequencies
    ins_f_end: ti.f64
    int_f_end: ti.f64
    MRD_f_falloff: ti.f64
    # common factor
    common_factor: ti.f64  # sqrt(2.0/3.0/pi^(1/3)*eta) * (2/m)^(-7/6)

    """
    In release 122022: 

    InspiralAmpFreqsVersion     = 122022
    IntermediateAmpFreqsVersion = 0
    RingdownAmpFreqsVersion     = 122022

    InspiralAmpFitsVersion      = 122022
    IntermediateAmpFitsVersion  = 122022
    RingdownAmpFitsVersion      = 122022

    InspiralAmpVersion          = 123
    IntermediateAmpVersion      = 211112 (110102 for mode 21)
    RingdownAmpVersion          = 2 (1 for mode 32)
    """

    @ti.func
    def _fit_ins_f_end_EMR(self, m: ti.f64, source_params: ti.template()) -> ti.f64:
        """The end frequency of inspiral amplitude for extreme mass ratio."""
        return (
            m
            * 1.25
            * (
                0.011671068725758493
                - 0.0000858396080377194 * source_params.chi_1
                + 0.000316707064291237 * source_params.chi_1_pow2
            )
            * (0.8447212540381764 + 6.2873167352395125 * source_params.eta)
            / (1.2857082764038923 - 0.9977728883419751 * source_params.chi_1)
        )

    @ti.func
    def _get_ins_f_end(
        self, m: ti.f64, f_MECO_lm: ti.f64, source_params: ti.template()
    ) -> ti.f64:
        f_end = 0.0
        if source_params.eta < 0.04535147392290249:  # for extreme mass ratios (q>20)
            # using a smoothing function to perform a smooth transition from fit_ins_f_end_EMR
            # to f_MECO, avoiding abrupt changes between two frequencies.
            # eta >> transition, smooth_weight -> 1, ins_f_end -> f_MECO
            # eta << transition, smooth_weight -> 0, ins_f_end -> ins_f_end_EMR
            transition = 0.0192234  # q=50
            sharpness = 0.004
            smooth_weight = 0.5 + 0.5 * tm.tanh(
                (source_params.eta - transition) / sharpness
            )
            f_end = (
                (1 - smooth_weight) * self._fit_ins_f_end_EMR(m, source_params)
            ) + smooth_weight * f_MECO_lm
        else:  # for comparable mass ratios
            f_end = f_MECO_lm

        return f_end

    @ti.func
    def _set_joint_frequencies(
        self,
        m: ti.f64,
        f_MECO_lm: ti.f64,
        QNM_freqs_lm: ti.template(),
        source_params: ti.template(),
    ):
        """
        Used for modes with no-mixing
        """
        # joint between inspiral and intermediate
        self.ins_f_end = self._get_ins_f_end(m, f_MECO_lm, source_params)
        # joint between intermediate and merge-ringdown
        self.int_f_end = QNM_freqs_lm.f_ring - QNM_freqs_lm.f_damp
        # transition in merge-ringdown, from  Lorentzian with falloff to entire exponential falloff
        self.MRD_f_falloff = QNM_freqs_lm.f_ring + 2.0 * QNM_freqs_lm.f_damp

    @ti.func
    def _get_ins_colloc_points(self) -> ti.types.vector(3, dtype=ti.f64):
        return ti.Vector(
            [0.5 * self.ins_f_end, 0.75 * self.ins_f_end, self.ins_f_end],
            dt=ti.f64,
        )

    @ti.func
    def _get_int_colloc_points(self) -> ti.types.vector(6, dtype=ti.f64):
        int_f_space = (self.int_f_end - self.ins_f_end) / 5.0
        return ti.Vector(
            [
                self.ins_f_end,
                self.ins_f_end + int_f_space,
                self.ins_f_end + 2.0 * int_f_space,
                self.ins_f_end + 3.0 * int_f_space,
                self.ins_f_end + 4.0 * int_f_space,
                self.int_f_end,
            ],
            dt=ti.f64,
        )

    @ti.func
    def _set_inspiral_coefficients(
        self,
        pn_coefficients_lm: ti.template(),
        source_params: ti.template(),
    ):
        """ """
        ins_colloc_points = self._get_ins_colloc_points()
        powers_ins_f0 = UsefulPowers()
        powers_ins_f0.update(ins_colloc_points[0])
        powers_ins_f1 = UsefulPowers()
        powers_ins_f1.update(ins_colloc_points[1])
        powers_ins_f2 = UsefulPowers()
        powers_ins_f2.update(ins_colloc_points[2])
        # the pseudeo-PN coeffs obtained here are with the denominator of f_lm^Ins powers
        ins_colloc_values = ti.Vector([0.0] * 3, dt=ti.f64)
        ins_colloc_values[0] = (
            self._ins_fit_v0(source_params)
            * powers_ins_f0.seven_sixths
            / self.common_factor
        ) - pn_coefficients_lm.PN_amplitude(powers_ins_f0)
        ins_colloc_values[1] = (
            self._ins_fit_v1(source_params)
            * powers_ins_f1.seven_sixths
            / self.common_factor
        ) - pn_coefficients_lm.PN_amplitude(powers_ins_f1)
        ins_colloc_values[2] = (
            self._ins_fit_v2(source_params)
            * powers_ins_f2.seven_sixths
            / self.common_factor
        ) - pn_coefficients_lm.PN_amplitude(powers_ins_f2)
        Ab_ins = ti.Matrix(
            [
                [
                    powers_ins_f0.seven_thirds,
                    powers_ins_f0.eight_thirds,
                    powers_ins_f0.three,
                    ins_colloc_values[0],
                ],
                [
                    powers_ins_f1.seven_thirds,
                    powers_ins_f1.eight_thirds,
                    powers_ins_f1.three,
                    ins_colloc_values[1],
                ],
                [
                    powers_ins_f2.seven_thirds,
                    powers_ins_f2.eight_thirds,
                    powers_ins_f2.three,
                    ins_colloc_values[2],
                ],
            ],
            dt=ti.f64,
        )
        self.rho_1, self.rho_2, self.rho_3 = gauss_elimination(Ab_ins)

    @ti.func
    def _set_merge_ringdown_coefficients(
        self,
        QNM_freqs_lm: ti.template(),
        source_params: ti.template(),
    ):
        """
        We use a different notation from lalsim, where
        gamma_1 = a_R * f_damp * sigma = RDCoefficient[0] * f_damp,
        gamma_2 = lambda / (f_damp * sigma) = RDCoefficient[1] / (RDCoefficient[2] * f_damp)
        gamma_3 = (f_damp * sigma)^2 = (RDCoefficient[2] * f_damp)^2
        The ansatz is given by
        gamma_1 / ((f - fring)^2 + gamma_3) * exp(-gamma_2 * (f - fring))
        """
        v0 = self._MRD_fit_v0(source_params)
        v1 = self._MRD_fit_v1(source_params)
        v2 = self._MRD_fit_v2(source_params)

        if v2 >= v1**2 / v0:
            v2 = 0.5 * v1**2 / v0
        if v2 > v1:
            v2 = 0.5 * v1
        if (v0 < v1) and (v2 > v0):
            v2 = v0

        self.gamma_2 = 0.5 * tm.log(v0 / v2) / QNM_freqs_lm.f_damp
        self.gamma_3 = (
            QNM_freqs_lm.f_damp_pow2
            * tm.sqrt(v0 * v2)
            / (v1 - tm.sqrt(v0 * v2))  # v0*v2 <= v1^2
        )
        self.gamma_1 = v1 * self.gamma_3

        self.falloff_gamma_1 = self._merge_ringdown_amplitude_Lorentzian(
            self.MRD_f_falloff - QNM_freqs_lm.f_ring
        )
        self.falloff_gamma_2 = (
            -self._merge_ringdown_d_amplitude_Lorentzian(
                self.MRD_f_falloff - QNM_freqs_lm.f_ring
            )
            / self.falloff_gamma_1
        )

    @ti.func
    def _set_intermediate_coefficients(
        self,
        QNM_freqs_lm: ti.template(),
        pn_coefficients_lm: ti.template(),
        source_params: ti.template(),
    ):
        """
        Require inspiral and merge-ringdown amplitude to set boundaries, can only be
        called after updating inspiral and merge-ringdown coefficients.

        Corresponding to IntermediateAmpVersion=211112, not used for mode 21.
        colloc-points 6;
        colloc-values 8 (values at 6 colloc-points and 2 derivatives at boundaries);
        ansatz-coeffs 8;
        augmented matrix 9x8
        """
        int_colloc_points = self._get_int_colloc_points()
        powers_int_f0 = UsefulPowers()
        powers_int_f0.update(int_colloc_points[0])

        int_colloc_values = ti.Vector([0.0] * 8, dt=ti.f64)
        # left boundary
        int_colloc_values[0] = self._inspiral_amplitude(
            pn_coefficients_lm, powers_int_f0
        )
        # fit values at collocation points
        int_colloc_values[1] = self._int_fit_v1(source_params)
        int_colloc_values[2] = self._int_fit_v2(source_params)
        int_colloc_values[3] = self._int_fit_v3(source_params)
        int_colloc_values[4] = self._int_fit_v4(source_params)
        # right boundary
        int_colloc_values[5] = self._merge_ringdown_amplitude_Lorentzian(
            int_colloc_points[5] - QNM_freqs_lm.f_ring
        )  # always have int_f_end < MRE_f_falloff
        # derivative at the left boundary
        int_colloc_values[6] = self._inspiral_d_amplitude(
            pn_coefficients_lm, powers_int_f0
        )
        # derivative at the right boundary
        int_colloc_values[7] = self._merge_ringdown_d_amplitude_Lorentzian(
            int_colloc_points[5] - QNM_freqs_lm.f_ring
        )

        # set the augmented matrix
        Ab = ti.Matrix([[0.0] * 9 for _ in range(8)], dt=ti.f64)
        row_idx = 0
        for i in ti.static(range(6)):
            # set the value at the collocation point
            Ab[row_idx, 8] = int_colloc_values[row_idx]
            # set the coefficient matrix of frequency powers
            # [1, fi, fi^2, fi^3, fi^4, fi^5, fi^6, fi^7] * fi^(-7/6)
            fi = int_colloc_points[i]
            fpower = fi ** (-7.0 / 6.0)
            for j in ti.static(range(8)):
                Ab[row_idx, j] = fpower
                fpower *= fi
            # next row
            row_idx += 1
        # for the derivatives at 2 boundaries
        for i in ti.static([0, 5]):
            Ab[row_idx, 8] = int_colloc_values[row_idx]
            # set the coefficient matrix of frequency powers for derivative
            # [ (-7/6)fi_-1,  (-7/6+1),     (-7/6+2)fi,   (-7/6+3)fi^2,
            #   (-7/6+4)fi^3, (-7/6+5)fi^4, (-7/6+6)fi^5, (-7/6+7)fi^6 ] * fi^(-7/6)
            fi = int_colloc_points[i]
            fpower = fi ** (-13.0 / 6.0)
            for j in ti.static(range(8)):
                Ab[row_idx, j] = (-7.0 / 6.0 + j) * fpower
                fpower *= fi
            # next row
            row_idx += 1

        self.int_ansatz_coeffs = gauss_elimination(Ab)

    @ti.func
    def _inspiral_amplitude(
        self,
        pn_coefficients_lm: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> ti.f64:
        """ """
        return (
            self.common_factor
            / powers_of_Mf.seven_sixths
            * (
                pn_coefficients_lm.PN_amplitude(powers_of_Mf)
                + self.rho_1 * powers_of_Mf.seven_thirds
                + self.rho_2 * powers_of_Mf.eight_thirds
                + self.rho_3 * powers_of_Mf.three
            )
        )

    @ti.func
    def _inspiral_d_amplitude(
        self,
        pn_coefficients_lm: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> ti.f64:
        """ """
        return (
            self.common_factor
            / powers_of_Mf.seven_sixths
            * (
                pn_coefficients_lm.PN_d_amplitude(powers_of_Mf)
                + 7.0 / 3.0 * self.rho_1 * powers_of_Mf.four_thirds
                + 8.0 / 3.0 * self.rho_2 * powers_of_Mf.five_thirds
                + 3.0 * self.rho_3 * powers_of_Mf.two
            )
        ) - (
            7.0
            / 6.0
            / powers_of_Mf.one
            * self._inspiral_amplitude(pn_coefficients_lm, powers_of_Mf)
        )

    @ti.func
    def _intermediate_amplitude(self, powers_of_Mf: ti.template()) -> ti.f64:
        """ """
        fpower = 1.0 / powers_of_Mf.seven_sixths
        ret = 0.0
        for i in ti.static(range(self.int_ansatz_coeffs.n)):
            ret += self.int_ansatz_coeffs[i] * fpower
            fpower *= powers_of_Mf.ones
        return ret

    @ti.func
    def _merge_ringdown_amplitude_Lorentzian(self, f_minus_fring: ti.f64) -> ti.f64:
        """
        Lorentzian with exponential falloff, f < MRD_f_falloff
        """
        return (
            self.gamma_1
            / (f_minus_fring**2 + self.gamma_3)
            * tm.exp(-self.gamma_2 * f_minus_fring)
        )

    @ti.func
    def _merge_ringdown_d_amplitude_Lorentzian(self, f_minus_fring: ti.f64) -> ti.f64:
        divisor = f_minus_fring**2 + self.gamma_3
        return (
            -self.gamma_1
            / divisor
            * tm.exp(-self.gamma_2 * f_minus_fring)
            * (self.gamma_2 + 2 * f_minus_fring / divisor)
        )

    @ti.func
    def _merge_ringdown_amplitude_falloff(self, Mf: ti.f64) -> ti.f64:
        """
        entire exponential falloff, f > MRD_f_falloff
        """
        return self.falloff_gamma_1 * tm.exp(
            -self.falloff_gamma_2 * (Mf - self.MRD_f_falloff)
        )

    @ti.func
    def compute_amplitude(
        self,
        QNM_freqs_lm: ti.template(),
        pn_coefficients_lm: ti.template(),
        source_params: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> ti.f64:
        amplitude = 0.0
        f_minus_fring = powers_of_Mf.one - QNM_freqs_lm.f_ring

        if powers_of_Mf.one < self.ins_f_end:
            amplitude = self._inspiral_amplitude(pn_coefficients_lm, powers_of_Mf)
        elif powers_of_Mf.one > self.MRD_f_falloff:
            amplitude = self._merge_ringdown_amplitude_falloff(powers_of_Mf.one)
        elif powers_of_Mf.one > self.int_f_end:
            amplitude = self._merge_ringdown_amplitude_Lorentzian(f_minus_fring)
        else:
            amplitude = self._intermediate_amplitude(powers_of_Mf)

        return source_params.dimension_factor * amplitude


@ti.dataclass
class PhaseCoefficientsHighModesBase:
    # Inspiral (only 4 pseudo-PN coefficients in the default 104 inspiral configuration of XAS model)
    sigma_1: ti.f64
    sigma_2: ti.f64
    sigma_3: ti.f64
    sigma_4: ti.f64
    Lambda_lm: ti.f64  # corrections for the complex PN amplitudes, eq 4.9
    # Intermediate
    c_0: ti.f64
    c_1: ti.f64
    c_2: ti.f64
    c_3: ti.f64  # only used for mode 32, set to 0.0 for mode 21, 33, 44
    c_4: ti.f64
    c_L: ti.f64
    _int_colloc_points: ti.types.vector(6, dtype=ti.f64)
    _int_colloc_values: ti.types.vector(6, dtype=ti.f64)
    # Merge-ringdown
    alpha_2: ti.f64
    alpha_L: ti.f64
    # joint frequencies
    ins_f_end: ti.f64
    int_f_end: ti.f64
    # continuity condition coefficients
    ins_C0: ti.f64
    ins_C1: ti.f64
    MRD_C0: ti.f64
    MRD_C1: ti.f64
    # constant for aligning each mode under the choice of tetrad convention
    delta_phi_lm: ti.f64
    # cache powers of some collocation points for converience
    _useful_powers: ti.types.struct(
        ins_f_end=UsefulPowers,
        int_f_end=UsefulPowers,
    )

    """
    In release 122022: 

    InspiralPhaseVersion = 122019
    IntermediatePhaseVersion = 122019
    RingdownPhaseVersion = 122019
    """

    @ti.func
    def _set_joint_frequencies_no_mixing(
        self, f_MECO_lm: ti.f64, QNM_freqs_lm: ti.template()
    ):
        self.ins_f_end = f_MECO_lm
        self.int_f_end = QNM_freqs_lm.f_ring - QNM_freqs_lm.f_damp
        self._useful_powers.ins_f_end.update(self.ins_f_end)
        self._useful_powers.int_f_end.update(self.int_f_end)

    @ti.func
    def _set_int_colloc_points_no_mixing(self, f_ring_lm: ti.f64, eta: ti.f64):
        # shifting forward the frequency of the first collocation points for small eta
        beta = 1.0 + 0.001 * (0.25 / eta - 1.0)

        self._int_colloc_points[0] = beta * self.ins_f_end
        self._int_colloc_points[1] = (
            tm.sqrt(3.0) * (self._int_colloc_points[0] - f_ring_lm)
            + 2.0 * (self._int_colloc_points[0] + f_ring_lm)
        ) / 4.0
        self._int_colloc_points[2] = (
            3.0 * self._int_colloc_points[0] + f_ring_lm
        ) / 4.0
        self._int_colloc_points[3] = (self._int_colloc_points[0] + f_ring_lm) / 2.0
        self._int_colloc_points[4] = (
            self._int_colloc_points[0] + 3.0 * f_ring_lm
        ) / 4.0
        self._int_colloc_points[5] = (
            self._int_colloc_points[0] + 7.0 * f_ring_lm
        ) / 8.0

    @ti.func
    def _get_int_augmented_matrix_no_mixing(
        self, QNM_freqs_lm: ti.template(), idx: ti.template()
    ) -> ti.types.matrix(5, 6, ti.f64):
        Ab = ti.Matrix([[0.0] * 6 for _ in range(5)], dt=ti.f64)
        for i in ti.static(range(5)):
            row = [
                1.0,
                self._int_colloc_points[idx[i]] ** (-1),
                self._int_colloc_points[idx[i]] ** (-2),
                self._int_colloc_points[idx[i]] ** (-4),
                QNM_freqs_lm.f_damp
                / (
                    QNM_freqs_lm.f_damp_pow2
                    + (self._int_colloc_points[idx[i]] - QNM_freqs_lm.f_ring) ** 2
                ),
                self._int_colloc_values[idx[i]],
            ]
            for j in ti.static(range(6)):
                Ab[i, j] = row[j]
        return Ab

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
        source_params: ti.template(),
    ):
        """
        Used for no mixing modes

        for 33, 44 mode, w_lm = 2;
        for 21 mode, w_lm = 1/3;
        """
        self.alpha_L = self._fit_alpha_L(source_params)
        self.alpha_2 = w_lm * self._fit_alpha_2(source_params)

    @ti.func
    def _fit_alpha_2(self, source_params: ti.template()) -> ti.f64:
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
                0.9088578269496244 * source_params.eta_pow2 * source_params.eta_sqrt
                + 15.619592332008951
                * source_params.delta_chi
                * source_params.eta_pow3
                * source_params.eta_sqrt
            )
            * source_params.delta
        )

    @ti.func
    def _fit_alpha_L(self, source_params: ti.template()) -> ti.f64:
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
    def _set_connection_coefficients(
        self,
        QNM_freqs_lm: ti.template(),
        pn_coefficients_lm: ti.template(),
    ):
        self.ins_C1 = self._intermediate_d_phase(
            QNM_freqs_lm, self._useful_powers.ins_f_end
        ) - self._inspiral_d_phase(pn_coefficients_lm, self._useful_powers.ins_f_end)
        # Note we have dropped the constant of phi_5, ins_C0 is different with CINSP in
        # lalsimulation. ins_C0 (tiwave) = CINSP (lalsim) - phi_5
        self.ins_C0 = (
            self._intermediate_phase(QNM_freqs_lm, self._useful_powers.ins_f_end)
            - self._inspiral_phase(pn_coefficients_lm, self._useful_powers.ins_f_end)
            - self.ins_C1 * self.ins_f_end
        )

        self.MRD_C1 = self._intermediate_d_phase(
            QNM_freqs_lm, self._useful_powers.int_f_end
        ) - self._merge_ringdown_d_phase(QNM_freqs_lm, self._useful_powers.int_f_end)
        self.MRD_C0 = (
            self._intermediate_phase(QNM_freqs_lm, self._useful_powers.int_f_end)
            - self._merge_ringdown_phase(QNM_freqs_lm, self._useful_powers.int_f_end)
            - self.MRD_C1 * self.int_f_end
        )

    @ti.func
    def _set_delta_phi_lm(
        self,
        m: ti.f64,
        f_MECO_lm: ti.f64,
        pn_coefficients_22: ti.template(),
        pn_coefficients_lm: ti.template(),
        phase_coefficients_22: ti.template(),
        source_params: ti.template(),
    ):
        """
        Setting delta_phi_lm for aligning different modes according to Eq. 4.13, call
        after the parameters of continuity condition are updated.
        """
        f_align = 0.0
        powers_of_falign = UsefulPowers()
        f_align_22 = 0.0
        powers_of_falign22 = UsefulPowers()
        if source_params.eta > eta_EMR:
            f_align = 0.6 * f_MECO_lm
            f_align_22 = 0.6 * source_params.f_MECO
        else:
            f_align = f_MECO_lm
            f_align_22 = source_params.f_MECO
        powers_of_falign.update(f_align)
        powers_of_falign22.update(f_align_22)

        # note there is no time-shift term for high modes here.
        delta_phi_lm = (
            0.5
            * m
            * phase_coefficients_22.compute_phase(
                pn_coefficients_22, source_params, powers_of_falign22
            )
            - 3.0 / 4.0 * PI * (1 - 0.5 * m)
            - (
                self._inspiral_phase(pn_coefficients_lm, powers_of_falign)
                + self.ins_C1 * f_align
                + self.ins_C0
            )
        )
        self.delta_phi_lm = tm.mod(delta_phi_lm, 2.0 * PI)

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
            pn_coefficients_lm.PN_d_phase(powers_of_Mf)
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
        self,
        QNM_freqs_lm: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> ti.f64:
        """for no mixing modes"""
        return -self.alpha_2 / powers_of_Mf.one + self.alpha_L * tm.atan2(
            (powers_of_Mf.one - QNM_freqs_lm.f_ring), QNM_freqs_lm.f_damp
        )

    @ti.func
    def _merge_ringdown_d_phase(
        self,
        QNM_freqs_lm: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> ti.f64:
        """for no mixing modes"""
        return self.alpha_2 / powers_of_Mf.two + self.alpha_L * QNM_freqs_lm.f_damp / (
            QNM_freqs_lm.f_damp_pow2 + (powers_of_Mf.one - QNM_freqs_lm.f_ring) ** 2
        )

    @ti.func
    def compute_phase(
        self,
        QNM_freqs_lm: ti.template(),
        pn_coefficients_lm: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> ti.f64:
        # Note the time-shift for making the peak around t=0 has been incorporated in the construction of intermediate phase, here only the constants delta_phi_lm for aligning different modes are needed. And thus the continuity condition parameters needs to be added in the inspiral and merge-ringdown phase.
        phase = 0.0
        if powers_of_Mf.one < self.ins_f_end:
            phase = (
                self._inspiral_phase(pn_coefficients_lm, powers_of_Mf)
                + self.ins_C0
                + self.ins_C1 * powers_of_Mf.one
            )
        elif powers_of_Mf.one > self.int_f_end:
            phase(
                self._merge_ringdown_phase(QNM_freqs_lm, powers_of_Mf)
                + self.MRD_C0
                + self.MRD_C1 * powers_of_Mf.one
            )
        else:
            phase = self._intermediate_phase(QNM_freqs_lm, powers_of_Mf)
        return phase + self.delta_phi_lm

    @ti.func
    def compute_d_phase(
        self,
        QNM_freqs_lm: ti.template(),
        pn_coefficients_lm: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> ti.f64:
        d_phase = 0.0
        if powers_of_Mf.one < self.ins_f_end:
            d_phase = (
                self._inspiral_d_phase(pn_coefficients_lm, powers_of_Mf) + self.ins_C1
            )
        elif powers_of_Mf.one > self.int_f_end:
            d_phase = (
                self._merge_ringdown_d_phase(QNM_freqs_lm, powers_of_Mf) + self.MRD_C1
            )
        else:
            d_phase = self._intermediate_d_phase(QNM_freqs_lm, powers_of_Mf)
        return d_phase


@sub_struct_from(SourceParametersMode22)
class SourceParametersHighModes:
    eta_sqrt: ti.f64
    eta_pow7: ti.f64
    eta_pow8: ti.f64

    chi_1_pow2: ti.f64
    chi_2_pow2: ti.f64
    delta_chi_half: ti.f64
    delta_chi_half_pow2: ti.f64

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
    # TODO: f_ISCO_lm used??
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
        reference_frequency: ti.f64,
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
            reference_frequency,
        )
        self.eta_sqrt = tm.sqrt(self.eta)
        self.eta_pow7 = self.eta * self.eta_pow6
        self.eta_pow8 = self.eta * self.eta_pow7

        self.chi_1_pow2 = self.chi_1 * self.chi_1
        self.chi_2_pow2 = self.chi_2 * self.chi_2
        self.delta_chi_half = self.delta_chi * 0.5
        self.delta_chi_half_pow2 = self.delta_chi_half * self.delta_chi_half

        if ti.static("21" in high_modes):
            self._set_QNM_frequencies_21()
            self.QNM_freqs_lm["21"].f_damp_pow2 = self.QNM_freqs_lm["21"].f_damp ** 2
            self.f_MECO_lm["21"] = 0.5 * self.f_MECO
            self.f_ISCO_lm["21"] = 0.5 * self.f_ISCO
        if ti.static("33" in high_modes):
            self._set_QNM_frequencies_33()
            self.QNM_freqs_lm["33"].f_damp_pow2 = self.QNM_freqs_lm["33"].f_damp ** 2
            self.f_MECO_lm["33"] = 1.5 * self.f_MECO
            self.f_ISCO_lm["33"] = 1.5 * self.f_ISCO
        if ti.static("32" in high_modes):
            self._set_QNM_frequencies_32()
            self.QNM_freqs_lm["32"].f_damp_pow2 = self.QNM_freqs_lm["32"].f_damp ** 2
            self.f_MECO_lm["32"] = self.f_MECO
            self.f_ISCO_lm["32"] = self.f_ISCO
        if ti.static("44" in high_modes):
            self._set_QNM_frequencies_44()
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


@ti.dataclass
class MergeRingdownMode32:
    # for amplitude ansatz of spheroidal
    amp_aux_coeffs: ti.types.vector(4, dtype=ti.f64)
    gamma_1: ti.f64  # in lalsim: RDCoefficient[0] * f_damp
    gamma_2: ti.f64  # in lalsim: RDCoefficient[1] / (RDCoefficient[2] * f_damp)
    gamma_3: ti.f64  # in lalsim: (RDCoefficient[2] * f_damp)^2
    falloff_gamma_1: ti.f64  # in lalsim: RDCoefficient[3]
    falloff_gamma_2: ti.f64  # in lalsim: RDCoefficient[4]
    amp_f_falloff: ti.f64
    amp_f_aux: ti.f64
    # for phase ansatz of spheroidal
    alpha_0: ti.f64
    alpha_2: ti.f64
    alpha_4: ti.f64
    alpha_L: ti.f64
    dphi0: ti.f64
    phi0: ti.f64
    # mixing coefficients
    a_322: ComplexNumber
    a_332: ComplexNumber

    @ti.func
    def _spheroidal_amplitude_Lorentzian(self, f_minus_fring: ti.f64) -> ti.f64:
        """
        Lorentzian with exponential falloff, f < amp_f_falloff
        """
        return (
            self.gamma_1
            / (f_minus_fring**2 + self.gamma_3)
            * tm.exp(-self.gamma_2 * f_minus_fring)
        )

    @ti.func
    def _spheroidal_d_amplitude_Lorentzian(self, f_minus_fring: ti.f64) -> ti.f64:
        divisor = f_minus_fring**2 + self.gamma_3
        return (
            -self.gamma_1
            / divisor
            * tm.exp(-self.gamma_2 * f_minus_fring)
            * (self.gamma_2 + 2 * f_minus_fring / divisor)
        )

    @ti.func
    def _spheroidal_amplitude_falloff(self, Mf: ti.f64) -> ti.f64:
        """
        entire exponential falloff, f > amp_f_falloff
        """
        return self.falloff_gamma_1 * tm.exp(
            -self.falloff_gamma_2 * (Mf - self.amp_f_falloff)
        )

    @ti.func
    def _spheroidal_amplitude_powers(self, Mf: ti.f64) -> ti.f64:
        amp = 0.0
        fpower = 1.0
        for i in ti.static(range(4)):
            amp += fpower * self.amp_aux_coeffs[i]
            fpower *= Mf
        return amp

    @ti.func
    def _spheroidal_amplitude(self, f_ring_32: ti.f64, Mf: ti.f64) -> ti.f64:
        amp = 0.0
        if Mf < self.amp_f_aux:
            amp = self._spheroidal_amplitude_powers(Mf)
        elif Mf > self.amp_f_falloff:
            amp = self._spheroidal_amplitude_falloff(Mf)
        else:
            amp = self._spheroidal_amplitude_Lorentzian(Mf - f_ring_32)
        return amp

    @ti.func
    def _spheroidal_d_phase_core(
        self, QNM_freqs_32: ti.template(), powers_of_Mf: ti.template()
    ) -> ti.f64:
        return (
            self.alpha_0
            + self.alpha_2 / powers_of_Mf.two
            + self.alpha_4 / powers_of_Mf.four
            + self.alpha_L
            * QNM_freqs_32.f_damp
            / (QNM_freqs_32.f_damp_pow2 + (powers_of_Mf.one - QNM_freqs_32.f_ring) ** 2)
        )

    @ti.func
    def _spheroidal_d_phase(
        self, QNM_freqs_32: ti.template(), powers_of_Mf: ti.template()
    ) -> ti.f64:
        return self._spheroidal_d_phase_core(QNM_freqs_32, powers_of_Mf) + self.dphi0

    @ti.func
    def _spheroidal_phase_core(
        self, QNM_freqs_32: ti.template(), powers_of_Mf: ti.template()
    ) -> ti.f64:
        return (
            self.alpha_0 * powers_of_Mf.one
            - self.alpha_2 / powers_of_Mf.one
            - 1.0 / 3.0 * self.alpha_4 / powers_of_Mf.three
            + self.alpha_L
            * tm.atan2((powers_of_Mf.one - QNM_freqs_32.f_ring), QNM_freqs_32.f_damp)
        )

    @ti.func
    def _spheroidal_phase(
        self, QNM_freqs_32: ti.template(), powers_of_Mf: ti.template()
    ) -> ti.f64:
        return (
            self._spheroidal_phase_core(QNM_freqs_32, powers_of_Mf)
            + self.dphi0 * powers_of_Mf.one
            + self.phi0
        )

    @ti.func
    def _spheroidal_h32(
        self, QNM_freqs_32: ti.template(), powers_of_Mf: ti.template()
    ) -> ComplexNumber:
        amp = self._spheroidal_amplitude(QNM_freqs_32.f_ring, powers_of_Mf.one)
        phi = self._spheroidal_phase(QNM_freqs_32, powers_of_Mf)
        return amp * tm.cexp(ComplexNumber([0.0, 1.0]) * phi)

    @ti.func
    def spherical_h32(
        self,
        h22: ti.template(),
        QNM_freqs_32: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> ComplexNumber:
        """
        The passed-in h22 has incorporated the common constant factor and f^(-7/6) factor,
        but does not include the dimension factor.
        """
        h32 = self._spheroidal_h32(QNM_freqs_32, powers_of_Mf)
        return tm.cmul(self.a_322, h22) + tm.cmul(self.a_332, h32)

    @ti.func
    def update_amplitude_coefficients(self, source_params: ti.template()):
        f_ring_32 = source_params.QNM_freqs_lm["32"].f_ring
        f_damp_32 = source_params.QNM_freqs_lm["32"].f_damp
        self.amp_f_falloff = f_ring_32 + 2.0 * f_damp_32
        self.amp_f_aux = f_ring_32 - f_damp_32

        # RDCoefficient[0] in lalsim
        fit_alambda = ti.abs(
            source_params.delta_chi_half_pow2
            * (
                -3.4614418482110163 * source_params.eta_pow3
                + 35.464117772624164 * source_params.eta_pow4
                - 85.19723511005235 * source_params.eta_pow5
            )
            + source_params.delta_chi_half
            * source_params.delta
            * (
                2.0328561081997463 * source_params.eta_pow3
                - 46.18751757691501 * source_params.eta_pow4
                + 170.9266105597438 * source_params.eta_pow5
            )
            + source_params.delta_chi_half_pow2
            * (
                -0.4600401291210382 * source_params.eta_pow3
                + 12.23450117663151 * source_params.eta_pow4
                - 42.74689906831975 * source_params.eta_pow5
            )
            * source_params.S_tot_hat
            + source_params.delta_chi_half
            * source_params.delta
            * (
                5.786292428422767 * source_params.eta_pow3
                - 53.60467819078566 * source_params.eta_pow4
                + 117.66195692191727 * source_params.eta_pow5
            )
            * source_params.S_tot_hat
            + source_params.S_tot_hat
            * (
                -0.0013330716557843666
                * (
                    56.35538385647113 * source_params.eta
                    - 1218.1550992423377 * source_params.eta_pow2
                    + 16509.69605686402 * source_params.eta_pow3
                    - 102969.88022112886 * source_params.eta_pow4
                    + 252228.94931931415 * source_params.eta_pow5
                    - 150504.2927996263 * source_params.eta_pow6
                )
                + 0.0010126460331462495
                * (
                    -33.87083889060834 * source_params.eta
                    + 502.6221651850776 * source_params.eta_pow2
                    - 1304.9210590188136 * source_params.eta_pow3
                    - 36980.079328277505 * source_params.eta_pow4
                    + 295469.28617550555 * source_params.eta_pow5
                    - 597155.7619486618 * source_params.eta_pow6
                )
                * source_params.S_tot_hat
                - 0.00043088431510840695
                * (
                    -30.014415072587354 * source_params.eta
                    - 1900.5495690280086 * source_params.eta_pow2
                    + 76517.21042363928 * source_params.eta_pow3
                    - 870035.1394696251 * source_params.eta_pow4
                    + 3.9072674134789007e6 * source_params.eta_pow5
                    - 6.094089675611567e6 * source_params.eta_pow6
                )
                * source_params.S_tot_hat_pow2
            )
            + (
                0.08408469319155859 * source_params.eta
                - 1.223794846617597 * source_params.eta_pow2
                + 6.5972460654253515 * source_params.eta_pow3
                - 15.707327897569396 * source_params.eta_pow4
                + 14.163264397061505 * source_params.eta_pow5
            )
            / (
                1
                - 8.612447115134758 * source_params.eta
                + 18.93655612952139 * source_params.eta_pow2
            )
        )
        # RDCoefficient[1] in lalsim
        fit_lambda = (
            0.978510781593996
            + 0.36457571743142897 * source_params.eta
            - 12.259851752618998 * source_params.eta_pow2
            + 49.19719473681921 * source_params.eta_pow3
            + source_params.delta_chi_half
            * source_params.delta
            * (
                -188.37119473865533 * source_params.eta_pow3
                + 2151.8731700399308 * source_params.eta_pow4
                - 6328.182823770599 * source_params.eta_pow5
            )
            + source_params.delta_chi_half_pow2
            * (
                115.3689949926392 * source_params.eta_pow3
                - 1159.8596972989067 * source_params.eta_pow4
                + 2657.6998831179444 * source_params.eta_pow5
            )
            + source_params.S_tot_hat
            * (
                0.22358643406992756
                * (
                    0.48943645614341924
                    - 32.06682257944444 * source_params.eta
                    + 365.2485484044132 * source_params.eta_pow2
                    - 915.2489655397206 * source_params.eta_pow3
                )
                + 0.0792473022309144
                * (
                    1.877251717679991
                    - 103.65639889587327 * source_params.eta
                    + 1202.174780792418 * source_params.eta_pow2
                    - 3206.340850767219 * source_params.eta_pow3
                )
                * source_params.S_tot_hat
            )
        )
        # RDCoefficient[3] in lalsim
        fit_sigma = (
            1.3353917551819414
            + 0.13401718687342024 * source_params.eta
            + source_params.delta_chi_half
            * source_params.delta
            * (
                144.37065005786636 * source_params.eta_pow3
                - 754.4085447486738 * source_params.eta_pow4
                + 123.86194078913776 * source_params.eta_pow5
            )
            + source_params.delta_chi_half_pow2
            * (
                209.09202210427972 * source_params.eta_pow3
                - 1769.4658099037918 * source_params.eta_pow4
                + 3592.287297392387 * source_params.eta_pow5
            )
            + source_params.S_tot_hat
            * (
                -0.012086025709597246
                * (
                    -6.230497473791485
                    + 600.5968613752918 * source_params.eta
                    - 6606.1009717965735 * source_params.eta_pow2
                    + 17277.60594350428 * source_params.eta_pow3
                )
                - 0.06066548829900489
                * (
                    -0.9208054306316676
                    + 142.0346574366267 * source_params.eta
                    - 1567.249168668069 * source_params.eta_pow2
                    + 4119.373703246675 * source_params.eta_pow3
                )
                * source_params.S_tot_hat
            )
        )
        # coefficients for Lorentzian with exponential falloff part
        self.gamma_1 = fit_alambda * f_damp_32
        self.gamma_2 = fit_lambda / (fit_sigma * f_damp_32)
        self.gamma_3 = (fit_sigma * f_damp_32) ** 2
        # coefficients for entire exponential falloff part
        self.falloff_gamma_1 = self._spheroidal_amplitude_Lorentzian(
            self.amp_f_falloff - f_ring_32
        )
        self.falloff_gamma_2 = (
            -self._spheroidal_d_amplitude_Lorentzian(self.amp_f_falloff - f_ring_32)
            / self.falloff_gamma_1
        )
        # coefficients for polynomial part
        # 4 ansatz coefficients;
        # 3 collocation points;
        # 4 equations: 2 fit values + value at f_aux + derivative at f_aux)
        fit_aux_0 = ti.abs(
            source_params.delta_chi_half_pow2
            * (
                -4.188795724777721 * source_params.eta_pow2
                + 53.39200466700963 * source_params.eta_pow3
                - 131.19660856923554 * source_params.eta_pow4
            )
            + source_params.delta_chi_half
            * source_params.delta
            * (
                14.284921364132623 * source_params.eta_pow2
                - 321.26423637658746 * source_params.eta_pow3
                + 1242.865584938088 * source_params.eta_pow4
            )
            + source_params.S_tot_hat
            * (
                -0.022968727462555794
                * (
                    83.66854837403105 * source_params.eta
                    - 3330.6261333413177 * source_params.eta_pow2
                    + 77424.12614733395 * source_params.eta_pow3
                    - 710313.3016672594 * source_params.eta_pow4
                    + 2.6934917075009225e6 * source_params.eta_pow5
                    - 3.572465179268999e6 * source_params.eta_pow6
                )
                + 0.0014795114305436387
                * (
                    -1672.7273629876313 * source_params.eta
                    + 90877.38260964208 * source_params.eta_pow2
                    - 1.6690169155105734e6 * source_params.eta_pow3
                    + 1.3705532554135624e7 * source_params.eta_pow4
                    - 5.116110998398143e7 * source_params.eta_pow5
                    + 7.06066766311127e7 * source_params.eta_pow6
                )
                * source_params.S_tot_hat
            )
            + (
                4.45156488896258 * source_params.eta
                - 77.39303992494544 * source_params.eta_pow2
                + 522.5070635563092 * source_params.eta_pow3
                - 1642.3057499049708 * source_params.eta_pow4
                + 2048.333892310575 * source_params.eta_pow5
            )
            * pow(
                1
                - 9.611489164758915 * source_params.eta
                + 24.249594730050312 * source_params.eta_pow2,
                -1,
            )
        )
        fit_aux_1 = ti.abs(
            source_params.delta_chi_half_pow2
            * (
                -18.550171209458394 * source_params.eta_pow2
                + 188.99161055445936 * source_params.eta_pow3
                - 440.26516625611 * source_params.eta_pow4
            )
            + source_params.delta_chi_half
            * source_params.delta
            * (
                13.132625215315063 * source_params.eta_pow2
                - 340.5204040505528 * source_params.eta_pow3
                + 1327.1224176812448 * source_params.eta_pow4
            )
            + source_params.chi_PN_hat
            * (
                -0.16707403272774676
                * (
                    6.678916447469937 * source_params.eta
                    + 1331.480396625797 * source_params.eta_pow2
                    - 41908.45179140144 * source_params.eta_pow3
                    + 520786.0225074669 * source_params.eta_pow4
                    - 3.1894624909922685e6 * source_params.eta_pow5
                    + 9.51553823212259e6 * source_params.eta_pow6
                    - 1.1006903622406831e7 * source_params.eta_pow7
                )
                + 0.015205286051218441
                * (
                    108.10032279461095 * source_params.eta
                    - 16084.215590200103 * source_params.eta_pow2
                    + 462957.5593513407 * source_params.eta_pow3
                    - 5.635028227588545e6 * source_params.eta_pow4
                    + 3.379925277713386e7 * source_params.eta_pow5
                    - 9.865815275452062e7 * source_params.eta_pow6
                    + 1.1201307979786257e8 * source_params.eta_pow7
                )
                * source_params.chi_PN_hat
            )
            + (
                3.902154247490771 * source_params.eta
                - 55.77521071924907 * source_params.eta_pow2
                + 294.9496843041973 * source_params.eta_pow3
                - 693.6803787318279 * source_params.eta_pow4
                + 636.0141528226893 * source_params.eta_pow5
            )
            / (
                1.0
                - 8.56699762573719 * source_params.eta
                + 19.119341007236955 * source_params.eta_pow2
            )
        )
        aux_colloc_values = ti.Vector(
            [
                fit_aux_0,
                fit_aux_1,
                self._spheroidal_amplitude_Lorentzian(self.amp_f_aux - f_ring_32),
                self._spheroidal_d_amplitude_Lorentzian(self.amp_f_aux - f_ring_32),
            ],
            dt=ti.f64,
        )
        int_f_end = source_params.f_ring - 0.5 * source_params.f_damp
        aux_colloc_points = ti.Vector(
            [
                int_f_end,
                0.5 * (int_f_end + self.amp_f_aux),
                self.amp_f_aux,
            ],
            dt=ti.f64,
        )
        Ab = ti.Matrix([[0.0] * 5 for _ in range(4)], dt=ti.f64)
        row_idx = 0
        for i in ti.static(range(3)):
            # set the value for 3 collocation points
            Ab[row_idx, 4] = aux_colloc_values[row_idx]
            # set the coefficient matrix, [1, fi, fi^2, fi^3]
            fi = aux_colloc_points[i]
            fpower = 1.0
            for j in ti.static(range(4)):
                Ab[row_idx, j] = fpower
                fpower *= fi
            # next row
            row_idx += 1
        # the last row for derivative at f_aux
        Ab[row_idx, 4] = aux_colloc_values[row_idx]
        # the coefficient matrix, [0, 1, 2*fi, 3*fi^2]
        Ab[row_idx, 0] = 0.0
        fi = aux_colloc_points[2]
        fpower = 1.0
        for j in ti.static(range(1, 4)):
            Ab[row_idx, j] = j * fpower
            fpower *= fi
        self.amp_aux_coeffs = gauss_elimination(Ab)

    @ti.func
    def update_phase_coefficients(
        self,
        pn_coefficients_22: ti.template(),
        phase_coefficients_22: ti.template(),
        source_params: ti.template(),
    ):
        colloc_points = ti.Vector([0.0] * 4, dt=ti.f64)
        colloc_values = ti.Vector([0.0] * 4, dt=ti.f64)

        colloc_points[0] = source_params.f_ring
        colloc_points[1] = (
            source_params.QNM_freqs_lm["32"].f_ring
            - 1.5 * source_params.QNM_freqs_lm["32"].f_damp
        )
        colloc_points[2] = (
            source_params.QNM_freqs_lm["32"].f_ring
            - 0.5 * source_params.QNM_freqs_lm["32"].f_damp
        )
        colloc_points[3] = (
            source_params.QNM_freqs_lm["32"].f_ring
            + 0.5 * source_params.QNM_freqs_lm["32"].f_damp
        )

        colloc_values[0] = (
            3169.372056189274
            + 426.8372805022653 * source_params.eta
            - 12569.748101922158 * source_params.eta_pow2
            + 149846.7281073725 * source_params.eta_pow3
            - 817182.2896823225 * source_params.eta_pow4
            + 1.5674053633767858e6 * source_params.eta_pow5
            + (
                19.23408352151287
                - 1762.6573670619173 * source_params.eta
                + 7855.316419853637 * source_params.eta_pow2
                - 3785.49764771212 * source_params.eta_pow3
            )
            * source_params.S_tot_hat
            + (
                -42.88446003698396
                + 336.8340966473415 * source_params.eta
                - 5615.908682338113 * source_params.eta_pow2
                + 20497.5021807654 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow2
            + (
                13.918237996338371
                + 10145.53174542332 * source_params.eta
                - 91664.12621864353 * source_params.eta_pow2
                + 201204.5096556517 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow3
            + (
                -24.72321125342808
                - 4901.068176970293 * source_params.eta
                + 53893.9479532688 * source_params.eta_pow2
                - 139322.02687945773 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow4
            + (
                -61.01931672442576
                - 16556.65370439302 * source_params.eta
                + 162941.8009556697 * source_params.eta_pow2
                - 384336.57477596396 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow5
            + source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
            * (
                641.2473192044652
                - 1600.240100295189 * source_params.chi_1 * source_params.eta
                + 1600.240100295189 * source_params.chi_2 * source_params.eta
                + 13275.623692212472 * source_params.eta * source_params.S_tot_hat
            )
        )
        colloc_values[1] = (
            3131.0260952676376
            + 206.09687819102305 * source_params.eta
            - 2636.4344627081873 * source_params.eta_pow2
            + 7475.062269742079 * source_params.eta_pow3
            + (
                49.90874152040307
                - 691.9815135740145 * source_params.eta
                - 434.60154548208334 * source_params.eta_pow2
                + 10514.68111669422 * source_params.eta_pow3
            )
            * source_params.S_tot_hat
            + (
                97.3078084654917
                - 3458.2579971189534 * source_params.eta
                + 26748.805404989867 * source_params.eta_pow2
                - 56142.13736008524 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow2
            + (
                -132.49105074500454
                + 429.0787542102207 * source_params.eta
                + 7269.262546204149 * source_params.eta_pow2
                - 27654.067482558712 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow3
            + (
                -227.8023564332453
                + 5119.138772157134 * source_params.eta
                - 34444.2579678986 * source_params.eta_pow2
                + 69666.01833764123 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow4
            + 477.51566939885424
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
        )
        colloc_values[2] = (
            3082.803556599222
            + 76.94679795837645 * source_params.eta
            - 586.2469821978381 * source_params.eta_pow2
            + 977.6115755788503 * source_params.eta_pow3
            + (
                45.08944710349874
                - 807.7353772747749 * source_params.eta
                + 1775.4343704616288 * source_params.eta_pow2
                + 2472.6476419567534 * source_params.eta_pow3
            )
            * source_params.S_tot_hat
            + (
                95.57355060136699
                - 2224.9613131172046 * source_params.eta
                + 13821.251641893134 * source_params.eta_pow2
                - 25583.314298758105 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow2
            + (
                -144.96370424517866
                + 2268.4693587493093 * source_params.eta
                - 10971.864789147161 * source_params.eta_pow2
                + 16259.911572457446 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow3
            + (
                -227.8023564332453
                + 5119.138772157134 * source_params.eta
                - 34444.2579678986 * source_params.eta_pow2
                + 69666.01833764123 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow4
            + 378.2359918274837
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
        )
        colloc_values[3] = (
            3077.0657367004565
            + 64.99844502520415 * source_params.eta
            - 357.38692756785395 * source_params.eta_pow2
            + (
                34.793450080444714
                - 986.7751755509875 * source_params.eta
                - 9490.641676924794 * source_params.eta_pow3
                + 5700.682624203565 * source_params.eta_pow2
            )
            * source_params.S_tot_hat
            + (
                57.38106384558743
                - 1644.6690499868596 * source_params.eta
                - 19906.416384606226 * source_params.eta_pow3
                + 11008.881935880598 * source_params.eta_pow2
            )
            * source_params.S_tot_hat_pow2
            + (
                -126.02362949830213
                + 3169.3397351803583 * source_params.eta
                + 62863.79877094988 * source_params.eta_pow3
                - 26766.730897942085 * source_params.eta_pow2
            )
            * source_params.S_tot_hat_pow3
            + (
                -169.30909412804587
                + 4900.706039920717 * source_params.eta
                + 95314.99988114933 * source_params.eta_pow3
                - 41414.05689348732 * source_params.eta_pow2
            )
            * source_params.S_tot_hat_pow4
            + 390.5443469721231
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
        )

        Ab = ti.Matrix([[0.0] * 5 for _ in range(4)], dt=ti.f64)
        for i in ti.static(range(4)):
            row = [
                1,
                colloc_points[i] ** (-2),
                colloc_points[i] ** (-4),
                source_params.QNM_freqs_lm["32"].f_damp
                / (
                    source_params.QNM_freqs_lm["32"].f_damp_pow2
                    + (colloc_points[i] - source_params.QNM_freqs_lm["32"].f_ring) ** 2
                ),
                colloc_values[i],
            ]
            for j in ti.static(range(5)):
                Ab[i, j] = row[j]

        self.alpha_0, self.alpha_2, self.alpha_4, self.alpha_L = gauss_elimination(Ab)

        # Using two extra fits to ensure the phase in spheroidal basis has correct
        # relative time and phase shift wrt to the 22 mode.
        # time shift:
        # fit for dphi32_spheroidal(fref) - dphi22(fref)
        dphi_diff_fit = (
            11.851438981981772
            + 167.95086712701223 * source_params.eta
            - 4565.033758777737 * source_params.eta_pow2
            + 61559.132976189896 * source_params.eta_pow3
            - 364129.24735853914 * source_params.eta_pow4
            + 739270.8814129328 * source_params.eta_pow5
            + (
                9.506768471271634
                + 434.31707030999445 * source_params.eta
                - 8046.364492927503 * source_params.eta_pow2
                + 26929.677144312944 * source_params.eta_pow3
            )
            * source_params.S_tot_hat
            + (
                -5.949655484033632
                - 307.67253970367034 * source_params.eta
                + 1334.1062451631644 * source_params.eta_pow2
                + 3575.347142399199 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow2
            + (
                3.4881615575084797
                - 2244.4613237912527 * source_params.eta
                + 24145.932943269272 * source_params.eta_pow2
                - 60929.87465551446 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow3
            + (
                15.585154698977842
                - 2292.778112523392 * source_params.eta
                + 24793.809334683185 * source_params.eta_pow2
                - 65993.84497923202 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow4
            + 465.7904934097202
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
        )
        # impose that dphi_spheroidal_32(fref)-dphi_22(fref) = dphi_diff_fit
        powers_of_fref = UsefulPowers()
        powers_of_fref.update(source_params.f_ring + source_params.f_damp)
        dphi_ref_22 = phase_coefficients_22.compute_d_phase(
            pn_coefficients_22, source_params, powers_of_fref
        )
        self.dphi0 = (
            dphi_diff_fit
            + dphi_ref_22
            - self._spheroidal_d_phase_core(
                source_params.QNM_freqs_lm["32"], powers_of_fref
            )
        )

        # phase shift:
        phi_diff_fit = (
            -1.3328895897490733
            - 22.209549522908667 * source_params.eta
            + 1056.2426481245027 * source_params.eta_pow2
            - 21256.376324666326 * source_params.eta_pow3
            + 246313.12887984765 * source_params.eta_pow4
            - 1.6312968467540336e6 * source_params.eta_pow5
            + 5.614617173188322e6 * source_params.eta_pow6
            - 7.612233821752137e6 * source_params.eta_pow7
            + (
                source_params.S_tot_hat
                * (
                    -1.622727240110213
                    + 0.9960210841611344 * source_params.S_tot_hat
                    - 1.1239505323267036 * source_params.S_tot_hat_pow2
                    - 1.9586085340429995 * source_params.S_tot_hat_pow3
                    + source_params.eta_pow2
                    * (
                        196.7055281997748
                        + 135.25216875394943 * source_params.S_tot_hat
                        + 1086.7504825459278 * source_params.S_tot_hat_pow2
                        + 546.6246807461155 * source_params.S_tot_hat_pow3
                        - 312.1010566468068 * source_params.S_tot_hat_pow4
                    )
                    + 0.7638287749489343 * source_params.S_tot_hat_pow4
                    + source_params.eta
                    * (
                        -47.475568056234245
                        - 35.074072557604445 * source_params.S_tot_hat
                        - 97.16014978329918 * source_params.S_tot_hat_pow2
                        - 34.498125910065156 * source_params.S_tot_hat_pow3
                        + 24.02858084544326 * source_params.S_tot_hat_pow4
                    )
                    + source_params.eta_pow3
                    * (
                        62.632493533037625
                        - 22.59781899512552 * source_params.S_tot_hat
                        - 2683.947280170815 * source_params.S_tot_hat_pow2
                        - 1493.177074873678 * source_params.S_tot_hat_pow3
                        + 805.0266029288334 * source_params.S_tot_hat_pow4
                    )
                )
            )
            / (-2.950271397057221 + source_params.S_tot_hat)
            + (
                source_params.delta
                * (
                    source_params.chi_2
                    * source_params.eta_pow2
                    * source_params.eta_sqrt
                    * (88.56162028006072 - 30.01812659282717 * source_params.S_tot_hat)
                    + source_params.chi_2
                    * source_params.eta_pow2
                    * (
                        43.126266433486435
                        - 14.617728550838805 * source_params.S_tot_hat
                    )
                    + source_params.chi_1
                    * source_params.eta_pow2
                    * (
                        -43.126266433486435
                        + 14.617728550838805 * source_params.S_tot_hat
                    )
                    + source_params.chi_1
                    * source_params.eta_pow2
                    * source_params.eta_sqrt
                    * (-88.56162028006072 + 30.01812659282717 * source_params.S_tot_hat)
                )
            )
            / (-2.950271397057221 + source_params.S_tot_hat)
        )
        powers_of_fref.update(source_params.f_ring)
        phi_ref_22 = phase_coefficients_22.compute_phase(
            pn_coefficients_22, source_params, powers_of_fref
        )
        self.phi0 = (
            phi_diff_fit
            + phi_ref_22
            - (
                self._spheroidal_phase_core(
                    source_params.QNM_freqs_lm["32"], powers_of_fref
                )
                + self.dphi0 * powers_of_fref.one
            )
        )

    @ti.func
    def update_mixing_coefficients(self, source_params: ti.template()):
        self.a_322[0] = (
            -source_params.final_spin
            * (
                0.47513455283841244
                - 0.9016636384605536 * source_params.final_spin
                + 0.3844811236426182 * source_params.final_spin_pow2
                + 0.0855565148647794 * source_params.final_spin_pow3
                - 0.03620067426672167 * source_params.final_spin_pow4
                - 0.006557249133752502 * source_params.final_spin_pow5
            )
            / (
                -6.76894063440646
                + 15.170831931186493 * source_params.final_spin
                - 9.406169787571082 * source_params.final_spin_pow2
                + source_params.final_spin_pow4
            )
        )
        self.a_322[1] = (
            source_params.final_spin
            * (
                -2.8704762147145533
                + 4.436434016918535 * source_params.final_spin
                - 1.0115343326360486 * source_params.final_spin_pow2
                - 0.08965314412106505 * source_params.final_spin_pow3
                - 0.4236810894599512 * source_params.final_spin_pow4
                - 0.041787576033810676 * source_params.final_spin_pow5
            )
            / (
                -171.80908957903395
                + 272.362882450877 * source_params.final_spin
                - 76.68544453077854 * source_params.final_spin_pow2
                - 25.14197656531123 * source_params.final_spin_pow4
                + source_params.final_spin_pow6
            )
        )
        self.a_332[0] = -(
            1.0
            - 2.107852425643677 * source_params.final_spin
            + 1.1906393634562715 * source_params.final_spin_pow2
            + 0.02244848864087732 * source_params.final_spin_pow3
            - 0.09593447799423722 * source_params.final_spin_pow4
            - 0.0021343381708933025 * source_params.final_spin_pow5
            - 0.005319515989331159 * source_params.final_spin_pow6
        ) / (
            1.0
            - 2.1078515887706324 * source_params.final_spin
            + 1.2043484690080966 * source_params.final_spin_pow2
            - 0.08910191596778137 * source_params.final_spin_pow4
            - 0.005471749827809503 * source_params.final_spin_pow6
        )
        self.a_332[1] = (
            source_params.final_spin
            * (
                12.45701482868677
                - 29.398484595717147 * source_params.final_spin
                + 18.26221675782779 * source_params.final_spin_pow2
                + 1.9308599142669403 * source_params.final_spin_pow3
                - 3.159763242921214 * source_params.final_spin_pow4
                - 0.0910871567367674 * source_params.final_spin_pow5
            )
            / (
                345.52914639836257
                - 815.4349339779621 * source_params.final_spin
                + 538.3888932415709 * source_params.final_spin_pow2
                - 69.3840921447381 * source_params.final_spin_pow4
                + source_params.final_spin_pow6
            )
        )

    @ti.func
    def update_all_coefficients(
        self,
        pn_coefficients_22: ti.template(),
        phase_coefficients_22: ti.template(),
        source_params: ti.template(),
    ):
        self.update_amplitude_coefficients(source_params)
        self.update_phase_coefficients(
            pn_coefficients_22, phase_coefficients_22, source_params
        )
        self.update_mixing_coefficients(source_params)


@sub_struct_from(PostNewtonianCoefficientsHighModesBase)
class PostNewtonianCoefficientsMode21:

    @ti.func
    def update_pn_coefficients(
        self,
        pn_coefficients_22: ti.template(),
        source_params: ti.template(),
    ):
        self._set_rescaling_phase_coefficients(1.0, pn_coefficients_22)

        # since only absolute values are used, the minus and i are dropped.
        amp_global = tm.sqrt(2.0) / 3.0
        self.A_0 = amp_global * ComplexNumber([0.0, 0.0])
        self.A_1 = (
            amp_global
            * ComplexNumber([source_params.delta, 0.0])
            * useful_powers_2pi.third
        )
        self.A_2 = (
            amp_global
            * ComplexNumber(
                [
                    -3.0
                    / 2.0
                    * (source_params.chi_a + source_params.chi_s * source_params.delta),
                    0.0,
                ]
            )
            * useful_powers_2pi.two_thirds
        )
        self.A_3 = (
            amp_global
            * ComplexNumber(
                [
                    (
                        3.35 / 6.72 * source_params.delta
                        + 11.7 / 5.6 * source_params.delta * source_params.eta
                    ),
                    0.0,
                ]
            )
            * useful_powers_2pi.one
        )
        self.A_4 = (
            amp_global
            * ComplexNumber(
                [
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
                    (-0.5 - 2.0 * tm.log(2.0)) * source_params.delta,
                ]
            )
            * useful_powers_2pi.four_thirds
        )
        self.A_5 = (
            amp_global
            * ComplexNumber(
                [
                    (
                        -0.964357 / 8.128512 * source_params.delta
                        - 3.6529 / 1.2544 * source_params.delta * source_params.eta
                        + 21.365 / 8.064 * source_params.delta * source_params.eta_pow2
                        + 3.0 * source_params.chi_a * PI
                        + 3.0 * source_params.chi_s * source_params.delta * PI
                        - 30.7 / 3.2 * source_params.chi_a_pow2 * source_params.delta
                        + 10.0
                        * source_params.chi_a_pow2
                        * source_params.delta
                        * source_params.eta
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
                    ),
                    0.0,
                ]
            )
            * useful_powers_2pi.five_thirds
        )
        self.A_6 = (
            amp_global
            * ComplexNumber(
                [
                    (
                        -2.455 / 1.344 * source_params.delta * PI
                        + 4.17 / 1.12 * source_params.delta * source_params.eta * PI
                        + 143.063173 / 5.419008 * source_params.chi_a
                        - 227.58317 / 2.25792 * source_params.chi_a * source_params.eta
                        + 42.617 / 1.792 * source_params.chi_a * source_params.eta_pow2
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
                        + 24.3 / 6.4 * source_params.chi_s_pow3 * source_params.delta
                        - 3.0
                        / 16.0
                        * source_params.chi_s_pow3
                        * source_params.delta
                        * source_params.eta
                        + 72.9 / 6.4 * source_params.chi_a * source_params.chi_s_pow2
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
                    (
                        -(3.35 / 13.44 + 3.35 / 3.36 * tm.log(2.0))
                        * source_params.delta
                        - (14.89 / 1.12 + 8.9 / 2.8 * tm.log(2.0))
                        * source_params.delta
                        * source_params.eta
                    ),
                ]
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

        # since only absolute values are used, the minus and i are dropped.
        amp_global = 0.75 * tm.sqrt(5.0 / 7.0)
        self.A_0 = amp_global * ComplexNumber([0.0, 0.0])
        self.A_1 = (
            amp_global
            * ComplexNumber([source_params.delta, 0.0])
            * useful_powers_2pi_over_3.third
        )
        self.A_2 = amp_global * ComplexNumber([0.0, 0.0])
        self.A_3 = (
            amp_global
            * ComplexNumber(
                [
                    (
                        -19.45 / 6.72 * source_params.delta
                        + 27.0 / 8.0 * source_params.delta * source_params.eta
                    ),
                    0.0,
                ]
            )
            * useful_powers_2pi_over_3.one
        )
        self.A_4 = (
            amp_global
            * ComplexNumber(
                [
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
                    (-21.0 / 5.0 + 6.0 * tm.log(1.5)) * source_params.delta,
                ]
            )
            * useful_powers_2pi_over_3.four_thirds
        )
        self.A_5 = (
            amp_global
            * ComplexNumber(
                [
                    (
                        -10.77664867 / 4.47068160 * source_params.delta
                        - 117.58073 / 8.87040 * source_params.delta * source_params.eta
                        + 42.0389
                        / 6.3360
                        * source_params.delta
                        * source_params.eta_pow2
                        - 8.1 / 3.2 * source_params.chi_a_pow2 * source_params.delta
                        - 8.1 / 3.2 * source_params.chi_s_pow2 * source_params.delta
                        + 10.0
                        * source_params.chi_a_pow2
                        * source_params.delta
                        * source_params.eta
                        + 1.0
                        / 8.0
                        * source_params.chi_s_pow2
                        * source_params.delta
                        * source_params.eta
                        - 8.1 / 1.6 * source_params.chi_a * source_params.chi_s
                        + 81.0
                        / 4.0
                        * source_params.chi_a
                        * source_params.chi_s
                        * source_params.eta
                    ),
                    0.0,
                ]
            )
            * useful_powers_2pi_over_3.five_thirds
        )
        self.A_6 = (
            amp_global
            * ComplexNumber(
                [
                    (
                        -5.675 / 1.344 * source_params.delta * PI
                        + 13.1 / 1.6 * source_params.delta * source_params.eta * PI
                        + 16.3021 / 1.6128 * source_params.chi_a
                        - 148.501 / 4.032 * source_params.chi_a * source_params.eta
                        - 13.7 / 2.4 * source_params.chi_a * source_params.eta_pow2
                        + 16.3021 / 1.6128 * source_params.chi_s * source_params.delta
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
                    (
                        (38.9 / 3.2 - 19.45 / 1.12 * tm.log(1.5)) * source_params.delta
                        + (-440.957 / 9.720 + 69.0 / 4.0 * tm.log(1.5))
                        * source_params.delta
                        * source_params.eta
                    ),
                ]
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

        # since only absolute values are used, the minus and i are dropped.
        amp_global = tm.sqrt(5.0 / 7.0) / 3.0
        self.A_0 = amp_global * ComplexNumber([0.0, 0.0])
        self.A_1 = amp_global * ComplexNumber([0.0, 0.0])
        self.A_2 = (
            amp_global
            * ComplexNumber([(-1.0 + 3.0 * source_params.eta), 0.0])
            * useful_powers_pi.two_thirds
        )
        self.A_3 = (
            amp_global
            * ComplexNumber([(-4.0 * source_params.chi_s * source_params.eta), 0.0])
            * useful_powers_pi.one
        )
        self.A_4 = (
            amp_global
            * ComplexNumber(
                [
                    (
                        1.0471 / 1.0080
                        - 12.325 / 2.016 * source_params.eta
                        + 58.9 / 7.2 * source_params.eta_pow2
                    ),
                    0.0,
                ]
            )
            * useful_powers_pi.four_thirds
        )
        self.A_5 = (
            amp_global
            * ComplexNumber(
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
            * useful_powers_pi.five_thirds
        )
        self.A_6 = (
            amp_global
            * ComplexNumber(
                [
                    (
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
                    ),
                    0.0,
                ]
            )
            * useful_powers_pi.two
        )


@sub_struct_from(PostNewtonianCoefficientsHighModesBase)
class PostNewtonianCoefficientsMode44:

    @ti.func
    def update_pn_coefficients(
        self,
        pn_coefficients_22: ti.template(),
        source_params: ti.template(),
    ):
        self._set_rescaling_phase_coefficients(4.0, pn_coefficients_22)

        # since only absolute values are used, the minus and i are dropped.
        amp_global = 4.0 / 9.0 * tm.sqrt(10.0 / 7.0)
        self.A_0 = amp_global * ComplexNumber([0.0, 0.0])
        self.A_1 = amp_global * ComplexNumber([0.0, 0.0])
        self.A_2 = (
            amp_global
            * ComplexNumber([(1.0 - 3.0 * source_params.eta), 0.0])
            * useful_powers_pi_over_2.two_thirds
        )
        self.A_3 = amp_global * ComplexNumber([0.0, 0.0])
        self.A_4 = (
            amp_global
            * ComplexNumber(
                [
                    (
                        -15.8383 / 3.6960
                        + 128.221 / 7.392 * source_params.eta
                        - 106.3 / 8.8 * source_params.eta_pow2
                    ),
                    0.0,
                ]
            )
            * useful_powers_pi_over_2.four_thirds
        )
        self.A_5 = (
            amp_global
            * ComplexNumber(
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
            * useful_powers_pi_over_2.five_thirds
        )
        self.A_6 = (
            amp_global
            * ComplexNumber(
                [
                    (
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
                    ),
                    0.0,
                ]
            )
            * useful_powers_pi_over_2.two
        )


@sub_struct_from(AmplitudeCoefficientsHighModesBase)
class AmplitudeCoefficientsMode21:
    """
    In the intermediate of 21 mode, the derivative of left boundary and two collocation
    points in the middle are dropped to avoid wavy behavior. There are only 5 coefficints
    for the intermediate of mode 21.
    """

    int_ansatz_coeffs: ti.types.vector(5, ti.f64)

    @ti.func
    def _ins_fit_v0(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            source_params.delta_chi_half
            * source_params.eta_pow5
            * (
                -3962.5020052272976
                + 987.635855365408 * source_params.chi_PN_hat
                - 134.98527058315528 * source_params.chi_PN_hat_pow2
            )
            + source_params.delta
            * (
                19.30531354642419
                + 16.6640319856064 * source_params.eta
                - 120.58166037019478 * source_params.eta_pow2
                + 220.77233521626252 * source_params.eta_pow3
            )
            * source_params.eta_sqrt
            + source_params.delta_chi_half
            * source_params.delta
            * (
                31.364509907424765 * source_params.eta
                - 843.6414532232126 * source_params.eta_pow2
                + 2638.3077554662905 * source_params.eta_pow3
            )
            * source_params.eta_sqrt
            + source_params.delta_chi_half
            * source_params.delta
            * (
                32.374226994179054 * source_params.eta
                - 202.86279451816662 * source_params.eta_pow2
                + 347.1621871204769 * source_params.eta_pow3
            )
            * source_params.chi_PN_hat
            * source_params.eta_sqrt
            + source_params.delta
            * source_params.chi_PN_hat
            * (
                -16.75726972301224
                * (
                    1.1787350890261943
                    - 7.812073811917883 * source_params.eta
                    + 99.47071002831267 * source_params.eta_pow2
                    - 500.4821414428368 * source_params.eta_pow3
                    + 876.4704270866478 * source_params.eta_pow4
                )
                + 2.3439955698372663
                * (
                    0.9373952326655807
                    + 7.176140122833879 * source_params.eta
                    - 279.6409723479635 * source_params.eta_pow2
                    + 2178.375177755584 * source_params.eta_pow3
                    - 4768.212511142035 * source_params.eta_pow4
                )
                * source_params.chi_PN_hat
            )
            * source_params.eta_sqrt
        )

    @ti.func
    def _ins_fit_v1(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            source_params.delta_chi_half
            * source_params.eta_pow5
            * (
                -2898.9172078672705
                + 580.9465034962822 * source_params.chi_PN_hat
                + 22.251142639924076 * source_params.chi_PN_hat_pow2
            )
            + source_params.delta
            * (
                source_params.delta_chi_half_pow2
                * (
                    -18.541685007214625 * source_params.eta
                    + 166.7427445020744 * source_params.eta_pow2
                    - 417.5186332459383 * source_params.eta_pow3
                )
                + source_params.delta_chi_half
                * (
                    41.61457952037761 * source_params.eta
                    - 779.9151607638761 * source_params.eta_pow2
                    + 2308.6520892707795 * source_params.eta_pow3
                )
            )
            * source_params.eta_sqrt
            + source_params.delta
            * (
                11.414934585404561
                + 30.883118528233638 * source_params.eta
                - 260.9979123967537 * source_params.eta_pow2
                + 1046.3187137392433 * source_params.eta_pow3
                - 1556.9475493549746 * source_params.eta_pow4
            )
            * source_params.eta_sqrt
            + source_params.delta
            * source_params.chi_PN_hat
            * (
                -10.809007068469844
                * (
                    1.1408749895922659
                    - 18.140470190766937 * source_params.eta
                    + 368.25127088896744 * source_params.eta_pow2
                    - 3064.7291458207815 * source_params.eta_pow3
                    + 11501.848278358668 * source_params.eta_pow4
                    - 16075.676528787526 * source_params.eta_pow5
                )
                + 1.0088254664333147
                * (
                    1.2322739396680107
                    - 192.2461213084741 * source_params.eta
                    + 4257.760834055382 * source_params.eta_pow2
                    - 35561.24587952242 * source_params.eta_pow3
                    + 130764.22485304279 * source_params.eta_pow4
                    - 177907.92440833704 * source_params.eta_pow5
                )
                * source_params.chi_PN_hat
            )
            * source_params.eta_sqrt
            + source_params.delta
            * (
                source_params.delta_chi_half
                * (
                    36.88578491943111 * source_params.eta
                    - 321.2569602623214 * source_params.eta_pow2
                    + 748.6659668096737 * source_params.eta_pow3
                )
                * source_params.chi_PN_hat
                + source_params.delta_chi_half
                * (
                    -95.42418611585117 * source_params.eta
                    + 1217.338674959742 * source_params.eta_pow2
                    - 3656.192371615541 * source_params.eta_pow3
                )
                * source_params.chi_PN_hat_pow2
            )
            * source_params.eta_sqrt
        )

    @ti.func
    def _ins_fit_v2(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            source_params.delta_chi_half
            * source_params.eta_pow5
            * (
                -2282.9983216879655
                + 157.94791186394787 * source_params.chi_PN_hat
                + 16.379731479465033 * source_params.chi_PN_hat_pow2
            )
            + source_params.delta_chi_half
            * source_params.delta
            * (
                21.935833431534224 * source_params.eta
                - 460.7130131927895 * source_params.eta_pow2
                + 1350.476411541137 * source_params.eta_pow3
            )
            * source_params.eta_sqrt
            + source_params.delta
            * (
                5.390240326328237
                + 69.01761987509603 * source_params.eta
                - 568.0027716789259 * source_params.eta_pow2
                + 2435.4098320959706 * source_params.eta_pow3
                - 3914.3390484239667 * source_params.eta_pow4
            )
            * source_params.eta_sqrt
            + source_params.delta_chi_half
            * source_params.delta
            * (
                29.731007410186827 * source_params.eta
                - 372.09609843131386 * source_params.eta_pow2
                + 1034.4897198648962 * source_params.eta_pow3
            )
            * source_params.chi_PN_hat
            * source_params.eta_sqrt
            + source_params.delta
            * source_params.chi_PN_hat
            * (
                -7.1976397556450715
                * (
                    0.7603360145475428
                    - 6.587249958654174 * source_params.eta
                    + 120.87934060776237 * source_params.eta_pow2
                    - 635.1835857158857 * source_params.eta_pow3
                    + 1109.0598539312573 * source_params.eta_pow4
                )
                - 0.0811847192323969
                * (
                    7.951454648295709
                    + 517.4039644814231 * source_params.eta
                    - 9548.970156895082 * source_params.eta_pow2
                    + 52586.63520999897 * source_params.eta_pow3
                    - 93272.17990295641 * source_params.eta_pow4
                )
                * source_params.chi_PN_hat
                - 0.28384547935698246
                * (
                    -0.8870770459576875
                    + 180.0378964169756 * source_params.eta
                    - 2707.9572896559484 * source_params.eta_pow2
                    + 14158.178124971111 * source_params.eta_pow3
                    - 24507.800226675925 * source_params.eta_pow4
                )
                * source_params.chi_PN_hat_pow2
            )
            * source_params.eta_sqrt
        )

    @ti.func
    def _MRD_fit_v0(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            source_params.delta
            * source_params.eta
            * (
                12.880905080761432
                - 23.5291063016996 * source_params.eta
                + 92.6090002736012 * source_params.eta_pow2
                - 175.16681482428694 * source_params.eta_pow3
            )
            + source_params.delta_chi_half
            * source_params.delta
            * source_params.eta
            * (
                26.89427230731867 * source_params.eta
                - 710.8871223808559 * source_params.eta_pow2
                + 2255.040486907459 * source_params.eta_pow3
            )
            + source_params.delta_chi_half
            * source_params.delta
            * source_params.eta
            * (
                21.402708785047853 * source_params.eta
                - 232.07306353130417 * source_params.eta_pow2
                + 591.1097623278739 * source_params.eta_pow3
            )
            * source_params.chi_PN_hat
            + source_params.delta
            * source_params.eta
            * source_params.chi_PN_hat
            * (
                -10.090867481062709
                * (
                    0.9580746052260011
                    + 5.388149112485179 * source_params.eta
                    - 107.22993216128548 * source_params.eta_pow2
                    + 801.3948756800821 * source_params.eta_pow3
                    - 2688.211889175019 * source_params.eta_pow4
                    + 3950.7894052628735 * source_params.eta_pow5
                    - 1992.9074348833092 * source_params.eta_pow6
                )
                - 0.42972412296628143
                * (
                    1.9193131231064235
                    + 139.73149069609775 * source_params.eta
                    - 1616.9974609915555 * source_params.eta_pow2
                    - 3176.4950303461164 * source_params.eta_pow3
                    + 107980.65459735804 * source_params.eta_pow4
                    - 479649.75188253267 * source_params.eta_pow5
                    + 658866.0983367155 * source_params.eta_pow6
                )
                * source_params.chi_PN_hat
            )
            + source_params.delta_chi_half
            * source_params.eta_pow5
            * (
                -1512.439342647443
                + 175.59081294852444 * source_params.chi_PN_hat
                + 10.13490934572329 * source_params.chi_PN_hat_pow2
            )
        )

    @ti.func
    def _MRD_fit_v1(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            source_params.delta
            * (9.112452928978168 - 7.5304766811877455 * source_params.eta)
            * source_params.eta
            + source_params.delta_chi_half
            * source_params.delta
            * source_params.eta
            * (
                16.236533863306132 * source_params.eta
                - 500.11964987628926 * source_params.eta_pow2
                + 1618.0818430353293 * source_params.eta_pow3
            )
            + source_params.delta_chi_half
            * source_params.delta
            * source_params.eta
            * (
                2.7866868976718226 * source_params.eta
                - 0.4210629980868266 * source_params.eta_pow2
                - 20.274691328125606 * source_params.eta_pow3
            )
            * source_params.chi_PN_hat
            + source_params.delta_chi_half
            * source_params.eta_pow5
            * (
                -1116.4039232324135
                + 245.73200219767514 * source_params.chi_PN_hat
                + 21.159179960295855 * source_params.chi_PN_hat_pow2
            )
            + source_params.delta
            * source_params.eta
            * source_params.chi_PN_hat
            * (
                -8.236485576091717
                * (
                    0.8917610178208336
                    + 5.1501231412520285 * source_params.eta
                    - 87.05136337926156 * source_params.eta_pow2
                    + 519.0146702141192 * source_params.eta_pow3
                    - 997.6961311502365 * source_params.eta_pow4
                )
                + 0.2836840678615208
                * (
                    -0.19281297100324718
                    - 57.65586769647737 * source_params.eta
                    + 586.7942442434971 * source_params.eta_pow2
                    - 1882.2040277496196 * source_params.eta_pow3
                    + 2330.3534917059906 * source_params.eta_pow4
                )
                * source_params.chi_PN_hat
                + 0.40226131643223145
                * (
                    -3.834742668014861
                    + 190.42214703482531 * source_params.eta
                    - 2885.5110686004946 * source_params.eta_pow2
                    + 16087.433824017446 * source_params.eta_pow3
                    - 29331.524552164105 * source_params.eta_pow4
                )
                * source_params.chi_PN_hat_pow2
            )
        )

    @ti.func
    def _MRD_fit_v2(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            source_params.delta
            * (2.920930733198033 - 3.038523690239521 * source_params.eta)
            * source_params.eta
            + source_params.delta_chi_half
            * source_params.delta
            * source_params.eta
            * (
                6.3472251472354975 * source_params.eta
                - 171.23657247338042 * source_params.eta_pow2
                + 544.1978232314333 * source_params.eta_pow3
            )
            + source_params.delta_chi_half
            * source_params.delta
            * source_params.eta
            * (
                1.9701247529688362 * source_params.eta
                - 2.8616711550845575 * source_params.eta_pow2
                - 0.7347258030219584 * source_params.eta_pow3
            )
            * source_params.chi_PN_hat
            + source_params.delta_chi_half
            * source_params.eta_pow5
            * (
                -334.0969956136684
                + 92.91301644484749 * source_params.chi_PN_hat
                - 5.353399481074393 * source_params.chi_PN_hat_pow2
            )
            + source_params.delta
            * source_params.eta
            * source_params.chi_PN_hat
            * (
                -2.7294297839371824
                * (
                    1.148166706456899
                    - 4.384077347340523 * source_params.eta
                    + 36.120093043420326 * source_params.eta_pow2
                    - 87.26454353763077 * source_params.eta_pow3
                )
                + 0.23949142867803436
                * (
                    -0.6931516433988293
                    + 33.33372867559165 * source_params.eta
                    - 307.3404155231787 * source_params.eta_pow2
                    + 862.3123076782916 * source_params.eta_pow3
                )
                * source_params.chi_PN_hat
                + 0.1930861073906724
                * (
                    3.7735099269174106
                    - 19.11543562444476 * source_params.eta
                    - 78.07256429516346 * source_params.eta_pow2
                    + 485.67801863289293 * source_params.eta_pow3
                )
                * source_params.chi_PN_hat_pow2
            )
        )

    @ti.func
    def _set_intermediate_coefficients(
        self,
        pn_coefficients_21: ti.template(),
        source_params: ti.template(),
    ):
        """
        Require inspiral and merge-ringdown amplitude to set boundaries, can only be
        called after updating inspiral and merge-ringdown coefficients.

        Note IntermediateAmpVersion = 110102 for mode 21, where the derivative of left
        boundary is skip and the two collocation points in the middle are dropped (using
        collocation points of 0, 1, 3, 5).
        """
        int_colloc_points = ti.Vector([0.0] * 4, dt=ti.f64)
        int_f_space = (self.int_f_end - self.ins_f_end) / 5.0
        int_colloc_points[0] = self.ins_f_end
        int_colloc_points[1] = self.ins_f_end + int_f_space
        int_colloc_points[2] = self.ins_f_end + 3.0 * int_f_space
        int_colloc_points[3] = self.int_f_end
        powers_int_f0 = UsefulPowers()
        powers_int_f0.update(int_colloc_points[0])

        int_colloc_values = ti.Vector([0.0] * 5, dt=ti.f64)
        # left boundary
        int_colloc_values[0] = self._inspiral_amplitude(
            pn_coefficients_21, powers_int_f0
        )
        # fit of collocation point (ins_f_end + int_f_space)
        # IMRPhenomXHM_Inter_Amp_21_int1
        int_colloc_values[1] = ti.abs(
            source_params.delta
            * source_params.eta
            * (
                source_params.delta_chi_half_pow2
                * (
                    5.159755997682368 * source_params.eta
                    - 30.293198248154948 * source_params.eta_pow2
                    + 63.70715919820867 * source_params.eta_pow3
                )
                + source_params.delta_chi_half
                * (
                    8.262642080222694 * source_params.eta
                    - 415.88826990259116 * source_params.eta_pow2
                    + 1427.5951158851076 * source_params.eta_pow3
                )
            )
            + source_params.delta
            * source_params.eta
            * (
                18.55363583212328
                - 66.46950491124205 * source_params.eta
                + 447.2214642597892 * source_params.eta_pow2
                - 1614.178472020212 * source_params.eta_pow3
                + 2199.614895727586 * source_params.eta_pow4
            )
            + source_params.delta_chi_half
            * source_params.eta_pow5
            * (
                -1698.841763891122
                - 195.27885562092342 * source_params.S_tot_hat
                - 1.3098861736238572 * source_params.S_tot_hat_pow2
            )
            + source_params.delta
            * source_params.eta
            * (
                source_params.delta_chi_half
                * (
                    34.17829404207186 * source_params.eta
                    - 386.34587928670015 * source_params.eta_pow2
                    + 1022.8553774274128 * source_params.eta_pow3
                )
                * source_params.S_tot_hat
                + source_params.delta_chi_half
                * (
                    56.76554600963724 * source_params.eta
                    - 491.4593694689354 * source_params.eta_pow2
                    + 1016.6019654342113 * source_params.eta_pow3
                )
                * source_params.S_tot_hat_pow2
            )
            + source_params.delta
            * source_params.eta
            * source_params.S_tot_hat
            * (
                -8.276366844994188
                * (
                    1.0677538075697492
                    - 24.12941323757896 * source_params.eta
                    + 516.7886322104276 * source_params.eta_pow2
                    - 4389.799658723288 * source_params.eta_pow3
                    + 16770.447637953577 * source_params.eta_pow4
                    - 23896.392706809565 * source_params.eta_pow5
                )
                - 1.6908277400304084
                * (
                    3.4799140066657928
                    - 29.00026389706585 * source_params.eta
                    + 114.8330693231833 * source_params.eta_pow2
                    - 184.13091281984674 * source_params.eta_pow3
                    + 592.300353344717 * source_params.eta_pow4
                    - 2085.0821513466053 * source_params.eta_pow5
                )
                * source_params.S_tot_hat
                - 0.46006975902558517
                * (
                    -2.1663474937625975
                    + 826.026625945615 * source_params.eta
                    - 17333.549622759732 * source_params.eta_pow2
                    + 142904.08962903373 * source_params.eta_pow3
                    - 528521.6231015554 * source_params.eta_pow4
                    + 731179.456702448 * source_params.eta_pow5
                )
                * source_params.S_tot_hat_pow2
            )
        )
        # fit of collocation point (ins_f_end + 3*int_f_space)
        # IMRPhenomXHM_Inter_Amp_21_int3
        int_colloc_values[2] = ti.abs(
            source_params.delta
            * source_params.eta
            * (
                13.318990196097973
                - 21.755549987331054 * source_params.eta
                + 76.14884211156267 * source_params.eta_pow2
                - 127.62161159798488 * source_params.eta_pow3
            )
            + source_params.delta_chi_half
            * source_params.delta
            * source_params.eta
            * (
                17.704321326939414 * source_params.eta
                - 434.4390350012534 * source_params.eta_pow2
                + 1366.2408490833282 * source_params.eta_pow3
            )
            + source_params.delta_chi_half
            * source_params.delta
            * source_params.eta
            * (
                11.877985158418596 * source_params.eta
                - 131.04937626836355 * source_params.eta_pow2
                + 343.79587860999874 * source_params.eta_pow3
            )
            * source_params.S_tot_hat
            + source_params.delta_chi_half
            * source_params.eta_pow5
            * (
                -1522.8543551416456
                - 16.639896279650678 * source_params.S_tot_hat
                + 3.0053086651515843 * source_params.S_tot_hat_pow2
            )
            + source_params.delta
            * source_params.eta
            * source_params.S_tot_hat
            * (
                -8.665646058245033
                * (
                    0.7862132291286934
                    + 8.293609541933655 * source_params.eta
                    - 111.70764910503321 * source_params.eta_pow2
                    + 576.7172598056907 * source_params.eta_pow3
                    - 1001.2370065269745 * source_params.eta_pow4
                )
                - 0.9459820574514348
                * (
                    1.309016452198605
                    + 48.94077040282239 * source_params.eta
                    - 817.7854010574645 * source_params.eta_pow2
                    + 4331.56002883546 * source_params.eta_pow3
                    - 7518.309520232795 * source_params.eta_pow4
                )
                * source_params.S_tot_hat
                - 0.4308267743835775
                * (
                    9.970654092010587
                    - 302.9708323417439 * source_params.eta
                    + 3662.099161055873 * source_params.eta_pow2
                    - 17712.883990278668 * source_params.eta_pow3
                    + 29480.158198408903 * source_params.eta_pow4
                )
                * source_params.S_tot_hat_pow2
            )
        )
        # right boundary
        int_colloc_values[3] = self._merge_ringdown_amplitude_Lorentzian(
            int_colloc_points[3] - source_params.QNM_freqs_lm["21"].f_ring
        )  # always have int_f_end < MRE_f_falloff
        # derivative at the right boundary
        int_colloc_values[4] = self._merge_ringdown_d_amplitude_Lorentzian(
            int_colloc_points[3] - source_params.QNM_freqs_lm["21"].f_ring
        )

        # set the augmented matrix
        Ab = ti.Matrix([[0.0] * 6 for _ in range(5)], dt=ti.f64)
        row_idx = 0
        for i in ti.static(range(4)):
            # set the value at the collocation point
            Ab[row_idx, 5] = int_colloc_values[row_idx]
            # set the coefficient matrix of frequency powers
            # note there are only 5 terms for mode 21
            # [1, fi, fi^2, fi^3, fi^4] * fi^(-7/6)
            fi = int_colloc_points[i]
            fpower = fi ** (-7.0 / 6.0)
            for j in ti.static(range(5)):
                Ab[row_idx, j] = fpower
                fpower *= fi
            # next row
            row_idx += 1
        # for the derivatives at the boundaries
        # note there is only the derivative of right boundary for mode 21
        Ab[row_idx, 5] = int_colloc_values[row_idx]
        # set the coefficient matrix of frequency powers for derivative
        # [(-7/6)fi_-1, (-7/6+1), (-7/6+2)fi, (-7/6+3)fi^2, (-7/6+4fi^3] * fi^(-7/6)
        fi = int_colloc_points[3]
        fpower = fi ** (-13.0 / 6.0)
        for j in ti.static(range(5)):
            Ab[row_idx, j] = (-7.0 / 6.0 + j) * fpower
            fpower *= fi

        self.int_ansatz_coeffs = gauss_elimination(Ab)

    @ti.func
    def update_amplitude_coefficients(
        self,
        pn_coefficients_21: ti.template(),
        source_params: ti.template(),
    ):
        self.common_factor = 2.0 ** (-7.0 / 6.0) * tm.sqrt(
            2.0 * source_params.eta / 3.0 / useful_powers_pi.third
        )
        self._set_joint_frequencies(
            1.0,
            source_params.f_MECO_lm["21"],
            source_params.QNM_freqs_lm["21"],
            source_params,
        )
        self._set_inspiral_coefficients(pn_coefficients_21, source_params)
        self._set_merge_ringdown_coefficients(
            source_params.QNM_freqs_lm["21"], source_params
        )
        self._set_intermediate_coefficients(pn_coefficients_21, source_params)


@sub_struct_from(AmplitudeCoefficientsHighModesBase)
class AmplitudeCoefficientsMode33:
    """ """

    @ti.func
    def _ins_fit_v0(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            source_params.delta_chi_half
            * source_params.eta_pow5
            * (
                155.1434307076563
                + 26.852777193715088 * source_params.chi_PN_hat
                + 1.4157230717300835 * source_params.chi_PN_hat_pow2
            )
            + source_params.delta_chi_half
            * source_params.delta
            * (
                6.296698171560171 * source_params.eta
                + 15.81328761563562 * source_params.eta_pow2
                - 141.85538063933927 * source_params.eta_pow3
            )
            * source_params.eta_sqrt
            + source_params.delta
            * (
                20.94372147101354
                + 68.14577638017842 * source_params.eta
                - 898.470298591732 * source_params.eta_pow2
                + 4598.64854748635 * source_params.eta_pow3
                - 8113.199260593833 * source_params.eta_pow4
            )
            * source_params.eta_sqrt
            + source_params.delta_chi_half
            * source_params.delta
            * (
                29.221863857271703 * source_params.eta
                - 348.1658322276406 * source_params.eta_pow2
                + 965.4670353331536 * source_params.eta_pow3
            )
            * source_params.chi_PN_hat
            * source_params.eta_sqrt
            + source_params.delta
            * source_params.chi_PN_hat
            * (
                -9.753610761811967
                * (
                    1.7819678168496158
                    - 44.07982999150369 * source_params.eta
                    + 750.8933447725581 * source_params.eta_pow2
                    - 5652.44754829634 * source_params.eta_pow3
                    + 19794.855873435758 * source_params.eta_pow4
                    - 26407.40988450443 * source_params.eta_pow5
                )
                + 0.014210376114848208
                * (
                    -196.97328616330392
                    + 7264.159472864562 * source_params.eta
                    - 125763.47850622259 * source_params.eta_pow2
                    + 1.1458022059130718e6 * source_params.eta_pow3
                    - 4.948175330328345e6 * source_params.eta_pow4
                    + 7.911048294733888e6 * source_params.eta_pow5
                )
                * source_params.chi_PN_hat
                - 0.26859293613553986
                * (
                    -8.029069605349488
                    + 888.7768796633982 * source_params.eta
                    - 16664.276483466252 * source_params.eta_pow2
                    + 128973.72291098491 * source_params.eta_pow3
                    - 462437.2690007375 * source_params.eta_pow4
                    + 639989.1197424605 * source_params.eta_pow5
                )
                * source_params.chi_PN_hat_pow2
            )
            * source_params.eta_sqrt
        )

    @ti.func
    def _ins_fit_v1(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            source_params.delta_chi_half
            * source_params.eta_pow5
            * (
                161.62678370819597
                + 37.141092711336846 * source_params.chi_PN_hat
                - 0.16889712161410445 * source_params.chi_PN_hat_pow2
            )
            + source_params.delta_chi_half
            * source_params.delta
            * (
                3.4895829486899825 * source_params.eta
                + 51.07954458810889 * source_params.eta_pow2
                - 249.71072528701757 * source_params.eta_pow3
            )
            * source_params.eta_sqrt
            + source_params.delta
            * (
                12.501397517602173
                + 35.75290806646574 * source_params.eta
                - 357.6437296928763 * source_params.eta_pow2
                + 1773.8883882162215 * source_params.eta_pow3
                - 3100.2396041211605 * source_params.eta_pow4
            )
            * source_params.eta_sqrt
            + source_params.delta_chi_half
            * source_params.delta
            * (
                13.854211287141906 * source_params.eta
                - 135.54916401086845 * source_params.eta_pow2
                + 327.2467193417936 * source_params.eta_pow3
            )
            * source_params.chi_PN_hat
            * source_params.eta_sqrt
            + source_params.delta
            * source_params.chi_PN_hat
            * (
                -5.2580116732827085
                * (
                    1.7794900975289085
                    - 48.20753331991333 * source_params.eta
                    + 861.1650630146937 * source_params.eta_pow2
                    - 6879.681319382729 * source_params.eta_pow3
                    + 25678.53964955809 * source_params.eta_pow4
                    - 36383.824902258915 * source_params.eta_pow5
                )
                + 0.028627002336747746
                * (
                    -50.57295946557892
                    + 734.7581857539398 * source_params.eta
                    - 2287.0465658878725 * source_params.eta_pow2
                    + 15062.821881048358 * source_params.eta_pow3
                    - 168311.2370167227 * source_params.eta_pow4
                    + 454655.37836367317 * source_params.eta_pow5
                )
                * source_params.chi_PN_hat
                - 0.15528289788512326
                * (
                    -12.738184090548508
                    + 1129.44485109116 * source_params.eta
                    - 25091.14888164863 * source_params.eta_pow2
                    + 231384.03447562453 * source_params.eta_pow3
                    - 953010.5908118751 * source_params.eta_pow4
                    + 1.4516597366230418e6 * source_params.eta_pow5
                )
                * source_params.chi_PN_hat_pow2
            )
            * source_params.eta_sqrt
        )

    @ti.func
    def _ins_fit_v2(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            source_params.delta_chi_half
            * source_params.delta
            * (
                -0.5869777957488564 * source_params.eta
                + 32.65536124256588 * source_params.eta_pow2
                - 110.10276573567405 * source_params.eta_pow3
            )
            + source_params.delta_chi_half
            * source_params.delta
            * (
                3.524800489907584 * source_params.eta
                - 40.26479860265549 * source_params.eta_pow2
                + 113.77466499598913 * source_params.eta_pow3
            )
            * source_params.chi_PN_hat
            + source_params.delta
            * source_params.chi_PN_hat
            * (
                -1.2846335585108297
                * (
                    0.09991079016763821
                    + 1.37856806162599 * source_params.eta
                    + 23.26434219690476 * source_params.eta_pow2
                    - 34.842921754693386 * source_params.eta_pow3
                    - 70.83896459998664 * source_params.eta_pow4
                )
                - 0.03496714763391888
                * (
                    -0.230558571912664
                    + 188.38585449575902 * source_params.eta
                    - 3736.1574640444287 * source_params.eta_pow2
                    + 22714.70643022915 * source_params.eta_pow3
                    - 43221.0453556626 * source_params.eta_pow4
                )
                * source_params.chi_PN_hat
            )
            + source_params.delta_chi_half
            * source_params.eta_pow7
            * (
                2667.3441342894776
                + 47.94869769580204 * source_params.delta_chi_half_pow2
                + 793.5988192446642 * source_params.chi_PN_hat
                + 293.89657731755483 * source_params.chi_PN_hat_pow2
            )
            + source_params.delta
            * (
                5.148353856800232
                + 148.98231189649468 * source_params.eta
                - 2774.5868652930294 * source_params.eta_pow2
                + 29052.156454239772 * source_params.eta_pow3
                - 162498.31493332976 * source_params.eta_pow4
                + 460912.76402476896 * source_params.eta_pow5
                - 521279.50781871413 * source_params.eta_pow6
            )
            * source_params.eta_sqrt
        )

    @ti.func
    def _MRD_fit_v0(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            source_params.delta
            * source_params.eta
            * (
                12.439702602599235
                - 4.436329538596615 * source_params.eta
                + 22.780673360839497 * source_params.eta_pow2
            )
            + source_params.delta
            * source_params.eta
            * (
                source_params.delta_chi_half
                * (
                    -41.04442169938298 * source_params.eta
                    + 502.9246970179746 * source_params.eta_pow2
                    - 1524.2981907688634 * source_params.eta_pow3
                )
                + source_params.delta_chi_half_pow2
                * (
                    32.23960072974939 * source_params.eta
                    - 365.1526474476759 * source_params.eta_pow2
                    + 1020.6734178547847 * source_params.eta_pow3
                )
            )
            + source_params.delta_chi_half
            * source_params.delta
            * source_params.eta
            * (
                -52.85961155799673 * source_params.eta
                + 577.6347407795782 * source_params.eta_pow2
                - 1653.496174539196 * source_params.eta_pow3
            )
            * source_params.S_tot_hat
            + source_params.delta_chi_half
            * source_params.eta_pow5
            * (
                257.33227387984863
                - 34.5074027042393 * source_params.delta_chi_half_pow2
                - 21.836905132600755 * source_params.S_tot_hat
                - 15.81624534976308 * source_params.S_tot_hat_pow2
            )
            + 13.499999999999998
            * source_params.delta
            * source_params.eta
            * source_params.S_tot_hat
            * (
                -0.13654149379906394
                * (
                    2.719687834084113
                    + 29.023992126142304 * source_params.eta
                    - 742.1357702210267 * source_params.eta_pow2
                    + 4142.974510926698 * source_params.eta_pow3
                    - 6167.08766058184 * source_params.eta_pow4
                    - 3591.1757995710486 * source_params.eta_pow5
                )
                - 0.06248535354306988
                * (
                    6.697567446351289
                    - 78.23231700361792 * source_params.eta
                    + 444.79350113344543 * source_params.eta_pow2
                    - 1907.008984765889 * source_params.eta_pow3
                    + 6601.918552659412 * source_params.eta_pow4
                    - 10056.98422430965 * source_params.eta_pow5
                )
                * source_params.S_tot_hat
            )
            / (-3.9329308614837704 + source_params.S_tot_hat)
        )

    @ti.func
    def _MRD_fit_v1(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            source_params.delta
            * source_params.eta
            * (8.425057692276933 + 4.543696144846763 * source_params.eta)
            + source_params.delta_chi_half
            * source_params.delta
            * source_params.eta
            * (
                -32.18860840414171 * source_params.eta
                + 412.07321398189293 * source_params.eta_pow2
                - 1293.422289802462 * source_params.eta_pow3
            )
            + source_params.delta_chi_half
            * source_params.delta
            * source_params.eta
            * (
                -17.18006888428382 * source_params.eta
                + 190.73514518113845 * source_params.eta_pow2
                - 636.4802385540647 * source_params.eta_pow3
            )
            * source_params.S_tot_hat
            + source_params.delta
            * source_params.eta
            * source_params.S_tot_hat
            * (
                0.1206817303851239
                * (
                    8.667503604073314
                    - 144.08062755162752 * source_params.eta
                    + 3188.189172446398 * source_params.eta_pow2
                    - 35378.156133055556 * source_params.eta_pow3
                    + 163644.2192178668 * source_params.eta_pow4
                    - 265581.70142471837 * source_params.eta_pow5
                )
                + 0.08028332044013944
                * (
                    12.632478544060636
                    - 322.95832000179297 * source_params.eta
                    + 4777.45310151897 * source_params.eta_pow2
                    - 35625.58409457366 * source_params.eta_pow3
                    + 121293.97832549023 * source_params.eta_pow4
                    - 148782.33687815256 * source_params.eta_pow5
                )
                * source_params.S_tot_hat
            )
            + source_params.delta_chi_half
            * source_params.eta_pow5
            * (
                159.72371180117415
                - 29.10412708633528 * source_params.delta_chi_half_pow2
                - 1.873799747678187 * source_params.S_tot_hat
                + 41.321480132899524 * source_params.S_tot_hat_pow2
            )
        )

    @ti.func
    def _MRD_fit_v2(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            source_params.delta
            * source_params.eta
            * (2.485784720088995 + 2.321696430921996 * source_params.eta)
            + source_params.delta
            * source_params.eta
            * (
                source_params.delta_chi_half
                * (
                    -10.454376404653859 * source_params.eta
                    + 147.10344302665484 * source_params.eta_pow2
                    - 496.1564538739011 * source_params.eta_pow3
                )
                + source_params.delta_chi_half_pow2
                * (
                    -5.9236399792925996 * source_params.eta
                    + 65.86115501723127 * source_params.eta_pow2
                    - 197.51205149250532 * source_params.eta_pow3
                )
            )
            + source_params.delta_chi_half
            * source_params.delta
            * source_params.eta
            * (
                -10.27418232676514 * source_params.eta
                + 136.5150165348149 * source_params.eta_pow2
                - 473.30988537734174 * source_params.eta_pow3
            )
            * source_params.S_tot_hat
            + source_params.delta_chi_half
            * source_params.eta_pow5
            * (
                32.07819766300362
                - 3.071422453072518 * source_params.delta_chi_half_pow2
                + 35.09131921815571 * source_params.S_tot_hat
                + 67.23189816732847 * source_params.S_tot_hat_pow2
            )
            + 13.499999999999998
            * source_params.delta
            * source_params.eta
            * source_params.S_tot_hat
            * (
                0.0011484326782460882
                * (
                    4.1815722950796035
                    - 172.58816646768219 * source_params.eta
                    + 5709.239330076732 * source_params.eta_pow2
                    - 67368.27397765424 * source_params.eta_pow3
                    + 316864.0589150127 * source_params.eta_pow4
                    - 517034.11171277676 * source_params.eta_pow5
                )
                - 0.009496797093329243
                * (
                    0.9233282181397624
                    - 118.35865186626413 * source_params.eta
                    + 2628.6024206791726 * source_params.eta_pow2
                    - 23464.64953722729 * source_params.eta_pow3
                    + 94309.57566199072 * source_params.eta_pow4
                    - 140089.40725211444 * source_params.eta_pow5
                )
                * source_params.S_tot_hat
            )
            / (
                0.09549360183532198
                - 0.41099904730526465 * source_params.S_tot_hat
                + source_params.S_tot_hat_pow2
            )
        )

    @ti.func
    def _int_fit_v1(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            source_params.delta_chi_half
            * source_params.delta
            * source_params.eta
            * (
                -0.3516244197696068 * source_params.eta
                + 40.425151307421416 * source_params.eta_pow2
                - 148.3162618111991 * source_params.eta_pow3
            )
            + source_params.delta
            * source_params.eta
            * (
                26.998512565991778
                - 146.29035440932105 * source_params.eta
                + 914.5350366065115 * source_params.eta_pow2
                - 3047.513201789169 * source_params.eta_pow3
                + 3996.417635728702 * source_params.eta_pow4
            )
            + source_params.delta_chi_half
            * source_params.delta
            * source_params.eta
            * (
                5.575274516197629 * source_params.eta
                - 44.592719238427094 * source_params.eta_pow2
                + 99.91399033058927 * source_params.eta_pow3
            )
            * source_params.S_tot_hat
            + source_params.delta
            * source_params.eta
            * source_params.S_tot_hat
            * (
                -0.5383304368673182
                * (
                    -7.456619067234563
                    + 129.36947401891433 * source_params.eta
                    - 843.7897535238325 * source_params.eta_pow2
                    + 3507.3655567272644 * source_params.eta_pow3
                    - 9675.194644814854 * source_params.eta_pow4
                    + 11959.83533107835 * source_params.eta_pow5
                )
                - 0.28042799223829407
                * (
                    -6.212827413930676
                    + 266.69059813274475 * source_params.eta
                    - 4241.537539226717 * source_params.eta_pow2
                    + 32634.43965039936 * source_params.eta_pow3
                    - 119209.70783201039 * source_params.eta_pow4
                    + 166056.27237509796 * source_params.eta_pow5
                )
                * source_params.S_tot_hat
            )
            + source_params.delta_chi_half
            * source_params.eta_pow5
            * (
                199.6863414922219
                + 53.36849263931051 * source_params.S_tot_hat
                + 7.650565415855383 * source_params.S_tot_hat_pow2
            )
        )

    @ti.func
    def _int_fit_v2(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            source_params.delta
            * source_params.eta
            * (
                17.42562079069636
                - 28.970875603981295 * source_params.eta
                + 50.726220750178435 * source_params.eta_pow2
            )
            + source_params.delta_chi_half
            * source_params.delta
            * source_params.eta
            * (
                -7.861956897615623 * source_params.eta
                + 93.45476935080045 * source_params.eta_pow2
                - 273.1170921735085 * source_params.eta_pow3
            )
            + source_params.delta_chi_half
            * source_params.delta
            * source_params.eta
            * (
                -0.3265505633310564 * source_params.eta
                - 9.861644053348053 * source_params.eta_pow2
                + 60.38649425562178 * source_params.eta_pow3
            )
            * source_params.S_tot_hat
            + source_params.delta_chi_half
            * source_params.eta_pow5
            * (
                234.13476431269862
                + 51.2153901931183 * source_params.S_tot_hat
                - 10.05114600643587 * source_params.S_tot_hat_pow2
            )
            + source_params.delta
            * source_params.eta
            * source_params.S_tot_hat
            * (
                0.3104472390387834
                * (
                    6.073591341439855
                    + 169.85423386969634 * source_params.eta
                    - 4964.199967099143 * source_params.eta_pow2
                    + 42566.59565666228 * source_params.eta_pow3
                    - 154255.3408672655 * source_params.eta_pow4
                    + 205525.13910847943 * source_params.eta_pow5
                )
                + 0.2295327944679772
                * (
                    19.236275867648594
                    - 354.7914372697625 * source_params.eta
                    + 1876.408148917458 * source_params.eta_pow2
                    + 2404.4151687877525 * source_params.eta_pow3
                    - 41567.07396803811 * source_params.eta_pow4
                    + 79210.33893514868 * source_params.eta_pow5
                )
                * source_params.S_tot_hat
                + 0.30983324991828787
                * (
                    11.302200127272357
                    - 719.9854052004307 * source_params.eta
                    + 13278.047199998868 * source_params.eta_pow2
                    - 104863.50453518033 * source_params.eta_pow3
                    + 376409.2335857397 * source_params.eta_pow4
                    - 504089.07690692553 * source_params.eta_pow5
                )
                * source_params.S_tot_hat_pow2
            )
        )

    @ti.func
    def _int_fit_v3(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            source_params.delta
            * source_params.eta
            * (
                14.555522136327964
                - 12.799844096694798 * source_params.eta
                + 16.79500349318081 * source_params.eta_pow2
            )
            + source_params.delta_chi_half
            * source_params.delta
            * source_params.eta
            * (
                -16.292654447108134 * source_params.eta
                + 190.3516012682791 * source_params.eta_pow2
                - 562.0936797781519 * source_params.eta_pow3
            )
            + source_params.delta_chi_half
            * source_params.delta
            * source_params.eta
            * (
                -7.048898856045782 * source_params.eta
                + 49.941617405768135 * source_params.eta_pow2
                - 73.62033985436068 * source_params.eta_pow3
            )
            * source_params.S_tot_hat
            + source_params.delta_chi_half
            * source_params.eta_pow5
            * (
                263.5151703818307
                + 44.408527093031566 * source_params.S_tot_hat
                + 10.457035444964653 * source_params.S_tot_hat_pow2
            )
            + source_params.delta
            * source_params.eta
            * source_params.S_tot_hat
            * (
                0.4590550434774332
                * (
                    3.0594364612798635
                    + 207.74562213604057 * source_params.eta
                    - 5545.0086137386525 * source_params.eta_pow2
                    + 50003.94075934942 * source_params.eta_pow3
                    - 195187.55422847517 * source_params.eta_pow4
                    + 282064.174913521 * source_params.eta_pow5
                )
                + 0.657748992123043
                * (
                    5.57939137343977
                    - 124.06189543062042 * source_params.eta
                    + 1276.6209573025596 * source_params.eta_pow2
                    - 6999.7659193505915 * source_params.eta_pow3
                    + 19714.675715229736 * source_params.eta_pow4
                    - 20879.999628681435 * source_params.eta_pow5
                )
                * source_params.S_tot_hat
                + 0.3695850566805098
                * (
                    6.077183107132255
                    - 498.95526910874986 * source_params.eta
                    + 10426.348944657859 * source_params.eta_pow2
                    - 91096.64982858274 * source_params.eta_pow3
                    + 360950.6686625352 * source_params.eta_pow4
                    - 534437.8832860565 * source_params.eta_pow5
                )
                * source_params.S_tot_hat_pow2
            )
        )

    @ti.func
    def _int_fit_v4(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            source_params.delta
            * source_params.eta
            * (
                13.312095699772305
                - 7.449975618083432 * source_params.eta
                + 17.098576301150125 * source_params.eta_pow2
            )
            + source_params.delta
            * source_params.eta
            * (
                source_params.delta_chi_half
                * (
                    -31.171150896110156 * source_params.eta
                    + 371.1389274783572 * source_params.eta_pow2
                    - 1103.1917047361735 * source_params.eta_pow3
                )
                + source_params.delta_chi_half_pow2
                * (
                    32.78644599730888 * source_params.eta
                    - 395.15713118955387 * source_params.eta_pow2
                    + 1164.9282236341376 * source_params.eta_pow3
                )
            )
            + source_params.delta_chi_half
            * source_params.delta
            * source_params.eta
            * (
                -46.85669289852532 * source_params.eta
                + 522.3965959942979 * source_params.eta_pow2
                - 1485.5134187612182 * source_params.eta_pow3
            )
            * source_params.S_tot_hat
            + source_params.delta_chi_half
            * source_params.eta_pow5
            * (
                287.90444670305715
                - 21.102665129433042 * source_params.delta_chi_half_pow2
                + 7.635582066682054 * source_params.S_tot_hat
                - 29.471275170013012 * source_params.S_tot_hat_pow2
            )
            + source_params.delta
            * source_params.eta
            * source_params.S_tot_hat
            * (
                0.6893003654021495
                * (
                    3.1014226377197027
                    - 44.83989278653052 * source_params.eta
                    + 565.3767256471909 * source_params.eta_pow2
                    - 4797.429130246123 * source_params.eta_pow3
                    + 19514.812242035154 * source_params.eta_pow4
                    - 27679.226582207506 * source_params.eta_pow5
                )
                + 0.7068016563068026
                * (
                    4.071212304920691
                    - 118.51094098279343 * source_params.eta
                    + 1788.1730303291356 * source_params.eta_pow2
                    - 13485.270489656365 * source_params.eta_pow3
                    + 48603.96661003743 * source_params.eta_pow4
                    - 65658.74746265226 * source_params.eta_pow5
                )
                * source_params.S_tot_hat
                + 0.2181399561677432
                * (
                    -1.6754158383043574
                    + 303.9394443302189 * source_params.eta
                    - 6857.936471898544 * source_params.eta_pow2
                    + 59288.71069769708 * source_params.eta_pow3
                    - 216137.90827404748 * source_params.eta_pow4
                    + 277256.38289831823 * source_params.eta_pow5
                )
                * source_params.S_tot_hat_pow2
            )
        )

    @ti.func
    def update_amplitude_coefficients(
        self,
        pn_coefficients_33: ti.template(),
        source_params: ti.template(),
    ):
        self.common_factor = (2.0 / 3.0) ** (-7.0 / 6.0) * tm.sqrt(
            2.0 * source_params.eta / 3.0 / useful_powers_pi.third
        )
        self._set_joint_frequencies(
            3.0,
            source_params.f_MECO_lm["33"],
            source_params.QNM_freqs_lm["33"],
            source_params,
        )
        self._set_inspiral_coefficients(pn_coefficients_33, source_params)
        self._set_merge_ringdown_coefficients(
            source_params.QNM_freqs_lm["33"], source_params
        )
        self._set_intermediate_coefficients(
            source_params.QNM_freqs_lm["33"], pn_coefficients_33, source_params
        )


@sub_struct_from(AmplitudeCoefficientsHighModesBase)
class AmplitudeCoefficientsMode32:
    """ """

    removed_members = [
        "gamma_1",
        "gamma_2",
        "gamma_3",
        "falloff_gamma_1",
        "falloff_gamma_2",
        "MRD_f_falloff",
    ]

    @ti.func
    def _set_joint_frequencies(
        self,
        m: ti.f64,
        f_MECO_lm: ti.f64,
        source_params: ti.template(),
    ):
        # joint between inspiral and intermediate
        self.ins_f_end = self._get_ins_f_end(m, f_MECO_lm, source_params)
        # joint between intermediate and merge-ringdown
        self.int_f_end = source_params.f_ring - 0.5 * source_params.f_damp

    @ti.func
    def _set_intermediate_coefficients(
        self,
        pn_coefficients_22: ti.template(),
        pn_coefficients_32: ti.template(),
        amplitude_coefficients_22: ti.template(),
        phase_coefficients_22: ti.template(),
        spheroidal_merge_ringdown_32: ti.template(),
        source_params: ti.template(),
    ):
        """
        Require inspiral and merge-ringdown amplitude to set boundaries, can only be
        called after updating inspiral and merge-ringdown coefficients.

        Corresponding to IntermediateAmpVersion=211112, not used for mode 21.
        colloc-points 6;
        colloc-values 8 (values at 6 colloc-points and 2 derivatives at boundaries);
        ansatz-coeffs 8;
        augmented matrix 9x8
        """
        int_colloc_points = self._get_int_colloc_points()
        powers_int_f0 = UsefulPowers()
        powers_int_f0.update(int_colloc_points[0])
        powers_int_f5 = UsefulPowers()
        powers_int_f5.update(int_colloc_points[5])

        int_colloc_values = ti.Vector([0.0] * 8, dt=ti.f64)
        # left boundary
        int_colloc_values[0] = self._inspiral_amplitude(
            pn_coefficients_32, powers_int_f0
        )
        # fit values at collocation points
        int_colloc_values[1] = self._int_fit_v1(source_params)
        int_colloc_values[2] = self._int_fit_v2(source_params)
        int_colloc_values[3] = self._int_fit_v3(source_params)
        int_colloc_values[4] = self._int_fit_v4(source_params)
        # right boundary
        int_colloc_values[5] = self._merge_ringdown_amplitude(
            pn_coefficients_22,
            amplitude_coefficients_22,
            phase_coefficients_22,
            spheroidal_merge_ringdown_32,
            source_params,
            powers_int_f5,
        )
        # derivative at the left boundary
        int_colloc_values[6] = self._inspiral_d_amplitude(
            pn_coefficients_32, powers_int_f0
        )
        # derivative at the right boundary
        int_colloc_values[7] = self._merge_ringdown_d_amplitude(
            pn_coefficients_22,
            amplitude_coefficients_22,
            phase_coefficients_22,
            spheroidal_merge_ringdown_32,
            source_params,
            powers_int_f5,
        )
        print(int_colloc_points)
        print(int_colloc_values)

        # set the augmented matrix
        Ab = ti.Matrix([[0.0] * 9 for _ in range(8)], dt=ti.f64)
        row_idx = 0
        for i in ti.static(range(6)):
            # set the value at the collocation point
            Ab[row_idx, 8] = int_colloc_values[row_idx]
            # set the coefficient matrix of frequency powers
            # [1, fi, fi^2, fi^3, fi^4, fi^5, fi^6, fi^7] * fi^(-7/6)
            fi = int_colloc_points[i]
            fpower = fi ** (-7.0 / 6.0)
            for j in ti.static(range(8)):
                Ab[row_idx, j] = fpower
                fpower *= fi
            # next row
            row_idx += 1
        # for the derivatives at 2 boundaries
        for i in ti.static([0, 5]):
            Ab[row_idx, 8] = int_colloc_values[row_idx]
            # set the coefficient matrix of frequency powers for derivative
            # [ (-7/6)fi_-1,  (-7/6+1),     (-7/6+2)fi,   (-7/6+3)fi^2,
            #   (-7/6+4)fi^3, (-7/6+5)fi^4, (-7/6+6)fi^5, (-7/6+7)fi^6 ] * fi^(-7/6)
            fi = int_colloc_points[i]
            fpower = fi ** (-13.0 / 6.0)
            for j in ti.static(range(8)):
                Ab[row_idx, j] = (-7.0 / 6.0 + j) * fpower
                fpower *= fi
            # next row
            row_idx += 1

        self.int_ansatz_coeffs = gauss_elimination(Ab)

    @ti.func
    def _merge_ringdown_amplitude(
        self,
        pn_coefficients_22: ti.template(),
        amplitude_coefficients_22: ti.template(),
        phase_coefficients_22: ti.template(),
        spheroidal_merge_ringdown_32: ti.template(),
        source_params: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> ti.f64:
        amp_22 = (
            amplitude_coefficients_22._merge_ringdown_amplitude(
                source_params, powers_of_Mf
            )
            / powers_of_Mf.seven_sixths
            * amplitude_coefficients_22.common_factor
        )
        phi_22 = phase_coefficients_22.compute_phase(
            pn_coefficients_22, source_params, powers_of_Mf
        )
        h22 = amp_22 * tm.cexp(ComplexNumber([0.0, 1.0]) * phi_22)
        return self._merge_ringdown_amplitude_core(
            h22,
            spheroidal_merge_ringdown_32,
            source_params.QNM_freqs_lm["32"],
            powers_of_Mf,
        )

    @ti.func
    def _merge_ringdown_amplitude_core(
        self,
        h22: ti.template(),
        spheroidal_merge_ringdown_32: ti.template(),
        QNM_freqs_32: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> ti.f64:
        h32 = spheroidal_merge_ringdown_32.spherical_h32(
            h22, QNM_freqs_32, powers_of_Mf
        )
        return tm.length(h32)

    @ti.func
    def _merge_ringdown_d_amplitude(
        self,
        pn_coefficients_22: ti.template(),
        amplitude_coefficients_22: ti.template(),
        phase_coefficients_22: ti.template(),
        spheroidal_merge_ringdown_32: ti.template(),
        source_params: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> ti.f64:

        powers_Mf_left2 = UsefulPowers()
        powers_Mf_left = UsefulPowers()
        powers_Mf_right = UsefulPowers()
        powers_Mf_right2 = UsefulPowers()

        step = 1e-9
        powers_Mf_left2.update(powers_of_Mf.one - 2.0 * step)
        powers_Mf_left.update(powers_of_Mf.one - step)
        powers_Mf_right.update(powers_of_Mf.one + step)
        powers_Mf_right2.update(powers_of_Mf.one + 2.0 * step)

        amp_left2 = self._merge_ringdown_amplitude(
            pn_coefficients_22,
            amplitude_coefficients_22,
            phase_coefficients_22,
            spheroidal_merge_ringdown_32,
            source_params,
            powers_Mf_left2,
        )
        amp_left = self._merge_ringdown_amplitude(
            pn_coefficients_22,
            amplitude_coefficients_22,
            phase_coefficients_22,
            spheroidal_merge_ringdown_32,
            source_params,
            powers_Mf_left,
        )
        amp_right = self._merge_ringdown_amplitude(
            pn_coefficients_22,
            amplitude_coefficients_22,
            phase_coefficients_22,
            spheroidal_merge_ringdown_32,
            source_params,
            powers_Mf_right,
        )
        amp_right2 = self._merge_ringdown_amplitude(
            pn_coefficients_22,
            amplitude_coefficients_22,
            phase_coefficients_22,
            spheroidal_merge_ringdown_32,
            source_params,
            powers_Mf_right2,
        )
        return (amp_left2 - 8.0 * amp_left + 8.0 * amp_right - amp_right2) / (
            12.0 * step
        )

    @ti.func
    def _ins_fit_v0(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            (
                source_params.delta_chi_half
                * source_params.delta
                * (
                    -0.739317114582042 * source_params.eta
                    - 47.473246070362634 * source_params.eta_pow2
                    + 278.9717709112207 * source_params.eta_pow3
                    - 566.6420939162068 * source_params.eta_pow4
                )
                + source_params.delta_chi_half_pow2
                * (
                    -0.5873680378268906 * source_params.eta
                    + 6.692187014925888 * source_params.eta_pow2
                    - 24.37776782232888 * source_params.eta_pow3
                    + 23.783684827838247 * source_params.eta_pow4
                )
            )
            * source_params.eta_sqrt
            + (
                3.2940434453819694
                + 4.94285331708559 * source_params.eta
                - 343.3143244815765 * source_params.eta_pow2
                + 3585.9269057886418 * source_params.eta_pow3
                - 19279.186145681153 * source_params.eta_pow4
                + 51904.91007211022 * source_params.eta_pow5
                - 55436.68857586653 * source_params.eta_pow6
            )
            * source_params.eta_sqrt
            + source_params.delta_chi_half
            * source_params.delta
            * (
                12.488240781993923 * source_params.eta
                - 209.32038774208385 * source_params.eta_pow2
                + 1160.9833883184604 * source_params.eta_pow3
                - 2069.5349737049073 * source_params.eta_pow4
            )
            * source_params.chi_PN_hat
            * source_params.eta_sqrt
            + source_params.chi_PN_hat
            * (
                0.6343034651912586
                * (
                    -2.5844888818001737
                    + 78.98200041834092 * source_params.eta
                    - 1087.6241783616488 * source_params.eta_pow2
                    + 7616.234910399297 * source_params.eta_pow3
                    - 24776.529123239357 * source_params.eta_pow4
                    + 30602.210950069973 * source_params.eta_pow5
                )
                - 0.062088720220899465
                * (
                    6.5586380356588565
                    + 36.01386705325694 * source_params.eta
                    - 3124.4712274775407 * source_params.eta_pow2
                    + 33822.437731298516 * source_params.eta_pow3
                    - 138572.93700180828 * source_params.eta_pow4
                    + 198366.10615196894 * source_params.eta_pow5
                )
                * source_params.chi_PN_hat
            )
            * source_params.eta_sqrt
        )

    @ti.func
    def _ins_fit_v1(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            (
                source_params.delta_chi_half_pow2
                * (
                    -0.03940151060321499 * source_params.eta
                    + 1.9034209537174116 * source_params.eta_pow2
                    - 8.78587250202154 * source_params.eta_pow3
                )
                + source_params.delta_chi_half
                * source_params.delta
                * (
                    -1.704299788495861 * source_params.eta
                    - 4.923510922214181 * source_params.eta_pow2
                    + 0.36790005839460627 * source_params.eta_pow3
                )
            )
            * source_params.eta_sqrt
            + (
                2.2911849711339123
                - 5.1846950040514335 * source_params.eta
                + 60.10368251688146 * source_params.eta_pow2
                - 1139.110227749627 * source_params.eta_pow3
                + 7970.929280907627 * source_params.eta_pow4
                - 25472.73682092519 * source_params.eta_pow5
                + 30950.67053883646 * source_params.eta_pow6
            )
            * source_params.eta_sqrt
            + source_params.chi_PN_hat
            * (
                0.7718201508695763
                * (
                    -1.3012906461000349
                    + 26.432880113146012 * source_params.eta
                    - 186.5001124789369 * source_params.eta_pow2
                    + 712.9101229418721 * source_params.eta_pow3
                    - 970.2126139442341 * source_params.eta_pow4
                )
                + 0.04832734931068797
                * (
                    -5.9999628512498315
                    + 78.98681284391004 * source_params.eta
                    + 1.8360177574514709 * source_params.eta_pow2
                    - 2537.636347529708 * source_params.eta_pow3
                    + 6858.003573909322 * source_params.eta_pow4
                )
                * source_params.chi_PN_hat
            )
            * source_params.eta_sqrt
        )

    @ti.func
    def _ins_fit_v2(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            (
                source_params.delta_chi_half_pow2
                * (
                    -0.6358511175987503 * source_params.eta
                    + 5.555088747533164 * source_params.eta_pow2
                    - 14.078156877577733 * source_params.eta_pow3
                )
                + source_params.delta_chi_half
                * source_params.delta
                * (
                    0.23205448591711159 * source_params.eta
                    - 19.46049432345157 * source_params.eta_pow2
                    + 36.20685853857613 * source_params.eta_pow3
                )
            )
            * source_params.eta_sqrt
            + (
                1.1525594672495008
                + 7.380126197972549 * source_params.eta
                - 17.51265776660515 * source_params.eta_pow2
                - 976.9940395257111 * source_params.eta_pow3
                + 8880.536804741967 * source_params.eta_pow4
                - 30849.228936891763 * source_params.eta_pow5
                + 38785.53683146884 * source_params.eta_pow6
            )
            * source_params.eta_sqrt
            + source_params.delta_chi_half
            * source_params.delta
            * (
                1.904350804857431 * source_params.eta
                - 25.565242391371093 * source_params.eta_pow2
                + 80.67120303906654 * source_params.eta_pow3
            )
            * source_params.chi_PN_hat
            * source_params.eta_sqrt
            + source_params.chi_PN_hat
            * (
                0.785171689871352
                * (
                    -0.4634745514643032
                    + 18.70856733065619 * source_params.eta
                    - 167.9231114864569 * source_params.eta_pow2
                    + 744.7699462372949 * source_params.eta_pow3
                    - 1115.008825153004 * source_params.eta_pow4
                )
                + 0.13469300326662165
                * (
                    -2.7311391326835133
                    + 72.17373498208947 * source_params.eta
                    - 483.7040402103785 * source_params.eta_pow2
                    + 1136.8367114738041 * source_params.eta_pow3
                    - 472.02962341590774 * source_params.eta_pow4
                )
                * source_params.chi_PN_hat
            )
            * source_params.eta_sqrt
        )

    @ti.func
    def _int_fit_v1(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            (
                source_params.delta_chi_half_pow2
                * (
                    -0.2341404256829785 * source_params.eta
                    + 2.606326837996192 * source_params.eta_pow2
                    - 8.68296921440857 * source_params.eta_pow3
                )
                + source_params.delta_chi_half
                * source_params.delta
                * (
                    0.5454562486736877 * source_params.eta
                    - 25.19759222940851 * source_params.eta_pow2
                    + 73.40268975811729 * source_params.eta_pow3
                )
            )
            * source_params.eta_sqrt
            + source_params.delta_chi_half
            * source_params.delta
            * (
                0.4422257616009941 * source_params.eta
                - 8.490112284851655 * source_params.eta_pow2
                + 32.22238925527844 * source_params.eta_pow3
            )
            * source_params.chi_PN_hat
            * source_params.eta_sqrt
            + source_params.chi_PN_hat
            * (
                0.7067243321652764
                * (
                    0.12885110296881636
                    + 9.608999847549535 * source_params.eta
                    - 85.46581740280585 * source_params.eta_pow2
                    + 325.71940024255775 * source_params.eta_pow3
                    + 175.4194342269804 * source_params.eta_pow4
                    - 1929.9084724384807 * source_params.eta_pow5
                )
                + 0.1540566313813899
                * (
                    -0.3261041495083288
                    + 45.55785402900492 * source_params.eta
                    - 827.591235943271 * source_params.eta_pow2
                    + 7184.647314370326 * source_params.eta_pow3
                    - 28804.241518798244 * source_params.eta_pow4
                    + 43309.69769878964 * source_params.eta_pow5
                )
                * source_params.chi_PN_hat
            )
            * source_params.eta_sqrt
            + (
                480.0434256230109 * source_params.eta
                + 25346.341240810478 * source_params.eta_pow2
                - 99873.4707358776 * source_params.eta_pow3
                + 106683.98302194536 * source_params.eta_pow4
            )
            * source_params.eta_sqrt
            / (
                1.0
                + 1082.6574834474493 * source_params.eta
                + 10083.297670051445 * source_params.eta_pow2
            )
        )

    @ti.func
    def _int_fit_v2(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            source_params.eta
            * (
                source_params.delta_chi_half_pow2
                * (
                    -4.175680729484314 * source_params.eta
                    + 47.54281549129226 * source_params.eta_pow2
                    - 128.88334273588077 * source_params.eta_pow3
                )
                + source_params.delta_chi_half
                * source_params.delta
                * (
                    -0.18274358639599947 * source_params.eta
                    - 71.01128541687838 * source_params.eta_pow2
                    + 208.07105580635888 * source_params.eta_pow3
                )
            )
            + source_params.eta
            * (
                4.760999387359598
                - 38.57900689641654 * source_params.eta
                + 456.2188780552874 * source_params.eta_pow2
                - 4544.076411013166 * source_params.eta_pow3
                + 24956.9592553473 * source_params.eta_pow4
                - 69430.10468748478 * source_params.eta_pow5
                + 77839.74180254337 * source_params.eta_pow6
            )
            + source_params.delta_chi_half
            * source_params.delta
            * source_params.eta
            * (
                1.2198776533959694 * source_params.eta
                - 26.816651899746475 * source_params.eta_pow2
                + 68.72798751937934 * source_params.eta_pow3
            )
            * source_params.S_tot_hat
            + source_params.eta
            * source_params.S_tot_hat
            * (
                1.5098291294292217
                * (
                    0.4844667556328104
                    + 9.848766999273414 * source_params.eta
                    - 143.66427232396376 * source_params.eta_pow2
                    + 856.9917885742416 * source_params.eta_pow3
                    - 1633.3295758142904 * source_params.eta_pow4
                )
                + 0.32413108737204144
                * (
                    2.835358206961064
                    - 62.37317183581803 * source_params.eta
                    + 761.6103793011912 * source_params.eta_pow2
                    - 3811.5047139343505 * source_params.eta_pow3
                    + 6660.304740652403 * source_params.eta_pow4
                )
                * source_params.S_tot_hat
            )
        )

    @ti.func
    def _int_fit_v3(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            3.881450518842405 * source_params.eta
            - 12.580316392558837 * source_params.eta_pow2
            + 1.7262466525848588 * source_params.eta_pow3
            + source_params.delta_chi_half_pow2
            * (
                -7.065118823041031 * source_params.eta_pow2
                + 77.97950589523865 * source_params.eta_pow3
                - 203.65975422378446 * source_params.eta_pow4
            )
            - 58.408542930248046 * source_params.eta_pow4
            + source_params.delta_chi_half
            * source_params.delta
            * (
                1.924723094787216 * source_params.eta_pow2
                - 90.92716917757797 * source_params.eta_pow3
                + 387.00162600306226 * source_params.eta_pow4
            )
            + 403.5748987560612 * source_params.eta_pow5
            + source_params.delta_chi_half
            * source_params.delta
            * (
                -0.2566958540737833 * source_params.eta_pow2
                + 14.488550203412675 * source_params.eta_pow3
                - 26.46699529970884 * source_params.eta_pow4
            )
            * source_params.chi_PN_hat
            + source_params.chi_PN_hat
            * (
                0.3650871458400108
                * (
                    71.57390929624825 * source_params.eta_pow2
                    - 994.5272351916166 * source_params.eta_pow3
                    + 6734.058809060536 * source_params.eta_pow4
                    - 18580.859291282686 * source_params.eta_pow5
                    + 16001.318492586077 * source_params.eta_pow6
                )
                + 0.0960146077440495
                * (
                    451.74917589707513 * source_params.eta_pow2
                    - 9719.470997418284 * source_params.eta_pow3
                    + 83403.5743434538 * source_params.eta_pow4
                    - 318877.43061174755 * source_params.eta_pow5
                    + 451546.88775684836 * source_params.eta_pow6
                )
                * source_params.chi_PN_hat
                - 0.03985156529181297
                * (
                    -304.92981902871617 * source_params.eta_pow2
                    + 3614.518459296278 * source_params.eta_pow3
                    - 7859.4784979916085 * source_params.eta_pow4
                    - 46454.57664737511 * source_params.eta_pow5
                    + 162398.81483375572 * source_params.eta_pow6
                )
                * source_params.chi_PN_hat_pow2
            )
        )

    @ti.func
    def _int_fit_v4(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            source_params.eta
            * (
                source_params.delta_chi_half_pow2
                * (
                    -8.572797326909152 * source_params.eta
                    + 92.95723645687826 * source_params.eta_pow2
                    - 236.2438921965621 * source_params.eta_pow3
                )
                + source_params.delta_chi_half
                * source_params.delta
                * (
                    6.674358856924571 * source_params.eta
                    - 171.4826985994883 * source_params.eta_pow2
                    + 645.2760206304703 * source_params.eta_pow3
                )
            )
            + source_params.eta
            * (
                3.921660532875504
                - 16.57299637423352 * source_params.eta
                + 25.254017911686333 * source_params.eta_pow2
                - 143.41033155133266 * source_params.eta_pow3
                + 692.926425981414 * source_params.eta_pow4
            )
            + source_params.delta_chi_half
            * source_params.delta
            * source_params.eta
            * (
                -3.582040878719185 * source_params.eta
                + 57.75888914133383 * source_params.eta_pow2
                - 144.21651114700492 * source_params.eta_pow3
            )
            * source_params.S_tot_hat
            + source_params.eta
            * source_params.S_tot_hat
            * (
                1.242750265695504
                * (
                    -0.522172424518215
                    + 25.168480118950065 * source_params.eta
                    - 303.5223688400309 * source_params.eta_pow2
                    + 1858.1518762309654 * source_params.eta_pow3
                    - 3797.3561904195085 * source_params.eta_pow4
                )
                + 0.2927045241764365
                * (
                    0.5056957789079993
                    - 15.488754837330958 * source_params.eta
                    + 471.64047356915603 * source_params.eta_pow2
                    - 3131.5783196211587 * source_params.eta_pow3
                    + 6097.887891566872 * source_params.eta_pow4
                )
                * source_params.S_tot_hat
            )
        )

    @ti.func
    def update_amplitude_coefficients(
        self,
        pn_coefficients_22: ti.template(),
        pn_coefficients_32: ti.template(),
        amplitude_coefficients_22,
        phase_coefficients_22,
        spheroidal_merge_ringdown_32: ti.template(),
        source_params: ti.template(),
    ):
        self.common_factor = tm.sqrt(
            2.0 * source_params.eta / 3.0 / useful_powers_pi.third
        )
        self._set_joint_frequencies(
            2.0,
            source_params.f_MECO_lm["32"],
            source_params,
        )
        self._set_inspiral_coefficients(pn_coefficients_32, source_params)
        self._set_intermediate_coefficients(
            pn_coefficients_22,
            pn_coefficients_32,
            amplitude_coefficients_22,
            phase_coefficients_22,
            spheroidal_merge_ringdown_32,
            source_params,
        )


@sub_struct_from(AmplitudeCoefficientsHighModesBase)
class AmplitudeCoefficientsMode44:

    @ti.func
    def _ins_fit_v0(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            (
                source_params.delta_chi_half
                * source_params.delta
                * (
                    0.5697308729057493 * source_params.eta
                    + 8.895576813118867 * source_params.eta_pow2
                    - 34.98399465240273 * source_params.eta_pow3
                )
                + source_params.delta_chi_half_pow2
                * (
                    1.6370346538130884 * source_params.eta
                    - 14.597095790380884 * source_params.eta_pow2
                    + 33.182723737396294 * source_params.eta_pow3
                )
            )
            * source_params.eta_sqrt
            + (
                5.2601381002242595
                - 3.557926105832778 * source_params.eta
                - 138.9749850448088 * source_params.eta_pow2
                + 603.7453704122706 * source_params.eta_pow3
                - 923.5495700703648 * source_params.eta_pow4
            )
            * source_params.eta_sqrt
            + source_params.chi_PN_hat
            * (
                -0.41839636169678796
                * (
                    5.143510231379954
                    + 104.62892421207803 * source_params.eta
                    - 4232.508174045782 * source_params.eta_pow2
                    + 50694.024801783446 * source_params.eta_pow3
                    - 283097.33358214336 * source_params.eta_pow4
                    + 758333.2655404843 * source_params.eta_pow5
                    - 788783.0559069642 * source_params.eta_pow6
                )
                - 0.05653522061311774
                * (
                    5.605483124564013
                    + 694.00652410087 * source_params.eta
                    - 17551.398321516353 * source_params.eta_pow2
                    + 165236.6480734229 * source_params.eta_pow3
                    - 761661.9645651339 * source_params.eta_pow4
                    + 1.7440315410044065e6 * source_params.eta_pow5
                    - 1.6010489769238676e6 * source_params.eta_pow6
                )
                * source_params.chi_PN_hat
                - 0.023693246676754775
                * (
                    16.437107575918503
                    - 2911.2154288136217 * source_params.eta
                    + 89338.32554683842 * source_params.eta_pow2
                    - 1.0803340811860575e6 * source_params.eta_pow3
                    + 6.255666490084672e6 * source_params.eta_pow4
                    - 1.7434160932177313e7 * source_params.eta_pow5
                    + 1.883460394974573e7 * source_params.eta_pow6
                )
                * source_params.chi_PN_hat_pow2
            )
            * source_params.eta_sqrt
        )

    @ti.func
    def _ins_fit_v1(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            (
                source_params.delta_chi_half_pow2
                * (
                    -0.8318312659717388 * source_params.eta
                    + 7.6541168007977864 * source_params.eta_pow2
                    - 16.648660653220123 * source_params.eta_pow3
                )
                + source_params.delta_chi_half
                * source_params.delta
                * (
                    2.214478316304753 * source_params.eta
                    - 7.028104574328955 * source_params.eta_pow2
                    + 5.56587823143958 * source_params.eta_pow3
                )
            )
            * source_params.eta_sqrt
            + (
                3.173191054680422
                + 6.707695566702527 * source_params.eta
                - 155.22519772642607 * source_params.eta_pow2
                + 604.0067075996933 * source_params.eta_pow3
                - 876.5048298377644 * source_params.eta_pow4
            )
            * source_params.eta_sqrt
            + source_params.delta_chi_half
            * source_params.delta
            * (
                4.749663394334708 * source_params.eta
                - 42.62996105525792 * source_params.eta_pow2
                + 97.01712147349483 * source_params.eta_pow3
            )
            * source_params.chi_PN_hat
            * source_params.eta_sqrt
            + source_params.chi_PN_hat
            * (
                -0.2627203100303006
                * (
                    6.460396349297595
                    - 52.82425783851536 * source_params.eta
                    - 552.1725902144143 * source_params.eta_pow2
                    + 12546.255587592654 * source_params.eta_pow3
                    - 81525.50289542897 * source_params.eta_pow4
                    + 227254.37897941095 * source_params.eta_pow5
                    - 234487.3875219032 * source_params.eta_pow6
                )
                - 0.008424003742397579
                * (
                    -109.26773035716548
                    + 15514.571912666677 * source_params.eta
                    - 408022.6805482195 * source_params.eta_pow2
                    + 4.620165968920881e6 * source_params.eta_pow3
                    - 2.6446950627957724e7 * source_params.eta_pow4
                    + 7.539643948937692e7 * source_params.eta_pow5
                    - 8.510662871580401e7 * source_params.eta_pow6
                )
                * source_params.chi_PN_hat
                - 0.008830881730801855
                * (
                    -37.49992494976597
                    + 1359.7883958101172 * source_params.eta
                    - 23328.560285901796 * source_params.eta_pow2
                    + 260027.4121353132 * source_params.eta_pow3
                    - 1.723865744472182e6 * source_params.eta_pow4
                    + 5.858455766230802e6 * source_params.eta_pow5
                    - 7.756341721552802e6 * source_params.eta_pow6
                )
                * source_params.chi_PN_hat_pow2
                - 0.027167813927224657
                * (
                    34.281932237450256
                    - 3312.7658728016568 * source_params.eta
                    + 84126.14531363266 * source_params.eta_pow2
                    - 956052.0170024392 * source_params.eta_pow3
                    + 5.570748509263883e6 * source_params.eta_pow4
                    - 1.6270212243584689e7 * source_params.eta_pow5
                    + 1.8855858173287075e7 * source_params.eta_pow6
                )
                * source_params.chi_PN_hat_pow3
            )
            * source_params.eta_sqrt
        )

    @ti.func
    def _ins_fit_v2(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            (
                source_params.delta_chi_half
                * source_params.delta
                * (
                    1.4739380748149558 * source_params.eta
                    + 0.06541707987699942 * source_params.eta_pow2
                    - 9.473290540936633 * source_params.eta_pow3
                )
                + source_params.delta_chi_half_pow2
                * (
                    -0.3640838331639651 * source_params.eta
                    + 3.7369795937033756 * source_params.eta_pow2
                    - 8.709159662885131 * source_params.eta_pow3
                )
            )
            * source_params.eta_sqrt
            + (
                1.7335503724888923
                + 12.656614578053683 * source_params.eta
                - 139.6610487470118 * source_params.eta_pow2
                + 456.78649322753824 * source_params.eta_pow3
                - 599.2709938848282 * source_params.eta_pow4
            )
            * source_params.eta_sqrt
            + source_params.delta_chi_half
            * source_params.delta
            * (
                2.3532739003216254 * source_params.eta
                - 21.37216554136868 * source_params.eta_pow2
                + 53.35003268489743 * source_params.eta_pow3
            )
            * source_params.chi_PN_hat
            * source_params.eta_sqrt
            + source_params.chi_PN_hat
            * (
                -0.15782329022461472
                * (
                    6.0309399412954345
                    - 229.16361598098678 * source_params.eta
                    + 3777.477006415653 * source_params.eta_pow2
                    - 31109.307191210424 * source_params.eta_pow3
                    + 139319.8239886073 * source_params.eta_pow4
                    - 324891.4001578353 * source_params.eta_pow5
                    + 307714.3954026392 * source_params.eta_pow6
                )
                - 0.03050157254864058
                * (
                    4.232861441291087
                    + 1609.4251694451375 * source_params.eta
                    - 51213.27604422822 * source_params.eta_pow2
                    + 612317.1751155312 * source_params.eta_pow3
                    - 3.5589766538499263e6 * source_params.eta_pow4
                    + 1.0147654212772278e7 * source_params.eta_pow5
                    - 1.138861230369246e7 * source_params.eta_pow6
                )
                * source_params.chi_PN_hat
                - 0.026407497690308382
                * (
                    -17.184685557542196
                    + 744.4743953122965 * source_params.eta
                    - 10494.512487701073 * source_params.eta_pow2
                    + 66150.52694069289 * source_params.eta_pow3
                    - 184787.79377504133 * source_params.eta_pow4
                    + 148102.4257785174 * source_params.eta_pow5
                    + 128167.89151782403 * source_params.eta_pow6
                )
                * source_params.chi_PN_hat_pow2
            )
            * source_params.eta_sqrt
        )

    @ti.func
    def _MRD_fit_v0(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            source_params.eta
            * (
                source_params.delta_chi_half
                * source_params.delta
                * (
                    -8.51952446214978 * source_params.eta
                    + 117.76530248141987 * source_params.eta_pow2
                    - 297.2592736781142 * source_params.eta_pow3
                )
                + source_params.delta_chi_half_pow2
                * (
                    -0.2750098647982238 * source_params.eta
                    + 4.456900599347149 * source_params.eta_pow2
                    - 8.017569928870929 * source_params.eta_pow3
                )
            )
            + source_params.eta
            * (
                5.635069974807398
                - 33.67252878543393 * source_params.eta
                + 287.9418482197136 * source_params.eta_pow2
                - 3514.3385364216438 * source_params.eta_pow3
                + 25108.811524802128 * source_params.eta_pow4
                - 98374.18361532023 * source_params.eta_pow5
                + 158292.58792484726 * source_params.eta_pow6
            )
            + source_params.eta
            * source_params.S_tot_hat
            * (
                -0.4360849737360132
                * (
                    -0.9543114627170375
                    - 58.70494649755802 * source_params.eta
                    + 1729.1839588870455 * source_params.eta_pow2
                    - 16718.425586396803 * source_params.eta_pow3
                    + 71236.86532610047 * source_params.eta_pow4
                    - 111910.71267453219 * source_params.eta_pow5
                )
                - 0.024861802943501172
                * (
                    -52.25045490410733
                    + 1585.462602954658 * source_params.eta
                    - 15866.093368857853 * source_params.eta_pow2
                    + 35332.328181283 * source_params.eta_pow3
                    + 168937.32229060197 * source_params.eta_pow4
                    - 581776.5303770923 * source_params.eta_pow5
                )
                * source_params.S_tot_hat
                + 0.005856387555754387
                * (
                    186.39698091707513
                    - 9560.410655118145 * source_params.eta
                    + 156431.3764198244 * source_params.eta_pow2
                    - 1.0461268207440731e6 * source_params.eta_pow3
                    + 3.054333578686424e6 * source_params.eta_pow4
                    - 3.2369858387064277e6 * source_params.eta_pow5
                )
                * source_params.S_tot_hat_pow2
            )
        )

    @ti.func
    def _MRD_fit_v1(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            source_params.eta
            * (
                source_params.delta_chi_half
                * source_params.delta
                * (
                    -2.861653255976984 * source_params.eta
                    + 50.50227103211222 * source_params.eta_pow2
                    - 123.94152825700999 * source_params.eta_pow3
                )
                + source_params.delta_chi_half_pow2
                * (
                    2.9415751419018865 * source_params.eta
                    - 28.79779545444817 * source_params.eta_pow2
                    + 72.40230240887851 * source_params.eta_pow3
                )
            )
            + source_params.eta
            * (
                3.2461722686239307
                + 25.15310593958783 * source_params.eta
                - 792.0167314124681 * source_params.eta_pow2
                + 7168.843978909433 * source_params.eta_pow3
                - 30595.4993786313 * source_params.eta_pow4
                + 49148.57065911245 * source_params.eta_pow5
            )
            + source_params.eta
            * source_params.S_tot_hat
            * (
                -0.23311779185707152
                * (
                    -1.0795711755430002
                    - 20.12558747513885 * source_params.eta
                    + 1163.9107546486134 * source_params.eta_pow2
                    - 14672.23221502075 * source_params.eta_pow3
                    + 73397.72190288734 * source_params.eta_pow4
                    - 127148.27131388368 * source_params.eta_pow5
                )
                + 0.025805905356653
                * (
                    11.929946153728276
                    + 350.93274421955806 * source_params.eta
                    - 14580.02701600596 * source_params.eta_pow2
                    + 174164.91607515427 * source_params.eta_pow3
                    - 819148.9390278616 * source_params.eta_pow4
                    + 1.3238624538095295e6 * source_params.eta_pow5
                )
                * source_params.S_tot_hat
                + 0.019740635678180102
                * (
                    -7.046295936301379
                    + 1535.781942095697 * source_params.eta
                    - 27212.67022616794 * source_params.eta_pow2
                    + 201981.0743810629 * source_params.eta_pow3
                    - 696891.1349708183 * source_params.eta_pow4
                    + 910729.0219043035 * source_params.eta_pow5
                )
                * source_params.S_tot_hat_pow2
            )
        )

    @ti.func
    def _MRD_fit_v2(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            source_params.eta
            * (
                source_params.delta_chi_half
                * source_params.delta
                * (
                    2.4286414692113816 * source_params.eta
                    - 23.213332913737403 * source_params.eta_pow2
                    + 66.58241012629095 * source_params.eta_pow3
                )
                + source_params.delta_chi_half_pow2
                * (
                    3.085167288859442 * source_params.eta
                    - 31.60440418701438 * source_params.eta_pow2
                    + 78.49621016381445 * source_params.eta_pow3
                )
            )
            + source_params.eta
            * (
                0.861883217178703
                + 13.695204704208976 * source_params.eta
                - 337.70598252897696 * source_params.eta_pow2
                + 2932.3415281149432 * source_params.eta_pow3
                - 12028.786386004691 * source_params.eta_pow4
                + 18536.937955014455 * source_params.eta_pow5
            )
            + source_params.eta
            * source_params.S_tot_hat
            * (
                -0.048465588779596405
                * (
                    -0.34041762314288154
                    - 81.33156665674845 * source_params.eta
                    + 1744.329802302927 * source_params.eta_pow2
                    - 16522.343895064576 * source_params.eta_pow3
                    + 76620.18243090731 * source_params.eta_pow4
                    - 133340.93723954144 * source_params.eta_pow5
                )
                + 0.024804027856323612
                * (
                    -8.666095805675418
                    + 711.8727878341302 * source_params.eta
                    - 13644.988225595187 * source_params.eta_pow2
                    + 112832.04975245205 * source_params.eta_pow3
                    - 422282.0368440555 * source_params.eta_pow4
                    + 584744.0406581408 * source_params.eta_pow5
                )
                * source_params.S_tot_hat
            )
        )

    @ti.func
    def _int_fit_v1(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            source_params.eta
            * (
                source_params.delta_chi_half
                * source_params.delta
                * (
                    1.5378890240544967 * source_params.eta
                    - 3.4499418893734903 * source_params.eta_pow2
                    + 16.879953490422782 * source_params.eta_pow3
                )
                + source_params.delta_chi_half_pow2
                * (
                    1.720226708214248 * source_params.eta
                    - 11.87925165364241 * source_params.eta_pow2
                    + 23.259283336239545 * source_params.eta_pow3
                )
            )
            + source_params.eta
            * (
                8.790173464969538
                - 64.95499142822892 * source_params.eta
                + 324.1998823562892 * source_params.eta_pow2
                - 1111.9864921907126 * source_params.eta_pow3
                + 1575.602443847111 * source_params.eta_pow4
            )
            + source_params.eta
            * source_params.chi_PN_hat
            * (
                -0.062333275821238224
                * (
                    -21.630297087123807
                    + 137.4395894877131 * source_params.eta
                    + 64.92115530780129 * source_params.eta_pow2
                    - 1013.1110639471394 * source_params.eta_pow3
                )
                - 0.11014697070998722
                * (
                    4.149721483857751
                    - 108.6912882442823 * source_params.eta
                    + 831.6073263887092 * source_params.eta_pow2
                    - 1828.2527520190122 * source_params.eta_pow3
                )
                * source_params.chi_PN_hat
                - 0.07704777584463054
                * (
                    4.581767671445529
                    - 50.35070009227704 * source_params.eta
                    + 344.9177692251726 * source_params.eta_pow2
                    - 858.9168637051405 * source_params.eta_pow3
                )
                * source_params.chi_PN_hat_pow2
            )
        )

    @ti.func
    def _int_fit_v2(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            source_params.eta
            * (
                source_params.delta_chi_half
                * source_params.delta
                * (
                    2.3123974306694057 * source_params.eta
                    - 12.237594841284904 * source_params.eta_pow2
                    + 44.78225529547671 * source_params.eta_pow3
                )
                + source_params.delta_chi_half_pow2
                * (
                    2.9282931698944292 * source_params.eta
                    - 25.624210264341933 * source_params.eta_pow2
                    + 61.05270871360041 * source_params.eta_pow3
                )
            )
            + source_params.eta
            * (
                6.98072197826729
                - 46.81443520117986 * source_params.eta
                + 236.76146303619544 * source_params.eta_pow2
                - 920.358408667518 * source_params.eta_pow3
                + 1478.050456337336 * source_params.eta_pow4
            )
            + source_params.eta
            * source_params.chi_PN_hat
            * (
                -0.07801583359561987
                * (
                    -28.29972282146242
                    + 752.1603553640072 * source_params.eta
                    - 10671.072606753183 * source_params.eta_pow2
                    + 83447.0461509547 * source_params.eta_pow3
                    - 350025.2112501252 * source_params.eta_pow4
                    + 760889.6919776166 * source_params.eta_pow5
                    - 702172.2934567826 * source_params.eta_pow6
                )
                + 0.013159545629626014
                * (
                    91.1469833190294
                    - 3557.5003799977294 * source_params.eta
                    + 52391.684517955284 * source_params.eta_pow2
                    - 344254.9973814295 * source_params.eta_pow3
                    + 1.0141877915334814e6 * source_params.eta_pow4
                    - 1.1505186449682908e6 * source_params.eta_pow5
                    + 268756.85659532435 * source_params.eta_pow6
                )
                * source_params.chi_PN_hat
            )
        )

    @ti.func
    def _int_fit_v3(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            source_params.eta
            * (
                source_params.delta_chi_half
                * source_params.delta
                * (
                    -0.8765502142143329 * source_params.eta
                    + 22.806632458441996 * source_params.eta_pow2
                    - 43.675503209991184 * source_params.eta_pow3
                )
                + source_params.delta_chi_half_pow2
                * (
                    0.48698617426180074 * source_params.eta
                    - 4.302527065360426 * source_params.eta_pow2
                    + 16.18571810759235 * source_params.eta_pow3
                )
            )
            + source_params.eta
            * (
                6.379772583015967
                - 44.10631039734796 * source_params.eta
                + 269.44092930942793 * source_params.eta_pow2
                - 1285.7635006711453 * source_params.eta_pow3
                + 2379.538739132234 * source_params.eta_pow4
            )
            + source_params.eta
            * source_params.chi_PN_hat
            * (
                -0.23316184683282615
                * (
                    -1.7279023138971559
                    - 23.606399143993716 * source_params.eta
                    + 409.3387618483284 * source_params.eta_pow2
                    - 1115.4147472977265 * source_params.eta_pow3
                )
                - 0.09653777612560172
                * (
                    -5.310643306559746
                    - 2.1852511802701264 * source_params.eta
                    + 541.1248219096527 * source_params.eta_pow2
                    - 1815.7529908827103 * source_params.eta_pow3
                )
                * source_params.chi_PN_hat
                - 0.060477799540741804
                * (
                    -14.578189130145661
                    + 175.6116682068523 * source_params.eta
                    - 569.4799973930861 * source_params.eta_pow2
                    + 426.0861915646515 * source_params.eta_pow3
                )
                * source_params.chi_PN_hat_pow2
            )
        )

    @ti.func
    def _int_fit_v4(self, source_params: ti.template()) -> ti.f64:
        return ti.abs(
            source_params.eta
            * (
                source_params.delta_chi_half
                * source_params.delta
                * (
                    -2.461738962276138 * source_params.eta
                    + 45.3240543970684 * source_params.eta_pow2
                    - 112.2714974622516 * source_params.eta_pow3
                )
                + source_params.delta_chi_half_pow2
                * (
                    0.9158352037567031 * source_params.eta
                    - 8.724582331021695 * source_params.eta_pow2
                    + 28.44633544874233 * source_params.eta_pow3
                )
            )
            + source_params.eta
            * (
                6.098676337298138
                - 45.42463610529546 * source_params.eta
                + 350.97192927929433 * source_params.eta_pow2
                - 2002.2013283876834 * source_params.eta_pow3
                + 4067.1685640401033 * source_params.eta_pow4
            )
            + source_params.eta
            * source_params.chi_PN_hat
            * (
                -0.36068516166901304
                * (
                    -2.120354236840677
                    - 47.56175350408845 * source_params.eta
                    + 1618.4222330016048 * source_params.eta_pow2
                    - 14925.514654896673 * source_params.eta_pow3
                    + 60287.45399959349 * source_params.eta_pow4
                    - 91269.3745059139 * source_params.eta_pow5
                )
                - 0.09635801207669747
                * (
                    -11.824692837267394
                    + 371.7551657959369 * source_params.eta
                    - 4176.398139238679 * source_params.eta_pow2
                    + 16655.87939259747 * source_params.eta_pow3
                    - 4102.218189945819 * source_params.eta_pow4
                    - 67024.98285179552 * source_params.eta_pow5
                )
                * source_params.chi_PN_hat
                - 0.06565232123453196
                * (
                    -26.15227471380236
                    + 1869.0168486099005 * source_params.eta
                    - 33951.35186039629 * source_params.eta_pow2
                    + 253694.6032002248 * source_params.eta_pow3
                    - 845341.6001856657 * source_params.eta_pow4
                    + 1.0442282862506858e6 * source_params.eta_pow5
                )
                * source_params.chi_PN_hat_pow2
            )
        )

    @ti.func
    def update_amplitude_coefficients(
        self,
        pn_coefficients_44: ti.template(),
        source_params: ti.template(),
    ):
        self.common_factor = 0.5 ** (-7.0 / 6.0) * tm.sqrt(
            2.0 * source_params.eta / 3.0 / useful_powers_pi.third
        )

        self._set_joint_frequencies(
            4.0,
            source_params.f_MECO_lm["44"],
            source_params.QNM_freqs_lm["44"],
            source_params,
        )
        self._set_inspiral_coefficients(pn_coefficients_44, source_params)
        self._set_merge_ringdown_coefficients(
            source_params.QNM_freqs_lm["44"], source_params
        )
        self._set_intermediate_coefficients(
            source_params.QNM_freqs_lm["44"], pn_coefficients_44, source_params
        )


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

        Rules for choosing collocation nodes:
        for situation with (eta < etaEMR) or (emm == ell and STotR >= 0.8) or (modeTag == 33 and STotR < 0), using collocation nodes: 0, 1, 3, 4, 5.
        for situation with (STotR >= 0.8) and (modeTag == 21), using collocation nodes: 0, 1, 2, 4, 5.
        for remaining parameter space, using collocation nodes: 0, 1, 2, 3, 5.
        """
        self.c_3 = 0.0

        self._set_int_colloc_value_0(source_params)
        self._set_int_colloc_value_1(source_params)
        self._set_int_colloc_value_2(source_params)
        self._set_int_colloc_value_5(source_params)

        # special operation for 21 mode to avoide sharp transitions in high-spin cases
        if source_params.S_tot_hat >= 0.8:
            int_22_v0 = phase_coefficients_22.compute_d_phase(
                pn_coefficients_22, source_params, self._useful_powers.int_f0
            )
            int_22_v1 = phase_coefficients_22.compute_d_phase(
                pn_coefficients_22, source_params, self._useful_powers.int_f1
            )
            int_22_v2 = phase_coefficients_22.compute_d_phase(
                pn_coefficients_22, source_params, self._useful_powers.int_f2
            )
            diff_01 = int_22_v0 - int_22_v1
            diff_12 = int_22_v1 - int_22_v2
            self._int_colloc_values[1] = self._int_colloc_values[2] + diff_12
            self._int_colloc_values[0] = self._int_colloc_values[1] + diff_01

        # simplified the conditional structure in LALSimIMRPhenomXHM_internals.c l.2108 for mode 21
        if source_params.eta < eta_EMR:  # using collocation nodes: 0, 1, 3, 4, 5
            self._set_intermediate_coefficients_01345(source_params)
        elif source_params.S_tot_hat >= 0.8:
            self._set_intermediate_coefficients_01245(source_params)
        else:
            self._set_intermediate_coefficients_01235(source_params)

    @ti.func
    def _set_intermediate_coefficients_01345(self, source_params: ti.template()):
        self._int_colloc_values[2] = 0.0
        self._set_int_colloc_value_3(source_params)
        self._set_int_colloc_value_4(source_params)

        Ab = self._get_int_augmented_matrix_no_mixing(
            source_params.QNM_freqs_lm["21"], [0, 1, 3, 4, 5]
        )
        self.c_0, self.c_1, self.c_2, self.c_4, self.c_L = gauss_elimination(Ab)

    @ti.func
    def _set_intermediate_coefficients_01245(self, source_params: ti.template()):
        self._int_colloc_values[3] = 0.0
        self._set_int_colloc_value_4(source_params)

        Ab = self._get_int_augmented_matrix_no_mixing(
            source_params.QNM_freqs_lm["21"], [0, 1, 2, 4, 5]
        )
        self.c_0, self.c_1, self.c_2, self.c_4, self.c_L = gauss_elimination(Ab)

    @ti.func
    def _set_intermediate_coefficients_01235(self, source_params: ti.template()):
        self._set_int_colloc_value_3(source_params)
        self._int_colloc_values[4] = 0.0

        Ab = self._get_int_augmented_matrix_no_mixing(
            source_params.QNM_freqs_lm["21"], [0, 1, 2, 3, 5]
        )
        self.c_0, self.c_1, self.c_2, self.c_4, self.c_L = gauss_elimination(Ab)

    @ti.func
    def _set_int_colloc_value_0(self, source_params: ti.template()):
        self._int_colloc_values[0] = -source_params.peak_time_diff + (
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
    def _set_int_colloc_value_1(self, source_params: ti.template()):
        self._int_colloc_values[1] = -source_params.peak_time_diff + (
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
    def _set_int_colloc_value_2(self, source_params: ti.template()):
        self._int_colloc_values[2] = -source_params.peak_time_diff + (
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
    def _set_int_colloc_value_3(self, source_params: ti.template()):
        self._int_colloc_values[3] = -source_params.peak_time_diff + (
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
    def _set_int_colloc_value_4(self, source_params: ti.template()):
        self._int_colloc_values[4] = -source_params.peak_time_diff + (
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
    def _set_int_colloc_value_5(self, source_params: ti.template()):
        self._int_colloc_values[5] = -source_params.peak_time_diff + (
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
        pn_coefficients_21: ti.template(),
        phase_coefficients_22: ti.template(),
        source_params: ti.template(),
    ):
        self._set_joint_frequencies_no_mixing(
            source_params.f_MECO_lm["21"], source_params.QNM_freqs_lm["21"]
        )
        # intermediate
        self._set_int_colloc_points_no_mixing(
            source_params.QNM_freqs_lm["21"].f_ring, source_params.eta
        )
        # used for special operation for 21 mode to avoide sharp transitions in high-spin cases
        self._useful_powers.int_f0.update(self._int_colloc_points[0])
        self._useful_powers.int_f1.update(self._int_colloc_points[1])
        self._useful_powers.int_f2.update(self._int_colloc_points[2])
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
        self._set_MRD_rescaling_coefficients(1.0 / 3.0, source_params)
        # continuity conditions
        self._set_connection_coefficients(
            source_params.QNM_freqs_lm["21"], pn_coefficients_21
        )
        # the constant phase for aligning modes
        self._set_delta_phi_lm(
            1.0,
            source_params.f_MECO_lm["21"],
            pn_coefficients_22,
            pn_coefficients_21,
            phase_coefficients_22,
            source_params,
        )


@sub_struct_from(PhaseCoefficientsHighModesBase)
class PhaseCoefficientsMode33:

    @ti.func
    def _Lambda_33_PN(self) -> ti.f64:
        return 2.0 / 3.0 * PI * (21.0 / 5.0 - 6.0 * tm.log(1.5))

    @ti.func
    def _Lambda_33_fit(self, source_params: ti.template()) -> ti.f64:
        return (
            4.1138398568400705
            + 9.772510519809892 * source_params.eta
            - 103.92956504520747 * source_params.eta_pow2
            + 242.3428625556764 * source_params.eta_pow3
            + (
                (
                    -0.13253553909611435
                    + 26.644159828590055 * source_params.eta
                    - 105.09339163109497 * source_params.eta_pow2
                )
                * source_params.S_tot_hat
            )
            / (1.0 + 0.11322426762297967 * source_params.S_tot_hat)
            - 19.705359163581168
            * source_params.delta_chi
            * source_params.eta_pow2
            * source_params.delta
        )

    @ti.func
    def _set_intermediate_coefficients(self, source_params: ti.template()):
        """
        Setting intermediate coefficients for mode 33.
        For modes without significant mode-mixing, using 5 out of 6 collocation nodes determined according to spin and mass ratio, and setting c_3 = 0.

        Rules for choosing collocation nodes:
        for situation with (eta < etaEMR) or (emm == ell and STotR >= 0.8) or (modeTag == 33 and STotR < 0), using collocation nodes: 0, 1, 3, 4, 5.
        for situation with (STotR >= 0.8) and (modeTag == 21), using collocation nodes: 0, 1, 2, 4, 5.
        for remaining parameter space, using collocation nodes: 0, 1, 2, 3, 5
        """
        self.c_3 = 0.0

        self._set_int_colloc_value_0(source_params)
        self._set_int_colloc_value_1(source_params)
        self._set_int_colloc_value_5(source_params)

        # simplified the conditional structure in LALSimIMRPhenomXHM_internals.c l.2108 for mode 33
        if (
            (source_params.eta < eta_EMR)
            or (source_params.S_tot_hat >= 0.8)
            or (source_params.S_tot_hat < 0.0)
        ):  # using collocation nodes: 0, 1, 3, 4, 5
            self._set_intermediate_coefficients_01345(source_params)
        else:  # using collocation nodes: 0, 1, 2, 3, 5
            self._set_intermediate_coefficients_01235(source_params)

    @ti.func
    def _set_intermediate_coefficients_01345(self, source_params: ti.template()):
        self._int_colloc_values[2] = 0.0
        self._set_int_colloc_value_3(source_params)
        self._set_int_colloc_value_4(source_params)

        Ab = self._get_int_augmented_matrix_no_mixing(
            source_params.QNM_freqs_lm["33"], [0, 1, 3, 4, 5]
        )
        self.c_0, self.c_1, self.c_2, self.c_4, self.c_L = gauss_elimination(Ab)

    @ti.func
    def _set_intermediate_coefficients_01235(self, source_params: ti.template()):
        self._int_colloc_values[4] = 0.0
        self._set_int_colloc_value_2(source_params)
        self._set_int_colloc_value_3(source_params)

        Ab = self._get_int_augmented_matrix_no_mixing(
            source_params.QNM_freqs_lm["33"], [0, 1, 2, 3, 5]
        )
        self.c_0, self.c_1, self.c_2, self.c_4, self.c_L = gauss_elimination(Ab)

    @ti.func
    def _set_int_colloc_value_0(self, source_params: ti.template()):
        self._int_colloc_values[0] = -source_params.peak_time_diff + (
            4360.19
            + 4.27128 / source_params.eta
            - 8727.4 * source_params.eta
            + 18485.9 * source_params.eta_pow2
            + 371303.00000000006 * source_params.eta_pow3
            - 3.22792e6 * source_params.eta_pow4
            + 1.01799e7 * source_params.eta_pow5
            - 1.15659e7 * source_params.eta_pow6
            + (
                (
                    11.6635
                    - 251.579 * source_params.eta
                    - 3255.6400000000003 * source_params.eta_pow2
                    + 19614.6 * source_params.eta_pow3
                    - 34860.2 * source_params.eta_pow4
                )
                * source_params.S_tot_hat
                + (
                    14.8017
                    + 204.025 * source_params.eta
                    - 5421.92 * source_params.eta_pow2
                    + 36587.3 * source_params.eta_pow3
                    - 74299.5 * source_params.eta_pow4
                )
                * source_params.S_tot_hat_pow2
            )
            / source_params.eta
            + source_params.eta
            * (
                223.65100000000004
                * source_params.chi_1
                * source_params.delta
                * (3.9201300240106223 + 1.0 * source_params.eta)
                - 223.65100000000004
                * source_params.chi_2
                * source_params.delta
                * (3.9201300240106223 + 1.0 * source_params.eta)
            )
        )

    @ti.func
    def _set_int_colloc_value_1(self, source_params: ti.template()):
        self._int_colloc_values[1] = -source_params.peak_time_diff + (
            (
                3797.06
                + 0.786684 / source_params.eta
                - 2397.09 * source_params.eta
                - 25514.0 * source_params.eta_pow2
                + 518314.99999999994 * source_params.eta_pow3
                - 3.41708e6 * source_params.eta_pow4
                + 1.01799e7 * source_params.eta_pow5
                - 1.15659e7 * source_params.eta_pow6
            )
            + (
                (
                    6.7812399999999995
                    + 39.4668 * source_params.eta
                    - 3520.37 * source_params.eta_pow2
                    + 19614.6 * source_params.eta_pow3
                    - 34860.2 * source_params.eta_pow4
                )
                * source_params.S_tot_hat
                + (
                    4.80384
                    + 293.215 * source_params.eta
                    - 5914.61 * source_params.eta_pow2
                    + 36587.3 * source_params.eta_pow3
                    - 74299.5 * source_params.eta_pow4
                )
                * source_params.S_tot_hat_pow2
            )
            / source_params.eta
            - 223.65100000000004
            * source_params.delta
            * source_params.eta
            * (
                source_params.chi_1 * (-1.3095134830606614 - source_params.eta)
                + source_params.chi_2 * (1.3095134830606614 + source_params.eta)
            )
        )

    @ti.func
    def _set_int_colloc_value_2(self, source_params: ti.template()):
        self._int_colloc_values[2] = -source_params.peak_time_diff + (
            3321.83
            + 1796.03 * source_params.eta
            - 52406.1 * source_params.eta_pow2
            + 605028.0 * source_params.eta_pow3
            - 3.52532e6 * source_params.eta_pow4
            + 1.01799e7 * source_params.eta_pow5
            - 1.15659e7 * source_params.eta_pow6
            + (
                223.601
                - 3714.77 * source_params.eta
                + 19614.6 * source_params.eta_pow2
                - 34860.2 * source_params.eta_pow3
            )
            * source_params.S_tot_hat
            + (
                314.317
                - 5906.46 * source_params.eta
                + 36587.3 * source_params.eta_pow2
                - 74299.5 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow2
            + 223.651
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
        )

    @ti.func
    def _set_int_colloc_value_3(self, source_params: ti.template()):
        self._int_colloc_values[3] = -source_params.peak_time_diff + (
            3239.44
            - 661.15 * source_params.eta
            + 5139.79 * source_params.eta_pow2
            + 3456.2 * source_params.eta_pow3
            - 248477.0 * source_params.eta_pow4
            + 1.17255e6 * source_params.eta_pow5
            - 1.70363e6 * source_params.eta_pow6
            + (
                225.859
                - 4150.09 * source_params.eta
                + 24364.0 * source_params.eta_pow2
                - 46537.3 * source_params.eta_pow3
            )
            * source_params.S_tot_hat
            + (
                35.2439
                - 994.971 * source_params.eta
                + 8953.98 * source_params.eta_pow2
                - 23603.5 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow2
            + (
                -310.489
                + 5946.15 * source_params.eta
                - 35337.1 * source_params.eta_pow2
                + 67102.4 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow3
            + 30.484
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
        )

    @ti.func
    def _set_int_colloc_value_4(self, source_params: ti.template()):
        self._int_colloc_values[4] = -source_params.peak_time_diff + (
            3114.3
            + 2143.06 * source_params.eta
            - 49428.3 * source_params.eta_pow2
            + 563997.0 * source_params.eta_pow3
            - 3.35991e6 * source_params.eta_pow4
            + 9.99745e6 * source_params.eta_pow5
            - 1.17123e7 * source_params.eta_pow6
            + (
                190.051
                - 3705.08 * source_params.eta
                + 23046.2 * source_params.eta_pow2
                - 46537.3 * source_params.eta_pow3
            )
            * source_params.S_tot_hat
            + (
                63.6615
                - 1414.2 * source_params.eta
                + 10166.1 * source_params.eta_pow2
                - 23603.5 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow2
            + (
                -257.524
                + 5179.97 * source_params.eta
                - 33001.4 * source_params.eta_pow2
                + 67102.4 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow3
            + 54.9833
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
        )

    @ti.func
    def _set_int_colloc_value_5(self, source_params: ti.template()):
        self._int_colloc_values[5] = -source_params.peak_time_diff + (
            3111.46
            + 384.121 * source_params.eta
            - 13003.6 * source_params.eta_pow2
            + 179537.0 * source_params.eta_pow3
            - 1.19313e6 * source_params.eta_pow4
            + 3.79886e6 * source_params.eta_pow5
            - 4.64858e6 * source_params.eta_pow6
            + (
                182.864
                - 3834.22 * source_params.eta
                + 24532.9 * source_params.eta_pow2
                - 50165.9 * source_params.eta_pow3
            )
            * source_params.S_tot_hat
            + (
                21.0158
                - 746.957 * source_params.eta
                + 6701.33 * source_params.eta_pow2
                - 17842.3 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow2
            + (
                -292.855
                + 5886.62 * source_params.eta
                - 37382.4 * source_params.eta_pow2
                + 75501.8 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow3
            + 75.5162
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
        )

    @ti.func
    def update_phase_coefficients(
        self,
        pn_coefficients_22: ti.template(),
        pn_coefficients_33: ti.template(),
        phase_coefficients_22: ti.template(),
        source_params: ti.template(),
    ):
        self._set_joint_frequencies_no_mixing(
            source_params.f_MECO_lm["33"], source_params.QNM_freqs_lm["33"]
        )
        # intermediate
        self._set_int_colloc_points_no_mixing(
            source_params.QNM_freqs_lm["33"].f_ring, source_params.eta
        )
        self._set_intermediate_coefficients(source_params)
        # inspiral
        self._set_ins_rescaling_coefficients(3.0, phase_coefficients_22)
        if source_params.eta > 0.01:
            self.Lambda_lm = self._Lambda_33_PN()
        else:
            self.Lambda_lm = self._Lambda_33_fit(source_params)
        # merge-ringdown
        self._set_MRD_rescaling_coefficients(2.0, source_params)
        # continuity conditions
        self._set_connection_coefficients(
            source_params.QNM_freqs_lm["33"], pn_coefficients_33
        )
        # the constant phase for aligning modes
        self._set_delta_phi_lm(
            3.0,
            source_params.f_MECO_lm["33"],
            pn_coefficients_22,
            pn_coefficients_33,
            phase_coefficients_22,
            source_params,
        )


@sub_struct_from(PhaseCoefficientsHighModesBase)
class PhaseCoefficientsMode32:

    _int_phi_fend: ti.f64  # phase at the int_f_end
    _int_dphi_fend: ti.f64  # derivative of phase at the int_f_end

    removed_members = ["alpha_2", "alpha_L"]

    @ti.func
    def _Lambda_32_PN(self, source_params: ti.template()) -> ti.f64:
        return (2376.0 * PI * (-5.0 + 22.0 * source_params.eta)) / (
            -3960.0 + 11880 * source_params.eta
        )

    @ti.func
    def _Lambda_32_fit(self, source_params: ti.template()) -> ti.f64:
        return (
            (
                9.913819875501506
                + 18.424900617803107 * source_params.eta
                - 574.8672384388947 * source_params.eta_pow2
                + 2671.7813055097877 * source_params.eta_pow3
                - 6244.001932443913 * source_params.eta_pow4
            )
            / (1.0 - 0.9103118343073325 * source_params.eta)
            + (
                -4.367632806613781
                + 245.06757304950986 * source_params.eta
                - 2233.9319708029775 * source_params.eta_pow2
                + 5894.355429022858 * source_params.eta_pow3
            )
            * source_params.S_tot_hat
            + (
                -1.375112297530783
                - 1876.760129419146 * source_params.eta
                + 17608.172965575013 * source_params.eta_pow2
                - 40928.07304790013 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow2
            + (
                -1.28324755577382
                - 138.36970336658558 * source_params.eta
                + 708.1455154504333 * source_params.eta_pow2
                - 273.23750933544176 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow3
            + (
                1.8403161863444328
                + 2009.7361967331492 * source_params.eta
                - 18636.271414571278 * source_params.eta_pow2
                + 42379.205045791656 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow4
            + source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
            * (
                -105.34550407768225
                - 1566.1242344157668 * source_params.chi_1 * source_params.eta
                + 1566.1242344157668 * source_params.chi_2 * source_params.eta
                + 2155.472229664981 * source_params.eta * source_params.S_tot_hat
            )
        )

    @ti.func
    def _set_joint_frequencies_mode_32(self, source_params: ti.template()):
        self.ins_f_end = source_params.f_MECO_lm["32"]
        self.int_f_end = source_params.f_ring - 0.5 * source_params.f_damp
        # correct int_f_end for EMR with negative spins
        if source_params.eta < 0.01 and source_params.chi_1 < 0.0:
            self.int_f_end *= 1.2 - 0.25 * source_params.chi_1
        self._useful_powers.ins_f_end.update(self.ins_f_end)
        self._useful_powers.int_f_end.update(self.int_f_end)

    @ti.func
    def _set_int_colloc_points_mode_32(self, source_params: ti.template()):
        # shifting forward the frequency of the first collocation points for small eta
        beta = 1.0 + 0.001 * (0.25 / source_params.eta - 1.0)
        f_end = source_params.f_ring - 0.5 * source_params.f_damp

        self._int_colloc_points[0] = beta * self.ins_f_end
        self._int_colloc_points[1] = (
            tm.sqrt(3.0) * (self._int_colloc_points[0] - f_end)
            + 2.0 * (self._int_colloc_points[0] + f_end)
        ) / 4.0
        self._int_colloc_points[2] = (3.0 * self._int_colloc_points[0] + f_end) / 4.0
        self._int_colloc_points[3] = (self._int_colloc_points[0] + f_end) / 2.0
        # note the l.2226 in LALSimIMRPhenomXHM_internals.c for setting int_colloc_points_4
        # does not actually work. When eta>eta_EMR(0.05), the int_f_end is not corrected.
        self._int_colloc_points[4] = f_end
        self._int_colloc_points[5] = self.int_f_end

    @ti.func
    def _set_intermediate_coefficients(
        self,
        pn_coefficients_22: ti.template(),
        amplitude_coefficients_22: ti.template(),
        phase_coefficients_22: ti.template(),
        spheroidal_merge_ringdown_32: ti.template(),
        source_params: ti.template(),
    ):
        """
        Setting intermediate coefficients for mode 32.

        For the mode 32, c_3 is allowed to be non-zero for better control on effects
        of mode-mixing. Fixing the extra degree of freedom requires one more equation
        which is given by the second-order derivative at the left boundary.
        """
        # get the derivatives at the boundary using finite-difference
        step = 1e-7
        phi = ti.Vector([0.0] * 3, dt=ti.f64)
        powers_fi = UsefulPowers()
        for i in ti.static(range(3)):
            fi = self.int_f_end + (i - 1) * step
            powers_fi.update(fi)
            amp_22 = (
                amplitude_coefficients_22._merge_ringdown_amplitude(
                    source_params, powers_fi
                )
                / powers_fi.seven_sixths
                * amplitude_coefficients_22.common_factor
            )
            phi_22 = phase_coefficients_22.compute_phase(
                pn_coefficients_22, source_params, powers_fi
            )
            h22 = amp_22 * tm.cexp(ComplexNumber([0.0, 1.0]) * phi_22)
            phi_i = self._merge_ringdown_phase(
                h22,
                spheroidal_merge_ringdown_32,
                source_params.QNM_freqs_lm["32"],
                powers_fi,
            )
            # make sure that all the three points belong to the same branch
            # TODO is this really work?? if three phi across 0, blow operation cannot make them in the same branch
            # consider using function like np.unwrap??
            if phi_i > 0.0:
                phi_i -= 2.0 * PI
            phi[i] = phi_i
        d_phi = 0.5 * (phi[2] - phi[0]) / step
        d2_phi = (phi[2] - 2.0 * phi[1] + phi[0]) / step**2
        # will be used to set the connection coefficients
        self._int_phi_fend = phi[1]
        self._int_dphi_fend = d_phi

        self._set_int_colloc_value_0(source_params)
        self._set_int_colloc_value_1(source_params)
        self._set_int_colloc_value_2(source_params)
        self._set_int_colloc_value_3(source_params)
        # If the mass-ratio is not extreme, use the numerical derivative for int_colloc_values[4],
        # otherwise using the value of the fit.
        # Note if eta>eta_EMR, int_f_end is not corrected, int_colloc_points[4] does not
        # need to re-set as l.2226 in LALSimIMRPhenomXHM_internals.c
        if source_params.eta > eta_EMR:
            self._int_colloc_values[4] = d_phi
        else:
            self._set_int_colloc_value_4(source_params)
        self._int_colloc_values[5] = d2_phi

        Ab = ti.Matrix([[0.0] * 7 for _ in range(6)], dt=ti.f64)
        for i in ti.static(range(5)):
            row = [
                1.0,
                self._int_colloc_points[i] ** (-1),
                self._int_colloc_points[i] ** (-2),
                self._int_colloc_points[i] ** (-3),
                self._int_colloc_points[i] ** (-4),
                source_params.QNM_freqs_lm["32"].f_damp
                / (
                    source_params.QNM_freqs_lm["32"].f_damp_pow2
                    + (
                        self._int_colloc_points[i]
                        - source_params.QNM_freqs_lm["32"].f_ring
                    )
                    ** 2
                ),
                self._int_colloc_values[i],
            ]
            for j in ti.static(range(7)):
                Ab[i, j] = row[j]
        # the 2ed derivative at the left boundary
        row = [
            0.0,
            -self._int_colloc_points[5] ** (-2),
            -2.0 * self._int_colloc_points[5] ** (-3),
            -3.0 * self._int_colloc_points[5] ** (-4),
            -4.0 * self._int_colloc_points[5] ** (-5),
            -2.0
            * (self._int_colloc_points[5] - source_params.QNM_freqs_lm["32"].f_ring)
            * source_params.QNM_freqs_lm["32"].f_damp
            / (
                source_params.QNM_freqs_lm["32"].f_damp_pow2
                + (self._int_colloc_points[5] - source_params.QNM_freqs_lm["32"].f_ring)
                ** 2
            )
            ** 2,
            self._int_colloc_values[5],
        ]
        for j in ti.static(range(7)):
            Ab[5, j] = row[j]

        self.c_0, self.c_1, self.c_2, self.c_3, self.c_4, self.c_L = gauss_elimination(Ab)  # fmt: skip

    @ti.func
    def _set_connection_coefficients(
        self,
        QNM_freqs_32: ti.template(),
        pn_coefficients_32: ti.template(),
    ):
        self.ins_C1 = self._intermediate_d_phase(
            QNM_freqs_32, self._useful_powers.ins_f_end
        ) - self._inspiral_d_phase(pn_coefficients_32, self._useful_powers.ins_f_end)
        # Note we have dropped the constant of phi_5, ins_C0 is different with CINSP in
        # lalsimulation. ins_C0 (tiwave) = CINSP (lalsim) - phi_5
        self.ins_C0 = (
            self._intermediate_phase(QNM_freqs_32, self._useful_powers.ins_f_end)
            - self._inspiral_phase(pn_coefficients_32, self._useful_powers.ins_f_end)
            - self.ins_C1 * self.ins_f_end
        )

        self.MRD_C1 = (
            self._intermediate_d_phase(QNM_freqs_32, self._useful_powers.int_f_end)
            - self._int_dphi_fend
        )
        self.MRD_C0 = (
            self._intermediate_phase(QNM_freqs_32, self._useful_powers.int_f_end)
            - self._int_phi_fend
            - self.MRD_C1 * self.int_f_end
        )

    @ti.func
    def _set_int_colloc_value_0(self, source_params: ti.template()):
        self._int_colloc_values[0] = -source_params.peak_time_diff + (
            4414.11
            + 4.21564 / source_params.eta
            - 10687.8 * source_params.eta
            + 58234.6 * source_params.eta_pow2
            - 64068.40000000001 * source_params.eta_pow3
            - 704442.0 * source_params.eta_pow4
            + 2.86393e6 * source_params.eta_pow5
            - 3.26362e6 * source_params.eta_pow6
            + (
                (
                    6.39833
                    - 610.267 * source_params.eta
                    + 2095.72 * source_params.eta_pow2
                    - 3970.89 * source_params.eta_pow3
                )
                * source_params.S_tot_hat
                + (
                    22.956700000000005
                    - 99.1551 * source_params.eta
                    + 331.593 * source_params.eta_pow2
                    - 794.79 * source_params.eta_pow3
                )
                * source_params.S_tot_hat_pow2
                + (
                    10.4333
                    + 43.8812 * source_params.eta
                    - 541.261 * source_params.eta_pow2
                    + 294.289 * source_params.eta_pow3
                )
                * source_params.S_tot_hat_pow3
                + source_params.eta
                * (
                    106.047
                    - 1569.0299999999997 * source_params.eta
                    + 4810.61 * source_params.eta_pow2
                )
                * source_params.S_tot_hat_pow4
            )
            / source_params.eta
            + 132.244
            * source_params.delta
            * source_params.eta
            * (
                source_params.chi_1 * (6.227738120444028 - 1.0 * source_params.eta)
                + source_params.chi_2 * (-6.227738120444028 + 1.0 * source_params.eta)
            )
        )

    @ti.func
    def _set_int_colloc_value_1(self, source_params: ti.template()):
        self._int_colloc_values[1] = -source_params.peak_time_diff + (
            3980.7
            + 0.956703 / source_params.eta
            - 6202.38 * source_params.eta
            + 29218.1 * source_params.eta_pow2
            + 24484.2 * source_params.eta_pow3
            - 807629.0 * source_params.eta_pow4
            + 2.86393e6 * source_params.eta_pow5
            - 3.26362e6 * source_params.eta_pow6
            + (
                (
                    1.92692
                    - 226.825 * source_params.eta
                    + 75.246 * source_params.eta_pow2
                    + 1291.56 * source_params.eta_pow3
                )
                * source_params.S_tot_hat
                + (
                    15.328700000000001
                    - 99.1551 * source_params.eta
                    + 608.328 * source_params.eta_pow2
                    - 2402.94 * source_params.eta_pow3
                )
                * source_params.S_tot_hat_pow2
                + (
                    10.4333
                    + 43.8812 * source_params.eta
                    - 541.261 * source_params.eta_pow2
                    + 294.289 * source_params.eta_pow3
                )
                * source_params.S_tot_hat_pow3
                + source_params.eta
                * (
                    106.047
                    - 1569.0299999999997 * source_params.eta
                    + 4810.61 * source_params.eta_pow2
                )
                * source_params.S_tot_hat_pow4
            )
            / source_params.eta
            + 132.244
            * source_params.delta
            * source_params.eta
            * (
                source_params.chi_1 * (2.5769789177580837 - 1.0 * source_params.eta)
                + source_params.chi_2 * (-2.5769789177580837 + 1.0 * source_params.eta)
            )
        )

    @ti.func
    def _set_int_colloc_value_2(self, source_params: ti.template()):
        self._int_colloc_values[2] = -source_params.peak_time_diff + (
            3416.57
            + 2308.63 * source_params.eta
            - 84042.9 * source_params.eta_pow2
            + 1.01936e6 * source_params.eta_pow3
            - 6.0644e6 * source_params.eta_pow4
            + 1.76399e7 * source_params.eta_pow5
            - 2.0065e7 * source_params.eta_pow6
            + (
                24.6295
                - 282.354 * source_params.eta
                - 2582.55 * source_params.eta_pow2
                + 12750.0 * source_params.eta_pow3
            )
            * source_params.S_tot_hat
            + (
                433.675
                - 8775.86 * source_params.eta
                + 56407.8 * source_params.eta_pow2
                - 114798.0 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow2
            + (
                559.705
                - 10627.4 * source_params.eta
                + 61581.0 * source_params.eta_pow2
                - 114029.0 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow3
            + (106.047 - 1569.03 * source_params.eta + 4810.61 * source_params.eta_pow2)
            * source_params.S_tot_hat_pow4
            + 63.9466
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
        )

    @ti.func
    def _set_int_colloc_value_3(self, source_params: ti.template()):
        self._int_colloc_values[3] = -source_params.peak_time_diff + (
            3307.49
            - 476.909 * source_params.eta
            - 5980.37 * source_params.eta_pow2
            + 127610.0 * source_params.eta_pow3
            - 919108.0 * source_params.eta_pow4
            + 2.86393e6 * source_params.eta_pow5
            - 3.26362e6 * source_params.eta_pow6
            + (
                -5.02553
                - 282.354 * source_params.eta
                + 1291.56 * source_params.eta_pow2
            )
            * source_params.S_tot_hat
            + (
                -43.8823
                + 740.123 * source_params.eta
                - 2402.94 * source_params.eta_pow2
            )
            * source_params.S_tot_hat_pow2
            + (43.8812 - 370.362 * source_params.eta + 294.289 * source_params.eta_pow2)
            * source_params.S_tot_hat_pow3
            + (106.047 - 1569.03 * source_params.eta + 4810.61 * source_params.eta_pow2)
            * source_params.S_tot_hat_pow4
            - 132.244
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
        )

    @ti.func
    def _set_int_colloc_value_4(self, source_params: ti.template()):
        self._int_colloc_values[4] = -source_params.peak_time_diff + (
            3259.03
            - 3967.58 * source_params.eta
            + 111203.0 * source_params.eta_pow2
            - 1.81883e6 * source_params.eta_pow3
            + 1.73811e7 * source_params.eta_pow4
            - 9.56988e7 * source_params.eta_pow5
            + 2.75056e8 * source_params.eta_pow6
            - 3.15866e8 * source_params.eta_pow7
            + (19.7509 - 1104.53 * source_params.eta + 3810.18 * source_params.eta_pow2)
            * source_params.S_tot_hat
            + (-230.07 + 2314.51 * source_params.eta - 5944.49 * source_params.eta_pow2)
            * source_params.S_tot_hat_pow2
            + (
                -201.633
                + 2183.43 * source_params.eta
                - 6233.99 * source_params.eta_pow2
            )
            * source_params.S_tot_hat_pow3
            + (106.047 - 1569.03 * source_params.eta + 4810.61 * source_params.eta_pow2)
            * source_params.S_tot_hat_pow4
            + 112.714
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
        )

    @ti.func
    def _merge_ringdown_phase(
        self,
        h22: ti.template(),
        spheroidal_merge_ringdown_32: ti.template(),
        QNM_freqs_32: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> ti.f64:
        h32 = spheroidal_merge_ringdown_32.spherical_h32(
            h22, QNM_freqs_32, powers_of_Mf
        )
        return tm.atan2(h32[1], h32[0])  # [-pi, pi]

    @ti.func
    def update_phase_coefficients(
        self,
        pn_coefficients_22: ti.template(),
        pn_coefficients_32: ti.template(),
        amplitude_coefficients_22: ti.template(),
        phase_coefficients_22: ti.template(),
        spheroidal_merge_ringdown_32: ti.template(),
        source_params: ti.template(),
    ):
        self._set_joint_frequencies_mode_32(source_params)
        # intermediate
        self._set_int_colloc_points_mode_32(source_params)
        self._set_intermediate_coefficients(
            pn_coefficients_22,
            amplitude_coefficients_22,
            phase_coefficients_22,
            spheroidal_merge_ringdown_32,
            source_params,
        )
        # inspiral
        self._set_ins_rescaling_coefficients(2.0, phase_coefficients_22)
        if source_params.eta > 0.01:
            self.Lambda_lm = self._Lambda_32_PN(source_params)
        else:
            self.Lambda_lm = self._Lambda_32_fit(source_params)
        # continuity conditions
        self._set_connection_coefficients(
            source_params.QNM_freqs_lm["32"], pn_coefficients_32
        )
        # the constant phase for aligning modes
        self._set_delta_phi_lm(
            2.0,
            source_params.f_MECO_lm["32"],
            pn_coefficients_22,
            pn_coefficients_32,
            phase_coefficients_22,
            source_params,
        )


@sub_struct_from(PhaseCoefficientsHighModesBase)
class PhaseCoefficientsMode44:

    @ti.func
    def _Lambda_44_PN(self, source_params: ti.template()) -> ti.f64:
        return (
            45045.0
            * PI
            * (
                336.0
                - 1193.0 * source_params.eta
                + 320.0 * (-1.0 + 3.0 * source_params.eta) * tm.log(2.0)
            )
            / (2.0 * (1801800.0 - 5405400.0 * source_params.eta))
        )

    @ti.func
    def _Lambda_44_fit(self, source_params: ti.template()) -> ti.f64:
        return (
            5.254484747463392
            - 21.277760168559862 * source_params.eta
            + 160.43721442910618 * source_params.eta_pow2
            - 1162.954360723399 * source_params.eta_pow3
            + 1685.5912722190276 * source_params.eta_pow4
            - 1538.6661348106031 * source_params.eta_pow5
            + (
                0.007067861615983771
                - 10.945895160727437 * source_params.eta
                + 246.8787141453734 * source_params.eta_pow2
                - 810.7773268493444 * source_params.eta_pow3
            )
            * source_params.S_tot_hat
            + (
                0.17447830920234977
                + 4.530539154777984 * source_params.eta
                - 176.4987316167203 * source_params.eta_pow2
                + 621.6920322846844 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow2
            - 8.384066369867833
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
        )

    @ti.func
    def _set_intermediate_coefficients(self, source_params: ti.template()):
        """
        Setting intermediate coefficients for mode 44.
        For modes without significant mode-mixing, using 5 out of 6 collocation nodes determined according to spin and mass ratio, and setting c_3 = 0.

        Rules for choosing collocation nodes:
        for situation with (eta < etaEMR) or (emm == ell and STotR >= 0.8) or (modeTag == 33 and STotR < 0), using collocation nodes: 0, 1, 3, 4, 5.
        for situation with (STotR >= 0.8) and (modeTag == 21), using collocation nodes: 0, 1, 2, 4, 5.
        for remaining parameter space, using collocation nodes: 0, 1, 2, 3, 5
        """
        self.c_3 = 0.0

        self._set_int_colloc_value_0(source_params)
        self._set_int_colloc_value_1(source_params)
        self._set_int_colloc_value_5(source_params)

        # simplified the conditional structure in LALSimIMRPhenomXHM_internals.c l.2108 for modeTag=33
        if (source_params.eta < eta_EMR) or (
            source_params.S_tot_hat >= 0.8
        ):  # using collocation nodes: 0, 1, 3, 4, 5
            self._set_intermediate_coefficients_01345(source_params)
        else:  # using collocation nodes: 0, 1, 2, 3, 5
            self._set_intermediate_coefficients_01235(source_params)

    @ti.func
    def _set_intermediate_coefficients_01345(self, source_params: ti.template()):
        self._int_colloc_values[2] = 0.0
        self._set_int_colloc_value_3(source_params)
        self._set_int_colloc_value_4(source_params)

        Ab = self._get_int_augmented_matrix_no_mixing(
            source_params.QNM_freqs_lm["44"], [0, 1, 3, 4, 5]
        )
        self.c_0, self.c_1, self.c_2, self.c_4, self.c_L = gauss_elimination(Ab)

    @ti.func
    def _set_intermediate_coefficients_01235(self, source_params: ti.template()):
        self._int_colloc_values[4] = 0.0
        self._set_int_colloc_value_2(source_params)
        self._set_int_colloc_value_3(source_params)

        Ab = self._get_int_augmented_matrix_no_mixing(
            source_params.QNM_freqs_lm["44"], [0, 1, 2, 3, 5]
        )
        self.c_0, self.c_1, self.c_2, self.c_4, self.c_L = gauss_elimination(Ab)

    @ti.func
    def _set_int_colloc_value_0(self, source_params: ti.template()):
        self._int_colloc_values[0] = -source_params.peak_time_diff + (
            4349.66
            + 4.34125 / source_params.eta
            - 8202.33 * source_params.eta
            + 5534.1 * source_params.eta_pow2
            + 536500.0 * source_params.eta_pow3
            - 4.33197e6 * source_params.eta_pow4
            + 1.37792e7 * source_params.eta_pow5
            - 1.60802e7 * source_params.eta_pow6
            + (
                (
                    12.0704
                    - 528.098 * source_params.eta
                    + 1822.9100000000003 * source_params.eta_pow2
                    - 9349.73 * source_params.eta_pow3
                    + 17900.9 * source_params.eta_pow4
                )
                * source_params.S_tot_hat
                + (
                    10.4092
                    + 253.334 * source_params.eta
                    - 5452.04 * source_params.eta_pow2
                    + 35416.6 * source_params.eta_pow3
                    - 71523.0 * source_params.eta_pow4
                )
                * source_params.S_tot_hat_pow2
                + source_params.eta
                * (
                    492.60300000000007
                    - 9508.5 * source_params.eta
                    + 57303.4 * source_params.eta_pow2
                    - 109418.0 * source_params.eta_pow3
                )
                * source_params.S_tot_hat_pow3
            )
            / source_params.eta
            - 262.143
            * source_params.delta
            * source_params.eta
            * (
                source_params.chi_1 * (-3.0782778864970646 - 1.0 * source_params.eta)
                + source_params.chi_2 * (3.0782778864970646 + 1.0 * source_params.eta)
            )
        )

    @ti.func
    def _set_int_colloc_value_1(self, source_params: ti.template()):
        self._int_colloc_values[1] = -source_params.peak_time_diff + (
            3804.19
            + 0.66144 / source_params.eta
            - 2421.77 * source_params.eta
            - 33475.8 * source_params.eta_pow2
            + 665951.0 * source_params.eta_pow3
            - 4.50145e6 * source_params.eta_pow4
            + 1.37792e7 * source_params.eta_pow5
            - 1.60802e7 * source_params.eta_pow6
            + (
                (
                    5.83038
                    - 172.047 * source_params.eta
                    + 926.576 * source_params.eta_pow2
                    - 7676.87 * source_params.eta_pow3
                    + 17900.9 * source_params.eta_pow4
                )
                * source_params.S_tot_hat
                + (
                    6.17601
                    + 253.334 * source_params.eta
                    - 5672.02 * source_params.eta_pow2
                    + 35722.1 * source_params.eta_pow3
                    - 71523.0 * source_params.eta_pow4
                )
                * source_params.S_tot_hat_pow2
                + source_params.eta
                * (
                    492.60300000000007
                    - 9508.5 * source_params.eta
                    + 57303.4 * source_params.eta_pow2
                    - 109418.0 * source_params.eta_pow3
                )
                * source_params.S_tot_hat_pow3
            )
            / source_params.eta
            - 262.143
            * source_params.delta
            * source_params.eta
            * (
                source_params.chi_1 * (-1.0543062374352932 - 1.0 * source_params.eta)
                + source_params.chi_2 * (1.0543062374352932 + 1.0 * source_params.eta)
            )
        )

    @ti.func
    def _set_int_colloc_value_2(self, source_params: ti.template()):
        self._int_colloc_values[2] = -source_params.peak_time_diff + (
            3308.97
            + 2353.58 * source_params.eta
            - 66340.1 * source_params.eta_pow2
            + 777272.0 * source_params.eta_pow3
            - 4.64438e6 * source_params.eta_pow4
            + 1.37792e7 * source_params.eta_pow5
            - 1.60802e7 * source_params.eta_pow6
            + (
                -21.5697
                + 926.576 * source_params.eta
                - 7989.26 * source_params.eta_pow2
                + 17900.9 * source_params.eta_pow3
            )
            * source_params.S_tot_hat
            + (
                353.539
                - 6403.24 * source_params.eta
                + 37599.5 * source_params.eta_pow2
                - 71523.0 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow2
            + (
                492.603
                - 9508.5 * source_params.eta
                + 57303.4 * source_params.eta_pow2
                - 109418.0 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow3
            + 262.143
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
        )

    @ti.func
    def _set_int_colloc_value_3(self, source_params: ti.template()):
        self._int_colloc_values[3] = -source_params.peak_time_diff + (
            3245.63
            - 928.56 * source_params.eta
            + 8463.89 * source_params.eta_pow2
            - 17422.6 * source_params.eta_pow3
            - 165169.0 * source_params.eta_pow4
            + 908279.0 * source_params.eta_pow5
            - 1.31138e6 * source_params.eta_pow6
            + (
                32.506
                - 590.293 * source_params.eta
                + 3536.61 * source_params.eta_pow2
                - 6758.52 * source_params.eta_pow3
            )
            * source_params.S_tot_hat
            + (
                -25.7716
                + 738.141 * source_params.eta
                - 4867.87 * source_params.eta_pow2
                + 9129.45 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow2
            + (
                -15.7439
                + 620.695 * source_params.eta
                - 4679.24 * source_params.eta_pow2
                + 9582.58 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow3
            + 87.0832
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
        )

    @ti.func
    def _set_int_colloc_value_4(self, source_params: ti.template()):
        self._int_colloc_values[4] = -source_params.peak_time_diff + (
            3108.38
            + 3722.46 * source_params.eta
            - 119588.0 * source_params.eta_pow2
            + 1.92148e6 * source_params.eta_pow3
            - 1.69796e7 * source_params.eta_pow4
            + 8.39194e7 * source_params.eta_pow5
            - 2.17143e8 * source_params.eta_pow6
            + 2.2829700000000003e8 * source_params.eta_pow7
            + (118.319 - 529.854 * source_params.eta)
            * source_params.eta
            * source_params.S_tot_hat
            + (21.0314 - 240.648 * source_params.eta + 516.333 * source_params.eta_pow2)
            * source_params.S_tot_hat_pow2
            + (20.3384 - 356.241 * source_params.eta + 999.417 * source_params.eta_pow2)
            * source_params.S_tot_hat_pow3
            + 97.1364
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
        )

    @ti.func
    def _set_int_colloc_value_5(self, source_params: ti.template()):
        self._int_colloc_values[5] = -source_params.peak_time_diff + (
            3096.03
            + 986.752 * source_params.eta
            - 20371.1 * source_params.eta_pow2
            + 220332.0 * source_params.eta_pow3
            - 1.31523e6 * source_params.eta_pow4
            + 4.29193e6 * source_params.eta_pow5
            - 6.01179e6 * source_params.eta_pow6
            + (
                -9.96292
                - 118.526 * source_params.eta
                + 2255.76 * source_params.eta_pow2
                - 6758.52 * source_params.eta_pow3
            )
            * source_params.S_tot_hat
            + (
                -14.4869
                + 370.039 * source_params.eta
                - 3605.8 * source_params.eta_pow2
                + 9129.45 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow2
            + (
                17.0209
                + 70.1931 * source_params.eta
                - 3070.08 * source_params.eta_pow2
                + 9582.58 * source_params.eta_pow3
            )
            * source_params.S_tot_hat_pow3
            + 23.0759
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
        )

    @ti.func
    def update_phase_coefficients(
        self,
        pn_coefficients_22: ti.template(),
        pn_coefficients_44: ti.template(),
        phase_coefficients_22: ti.template(),
        source_params: ti.template(),
    ):
        self._set_joint_frequencies_no_mixing(
            source_params.f_MECO_lm["44"], source_params.QNM_freqs_lm["44"]
        )
        # intermediate
        self._set_int_colloc_points_no_mixing(
            source_params.QNM_freqs_lm["44"].f_ring, source_params.eta
        )
        self._set_intermediate_coefficients(source_params)
        # inspiral
        self._set_ins_rescaling_coefficients(4.0, phase_coefficients_22)
        if source_params.eta > 0.01:
            self.Lambda_lm = self._Lambda_44_PN(source_params)
        else:
            self.Lambda_lm = self._Lambda_44_fit(source_params)
        # merge-ringdown
        self._set_MRD_rescaling_coefficients(2.0, source_params)
        # continuity conditions
        self._set_connection_coefficients(
            source_params.QNM_freqs_lm["44"], pn_coefficients_44
        )
        # the constant phase for aligning modes
        self._set_delta_phi_lm(
            4.0,
            source_params.f_MECO_lm["44"],
            pn_coefficients_22,
            pn_coefficients_44,
            phase_coefficients_22,
            source_params,
        )


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
        # # zero for odd modes (21, 33) with equal black holes
        # if (mass_1 == mass_2) and (chi_1 == chi_2):
        #     return 0.0
        pass

    @ti.func
    def _parameter_check(self):
        # TODO
        pass
