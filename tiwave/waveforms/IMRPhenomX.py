import os
import warnings
from typing import Optional

import taichi as ti
import taichi.math as tm
import numpy as np

from ..constants import *
from ..utils import ComplexNumber, gauss_elimination
from .base_waveform import BaseWaveform


# Amplitude ansatz
@ti.func
def _amplitude_inspiral_ansatz(powers_of_Mf, amplitude_coefficients, pn_prefactors):
    """
    Eq.30 in arXiv:1508.07253, without amp0
    """
    return (
        1.0
        + pn_prefactors.prefactor_A_2 * powers_of_Mf.two_thirds
        + pn_prefactors.prefactor_A_3 * powers_of_Mf.one
        + pn_prefactors.prefactor_A_4 * powers_of_Mf.four_thirds
        + pn_prefactors.prefactor_A_5 * powers_of_Mf.five_thirds
        + pn_prefactors.prefactor_A_6 * powers_of_Mf.two
        + amplitude_coefficients.rho_1 * powers_of_Mf.seven_thirds
        + amplitude_coefficients.rho_2 * powers_of_Mf.eight_thirds
        + amplitude_coefficients.rho_3 * powers_of_Mf.three
    )


@ti.func
def _d_amplitude_inspiral_ansatz(powers_of_Mf, amplitude_coefficients, pn_prefactors):
    """
    without amp0
    """
    return (
        2.0 * pn_prefactors.prefactor_A_2 / powers_of_Mf.third
        + 3.0 * pn_prefactors.prefactor_A_3
        + 4.0 * pn_prefactors.prefactor_A_4 * powers_of_Mf.third
        + 5.0 * pn_prefactors.prefactor_A_5 * powers_of_Mf.two_thirds
        + 6.0 * pn_prefactors.prefactor_A_6 * powers_of_Mf.one
        + 7.0 * amplitude_coefficients.rho_1 * powers_of_Mf.four_thirds
        + 8.0 * amplitude_coefficients.rho_2 * powers_of_Mf.five_thirds
        + 9.0 * amplitude_coefficients.rho_3 * powers_of_Mf.two
    ) / 3.0


@ti.func
def _amplitude_intermediate_ansatz(powers_of_Mf, amplitude_coefficients):
    """
    without amp0
    """
    return (
        amplitude_coefficients.delta_0
        + amplitude_coefficients.delta_1 * powers_of_Mf.one
        + amplitude_coefficients.delta_2 * powers_of_Mf.two
        + amplitude_coefficients.delta_3 * powers_of_Mf.three
        + amplitude_coefficients.delta_4 * powers_of_Mf.four
    )


@ti.func
def _amplitude_merge_ringdown_ansatz(
    powers_of_Mf, amplitude_coefficients, f_ring, f_damp
):
    """
    without amp0
    """
    f_minus_fring = powers_of_Mf.one - f_ring
    fdamp_gamma3 = amplitude_coefficients.gamma_3 * f_damp
    return (
        amplitude_coefficients.gamma_1
        * fdamp_gamma3
        / (f_minus_fring**2 + fdamp_gamma3**2)
        * tm.exp(-f_minus_fring * amplitude_coefficients.gamma_2 / fdamp_gamma3)
    )


@ti.func
def _d_amplitude_merge_ringdown_ansatz(
    powers_of_Mf, amplitude_coefficients, f_ring, f_damp
):
    """
    without amp0
    """
    fdamp_gamma3 = amplitude_coefficients.gamma_3 * f_damp
    pow2_fdamp_gamma3 = fdamp_gamma3 * fdamp_gamma3
    f_minus_fring = powers_of_Mf.one - f_ring
    exp_factor = tm.exp(-f_minus_fring * amplitude_coefficients.gamma_2 / fdamp_gamma3)
    pow2_plus_pow2 = f_minus_fring**2 + pow2_fdamp_gamma3
    return (
        exp_factor
        / pow2_plus_pow2
        * (
            -2
            * f_damp
            * amplitude_coefficients.gamma_1
            * amplitude_coefficients.gamma_3
            * f_minus_fring
            / pow2_plus_pow2
            - amplitude_coefficients.gamma_1 * amplitude_coefficients.gamma_2
        )
    )


# Phase ansatz
@ti.func
def _phase_inspiral_ansatz(powers_of_Mf, phase_coefficients, pn_prefactors):
    """
    without 1/eta
    """
    return 3.0 / 128.0 * (
        pn_prefactors.prefactor_varphi_0 / powers_of_Mf.five_thirds
        + pn_prefactors.prefactor_varphi_1 / powers_of_Mf.four_thirds
        + pn_prefactors.prefactor_varphi_2 / powers_of_Mf.one
        + pn_prefactors.prefactor_varphi_3 / powers_of_Mf.two_thirds
        + pn_prefactors.prefactor_varphi_4 / powers_of_Mf.third
        + pn_prefactors.prefactor_varphi_5
        + pn_prefactors.prefactor_varphi_5l * (powers_of_Mf.log + useful_powers_pi.log)
        + pn_prefactors.prefactor_varphi_6 * powers_of_Mf.third
        + pn_prefactors.prefactor_varphi_6l
        * powers_of_Mf.third
        * (powers_of_Mf.log + useful_powers_pi.log)
        + pn_prefactors.prefactor_varphi_7 * powers_of_Mf.two_thirds
    ) + (
        phase_coefficients.sigma_1 * powers_of_Mf.one
        + 0.75 * phase_coefficients.sigma_2 * powers_of_Mf.four_thirds
        + 0.6 * phase_coefficients.sigma_3 * powers_of_Mf.five_thirds
        + 0.5 * phase_coefficients.sigma_4 * powers_of_Mf.two
    )


@ti.func
def _d_phase_inspiral_ansatz(powers_of_Mf, phase_coefficients, pn_prefactors):
    """
    without 1/eta
    """
    return 3.0 / 128.0 * (
        -5.0 * pn_prefactors.prefactor_varphi_0 / powers_of_Mf.eight_thirds
        - 4.0 * pn_prefactors.prefactor_varphi_1 / powers_of_Mf.seven_thirds
        - 3.0 * pn_prefactors.prefactor_varphi_2 / powers_of_Mf.two
        - 2.0 * pn_prefactors.prefactor_varphi_3 / powers_of_Mf.five_thirds
        - 1.0 * pn_prefactors.prefactor_varphi_4 / powers_of_Mf.four_thirds
        + 3 * pn_prefactors.prefactor_varphi_5l / powers_of_Mf.one
        + pn_prefactors.prefactor_varphi_6 / powers_of_Mf.two_thirds
        + pn_prefactors.prefactor_varphi_6l
        / powers_of_Mf.two_thirds
        * (3.0 + powers_of_Mf.log + useful_powers_pi.log)
        + 2.0 * pn_prefactors.prefactor_varphi_7 / powers_of_Mf.third
    ) / 3.0 + (
        phase_coefficients.sigma_1
        + phase_coefficients.sigma_2 * powers_of_Mf.third
        + phase_coefficients.sigma_3 * powers_of_Mf.two_thirds
        + phase_coefficients.sigma_4 * powers_of_Mf.one
    )


@ti.func
def _phase_intermediate_ansatz(powers_of_Mf, phase_coefficients):
    """
    without 1/eta
    """
    return (
        phase_coefficients.beta_1 * powers_of_Mf.one
        + phase_coefficients.beta_2 * powers_of_Mf.log
        - phase_coefficients.beta_3 / 3.0 / powers_of_Mf.three
    )


@ti.func
def _d_phase_intermediate_ansatz(powers_of_Mf, phase_coefficients):
    """
    without 1/eta
    """
    return (
        phase_coefficients.beta_1
        + phase_coefficients.beta_2 / powers_of_Mf.one
        + phase_coefficients.beta_3 / powers_of_Mf.four
    )


@ti.func
def _phase_merge_ringdown_ansatz(powers_of_Mf, phase_coefficients, f_ring, f_damp):
    """
    without 1/eta
    """
    return (
        phase_coefficients.alpha_1 * powers_of_Mf.one
        - phase_coefficients.alpha_2 / powers_of_Mf.one
        + 4.0 / 3.0 * phase_coefficients.alpha_3 * powers_of_Mf.three_fourths
        +
        # note that tm.atan2 return the value in [-pi, pi], make sure f_damp > 0
        phase_coefficients.alpha_4
        * tm.atan2((powers_of_Mf.one - phase_coefficients.alpha_5 * f_ring), f_damp)
    )


@ti.func
def _d_phase_merge_ringdown_ansatz(powers_of_Mf, phase_coefficients, f_ring, f_damp):
    """
    without 1/eta
    """
    return (
        phase_coefficients.alpha_1
        + phase_coefficients.alpha_2 / powers_of_Mf.two
        + phase_coefficients.alpha_3 / powers_of_Mf.fourth
        + phase_coefficients.alpha_4
        * f_damp
        / (f_damp**2 + (powers_of_Mf.one - phase_coefficients.alpha_5 * f_ring) ** 2)
    )


@ti.dataclass
class SourceParameters:
    # passed in parameters
    M: ti.f64  # total mass
    q: ti.f64
    chi_1: ti.f64
    chi_2: ti.f64
    dL_Mpc: ti.f64
    iota: ti.f64
    phase_ref: ti.f64
    tc: ti.f64
    # base parameters
    dL_SI: ti.f64
    mass_1: ti.f64
    mass_2: ti.f64
    M_sec: ti.f64  # total mass in second
    eta: ti.f64  # symmetric_mass_ratio
    eta2: ti.f64  # eta^2
    eta3: ti.f64
    delta: ti.f64
    chi_s: ti.f64
    chi_a: ti.f64
    chi_s2: ti.f64
    chi_a2: ti.f64
    chi_PN: ti.f64
    xi: ti.f64
    # derived parameters
    f_amp_ins_max: ti.f64

    @ti.func
    def update_all_source_parameters(self, parameters):
        # total 11 parameters: m1, m1, chi1, chi2, iota, psi, tc, phi0, dL, lon, lat
        # 3 only used in response function: psi, lon, lat
        # mass is in the unit of solar mass
        # dL is in the unit of Mpc
        self.M = parameters["total_mass"]
        self.q = parameters["mass_ratio"]
        self.chi_1 = parameters["chi_1"]
        self.chi_2 = parameters["chi_2"]
        self.dL_Mpc = parameters["luminosity_distance"]
        self.iota = parameters["inclination"]
        self.phase_ref = parameters["reference_phase"]
        self.tc = parameters["coalescence_time"]
        # base parameters
        self.mass_1 = self.M / (1 + self.q)
        self.mass_2 = self.M - self.mass_1
        self.dL_SI = self.dL_Mpc * 1e6 * PC_SI
        self.M_sec = self.M * MTSUN_SI
        self.eta = self.mass_1 * self.mass_2 / (self.M * self.M)
        self.eta2 = self.eta * self.eta
        self.eta3 = self.eta * self.eta2
        self.eta4 = self.eta + self.eta3

        self.delta_chi = self.chi_1 - self.chi_2
        self.delta_chi_pow2 = self.delta_chi * self.delta_chi

        self.delta = tm.sqrt(1.0 - 4.0 * self.eta)
        self.chi_PN = (
            self.mass_1 * self.chi_1 + self.mass_2 * self.chi_2
        ) / self.M - 38.0 / 113.0 * self.eta * (self.chi_1 + self.chi_2)
        self.chi_PN_hat = (
            (self.mass_1 * self.chi_1 + self.mass_2 * self.chi_2) / self.M
            - 38.0 / 113.0 * self.eta * (self.chi_1 + self.chi_2)
        ) / (1.0 - (76.0 / 113.0 * self.eta))
        self.chi_PN_hat_pow2 = self.chi_PN_hat * self.chi_PN_hat
        self.chi_PN_hat_pow3 = self.chi_PN_hat * self.chi_PN_hat_pow2

        self.chi_f = self._final_spin()
        self.chi_f_pow2 = self.chi_f * self.chi_f
        self.chi_f_pow3 = self.chi_f * self.chi_f_pow2
        self.chi_f_pow4 = self.chi_f * self.chi_f_pow3
        self.chi_f_pow5 = self.chi_f * self.chi_f_pow4
        self.chi_f_pow6 = self.chi_f * self.chi_f_pow5
        self.chi_f_pow7 = self.chi_f * self.chi_f_pow6

        self.f_MECO = self._f_MECO()
        self.f_ISCO = self._f_ISCO()

        self.f_amp_ins_max = self.f_MECO + 0.25 * (self.f_ISCO - self.f_MECO)

        self.S_tot_hat = (
            self.mass_1 * self.mass_1 * self.chi_1
            + self.mass_2 * self.mass_2 * self.chi_2
        ) / (self.M * self.M)

        self.f_ring = (
            0.05947169566573468
            - 0.14989771215394762 * self.chi_f
            + 0.09535606290986028 * self.chi_f_pow2
            + 0.02260924869042963 * self.chi_f_pow3
            - 0.02501704155363241 * self.chi_f_pow4
            - 0.005852438240997211 * self.chi_f_pow5
            + 0.0027489038393367993 * self.chi_f_pow6
            + 0.0005821983163192694 * self.chi_f_pow7
        ) / (
            1
            - 2.8570126619966296 * self.chi_f
            + 2.373335413978394 * self.chi_f_pow2
            - 0.6036964688511505 * self.chi_f_pow4
            + 0.0873798215084077 * self.chi_f_pow6
        )
        self.f_damp = (
            0.014158792290965177
            - 0.036989395871554566 * self.chi_f
            + 0.026822526296575368 * self.chi_f_pow2
            + 0.0008490933750566702 * self.chi_f_pow3
            - 0.004843996907020524 * self.chi_f_pow4
            - 0.00014745235759327472 * self.chi_f_pow5
            + 0.0001504546201236794 * self.chi_f_pow6
        ) / (
            1
            - 2.5900842798681376 * self.chi_f
            + 1.8952576220623967 * self.chi_f_pow2
            - 0.31416610693042507 * self.chi_f_pow4
            + 0.009002719412204133 * self.chi_f_pow6
        )

    @ti.func
    def _final_spin(self):
        """Final dimensionless spin, PhysRevD.95.064024"""
        no_spin = (
            3.4641016151377544 * self.eta
            + 20.0830030082033 * self.eta2
            - 12.333573402277912 * self.eta3
        ) / (1 + 7.2388440419467335 * self.eta)
        eq_spin = (self.m1_pow2 + self.m2_pow2) * self.S_tot + (
            (
                -0.8561951310209386 * self.eta
                - 0.09939065676370885 * self.eta2
                + 1.668810429851045 * self.eta3
            )
            * self.S_tot
            + (
                0.5881660363307388 * self.eta
                - 2.149269067519131 * self.eta2
                + 3.4768263932898678 * self.eta3
            )
            * self.S_tot_pow2
            + (
                0.142443244743048 * self.eta
                - 0.9598353840147513 * self.eta2
                + 1.9595643107593743 * self.eta3
            )
            * self.S_tot_pow3
        ) / (
            1
            + (
                -0.9142232693081653
                + 2.3191363426522633 * self.eta
                - 9.710576749140989 * self.eta3
            )
            * self.S_tot
        )
        uneq_spin = (
            0.3223660562764661
            * self.delta_chi
            * self.delta
            * (1 + 9.332575956437443 * self.eta)
            * self.eta2
            - 0.059808322561702126 * self.delta_chi_pow2 * self.eta3
            + 2.3170397514509933
            * self.delta_chi
            * self.delta
            * (1 - 3.2624649875884852 * self.eta)
            * self.eta3
            * self.delta_chi
        )
        return no_spin + eq_spin + uneq_spin

    @ti.func
    def _f_MECO(self):
        # Frequency of the minimum energy circular orbit (MECO).
        no_spin = (
            0.018744340279608845
            + 0.0077903147004616865 * self.eta
            + 0.003940354686136861 * self.eta2
            - 0.00006693930988501673 * self.eta3
        ) / (1.0 - 0.10423384680638834 * self.eta)
        eq_spin = (
            self.chi_PN_hat
            * (
                0.00027180386951683135
                - 0.00002585252361022052 * self.chi_PN_hat
                + self.eta4
                * (
                    -0.0006807631931297156
                    + 0.022386313074011715 * self.chi_PN_hat
                    - 0.0230825153005985 * self.chi_PN_hat_pow2
                )
                + self.eta2
                * (
                    0.00036556167661117023
                    - 0.000010021140796150737 * self.chi_PN_hat
                    - 0.00038216081981505285 * self.chi_PN_hat_pow2
                )
                + self.eta
                * (
                    0.00024422562796266645
                    - 0.00001049013062611254 * self.chi_PN_hat
                    - 0.00035182990586857726 * self.chi_PN_hat_pow2
                )
                + self.eta3
                * (
                    -0.0005418851224505745
                    + 0.000030679548774047616 * self.chi_PN_hat
                    + 4.038390455349854e-6 * self.chi_PN_hat_pow2
                )
                - 0.00007547517256664526 * self.chi_PN_hat_pow2
            )
        ) / (
            0.026666543809890402
            + (
                -0.014590539285641243
                - 0.012429476486138982 * self.eta
                + 1.4861197211952053 * self.eta4
                + 0.025066696514373803 * self.eta2
                + 0.005146809717492324 * self.eta3
            )
            * self.chi_PN_hat
            + (
                -0.0058684526275074025
                - 0.02876774751921441 * self.eta
                - 2.551566872093786 * self.eta4
                - 0.019641378027236502 * self.eta2
                - 0.001956646166089053 * self.eta3
            )
            * self.chi_PN_hat_pow2
            + (
                0.003507640638496499
                + 0.014176504653145768 * self.eta
                + 1.0 * self.eta4
                + 0.012622225233586283 * self.eta2
                - 0.00767768214056772 * self.eta3
            )
            * self.chi_PN_hat_pow3
        )
        uneq_spin = self.delta_chi_pow2 * (
            0.00034375176678815234 + 0.000016343732281057392 * self.eta
        ) * self.eta2 + self.delta_chi * self.delta * self.eta * (
            0.08064665214195679 * self.eta2
            + self.eta
            * (-0.028476219509487793 - 0.005746537021035632 * self.chi_PN_hat)
            - 0.0011713735642446144 * self.chi_PN_hat
        )
        return no_spin + eq_spin + uneq_spin

    @ti.func
    def _f_ISCO(self):
        """Frequency of the innermost stable circular orbit (ISCO)."""
        Z1 = 1.0 + (1.0 - self.chi_f_pow2) ** (1 / 3) * (
            (1 + self.chi_f) ** (1 / 3) + (1 - self.chi_f) ** (1 / 3)
        )
        if Z1 > 3.0:
            Z1 = 3.0
        Z2 = tm.sqrt(3.0 * self.chi_f_pow2 + Z1 * Z1)

        return (
            1.0
            / (
                (3.0 + Z2 - tm.sign(self.chi_f) * tm.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2)))
                ** (3 / 2)
                + self.chi_f
            )
            / PI
        )


# Amplitude coefficients
@ti.dataclass
class AmplitudeCoefficients:
    # Inspiral
    rho_1: ti.f64
    rho_2: ti.f64
    rho_3: ti.f64
    # Intermediate
    alpha_0: ti.f64
    alpha_1: ti.f64
    alpha_2: ti.f64
    alpha_3: ti.f64
    alpha_4: ti.f64
    alpha_5: ti.f64
    # Merge-ringdown
    gamma_1: ti.f64
    gamma_2: ti.f64
    gamma_3: ti.f64
    # derived coefficients
    amp0: ti.f64

    @ti.func
    def compute_amplitude_coefficients(self, source_params, pn_prefactors, amp_int_ver):
        # Inspiral coefficients: rho_1, rho_2, rho_3

        # The amplitude calibrating collocation points.
        # TODO: the equation of Eq. 6.5 in arXiv: 2001.11412 uses fMECO, while fAT (Eq. 5.7)
        # is used in lalsim (L. 781 in LALSimIMRPhenomX_internals.c). Confirm this!
        f1_ins = 0.5 * source_params.f_amp_ins_max
        f2_ins = 0.75 * source_params.f_amp_ins_max
        f3_ins = source_params.f_amp_ins_max

        # Value for amplitude collocation point at 0.5 f^A_T,
        v1_ins = (
            (
                -0.015178276424448592
                - 0.06098548699809163 * source_params.eta
                + 0.4845148547154606 * source_params.eta2
            )
            / (1.0 + 0.09799277215675059 * source_params.eta)
            + (
                (0.02300153747158323 + 0.10495263104245876 * source_params.eta2)
                * source_params.chi_PN_hat
                + (0.04834642258922544 - 0.14189350657140673 * source_params.eta)
                * source_params.eta
                * source_params.chi_PN_hat_pow3
                + (0.01761591799745109 - 0.14404522791467844 * source_params.eta2)
                * source_params.chi_PN_hat_pow2
            )
            / (1.0 - 0.7340448493183307 * source_params.chi_PN_hat)
            + source_params.delta_chi
            * source_params.delta
            * source_params.eta4
            * (0.0018724905795891192 + 34.90874132485147 * source_params.eta)
        )
        # Value for amplitude collocation point at 0.75 f^A_T,
        v2_ins = (
            (
                -0.058572000924124644
                - 1.1970535595488723 * source_params.eta
                + 8.4630293045015 * source_params.eta2
            )
            / (1.0 + 15.430818840453686 * source_params.eta)
            + (
                (
                    -0.08746408292050666
                    + source_params.eta
                    * (
                        -0.20646621646484237
                        - 0.21291764491897636 * source_params.chi_PN_hat
                    )
                    + source_params.eta2
                    * (
                        0.788717372588848
                        + 0.8282888482429105 * source_params.chi_PN_hat
                    )
                    - 0.018924013869130434 * source_params.chi_PN_hat
                )
                * source_params.chi_PN_hat
            )
            / (-1.332123330797879 + 1.0 * source_params.chi_PN_hat)
            + source_params.dchi
            * source_params.delta
            * source_params.eta4
            * (0.004389995099201855 + 105.84553997647659 * source_params.eta)
        )
        # Value for amplitude collocation point at 1.0 f^A_T,
        v3_ins = (
            (
                -0.16212854591357853
                + 1.617404703616985 * source_params.eta
                - 3.186012733446088 * source_params.eta2
                + 5.629598195000046 * source_params.eta3
            )
            / (1.0 + 0.04507019231274476 * source_params.eta)
            + (
                source_params.chi_PN_hat
                * (
                    1.0055835408962206
                    + source_params.eta2
                    * (
                        18.353433894421833
                        - 18.80590889704093 * source_params.chi_PN_hat
                    )
                    - 0.31443470118113853 * source_params.chi_PN_hat
                    + source_params.eta
                    * (
                        -4.127597118865669
                        + 5.215501942120774 * source_params.chi_PN_hat
                    )
                    + source_params.eta3
                    * (
                        -41.0378120175805
                        + 19.099315016873643 * source_params.chi_PN_hat
                    )
                )
            )
            / (
                5.852706459485663
                - 5.717874483424523 * source_params.chi_PN_hat
                + 1.0 * source_params.chi_PN_hat_pow2
            )
            + source_params.delta_chi
            * source_params.delta
            * source_params.eta4
            * (0.05575955418803233 + 208.92352600701068 * source_params.eta)
        )

        Ab_ins = ti.Matrix(
            [
                [f1_ins ** (7 / 3), f1_ins ** (8 / 3), f1_ins**3, v1_ins],
                [f2_ins ** (7 / 3), f2_ins ** (8 / 3), f2_ins**3, v2_ins],
                [f3_ins ** (7 / 3), f3_ins ** (8 / 3), f3_ins**3, v3_ins],
            ],
            dt=ti.f64,
        )
        self.rho_1, self.rho_2, self.rho_3 = gauss_elimination(Ab_ins)

        # Inermediate coefficients:
        # version 104: alpha_0, alpha_1, alpha_2, alpha_3, alpha_4 (default)
        # version 105: alpha_0, alpha_1, alpha_2, alpha_3, alpha_4, alpha_5
        if ti.static(amp_int_ver == 104):
            f1_int = source_params.f_amp_ins_max
            f3_int = source_params.f_peak
            f2_int = (f1_int + f3_int) / 2.0

            v1_int = 1 / _amplitude_inspiral_ansatz()
            v2_int = (
                (
                    1.4873184918202145
                    + 1974.6112656679577 * source_params.eta
                    + 27563.641024162127 * source_params.eta2
                    - 19837.908020966777 * source_params.eta3
                )
                / (
                    1.0
                    + 143.29004876335128 * source_params.eta
                    + 458.4097306093354 * source_params.eta2
                )
                + (
                    source_params.S_tot_hat
                    * (
                        27.952730865904343
                        + source_params.eta
                        * (
                            -365.55631765202895
                            - 260.3494489873286 * source_params.S_tot_hat
                        )
                        + 3.2646808851249016 * source_params.S_tot_hat
                        + 3011.446602208493
                        * source_params.eta2
                        * source_params.S_tot_hat
                        - 19.38970173389662 * source_params.S_tot_hat_pow2
                        + source_params.eta3
                        * (
                            1612.2681322644232
                            - 6962.675551371755 * source_params.S_tot_hat
                            + 1486.4658089990298 * source_params.S_tot_hat_pow2
                        )
                    )
                )
                / (
                    12.647425554323242
                    - 10.540154508599963 * source_params.S_tot_hat
                    + 1.0 * source_params.S_tot_hat_pow2
                )
                + self.delta_chi
                * self.delta
                * (-0.016404056649860943 - 296.473359655246 * source_params.eta)
                * source_params.eta2
            )

            v3_int = 1 / _amplitude_merge_ringdown_ansatz()
            d1_int = _d_amplitude_inspiral_ansatz()
            d3_int = _d_amplitude_merge_ringdown_ansatz()

            Ab_int = ti.Matrix([[], [], [], [], []])

            (
                self.alpha_0,
                self.alpha_1,
                self.alpha_2,
                self.alpha_3,
                self.alpha_4,
            ) = gauss_elimination(Ab_int)

        elif ti.static(amp_int_ver == 105):
            f1_int = source_params.f_amp_ins_max
            f4_int = source_params.f_peak
            f2_int = f1_int + (f4_int - f1_int) / 3.0
            f3_int = f1_int + (f4_int - f1_int) * 2.0 / 3.0

            v1_int = 1 / _amplitude_inspiral_ansatz()
            v2_int = (
                (
                    2.2436523786378983
                    + 2162.4749081764216 * source_params.eta
                    + 24460.158604784723 * source_params.eta2
                    - 12112.140570900956 * source_params.eta3
                )
                / (
                    1.0
                    + 120.78623282522702 * source_params.eta
                    + 416.4179522274108 * source_params.eta2
                )
                + (
                    source_params.S_tot_hat
                    * (
                        6.727511603827924
                        + source_params.eta2
                        * (
                            414.1400701039126
                            - 234.3754066885935 * source_params.S_tot_hat
                        )
                        - 5.399284768639545 * source_params.S_tot_hat
                        + source_params.eta
                        * (
                            -186.87972530996245
                            + 128.7402290554767 * source_params.S_tot_hat
                        )
                    )
                )
                / (
                    3.24359204029217
                    - 3.975650468231452 * source_params.S_tot_hat
                    + 1.0 * source_params.S_tot_hat_pow2
                )
                + source_params.delta_chi
                * self.delta
                * (-59.52510939953099 + 13.12679437100751 * source_params.eta)
                * source_params.eta2
            )

            v3_int = (
                (
                    1.195392410912163
                    + 1677.2558976605421 * source_params.eta
                    + 24838.37133975971 * source_params.eta2
                    - 17277.938868280915 * source_params.eta3
                )
                / (
                    1.0
                    + 144.78606839716073 * source_params.eta
                    + 428.8155916011666 * source_params.eta2
                )
                + (
                    source_params.S_tot_hat
                    * (
                        -2.1413952025647793
                        + 0.5719137940424858 * source_params.S_tot_hat
                        + source_params.eta
                        * (
                            46.61350006858767
                            + 0.40917927503842105 * source_params.S_tot_hat
                            - 11.526500209146906 * source_params.S_tot_hat_pow2
                        )
                        + 1.1833965566688387 * source_params.S_tot_hat_pow2
                        + source_params.eta2
                        * (
                            -84.82318288272965
                            - 34.90591158988979 * source_params.S_tot_hat
                            + 19.494962340530186 * source_params.S_tot_hat_pow2
                        )
                    )
                )
                / (-1.4786392693666195 + 1.0 * source_params.S_tot_hat)
                + source_params.delta_chi
                * source_params.delta
                * (-333.7662575986524 + 532.2475589084717 * source_params.eta)
                * source_params.eta3
            )

            v4_int = 1 / _amplitude_merge_ringdown_ansatz()
            d1_int = _d_amplitude_inspiral_ansatz()
            d4_int = _d_amplitude_merge_ringdown_ansatz()

            Ab_int = ti.Matrix([[], [], [], [], [], []])

            (
                self.alpha_0,
                self.alpha_1,
                self.alpha_2,
                self.alpha_3,
                self.alpha_4,
                self.alpha_5,
            ) = gauss_elimination(Ab_int)

        # Merge-ringdown coefficients: lambda (gamma_2), sigma (gamma_3), a_R (gamma_1)
        self.gamma_2 = (
            (
                0.8312293675316895
                + 7.480371544268765 * source_params.eta
                - 18.256121237800397 * source_params.eta2
            )
            / (
                1.0
                + 10.915453595496611 * source_params.eta
                - 30.578409433912874 * source_params.eta2
            )
            + (
                source_params.S_tot_hat
                * (
                    0.5869408584532747
                    + source_params.eta
                    * (
                        -0.1467158405070222
                        - 2.8489481072076472 * source_params.S_tot_hat
                    )
                    + 0.031852563636196894 * source_params.S_tot_hat
                    + source_params.eta2
                    * (
                        0.25295441250444334
                        + 4.6849496672664594 * source_params.S_tot_hat
                    )
                )
            )
            / (
                3.8775263105069953
                - 3.41755361841226 * source_params.S_tot_hat
                + 1.0 * source_params.S_tot_hat_pow2
            )
            + -0.00548054788508203
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta
        )
        self.gamma_3 = (
            1.3666000000000007
            - 4.091333144596439 * source_params.eta
            + 2.109081209912545 * source_params.eta2
            - 4.222259944408823 * source_params.eta3
        ) / (1.0 - 2.7440263888207594 * source_params.eta) + (
            0.07179105336478316
            + source_params.eta2
            * (2.331724812782498 - 0.6330998412809531 * source_params.S_tot_hat)
            + source_params.eta
            * (-0.8752427297525086 + 0.4168560229353532 * source_params.S_tot_hat)
            - 0.05633734476062242 * source_params.S_tot_hat
        ) * source_params.S_tot_hat


@ti.data_oriented
class IMRPhenomXAS(BaseWaveform):
    def __init__(
        self,
        frequencies: ti.ScalarField,
        waveform_container: Optional[ti.StructField] = None,
        reference_frequency: Optional[float] = None,
        returned_form: str = "polarizations",
        include_tf: bool = True,
        parameter_sanity_check: bool = False,
    ) -> None:
        """
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
        """
        self.frequencies = frequencies
        if reference_frequency is None:
            self.reference_frequency = self.frequencies[0]
        elif reference_frequency <= 0.0:
            raise ValueError(
                f"you are set reference_frequency={reference_frequency}, which must be postive."
            )
        else:
            self.reference_frequency = reference_frequency

        # TODO: make the sanity checks do not depend on the taichi debug mode
        self.parameter_sanity_check = parameter_sanity_check
        if self.parameter_sanity_check:
            warnings.warn(
                "`parameter_sanity_check` is turn-on, make sure taichi is initialized with debug mode"
            )
        else:
            warnings.warn(
                "`parameter_sanity_check` is disable, make sure all parameters passed in are valid."
            )

        if waveform_container is not None:
            if not (waveform_container.shape == frequencies.shape):
                raise ValueError(
                    "passed in `waveform_container` and `frequencies` have different shape"
                )
            self.waveform_container = waveform_container
            ret_content = self.waveform_container.keys
            if "tf" in ret_content:
                include_tf = True
                ret_content.remove("tf")
            else:
                include_tf = False
            if all([item in ret_content for item in ["hplus", "hcross"]]):
                returned_form = "polarizations"
                [ret_content.remove(item) for item in ["hplus", "hcross"]]
            elif all([item in ret_content for item in ["amplitude", "phase"]]):
                returned_form = "amplitude_phase"
                [ret_content.remove(item) for item in ["amplitude", "phase"]]
            if len(ret_content) > 0:
                raise ValueError(
                    f"`waveform_container` contains additional unknown keys {ret_content}."
                )
            self.returned_form = returned_form
            self.include_tf = include_tf
            print(
                f"Using `waveform_container` passed in, updating returned_form={self.returned_form}, include_tf={self.include_tf}"
            )
        else:
            self._initialize_waveform_container(returned_form, include_tf)
            self.returned_form = returned_form
            self.include_tf = include_tf
            print(
                f"`waveform_container` is not given, initializing one with returned_form={returned_form}, include_tf={include_tf}"
            )

        # initializing data struct with 0, and instantiating fields for global accessing
        self.source_parameters = SourceParameters.field(shape=())
        self.phase_coefficients = PhaseCoefficients.field(shape=())
        self.amplitude_coefficients = AmplitudeCoefficients.field(shape=())
        self.pn_prefactors = PostNewtonianPrefactors.field(shape=())

    def _initialize_waveform_container(
        self, returned_form: str, include_tf: bool
    ) -> None:
        ret_content = {}
        if returned_form == "polarizations":
            ret_content.update({"hplus": ComplexNumber, "hcross": ComplexNumber})
        elif returned_form == "amplitude_phase":
            ret_content.update({"amplitude": ti.f64, "phase": ti.f64})
        else:
            raise Exception(
                f"{returned_form} is unknown. `returned_form` can only be one of `polarizations` and `amplitude_phase`"
            )

        if include_tf:
            ret_content.update({"tf": ti.f64})

        self.waveform_container = ti.Struct.field(
            ret_content,
            shape=(self.frequencies.length,),
        )
        return None

    def update_waveform(self, parameters: dict[str, float]):
        """
        necessary preparation which need to be finished in python scope for waveform computation
        (this function may be awkward, since no interpolation function in taichi-lang)
        """
        self.source_parameters[None].generate_all_source_parameters(parameters)
        self._update_waveform_kernel()

    @ti.kernel
    def _update_waveform_kernel(self):
        if ti.static(self.parameter_sanity_check):
            self._parameter_check()

        self.pn_prefactors[None].compute_PN_prefactors(self.source_parameters[None])
        self.amplitude_coefficients[None].compute_amplitude_coefficients(
            self.source_parameters[None], self.pn_prefactors[None]
        )
        self.phase_coefficients[None].compute_phase_coefficients(
            self.source_parameters[None], self.pn_prefactors[None]
        )

        powers_of_Mf = UsefulPowers()

        powers_of_Mf.updating(self.amplitude_coefficients[None].f_peak)
        t0 = (
            _d_phase_merge_ringdown_ansatz(
                powers_of_Mf,
                self.phase_coefficients[None],
                self.source_parameters[None].f_ring,
                self.source_parameters[None].f_damp,
            )
            / self.source_parameters[None].eta
        )
        # t0 = (_d_phase_merge_ringdown_ansatz(powers_of_Mf, self.phase_coefficients[None], self.source_parameters[None].f_ring, self.source_parameters[None].f_damp) + self.phase_coefficients[None].C2_merge_ringdown)/self.source_parameters[None].eta
        time_shift = (
            t0
            - 2
            * PI
            * self.source_parameters[None].tc
            / self.source_parameters[None].M_sec
        )
        Mf_ref = self.source_parameters[None].M_sec * self.reference_frequency
        powers_of_Mf.updating(Mf_ref)
        phase_ref_temp = _compute_phase(
            powers_of_Mf,
            self.phase_coefficients[None],
            self.pn_prefactors[None],
            self.source_parameters[None].f_ring,
            self.source_parameters[None].f_damp,
            self.source_parameters[None].eta,
        )
        phase_shift = 2.0 * self.source_parameters[None].phase_ref + phase_ref_temp

        for idx in self.frequencies:
            Mf = self.source_parameters[None].M_sec * self.frequencies[idx]
            if Mf < FREQUENCY_CUT:
                powers_of_Mf.updating(Mf)
                amplitude = _compute_amplitude(
                    powers_of_Mf,
                    self.amplitude_coefficients[None],
                    self.pn_prefactors[None],
                    self.source_parameters[None].f_ring,
                    self.source_parameters[None].f_damp,
                )
                phase = _compute_phase(
                    powers_of_Mf,
                    self.phase_coefficients[None],
                    self.pn_prefactors[None],
                    self.source_parameters[None].f_ring,
                    self.source_parameters[None].f_damp,
                    self.source_parameters[None].eta,
                )
                phase -= time_shift * (Mf - Mf_ref) + phase_shift
                # remember multiple amp0 and shift phase and 1/eta
                if ti.static(self.returned_form == "amplitude_phase"):
                    self.waveform_container[idx].amplitude = amplitude
                    self.waveform_container[idx].phase = phase
                if ti.static(self.returned_form == "polarizations"):
                    (
                        self.waveform_container[idx].hcross,
                        self.waveform_container[idx].hplus,
                    ) = _get_polarization_from_amplitude_phase(
                        amplitude, phase, self.source_parameters[None].iota
                    )
                if ti.static(self.include_tf):
                    tf = _compute_tf(
                        powers_of_Mf,
                        self.phase_coefficients[None],
                        self.pn_prefactors[None],
                        self.source_parameters[None].f_ring,
                        self.source_parameters[None].f_damp,
                        self.source_parameters[None].eta,
                    )
                    tf -= time_shift
                    tf *= self.source_parameters[None].M_sec / PI / 2
                    self.waveform_container[idx].tf = tf
            else:
                if ti.static(self.returned_form == "amplitude_phase"):
                    self.waveform_container[idx].amplitude = 0.0
                    self.waveform_container[idx].phase = 0.0
                if ti.static(self.returned_form == "polarization"):
                    self.waveform_container[idx].hcross.fill(0.0)
                    self.waveform_container[idx].hplus.fill(0.0)
                if ti.static(self.include_tf):
                    self.waveform_container[idx].tf = 0.0

    @ti.func
    def _parameter_check(self):
        assert (
            self.source_parameters[None].mass_1 > self.source_parameters[None].mass_2
        ), f"require m1 > m2, you are passing m1: {self.source_parameters[None].mass_1}, m2:{self.source_parameters[None].mass_2}"
        assert (
            self.source_parameters[None].q > 0.0
            and self.source_parameters[None].q < 1.0
        ), f"require 0 < q < 1, you are passing q: {self.source_parameters[None].q}"
        assert (
            self.source_parameters[None].chi_1 > -1.0
            and self.source_parameters[None].chi_1 < 1.0
        ), f"require -1 < chi_1 < 1, you are passing chi_1: {self.source_parameters[None].chi_1}"
        assert (
            self.source_parameters[None].chi_2 > -1.0
            and self.source_parameters[None].chi_2 < 1.0
        ), f"require -1 < chi_2 < 1, you are passing chi_2: {self.source_parameters[None].chi_2}"

        # TODO more parameter check

    def np_array_of_waveform_container(self):
        ret = {}
        if self.returned_form == "polarizations":
            hcross_array = (
                self.waveform_container.hcross.to_numpy()
                .view(dtype=np.complex128)
                .reshape((self.frequencies.shape))
            )
            hplus_array = (
                self.waveform_container.hplus.to_numpy()
                .view(dtype=np.complex128)
                .reshape((self.frequencies.shape))
            )
            ret["hcross"] = hcross_array
            ret["hplus"] = hplus_array
        elif self.returned_form == "amplitude_phase":
            amp_array = self.waveform_container.amplitude.to_numpy()
            phase_array = self.waveform_container.phase.to_numpy()
            ret["amplitude"] = amp_array
            ret["phase"] = phase_array
        if self.include_tf:
            tf_array = self.waveform_container.tf.to_numpy()
            ret["tf"] = tf_array
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
