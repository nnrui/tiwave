import os
import warnings
from typing import Optional

import taichi as ti
import taichi.math as tm
import numpy as np

from ..constants import *
from ..utils import ComplexNumber, gauss_elimination, UsefulPowers
from .base_waveform import BaseWaveform


# Frequently used constants
sqrt2 = tm.sqrt(2)
useful_powers_pi = UsefulPowers()
useful_powers_pi.update(PI)
# Prepare an instance of UsefulPowers for later use
useful_powers_f1_int = UsefulPowers()
useful_powers_f3_int = UsefulPowers()


# Amplitude ansatz
@ti.func
def _amplitude_inspiral_ansatz(powers_of_Mf:ti.template(), amplitude_coefficients:ti.template(), pn_prefactors:ti.template()):
    """
    Eq. 6.3 in arXiv:2001.11412.
    Without amp0.
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
def _d_amplitude_inspiral_ansatz(powers_of_Mf:ti.template(), amplitude_coefficients:ti.template(), pn_prefactors:ti.template()):
    """
    Without amp0.
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
def _amplitude_intermediate_ansatz(powers_of_Mf:ti.template(), amplitude_coefficients:ti.template()):
    """
    Eq. 6.7 in arXiv:2001.11412.
    Only the recommended fitting model `104` with 4th order polynomial ansatz are implemented.
    Without amp0.
    """
    return 1.0 / (
        amplitude_coefficients.alpha_0
        + amplitude_coefficients.alpha_1 * powers_of_Mf.one
        + amplitude_coefficients.alpha_2 * powers_of_Mf.two
        + amplitude_coefficients.alpha_3 * powers_of_Mf.three
        + amplitude_coefficients.alpha_4 * powers_of_Mf.four
    )


@ti.func
def _amplitude_merge_ringdown_ansatz(powers_of_Mf:ti.template(), amplitude_coefficients:ti.template(), source_params:ti,template()):
    """
    Eq. 6.19 in arXiv:2001.11412.
    Different notation with Eq. 6.19: gamma_1: a_R, gamma_2: lambda, gamma_3: sigma
    gamma1_gamma3_fdamp: a_R * f_damp * sigma
    gamma2_over_gamma3_fdamp: lambda / (f_damp * sigma)
    gamma3_fdamp: f_damp * sigma
    gamma3_fdamp_pow2: (f_damp * sigma)^2
    """
    f_minus_fring = powers_of_Mf.one - source_params.f_ring
    return (
        amplitude_coefficients.gamma1_gamma3_fdamp
        / (f_minus_fring * f_minus_fring + amplitude_coefficients.gamma3_fdamp_pow2)
        * tm.exp(-f_minus_fring * amplitude_coefficients.gamma2_over_gamma3_fdamp)
    )


@ti.func
def _d_amplitude_merge_ringdown_ansatz(powers_of_Mf:ti.template(), amplitude_coefficients:ti.template(), source_params:ti.template()):
    """
    Derivative with respect to f of the amplitude merge-ringdown ansatz.
    """
    # f - f_ring
    f_minus_fring = powers_of_Mf.one - source_params.f_ring
    # (f - f_ring)^2
    f_minus_fring_pow2 = f_minus_fring * f_minus_fring
    # (f - f_ring)^2 + (gamma_3 * f_damp)^2
    common_term = f_minus_fring_pow2 + amplitude_coefficients.gamma3_fdamp_pow2
    return (
        -amplitude_coefficients.gamma_1
        * tm.exp(-f_minus_fring * amplitude_coefficients.gamma2_over_gamma3_fdamp)
        * (
            amplitude_coefficients.gamma_2 * common_term
            + 2.0 * amplitude_coefficients.gamma3_fdamp * f_minus_fring
        )
        / common_term**2
    )


# Phase ansatz
@ti.func
def _phase_inspiral_ansatz(powers_of_Mf:ti.template(), phase_coefficients:ti.template(), pn_prefactors:ti.template()):
    """
    Eq. 7.1 in arXiv:2001.11412.
    Only the recommended fitting model `104` are implemented.
    Without :math:`1/\eta`
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
        phase_coefficients.sigma_0
        + phase_coefficients.sigma_1 * powers_of_Mf.one
        + 0.75 * phase_coefficients.sigma_2 * powers_of_Mf.four_thirds
        + 0.6 * phase_coefficients.sigma_3 * powers_of_Mf.five_thirds
        + 0.5 * phase_coefficients.sigma_4 * powers_of_Mf.two
    )


@ti.func
def _d_phase_inspiral_ansatz(powers_of_Mf:ti.template(), phase_coefficients:ti.template(), pn_prefactors:ti.template()):
    """
    Without :math:`1/\eta`
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
def _phase_intermediate_ansatz(powers_of_Mf:ti.template(), phase_coefficients:ti.template()):
    """
    without 1/eta
    """
    return (
        phase_coefficients.beta_1 * powers_of_Mf.one
        + phase_coefficients.beta_2 * powers_of_Mf.log
        - phase_coefficients.beta_3 / 3.0 / powers_of_Mf.three
    )


@ti.func
def _d_phase_intermediate_ansatz(powers_of_Mf:ti.template(), phase_coefficients:ti.template()):
    """
    without 1/eta
    """
    return (
        phase_coefficients.beta_1
        + phase_coefficients.beta_2 / powers_of_Mf.one
        + phase_coefficients.beta_3 / powers_of_Mf.four
    )


@ti.func
def _d_phase_merge_ringdown_ansatz(
    powers_of_Mf: ti.template(),
    phase_coefficients: ti.template(),
    source_params: ti.template(),
):
    """
    without 1/eta
    Eq. 7.12 of arXiv:2001.11412.
    """

    return (
        phase_coefficients.c_0
        + phase_coefficients.c_1 / powers_of_Mf.one_third
        + phase_coefficients.c_2 / powers_of_Mf.two
        + phase_coefficients.c_4 / powers_of_Mf.four
        + phase_coefficients.c_L
        / (source_params.f_damp_pow2 + (powers_of_Mf.one - source_params.f_ring) ** 2)
    )


@ti.func
def _phase_merge_ringdown_ansatz(
    powers_of_Mf: ti.template(),
    phase_coefficients: ti.template(),
    source_params: ti.template(),
):
    """
    without 1/eta
    """
    return (
        phase_coefficients.c_0 * powers_of_Mf.one
        + 1.5 * phase_coefficients.c_1 * powers_of_Mf.two_third
        - phase_coefficients.c_2 / powers_of_Mf.one
        - phase_coefficients.c_4 / 3.0 / powers_of_Mf.three
        + phase_coefficients.c_L
        / source_params.f_damp
        * tm.atan2((powers_of_Mf.one - source_params.f_ring), source_params.f_damp)
    )


@ti.dataclass
class SourceParameters:
    # TODO: doc detail defination and unit!
    # passed in parameters
    M: ti.f64  # total mass (solar mass)
    q: ti.f64
    chi_1: ti.f64
    chi_2: ti.f64
    dL_Mpc: ti.f64  # luminosity distance (Mpc)
    iota: ti.f64
    phase_ref: ti.f64
    tc: ti.f64
    # derived parameters
    dL_SI: ti.f64
    m_1: ti.f64
    m_2: ti.f64
    M_sec: ti.f64  # total mass in second
    eta: ti.f64  # symmetric_mass_ratio
    delta: ti.f64
    chi_s: ti.f64
    chi_a: ti.f64
    chi_s2: ti.f64
    chi_a2: ti.f64
    chi_PN: ti.f64
    xi: ti.f64
    # cache frequently used parameters
    eta_pow2: ti.f64  # eta^2
    eta_pow3: ti.f64
    eta_pow4: ti.f64

    @ti.func
    def update_all_source_parameters(self, parameters: ti.template()):
        """
        Totally 9 parameters are needed: M, q, chi1, chi2, iota, tc, phi0, dL.
        """
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
        # derived parameters
        self.m_1 = self.M / (1 + self.q)
        self.m_2 = self.M - self.m_1
        self.dL_SI = self.dL_Mpc * 1e6 * PC_SI
        self.M_sec = self.M * MTSUN_SI
        self.eta = self.m_1 * self.m_2 / (self.M * self.M)
        self.eta_pow2 = self.eta * self.eta
        self.eta_pow3 = self.eta * self.eta_pow2
        self.eta_pow4 = self.eta + self.eta_pow3

        self.delta_chi = self.chi_1 - self.chi_2
        self.delta_chi_pow2 = self.delta_chi * self.delta_chi

        self.delta = tm.sqrt(1.0 - 4.0 * self.eta)
        self.chi_PN = (
            self.m_1 * self.chi_1 + self.m_2 * self.chi_2
        ) / self.M - 38.0 / 113.0 * self.eta * (self.chi_1 + self.chi_2)
        self.chi_PN_hat = (
            (self.m_1 * self.chi_1 + self.m_2 * self.chi_2) / self.M
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

        self.S_tot_hat = (
            self.m_1 * self.m_1 * self.chi_1 + self.m_2 * self.m_2 * self.chi_2
        ) / (self.M * self.M)
        self.S_tot_hat_pow2 = self.S_tot_hat * self.S_tot_hat
        self.S_tot_hat_pow3 = self.S_tot_hat * self.S_tot_hat_pow2
        self.S_tot_hat_pow4 = self.S_tot_hat * self.S_tot_hat_pow3

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
            + 20.0830030082033 * self.eta_pow2
            - 12.333573402277912 * self.eta_pow3
        ) / (1 + 7.2388440419467335 * self.eta)
        eq_spin = (self.m1_pow2 + self.m2_pow2) * self.S_tot + (
            (
                -0.8561951310209386 * self.eta
                - 0.09939065676370885 * self.eta_pow2
                + 1.668810429851045 * self.eta_pow3
            )
            * self.S_tot
            + (
                0.5881660363307388 * self.eta
                - 2.149269067519131 * self.eta_pow2
                + 3.4768263932898678 * self.eta_pow3
            )
            * self.S_tot_pow2
            + (
                0.142443244743048 * self.eta
                - 0.9598353840147513 * self.eta_pow2
                + 1.9595643107593743 * self.eta_pow3
            )
            * self.S_tot_pow3
        ) / (
            1
            + (
                -0.9142232693081653
                + 2.3191363426522633 * self.eta
                - 9.710576749140989 * self.eta_pow3
            )
            * self.S_tot
        )
        uneq_spin = (
            0.3223660562764661
            * self.delta_chi
            * self.delta
            * (1 + 9.332575956437443 * self.eta)
            * self.eta_pow2
            - 0.059808322561702126 * self.delta_chi_pow2 * self.eta_pow3
            + 2.3170397514509933
            * self.delta_chi
            * self.delta
            * (1 - 3.2624649875884852 * self.eta)
            * self.eta_pow3
            * self.delta_chi
        )
        return no_spin + eq_spin + uneq_spin

    @ti.func
    def _f_MECO(self):
        # Frequency of the minimum energy circular orbit (MECO).
        no_spin = (
            0.018744340279608845
            + 0.0077903147004616865 * self.eta
            + 0.003940354686136861 * self.eta_pow2
            - 0.00006693930988501673 * self.eta_pow3
        ) / (1.0 - 0.10423384680638834 * self.eta)
        eq_spin = (
            self.chi_PN_hat
            * (
                0.00027180386951683135
                - 0.00002585252361022052 * self.chi_PN_hat
                + self.eta_pow4
                * (
                    -0.0006807631931297156
                    + 0.022386313074011715 * self.chi_PN_hat
                    - 0.0230825153005985 * self.chi_PN_hat_pow2
                )
                + self.eta_pow2
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
                + self.eta_pow3
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
                + 1.4861197211952053 * self.eta_pow4
                + 0.025066696514373803 * self.eta_pow2
                + 0.005146809717492324 * self.eta_pow3
            )
            * self.chi_PN_hat
            + (
                -0.0058684526275074025
                - 0.02876774751921441 * self.eta
                - 2.551566872093786 * self.eta_pow4
                - 0.019641378027236502 * self.eta_pow2
                - 0.001956646166089053 * self.eta_pow3
            )
            * self.chi_PN_hat_pow2
            + (
                0.003507640638496499
                + 0.014176504653145768 * self.eta
                + 1.0 * self.eta_pow4
                + 0.012622225233586283 * self.eta_pow2
                - 0.00767768214056772 * self.eta_pow3
            )
            * self.chi_PN_hat_pow3
        )
        uneq_spin = self.delta_chi_pow2 * (
            0.00034375176678815234 + 0.000016343732281057392 * self.eta
        ) * self.eta_pow2 + self.delta_chi * self.delta * self.eta * (
            0.08064665214195679 * self.eta_pow2
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
    ins_colloc_points: ti.types.vector(3, ti.f64)
    ins_colloc_values: ti.types.vector(3, ti.f64)
    # Intermediate (104 model)
    alpha_0: ti.f64
    alpha_1: ti.f64
    alpha_2: ti.f64
    alpha_3: ti.f64
    alpha_4: ti.f64
    int_colloc_points: ti.types.vector(3, ti.f64)
    int_colloc_values: ti.types.vector(5, ti.f64)
    # Merge-ringdown
    gamma_1: ti.f64  # a_R in arXiv:2001.11412
    gamma_2: ti.f64  # lambda in arXiv:2001.11412
    gamma_3: ti.f64  # sigma in arXiv:2001.11412
    f_peak: ti.f64
    # cached parameters
    gamma1_gamma3_fdamp: ti.f64  # a_R * f_damp * sigma
    gamma2_over_gamma3_fdamp: ti.f64  # lambda / (f_damp * sigma)
    gamma3_fdamp: ti.f64  # f_damp * sigma
    gamma3_fdamp_pow2: ti.f64  # (f_damp * sigma)^2
    amp_0: ti.f64

    @ti.func
    def _ins_int_colloc_points(self, source_params: ti.template()):
        """
        Computing collocation points in insprial and intermediate range.
        Only can be called after updating merge-ringdown coefficient, since the f_peak
        is needed for intermediate collocation points.
        """
        # Insprial, Eq.6.5.
        fmax_ins = source_params.f_MECO + 0.25 * (
            source_params.f_ISCO - source_params.f_MECO
        )
        # Note the equation of Eq. 6.5 uses f_MECO, while fAT (Eq. 5.7) is used in
        # lalsimumation (l. 781 in LALSimIMRPhenomX_internals.c).
        self.ins_colloc_points[0] = fmax_ins * 0.5
        self.ins_colloc_points[1] = fmax_ins * 0.75
        self.ins_colloc_points[2] = fmax_ins

        # Intermediate, Tab. II
        self.int_colloc_points[0] = fmax_ins
        self.int_colloc_points[1] = (fmax_ins + self.f_peak) * 0.5
        self.int_colloc_points[2] = self.f_peak

    @ti.func
    def _inspiral_coefficients(self, source_params: ti.template()):
        # Value for amplitude collocation point at 0.5 f^A_T,
        self.ins_colloc_values[0] = (
            (
                -0.015178276424448592
                - 0.06098548699809163 * source_params.eta
                + 0.4845148547154606 * source_params.eta_pow2
            )
            / (1.0 + 0.09799277215675059 * source_params.eta)
            + (
                (0.02300153747158323 + 0.10495263104245876 * source_params.eta_pow2)
                * source_params.chi_PN_hat
                + (0.04834642258922544 - 0.14189350657140673 * source_params.eta)
                * source_params.eta
                * source_params.chi_PN_hat_pow3
                + (0.01761591799745109 - 0.14404522791467844 * source_params.eta_pow2)
                * source_params.chi_PN_hat_pow2
            )
            / (1.0 - 0.7340448493183307 * source_params.chi_PN_hat)
            + source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow4
            * (0.0018724905795891192 + 34.90874132485147 * source_params.eta)
        )
        # Value for amplitude collocation point at 0.75 f^A_T,
        self.ins_colloc_values[1] = (
            (
                -0.058572000924124644
                - 1.1970535595488723 * source_params.eta
                + 8.4630293045015 * source_params.eta_pow2
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
                    + source_params.eta_pow2
                    * (
                        0.788717372588848
                        + 0.8282888482429105 * source_params.chi_PN_hat
                    )
                    - 0.018924013869130434 * source_params.chi_PN_hat
                )
                * source_params.chi_PN_hat
            )
            / (-1.332123330797879 + 1.0 * source_params.chi_PN_hat)
            + source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow4
            * (0.004389995099201855 + 105.84553997647659 * source_params.eta)
        )
        # Value for amplitude collocation point at 1.0 f^A_T,
        self.ins_colloc_values[2] = (
            (
                -0.16212854591357853
                + 1.617404703616985 * source_params.eta
                - 3.186012733446088 * source_params.eta_pow2
                + 5.629598195000046 * source_params.eta_pow3
            )
            / (1.0 + 0.04507019231274476 * source_params.eta)
            + (
                source_params.chi_PN_hat
                * (
                    1.0055835408962206
                    + source_params.eta_pow2
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
                    + source_params.eta_pow3
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
            * source_params.eta_pow4
            * (0.05575955418803233 + 208.92352600701068 * source_params.eta)
        )

        Ab_ins = ti.Matrix(
            [
                [
                    self.ins_colloc_points[0] ** (7 / 3),
                    self.ins_colloc_points[0] ** (8 / 3),
                    self.ins_colloc_points[0] ** 3,
                    self.ins_colloc_values[0],
                ],
                [
                    self.ins_colloc_points[1] ** (7 / 3),
                    self.ins_colloc_points[1] ** (8 / 3),
                    self.ins_colloc_points[1] ** 3,
                    self.ins_colloc_values[1],
                ],
                [
                    self.ins_colloc_points[2] ** (7 / 3),
                    self.ins_colloc_points[2] ** (8 / 3),
                    self.ins_colloc_points[2] ** 3,
                    self.ins_colloc_values[2],
                ],
            ],
            dt=ti.f64,
        )
        self.rho_1, self.rho_2, self.rho_3 = gauss_elimination(Ab_ins)

    @ti.func
    def _intermediate_coefficients(
        self, source_params: ti.template(), pn_prefactors: ti.template()
    ):
        """
        Only the recommended fit model `104` is implemented, and only can be called after
        updated the inspiral and merge-ringdown coefficients. 

        Different with the implementaion in lalsimulation, here we fully simply :math:`A_0` as:
        .. math::
            \begin{aligned}
            \frac{A_0(f_1)}{A_{\mathrm{int}}^{\mathrm{inv}}(f_1)} &= A_0(f_1)A_{\mathrm{ins}}(f_1),\\
            A_{\mathrm{int}}^{\mathrm{inv}}(f_1) &= 1/A_{\mathrm{ins}}(f_1),
            \end{aligned}      
        and 
        .. math::
            \begin{aligned}
            \left[\frac{A_0(f_1)}{A_{\mathrm{int}}^{\mathrm{inv}}(f_1)}\right]' &= \left[A_0(f_1)A_{\mathrm{ins}}(f_1)\right]', \\
            \left[A_{\mathrm{int}}^{\mathrm{inv}}(f_1)\right]' &= \frac{A'_{\mathrm{ins}}(f_1)}{A_{\mathrm{ins}}^{2}(f_1)}
            \end{aligned}        
        The case of point at :math:`f_3` is similar. Thus there will be a different of 
        factor :math:`f^{7/6}` in amplitude_intermediate_ansatz function.
        """
        useful_powers_f1_int.update(self.int_colloc_points[0])
        useful_powers_f3_int.update(self.int_colloc_points[2])

        self.int_colloc_values[0] = 1.0 / _amplitude_inspiral_ansatz(
            useful_powers_f1_int, self, pn_prefactors
        )
        self.int_colloc_values[1] = self.int_colloc_points[1] ** (-7 / 6) / (
            (
                1.4873184918202145
                + 1974.6112656679577 * source_params.eta
                + 27563.641024162127 * source_params.eta_pow2
                - 19837.908020966777 * source_params.eta_pow3
            )
            / (
                1.0
                + 143.29004876335128 * source_params.eta
                + 458.4097306093354 * source_params.eta_pow2
            )
            + source_params.S_tot_hat
            * (
                27.952730865904343
                + source_params.eta
                * (-365.55631765202895 - 260.3494489873286 * source_params.S_tot_hat)
                + 3.2646808851249016 * source_params.S_tot_hat
                + 3011.446602208493 * source_params.eta_pow2 * source_params.S_tot_hat
                - 19.38970173389662 * source_params.S_tot_hat_pow2
                + source_params.eta_pow3
                * (
                    1612.2681322644232
                    - 6962.675551371755 * source_params.S_tot_hat
                    + 1486.4658089990298 * source_params.S_tot_hat_pow2
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
            * source_params.eta_pow2
        )
        self.int_colloc_values[2] = 1.0 / _amplitude_merge_ringdown_ansatz(
            useful_powers_f3_int, self, source_params.f_ring
        )
        self.int_colloc_values[3] = (
            _d_amplitude_inspiral_ansatz(useful_powers_f1_int, self, pn_prefactors)
            / self.int_colloc_values[0] ** 2
        )
        self.int_colloc_values[4] = (
            _d_amplitude_merge_ringdown_ansatz(
                useful_powers_f3_int, self, source_params.f_ring
            )
            / self.int_colloc_values[2] ** 2
        )

        Ab_int = ti.Matrix(
            [
                [
                    1.0,
                    self.ins_colloc_points[0],
                    self.ins_colloc_points[0] ** 2,
                    self.ins_colloc_points[0] ** 3,
                    self.ins_colloc_points[0] ** 4,
                    self.ins_colloc_values[0],
                ],
                [
                    1.0,
                    self.ins_colloc_points[1],
                    self.ins_colloc_points[1] ** 2,
                    self.ins_colloc_points[1] ** 3,
                    self.ins_colloc_points[1] ** 4,
                    self.ins_colloc_values[1],
                ],
                [
                    1.0,
                    self.ins_colloc_points[2],
                    self.ins_colloc_points[2] ** 2,
                    self.ins_colloc_points[2] ** 3,
                    self.ins_colloc_points[2] ** 4,
                    self.ins_colloc_values[2],
                ],
                [
                    0.0,
                    1.0,
                    2.0 * self.ins_colloc_points[0],
                    3.0 * self.ins_colloc_points[0] ** 2,
                    4.0 * self.ins_colloc_points[0] ** 3,
                    self.ins_colloc_values[3],
                ],
                [
                    0.0,
                    1.0,
                    2.0 * self.ins_colloc_points[2],
                    3.0 * self.ins_colloc_points[2] ** 2,
                    4.0 * self.ins_colloc_points[2] ** 3,
                    self.ins_colloc_values[4],
                ],
            ]
        )
        (
            self.alpha_0,
            self.alpha_1,
            self.alpha_2,
            self.alpha_3,
            self.alpha_4,
        ) = gauss_elimination(Ab_int)

    @ti.func
    def _merge_ringdown_coefficients(self, source_params):
        """
        Computing merge-ringdown coefficients. Using different notation in arXiv:2001.11412,
        a_R (gamma_1), lambda (gamma_2), sigma (gamma_3)
        """
        self.gamma_2 = (
            (
                0.8312293675316895
                + 7.480371544268765 * source_params.eta
                - 18.256121237800397 * source_params.eta_pow2
            )
            / (
                1.0
                + 10.915453595496611 * source_params.eta
                - 30.578409433912874 * source_params.eta_pow2
            )
            + source_params.S_tot_hat
            * (
                0.5869408584532747
                + source_params.eta
                * (-0.1467158405070222 - 2.8489481072076472 * source_params.S_tot_hat)
                + 0.031852563636196894 * source_params.S_tot_hat
                + source_params.eta_pow2
                * (0.25295441250444334 + 4.6849496672664594 * source_params.S_tot_hat)
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
            + 2.109081209912545 * source_params.eta_pow2
            - 4.222259944408823 * source_params.eta_pow3
        ) / (1.0 - 2.7440263888207594 * source_params.eta) + (
            0.07179105336478316
            + source_params.eta_pow2
            * (2.331724812782498 - 0.6330998412809531 * source_params.S_tot_hat)
            + source_params.eta
            * (-0.8752427297525086 + 0.4168560229353532 * source_params.S_tot_hat)
            - 0.05633734476062242 * source_params.S_tot_hat
        ) * source_params.S_tot_hat
        # cache frequently used parameters
        # lambda / (f_damp * sigma)
        self.gamma2_over_gamma3_fdamp = self.gamma_2 / (
            self.gamma_3 * source_params.f_damp
        )
        # f_damp * sigma
        self.gamma3_fdamp = self.gamma_3 * source_params.f_damp
        # (f_damp * sigma)^2
        self.gamma3_fdamp_pow2 = self.gamma3_fdamp * self.gamma3_fdamp

        if self.gamma_2 > 1.0:
            self.f_peak = ti.abs(
                source_params.f_ring
                - source_params.f_damp * self.gamma_3 / self.gamma_2
            )
        else:
            self.f_peak = ti.abs(
                source_params.f_ring
                + (tm.sqrt(1 - self.gamma_2 * self.gamma_2) - 1)
                * source_params.f_damp
                * self.gamma_3
                / self.gamma_2
            )
        value_peak = (
            (
                0.03689164742964719
                + 25.417967754401182 * source_params.eta
                + 162.52904393600332 * source_params.eta_pow2
            )
            / (
                1.0
                + 61.19874463331437 * source_params.eta
                - 29.628854485544874 * source_params.eta_pow2
            )
            + source_params.S_tot_hat
            * (
                -0.14352506969368556
                + 0.026356911108320547 * source_params.S_tot_hat
                + 0.19967405175523437 * source_params.S_tot_hat_pow2
                - 0.05292913111731128 * source_params.S_tot_hat_pow3
                + source_params.eta_pow3
                * (
                    -48.31945248941757
                    - 3.751501972663298 * source_params.S_tot_hat
                    + 81.9290740950083 * source_params.S_tot_hat_pow2
                    + 30.491948143930266 * source_params.S_tot_hat_pow3
                    - 132.77982622925845 * source_params.S_tot_hat_pow4
                )
                + source_params.eta
                * (
                    -4.805034453745424
                    + 1.11147906765112 * source_params.S_tot_hat
                    + 6.176053843938542 * source_params.S_tot_hat_pow2
                    - 0.2874540719094058 * source_params.S_tot_hat_pow3
                    - 8.990840289951514 * source_params.S_tot_hat_pow4
                )
                - 0.18147275151697131 * source_params.S_tot_hat_pow4
                + source_params.eta_pow2
                * (
                    27.675454081988036
                    - 2.398327419614959 * source_params.S_tot_hat
                    - 47.99096500250743 * source_params.S_tot_hat_pow2
                    - 5.104257870393138 * source_params.S_tot_hat_pow3
                    + 72.08174136362386 * source_params.S_tot_hat_pow4
                )
            )
            / (-1.4160870461211452 + 1.0 * source_params.S_tot_hat)
            - 0.04426571511345366
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
        )
        fpeak_minus_fring = self.f_peak - source_params.f_ring
        self.gamma_1 = (
            value_peak
            / self.gamma3_fdamp
            * (fpeak_minus_fring**2 + self.gamma3_fdamp_pow2)
            * tm.exp(fpeak_minus_fring * self.gamma2_over_gamma3_fdamp)
        )
        # cache frequently used parameters
        # a_R * f_damp * sigma
        self.gamma1_gamma3_fdamp = self.gamma_1 * self.gamma_3 * source_params.f_damp

    @ti.func
    def compute_amplitude_coefficients(self, source_params, pn_prefactors):
        self._merge_ringdown_coefficients(source_params)
        self._ins_int_colloc_points(source_params)
        self._inspiral_coefficients(source_params)
        self._intermediate_coefficients(source_params, pn_prefactors)

        # The common prefactor, A0 (without f^{-7/6})
        self.amp0 = (
            0.25
            * tm.sqrt(10.0 / 3.0 * source_params.eta / useful_powers_pi.four_thirds)
            * source_params.M**2
            / source_params.dL_SI
            * MRSUN_SI
            * MTSUN_SI
        )


@ti.dataclass
class PhaseCoefficients:
    # Inspiral (104 fitting model)
    sigma_0: ti.f64
    sigma_1: ti.f64
    sigma_2: ti.f64
    sigma_3: ti.f64
    sigma_4: ti.f64
    # Intermediate
    beta_0: ti.f64
    beta_1: ti.f64
    beta_2: ti.f64
    beta_3: ti.f64
    beta_4: ti.f64
    # Merge_ringdown
    c_0: ti.f64
    c_1: ti.f64  # f^-1/3
    c_2: ti.f64  # f^-2
    c_4: ti.f64  # f_-4
    c_L: ti.f64  # Lorentzian term
    MRD_colloc_points: ti.types.vector(4, ti.f64)
    MRD_colloc_values: ti.types.vector(4, ti.f64)

    @ti.func
    def _all_colloc_points(self, source_params: ti.template()):
        # Merge-ringdown
        fmin_MRD = 0.3 * source_params.f_ring + 0.6 * source_params.f_ISCO
        fmax_MRD = source_params.f_ring + 1.25 * source_params.f_damp
        frange_MRD = fmax_MRD - fmin_MRD
        self.MRD_colloc_points[0] = fmin_MRD
        self.MRD_colloc_points[1] = fmin_MRD + 0.5 * (1 - 1 / sqrt2) * frange_MRD
        self.MRD_colloc_points[2] = fmin_MRD + 0.5 * frange_MRD
        self.MRD_colloc_points[3] = source_params.f_ring
        self.MRD_colloc_points[4] = fmax_MRD

        # Inspiral
        # Intermediate

    @ti.func
    def _merge_ringdown_coefficients(self, source_params: ti.template()):
        # Difference between collocation points 1 and 2 (v1 - v2)
        d12_MRD = (
            source_params.eta
            * (
                0.7207992174994245
                - 1.237332073800276 * source_params.eta
                + 6.086871214811216 * source_params.eta_pow2
            )
            / (
                0.006851189888541745
                + 0.06099184229137391 * source_params.eta
                - 0.15500218299268662 * source_params.eta_pow2
                + 1.0 * source_params.eta_pow3
            )
            + (
                (
                    0.06519048552628343
                    - 25.25397971063995 * source_params.eta
                    - 308.62513664956975 * source_params.eta_pow4
                    + 58.59408241189781 * source_params.eta_pow2
                    + 160.14971486043524 * source_params.eta_pow3
                )
                * source_params.S_tot_hat
                + source_params.eta
                * (
                    -5.215945111216946
                    + 153.95945758807616 * source_params.eta
                    - 693.0504179144295 * source_params.eta_pow2
                    + 835.1725103648205 * source_params.eta_pow3
                )
                * source_params.S_tot_hat_pow2
                + (
                    0.20035146870472367
                    - 0.28745205203100666 * source_params.eta
                    - 47.56042058800358 * source_params.eta_pow4
                )
                * source_params.S_tot_hat_pow3
                + source_params.eta
                * (
                    5.7756520242745735
                    - 43.97332874253772 * source_params.eta
                    + 338.7263666984089 * source_params.eta_pow3
                )
                * source_params.S_tot_hat_pow4
                + (
                    -0.2697933899920511
                    + 4.917070939324979 * source_params.eta
                    - 22.384949087140086 * source_params.eta_pow4
                    - 11.61488280763592 * source_params.eta_pow2
                )
                * source_params.S_tot_hat_pow5
            )
            / (1.0 - 0.6628745847248266 * source_params.S_tot_hat)
            - 23.504907495268824
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
        )
        # Difference between collocation points 2 and 4 (v2 - v4)
        d24_MRD = (
            (
                source_params.eta
                * (
                    -9.460253118496386
                    + 9.429314399633007 * source_params.eta
                    + 64.69109972468395 * source_params.eta_pow2
                )
            )
            / (
                -0.0670554310666559
                - 0.09987544893382533 * source_params.eta
                + 1.0 * source_params.eta_pow2
            )
            + (
                17.36495157980372 * source_params.eta * source_params.S_tot_hat
                + source_params.eta_pow3
                * source_params.S_tot_hat
                * (930.3458437154668 + 808.457330742532 * source_params.S_tot_hat)
                + source_params.eta_pow4
                * source_params.S_tot_hat
                * (
                    -774.3633787391745
                    - 2177.554979351284 * source_params.S_tot_hat
                    - 1031.846477275069 * source_params.S_tot_hat_pow2
                )
                + source_params.eta_pow2
                * source_params.S_tot_hat
                * (
                    -191.00932194869588
                    - 62.997389062600035 * source_params.S_tot_hat
                    + 64.42947340363101 * source_params.S_tot_hat_pow2
                )
                + 0.04497628581617564 * source_params.S_tot_hat_pow3
            )
            / (1.0 - 0.7267610313751913 * source_params.S_tot_hat)
            + source_params.delta_chi
            * source_params.delta
            * (-36.66374091965371 + 91.60477826830407 * source_params.eta)
            * source_params.eta_pow2
        )
        # Difference between collocation points 3 and 4 (v3 - v4)
        d34_MRD = (
            (
                source_params.eta
                * (-8.506898502692536 + 13.936621412517798 * source_params.eta)
            )
            / (-0.40919671232073945 + 1.0 * source_params.eta)
            + (
                source_params.eta
                * (
                    1.7280582989361533 * source_params.S_tot_hat
                    + 18.41570325463385 * source_params.S_tot_hat_pow3
                    - 13.743271480938104 * source_params.S_tot_hat_pow4
                )
                + source_params.eta_pow2
                * (
                    73.8367329022058 * source_params.S_tot_hat
                    - 95.57802408341716 * source_params.S_tot_hat_pow3
                    + 215.78111099820157 * source_params.S_tot_hat_pow4
                )
                + 0.046849371468156265 * source_params.S_tot_hat_pow2
                + source_params.eta_pow3
                * source_params.S_tot_hat
                * (
                    -27.976989112929353
                    + 6.404060932334562 * source_params.S_tot_hat
                    - 633.1966645925428 * source_params.S_tot_hat_pow3
                    + 109.04824706217418 * source_params.S_tot_hat_pow2
                )
            )
            / (1.0 - 0.6862449113932192 * source_params.S_tot_hat)
            + 641.8965762829259
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow5
        )
        # Merge-ringdown phase at collocation point f4 (f_ring)
        v4_MRD = (
            (
                -85.86062966719405
                - 4616.740713893726 * source_params.eta
                - 4925.756920247186 * source_params.eta_pow2
                + 7732.064464348168 * source_params.eta_pow3
                + 12828.269960300782 * source_params.eta_pow4
                - 39783.51698102803 * source_params.eta_pow5
            )
            / (1.0 + 50.206318806624004 * source_params.eta)
            + (
                source_params.S_tot_hat
                * (
                    33.335857451144356
                    - 36.49019206094966 * source_params.S_tot_hat
                    + source_params.eta_pow3
                    * (
                        1497.3545918387515
                        - 101.72731770500685 * source_params.S_tot_hat
                    )
                    * source_params.S_tot_hat
                    - 3.835967351280833 * source_params.S_tot_hat_pow2
                    + 2.302712009652155 * source_params.S_tot_hat_pow3
                    + source_params.eta_pow2
                    * (
                        93.64156367505917
                        - 18.184492163348665 * source_params.S_tot_hat
                        + 423.48863373726243 * source_params.S_tot_hat_pow2
                        - 104.36120236420928 * source_params.S_tot_hat_pow3
                        - 719.8775484010988 * source_params.S_tot_hat_pow4
                    )
                    + 1.6533417657003922 * source_params.S_tot_hat_pow4
                    + source_params.eta
                    * (
                        -69.19412903018717
                        + 26.580344399838758 * source_params.S_tot_hat
                        - 15.399770764623746 * source_params.S_tot_hat_pow2
                        + 31.231253209893488 * source_params.S_tot_hat_pow3
                        + 97.69027029734173 * source_params.S_tot_hat_pow4
                    )
                    + source_params.eta_pow4
                    * (
                        1075.8686153198323
                        - 3443.0233614187396 * source_params.S_tot_hat
                        - 4253.974688619423 * source_params.S_tot_hat_pow2
                        - 608.2901586790335 * source_params.S_tot_hat_pow3
                        + 5064.173605639933 * source_params.S_tot_hat_pow4
                    )
                )
            )
            / (-1.3705601055555852 + 1.0 * source_params.S_tot_hat)
            + source_params.delta_chi
            * source_params.delta
            * source_params.eta
            * (22.363215261437862 + 156.08206945239374 * source_params.eta)
        )
        # Difference between collocation points 5 and 4 (v5 - v4)
        d54_MRD = (
            (
                source_params.eta
                * (
                    7.05731400277692
                    + 22.455288821807095 * source_params.eta
                    + 119.43820622871043 * source_params.eta_pow2
                )
            )
            / (0.26026709603623255 + 1.0 * source_params.eta)
            + (
                source_params.eta_pow2
                * (134.88158268621922 - 56.05992404859163 * source_params.S_tot_hat)
                * source_params.S_tot_hat
                + source_params.eta
                * source_params.S_tot_hat
                * (-7.9407123129681425 + 9.486783128047414 * source_params.S_tot_hat)
                + source_params.eta_pow3
                * source_params.S_tot_hat
                * (-316.26970506215554 + 90.31815139272628 * source_params.S_tot_hat)
            )
            / (1.0 - 0.7162058321905909 * source_params.S_tot_hat)
            + 43.82713604567481
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow3
        )
        self.MRD_colloc_values[4] = d54_MRD + v4_MRD
        self.MRD_colloc_values[3] = v4_MRD
        self.MRD_colloc_values[2] = d34_MRD + v4_MRD
        self.MRD_colloc_values[1] = d24_MRD + v4_MRD
        self.MRD_colloc_values[0] = d12_MRD + self.MRD_colloc_values[1]

        Ab_MRD = ti.Matrix(
            [
                [
                    1,
                    self.MRD_colloc_points[0] ** (-1 / 3),
                    self.MRD_colloc_points[0] ** (-2),
                    self.MRD_colloc_points[0] ** (-4),
                    1.0
                    / (
                        source_params.f_damp_pow2
                        + (self.MRD_colloc_points[0] - source_params.f_ring) ** 2
                    ),
                    self.MRD_colloc_values[0],
                ],
                [
                    1,
                    self.MRD_colloc_points[1] ** (-1 / 3),
                    self.MRD_colloc_points[1] ** (-2),
                    self.MRD_colloc_points[1] ** (-4),
                    1.0
                    / (
                        source_params.f_damp_pow2
                        + (self.MRD_colloc_points[1] - source_params.f_ring) ** 2
                    ),
                    self.MRD_colloc_values[1],
                ],
                [
                    1,
                    self.MRD_colloc_points[2] ** (-1 / 3),
                    self.MRD_colloc_points[2] ** (-2),
                    self.MRD_colloc_points[2] ** (-4),
                    1.0
                    / (
                        source_params.f_damp_pow2
                        + (self.MRD_colloc_points[2] - source_params.f_ring) ** 2
                    ),
                    self.MRD_colloc_values[2],
                ],
                [
                    1,
                    self.MRD_colloc_points[3] ** (-1 / 3),
                    self.MRD_colloc_points[3] ** (-2),
                    self.MRD_colloc_points[3] ** (-4),
                    1.0
                    / (
                        source_params.f_damp_pow2
                        + (self.MRD_colloc_points[3] - source_params.f_ring) ** 2
                    ),
                    self.MRD_colloc_values[3],
                ],
                [
                    1,
                    self.MRD_colloc_points[4] ** (-1 / 3),
                    self.MRD_colloc_points[4] ** (-2),
                    self.MRD_colloc_points[4] ** (-4),
                    1.0
                    / (
                        source_params.f_damp_pow2
                        + (self.MRD_colloc_points[4] - source_params.f_ring) ** 2
                    ),
                    self.MRD_colloc_values[4],
                ],
            ]
        )
        self.c_0, self.c_1, self.c_2, self.c_4, self.c_L = gauss_elimination(Ab_MRD)

    @ti.func
    def _inspiral_coefficients(self, source_params: ti.template()):
        """104 model"""

    @ti.func
    def compute_phase_coefficients(
        self, source_params, pn_prefactors, phase_ins_ver: ti.int
    ):
        # Inspiral coefficients:
        # ver 104: Canonical TaylorF2, with 4 pseudo-PN coefficients.
        # ver 105: Canonical TaylorF2, with 5 pseudo-PN coefficients.
        # ver 114: Extended TaylorF2, with 4 pseudo-PN coefficients.
        # ver 115: Extended TaylorF2, with 5 pseudo-PN coefficients.
        if ti.static(phase_ins_ver == 104):
            pass
        elif ti.static(phase_ins_ver == 105):
            pass
        elif ti.static(phase_ins_ver == 114):
            pass
        elif ti.static(phase_ins_ver == 115):
            pass
        # Intermediate coefficients: beta_0, beta_1, beta_2, beta_3, beta_4
        # ver 104: 4 coefficients with beta_3 = 0.
        # ver 105: 5 coefficients.
        if ti.static(phase_ins_ver == 104):
            pass
        elif ti.static(phase_ins_ver == 105):
            pass

        # Merge-ringdown coefficients
        dphase0 = 5.0 / (128.0 * useful_power_pi.five_third)
        fmin_mr = 0.3 * source_params.f_ring + 0.6 * source_params.f_ISCO
        fmax_mr = source_params.f_ring + 1.25 * source_params.f_damp
        df_mr = fmax_mr - fmin_mr
        f2_mr = fmin_mr + 0.5 * (1 - 1 / sqrt2) * df_mr
        f3_mr = fmin_mr + 0.5 * df_mr

        # Difference between collocation points 1 and 2 (v1 - v2)
        v1_mr = (
            (
                (
                    source_params.eta
                    * (
                        0.7207992174994245
                        - 1.237332073800276 * source_params.eta
                        + 6.086871214811216 * source_params.eta_pow2
                    )
                )
                / (
                    0.006851189888541745
                    + 0.06099184229137391 * source_params.eta
                    - 0.15500218299268662 * source_params.eta_pow2
                    + 1.0 * source_params.eta_pow2 * source_params.eta
                )
            )
            + (
                (
                    (
                        0.06519048552628343
                        - 25.25397971063995 * source_params.eta
                        - 308.62513664956975
                        * source_params.eta_pow2
                        * source_params.eta_pow2
                        + 58.59408241189781 * source_params.eta_pow2
                        + 160.14971486043524 * source_params.ta2 * source_params.eta
                    )
                    * source_params.S_tot_hat
                    + source_params.eta
                    * (
                        -5.215945111216946
                        + 153.95945758807616 * source_params.eta
                        - 693.0504179144295 * source_params.eta_pow2
                        + 835.1725103648205 * source_params.eta_pow2 * source_params.eta
                    )
                    * source_params.S_tot_hat_pow2
                    + (
                        0.20035146870472367
                        - 0.28745205203100666 * source_params.eta
                        - 47.56042058800358
                        * source_params.eta_pow2
                        * source_params.eta_pow2
                    )
                    * source_params.S_tot_hat_pow2
                    * source_params.S_tot_hat
                    + source_params.eta
                    * (
                        5.7756520242745735
                        - 43.97332874253772 * source_params.eta
                        + 338.7263666984089 * source_params.eta_pow2 * source_params.eta
                    )
                    * source_params.S_tot_hat_pow2
                    * source_params.S_tot_hat_pow2
                    + (
                        -0.2697933899920511
                        + 4.917070939324979 * source_params.eta
                        - 22.384949087140086
                        * source_params.eta_pow2
                        * source_params.eta_pow2
                        - 11.61488280763592 * source_params.eta_pow2
                    )
                    * source_params.S_tot_hat_pow2
                    * source_params.S_tot_hat_pow2
                    * source_params.S_tot_hat
                )
                / (1.0 - 0.6628745847248266 * source_params.S_tot_hat)
            )
            + (
                -23.504907495268824
                * source_params.delta_chi
                * source_params.delta
                * self.eta_pow2
            )
        )
        # Difference between collocation points 2 and 4 (v2 - v4)
        v2_mr = (
            (
                (
                    source_params.eta
                    * (
                        -9.460253118496386
                        + 9.429314399633007 * source_params.eta
                        + 64.69109972468395 * source_params.eta_pow2
                    )
                )
                / (
                    -0.0670554310666559
                    - 0.09987544893382533 * source_params.eta
                    + 1.0 * source_params.eta_pow2
                )
            )
            + (
                (
                    17.36495157980372 * source_params.eta * source_params.S_tot_hat
                    + source_params.eta_pow2
                    * source_params.eta
                    * source_params.S_tot_hat
                    * (930.3458437154668 + 808.457330742532 * source_params.S_tot_hat)
                    + source_params.eta_pow2
                    * source_params.eta_pow2
                    * source_params.S_tot_hat
                    * (
                        -774.3633787391745
                        - 2177.554979351284 * source_params.S_tot_hat
                        - 1031.846477275069 * source_params.S_tot_hat_pow2
                    )
                    + source_params.eta_pow2
                    * source_params.S_tot_hat
                    * (
                        -191.00932194869588
                        - 62.997389062600035 * source_params.S_tot_hat
                        + 64.42947340363101 * source_params.S_tot_hat_pow2
                    )
                    + 0.04497628581617564
                    * source_params.S_tot_hat_pow2
                    * source_params.S_tot_hat
                )
                / (1.0 - 0.7267610313751913 * source_params.S_tot_hat)
            )
            + (
                source_params.delta_chi
                * source_params.delta
                * (-36.66374091965371 + 91.60477826830407 * source_params.eta)
                * source_params.eta_pow2
            )
        )
        # Difference between collocation points 3 and 4 (v3 - v4)
        v3_mr = (
            (
                (
                    source_params.eta
                    * (-8.506898502692536 + 13.936621412517798 * source_params.eta)
                )
                / (-0.40919671232073945 + 1.0 * source_params.eta)
            )
            + (
                (
                    source_params.eta
                    * (
                        1.7280582989361533 * source_params.S_tot_hat
                        + 18.41570325463385
                        * source_params.S_tot_hat_pow2
                        * source_params.S_tot_hat
                        - 13.743271480938104
                        * source_params.S_tot_hat_pow2
                        * source_params.S_tot_hat_pow2
                    )
                    + source_params.eta_pow2
                    * (
                        73.8367329022058 * source_params.S_tot_hat
                        - 95.57802408341716
                        * source_params.S_tot_hat_pow2
                        * source_params.S_tot_hat
                        + 215.78111099820157
                        * source_params.S_tot_hat_pow2
                        * source_params.S_tot_hat_pow2
                    )
                    + 0.046849371468156265 * source_params.S_tot_hat_pow2
                    + source_params.eta_pow2
                    * source_params.eta
                    * source_params.S_tot_hat
                    * (
                        -27.976989112929353
                        + 6.404060932334562 * source_params.S_tot_hat
                        - 633.1966645925428
                        * source_params.S_tot_hat_pow2
                        * source_params.S_tot_hat
                        + 109.04824706217418 * source_params.S_tot_hat_pow2
                    )
                )
                / (1.0 - 0.6862449113932192 * source_params.S_tot_hat)
            )
            + (
                641.8965762829259
                * source_params.delta_chi
                * source_params.delta
                * source_params.eta_pow2
                * source_params.eta_pow2
                * source_params.eta
            )
        )
        # Merge-ringdown phase at collocation point f4 (f_ring)
        v4_mr = (
            (
                (
                    -85.86062966719405
                    - 4616.740713893726 * source_params.eta
                    - 4925.756920247186 * source_params.eta_pow2
                    + 7732.064464348168 * source_params.eta_pow2 * source_params.eta
                    + 12828.269960300782
                    * source_params.eta_pow2
                    * source_params.eta_pow2
                    - 39783.51698102803
                    * source_params.eta_pow2
                    * source_params.eta_pow2
                    * source_params.eta
                )
                / (1.0 + 50.206318806624004 * source_params.eta)
            )
            + (
                (
                    source_params.S_tot_hat
                    * (
                        33.335857451144356
                        - 36.49019206094966 * source_params.S_tot_hat
                        + source_params.eta_pow2
                        * source_params.eta
                        * (
                            1497.3545918387515
                            - 101.72731770500685 * source_params.S_tot_hat
                        )
                        * source_params.S_tot_hat
                        - 3.835967351280833 * source_params.S_tot_hat_pow2
                        + 2.302712009652155
                        * source_params.S_tot_hat_pow2
                        * source_params.S_tot_hat
                        + source_params.eta_pow2
                        * (
                            93.64156367505917
                            - 18.184492163348665 * source_params.S_tot_hat
                            + 423.48863373726243 * source_params.S_tot_hat_pow2
                            - 104.36120236420928
                            * source_params.S_tot_hat_pow2
                            * source_params.S_tot_hat
                            - 719.8775484010988
                            * source_params.S_tot_hat_pow2
                            * source_params.S_tot_hat_pow2
                        )
                        + 1.6533417657003922
                        * source_params.S_tot_hat_pow2
                        * source_params.S_tot_hat_pow2
                        + source_params.eta
                        * (
                            -69.19412903018717
                            + 26.580344399838758 * source_params.S_tot_hat
                            - 15.399770764623746 * source_params.S_tot_hat_pow2
                            + 31.231253209893488
                            * source_params.S_tot_hat_pow2
                            * source_params.S_tot_hat
                            + 97.69027029734173
                            * source_params.S_tot_hat_pow2
                            * source_params.S_tot_hat_pow2
                        )
                        + source_params.eta_pow2
                        * source_params.eta_pow2
                        * (
                            1075.8686153198323
                            - 3443.0233614187396 * source_params.S_tot_hat
                            - 4253.974688619423 * source_params.S_tot_hat_pow2
                            - 608.2901586790335
                            * source_params.S_tot_hat_pow2
                            * source_params.S_tot_hat
                            + 5064.173605639933
                            * source_params.S_tot_hat_pow2
                            * source_params.S_tot_hat_pow2
                        )
                    )
                )
                / (-1.3705601055555852 + 1.0 * source_params.S_tot_hat)
            )
            + (
                source_params.delta_chi
                * source_params.delta
                * source_params.eta
                * (22.363215261437862 + 156.08206945239374 * source_params.eta)
            )
        )
        # Difference between collocation points 5 and 4 (v5 - v4)
        v5_mr = (
            (
                (
                    source_params.eta
                    * (
                        7.05731400277692
                        + 22.455288821807095 * source_params.eta
                        + 119.43820622871043 * source_params.eta_pow2
                    )
                )
                / (0.26026709603623255 + 1.0 * source_params.eta)
            )
            + (
                (
                    source_params.eta_pow2
                    * (134.88158268621922 - 56.05992404859163 * source_params.S_tot_hat)
                    * source_params.S_tot_hat
                    + source_params.eta
                    * source_params.S_tot_hat
                    * (
                        -7.9407123129681425
                        + 9.486783128047414 * source_params.S_tot_hat
                    )
                    + source_params.eta_pow2
                    * source_params.eta
                    * source_params.S_tot_hat
                    * (
                        -316.26970506215554
                        + 90.31815139272628 * source_params.S_tot_hat
                    )
                )
                / (1.0 - 0.7162058321905909 * source_params.S_tot_hat)
            )
            + (
                43.82713604567481
                * source_params.delta_chi
                * source_params.delta
                * source_params.eta_pow2
                * source_params.eta
            )
        )
        v5_mr = v5_mr + v4_mr
        v3_mr = v3_mr + v4_mr
        v2_mr = v2_mr + v4_mr
        v1_mr = v1_mr + v2_mr

        Ab_mr = ti.Matrix(
            [
                [
                    1,
                    fmin_mr ** (-1 / 3),
                    fmin_mr ** (-2),
                    fmin_mr ** (-4),
                    dphase0 / (f_damp_pow2 + (fmin_mr - source_params.f_ring) ** 2),
                    v1_mr,
                ],
                [
                    1,
                    f2_mr ** (-1 / 3),
                    f2_mr ** (-2),
                    f2_mr ** (-4),
                    dphase0 / (f_damp_pow2 + (f2_mr - source_params.f_ring) ** 2),
                    v2_mr,
                ],
                [
                    1,
                    f3_mr ** (-1 / 3),
                    f3_mr ** (-2),
                    f3_mr ** (-4),
                    dphase0 / (f_damp_pow2 + (f3_mr - source_params.f_ring) ** 2),
                    v3_mr,
                ],
                [
                    1,
                    f3_mr ** (-1 / 3),
                    f3_mr ** (-2),
                    f3_mr ** (-4),
                    dphase0 / (f_damp_pow2 + (f3_mr - source_params.f_ring) ** 2),
                    v3_mr,
                ],
                [
                    1,
                    fmax_mr ** (-1 / 3),
                    fmax_mr ** (-2),
                    fmax_mr ** (-4),
                    dphase0 / (f_damp_pow2 + (fmax_mr - source_params.f_ring) ** 2),
                    v4_mr,
                ],
            ]
        )
        self.c_0, self.c_1, self.c_2, self.c_4, self.cRD = gauss_elimination(Ab_mr)

        # Inspiral (Only the recommended fitting model with 4 pseudo-PN coefficients is implemented)
        f_phase_ins_max = 1.02 * source_params.f_MECO
        f_range_ins = f_phase_ins_max - f_phase_ins_min
        f2_ins = f_phase_ins_min + 1.0 / 4.0 * f_range_ins
        f3_ins = f_phase_ins_min + 3.0 / 4.0 * f_range_ins

        # Value of v1 - v3
        d13_ins = (
            (
                -17294.000000000007
                - 19943.076428555978 * source_params.eta
                + 483033.0998073767 * source_params.eta_pow2
            )
            / (1.0 + 4.460294035404433 * source_params.eta)
            + (
                source_params.S_tot_hat
                * (
                    68384.62786426462
                    + 67663.42759836042 * source_params.S_tot_hat
                    - 2179.3505885609297 * source_params.S_tot_hat_pow2
                    + source_params.eta
                    * (
                        -58475.33302037833
                        + 62190.404951852535 * source_params.S_tot_hat
                        + 18298.307770807573 * source_params.S_tot_hat_pow2
                        - 303141.1945565486 * source_params.S_tot_hat_pow3
                    )
                    + 19703.894135534803 * source_params.S_tot_hat_pow3
                    + source_params.eta_pow2
                    * (
                        -148368.4954044637
                        - 758386.5685734496 * source_params.S_tot_hat
                        - 137991.37032619823 * source_params.S_tot_hat_pow2
                        + 1.0765877367729193e6 * source_params.S_tot_hat_pow3
                    )
                    + 32614.091002011017 * source_params.S_tot_hat_pow4
                )
            )
            / (2.0412979553629143 + 1.0 * source_params.S_tot_hat)
            + 12017.062595934838
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta
        )
        # Value of v2 - v3
        d23_ins = (
            (
                -7579.300000000004
                - 120297.86185566607 * source_params.eta
                + 1.1694356931282217e6 * source_params.eta_pow2
                - 557253.0066989232 * source_params.eta_pow3
            )
            / (1.0 + 18.53018618227582 * source_params.eta)
            + (
                source_params.S_tot_hat
                * (
                    -27089.36915061857
                    - 66228.9369155027 * source_params.S_tot_hat
                    + source_params.eta_pow2
                    * (
                        150022.21343386435
                        - 50166.382087278434 * source_params.S_tot_hat
                        - 399712.22891153296 * source_params.S_tot_hat_pow2
                    )
                    - 44331.41741405198 * source_params.S_tot_hat_pow2
                    + source_params.eta
                    * (
                        50644.13475990821
                        + 157036.45676788126 * source_params.S_tot_hat
                        + 126736.43159783827 * source_params.S_tot_hat_pow2
                    )
                    + source_params.eta_pow3
                    * (
                        -593633.5370110178
                        - 325423.99477314285 * source_params.S_tot_hat
                        + 847483.2999508682 * source_params.S_tot_hat_pow2
                    )
                )
            )
            / (
                -1.5232497464826662
                - 3.062957826830017 * source_params.S_tot_hat
                - 1.130185486082531 * source_params.S_tot_hat_pow2
                + 1.0 * source_params.S_tot_hat_pow3
            )
            + 3843.083992827935
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta
        )
        # Value of v3
        v3_ins = (
            (
                15415.000000000007
                + 873401.6255736464 * source_params.eta
                + 376665.64637025696 * source_params.eta_pow2
                - 3.9719980569125614e6 * source_params.eta_pow3
                + 8.913612508054944e6 * source_params.eta_pow4
            )
            / (1.0 + 46.83697749859996 * source_params.eta)
            + (
                source_params.S_tot_hat
                * (
                    397951.95299014193
                    - 207180.42746987 * source_params.S_tot_hat
                    + source_params.eta_pow3
                    * (
                        4.662143741417853e6
                        - 584728.050612325 * source_params.S_tot_hat
                        - 1.6894189124921719e6 * source_params.S_tot_hat_pow2
                    )
                    + source_params.eta
                    * (
                        -1.0053073129700898e6
                        + 1.235279439281927e6 * source_params.S_tot_hat
                        - 174952.69161683554 * source_params.S_tot_hat_pow2
                    )
                    - 130668.37221912303 * source_params.S_tot_hat_pow2
                    + source_params.eta_pow2
                    * (
                        -1.9826323844247842e6
                        + 208349.45742548333 * source_params.S_tot_hat
                        + 895372.155565861 * source_params.S_tot_hat_pow2
                    )
                )
            )
            / (
                -9.675704197652225
                + 3.5804521763363075 * source_params.S_tot_hat
                + 2.5298346636273306 * source_params.S_tot_hat_pow2
                + 1.0 * source_params.S_tot_hat_pow3
            )
            + -1296.9289110696955 * source_params.delta_chi_pow2 * source_params.eta
            + source_params.delta_chi
            * source_params.delta
            * source_params.eta
            * (
                -24708.109411857182
                + 24703.28267342699 * source_params.eta
                + 47752.17032707405 * source_params.S_tot_hat
            )
        )
        # Value of v4 - v3
        d43_ins = (
            (
                2439.000000000001
                - 31133.52170083207 * source_params.eta
                + 28867.73328134167 * source_params.eta_pow2
            )
            / (1.0 + 0.41143032589262585 * source_params.eta)
            + (
                source_params.S_tot_hat
                * (
                    16116.057657391262
                    + source_params.eta_pow3
                    * (
                        -375818.0132734753
                        - 386247.80765802023 * source_params.S_tot_hat
                    )
                    + source_params.eta
                    * (-82355.86732027541 - 25843.06175439942 * source_params.S_tot_hat)
                    + 9861.635308837876 * source_params.S_tot_hat
                    + source_params.eta_pow2
                    * (
                        229284.04542668918
                        + 117410.37432997991 * source_params.S_tot_hat
                    )
                )
            )
            / (
                -3.7385208695213668
                + 0.25294420589064653 * source_params.S_tot_hat
                + 1.0 * source_params.S_tot_hat_pow2
            )
            + 194.5554531509207
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta
        )
        v1_ins = d13_ins + v3_ins
        v2_ins = d23_ins + v3_ins
        v4_ins = d43_ins + v3_ins
        Ab_ins = ti.Matrix(
            [
                [
                    1,
                    f_phase_ins_min ** (1 / 3),
                    f_phase_ins_min ** (2 / 3),
                    f_phase_ins_min,
                    v1_ins,
                ],
                [
                    1,
                    f2_ins ** (1 / 3),
                    f2_ins ** (2 / 3),
                    f2_ins,
                    v2_ins,
                ],
                [
                    1,
                    f3_ins ** (1 / 3),
                    f3_ins ** (2 / 3),
                    f3_ins,
                    v3_ins,
                ],
                [
                    1,
                    f_phase_ins_max ** (1 / 3),
                    f_phase_ins_max ** (2 / 3),
                    f_phase_ins_max,
                    v4_ins,
                ],
            ]
        )
        sigma_1, sigma_2, sigma_3, sigma_4 = gauss_elimination(Ab_ins)
        self.sigma_1 = -5.0 / 3.0 * sigma_1
        self.sigma_2 = -5.0 / 4.0 * sigma_2
        self.sigma_3 = -1.0 * sigma_3
        self.sigma_4 = -5.0 / 6.0 * sigma_4
        # intermediate (105, 5 collocation coefficients)
        # !!! check again
        f_phase_T = 0.6 * (0.5 * source_params.f_ring + source_params.f_ISCO)
        delta_R = 0.03 * (f_phase_T - source_params.f_MECO)
        f_phase_int_min = source_params.f_MECO - delta_R
        f_phase_int_max = f_phase_T + 0.5 * delta_R
        f_range_int = f_phase_int_max - f_phase_int_min
        f2_int = f_phase_int_min + (0.5 - 1 / (2 * sqrt2)) * f_range_int
        f3_int = f_phase_int_min + 0.5 * f_range_int
        f4_int = f_phase_int_min + (0.5 + 1 / (2 * sqrt2)) * f_range_int

        # v2_int_bar - v4_MRD.
        d_v2intbar_v4MRD = (
            (
                source_params.eta
                * (
                    0.9951733419499662
                    + 101.21991715215253 * source_params.eta
                    + 632.4731389009143 * eta_pow2
                )
            )
            / (
                0.00016803066316882238
                + 0.11412314719189287 * source_params.eta
                + 1.8413983770369362 * eta_pow2
                + 1.0 * eta_pow3
            )
            + (
                source_params.S_tot_hat
                * (
                    18.694178521101332
                    + 16.89845522539974 * source_params.S_tot_hat
                    + 4941.31613710257 * eta_pow2 * source_params.S_tot_hat
                    + source_params.eta
                    * (
                        -697.6773920613674
                        - 147.53381808989846 * source_params.S_tot_hat_pow2
                    )
                    + 0.3612417066833153 * source_params.S_tot_hat_pow2
                    + eta_pow3
                    * (
                        3531.552143264721
                        - 14302.70838220423 * source_params.S_tot_hat
                        + 178.85850322465944 * source_params.S_tot_hat_pow2
                    )
                )
            )
            / (
                2.965640445745779
                - 2.7706595614504725 * source_params.S_tot_hat
                + 1.0 * source_params.S_tot_hat_pow2
            )
            + source_params.delta_chi
            * source_params.delta
            * eta_pow2
            * (
                356.74395864902294
                + 1693.326644293169 * eta_pow2 * source_params.S_tot_hat
            )
        )
        # v3_int - v4_MRD.
        d_v3int_v4MRD = (
            (
                source_params.eta
                * (
                    -5.126358906504587
                    - 227.46830225846668 * source_params.eta
                    + 688.3609087244353 * eta_pow2
                    - 751.4184178636324 * eta_pow3
                )
            )
            / (
                -0.004551938711031158
                - 0.7811680872741462 * source_params.eta
                + 1.0 * eta_pow2
            )
            + (
                source_params.S_tot_hat
                * (
                    0.1549280856660919
                    - 0.9539250460041732 * source_params.S_tot_hat
                    - 539.4071941841604 * eta_pow2 * source_params.S_tot_hat
                    + source_params.eta
                    * (
                        73.79645135116367
                        - 8.13494176717772 * source_params.S_tot_hat_pow2
                    )
                    - 2.84311102369862 * source_params.S_tot_hat_pow2
                    + eta_pow3
                    * (
                        -936.3740515136005
                        + 1862.9097047992134 * source_params.S_tot_hat
                        + 224.77581754671272 * source_params.S_tot_hat_pow2
                    )
                )
            )
            / (-1.5308507364054487 + 1.0 * source_params.S_tot_hat)
            + 2993.3598520496153
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow6
        )
        # v4_int - v3_int.
        d43_int = (
            (
                0.4248820426833804
                - 906.746595921514 * source_params.eta
                - 282820.39946006844 * eta_pow2
                - 967049.2793750163 * eta_pow3
                + 670077.5414916876 * eta_pow4
            )
            / (
                1.0
                + 1670.9440812294847 * source_params.eta
                + 19783.077247023448 * eta_pow2
            )
            + (
                source_params.S_tot_hat
                * (
                    0.22814271667259703
                    + 1.1366593671801855 * source_params.S_tot_hat
                    + eta_pow3
                    * (
                        3499.432393555856
                        - 877.8811492839261 * source_params.S_tot_hat
                        - 4974.189172654984 * source_params.S_tot_hat_pow2
                    )
                    + source_params.eta
                    * (
                        12.840649528989287
                        - 61.17248283184154 * source_params.S_tot_hat_pow2
                    )
                    + 0.4818323187946999 * source_params.S_tot_hat_pow2
                    + eta_pow2
                    * (
                        -711.8532052499075
                        + 269.9234918621958 * source_params.S_tot_hat
                        + 941.6974723887743 * source_params.S_tot_hat_pow2
                    )
                    + eta_pow4
                    * (
                        -4939.642457025497
                        - 227.7672020783411 * source_params.S_tot_hat
                        + 8745.201037897836 * source_params.S_tot_hat_pow2
                    )
                )
            )
            / (-1.2442293719740283 + 1.0 * source_params.S_tot_hat)
            + source_params.delta_chi
            * source_params.delta
            * (-514.8494071830514 + 1493.3851099678195 * source_params.eta)
            * eta_pow3
        )
        # v2_int_bar
        v2_int_bar = (
            (
                -82.54500000000004
                - 5.58197349185435e6 * source_params.eta
                - 3.5225742421184325e8 * eta_pow2
                + 1.4667258334378073e9 * eta_pow3
            )
            / (
                1.0
                + 66757.12830903867 * source_params.eta
                + 5.385164380400193e6 * eta_pow2
                + 2.5176585751772933e6 * eta_pow3
            )
            + (
                source_params.S_tot_hat
                * (
                    19.416719811164853
                    - 36.066611959079935 * source_params.S_tot_hat
                    - 0.8612656616290079 * source_params.S_tot_hat_pow2
                    + eta_pow2
                    * (
                        170.97203068800542
                        - 107.41099349364234 * source_params.S_tot_hat
                        - 647.8103976942541 * source_params.S_tot_hat_pow3
                    )
                    + 5.95010003393006 * source_params.S_tot_hat_pow3
                    + eta_pow3
                    * (
                        -1365.1499998427248
                        + 1152.425940764218 * source_params.S_tot_hat
                        + 415.7134909564443 * source_params.S_tot_hat_pow2
                        + 1897.5444343138167 * source_params.S_tot_hat_pow3
                        - 866.283566780576 * source_params.S_tot_hat_pow4
                    )
                    + 4.984750041013893 * source_params.S_tot_hat_pow4
                    + source_params.eta
                    * (
                        207.69898051583655
                        - 132.88417400679026 * source_params.S_tot_hat
                        - 17.671713040498304 * source_params.S_tot_hat_pow2
                        + 29.071788188638315 * source_params.S_tot_hat_pow3
                        + 37.462217031512786 * source_params.S_tot_hat_pow4
                    )
                )
            )
            / (-1.1492259468169692 + 1.0 * source_params.S_tot_hat)
            + source_params.delta_chi
            * source_params.delta
            * eta_pow3
            * (
                7343.130973149263
                - 20486.813161100774 * source_params.eta
                + 515.9898508588834 * source_params.S_tot_hat
            )
        )

        v1_int = _d_phase_inspiral_ansatz()
        v2_int = 0.75 * (d_v2intbar_v4MRD + v4_MRD) + 0.25 * v2_int_bar
        v3_int = d_v3int_v4MRD + v4_MRD
        v4_int = d43_int + v3_int
        v5_int = _d_phase_merge_ringdown_ansatz()

        Ab_int = ti.Matrix(
            [
                [
                    1.0,
                    source_params.f_ring / f_phase_int_min,
                    (source_params.f_ring / f_phase_int_min) ** 2,
                    (source_params.f_ring / f_phase_int_min) ** 3,
                    (source_params.f_ring / f_phase_int_min) ** 4,
                    v1_int,
                ],
                [
                    1.0,
                    source_params.f_ring / f2_int,
                    (source_params.f_ring / f2_int) ** 2,
                    (source_params.f_ring / f2_int) ** 3,
                    (source_params.f_ring / f2_int) ** 4,
                    v2_int,
                ],
                [
                    1.0,
                    source_params.f_ring / f3_int,
                    (source_params.f_ring / f3_int) ** 2,
                    (source_params.f_ring / f3_int) ** 3,
                    (source_params.f_ring / f3_int) ** 4,
                    v3_int,
                ],
                [
                    1.0,
                    source_params.f_ring / f4_int,
                    (source_params.f_ring / f4_int) ** 2,
                    (source_params.f_ring / f4_int) ** 3,
                    (source_params.f_ring / f4_int) ** 4,
                    v4_int,
                ],
                [
                    1.0,
                    source_params.f_ring / f_phase_int_max,
                    (source_params.f_ring / f_phase_int_max) ** 2,
                    (source_params.f_ring / f_phase_int_max) ** 3,
                    (source_params.f_ring / f_phase_int_max) ** 4,
                    v5_int,
                ],
            ]
        )

        beta_0, beta_1, beta_2, beta_3, beta_4 = gauss_elimination(Ab_int)
        self.beta_0 = beta_0
        self.beta_1 = beta_1 * source_params.f_ring
        self.beta_2 = beta_2 * source_params.f_ring**2
        self.beta_3 = beta_3 * source_params.f_ring**3
        self.beta_4 = beta_4 * source_params.f_ring**4


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
        # t0 = (_d_phase_merge_ringdown_ansatz(powers_of_Mf, self.phase_coefficients[None], self.source_parameters[None].f_ring, self.source_parameters[None].f_damp) + self.phase_coefficients[None].c_2_merge_ringdown)/self.source_parameters[None].eta
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
            self.source_parameters[None].m_1 > self.source_parameters[None].m_2
        ), f"require m1 > m2, you are passing m1: {self.source_parameters[None].m_1}, m2:{self.source_parameters[None].m_2}"
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
