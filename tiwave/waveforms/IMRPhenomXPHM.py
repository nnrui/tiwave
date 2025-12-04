from typing import Callable

import taichi as ti
import taichi.math as tm
import numpy as np
from numpy.typing import NDArray

from ..constants import *
from ..utils import UsefulPowers, ti_complex, sub_struct_from
from .base_waveform import BaseWaveform
from .common import PostNewtonianCoefficients
from .IMRPhenomXAS import (
    SourceParameters,
    PhaseCoefficients,
    AmplitudeCoefficients,
    PHENOMXAS_HIGH_FREQUENCY_CUT,
)


powers_of_pi = UsefulPowers()
powers_of_pi.update(PI)


@ti.func
def rotate_z(angle: float, vec: ti.types.vector(3, float)) -> ti.types.vector(3, float):
    cos_angle = tm.cos(angle)
    sin_angle = tm.sin(angle)
    return ti.Vector(
        [
            vec.x * cos_angle - vec.y * sin_angle,
            vec.x * sin_angle + vec.y * cos_angle,
            vec.z,
        ],
        dt=float,
    )


@ti.func
def rotate_y(angle: float, vec: ti.types.vector(3, float)) -> ti.types.vector(3, float):
    cos_angle = tm.cos(angle)
    sin_angle = tm.sin(angle)
    return ti.Vector(
        [
            vec.x * cos_angle + vec.z * sin_angle,
            vec.y,
            -vec.x * sin_angle + vec.z * cos_angle,
        ],
        dt=float,
    )


@sub_struct_from(SourceParameters)
class SourceParametersPrecessionNNLO:
    chi1_vec: ti.types.vector(3, dtype=float)
    chi2_vec: ti.types.vector(3, dtype=float)
    S1_vec: ti.types.vector(3, dtype=float)
    S2_vec: ti.types.vector(3, dtype=float)
    S_para: float
    S_perp: float
    chi_l: float
    chi_p: float
    chi_l_pow2: float
    chi_p_pow2: float
    delta_pow2: float
    delta_pow3: float
    m1_dimless_pow2: float
    m1_dimless_pow3: float
    m1_dimless_pow4: float
    m1_dimless_pow5: float
    m1_dimless_pow6: float
    m1_dimless_pow7: float
    m1_dimless_pow8: float
    phase_ref_in: float

    @ti.func
    def _update_precessing_final_state(self):
        af_prec = tm.sign(self.final_spin) * tm.sqrt(
            self.S_perp**2 + self.final_spin_pow2
        )
        self.final_spin = tm.clamp(af_prec, -1.0, 1.0)

        self.final_spin_pow2 = self.final_spin * self.final_spin
        self.final_spin_pow3 = self.final_spin * self.final_spin_pow2
        self.final_spin_pow4 = self.final_spin * self.final_spin_pow3
        self.final_spin_pow5 = self.final_spin * self.final_spin_pow4
        self.final_spin_pow6 = self.final_spin * self.final_spin_pow5
        self.final_spin_pow7 = self.final_spin * self.final_spin_pow6

        self.f_ring = self._get_f_ring()
        self.f_damp = self._get_f_damp()
        self.f_damp_pow2 = self.f_damp * self.f_damp

    @ti.func
    def update_source_parameters(
        self,
        mass_1: float,
        mass_2: float,
        chi1_x: float,
        chi1_y: float,
        chi1_z: float,
        chi2_x: float,
        chi2_y: float,
        chi2_z: float,
        luminosity_distance: float,
        inclination: float,
        reference_phase: float,
        reference_frequency: float,
    ):
        self._parent_update_source_parameters(
            mass_1,
            mass_2,
            chi1_z,
            chi2_z,
            luminosity_distance,
            inclination,
            reference_phase,
            reference_frequency,
        )
        # in the default convention, reference phase passed to nonprecessing modes is 0.0
        self.phase_ref_in = self.phase_ref  # input phase_ref
        self.phase_ref = 0.0  # passed to nonprecessing modes

        self.delta_pow2 = self.delta * self.delta
        self.delta_pow3 = self.delta * self.delta_pow2
        self.m1_dimless_pow2 = self.m1_dimless * self.m1_dimless
        self.m1_dimless_pow3 = self.m1_dimless * self.m1_dimless_pow2
        self.m1_dimless_pow4 = self.m1_dimless * self.m1_dimless_pow3
        self.m1_dimless_pow5 = self.m1_dimless * self.m1_dimless_pow4
        self.m1_dimless_pow6 = self.m1_dimless * self.m1_dimless_pow5
        self.m1_dimless_pow7 = self.m1_dimless * self.m1_dimless_pow6
        self.m1_dimless_pow8 = self.m1_dimless * self.m1_dimless_pow7

        self.chi1_vec = ti.Vector([chi1_x, chi1_y, chi1_z], dt=float)
        self.chi2_vec = ti.Vector([chi2_x, chi2_y, chi2_z], dt=float)
        self.S1_vec = self.chi1_vec * self.m1_dimless_pow2
        self.S2_vec = self.chi2_vec * self.m2_dimless * self.m2_dimless

        A1 = 2.0 + (3.0 * mass_2) / (2.0 * mass_1)
        A2 = 2.0 + (3.0 * mass_1) / (2.0 * mass_2)
        S1_perp = tm.sqrt(self.S1_vec.x * self.S1_vec.x + self.S1_vec.y * self.S1_vec.y)
        S2_perp = tm.sqrt(self.S2_vec.x * self.S2_vec.x + self.S2_vec.y * self.S2_vec.y)

        self.chi_p = tm.max(A1 * S1_perp, A2 * S2_perp) / (A1 * self.m1_dimless_pow2)
        self.chi_l = self.chi_eff / self.m1_dimless
        self.chi_p_pow2 = self.chi_p * self.chi_p
        self.chi_l_pow2 = self.chi_l * self.chi_l

        self.S_para = self.S1_vec.z + self.S2_vec.z
        self.S_perp = self.chi_p * self.m1_dimless_pow2

        self._update_precessing_final_state()


# @ti.func
# def spin_minus_2_spherical_harmonic(
#     theta: float, phi: float, l: int, m: int
# ) -> ti_complex:
#     fac = 0.0
#     if ti.static(l == 2):
#         if ti.static(m == -2):
#             pass

#     return fac * tm.cexp(ti_complex[0.0, m * phi])


# @ti.dataclass
# class SphericalHarmonicL2:
#     Y_2m2: ti_complex
#     Y_2m1: ti_complex
#     Y_20: ti_complex
#     Y_21: ti_complex
#     Y_22: ti_complex

#     @ti.func
#     def update(self, theta: float, phi: float):
#         pass


# @ti.dataclass
# class SphericalHarmonicL3:
#     Y_3m3: ti_complex
#     Y_3m2: ti_complex
#     Y_3m1: ti_complex
#     Y_30: ti_complex
#     Y_31: ti_complex
#     Y_32: ti_complex
#     Y_33: ti_complex

#     @ti.func
#     def update(self, theta: float, phi: float):
#         pass


# @ti.dataclass
# class SphericalHarmonicL4:
#     Y_4m4: ti_complex
#     Y_4m3: ti_complex
#     Y_4m2: ti_complex
#     Y_4m1: ti_complex
#     Y_40: ti_complex
#     Y_41: ti_complex
#     Y_42: ti_complex
#     Y_43: ti_complex
#     Y_44: ti_complex

#     @ti.func
#     def update(self, theta: float, phi: float):
#         pass


# @ti.dataclass
# class PrecessionCoefficients:

#     spher_harms_l2: SphericalHarmonicL2
#     spher_harms_l3: SphericalHarmonicL3
#     spher_harms_l4: SphericalHarmonicL4

#     @ti.func
#     def update_precession_coefficients(
#         self, source_params: ti.template(), high_modes: ti.template()
#     ):
#         pass


@ti.dataclass
class PrecessionCoefficientsNNLO:
    alpha_m3: float
    alpha_m2: float
    alpha_m1: float
    alpha_1: float
    alpha_log: float

    gamma_m3: float
    gamma_m2: float
    gamma_m1: float
    gamma_1: float
    gamma_log: float

    L_0: float
    L_1: float
    L_2: float
    L_3: float
    L_4: float
    L_5: float
    L_6: float

    alpha_0: float
    gamma_0: float
    cos_2zeta: float
    sin_2zeta: float
    spin_minus2_Y2m: ti.types.vector(5, dtype=float)

    @ti.func
    def _set_convention_constants(self, source_params: ti.template()):
        """
        Set alpha_0, gamma_0, and zeta, referring Appendix C and D in 2004.06503
        """
        powers_Mf_ref = UsefulPowers()
        powers_Mf_ref.update(source_params.Mf_ref)

        # total angular momentum in L0 frame, Eq C5 in 2004.06503
        L0_norm = self._compute_L_norm_3PN(source_params.eta, powers_Mf_ref)
        J_vec_L0 = ti.Vector(
            [
                source_params.S1_vec.x + source_params.S2_vec.x,
                source_params.S1_vec.y + source_params.S2_vec.y,
                source_params.S1_vec.z + source_params.S2_vec.z + L0_norm,
            ],
            dt=float,
        )
        J_norm = tm.length(J_vec_L0)

        theta_J_L0 = 0.0
        if J_norm < 1.0e-10:
            theta_J_L0 = 0.0
        else:
            theta_J_L0 = tm.acos(J_vec_L0.z / J_norm)  # Eq. C6 in 2004.06503

        phi_J_L0 = 0.0
        if ti.abs(J_vec_L0.x) < 1.0e-15 and ti.abs(J_vec_L0.y) < 1.0e-15:
            phi_J_L0 = 0.0
        else:
            phi_J_L0 = tm.atan2(J_vec_L0.y, J_vec_L0.x)  # Eq. C7 in 2004.06503

        # get Nhat in Jprime frame
        N_vec_L0 = ti.Vector(
            [
                tm.sin(source_params.iota)
                * tm.cos(PI / 2.0 - source_params.phase_ref_in),
                tm.sin(source_params.iota)
                * tm.sin(PI / 2.0 - source_params.phase_ref_in),
                tm.cos(source_params.iota),
            ]
        )  # Eq. C4
        N_vec_Jprime = rotate_y(-theta_J_L0, rotate_z(-phi_J_L0, N_vec_L0))  # Eq. C11

        kappa = 0.0
        if ti.abs(N_vec_Jprime.x) < 1.0e-15 and ti.abs(N_vec_Jprime.y) < 1.0e-15:
            kappa = 0.0
        else:
            kappa = tm.atan2(N_vec_Jprime.y, N_vec_Jprime.x)

        alpha_0 = PI - kappa
        gamma_0 = PI - phi_J_L0  # note gamma = -epsilon

        alpha_ref = self._compute_alpha_core(powers_Mf_ref)
        gamma_ref = self._compute_gamma_core(powers_Mf_ref)

        self.alpha_0 = alpha_0 - alpha_ref
        self.gamma_0 = gamma_0 - gamma_ref

        # set zeta for polarization convention, adopting the P, Q basis from 0810.5336 (Arun et al)
        X_vec_L0 = ti.Vector(
            [
                -tm.cos(source_params.iota) * tm.sin(source_params.phase_ref_in),
                -tm.cos(source_params.iota) * tm.cos(source_params.phase_ref_in),
                tm.sin(source_params.iota),
            ],
            dt=float,
        )  # Eq. C24 in 2004.06503
        X_vec_J = rotate_z(
            -kappa, (rotate_y(-theta_J_L0, rotate_z(-phi_J_L0, X_vec_L0)))
        )  # Eq. C13 in 2004.06503
        # the angle between J and N, Eq. C10
        theta_J_N = tm.acos(tm.dot(J_vec_L0, N_vec_L0) / J_norm)
        cos_theta_J_N = tm.cos(theta_J_N)
        sin_theta_J_N = tm.sin(theta_J_N)
        # adopting convention from Arun et al, Eq. C23 in 2004.06503
        P_vec_J = ti.Vector([cos_theta_J_N, 0.0, -sin_theta_J_N], dt=float)
        Q_vec_J = ti.Vector([0.0, 1.0, 0.0], dt=float)
        zeta = tm.atan2(tm.dot(X_vec_J, Q_vec_J), tm.dot(X_vec_J, P_vec_J))
        self.cos_2zeta = tm.cos(2.0 * zeta)
        self.sin_2zeta = tm.sin(2.0 * zeta)

        # set the sperical harmonic factors [Y_{2,-2}, Y_{2,-1}, Y_{2,0}, Y_{2,1}, Y_{2,2}]
        self.spin_minus2_Y2m = ti.Vector(
            [
                tm.sqrt(5.0 / (64.0 * PI)) * (1.0 - cos_theta_J_N) ** 2,
                tm.sqrt(5.0 / (16.0 * PI)) * sin_theta_J_N * (1.0 - cos_theta_J_N),
                tm.sqrt(15.0 / (32.0 * PI)) * sin_theta_J_N**2,
                tm.sqrt(5.0 / (16.0 * PI)) * sin_theta_J_N * (1.0 + cos_theta_J_N),
                tm.sqrt(5.0 / (64.0 * PI)) * (1.0 + cos_theta_J_N) ** 2,
            ],
            dt=float,  # phi=0, the harmonic factors are real
        )

    @ti.func
    def _set_alpha_coefficients(self, source_params: ti.template()):
        # Eq. G9a - G9e in 2004.06503
        self.alpha_m3 = (
            -3.5 / 19.2 + 5.0 * source_params.delta / (64.0 * source_params.m1_dimless)
        ) / powers_of_pi.one
        self.alpha_m2 = (
            5.0
            * source_params.m1_dimless
            * source_params.chi_l
            * (3.0 * source_params.delta - 7.0 * source_params.m1_dimless)
            / (128.0 * source_params.eta)
        ) / powers_of_pi.two_thirds
        self.alpha_m1 = (
            -5.515 / 3.072
            + (4.555 * source_params.delta) / (7.168 * source_params.m1_dimless)
            + (
                -5.15 / 3.84
                - (1.5 * source_params.delta_pow2)
                / (25.6 * source_params.m1_dimless_pow2)
                + (1.75 * source_params.delta) / (2.56 * source_params.m1_dimless)
            )
            * source_params.eta
            + (
                (1.5 * source_params.delta * source_params.m1_dimless_pow3)
                - (3.5 * source_params.m1_dimless_pow4)
            )
            * source_params.chi_p_pow2
            / (12.8 * source_params.eta_pow2)
        ) / powers_of_pi.third
        self.alpha_1 = (
            40.121485 / 9.289728
            + 3.9695 / 8.6016 * source_params.eta
            + 9.55 / 5.76 * source_params.eta_pow2
            - 1.5
            / 102.4
            * source_params.delta_pow3
            * source_params.eta_pow2
            / source_params.m1_dimless_pow3
            + (1.615 / 28.672 * source_params.eta + 3.5 / 25.6 * source_params.eta_pow2)
            * source_params.delta_pow2
            / source_params.m1_dimless_pow2
            + (
                -2.7895885 / 2.1676032
                + 2.65 / 143.36 * source_params.eta
                - 2.725 / 3.072 * source_params.eta_pow2
            )
            * source_params.delta
            / source_params.m1_dimless
            + (
                4.85 / 143.36 * source_params.chi_p_pow2 / source_params.eta_pow2
                + (
                    -18.15 / 2.56 * source_params.chi_l_pow2
                    - 1.45 / 5.12 * source_params.chi_p_pow2
                )
                / source_params.eta
            )
            * source_params.delta
            * source_params.m1_dimless_pow3
            + (
                4.75 / 61.44 * source_params.chi_p_pow2 / source_params.eta_pow2
                + (
                    16.45 / 1.92 * source_params.chi_l_pow2
                    + 5.75 / 15.36 * source_params.chi_p_pow2
                )
                / source_params.eta
            )
            * source_params.m1_dimless_pow4
            + (
                -1.5 / 12.8 * source_params.chi_l_pow2
                + 1.5 / 51.2 * source_params.chi_p_pow2
            )
            * source_params.chi_p_pow2
            * source_params.delta
            * source_params.m1_dimless_pow7
            / source_params.eta_pow4
            + (
                3.5 / 12.8 * source_params.chi_l_pow2
                - 3.5 / 51.2 * source_params.chi_p_pow2
            )
            * source_params.chi_p_pow2
            * source_params.m1_dimless_pow8
            / source_params.eta_pow4
            + (
                1.5
                / 1.6
                * source_params.chi_l
                * source_params.delta
                * PI
                / source_params.eta
                * source_params.m1_dimless
            )
            + (
                (
                    3.75 / 2.56 * source_params.chi_l_pow2
                    + 1.5 / 25.6 * source_params.chi_p_pow2
                )
                * source_params.delta_pow2
                - 3.5 / 1.6 * source_params.chi_l * PI
            )
            / source_params.eta
            * source_params.m1_dimless_pow2
        ) * powers_of_pi.third
        self.alpha_log = (
            -3.5 * PI / 4.8
            + (5.0 * source_params.delta * PI) / (16.0 * source_params.m1_dimless)
            + (5.0 / 16.0 * source_params.chi_l * source_params.delta_pow2)
            - (
                5.0
                / 3.0
                * source_params.chi_l
                * source_params.delta
                * source_params.m1_dimless
            )
            + (2.545 / 1.152 * source_params.chi_l * source_params.m1_dimless_pow2)
            + (
                -2.035
                / 21.504
                * source_params.chi_l
                * source_params.delta
                * source_params.m1_dimless
                + 2.995 / 9.216 * source_params.chi_l * source_params.m1_dimless_pow2
            )
            / source_params.eta
            + (
                5.0
                / 128.0
                * source_params.chi_l
                * source_params.chi_p_pow2
                * source_params.delta
                * source_params.m1_dimless_pow5
                - 3.5
                / 38.4
                * source_params.chi_l
                * source_params.chi_p_pow2
                * source_params.m1_dimless_pow6
            )
            / source_params.eta_pow3
        )

    @ti.func
    def _set_gamma_coefficients(self, source_params: ti.template()):
        # Eq. G9f - G9j in 2004.06503, note gamma = -epsilon
        self.gamma_m3 = -self.alpha_m3
        self.gamma_m2 = -self.alpha_m2
        self.gamma_m1 = (
            -(
                -5.515 / 3.072
                + source_params.eta
                * (
                    -5.15 / 3.84
                    - (1.5 * source_params.delta_pow2)
                    / (25.6 * source_params.m1_dimless_pow2)
                    + (1.75 * source_params.delta) / (2.56 * source_params.m1_dimless)
                )
                + (4.555 * source_params.delta) / (7.168 * source_params.m1_dimless)
            )
            / powers_of_pi.third
        )
        self.gamma_1 = (
            -(
                40.121485 / 9.289728
                + 3.9695 / 8.6016 * source_params.eta
                + 9.55 / 5.76 * source_params.eta_pow2
                - 1.5
                / 102.4
                * source_params.delta_pow3
                * source_params.eta_pow2
                / source_params.m1_dimless_pow3
                + (
                    1.615 / 28.672 * source_params.eta
                    + 3.5 / 25.6 * source_params.eta_pow2
                )
                * source_params.delta_pow2
                / source_params.m1_dimless_pow2
                + (
                    -2.7895885 / 2.1676032
                    + 2.65 / 143.36 * source_params.eta
                    - 2.725 / 3.072 * source_params.eta_pow2
                )
                * source_params.delta
                / source_params.m1_dimless
                - 18.15
                / 2.56
                * source_params.chi_l_pow2
                * source_params.delta
                * source_params.m1_dimless_pow3
                / source_params.eta
                + 16.45
                / 1.92
                * source_params.chi_l_pow2
                * source_params.m1_dimless_pow4
                / source_params.eta
                + 1.5
                / 1.6
                * source_params.chi_l
                * source_params.delta
                * source_params.m1_dimless
                * PI
                / source_params.eta
                + (
                    3.75
                    / 2.56
                    * source_params.chi_l_pow2
                    * source_params.delta_pow2
                    / source_params.eta
                    - 3.5 / 1.6 * source_params.chi_l * PI / source_params.eta
                )
                * source_params.m1_dimless_pow2
            )
            * powers_of_pi.third
        )
        self.gamma_log = -(
            -3.5 / 4.8 * PI
            + 5.0 / 16.0 * source_params.chi_l * source_params.delta_pow2
            - 5.0
            / 3.0
            * source_params.chi_l
            * source_params.delta
            * source_params.m1_dimless
            + 2.545 / 1.152 * source_params.chi_l * source_params.m1_dimless_pow2
            + (
                -2.035
                / 21.504
                * source_params.chi_l
                * source_params.delta
                * source_params.m1_dimless
                + 2.995 / 9.216 * source_params.chi_l * source_params.m1_dimless_pow2
            )
            / source_params.eta
            + (5.0 * source_params.delta * PI) / (16.0 * source_params.m1_dimless)
        )

    @ti.func
    def _set_orbital_angular_momentum_coefficients(self, source_params: ti.template()):
        self.L_0 = 1.0 / powers_of_pi.third
        self.L_1 = 0.0
        self.L_2 = (3.0 / 2.0 + source_params.eta / 6.0) * powers_of_pi.third
        self.L_3 = (
            5.0
            / 6.0
            * (
                source_params.chi_1
                * (-2.0 - 2.0 * source_params.delta + source_params.eta)
                + source_params.chi_2
                * (-2.0 + 2.0 * source_params.delta + source_params.eta)
            )
        ) * powers_of_pi.two_thirds
        self.L_4 = (
            (81.0 + (-57.0 + source_params.eta) * source_params.eta) / 24.0
        ) * powers_of_pi.one
        self.L_5 = (
            -7.0
            / 144.0
            * (
                source_params.chi_1
                * (
                    72.0
                    + source_params.delta * (72.0 - 31.0 * source_params.eta)
                    + source_params.eta * (-121.0 + 2.0 * source_params.eta)
                )
                + source_params.chi_2
                * (
                    72.0
                    + source_params.eta * (-121.0 + 2.0 * source_params.eta)
                    + source_params.delta * (-72.0 + 31.0 * source_params.eta)
                )
            )
        ) * powers_of_pi.four_thirds
        self.L_6 = (
            (
                10935.0
                + source_params.eta
                * (
                    -62001.0
                    + source_params.eta * (1674.0 + 7.0 * source_params.eta)
                    + 2214.0 * powers_of_pi.two
                )
            )
            / 1296.0
        ) * powers_of_pi.five_thirds

    @ti.func
    def update_precession_coefficients(self, source_params: ti.template()):
        self._set_alpha_coefficients(source_params)
        self._set_gamma_coefficients(source_params)
        self._set_orbital_angular_momentum_coefficients(source_params)
        self._set_convention_constants(source_params)

    @ti.func
    def _compute_alpha_core(self, powers_of_Mf: ti.template()) -> float:
        return (
            self.alpha_m3 / powers_of_Mf.one
            + self.alpha_m2 / powers_of_Mf.two_thirds
            + self.alpha_m1 / powers_of_Mf.third
            + self.alpha_1 * powers_of_Mf.third
            + self.alpha_log * powers_of_Mf.log
        )

    @ti.func
    def _compute_alpha(self, powers_of_Mf: ti.template()) -> float:
        return self._compute_alpha_core(powers_of_Mf) + self.alpha_0

    @ti.func
    def _compute_gamma_core(self, powers_of_Mf: ti.template()) -> float:
        return (
            self.gamma_m3 / powers_of_Mf.one
            + self.gamma_m2 / powers_of_Mf.two_thirds
            + self.gamma_m1 / powers_of_Mf.third
            + self.gamma_1 * powers_of_Mf.third
            + self.gamma_log * powers_of_Mf.log
        )

    @ti.func
    def _compute_gamma(self, powers_of_Mf: ti.template()) -> float:
        return self._compute_gamma_core(powers_of_Mf) + self.gamma_0

    @ti.func
    def _compute_cos_beta(
        self,
        source_params: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> float:
        L_norm = self._compute_L_norm_3PN(source_params.eta, powers_of_Mf)
        J_para = L_norm + source_params.S_para
        s = source_params.S_perp / J_para
        return tm.sign(J_para) / tm.sqrt(1.0 + s * s)

    @ti.func
    def _compute_L_norm_3PN(self, eta: float, powers_of_Mf: ti.template()) -> float:
        return eta * (
            self.L_0 / powers_of_Mf.third
            + self.L_1
            + self.L_2 * powers_of_Mf.third
            + self.L_3 * powers_of_Mf.two_thirds
            + self.L_4 * powers_of_Mf.one
            + self.L_5 * powers_of_Mf.four_thirds
            + self.L_6 * powers_of_Mf.five_thirds
        )

    @ti.func
    def compute_euler_angles(
        self, source_params: ti.template(), powers_of_Mf: ti.template()
    ) -> ti.types.vector(3, float):
        return ti.Vector(
            [
                self._compute_alpha(powers_of_Mf),
                self._compute_gamma(powers_of_Mf),
                self._compute_cos_beta(source_params, powers_of_Mf),
            ],
            dt=float,
        )


@sub_struct_from(SourceParameters)
class SourceParametersPrecessionMSA:
    pass


@ti.dataclass
class PrecessionCoefficientsMSA:
    pass


#     """
#     Note the notation in 1703.03967 uses phiz->alpha, zeta->-gamma, thetaL->beta
#     """
#     # Eq. D15 - D20 in 1703.03967
#     Omega_phiz_0_avg: float
#     Omega_phiz_1_avg: float
#     Omega_phiz_2_avg: float
#     Omega_phiz_3_avg: float
#     Omega_phiz_4_avg: float
#     Omega_phiz_5_avg: float

#     # Eq. F6 - F11 in 1703.03967
#     Omega_zeta_0_avg: float
#     Omega_zeta_1_avg: float
#     Omega_zeta_2_avg: float
#     Omega_zeta_3_avg: float
#     Omega_zeta_4_avg: float
#     Omega_zeta_5_avg: float

#     S_avg: float
#     S_avg_pow2: float # Eq. 45 in 1703.03967

#     c1: float # Eq.41 in 1703.03967
#     c1_pow2:float

#     psi_0:float
#     psi_1:float
#     psi_2:float


#     @ti.func
#     def update_MSA_coefficients(
#         self, source_params: ti.template(), source_params: ti.template()
#     ):
#         S_3_pow2, S_minus_pow2, S_plus_pow2 = self._get_roots_spin_evolution_equation()
#         self.S_avg_pow2 = 0.5*(S_plus_pow2 + S_minus_pow2)
#         self.S_avg = tm.sqrt(self.S_avg_pow2)

#         self.c1 = 0.5 * (J0norm*J0norm - L0norm*L0norm - self.SAv2)/ L_0_norm * eta
#         self.c1_pow2 = self.c1 * self.c1

#         # PN coefficients for evolution of orbital frequency
#         # different with equations in 1703.03967
#         a0 =  96.0/5.0 * source_params.eta
#         a2 = -1486.0/35.0 * source_params.eta - 264.0/5.0*source_params.eta_pow2
#         a3 = (384.0*PI/5.0 +
#               self._get_PN_beta(-904./5., -120., source_params)
#                     )* source_params.eta
#         a4 = eta * (34103./945. + eta*(13661./105. + eta*(944./15.)) +
#                     self._get_PN_sigma(-494./5., -1442./5., pPrec)
#                     + self._get_PN_tau(-233./5., -719./5., pPrec)
#                     )
#         a5 = eta * (PI*(-4159./35.) + eta*(PI*(-2268./5.)) +
#                     self._get_PN_beta((-62638./105. + eta*(4636./5.)), (-6472./35. + eta*(3372./5.)), pPrec)
#                     )

#         a0_pow2 = a0 * a0
#         a0_pow3 = a0_pow2 * a0
#         a2_pow2 = a2*a2
#         # Eq. A1 - A5 in 1703.03967
#         g0 = 1.0 / a0
#         g2 = -(a2/a0_pow2)
#         g3 = -(a3/a0_pow2)
#         g4 = -(a4*a0 - a2_pow2) / a0_pow3
#         g5 = -(a5*a0 - 2.0*a3*a2) / a0_pow3

#         Rm = S_plus_pow2 - S_minus_pow2 # Eq. D1
#         Rm_pow2 = Rm * Rm
#         cp = S_plus_pow2* eta_pow2 - c1_pow2 # Eq. D2
#         cm = S_plus_pow2* eta_pow2 - c1_pow2 # Eq. D3
#         cp_cm = tm.abs(cp * cm )
#         sqrt_cp_cm = tm.sqrt(cp_cm)
#         a1 = 0.5 +0.75*eta # Eq. D4
#         a2 = -0.75*chi_eff /eta # Eq. D5

#         D2_Rm = (cp - sqrt_cp_cm)/eta_pow2 # Eq. E3
#         D4_Rm2 =  -0.5*Rm*sqrt_cpcm/eta2 - cp/eta4*(sqrt_cpcm - cp) # Rm^2 * D4, Eq. E4

#         ad = aw / dw
#         hd = hw / dw
#         cd = cw / dw
#         fd = fw / dw
#         gd = gw / dw

#         # Eq. D9 - D14 in 1703.03967
#         Omega_phiz_0 = a1 + ad
#         Omega_phiz_1 = a2 - ad*Seff - ad*hd
#         Omega_phiz_2 = ad*hd*Seff + cd - ad*fd + ad*hd_2
#         Omega_phiz_3 = (adfd - cd - adhd_2)*(Seff + hd) + adfdhd
#         Omega_phiz_4 = (cd + adhd_2 - 2.0*adfd)*(hdD*Seff + hdD_2 - fdD) - adD*fdD*fdD
#         Omega_phiz_5 = (cd - adfd + adhd_2) * fd * (Seff + 2.0*hd) - (cd + adhd_2 - 2.0*adfd) * hd_2 * (Seff + hd) - adfd*fd*hd

#         # Eq. D15 - D20 in 1703.03967
#         self.Omega_phiz_0_avg = 3.0 * g0 * Omega_phiz_0
#         self.Omega_phiz_1_avg = 3.0 * g0 * Omega_phiz_1
#         self.Omega_phiz_2_avg = 3.0 * (g0 * Omega_phiz_2 + g2 * Omega_phiz_0)
#         self.Omega_phiz_3_avg = 3.0 * (g0 * Omega_phiz_3 + g2 * Omega_phiz_1 + g3 * Omega_phiz_0)
#         self.Omega_phiz_4_avg = 3.0 * (g0 * Omega_phiz_4 + g2 * Omega_phiz_2 + g3 * Omega_phiz_1 + g4*Omega_phiz_0)
#         self.Omega_phiz_5_avg = 3.0 * (g0 * Omega_phiz_5 + g2 * Omega_phiz_3 + g3 * Omega_phiz_2 + g4*Omega_phiz_1 + g5*Omega_phiz_0)


#         # Eq. F12 - F17 in 1703.03967
#         Omega_zeta_0 = Omega_phiz_0
#         Omega_zeta_1 = Omega_phiz_1 + c1_over_eta2* Omega_phiz_0
#         Omega_zeta_2 = Omega_phiz_2 + c1_over_eta2* Omega_phiz_1
#         Omega_zeta_3 = Omega_phiz_3 + c1_over_eta2* Omega_phiz_2 + gd
#         Omega_zeta_4 = Omega_phiz_4 + c1_over_eta2* Omega_phiz_3 - gd*chi_eff - gd*hd

#         # Eq. F6 - F11 in 1703.03967
#         self.Omega_zeta_0_avg = -g0*Omega_zeta_0
#         self.Omega_zeta_1_avg = -1.5 * Omega_zeta_1
#         self.Omega_zeta_2_avg = -3.0*(g0 * Omega_zeta_2 + g2*Omega_zeta_0)
#         self.Omega_zeta_3_avg = 3.0*( g0 *  Omega_zeta_3 +  g2* Omega_zeta_1 +  g3* Omega_zeta_0)
#         self.Omega_zeta_4_avg = 3.0*( g0 *  Omega_zeta_4 +  g2* Omega_zeta_2 +  g3* Omega_zeta_1 +  g4* Omega_zeta_0)

#         # Eq. C1 and C2 in 1703.03967
#         self.phi_1 = 3.0 * (2.0 * eta2 * Seff - c_1) / (eta * delta2)
#         self.phi_2 = 0.0
#         # the integration constant
#         self.psi_0 = 0.0


#     @ti.func
#     def _get_PN_beta(a:float, b:float, source_params:ti.template()):
#         """
#         The spin-orbit couplings for post-Newtonian orbital angular momentum
#         TODO: reference??
#         """
#         return ( dotS1L* (a + b* qq) +  dotS2L*(a + b/ qq))

#     @ti.func
#     def _get_PN_sigma(a:float, b:float, source_params:ti.template()):
#         """
#         The spin-spin couplings for post-Newtonian orbital angular momentum
#         TODO: reference??
#         """
#         return ( inveta * (a* dotS1S2 - b* dotS1L* dotS2L))

#     @ti.func
#     def _get_PN_tau(a:float, b:float, source_params:ti.template()):
#         """
#         The spin-spin couplings for post-Newtonian orbital angular momentum
#         TODO: reference??
#         """
#         return (( qq * ( ( S1_norm_2 * a) - b* dotS1L* dotS1L) + (a* S2_norm_2 - b* dotS2L* dotS2L) /  qq) /  eta)

#     @ti.func
#     def _get_roots_spin_evolution_equation(self)->tuple[float, float, float]:
#         """
#         Get roots of Eq. 21 in 1703.03967, returns S_3_pow2, S_minus_pow2, S_plus_pow2.
#         Using trigonometric solution for three real roots of cubic equation, referring https://en.wikipedia.org/wiki/Cubic_equation
#         """
#         B, C, D = self._get_spin_evolution_coefficients()
#         B_pow2 = B * B
#         B_pow3 = B_pow2 * B
#         # p and q are coefficients in the depressed cubic form
#         p = C - B_pow2/3.0
#         q = 2.0/27.0*B_pow3 - B * C / 3.0 + D
#         sqrt_val = tm.sqrt(-p/3.0)
#         acos_arg = 1.5 * q/p/sqrt_val
#         acos_arg = tm.clamp(acos_arg, -1.0, 1.0)
#         theta = tm.acos(acos_arg) / 3.0

#         root_0 = 0.0
#         root_1 = 0.0
#         root_2 = 0.0
#         if (tm.isnan(theta) or
#             tm.isnan(sqrt_val) or
#             dotS1Ln == 1.0 or
#             dotS2Ln == 1.0 or
#             dotS1Ln == -1.0 or
#             dotS2Ln == -1.0 or
#             a_1 == 0.0 or
#             a_2 == 0.0
#             ):
#             root_0 = 0.0
#             root_1 = a_total_pow2
#             # Add a numerical perturbation to prevent azimuthal precession angle from diverging
#             root_2 = root_1 + 1.0e-9
#         else:
#             root_0 =2.0* sqrt_val * tm.cos(theta - 2.0*PI/3.0) - B/3.0
#             root_1 =2.0* sqrt_val * tm.cos(theta - PI/3.0) - B/3.0
#             root_2 =2.0* sqrt_val * tm.cos(theta) - B/3.0
#             # sort the roots
#             if root_0 > root_1: root_0, root_1 = root_1, root_0
#             if root_1 > root_2: root_1, root_2 = root_2, root_1
#             if root_0 > root_1: root_0, root_1 = root_1, root_0

#         return root_0, root_1, root_2

#     @ti.func
#     def _get_spin_evolution_coefficients(self, )->tuple[float, float, float]:
#         """
#         Coefficients B, C, D in Eq. 21 of 1703.03967, given by Eq. B2, B3 and B4
#         """
#         B = ((LNorm2 + S1Norm2)*q + 2.0*LNorm*Seff - 2.0*JNorm2 -
#                   S1Norm2 - S2Norm2 + (LNorm2 + S2Norm2)/q)
#         C = (J2mL2Sq - 2.0*LNorm*Seff*J2mL2 - 2.0*((1.0 - q)/q)*LNorm2*(S1Norm2 - q*S2Norm2) +
#                       4.0*eta*LNorm2*Seff*Seff - 2.0*delta*(S1Norm2 - S2Norm2)*Seff*LNorm +
#                       2.0*((1.0 - q)/q)*(q*S1Norm2 - S2Norm2)*JNorm2)
#         D = (((1.0 - q)/q)*(S2Norm2 - q*S1Norm2)*J2mL2Sq
#                       + deltaSq*(S1Norm2 - S2Norm2)*(S1Norm2 - S2Norm2)*LNorm2/eta
#                       + 2.0*delta*LNorm*Seff*(S1Norm2 - S2Norm2)*J2mL2)
#         return B, C, D


#     @ti.func
#     def _compute_phiz_core(self, J_norm, L_norm, source_params: ti.template(), source_params:ti.template()) -> float:


#         phiz_0 = ((JNorm *   inveta4) * (0.5*c12 - c1*  eta2*invv/6.0 - SAv2*  eta2/3.0 -   eta4*invv2/3.0)
#                           - (c1 * 0.5 *   inveta)*(c12 *   inveta4 - SAv2 *   inveta2)*log1)
#         phiz_1 =  - 0.5 * JNorm *   inveta2 * (c1 +   eta * LNewt) + 0.5*  inveta3*(c12 -   eta2*SAv2)*log1

#         phiz_2 =  -JNorm + SAv*log2 - c1*log1*  inveta

#         phiz_3 =  JNorm*v -   eta*log1 + c1*log2*  invSAv

#         phiz_4 = (0.5*JNorm*invSAv2*v)*(c1 + v*SAv2) - (0.5*invSAv2*invSAv)*(c12 -   eta2*SAv2)*log2

#         # phiz_5 =  (-JNorm*v*( (0.5*c12*invSAv2*invSAv2) - (c1*v*invSAv2/6.0) - v*v/3.0 -   eta2*invSAv2/3.0)
#         #                 + (0.5*c1*invSAv2*invSAv2*invSAv)*(c12 -   eta2*SAv2)*log2)


#         return (
#             alpha_correction
#             + source_params.MSA_alpha_integration_constant
#             + alpha_0 * source_params.Omega_alpha_0
#             + alpha_1 * source_params.Omega_alpha_1
#             + alpha_2 * source_params.Omega_alpha_2
#             + alpha_3 * source_params.Omega_alpha_3
#             + alpha_4 * source_params.Omega_alpha_4
#             # + alpha_5 * source_params.Omega_alpha_5
#         )


#     @ti.func
#     def _compute_phiz(self)->float:
#         ret =  self._compute_phiz_core() + self.alpha_0
#         if tm.isnan(ret):
#             ret = 0.0
#         return ret

#     @ti.func
#     def _compute_zeta_core(
#         self, source_params: ti.template(), source_params: ti.template()
#     ) -> float:
#         """
#         Eq. F5 in 1703.03967
#         """
#         return (
#             source_params.eta
#             * (
#                 source_params.Omega_gamma_0 / powers_of_piMf.one
#                 + source_params.Omega_gamma_1 / powers_of_piMf.two_third
#                 + source_params.Omega_gamma_1 / powers_of_piMf.one_third
#                 + source_params.Omega_gamma_1 / powers_of_piMf.log
#                 + source_params.Omega_gamma_1 * powers_of_piMf.third
#                 + source_params.Omega_gamma_1 * powers_of_piMf.two_third
#             )
#             + source_params.gamma_constant
#         )

#     @ti.func
#     def _compute_zeta(self)->float:
#         ret = self._compute_zeta_core() + self.zeta_0
#         if tm.isnan(ret):
#             ret = 0.0
#         return ret

#     @ti.func
#     def _compute_cos_thetaL(self, J_norm:float, L_norm:float, S_norm:float) -> float:
#         """
#         Eq. 8 in 1703.03967
#         """
#         cos_thetaL = (
#             0.5
#             * (J_norm * J_norm + L_norm * L_norm - S_norm * S_norm)
#             / (L_norm * J_norm)
#         )

#         return tm.clamp(cos_thetaL, -1.0, 1.0)


#     @ti.func
#     def _get_L_norm_3PN(self, L_norm_Newt:float)->float:
#         return L_norm_Newt*(1. + v2*( constants_L[0] + v* constants_L[1] + v2*( constants_L[2] + v* constants_L[3] + v2*( constants_L[4]))))

#     @ti.func
#     def _get_J_norm(self, L_norm:float)->float:
#         """The magnitude of the total angular momentum, Eq. 41 in 1703.03967"""
#         J_norm_pow2 = L_norm*L_norm + 2.0*L_norm*self.c1_over_eta + self.S_avg_pow2
#         return tm.sqrt(J_norm_pow2)

#     @ti.func
#     def _get_S_norm(self, )->float:
#         """
#         The magnitude of the total spin angular momentum, Eq. 23 in 1703.03967
#         """
#         sn = 0.0
#         if ti.abs(S_plus_pow2 - S_minus_pow2) < 1.0e-5:
#             sn = 0.0
#         else:
#             m = (S_plus_pow2 - S_minus_pow2)/(S_plus_pow2 - S_3_pow2)
#             # Eq. 51 in 1703.03967
#             psi = self.psi_0 -0.75*self.g0 * source_params.delta * (
#                 v**(-3)
#                 + self.psi_1 * v**(-2)
#                 + self.psi_2 * v**(-1)
#             )
#             sn, _, _ = jacobi_elliptic_funcs(psi, m)

#         S_norm_pow2 = S_plus_pow2 + (S_minus_pow2 - S_plus_pow2) * sn * sn

#         return tm.sqrt(S_norm_pow2)

#     @ti.func
#     def _get_MSA_corrections(self, L_norm, J_norm)->tuple[float, float]:
#         """
#         MSA corrections to phiz and zeta, Eq. 67 and F19 in 1703.03967
#         """
#         # Eq. B6 - B8 in 1703.03967
#         c0 = JNorm * (
#             0.75*(1.0 - Seff*v) * v2 * (
#                  eta3 + 4.0* eta3*Seff*v
#                   - 2.0* eta*(JNorm2 -  Spl2 + 2.0*( S1_norm_2 -  S2_norm_2)* delta_qq)*v2
#                   - 4.0* eta*Seff*(JNorm2 -  Spl2)*v3
#                   + (JNorm2 -  Spl2)*(JNorm2 -  Spl2)*v4* inveta
#                   )
#                   )

#         c2 = JNorm * (
#              -1.5 *  eta * ( Spl2 -  Smi2)*(1.0 + 2.0*Seff*v - (JNorm2 -  Spl2)*v2* inveta2) * (1.0 - Seff*v)*v4
#              )

#         c4 = JNorm * ( 0.75 *  inveta * ( Spl2 -  Smi2)*( Spl2 -  Smi2)*(1.0 - Seff * v)*v6 )

#         # Eq. B9 - B11 in 1703.03967
#         d0 =-( JNorm2 - (LNorm +  Spl)*(LNorm +  Spl)) * ( (JNorm2 - (LNorm -  Spl)*(LNorm -  Spl)) )
#         d2 =-2.0*( Spl2 -  Smi2)*(JNorm2 + LNorm2 -  Spl2)
#         d4 =-( Spl2 -  Smi2)*( Spl2 -  Smi2)

#         # Eq. B18 and B19 in 1703.03967
#         C1 = -0.5 * (c0/d0 - 2.0*(c0+c2+c4)/nc_num)
#         C2 = (c0*( -2.0*d0*d4 + d2*d2 + d2*d4 ) - c2*d0*( d2 + 2.0*d4 ) + c4*d0*( two_d0 + d2 ))/(2.0 * d0 * sd * (d0 + d2 + d4))

#         # Eq. B14 and B15 in 1703.03967
#         Cphi = C1 + C2
#         Dphi = C1 - C2

#         # Eq. B20 in 1703.03967
#         sd = sqrt( fabs(d2*d2 - 4.0*d0*d4) )

#         # Eq. F20 and F21 in 1703.03967
#         A_thetaL = 0.5 * ( J_norm_pow2 + L_norm_pow2 - S_plus_pow2)/ (JNorm * LNorm)
#         B_thetaL = 0.5 * Spl2mSmi2 / (JNorm * LNorm)

#         # Eq. B16 and B17 in 1703.03967
#         nc = 2.0*(d0 + d2 +d4)/(2*d0 + d2+sd)
#         nd = (2*d0 + d2+sd)/(2.0*d0)

#         psi = self._get_psi()
#         d_psi = self._get_d_psi()

#         if nc == 1.0:
#             Cphi_term = 0.0
#         else:
#             Cphi_term =  fabs( (c4 * d0 * ((2*d0+d2) + sd) - c2 * d0 * ((d2+2.*d4) - sd) - c0 * ((2*d0*d4) - (d2+d4) * (d2 - sd))) / (C2den)) * (sqrt_nc / (nc - 1.) * (atan_psi - atan(sqrt_nc * tan_psi))) / psi_dot
#         if nd == 1.0:
#             Dphi_term = 0.0
#         else:
#             Dphi_term = fabs( (-c4 * d0 * ((2*d0+d2) - sd) + c2 * d0 * ((d2+2.*d4) + sd) - c0 * (-(2*d0*d4) + (d2+d4) * (d2 + sd)))) / (C2den) * (sqrt_nd / (nd - 1.) * (atan_psi - atan(sqrt_nd * tan_psi))) / psi_dot

#         phiz_corr = Cphi_term + Dphi_term
#         zeta_corr = A_thetaL * phiz_corr +  2.0*B_thetaL * d0 * (Cphi_term/(sd-d2) - Dphi_term/(sd+d2))

#         if tm.isnan(phiz_corr):
#             phiz_corr = 0.0
#         if tm.isnan(zeta_corr):
#             zeta_corr = 0.0

#         return phiz_corr, zeta_corr


#     @ti.func
#     def compute_euler_angles(
#         self, source_params: ti.template(), source_params: ti.template()
#     ) -> ti.types.vector(3, float):
#         """
#         Note the notations used in 1703.03967 are different with 2004.06503, where phiz->alpha, zeta->-gamma, thetaL->beta
#         """
#         L_norm_Newt = source_params.eta / v
#         J_norm_Newt = self._get_J_norm(L_norm_Newt)
#         L_norm_3PN = self._get_L_norm_3PN(L_norm_Newt)
#         J_norm_3PN = self._get_J_norm(L_norm_3PN)

#         S_3_pow2, S_minus_pow2, S_plus_pow2 = self._get_roots_spin_evolution_equation(L_norm_Newt, J_norm_Newt)

#         S_norm = self._get_S_norm()
#         phiz_corr, zeta_corr = self._get_MSA_corrections(L_norm_Newt, J_norm_Newt)

#         phiz = self._compute_phiz() + phiz_corr
#         zeta = self._compute_zeta() + zeta_corr
#         cos_thetaL = self._compute_cos_thetaL()

#         return ti.Vector([phiz, zeta, cos_thetaL])


class IMRPhenomXPHM(BaseWaveform):

    def __init__(
        self,
        frequencies,
        reference_frequency=None,
        return_form="polarizations",
        include_tf=True,
        check_parameters=False,
    ):
        super().__init__(
            frequencies, reference_frequency, return_form, include_tf, check_parameters
        )

    def update_waveform(self, parameters):
        pass

    def update_waveform_kernel_NNLO(
        self,
    ):
        pass

    def update_waveform_kernel_MSA(
        self,
    ):
        pass

    def update_waveform_kernel_try_MSA_fallback_NNLO(
        self,
    ):
        pass


@ti.data_oriented
class IMRPhenomXP(BaseWaveform):

    def __init__(
        self,
        frequencies: ti.ScalarField | NDArray,
        reference_frequency: float | None = None,
        return_form: str = "polarizations",  # "amplitude_phase_eulerangles" or "polarizations"
        include_tf: bool = True,
        check_parameters: bool = False,
        parameter_conversion: Callable | None = None,
        precession_model: str = "try_MSA_fallback_NNLO",  # one of "NNLO", "MSA", "try_MSA_fallback_NNLO"
    ):
        super().__init__(
            frequencies,
            reference_frequency,
            return_form,
            include_tf,
            check_parameters,
            parameter_conversion,
        )

        self.phase_coefficients = PhaseCoefficients.field(shape=())
        self.amplitude_coefficients = AmplitudeCoefficients.field(shape=())
        self.pn_coefficients = PostNewtonianCoefficients.field(shape=())

        if precession_model == "NNLO":
            self.source_parameters = SourceParametersPrecessionNNLO.field(shape=())
            self.precession_coefficients = PrecessionCoefficientsNNLO.field(shape=())
            self._update_waveform_kernel = self._update_waveform_kernel_NNLO
        elif precession_model == "MSA":
            self.source_parameters = SourceParametersPrecessionMSA.field(shape=())
            self.precession_coefficients = PrecessionCoefficientsMSA.field(shape=())
            self._update_waveform_kernel = self._update_waveform_kernel_MSA
        elif precession_model == "try_MSA_fallback_NNLO":
            self._source_params_NNLO = SourceParametersPrecessionNNLO.field(shape=())
            self._prec_coeffs_NNLO = PrecessionCoefficientsNNLO.field(shape=())
            self._source_params_MSA = SourceParametersPrecessionMSA.field(shape=())
            self._prec_coeffs_MSA = PrecessionCoefficientsMSA.field(shape=())
            self.source_parameters = self._source_params_MSA
            self.precession_coefficients = self._prec_coeffs_MSA
            self._update_waveform_kernel = (
                self._update_waveform_kernel_try_MSA_fallback_NNLO
            )
        else:
            raise ValueError(
                f"Unrecognized precession model: {precession_model}, please select from NNLO, MSA, or try_MSA_fallback_NNLO"
            )

    def _initialize_waveform_container(self) -> None:
        ret_content = {}
        if self.return_form == "polarizations":
            ret_content.update({"plus": ti_complex, "cross": ti_complex})
        elif self.return_form == "amplitude_phase_eulerangles":
            ret_content.update(
                {
                    "amplitude": float,
                    "phase": float,
                    "euler_angles": ti.types.vector(3, float),
                }
            )
        else:
            raise ValueError(
                f"{self.return_form} is unknown. `return_form` can only be one of `polarizations` and `amplitude_phase_eulerangles`"
            )

        if self.include_tf:
            ret_content.update({"tf": float})

        self.waveform_container = ti.Struct.field(
            ret_content,
            shape=self.frequencies.shape,
        )
        return None

    def update_waveform(self, input_params: dict[str, float]) -> None:
        params = self.parameter_conversion(input_params)
        self._update_waveform_kernel(
            params["mass_1"],
            params["mass_2"],
            params["chi1_x"],
            params["chi1_y"],
            params["chi1_z"],
            params["chi2_x"],
            params["chi2_y"],
            params["chi2_z"],
            params["luminosity_distance"],
            params["inclination"],
            params["reference_phase"],
            self.reference_frequency,
        )

    @ti.kernel
    def _update_waveform_kernel_NNLO(
        self,
        mass_1: float,
        mass_2: float,
        chi1_x: float,
        chi1_y: float,
        chi1_z: float,
        chi2_x: float,
        chi2_y: float,
        chi2_z: float,
        luminosity_distance: float,
        inclination: float,
        reference_phase: float,
        reference_frequency: float,
    ):
        self.source_parameters[None].update_source_parameters(
            mass_1,
            mass_2,
            chi1_x,
            chi1_y,
            chi1_z,
            chi2_x,
            chi2_y,
            chi2_z,
            luminosity_distance,
            inclination,
            reference_phase,
            reference_frequency,
        )
        self.pn_coefficients[None].update_pn_coefficients(self.source_parameters[None])
        self.amplitude_coefficients[None].update_amplitude_coefficients(
            self.pn_coefficients[None], self.source_parameters[None]
        )
        self.phase_coefficients[None].update_phase_coefficients(
            self.pn_coefficients[None], self.source_parameters[None]
        )
        self.precession_coefficients[None].update_precession_coefficients(
            self.source_parameters[None]
        )

        # main loop for building the waveform, auto-parallelized.
        powers_of_Mf = UsefulPowers()
        for idx in self.frequencies:
            Mf = self.source_parameters[None].M_sec * self.frequencies[idx]
            if Mf < PHENOMXAS_HIGH_FREQUENCY_CUT:
                powers_of_Mf.update(Mf)
                amplitude = self.amplitude_coefficients[None].compute_amplitude(
                    self.pn_coefficients[None],
                    self.source_parameters[None],
                    powers_of_Mf,
                )
                amplitude *= self.source_parameters[None].dimension_factor
                phase = self.phase_coefficients[None].compute_phase(
                    self.pn_coefficients[None],
                    self.source_parameters[None],
                    powers_of_Mf,
                )
                euler_angles = self.precession_coefficients[None].compute_euler_angles(
                    self.source_parameters[None], powers_of_Mf
                )

                if ti.static(self.return_form == "amplitude_phase_eulerangles"):
                    self.waveform_container[idx].amplitude = amplitude
                    self.waveform_container[idx].phase = phase
                    self.waveform_container[idx].euler_angles = euler_angles
                if ti.static(self.return_form == "polarizations"):
                    h22_align = amplitude * tm.cexp(ti_complex([0.0, phase]))
                    twist_fac_p, twist_fac_c = self._get_twist_up_factors(euler_angles)
                    self.waveform_container[idx].plus = tm.cmul(twist_fac_p, h22_align)
                    self.waveform_container[idx].cross = tm.cmul(twist_fac_c, h22_align)
                if ti.static(self.include_tf):
                    dphi = self.phase_coefficients[None].compute_d_phase(
                        self.pn_coefficients[None],
                        self.source_parameters[None],
                        powers_of_Mf,
                    )
                    dphi *= self.source_parameters[None].M_sec / PI / 2  # to second
                    self.waveform_container[idx].tf = -dphi
            else:
                if ti.static(self.return_form == "amplitude_phase_eulerangles"):
                    self.waveform_container[idx].amplitude = 0.0
                    self.waveform_container[idx].phase = 0.0
                    self.waveform_container[idx].euler_angles = ti.Vector(
                        [0.0, 0.0, 0.0]
                    )
                if ti.static(self.return_form == "polarizations"):
                    self.waveform_container[idx].plus.fill(0.0)
                    self.waveform_container[idx].cross.fill(0.0)
                if ti.static(self.include_tf):
                    self.waveform_container[idx].tf = 0.0

    def _update_waveform_kernel_MSA(
        self,
        mass_1,
        mass_2,
        chi1_x,
        chi1_y,
        chi1_z,
        chi2_x,
        chi2_y,
        chi2_z,
        luminosity_distance,
        inclination,
        reference_phase,
        reference_frequency,
    ) -> None:
        pass

    def _update_waveform_kernel_try_MSA_fallback_NNLO(
        self,
        mass_1,
        mass_2,
        chi1_x,
        chi1_y,
        chi1_z,
        chi2_x,
        chi2_y,
        chi2_z,
        luminosity_distance,
        inclination,
        reference_phase,
        reference_frequency,
    ) -> None:
        pass

    @ti.func
    def _get_twist_up_factors(
        self,
        euler_angles: ti.template(),
    ) -> tuple[ti_complex, ti_complex]:
        """Eq. 3.5, 3.6 in 2004.06503"""
        alpha = euler_angles.x
        gamma = euler_angles.y
        cos_beta = euler_angles.z

        cos_half_beta_pow2 = ti.abs((1.0 + cos_beta) / 2.0)
        cos_half_beta = tm.sqrt(cos_half_beta_pow2)
        cos_half_beta_pow3 = cos_half_beta * cos_half_beta_pow2
        cos_half_beta_pow4 = cos_half_beta_pow2 * cos_half_beta_pow2

        sin_half_beta_pow2 = ti.abs((1.0 - cos_beta) / 2.0)
        sin_half_beta = tm.sqrt(sin_half_beta_pow2)
        sin_half_beta_pow3 = sin_half_beta * sin_half_beta_pow2
        sin_half_beta_pow4 = sin_half_beta_pow2 * sin_half_beta_pow2

        # d_2_m_2 = [d^2_{-2,2}, d^2_{-1,2}, d^2_{0,2}, d^2_{1,2}, d^2_{2,2}]
        d_2_m_2 = ti.Vector(
            [
                sin_half_beta_pow4,
                2.0 * cos_half_beta * sin_half_beta_pow3,
                tm.sqrt(6.0) * cos_half_beta_pow2 * sin_half_beta_pow2,
                2.0 * cos_half_beta_pow3 * sin_half_beta,
                cos_half_beta_pow4,
            ],
            dt=float,
        )
        # d_2_m_minus2 = [d^2_{-2,-2}, d^2_{-1,-2}, d^2_{0,-2}, d^2_{1,-2}, d^2_{2,-2}]
        d_2_m_minus2 = ti.Vector(
            [
                d_2_m_2[4],
                -d_2_m_2[3],
                d_2_m_2[2],
                -d_2_m_2[1],
                d_2_m_2[0],
            ],
            dt=float,
        )

        # e^{i 2 gamma}
        gamma_term = tm.cexp(ti_complex([0.0, 2.0 * gamma]))

        # e^{i m alpha}
        cexp_i_alpha = tm.cexp(ti_complex([0.0, alpha]))
        cexp_i_2alpha = tm.cmul(cexp_i_alpha, cexp_i_alpha)
        cexp_minus_i_alpha = tm.cconj(cexp_i_alpha)
        cexp_minus_i_2alpha = tm.cconj(cexp_i_2alpha)

        # exp(-i*m*alpha) * d^2_{m,-2} * Y_{2,m}, note Y_{2,m} are all real here
        A_2_m_minus2 = ti.Matrix.rows(
            [
                cexp_i_2alpha
                * d_2_m_minus2[0]
                * self.precession_coefficients[None].spin_minus2_Y2m[0],
                cexp_i_alpha
                * d_2_m_minus2[1]
                * self.precession_coefficients[None].spin_minus2_Y2m[1],
                ti_complex([1.0, 0.0])
                * d_2_m_minus2[2]
                * self.precession_coefficients[None].spin_minus2_Y2m[2],
                cexp_minus_i_alpha
                * d_2_m_minus2[3]
                * self.precession_coefficients[None].spin_minus2_Y2m[3],
                cexp_minus_i_2alpha
                * d_2_m_minus2[4]
                * self.precession_coefficients[None].spin_minus2_Y2m[4],
            ]
        )
        # exp(i*m*alpha) * d^2_{m,2} * Y_conj_{2,m}, note Y_{2,m} are all real here
        A_conj_2_m_2 = ti.Matrix.rows(
            [
                cexp_minus_i_2alpha
                * d_2_m_2[0]
                * self.precession_coefficients[None].spin_minus2_Y2m[0],
                cexp_minus_i_alpha
                * d_2_m_2[1]
                * self.precession_coefficients[None].spin_minus2_Y2m[1],
                ti_complex([1.0, 0.0])
                * d_2_m_2[2]
                * self.precession_coefficients[None].spin_minus2_Y2m[2],
                cexp_i_alpha
                * d_2_m_2[3]
                * self.precession_coefficients[None].spin_minus2_Y2m[3],
                cexp_i_2alpha
                * d_2_m_2[4]
                * self.precession_coefficients[None].spin_minus2_Y2m[4],
            ]
        )
        p_fac = ti.Vector([1.0] * 5, dt=float) @ (A_2_m_minus2 + A_conj_2_m_2)
        c_fac = ti.Vector([1.0] * 5, dt=float) @ (A_2_m_minus2 - A_conj_2_m_2)

        p_fac = 0.5 * tm.cmul(gamma_term, p_fac)
        c_fac = 0.5 * tm.cmul(tm.cmul(gamma_term, c_fac), ti_complex([0.0, 1.0]))

        # additional rotation for the convention of polarization vectors, Appendix D in 2004.06503
        return (
            (
                self.precession_coefficients[None].cos_2zeta * p_fac
                + self.precession_coefficients[None].sin_2zeta * c_fac
            ),
            (
                -self.precession_coefficients[None].sin_2zeta * p_fac
                + self.precession_coefficients[None].cos_2zeta * c_fac
            ),
        )


# # The end of the waveform is defined as 0.3. while if the effective spin is very
# # high, the ringdown of 44 mode is almost cut out at 0.3, so increasing to 0.33.
# f_max = 0.0
# if self.source_parameters[None].chi_eff > 0.99:
#     f_max = 0.33
# else:
#     f_max = PHENOMXAS_HIGH_FREQUENCY_CUT


"""
DEFINE_ISDEFAULT_FUNC(PhenomXPrecVersion, INT4, "PrecVersion", 300)
DEFINE_ISDEFAULT_FUNC(PhenomXReturnCoPrec, INT4, "ReturnCoPrec", 0)
DEFINE_ISDEFAULT_FUNC(PhenomXPExpansionOrder, INT4, "ExpansionOrder", 5)
DEFINE_ISDEFAULT_FUNC(PhenomXPConvention, INT4, "Convention", 1)
DEFINE_ISDEFAULT_FUNC(PhenomXPFinalSpinMod, INT4, "FinalSpinMod", 4)
DEFINE_ISDEFAULT_FUNC(PhenomXPTransPrecessionMethod, INT4, "TransPrecessionMethod", 1)
DEFINE_ISDEFAULT_FUNC(PhenomXPSpinTaylorVersion, String, "SpinTaylorVersion", NULL)
DEFINE_ISDEFAULT_FUNC(PhenomXPSpinTaylorCoarseFactor, INT4, "SpinTaylorCoarseFactor",10);

/* IMRPhenomXPHM */
DEFINE_ISDEFAULT_FUNC(PhenomXPHMMBandVersion, INT4, "MBandPrecVersion", 0)
DEFINE_ISDEFAULT_FUNC(PhenomXPHMThresholdMband, REAL8, "PrecThresholdMband", 0.001)
DEFINE_ISDEFAULT_FUNC(PhenomXPHMUseModes, INT4, "UseModes", 0)
DEFINE_ISDEFAULT_FUNC(PhenomXPHMModesL0Frame, INT4, "ModesL0Frame", 0)
DEFINE_ISDEFAULT_FUNC(PhenomXPHMPrecModes, INT4, "PrecModes", 0)
DEFINE_ISDEFAULT_FUNC(PhenomXPHMTwistPhenomHM, INT4, "TwistPhenomHM", 0)

/* IMRPhenomX_PNR Parameters */
DEFINE_ISDEFAULT_FUNC(PhenomXPNRUseTunedAngles, INT4, "PNRUseTunedAngles", 0)
DEFINE_ISDEFAULT_FUNC(PhenomXPNRUseTunedCoprec, INT4, "PNRUseTunedCoprec", 0)
DEFINE_ISDEFAULT_FUNC(PhenomXPNRUseTunedCoprec33, INT4, "PNRUseTunedCoprec33", 0)
// Option to only be used when actively tuning PNR Coprec relative to XHM wherein the non-precessing final spin is used
DEFINE_ISDEFAULT_FUNC(PhenomXPNRUseInputCoprecDeviations, INT4, "PNRUseInputCoprecDeviations", 0)
// Dev option for forcing 22 phase derivative inspiral values to align with XHM at a low ref frequency
DEFINE_ISDEFAULT_FUNC(PhenomXPNRForceXHMAlignment, INT4, "PNRForceXHMAlignment", 0)
/* Toggle output of XAS phase for debugging purposes */
DEFINE_ISDEFAULT_FUNC(PhenomXOnlyReturnPhase, INT4, "PhenomXOnlyReturnPhase", 0)
DEFINE_ISDEFAULT_FUNC(PhenomXPNRInterpTolerance, REAL8, "PNRInterpTolerance", 0.01)
/* IMRPhenomX_PNR_Asymmetry Parameters */
DEFINE_ISDEFAULT_FUNC(PhenomXAntisymmetricWaveform, INT4, "AntisymmetricWaveform", 0)

"""
