import taichi as ti
import taichi.math as tm
import numpy as np
import os

from ..constants import *
from ..utils import ComplexNumber


'''
TODO:
check _get_polarization_from_amplitude_phase, add shperical harmonic
1. source parameter check, (m1>m2, q<1, q<18, using warning and error rather assert); 
4. using one matrix to compute all phenomenology coefficients
5. add loop config for waveform_kernel
'''

@ti.dataclass
class UsefulPowers:
    third: ti.f64
    two_thirds: ti.f64
    one: ti.f64
    four_thirds: ti.f64
    five_thirds: ti.f64
    two: ti.f64
    seven_thirds: ti.f64
    eight_thirds: ti.f64
    three: ti.f64
    four: ti.f64
    fourth: ti.f64
    three_fourths: ti.f64
    seven_sixths: ti.f64
    log: ti.f64

    @ti.pyfunc
    def updating(self, number: ti.f64):
        self.third = number**(1/3)
        self.two_thirds = self.third * self.third
        self.one = number
        self.four_thirds = number * self.third
        self.five_thirds = number * self.two_thirds
        self.two = number * number
        self.seven_thirds = self.two * self.third
        self.eight_thirds = self.two * self.two_thirds
        self.three = self.two * number
        self.four = self.three * number
        self.fourth = number**(1/4)
        self.three_fourths = self.fourth**3
        self.seven_sixths = ti.sqrt(self.seven_thirds)
        self.log = ti.log(number)


QNMgrid_a     = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data/QNMData_a.txt'))
QNMgrid_fring = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data/QNMData_fring.txt'))
QNMgrid_fdamp = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data/QNMData_fdamp.txt'))
AMPLITUDE_INSPIRAL_fJoin = 0.014
PHASE_INSPIRAL_fJoin = 0.018
FREQUENCY_CUT = 0.2

useful_powers_pi = UsefulPowers()
useful_powers_pi.updating(PI)

powers_phase_intermediate_f_join = UsefulPowers()
powers_phase_intermediate_f_join.updating(PHASE_INSPIRAL_fJoin)

powers_amplitude_intermediate_f_join = UsefulPowers()
powers_amplitude_intermediate_f_join.updating(AMPLITUDE_INSPIRAL_fJoin)


@ti.func
def _solve_delta_i(f1, f2, f3, v1, v2, v3, d1, d2):
    '''
    solving linear equation system to get delta_i for intermediate amplitude
    '''
    f1_2 = f1 * f1
    f1_3 = f1_2 * f1
    f1_4 = f1_3 * f1

    f2_2 = f2 * f2
    f2_3 = f2_2 * f2
    f2_4 = f2_3 * f2

    f3_2 = f3 * f3
    f3_3 = f3_2 * f3
    f3_4 = f3_3 * f3
    Ab =  ti.Matrix([[1.0,  f1, f1_2,   f1_3,   f1_4, v1],
                     [1.0,  f2, f2_2,   f2_3,   f2_4, v2],
                     [1.0,  f3, f3_2,   f3_3,   f3_4, v3],
                     [0.0, 1.0, 2*f1, 3*f1_2, 4*f1_3, d1],
                     [0.0, 1.0, 2*f3, 3*f3_2, 4*f3_3, d2]], dt=ti.f64)

    return _gauss_elimination_5x5(Ab)


@ti.func
def _gauss_elimination_5x5(Ab):
    for i in ti.static(range(5)):
        for j in ti.static(range(i + 1, 5)):
            scale = Ab[j, i] / Ab[i, i]
            Ab[j, i] = 0.0
            for k in ti.static(range(i + 1, 6)):
                Ab[j, k] -= Ab[i, k] * scale
    # Back substitution
    x = ti.Vector.zero(ti.f64, 5)
    for i in ti.static(range(4, -1, -1)):
        x[i] = Ab[i, 5]
        for k in ti.static(range(i + 1, 5)):
            x[i] -= Ab[i, k] * x[k]
        x[i] = x[i] / Ab[i, i]
    return x


# Amplitude ansatz
@ti.func
def _amplitude_inspiral_ansatz(powers_of_Mf, amplitude_coefficients, pn_prefactors):
    '''
    Eq.30 in arXiv:1508.07253, without amp0
    '''
    return (1.0 + 
            pn_prefactors.prefactor_A_2 * powers_of_Mf.two_thirds + 
            pn_prefactors.prefactor_A_3 * powers_of_Mf.one + 
            pn_prefactors.prefactor_A_4 * powers_of_Mf.four_thirds +
            pn_prefactors.prefactor_A_5 * powers_of_Mf.five_thirds +
            pn_prefactors.prefactor_A_6 * powers_of_Mf.two +
            amplitude_coefficients.rho_1 * powers_of_Mf.seven_thirds + 
            amplitude_coefficients.rho_2 * powers_of_Mf.eight_thirds +
            amplitude_coefficients.rho_3 * powers_of_Mf.three
            )

@ti.func
def _d_amplitude_inspiral_ansatz(powers_of_Mf, amplitude_coefficients, pn_prefactors):
    '''
    without amp0
    '''
    return (2.0 * pn_prefactors.prefactor_A_2 / powers_of_Mf.third + 
            3.0 * pn_prefactors.prefactor_A_3 + 
            4.0 * pn_prefactors.prefactor_A_4 * powers_of_Mf.third +
            5.0 * pn_prefactors.prefactor_A_5 * powers_of_Mf.two_thirds +
            6.0 * pn_prefactors.prefactor_A_6 * powers_of_Mf.one +
            7.0 * amplitude_coefficients.rho_1 * powers_of_Mf.four_thirds + 
            8.0 * amplitude_coefficients.rho_2 * powers_of_Mf.five_thirds +
            9.0 * amplitude_coefficients.rho_3 * powers_of_Mf.two
            ) / 3.0

@ti.func
def _amplitude_intermediate_ansatz(powers_of_Mf, amplitude_coefficients):
    '''
    without amp0
    '''
    return (amplitude_coefficients.delta_0 +
            amplitude_coefficients.delta_1 * powers_of_Mf.one + 
            amplitude_coefficients.delta_2 * powers_of_Mf.two + 
            amplitude_coefficients.delta_3 * powers_of_Mf.three + 
            amplitude_coefficients.delta_4 * powers_of_Mf.four
            )

@ti.func
def _amplitude_merge_ringdown_ansatz(powers_of_Mf, amplitude_coefficients, f_ring, f_damp):
    '''
    without amp0
    '''
    f_minus_fring = powers_of_Mf.one - f_ring
    fdamp_gamma3 = amplitude_coefficients.gamma_3 * f_damp
    return (amplitude_coefficients.gamma_1 * fdamp_gamma3 / 
            (f_minus_fring**2 + fdamp_gamma3**2) *
            tm.exp(-f_minus_fring * amplitude_coefficients.gamma_2 / fdamp_gamma3)
            )

@ti.func
def _d_amplitude_merge_ringdown_ansatz(powers_of_Mf, amplitude_coefficients, f_ring, f_damp):
    '''
    without amp0
    '''
    fdamp_gamma3 = amplitude_coefficients.gamma_3 * f_damp
    pow2_fdamp_gamma3 = fdamp_gamma3 * fdamp_gamma3
    f_minus_fring = powers_of_Mf.one - f_ring
    exp_factor = tm.exp(-f_minus_fring * amplitude_coefficients.gamma_2 / fdamp_gamma3)
    pow2_plus_pow2 = f_minus_fring**2 + pow2_fdamp_gamma3
    return (exp_factor / pow2_plus_pow2 * (-2*f_damp*amplitude_coefficients.gamma_1*amplitude_coefficients.gamma_3*f_minus_fring/pow2_plus_pow2 - 
                                           amplitude_coefficients.gamma_1*amplitude_coefficients.gamma_2)
            )

# Phase ansatz
@ti.func
def _phase_inspiral_ansatz(powers_of_Mf, phase_coefficients, pn_prefactors):
    '''
    without 1/eta
    '''
    return (3.0/128.0 * (pn_prefactors.prefactor_varphi_0 / powers_of_Mf.five_thirds + 
                         pn_prefactors.prefactor_varphi_1 / powers_of_Mf.four_thirds + 
                         pn_prefactors.prefactor_varphi_2 / powers_of_Mf.one + 
                         pn_prefactors.prefactor_varphi_3 / powers_of_Mf.two_thirds + 
                         pn_prefactors.prefactor_varphi_4 / powers_of_Mf.third + 
                         pn_prefactors.prefactor_varphi_5 + 
                         pn_prefactors.prefactor_varphi_5l * (powers_of_Mf.log + useful_powers_pi.log) + 
                         pn_prefactors.prefactor_varphi_6 * powers_of_Mf.third + 
                         pn_prefactors.prefactor_varphi_6l * powers_of_Mf.third * (powers_of_Mf.log + useful_powers_pi.log) + 
                         pn_prefactors.prefactor_varphi_7 * powers_of_Mf.two_thirds
                        ) + 
            (phase_coefficients.sigma_1 * powers_of_Mf.one + 
             0.75 * phase_coefficients.sigma_2 * powers_of_Mf.four_thirds + 
             0.6 * phase_coefficients.sigma_3 * powers_of_Mf.five_thirds + 
             0.5 * phase_coefficients.sigma_4 * powers_of_Mf.two
             )
            )

@ti.func
def _d_phase_inspiral_ansatz(powers_of_Mf, phase_coefficients, pn_prefactors):
    '''
    without 1/eta
    '''
    return (3.0/128.0 * (-5.0 * pn_prefactors.prefactor_varphi_0 / powers_of_Mf.eight_thirds - 
                         4.0 * pn_prefactors.prefactor_varphi_1 / powers_of_Mf.seven_thirds - 
                         3.0 * pn_prefactors.prefactor_varphi_2 / powers_of_Mf.two - 
                         2.0 * pn_prefactors.prefactor_varphi_3 / powers_of_Mf.five_thirds - 
                         1.0 * pn_prefactors.prefactor_varphi_4 / powers_of_Mf.four_thirds + 
                         3 * pn_prefactors.prefactor_varphi_5l / powers_of_Mf.one+ 
                         pn_prefactors.prefactor_varphi_6 / powers_of_Mf.two_thirds + 
                         pn_prefactors.prefactor_varphi_6l / powers_of_Mf.two_thirds * (3.0 + powers_of_Mf.log + useful_powers_pi.log) + 
                         2.0 * pn_prefactors.prefactor_varphi_7 / powers_of_Mf.third
                        ) / 3.0 + 
            (phase_coefficients.sigma_1 + 
             phase_coefficients.sigma_2 * powers_of_Mf.third + 
             phase_coefficients.sigma_3 * powers_of_Mf.two_thirds + 
             phase_coefficients.sigma_4 * powers_of_Mf.one
             )
            )

@ti.func
def _phase_intermediate_ansatz(powers_of_Mf, phase_coefficients):
    '''
    without 1/eta
    '''
    return (phase_coefficients.beta_1 * powers_of_Mf.one + 
            phase_coefficients.beta_2 * powers_of_Mf.log -
            phase_coefficients.beta_3 / 3.0 / powers_of_Mf.three
            )

@ti.func
def _d_phase_intermediate_ansatz(powers_of_Mf, phase_coefficients):
    '''
    without 1/eta
    '''
    return (phase_coefficients.beta_1 + 
            phase_coefficients.beta_2 / powers_of_Mf.one +
            phase_coefficients.beta_3 / powers_of_Mf.four
            )

@ti.func
def _phase_merge_ringdown_ansatz(powers_of_Mf, phase_coefficients, f_ring, f_damp):
    '''
    without 1/eta
    '''
    return (phase_coefficients.alpha_1 * powers_of_Mf.one - 
            phase_coefficients.alpha_2 / powers_of_Mf.one +
            4.0/3.0 * phase_coefficients.alpha_3 * powers_of_Mf.three_fourths + 
            # note that tm.atan2 return the value in [-pi, pi], make sure f_damp > 0
            phase_coefficients.alpha_4 * tm.atan2((powers_of_Mf.one - phase_coefficients.alpha_5*f_ring), f_damp)
            )

@ti.func
def _d_phase_merge_ringdown_ansatz(powers_of_Mf, phase_coefficients, f_ring, f_damp):
    '''
    without 1/eta
    '''
    return (phase_coefficients.alpha_1 + 
            phase_coefficients.alpha_2 / powers_of_Mf.two +
            phase_coefficients.alpha_3 / powers_of_Mf.fourth +
            phase_coefficients.alpha_4*f_damp / (f_damp**2 + (powers_of_Mf.one - phase_coefficients.alpha_5*f_ring)**2)
            )


@ti.dataclass
class SourceParameters:
    # passed in parameters
    M: ti.f64       # total mass
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
    M_sec: ti.f64   # total mass in second
    eta: ti.f64     # symmetric_mass_ratio
    eta2: ti.f64    # eta^2
    eta3: ti.f64
    delta: ti.f64
    chi_s: ti.f64
    chi_a: ti.f64
    chi_s2: ti.f64
    chi_a2: ti.f64
    chi_PN: ti.f64
    xi: ti.f64
    # derived parameters
    final_spin: ti.f64
    E_rad: ti.f64
    f_ring: ti.f64
    f_damp: ti.f64

    def generate_all_source_parameters(self, parameters):
        # total 11 parameters: m1, m1, chi1, chi2, iota, psi, tc, phi0, dL, lon, lat
        # 3 only used in response function: psi, lon, lat
        # mass is in the unit of solar mass
        # dL is in the unit of Mpc
        self.M = parameters['total_mass']
        self.q = parameters['mass_ratio']
        self.chi_1 = parameters['chi_1']
        self.chi_2 = parameters['chi_2']
        self.dL_Mpc = parameters['luminosity_distance']
        self.iota = parameters['inclination']
        self.phase_ref = parameters['reference_phase']
        self.tc = parameters['coalescence_time']
        # base parameters
        self.mass_1 = self.M / (1+self.q)
        self.mass_2 = self.M - self.mass_1
        self.dL_SI = self.dL_Mpc * 1e6 * PC_SI
        self.M_sec = self.M * MTSUN_SI
        self.eta = self.mass_1 * self.mass_2 / (self.M * self.M)
        self.eta2 = self.eta * self.eta
        self.eta3 = self.eta2 * self.eta
        self.delta = (self.mass_1 - self.mass_2)/self.M     # make sure self.mass_1 > self.mass_2
        self.chi_s = (self.chi_1 + self.chi_2) * 0.5
        self.chi_a = (self.chi_1 - self.chi_2) * 0.5
        self.chi_s2 = self.chi_s * self.chi_s
        self.chi_a2 = self.chi_a * self.chi_a
        self.chi_PN = (self.mass_1*self.chi_1 + self.mass_2*self.chi_2)/self.M - 38.0/113.0*self.eta*(self.chi_1+self.chi_2)
        self.xi = self.chi_PN - 1.0
        # final spin (FinalSpin0815, Eq. (3.6) in arXiv:1508.07250)
        S = (self.mass_1*self.mass_1*self.chi_1 + self.mass_2*self.mass_2*self.chi_2)/(self.M * self.M)
        self.final_spin = S + self.eta * (3.4641016151377544 - 
                                          4.399247300629289*self.eta +
                                          9.397292189321194*self.eta2 - 
                                          13.180949901606242*self.eta3 +
                                          S*((-0.0850917821418767 - 5.837029316602263*self.eta) +
                                             (0.1014665242971878 - 2.0967746996832157*self.eta)*S +
                                             (-1.3546806617824356 + 4.108962025369336*self.eta)*S**2 +
                                             (-0.8676969352555539 + 2.064046835273906*self.eta)*S**3
                                             )
                                         )
        # total radiated energy (EradRational0815, Eq. (3.7) and (3.8) in arXiv:1508.07250)
        S_hat = S/(1.0 - 2*self.eta)
        self.E_rad = (self.eta * (0.055974469826360077 + 
                                  0.5809510763115132 * self.eta - 
                                  0.9606726679372312 * self.eta2 + 
                                  3.352411249771192 * self.eta3
                                 ) *
                      (1.0 + (-0.0030302335878845507 - 2.0066110851351073*self.eta + 7.7050567802399215*self.eta2)*S_hat) /
                      (1.0 + (-0.6714403054720589 - 1.4756929437702908*self.eta + 7.304676214885011*self.eta2)*S_hat)
        )
        # interpolation parameters
        self.f_ring = np.interp(self.final_spin, QNMgrid_a, QNMgrid_fring) / (1.0 - self.E_rad)
        self.f_damp = np.interp(self.final_spin, QNMgrid_a, QNMgrid_fdamp) / (1.0 - self.E_rad)



# PN expansion coefficients
@ti.dataclass
class PostNewtonianPrefactors:
    # 3.5 PN phase
    prefactor_varphi_0: ti.f64
    prefactor_varphi_1: ti.f64
    prefactor_varphi_2: ti.f64
    prefactor_varphi_3: ti.f64
    prefactor_varphi_4: ti.f64
    prefactor_varphi_5: ti.f64
    prefactor_varphi_5l: ti.f64
    prefactor_varphi_6: ti.f64
    prefactor_varphi_6l: ti.f64
    prefactor_varphi_7: ti.f64
    # 3 PN amplitude
    prefactor_A_0: ti.f64
    prefactor_A_1: ti.f64
    prefactor_A_2: ti.f64
    prefactor_A_3: ti.f64
    prefactor_A_4: ti.f64
    prefactor_A_5: ti.f64
    prefactor_A_6: ti.f64

    @ti.func
    def compute_PN_prefactors(self, source_params):
        '''
        Using Eq.B6 - B13 and Eq.B14 - B19 in arXiv:1508.07253
        3PN spin-spin term not included
        '''
        # Phase
        self.prefactor_varphi_0 = 1.0 / useful_powers_pi.five_thirds
        self.prefactor_varphi_1 = 0.0 / useful_powers_pi.four_thirds
        self.prefactor_varphi_2 = (37.15/7.56 + 55.0/9.0*source_params.eta) / useful_powers_pi.one
        self.prefactor_varphi_3 = (-16.0*PI +
                                   (113.0/3.0*source_params.delta*source_params.chi_a) + 
                                   (113.0/3.0 - 76.0/3.0*source_params.eta)*source_params.chi_s 
                                  ) / useful_powers_pi.two_thirds
        self.prefactor_varphi_4 = (152.93365/5.08032 + 271.45/5.04*source_params.eta + 308.5/7.2*source_params.eta2 +
                                   (-405.0/8.0 + 200.0*source_params.eta) * source_params.chi_a2 - 
                                   405.0/4.0 * source_params.delta * source_params.chi_a * source_params.chi_s + 
                                   (-405.0/8.0 + 5.0/2.0*source_params.eta) * source_params.chi_s2
                                  ) / useful_powers_pi.third
        self.prefactor_varphi_5 = ((386.45/7.56 - 65.0/9.0*source_params.eta) * PI +
                                   (-732.985/2.268 - 140.0/9.0*source_params.eta) * source_params.delta * source_params.chi_a +
                                   (-732.985/2.268 + 2426.0/8.1*source_params.eta + 340.0/9.0*source_params.eta2) * source_params.chi_s
                                  )
        self.prefactor_varphi_5l= self.prefactor_varphi_5
        self.prefactor_varphi_6 = ((11583.231236531/4.694215680 - 640.0/3.0*PI*PI - 684.8/2.1*EULER_GAMMA + 
                                        (-15737.765635/3.048192 + 225.5/1.2*PI*PI)*source_params.eta + 
                                        76.055/1.728*source_params.eta2 - 
                                        127.825/1.296*source_params.eta3 - 
                                        tm.log(4.0)*684.8/2.1) +
                                   2270.0/3.0*PI*source_params.delta*source_params.chi_a + 
                                   (2270.0/3.0 - 520.0*source_params.eta)*PI*source_params.chi_s
                                  ) * useful_powers_pi.third
        self.prefactor_varphi_6l=-684.8/6.3 * useful_powers_pi.third
        self.prefactor_varphi_7 = ((770.96675/2.54016 + 378.515/1.512*source_params.eta - 740.45/7.56*source_params.eta2)*PI + 
                                   (-25150.083775/3.048192 + 26804.935/6.048*source_params.eta - 198.5/4.8*source_params.eta2) * source_params.delta * source_params.chi_a + 
                                   (-25150.083775/3.048192 + 105666.55595/7.62048*source_params.eta - 1042.165/3.024*source_params.eta2 + 534.5/3.6*source_params.eta3) * source_params.chi_s
                                  ) * useful_powers_pi.two_thirds
        # Amplitude
        self.prefactor_A_0 = 1.0 
        self.prefactor_A_1 = 0.0
        self.prefactor_A_2 = (-3.23/2.24 + 4.51/1.68*source_params.eta) * useful_powers_pi.two_thirds
        self.prefactor_A_3 = (27.0/8.0 * source_params.delta * source_params.chi_a + 
                              (27.0/8.0 - 11.0/6.0*source_params.eta) * source_params.chi_s
                              ) * useful_powers_pi.one
        self.prefactor_A_4 = (-27.312085/8.128512 -
                              19.75055/3.38688*source_params.eta + 
                              10.5271/2.4192*source_params.eta2 +
                              (-8.1/3.2 + 8.0*source_params.eta) * source_params.chi_a2 -
                              8.1/1.6 * source_params.delta * source_params.chi_a * source_params.chi_s +
                              (-8.1/3.2 + 17.0/8.0*source_params.eta) * source_params.chi_s2
                              ) * useful_powers_pi.four_thirds
        self.prefactor_A_5 = (-8.5/6.4*PI+
                              8.5/1.6*PI*source_params.eta+
                              (28.5197/1.6128 - 1.579/4.032*source_params.eta) * source_params.delta * source_params.chi_a +
                              (28.5197/1.6128 - 153.17/6.72*source_params.eta - 2.227/1.008*source_params.eta2) * source_params.chi_s
                              ) * useful_powers_pi.five_thirds
        self.prefactor_A_6 = (-177.520268561/8.583708672 +
                              (545.384828789/5.007163392 - 20.5/4.8*useful_powers_pi.two) * source_params.eta -
                              32.48849057/1.78827264 * source_params.eta2 +
                              34.473079/6.386688 * source_params.eta3 + 
                              (3.1/1.2*PI - 7.0/3.0*PI*source_params.eta) * source_params.chi_s +
                              (161.4569/6.4512 - 187.3643/1.6128*source_params.eta + 216.7/4.2*source_params.eta2) * source_params.chi_a2 +
                              (161.4569/6.4512 - 61.391/1.344*source_params.eta + 57.451/4.032*source_params.eta2) * source_params.chi_s2 +
                              (3.1/1.2*PI + (161.4569/3.2256 - 165.961/2.688*source_params.eta)*source_params.chi_s) * source_params.delta * source_params.chi_a
                              ) * useful_powers_pi.two


# Phase coefficients
@ti.dataclass
class PhaseCoefficients:
    # Inspiral (Region I)
    sigma_1: ti.f64
    sigma_2: ti.f64
    sigma_3: ti.f64
    sigma_4: ti.f64
    # Intermediate (Region IIa)
    beta_1: ti.f64
    beta_2: ti.f64
    beta_3: ti.f64
    # Merge-ringdown (Region IIb)
    alpha_1: ti.f64
    alpha_2: ti.f64
    alpha_3: ti.f64
    alpha_4: ti.f64
    alpha_5: ti.f64
    # connection coefficients
    C1_intermediate: ti.f64
    C2_intermediate: ti.f64
    C1_merge_ringdown: ti.f64
    C2_merge_ringdown: ti.f64
    phase_merge_ringdown_f_join: ti.f64

    @ti.func
    def compute_phase_coefficients(self, source_params, pn_prefactors):
        '''
        Compute phase coefficients in Eq. 28, 16, 14 of arXiv:1508.07253
        '''
        # phenomenological fitting coefficients
        # Inspiral (Region I)
        self.sigma_1 = (2096.551999295543 + 
                   1463.7493168261553*source_params.eta + 
                   (1312.5493286098522 + 18307.330017082117*source_params.eta - 43534.1440746107*source_params.eta2 + 
                       (-833.2889543511114 + 32047.31997183187*source_params.eta - 108609.45037520859*source_params.eta2) * source_params.xi + 
                       (452.25136398112204 + 8353.439546391714*source_params.eta - 44531.3250037322*source_params.eta2) * source_params.xi * source_params.xi
                   ) * source_params.xi
                   )
        self.sigma_2 = (-10114.056472621156 - 
                   44631.01109458185*source_params.eta + 
                   (-6541.308761668722 - 266959.23419307504*source_params.eta + 686328.3229317984*source_params.eta2 + 
                       (3405.6372187679685 - 437507.7208209015*source_params.eta + 1.6318171307344697e6*source_params.eta2) * source_params.xi + 
                       (-7462.648563007646 - 114585.25177153319*source_params.eta + 674402.4689098676*source_params.eta2) * source_params.xi * source_params.xi
                   ) * source_params.xi
                   )
        self.sigma_3 = (22933.658273436497 + 
                   230960.00814979506*source_params.eta + 
                   (14961.083974183695 + 1.1940181342318142e6*source_params.eta - 3.1042239693052764e6*source_params.eta2 + 
                       (-3038.166617199259 + 1.8720322849093592e6*source_params.eta - 7.309145012085539e6*source_params.eta2) * source_params.xi + 
                       (42738.22871475411 + 467502.018616601*source_params.eta - 3.064853498512499e6*source_params.eta2) * source_params.xi * source_params.xi
                   ) * source_params.xi
                   )
        self.sigma_4 = (-14621.71522218357 - 
                   377812.8579387104*source_params.eta + 
                   (-9608.682631509726 - 1.7108925257214056e6*source_params.eta + 4.332924601416521e6*source_params.eta2 + 
                       (-22366.683262266528 - 2.5019716386377467e6*source_params.eta + 1.0274495902259542e7*source_params.eta2) * source_params.xi + 
                       (-85360.30079034246 - 570025.3441737515*source_params.eta + 4.396844346849777e6*source_params.eta2) * source_params.xi * source_params.xi
                   ) * source_params.xi
                   )
        # Intermediate (Region IIa)
        self.beta_1 = (97.89747327985583 - 
                  42.659730877489224*source_params.eta + 
                  (153.48421037904913 - 1417.0620760768954*source_params.eta + 2752.8614143665027*source_params.eta2 + 
                       (138.7406469558649 - 1433.6585075135881*source_params.eta + 2857.7418952430758*source_params.eta2) * source_params.xi + 
                       (41.025109467376126 - 423.680737974639*source_params.eta + 850.3594335657173*source_params.eta2) * source_params.xi * source_params.xi
                   ) * source_params.xi
                   )
        self.beta_2 = (-3.282701958759534 - 
                  9.051384468245866*source_params.eta + 
                  (-12.415449742258042 + 55.4716447709787*source_params.eta - 106.05109938966335*source_params.eta2 + 
                       (-11.953044553690658 + 76.80704618365418*source_params.eta - 155.33172948098394*source_params.eta2) * source_params.xi + 
                       (-3.4129261592393263 + 25.572377569952536*source_params.eta - 54.408036707740465*source_params.eta2) * source_params.xi * source_params.xi
                   ) * source_params.xi
                   )
        self.beta_3 = (-0.000025156429818799565 + 
                  0.000019750256942201327*source_params.eta + 
                  (-0.000018370671469295915 + 0.000021886317041311973*source_params.eta + 0.00008250240316860033*source_params.eta2 + 
                       (7.157371250566708e-6 - 0.000055780000112270685*source_params.eta + 0.00019142082884072178*source_params.eta2) * source_params.xi + 
                       (5.447166261464217e-6 - 0.00003220610095021982*source_params.eta + 0.00007974016714984341*source_params.eta2) * source_params.xi * source_params.xi
                   ) * source_params.xi
                   )
        # Merge-ringdown (Region IIb)
        self.alpha_1 = (43.31514709695348 + 
                   638.6332679188081*source_params.eta + 
                   (-32.85768747216059 + 2415.8938269370315*source_params.eta - 5766.875169379177*source_params.eta2 + 
                       (-61.85459307173841 + 2953.967762459948*source_params.eta - 8986.29057591497*source_params.eta2) * source_params.xi + 
                       (-21.571435779762044 + 981.2158224673428*source_params.eta - 3239.5664895930286*source_params.eta2) * source_params.xi * source_params.xi
                   ) * source_params.xi
                   )
        self.alpha_2 = (-0.07020209449091723 - 
                   0.16269798450687084*source_params.eta + 
                   (-0.1872514685185499 + 1.138313650449945*source_params.eta - 2.8334196304430046*source_params.eta2 + 
                       (-0.17137955686840617 + 1.7197549338119527*source_params.eta - 4.539717148261272*source_params.eta2) * source_params.xi + 
                       (-0.049983437357548705 + 0.6062072055948309*source_params.eta - 1.682769616644546*source_params.eta2) * source_params.xi * source_params.xi
                   ) * source_params.xi
                   )
        self.alpha_3 = (9.5988072383479 - 
                   397.05438595557433*source_params.eta + 
                   (16.202126189517813 - 1574.8286986717037*source_params.eta + 3600.3410843831093*source_params.eta2 + 
                       (27.092429659075467 - 1786.482357315139*source_params.eta + 5152.919378666511*source_params.eta2) * source_params.xi + 
                       (11.175710130033895 - 577.7999423177481*source_params.eta + 1808.730762932043*source_params.eta2) * source_params.xi * source_params.xi
                   ) * source_params.xi
                   )
        self.alpha_4 = (-0.02989487384493607 + 
                   1.4022106448583738*source_params.eta + 
                   (-0.07356049468633846 + 0.8337006542278661*source_params.eta + 0.2240008282397391*source_params.eta2 + 
                       (-0.055202870001177226 + 0.5667186343606578*source_params.eta + 0.7186931973380503*source_params.eta2) * source_params.xi + 
                       (-0.015507437354325743 + 0.15750322779277187*source_params.eta + 0.21076815715176228*source_params.eta2) * source_params.xi * source_params.xi
                   ) * source_params.xi
                   )
        self.alpha_5 = (0.9974408278363099 - 
                   0.007884449714907203*source_params.eta + 
                   (-0.059046901195591035 + 1.3958712396764088*source_params.eta - 4.516631601676276*source_params.eta2 + 
                       (-0.05585343136869692 + 1.7516580039343603*source_params.eta - 5.990208965347804*source_params.eta2) * source_params.xi + 
                       (-0.017945336522161195 + 0.5965097794825992*source_params.eta - 2.0608879367971804*source_params.eta2) * source_params.xi * source_params.xi
                   ) * source_params.xi
                   )
        # compute connection coefficients
        # transition between inspiral (region I) and intermediate (region IIa)
        self.C2_intermediate = _d_phase_inspiral_ansatz(powers_phase_intermediate_f_join, self, pn_prefactors) - _d_phase_intermediate_ansatz(powers_phase_intermediate_f_join, self)
        self.C1_intermediate = _phase_inspiral_ansatz(powers_phase_intermediate_f_join, self, pn_prefactors) - _phase_intermediate_ansatz(powers_phase_intermediate_f_join, self) - self.C2_intermediate * PHASE_INSPIRAL_fJoin
        # transition between intermediate (region IIa) and merge_ringdown (region IIb)
        # note that incorporating C2_intermediate and C1_intermediate in intermediate part
        phase_MRD_f_join = 0.5 * source_params.f_ring
        powers_phase_MRD_f_join = UsefulPowers()
        powers_phase_MRD_f_join.updating(phase_MRD_f_join)
        self.C2_merge_ringdown = (_d_phase_intermediate_ansatz(powers_phase_MRD_f_join, self) + self.C2_intermediate) - _d_phase_merge_ringdown_ansatz(powers_phase_MRD_f_join, self, source_params.f_ring, source_params.f_damp)
        self.C1_merge_ringdown = (_phase_intermediate_ansatz(powers_phase_MRD_f_join, self) + self.C1_intermediate + self.C2_intermediate*phase_MRD_f_join) - _phase_merge_ringdown_ansatz(powers_phase_MRD_f_join, self, source_params.f_ring, source_params.f_damp) - self.C2_merge_ringdown*phase_MRD_f_join
        self.phase_merge_ringdown_f_join = phase_MRD_f_join

# Amplitude coefficients
@ti.dataclass
class AmplitudeCoefficients:
    # Inspiral (Region I)
    rho_1: ti.f64
    rho_2: ti.f64
    rho_3: ti.f64
    # Intermediate (Region IIa)
    delta_0: ti.f64
    delta_1: ti.f64
    delta_2: ti.f64
    delta_3: ti.f64
    delta_4: ti.f64
    # Merge-ringdown (Region IIb)
    gamma_1: ti.f64
    gamma_2: ti.f64
    gamma_3: ti.f64
    # derived coefficients
    f_peak: ti.f64
    amp0: ti.f64

    @ti.func
    def compute_amplitude_coefficients(self, source_params, pn_prefactors):
        # Inspiral (Region I)
        self.rho_1 = (3931.8979897196696 - 
                      17395.758706812805*source_params.eta + 
                      (3132.375545898835 + 343965.86092361377*source_params.eta - 1.2162565819981997e6*source_params.eta2 + 
                            (-70698.00600428853 + 1.383907177859705e6*source_params.eta - 3.9662761890979446e6*source_params.eta2) * source_params.xi + 
                            (-60017.52423652596 + 803515.1181825735*source_params.eta - 2.091710365941658e6*source_params.eta2) * source_params.xi * source_params.xi
                        ) * source_params.xi
                        )
        self.rho_2 = (-40105.47653771657 + 
                      112253.0169706701*source_params.eta + 
                      (23561.696065836168 - 3.476180699403351e6*source_params.eta + 1.137593670849482e7*source_params.eta2 + 
                            (754313.1127166454 - 1.308476044625268e7*source_params.eta + 3.6444584853928134e7*source_params.eta2) * source_params.xi + 
                            (596226.612472288 - 7.4277901143564405e6*source_params.eta + 1.8928977514040343e7*source_params.eta2) * source_params.xi * source_params.xi
                        ) * source_params.xi
                        )
        self.rho_3 = (83208.35471266537 - 
                      191237.7264145924*source_params.eta + 
                      (-210916.2454782992 + 8.71797508352568e6*source_params.eta - 2.6914942420669552e7*source_params.eta2 + 
                            (-1.9889806527362722e6 + 3.0888029960154563e7*source_params.eta - 8.390870279256162e7*source_params.eta2) * source_params.xi + 
                            (-1.4535031953446497e6 + 1.7063528990822166e7*source_params.eta - 4.2748659731120914e7*source_params.eta2) * source_params.xi * source_params.xi
                        ) * source_params.xi
                        )
        # Merge-ringdown (Region IIb)
        self.gamma_1 = (0.006927402739328343 + 
                        0.03020474290328911*source_params.eta + 
                        (0.006308024337706171 - 0.12074130661131138*source_params.eta + 0.26271598905781324*source_params.eta2 + 
                            (0.0034151773647198794 - 0.10779338611188374*source_params.eta + 0.27098966966891747*source_params.eta2) * source_params.xi+ 
                            (0.0007374185938559283 - 0.02749621038376281*source_params.eta + 0.0733150789135702*source_params.eta2) * source_params.xi * source_params.xi
                        ) * source_params.xi
                        )
        self.gamma_2 = (1.010344404799477 + 
                        0.0008993122007234548*source_params.eta + 
                        (0.283949116804459 - 4.049752962958005*source_params.eta + 13.207828172665366*source_params.eta2 + 
                            (0.10396278486805426 - 7.025059158961947*source_params.eta + 24.784892370130475*source_params.eta2) * source_params.xi + 
                            (0.03093202475605892 - 2.6924023896851663*source_params.eta + 9.609374464684983*source_params.eta2) * source_params.xi * source_params.xi
                        ) * source_params.xi
                        )
        self.gamma_3 = (1.3081615607036106 - 
                        0.005537729694807678*source_params.eta +
                        (-0.06782917938621007 - 0.6689834970767117*source_params.eta + 3.403147966134083*source_params.eta2 + 
                            (-0.05296577374411866 - 0.9923793203111362*source_params.eta + 4.820681208409587*source_params.eta2) * source_params.xi + 
                            (-0.006134139870393713 - 0.38429253308696365*source_params.eta + 1.7561754421985984*source_params.eta2) * source_params.xi * source_params.xi
                        ) * source_params.xi
                        )
        # compute delta_s in intermediate (Region IIa) and the derived coefficients
        if self.gamma_2 > 1.0:
            self.f_peak = ti.abs(source_params.f_ring - source_params.f_damp*self.gamma_3/self.gamma_2)
        else:
            self.f_peak = ti.abs(source_params.f_ring + (tm.sqrt(1-self.gamma_2*self.gamma_2) - 1) * source_params.f_damp*self.gamma_3/self.gamma_2)
        powers_amplitude_f_peak = UsefulPowers()
        powers_amplitude_f_peak.updating(self.f_peak)

        f_mid = 0.5*(AMPLITUDE_INSPIRAL_fJoin + self.f_peak)
        v1 = _amplitude_inspiral_ansatz(powers_amplitude_intermediate_f_join, self, pn_prefactors)
        v2 = (0.8149838730507785 + 
              2.5747553517454658*source_params.eta + 
              (1.1610198035496786 - 2.3627771785551537*source_params.eta + 6.771038707057573*source_params.eta2 + 
                    (0.7570782938606834 - 2.7256896890432474*source_params.eta + 7.1140380397149965*source_params.eta2) * source_params.xi + 
                    (0.1766934149293479 - 0.7978690983168183*source_params.eta + 2.1162391502005153*source_params.eta2) * source_params.xi * source_params.xi
              ) * source_params.xi
              )
        v3 = _amplitude_merge_ringdown_ansatz(powers_amplitude_f_peak, self, source_params.f_ring, source_params.f_damp)
        d1 = _d_amplitude_inspiral_ansatz(powers_amplitude_intermediate_f_join, self, pn_prefactors)
        d2 = _d_amplitude_merge_ringdown_ansatz(powers_amplitude_f_peak, self, source_params.f_ring, source_params.f_damp)
        self.delta_0, self.delta_1, self.delta_2, self.delta_3, self.delta_4 = _solve_delta_i(AMPLITUDE_INSPIRAL_fJoin, f_mid, self.f_peak, v1, v2, v3, d1, d2)

        self.amp0 = 0.25 * tm.sqrt(10.0/3.0*source_params.eta/useful_powers_pi.four_thirds) * source_params.M**2 / source_params.dL_SI * MRSUN_SI * MTSUN_SI


@ti.func
def _compute_amplitude(powers_of_Mf, amplitude_coefficients, pn_prefactors, f_ring, f_damp):
    amplitude = 0.0
    if powers_of_Mf.one < AMPLITUDE_INSPIRAL_fJoin:
        amplitude = _amplitude_inspiral_ansatz(powers_of_Mf, amplitude_coefficients, pn_prefactors)
    elif powers_of_Mf.one > amplitude_coefficients.f_peak:
        amplitude = _amplitude_merge_ringdown_ansatz(powers_of_Mf, amplitude_coefficients, f_ring, f_damp)
    else:
        amplitude = _amplitude_intermediate_ansatz(powers_of_Mf, amplitude_coefficients)
    return amplitude * amplitude_coefficients.amp0 / powers_of_Mf.seven_sixths

@ti.func
def _compute_phase(powers_of_Mf, phase_coefficients, pn_prefactors, f_ring, f_damp, eta):
    '''
    note that all phase ansatz are without 1/eta
    '''
    phase = 0.0
    if powers_of_Mf.one < PHASE_INSPIRAL_fJoin:
        phase = _phase_inspiral_ansatz(powers_of_Mf, phase_coefficients, pn_prefactors)
    elif powers_of_Mf.one > phase_coefficients.phase_merge_ringdown_f_join:
        phase = _phase_merge_ringdown_ansatz(powers_of_Mf, phase_coefficients, f_ring, f_damp) + phase_coefficients.C1_merge_ringdown + phase_coefficients.C2_merge_ringdown * powers_of_Mf.one
    else:
        phase = _phase_intermediate_ansatz(powers_of_Mf, phase_coefficients) + phase_coefficients.C1_intermediate + phase_coefficients.C2_intermediate * powers_of_Mf.one
    return phase/eta

@ti.func
def _compute_tf(powers_of_Mf, phase_coefficients, pn_prefactors, f_ring, f_damp, eta):
    '''
    note that all phase ansatz are without 1/eta
    '''
    tf = 0.0
    if powers_of_Mf.one < PHASE_INSPIRAL_fJoin:
        tf = _d_phase_inspiral_ansatz(powers_of_Mf, phase_coefficients, pn_prefactors)
    elif powers_of_Mf.one > phase_coefficients.phase_merge_ringdown_f_join:
        tf = _d_phase_merge_ringdown_ansatz(powers_of_Mf, phase_coefficients, f_ring, f_damp) + phase_coefficients.C2_merge_ringdown
    else:
        tf = _d_phase_intermediate_ansatz(powers_of_Mf, phase_coefficients) + phase_coefficients.C2_intermediate
    return tf/eta


@ti.func
def _get_polarization_from_amplitude_phase(amplitude, phase, iota):
    cross_prefactor = tm.cos(iota)
    plus_prefactor = 0.5 * (1.0 + cross_prefactor**2)
    plus = amplitude * tm.cexp(ComplexNumber([0.0, -1.0]*phase))
    cross = tm.cmul(ComplexNumber([0.0, -1.0]), plus) * cross_prefactor
    plus *= plus_prefactor
    return cross, plus
    



@ti.data_oriented
class IMRPhenomD(object):

    def __init__(self, frequencies, waveform_container=None, reference_frequency=None, returned_form='polarizations', include_tf=True, parameter_sanity_check=False):
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
            raise Exception(f'you are set reference_frequency={reference_frequency}, which must be postive.')
        else:
            self.reference_frequency = reference_frequency
            
        self.parameter_sanity_check = parameter_sanity_check
        if self.parameter_sanity_check:
            print('`parameter_sanity_check` is turn-on, make sure taichi is initialized with debug mode')
        else:
            print('`parameter_sanity_check` is disable, make sure all parameters passed in are valid.')

        if waveform_container is not None:
            if not (waveform_container.shape == frequencies.shape):
                raise Exception('passed in `waveform_container` and `frequencies` have different shape')
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
                raise Exception(f'`waveform_container` contains additional unknown keys {ret_content}, check spellings')
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

    def _initialize_waveform_container(self, returned_form, include_tf):
        ret_content = {}
        if returned_form == 'polarizations':
            ret_content.update({'hplus': ComplexNumber, 'hcross': ComplexNumber})
        elif returned_form == 'amplitude_phase':
            ret_content.update({'amplitude': ti.f64, 'phase': ti.f64})
        else:
            raise Exception(f'{returned_form} is unknown. `returned_form` can be one of `polarizations` and `amplitude_phase`')
        if include_tf:
            ret_content.update({'tf': ti.f64})
        
        waveform_field = ti.Struct.field(ret_content)
        ti.root.dense(ti.i, self.frequencies.shape).place(waveform_field)
        if hasattr(self, 'waveform_container') and self.waveform_container is not None:
            raise Exception('`waveform_container` is not None, you are trying to override the original')
        else:
            self.waveform_container = waveform_field

    def update_waveform(self, parameters):
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
        

