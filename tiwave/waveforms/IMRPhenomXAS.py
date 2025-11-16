# TODO:
# - improve amplitude merge-ringdown coefficients
from typing import Callable

import taichi as ti
import taichi.math as tm
import numpy as np
from numpy.typing import NDArray

from ..constants import *
from ..utils import ti_complex, gauss_elimination, UsefulPowers
from .common import PostNewtonianCoefficients
from .base_waveform import BaseWaveform


# Constants
PHENOMXAS_HIGH_FREQUENCY_CUT = 0.3
sqrt2 = tm.sqrt(2)
useful_powers_pi = UsefulPowers()
useful_powers_pi.update(PI)


@ti.dataclass
class SourceParameters:
    # TODO: doc detail defination and unit!
    # passed-in parameters
    m_1: float  # mass of primary (solar mass)
    m_2: float  # mass of secondary m_1 > m_2
    chi_1: float
    chi_2: float
    dL_Mpc: float  # luminosity distance (Mpc)
    iota: float
    phase_ref: float
    # derived parameters
    M: float  # total mass
    m1_dimless: float
    m2_dimless: float
    dL_SI: float  # luminosity distance in meter
    M_sec: float  # total mass in second
    Mf_ref: float  # dimensionless reference frequency
    dimension_factor: float  # dimension factor of mass and distance in amplitude
    eta: float  # symmetric_mass_ratio
    eta_pow2: float  # eta^2
    eta_pow3: float
    eta_pow4: float
    eta_pow5: float
    eta_pow6: float
    delta: float  # (m_1-m_2)/M
    delta_chi: float  # chi_1 - chi_2
    delta_chi_pow2: float
    chi_s: float  # (chi_1 + chi_2)/2
    chi_s_pow2: float
    chi_s_pow3: float
    chi_a: float  # (chi_1 - chi_2)/2
    chi_a_pow2: float
    chi_a_pow3: float
    chi_eff: float
    chi_PN_hat: float  # Eq. 4.17 in arXiv:2001.11412
    chi_PN_hat_pow2: float
    chi_PN_hat_pow3: float
    chi_PN_hat_pow4: float
    S_tot_hat: float  # Eq. 4.18
    S_tot_hat_pow2: float
    S_tot_hat_pow3: float
    S_tot_hat_pow4: float
    S_tot_hat_pow5: float
    # fitting parameters
    final_mass: float
    final_spin: float
    final_spin_pow2: float
    final_spin_pow3: float
    final_spin_pow4: float
    final_spin_pow5: float
    final_spin_pow6: float
    final_spin_pow7: float
    f_ring: float
    f_damp: float
    f_damp_pow2: float
    f_MECO: float
    f_ISCO: float
    peak_time_diff: float

    @ti.func
    def update_source_parameters(
        self,
        mass_1: float,
        mass_2: float,
        chi_1: float,
        chi_2: float,
        luminosity_distance: float,
        inclination: float,
        reference_phase: float,
        reference_frequency: float,
    ):
        """
        TODO: doc parameters definition and units,
        """
        # mass is in the unit of solar mass
        # dL is in the unit of Mpc
        self.m_1 = mass_1
        self.m_2 = mass_2
        self.chi_1 = chi_1
        self.chi_2 = chi_2
        self.dL_Mpc = luminosity_distance
        self.iota = inclination
        self.phase_ref = reference_phase

        # derived parameters
        self.M = self.m_1 + self.m_2
        self.m1_dimless = self.m_1 / self.M
        self.m2_dimless = self.m_2 / self.M
        self.dL_SI = self.dL_Mpc * 1e6 * PC_SI
        self.M_sec = self.M * MTSUN_SI
        self.Mf_ref = self.M_sec * reference_frequency
        self.dimension_factor = self.M**2 / self.dL_SI * MRSUN_SI * MTSUN_SI

        self.eta = self.m_1 * self.m_2 / (self.M * self.M)
        self.eta_pow2 = self.eta * self.eta
        self.eta_pow3 = self.eta * self.eta_pow2
        self.eta_pow4 = self.eta * self.eta_pow3
        self.eta_pow5 = self.eta * self.eta_pow4
        self.eta_pow6 = self.eta * self.eta_pow5

        self.delta = tm.sqrt(1.0 - 4.0 * self.eta)
        self.delta_chi = self.chi_1 - self.chi_2
        self.delta_chi_pow2 = self.delta_chi * self.delta_chi

        self.chi_s = (self.chi_1 + self.chi_2) * 0.5
        self.chi_s_pow2 = self.chi_s * self.chi_s
        self.chi_s_pow3 = self.chi_s * self.chi_s_pow2
        self.chi_a = (self.chi_1 - self.chi_2) * 0.5
        self.chi_a_pow2 = self.chi_a * self.chi_a
        self.chi_a_pow3 = self.chi_a * self.chi_a_pow2

        self.chi_eff = self.m1_dimless * self.chi_1 + self.m2_dimless * self.chi_2
        self.chi_PN_hat = (
            self.chi_eff - 38.0 / 113.0 * self.eta * (self.chi_1 + self.chi_2)
        ) / (1.0 - (76.0 / 113.0 * self.eta))
        self.chi_PN_hat_pow2 = self.chi_PN_hat * self.chi_PN_hat
        self.chi_PN_hat_pow3 = self.chi_PN_hat * self.chi_PN_hat_pow2
        self.chi_PN_hat_pow4 = self.chi_PN_hat * self.chi_PN_hat_pow3

        self.S_tot_hat = (
            self.m1_dimless * self.m1_dimless * self.chi_1
            + self.m2_dimless * self.m2_dimless * self.chi_2
        ) / (self.m1_dimless * self.m1_dimless + self.m2_dimless * self.m2_dimless)
        self.S_tot_hat_pow2 = self.S_tot_hat * self.S_tot_hat
        self.S_tot_hat_pow3 = self.S_tot_hat * self.S_tot_hat_pow2
        self.S_tot_hat_pow4 = self.S_tot_hat * self.S_tot_hat_pow3
        self.S_tot_hat_pow5 = self.S_tot_hat * self.S_tot_hat_pow4

        # fitting parameters
        self.final_mass = self._get_final_mass()
        self.final_spin = self._get_final_spin()
        self.final_spin_pow2 = self.final_spin * self.final_spin
        self.final_spin_pow3 = self.final_spin * self.final_spin_pow2
        self.final_spin_pow4 = self.final_spin * self.final_spin_pow3
        self.final_spin_pow5 = self.final_spin * self.final_spin_pow4
        self.final_spin_pow6 = self.final_spin * self.final_spin_pow5
        self.final_spin_pow7 = self.final_spin * self.final_spin_pow6
        self.f_ring = self._get_f_ring()
        self.f_damp = self._get_f_damp()
        self.f_damp_pow2 = self.f_damp * self.f_damp
        self.f_MECO = self._get_f_MECO()
        self.f_ISCO = self._get_f_ISCO()
        self.peak_time_diff = 2.0 * PI * (500.0 + self._psi4_to_strain())

    @ti.func
    def _psi4_to_strain(self) -> float:
        """
        The fit of the time internal between peak of strain and psi4.
        """
        return (
            13.39320482758057
            - 175.42481512989315 * self.eta
            + 2097.425116152503 * self.eta_pow2
            - 9862.84178637907 * self.eta_pow3
            + 16026.897939722587 * self.eta_pow4
            + (
                4.7895602776763
                - 163.04871764530466 * self.eta
                + 609.5575850476959 * self.eta_pow2
            )
            * self.S_tot_hat
            + (
                1.3934428041390161
                - 97.51812681228478 * self.eta
                + 376.9200932531847 * self.eta_pow2
            )
            * self.S_tot_hat_pow2
            + (
                15.649521097877374
                + 137.33317057388916 * self.eta
                - 755.9566456906406 * self.eta_pow2
            )
            * self.S_tot_hat_pow3
            + (
                13.097315867845788
                + 149.30405703643288 * self.eta
                - 764.5242164872267 * self.eta_pow2
            )
            * self.S_tot_hat_pow4
            + 105.37711654943146 * self.delta_chi * self.delta * self.eta_pow2
        )

    @ti.func
    def _get_final_mass(self) -> float:
        """
        arXiv:1611.00332
        """
        return 1.0 - (
            (
                0.057190958417936644 * self.eta
                + 0.5609904135313374 * self.eta_pow2
                - 0.84667563764404 * self.eta_pow3
                + 3.145145224278187 * self.eta_pow4
            )
            * (
                1.0
                + (
                    -0.13084389181783257
                    - 1.1387311580238488 * self.eta
                    + 5.49074464410971 * self.eta_pow2
                )
                * self.S_tot_hat
                + (-0.17762802148331427 + 2.176667900182948 * self.eta_pow2)
                * self.S_tot_hat_pow2
                + (
                    -0.6320191645391563
                    + 4.952698546796005 * self.eta
                    - 10.023747993978121 * self.eta_pow2
                )
                * self.S_tot_hat_pow3
            )
            / (
                1.0
                + (
                    -0.9919475346968611
                    + 0.367620218664352 * self.eta
                    + 4.274567337924067 * self.eta_pow2
                )
                * self.S_tot_hat
            )
            + (
                -0.09803730445895877
                * self.delta_chi
                * self.delta
                * (1 - 3.2283713377939134 * self.eta)
                * self.eta_pow2
                + 0.01118530335431078 * self.delta_chi_pow2 * self.eta_pow3
                - 0.01978238971523653
                * self.delta_chi
                * self.delta
                * (1 - 4.91667749015812 * self.eta)
                * self.eta
                * self.S_tot_hat
            )
        )

    @ti.func
    def _get_final_spin(self) -> float:
        return (
            (
                3.4641016151377544 * self.eta
                + 20.0830030082033 * self.eta_pow2
                - 12.333573402277912 * self.eta_pow3
            )
            / (1 + 7.2388440419467335 * self.eta)
            + (
                (self.m1_dimless * self.m1_dimless + self.m2_dimless * self.m2_dimless)
                * self.S_tot_hat
                + (
                    (
                        -0.8561951310209386 * self.eta
                        - 0.09939065676370885 * self.eta_pow2
                        + 1.668810429851045 * self.eta_pow3
                    )
                    * self.S_tot_hat
                    + (
                        0.5881660363307388 * self.eta
                        - 2.149269067519131 * self.eta_pow2
                        + 3.4768263932898678 * self.eta_pow3
                    )
                    * self.S_tot_hat_pow2
                    + (
                        0.142443244743048 * self.eta
                        - 0.9598353840147513 * self.eta_pow2
                        + 1.9595643107593743 * self.eta_pow3
                    )
                    * self.S_tot_hat_pow3
                )
                / (
                    1
                    + (
                        -0.9142232693081653
                        + 2.3191363426522633 * self.eta
                        - 9.710576749140989 * self.eta_pow3
                    )
                    * self.S_tot_hat
                )
            )
            + (
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
                * self.S_tot_hat
            )
        )

    @ti.func
    def _get_f_ring(self) -> float:
        return (
            (
                0.05947169566573468
                - 0.14989771215394762 * self.final_spin
                + 0.09535606290986028 * self.final_spin_pow2
                + 0.02260924869042963 * self.final_spin_pow3
                - 0.02501704155363241 * self.final_spin_pow4
                - 0.005852438240997211 * self.final_spin_pow5
                + 0.0027489038393367993 * self.final_spin_pow6
                + 0.0005821983163192694 * self.final_spin_pow7
            )
            / (
                1.0
                - 2.8570126619966296 * self.final_spin
                + 2.373335413978394 * self.final_spin_pow2
                - 0.6036964688511505 * self.final_spin_pow4
                + 0.0873798215084077 * self.final_spin_pow6
            )
            / self.final_mass
        )

    @ti.func
    def _get_f_damp(self) -> float:
        return (
            (
                0.014158792290965177
                - 0.036989395871554566 * self.final_spin
                + 0.026822526296575368 * self.final_spin_pow2
                + 0.0008490933750566702 * self.final_spin_pow3
                - 0.004843996907020524 * self.final_spin_pow4
                - 0.00014745235759327472 * self.final_spin_pow5
                + 0.0001504546201236794 * self.final_spin_pow6
            )
            / (
                1.0
                - 2.5900842798681376 * self.final_spin
                + 1.8952576220623967 * self.final_spin_pow2
                - 0.31416610693042507 * self.final_spin_pow4
                + 0.009002719412204133 * self.final_spin_pow6
            )
            / self.final_mass
        )

    @ti.func
    def _get_f_MECO(self) -> float:
        return (
            (
                0.018744340279608845
                + 0.0077903147004616865 * self.eta
                + 0.003940354686136861 * self.eta_pow2
                - 0.00006693930988501673 * self.eta_pow3
            )
            / (1.0 - 0.10423384680638834 * self.eta)
            + self.chi_PN_hat
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
            / (
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
            + (
                self.delta_chi_pow2
                * (0.00034375176678815234 + 0.000016343732281057392 * self.eta)
                * self.eta_pow2
                + self.delta_chi
                * self.delta
                * self.eta
                * (
                    0.08064665214195679 * self.eta_pow2
                    + self.eta
                    * (-0.028476219509487793 - 0.005746537021035632 * self.chi_PN_hat)
                    - 0.0011713735642446144 * self.chi_PN_hat
                )
            )
        )

    @ti.func
    def _get_f_ISCO(self) -> float:
        """
        Frequency of the innermost stable circular orbit (ISCO).
        """
        Z1 = 1.0 + (1.0 - self.final_spin_pow2) ** (1 / 3) * (
            (1 + self.final_spin) ** (1 / 3) + (1 - self.final_spin) ** (1 / 3)
        )
        if Z1 > 3.0:
            Z1 = 3.0
        Z2 = tm.sqrt(3.0 * self.final_spin_pow2 + Z1 * Z1)

        return (
            1.0
            / (
                (
                    3.0
                    + Z2
                    - tm.sign(self.final_spin) * tm.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))
                )
                ** (3 / 2)
                + self.final_spin
            )
            / PI
        )


# Amplitude coefficients
@ti.dataclass
class AmplitudeCoefficients:
    # Inspiral
    rho_1: float
    rho_2: float
    rho_3: float
    ins_colloc_points: ti.types.vector(3, float)
    ins_colloc_values: ti.types.vector(3, float)
    # Intermediate (104 model)
    alpha_0: float
    alpha_1: float
    alpha_2: float
    alpha_3: float
    alpha_4: float
    int_colloc_points: ti.types.vector(3, float)
    int_colloc_values: ti.types.vector(5, float)
    # Merge-ringdown
    gamma_1: float  # a_R in arXiv:2001.11412
    gamma_2: float  # lambda
    gamma_3: float  # sigma
    f_peak: float
    # joint frequencies
    fjoin_int_ins: float
    fjoin_MRD_int: float
    _useful_powers: ti.types.struct(
        fjoin_int_ins=UsefulPowers,
        fjoin_MRD_int=UsefulPowers,
    )
    # cached parameters
    gamma1_gamma3_fdamp: float  # a_R * f_damp * sigma
    gamma2_over_gamma3_fdamp: float  # lambda / (f_damp * sigma)
    gamma3_fdamp: float  # f_damp * sigma
    gamma3_fdamp_pow2: float  # (f_damp * sigma)^2
    common_factor: float  # sqrt(2.0/3.0/pi^(1/3)*eta) (NOT including Y22)

    @ti.func
    def _set_ins_int_colloc_points(self, source_params: ti.template()):
        """
        Computing collocation points in insprial and intermediate range.
        Only can be called after updating merge-ringdown coefficient, since the f_peak
        is needed for intermediate collocation points.
        """
        # Joint frequencies
        self.fjoin_int_ins = source_params.f_MECO + 0.25 * (
            source_params.f_ISCO - source_params.f_MECO
        )  # Eq.6.5.
        self.fjoin_MRD_int = self.f_peak
        self._useful_powers.fjoin_int_ins.update(self.fjoin_int_ins)
        self._useful_powers.fjoin_MRD_int.update(self.fjoin_MRD_int)

        # Note the equation of Eq. 6.5 uses f_MECO, while fAT (Eq. 5.7) is used in
        # lalsimumation (l. 781 in LALSimIMRPhenomX_internals.c).
        self.ins_colloc_points[0] = self.fjoin_int_ins * 0.5
        self.ins_colloc_points[1] = self.fjoin_int_ins * 0.75
        self.ins_colloc_points[2] = self.fjoin_int_ins
        # Intermediate, Tab. II
        self.int_colloc_points[0] = self.fjoin_int_ins
        self.int_colloc_points[1] = (self.fjoin_int_ins + self.fjoin_MRD_int) * 0.5
        self.int_colloc_points[2] = self.fjoin_MRD_int

    @ti.func
    def _set_inspiral_coefficients(self, source_params: ti.template()):
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
            dt=float,
        )
        self.rho_1, self.rho_2, self.rho_3 = gauss_elimination(Ab_ins)

    @ti.func
    def _set_intermediate_coefficients(
        self, pn_coefficients: ti.template(), source_params: ti.template()
    ):
        """
        Only the recommended fit model `104` is implemented, and only can be called after
        updated the inspiral and merge-ringdown coefficients.
        """
        # # rigidly follow Eq. 6.6 in 2001.11412 (including f^(-7/6))
        # # different with lalsim
        # # if this part is used to get the intermediate amplitude coefficients, the
        # # _compute_amplitdue() must be modified accordingly.
        # self.int_colloc_values[0] = 1.0 / self._inspiral_amplitude(
        #     pn_coefficients, self._useful_powers.fjoin_int_ins
        # )
        # self.int_colloc_values[1] = self.int_colloc_points[1] ** (-7 / 6) / (
        #     (
        #         1.4873184918202145
        #         + 1974.6112656679577 * source_params.eta
        #         + 27563.641024162127 * source_params.eta_pow2
        #         - 19837.908020966777 * source_params.eta_pow3
        #     )
        #     / (
        #         1.0
        #         + 143.29004876335128 * source_params.eta
        #         + 458.4097306093354 * source_params.eta_pow2
        #     )
        #     + source_params.S_tot_hat
        #     * (
        #         27.952730865904343
        #         + source_params.eta
        #         * (-365.55631765202895 - 260.3494489873286 * source_params.S_tot_hat)
        #         + 3.2646808851249016 * source_params.S_tot_hat
        #         + 3011.446602208493 * source_params.eta_pow2 * source_params.S_tot_hat
        #         - 19.38970173389662 * source_params.S_tot_hat_pow2
        #         + source_params.eta_pow3
        #         * (
        #             1612.2681322644232
        #             - 6962.675551371755 * source_params.S_tot_hat
        #             + 1486.4658089990298 * source_params.S_tot_hat_pow2
        #         )
        #     )
        #     / (
        #         12.647425554323242
        #         - 10.540154508599963 * source_params.S_tot_hat
        #         + 1.0 * source_params.S_tot_hat_pow2
        #     )
        #     + source_params.delta_chi
        #     * source_params.delta
        #     * (-0.016404056649860943 - 296.473359655246 * source_params.eta)
        #     * source_params.eta_pow2
        # )
        # self.int_colloc_values[2] = 1.0 / self._merge_ringdown_amplitude(
        #     source_params,self._useful_powers.fjoin_MRD_int
        # )
        # self.int_colloc_values[3] = -(
        #     self._inspiral_d_amplitude(
        #         pn_coefficients, self._useful_powers.fjoin_int_ins
        #     )
        #     * self.int_colloc_values[0] ** 2
        # )
        # self.int_colloc_values[4] = -(
        #     self._merge_ringdown_d_amplitude
        #         source_params, self._useful_powers.fjoin_MRD_int
        #     )
        #     * self.int_colloc_values[2] ** 2
        # )

        # The way to compute intermediate coefficients used in lalsimulation.
        # the f^(-7/6) factor is absorbed into the coefficients, which may not be an
        # appropriate approach?? Since the coefficients are no longer constants.
        # This is equivalent to use the ansatz of the form 1/[...] rather than f^(-7/6)/[...]
        v0_ins = self._inspiral_amplitude(
            pn_coefficients, self._useful_powers.fjoin_int_ins
        )
        v2_MRD = self._merge_ringdown_amplitude(
            source_params, self._useful_powers.fjoin_MRD_int
        )

        self.int_colloc_values[0] = (
            self._useful_powers.fjoin_int_ins.seven_sixths / v0_ins
        )
        self.int_colloc_values[1] = 1.0 / (
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
            + source_params.delta_chi
            * source_params.delta
            * (-0.016404056649860943 - 296.473359655246 * source_params.eta)
            * source_params.eta_pow2
        )
        self.int_colloc_values[2] = (
            self._useful_powers.fjoin_MRD_int.seven_sixths / v2_MRD
        )
        self.int_colloc_values[3] = (
            7 / 6 * self.int_colloc_points[0] ** (1 / 6) / v0_ins
            - self.int_colloc_points[0] ** (7 / 6)
            * self._inspiral_d_amplitude(
                pn_coefficients, self._useful_powers.fjoin_int_ins
            )
            / v0_ins**2
        )
        self.int_colloc_values[4] = (
            7 / 6 * self.int_colloc_points[2] ** (1 / 6) / v2_MRD
            - self.int_colloc_points[2] ** (7 / 6)
            * self._merge_ringdown_d_amplitude(
                source_params, self._useful_powers.fjoin_MRD_int
            )
            / v2_MRD**2
        )

        Ab_int = ti.Matrix(
            [
                [
                    1.0,
                    self.int_colloc_points[0],
                    self.int_colloc_points[0] ** 2,
                    self.int_colloc_points[0] ** 3,
                    self.int_colloc_points[0] ** 4,
                    self.int_colloc_values[0],
                ],
                [
                    1.0,
                    self.int_colloc_points[1],
                    self.int_colloc_points[1] ** 2,
                    self.int_colloc_points[1] ** 3,
                    self.int_colloc_points[1] ** 4,
                    self.int_colloc_values[1],
                ],
                [
                    1.0,
                    self.int_colloc_points[2],
                    self.int_colloc_points[2] ** 2,
                    self.int_colloc_points[2] ** 3,
                    self.int_colloc_points[2] ** 4,
                    self.int_colloc_values[2],
                ],
                [
                    0.0,
                    1.0,
                    2.0 * self.int_colloc_points[0],
                    3.0 * self.int_colloc_points[0] ** 2,
                    4.0 * self.int_colloc_points[0] ** 3,
                    self.int_colloc_values[3],
                ],
                [
                    0.0,
                    1.0,
                    2.0 * self.int_colloc_points[2],
                    3.0 * self.int_colloc_points[2] ** 2,
                    4.0 * self.int_colloc_points[2] ** 3,
                    self.int_colloc_values[4],
                ],
            ],
            dt=float,
        )
        (
            self.alpha_0,
            self.alpha_1,
            self.alpha_2,
            self.alpha_3,
            self.alpha_4,
        ) = gauss_elimination(Ab_int)

    @ti.func
    def _set_merge_ringdown_coefficients(self, source_params: ti.template()):
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
    def _inspiral_amplitude(
        self, pn_coefficients: ti.template(), powers_of_Mf: ti.template()
    ) -> float:
        """
        Eq. 6.3 in arXiv:2001.11412.
        only have the 103 fit model.
        """
        return (
            pn_coefficients.PN_amplitude(powers_of_Mf)
            + self.rho_1 * powers_of_Mf.seven_thirds
            + self.rho_2 * powers_of_Mf.eight_thirds
            + self.rho_3 * powers_of_Mf.three
        )

    @ti.func
    def _inspiral_d_amplitude(
        self, pn_coefficients: ti.template(), powers_of_Mf: ti.template()
    ) -> float:
        """ """
        return (
            pn_coefficients.PN_d_amplitude(powers_of_Mf)
            + 7.0 / 3.0 * self.rho_1 * powers_of_Mf.four_thirds
            + 8.0 / 3.0 * self.rho_2 * powers_of_Mf.five_thirds
            + 3.0 * self.rho_3 * powers_of_Mf.two
        )

    @ti.func
    def _intermediate_amplitude(self, powers_of_Mf: ti.template()) -> float:
        """
        Eq. 6.7 in arXiv:2001.11412.
        Only the recommended fitting model `104` with 4th order polynomial ansatz are implemented.
        """
        # NOTE the ansatz used in lalsimulation where the intermediate coefficients have absorbed
        # common f^(-7/6) factor.
        return 1.0 / (
            self.alpha_0
            + self.alpha_1 * powers_of_Mf.one
            + self.alpha_2 * powers_of_Mf.two
            + self.alpha_3 * powers_of_Mf.three
            + self.alpha_4 * powers_of_Mf.four
        )

    @ti.func
    def _merge_ringdown_amplitude(
        self, source_params: ti.template(), powers_of_Mf: ti.template()
    ) -> float:
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
            self.gamma1_gamma3_fdamp
            / (f_minus_fring * f_minus_fring + self.gamma3_fdamp_pow2)
            * tm.exp(-f_minus_fring * self.gamma2_over_gamma3_fdamp)
        )

    @ti.func
    def _merge_ringdown_d_amplitude(
        self, source_params: ti.template(), powers_of_Mf: ti.template()
    ) -> float:
        """
        Derivative with respect to f of the amplitude merge-ringdown ansatz.
        """
        # f - f_ring
        f_minus_fring = powers_of_Mf.one - source_params.f_ring
        # (f - f_ring)^2
        f_minus_fring_pow2 = f_minus_fring * f_minus_fring
        # (f - f_ring)^2 + (gamma_3 * f_damp)^2
        common_term = f_minus_fring_pow2 + self.gamma3_fdamp_pow2
        return (
            -self.gamma_1
            * tm.exp(-f_minus_fring * self.gamma2_over_gamma3_fdamp)
            * (self.gamma_2 * common_term + 2.0 * self.gamma3_fdamp * f_minus_fring)
            / common_term**2
        )

    @ti.func
    def update_amplitude_coefficients(
        self, pn_coefficients: ti.template(), source_params: ti.template()
    ):
        self._set_merge_ringdown_coefficients(source_params)
        self._set_ins_int_colloc_points(source_params)
        self._set_inspiral_coefficients(source_params)
        self._set_intermediate_coefficients(pn_coefficients, source_params)

        # The common factor (without f^{-7/6} and the constant factor in Y22)
        self.common_factor = tm.sqrt(
            2.0 * source_params.eta / 3.0 / useful_powers_pi.third
        )

    @ti.func
    def compute_amplitude(
        self,
        pn_coefficients: ti.template(),
        source_params: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> float:
        # amplitude = 0.0
        # if powers_of_Mf.one < self.fjoin_int_ins:
        #     amplitude = self._inspiral_amplitude(pn_coefficients, powers_of_Mf)
        # elif powers_of_Mf.one > self.fjoin_MRD_int:
        #     amplitude = self._merge_ringdown_amplitude(source_params, powers_of_Mf)
        # else:
        #     amplitude = self._intermediate_amplitude(powers_of_Mf)
        # return (
        #       source_params.dimension_factor
        #       * self.common_factor
        #       * amplitude
        #       / powers_of_Mf.seven_sixths
        # )
        amplitude = 0.0
        if powers_of_Mf.one < self.fjoin_int_ins:
            amplitude = (
                self._inspiral_amplitude(pn_coefficients, powers_of_Mf)
                / powers_of_Mf.seven_sixths
            )
        elif powers_of_Mf.one > self.fjoin_MRD_int:
            amplitude = (
                self._merge_ringdown_amplitude(source_params, powers_of_Mf)
                / powers_of_Mf.seven_sixths
            )
        else:
            # Note the intermediate coefficients used in lalsimulation have absorbed the factor of f^(-7/6)
            amplitude = self._intermediate_amplitude(powers_of_Mf)
        return self.common_factor * amplitude


@ti.dataclass
class PhaseCoefficients:
    # Inspiral (104 fitting model)
    sigma_1: float
    sigma_2: float
    sigma_3: float
    sigma_4: float
    ins_colloc_points: ti.types.vector(4, float)
    ins_colloc_values: ti.types.vector(4, float)
    # Intermediate (105 fitting model)
    beta_0: float
    beta_1: float
    beta_2: float
    beta_3: float
    beta_4: float
    int_colloc_points: ti.types.vector(5, float)
    int_colloc_values: ti.types.vector(5, float)
    C0_int: float
    C1_int: float
    # Merge-ringdown
    c_0: float
    c_1: float  # f^-1/3
    c_2: float  # f^-2
    c_4: float  # f_-4
    c_L: float  # Lorentzian term
    MRD_colloc_points: ti.types.vector(5, float)
    MRD_colloc_values: ti.types.vector(5, float)
    C0_MRD: float
    C1_MRD: float
    # joint frequencies
    fjoin_int_ins: float
    fjoin_MRD_int: float
    _useful_powers: ti.types.struct(
        fjoin_int_ins=UsefulPowers,  # fmax_ins is not same with fmin_int, using fmin_int as the joint point when computing the phase and the connection coefficients
        fjoin_MRD_int=UsefulPowers,  # fmin_MRD is not same with fmax_int, using fmax_int as the joint point when computing the phase and the connection coefficients
    )
    # time and phase shift
    time_shift: float
    phase_shift: float

    @ti.func
    def _set_all_colloc_points(self, source_params: ti.template()):
        # Merge-ringdown
        fmin_MRD = 0.3 * source_params.f_ring + 0.6 * source_params.f_ISCO
        fmax_MRD = source_params.f_ring + 1.25 * source_params.f_damp
        frange_MRD = fmax_MRD - fmin_MRD
        self.MRD_colloc_points[0] = fmin_MRD
        self.MRD_colloc_points[1] = fmin_MRD + 0.5 * (1 - 1 / sqrt2) * frange_MRD
        self.MRD_colloc_points[2] = fmin_MRD + 0.5 * frange_MRD
        self.MRD_colloc_points[3] = source_params.f_ring
        self.MRD_colloc_points[4] = fmax_MRD
        # Inspiral (fitting model 104)
        fmin_ins = 0.0026
        fmax_ins = 1.02 * source_params.f_MECO
        frange_ins = fmax_ins - fmin_ins
        self.ins_colloc_points[0] = fmin_ins
        self.ins_colloc_points[1] = fmin_ins + 0.25 * frange_ins
        self.ins_colloc_points[2] = fmin_ins + 0.75 * frange_ins
        self.ins_colloc_points[3] = fmax_ins
        # Intermediate (fitting model 105)
        delta_R = 0.03 * (fmin_MRD - source_params.f_MECO)
        fmin_int = source_params.f_MECO - delta_R
        fmax_int = fmin_MRD + 0.5 * delta_R
        frange_int = fmax_int - fmin_int
        self.int_colloc_points[0] = fmin_int
        self.int_colloc_points[1] = fmin_int + 0.5 * (1.0 - 1 / sqrt2) * frange_int
        self.int_colloc_points[2] = fmin_int + 0.5 * frange_int
        self.int_colloc_points[3] = fmin_int + 0.5 * (1.0 + 1 / sqrt2) * frange_int
        self.int_colloc_points[4] = fmax_int
        # Note fmax_ins is not same with fmin_int, using fmin_int as the joint point
        # when computing the phase and the connection coefficients.
        # Similarly, fmin_MRD is not same with fmax_int, using fmax_int as the joint
        # point when computing the phase and the connection coefficients.
        self.fjoin_int_ins = fmin_int
        self.fjoin_MRD_int = fmax_int
        self._useful_powers.fjoin_int_ins.update(fmin_int)
        self._useful_powers.fjoin_MRD_int.update(fmax_int)

    @ti.func
    def _set_merge_ringdown_coefficients(self, source_params: ti.template()):
        # Note we use 0 as the first index
        # Difference between values on collocation points 0 and 1 (v0 - v1)
        d01_MRD = (
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
        # Difference between values on collocation points 1 and 3 (v1 - v3)
        d13_MRD = (
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
        # Difference between values on collocation points 2 and 3 (v2 - v3)
        d23_MRD = (
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
        # the value on collocation point 3 (v3)
        v3_MRD = (
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
        # Difference between values on collocation points 4 and 3 (v4 - v3)
        d43_MRD = (
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
        self.MRD_colloc_values[4] = d43_MRD + v3_MRD
        self.MRD_colloc_values[3] = v3_MRD
        self.MRD_colloc_values[2] = d23_MRD + v3_MRD
        self.MRD_colloc_values[1] = d13_MRD + v3_MRD
        self.MRD_colloc_values[0] = d01_MRD + self.MRD_colloc_values[1]

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
            ],
            dt=float,
        )
        self.c_0, self.c_1, self.c_2, self.c_4, self.c_L = (
            gauss_elimination(Ab_MRD) / source_params.eta
        )

    @ti.func
    def _set_inspiral_coefficients(self, source_params: ti.template()):
        """104 model"""
        # Note we use 0 as the first index
        # Value of v0 - v2
        d02_ins = (
            (
                -17294.000000000007
                - 19943.076428555978 * source_params.eta
                + 483033.0998073767 * source_params.eta_pow2
            )
            / (1.0 + 4.460294035404433 * source_params.eta)
            + source_params.chi_PN_hat
            * (
                68384.62786426462
                + 67663.42759836042 * source_params.chi_PN_hat
                - 2179.3505885609297 * source_params.chi_PN_hat_pow2
                + source_params.eta
                * (
                    -58475.33302037833
                    + 62190.404951852535 * source_params.chi_PN_hat
                    + 18298.307770807573 * source_params.chi_PN_hat_pow2
                    - 303141.1945565486 * source_params.chi_PN_hat_pow3
                )
                + 19703.894135534803 * source_params.chi_PN_hat_pow3
                + source_params.eta_pow2
                * (
                    -148368.4954044637
                    - 758386.5685734496 * source_params.chi_PN_hat
                    - 137991.37032619823 * source_params.chi_PN_hat_pow2
                    + 1.0765877367729193e6 * source_params.chi_PN_hat_pow3
                )
                + 32614.091002011017 * source_params.chi_PN_hat_pow4
            )
            / (2.0412979553629143 + 1.0 * source_params.chi_PN_hat)
            + 12017.062595934838
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta
        )
        # Value of v1 - v2
        d12_ins = (
            (
                -7579.300000000004
                - 120297.86185566607 * source_params.eta
                + 1.1694356931282217e6 * source_params.eta_pow2
                - 557253.0066989232 * source_params.eta_pow3
            )
            / (1.0 + 18.53018618227582 * source_params.eta)
            + source_params.chi_PN_hat
            * (
                -27089.36915061857
                - 66228.9369155027 * source_params.chi_PN_hat
                + source_params.eta_pow2
                * (
                    150022.21343386435
                    - 50166.382087278434 * source_params.chi_PN_hat
                    - 399712.22891153296 * source_params.chi_PN_hat_pow2
                )
                - 44331.41741405198 * source_params.chi_PN_hat_pow2
                + source_params.eta
                * (
                    50644.13475990821
                    + 157036.45676788126 * source_params.chi_PN_hat
                    + 126736.43159783827 * source_params.chi_PN_hat_pow2
                )
                + source_params.eta_pow3
                * (
                    -593633.5370110178
                    - 325423.99477314285 * source_params.chi_PN_hat
                    + 847483.2999508682 * source_params.chi_PN_hat_pow2
                )
            )
            / (
                -1.5232497464826662
                - 3.062957826830017 * source_params.chi_PN_hat
                - 1.130185486082531 * source_params.chi_PN_hat_pow2
                + 1.0 * source_params.chi_PN_hat_pow3
            )
            + 3843.083992827935
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta
        )
        # Value of v2
        v2_ins = (
            (
                15415.000000000007
                + 873401.6255736464 * source_params.eta
                + 376665.64637025696 * source_params.eta_pow2
                - 3.9719980569125614e6 * source_params.eta_pow3
                + 8.913612508054944e6 * source_params.eta_pow4
            )
            / (1.0 + 46.83697749859996 * source_params.eta)
            + source_params.chi_PN_hat
            * (
                397951.95299014193
                - 207180.42746987 * source_params.chi_PN_hat
                + source_params.eta_pow3
                * (
                    4.662143741417853e6
                    - 584728.050612325 * source_params.chi_PN_hat
                    - 1.6894189124921719e6 * source_params.chi_PN_hat_pow2
                )
                + source_params.eta
                * (
                    -1.0053073129700898e6
                    + 1.235279439281927e6 * source_params.chi_PN_hat
                    - 174952.69161683554 * source_params.chi_PN_hat_pow2
                )
                - 130668.37221912303 * source_params.chi_PN_hat_pow2
                + source_params.eta_pow2
                * (
                    -1.9826323844247842e6
                    + 208349.45742548333 * source_params.chi_PN_hat
                    + 895372.155565861 * source_params.chi_PN_hat_pow2
                )
            )
            / (
                -9.675704197652225
                + 3.5804521763363075 * source_params.chi_PN_hat
                + 2.5298346636273306 * source_params.chi_PN_hat_pow2
                + 1.0 * source_params.chi_PN_hat_pow3
            )
            + (
                -1296.9289110696955 * source_params.delta_chi_pow2 * source_params.eta
                + source_params.delta_chi
                * source_params.delta
                * source_params.eta
                * (
                    -24708.109411857182
                    + 24703.28267342699 * source_params.eta
                    + 47752.17032707405 * source_params.chi_PN_hat
                )
            )
        )
        # Value for v3 - v2
        d32_ins = (
            (
                2439.000000000001
                - 31133.52170083207 * source_params.eta
                + 28867.73328134167 * source_params.eta_pow2
            )
            / (1.0 + 0.41143032589262585 * source_params.eta)
            + source_params.chi_PN_hat
            * (
                16116.057657391262
                + source_params.eta_pow3
                * (-375818.0132734753 - 386247.80765802023 * source_params.chi_PN_hat)
                + source_params.eta
                * (-82355.86732027541 - 25843.06175439942 * source_params.chi_PN_hat)
                + 9861.635308837876 * source_params.chi_PN_hat
                + source_params.eta_pow2
                * (229284.04542668918 + 117410.37432997991 * source_params.chi_PN_hat)
            )
            / (
                -3.7385208695213668
                + 0.25294420589064653 * source_params.chi_PN_hat
                + 1.0 * source_params.chi_PN_hat_pow2
            )
            + 194.5554531509207
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta
        )
        self.ins_colloc_values[0] = d02_ins + v2_ins
        self.ins_colloc_values[1] = d12_ins + v2_ins
        self.ins_colloc_values[2] = v2_ins
        self.ins_colloc_values[3] = d32_ins + v2_ins

        Ab_ins = ti.Matrix(
            [
                [
                    1.0,
                    self.ins_colloc_points[0] ** (1 / 3),
                    self.ins_colloc_points[0] ** (2 / 3),
                    self.ins_colloc_points[0],
                    self.ins_colloc_values[0],
                ],
                [
                    1.0,
                    self.ins_colloc_points[1] ** (1 / 3),
                    self.ins_colloc_points[1] ** (2 / 3),
                    self.ins_colloc_points[1],
                    self.ins_colloc_values[1],
                ],
                [
                    1.0,
                    self.ins_colloc_points[2] ** (1 / 3),
                    self.ins_colloc_points[2] ** (2 / 3),
                    self.ins_colloc_points[2],
                    self.ins_colloc_values[2],
                ],
                [
                    1.0,
                    self.ins_colloc_points[3] ** (1 / 3),
                    self.ins_colloc_points[3] ** (2 / 3),
                    self.ins_colloc_points[3],
                    self.ins_colloc_values[3],
                ],
            ],
            dt=float,
        )
        # absorbing common factor into the pseudo PN parameters
        # note the normalizing factor used in lalsim: dphase0=(5/128/pi^(5/3))
        common_factor = 5.0 / 128.0 / source_params.eta / useful_powers_pi.five_thirds
        self.sigma_1, self.sigma_2, self.sigma_3, self.sigma_4 = (
            common_factor * gauss_elimination(Ab_ins)
        )

    @ti.func
    def _set_intermediate_coefficients(
        self, pn_coefficients: ti.template(), source_params: ti.template()
    ):
        """
        Require inspiral and merge-ringdown coefficients, can only be called after updating
        inspiral and merge-ringdown coefficients.
        """
        # v1_int - v3_MRD
        d_v1int_v3MRD = (
            (
                source_params.eta
                * (
                    0.9951733419499662
                    + 101.21991715215253 * source_params.eta
                    + 632.4731389009143 * source_params.eta_pow2
                )
            )
            / (
                0.00016803066316882238
                + 0.11412314719189287 * source_params.eta
                + 1.8413983770369362 * source_params.eta_pow2
                + 1.0 * source_params.eta_pow3
            )
            + source_params.S_tot_hat
            * (
                18.694178521101332
                + 16.89845522539974 * source_params.S_tot_hat
                + 4941.31613710257 * source_params.eta_pow2 * source_params.S_tot_hat
                + source_params.eta
                * (
                    -697.6773920613674
                    - 147.53381808989846 * source_params.S_tot_hat_pow2
                )
                + 0.3612417066833153 * source_params.S_tot_hat_pow2
                + source_params.eta_pow3
                * (
                    3531.552143264721
                    - 14302.70838220423 * source_params.S_tot_hat
                    + 178.85850322465944 * source_params.S_tot_hat_pow2
                )
            )
            / (
                2.965640445745779
                - 2.7706595614504725 * source_params.S_tot_hat
                + 1.0 * source_params.S_tot_hat_pow2
            )
            + source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
            * (
                356.74395864902294
                + 1693.326644293169 * source_params.eta_pow2 * source_params.S_tot_hat
            )
        )
        # v2_int - v3_MRD.
        d_v2int_v3MRD = (
            source_params.eta
            * (
                -5.126358906504587
                - 227.46830225846668 * source_params.eta
                + 688.3609087244353 * source_params.eta_pow2
                - 751.4184178636324 * source_params.eta_pow3
            )
            / (
                -0.004551938711031158
                - 0.7811680872741462 * source_params.eta
                + 1.0 * source_params.eta_pow2
            )
            + source_params.S_tot_hat
            * (
                0.1549280856660919
                - 0.9539250460041732 * source_params.S_tot_hat
                - 539.4071941841604 * source_params.eta_pow2 * source_params.S_tot_hat
                + source_params.eta
                * (73.79645135116367 - 8.13494176717772 * source_params.S_tot_hat_pow2)
                - 2.84311102369862 * source_params.S_tot_hat_pow2
                + source_params.eta_pow3
                * (
                    -936.3740515136005
                    + 1862.9097047992134 * source_params.S_tot_hat
                    + 224.77581754671272 * source_params.S_tot_hat_pow2
                )
            )
            / (-1.5308507364054487 + 1.0 * source_params.S_tot_hat)
            + 2993.3598520496153
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow6
        )
        # v3_int - v2_int.
        d32_int = (
            (
                0.4248820426833804
                - 906.746595921514 * source_params.eta
                - 282820.39946006844 * source_params.eta_pow2
                - 967049.2793750163 * source_params.eta_pow3
                + 670077.5414916876 * source_params.eta_pow4
            )
            / (
                1.0
                + 1670.9440812294847 * source_params.eta
                + 19783.077247023448 * source_params.eta_pow2
            )
            + source_params.S_tot_hat
            * (
                0.22814271667259703
                + 1.1366593671801855 * source_params.S_tot_hat
                + source_params.eta_pow3
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
                + source_params.eta_pow2
                * (
                    -711.8532052499075
                    + 269.9234918621958 * source_params.S_tot_hat
                    + 941.6974723887743 * source_params.S_tot_hat_pow2
                )
                + source_params.eta_pow4
                * (
                    -4939.642457025497
                    - 227.7672020783411 * source_params.S_tot_hat
                    + 8745.201037897836 * source_params.S_tot_hat_pow2
                )
            )
            / (-1.2442293719740283 + 1.0 * source_params.S_tot_hat)
            + source_params.delta_chi
            * source_params.delta
            * (-514.8494071830514 + 1493.3851099678195 * source_params.eta)
            * source_params.eta_pow3
        )
        # v1_int_bar
        v1_int_bar = (
            (
                -82.54500000000004
                - 5.58197349185435e6 * source_params.eta
                - 3.5225742421184325e8 * source_params.eta_pow2
                + 1.4667258334378073e9 * source_params.eta_pow3
            )
            / (
                1.0
                + 66757.12830903867 * source_params.eta
                + 5.385164380400193e6 * source_params.eta_pow2
                + 2.5176585751772933e6 * source_params.eta_pow3
            )
            + source_params.S_tot_hat
            * (
                19.416719811164853
                - 36.066611959079935 * source_params.S_tot_hat
                - 0.8612656616290079 * source_params.S_tot_hat_pow2
                + source_params.eta_pow2
                * (
                    170.97203068800542
                    - 107.41099349364234 * source_params.S_tot_hat
                    - 647.8103976942541 * source_params.S_tot_hat_pow3
                )
                + 5.95010003393006 * source_params.S_tot_hat_pow3
                + source_params.eta_pow3
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
            / (-1.1492259468169692 + 1.0 * source_params.S_tot_hat)
            + source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow3
            * (
                7343.130973149263
                - 20486.813161100774 * source_params.eta
                + 515.9898508588834 * source_params.S_tot_hat
            )
        )
        # Note different with lalsim, we have absorbed 1/eta into inspiral and
        # merge-ringdown phase coefficients. Remember multiply it in int_colloc_values.
        self.int_colloc_values[0] = self._inspiral_d_phase(
            pn_coefficients, self._useful_powers.fjoin_int_ins
        )
        self.int_colloc_values[1] = (
            0.75 * (d_v1int_v3MRD + self.MRD_colloc_values[3]) + 0.25 * v1_int_bar
        ) / source_params.eta
        self.int_colloc_values[2] = (
            d_v2int_v3MRD + self.MRD_colloc_values[3]
        ) / source_params.eta
        self.int_colloc_values[3] = (
            d32_int / source_params.eta + self.int_colloc_values[2]
        )
        # note the fmax_int is different with the fmin_MRD, it is more appropriate to
        # use fjoin_MRD_int (fmax_int) to set the last int_colloc_values??
        # self.int_colloc_values[4] = self._merge_ringdown_d_phase(
        #     source_params, self._useful_powers.fjoin_MRD_int
        # )
        # Following lalsim, using MRD_colloc_values[0] (at f_phi_T) rather recalculating
        # the value at (f_phi_T + 0.5deltaR), it may be a potential bug??
        self.int_colloc_values[4] = self.MRD_colloc_values[0] / source_params.eta

        # the factor of f_ring is used to enhance the numerical stability??
        int_colloc_points_scaled = self.int_colloc_points / source_params.f_ring
        Ab_int = ti.Matrix(
            [
                [
                    1.0,
                    int_colloc_points_scaled[0] ** (-1),
                    int_colloc_points_scaled[0] ** (-2),
                    int_colloc_points_scaled[0] ** (-3),
                    int_colloc_points_scaled[0] ** (-4),
                    self.int_colloc_values[0]
                    - (
                        self.c_L
                        / (
                            source_params.f_damp_pow2
                            + 0.25
                            * (self.int_colloc_points[0] - source_params.f_ring) ** 2
                        )
                    ),
                ],
                [
                    1.0,
                    int_colloc_points_scaled[1] ** (-1),
                    int_colloc_points_scaled[1] ** (-2),
                    int_colloc_points_scaled[1] ** (-3),
                    int_colloc_points_scaled[1] ** (-4),
                    self.int_colloc_values[1]
                    - (
                        self.c_L
                        / (
                            source_params.f_damp_pow2
                            + 0.25
                            * (self.int_colloc_points[1] - source_params.f_ring) ** 2
                        )
                    ),
                ],
                [
                    1.0,
                    int_colloc_points_scaled[2] ** (-1),
                    int_colloc_points_scaled[2] ** (-2),
                    int_colloc_points_scaled[2] ** (-3),
                    int_colloc_points_scaled[2] ** (-4),
                    self.int_colloc_values[2]
                    - (
                        self.c_L
                        / (
                            source_params.f_damp_pow2
                            + 0.25
                            * (self.int_colloc_points[2] - source_params.f_ring) ** 2
                        )
                    ),
                ],
                [
                    1.0,
                    int_colloc_points_scaled[3] ** (-1),
                    int_colloc_points_scaled[3] ** (-2),
                    int_colloc_points_scaled[3] ** (-3),
                    int_colloc_points_scaled[3] ** (-4),
                    self.int_colloc_values[3]
                    - (
                        self.c_L
                        / (
                            source_params.f_damp_pow2
                            + 0.25
                            * (self.int_colloc_points[3] - source_params.f_ring) ** 2
                        )
                    ),
                ],
                [
                    1.0,
                    int_colloc_points_scaled[4] ** (-1),
                    int_colloc_points_scaled[4] ** (-2),
                    int_colloc_points_scaled[4] ** (-3),
                    int_colloc_points_scaled[4] ** (-4),
                    self.int_colloc_values[4]
                    - (
                        self.c_L
                        / (
                            source_params.f_damp_pow2
                            + 0.25
                            * (self.int_colloc_points[4] - source_params.f_ring) ** 2
                        )
                    ),
                ],
            ],
            dt=float,
        )

        (
            beta_0,
            beta_1,
            beta_2,
            beta_3,
            beta_4,
        ) = gauss_elimination(Ab_int)
        self.beta_0 = beta_0
        self.beta_1 = beta_1 * source_params.f_ring
        self.beta_2 = beta_2 * source_params.f_ring**2
        self.beta_3 = beta_3 * source_params.f_ring**3
        self.beta_4 = beta_4 * source_params.f_ring**4

    @ti.func
    def _set_connection_coefficients(
        self, pn_coefficients: ti.template(), source_params: ti.template()
    ):
        """
        Since the fmax_ins and fmin_MRD are not same with fmin_int and fmax_int, addition
        connection coefficients are required to keep C0 and C1 continuity condition.
        .. math::
        \begin{aligned}
            \phi_{\mathrm{ins}}(f_{\mathrm{join}}) &= \phi_{\mathrm{int}}(f_{\mathrm{join}}) + C_0 + C_1 f_{\mathrm{join}}, \\
            \phi_{\mathrm{ins}}'(f_{\mathrm{join}}) &=\phi_{\mathrm{int}}'(f_{\mathrm{join}}) + C_1,
        \end{aligned}
        from which we can have
        \begin{aligned}
            C_0 &= \phi_{\mathrm{ins}}(f_{\mathrm{join}}) - \phi_{\mathrm{int}}(f_{\mathrm{join}}) - C_1 f_{\mathrm{join}}, \\
            C_1 &= \phi_{\mathrm{int}}'(f_{\mathrm{join}}) - \phi_{\mathrm{ins}}'(f_{\mathrm{join}}),
        \end{aligned}
        The case for joint point of intermediate and merge-ringdown ranges is similar.
        """
        # Connection coefficients added into intermediate
        self.C1_int = self._inspiral_d_phase(
            pn_coefficients, self._useful_powers.fjoin_int_ins
        ) - self._intermediate_d_phase(source_params, self._useful_powers.fjoin_int_ins)
        self.C0_int = (
            self._inspiral_phase(pn_coefficients, self._useful_powers.fjoin_int_ins)
            - self._intermediate_phase(source_params, self._useful_powers.fjoin_int_ins)
            - self.C1_int * self.fjoin_int_ins
        )
        # Connection coefficients added into merge-ringdown
        self.C1_MRD = (
            self._intermediate_d_phase(source_params, self._useful_powers.fjoin_MRD_int)
            + self.C1_int
            - self._merge_ringdown_d_phase(
                source_params, self._useful_powers.fjoin_MRD_int
            )
        )
        self.C0_MRD = (
            self._intermediate_phase(source_params, self._useful_powers.fjoin_MRD_int)
            + self.C0_int
            + self.C1_int * self.fjoin_MRD_int
            - self._merge_ringdown_phase(
                source_params, self._useful_powers.fjoin_MRD_int
            )
            - self.C1_MRD * self.fjoin_MRD_int
        )

    @ti.func
    def _fit_of_time_at_fit_frequency(self, source_params: ti.template()) -> float:
        """
        The fit of dphi at the fit frequency, dphi(fring-fdamp). evaluated on the
        calibration dataset
        """
        return (
            3155.1635543201924
            + 1257.9949740608242 * source_params.eta
            - 32243.28428870599 * source_params.eta_pow2
            + 347213.65466875216 * source_params.eta_pow3
            - 1.9223851649491738e6 * source_params.eta_pow4
            + 5.3035911346921865e6 * source_params.eta_pow5
            - 5.789128656876938e6 * source_params.eta_pow6
            + (
                -24.181508118588667
                + 115.49264174560281 * source_params.eta
                - 380.19778216022763 * source_params.eta_pow2
            )
            * source_params.S_tot_hat
            + (
                24.72585609641552
                - 328.3762360751952 * source_params.eta
                + 725.6024119989094 * source_params.eta_pow2
            )
            * source_params.S_tot_hat_pow2
            + (
                23.404604124552
                - 646.3410199799737 * source_params.eta
                + 1941.8836639529036 * source_params.eta_pow2
            )
            * source_params.S_tot_hat_pow3
            + (
                -12.814828278938885
                - 325.92980012408367 * source_params.eta
                + 1320.102640190539 * source_params.eta_pow2
            )
            * source_params.S_tot_hat_pow4
            - 148.17317525117338
            * source_params.delta_chi
            * source_params.delta
            * source_params.eta_pow2
        )

    @ti.func
    def _set_time_and_phase_shift(
        self, pn_coefficients: ti.template(), source_params: ti.template()
    ):
        # initializing an instance of UsefulPowers for later use
        powers_of_Mf = UsefulPowers()

        # time shift so that peak amplitude is near t=0
        f_fit = source_params.f_ring - source_params.f_damp
        powers_of_Mf.update(f_fit)
        time_at_f_fit = self._compute_d_phase_core(
            pn_coefficients, source_params, powers_of_Mf
        )
        self.time_shift = (
            self._fit_of_time_at_fit_frequency(source_params)
            - time_at_f_fit
            - source_params.peak_time_diff
        )

        # phase shift to the reference_phase
        powers_of_Mf.update(source_params.Mf_ref)
        self.phase_shift = (
            2.0 * source_params.phase_ref
            + PI / 4
            - (
                self._compute_phase_core(pn_coefficients, source_params, powers_of_Mf)
                + self.time_shift * source_params.Mf_ref
            )
        )

    @ti.func
    def _inspiral_phase(
        self,
        pn_coefficients: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> float:
        return (
            pn_coefficients.PN_phase(powers_of_Mf)
            + self.sigma_1 * powers_of_Mf.one
            + 0.75 * self.sigma_2 * powers_of_Mf.four_thirds
            + 0.6 * self.sigma_3 * powers_of_Mf.five_thirds
            + 0.5 * self.sigma_4 * powers_of_Mf.two
        )

    @ti.func
    def _inspiral_d_phase(
        self,
        pn_coefficients: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> float:
        return (
            pn_coefficients.PN_d_phase(powers_of_Mf)
            + self.sigma_1
            + self.sigma_2 * powers_of_Mf.third
            + self.sigma_3 * powers_of_Mf.two_thirds
            + self.sigma_4 * powers_of_Mf.one
        )

    @ti.func
    def _intermediate_phase(
        self,
        source_params: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> float:
        return (
            self.beta_0 * powers_of_Mf.one
            + self.beta_1 * powers_of_Mf.log
            - self.beta_2 / powers_of_Mf.one
            - self.beta_3 / 2.0 / powers_of_Mf.two
            - self.beta_4 / 3.0 / powers_of_Mf.three
            + 2.0
            * self.c_L
            / source_params.f_damp
            * tm.atan2(
                (powers_of_Mf.one - source_params.f_ring), 2.0 * source_params.f_damp
            )
        )

    @ti.func
    def _intermediate_d_phase(
        self,
        source_params: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> float:
        return (
            self.beta_0
            + self.beta_1 / powers_of_Mf.one
            + self.beta_2 / powers_of_Mf.two
            + self.beta_3 / powers_of_Mf.three
            + self.beta_4 / powers_of_Mf.four
            + self.c_L
            / (
                source_params.f_damp_pow2
                + 0.25 * (powers_of_Mf.one - source_params.f_ring) ** 2
            )
        )

    @ti.func
    def _merge_ringdown_phase(
        self,
        source_params: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> float:
        return (
            self.c_0 * powers_of_Mf.one
            + 1.5 * self.c_1 * powers_of_Mf.two_thirds
            - self.c_2 / powers_of_Mf.one
            - self.c_4 / 3.0 / powers_of_Mf.three
            + self.c_L
            / source_params.f_damp
            * tm.atan2((powers_of_Mf.one - source_params.f_ring), source_params.f_damp)
        )

    @ti.func
    def _merge_ringdown_d_phase(
        self,
        source_params: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> float:
        return (
            self.c_0
            + self.c_1 / powers_of_Mf.third
            + self.c_2 / powers_of_Mf.two
            + self.c_4 / powers_of_Mf.four
            + self.c_L
            / (
                source_params.f_damp_pow2
                + (powers_of_Mf.one - source_params.f_ring) ** 2
            )
        )

    @ti.func
    def update_phase_coefficients(
        self,
        pn_coefficients: ti.template(),
        source_params: ti.template(),
    ):
        # note the func need to be called in proper order
        self._set_all_colloc_points(source_params)
        self._set_merge_ringdown_coefficients(source_params)
        self._set_inspiral_coefficients(source_params)
        self._set_intermediate_coefficients(pn_coefficients, source_params)
        self._set_connection_coefficients(pn_coefficients, source_params)
        self._set_time_and_phase_shift(pn_coefficients, source_params)

    @ti.func
    def _compute_phase_core(
        self,
        pn_coefficients: ti.template(),
        source_params: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> float:
        """ """
        phase = 0.0
        # The fmax_ins and fmin_MRD are not same with fmin_int and fmax_int. Taking the fmin_int
        # and fmax_int as the transtion points (l. 1020 in LALSimIMRPhenomX_internals.c)
        if powers_of_Mf.one < self.fjoin_int_ins:
            phase = self._inspiral_phase(pn_coefficients, powers_of_Mf)
        elif powers_of_Mf.one > self.fjoin_MRD_int:
            phase = (
                self._merge_ringdown_phase(source_params, powers_of_Mf)
                + self.C0_MRD
                + self.C1_MRD * powers_of_Mf.one
            )
        else:
            phase = (
                self._intermediate_phase(source_params, powers_of_Mf)
                + self.C0_int
                + self.C1_int * powers_of_Mf.one
            )
        return phase

    @ti.func
    def compute_phase(
        self,
        pn_coefficients: ti.template(),
        source_params: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> float:
        """ """
        return (
            self._compute_phase_core(pn_coefficients, source_params, powers_of_Mf)
            + self.time_shift * powers_of_Mf.one
            + self.phase_shift
        )

    @ti.func
    def _compute_d_phase_core(
        self,
        pn_coefficients: ti.template(),
        source_params: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> float:
        """ """
        d_phase = 0.0
        if powers_of_Mf.one < self.fjoin_int_ins:
            d_phase = self._inspiral_d_phase(pn_coefficients, powers_of_Mf)
        elif powers_of_Mf.one > self.fjoin_MRD_int:
            d_phase = (
                self._merge_ringdown_d_phase(source_params, powers_of_Mf) + self.C1_MRD
            )
        else:
            d_phase = (
                self._intermediate_d_phase(source_params, powers_of_Mf) + self.C1_int
            )
        return d_phase

    @ti.func
    def compute_d_phase(
        self,
        pn_coefficients: ti.template(),
        source_params: ti.template(),
        powers_of_Mf: ti.template(),
    ) -> float:

        return (
            self._compute_d_phase_core(pn_coefficients, source_params, powers_of_Mf)
            + self.time_shift
        )


@ti.data_oriented
class IMRPhenomXAS(BaseWaveform):
    """
    TODO:
    - description of proper parameter range, like q<1000, chi<0.99 etc..

    """

    def __init__(
        self,
        frequencies: ti.ScalarField | NDArray[np.float64],
        reference_frequency: float | None = None,
        return_form: str = "polarizations",
        include_tf: bool = True,
        check_parameters: bool = False,
        parameter_conversion: Callable | None = None,
    ) -> None:
        super().__init__(
            frequencies,
            reference_frequency,
            return_form,
            include_tf,
            check_parameters,
            parameter_conversion,
        )
        # instantiating scalar fields for global accessing
        self.source_parameters = SourceParameters.field(shape=())
        self.phase_coefficients = PhaseCoefficients.field(shape=())
        self.amplitude_coefficients = AmplitudeCoefficients.field(shape=())
        self.pn_coefficients = PostNewtonianCoefficients.field(shape=())

    def update_waveform(self, input_params: dict[str, float]) -> None:
        """
        necessary preparation which need to be finished in python scope
        """
        params = self.parameter_conversion(input_params)
        self._update_waveform_kernel(
            params["mass_1"],
            params["mass_2"],
            params["chi_1"],
            params["chi_2"],
            params["luminosity_distance"],
            params["inclination"],
            params["reference_phase"],
            self.reference_frequency,
        )

    @ti.func
    def _set_harmonic_factors(self, harmonic_factors: ti.template()):
        common = 0.125 * tm.sqrt(5.0 / PI)
        cos_iota = tm.cos(self.source_parameters[None].iota)
        harmonic_factors.plus = (
            -ti_complex([1.0, 0.0]) * common * (1.0 + cos_iota * cos_iota)
        )
        harmonic_factors.cross = ti_complex([0.0, 1.0]) * common * (2 * cos_iota)

    @ti.kernel
    def _update_waveform_kernel(
        self,
        mass_1: float,
        mass_2: float,
        chi_1: float,
        chi_2: float,
        luminosity_distance: float,
        inclination: float,
        reference_phase: float,
        reference_frequency: float,
    ):

        self.source_parameters[None].update_source_parameters(
            mass_1,
            mass_2,
            chi_1,
            chi_2,
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

        harm_fac = ti.Struct(
            plus=ti_complex([0.0, 0.0]), cross=ti_complex([0.0, 0.0])
        )
        if ti.static(self.return_form == "polarizations"):
            self._set_harmonic_factors(harm_fac)

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

                if ti.static(self.return_form == "amplitude_phase"):
                    self.waveform_container[idx].amplitude = amplitude
                    self.waveform_container[idx].phase = phase
                if ti.static(self.return_form == "polarizations"):
                    h_22 = amplitude * tm.cexp(ti_complex([0.0, phase]))
                    self.waveform_container[idx].plus = tm.cmul(harm_fac.plus, h_22)
                    self.waveform_container[idx].cross = tm.cmul(harm_fac.cross, h_22)
                if ti.static(self.include_tf):
                    dphi = self.phase_coefficients[None].compute_d_phase(
                        self.pn_coefficients[None],
                        self.source_parameters[None],
                        powers_of_Mf,
                    )
                    dphi *= self.source_parameters[None].M_sec / PI / 2  # to second
                    self.waveform_container[idx].tf = -dphi
            else:
                if ti.static(self.return_form == "amplitude_phase"):
                    self.waveform_container[idx].amplitude = 0.0
                    self.waveform_container[idx].phase = 0.0
                if ti.static(self.return_form == "polarizations"):
                    self.waveform_container[idx].plus.fill(0.0)
                    self.waveform_container[idx].cross.fill(0.0)
                if ti.static(self.include_tf):
                    self.waveform_container[idx].tf = 0.0

    def parameter_validity_check(self, parameters):
        # TODO: check paramters in taichi scope for improving performance
        # super().parameter_validity_check(parameters)
        pass
