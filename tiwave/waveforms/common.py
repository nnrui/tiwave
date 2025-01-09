import taichi as ti
import taichi.math as tm

from ..constants import *
from ..utils import UsefulPowers


useful_powers_pi = UsefulPowers()
useful_powers_pi.update(PI)


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
    prefactor_varphi_8: ti.f64
    prefactor_varphi_8l: ti.f64
    # 3 PN amplitude
    prefactor_A_0: ti.f64
    prefactor_A_1: ti.f64
    prefactor_A_2: ti.f64
    prefactor_A_3: ti.f64
    prefactor_A_4: ti.f64
    prefactor_A_5: ti.f64
    prefactor_A_6: ti.f64

    @ti.func
    def compute_PN_prefactors(self, source_params: ti.template()):
        """
        Using Eq.B6 - B13 and Eq.B14 - B19 in arXiv:1508.07253
        3PN spin-spin term not included
        """
        # Phase
        self.prefactor_varphi_0 = 1.0 / useful_powers_pi.five_thirds
        self.prefactor_varphi_1 = 0.0 / useful_powers_pi.four_thirds
        self.prefactor_varphi_2 = (
            37.15 / 7.56 + 55.0 / 9.0 * source_params.eta
        ) / useful_powers_pi.one
        self.prefactor_varphi_3 = (
            -16.0 * PI
            + (113.0 / 3.0 * source_params.delta * source_params.chi_a)
            + (113.0 / 3.0 - 76.0 / 3.0 * source_params.eta) * source_params.chi_s
        ) / useful_powers_pi.two_thirds
        self.prefactor_varphi_4 = (
            152.93365 / 5.08032
            + 271.45 / 5.04 * source_params.eta
            + 308.5 / 7.2 * source_params.eta_pow2
            + (-405.0 / 8.0 + 200.0 * source_params.eta) * source_params.chi_a_pow2
            - 405.0
            / 4.0
            * source_params.delta
            * source_params.chi_a
            * source_params.chi_s
            + (-405.0 / 8.0 + 5.0 / 2.0 * source_params.eta) * source_params.chi_s_pow2
        ) / useful_powers_pi.third
        self.prefactor_varphi_5 = (
            (386.45 / 7.56 - 65.0 / 9.0 * source_params.eta) * PI
            + (-732.985 / 2.268 - 140.0 / 9.0 * source_params.eta)
            * source_params.delta
            * source_params.chi_a
            + (
                -732.985 / 2.268
                + 2426.0 / 8.1 * source_params.eta
                + 340.0 / 9.0 * source_params.eta_pow2
            )
            * source_params.chi_s
        )
        self.prefactor_varphi_5l = self.prefactor_varphi_5
        self.prefactor_varphi_6 = (
            (
                11583.231236531 / 4.694215680
                - 640.0 / 3.0 * PI * PI
                - 684.8 / 2.1 * EULER_GAMMA
                + (-15737.765635 / 3.048192 + 225.5 / 1.2 * PI * PI) * source_params.eta
                + 76.055 / 1.728 * source_params.eta_pow2
                - 127.825 / 1.296 * source_params.eta_pow3
                - tm.log(2.0) * 1369.6 / 2.1
            )
            + 2270.0 / 3.0 * PI * source_params.delta * source_params.chi_a
            + (2270.0 / 3.0 - 520.0 * source_params.eta) * PI * source_params.chi_s
            + (755.15 / 1.44 - 822.5 / 1.8 * source_params.eta)
            * source_params.delta
            * source_params.chi_s
            * source_params.chi_a
            + (
                755.15 / 2.88
                - 2632.45 / 2.52 * source_params.eta
                - 480.0 * source_params.eta_pow2
            )
            * source_params.chi_a_pow2
            + (
                755.15 / 2.88
                - 2324.15 / 5.04 * source_params.eta
                + 1255.0 / 9.0 * source_params.eta_pow2
            )
            * source_params.chi_s_pow2
        ) * useful_powers_pi.third
        self.prefactor_varphi_6l = -684.8 / 6.3 * useful_powers_pi.third
        self.prefactor_varphi_7 = (
            (
                770.96675 / 2.54016
                + 378.515 / 1.512 * source_params.eta
                - 740.45 / 7.56 * source_params.eta_pow2
            )
            * PI
            + (
                -25150.083775 / 3.048192
                + 26804.935 / 6.048 * source_params.eta
                - 198.5 / 4.8 * source_params.eta_pow2
            )
            * source_params.delta
            * source_params.chi_a
            + (
                -25150.083775 / 3.048192
                + 105666.55595 / 7.62048 * source_params.eta
                - 1042.165 / 3.024 * source_params.eta_pow2
                + 534.5 / 3.6 * source_params.eta_pow3
            )
            * source_params.chi_s
            - 1140.0
            * PI
            * source_params.delta
            * source_params.chi_a
            * source_params.chi_s
            + (-570.0 + 2240.0 * source_params.eta) * PI * source_params.chi_a_pow2
            + (-570.0 + 40.0 * source_params.eta) * PI * source_params.chi_s_pow2
            + (14585.0 / 8.0 - 215.0 / 2.0 * source_params.eta)
            * source_params.delta
            * source_params.chi_a
            * source_params.chi_s_pow2
            + (
                14585.0 / 8.0
                - 7270.0 * source_params.eta
                + 80.0 * source_params.eta_pow2
            )
            * source_params.chi_a_pow2
            * source_params.chi_s
            + (1458.5 / 2.4 - 2380.0 * source_params.eta)
            * source_params.delta
            * source_params.chi_a_pow3
            + (
                1458.5 / 2.4
                - 475.0 / 6.0 * source_params.eta
                + 100.0 / 3.0 * source_params.eta_pow2
            )
            * source_params.chi_s_pow3
        ) * useful_powers_pi.two_thirds
        self.prefactor_varphi_8 = (
            (2339.15 / 1.68 - 991.85 / 2.52 * source_params.eta)
            * source_params.delta
            * source_params.chi_a
            + (
                2339.15 / 1.68
                - 3970.375 / 2.268 * source_params.eta
                + 196.55 / 1.89 * source_params.eta_pow2
            )
            * source_params.chi_s
        ) * useful_powers_pi.two
        self.prefactor_varphi_8l = -self.prefactor_varphi_8
        # Amplitude
        # (equatoins in PhenomD paper have some difference with PhenomX paper in A5 and A6, using equations in PhenomX paper here.)
        self.prefactor_A_0 = 1.0
        self.prefactor_A_1 = 0.0
        self.prefactor_A_2 = (
            -3.23 / 2.24 + 4.51 / 1.68 * source_params.eta
        ) * useful_powers_pi.two_thirds
        self.prefactor_A_3 = (
            27.0 / 8.0 * source_params.delta * source_params.chi_a
            + (27.0 / 8.0 - 11.0 / 6.0 * source_params.eta) * source_params.chi_s
        ) * useful_powers_pi.one
        self.prefactor_A_4 = (
            -27.312085 / 8.128512
            - 19.75055 / 3.38688 * source_params.eta
            + 10.5271 / 2.4192 * source_params.eta_pow2
            + (-8.1 / 3.2 + 8.0 * source_params.eta) * source_params.chi_a_pow2
            - 8.1
            / 1.6
            * source_params.delta
            * source_params.chi_a
            * source_params.chi_s
            + (-8.1 / 3.2 + 17.0 / 8.0 * source_params.eta) * source_params.chi_s_pow2
        ) * useful_powers_pi.four_thirds
        self.prefactor_A_5 = (
            -8.5 / 6.4 * PI
            + 8.5 / 1.6 * PI * source_params.eta
            - 3.0 / 8.0 * (-1.0 + 3.0 * source_params.eta) * source_params.chi_s_pow3
            - 3.0
            / 8.0
            * (-1.0 + source_params.eta)
            * source_params.delta
            * source_params.chi_a_pow3
            - 9.0
            / 8.0
            * (-1.0 + source_params.eta)
            * source_params.delta
            * source_params.chi_s_pow2
            * source_params.chi_a
            - 9.0
            / 8.0
            * (-1.0 + 3.0 * source_params.eta)
            * source_params.chi_s
            * source_params.chi_a_pow2
            + (28.7213 / 1.6128 - 2.083 / 4.032 * source_params.eta)
            * source_params.delta
            * source_params.chi_a
            + (
                28.7213 / 1.6128
                - 155.69 / 6.72 * source_params.eta
                - 2.227 / 1.008 * source_params.eta_pow2
            )
            * source_params.chi_s
        ) * useful_powers_pi.five_thirds
        self.prefactor_A_6 = (
            -177.520268561 / 8.583708672
            + (545.384828789 / 5.007163392 - 20.5 / 4.8 * useful_powers_pi.two)
            * source_params.eta
            - 32.48849057 / 1.78827264 * source_params.eta_pow2
            + 34.473079 / 6.386688 * source_params.eta_pow3
            + (
                -49.039 / 7.168
                - 10.043 / 2.016 * source_params.eta
                + 14.975 / 4.032 * source_params.eta_pow2
            )
            * source_params.chi_s_pow2
            + (
                -49.039 / 7.168
                + 26.553 / 1.792 * source_params.eta
                + 77.2 / 2.1 * source_params.eta_pow2
            )
            * source_params.chi_a_pow2
            - (49.039 / 3.584 + 141.359 / 8.064 * source_params.eta)
            * source_params.delta
            * source_params.chi_s
            * source_params.chi_a
            + (-17.0 / 6.0 + 10.0 / 3.0 * source_params.eta) * PI * source_params.chi_s
            - 17.0 / 6.0 * PI * source_params.delta * source_params.chi_a
        ) * useful_powers_pi.two
