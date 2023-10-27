import taichi as ti
import taichi.math as tm

from ..constants import *


'''
TODO:
1. source parameter check, (m1>m2, q<1)
'''


@ti.dataclass
class SourceParameters:
    M: ti.f64       # total mass
    M_sec: ti.f64   # total mass in second
    eta: ti.f64     # symmetric_mass_ratio
    eta2: ti.f64    # eta^2
    chi_PN: ti.f64

    @ti.func
    def generate_all_source_parameters(self, mass_1, mass_2, chi_1, chi_2):
        self.M = mass_1 + mass_2
        self.eta = mass_1 * mass_2 / (self.M * self.M)
        self.eta2 = self.eta * self.eta
        self.eta3 = self.eta2 * self.eta
        self.delta = mass_1 - mass_2     # mass_1 > mass_2
        self.chi_s = (chi_1 + chi_2) * 0.5
        self.chi_a = (chi_1 - chi_2) * 0.5
        self.chi_PN = (mass_1*chi_1 + mass_2*chi_2)/self.M - 38.0/113.0*self.eta*(chi_1+chi_2)




# PN expansion coefficients
@ti.dataclass
class PostNewtonianCoefficients:
    # 3.5 PN phase
    varphi_0: ti.f64
    varphi_1: ti.f64
    varphi_2: ti.f64
    varphi_3: ti.f64
    varphi_4: ti.f64
    varphi_5: ti.f64
    varphi_5l: ti.f64
    varphi_6: ti.f64
    varphi_6l: ti.f64
    varphi_7: ti.f64
    # 3 PN amplitude
    A_0: ti.f64
    A_1: ti.f64
    A_2: ti.f64
    A_3: ti.f64
    A_4: ti.f64
    A_5: ti.f64
    A_6: ti.f64

    @ti.func
    def compute_PN_coefficients(self, source_params):
        '''
        Using Eq.B6 - B13 and Eq.B14 - B19 in arXiv:1508.07253
        3PN spin-spin term not included
        '''
        self.varphi_0 = 1.0
        self.varphi_1 = 0.0
        self.varphi_2 = 3715.0/756.0 + 55.0/9.0*source_params.eta
        self.varphi_3 = (-16.0*PI +
                         (113.0/3.0*source_params.delta*source_params.chi_a) + 
                         (113.0/3.0 - 76.0/3.0*source_params.eta)*source_params.chi_s 
                        )
        self.varphi_4 = (5.0/72.0*(3058.673/7.056 + 5429.0/7.0*source_params.eta + 617.0*source_params.eta2) +
                         (-405.0/8.0 + 200.0*source_params.eta) * source_params.chi_a * source_params.chi_a - 
                         405.0/4.0 * source_params.delta * source_params.chi_a * source_params.chi_s + 
                         (-405.0/8.0 + 5.0/2.0*source_params.eta) * source_params.chi_s * source_params.chi_s
                        )
        self.varphi_5 = (5.0/9.0 * (772.9/8.4 - 13.0*source_params.eta) * PI +
                         (-732.985/2.268 - 140.0/9.0*source_params.eta) * source_params.delta * source_params.chi_a +
                         (-732.985/2.268 + 2426.0/8.1*source_params.eta + 340.0/9.0*source_params.eta2) * source_params.chi_s
                        )
        self.varphi_5l= 3.0 * self.varphi_5
        self.varphi_6 = ((11583.231236531/4.694215680 - 640.0/3.0*PI*PI - 684.8/2.1*EULER_GAMMA + 
                            (-15737.765635/3.048192 + 225.5/1.2*PI*PI)*source_params.eta + 
                            76.055/1.728*source_params.eta2 - 
                            127.825/1.296*source_params.eta3 - 
                            tm.log(4.)*684.8/2.1) +
                         2270.0/3.0*PI*source_params.delta*source_params.chi_a
                         (2270.0/3.0 - 520.0*source_params.eta)*PI*source_params.chi_s
                        )
        self.varphi_6l=-684.8/2.1
        self.varphi_7 = ((770.96675/2.54016 + 378.515/1.512*source_params.eta- 740.45/7.56*source_params.eta2)*PI + 
                         (-25150.083775/3.048192 + 26804.935/6.048*source_params.eta - 198.5/4.8*source_params.eta2) * source_params.delta * source_params.chi_a
                         (-25150.083775/3.048192 + 105666.55595/7.62048*source_params.eta - 1042.165/3.024*source_params.eta2 + 534.5/3.6*source_params.eta3) * source_params.chi_s
                        )


# 14 phase coefficients
@ti.dataclass
class PhaseCoefficients:
    # Inspiral (Region I)
    sigma_0: ti.f64
    sigma_1: ti.f64
    sigma_2: ti.f64
    sigma_3: ti.f64
    sigma_4: ti.f64
    # Intermediate (Region IIa)
    beta_0: ti.f64
    beta_1: ti.f64
    beta_2: ti.f64
    beta_3: ti.f64
    # Merge-ringdown (Region IIb)
    alpha_0: ti.f64
    alpha_1: ti.f64
    alpha_2: ti.f64
    alpha_3: ti.f64
    alpha_4: ti.f64
    alpha_5: ti.f64

    @ti.func
    def compute_phase_coefficients(self, source_params):
        '''
        Compute phase coefficients in Eq. 28, 16, 14 of arXiv:1508.07253
        '''
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
 

# 11 amplitude coefficients
@ti.dataclass
class AmplitudeCoefficient:
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

    @ti.func
    def compute_amplitude_coefficients(self, source_params):
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




@ti.data_oriented
class IMRPhenomD(object):

    def __init__(self, parameter_sanity_check=True):

        self.Mf_phase_ins_join = 0.018
        self.Mf_amp_ins_join = 0.014
        self.Mf_cut = 0.2

        self.parameter_sanity_check = parameter_sanity_check

    @ti.func
    def _parameter_check(self):
        return SUCCESS
        
    @ti.func
    def _amplitude_coefficients(self):
        pass

    @ti.func
    def _phase_coefficients(self):
        pass

    @ti.func
    def _connection_coefficients(self):
        pass

    @ti.kernel
    def update_amplitude_phase_tf_field(self):
        pass

    @ti.kernel
    def updata_hplus_hcross_tf_field(self):
        pass

    def np_array_view_amplitude_phase_tf(self):
        pass
        
    def np_array_view_hplus_hcross_tf(self):
        pass