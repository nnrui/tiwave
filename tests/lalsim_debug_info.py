import numpy as np
import bilby
import pandas as pd
import lal
import lalsimulation as lalsim
from lalsimulation import SimInspiralChooseFDWaveformSequence


extraParams = lal.CreateDict()
approximant = lalsim.GetApproximantFromString('IMRPhenomXAS')
params_in = dict(mass_1=159059.325070,
                 mass_2=108817.001497,
                 chi_1=-0.448101,
                 chi_2=0.014488,
                 luminosity_distance=7205.907347,
                 inclination=2.123734,
                 reference_phase=3.211444,
                 coalescence_time=0.000000,)
parameters = bilby.gw.conversion.generate_mass_parameters(params_in)

minimum_frequency = 1e-5
maximum_frequency = 0.1
duration = 3600*24*30*6
cadance = 10
f_array = np.arange(0, 1.0/(2*cadance), 1.0/duration)
bound = ((f_array >= minimum_frequency) * (f_array <= maximum_frequency))
f_array = f_array[bound]
data_length = len(f_array)
f_array_lal = lal.CreateREAL8Vector(data_length)
f_array_lal.data = f_array
plus_lal, cross_lal = SimInspiralChooseFDWaveformSequence(parameters['reference_phase'], 
                                                          parameters['mass_1']*lal.MSUN_SI, 
                                                          parameters['mass_2']*lal.MSUN_SI, 
                                                          0.0,
                                                          0.0,
                                                          parameters['chi_1'], 
                                                          0.0,
                                                          0.0,
                                                          parameters['chi_2'], 
                                                          0.0,
                                                          parameters['luminosity_distance']*1e6*lal.PC_SI,
                                                          parameters['inclination'],
                                                          extraParams, 
                                                          approximant,
                                                          f_array_lal
                                                          )
print(parameters)
"""
Inspiral Amp Version		: 103
Intermediate Amp Version	: 104
Ringdown Amp Version		: 103

Inspiral Phase Version		: 104
Intermediate Phase Version	: 105
Ringdown Phase Version		: 105

m1 (Msun) = 159059.325070
m2 (Msun) = 108817.001497
m1_SI     = 316275131995757910222824318246584320.000000
m2_SI     = 216372799876399367941314981536989184.000000
m1        = 0.593779
m2        = 0.406221
eta     : 2.4120550773087363e-01
q       : 1.4617139131001062e+00
chi1L   : -4.4810100000000003e-01
chi2L   : 1.4487999999999999e-02
chi_eff : -2.6018761268951091e-01
chi_hat : -2.6858794656933860e-01
STotR   : -3.0062032522700266e-01
fRef : 0.000010
phi0 : 3.211444
fCut : 0.227372
fMin : 0.000010
fMax : 0.050000
fMin        = 0.000010
fMax        = 0.050000
f_max_prime = 0.050000
Mf  = 0.961244
af  = 0.560358
frd = 0.079700
fda = 0.014008
fMECO = 0.018423
fISCO = 0.037284


eta     = 0.241206

ampNorm = 3.313521e-01
amp0 : 2.347195e-18

 **** Sanity checks complete. Waveform struct has been initialized. ****


 **** Now in IMRPhenomXASGenerateFD... ****


 **** Initializing amplitude struct... ****

chi1L  = -0.448101
chi1L2 = 0.200795
chi1L3 = -0.089976
chi2L2 = 0.000210
delta  = 0.187558
eta2   = 0.058180
eta3   = 0.014033
V1     = 1.043634
gamma1 = 0.016014
gamma2 = 0.807845
gamma3 = 1.312529
fmax   = 0.070355

Amplitude pseudo PN coeffcients
fAmpMatchIN = 0.023138
V1   = -0.010162
V2   = 0.008887
V3   = 0.088581
F1   = 1.156899e-02
F2   = 1.735348e-02
F3   = 2.313797e-02

TaylorF2 PN Amplitude Coefficients
pnTwoThirds   = -1.704102
pnThreeThirds = -2.457540
pnFourThirds  = -21.570839
pnFiveThirds  = -23.872240
pnSixThirds   = -43.283461

powers_of_lalpi.itself = 3.141593
powers_of_lalpi.four_thirds = 4.601151
powers_of_lalpi.five_thirds = 6.738809

Pseudo-PN Amplitude Coefficients (Agrees with MMA).
alpha1 = 710.843031
alpha2 = -20672.518153
alpha3  = 70943.444105
d1 = 0.873753
d4 = 0.718260

Intermediate Region:
F1 = 0.023138
F2 = 0.046747
F3 = 0.000000
F4 = 0.070355

Insp@F1 = 0.683102
d1 = 0.873753
d4 = 0.718260
V1 = 0.018081
V2 = 0.033174
V3 = 0.000000
V4 = 0.043314
delta0 = -0.001525
delta1 = 0.568562
delta2 = 25.643012
delta3 = -703.633528
delta4 = 5018.045605
delta5 = 0.000000

 **** Amplitude struct initialized. ****

 **** Initializing phase struct... ****

Solving system of equations for RD phase...
Rigndown collocation points :
F1 : 0.046280
F2 : 0.053739
F3 : 0.071745
F4 : 0.079700
F5 : 0.097210
NCollRD = 5

Ringdown Collocation Points:
v1 : -67.831458
v2 : -75.412049
v3 : -93.405762
v4 : -99.674816
v5 : -90.733650

For row 0: a0 + a1 2.785273 + a2 466.881832 + a4 217978.644800 + aRD -4.414410

Ringdown Coefficients:
c0  : -45.080870
c1  : -23.422597
c2  : 0.122491
c4  : -5.504437e-05
cRD : 0.612227
d0  : 0.005797
cL  : -3.548862e-03

Freeing arrays...

NPseudoPN : 4
NColl : 4


Inspiral Collocation Points and Values:
F1 : 0.002600
F2 : 0.006648
F3 : 0.014743
F4 : 0.018791

V1 : 17823.495830
V2 : 22473.842470
V3 : 20468.455811
V4 : 17592.647788


3pPN
Inspiral Pseudo-PN Coefficients:
a0 : -17535.380013
a1 : 346008.287337
a2 : -476763.485022
a3 : -1232666.902429
a4 : 0.000000

TaylorF2 PN Coefficients:
phi0   : 1.000000
phi1   : 0.000000
phi2   : 13.702565
phi3   : -184.540539
phi4   : 193.966223
phi5   : 0.000000
phi6   : -16996.800941
phi7   : 34294.620558
phi8   : 2578.950716
phi5L  : 1504.309510
phi6L  : -1072.810332
phi8L  : 17819.061354
phi8P  : 29225.633354
phi9P  : -432510.359171
phi10P : 476763.485022
phi11P : 1027222.418691
phi12P : -0.000000

TaylorF2 PN Derivative Coefficients
dphi0  : 1.000000
dphi1  : 0.000000
dphi2  : 8.221539
dphi3  : -73.816215
dphi4  : 38.793245
dphi5  : -902.585706
dphi6  : 4043.046388
dphi7  : -13717.848223
dphi8  : -12238.807242
dphi9  : -0.000000

dphi6L : 214.562066
dphi8L : -10691.436812
dphi9L : -0.000000

Transition frequency for ins to int : 0.017587

NColPointsInt : 5
For row 0: a0 + a1 4.531779 + a2 20.537019 + a3 93.069229 + a4 421.769160 = 79.483149 , ff0 = -3.057418, ff = 0.017587
For row 1: a0 + a1 3.647576 + a2 13.304807 + a3 48.530289 + a4 177.017895 = 13.304950
For row 2: a0 + a1 2.479587 + a2 6.148353 + a3 15.245377 + a4 37.802242 = -39.436953
For row 3: a0 + a1 1.878177 + a2 3.527549 + a3 6.625362 + a4 12.443603 = -56.691133
For row 4: a0 + a1 1.706712 + a2 2.912866 + a3 4.971422 + a4 8.484786 = -60.256613

Intermediate Collocation Points and Values:
F1 : 0.0175870
F2 : 0.0218502
F3 : 0.0321426
F4 : 0.0424350
F5 : 0.0466982

V's agree with Mathematica...
V1 : 76.4257312
V2 : 9.8690605
V3 : -44.0963559
V4 : -63.2220021
V5 : -67.8314575

g0 : -34.5118361
g1 : -65.9662410
g2 : 42.2830464
g3 : -8.7644819
g4 : 0.8542000

b0 : -34.5118361
b1 : -5.2575343
b2 : 0.2685883
b3 : -0.0044372
b4 : 0.0000345



 **** Phase struct initialized. ****


dphiIM = 76.425731 and dphiIN = 76.425731
phiIN(fIns)  : 32.0498792
phiIM(fIns)  : 11.0066829
fIns         : 0.0175870
C2           : -0.0000000


phiIMC(fInt) : 31.1331465
phiRD(fInt)  : -8.8078947
fInt         : 0.0466982
C2           : -0.0000000

dphiIM = -67.831458 and dphiRD = -68.289810

Continuity Coefficients
C1Int : 21.043196
C2Int : -0.000000
C1MRD : 39.919637
C2MRD : 0.458352


 **** Phase struct initialized. ****

C1IM     = 21.0432
C2IM     = -0.0000
C1RD     = 39.9196
C2RD     = 0.4584
fIN      = 0.0176
fIM      = 0.0467
{'mass_1': 159059.32507, 'mass_2': 108817.001497, 'chi_1': -0.448101, 'chi_2': 0.014488, 'luminosity_distance': 7205.907347, 'inclination': 2.123734, 'reference_phase': 3.211444, 'coalescence_time': 0.0, 'chirp_mass': 114121.28862886243, 'total_mass': 267876.326567, 'symmetric_mass_ratio': 0.2412055077308736, 'mass_ratio': 0.6841283995711098}
"""