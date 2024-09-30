import numpy as np
import bilby
import pandas as pd
import lal
import lalsimulation as lalsim
from lalsimulation import SimInspiralChooseFDWaveformSequence


extraParams = lal.CreateDict()
approximant = lalsim.GetApproximantFromString('IMRPhenomXAS')
num_tests = 100
rng = np.random.default_rng()
parameters = {}
parameters['total_mass'] = rng.uniform(1e3, 1e6, num_tests)
parameters['mass_ratio'] = rng.uniform(0.2, 1.0, num_tests)
parameters['chi_1'] = rng.uniform(-1.0, 1.0, num_tests)
parameters['chi_2'] = rng.uniform(-1.0, 1.0, num_tests)
parameters['luminosity_distance'] = rng.uniform(1000.0, 10000.0, num_tests)
parameters['inclination'] = rng.uniform(0, np.pi, num_tests)
parameters['reference_phase'] = rng.uniform(0, 2*np.pi, num_tests)
parameters['coalescence_time'] = np.zeros(num_tests)
parameters = bilby.gw.conversion.generate_mass_parameters(parameters)
parameters = pd.DataFrame(parameters)

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
plus_lal, cross_lal = SimInspiralChooseFDWaveformSequence(parameters.iloc[0]['reference_phase'], 
                                                          parameters.iloc[0]['mass_1']*lal.MSUN_SI, 
                                                          parameters.iloc[0]['mass_2']*lal.MSUN_SI, 
                                                          0.0,
                                                          0.0,
                                                          parameters.iloc[0]['chi_1'], 
                                                          0.0,
                                                          0.0,
                                                          parameters.iloc[0]['chi_2'], 
                                                          0.0,
                                                          parameters.iloc[0]['luminosity_distance']*1e6*lal.PC_SI,
                                                          parameters.iloc[0]['inclination'],
                                                          extraParams, 
                                                          approximant,
                                                          f_array_lal
                                                          )


"""
m1 (Msun) = 539736.360155
m2 (Msun) = 152808.142120
m1_SI     = 1073217106106998369241788735247351808.000000
m2_SI     = 303845218113819457791843492691968000.000000
m1        = 0.779353
m2        = 0.220647
eta     : 1.7196212350860812e-01
q       : 3.5321178090911229e+00
chi1L   : -6.0839473244999986e-01
chi2L   : 9.9562935272995312e-01
chi_eff : -2.5447099243547744e-01
chi_hat : -3.1307270293336120e-01
STotR   : -4.8936525786114943e-01
fRef : 0.000010
phi0 : 4.113077
fCut : 0.087948
fMin : 0.000010
fMax : 0.050000
fMin        = 0.000010
fMax        = 0.050000
f_max_prime = 0.050000
Mf  = 0.979959
af  = 0.224286
frd = 0.065959
fda = 0.014323
fMECO = 0.017638
fISCO = 0.026011


eta     = 0.171962

ampNorm = 2.797773e-01
amp0 : 2.135153e-17

 **** Sanity checks complete. Waveform struct has been initialized. ****


 **** Now in IMRPhenomXASGenerateFD... ****


 **** Initializing amplitude struct... ****

chi1L  = -0.608395
chi1L2 = 0.370144
chi1L3 = -0.225194
chi2L2 = 0.991278
delta  = 0.558705
eta2   = 0.029571
eta3   = 0.005085
V1     = 0.763408
gamma1 = 0.012603
gamma2 = 0.739247
gamma3 = 1.336864
fmax   = 0.057501

Amplitude pseudo PN coeffcients
fAmpMatchIN = 0.019731
V1   = -0.021540
V2   = -0.035255
V3   = -0.002079
F1   = 9.865471e-03
F2   = 1.479821e-02
F3   = 1.973094e-02

TaylorF2 PN Amplitude Coefficients
pnTwoThirds   = -2.102832
pnThreeThirds = -2.889889
pnFourThirds  = -21.254615
pnFiveThirds  = -44.572895
pnSixThirds   = -78.892285

powers_of_lalpi.itself = 3.141593
powers_of_lalpi.four_thirds = 4.601151
powers_of_lalpi.five_thirds = 6.738809

Pseudo-PN Amplitude Coefficients (Agrees with MMA).
alpha1 = 9234.367876
alpha2 = -100269.959469
alpha3  = 244332.120110
d1 = 1.330422
d4 = 0.949442

Intermediate Region:
F1 = 0.019731
F2 = 0.038616
F3 = 0.000000
F4 = 0.057501

Insp@F1 = 0.579102
d1 = 1.330422
d4 = 0.949442
V1 = 0.017711
V2 = 0.034204
V3 = 0.000000
V4 = 0.046794
delta0 = -0.024380
delta1 = 3.084309
delta2 = -55.241985
delta3 = 333.227929
delta4 = 1200.270334
delta5 = 0.000000


 **** Amplitude struct initialized. ****



 **** Initializing phase struct... ****


Solving system of equations for RD phase...
Rigndown collocation points :
F1 : 0.035394
F2 : 0.042492
F3 : 0.059628
F4 : 0.065959
F5 : 0.083863
NCollRD = 5

Ringdown Collocation Points:
v1 : -72.385463
v2 : -80.520229
v3 : -95.729254
v4 : -99.185237
v5 : -93.926738

For row 0: a0 + a1 3.045711 + a2 798.238299 + a4 637184.381470 + aRD -5.087702

Ringdown Coefficients:
c0  : -70.915772
c1  : -11.206612
c2  : 0.053247
c4  : -1.201581e-05
cRD : 0.429540
d0  : 0.005797
cL  : -2.489892e-03

Freeing arrays...

NPseudoPN : 4
NColl : 4


Inspiral Collocation Points and Values:
F1 : 0.002600
F2 : 0.006448
F3 : 0.014143
F4 : 0.017990

V1 : 16068.170200
V2 : 25460.495495
V3 : 28168.856125
V4 : 26605.708890


3pPN
Inspiral Pseudo-PN Coefficients:
a0 : -32332.602160
a1 : 412927.136890
a2 : -140272.430899
a3 : -2202792.831793
a4 : 0.000000

TaylorF2 PN Coefficients:
phi0   : 1.000000
phi1   : 0.000000
phi2   : 12.794888
phi3   : -190.675906
phi4   : 170.671701
phi5   : 0.000000
phi6   : -15966.855184
phi7   : 45642.135519
phi8   : 3673.971541
phi5L  : 1686.842842
phi6L  : -1072.810332
phi8L  : 25385.023412
phi8P  : 53887.670267
phi9P  : -516158.921112
phi10P : 140272.430899
phi11P : 1835660.693161
phi12P : -0.000000

TaylorF2 PN Derivative Coefficients
dphi0  : 1.000000
dphi1  : 0.000000
dphi2  : 7.676933
dphi3  : -76.270362
dphi4  : 34.134340
dphi5  : -1012.105705
dphi6  : 3837.057236
dphi7  : -18256.854208
dphi8  : -17435.396971
dphi9  : -0.000000

dphi6L : 214.562066
dphi8L : -15231.014047
dphi9L : -0.000000

Transition frequency for ins to int : 0.017105

NColPointsInt : 5
For row 0: a0 + a1 3.856155 + a2 14.869930 + a3 57.340754 + a4 221.114828 = 29.328141 , ff0 = -3.105280, ff = 0.017105
For row 1: a0 + a1 3.327516 + a2 11.072361 + a3 36.843455 + a4 122.597174 = -9.954363
For row 2: a0 + a1 2.500080 + a2 6.250399 + a3 15.626495 + a4 39.067483 = -48.834704
For row 3: a0 + a1 2.002202 + a2 4.008815 + a3 8.026459 + a4 16.070596 = -63.387638
For row 4: a0 + a1 1.849630 + a2 3.421129 + a3 6.327822 + a4 11.704127 = -66.656839

Intermediate Collocation Points and Values:
F1 : 0.0171049
F2 : 0.0198223
F3 : 0.0263828
F4 : 0.0329432
F5 : 0.0356607

V's agree with Mathematica...
V1 : 26.2228614
V2 : -13.3314376
V3 : -53.0073826
V4 : -68.6003949
V5 : -72.3854625

g0 : -29.3704096
g1 : -83.9808285
g2 : 52.1567987
g3 : -12.0220688
g4 : 1.3401551

b0 : -29.3704096
b1 : -5.5392959
b2 : 0.2269132
b3 : -0.0034499
b4 : 0.0000254



 **** Phase struct initialized. ****


dphiIM = 26.222861 and dphiIN = 26.222861
phiIN(fIns)  : 33.9577733
phiIM(fIns)  : 13.3355180
fIns         : 0.0171049
C2           : -0.0000000


phiIMC(fInt) : 33.1309356
phiRD(fInt)  : -5.5586530
fInt         : 0.0356607
C2           : -0.0000000

dphiIM = -72.385463 and dphiRD = -72.738183

Continuity Coefficients
C1Int : 20.622255
C2Int : -0.000000
C1MRD : 38.677010
C2MRD : 0.352720


 **** Phase struct initialized. ****

C1IM     = 20.6223
C2IM     = -0.0000
C1RD     = 38.6770
C2RD     = 0.3527
fIN      = 0.0171
fIM      = 0.0357
"""