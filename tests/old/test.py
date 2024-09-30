import sys
sys.path.append('/home/hydrogen/workspace/Space_GW/pespace')
sys.path.append('/home/hydrogen/workspace/Space_GW/tiwave')

import taichi as ti
import numpy as np
from tiwave.waveforms.IMRPhenomD import _gauss_elimination_5x5
from tiwave.utils import _gauss_elimination

ti.init(default_fp=ti.f64)

a = np.array([[3., 2., 0.], [1., -1., 0.], [0., 5., 1.]])
b = np.array([2., 4., -1.])

x = np.linalg.solve(a, b)
print(x)

ab = np.concatenate((a, np.atleast_2d(b).T), axis=1)
print(ab)
ti_ab = ti.Matrix(ab)

ti_a = ti.Matrix(a)
ti_b = ti.Vector(b)

@ti.kernel
def _solve():
    x = ti.solve(ti_a, ti_b)
    print(x)

n=3
@ti.kernel
def main():
    x = _gauss_elimination(ti_ab)
    print(x)

main()




